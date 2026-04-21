# GS-LRM: Large Reconstruction Model for 3D Gaussian Splatting

---

## 1. 핵심 주장과 주요 기여 요약

### 핵심 주장

GS-LRM은 **2~4장의 포즈(카메라 파라미터)가 알려진 희소 입력 이미지**로부터 단일 A100 GPU에서 **약 0.23초** 만에 고품질 3D Gaussian primitives를 예측하는 **확장 가능한(scalable) 대형 재구성 모델**이다. 핵심은 기존 LRM들이 채택하던 **Triplane NeRF** 표현을 버리고, **픽셀 정렬(pixel-aligned) 3D Gaussian Splatting**을 채택함으로써 속도·품질·범용성을 동시에 향상시킨 것이다.

### 주요 기여

| 기여 항목 | 내용 |
|---|---|
| **단순하고 확장 가능한 아키텍처** | ViT 기반 순수 Transformer로 구현, triplane 등 별도 3D 표현 불필요 |
| **픽셀 정렬 Gaussian 예측** | 각 픽셀에 하나의 3D Gaussian을 대응, 고주파 디테일 보존 |
| **객체 + 장면 통합 지원** | Objaverse(객체)와 RealEstate10K(실내외 장면) 모두에서 SOTA 달성 |
| **정량적 성능 향상** | 객체: +4dB PSNR, 장면: +2.2dB PSNR (기존 SOTA 대비) |
| **다운스트림 응용 데모** | Text-to-3D, Image-to-3D, Text-to-Scene 파이프라인 연계 시연 |

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

기존의 문제들을 다음과 같이 정리할 수 있다:

1. **Triplane NeRF의 해상도 한계**: 기존 LRM들(LRM, Instant3D, DMV3D 등)은 triplane NeRF를 사용하나, 고정된 triplane 해상도로 인해 고주파 디테일 재현이 어렵다.
2. **느린 렌더링**: Volume rendering 기반이므로 추론 속도가 느리다.
3. **객체 중심 편향**: 기존 LRM들은 주로 object-centric reconstruction에 초점을 맞춰 대규모 장면 처리에 취약하다.
4. **복잡한 아키텍처**: 기존 방법들은 epipolar line 기반 feature aggregation 등 복잡한 3D inductive bias를 사용한다.

---

### 2.2 제안하는 방법 및 수식

#### (1) 입력 토큰화 (Tokenizing Posed Images)

입력: $N$개의 멀티뷰 이미지 $\{\mathbf{I}_i \in \mathbb{R}^{H \times W \times 3} \mid i = 1, 2, \ldots, N\}$과 카메라 내/외부 파라미터.

카메라 파라미터로부터 **Plücker 광선 좌표** $\{\mathbf{P}_i \in \mathbb{R}^{H \times W \times 6}\}$를 계산하여 포즈 조건부 임베딩으로 사용한다. RGB와 Plücker 좌표를 채널 방향으로 concat하면 9채널 feature map이 된다.

패치 크기 $p$로 patchify 후 선형 투영:

$$\{\mathbf{T}_{ij}\}_{j=1,2,\ldots,HW/p^2} = \text{Linear}\Big(\text{Patchify}_p\Big(\text{Concat}(\mathbf{I}_i, \mathbf{P}_i)\Big)\Big) $$

여기서 $\mathbf{T}_{ij} \in \mathbb{R}^d$는 이미지 $i$의 $j$번째 패치 토큰이며, 이미지당 총 $HW/p^2$개의 토큰이 생성된다.

> Plücker 좌표는 픽셀/뷰마다 고유하게 달라지므로 **추가적인 위치 임베딩 없이** 공간적 구별 역할을 수행한다.

#### (2) Transformer 처리

모든 멀티뷰 토큰을 concat하여 $L$개의 Transformer 블록을 통과시킨다:

$$\{\mathbf{T}_{ij}\}^0 = \{\mathbf{T}_{ij}\} $$

$$\{\mathbf{T}_{ij}\}^l = \text{TransformerBlock}^l(\{\mathbf{T}_{ij}\}^{l-1}), \quad l = 1, 2, \ldots, L $$

각 블록은 **Pre-LayerNorm + Multi-Head Self-Attention + MLP + Residual Connection**으로 구성된다.

#### (3) 출력 토큰 → 픽셀별 Gaussian 디코딩

출력 토큰으로부터 단일 선형 레이어로 Gaussian 파라미터를 디코딩:

$$\{\mathbf{G}_{ij}\} = \text{Linear}(\{\mathbf{T}_{ij}\}^L) $$

여기서 $\mathbf{G}_{ij} \in \mathbb{R}^{p^2 \cdot q}$이고, $q = 12$ (RGB: 3, scale: 3, rotation quaternion: 4, opacity: 1, ray distance: 1).

Unpatchify 후 각 이미지당 $HW$개의 Gaussian이 생성되며, $N$개 뷰를 합치면 총 $N \cdot HW$개의 Gaussian이 출력된다.

#### (4) Gaussian 중심 위치 계산

$$\text{xyz} = \text{ray}_o + t \cdot \text{ray}_d $$

ray distance $t$로의 변환:

$$\omega = \sigma(\mathbf{G}_\text{distance}) $$

$$t = (1 - \omega)\, d_\text{near} + \omega\, d_\text{far} $$

#### (5) 스케일 및 Opacity 파라미터화

$$\text{scale} = \min\{\exp(\mathbf{G}_\text{scale} - 2.3),\ 0.3\} $$

$$\text{opacity} = \sigma(\mathbf{G}_\text{opacity} - 2.0) $$

> 초기값이 약 0.1이 되도록 bias를 설정하여 학습 안정성 확보. 최대 스케일 0.3 clipping으로 선형 Gaussian 퇴화 방지.

#### (6) 손실 함수

MSE + Perceptual loss의 조합:

```math
\mathcal{L} = \frac{1}{M} \sum_{i'=1}^{M} \left( \text{MSE}\left(\hat{\mathbf{I}}^*_{i'}, \mathbf{I}^*_{i'}\right) + \lambda \cdot \text{Perceptual}\left(\hat{\mathbf{I}}^*_{i'}, \mathbf{I}^*_{i'}\right) \right)
```

여기서 $\lambda = 0.5$, Perceptual loss는 VGG-19 기반 (LPIPS보다 학습 안정적).

---

### 2.3 모델 구조 상세

```
입력: N × (H×W×3 RGB + H×W×6 Plücker) → 9채널
↓ Patchify (8×8 패치) + Linear → 토큰화
↓ 모든 뷰 토큰 Concatenate (총 N·HW/64개 토큰)
↓ 24층 Transformer (hidden dim=1024, 16-head self-attention, MLP dim=4096)
↓ Linear 디코더 → 픽셀당 Gaussian 파라미터 (q=12)
↓ Unpatchify → 뷰당 HW개 Gaussian
↓ 모든 뷰 Merge → N·HW개 Gaussian
출력: 3D Gaussian Splats → 차분 렌더링
```

**주요 구성요소:**
- **패치 크기**: $8 \times 8$
- **모델 파라미터**: 약 300M
- **최대 토큰 길이**: 최대 16K (512×512×4뷰 입력 시)
- **학습 최적화**: FlashAttention-v2, Gradient Checkpointing, BF16 mixed precision, Deferred Backpropagation

---

### 2.4 성능 향상

**객체 수준 (GSO / ABO 데이터셋):**

| 방법 | PSNR (GSO) | SSIM | LPIPS | PSNR (ABO) | SSIM | LPIPS |
|---|---|---|---|---|---|---|
| Triplane-LRM [Instant3D] | 26.54 | 0.893 | 0.064 | 27.50 | 0.896 | 0.093 |
| **GS-LRM (Res-512)** | **30.52** | **0.952** | **0.050** | **29.09** | **0.925** | **0.085** |
| LGM | 21.44 | 0.832 | 0.122 | 20.79 | 0.813 | 0.158 |
| **GS-LRM (Res-256)** | **29.59** | **0.944** | **0.051** | **28.98** | **0.926** | **0.074** |

**장면 수준 (RealEstate10K 데이터셋):**

| 방법 | PSNR | SSIM | LPIPS |
|---|---|---|---|
| pixelNeRF | 20.43 | 0.589 | 0.550 |
| GPNR | 24.11 | 0.793 | 0.255 |
| Du et al. | 24.78 | 0.820 | 0.213 |
| pixelSplat | 25.89 | 0.858 | 0.142 |
| **GS-LRM (Ours)** | **28.10** | **0.892** | **0.114** |

---

### 2.5 한계점

논문에서 명시한 한계점:

1. **해상도 제한**: 현재 최대 $512 \times 904$ 해상도; 1K~2K 이상으로의 확장이 필요하다.
2. **카메라 파라미터 의존성**: 입력 이미지의 포즈를 알아야 한다는 전제가 실용적이지 않을 수 있다.
3. **픽셀 정렬 표현의 한계**: 뷰 프러스텀(view frustum) 내부 표면만 명시적으로 모델링하므로, **미관측 영역(unseen regions)** 재구성 능력이 제한적이다.
4. **정적 장면만 처리**: 동적 장면(dynamic scene)에 대한 지원이 없다.
5. **View-dependent 표현 부재**: 0차 구형 조화 함수(Spherical Harmonics)만 사용하여 시점 의존적 외관 모델링이 단순하다.

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 현재 일반화 성능의 기반

GS-LRM의 일반화 성능은 다음 구조적 요소에서 비롯된다:

#### (a) Self-Attention의 멀티뷰 대응 학습
기존 epipolar-based 방법들은 사전 정의된 기하학적 제약(에피폴라 라인)에서만 특징을 집계하지만, GS-LRM의 **dense self-attention**은 모든 뷰의 모든 픽셀 간의 대응 관계를 데이터로부터 직접 학습한다:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

이를 통해 특정 기하학적 구조에 편향되지 않은 **범용 재구성 사전(general reconstruction prior)**을 학습한다.

#### (b) Plücker 좌표 기반 포즈 조건화
Plücker 광선 좌표는 카메라 파라미터를 픽셀 수준에서 인코딩한다:

$$\mathbf{P}_i = (\text{ray}_o \times \text{ray}_d,\ \text{ray}_d) \in \mathbb{R}^{H \times W \times 6}$$

이 표현은 카메라 배치에 무관하게 **상대적인 3D 공간 정보**를 제공하므로, 다양한 카메라 구성에 대해 일반화된다.

#### (c) 해상도 적응성 (Resolution Scalability)
Triplane LRM은 고정 해상도의 triplane에 재구성 결과를 저장하지만, GS-LRM은 **입력 해상도에 비례하여 Gaussian 수가 증가**하므로($N \cdot HW$개), 고해상도 입력에서도 디테일을 보존한다.

#### (d) 도메인 독립적 아키텍처
동일한 Transformer 아키텍처를 Objaverse(객체)와 RealEstate10K(장면)에 **최소한의 도메인 특화 파라미터 변경**으로 학습할 수 있음을 증명하였다. 이는 아키텍처 자체의 범용성을 시사한다.

### 3.2 일반화 성능 향상을 위한 향후 방향

| 방향 | 내용 |
|---|---|
| **포즈 비의존 학습** | 카메라 파라미터 없이 동작하는 pose-free 방식 연구 (DUSt3R 등과 결합) |
| **더 다양한 데이터셋 학습** | 현재는 Objaverse + RealEstate10K에 한정; 야외 장면, 의료 영상 등 다양한 도메인 확장 |
| **Zero-shot 크로스 도메인** | 단일 통합 모델로 객체+장면+인체 등 모든 도메인 처리 |
| **고해상도 입력 지원** | 1K~4K 해상도 지원으로 산업·의료 적용 가능성 확대 |
| **동적 장면 대응** | 4D Gaussian(시간 축 추가)으로 동적 콘텐츠 재구성 |
| **미관측 영역 처리** | Hallucination 능력 강화를 위한 생성 모델과의 결합 |
| **더 높은 차수의 SH** | 고차 구형 조화 함수로 view-dependent 효과 향상 |

---

## 4. 2020년 이후 관련 최신 연구 비교 분석

### 4.1 기술 계보 및 비교

```
NeRF (2020) → pixelNeRF (2021) → IBRNet (2021) → MVSNeRF (2021)
                                                          ↓
3DGS (2023) → PixelSplat (2023) → LGM (2024) → GS-LRM (2024)
                                                          ↑
LRM (2023) → Instant3D (2023) → DMV3D (2023) ──────────→
```

### 4.2 주요 방법론 비교표

| 방법 | 연도 | 표현 | 아키텍처 | 입력 뷰 | 객체/장면 | 렌더링 속도 | 특이사항 |
|---|---|---|---|---|---|---|---|
| **NeRF** | 2020 | MLP NeRF | Per-scene 최적화 | 다수 | 장면 | 느림 | 범용 표준 |
| **pixelNeRF** | 2021 | NeRF | CNN+NeRF | 1~몇 장 | 장면 | 중간 | 최초 generalizable NeRF |
| **IBRNet** | 2021 | NeRF | Transformer | 다수 | 장면 | 중간 | 멀티뷰 feature 집계 |
| **MVSNeRF** | 2021 | NeRF | Cost volume | 3장 | 장면 | 중간 | MVS+NeRF 결합 |
| **3DGS** | 2023 | Gaussian | Per-scene 최적화 | 다수 | 객체/장면 | **실시간** | 실시간 렌더링 표준 |
| **LRM** | 2023 | Triplane NeRF | Transformer | 1장 | 객체 | 느림 | 최초 대형 재구성 모델 |
| **Instant3D** | 2023 | Triplane NeRF | Transformer | 4장 | 객체 | 느림 | Text-to-3D 파이프라인 |
| **DMV3D** | 2023 | Triplane NeRF | Transformer | 다수 | 객체 | 느림 | 확산 모델 결합 |
| **pixelSplat** | 2023 | 3D Gaussian | CNN+Epipolar | 2장 | **장면** | 빠름 | Epipolar 기반 GS |
| **LGM** | 2024 | 3D Gaussian | U-Net | 4장 | **객체** | 빠름 | U-Net 기반 GS |
| **GS-LRM** | 2024 | 3D Gaussian | **순수 Transformer** | 2~4장 | **객체+장면** | **빠름** | 픽셀 정렬 GS+범용성 |

### 4.3 핵심 차별점 분석

**vs. Triplane LRM 계열 (LRM, Instant3D, DMV3D)**:
- GS-LRM은 triplane을 제거하고 pixel-aligned Gaussian으로 대체 → 고주파 디테일 보존, 빠른 렌더링, 대규모 장면 처리 가능

**vs. pixelSplat**:
- pixelSplat은 epipolar line 기반 feature aggregation을 사용하지만 GS-LRM은 전역 self-attention 사용 → PSNR +2.2dB 향상
- pixelSplat은 장면만 지원하지만 GS-LRM은 객체+장면 모두 지원

**vs. LGM**:
- LGM은 U-Net 기반이나 GS-LRM은 Transformer 기반 → 더 쉬운 확장성
- PSNR 약 +8dB 향상 (대등한 compute에서)

---

## 5. 향후 연구에 미치는 영향과 고려할 점

### 5.1 향후 연구에 미치는 영향

#### (a) 패러다임 전환: NeRF → Gaussian Splatting in Feed-forward Models
GS-LRM은 feed-forward 3D 재구성에서 Triplane NeRF 대신 Gaussian Splatting이 더 효과적임을 대규모로 검증하였다. 이후 연구들이 Gaussian 기반 feed-forward 모델로 전환하는 데 강력한 근거를 제공한다.

#### (b) 단순성의 힘: 3D Inductive Bias 없는 Transformer
Epipolar line, cost volume 등의 복잡한 3D prior 없이도 순수 Transformer + 대용량 데이터로 SOTA를 달성할 수 있음을 증명하였다. 이는 **Scaling Law가 3D 재구성에도 적용됨**을 시사한다.

#### (c) 통합 모델의 가능성
동일 아키텍처로 객체·장면 양쪽을 처리한 최초의 사례로서, **단일 통합 3D 재구성 모델** 연구를 촉진한다.

#### (d) 3D 생성 파이프라인의 강화
Text-to-3D, Image-to-3D 파이프라인의 재구성 백본으로 GS-LRM을 활용할 수 있음을 실증하여, **3D 생성 분야의 발전**을 가속화한다.

### 5.2 앞으로의 연구에서 고려할 점

#### 기술적 고려사항

1. **포즈 추정과의 결합**: 현재 GS-LRM은 정확한 카메라 파라미터를 요구한다. DUSt3R, PF-LRM처럼 포즈 추정을 내재화하거나, COLMAP 없이 동작하는 방향을 모색해야 한다.

2. **메모리 효율화**: $N \cdot HW$개의 Gaussian은 해상도에 따라 기하급수적으로 증가한다. 예) $4 \times 512 \times 512 = 1{,}048{,}576$개. 효율적인 Gaussian pruning 또는 압축 기법 연구가 필요하다.

3. **미관측 영역(Hallucination)**: 픽셀 정렬 표현은 입력 뷰 내부만 명시적으로 모델링하므로, 생성 모델(Diffusion Model)과의 결합으로 미관측 영역을 채우는 연구가 필요하다.

4. **View-Dependent 외관**: 현재 0차 Spherical Harmonics만 사용하므로 고차 SH나 Neural Radiance Cache와의 결합으로 반사, 광택 등의 표현력을 높여야 한다.

5. **동적 장면 확장**: 4D Gaussian Splatting과 결합하여 동영상 입력에서 시공간 재구성을 수행하는 연구가 유망하다.

#### 데이터 및 학습 관련 고려사항

6. **다양한 도메인 데이터**: 현재 합성 데이터(Objaverse)와 실내 장면(RealEstate10K)에 편향되어 있으므로, 야외·의료·산업 등 다양한 도메인 데이터로의 확장이 필요하다.

7. **데이터 품질 vs. 양**: Scaling Law 관점에서 데이터 품질과 양 사이의 트레이드오프를 체계적으로 분석해야 한다.

8. **도메인 일반화 평가**: 학습 데이터와 다른 분포(out-of-distribution)에서의 성능 저하를 체계적으로 평가하고 개선책을 마련해야 한다.

#### 응용 관련 고려사항

9. **실시간 처리**: 현재 0.23초 추론이지만 모바일/엣지 디바이스에서의 경량화 연구가 필요하다.

10. **3D 표준 파이프라인 통합**: CAD, 게임, AR/VR, 의료 영상 등 산업 파이프라인과의 통합 시 요구되는 정밀도와 포맷 호환성을 고려해야 한다.

---

## 참고자료 (출처)

1. **Zhang, K., Bi, S., Tan, H., et al. "GS-LRM: Large Reconstruction Model for 3D Gaussian Splatting."** arXiv:2404.19702v1 (2024). — *본 분석의 주요 원본 논문*

2. **Kerbl, B., et al. "3D Gaussian Splatting for Real-Time Radiance Field Rendering."** ACM Transactions on Graphics, 42(4), 2023. — *3DGS 원본 논문 (논문 내 [30])*

3. **Mildenhall, B., et al. "NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis."** ECCV 2020. — *NeRF 원본 (논문 내 [41])*

4. **Li, J., et al. "Instant3D: Fast Text-to-3D with Sparse-View Generation and Large Reconstruction Model."** 2023. — *Triplane-LRM 베이스라인 (논문 내 [32])*

5. **Hong, Y., et al. "LRM: Large Reconstruction Model for Single Image to 3D."** 2023. — *최초 LRM (논문 내 [27])*

6. **Charatan, D., et al. "pixelSplat: 3D Gaussian Splats from Image Pairs for Scalable Generalizable 3D Reconstruction."** arXiv:2312.12337, 2023. — *장면 레벨 GS 베이스라인 (논문 내 [8])*

7. **Tang, J., et al. "LGM: Large Multi-View Gaussian Model for High-Resolution 3D Content Creation."** arXiv:2402.05054, 2024. — *객체 레벨 GS 베이스라인 (논문 내 [61])*

8. **Dosovitskiy, A., et al. "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale."** ICLR 2021. — *ViT 아키텍처 기반 (논문 내 [20])*

9. **Yu, A., et al. "pixelNeRF: Neural Radiance Fields from One or Few Images."** CVPR 2021. — *generalizable NeRF 비교 (논문 내 [72])*

10. **Wang, S., et al. "DUSt3R: Geometric 3D Vision Made Easy."** arXiv:2312.14132, 2023. — *포즈 프리 재구성 관련 미래 방향 (논문 내 [65])*

11. **Dao, T. "FlashAttention-2: Faster Attention with Better Parallelism."** arXiv:2307.08691, 2023. — *학습 최적화 (논문 내 [18])*

12. **GS-LRM 프로젝트 페이지**: https://sai-bi.github.io/project/gs-lrm/
