# Depth Any Video with Scalable Synthetic Data

---

## 1. 핵심 주장과 주요 기여 요약

### 핵심 주장
"Depth Any Video"는 비디오 깊이 추정(Video Depth Estimation) 분야에서 두 가지 근본적인 문제를 동시에 해결한다:
1. **데이터 부족 문제**: 일관성 있고 확장 가능한 실측 깊이 데이터의 희소성
2. **시간적 일관성 문제**: 프레임 간 일관되지 않은 깊이 예측으로 인한 플리커링(flickering) 아티팩트

### 주요 기여

| 기여 항목 | 내용 |
|---|---|
| **DA-V 데이터셋** | 다양한 가상 환경에서 수집한 40,000개의 5초 비디오 클립 (6M 프레임) |
| **생성 모델 기반 프레임워크** | SVD(Stable Video Diffusion) 기반의 조건부 디노이징 프로세스 |
| **혼합 지속시간 학습 전략** | 가변 길이/프레임률 비디오 처리를 위한 Frame Dropout + Video Packing |
| **장시간 비디오 추론** | 키프레임 예측 + 프레임 보간을 통한 최대 150프레임 고해상도 추론 |
| **조건부 플로우 매칭** | 추론 속도 6.5배 향상 (25스텝 → 3스텝) |

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

**기존 방법의 한계:**
- **LiDAR/구조광 센서**: 비용이 높고, 특정 조명 조건이나 반사 표면에서 실패
- **스테레오 매칭 기반**: 약한 텍스처 영역에서 실패하며, 계산 비용이 큼
- **기존 데이터셋의 제한**: KITTI(25K), DynamicReplica(169K) 등은 규모·다양성 부족
- **시간적 일관성**: 단일 이미지 깊이 추정 모델을 비디오에 적용 시 프레임 간 불일치

### 2.2 제안하는 방법 (수식 포함)

#### (A) 깊이 정규화

VAE의 입력 범위 $[-1, 1]$에 맞추기 위한 정규화:

$$\tilde{x}_d = \left(\frac{x_d - d_2}{d_{98} - d_2} - 0.5\right) \times 2 $$

여기서 $d_2$와 $d_{98}$은 각각 깊이값 $x_d$의 2번째 및 98번째 백분위수이다.

#### (B) 조건부 플로우 매칭 (Conditional Flow Matching)

가우시안 노이즈 $\epsilon \sim \mathcal{N}(0, I)$와 데이터 $x \sim p(x)$ 사이의 선형 보간:

$$\phi_t(x) = tx + (1-t)\epsilon $$

시간 의존적 속도장 (noise → data 방향):

$$v_t(x) = x - \epsilon $$

이 속도장은 ODE를 정의한다:

$$d\phi_t(x) = v_t(\phi_t(x))\,dt $$

학습 시 플로우 매칭 목적 함수:

$$\mathcal{L}_\theta = \mathbb{E}_t \left\| v_\theta\left(\phi_t(z_d),\, z_c,\, t\right) - v_t(z_d) \right\|^2 $$

여기서 $z_d = \mathcal{E}(\tilde{x}_d)$는 latent 깊이 코드, $z_c = \mathcal{E}(x_c)$는 latent 비디오 코드이다.

#### (C) 프레임 보간 (Long Video Inference)

장시간 비디오 추론을 위한 프레임 보간 공식:

$$\tilde{z}_d = v_\theta\left(\phi_t(z_d),\, z_c,\, \hat{z}_d,\, m,\, t\right) $$

여기서 $\hat{z}_d$는 예측된 키프레임, $m$은 알려진 키프레임 위치를 나타내는 마스킹 맵이다.

#### (D) 시간적 정렬 오차 (TAE, Temporal Alignment Error)

비디오 깊이의 시간적 일관성을 정량적으로 평가하기 위한 새로운 메트릭:

$$\text{TAE} = \frac{1}{2(T-2)} \sum_{k=0}^{T-1} \left[ \text{AbsRel}\left(f(\hat{x}_d^k, p^k),\, \hat{x}_d^{k+1}\right) + \text{AbsRel}\left(f(\hat{x}_d^{k+1}, p_-^{k+1}),\, \hat{x}_d^k\right) \right] $$

여기서 $T$는 프레임 수, $f$는 변환 행렬 $p^k$를 이용해 $k$번째 프레임의 깊이를 $(k+1)$번째 프레임으로 투영하는 함수이다.

### 2.3 모델 구조

```
입력 비디오 x_c ──→ Latent Encoder E ──→ z_c ─┐
                                                 ├→ Concat → Denoising U-Net v_θ → z_d_hat → Decoder D → x_d_hat
깊이 GT x_d ──→ Normalize → Encoder E ──→ z_d ─┘
                                    ↑
                              Add Noise φ_t(z_d)
```

**핵심 구조 요소:**

| 구성 요소 | 상세 내용 |
|---|---|
| **기반 모델** | Stable Video Diffusion (SVD) |
| **VAE** | 공간 차원만 압축하는 2D 공간 VAE (시간 압축 제외) |
| **디노이저** | 3D UNet with Temporal Transformer |
| **위치 인코딩** | Sinusoidal → RoPE (Rotary Position Encoding)로 교체 |
| **조건 임베딩** | CLIP 임베딩 제거 → Zero 임베딩으로 대체 |
| **스케줄러** | EDM → Conditional Flow Matching으로 교체 |

**혼합 지속시간 학습 전략:**
- **Frame Dropout**: 긴 비디오에서 원래 인덱스를 유지한 채 $K$개 프레임을 랜덤 샘플링
- **Video Packing**: 유사 해상도 비디오를 그룹화하여 가변 배치 크기로 메모리 효율 최적화

### 2.4 성능 향상

#### 단일 프레임 깊이 추정 (Zero-shot, Affine-invariant)

| 방법 | NYUv2 AbsRel↓ | NYUv2 δ₁↑ | KITTI AbsRel↓ | ETH3D AbsRel↓ | ScanNet AbsRel↓ |
|---|---|---|---|---|---|
| Marigold (2024) | 5.5 | 96.4 | 9.9 | 6.5 | 6.4 |
| GeoWizard (2024) | 5.2 | 96.6 | 9.7 | 6.4 | 6.1 |
| **Depth Any Video (Ours)** | **5.1** | **97.0** | **7.3** | **4.7** | **5.3** |

#### 비디오 깊이 추정 (ScanNet++)

| 방법 | AbsRel↓ | δ₁↑ | TAE↓ |
|---|---|---|---|
| NVDS (2023) | 22.2 | 61.9 | 3.7 |
| ChronoDepth (2024) | 10.4 | 90.7 | 2.3 |
| DepthCrafter (2024) | 11.5 | 88.1 | 2.2 |
| **Depth Any Video (Ours)** | **9.3** | **93.4** | **2.1** |

#### 추론 효율성 (ScanNet)

| 방법 | 디노이징 스텝↓ | 파라미터↓ | 런타임↓ | δ₁↑ |
|---|---|---|---|---|
| Marigold | 50 | 865.9M | 2.06s | 94.5 |
| ChronoDepth | 10 | 1524.6M | 1.04s | 93.4 |
| DepthCrafter | 25 | 2156.7M | 4.80s | 93.8 |
| **Depth Any Video** | **3** | 1422.8M | **0.37s** | **96.1** |

### 2.5 한계점

논문이 명시한 한계:
- **거울·수면 반사** 처리 어려움: 반사 표면에서의 깊이 추정 실패
- **극단적으로 긴 비디오**: 150프레임 이상의 영상에서 품질 저하
- **3D VAE의 한계**: 시간 압축을 포함한 3D VAE 사용 시 빠른 모션에서 블러 아티팩트 발생
- **저품질 VAE의 상한선**: 현재 사용하는 SVD의 3D VAE보다 SD3의 2D VAE가 재구성 품질이 더 높으나, 비디오 시간적 처리를 위해 타협

---

## 3. 모델의 일반화 성능 향상 가능성 (중점 분석)

### 3.1 일반화 성능 향상의 핵심 메커니즘

#### (A) 합성 데이터 파이프라인 (DA-V Dataset)

기존 합성 데이터셋과의 비교:

| 데이터셋 | 실외 | 실내 | 동적 | 비디오 | 프레임 수 |
|---|---|---|---|---|---|
| Hypersim (2021) | ✗ | ✓ | ✗ | ✗ | 68K |
| VKITTI (2020) | ✓ | ✗ | ✗ | ✓ | 25K |
| DynamicReplica (2023) | ✗ | ✓ | ✓ | ✓ | 169K |
| **DA-V (Ours)** | **✓** | **✓** | **✓** | **✓** | **6M** |

DA-V는 실내·실외, 정적·동적 장면 모두를 포괄하며, 기존 데이터셋 대비 압도적인 규모(6M 프레임)를 달성했다. 이는 게임 엔진(Cities: Skylines II, Grand Theft Auto V)의 깊이 버퍼를 ReShade를 통해 실시간으로 추출하는 방식으로 구현되었다.

**데이터 품질 관리 파이프라인:**
1. Scene cut detection (PySceneDetect)으로 장면 전환 감지
2. 학습된 깊이 모델로 depth metric score 기반 필터링
3. CLIP 모델을 이용한 semantic similarity 검증 (colorized depth 채널 비교)
4. 위 두 점수가 임계값 미만인 세그먼트 제거

#### (B) 생성 모델의 Prior 활용

SVD의 사전 학습된 생성 prior는 다양한 실세계 데이터로부터 학습된 강력한 시각적 특성을 내포하고 있다. Ablation study 결과:

$$\text{Prior 없음: AbsRel}=21.0, \delta_1=65.1$$
$$\text{Prior 있음: AbsRel}=7.5, \delta_1=93.8$$

이는 단순 지도 학습 대비 약 **3배의 정확도 향상**을 보여준다.

#### (C) 혼합 지속시간 학습 전략의 일반화 기여

**Frame Dropout의 효과:**
- RoPE를 단순 적용하면 학습되지 않은 프레임 위치에 대해 일반화 실패
- 원래의 프레임 위치 인덱스 $i = [0, \cdots, T-1]$을 유지한 채로 $K$개 프레임을 랜덤 샘플링함으로써, temporal layer가 다양한 프레임 길이에 일반화

**Video Packing의 효과:**
- 학습 시간 33% 절약, GPU 메모리 활용도 40% 향상
- 다양한 길이/해상도의 비디오를 균형 있게 학습 가능

#### (D) 조건부 플로우 매칭의 역할

기존 EDM 스케줄러 → 조건부 플로우 매칭 전환으로:
- 추론 시간: $6.5\times$ 가속 (2.4s → 0.37s)
- 정확도: AbsRel 0.6 개선, $\delta_1$ 1.1 개선
- 단 1 스텝에서도 강한 결과 달성 (최적: 3스텝)

#### (E) 앙상블 전략

다양한 노이즈 초기화로 20개 예측을 앙상블하여 예측의 분산을 줄이고, 최종 성능을 안정적으로 향상시킴. 5개 이상부터 개선 폭이 감소.

### 3.2 일반화 성능의 정량적 검증

Ablation study (Table 5)에서 각 컴포넌트의 일반화 기여도:

| 구성 요소 | AbsRel↓ | δ₁↑ | 개선 폭 |
|---|---|---|---|
| 기준선 (없음) | 21.0 | 65.1 | - |
| + Generative Prior | 7.5 | 93.8 | **△13.5 / +28.7** |
| + Flow Matching | 6.9 | 94.9 | △0.6 / +1.1 |
| + Synthetic Data | 6.5 | 95.6 | △0.4 / +0.7 |
| + Mixed-duration | 6.4 | 95.8 | △0.1 / +0.2 |

특히 야외 장면에서의 정확도 향상이 두드러지며, 이는 DA-V 데이터셋의 풍부한 야외 환경 덕분임을 논문이 명시하고 있다.

---

## 4. 최신 연구 비교 분석 (2020년 이후)

### 4.1 주요 관련 연구 계보

```
2020: DiverseDepth, Consistent Video Depth Estimation (Luo et al.)
2021: MiDaS (Lasinger et al.), DPT (Ranftl et al.)
2022: HDN (Zhang et al.)
2023: Marigold (Ke et al.), ZoeDepth, Metric3D, NVDS (Wang et al.)
2024: Depth Anything V1/V2, GeoWizard, ChronoDepth, DepthCrafter, DepthFM
2024: Depth Any Video (본 논문)
```

### 4.2 방법론적 비교

| 방법 | 유형 | 기반 모델 | 비디오 | 시간적 일관성 | 데이터 전략 |
|---|---|---|---|---|---|
| **MiDaS** (2021) | 판별적 | - | ✗ | ✗ | 혼합 실데이터 |
| **DPT** (2021) | 판별적 | ViT | ✗ | ✗ | 실데이터 |
| **Marigold** (2024) | 생성적 | Stable Diffusion | ✗ | ✗ | 합성 74K |
| **DepthFM** (2024) | 생성적 | Flow Matching | ✗ | ✗ | 합성 63K |
| **GeoWizard** (2024) | 생성적 | Stable Diffusion | ✗ | ✗ | 합성 0.3M |
| **Depth Anything V2** (2024) | 판별적 | DINOv2 | ✗ | ✗ | 실데이터 63.5M |
| **NVDS** (2023) | 판별적+최적화 | - | ✓ | 부분적 | 실데이터 |
| **ChronoDepth** (2024) | 생성적 | SVD | ✓ | ✓ | 실데이터(스테레오) |
| **DepthCrafter** (2024) | 생성적 | SVD | ✓ | ✓ | 실데이터(스테레오) |
| **Depth Any Video** (본 논문) | 생성적 | SVD + Flow Matching | ✓ | ✓ | 합성 6M |

### 4.3 접근법의 차별성

**vs. Marigold (Ke et al., 2024):**
- Marigold는 이미지 기반 Stable Diffusion 활용, 25~50스텝 필요
- 본 논문은 SVD + 플로우 매칭으로 3스텝 달성, 비디오 시간적 일관성 추가

**vs. ChronoDepth (Shao et al., 2024):**
- ChronoDepth는 학습 가능한 절대 위치 임베딩 사용 → 고정 프레임 수 제한
- 본 논문은 파라미터 없는 RoPE 사용 → 가변 프레임 수 대응, 파라미터 감소

**vs. DepthCrafter (Hu et al., 2024):**
- DepthCrafter는 CLIP 임베딩 + 분류기 없는 가이던스(CFG) 사용 → 복잡도 높음
- 본 논문은 이를 제거하여 효율성 향상, 추론 시간 4.80s → 0.37s

**vs. Depth Anything V2 (Yang et al., 2024):**
- 63.5M 실데이터로 학습된 판별적 모델
- 본 논문은 합성 데이터만으로 KITTI, ETH3D에서 동등 이상의 성능 달성

---

## 5. 향후 연구에 미치는 영향과 고려 사항

### 5.1 향후 연구에 미치는 영향

#### (A) 합성 데이터 패러다임의 확립
이 논문은 **게임 엔진을 통한 대규모 합성 데이터 수집이 실세계 일반화에 실질적으로 효과적임**을 처음으로 비디오 깊이 추정 도메인에서 체계적으로 검증했다. 이는 다음 연구 방향을 열어준다:
- 합성 데이터의 다양성·규모 확장이 모델 성능의 핵심 레버로 작용
- 반사·투명 물체 등 어려운 장면의 합성 데이터 수집 방향 제시

#### (B) 생성 모델의 Prior를 활용한 Dense Prediction
Marigold에서 시작된 **Stable Diffusion의 prior를 Dense Prediction에 활용**하는 패러다임이 비디오로 확장되었다. 이 접근법은:
- 표면 법선 추정, 광학 흐름, 3D 재구성 등 다른 Dense Prediction 태스크로 확장 가능
- Video Foundation Model의 prior가 시간적 일관성 유지에 기여함을 입증

#### (C) 플로우 매칭의 실용화
확산 모델 기반 비전 태스크에서 **EDM → 플로우 매칭으로의 전환**이 효율성과 성능 모두를 향상시킴을 보임으로써, 향후 유사 태스크에서 플로우 매칭 채택을 가속화할 것으로 예상된다.

#### (D) 가변 길이 입력 처리 방법론
RoPE + Frame Dropout의 조합은 **LLM의 맥락 길이 확장 기법을 비전 도메인에 적용**한 사례로, 가변 길이 비디오 처리의 새로운 표준을 제시한다.

### 5.2 앞으로 연구 시 고려할 점

#### (A) 데이터 측면
- **합성-실제 도메인 갭(Synthetic-to-Real Gap)**: 게임 엔진 그래픽은 현실과 유사하나 완전히 일치하지 않음. 도메인 어댑테이션 기법의 추가 적용이 필요
- **어려운 장면 데이터 수집**: 반사 표면(수면, 거울), 안개, 극단적 조명 조건에 대한 합성 데이터 확장
- **저작권 및 법적 문제**: 게임 컨텐츠 활용 시 EULA 준수 필요 (본 논문에서도 명시)
- **데이터 필터링 자동화**: 현재 CLIP 기반 필터링의 한계를 극복하는 더 정교한 품질 관리

#### (B) 모델 아키텍처 측면
- **2D VAE 활용 가능성**: 실험에서 SD3의 2D VAE가 더 높은 재구성 품질을 보였으나 비디오 시간적 처리를 위해 3D VAE를 사용. 2D VAE를 활용하면서 시간적 일관성을 유지하는 방법 연구 필요
- **더 강력한 Video Foundation Model**: SVD 이후의 최신 비디오 생성 모델(예: Wan, CogVideoX 등)을 기반으로 한 개선 가능성
- **Metric Depth 확장**: 현재 Affine-invariant (scale-shift 불변) 깊이 예측만 지원; 절대적 스케일을 가진 metric depth로의 확장

#### (C) 추론 효율성 측면
- **극단적으로 긴 비디오**: 현재 150프레임 한계를 넘기 위한 계층적 키프레임 전략 연구
- **실시간 처리**: 자율주행 등 실시간 응용을 위한 추가 경량화
- **모바일/엣지 배포**: 1.4B 파라미터 모델의 경량화 (지식 증류, 양자화 등)

#### (D) 평가 방법론 측면
- **TAE 메트릭의 표준화**: 본 논문이 제안한 Temporal Alignment Error가 향후 비디오 깊이 추정의 표준 평가 지표로 자리잡을 가능성이 있으나, 카메라 파라미터를 필요로 하는 한계 존재
- **In-the-wild 평가**: 카메라 파라미터 없는 실세계 비디오 평가 프로토콜 개발 필요
- **Human Study**: 정량적 메트릭 외에 사용자 연구를 통한 지각적 품질 평가

#### (E) 응용 확장 측면
- **3D 재구성 파이프라인 통합**: 시간적으로 일관된 비디오 깊이를 활용한 동적 3D 장면 재구성
- **비디오 편집 응용**: depth-aware 비디오 편집, 배경 분리, 깊이 기반 효과 등
- **멀티태스크 학습**: 깊이 추정과 광학 흐름, 카메라 포즈 추정을 동시에 학습하는 통합 모델

---

## 참고 자료

**주요 참고 논문 (논문 내 인용 기준):**

1. **Depth Any Video with Scalable Synthetic Data** - Yang et al., arXiv:2410.10815v2 (2025) *(본 논문)*
2. **Stable Video Diffusion** - Blattmann et al., arXiv:2311.15127, 2023
3. **Marigold** - Ke et al., CVPR 2024, "Repurposing Diffusion-based Image Generators for Monocular Depth Estimation"
4. **Flow Matching for Generative Modeling** - Lipman et al., ICLR 2023
5. **RoFormer: Enhanced Transformer with Rotary Position Embedding** - Su et al., arXiv:2104.09864, 2021
6. **Depth Anything** - Yang et al., CVPR 2024
7. **Depth Anything V2** - Yang et al., arXiv:2406.09414, 2024
8. **GeoWizard** - Fu et al., arXiv:2403.12013, 2024
9. **DepthCrafter** - Hu et al., arXiv:2409.02095, 2024
10. **ChronoDepth** - Shao et al., arXiv:2406.01493, 2024
11. **NVDS: Neural Video Depth Stabilizer** - Wang et al., ICCV 2023
12. **DepthFM** - Gui et al., arXiv:2403.13788, 2024
13. **Metric3D** - Yin et al., ICCV 2023
14. **Elucidating the Design Space of Diffusion-based Generative Models (EDM)** - Karras et al., NeurIPS 2022
15. **High-Resolution Image Synthesis with Latent Diffusion Models** - Rombach et al., CVPR 2022
16. **Playing for Data: Ground Truth from Computer Games** - Richter et al., ECCV 2016
17. **DPT: Vision Transformers for Dense Prediction** - Ranftl et al., ICCV 2021
18. **Hypersim** - Roberts et al., ICCV 2021
19. **Virtual KITTI 2** - Cabon et al., arXiv:2001.10773, 2020
20. **KITTI Vision Benchmark** - Geiger et al., CVPR 2012

**프로젝트 페이지:** https://depthanyvideo.github.io

> **⚠️ 정확도 고지**: 본 답변은 제공된 논문 PDF(arXiv:2410.10815v2)의 내용을 직접 기반으로 작성되었습니다. 최신 연구 비교 분석 중 논문에 직접 인용되지 않은 2024년 이후 후속 연구(예: Wan, CogVideoX 등의 최신 모델과의 비교)는 논문 원문에 없는 내용이므로 포함하지 않았습니다.
