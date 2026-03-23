# SparseGS: Real-Time 360° Sparse View Synthesis using Gaussian Splatting

---

## 1. 핵심 주장 및 주요 기여 요약

SparseGS는 희소 학습 뷰(sparse training views) 환경에서 3DGS의 한계를 극복하기 위해 설계된 효율적인 학습 파이프라인이다. 이 방법은 **깊이 사전정보(depth priors)**, **새로운 깊이 렌더링 기법**, **가지치기 휴리스틱(pruning heuristic)**을 통합하여 floater 아티팩트를 완화하고, **Unseen Viewpoint Regularization** 모듈을 통해 배경 붕괴(background collapse)를 해소한다.

### 주요 기여 (Contributions)

첫째, 기존 알파-블렌딩 깊이를 넘어 floater를 효과적으로 관리하는 **두 가지 새로운 깊이 렌더링 기법**(softmax-scaling depth, mode-selection depth)을 제안한다.

둘째, 2D 생성적 확산 사전정보(diffusion prior)와 깊이 워핑(depth warping)을 활용하여 배경 붕괴를 해결하는 모듈을 도입한다.

셋째, 3DGS의 명시적(explicit) 표현 특성을 활용하여 불필요한 가우시안을 직접 식별·제거하는 **floater pruning 절차**를 제시한다.

실험 결과, SparseGS는 MipNeRF-360 데이터셋에서 기본 3DGS 대비 LPIPS 6.4%, PSNR 12.2% 향상을 달성하였으며, NeRF 기반 방법들 대비 LPIPS에서 최소 17.6% 이상 개선하면서 학습 및 추론 비용을 크게 절감하였다.

---

## 2. 상세 분석: 문제 정의, 제안 방법, 모델 구조, 성능, 한계

### 2.1 해결하고자 하는 문제

3D Gaussian Splatting(3DGS)는 비한정(unbounded) 3D 장면의 실시간 렌더링을 가능하게 했지만, 정확한 3D 기하 복원을 위해 조밀한(dense) 학습 뷰를 필요로 한다. 입력 뷰의 수가 제한되면 재구성 품질이 크게 저하되어 보이지 않는 시점에서 "floaters"와 "background collapse" 같은 아티팩트가 발생한다.

3DGS는 일관된 장면 표현을 생성하기 위해 풍부한 학습 뷰가 필요하며, few-shot 설정에서는 NeRF와 유사하게 학습 뷰에 과적합(overfit)되어 배경 붕괴 및 과도한 floater 현상이 발생한다.

### 2.2 제안하는 방법 (수식 포함)

SparseGS 파이프라인은 **4가지 핵심 구성요소**로 이루어져 있다:

깊이 상관 손실(depth correlation loss), 확산 손실(diffusion loss), 이미지 재투영 손실(image re-projection loss), 그리고 floater pruning 연산이다.

#### (A) 깊이 렌더링 기법 (3가지)

이 방법은 알파-블렌딩(alpha-blending), 모드 선택(mode-selection), 소프트맥스 스케일링(softmax scaling)의 세 가지 깊이 렌더링 기법을 활용하며, 모드 선택과 소프트맥스 스케일링은 본 논문에서 새롭게 제안된 기법이다.

**① Alpha-blending Depth:**

알파-블렌딩은 컬러 이미지 렌더링과 동일한 절차를 따른다. 기본 3DGS의 깊이 렌더링 공식은 다음과 같다:

$$d^{\alpha}_{x,y} = \sum_{i \in \mathcal{N}} w_i \cdot d_i, \quad w_i = \alpha_i \prod_{j=1}^{i-1}(1 - \alpha_j)$$

여기서 $d_i$는 각 가우시안의 깊이값, $w_i$는 알파 블렌딩 가중치, $\alpha_i$는 불투명도이다.

**② Softmax-scaling Depth:**

소프트맥스 함수를 가우시안 깊이 값에 적용하여 더 나은 깊이 그래디언트 제어를 달성한다. 소프트맥스 스케일링은 가중치를 다음과 같이 재정의한다:

$$d^{\text{softmax}}_{x,y} = \sum_{i \in \mathcal{N}} \tilde{w}_i \cdot d_i, \quad \tilde{w}_i = \frac{\exp(\beta \cdot w_i)}{\sum_{j \in \mathcal{N}} \exp(\beta \cdot w_j)}$$

여기서 $\beta$는 온도 파라미터(temperature parameter)로, 가중치 분포의 날카로움을 제어한다.

**③ Mode-selection Depth:**

Mode 깊이는 가장 높은 기여도를 가진 가우시안의 깊이를 선택하는 방식이다:

$$d^{\text{mode}}_{x,y} = d_{k}, \quad k = \arg\max_{i \in \mathcal{N}} w_i$$

#### (B) Depth Correlation Loss (Pearson 상관계수 기반)

Pearson 상관계수를 이미지 패치 전체에 걸쳐 계산하여 깊이 맵 간의 유사도를 측정한다. 이는 기존의 두 점만 비교하던 깊이 랭킹 손실과 달리, 전체 패치를 비교하므로 더 넓은 이미지 영역에 영향을 줄 수 있으며 더 많은 로컬 구조를 학습할 수 있다. Pearson 상관계수는 정규화된 교차상관(normalized cross-correlation)과 밀접하게 관련되어 있어, 깊이 값 범위의 차이와 무관하게 동일 위치의 패치가 높은 교차 상관을 갖도록 유도한다.

$$\mathcal{L}_{\text{depth}} = \frac{1}{|\mathcal{P}|} \sum_{p \in \mathcal{P}} \left(1 - \rho\left(p^{\text{softmax}}, p^{\text{pt}}\right)\right)$$

여기서 $\rho$는 Pearson 상관계수이다:

$$\rho(X, Y) = \frac{\text{Cov}(X, Y)}{\sigma_X \cdot \sigma_Y} = \frac{\sum_{i}(x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_{i}(x_i - \bar{x})^2} \cdot \sqrt{\sum_{i}(y_i - \bar{y})^2}}$$

$p^{\text{softmax}}$는 소프트맥스 렌더링 깊이 패치, $p^{\text{pt}}$는 단안 깊이 추정(monocular depth, 예: DPT)으로부터의 깊이 패치이다.

#### (C) Unseen Viewpoint Regularization (확산 모델 기반 SDS 손실)

입력 뷰의 커버리지가 부족한 영역에서는 Score Distillation Sampling(SDS)과 Depth Warping을 활용하여 기하학적 붕괴와 텍스처 노이즈를 줄이면서 세밀한 디테일을 보존한다.

SDS 손실은 사전학습된 2D diffusion 모델 $\epsilon_\phi$를 활용하여:

$$\nabla_\theta \mathcal{L}_{\text{SDS}} = \mathbb{E}_{t, \epsilon}\left[w(t)\left(\epsilon_\phi(z_t; y, t) - \epsilon\right) \frac{\partial z}{\partial \theta}\right]$$

여기서 $z_t$는 렌더링 이미지에 노이즈를 추가한 잠재 벡터, $y$는 텍스트 프롬프트, $t$는 타임스텝, $w(t)$는 가중치 함수이다.

#### (D) Floater Pruning

"Floaters"를 제거하기 위해 3D 가우시안의 명시적 표현을 활용하는 새로운 연산을 제안한다. Floater는 카메라 평면에 가까이 위치한 낮은 불투명도의 가우시안으로 나타나며, softmax 깊이에서는 "평균화"되어 두드러지지 않지만 mode-selection 깊이에서는 두드러진다. 이 차이를 활용하여 각 학습 뷰에 대한 floater mask $F$를 생성하고, mode 가우시안까지 포함한 모든 가우시안을 선택적으로 가지치기한다.

구체적으로, SparseGS 접근법은 알파 블렌딩과 모드 선택 방법을 사용하여 깊이 불일치를 계산한 다음, dip test 기반의 이변량 탐지(bivariate detection)를 수행하여 깊이 차이의 이변량 점수를 평가한다. 이 점수를 기반으로 적응적 가지치기 임계값을 설정하고, 부정확한 가우시안 포인트를 제거하는 마스크를 생성한다.

### 2.3 전체 손실 함수

전체 학습 손실은 다음과 같이 구성된다:

```math
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{recon}} + \lambda_{\text{pearson}} \mathcal{L}_{\text{depth}} + \lambda_{\text{local}} \mathcal{L}_{\text{local\_depth}} + \lambda_{\text{diffusion}} \mathcal{L}_{\text{SDS}} + \lambda_{\text{reg}} \mathcal{L}_{\text{reg}}
```

여기서:
- $\mathcal{L}_{\text{recon}}$: 기본 3DGS 재구성 손실 (L1 + D-SSIM)
- $\mathcal{L}_{\text{depth}}$: 글로벌 Pearson 상관 깊이 손실
- $\mathcal{L}_{\text{local depth}}$: 로컬 패치 기반 Pearson 깊이 손실
- $\mathcal{L}_{\text{SDS}}$: Score Distillation Sampling 손실
- $\mathcal{L}_{\text{reg}}$: 정규화 항

### 2.4 모델 구조 요약

```
┌─────────────────────────────────────────────┐
│           SparseGS Pipeline                 │
├─────────────────────────────────────────────┤
│  Input: Sparse Views (12장 for 360°,        │
│         3장 for forward-facing)             │
│                                             │
│  ① Base 3DGS Initialization (SfM→COLMAP)   │
│  ② Depth Rendering Module                  │
│     ├─ Alpha-blending depth                 │
│     ├─ Softmax-scaling depth (NEW)          │
│     └─ Mode-selection depth (NEW)           │
│  ③ Depth Correlation Loss                  │
│     └─ Pearson correlation with DPT depth   │
│  ④ Unseen Viewpoint Regularization         │
│     ├─ SDS (Score Distillation Sampling)    │
│     └─ Depth Warping                        │
│  ⑤ Floater Pruning (at preset intervals)   │
│     └─ Adaptive threshold via dip-test      │
│                                             │
│  Output: Coherent 3D Gaussian Scene         │
│          → Real-time Rendering              │
└─────────────────────────────────────────────┘
```

### 2.5 성능 향상

Mip-NeRF360, LLFF, DTU 데이터셋에 대한 광범위한 평가에서 SparseGS는 비한정(unbounded) 및 전방향(forward-facing) 시나리오 모두에서 고품질 재구성을 달성하며, 각각 최소 12장과 3장의 입력 이미지만으로도 높은 성능을 유지한다.

| 비교 대상 | 지표 | 개선폭 |
|-----------|------|--------|
| 기본 3DGS | LPIPS | **6.4%** 향상 |
| 기본 3DGS | PSNR | **12.2%** 향상 |
| NeRF 기반 방법들 | LPIPS | **≥ 17.6%** 향상 |

SparseGS는 희소 입력 설정에서 새로운 뷰 합성 품질을 크게 향상시키면서도 빠른 학습과 실시간 렌더링을 유지한다.

### 2.6 한계

SparseGS는 DTU 데이터셋에서 전방향(frontal) 장면에 특화된 이전 방법들을 초월하지는 못하지만, 기본 3DGS 방법을 크게 개선하여 높은 시각적 충실도와 경쟁력 있는 메트릭을 달성한다.

재구성이 불충분한 영역에서 NeRF 기반 방법은 과도하게 부드러운 외관을 생성하는 반면, 3DGS 기반 표현은 고립된 가우시안을 배치하여 고주파 아티팩트로 나타난다. PSNR은 과도하게 부드러운 이미지를 허용하고 날카로운 아티팩트를 크게 벌점하는 반면, LPIPS 같은 지각적 메트릭은 그 반대를 강조한다.

추가적 한계점:
- **확산 모델 의존성**: SDS 손실은 사전학습된 diffusion 모델에 의존하므로 해당 모델의 도메인 편향에 영향을 받을 수 있음
- **단안 깊이 추정의 노이즈**: 이러한 방법들의 정밀도는 단안 깊이 추정의 정확도에 달려 있으며, 깊이 맵의 정확도가 부족하면 재구성 품질이 저하된다. 이는 단안 깊이 추정에서 정확성과 안정성을 모두 달성하는 것의 중요성을 강조한다.
- **하이퍼파라미터 민감도**: 다수의 손실 가중치($\lambda_{\text{pearson}}, \lambda_{\text{local}}, \lambda_{\text{diffusion}}$, 프루닝 스케줄 등)에 대한 세밀한 조정이 필요함

---

## 3. 모델의 일반화 성능 향상 가능성

SparseGS의 일반화 성능과 관련하여 다음과 같은 핵심 사항들이 있다:

### 3.1 일반화 강점

SparseGS 파이프라인은 전방향 데이터셋뿐만 아니라 360도 비한정 장면에서도 SOTA 성능을 달성하여, 대부분의 기존 few-shot 기법들이 효과적으로 처리하지 못하는 시나리오를 처리할 수 있다.

LLFF와 DTU라는 전방향 데이터셋에서도 파이프라인의 견고성을 입증하는 평가를 제공하며, LLFF 데이터셋은 8개의 복잡한 전방향 실제 장면으로 구성되고 DTU 데이터셋은 전경 마스크가 있는 객체 중심 장면을 포함한다.

### 3.2 일반화 향상을 위한 핵심 메커니즘

**① Scale-invariant 깊이 정규화:**
Pearson 상관계수를 사용한 깊이 상관 손실은 절대 깊이 스케일에 민감하지 않아 다양한 장면에서 일반화가 가능:

$$\rho(X, Y) \text{는 선형 변환에 불변 → scale ambiguity 문제 자연 해결}$$

**② Off-the-shelf 사전 모델 활용:**
기성 깊이 추정 모델(off-the-shelf depth estimation models)을 활용하여 새로운 뷰 출력을 정규화하고, 확산 모델 가이던스로 학습 뷰의 낮은 커버리지 영역을 재구성한다. 이는 장면 특정 학습 없이도 일반적인 깊이 사전정보를 제공한다.

**③ 적응적 Pruning:**
Dip-test 기반의 적응적 임계값은 장면 특성에 맞게 자동 조정되어 다양한 장면에 대한 일반화를 지원한다.

### 3.3 일반화 성능의 제약 및 개선 방향

깊이 추정 네트워크는 보이지 않는 장면에서 스케일 불일치로 깊이를 예측하므로, 도메인 이동(domain shift)에 민감하다.

| 일반화 제약 | 개선 방향 |
|------------|----------|
| 단안 깊이 모델 의존성 | Foundation model (예: Depth Anything V2) 통합 |
| 확산 모델 도메인 편향 | 장면 적응적 LoRA fine-tuning |
| SfM 초기화 품질 의존 | MVS 기반 dense initialization (MASt3R 등) |
| 하이퍼파라미터 민감도 | 자동 하이퍼파라미터 탐색 도입 |

---

## 4. 연구 영향 및 향후 연구 고려사항

### 4.1 이 논문이 후속 연구에 미친 영향

SparseGS는 3DGS 기반 sparse view synthesis의 초기 연구 중 하나로서, 이후의 연구 방향에 중요한 영향을 미쳤다:

1. **깊이 정규화 패러다임 확립**: 3DGS 기반의 few-shot novel view synthesis 방법들은 거의 모두 사전학습된 깊이 추정 네트워크를 파이프라인에 통합하려는 경향을 보인다. SparseGS의 Pearson 상관계수 기반 접근은 후속 연구의 표준이 되었다.

2. **명시적 표현의 장점 활용**: 명시적 표현의 특성을 활용한 직접적 floater pruning이라는 핵심 연산을 도입하여, 렌더링된 이미지에서 floater나 배경 붕괴로 인한 문제 영역을 식별하고 3D 표현을 직접 편집하여 아티팩트를 제거할 수 있게 하였다.

3. **360° unbounded 희소 뷰 합성**: 이 연산이 일반적인 뷰 합성 메트릭에서 상당한 이점을 제공하며, 대부분의 기존 few-shot 기법이 처리하지 못하는 전체 360° 비한정 장면에서도 작동할 수 있음을 보여주었다.

### 4.2 향후 연구 시 고려할 점

1. **보다 강건한 깊이 사전정보**: 최신 foundation model (예: Depth Anything V2, UniDepth 등)의 통합으로 도메인 일반화 성능 향상 가능
2. **SfM-free 접근**: InstantSplat처럼 SfM(Structure-from-Motion) 없이 feed-forward 모델의 풍부한 기하학적 사전정보와 point-based 표현을 결합하는 접근이 유망
3. **다중 스케일 깊이 정렬**: HDGS에서 제안된 것처럼 coarse-to-fine 수준에서 점진적으로 기하학을 정제하는 깊이 감독 프레임워크 도입 가능
4. **장면 적응적 확산 모델**: GS-GS처럼 기본 3D/4D 가우시안 스플래팅 모델과 사전학습된 확산 모델에 삽입된 LoRA 모듈을 공동 최적화하여 pseudo-view 이미지를 반복적으로 생성하는 전략이 효과적

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

| 방법 | 연도 | 핵심 접근 | 표현 | 장점 | 한계 |
|------|------|----------|------|------|------|
| **RegNeRF** | 2022 | 정규화 기반 NeRF | 암묵적 | 보이지 않는 뷰에 정규화 | 느린 학습·추론 |
| **FreeNeRF** | 2023 | 주파수 정규화 | 암묵적 | 추가 입력 불필요 | 실시간 렌더링 불가 |
| **SparseNeRF** | 2023 | 공간적 연속성 손실 | 암묵적 | 단안 깊이 활용 | 전방향 장면에 제한 |
| **3DGS** (Kerbl et al.) | 2023 | Gaussian Splatting | 명시적 | 실시간 렌더링 | 희소 뷰에서 과적합 |
| **SparseGS** | 2023 | 깊이 정규화 + SDS + Pruning | 명시적 | 360° 실시간, floater 제거 | 다수 하이퍼파라미터 |
| **FSGS** | 2023 | Gaussian Unpooling + 뷰 증강 | 명시적 | 추론 속도가 기존 SOTA 방법 대비 2000배 이상 빠름 | Dense stereo 전처리 필요 |
| **DNGaussian** | 2024 | Global-Local Depth Normalization | 명시적 | 메모리 비용 대폭 절감, 학습 시간 25배 단축, 렌더링 속도 3000배 이상 향상 | 초기화되지 않은 영역 처리 불가 |
| **InstantSplat** | 2024 | 매우 희소한 뷰에서 몇 초 만에 강건한 최적화를 완료하는 사실적 3D 재구성 방법으로, feed-forward 모델의 풍부한 기하학적 사전정보와 point-based 표현을 결합 | 명시적 | SfM 불필요 | MVS 모델 의존성 |
| **FewViewGS** | 2024 | 깊이 추정이나 확산 모델에 의존하지 않고 매칭 기반 일관성 제약과 다단계 학습을 활용 | 명시적 | 외부 사전모델 불필요 | 3-view 이하 극한 상황 도전적 |
| **GS-GS** (CVPR 2025) | 2025 | 정적 및 동적 장면 재구성을 위한 일반 파이프라인으로 3장의 학습 뷰만으로 높은 충실도 품질 달성 | 명시적 | 정적·동적 장면 모두 지원 | LoRA fine-tuning 오버헤드 |
| **D2GS** | 2025 | 깊이 및 밀도 기반 안내 | 명시적 | LLFF와 MipNeRF360에서 일관되게 최고 성능 달성 | 학습 불안정성 보고 |

### 핵심 트렌드 분석

1. **NeRF → 3DGS 패러다임 전환**: NeRF 프레임워크 기반의 주류 희소 뷰 3D 재구성 알고리즘은 생성 품질과 실시간 성능의 균형을 맞추기 어려웠으나, 최근 3D Gaussian Splatting 기술의 등장으로 뛰어난 결과를 보여주고 있다.

2. **깊이 사전정보의 보편화**: 현재 3DGS 기반 sparse view synthesis 방법(FSGS, SparseGS, DNGaussian, CoherentGS 등)들은 사전학습된 모델에서 얻은 깊이 사전정보를 감독 신호로 활용한다. SparseGS가 이 접근의 선구적 역할을 하였다.

3. **SfM-free 방향**: InstantSplat, MASt3R 기반 방법들이 SfM 의존성을 제거하는 추세

4. **Cascade/Hierarchical 깊이 감독**: HDGS는 coarse에서 fine 수준으로 점진적으로 기하학을 정제하는 계층적 깊이 감독 프레임워크를 도입하여 SparseGS의 Pearson 상관 기반 접근을 확장

---

## 참고 자료 (References)

1. **Xiong, H., Muttukuru, S., Upadhyay, R., Chari, P., Kadambi, A.** "SparseGS: Real-Time 360° Sparse View Synthesis using Gaussian Splatting." *arXiv:2312.00206* (2023), accepted at *3DV 2025*. — [https://arxiv.org/abs/2312.00206](https://arxiv.org/abs/2312.00206)

2. **SparseGS Project Page** — [https://formycat.github.io/SparseGS-Real-Time-360-Sparse-View-Synthesis-using-Gaussian-Splatting/](https://formycat.github.io/SparseGS-Real-Time-360-Sparse-View-Synthesis-using-Gaussian-Splatting/)

3. **SparseGS GitHub Repository** — [https://github.com/ForMyCat/SparseGS](https://github.com/ForMyCat/SparseGS)

4. **eScholarship (UC)** "SparseGS: Real-Time 360° Sparse View Synthesis using Gaussian Splatting" — [https://escholarship.org/uc/item/52z2695z](https://escholarship.org/uc/item/52z2695z)

5. **Springer Nature** "A review on 3D Gaussian splatting for sparse view reconstruction." *Artificial Intelligence Review* (2025) — [https://link.springer.com/article/10.1007/s10462-025-11171-4](https://link.springer.com/article/10.1007/s10462-025-11171-4)

6. **Zhu, Z. et al.** "FSGS: Real-Time Few-shot View Synthesis using Gaussian Splatting." *ECCV 2024*, arXiv:2312.00451 — [https://arxiv.org/html/2312.00451v2](https://arxiv.org/html/2312.00451v2)

7. **Li, J. et al.** "DNGaussian: Optimizing Sparse-View 3D Gaussian Radiance Fields with Global-Local Depth Normalization." *CVPR 2024*, arXiv:2403.06912 — [https://github.com/Fictionarry/DNGaussian](https://github.com/Fictionarry/DNGaussian)

8. **Fan, Z. et al.** "InstantSplat: Sparse-view Gaussian Splatting in Seconds." (2024) — [https://instantsplat.github.io/](https://instantsplat.github.io/)

9. **Kong, H., Yang, X.** "Generative Sparse-View Gaussian Splatting." *CVPR 2025* — [https://openaccess.thecvf.com/content/CVPR2025/papers/Kong_Generative_Sparse-View_Gaussian_Splatting_CVPR_2025_paper.pdf](https://openaccess.thecvf.com/content/CVPR2025/papers/Kong_Generative_Sparse-View_Gaussian_Splatting_CVPR_2025_paper.pdf)

10. **FewViewGS** "Gaussian Splatting with Few View Matching and Multi-stage Training." *NeurIPS 2024* — [https://proceedings.neurips.cc/paper_files/paper/2024/file/e6209a02394269f14c29e049f4e05c42-Paper-Conference.pdf](https://proceedings.neurips.cc/paper_files/paper/2024/file/e6209a02394269f14c29e049f4e05c42-Paper-Conference.pdf)

11. **HDGS** "Learning Fine-Grained Geometry for Sparse-View Splatting via Cascade Depth Loss." arXiv:2505.22279 (2025) — [https://arxiv.org/html/2505.22279](https://arxiv.org/html/2505.22279)

12. **D2GS** "Depth-and-Density Guided Gaussian Splatting for Stable and Accurate Sparse-View Reconstruction." arXiv:2510.08566 (2025) — [https://arxiv.org/html/2510.08566v1](https://arxiv.org/html/2510.08566v1)

13. **Semantic Scholar — SparseGS** — [https://www.semanticscholar.org/paper/SparseGS](https://www.semanticscholar.org/paper/SparseGS:-Real-Time-360%7B%5Cdeg%7D-Sparse-View-Synthesis-Xiong-Muttukuru/4c7cf6d5f4de22d0dfcce35f77a6b5d21b2c01f0)

14. **Springer Nature** "Recent advances in 3D Gaussian splatting." *Computational Visual Media* (2024) — [https://link.springer.com/article/10.1007/s41095-024-0436-y](https://link.springer.com/article/10.1007/s41095-024-0436-y)

> **주의**: 본 분석에서 제시한 수식 중 전체 손실 함수($\mathcal{L}_{\text{total}}$)의 정확한 가중치 조합은 논문의 공식 코드 repository의 학습 인자(arguments)를 참고하여 재구성한 것이며, 논문의 표기와 미세한 차이가 있을 수 있습니다.
