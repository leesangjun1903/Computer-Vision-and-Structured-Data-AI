# Marigold: Repurposing Diffusion-Based Image Generators for Monocular Depth Estimation

**논문 정보**: Bingxin Ke, Anton Obukhov, Shengyu Huang, Nando Metzger, Rodrigo Caye Daudt, Konrad Schindler — CVPR 2024 (Oral, Best Paper Award Candidate)

---

## 1. 핵심 주장 및 주요 기여 요약

단안(Monocular) 깊이 추정은 컴퓨터 비전의 기본 과제이며, 단일 이미지로부터 3D 깊이를 복원하는 것은 기하학적으로 ill-posed 문제로 장면 이해를 요구한다. 기존 단안 깊이 추정기들은 훈련 데이터에 제한되어 새로운 도메인에 대한 zero-shot 일반화에 어려움을 겪고 있으며, 이에 따라 최근의 생성적 확산 모델에 내재된 광범위한 사전 지식(prior)이 더 나은 일반화 성능을 가능하게 할 수 있는지를 탐구한다.

**핵심 기여**:

1. Stable Diffusion에서 파생되어 풍부한 사전 지식을 보존하는 affine-invariant 단안 깊이 추정 방법인 Marigold를 제안한다.
2. 단일 GPU에서 며칠 내에 합성 데이터만으로 fine-tuning할 수 있다.
3. 다양한 데이터셋에서 SOTA 성능을 달성하며, 특정 경우 20% 이상의 성능 향상을 보인다.

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

단안 깊이 추정은 사진 이미지를 깊이 맵으로 변환하는 것, 즉 모든 픽셀에 대한 거리 값을 회귀하는 것을 목표로 하며, 3D 세계에서 2D 이미지로의 투영을 되돌리는 것은 기하학적으로 ill-posed 문제로, 전형적인 물체 형상과 크기, 가능한 장면 레이아웃, 가림(occlusion) 패턴 등의 사전 지식 없이는 해결할 수 없다.

기존 깊이 추정기의 발전은 비교적 작은 CNN부터 대규모 Transformer 아키텍처까지 모델 용량의 성장을 따라왔으나, 훈련 데이터에 의해 시각 세계에 대한 지식이 제한되어 낯선 콘텐츠와 레이아웃의 이미지에서 어려움을 겪는다.

### 2.2 제안하는 방법 (수식 포함)

Marigold는 단안 깊이 추정을 **조건부 디노이징 확산 생성(conditional denoising diffusion generation)** 과제로 정의한다.

#### (a) 잠재 공간 인코딩

사전 훈련된 Stable Diffusion의 VAE를 사용하여 이미지 $x$와 깊이 맵 $d$를 잠재 공간으로 인코딩하고, U-Net만을 fine-tuning하며, 표준 확산 목적 함수(standard diffusion objective)를 깊이 잠재 코드에 대해 최적화한다. 이미지 조건화는 두 잠재 코드를 연결(concatenation)하여 U-Net에 입력함으로써 달성된다.

구체적으로, 이미지 $\mathbf{x} \in \mathbb{R}^{W \times H \times 3}$와 깊이 맵 $\mathbf{d} \in \mathbb{R}^{W \times H}$가 주어지면:

$$\mathbf{z}^{(x)} = \mathcal{E}(\mathbf{x}), \quad \mathbf{z}^{(d)}_0 = \mathcal{E}(\mathbf{d})$$

여기서 $\mathcal{E}$는 Stable Diffusion의 사전 훈련된 VAE 인코더이다.

#### (b) Forward Diffusion Process

깊이 잠재 코드에 점진적으로 가우시안 노이즈를 추가한다:

$$\mathbf{z}^{(d)}_t = \sqrt{\bar{\alpha}_t} \, \mathbf{z}^{(d)}_0 + \sqrt{1 - \bar{\alpha}_t} \, \boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$

#### (c) 훈련 목적 함수 (Training Loss)

U-Net만을 fine-tuning하며 표준 확산 목적 함수를 깊이 잠재 코드에 대해 최적화한다. 이미지 조건화는 두 잠재 코드를 연결하여 U-Net에 입력함으로써 달성된다.

$$\mathcal{L} = \mathbb{E}_{\mathbf{z}^{(d)}_0,\, \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0},\mathbf{I}),\, t \sim \mathcal{U}(T)} \left\| \boldsymbol{\epsilon} - \hat{\boldsymbol{\epsilon}}_\theta\!\left(\mathbf{z}^{(d)}_t,\, \mathbf{z}^{(x)},\, t\right) \right\|^2_2$$

여기서 $\hat{\boldsymbol{\epsilon}}_\theta$는 U-Net 디노이저, $t$는 타임스텝, $T$는 전체 확산 스케줄 길이이다.

#### (d) 추론 (Inference)

입력 이미지 $x$가 주어지면, Stable Diffusion VAE로 잠재 코드 $z^{(x)}$를 인코딩하고, 매 디노이징 반복마다 깊이 잠재 $z^{(d)}_t$와 연결하여 수정된 fine-tuned U-Net에 입력한다. $T$ 단계의 스케줄을 실행한 후, 결과 깊이 잠재 $z^{(d)}_0$를 이미지로 디코딩하고, 3개 채널을 평균하여 최종 추정 $\hat{d}$를 얻는다.

$$\hat{\mathbf{d}} = \frac{1}{3}\sum_{c=1}^{3} \mathcal{D}\!\left(\mathbf{z}^{(d)}_0\right)_c$$

여기서 $\mathcal{D}$는 VAE 디코더이다.

### 2.3 모델 구조

| 구성 요소 | 세부 사항 |
|---|---|
| **Base Model** | 사전 훈련된 text-to-image LDM인 Stable Diffusion v2 기반 |
| **VAE** | 동결된(frozen) VAE를 사용하여 이미지와 해당 깊이 맵을 잠재 공간으로 인코딩 |
| **U-Net** | U-Net만 fine-tuning하며, 첫 번째 레이어를 수정하여 연결된 잠재 코드를 입력 받도록 변경 |
| **조건화 방식** | 텍스트 프롬프트 대신 이미지 잠재 코드와 깊이 잠재 코드의 채널 연결(concatenation) |
| **처리 해상도** | Stable Diffusion이 유도된 해상도인 768×768에서 최적 성능 |

### 2.4 성능 향상

이 방법은 실제 깊이 샘플을 한 번도 보지 않고도 실내·실외 장면 모두에서 대부분의 경우 다른 방법들을 능가한다.

Marigold는 벽과 가구 사이의 공간 관계 같은 장면 레이아웃을 정확히 포착할 뿐만 아니라 세밀한 디테일(의자 다리 등 얇은 구조)도 잘 포착하며, 평면(벽)의 재구성이 현저히 개선되었다. 또한 일반적인 형상과 레이아웃을 효과적으로 모델링하여 생성적 사전 지식에 대한 기대와 부합한다.

### 2.5 한계

1. **추론 속도**: 대규모 확산 모델을 재활용한 고정밀 단안 깊이 추정기로서 SOTA를 달성했으나, 다단계 추론에 따른 높은 계산 요구량이 많은 시나리오에서의 사용을 제한했다.
2. **해상도 편향**: Stable Diffusion에서 fine-tune된 모델은 원본 해상도에 대한 해상도 편향을 보이며, 처리 해상도인 768로 다운샘플링/업샘플링 시 고해상도 이미지에서 세부 정보가 크게 손실된다.
3. **합성-실제 도메인 갭**: 합성 데이터와 실제 데이터 간의 도메인 갭이 때때로 일반화 능력을 제한할 수 있다는 우려가 남아 있다.
4. **확률적 출력**: Marigold가 생성 모델이기 때문에, 확산 과정을 시작하는 초기 노이즈에 따라 예측이 달라진다.

---

## 3. 모델의 일반화 성능 향상 가능성 (핵심 분석)

Marigold의 **일반화 능력**은 이 논문의 가장 중요한 기여이다:

### 3.1 생성적 사전 지식의 활용

생성적 확산 모델에 포착된 광범위한 사전 지식(prior)이 더 나은, 더 일반화 가능한 깊이 추정을 가능하게 할 수 있다는 가설이 Marigold의 핵심 동기이다.

사전 훈련된 확산 모델의 잠재력을 해제하는 핵심은 그 사전 지식을 보존하는 것이다. Stable Diffusion은 수십억 장의 이미지-텍스트 쌍에서 학습되어 다양한 장면, 물체, 레이아웃에 대한 풍부한 시각적 이해를 내재하고 있다.

### 3.2 합성 데이터만으로의 Zero-Shot Transfer

Marigold는 단 ~74K개의 합성 깊이 샘플만으로 실제 이미지 데이터셋에서 SOTA 깊이 추정을 달성한다. 합성 데이터 사용의 장점:

- 합성 깊이는 본질적으로 조밀하고 완전하여 모든 픽셀에 유효한 GT 깊이 값이 있고, 렌더링 파이프라인에 의해 보장되는 가장 깨끗한 형태의 깊이이므로 짧은 fine-tuning 프로토콜에서 그래디언트 업데이트의 노이즈를 줄인다.

### 3.3 일반화 성능 향상을 위한 핵심 설계 원칙

| 원칙 | 설명 |
|---|---|
| **최소한의 아키텍처 수정** | 사전 훈련된 LDM 아키텍처의 최소한의 수정만 필요 |
| **짧은 Fine-tuning** | 과적합(overfitting)을 방지하여 사전 지식 보존 |
| **Affine-Invariant 예측** | 스케일/시프트에 불변인 상대 깊이 예측으로 도메인 간 전이 용이 |
| **앙상블 추론** | 다중 추론의 픽셀별 통계를 통해 예측 일관성 향상; multi-resolution noise 훈련이 예측 일관성을 증가시킨다. |

### 3.4 일반화에 대한 시사점

"Repurposing"이라는 용어는 확산 모델의 기본 원리가 다양한 과제에 적용될 수 있음을 강조하며, 하나의 모델 발전이 다른 분야의 돌파구를 열 수 있음을 시사한다.

---

## 4. 향후 연구에 미치는 영향 및 고려사항

### 4.1 연구 영향

1. **Foundation Model → Dense Prediction 패러다임 확립**: Marigold는 깊이 추정에서 시작하여 표면 법선 예측, 내재적 분해(intrinsic decomposition) 등 밀집 이미지 분석 과제로 확장되었다.
2. **후속 연구 촉발**:
   - Marigold-DC는 희소 깊이 완성(sparse depth completion)을 조건부 깊이 추정으로 재정의
   - Rolling Depth (CVPR 2025)는 비디오 깊이 추정에서 우수한 시간적 일관성을 달성
   - Better Depth (NeurIPS 2024)는 확산 모델을 통한 거친 예측의 정제를 시연

3. **효율성 개선 연구 촉진**: 추론 파이프라인의 결함을 수정하면 200배 이상 빠르게 동작하면서도 비슷한 성능을 달성할 수 있으며, 단일 스텝 모델에 대한 end-to-end fine-tuning으로 모든 확산 기반 깊이/법선 추정 모델을 능가하는 결정론적 모델을 얻을 수 있다.

### 4.2 향후 연구 시 고려할 점

| 고려 사항 | 세부 내용 |
|---|---|
| **추론 효율성** | 다단계 확산 과정의 단계 수 감소 (LCM distillation, flow matching 등) |
| **Metric Depth** | Affine-invariant에서 절대적 metric depth로의 확장 |
| **고해상도 처리** | 768p 해상도 편향을 극복하기 위한 tile 기반 고해상도 추론 전략 |
| **시간적 일관성** | 비디오 입력에 대한 프레임 간 깊이 일관성 보장 |
| **도메인 특화** | 의료, 위성, 수중 등 특수 도메인으로의 확장 가능성 |

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

| 모델/논문 | 연도 | 접근 방식 | 핵심 특징 | Marigold 대비 차이점 |
|---|---|---|---|---|
| **MiDaS v3.1** | 2020/2022 | Discriminative (Transformer) | MiDaS는 최대 12개 데이터셋에서 multi-objective optimization으로 학습하여 zero-shot cross-dataset transfer를 목표로 한다. | 대규모 real 데이터 혼합 필요; 생성적 prior 미활용 |
| **ZoeDepth** | 2023 | Discriminative + Metric binning | MiDAS 백본과 적응적 metric binning 모듈을 통합하여 정밀한 절대 깊이 추정을 가능하게 한 zero-shot metric depth 추정의 주요 진전. | Metric depth 지원; 그러나 ZoeDepth는 야외 자연 환경에서 성능 저하가 심각(MAE: 3.087m). |
| **Depth Anything V1** | 2024 (CVPR) | Semi-supervised DPT | 어떤 상황의 이미지도 처리하는 강력한 foundation 모델을 구축하고, ~62M 비라벨 데이터를 자동 레이블링하여 일반화 오차를 크게 줄인다. | 대규모 데이터 스케일링; 확산 모델 미사용; 빠른 추론 |
| **Depth Anything V2** | 2024 (NeurIPS) | Teacher-Student + Synthetic | 합성 이미지로 라벨된 실제 이미지를 대체하고, teacher 모델 용량 확대, 대규모 pseudo-labeled 실제 이미지로 student 훈련. Stable Diffusion 기반 모델보다 10배 이상 빠르고 더 정확하다. | 효율성 극대화; Marigold 대비 추론 속도 10x 이상 |
| **DepthFM** | 2024 (AAAI) | Flow Matching | 이미지 분포에서 깊이 분포로의 직접 매핑을 학습하는 flow matching 접근법으로, 확산 기반보다 현저히 효율적이면서도 세밀한 깊이 맵을 제공한다. | 단일 추론 단계로 깊이 맵 합성 가능 |
| **ML Depth Pro** (Apple) | 2024 | Discriminative (Foundation) | 카메라 메타데이터 없이 고도로 상세하고 metric하게 정확한 깊이 맵을 초 이하의 시간에 생성 | 산업용 실시간 적용에 초점 |
| **Metric3D v2** | 2024 | Geometric Foundation | zero-shot metric depth와 표면 법선 추정을 위한 다목적 단안 기하 foundation 모델 | Metric depth 직접 지원 |
| **E2E Fine-tuning** | 2024 | Single-step Diffusion | 수정된 모델이 이전 최고 성능과 비슷하면서 200배 이상 빠르며, task-specific loss로 end-to-end fine-tuning하여 모든 확산 기반 모델을 능가 | Marigold의 효율성 한계를 직접 해결 |

### 핵심 트렌드 분석

1. **생성적 모델 vs. 판별적 모델**: Marigold가 개척한 "생성적 prior 활용" 패러다임은 강력한 일반화를 입증했으나, Depth Anything V2 등 판별적 접근이 10배 이상 빠르고 더 정확한 결과를 보이면서 두 패러다임 간 경쟁이 활발하다.

2. **합성 데이터의 중요성**: Marigold, DepthFM, Depth Anything V2 모두 합성 데이터 훈련의 효과를 입증하여, 고품질 합성 데이터가 일반화의 핵심 요소임이 확인되었다.

3. **효율성 추구**: 확산 기반 모델의 느린 추론 속도가 주요 병목이며, Flow Matching, LCM distillation, single-step 추론 등의 해결책이 활발히 연구되고 있다.

---

### 참고자료 및 출처

1. Ke, B. et al., *"Repurposing Diffusion-Based Image Generators for Monocular Depth Estimation,"* CVPR 2024. — [arXiv:2312.02145](https://arxiv.org/abs/2312.02145)
2. Ke, B. et al., *"Marigold: Affordable Adaptation of Diffusion-Based Image Generators for Image Analysis,"* IEEE TPAMI / arXiv:2505.09358, 2025.
3. Marigold 공식 프로젝트 페이지 — [marigoldmonodepth.github.io](https://marigoldmonodepth.github.io/)
4. GitHub 공식 레포지토리 — [github.com/prs-eth/Marigold](https://github.com/prs-eth/Marigold)
5. HuggingFace 모델 카드 — [prs-eth/marigold-depth-v1-0](https://huggingface.co/prs-eth/marigold-depth-v1-0)
6. CVPR Open Access PDF — [thecvf.com](https://openaccess.thecvf.com/content/CVPR2024/papers/Ke_Repurposing_Diffusion-Based_Image_Generators_for_Monocular_Depth_Estimation_CVPR_2024_paper.pdf)
7. Ranftl, R. et al., *"Towards Robust Monocular Depth Estimation: Mixing Datasets for Zero-shot Cross-dataset Transfer,"* TPAMI 2022. (MiDaS)
8. Yang, L. et al., *"Depth Anything: Unleashing the Power of Large-Scale Unlabeled Data,"* CVPR 2024.
9. Yang, L. et al., *"Depth Anything V2,"* NeurIPS 2024.
10. Gui, M. et al., *"DepthFM: Fast Monocular Depth Estimation with Flow Matching,"* AAAI 2024.
11. Garcia, G. et al., *"Fine-Tuning Image-Conditional Diffusion Models is Easier than You Think,"* 2024. — [diffusion-e2e-ft](https://gonzalomartingarcia.github.io/diffusion-e2e-ft/)
12. Bochkovskii, A. et al., *"Depth Pro: Sharp Monocular Metric Depth in Less Than a Second,"* arXiv 2024. (Apple ML Depth Pro)
13. Viola, M. et al., *"Marigold-DC: Zero-Shot Monocular Depth Completion with Guided Diffusion,"* arXiv:2412.13389, 2024.
14. Datumo Blog, *"The Most Refined Depth Estimation Model: Marigold"*, Dec 2024 — [datumo.com](https://datumo.com/blog/tech/depth-estimation-marigold/)
15. *"Survey on Monocular Metric Depth Estimation,"* arXiv:2501.11841, 2025.
