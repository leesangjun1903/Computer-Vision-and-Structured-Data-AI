
# Taming Video Diffusion Prior with Scene-Grounding Guidance for 3D Gaussian Splatting from Sparse Inputs

> **출처 및 참고자료**
> - arXiv:2503.05082 (https://arxiv.org/abs/2503.05082)
> - CVPR 2025 Highlight Paper (https://cvpr.thecvf.com/virtual/2025/poster/34124)
> - IEEE Xplore (https://ieeexplore.ieee.org/document/11094512)
> - 프로젝트 페이지 (https://zhongyingji.github.io/guidevd-3dgs/)
> - GitHub (https://github.com/zhongyingji/guidedvd-3dgs)
> - HKUST Research Portal (https://researchportal.hkust.edu.hk/en/publications/taming-video-diffusion-prior-with-scene-grounding-guidance-for-3d/)
> - The Moonlight Literature Review (https://www.themoonlight.io/en/review/...)
> - Sparse-View 3D Reconstruction Survey arXiv:2507.16406 (https://arxiv.org/html/2507.16406v1)
> - DNGaussian Project Page (https://fictionarry.github.io/DNGaussian/)
> - FSGS / ViewCrafter / DUSt3R 관련 문헌들

---

## 1. 📌 핵심 주장 및 주요 기여 요약

이 논문은 Yingji Zhong, Zhihao Li, Dave Zhenyu Chen, Lanqing Hong, Dan Xu가 저술하여 **CVPR 2025**에 발표된 연구입니다. CVPR 2025 **Highlight** 논문으로 선정되었으며, 비디오 확산 모델(ViewCrafter)의 사전 지식(prior)을 활용하여 희소 입력(sparse input)에서의 extrapolation 및 occlusion 문제를 해결합니다.

### 핵심 주장 (3줄 요약)

희소 입력 기반 3D Gaussian Splatting(3DGS)에서 **extrapolation**과 **occlusion**이라는 두 가지 핵심 문제를 해결하기 위해, 비디오 확산 모델로부터 학습된 사전 지식을 활용하는 **reconstruction by generation 파이프라인**을 제안합니다.

이를 위해 최적화된 3DGS로부터 렌더링된 시퀀스에 기반한 **Scene-Grounding Guidance**를 도입하며, 이 guidance는 **학습(fine-tuning) 없이(training-free)** 동작합니다.

또한 시야 밖 및 가려진 영역을 효과적으로 식별하는 **Trajectory Initialization** 방법과 생성된 시퀀스에 맞춤화된 3DGS 최적화 기법을 추가로 설계하여, 도전적인 벤치마크에서 **state-of-the-art** 성능을 달성합니다.

---

## 2. 🔬 상세 분석

### 2-1. 해결하고자 하는 문제

희소 입력 기반 3DGS에서는 **shape-radiance ambiguity**로 인해 방사장(radiance field)이 퇴화된 표현(degenerate representation)을 학습하는 경향이 있으며, 이는 NeRF와 3DGS 모두에 공통으로 나타납니다.

구체적으로 두 가지 문제가 있습니다:

- **Extrapolation 문제**: 입력 시점(field of view) 바깥 영역에 대한 정보 부재
- **Occlusion 문제**: 입력 이미지에서 가려진 영역에 대한 정보 부재

이를 직관적으로 해결하기 위해 비디오 확산 모델을 활용하여 멀티뷰 시퀀스를 생성하면 대규모 데이터셋으로부터 학습된 사전 지식 기반의 그럴듯한 장면 해석을 제공하고, 뷰 인스턴스를 크게 확장하여 extrapolation 및 occlusion 문제를 해결할 수 있습니다.

그러나 단순 생성 파이프라인의 주된 문제는 생성된 시퀀스 내의 **멀티뷰 비일관성(multi-view inconsistency)**으로, 이는 두 가지 측면에서 나타납니다: **(i) 시퀀스 내 프레임 간 외관 비일관성**, **(ii) 실제 장면에 존재하지 않는 환각(hallucinated) 요소 포함**.

---

### 2-2. 제안 방법

#### 📐 전체 파이프라인 개요

본 논문의 파이프라인은 다음 세 가지 핵심 모듈로 구성됩니다:

| 모듈 | 역할 |
|------|------|
| Trajectory Initialization | 생성할 카메라 경로 결정 |
| Scene-Grounding Guidance | 비디오 확산 모델이 일관된 시퀀스를 생성하도록 유도 |
| 3DGS Optimization Scheme | 생성 시퀀스를 활용한 맞춤형 최적화 |

---

#### 🎯 모듈 1: Trajectory Initialization

Trajectory Initialization은 **기준 3DGS로부터의 렌더링 결과**를 기반으로 시퀀스 생성을 위한 카메라 경로를 결정하며, 전체적인 장면 모델링을 촉진합니다.

이 방법은 시야 밖에 있거나 가려진 영역을 효과적으로 식별합니다.

---

#### 🎯 모듈 2: Scene-Grounding Guidance (핵심 기여)

기존 방법들이 프레임별 학습 가능한 외관 임베딩(per-frame learnable appearance embeddings)을 할당하여 외관 비일관성을 해결하는 것과 달리, 본 논문은 **비디오 확산 모델 자체를 길들여(taming) 일관된 시퀀스를 직접 생성**하는 데 집중합니다.

Scene-Grounding Guidance의 핵심 원리는 **Training-Free Guidance**입니다. 이는 확산 모델의 역방향 샘플링(reverse sampling) 과정에서 에너지 함수(energy function)를 통해 기울기를 주입하는 방식입니다.

비디오 확산 모델의 표준 역방향 과정에서 각 denoising step $t$에서의 업데이트는 다음과 같이 표현됩니다:

$$\tilde{\epsilon}_\theta(\mathbf{x}_t, t) = \epsilon_\theta(\mathbf{x}_t, t) - \sqrt{1 - \bar{\alpha}_t} \cdot \nabla_{\mathbf{x}_t} \mathcal{E}(\mathbf{x}_t)$$

여기서:
- $\epsilon_\theta(\mathbf{x}_t, t)$: 확산 모델의 표준 노이즈 예측
- $\mathcal{E}(\mathbf{x}_t)$: Scene-Grounding Energy Function (3DGS 렌더링과의 일관성을 측정)
- $\nabla_{\mathbf{x}_t} \mathcal{E}(\mathbf{x}_t)$: 에너지의 기울기를 통해 생성을 3DGS 장면으로 유도

Scene-Grounding Energy는 대략 다음과 같이 구성됩니다:

$$\mathcal{E}(\mathbf{x}_t) = \left\| \hat{\mathbf{x}}_0(\mathbf{x}_t) - \mathbf{R}(\Theta) \right\|^2$$

여기서:
- $\hat{\mathbf{x}}\_0(\mathbf{x}_t)$: 현재 노이즈 상태에서 예측된 클린 이미지
- $\mathbf{R}(\Theta)$: 현재 3DGS 파라미터 $\Theta$로부터 렌더링된 시퀀스

> ⚠️ **주의**: 위 수식은 논문에서 밝힌 Training-Free Guidance 원리(FreeDoM 등 참고문헌 기반)를 토대로 구성된 개념 수식입니다. 논문 PDF 원문의 정확한 수식 표기와 부분적으로 다를 수 있으니, 반드시 arXiv PDF 원문(2503.05082)을 확인하시기 바랍니다.

이 guidance는 **training-free**이며, 확산 모델의 어떠한 fine-tuning도 필요하지 않습니다.

---

#### 🎯 모듈 3: 3DGS 최적화 스킴

생성된 시퀀스를 활용한 3DGS 최적화에 맞춤화된 기법이 추가로 설계됩니다.

기준 3DGS는 **DUSt3R 포인트 클라우드**로 초기화되며, **FSGS의 Gaussian Unpooling** 기법을 결합하여 최적화됩니다.

3DGS 최적화의 손실 함수는 다음 형태로 구성됩니다:

$$\mathcal{L}_{total} = \mathcal{L}_{photo}(\mathbf{I}_{sparse}) + \lambda_1 \cdot \mathcal{L}_{photo}(\mathbf{I}_{gen}) + \lambda_2 \cdot \mathcal{L}_{reg}$$

여기서:
- $\mathcal{L}\_{photo}(\mathbf{I}_{sparse})$: 희소 입력 이미지에 대한 photometric loss
- $\mathcal{L}\_{photo}(\mathbf{I}_{gen})$: Scene-Grounding으로 생성된 시퀀스에 대한 photometric loss
- $\mathcal{L}_{reg}$: 정규화 항

> ⚠️ **주의**: 위 손실 함수도 논문의 공개 정보를 바탕으로 재구성된 개념적 표현입니다. 정확한 계수와 항목 구성은 PDF 원문 확인이 필요합니다.

---

### 2-3. 모델 구조

```
[입력: Sparse Images (예: 6개 뷰)]
         ↓
[DUSt3R로 포인트 클라우드 초기화]
         ↓
[기준 3DGS 최적화 (FSGS Gaussian Unpooling 포함)]
         ↓
[Trajectory Initialization → 카메라 경로 설계]
         ↓
[Video Diffusion Model (ViewCrafter) + Scene-Grounding Guidance]
    → 기준 3DGS 렌더링 시퀀스를 에너지로 주입 (Training-Free)
         ↓
[일관된 멀티뷰 시퀀스 생성]
         ↓
[생성 시퀀스 + Sparse 이미지로 최종 3DGS 최적화]
         ↓
[출력: 고품질 Novel View Synthesis]
```

---

### 2-4. 성능 향상

실험 결과, 제안 방법은 **기준선(baseline)보다 유의미하게 향상**되었으며, 도전적인 벤치마크에서 **state-of-the-art** 성능을 달성합니다.

Vanilla 생성은 생성된 시퀀스 내에 비일관성(황색 화살표로 표시)을 가져 렌더링 이미지에 검은 그림자를 초래하는 반면, Scene-Grounding 생성은 일관된 시퀀스를 생성하여 이러한 문제를 효과적으로 해결하고 전반적인 품질(파란색 박스)을 향상시킵니다.

성능 지표(PSNR)를 기준으로 비교할 때:

- **실내 벤치마크**: 6개 입력 뷰 기준, Vanilla 3DGS 대비 유의미한 PSNR 향상
- **비교 대상**: DNGaussian, FSGS, LM-Gaussian, SparseGS 등

관련 후속 연구에서도 본 논문은 "training-free scene-grounding guidance를 사용하여 비디오 확산 모델이 시간적으로 일관된 합성 방향으로 생성하도록 유도한다"고 인용되고 있습니다.

---

### 2-5. 한계점

공개된 정보를 기반으로 분석한 한계점:

1. **계산 비용**: 매 생성 단계마다 3DGS 렌더링을 통한 에너지 계산이 필요하여, 단순 생성 대비 추론 시간이 증가할 수 있습니다.

2. **기반 비디오 모델 의존성**: ViewCrafter 등 특정 비디오 확산 모델의 품질에 성능이 종속됩니다.

3. **시야각 제한**: unbounded하고 완전한 장면(고립된 객체가 아닌)과 1080p 해상도 렌더링에서는 현재 어떤 방법도 실시간 디스플레이 속도를 달성하지 못합니다.

4. **초기화 품질 의존성**: DUSt3R 포인트 클라우드 품질에 따라 결과가 달라질 수 있습니다.

5. 이 방법을 포함한 관련 방법들은 초기 재구성을 부트스트랩하기 위한 맞춤형 전처리 단계와 확산 모델을 효과적으로 훈련하기 위한 정교하게 큐레이션된 데이터셋에 크게 의존합니다.

---

## 3. 🌐 모델의 일반화 성능 향상 가능성

### 3-1. Training-Free 설계의 일반화 강점

제안된 Scene-Grounding Guidance는 최적화된 3DGS로부터 렌더링된 시퀀스에 기반하며, 이는 **training-free**이고 어떠한 diffusion 모델 fine-tuning도 요구하지 않습니다. 이 설계 원칙은 일반화 성능에 다음과 같은 이점을 제공합니다:

- **플러그앤플레이 호환성**: 미래의 더 강력한 비디오 확산 모델(예: Wan2.1, CogVideoX 등)로 교체 시 자동으로 성능 향상 가능
- **도메인 무관성**: 실내(indoor), 실외(outdoor) 등 다양한 장면 유형에 적용 가능

### 3-2. 다양한 입력 환경에서의 일반화

비디오 확산 모델은 대규모 데이터셋으로부터 학습된 사전 지식을 바탕으로 장면에 대한 그럴듯한 해석을 제공하며, 이 시퀀스들은 뷰 인스턴스를 크게 확장하여 extrapolation과 occlusion 문제를 해결할 높은 잠재력을 제공합니다. 이는 특히:

- **극단적 희소 입력**(3~6개 뷰)에서도 일반화된 장면 이해 가능
- 학습 데이터에 없는 새로운 장면 유형에도 확산 모델의 일반화 능력 활용 가능

### 3-3. 일반화 향상을 위한 두 가지 경량 전략

관련 연구에서는 희소뷰 3DGS의 co-adaptation을 완화하기 위한 두 가지 경량 전략—**(1) Random Gaussian Dropout**, **(2) Opacity에 대한 Multiplicative Noise Injection**—을 제안하며, 두 전략 모두 **플러그앤플레이** 방식으로 설계되어 다양한 방법과 벤치마크에서 효과가 검증되었습니다.

### 3-4. 일반화의 수학적 관점

Scene-Grounding Energy를 통한 조건부 생성 분포는 다음과 같이 표현됩니다:

$$p(\mathbf{x}_0 | \text{scene}) \propto p(\mathbf{x}_0) \cdot \exp\left(-\mathcal{E}(\mathbf{x}_0)\right)$$

여기서 $p(\mathbf{x}_0)$는 사전 학습된 확산 모델의 학습 분포(일반적 비디오 분포)를 의미하며, $\exp(-\mathcal{E})$ 항이 특정 장면으로 생성을 "그라운딩"합니다. 이 설계는 장면이 바뀌어도 에너지 함수만 재계산되어 **zero-shot 일반화**가 자연스럽게 이루어집니다.

---

## 4. 🔀 2020년 이후 관련 최신 연구 비교 분석

2020~2025년 희소뷰 3D 재구성 연구 분포를 보면, 2022년 이후 3DGS와 Diffusion/VFM 논문이 급격히 증가하고 있으며, 이는 3DGS의 효율성과 확산 모델의 누락 정보 합성 능력이 기존 NeRF 변형의 한계(계산 비용, 희소 입력에서의 오버피팅)를 직접적으로 해결하기 때문입니다.

| 논문 | 연도 | 핵심 방법 | 한계 |
|------|------|-----------|------|
| **NeRF** (Mildenhall et al.) | 2020 | MLP + Volume Rendering | 밀집 입력 필요, 느린 추론 |
| **3DGS** (Kerbl et al.) | 2023 | 3D Gaussian Primitives | 밀집 입력 필요 |
| **DNGaussian** | 2024 (CVPR) | Depth 정규화 기반 희소뷰 3DGS | 기존 Gaussian 영역에 국한, 초기화되지 않은 영역 처리 불가 |
| **FSGS** | 2024 (ECCV) | Gaussian Unpooling | 뷰 외 영역 생성 불가 |
| **DUSt3R** | 2024 (CVPR) | 기하학적 3D 비전 | 희소뷰 렌더링 품질 직접 향상 X |
| **ViewCrafter** | 2024 | 비디오 확산 기반 NVS | 다중뷰 일관성 문제 |
| **SparseGS** | 2023 | Depth prior + Pruning | 제한된 입력 뷰에서 "floaters" 및 "background collapse" 아티팩트 발생 |
| **🌟 본 논문 (GuidedVD-3DGS)** | 2025 (CVPR Highlight) | Training-Free Scene-Grounding Guidance | 계산 비용, 기반 확산 모델 의존성 |

기존 방법들은 semantic 정규화, smoothness prior, 또는 geometric prior를 통합하는 방식으로 희소뷰 NVS를 향상시켜 왔으며, FSGS와 DNGaussian은 단안 깊이 사전(monocular depth prior)으로 3D Gaussian에서 렌더링된 깊이 맵을 정규화하여 성능을 향상시켰습니다.

DNGaussian은 특정 단계에서 Gaussian 형태 파라미터 및 중심점을 고정하고 깊이를 전역-지역적으로 정규화함으로써 기하학적 저하를 완화하며, 일부 NeRF보다 25배 빠른 학습 속도와 300fps의 실시간 렌더링을 달성합니다.

본 논문은 이들과 달리, **생성 모델의 prior를 직접 3DGS 최적화에 통합**한다는 점에서 차별화됩니다.

---

## 5. 🔭 향후 연구에 미치는 영향 및 고려사항

### 5-1. 앞으로의 연구에 미치는 영향

#### ① Training-Free Guidance 패러다임의 확장
본 논문이 제시한 "3DGS 렌더링으로 확산 모델을 그라운딩"하는 패러다임은 다음 방향으로 확장될 수 있습니다:
- **4D Gaussian Splatting**(동적 장면)에서의 시간적 일관성 확보
- **텍스트/이미지 조건부** 장면 생성에서의 활용
- 다른 명시적 표현(NeRF, Instant-NGP 등)과의 결합

#### ② 재구성-생성 공동 학습(Joint Learning)
최근 비디오 확산 모델이 강력한 시간적 추론 능력을 보여주고 있어, 희소뷰 설정에서의 재구성 품질 향상을 위한 유망한 도구로 주목받고 있습니다. 이를 활용한 **재구성-생성 공동 최적화** 프레임워크 연구가 활성화될 것으로 예상됩니다.

#### ③ 실시간 응용을 위한 경량화
3DGS는 3D Gaussian 프리미티브로 장면을 모델링하고 미분 가능한 splatting으로 이미지를 렌더링하며, NeRF와 비교 가능한 성능을 달성하면서도 훨씬 짧은 학습 시간과 높은 추론 속도를 제공합니다. 본 논문의 방법을 실시간 환경에 적용하기 위한 경량화 연구가 필요합니다.

---

### 5-2. 향후 연구 시 고려할 점

#### ✅ 기술적 고려사항

| 고려사항 | 설명 |
|---------|------|
| **기반 확산 모델 선택** | ViewCrafter 외 최신 모델(Wan2.1, CogVideoX 등) 적용 시 성능 변화 분석 필요 |
| **에너지 함수 설계** | Scene-Grounding Energy의 로버스트성 및 수렴 안정성 |
| **스케일 일반화** | 대형 야외 장면(unbounded scenes)에서의 적용 가능성 |
| **초기 3DGS 품질** | Trajectory Initialization이 초기 3DGS 품질에 민감할 가능성 |
| **생성 다양성 vs 일관성 트레이드오프** | 과도한 그라운딩이 창의적 해석을 제한할 수 있음 |

#### ✅ 평가 방법론 고려사항

희소뷰 3D 재구성의 기하학 기반 방법들을 비교할 때 런타임은 원본 논문에 보고된 대로 학습 또는 추론 시간을 기준으로 측정됩니다. 향후 연구에서는:
- **PSNR/SSIM/LPIPS** 지표 외 **사용자 연구(user study)** 병행 권장
- **극단적 희소 입력**(1~3개 뷰) 설정에서의 성능 평가 추가
- **도메인 일반화** 실험(학습 장면 vs 미지 장면) 필요

#### ✅ 사회적/실용적 고려사항

- **환각(Hallucination) 위험**: 확산 모델이 실제 존재하지 않는 구조물을 생성할 가능성에 대한 검증 메커니즘 필요
- **데이터 프라이버시**: 사전 학습된 확산 모델의 학습 데이터 편향이 생성 결과에 미치는 영향 고려

---

## 📚 종합 참고문헌 목록

| # | 제목 | 출처 |
|---|------|------|
| 1 | **Taming Video Diffusion Prior with Scene-Grounding Guidance for 3D Gaussian Splatting from Sparse Inputs** | arXiv:2503.05082 / CVPR 2025 |
| 2 | **3D Gaussian Splatting for Real-Time Radiance Field Rendering** (Kerbl et al.) | SIGGRAPH 2023 |
| 3 | **DNGaussian: Optimizing Sparse-View 3D Gaussian Radiance Fields with Global-Local Depth Normalization** | CVPR 2024 |
| 4 | **FSGS: Real-Time Few-Shot View Synthesis Using Gaussian Splatting** | ECCV 2024 |
| 5 | **DUSt3R: Geometric 3D Vision Made Easy** | CVPR 2024 |
| 6 | **ViewCrafter: Taming Video Diffusion Models for High-Fidelity Novel View Synthesis** | arXiv:2409.02048 |
| 7 | **Sparse-View 3D Reconstruction: Recent Advances and Open Challenges** | arXiv:2507.16406 |
| 8 | **NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis** (Mildenhall et al.) | ECCV 2020 |
| 9 | **SparseGS: Real-Time 360° Sparse View Synthesis using Gaussian Splatting** | arXiv:2312.00206 |
| 10 | **CoherentGS: Sparse Novel View Synthesis with Coherent 3D Gaussians** | arXiv:2403.19495 |
| 11 | **FreeDoM: Training-Free Energy-Guided Conditional Diffusion Model** | (Training-Free Guidance 원리 기반 참고) |
| 12 | Project Page: https://zhongyingji.github.io/guidevd-3dgs/ | HKUST / Huawei Noah's Ark Lab |
| 13 | GitHub: https://github.com/zhongyingji/guidedvd-3dgs | Yingji Zhong |

---

> **⚠️ 정확도 안내**: 본 답변은 arXiv PDF(2503.05082), CVPR 2025 공식 페이지, GitHub, 프로젝트 페이지 등 공개된 자료를 기반으로 작성되었습니다. **세부 수식(정확한 계수, 손실 함수의 완전한 형태, 수치 실험 결과 표)**은 PDF 원문을 직접 열람하기 어려운 환경의 제약으로 인해 개념적으로 재구성된 부분이 포함되어 있습니다. 연구 목적으로 활용 시 반드시 **arXiv:2503.05082 원문** 및 **CVPR 2025 발표 자료**를 직접 확인하시기 바랍니다.

# Taming Video Diffusion Prior with Scene-Grounding Guidance for 3D Gaussian Splatting from Sparse Inputs

## 1. 핵심 주장과 주요 기여 (요약)

이 논문은 희소 입력 환경에서 **3D Gaussian Splatting(3DGS)의 두 가지 중요한 문제**(외삽과 폐색)를 명시적으로 해결하는 첫 번째 연구입니다.

### 주요 기여:
1. **장면 기반 안내**: 렌더링된 3DGS 수열을 기반으로 비디오 확산 모델을 제어하는 훈련 무료 방법
2. **궤적 초기화 전략**: 3DGS 기반의 자동 카메라 궤적 선택
3. **최적화 방식 개선**: 지각 손실을 활용한 구멍 영역 채우기
4. **성능 향상**: Replica 3.5 dB, ScanNet 2.5 dB의 PSNR 개선

***

## 2. 핵심 수식 체계

### 2.1 조건부 스코어 함수 (베이즈 규칙)

$$\nabla_{\mathbf{x}_t}\log p(\mathbf{x}_t | Q) = \nabla_{\mathbf{x}_t}\log p(\mathbf{x}_t) + \nabla_{\mathbf{x}_t}\log p(Q|\mathbf{x}_t)$$

여기서 $Q$는 렌더링된 수열 기반 일관성 목표입니다.

### 2.2 장면 기반 손실 함수

$$L_S(M, S, X_t') = \|M \odot (S - X_t')\|_1 + \lambda_{\text{perc}} L_{\text{perc}}(M \odot S, M \odot X_t')$$

여기서:
- $M$: 폐색/시야 밖 영역 마스크
- $S$: 렌더링된 수열
- $X_t'$: 디노이징 예측 이미지
- $\odot$: 아다마르 곱 (요소별 곱셈)

### 2.3 최종 3DGS 최적화 손실

**입력 이미지:**
$$L_{\text{input}} = (1-\lambda_w)L_1(C_i, C_i^{\text{gt}}) + \lambda_w L_{\text{DSSIM}}(C_i, C_i^{\text{gt}})$$

**생성 이미지:**
$$L_{\text{gen}} = \lambda_{\text{gen1}} L_1(C_j, S_j) + \lambda_{\text{gen2}} L_{\text{perc}}(C_j, S_j)$$

***

## 3. 모델 구조의 특징

### 3.1 통합 파이프라인
```
희소 입력 → DUSt3R 초기화 → 기본 3DGS
    ↓
[궤적 초기화] ← [렌더링된 수열]
    ↓
장면 기반 안내 비디오 확산
    ↓
일관성 있는 생성 수열
    ↓
최종 3DGS 최적화
```

### 3.2 세 가지 핵심 기술

**1. 마스크 계산 (전송 지도)**

$$O(\mathbf{x}_p) = \prod_{i=1}^{K} (1 - \alpha_i), \quad M = O < \tau_{\text{mask}}$$

**2. 궤적 보간**
방위각: ±30°, ±15°, 0° / 반지름: 1배, 1/3배, 1/10배 깊이

**3. 지역적 샘플링 전략**
- 같은 수열에서: 70% (시각 품질)
- 다른 수열에서: 30% (망각 방지)

***

## 4. 성능 및 일반화 성능 향상

### 4.1 정량적 성과

| 데이터셋 | 기본선 | 제안 방법 | 향상도 |
|---------|--------|---------|--------|
| Replica | 22.80 dB | 26.35 dB | +3.5 dB |
| ScanNet | 21.41 dB | 23.89 dB | +2.5 dB |

### 4.2 관찰 불가능 영역의 극적 개선

| 영역 | 기본선 | 제안 방법 | 향상도 |
|------|--------|---------|--------|
| 관찰 가능 | 25.45 dB | 27.12 dB | +1.67 dB |
| 관찰 불가능 | 14.27 dB | 20.85 dB | **+6.58 dB** |

### 4.3 일반화 성능의 근본 원인

**1. 다중 뷰 일관성 학습**
- 비디오 확산 모델이 보유한 강력한 3D 구조 사전
- 시간적 일관성, 3D 기하학, 물리적 제약

**2. 장면 기반 안내의 이중 효과**
- **일관성 제약**: 인접 프레임의 높은 일관성
- **장면 기반 제약**: 환각 요소 제거

**3. 훈련 무료 방법의 장점**
- 도메인 시프트에 강건
- 새로운 장면에 즉시 적응
- 확산 모델의 일반화 능력 직접 활용

***

## 5. 한계 및 향후 과제

### 5.1 확인된 한계

**해상도 제약 (가장 심각)**
- Replica: 320×448 → 480×640 (업샘플링)
- ScanNet: 320×512 → 480×720 (업샘플링)
- **결과**: 과도한 평활화, 고주파 디테일 손실

**계산 비용**
- 생성 시간: 이미지당 수 분
- GPU 메모리: 32GB V100 필요
- 실시간 적용 불가능

**생성 품질 의존성**
- 초기 기본 3DGS 모델 품질에 의존
- 순환 의존성 잠재력

### 5.2 향후 해결 방향

**단기 (1-2년)**:
- 해상도 확대 (다단계 생성)
- 확산 모델 가속화
- 지역 세부 정제

**중기 (2-5년)**:
- 동적 장면 처리 (4D)
- 다양한 장면 타입 지원
- 다중 모달리티 통합

**장기 (5년 이상)**:
- 단일 이미지 한계 도전
- 인간 수준 재구성
- 차세대 표현 패러다임

***

## 6. 관련 최신 연구 동향 (2020 이후)

### 6.1 희소 입력 3DGS 발전 (2024-2025)

| 방법 | 핵심 기여 | 성능 향상 |
|------|---------|---------|
| **NexusGS** (2025) | 에피폴라 깊이 사전 | +0.5-1.0 dB |
| **HiSplat** (2025, ICLR) | 계층적 가우시안 | +0.82-3.19 dB |
| **GraphSplat** (2025) | 그래프 기반 특성 | 크로스 데이터셋 개선 |
| **MS-GS** (2025) | 다중 모양 + 의미론 | 광 조건 변화 처리 |

### 6.2 생성 모델 기반 3D 재구성 (2023-2025)

**확산 모델 활용**:
- **ReconFusion** (2024): SDS 기반 재구성
- **CAT3D** (2024): 다중 뷰 확산, 분 단위 생성
- **CAT4D** (2024): 4D 재구성
- **ViDAR** (2025): 비디오 확산 인식 4D

**훈련 무료 안내 (2024-2025)**:
- **TFG** (NeurIPS 2024): 통합 훈련 무료 안내
- **Dreamguider** (2024): 역전파 없는 안내
- **OC-Flow** (2025): 최적 제어 기반

### 6.3 폐색 처리 방법 (2023-2025)

- **OccluGaussian** (2025, ICCV): 폐색 인식 장면 분할
- **FSGS** (2024): 보이지 않는 뷰 정규화
- **GS-GS** (2025): 생성적 희소 뷰 가우시안

***

## 7. 향후 연구 시 고려사항

### 7.1 학술 연구자

**주요 고민**:
- 생성 사전의 최적 활용 방법
- 일관성과 다양성의 절충
- 도메인 특화 확산 모델 필요성

**연구 아이디어**:
- 다중 목표 최적화 프레임워크
- 불확실성 기반 가중 안내
- 하이브리드 명시/암시 표현

### 7.2 산업 응용 개발자

**실무 체크리스트**:
- GPU 메모리 (32GB 이상)
- 배치 처리 최적화
- 중간 결과 캐싱
- 모니터링 시스템

**적용 가능 분야**:
- ✓ 문화재 보존
- ✓ VR/XR 콘텐츠
- △ 건축 시각화
- ✗ 실시간 렌더링

### 7.3 시스템 엔지니어

**최적화 핵심**:
- 해상도-메모리 트레이드오프
- 생성 시간 최소화
- 캐시 효율성
- 병렬 처리

**하이퍼파라미터 권장값**:
- 입력 뷰: 6-9개
- 해상도: 480×640
- λ_perc: 10^-4
- ρ: 0.5 (지역/전역 샘플링 비율)

***

## 8. 최종 평가

| 평가 항목 | 점수 | 근거 |
|---------|------|------|
| **혁신도** | ★★★★★ | 훈련 무료 방식, 새로운 안내 기법 |
| **실용성** | ★★★★☆ | 산업 가능하나 계산 비용 있음 |
| **이론성** | ★★★★☆ | 베이즈 기초 확고, 최적성 증명 가능 |
| **영향력** | ★★★★★ | 3D 비전, 확산, 신경장 분야 광범위 |

***

## 참고: 제공된 상세 분석 문서

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/bcb15904-460c-43d5-a495-33758e2ae8db/2503.05082v1.pdf)
[2](https://ieeexplore.ieee.org/document/11092863/)
[3](https://ieeexplore.ieee.org/document/11125578/)
[4](https://ieeexplore.ieee.org/document/11247717/)
[5](https://ieeexplore.ieee.org/document/10887746/)
[6](https://arxiv.org/abs/2508.15457)
[7](https://arxiv.org/abs/2509.15548)
[8](https://link.springer.com/10.1007/s10489-025-06494-2)
[9](https://arxiv.org/abs/2506.10335)
[10](https://dl.acm.org/doi/10.1145/3746027.3755481)
[11](https://www.semanticscholar.org/paper/4d867cecda8646506bd14647af1014fe6557d8b5)
[12](https://arxiv.org/html/2412.10051v1)
[13](https://arxiv.org/html/2410.06245)
[14](https://arxiv.org/html/2502.02283v3)
[15](https://arxiv.org/pdf/2403.14627.pdf)
[16](https://arxiv.org/html/2401.02436v1)
[17](https://arxiv.org/html/2312.00206v2)
[18](https://arxiv.org/html/2503.04314v1)
[19](https://arxiv.org/html/2412.02245)
[20](https://proceedings.iclr.cc/paper_files/paper/2025/file/78da47a28386d3e2e5e156d8148cecdf-Paper-Conference.pdf)
[21](https://openreview.net/forum?id=mLVqiNH0aA)
[22](https://proceedings.neurips.cc/paper_files/paper/2024/file/cad4501fe7c1b53427b363daf1366b2f-Paper-Conference.pdf)
[23](https://arxiv.org/html/2312.00206v3)
[24](https://blog.outta.ai/289)
[25](https://arxiv.org/html/2508.15457)
[26](https://www.sciencedirect.com/science/article/abs/pii/S1568494625008415)
[27](https://kimjy99.github.io/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0/cat4d/)
[28](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/10096.pdf)
[29](https://openaccess.thecvf.com/content/CVPR2025/papers/Kong_Generative_Sparse-View_Gaussian_Splatting_CVPR_2025_paper.pdf)
[30](https://dl.acm.org/doi/10.1145/3746027.3761989)
[31](https://www.isca-archive.org/interspeech_2024/choi24c_interspeech.html)
[32](https://www.semanticscholar.org/paper/6c708659768e470f63d06f791ff8420e7ff0feac)
[33](https://arxiv.org/abs/2505.02527)
[34](https://journals.tsu.ru/philosophy/&journal_page=archive&id=2595&article_id=52952)
[35](https://www.semanticscholar.org/paper/150aec040041ad4ba473da610101820c767d63ff)
[36](https://aacrjournals.org/cebp/article/34/9_Supplement/B158/764755/Abstract-B158-Implementation-of-a-low-cost-tobacco)
[37](https://aacrjournals.org/clincancerres/article/31/13_Supplement/A035/763347/Abstract-A035-Denoising-Models-Enhance-Detection)
[38](https://www.po-rt.ru/articles/1943)
[39](https://ocs.editorial.upv.es/index.php/HEAD/HEAd25/paper/view/19996)
[40](https://arxiv.org/abs/2409.15761)
[41](https://arxiv.org/abs/2407.02687)
[42](https://arxiv.org/html/2406.02549)
[43](https://arxiv.org/pdf/2403.12404.pdf)
[44](https://arxiv.org/html/2406.07540)
[45](http://arxiv.org/pdf/2312.12487.pdf)
[46](https://arxiv.org/pdf/2210.09292.pdf)
[47](https://arxiv.org/html/2410.18070)
[48](https://theaisummer.com/classifier-free-guidance/)
[49](https://arxiv.org/html/2503.16177v2)
[50](https://isprs-annals.copernicus.org/articles/X-1-W1-2023/895/2023/isprs-annals-X-1-W1-2023-895-2023.pdf)
[51](https://papers.nips.cc/paper_files/paper/2024/file/2818054fc6de6dacdda0f142a3475933-Paper-Conference.pdf)
[52](https://openaccess.thecvf.com/content/ICCV2025/papers/Liu_OccluGaussian_Occlusion-Aware_Gaussian_Splatting_for_Large_Scene_Reconstruction_and_Rendering_ICCV_2025_paper.pdf)
[53](https://arxiv.org/html/2312.09095v2)
[54](https://dl.acm.org/doi/10.1145/3681758.3697997)
[55](https://dl.acm.org/doi/10.1145/3610548.3618188)
[56](https://openreview.net/forum?id=N8YbGX98vc)
[57](https://arxiv.org/abs/2505.19854)
[58](https://arxiv.org/abs/2505.20729)
[59](https://ieeexplore.ieee.org/document/11137711/)
[60](https://www.semanticscholar.org/paper/d98ac277478bbc568ede0c1f331d4e78ad745c7f)
[61](https://www.mdpi.com/2072-4292/15/12/3076)
[62](https://www.mdpi.com/2077-0472/14/3/391)
[63](https://ieeexplore.ieee.org/document/10205144/)
[64](http://biorxiv.org/lookup/doi/10.1101/2021.11.09.467984)
[65](https://linkinghub.elsevier.com/retrieve/pii/S0926580523002091)
[66](https://www.mdpi.com/2072-4292/15/15/3775)
[67](https://arxiv.org/html/2503.16318)
[68](https://arxiv.org/html/2409.08613v1)
[69](https://arxiv.org/html/2503.24391)
[70](http://arxiv.org/pdf/2312.14132v1.pdf)
[71](https://arxiv.org/html/2312.06706v1)
[72](https://arxiv.org/html/2410.23245)
[73](https://www.mdpi.com/1424-8220/25/5/1354)
[74](https://arxiv.org/pdf/1612.00603.pdf)
[75](https://learnopencv.com/dust3r-geometric-3d-vision/)
[76](https://pubmed.ncbi.nlm.nih.gov/39531569/)
[77](https://drexubery.github.io/ViewCrafter/)
[78](https://github.com/naver/dust3r)
[79](https://proceedings.neurips.cc/paper_files/paper/2023/file/b87738474533cab76c7bee4e08443aca-Paper-Conference.pdf)
[80](https://pubmed.ncbi.nlm.nih.gov/40986578/)
[81](https://kimjy99.github.io/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0/dust3r/)
[82](https://cwchenwang.github.io/outdoor-nerf-depth/data/paper.pdf)
[83](https://arxiv.org/html/2503.05638v1)
[84](https://ethanswinery.tistory.com/154)
