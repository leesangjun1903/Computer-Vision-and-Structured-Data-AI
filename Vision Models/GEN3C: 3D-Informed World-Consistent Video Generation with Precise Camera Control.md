
# GEN3C: 3D-Informed World-Consistent Video Generation with Precise Camera Control

> **논문 정보**
> - **제목:** GEN3C: 3D-Informed World-Consistent Video Generation with Precise Camera Control
> - **저자:** Xuanchi Ren\*, Tianchang Shen\*, Jiahui Huang, Huan Ling, Yifan Lu, Merlin Nimier-David, Thomas Müller, Alexander Keller, Sanja Fidler, Jun Gao (*equal contribution)
> - **학회:** CVPR 2025 **Highlight** (상위 3% 수락률)
> - **arXiv:** [2503.03751](https://arxiv.org/abs/2503.03751) (2025년 3월 5일)
> - **소속:** NVIDIA Toronto AI Lab

---

## 1. 핵심 주장 및 주요 기여 요약

### 1.1 핵심 주장

GEN3C는 정밀한 카메라 제어(Camera Control)와 시간적 3D 일관성(temporal 3D Consistency)을 갖춘 생성형 비디오 모델이다. 기존 비디오 모델들은 이미 현실적인 영상을 생성하지만, 3D 정보를 거의 활용하지 않아 객체가 갑자기 나타났다 사라지는 등의 비일관성 문제가 발생한다. 카메라 제어가 구현된 경우에도, 카메라 파라미터가 신경망의 단순 입력에 불과하기 때문에 모델이 카메라에 따른 영상 변화를 스스로 추론해야 해 부정확할 수밖에 없다.

이에 대한 핵심 해결책으로, GEN3C는 **3D 캐시(3D cache)**—시드 이미지 또는 이전에 생성된 프레임의 픽셀별 깊이 예측으로 얻은 포인트 클라우드—에 의해 가이드된다. 다음 프레임을 생성할 때, GEN3C는 사용자가 제공한 새로운 카메라 궤적으로 3D 캐시를 2D 렌더링한 결과를 컨디셔닝 입력으로 사용한다. 이는 곧 GEN3C가 이전에 생성한 내용을 기억하거나 카메라 포즈로부터 이미지 구조를 추론할 필요가 없음을 의미한다.

### 1.2 주요 기여 (Contributions)

GEN3C는 정밀한 카메라 제어로 길고 시간적으로 일관된 영상을 생성하며, 단일 뷰 및 스파스 뷰(sparse-view) 새로운 시점 합성(Novel View Synthesis), 단안 동영상 새로운 시점 합성, 자율주행 시뮬레이션 등 다양한 응용에 적용된다.

명시적 3D 캐시를 통해 Dolly Zoom처럼 포즈와 내부 파라미터를 동시에 변화시키는 영화적 효과(cinematic effects)나 3D 편집(3D editing)을 지원한다.

| 기여 항목 | 설명 |
|---|---|
| **3D 캐시 메커니즘** | 포인트 클라우드 기반 명시적 3D 표현으로 시간적 일관성 확보 |
| **정밀 카메라 제어** | 사용자 정의 카메라 궤적을 2D 렌더링으로 변환해 직접 컨디셔닝 |
| **다양한 입력 유연성** | 단일 이미지, 스파스 뷰, 동적 동영상 등 다양한 입력 지원 |
| **SOTA 성능** | 스파스 뷰 NVS에서 최첨단 성능 달성 |
| **영역 일반화** | 도메인 외 영상(Sora, MovieGen 생성 영상)에도 적용 가능 |

---

## 2. 해결하고자 하는 문제, 제안 방법, 모델 구조, 성능 및 한계

### 2.1 해결하고자 하는 문제

현재 비디오 모델들은 현실적인 영상을 만들 수 있지만, 3D 공간에서의 일관성 유지에 취약하다. 프레임 사이에서 객체가 갑자기 나타나거나 사라질 수 있으며, 카메라 움직임을 제어하고자 할 때 이를 정밀하게 다루지 못한다.

구체적으로는 두 가지 핵심 문제를 다룬다:

1. **3D 비일관성 문제**: 기존 모델은 3D 구조를 명시적으로 인코딩하지 않아, 긴 비디오에서 객체가 등장/소멸하거나 구조가 왜곡되는 현상이 발생.
2. **부정확한 카메라 제어**: 기존 방식은 카메라 파라미터를 네트워크에 단순 수치로 입력하므로, 네트워크가 그 관계를 암묵적으로 학습해야 해 부정확한 제어만 가능.

### 2.2 제안 방법 (수식 포함)

#### (1) 3D 캐시 구축: 깊이 예측 기반 포인트 클라우드 생성

시드 이미지 $I_0 \in \mathbb{R}^{H \times W \times 3}$가 주어지면, 픽셀별 깊이 예측 모델 $f_\text{depth}$를 사용해 깊이 맵을 계산한다:

$$D_0 = f_\text{depth}(I_0) \in \mathbb{R}^{H \times W}$$

이 깊이 맵과 카메라 내부 파라미터(intrinsics) $K$, 외부 파라미터(extrinsics) $[R|t]$를 이용해 픽셀 $(u, v)$를 3D 포인트로 역투영(unproject)한다:

$$\mathbf{p}_{uv} = R^{-1} \left( K^{-1} \begin{bmatrix} u \\ v \\ 1 \end{bmatrix} D_0(u,v) - t \right)$$

이렇게 생성된 포인트 집합 $\mathcal{P} = \{\mathbf{p}\_{uv}, \mathbf{c}_{uv}\}$ (포인트 + RGB 색상)가 **3D 캐시**를 형성한다.

#### (2) 새로운 카메라 시점에서 2D 렌더링

사용자가 정의한 새로운 카메라 포즈 $[R'|t']$와 내부 파라미터 $K'$을 이용해 3D 캐시를 새로운 2D 이미지 평면으로 투영한다:

$$\tilde{I}_t = \Pi\left(\mathcal{P};\, K',\, R',\, t'\right)$$

여기서 $\Pi(\cdot)$은 포인트 클라우드 렌더링 함수이며, 투영되지 않은 영역(occluded/unseen regions)은 빈 픽셀로 남는다.

#### (3) 비디오 확산 모델을 통한 생성

렌더링된 2D 이미지 $\tilde{I}_t$를 컨디셔닝 입력으로 사용해 비디오 확산 모델이 다음 프레임을 생성한다. 표준 DDPM(Denoising Diffusion Probabilistic Model) 역방향 프로세스를 따른다:

$$p_\theta(\mathbf{x}_{0:T}) = p(\mathbf{x}_T) \prod_{t=1}^{T} p_\theta(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \tilde{I}_t)$$

모델의 목적함수는 다음 노이즈 예측 손실이다:

$$\mathcal{L} = \mathbb{E}_{\mathbf{x}_0, \epsilon, t}\left[\left\|\epsilon - \epsilon_\theta\left(\mathbf{x}_t,\, t,\, \tilde{I}_t\right)\right\|^2\right]$$

#### (4) 오토레그레시브 슬라이딩 윈도우 생성

긴 동영상 생성을 위해 이전에 생성된 프레임으로 3D 캐시를 업데이트하며 순차적으로 생성한다. 생성된 프레임 $\hat{I}_t$로부터 추가 깊이를 예측하여 캐시를 점진적으로 확장한다:

```math
\mathcal{P} \leftarrow \mathcal{P} \cup \left\{f_\text{unproject}\left(\hat{I}_t,\, f_\text{depth}(\hat{I}_t),\, K'_t,\, R'_t,\, t'_t\right)\right\}
```

사용자 입력(단일 이미지, 다중 뷰 이미지, 동영상)으로부터 각 이미지의 깊이를 예측하여 3D로 역투영함으로써 시공간 3D 캐시를 구축하고, 사용자의 카메라 포즈로 캐시를 렌더링해 비디오 확산 모델에 공급하여 원하는 카메라 포즈에 정렬된 고품질 비디오를 생성한다.

### 2.3 모델 구조

GEN3C의 아키텍처 유형은 Convolutional Neural Network(CNN)과 Transformer Network이다.

입력 타입은 카메라 파라미터와 이미지이며, 입력 포맷은 1D 배열의 카메라 포즈와 2D 배열의 이미지다. 입력 이미지는 720×1080 해상도를 권장하며, 카메라 파라미터는 121 프레임을 권장한다.

출력은 MP4 형식의 비디오이며, N×H×W 크기의 RGB 시퀀스 형태다.

주요 백본 모델로는 두 가지 버전이 공개되어 있다:
- **GEN3C-SVD**: Stable Video Diffusion(SVD) 기반
- **GEN3C-Cosmos-7B**: NVIDIA Cosmos(7B 파라미터) 기반

GEN3C는 추가적으로 깊이 정보, 카메라 내부 파라미터(intrinsics), 외부 파라미터(extrinsics)를 필요로 하며, 이는 VGGT 같은 오프-더-쉘프 방법으로 얻을 수 있다.

### 2.4 성능 향상

결과는 기존 연구보다 더 정밀한 카메라 제어와 함께, 자율주행 장면 및 단안 동영상과 같은 도전적인 환경에서도 스파스 뷰 새로운 시점 합성에서 최첨단 결과를 보여준다.

GEN3C는 단 두 장의 입력 이미지로 현실적인 영상을 생성하며, 동적 조명이나 반사(예: 피아노 반사)와 같은 사실적인 시점 의존적 조명 효과도 포착한다. 기준 방법들(MVSplat, PixelSplat)과 비교했을 때 최소 오버랩 상황에서도 훨씬 자연스럽고 현실적인 새로운 시점을 생성한다.

다른 모델들이 객체를 사라지게 하거나 부자연스럽게 변형시키는 반면, GEN3C는 현실과 더 유사한 일관된 세계를 유지한다. GEN3C는 이전 비디오 생성 모델들에 비해 정밀한 카메라 제어에서 우수한 성능을 보이며, 3D 캐시 시스템을 통해 프레임 간 객체 일관성을 더 잘 유지하여 객체가 갑자기 나타나거나 사라지는 문제를 감소시킨다.

모델은 동적 장면에 잘 일반화되며, 시점을 정확하게 제어하고 3D 일관성 있는 고화질 영상을 생성하며, 3D 캐시에서 가려지거나 누락된 영역을 채우는 능력을 보인다. 새로운 시점 합성 외에도 명시적 3D 캐시를 통해 객체 제거나 장면 편집 등의 응용도 가능하다.

### 2.5 한계점

이 접근법의 핵심 한계로는 비디오 확산이 종종 시간적 비일관성과 입력 이미지와의 약한 정렬을 유발하며, 포인트 클라우드가 희소하거나 부정확할 수 있고, 전체 프레임 확산은 계산 비용이 크다는 점이 있다. GEN3C는 특정 궤적 패턴에서만 좋은 결과를 보였다는 평가도 있다.

추가적인 한계:
- **깊이 추정 정확도 의존성**: 초기 깊이 예측이 부정확하면 3D 캐시 품질이 저하됨.
- **미관측 영역 한계**: 카메라가 크게 이동할수록 기존 캐시로 커버되지 않는 영역이 늘어나 환각(hallucination) 가능성 증가.
- **높은 메모리 요구량**: H100과 A100 GPU에서만 테스트되었으며, 메모리가 제한된 GPU의 경우 풀 오프로드 시 약 43GB의 최대 메모리를 관측한다.

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 도메인 외 일반화

GEN3C의 모델은 도메인 외 영상에도 일반화된다. Sora나 MovieGen이 생성한 영상을 입력으로 받아 카메라 제어가 적용된 출력을 생성할 수 있다.

GEN3C는 단안 동영상 생성에서도 우수한 성능을 보이며, 단일 2D 영상 입력만으로도 설득력 있는 3D 일관성 영상을 생성할 수 있다. 이는 제한된 초기 데이터에서도 3D 정보를 추론하고 유지하는 모델의 능력을 보여준다.

### 3.2 다양한 입력 모달리티 일반화

GEN3C는 단일 이미지로부터의 비디오/장면 생성, 스파스 뷰 이미지(5장 사용)로부터의 생성, 그리고 돌리 줌(Dolly Zoom)과 같은 포즈와 내부 파라미터를 동시에 변경하는 영화적 효과를 포함한 동적 영상 처리까지 다양한 입력 조건에서 작동한다.

### 3.3 일반화 성능을 높이는 핵심 설계

GEN3C는 이전에 생성한 내용을 기억하거나 카메라 포즈로부터 이미지 구조를 추론할 필요가 없다. 대신, 모델은 이전에 관측되지 않은 영역을 채우고 다음 프레임으로 장면 상태를 진전시키는 데 모든 생성 능력을 집중할 수 있다.

이 설계 원칙은 일반화 성능 향상에 직접 기여한다:

- **명시적 3D 표현**: 포인트 클라우드라는 도메인-독립적 표현을 사용하므로 다양한 장면 유형에 적용 가능.
- **생성 부담 분리**: 기억(3D 캐시 담당)과 생성(확산 모델 담당)을 분리하여 각 모듈이 자신의 역할에 집중.

### 3.4 ViPE를 통한 추가 일반화

2025년 8월, GEN3C 팀은 비디오로부터 깊이와 카메라 포즈를 공동으로 예측하는 데이터 어노테이션 파이프라인인 ViPE를 공개했다. ViPE는 GEN3C의 훈련과 테스트 단계 모두에 적용된다.

2025년 9월에는 GEN3C를 실제 세계 데이터 없이 정적 및 동적 3D 가우시안 스플래팅(3DGS) 디코더로 증류하는 새로운 연구 Lyra도 공개되었다.

---

## 4. 2020년 이후 관련 최신 연구 비교 분석

### 4.1 주요 선행/동시 연구 비교표

| 방법 | 발표 | 카메라 제어 방식 | 3D 정보 활용 | 주요 특징 |
|---|---|---|---|---|
| **NeRF** | ECCV 2020 | 없음 | Neural Radiance Field | 정적 장면 재구성, 밀집 뷰 필요 |
| **MotionCtrl** | SIGGRAPH 2024 | 카메라 포즈를 latent에 직접 주입 | 없음 | 카메라·객체 모션 독립 제어 |
| **CameraCtrl** | 2024 | Plücker 임베딩 | 없음 | U-Net에 카메라 포즈 주입 |
| **ViewCrafter** | arXiv 2024 | 포인트 클라우드 prior 활용 | DUSt3R 재구성 | 스파스 뷰 NVS, 반복적 뷰 확장 |
| **GEN3C** | CVPR 2025 | **3D 캐시 렌더링으로 직접 컨디셔닝** | **픽셀별 깊이→포인트 클라우드** | 동적 장면, 오토레그레시브 생성 |
| **SEVA** | 2025 | 두-단계 앵커+보간 | 최소 3D | 긴 궤적 NVS, 다양한 입력 뷰 수 |

### 4.2 MotionCtrl vs. GEN3C

카메라와 객체 모션의 정확한 제어는 비디오 생성의 필수 요소다. 그러나 기존 연구들은 한 가지 유형의 모션에만 집중하거나 두 가지를 명확히 구분하지 못해 제어 능력과 다양성이 제한된다. 이에 MotionCtrl은 카메라와 객체 모션을 효과적으로 독립적으로 제어하는 통합 유연 모션 컨트롤러를 제안한다.

그러나 이러한 방법들은 개선된 제어 가능성을 보이지만, 3D 공간 관계를 모델링하는 능력은 여전히 제한적이다. GEN3C는 이를 명시적 3D 캐시로 해결한다.

### 4.3 ViewCrafter vs. GEN3C

ViewCrafter는 비디오 확산 모델의 강력한 생성 능력과 포인트 기반 표현이 제공하는 거친 3D 단서를 활용해 정밀한 카메라 포즈 제어로 고품질 비디오 프레임을 생성한다.

GEN3C는 이와 유사한 포인트 클라우드 접근법을 사용하지만, 더 일반적인 단안 깊이 예측 기반의 포인트 클라우드를 사용하고, 동적 장면 및 오토레그레시브 긴 영상 생성에서 ViewCrafter를 능가한다.

### 4.4 관련 후속 연구 (유사 논문)

HuggingFace에 의해 추천된 유사 논문들로는 CamCtrl3D(2025), VidCRAFT3(2025), Difix3D+(2025), F3D-Gaus(2025), Joint Learning of Depth and Appearance for Portrait Image Animation(2025), Towards Physical Understanding in Video Generation(2025), Matrix3D(2025) 등이 있다.

---

## 5. 미래 연구에 미치는 영향 및 고려사항

### 5.1 앞으로의 연구에 미치는 영향

#### (1) 명시적 3D 표현과 생성 모델의 통합 패러다임 확립
GEN3C는 뉴럴 렌더링(NeRF, 3D-GS)과 확산 모델 사이의 다리 역할을 하는 새로운 설계 패러다임을 제시했다. 이 결과들은 비디오 생성을 실제 세계 모델링에 적용하기 위한 한 걸음으로서 해당 접근법의 유효성을 검증한다.

#### (2) 월드 모델(World Model) 연구의 촉진
ReconDreamer, GEN3C, FreeVS와 같은 방법들은 투영된 LiDAR 포인트, 생성된 깊이, 또는 퇴화된 가우시안과 같은 기하학적 제어 신호를 조건으로 하는 사전 학습된 생성 모델을 도입하여 외삽 시점의 품질을 향상시키려 한다.

#### (3) 자율주행 시뮬레이션의 새로운 방향
GEN3C는 자율주행 시뮬레이션을 위해 원본 영상의 사실적인 시점 변환을 생성할 수 있다. 이는 데이터 증강, 시나리오 다양화, 안전 테스트 등에 즉시 활용 가능한 실용적 가치를 지닌다.

#### (4) 생성 모델의 역할 분리 원칙 확산
"3D 캐시가 기억을 담당하고, 확산 모델이 생성을 담당한다"는 역할 분리 원칙은 더 효율적이고 일반화 가능한 시스템 설계에 영감을 줄 것이다.

### 5.2 앞으로 연구 시 고려할 점

#### (1) 깊이 추정 품질 개선
포인트 클라우드의 품질은 깊이 예측 모델의 정확도에 강하게 의존한다. 특히 투명 표면, 반사면, 얇은 구조물 등은 깊이 추정이 어렵다. **메트릭 깊이(metric depth) 예측** 연구와의 긴밀한 연계가 필요하다.

#### (2) 미관측 영역 생성 품질 개선
카메라가 크게 이동할 때, 3D 캐시가 커버하지 못하는 빈 영역에 대한 생성 품질이 저하된다. 이 문제를 위한 **조건부 인페인팅(conditional inpainting)** 전략이나 **사전 지식 기반 장면 완성** 연구가 필요하다.

#### (3) 동적 객체 처리
포인트 클라우드가 희소하거나 부정확할 수 있고, 비디오 확산이 시간적 비일관성을 유발할 수 있다. 특히 움직이는 객체의 경우 시간에 따라 포인트 클라우드가 잘못된 위치에 남는 "ghost" 현상이 발생할 수 있으므로, **4D 시공간 캐시** 또는 **객체별 분리 표현**이 연구되어야 한다.

#### (4) 계산 효율성
H100/A100에서만 테스트되었으며, 최대 약 43GB의 메모리가 필요하다. 실용적 배포를 위한 **모델 경량화(knowledge distillation, quantization)** 연구가 필수적이며, 이미 Lyra라는 후속 연구에서 이를 3DGS 디코더로 증류하는 방향이 제시되었다.

#### (5) 다양한 도메인 데이터셋 확보
현재 학습 데이터는 실내/실외 정적 장면 중심이다. 의료 영상, 수중 영상, 위성 영상 등 특수 도메인에서의 일반화를 위해 도메인별 파인튜닝 전략과 데이터 수집 파이프라인 구축이 필요하다.

#### (6) 다른 3D 표현과의 통합 가능성
포인트 클라우드 대신 **Neural Radiance Field(NeRF)**, **3D Gaussian Splatting(3D-GS)**, **Voxel Grid** 등을 3D 캐시로 활용하는 연구도 활발히 이루어질 것으로 예상된다. 각 표현의 장단점을 고려한 하이브리드 접근이 유망하다.

---

## 📚 참고 자료 및 출처

| # | 제목/출처 | 유형 |
|---|---|---|
| 1 | [arXiv:2503.03751 — GEN3C (원본 논문)](https://arxiv.org/abs/2503.03751) | arXiv 논문 |
| 2 | [NVIDIA Research Project Page — GEN3C](https://research.nvidia.com/labs/toronto-ai/GEN3C/) | 공식 프로젝트 페이지 |
| 3 | [GitHub — nv-tlabs/GEN3C (CVPR 2025 Highlight)](https://github.com/nv-tlabs/GEN3C) | 공식 코드 저장소 |
| 4 | [CVPR 2025 Open Access — GEN3C Paper PDF](https://openaccess.thecvf.com/content/CVPR2025/papers/Ren_GEN3C_3D-Informed_World-Consistent_Video_Generation_with_Precise_Camera_Control_CVPR_2025_paper.pdf) | 학회 논문 (오픈 액세스) |
| 5 | [HuggingFace — nvidia/GEN3C-Cosmos-7B 모델 카드](https://huggingface.co/nvidia/GEN3C-Cosmos-7B) | 모델 카드 |
| 6 | [HuggingFace Papers — GEN3C](https://huggingface.co/papers/2503.03751) | 논문 페이지 |
| 7 | [ResearchGate — GEN3C PDF](https://www.researchgate.net/publication/389617126) | 학술 DB |
| 8 | [Xuanchi Ren 홈페이지 — GEN3C (CVPR 2025 Highlight, 3% 수락률)](https://xuanchiren.com/publication/ren-2025-gen3c/) | 저자 홈페이지 |
| 9 | [aimodels.fyi — GEN3C 논문 상세](https://www.aimodels.fyi/papers/arxiv/gen3c-3d-informed-world-consistent-video-generation) | AI 논문 DB |
| 10 | [NVIDIA Research — GEN3C Publication](https://research.nvidia.com/publication/2025-08_gen3c-3d-informed-world-consistent-video-generation-precise-camera-control) | NVIDIA 공식 |
| 11 | [CVPR 2025 Open Access HTML — GEN3C](https://openaccess.thecvf.com/content/CVPR2025/html/Ren_GEN3C_3D-Informed_World-Consistent_Video_Generation_with_Precise_Camera_Control_CVPR_2025_paper.html) | 학회 논문 |
| 12 | **비교 논문:** ViewCrafter: Taming Video Diffusion Models for High-fidelity Novel View Synthesis (Yu et al., arXiv:2409.02048, 2024) | 관련 연구 |
| 13 | **비교 논문:** MotionCtrl: A Unified and Flexible Motion Controller for Video Generation (Wang et al., SIGGRAPH 2024) | 관련 연구 |
| 14 | **비교 논문:** Stable Virtual Camera (SEVA): Generative View Synthesis with Diffusion Models (2025) | 관련 연구 |
| 15 | **비교 논문:** CameraCtrl: Enabling Camera Control for Text-to-Video Generation (He et al., 2024) | 관련 연구 |
| 16 | SemanticNVS: Improving Semantic Scene Understanding in Generative Novel View Synthesis (arXiv:2602.20079) | 후속/관련 연구 |
| 17 | OmniCam: Unified Multimodal Video Generation via Camera Control (arXiv:2504.02312) | 관련 연구 |
| 18 | CamPVG: Camera-Controlled Panoramic Video Generation with Epipolar-Aware Diffusion (arXiv:2509.19979) | 관련 연구 |

---

> ⚠️ **정확도 고지:** 본 답변은 공개된 arXiv 논문(2503.03751), CVPR 2025 공개 논문, 공식 GitHub 저장소, HuggingFace 모델 카드 및 NVIDIA 공식 프로젝트 페이지를 기반으로 작성되었습니다. 수식의 세부 표기는 논문 본문에서 직접 확인할 것을 권장합니다. 논문 전문은 [CVPR 2025 Open Access PDF](https://openaccess.thecvf.com/content/CVPR2025/papers/Ren_GEN3C_3D-Informed_World-Consistent_Video_Generation_with_Precise_Camera_Control_CVPR_2025_paper.pdf)에서 열람 가능합니다.
