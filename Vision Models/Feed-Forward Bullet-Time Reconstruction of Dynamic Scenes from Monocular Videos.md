
# Feed-Forward Bullet-Time Reconstruction of Dynamic Scenes from Monocular Videos

> **논문 정보**
> - **제목:** Feed-Forward Bullet-Time Reconstruction of Dynamic Scenes from Monocular Videos
> - **저자:** Hanxue Liang, Jiawei Ren, Ashkan Mirzaei, Antonio Torralba, Ziwei Liu, Igor Gilitschenski, Sanja Fidler, Cengiz Oztireli, Huan Ling, Zan Gojcic, Jiahui Huang
> - **소속:** NVIDIA, University of Cambridge, NTU, University of Toronto, MIT, Vector Institute
> - **발표:** arXiv:2412.03526 (Dec 2024) / **NeurIPS 2025** 채택
> - **프로젝트 페이지:** https://research.nvidia.com/labs/toronto-ai/bullet-timer/

---

## 1. 핵심 주장 및 주요 기여 요약

최근 정적 장면(static scene)의 feed-forward 재구성 분야에서는 고품질의 새로운 시점 합성(Novel View Synthesis)에서 상당한 진전이 있었다. 그러나 이러한 모델들은 다양한 환경에 대한 일반화 능력이 부족하고, 동적 콘텐츠를 효과적으로 처리하지 못하는 문제가 있었다.

이 논문의 **핵심 주장**은 다음과 같다:

**BTimer**(BulletTimer의 약어)는 동적 장면의 실시간 재구성 및 새로운 시점 합성을 위한 **최초의 motion-aware feed-forward 모델**이다. 이 접근법은 모든 컨텍스트 프레임으로부터 정보를 집계하여 주어진 목표('bullet') 타임스탬프에서의 전체 장면을 3D Gaussian Splatting(3DGS) 표현으로 재구성한다. 이러한 공식화 덕분에 BTimer는 정적 및 동적 장면 데이터셋을 모두 활용하여 확장성(scalability)과 일반화(generalization)를 확보한다. 단순한 모노큘러 동영상 입력으로부터 BTimer는 **150ms 이내**에 bullet-time 장면을 재구성하며, 최적화 기반 접근법과 비교해도 정적·동적 장면 데이터셋 모두에서 **최고 수준의 성능**을 달성한다.

---

## 2. 논문 상세 분석

### 2-1. 해결하고자 하는 문제

feed-forward 재구성 모델은 정적 장면에서는 발전해 왔지만, 동적 장면으로의 확장은 여전히 어렵다. 기존 방법들은 구하기 어려운 일관된 비디오 깊이(video depth)를 입력으로 요구하거나, 렌더링을 지원하지 않거나, 객체 규모의 데이터에서만 동작한다. 반면 BTimer는 모노큘러 동영상에서 동적 장면을 완전히 feed-forward 방식으로 재구성하며, 임의의 시점과 타임스탬프에서 렌더링이 가능하다.

핵심적으로 해결하는 문제는 세 가지이다:
1. **동적 콘텐츠 처리 불가** - 기존 feed-forward 정적 모델의 한계
2. **최적화 기반 방법의 느린 속도** - 장면별 per-scene optimization의 시간 비용
3. **일반화 능력의 부재** - 새로운 도메인/장면에 적용 불가

---

### 2-2. 제안하는 방법 및 수식

#### (1) Bullet-Time 공식화 (Core Formulation)

핵심 아이디어는 단순하면서도 효과적이다: 컨텍스트(입력) 프레임에 **bullet-time embedding**을 추가하여 출력 3DGS 표현에 원하는 타임스탬프를 지정한다. 모델은 컨텍스트 프레임들의 예측을 집계하여 해당 bullet 타임스탬프에서의 장면을 반영하도록 학습된다.

모노큘러 비디오 $\{I_1, I_2, \ldots, I_N\}$과 카메라 포즈 $\{P_1, P_2, \ldots, P_N\}$, 타임스탬프 집합 $T = \{t_1, t_2, \ldots, t_N\}$이 주어질 때, BTimer는 **목표 bullet 타임스탬프** $t_b \in T$에서 3DGS 표현을 직접 예측한다:

$$\hat{\mathcal{G}}_{t_b} = f_\theta\bigl(\{I_i, P_i, t_i\}_{i=1}^{N},\; t_b\bigr)$$

여기서 $f_\theta$는 파라미터 $\theta$를 가진 BTimer Transformer 모델이며, $\hat{\mathcal{G}}_{t_b}$는 bullet 타임스탬프에서의 3D Gaussian 집합이다.

각 3D Gaussian $g_k$는 다음 속성으로 정의된다:

$$g_k = \{\mu_k \in \mathbb{R}^3,\; \Sigma_k \in \mathbb{R}^{3\times3},\; \alpha_k \in [0,1],\; c_k \in \mathbb{R}^3\}$$

여기서 $\mu_k$는 위치, $\Sigma_k$는 공분산 행렬(스케일 $S_k$와 회전 $R_k$로 분해: $\Sigma_k = R_k S_k S_k^T R_k^T$), $\alpha_k$는 불투명도, $c_k$는 색상이다.

#### (2) 입력 임베딩 (Plücker Embedding + Timestamp Embedding)

모델은 컨텍스트 프레임의 시퀀스와 그들의 **Plücker 임베딩**, 그리고 컨텍스트 타임스탬프 및 목표('bullet') 타임스탬프 임베딩을 입력으로 받아, bullet 타임스탬프에서의 3DGS 표현을 직접 예측한다.

Plücker 임베딩은 카메라 레이를 표현하는 데 사용된다. 레이의 원점 $\mathbf{o}$와 방향 $\mathbf{d}$에 대해:

$$\pi(\mathbf{o}, \mathbf{d}) = (\mathbf{d},\; \mathbf{o} \times \mathbf{d}) \in \mathbb{R}^6$$

타임스탬프 임베딩은 컨텍스트 타임스탬프 $t_i$와 bullet 타임스탬프 $t_b$에 대해 각각 생성되며, 모델이 시간적 맥락을 인지하도록 한다:

$$\mathbf{e}_{t} = \text{SinusoidalEmbed}(t) \in \mathbb{R}^{d}$$

#### (3) 손실 함수 (Loss Function)

기본 손실 함수는 3DGS 출력에서 렌더링된 이미지와 실제(ground truth) RGB 이미지 간의 픽셀별 차이를 측정하는 **MSE(Mean Squared Error)**를 포함한다. 시각적 품질 향상을 위한 **보조 지각 손실(LPIPS)**도 적용된다. 이 설계는 종종 획득하기 어려운 3D 지상 진실 데이터를 엄격하게 요구하지 않고 학습할 수 있게 한다.

학습 손실은 다음과 같이 표현된다:

$$\mathcal{L} = \mathcal{L}_{\text{MSE}} + \lambda_{\text{LPIPS}} \cdot \mathcal{L}_{\text{LPIPS}}$$

여기서 렌더링 이미지 $\hat{I}$와 타겟 이미지 $I^*$에 대한 MSE 손실은:

$$\mathcal{L}_{\text{MSE}} = \frac{1}{HW}\sum_{p}\|\hat{I}(p) - I^*(p)\|^2$$

또한 **interpolation supervision**이 중요하다. interpolation loss 없이 훈련하면 모델이 손실을 속이기 위해 카메라에 너무 가까운 낮은 깊이 값의 3DGS를 생성하는 경향이 있다. 반면, interpolation supervision을 추가하면 모델이 장면 다이나믹스를 고려하도록 강제된다.

---

### 2-3. 모델 구조

#### 전체 파이프라인

BTimer의 전체 파이프라인은 다음과 같다:

```
모노큘러 비디오 입력
    ↓
[컨텍스트 프레임 + Plücker 임베딩 + 타임스탬프 임베딩]
    ↓
[ViT 기반 Transformer 백본]
    ↓
[pixel-aligned 3DGS 파라미터 예측]
    ↓
[3DGS 래스터라이제이션 → 새로운 시점 렌더링]
```

#### NTE (Novel Time Enhancer) 모듈

빠른 움직임이 있는 경우, BTimer는 주 모델에 프레임을 입력하기 전에 중간 프레임을 예측하는 **Novel Time Enhancer(NTE)** 모듈을 추가로 도입한다.

NTE는 BTimer 모델과 동일한 ViT 아키텍처를 사용하지만, 입력 컨텍스트 토큰의 시간 특성은 해당 컨텍스트 타임스탬프만 인코딩한다. 빠른 추론을 위해 KV-Cache 전략이 사용된다. Transformer 백본의 출력에서 target 토큰만 유지하며, 이를 unpatchify하고 단일 선형 레이어를 통해 원본 이미지 해상도의 RGB 값으로 프로젝션한다.

BTimer가 $t_b \notin T$ (즉, bullet 타임스탬프가 입력 컨텍스트 프레임에 없을 때)인 경우:

$t_b \notin T$인 경우, 먼저 NTE를 사용하여 $t_b$에서의 $I_b$를 합성한다. 이때 목표 포즈 $P_b$는 $P$ 내의 인접 컨텍스트 포즈에서 선형 보간되고, 컨텍스트 프레임은 $t_b$에 가장 가까운 프레임으로 선택된다.

$$I_b = \text{NTE}\bigl(\{I_i, t_i\}_{i \in \mathcal{N}(t_b)},\; t_b,\; P_b\bigr)$$

여기서 $\mathcal{N}(t_b)$는 $t_b$에 가장 가까운 컨텍스트 프레임 인덱스 집합이다.

#### 커리큘럼 학습 전략 (Curriculum Training)

**Stage 1: 저해상도 → 고해상도 정적 사전학습.** 이 단계에서는 타임 임베딩이 사용되지 않는다. 데이터셋은 객체 중심 데이터(Objaverse)와 실내/실외 장면(RE10K, MVImgNet, DL3DV)을 포함한다.

전체 학습 단계는 다음과 같다:

| 단계 | 내용 | 해상도 | 주요 데이터셋 |
|------|------|--------|--------------|
| Stage 1a | 정적 장면 저해상도 사전학습 | $128^2$ | Objaverse, RE10K |
| Stage 1b | 정적 장면 고해상도 학습 | $256^2 \to 512^2$ | MVImgNet, DL3DV |
| Stage 2 | 정적 장면 co-training 포함 동적 장면 학습 | $256^2$ | 동적+정적 데이터 |
| Stage 3 | 동적 장면 fine-tuning | $256^2$ | 동적 데이터셋 |

BTimer의 학습 반복 횟수는 Stage 1의 세 단계($128^2$, $256^2$, $512^2$ 해상도)에서 각각 90K, 90K, 50K으로 고정되며, Stage 2와 Stage 3 동적 장면 학습에서는 각각 10K, 5K이다. 초기 학습률은 세 단계에서 $4 \times 10^{-4}$, $2 \times 10^{-4}$, $1 \times 10^{-4}$이며, 코사인 어닐링 스케줄로 학습률을 0까지 부드럽게 감소시킨다. 학습은 **32개의 NVIDIA A100 GPU**에서 진행된다.

---

### 2-4. 성능 향상

BTimer는 최적화 기반 접근법과 비교하여 매우 경쟁력 있는 성능을 달성하며, SSIM과 LPIPS 지표에서 2위를 기록한다. 일관된 깊이 추정 없이도 PGDVS를 3개 지표 모두에서 능가한다. 이는 더 선명한 세부 사항을 제공할 수 있는 모델의 효율성과 강력한 일반화 능력을 보여준다.

모델은 이전 최적화 기반 방법들과 경쟁하거나 초월하는 성능을 보여주며, PSNR 기준으로 모든 기준선 중 3위를 차지한다. 명시적 3DGS 기반 표현과 비교하여 PSNR에서 5% 향상된 25.82dB를 달성한다.

모델은 복잡한 움직임을 가진 다양한 객체에 대해 고품질의 선명한 렌더링을 생성하면서 강력한 시간적·다시점 일관성을 유지하며 실제 환경 캡처에서 강력한 일반화 능력을 보여준다. 깊이 맵 또한 bullet-time 재구성 공식화를 통해 복구할 수 있는 올바른 기하학을 보여준다.

---

### 2-5. 한계점

비록 이 방법이 경쟁력 있는 새로운 시점 합성 결과를 제공하지만, 복원된 기하학(따라서 깊이 맵)은 최근의 깊이 예측 모델만큼 정확하지 않은 경우가 많다. 또한 모델은 시점 외삽(view extrapolation)에 대한 지원이 제한적이다. 루프에 생성적 사전(generative prior)을 통합하는 것이 미래의 유망한 방향이다.

논문은 BTimer가 새로운 시점 합성에서 탁월하지만, pixel-aligned 3D Gaussian 공식화로 인해 복원된 기하학이 정확도가 부족할 수 있음을 인정한다. 또한 시간적 변형(temporal deformation)이 명시적으로 모델링되지 않으며, 움직임 추출을 위한 추가 후처리가 필요할 수 있다고 언급한다.

---

## 3. 모델의 일반화 성능 향상 가능성

일반화 성능은 BTimer의 핵심 강점이자 핵심 설계 목표이다.

### 3-1. Bullet-Time 공식화의 일반화 기여

이 설계는 정적 및 동적 재구성 시나리오를 자연스럽게 통합할 뿐 아니라, 모델이 장면 다이나믹스를 캡처하는 과정에서 암묵적으로 motion-aware해지게 한다. 특히 제안된 공식화는 (i) 대량의 정적 장면 데이터로 모델을 사전학습 가능, (ii) 입력 비디오의 길이와 프레임 속도에 제약받지 않고 데이터셋 전반에 효과적으로 확장, (iii) 다중 시점을 본질적으로 지원하는 부피적 비디오 표현을 출력할 수 있다.

### 3-2. 정적 데이터셋 활용을 통한 일반화

모델의 일반화 능력은 데이터 다양성에 의해 크게 결정된다. Bullet-time 재구성 공식화가 정적(모든 $T$ 원소를 동일하게 설정) 및 동적 장면을 자연스럽게 지원하고, 약한 감독을 위한 RGB 손실만 필요로 하기 때문에 수많은 정적 데이터셋의 가용성을 활용하여 모델을 사전학습할 잠재력을 열어준다.

### 3-3. 혼합 데이터셋(Mixed-Dataset) 학습의 효과

실제 환경 테스트를 위해 DAVIS 데이터셋의 모노큘러 비디오를 사용하고, 커스터마이즈된 파이프라인으로 카메라 포즈를 추정하여 사용한다. 단일 데이터셋(Ours-Static)보다 혼합 데이터셋(Ours-Full)을 사용하는 것이 일반화 및 성능에 훨씬 더 좋다.

### 3-4. 커리큘럼 학습과 실세계 데이터 활용

4D 동적 데이터셋이 부족하기 때문에 안정적인 학습을 위해 정적 데이터셋을 함께 co-training으로 사용하고, 인터넷 비디오로부터 카메라 포즈를 추정하는 커스터마이즈 파이프라인을 구축하여 실제 환경 데이터에 대한 강건성을 향상시킨다.

### 3-5. 일반화를 위한 개선 방향

현재 BTimer는 다음과 같은 이유로 일반화 한계를 가진다:

1. **View extrapolation 불가**: 모델은 시점 외삽에 대한 지원이 제한적이며, 생성적 사전(generative prior)을 통합하는 것이 미래의 유망한 방향이다.
2. **픽셀 정렬 3DGS의 기하학 정확도 한계**: 깊이 정보가 정확하지 않을 수 있다.
3. **end-to-end 미학습**: BTimer와 NTE 모듈이 별도로 훈련되어 있어 통합 최적화가 불가능하다.

---

## 4. 관련 최신 연구 비교 분석 (2020년 이후)

| 방법 | 연도 | 유형 | 표현 | 주요 특징 | 한계 |
|------|------|------|------|----------|------|
| **NeRF** (Mildenhall et al.) | 2020 | 최적화 | implicit MLP | 정적 장면 고품질 합성의 선구적 연구 | 느린 학습/렌더링, 동적 장면 미지원 |
| **Nerfies / HyperNeRF** | 2021 | 최적화 | deformable NeRF | 변형 가능한 장면 처리 | per-scene 최적화 필요, 느린 속도 |
| **D-NeRF / TiNeuVox** | 2021-22 | 최적화 | 시간 조건부 NeRF | 동적 장면 NeRF 일반화 | 합성 데이터 중심 |
| **4DGS (Wu et al.)** | 2024 | 최적화 | 4D Gaussian primitives | 4D 볼륨을 4D primitives로 근사하여 임의 시간의 새로운 시점 합성 가능 | per-scene 최적화 필요 |
| **Deformable 3D Gaussians** | 2024 | 최적화 | deformable 3DGS | 변형 가능한 Gaussian으로 고충실도 재구성 | per-scene 최적화, 일반화 부족 |
| **Shape of Motion** | 2024 | 최적화 | 3DGS + 3D motion | 단일 비디오 4D 재구성 | per-scene 최적화 |
| **MoSca** | 2024 | 최적화 | 4D Motion Scaffolds | 동적 장면의 모션/변형을 컴팩트하게 인코딩하는 Motion Scaffold 표현 활용 | per-scene 최적화 |
| **pixelSplat / MVSplat** | 2024 | feed-forward | 3DGS | 희소 멀티뷰 이미지에서 피드포워드 3DGS | 정적 장면에만 적용 가능 |
| **BTimer (본 논문)** | 2024 | **feed-forward** | **3DGS** | **최초 동적 scene feed-forward, 150ms 추론** | 기하 정확도 한계, view extrapolation 불가 |
| **Lyra** | 2025 | generative+feed-forward | 3DGS | 정적·동적 3D 장면 생성에서 최고 성능 달성, PSNR 21 이상, BTimer 대비 향상된 LPIPS | 생성 모델 기반으로 복잡성 증가 |

초기 동적 NeRF 방법들은 주로 시간에 따라 변화하는 NeRF나, 변형 필드가 있는 canonical 공간 NeRF로 4D 장면을 표현했다. 최근 방법들은 효율적인 3DGS 표현을 동적 장면으로 확장하기 시작했다. 이들은 Gaussian 변형 학습, 모션 궤적, 또는 직접적인 4D Gaussian primitives를 통해 동적 콘텐츠를 모델링한다.

---

## 5. 향후 연구에 미치는 영향 및 고려할 점

### 5-1. 향후 연구에 미치는 영향

#### (1) Feed-Forward 패러다임의 동적 장면으로의 확장 가능성 증명
이 설계는 정적 및 동적 재구성 시나리오를 자연스럽게 통합할 뿐 아니라, 장면 다이나믹스를 캡처하는 과정에서 암묵적으로 motion-aware해지도록 한다. 이는 기존에는 최적화 기반 접근이 필수적이었던 동적 장면 재구성 분야에서, 대용량 데이터 학습을 통한 일반화 모델 구축이 가능함을 실증적으로 보여주었다.

#### (2) 정적-동적 데이터 통합 학습의 유효성
제안된 공식화는 (i) 대규모 정적 장면 데이터로 사전학습이 가능하고, (ii) 입력 비디오의 길이와 프레임 속도에 제약 없이 데이터셋 전반에 효과적으로 확장되며, (iii) 본질적으로 다중 시점을 지원하는 부피적 비디오 표현을 출력한다. 이는 향후 **4D 비전 기초 모델(4D Vision Foundation Model)** 구축의 밑거름이 될 수 있다.

#### (3) 실시간 동적 3D 재구성의 응용 확대
BTimer는 모노큘러 비디오 입력에서 실시간 동적 장면 재구성 및 새로운 시점 합성의 발전에서 중요한 발걸음을 나타낸다. 이 논문의 bullet-time embedding, NTE 모듈, 종합적인 학습 전략은 컴퓨터 비전 분야에서 효율성과 효과성을 재정의하며 기존 방법론에서 직면했던 핵심 도전 과제를 해결한다.

### 5-2. 향후 연구 시 고려할 점

#### ① 기하학 정확도 개선
BTimer는 새로운 시점 합성에서 탁월하지만, pixel-aligned 3D Gaussian 공식화로 인해 복원된 기하학의 정확도가 부족할 수 있다. 향후 연구에서는 깊이 예측 모델이나 명시적 기하 지도를 통합하여 정확한 3D 기하학을 복원하는 방법을 탐구해야 한다. 예를 들어, **MonST3R**, **MegaSaM** 등 강력한 동적 포즈 추정 모델과의 결합을 고려할 수 있다.

#### ② View Extrapolation 지원
현재 모델은 시점 외삽에 대한 지원이 제한적이다. 루프에 생성적 사전(generative prior)을 통합하는 것이 미래의 유망한 방향이다. **Diffusion 기반 생성 모델**을 통합하여 미관측 영역의 인페인팅(inpainting)을 가능하게 하는 방향이 유망하다.

#### ③ End-to-End 통합 학습
BTimer 본체와 NTE 모듈은 현재 별도로 학습된다. 두 모듈을 end-to-end로 공동 학습(joint training)하면 모션 처리와 재구성 품질을 더욱 향상시킬 가능성이 있다.

#### ④ 고해상도 및 대규모 장면으로의 확장
현재 feed-forward 추론은 12개 컨텍스트 프레임의 $256 \times 256$ 해상도에서만 150ms가 소요되고, 출력 3DGS는 실시간으로 렌더링될 수 있다. 더 높은 해상도($512^2$ 이상)와 더 많은 컨텍스트 프레임으로의 확장은 메모리와 계산 비용 문제를 야기하므로, 효율적인 어텐션 메커니즘(Flash Attention, FlexAttention 등)과 시퀀스 병렬화(Sequence Parallelism) 기법이 핵심적으로 고려되어야 한다.

#### ⑤ 더욱 다양한 도메인 데이터의 통합
모델의 일반화 능력은 데이터 다양성에 의해 크게 결정된다. Bullet-time 재구성 공식화는 정적·동적 장면을 자연스럽게 지원하고 RGB 손실만 필요로 하므로, 수많은 정적 데이터셋의 잠재력을 활용할 수 있다. 의료 영상, 자율주행, 로봇공학 등 특수 도메인 데이터를 포함하여 더 광범위한 일반화를 추구해야 한다.

#### ⑥ 명시적 모션 표현의 통합
시간적 변형(temporal deformation)이 명시적으로 모델링되지 않으며, 움직임 추출을 위한 추가 후처리가 필요할 수 있다. 향후 연구에서는 명시적인 모션 플로우(optical flow) 또는 3D 트래킹 정보를 모델에 통합하여 물리적으로 일관된 동적 재구성을 달성해야 한다.

---

## 📚 참고 자료 및 출처

1. **arXiv 논문 원문:** Liang et al., "Feed-Forward Bullet-Time Reconstruction of Dynamic Scenes from Monocular Videos," arXiv:2412.03526, Dec. 2024.
   - URL: https://arxiv.org/abs/2412.03526

2. **NVIDIA 공식 프로젝트 페이지:** https://research.nvidia.com/labs/toronto-ai/bullet-timer/

3. **OpenReview (NeurIPS 2025):** https://openreview.net/forum?id=oGc1qHAUBJ

4. **NeurIPS 2025 포스터:** https://neurips.cc/virtual/2025/poster/116056

5. **논문 PDF (NVIDIA):** https://research.nvidia.com/labs/toronto-ai/bullet-timer/files/btimer.pdf

6. **논문 PDF (OpenReview):** https://openreview.net/pdf/939a4ac9d46466a3f2b1feecd059aa6c78969756.pdf

7. **저자 개인 페이지:** https://huangjh-pub.github.io/publication/btimer/

8. **문헌 리뷰 (Moonlight):** https://www.themoonlight.io/en/review/feed-forward-bullet-time-reconstruction-of-dynamic-scenes-from-monocular-videos

9. **블로그 분석 (Semyeong Yu):** https://semyeong-yu.github.io/blog/2025/BTimer/

10. **관련 연구 (4DGS, ICLR 2024):** Wu et al., "Real-time Photorealistic Dynamic Scene Representation and Rendering with 4D Gaussian Splatting," ICLR 2024.

11. **관련 연구 (Lyra, 2025):** https://www.emergentmind.com/topics/lyra-generative-3d-scene-reconstruction

12. **관련 연구 (4D3R, 2025):** https://arxiv.org/html/2511.05229
