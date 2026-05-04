# DiffusionRenderer: Neural Inverse and Forward Rendering with Video Diffusion Models

DiffusionRenderer 논문의 핵심 주장은 “비디오 확산 모델을 이용해 역렌더링(G‑buffer 추출)과 정방향 렌더링(광 전달 시뮬레이션)을 하나의 통합 프레임워크로 해결하고, 명시적인 경로 추적이나 정확한 3D 기하 없이도 실세계 영상에서 고품질 재조명·재질 편집·객체 삽입을 가능하게 한다”는 것입니다.[^1][^2][^3]
이를 위해 저자들은 (1) 비디오 역렌더러로 RGB→G‑buffer를 추정하고, (2) 이를 이용해 대규모 실세계 비디오를 자동 라벨링한 뒤, (3) 합성+실세계 데이터를 함께 사용하는 비디오 정방향 렌더러(G‑buffer→RGB)를 학습하여 기존 PBR·NeRF·2D 확산 기반 방법들을 PSNR/SSIM/LPIPS 등에서 상회하는 결과를 보입니다.[^3][^4][^1]

***

## 논문의 핵심 주장

- 명시적인 경로 추적 없이 G‑buffer(법선, 깊이, 알베도, 거칠기, 금속도)와 환경맵만으로 실제 PBR 렌더러에 근접한 그림자·반사·간접조명을 합성하는 “신경 렌더러”를 비디오 확산 모델로 구현합니다.[^1][^3]
- 별도의 비디오 역렌더러를 먼저 합성 데이터만으로 학습한 후, DL3DV‑10K 같은 실세계 비디오에 적용해 G‑buffer를 자동 추출(pseudo‑label)하고, 이를 정방향 렌더러의 학습에 활용함으로써 합성–실세계 도메인 갭을 줄입니다.[^4][^1]
- 하나의 프레임/비디오 입력만으로 재조명, 재질 편집, 가상 객체 삽입 등을 수행하면서도 시간적 일관성이 좋은 비디오 결과를 생성하는 점을 강조합니다.[^3][^1]

***

## 해결하려는 문제

1. **고전 PBR·NeRF 기반 렌더링의 한계**
    - 정확한 메쉬, 고품질 재질 파라미터, HDR 환경 조명 등 정밀한 3D 장면 기술이 필요하며, 실세계 응용(모바일 AR/VR, 영상 편집)에서는 이를 얻기 어렵습니다.[^2][^1]
    - NeRF·3D Gaussian Splatting 기반 역렌더링/재조명은 주로 정적 장면, 다중 뷰, 장시간 per‑scene 최적화에 의존하고, 편집에는 제약이 크며 동적/복잡 장면에서는 불안정합니다.[^5][^1]
2. **기존 확산 기반 역/정방향 렌더링의 한계**
    - RGB↔X, Intrinsic Image Diffusion, Neural Gaffer, DiLightNet 등은 주로 단일 이미지에 초점을 맞추고 있어 시간 일관성과 비디오 편집에 제한이 있습니다.[^6][^7][^8][^9][^1]
    - 많은 방법이 특정 도메인(실내, 인물, 단일 객체)에 특화되어 있고, 합성/실세계 데이터의 도메인 갭으로 인해 실세계 일반화가 제한됩니다.[^10][^11][^1]

**DiffusionRenderer의 목표**는 “하나의 비디오 확산 모델 프레임워크로 역렌더링과 신경 정방향 렌더링을 동시에 해결하면서, 합성·실세계 비디오에 걸쳐 잘 일반화되는 실용적인 영상 편집 엔진을 만드는 것”입니다.[^1][^3]

***

## 제안 방법 개요

### PBR 렌더링 식

고전적 PBR에서 표면점 $p$ 방향 $\omega_o$로의 출력 복사 휘도는 다음과 같이 정의됩니다.[^1]

$$
L_o(p, \omega_o) = \int_{\Omega} f_r(p, \omega_o, \omega_i)\, L_i(p, \omega_i)\, \lvert n \cdot \omega_i \rvert \, d\omega_i \quad (1)
$$

여기서 $f_r$는 BRDF, $L_i$는 입사 복사 휘도, $n$은 법선, $\Omega$는 반구입니다.[^1]

DiffusionRenderer는 이 적분을 몬테카를로로 직접 풀지 않고, **비디오 확산 모델 $f_\theta$** 가 G‑buffer와 환경맵을 조건으로 받아 $L_o$에 상응하는 RGB 비디오를 생성하도록 학습함으로써 “데이터 기반 근사”를 수행합니다.[^3][^1]

### 비디오 확산 모델의 기본

Stable Video Diffusion 기반으로, VAE 인코더 $E$와 디코더 $D$, UNet 구조의 비디오 확산 네트워크 $f_\theta$를 사용합니다.[^10][^1]

- 입력 RGB 비디오 $I \in \mathbb{R}^{F \times H \times W \times 3}$를 잠복표현 $z = E(I)$로 인코딩합니다.[^1]
- 특정 시간 스텝 $\tau$에서 노이즈가 추가된 잠복표현은

$$
z_\tau = \alpha_\tau z_0 + \sigma_\tau \epsilon
$$

형태로 주어지고, $\epsilon \sim \mathcal{N}(0, I)$입니다.[^1]
- 학습 시 denoising score matching 손실을 사용해 $f_\theta$가 $\epsilon$ 또는 노이즈 없는 잠복표현을 예측하도록 최적화합니다.[^12][^1]

이 기본 틀 위에, 역렌더러와 정방향 렌더러를 각각 조건부 비디오 확산 모델로 구성합니다.[^3][^1]

***

## 수식과 학습 목표

### 역렌더링(영상 → G‑buffer)

역렌더링 모델은 RGB 비디오 $I$를 조건으로 받아, G‑buffer 속성(법선 $n$, 깊이 $d$, 알베도/베이스컬러 $a$, 거칠기 $r$, 금속도 $m$)를 추정합니다.[^1]

1. VAE 인코더로 입력 비디오 인코딩

$$
z = E(I)
$$

로 RGB 잠복표현을 얻습니다.[^1]
2. 대상 속성 $P \in \{n, d, a, r, m\}$에 대해, 해당 GT 버퍼 $s_P$를 인코딩하여

$$
g^P_0 := E(s_P)
$$

를 얻고, 여기에 노이즈를 추가해

$$
g^P_\tau = \alpha_\tau g^P_0 + \sigma_\tau \epsilon
$$

를 구성합니다.[^1]
3. 각 속성마다 하나의 **도메인 임베딩 $c^P_{\text{emb}}$** 를 cross‑attention 조건으로 넣어, 같은 UNet이 속성별로 다른 맵을 생성하도록 합니다.[^1]
4. 학습 목적 함수는

$$
L(\theta, c_{\text{emb}}) =
\left\| f_\theta\big(g^P_\tau; z, c^P_{\text{emb}}, \tau\big) - g^P_0 \right\|_2^2
\quad (3)
$$

로 주어지며, 여기서 $f_\theta$는 조건부 비디오 확산 UNet입니다.[^1]

이렇게 학습된 역렌더러는 합성 데이터에만 학습되지만, InteriorVerse 및 DL3DV‑10K 등 실세계 데이터에서도 높은 정밀도(특히 금속도·거칠기 RMSE 감소, 법선 각도 오차 감소)를 보이며 잘 일반화됩니다.[^1]

### 정방향 렌더링(G‑buffer → RGB)

정방향 렌더러는 G‑buffer $\{n,d,a,r,m\}$와 환경맵 $E$를 조건으로 RGB 비디오를 합성하는 모델입니다.[^3][^1]

1. G‑buffer 인코딩
    - 각 채널을 VAE 인코더 $E$로 인코딩해,

$$
g = \{E(n), E(d), E(a), E(r), E(m)\} \in \mathbb{R}^{F \times h \times w \times 20}
$$

과 같은 픽셀 정렬(latent) G‑buffer를 얻습니다.[^1]
2. 환경맵 인코딩
    - HDR 환경맵 $E$에 대해 Reinhard tone‑mapping 및 로그 스케일링을 적용해

$$
E_{\text{log}} = \log(E + 1) / E_{\max}
$$

를 계산하고, LDR $E_{\text{ldr}}$, 방향 인코딩 $E_{\text{dir}}$을 함께 사용합니다.[^1]
    - 이 세 장을 VAE 인코더에 통과시켜 $h_E$, 별도의 환경맵 인코더 $E_{\text{env}}$가 다중 해상도 feature $\{h^i_{\text{env}}\}_{i=1}^K$를 생성합니다.[^1]
3. 손실 함수
    - 합성(synth) 데이터와 실세계(real) 자동 라벨 데이터를 동시에 사용하는 공동 학습 손실은

$$
\begin{aligned}
L(\theta, \Delta\theta) &= 
\left\| f_\theta\big(z^{\text{synth}}_\tau; g^{\text{synth}}, c^{\text{synth}}_{\text{env}}, \tau\big) - z^{\text{synth}}_0 \right\|_2^2 \\
&\quad+
\left\| f_{\theta+\Delta\theta}\big(z^{\text{real}}_\tau; g^{\text{real}}, c^{\text{real}}_{\text{env}}, \tau\big) - z^{\text{real}}_0 \right\|_2^2
\end{aligned}
\quad (4)
$$

로 정의되며, $\Delta\theta$는 LoRA 형태의 소수 파라미터로 실세계 도메인에 적응합니다.[^1]

여기서 중요한 점은, **합성 데이터에서의 정확한 감독과, 실세계 자동 라벨의 다양성을 하나의 비디오 확산 프레임워크 안에서 통합**한다는 것입니다.[^4][^1]

***

## 모델 구조

### 공통 기반: Stable Video Diffusion

- 기본 백본은 Stable Video Diffusion이며, 공간 방향만 압축하고 시간 방향 $F$는 유지하는 잠복 비디오 표현을 사용합니다.[^10][^1]
- UNet은 다단계 self‑attention과 cross‑attention으로 구성되어 있으며, cross‑attention은 원래 텍스트/CLIP 조건에 사용되던 블록을 **조명/도메인 임베딩** 수용용으로 재활용합니다.[^3][^1]


### 역렌더러 구조

- 입력: RGB 비디오 잠복표현 $z = E(I)$, 노이즈가 섞인 G‑buffer 잠복표현 $g^P_\tau$, 도메인 임베딩 $c^P_{\text{emb}}$.[^1]
- 각 속성(법선, 깊이, 알베도, 거칠기, 금속도)은 동일한 네트워크를 여러 번 호출해 순차적으로 생성되며, 각 호출마다 다른 $c^P_{\text{emb}}$를 사용합니다.[^1]
- 20‑step 다단계 확산 샘플링과, 별도로 fine‑tuning한 1‑step deterministic 버전(속도·PSNR 향상)을 함께 제안합니다.[^1]


### 정방향 렌더러 구조

- 입력: 노이즈가 섞인 RGB 잠복표현 $z_\tau$, G‑buffer 잠복표현 $g$, 다중 해상도 환경맵 feature $\{h^i_{\text{env}}\}$.[^1]
- G‑buffer 잠복표현은 UNet의 입력 채널에 직접 concatenation(픽셀 정렬 조건)으로 들어가고, 환경맵 feature는 각 계층의 cross‑attention 쿼리/키/값으로 주입됩니다.[^1]
- 합성 데이터에는 기본 파라미터 $\theta$, 실세계 데이터에는 $\theta + \Delta\theta$ (LoRA)를 사용해 도메인 차이를 보정합니다.[^1]

***

## 데이터와 학습 전략

1. **합성 데이터 생성**
    - Objaverse LVIS 서브셋(36,500개 3D 자산), 4,260개 PBR 재질, 766개 HDR 환경맵을 사용해 150,000개 비디오(각 24 프레임, 512²)를 커스텀 OptiX path tracer로 렌더링합니다.[^1]
    - 각 장면은 평면+3D 객체 3개까지+기본 primitive(구·큐브·실린더)를 배치하고, 카메라 궤도/진동, 조명 회전, 객체 회전/이동 등 다양한 모션을 포함합니다.[^1]
2. **실세계 데이터 자동 라벨링**
    - 합성 데이터로 학습한 역렌더러를 DL3DV‑10K 실세계 비디오(10,510개)에 적용하여 G‑buffer를 추정하고, DiffusionLight로 환경맵을 추정하여 총 약 150,000개 실세계 비디오 샘플의 pseudo‑label을 만듭니다.[^13][^1]
3. **공동 학습 및 정규화**
    - 역렌더러는 합성+InteriorVerse+HyperSim 데이터로 초기 학습 후, 실세계에 직접 적용합니다.[^1]
    - 정방향 렌더러는 합성+실세계 자동 라벨을 모두 사용하며, 합성 쪽은 정확도, 실세계 쪽은 스타일/도메인 적응을 위해 LoRA 파라미터를 사용합니다.[^1]
    - 조건 채널 dropout(0.1)을 사용해 특정 조건에 과도하게 의존하지 않도록 하고, classifier‑free guidance(1.2)를 정방향 렌더링에 적용해 시각 품질을 향상합니다.[^1]

이 데이터·학습 전략이 **합성–실세계, 이미지–비디오, 단일/복수 객체 장면**에 걸친 일반화 성능의 핵심입니다.[^4][^1]

***

## 성능 향상

### 정방향 렌더링

- SyntheticObjects/Scenes 데이터셋에서 PSNR·SSIM·LPIPS 기준으로 SplitSum, SSRT, RGB↔X, DiLightNet을 일관되게 상회합니다.[^1]
- 복잡한 상호 반사와 그림자가 있는 SyntheticScenes에서, 다른 방법들은 PSNR이 크게 떨어지는 반면 DiffusionRenderer는 감소폭이 2.3dB 정도로 상대적으로 작아 복잡한 조명에 더 잘 대응합니다.[^1]


### 역렌더링

- SyntheticScenes에서 알베도 si‑PSNR/si‑LPIPS 및 금속도·거칠기 RMSE, 법선 각도 오차 등에서 RGB↔X 대비 큰 폭의 개선을 보입니다.[^1]
- InteriorVerse 실내 데이터셋에서도 기존 학습 기반 역렌더링(IIW, Kocsis et al., RGB↔X 등)을 모두 상회하는 알베도 복원 성능을 달성합니다.[^8][^1]


### 재조명

- DiLightNet, Neural Gaffer와 비교 시, 복잡한 하이라이트와 간접광·그림자를 더 정확히 재현하며 PSNR/SSIM/LPIPS 및 ColorVideoVDP(JOD)에서 높은 점수를 기록합니다.[^7][^9][^1]
- 사용자/ GPT‑4V 기반 주관 평가에서도, 기준 path‑traced 영상에 더 가깝다고 선택되는 비율이 높게 나타납니다.[^9][^1]

***

## 한계점

- Stable Video Diffusion 기반이므로 24프레임 512² 비디오를 20스텝 확산으로 처리하는 데 수 초 단위의 시간이 걸려, 실시간·인터랙티브 렌더링에는 부담이 큽니다.[^14][^1]
- 역렌더러가 생성한 G‑buffer 및 DiffusionLight 기반 환경맵은 완벽하지 않아, 실세계 학습 데이터에는 구조적인 노이즈와 편향이 포함됩니다.[^13][^1]
- 편집 시 원본 색감·텍스처를 크게 망가뜨리지는 않지만, 색조·질감이 미묘하게 변하는 artifacts가 발생할 수 있으며, 매우 복잡한 광학 효과(예: 카우스틱, 참여 매질 등)는 아직 제한적입니다.[^1]

***

## 일반화 성능 향상 가능성

논문에서 직접 보여주는 일반화 전략과, 향후 확장 가능성을 나누어 볼 수 있습니다.[^3][^1]

### 논문 내 전략

1. **합성+실세계 공동 학습**
    - 합성 데이터는 정확한 G‑buffer/조명 감독을 제공하고, 실세계 데이터는 장면 다양성과 현실적인 텍스처·노이즈·카메라 동작을 제공합니다.[^1]
    - LoRA를 통해 실세계 도메인에 특화된 적은 수의 파라미터만 조정함으로써, 합성 도메인에서 학습된 물리적 priors를 크게 망가뜨리지 않고도 실세계 스타일에 적응합니다.[^1]
2. **비디오 기반 모델링**
    - 동일 장면을 다양한 뷰에서 관찰하는 비디오 구조를 활용해, 특히 금속도·거칠기처럼 시점 의존성이 큰 속성 추정의 정확도를 높이고, 시간적 일관성을 확보합니다.[^1]
3. **환경맵 오토인코더**
    - 조명 자체를 VAE+전용 auto‑encoder로 모델링하여, HDR의 높은 dynamic range와 방향 정보를 multi‑scale feature로 추출함으로써 다양한 조명 분포에 대한 일반화 능력을 향상시킵니다.[^1]

### 향후 일반화 개선 방향

- **데이터 스케일·다양성 확대**: 현재 합성 데이터는 Select된 Objaverse·PolyHaven 자산과 수백 개 HDRI에 제한되므로, 더 폭넓은 3D 자산·재질·환경맵을 포함한 웹‑스케일 합성 데이터로 확장할 여지가 큽니다.[^15][^1]
- **조명·재질 추정기의 공동 학습**: 현재 실세계 조명은 외부 모델(DiffusionLight)에 의존하며, 이는 오차의 근원이 됩니다.  역·정방향 렌더러와 조명 추정기를 end‑to‑end 공동 학습하면 도메인 갭을 더 줄일 수 있습니다.[^13][^1]
- **더 강력한 비디오 확산 priors 활용**: Stable Video Diffusion 대신, 최신 대규모 비디오 확산/“physical AI” 모델(Cosmos World, FrameDiffuser 등)을 기반으로 distillation·fine‑tuning하면 고해상도·장기 비디오에 더 잘 일반화할 수 있습니다.[^14][^1]
- **1‑step/소수 스텝 모델의 범용화**: 역렌더러에서 보인 1‑step deterministic fine‑tuning 결과를 더 넓은 도메인으로 확장하고, distillation 기법과 결합하면 **실시간에 가까운 범용 신경 렌더러**로 진화할 수 있습니다.[^16][^1]

***

## 2020년 이후 관련 최신 연구 비교

### 3D‑기반 역/정방향 렌더링 (NeRF 계열)

- NeRF 및 후속 작업(NeRF‑Factor, PhySG, IRON, UrbanIR, GS‑IR, GIR 등)은 신경 복사장/3D Gaussian Splatting을 이용해 장면의 기하·재질·조명을 공동으로 복원하고, PBR 기반 재조명을 지원합니다.[^1]
- 장점: 물리적으로 해석 가능한 3D 표현, 뷰 변경·재조명·객체 삽입에 강력.[^1]
- 한계: 다중 뷰·정적 장면 가정, per‑scene 최적화 비용, 복잡한 구조(나무·도시·동적 객체)에 대한 취약성.[^1]
- **DiffusionRenderer와의 차이**: 3D 방식을 버리고 2.5D G‑buffer 기반 비디오 렌더링으로 전환함으로써, 단일 비디오 입력만으로도 빠르게 편집 가능하며, explicit 3D 재구성이 필요 없습니다.[^3][^1]


### 2D 확산 기반 역렌더링

- **Intrinsic Image Diffusion (Kocsis et al. 2023)**, **GeoWizard**, **IntrinsicAnything**, **Luminet** 등은 이미지 확산 모델을 이용해 알베도·법선·조명 등 intrinsic 채널을 추정하는 다양한 방법을 제안합니다.[^10][^1]
- 장점: 강력한 이미지 priors를 활용해 단일 이미지에서도 좋은 intrinsic 분해 성능.[^10]
- 한계: 주로 단일 프레임, 특정 도메인(실내/실외/물체)에 특화, 시간 일관성·비디오 편집에 대한 직접 지원 부족.[^1]
- **DiffusionRenderer와의 차이**: 비디오 확산 모델을 기반으로 하여 시간 일관성·specular 특성 추정이 개선되고, 정방향 렌더러 학습을 위한 실세계 pseudo‑label 생성에 직접 사용됩니다.[^1]


### 2D 확산 기반 재조명/신경 렌더링

- **Neural Gaffer (NeurIPS 2024)**는 단일 객체 이미지를 환경맵 조건으로 재조명하는 end‑to‑end 2D 확산 모델로, 명시적 intrinsic 분해 없이 고품질 재조명을 수행합니다.[^7][^9]
- **DiLightNet (SIGGRAPH 2024)**는 확산 기반 이미지 생성에 세밀한 조명 제어를 결합한 2D 재조명 네트워크입니다.[^9][^1]
- 장점: 단일 이미지 재조명에서 매우 높은 품질, 다양한 2D 편집 응용 지원.[^7][^9]
- 한계: 비디오 시간 일관성과 G‑buffer 기반 편집(재질 수치 조정, 객체 삽입)은 직접적으로 지원하지 않습니다.[^1]
- **DiffusionRenderer와의 차이**: 2D 이미지가 아닌 비디오를 직접 모델링하고, G‑buffer 기반 재질·기하 편집과 객체 삽입을 지원하며, 합성+실세계 데이터에 공동 학습된 신경 렌더링 엔진입니다.[^3][^1]


### RGB↔X: 양방향 이미지 렌더링

- **RGB↔X (SIGGRAPH 2024)**는 실내 장면에 대해 RGB→X(역렌더링)와 X→RGB(정방향 렌더링)를 하나의 2D 확산 프레임워크로 다루며, 이질적인 데이터셋들을 조합해 학습합니다.[^11][^6][^8][^1]
- DiffusionRenderer와 유사하게 “역·정방향 렌더링을 확산 모델로 통합”한다는 철학을 공유하지만, RGB↔X는 이미지 수준, 주로 실내 도메인에 초점을 두고 있습니다.[^11]
- **DiffusionRenderer와의 차이**: 비디오 모델(시간 축)을 사용하고, 대규모 합성 비디오+실세계 비디오를 공동 학습하며, 합성 장면에서 path‑traced ground truth와 직접 비교되는 수준의 신경 렌더링 품질을 목표로 합니다.[^3][^1]


### 최신 후속 연구

- **Photorealistic Object Insertion with Diffusion‑Guided Inverse Rendering**는 대형 개인화 확산 모델을 inverse rendering에 결합해, 단일 이미지에서의 물체 삽입 품질을 크게 향상시킵니다.[^13]
- **FrameDiffuser**는 DiffusionRenderer와 같은 비디오 신경 렌더러의 “오프라인·시퀀스 전체 필요” 한계를 지적하고, G‑buffer와 이전 프레임만으로 autoregressive하게 프레임별 렌더링을 수행해 상호작용 시나리오에 적합한 방식을 제안합니다.[^14]
이들 연구는 DiffusionRenderer 스타일의 신경 렌더링을 “인터랙티브/온라인” 환경으로 확장하려는 흐름으로 볼 수 있습니다.[^14][^13]

***

## 향후 연구에 미치는 영향과 고려 사항

### 연구적 영향

- **신경 렌더링의 패러다임 이동**: NeRF 등 explicit 3D 표현 중심에서, “G‑buffer + 환경맵 + 확산 priors” 기반의 2.5D 신경 렌더링으로의 전환을 촉진하며, PBR/역렌더링/생성 모델을 하나의 확산 프레임워크로 통합하는 방향성을 강화합니다.[^11][^3][^1]
- **합성–실세계 혼합 학습 전략의 정착**: 합성에서 정확한 감독을 얻고, 실세계 비디오를 역렌더러로 자동 라벨링해 정방향 렌더러를 학습하는 pipeline은, 다른 물리 기반 비전 과제(예: 유체, 재료, 기상 시뮬레이션)에도 응용될 수 있는 일반적인 레시피를 제공합니다.[^4][^1]
- **비디오 확산 모델의 새로운 응용**: 텍스트‑비디오 생성 외에 “물리적으로 일관된 조명·재질 편집”이라는 명확한 다운스트림 태스크를 제시함으로써, 비디오 확산 모델의 학습 및 평가 기준을 풍부하게 만듭니다.[^12][^1]


### 앞으로 연구 시 고려할 점

1. **계산 비용과 실시간성**
    - 현재 구조는 오프라인 비디오 편집에는 충분하지만, 게임 엔진·AR에서 요구되는 수십 Hz 수준 실시간 렌더링에는 부적합합니다.[^14][^1]
    - distillation, 1‑step/소수 스텝 샘플링, encoder 재설계(Faster Diffusion 계열) 등을 통해 실시간 G‑buffer→RGB, RGB→G‑buffer를 목표로 하는 연구가 필요합니다.[^16][^14][^1]
2. **라벨 노이즈와 도메인 편향**
    - 실세계 pseudo‑label은 역렌더러와 조명 추정기의 오류를 포함하므로, 이 노이즈가 정방향 렌더러에 어떤 편향을 유도하는지 정량 분석하고, 노이즈‑로버스트 학습, self‑training, confidence weighting 등이 요구됩니다.[^13][^1]
3. **물리적 일관성과 편집 안정성**
    - 현재도 상당한 수준이지만, 복잡한 효과(카우스틱, 참여 매질, 다중 산란)나 매우 고광택 재질에서 확산 모델이 물리적 일관성을 항상 유지하는지는 미지수입니다.[^1]
    - 물리 기반 손실(렌더링 방정식 잔차, 에너지 보존 제약 등)을 결합한 “physics‑guided diffusion” 설계가 중요한 다음 단계가 될 수 있습니다.[^10][^1]
4. **범용 도메인 확장**
    - 실내/실외, 도시/자연, 인물/제품 등 다양한 도메인에 걸쳐 하나의 모델이 안정적으로 작동하도록 데이터 설계와 조건 설계(카테고리 토큰, 장면 타입 토큰 등)를 정교화해야 합니다.[^11][^1]
    - RGB↔X, Neural Gaffer, FrameDiffuser 등과의 아이디어 결합(예: 텍스트 조건, 카메라 트랙 조건, geometry‑aware priors)도 유망합니다.[^6][^7][^14]

***

## 참고한 주요 자료

아래는 본 답변에서 분석·비교에 활용한 대표 논문/자료 제목입니다(본문에 모두 인용됨).

- Liang et al., “DiffusionRenderer: Neural Inverse and Forward Rendering with Video Diffusion Models”, CVPR 2025.[^2][^4][^1]
- Zeng et al., “RGB↔X: Image decomposition and synthesis using material‑ and lighting‑aware diffusion models”, SIGGRAPH 2024.[^8][^6][^11][^1]
- Jin et al., “Neural Gaffer: Relighting Any Object via Diffusion”, NeurIPS 2024.[^9][^7][^1]
- Poirier‑Ginter et al., “A Diffusion Approach to Radiance Field Relighting using Multi‑Illumination Synthesis”, CGF 2024.[^1]
- Kocsis et al., “Intrinsic Image Diffusion for Single‑View Material Estimation”, 2023.[^10][^1]
- Wang et al., “Neural Fields Meet Explicit Geometric Representations for Inverse Rendering of Urban Scenes (FEGR)”, CVPR 2023.[^1]
- Lin et al., “UrbanIR: Large‑Scale Urban Scene Inverse Rendering from a Single Video”, 2023.[^1]
- Liang et al., “Photorealistic Object Insertion with Diffusion‑Guided Inverse Rendering”, 2024.[^13]
- “FrameDiffuser: G‑Buffer‑Conditioned Diffusion for Neural Forward Frame Rendering”, 2025.[^14]

이 외에도 논문 본문에 인용된 NeRF, PhySG, IRON, GIR, GS‑IR, DiffusionLight, Stable Video Diffusion 등의 관련 연구 및 공식 프로젝트 페이지를 참고했습니다.[^15][^12][^10][^3][^1]
<span style="display:none">[^17][^18][^19][^20][^21][^22][^23][^24]</span>

<div align="center">⁂</div>

[^1]: 2501.18590v2.pdf

[^2]: https://arxiv.org/abs/2501.18590

[^3]: https://research.nvidia.com/labs/toronto-ai/DiffusionRenderer/

[^4]: https://cvpr.thecvf.com/virtual/2025/poster/34862

[^5]: https://arxiv.org/html/2412.15050v4

[^6]: https://zheng95z.github.io/publications/rgbx24

[^7]: https://arxiv.org/abs/2406.07520

[^8]: https://arxiv.org/abs/2405.00666

[^9]: https://proceedings.neurips.cc/paper_files/paper/2024/file/ff7373914a96956f2a7cacbdf3b0b8d8-Paper-Conference.pdf

[^10]: https://arxiv.org/html/2404.11593v1

[^11]: http://iliyan.com/publications/RGBX

[^12]: https://pmc.ncbi.nlm.nih.gov/articles/PMC10606505/

[^13]: http://arxiv.org/pdf/2408.09702.pdf

[^14]: https://arxiv.org/abs/2512.16670

[^15]: https://github.com/nv-tlabs/diffusion-renderer

[^16]: http://arxiv.org/pdf/2312.09608.pdf

[^17]: https://ieeexplore.ieee.org/document/11093962/

[^18]: https://arxiv.org/html/2501.18590v1

[^19]: https://github.com/Haian-Jin/Neural_Gaffer

[^20]: http://graphics.csie.ncku.edu.tw/2025 CG/Presentation/1119_Presentation.pdf

[^21]: https://github.com/zheng95z/rgbx

[^22]: https://openreview.net/forum?id=zV2GDsZb5a

[^23]: https://neurips.cc/virtual/2024/poster/92953

[^24]: https://www.emergentmind.com/topics/lighting-aware-material-attention-mechanism

