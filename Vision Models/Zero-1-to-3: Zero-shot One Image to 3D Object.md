# Zero-1-to-3: Zero-shot One Image to 3D Object

## 1. 핵심 주장과 주요 기여 (간결 요약)

**핵심 주장**

- 대규모 2D 텍스트-이미지 확산 모델(Stable Diffusion)이 **명시적 3D 감독 없이도 강한 3D 기하 priors**를 학습하고 있으며,  
- 여기에 **상대 카메라 포즈를 조절하는 얇은 제어 층만 미세조정(fine-tuning)** 하면,  
  - 단일 RGB 이미지만으로 **지정한 viewpoint의 novel view 합성**과  
  - **zero-shot 단일 뷰 3D 복원(single-view 3D reconstruction)** 이 가능하다는 것을 보여준다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/5ff301c3-d278-4247-8fc4-98849c01d360/2303.11328v1.pdf)

**주요 기여**

1. **Viewpoint-conditioned latent diffusion 모델 제안**  
   - 입력 이미지 \(x\)와 원하는 상대 카메라 변환 $\((R,T)\)$ 을 조건으로, 목표 뷰 $\(\hat{x}\_{R,T}\)$ 를 합성하는 함수

$$
     \hat{x}_{R,T} = f(x, R, T)
     $$
     
  를 학습하는 **이미지-조건 확산 모델**을 구성. [openaccess.thecvf](https://openaccess.thecvf.com/content/ICCV2023/papers/Liu_Zero-1-to-3_Zero-shot_One_Image_to_3D_Object_ICCV_2023_paper.pdf)

2. **Synthetic 3D 데이터(Objaverse) 기반 카메라 제어 학습**  
   - Objaverse의 3D 자산을 렌더링해 수천만 장의 멀티뷰 이미지를 생성하고,  
   - 이로부터 **상대 카메라 extrinsics $\((R,T)\)$ **를 조건으로 하는 제어 메커니즘을 학습. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/5ff301c3-d278-4247-8fc4-98849c01d360/2303.11328v1.pdf)

3. **두 스트림 조건화 구조(“posed CLIP + image concatenation”)**  
   - (1) CLIP 임베딩 + 포즈를 결합한 “posed CLIP”으로 **고수준 semantics & view control**,  
   - (2) 노이즈 이미지와 입력 뷰를 채널 concat하여 **로컬 appearance & identity**를 유지하는 하이브리드 구조 제안. [openaccess.thecvf](https://openaccess.thecvf.com/content/ICCV2023/papers/Liu_Zero-1-to-3_Zero-shot_One_Image_to_3D_Object_ICCV_2023_paper.pdf)

4. **Zero-shot novel view synthesis + 3D reconstruction SOTA**  
   - Google Scanned Objects(GSO), RTMV에서 **DietNeRF, Image Variations, SJC 기반 방법, MCC, Point-E** 등을 큰 margin으로 상회. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/5ff301c3-d278-4247-8fc4-98849c01d360/2303.11328v1.pdf)
   - 특히 **Real 2D 이미지·회화·DALL·E 2 이미지까지** 강한 **open-world zero-shot generalization** 시연. [arxiv](https://arxiv.org/abs/2303.11328)

5. **기존 text-to-3D distillation(SJC)와 결합한 단일 뷰 3D 복원 파이프라인**  
   - Zero-1-to-3를 score Jacobian chaining(SJC)과 통합해, **단일 RGB 이미지 → neural radiance field → 3D mesh** 복원 프레임워크를 제시. [openaccess.thecvf](https://openaccess.thecvf.com/content/ICCV2023/papers/Liu_Zero-1-to-3_Zero-shot_One_Image_to_3D_Object_ICCV_2023_paper.pdf)

***

## 2. 문제 정의와 제안 방법 (수식·모델 구조 중심 상세 설명)

### 2.1 논문이 다루는 문제

**문제 1: 단일 이미지 기반 novel view synthesis (NVS)**  
- 입력: 한 장의 RGB 이미지 $\(x \in \mathbb{R}^{H \times W \times 3}\)$ 
- 목표: 사용자가 지정한 **상대 카메라 회전·이동 \((R,T)\)**에 대해,  
  - 같은 객체를 그 viewpoint에서 본 이미지를 합성:

$$
    \hat{x}_{R,T} = f(x, R, T) \approx x_{R,T}^{\star}
    $$
  
  - 여기서 $\(x_{R,T}^{\star}\)$ 는 이상적인 (하지만 관측되지 않는) GT novel view.

**문제 2: 단일 이미지 기반 3D 복원(single-view 3D reconstruction)**  
- 위와 같은 설정에서,  
- **볼륨/NeRF/voxel radiance field** 형태의 3D 표현을 추정해 전체 3D mesh를 복원.  
- 본질적으로 **극단적으로 under-constrained** 문제이므로, 강한 **3D priors**가 필수.

기존 방법의 한계:

- NeRF류: 다중 뷰·정확한 포즈가 필요, 카테고리·씬 영역에 한정. [arxiv](https://arxiv.org/pdf/2310.15110.pdf)
- 단일 뷰 3D: 보통 카테고리 특화 priors 혹은 3D 자산(CAD, voxel, mesh) 대규모 필요 → open-world generalization 약함. [arxiv](https://arxiv.org/html/2507.05819v1)
- 최근 diffusion + NeRF (DreamFields, DreamFusion, NeRDi, RealFusion, NeuralLift-360 등)는  
  - 주로 **text-to-3D 또는 image-to-3D 최적화 기반**이라 느리고,  
  - 3D 표현을 매번 per-shape optimization 해야 하며,  
  - 여전히 multi-view 훈련이나 카메라 정보 의존성이 큼. [arxiv](http://arxiv.org/abs/2212.03267)

**Zero-1-to-3가 푸는 핵심**:  
- “**2D 인터넷 규모 확산 모델이 이미 암묵적으로 가진 3D priors를 어떻게 꺼내 쓸 것인가?**”  
- 학습 시 **명시적 3D 감독 없이**, synthetic multi-view image pairs + 상대 포즈만으로  
  - **카메라 viewpoint를 controllable condition으로 학습**  
  - → fast, feed-forward NVS & 3D reconstruction이 가능하게 만드는 것. [arxiv](https://arxiv.org/abs/2303.11328)

***

### 2.2 제안된 방법: Viewpoint-conditioned Latent Diffusion

#### 2.2.1 기본 정식화

입력 이미지 $\(x\)$ 와 원하는 상대 카메라 변환 $\((R,T)\)$ 에 대해, Zero-1-to-3는

$$
\hat{x}_{R,T} = f(x,R,T)
$$

를 생성하는 확산 모델 $\(f\)$ 를 학습한다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/5ff301c3-d278-4247-8fc4-98849c01d360/2303.11328v1.pdf)

여기서 $\((R,T)\)$ 는 실제 구현에서는 **spherical coordinate 기반 상대 viewpoint 파라미터**  
$\((\Delta \theta, \sin \Delta\phi, \cos \Delta\phi, \Delta r)\)$ 형태로 인코딩되어,  
**4차원 pose 벡터**로 입력된다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/5ff301c3-d278-4247-8fc4-98849c01d360/2303.11328v1.pdf)

#### 2.2.2 Latent Diffusion 기반 학습 목표

Stable Diffusion 스타일의 **latent diffusion** 구조를 따른다. [arxiv](https://arxiv.org/html/2409.07452v1)

- VAE encoder $\(E\)$ :  
  - 이미지 $\(x\)$ 를 latent $\(z = E(x)\)$ (예: $\(64 \times 64\)$ 해상도)로 매핑
- Denoising U-Net $\(\epsilon_\theta\)$ :  
  - 노이즈가 섞인 latent $\(z_t\)$ , 시간 $\(t\)$ , 조건 $\(c(x,R,T)\)$ 를 입력으로 노이즈 $\(\epsilon\)$ 를 예측
- Decoder $\(D\)$ : latent를 다시 RGB 이미지로 디코딩

학습 시:

1. latent에 가우시안 노이즈를 주입:

$$
   z_t = \alpha_t z + \sigma_t \epsilon,\quad \epsilon \sim \mathcal{N}(0, I)
   $$

2. 조건 $\(c(x,R,T)\)$ 을 사용해 노이즈를 예측하도록 U-Net을 미세조정:

$$
   \min_{\theta} \mathbb{E}_{z \sim E(x),\, t,\, \epsilon \sim \mathcal{N}(0,1)}
   \left\|
     \epsilon - \epsilon_\theta(z_t, t, c(x,R,T))
   \right\|_2^2
   $$

- 여기서 **조건 \(c(x,R,T)\)**는 아래의 이중 스트림(hybrid) 조건화를 통해 구성된다. [openaccess.thecvf](https://openaccess.thecvf.com/content/ICCV2023/papers/Liu_Zero-1-to-3_Zero-shot_One_Image_to_3D_Object_ICCV_2023_paper.pdf)

#### 2.2.3 View-conditioned conditioning: “posed CLIP + image concatenation”

Zero-1-to-3의 핵심 설계는 **“전역 semantics + 로컬 appearance”를 동시에 유지**하는 조건화 방식이다.

1. **전역 스트림: posed CLIP embedding**

   - 입력 이미지 $\(x\)$ 를 CLIP encoder에 통과시켜 임베딩 $\(e_\text{CLIP}(x)\)$ (차원 768) 획득. [arxiv](https://arxiv.org/pdf/2311.07885.pdf)
   - 상대 카메라 포즈 벡터 $\(p = (\Delta\theta, \sin \Delta\phi, \cos \Delta\phi, \Delta r)\)$ 를 concatenate:

$$
     c(x,R,T) = \text{MLP}\big([e_\text{CLIP}(x);\, p]\big) \in \mathbb{R}^{768}
     $$
   
   - 이 벡터를 Stable Diffusion의 **cross-attention 키/값**으로 사용해,  
     U-Net 전체에 **“어떤 object를, 어떤 viewpoint에서 그려야 하는지”**를 전역적으로 주입. [openaccess.thecvf](https://openaccess.thecvf.com/content/ICCV2023/papers/Liu_Zero-1-to-3_Zero-shot_One_Image_to_3D_Object_ICCV_2023_paper.pdf)

2. **로컬 스트림: 입력 이미지 concat**

   - denoising 과정에서의 현재 latent 이미지를 디코딩하지 않고,  
     **latent 공간에서 노이즈 이미지를 upsample → RGB 공간으로 변환한 feature와 입력 이미지 \(x\)를 채널 방향으로 concat**. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/5ff301c3-d278-4247-8fc4-98849c01d360/2303.11328v1.pdf)
   - 이렇게 하면 모델이 **입력 사진의 정확한 색·텍스처·로컬 디테일**을 복제·보존할 수 있다.

3. **Classifier-free guidance**

   - Stable Diffusion의 classifier-free guidance를 따라, [arxiv](https://arxiv.org/abs/2303.11328)
     - 일정 확률로 조건 $(\(x\), \(c(x,R,T)\))$ 을 null로 설정하고  
     - 샘플링 시 조건 신호를 scale해 고품질 샘플을 얻는다. [semanticscholar](https://www.semanticscholar.org/paper/622cab9477f190ec9ef1d12e5e71ba36146ad694)

결과적으로, Zero-1-to-3는 **“입력 이미지 + 포즈”를 조건으로 하는 이미지-조건 확산 모델**이 된다.  

***

### 2.3 단일 이미지 3D 복원: Score Jacobian Chaining(SJC)와의 결합

#### 2.3.1 기본 아이디어

- SJC는 기존 text-to-image diffusion 모델을 frozen critic으로 사용해 [liner](https://liner.com/review/makeit3d-highfidelity-3d-creation-from-single-image-with-diffusion-prior)
  **NeRF/voxel radiance field**를 최적화하는 text-to-3D 알고리즘이다. [arxiv](https://arxiv.org/html/2505.08239v3)
- Zero-1-to-3는 **텍스트 대신 “입력 이미지 + 포즈”를 condition으로 쓰는** image-조건 확산 모델.  
- 따라서, SJC의 **score distillation**을 Zero-1-to-3에 적용하면,  
  - random camera viewpoint에서 렌더링한 이미지를  
  - “입력 이미지와 같은 object를, 해당 viewpoint에서 본 이미지” 분포로 끌어당기도록 3D 표현을 최적화할 수 있다. [openaccess.thecvf](https://openaccess.thecvf.com/content/ICCV2023/papers/Liu_Zero-1-to-3_Zero-shot_One_Image_to_3D_Object_ICCV_2023_paper.pdf)

#### 2.3.2 SJC-style gradient

Neural field(예: voxel radiance field) 파라미터 $\(\phi\)$ 에 대해,  
랜덤 viewpoint $\((R,T)\)$ 에서 렌더링한 이미지 $\(x_\pi\)$ 를 Zero-1-to-3 확산 모델에 통과시켜 **score distillation gradient**를 사용:

$$
\nabla_{\phi} \mathcal{L}_\text{SJC}
= \nabla_{\phi} \log p_{\sqrt{2}\epsilon}(x_\pi)
$$

실제로는,

1. neural field → volumetric rendering → $\(x_\pi\)$
2. 노이즈 $\(\epsilon \sim \mathcal{N}(0,1)\)$ 주입 → $\(x_{\pi,t}\)$
3. Zero-1-to-3의 U-Net $\(\epsilon_\theta\big(x_{\pi,t}, t, c(x,R,T)\big)\)$ 출력을 이용해  
   SDS/PAAS 스타일의 gradient를 계산해 $\(\phi\)$ 를 업데이트한다. [liner](https://liner.com/review/makeit3d-highfidelity-3d-creation-from-single-image-with-diffusion-prior)

추가로,

- **입력 뷰 재구성 MSE $\(\mathcal{L}_\text{MSE}\)$ **: 입력 뷰에서 렌더링된 이미지가 원본과 가깝게.  
- **depth smoothness loss**: geometry에서 불필요한 high-frequency를 줄여 메쉬의 홀 제거. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/5ff301c3-d278-4247-8fc4-98849c01d360/2303.11328v1.pdf)
- **near-view consistency loss**: 가까운 viewpoint 간 appearance 변화를 regularize해 multi-view 일관성 강화. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/5ff301c3-d278-4247-8fc4-98849c01d360/2303.11328v1.pdf)

***

### 2.4 모델 구조 및 학습 설정 요약

| 구성 요소 | 내용 |
|---|---|
| **Base** | Stable Diffusion latent diffusion (VAE + U-Net + cross-attention) [arxiv](https://arxiv.org/html/2409.07452v1) |
| **조건** | (1) CLIP image embedding + pose → cross-attention, (2) 입력 이미지 채널 concat [openaccess.thecvf](https://openaccess.thecvf.com/content/ICCV2023/papers/Liu_Zero-1-to-3_Zero-shot_One_Image_to_3D_Object_ICCV_2023_paper.pdf) |
| **포즈 인코딩** | spherical coordinate 기반 $\((\Delta\theta, \sin \Delta\phi, \cos \Delta\phi, \Delta r)\)$ [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/5ff301c3-d278-4247-8fc4-98849c01d360/2303.11328v1.pdf) |
| **훈련 데이터** | Objaverse 800k+ 3D assets → 약 1,000만 렌더링 이미지, 12 views per object [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/5ff301c3-d278-4247-8fc4-98849c01d360/2303.11328v1.pdf) |
| **학습 전략** | 이미지 해상도 256×256, latent 32×32, batch size 1536로 크게 키워 안정 학습 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/5ff301c3-d278-4247-8fc4-98849c01d360/2303.11328v1.pdf) |
| **추론 속도** | 단일 novel view 합성 ~2초 (RTX A6000 기준), 3D 복원 ~30분 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/5ff301c3-d278-4247-8fc4-98849c01d360/2303.11328v1.pdf) |

***

## 3. 성능 향상 및 한계

### 3.1 Novel View Synthesis 성능

**벤치마크**  

- **Google Scanned Objects(GSO)**: 고품질 스캔된 단일 오브젝트 [arxiv](https://arxiv.org/html/2503.12929v3)
- **RTMV**: 20개 이상의 객체가 섞인 복잡한 장면 [arxiv](https://arxiv.org/abs/2303.14184)

**비교 대상** (모두 zero-shot, single-view 조건)

- DietNeRF (CLIP consistency regularized NeRF) [xoft.tistory](https://xoft.tistory.com/61)
- Image Variations (Stable Diffusion image-conditional) [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/5ff301c3-d278-4247-8fc4-98849c01d360/2303.11328v1.pdf)
- SJC-I: text-conditioned 확산을 image-conditioned로 바꾼 SJC 변형 [liner](https://liner.com/review/makeit3d-highfidelity-3d-creation-from-single-image-with-diffusion-prior)

**GSO 결과 (Table 1)** [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/5ff301c3-d278-4247-8fc4-98849c01d360/2303.11328v1.pdf)

| 방법 | PSNR ↑ | SSIM ↑ | LPIPS ↓ | FID ↓ |
|---|---|---|---|---|
| DietNeRF | 8.93 | 0.645 | 0.412 | 12.919 |
| Image Variations | 5.91 | 0.540 | 0.545 | 22.533 |
| SJC-I | 6.57 | 0.552 | 0.484 | 19.783 |
| **Zero-1-to-3** | **18.38** | **0.877** | **0.088** | **0.027** |

- PSNR·SSIM·LPIPS·FID 모두에서 **압도적 개선**을 달성.  
- 특히 FID 0.027은 사실상 training distribution에 매우 가까운 수준의 품질을 의미. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/5ff301c3-d278-4247-8fc4-98849c01d360/2303.11328v1.pdf)

**RTMV 결과 (Table 2)** [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/5ff301c3-d278-4247-8fc4-98849c01d360/2303.11328v1.pdf)

| 방법 | PSNR ↑ | SSIM ↑ | LPIPS ↓ | FID ↓ |
|---|---|---|---|---|
| DietNeRF | 7.13 | 0.406 | 0.507 | 5.143 |
| Image Variations | 6.56 | 0.442 | 0.564 | 10.218 |
| SJC-I | 7.95 | 0.456 | 0.545 | 10.202 |
| **Zero-1-to-3** | **10.41** | **0.606** | **0.323** | **0.319** |

- Objaverse로만 학습했음에도, **다중 객체·복잡 배경의 out-of-distribution 장면에서 여전히 SOTA**. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/5ff301c3-d278-4247-8fc4-98849c01d360/2303.11328v1.pdf)

**질적 분석**

- GSO/RTMV 모두에서,  
  - 경쟁 방법 대비 **고주파 텍스처 보존**,  
  - self-occlusion 영역에서 **그럴듯한 geometry·texture hallucination**을 보여줌. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/5ff301c3-d278-4247-8fc4-98849c01d360/2303.11328v1.pdf)
- In-the-wild 사진, 인터넷 이미지, 인상파 회화, DALL·E 2 생성 이미지에 대해  
  - object identity·스타일을 유지하며 다양한 viewpoint의 이미지를 생성. [arxiv](http://arxiv.org/pdf/2303.14184.pdf)

### 3.2 Single-View 3D Reconstruction 성능

**GSO (Table 3)** [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/5ff301c3-d278-4247-8fc4-98849c01d360/2303.11328v1.pdf)

| 방법 | Chamfer Distance ↓ | IoU ↑ |
|---|---|---|
| MCC | 0.2343 | 0.1230 |
| SJC-I | 0.2245 | 0.1332 |
| Point-E | 0.0804 | 0.2944 |
| **Zero-1-to-3 + SJC** | **0.0717** | **0.5052** |

- CD는 Point-E 수준으로 낮지만,  
- **volumetric IoU는 다른 모든 방법을 크게 상회(0.5+)** →  
  - 전체 볼륨을 더 정확히 채운다는 의미. [arxiv](https://arxiv.org/html/2511.22194v1)

**RTMV (Table 4)** [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/5ff301c3-d278-4247-8fc4-98849c01d360/2303.11328v1.pdf)

- 복잡 장면에서 전반적으로 모든 방법 성능이 떨어지지만,  
- Zero-1-to-3 기반 방법이 여전히 가장 높은 IoU 및 경쟁력 있는 CD를 기록.

**요약적 해석**

- Point-E는 sparse point cloud (~4096 points)로 인해 hole이 많아 IoU가 낮고, [arxiv](https://arxiv.org/html/2511.22194v1)
- MCC는 입력 뷰에서 보이는 면은 잘 맞지만 occluded 부분 geometry 추론이 약한 반면, [arxiv](https://arxiv.org/html/2309.03453v2)
- Zero-1-to-3는 **멀티뷰 priors를 활용하는 확산 모델 + NeRF 스타일 volumetric 표현**을 결합해  
  - **전역 geometry와 local 디테일을 함께 복원**하는 데 강점을 보인다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/5ff301c3-d278-4247-8fc4-98849c01d360/2303.11328v1.pdf)

### 3.3 주요 한계

1. **Scene-level generalization 부족**

   - 학습은 **단일 객체 + 깔끔한 배경** 위주(Objaverse 렌더)로 진행. [arxiv](http://arxiv.org/pdf/2306.16928v1.pdf)
   - RTMV(다중 객체, cluttered background)에서 성능이 떨어지며,  
   - 완전한 scene-level 3D 이해·재구성에는 아직 부족. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/5ff301c3-d278-4247-8fc4-98849c01d360/2303.11328v1.pdf)

2. **Multi-view consistency의 한계**

   - 각 novel view를 **독립 샘플링**하는 2D diffusion이므로,  
   - viewpoint들 사이의 geometry·색상 consistency가 완벽하진 않음.  
   - 이 문제는 후속 연구 SyncDreamer, Zero123++, Wonder3D 등이 3D-aware attention·joint diffusion·cross-domain normal 등을 통해 개선하는 방향으로 발전. [arxiv](https://arxiv.org/abs/2309.03453)

3. **3D reconstruction 비용**

   - 단일 novel view는 수 초 수준으로 빠르지만,  
   - SJC 기반 3D 복원은 여전히 **약 30분/객체**로 per-shape optimization이 필요. [liner](https://liner.com/review/makeit3d-highfidelity-3d-creation-from-single-image-with-diffusion-prior)
   - 후속 One-2-3-45, Wonder3D, LRM류는 **feed-forward 3D 복원**으로 이 한계를 완화. [arxiv](https://arxiv.org/abs/2306.16928)

4. **배경 분리 필요**

   - In-the-wild 이미지에서는 off-the-shelf background removal을 필수적으로 사용. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/5ff301c3-d278-4247-8fc4-98849c01d360/2303.11328v1.pdf)
   - object-ground 관계, 그림자, contact geometry 등은 명시적으로 다루지 않음.  
   - 최근 “Floating No More” 류 single-image object-ground 복원 연구들이 이 부분을 보완. [semanticscholar](https://www.semanticscholar.org/paper/84cce9b8aea35e4fa38eef63da439573f21c0728)

***

## 4. 일반화 성능 관점에서의 심층 분석

### 4.1 왜 Zero-1-to-3는 강한 zero-shot generalization을 가지는가?

1. **인터넷 규모 2D 사전 학습의 효과**

   - Stable Diffusion는 **LAION-5B 등 수십억 이미지**로 학습되어,  
     - 다양한 객체 카테고리, 스타일, 조명, viewpoint를 망라하는 **풍부한 2D priors**를 가진다. [semanticscholar](https://www.semanticscholar.org/paper/DreamFusion:-Text-to-3D-using-2D-Diffusion-Poole-Jain/4c94d04afa4309ec2f06bdd0fe3781f91461b362)
   - Zero-1-to-3는 이 거대한 2D representation을 거의 유지한 채,  
     - **추가로 “상대 viewpoint control”이라는 얇은 층만 학습**.  
   - 따라서 Objaverse에 없는 카테고리/스타일에도  
     - **2D 수준의 표현력 + 새로 학습한 viewpoint 조건**을 조합해 zero-shot 성능을 발휘. [arxiv](https://arxiv.org/pdf/2307.05663.pdf)

2. **Synthetic 3D 데이터(Objaverse) 기반 viewpoint coverage**

   - Objaverse는 80만+ 객체, 다양한 topology·재질·스타일의 large-scale 3D 자산. [arxiv](http://arxiv.org/pdf/2306.16928v1.pdf)
   - 무작위 카메라 sampling으로 렌더링한 약 1천만 장 multi-view 이미지에서  
     - **상대 viewpoint 변환 \((R,T)\)**에 대한 mapping을 학습. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/5ff301c3-d278-4247-8fc4-98849c01d360/2303.11328v1.pdf)
   - 후속 Objaverse-XL(10M+ 객체)는 Zero123 스타일 모델의 zero-shot 일반화를 더 끌어올린다는 결과를 보여,  
     - **synthetic multi-view 데이터 규모 확장이 일반화에 직접 기여**함을 시사. [arxiv](https://arxiv.org/pdf/2307.05663.pdf)

3. **입력 이미지 직접 conditioning으로 카테고리 priors 탈피**

   - NeRF+category prior 방식과 달리,  
     - Zero-1-to-3는 CLIP 임베딩 + 이미지 concat을 통해  
       - **입력 이미지 자체를 priors의 anchor**로 사용.  
   - 이로 인해 **카테고리 한정 priors 없이도** 새로운 객체, 회화 스타일, DALL·E 이미지까지 대응 가능. [arxiv](http://arxiv.org/abs/2212.03267)

4. **모호성(aleatoric uncertainty)을 확산 모델이 자연스럽게 모델링**

   - self-occlusion 영역의 unknown texture/shape는 본질적으로 다값성(multimodal).  
   - Zero-1-to-3는 diffusion 샘플링으로 이런 불확실성을 **다양한 plausible sample로 모델링**. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/5ff301c3-d278-4247-8fc4-98849c01d360/2303.11328v1.pdf)
   - 이는 deterministic NeRF-based methods가 가지기 어려운 강점이다.

### 4.2 일반화 한계를 드러내는 요소

1. **Object-centric training bias**

   - 훈련 데이터는 단일 객체 중심, 배경 단순화, 일정한 lighting.  
   - 실제 장면의 clutter, occluder, 복잡 조명에 대한 distribution shift가 존재.  
   - ZeroNVS, NeO360, studentSplat 등은 이 한계를 넘어 scene-level single-view reconstruction으로 확장 중. [ieeexplore.ieee](https://ieeexplore.ieee.org/document/10657980/)

2. **Viewpoint 조건의 표현력·정확도**

   - pose 인코딩은 간결하지만, 실제 카메라 파이프라인(렌즈 왜곡, intrinsics 등)은 단순화.  
   - elevation·radius 등의 분포도 synthetic 설정에 맞춰져 있어  
     - extreme viewpoints나 FOV 차이에는 약할 수 있다.  
   - 이러한 부분은 Stable Zero123에서 **elevation conditioning**을 도입하여 개선. [stability](https://stability.ai/news/stable-zero123-3d-generation)

3. **3D-consistent representation의 부재**

   - 본질적으로 **2D 이미지 diffusion**이므로,  
     - multi-view sample 간 joint consistency는 명시적으로 강제되지 않는다.  
   - SyncDreamer, Zero123++, Wonder3D, 3D-Adapter 등이  
     - **3D-aware feature attention, multiview joint diffusion, RGBD/normal cross-domain diffusion**으로 이를 보완. [arxiv](https://arxiv.org/abs/2410.18974)

### 4.3 “모델의 일반화 성능 향상 가능성” – 설계 관점 시사점

Zero-1-to-3와 후속 연구 흐름을 종합하면, 일반화 성능을 더 끌어올리기 위한 설계 방향은 다음과 같이 정리할 수 있다.

1. **2D diffusion priors + synthetic 3D multi-view의 결합을 더 크게**

   - Objaverse → Objaverse-XL처럼, 3D 자산·렌더링 규모를 키워 **viewpoint·geometry coverage**를 늘릴수록  
     - Zero-shot NVS/3D reconstruction의 일반화 범위가 확장되는 경향. [arxiv](https://arxiv.org/pdf/2307.05663.pdf)

2. **2D → 3D-aware diffusion으로의 구조적 전환**

   - SyncDreamer: multiview joint diffusion + 3D-aware feature volume attention으로 **view-consistency 강화**. [arxiv](https://arxiv.org/abs/2309.03453)
   - Zero123++: conditioning·훈련 스킴을 개선해, Zero-1-to-3 대비 **기하 정합성 및 품질 향상**. [arxiv](https://arxiv.org/abs/2310.15110)
   - Wonder3D: multi-view RGB + normal map을 동시에 생성하는 **cross-domain diffusion**으로 geometry 디테일 극대화. [cg.cs.tsinghua.edu](https://cg.cs.tsinghua.edu.cn/papers/CVPR-2024-Wonder3D.pdf)
   - 3D-Adapter: 기존 Zero123++/Instant3D 위에 plug-in으로 3D feedback loop를 추가해 geometry 일관성을 향상시킴. [arxiv](https://arxiv.org/abs/2410.18974)

3. **Scene-level, multi-object generalization**

   - ZeroNVS, NeO360, studentSplat 등은 single-view에서 **실세계 outdoor/scene 레벨**로 확장. [arxiv](https://arxiv.org/pdf/2308.12967.pdf)
   - Zero-1-to-3 계열도 object-centric pretraining 후 scene-level fine-tuning이나,  
     - segmentation·depth priors를 추가로 결합하는 방향이 필요.

4. **Downstream task-aware training**

   - Zero-1-to-3는 **NVS·3D reconstruction을 동시에 지원**하지만,  
     - 학습 objective는 여전히 **pixel-level reconstruction + diffusion loss** 중심.  
   - 후속 Ctrl123, ConsistNet 등은 **pose-consistent feature space alignment, 3D consistency regularizer**를 명시적으로 도입해  
     - NVS·3D reconstruction 지표 향상을 달성. [arxiv](https://arxiv.org/abs/2403.10953)

***

## 5. 2020년 이후 관련 최신 연구 비교 분석

Zero-1-to-3를 중심으로, 2020년 이후 단일 이미지 → 3D 관련 주요 흐름을 계열별로 정리하면 다음과 같다.

### 5.1 NeRF·implicit field 기반 single-view/few-view (pre-diffusion)

- **pixelNeRF, IBRNet, DietNeRF, MCC 등** [arxiv](https://arxiv.org/html/2312.11535v2)
  - 공통점:  
    - NeRF/implicit field를 backbone으로 하고,  
    - multi-view 데이터셋에서 학습하여 새로운 scene/object에 대한 generalizable NVS 수행.  
  - DietNeRF는 CLIP consistency loss로 semantic priors를 도입하지만,  
    - 여전히 **multi-view 훈련 데이터**에 의존,  
    - open-world single-view zero-shot에는 약함. [xoft.tistory](https://xoft.tistory.com/61)

**Zero-1-to-3와의 차이**

- NeRF 계열: **3D 표현이 explicit**, but 2D priors는 제한적.  
- Zero-1-to-3: **표현은 2D latent diffusion**, but **3D priors는 인터넷 규모 사전학습 + synthetic multi-view**에서 간접 획득.  
- 결과적으로 Zero-1-to-3가 **open-world single-view zero-shot NVS·3D reconstruction에서 우위**. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/5ff301c3-d278-4247-8fc4-98849c01d360/2303.11328v1.pdf)

### 5.2 Text-to-3D diffusion 계열 (DreamFields, DreamFusion 등)

- **DreamFields**: CLIP supervision으로 NeRF를 최적화하는 text-to-3D. [semanticscholar](https://www.semanticscholar.org/paper/Zero-1-to-3:-Zero-shot-One-Image-to-3D-Object-Liu-Wu/2c70684973bc4d7b6f8404a647b8031c4d3c8383)
- **DreamFusion**: text-to-image diffusion + Score Distillation Sampling(SDS)으로 NeRF를 최적화. [arxiv](https://arxiv.org/html/2505.08239v3)
- **Magic3D, ProlificDreamer, DreamGaussian 등**: SDS·variational loss 개선, Gaussian Splatting 등으로 고품질·고속 텍스트 기반 3D 생성. [kimjy99.github](https://kimjy99.github.io/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0/one-2-3-45/)

**Zero-1-to-3와의 관계**

- 위 계열은 **텍스트 → 3D**가 주목표,  
  - image-to-3D는 보통 “text from image” 캡셔닝 + text-to-3D로 우회.  
- Zero-1-to-3는 **image-conditioned diffusion**을 직접 학습하여,  
  - text 없이도 입력 이미지의 identity를 강하게 보존한 채 NVS/3D 복원.  
- 후속 **One-2-3-45, Make-It-3D, RealFusion, NeuralLift-360** 등은  
  - Zero-1-to-3/Zero123를 priors로 사용하거나,  
  - Stable Diffusion 기반 priors로 image-to-3D를 구현. [openaccess.thecvf](https://openaccess.thecvf.com/content/CVPR2023/papers/Xu_NeuralLift-360_Lifting_an_In-the-Wild_2D_Photo_to_a_3D_Object_CVPR_2023_paper.pdf)

### 5.3 Image-to-3D diffusion 계열 (Zero-1-to-3 이후)

#### (1) NeRDi, NeuralLift-360, RealFusion (Zero-1-to-3 동시기)

- **NeRDi**: single-view NeRF + 2D diffusion prior + language guidance(2-stage semantic feature) + depth regularization. [openaccess.thecvf](https://openaccess.thecvf.com/content/CVPR2023/papers/Deng_NeRDi_Single-View_NeRF_Synthesis_With_Language-Guided_Diffusion_As_General_Image_CVPR_2023_paper.pdf)
- **NeuralLift-360**: reference 이미지 + depth-aware NeRF + Stable Diffusion prior로 360° object 복원. [arxiv](https://arxiv.org/abs/2211.16431)
- **RealFusion**: Stable Diffusion + DreamFusion 스타일 SDS로  
  - 입력 이미지를 잘 맞추면서, diffusion prior로 novel view를 hallucinate. [lukemelas.github](https://lukemelas.github.io/realfusion/)

이들은 모두 **per-shape optimization** 기반으로 느리며,  
Zero-1-to-3와 달리 **feed-forward NVS 모델**은 아니다.

#### (2) Zero-1-to-3 계열 직접 후속

- **Zero123 / Zero123-XL / Stable Zero123**:  
  - Zero-1-to-3 아키텍처를 확장·개선한 open-source 계열. [github](https://github.com/cvlab-columbia/zero123)
  - Stable Zero123는  
    - 더 엄격히 필터링한 Objaverse 렌더 + elevation conditioning + 효율적 dataloader로  
    - Zero123-XL 대비 품질·일반화·훈련 효율 향상을 보고. [stability](https://stability.ai/news/stable-zero123-3d-generation)

- **Zero123++**:  
  - Stable Diffusion에서 최솟의 수정으로 **consistent multi-view 이미지**를 생성하는 base model. [kimjy99.github](https://kimjy99.github.io/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0/zero123plus/)
  - conditioning·training scheme을 개선해  
    - Zero-1-to-3 대비 **geometry misalignment·texture degradation 감소**,  
    - multi-view set을 통째로 생성하는 데 더 적합. [arxiv](https://arxiv.org/pdf/2310.15110.pdf)

- **Cascade-Zero123**:  
  - 두 단계 Zero-1-to-3 모델을 cascade로 구성,  
    - 1단계: 입력 뷰 근방 여러 뷰 생성,  
    - 2단계: 그 뷰들을 self-prompt로 활용해 보다 consistent novel view 합성. [arxiv](https://arxiv.org/abs/2312.04424)
  - self-prompted nearby views를 통해  
    - geometry/appearance inconsistency를 완화하고,  
    - 곤충, 투명체, 다중 객체 등 복잡 object에 강함.

- **SyncDreamer**:  
  - Zero123 초기화를 사용하되,  
  - multiview joint diffusion + 3D-aware feature attention으로  
    - 여러 viewpoint 이미지를 **하나의 확산 reverse 과정에서 동시 생성**. [liuyuan-pal.github](https://liuyuan-pal.github.io/SyncDreamer/)
  - view 간 state를 매 단계마다 동기화해  
    - geometry·색상의 multiview consistency를 크게 향상,  
    - image-to-3D, text-to-3D에서 보다 안정적인 3D 복원을 가능하게 함.

- **ConsistNet, Ctrl123, Consistent123, Repaint123 등**:  
  - pose-sensitive feature space alignment, closed-loop transcription, case-aware priors 등을 사용해,  
  - Zero-1-to-3 계열의 **pose accuracy·view consistency를 정교하게 개선**. [arxiv](https://arxiv.org/pdf/2310.10343.pdf)

#### (3) Fast feed-forward 3D 모델

- **One-2-3-45 / One-2-3-45++**:  
  - Zero123/Zero-1-to-3로 multi-view 이미지를 먼저 생성하고,  
  - 이를 SDF 기반 generalizable neural surface로 lifting → **45초 내 full 3D mesh 복원**. [openaccess.thecvf](https://openaccess.thecvf.com/content/CVPR2024/supplemental/Liu_One-2-3-45_Fast_Single_CVPR_2024_supplemental.pdf)
- **Wonder3D**:  
  - cross-domain diffusion으로 multi-view color + normal을 동시에 생성하고,  
  - fast normal fusion으로 고품질 textured mesh를 2–3분 내 복원. [xxlong](https://www.xxlong.site/Wonder3D/)
- 이 계열은 Zero-1-to-3의 **feed-forward NVS 장점**과 NeRF-free reconstruction을 결합한 형태.

***

## 6. 앞으로의 연구에 미치는 영향과 향후 고려 사항

### 6.1 연구적 영향

1. **“2D 확산 모델 → 3D priors 추출” 패러다임의 정착**

   - Zero-1-to-3는 Stable Diffusion 같은 **2D foundation model**이  
     - **geometry priors까지 상당 부분 내장**하고 있음을 실증했다. [arxiv](https://arxiv.org/abs/2303.11328)
   - 이후 text-to-3D, image-to-3D, scene reconstruction, 3D avatar 생성 등  
     - 수많은 연구가 **“대규모 2D 모델 ↔ 3D world”를 잇는 아키텍처**를 설계하는 방향으로 확장. [openaccess.thecvf](https://openaccess.thecvf.com/content/CVPR2023/papers/Xu_NeuralLift-360_Lifting_an_In-the-Wild_2D_Photo_to_a_3D_Object_CVPR_2023_paper.pdf)

2. **Synthetic multi-view 데이터의 중요성 부각**

   - Objaverse·Objaverse-XL 렌더를 통한 viewpoint control 학습은  
     - **고비용 3D annotation 없이도 강력한 3D-aware model을 만들 수 있음**을 보여줌. [arxiv](http://arxiv.org/pdf/2306.16928v1.pdf)
   - 이후 3D Gaussian splatting, 3D LRM, studentSplat 등도  
     - 유사한 synthetic multi-view supervision을 적극 활용. [arxiv](https://arxiv.org/html/2601.11772v1)

3. **Image-conditioned generative priors의 재조명**

   - Text-conditioned priors에 비해,  
     - image-conditioned priors가 **identity 보존·fine detail 재현**에서 훨씬 강하다는 점을 부각.  
   - Zero-1-to-3 계열은 이후 **인물·아바타·제품 사진·예술 스타일** 등  
     - 실제 산업적 활용에 더 가까운 image-to-3D pipeline의 기반이 되었다. [dl.acm](https://dl.acm.org/doi/10.1145/3610548.3618153)

### 6.2 앞으로 연구 시 고려할 점 (특히 “일반화 성능 향상” 관점)

연구자로서 향후 관련 연구를 설계할 때, 다음 포인트들을 전략적으로 고려하는 것이 유의미하다.

1. **데이터: 어떤 3D/2D 조합이 일반화를 최대화하는가?**

   - 3D synthetic (Objaverse-XL, OmniObject3D 등) + 2D 인터넷 이미지/영상(일반 사진, 회화, 스케치, 만화 등)을  
     - 어떻게 혼합 학습할지  
     - 어떤 domain gap regularization·domain adaptation을 둘지. [arxiv](https://arxiv.org/pdf/2211.16431.pdf)
   - scene-level 데이터(NeRDS 360, DTU, Mip-NeRF 360 등)까지 포함해  
     - **object ↔ scene generalization**을 동시에 달성할 수 있는 구성 탐색. [ieeexplore.ieee](https://ieeexplore.ieee.org/document/10657980/)

2. **아키텍처: 2D diffusion vs 3D-aware diffusion vs hybrid**

   - 순수 2D latent diffusion 위에 viewpoint control만 더하는 Zero-1-to-3 스타일은  
     - 구현·훈련이 간단하고, 2D priors 손상이 적다는 장점.  
   - 그러나 multi-view consistency·scene geometry 측면에서는  
     - SyncDreamer·Wonder3D·3D-Adapter처럼 **3D-aware feature volume, RGBD/normal cross-domain, 3D feedback loop**가 점점 필수. [openaccess.thecvf](http://openaccess.thecvf.com/content/CVPR2024/papers/Long_Wonder3D_Single_Image_to_3D_using_Cross-Domain_Diffusion_CVPR_2024_paper.pdf)
   - 향후 연구에서는  
     - “**2D priors는 유지하되, 3D inductive bias를 어떻게 최소 수정으로 주입할 것인가**”가 핵심 설계 문제가 될 것.

3. **학습 목표: NVS, 3D reconstruction, downstream task를 동시에 최적화**

   - Zero-1-to-3는 주로 **NVS pixel fidelity**를 중심으로 학습되지만,  
   - 향후에는:
     - multi-view photometric consistency,  
     - 3D IoU, Chamfer, normal alignment,  
     - depth·segmentation·pose 등의 auxiliary task를 함께 고려하는  
     - **multi-task / multi-objective training**이 일반화 성능을 크게 끌어올릴 수 있다. [cg.cs.tsinghua.edu](https://cg.cs.tsinghua.edu.cn/papers/CVPR-2024-Wonder3D.pdf)

4. **평가: open-world·scene-level·robustness 지표 설계**

   - GSO·RTMV 같은 synthetic·semi-synthetic 데이터뿐 아니라,  
     - 실제사진, 예술작품, generated images(DALL·E, SDXL 등)까지 포함하는  
     - **open-world benchmark**를 정의할 필요.  
   - 또한 scene-level 깊이·object-ground 관계·occlusion reasoning 등을 포함해  
     - 2D 지표(PSNR·SSIM·LPIPS·FID)뿐 아니라  
     - 3D 인지·이해 관점의 지표를 도입하는 것이 중요. [semanticscholar](https://www.semanticscholar.org/paper/84cce9b8aea35e4fa38eef63da439573f21c0728)

5. **효율성과 실용성: feed-forward vs per-shape optimization**

   - Zero-1-to-3 자체는 fast NVS를 제공하지만,  
     - 3D 복원은 여전히 per-shape optimization에 의존.  
   - One-2-3-45, Wonder3D, LRM, TripoSR 등은  
     - feed-forward 3D 네트워크로 이 문제를 해결하려 한다. [arxiv](https://arxiv.org/abs/2306.16928)
   - 연구 설계 시,  
     - **연산비용·추론시간·메모리 footprint**를 명시적으로 제약 조건에 두고  
     - 품질·일반화 성능과의 trade-off를 체계적으로 분석하는 것이 실제 응용(게임·AR·로보틱스 등)에 중요하다.

***

## 7. 정리

Zero-1-to-3는 “**범용 2D 확산 모델이 내장한 3D priors를 synthetic 3D 데이터와 viewpoint conditioning으로 꺼내 쓰는 방법**”을 처음으로 체계적으로 보여 준 작업이다. [arxiv](https://arxiv.org/abs/2303.11328)

- single-view NVS와 zero-shot 3D reconstruction에서 SOTA를 달성했고,  
- 이후 Zero123++·SyncDreamer·Wonder3D·Stable Zero123·Cascade-Zero123·Consistent123 등의 풍부한 후속 연구를 촉발하며,  
- 오늘날 **“Single image → consistent multi-view → 3D”**가 하나의 표준 파이프라인으로 자리 잡는 데 결정적 역할을 했다. [arxiv](https://arxiv.org/pdf/2309.17261.pdf)

향후 연구에서는,

- 더 큰 3D·2D 데이터,  
- 3D-aware diffusion·3D feedback 구조,  
- multi-task objective,  
- scene-level generalization,

을 결합해, **로버스트한 open-world single-view 3D 이해 모델**로 진화하는 것이 핵심 방향이 될 것이다.

<span style="display:none">[^1_100][^1_101][^1_102][^1_103][^1_104][^1_105][^1_106][^1_59][^1_60][^1_61][^1_62][^1_63][^1_64][^1_65][^1_66][^1_67][^1_68][^1_69][^1_70][^1_71][^1_72][^1_73][^1_74][^1_75][^1_76][^1_77][^1_78][^1_79][^1_80][^1_81][^1_82][^1_83][^1_84][^1_85][^1_86][^1_87][^1_88][^1_89][^1_90][^1_91][^1_92][^1_93][^1_94][^1_95][^1_96][^1_97][^1_98][^1_99]</span>

<div align="center">⁂</div>

[^1_1]: 2303.11328v1.pdf

[^1_2]: https://openaccess.thecvf.com/content/ICCV2023/papers/Liu_Zero-1-to-3_Zero-shot_One_Image_to_3D_Object_ICCV_2023_paper.pdf

[^1_3]: https://arxiv.org/abs/2303.11328

[^1_4]: https://arxiv.org/pdf/2310.15110.pdf

[^1_5]: https://arxiv.org/html/2503.12929v1

[^1_6]: https://xoft.tistory.com/61

[^1_7]: https://arxiv.org/html/2507.05819v1

[^1_8]: https://arxiv.org/html/2411.02293v3/

[^1_9]: http://arxiv.org/abs/2212.03267

[^1_10]: https://openaccess.thecvf.com/content/CVPR2023/papers/Xu_NeuralLift-360_Lifting_an_In-the-Wild_2D_Photo_to_a_3D_Object_CVPR_2023_paper.pdf

[^1_11]: https://openaccess.thecvf.com/content/CVPR2023/html/Melas-Kyriazi_RealFusion_360deg_Reconstruction_of_Any_Object_From_a_Single_Image_CVPR_2023_paper.html

[^1_12]: https://arxiv.org/html/2505.08239v3

[^1_13]: https://www.semanticscholar.org/paper/Zero-1-to-3:-Zero-shot-One-Image-to-3D-Object-Liu-Wu/2c70684973bc4d7b6f8404a647b8031c4d3c8383

[^1_14]: https://arxiv.org/html/2409.07452v1

[^1_15]: https://arxiv.org/pdf/2311.07885.pdf

[^1_16]: https://www.semanticscholar.org/paper/622cab9477f190ec9ef1d12e5e71ba36146ad694

[^1_17]: https://liner.com/review/makeit3d-highfidelity-3d-creation-from-single-image-with-diffusion-prior

[^1_18]: http://arxiv.org/pdf/2306.16928v1.pdf

[^1_19]: https://arxiv.org/html/2503.12929v3

[^1_20]: https://arxiv.org/abs/2303.14184

[^1_21]: http://arxiv.org/pdf/2303.14184.pdf

[^1_22]: https://arxiv.org/html/2511.22194v1

[^1_23]: https://arxiv.org/html/2309.03453v2

[^1_24]: https://arxiv.org/abs/2309.03453

[^1_25]: https://cg.cs.tsinghua.edu.cn/papers/CVPR-2024-Wonder3D.pdf

[^1_26]: http://openaccess.thecvf.com/content/CVPR2024/papers/Long_Wonder3D_Single_Image_to_3D_using_Cross-Domain_Diffusion_CVPR_2024_paper.pdf

[^1_27]: https://arxiv.org/abs/2306.16928

[^1_28]: https://openaccess.thecvf.com/content/CVPR2024/supplemental/Liu_One-2-3-45_Fast_Single_CVPR_2024_supplemental.pdf

[^1_29]: https://www.semanticscholar.org/paper/84cce9b8aea35e4fa38eef63da439573f21c0728

[^1_30]: https://www.semanticscholar.org/paper/DreamFusion:-Text-to-3D-using-2D-Diffusion-Poole-Jain/4c94d04afa4309ec2f06bdd0fe3781f91461b362

[^1_31]: https://arxiv.org/pdf/2403.08902.pdf

[^1_32]: https://arxiv.org/pdf/2307.05663.pdf

[^1_33]: https://ieeexplore.ieee.org/document/10657980/

[^1_34]: https://arxiv.org/pdf/2308.12967.pdf

[^1_35]: https://arxiv.org/html/2601.11772v1

[^1_36]: https://stability.ai/news/stable-zero123-3d-generation

[^1_37]: https://arxiv.org/abs/2410.18974

[^1_38]: https://arxiv.org/abs/2310.15110

[^1_39]: https://arxiv.org/abs/2403.10953

[^1_40]: https://arxiv.org/pdf/2310.10343.pdf

[^1_41]: https://arxiv.org/pdf/2309.17261.pdf

[^1_42]: https://arxiv.org/html/2312.11535v2

[^1_43]: https://kimjy99.github.io/논문리뷰/one-2-3-45/

[^1_44]: https://openaccess.thecvf.com/content/CVPR2023/papers/Deng_NeRDi_Single-View_NeRF_Synthesis_With_Language-Guided_Diffusion_As_General_Image_CVPR_2023_paper.pdf

[^1_45]: https://arxiv.org/abs/2211.16431

[^1_46]: https://openaccess.thecvf.com/content/CVPR2023/html/Xu_NeuralLift-360_Lifting_an_In-the-Wild_2D_Photo_to_a_3D_Object_CVPR_2023_paper.html

[^1_47]: https://lukemelas.github.io/realfusion/

[^1_48]: https://github.com/cvlab-columbia/zero123

[^1_49]: https://kimjy99.github.io/논문리뷰/zero123plus/

[^1_50]: https://arxiv.org/abs/2312.04424

[^1_51]: http://arxiv.org/pdf/2312.04424.pdf

[^1_52]: https://arxiv.org/html/2312.04424v2

[^1_53]: https://liuyuan-pal.github.io/SyncDreamer/

[^1_54]: https://arxiv.org/html/2312.13271v3

[^1_55]: https://www.xxlong.site/Wonder3D/

[^1_56]: https://dl.acm.org/doi/10.1145/3610548.3618153

[^1_57]: http://arxiv.org/pdf/2312.04784.pdf

[^1_58]: https://arxiv.org/pdf/2211.16431.pdf

[^1_59]: https://ieeexplore.ieee.org/document/11072208/

[^1_60]: https://www.semanticscholar.org/paper/7b788a3fe0223b15264676d93f3d9de1ae3c3e42

[^1_61]: https://arxiv.org/abs/2307.14770

[^1_62]: https://www.semanticscholar.org/paper/15ea382f142cdb123b7dd4500ddae3a9abfa48a9

[^1_63]: https://arxiv.org/abs/2310.17994

[^1_64]: https://ieeexplore.ieee.org/document/10377650/

[^1_65]: https://ieeexplore.ieee.org/document/10282148/

[^1_66]: http://arxiv.org/pdf/2411.18623.pdf

[^1_67]: https://arxiv.org/html/2503.02410v1

[^1_68]: https://arxiv.org/html/2403.18922v1

[^1_69]: https://arxiv.org/html/2412.17812v1

[^1_70]: https://liner.com/review/neurallift360-lifting-inthewild-2d-photo-to-3d-object-with-360

[^1_71]: https://velog.io/@jameskoo0503/RealFusion-360-Reconstruction-of-Any-Object-from-a-Single-Image

[^1_72]: https://www.youtube.com/watch?v=K0IcBBfEwCc

[^1_73]: https://liner.com/review/realfusion-360-reconstruction-any-object-from-single-image

[^1_74]: https://openaccess.thecvf.com/content/CVPR2023/html/Deng_NeRDi_Single-View_NeRF_Synthesis_With_Language-Guided_Diffusion_As_General_Image_CVPR_2023_paper.html

[^1_75]: https://github.com/VITA-Group/NeuralLift-360

[^1_76]: https://blog.csdn.net/NGUever15/article/details/128529425

[^1_77]: https://xoft.tistory.com/68

[^1_78]: https://kimjy99.github.io/논문리뷰/realfusion/

[^1_79]: https://waymo.com/research/nerdi-single-view-nerf-synthesis-with-language-guided-diffusion-as-general/

[^1_80]: https://arxiv.org/abs/2212.03267

[^1_81]: https://www.semanticscholar.org/paper/e3a72cc5c29c9b245a0e589384a2adbcfa4a03c0

[^1_82]: https://openaccess.thecvf.com/content/CVPR2023/supplemental/Melas-Kyriazi_RealFusion_360deg_Reconstruction_CVPR_2023_supplemental.pdf

[^1_83]: https://openaccess.thecvf.com/content/CVPR2023/supplemental/Deng_NeRDi_Single-View_NeRF_CVPR_2023_supplemental.pdf

[^1_84]: https://arxiv.org/pdf/2303.11328.pdf

[^1_85]: https://arxiv.org/abs/2302.10663

[^1_86]: https://www.semanticscholar.org/paper/NeRDi:-Single-View-NeRF-Synthesis-with-Diffusion-as-Deng-Jiang/e3f5a9251529f34bc15b89e3294e576efbc0af4c

[^1_87]: https://arxiv.org/pdf/2311.10123.pdf

[^1_88]: https://openaccess.thecvf.com/content/CVPR2023/papers/Melas-Kyriazi_RealFusion_360deg_Reconstruction_of_Any_Object_From_a_Single_Image_CVPR_2023_paper.pdf

[^1_89]: https://arxiv.org/pdf/2212.03267.pdf

[^1_90]: https://arxiv.org/abs/2407.10558

[^1_91]: https://academic.oup.com/jcde/article/12/12/70/8304017

[^1_92]: https://journals.viamedica.pl/rpor/article/view/98735

[^1_93]: https://ijgc.bmj.com/lookup/doi/10.1136/ijgc-2023-IGCS.171

[^1_94]: https://www.youtube.com/watch?v=9BAmgRK-29c

[^1_95]: https://www.youtube.com/watch?v=nfcuAXJXzCU

[^1_96]: https://liner.com/review/wonder3d-single-image-to-3d-using-crossdomain-diffusion

[^1_97]: https://liner.com/review/syncdreamer-generating-multiviewconsistent-images-from-a-singleview-image

[^1_98]: https://www.runcomfy.com/comfyui-workflows/wonder3d-single-view-3d-reconstruction

[^1_99]: https://openreview.net/forum?id=MN3yH2ovHb

[^1_100]: https://github.com/xxlong0/Wonder3D

[^1_101]: https://openaccess.thecvf.com/content/CVPR2024/supplemental/Long_Wonder3D_Single_Image_CVPR_2024_supplemental.pdf

[^1_102]: https://www.semanticscholar.org/paper/SyncDreamer:-Generating-Multiview-consistent-Images-Liu-Lin/fcd0de4066d93fa3822a14898008fa2dd99f7be6

[^1_103]: https://arxiv.org/abs/2310.15008

[^1_104]: https://arxiv.org/html/2507.02299v1

[^1_105]: https://arxiv.org/html/2412.06614v1

[^1_106]: https://www.semanticscholar.org/paper/Wonder3D:-Single-Image-to-3D-Using-Cross-Domain-Long-Guo/d2c5565a039f464b778e0f2263da418ef42e98b0
