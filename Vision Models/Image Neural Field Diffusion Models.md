
# Image Neural Field Diffusion Models

> **논문 정보**
> - **제목**: Image Neural Field Diffusion Models
> - **저자**: Yinbo Chen (UC San Diego), Oliver Wang (Google Research), Richard Zhang, Eli Shechtman, Michael Gharbi (Adobe Research), Xiaolong Wang (UC San Diego)
> - **학회**: CVPR 2024 Highlight
> - **arXiv**: [2406.07480](https://arxiv.org/abs/2406.07480) | **프로젝트 페이지**: [yinboc.github.io/infd](https://yinboc.github.io/infd/)

---

## 1. 핵심 주장 및 주요 기여 (간결 요약)

### 🔑 핵심 주장

Diffusion 모델은 GAN 대비 안정적 학습, 분포 모드의 우수한 커버리지, 역문제 해결 능력 등의 장점을 가지지만, 대부분의 Diffusion 모델은 **고정 해상도 이미지**의 분포만 학습한다. 이 논문은 임의의 해상도로 렌더링 가능한 **Image Neural Field**를 통해 연속적(continuous) 이미지의 분포를 학습하는 방법을 제안하고, 이것이 고정 해상도 모델 대비 우월함을 보인다.

### 🏆 주요 기여 (3가지)

① 혼합 해상도(mixed-resolution) 이미지 데이터셋으로 학습 가능, ② 고정 해상도 Diffusion 모델 + 별도 초해상도(SR) 모델 파이프라인을 성능에서 능가, ③ 서로 다른 스케일에서 조건이 부여된 역문제를 효율적으로 해결 가능.

---

## 2. 상세 분석

### 2-1. 해결하고자 하는 문제

고해상도 이미지(예: 2K)를 생성하기 위해 기존 LDM(Latent Diffusion Model)은 먼저 저해상도 이미지를 생성한 후, **별도의 초해상도 모델로 업샘플링하는 2단계 파이프라인**에 의존한다.

이러한 접근 방식은 다음의 구조적 문제를 내포한다:

- **해상도 고정성**: 학습 시 사용한 해상도 이외에는 직접 생성 불가
- **파이프라인 복잡성**: SR 모델을 별도로 학습해야 하며, 오류가 누적됨
- **스케일 불일치**: 서로 다른 스케일의 조건을 하나의 모델로 처리하기 어려움
- **혼합 해상도 학습 불가**: 다양한 해상도의 데이터셋을 균일하게 활용하기 어려움

### 2-2. 제안하는 방법

#### (1) 전체 프레임워크: INFD (Image Neural Field Diffusion)

이 논문은 **INFD(Image Neural Field Diffusion)**를 제안한다. 이 방법은 Latent Diffusion 프레임워크를 기반으로 하며, 먼저 임의의 해상도로 렌더링 가능한 이미지 뉴럴 필드를 표현하는 잠재 표현(latent representation)을 학습하고, 이후 이 잠재 표현 위에 Diffusion 모델을 학습한다.

#### (2) 인코더-디코더 구조

임의 해상도의 학습 이미지가 주어지면, 먼저 이를 고정 해상도로 다운샘플링하여 인코더 $E$에 통과시켜 잠재 표현 $z$를 얻는다. 디코더 $D$는 $z$를 입력으로 받아 특징 맵 $\phi$를 생성하고, 이것이 뉴럴 필드 렌더러 $R$을 구동한다. 렌더러는 적절한 픽셀 좌표 그리드 $c$와 픽셀 크기 $s$를 질의하여 이미지를 렌더링할 수 있다. 오토인코더는 랜덤으로 다운샘플된 이미지의 크롭을 기반으로 학습된다. 테스트 시에는 Diffusion 모델이 잠재 표현 $z$를 생성하고, 이를 디코딩하여 고해상도 이미지를 렌더링한다.

수식으로 표현하면:

$$z = E(I_{\downarrow})$$

$$\phi = D(z)$$

$$\hat{I}(c, s) = R(\phi, c, s)$$

여기서 $I_{\downarrow}$는 고정 해상도로 다운샘플된 이미지, $c$는 픽셀 좌표, $s$는 픽셀 크기(pixel size)이다.

#### (3) 핵심 모듈: CLIF (Convolutional Local Image Function)

기존의 LIIF(Local Implicit Image Function)를 직접 오토인코더에 구현하면 이미지 세부 묘사가 흐려지는 문제가 발생함을 발견하고, **CLIF(Convolutional Local Image Function)**를 제안한다. CLIF는 잠재 표현을 사실적인 고해상도 이미지로 렌더링하며, 서로 다른 해상도에서도 이미지 내용이 일관성을 유지한다.

CLIF는 특징 맵 $\phi$ (yellow dots)가 주어지면, 각 질의점 $x$ (green dot)에 대해 가장 가까운 특징 벡터와 상대 좌표, 픽셀 크기를 가져온다. 이 질의 정보의 그리드는 RGB 그리드를 렌더링하는 **합성곱 네트워크(convolutional network)**로 전달된다. 포인트 단위 함수인 LIIF와 달리, CLIF는 더 높은 생성 능력을 가지며 스케일 일관성(scale-consistent)을 학습한다.

CLIF 렌더링 수식:

$$\hat{I}_{patch}(c, s) = \text{Conv}\left(\left[\phi_{\text{nearest}(x)},\; \Delta x,\; s\right]\right)$$

여기서 $\Delta x = x - \text{nearest feature location}$, $s$는 픽셀 크기를 나타낸다.

#### (4) 학습 목적 함수

뉴럴 필드 오토인코더는 LDM을 따라 **L1 손실, 지각 손실(perceptual loss), GAN 손실**로 학습되며, AnyResGAN과 유사하게 멀티스케일 패치 기반으로 지도(supervision)한다.

$$\mathcal{L}_{total} = \mathcal{L}_{L1} + \lambda_p \mathcal{L}_{perceptual} + \lambda_{adv} \mathcal{L}_{GAN}$$

#### (5) Diffusion 모델 학습

표준 DDPM/DDIM 공식을 잠재 공간 $z$에 적용한다:

**Forward Process:**
$$q(z_t | z_{t-1}) = \mathcal{N}(z_t;\; \sqrt{1-\beta_t}\, z_{t-1},\; \beta_t \mathbf{I})$$

**Reverse Process (학습 목표):**

$$\mathcal{L}_{DM} = \mathbb{E}_{z, \epsilon \sim \mathcal{N}(0,1), t}\left[\|\epsilon - \epsilon_\theta(z_t, t)\|^2\right]$$

이 효율적인 Diffusion 프로세스는 $64 \times 64$ 해상도의 잠재 표현만으로도 $2\text{K}$ 해상도의 사실적인 고해상도 이미지를 렌더링할 수 있다.

---

### 2-3. 모델 구조 요약

```
[Training Image (임의 해상도)]
        ↓  Downsample
[Fixed-Resolution Input (e.g., 256×256)]
        ↓  Encoder E
[Latent z (64×64)]
        ↓  Decoder D
[Feature Map φ]
        ↓  CLIF Renderer R(φ, c, s)
[High-Resolution Output (e.g., 2048×2048)]
```

- **Diffusion 모델**: LDM 기반, 잠재 공간 $z$ 위에서 동작
- **오토인코더**: 기존 LDM의 오토인코더를 Neural Field 오토인코더로 변환 가능
- **CLIF**: 합성곱 네트워크 기반 뉴럴 필드 렌더러 (LIIF 대체)

텍스트-이미지 합성에서는 기존 **Stable Diffusion** 체크포인트를 고해상도 이미지를 포함한 LAION 데이터셋의 소규모 서브셋으로 파인튜닝하였다.

---

### 2-4. 성능 향상

INFD는 혼합 해상도 이미지 데이터셋으로 학습 가능하며, 고정 해상도 Diffusion 모델 + 초해상도 모델 파이프라인보다 우수한 성능을 달성하고, 다양한 스케일에서의 조건을 활용한 역문제를 효율적으로 해결한다.

정성적 비교에서 이미지 뉴럴 필드 Diffusion 모델은 Diffusion 기반으로 FFHQ(검은 점)와 산악 데이터셋(격자 패턴)에서 나타나는 **GAN 아티팩트를 방지**하며 AnyResGAN보다 우월하다.

Stable Diffusion을 CLIF 렌더러로 파인튜닝하여 $2048 \times 2048$ 해상도 출력이 가능하며, 멀티스케일 조건(사각 영역 + 텍스트 프롬프트)을 만족하는 고해상도 이미지를 생성할 수 있다. 이를 위해 $224 \times 224$로 해당 영역을 렌더링하여 사전 학습된 CLIP 모델에 통과시키고 CLIP 유사도를 최대화한다. 이로써 **추가적인 task-specific 학습 없이 레이아웃-이미지 생성(layout-to-image generation)**이 가능하다.

명시적인 일관성 목적 함수 없이도 CLIF가 **스케일 일관성(scale-consistent)**을 학습하며, 서로 다른 해상도로 렌더링해도 내용이 정렬됨을 확인하였다.

---

### 2-5. 한계점

텍스트-이미지 합성 시 두 가지 문제가 발생한다. 첫째, 사전 학습 모델의 학습 데이터에는 노이즈가 포함되어 있지만 고해상도 파인튜닝 데이터셋은 클린 이미지만 포함하여 스케일 일관성 가정을 위반한다. 이로 인해 고해상도 이미지를 생성하려면 **"4k" 같은 추가 프롬프트**가 필요하다. 둘째, 파인튜닝에 사용된 LAION 서브셋이 모든 객체 카테고리를 포괄하지 못하므로, **분포 외(out-of-distribution) 객체에서 성능이 저하**될 수 있다.

또한 논문은 뉴럴 필드 표현의 **계산 복잡도와 학습 요구사항**을 충분히 논의하지 않으며, 생성된 이미지의 해석 가능성(interpretability) 및 강건성(robustness)에 대한 탐구도 부족하다.

---

## 3. 일반화 성능 향상 가능성 🔍

INFD의 일반화 가능성은 다음 세 축에서 분석할 수 있다:

### 3-1. 해상도 일반화 (Resolution Generalization)

이미지 뉴럴 필드는 임의의 해상도로 렌더링 가능하므로, 학습 시 보지 못한 해상도에도 적용 가능한 구조를 갖는다.

CLIF는 LIIF와 달리 더 높은 생성 능력을 보유하며, 명시적인 일관성 목적 함수 없이도 스케일 일관성을 학습한다. 서로 다른 해상도에서 렌더링하더라도 이미지 내용이 정렬된다.

### 3-2. 데이터 일반화 (Data Generalization)

기존의 Latent Diffusion 오토인코더를 이미지 뉴럴 필드 오토인코더로 변환할 수 있는 방법을 제공함으로써, 대규모 사전 학습된 모델의 지식을 활용한 일반화가 가능하다.

혼합 해상도(mixed-resolution) 이미지 데이터셋을 활용한 학습이 가능하여, 현실 세계의 다양한 해상도 데이터로의 일반화 능력이 향상된다.

### 3-3. 역문제 일반화 (Inverse Problem Generalization)

멀티스케일 조건(사각 영역 + 텍스트 프롬프트)을 정의하고, 해당 영역을 렌더링한 뒤 사전 학습된 CLIP 모델과의 유사도를 최대화함으로써, **추가적인 task-specific 학습 없이** 레이아웃-이미지 생성이 가능하다. 이는 다양한 역문제로의 일반화 가능성을 보여준다.

### 3-4. 일반화의 한계

그러나 파인튜닝 데이터셋이 특정 카테고리에 편향되어 있어, **분포 외 객체에서 성능이 최적화되지 않을 수 있다**는 한계가 존재한다.

---

## 4. 향후 연구에 미치는 영향 및 고려할 점

### 4-1. 연구에 미치는 영향

#### 📌 (1) 연속 이미지 생성 패러다임 전환
뉴럴 필드를 활용하여 복잡한 공간적·의미적 관계를 포착하는 INFDMs는 이미지 합성, 인페인팅, 초해상도 등 다양한 작업에서 Diffusion 기반 이미지 생성의 최신 기술을 크게 진보시킬 잠재력을 가진다.

#### 📌 (2) 기존 모델과의 호환성
본 방법은 기존의 Latent Diffusion 오토인코더를 이미지 뉴럴 필드 오토인코더로 변환하는 데 활용될 수 있어, Stable Diffusion 등 기존 대형 모델의 업그레이드 경로를 제시한다.

#### 📌 (3) 멀티스케일 역문제 해결의 새 방향
이 논문은 단일 Diffusion 모델로 멀티스케일 조건을 처리하는 방법을 보여줌으로써, 이미지 편집, 인페인팅, 레이아웃 기반 생성 등 다양한 응용 연구에 영감을 준다.

#### 📌 (4) 후속 연구 자극: PixNerd
INFD의 영향을 받아 PixNerd(Pixel Neural Field Diffusion)가 제안되었으며, 이는 단일 스케일·단일 단계의 효율적인 엔드-투-엔드 솔루션으로 VAE 없이 ImageNet 512×512에서 FID 2.15를 달성하였다.

---

### 4-2. 앞으로 연구 시 고려할 점

| 고려 사항 | 설명 |
|-----------|------|
| **계산 복잡도 최적화** | 고해상도 뉴럴 필드 렌더링은 추가적인 계산 비용이 발생하므로, 효율적인 렌더러 설계 연구가 필요함 |
| **스케일 일관성 목적 함수** | 현재 CLIF는 암묵적으로 스케일 일관성을 학습하나, 명시적 일관성 목적 함수 도입 시 추가 성능 개선 여지가 있음 |
| **학습 데이터 편향 해결** | 파인튜닝 데이터셋이 모든 객체 카테고리를 포괄하지 못하는 문제를 해결하기 위한 더 다양하고 대규모적인 고해상도 데이터셋 구축이 필요 |
| **분포 외 해상도 성능** | 학습 범위를 크게 벗어난 초고해상도(예: 4K, 8K)에서의 품질 저하 가능성 분석 필요 |
| **해석 가능성 및 강건성** | 계산 복잡도 및 학습 요구사항, 그리고 생성 이미지의 해석 가능성과 강건성에 대한 심층 연구가 필요하다 |
| **3D 확장** | 2D 이미지 뉴럴 필드를 NeRF 등 3D 뉴럴 필드와 결합한 3D 콘텐츠 생성으로의 확장 탐색 |
| **비디오 생성 적용** | 시간 축으로 연속적인 뉴럴 필드를 학습하는 방식의 비디오 생성 모델로의 확장 가능성 |

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

| 모델 | 연도 | 핵심 방법 | 해상도 유연성 | 주요 장점 |
|------|------|-----------|--------------|-----------|
| **DDPM** (Ho et al.) | 2020 | Denoising score matching | 고정 | Diffusion 모델의 기초 |
| **LDM / Stable Diffusion** (Rombach et al.) | 2022 | 잠재 공간 기반 Diffusion | 고정 | 효율적, 대규모 적용 가능 |
| **LIIF** (Chen et al.) | 2021 | 로컬 암시적 이미지 함수 | 임의 | 임의 스케일 SR 가능 |
| **AnyResGAN** | ~2023 | GAN 기반 임의 해상도 생성 | 임의 | 해상도 유연성 |
| **NeuralField-LDM** | 2023 | 3D 뉴럴 필드 + LDM | 임의(3D) | 3D 씬 생성 가능 |
| **INFD (본 논문)** | 2024 | CLIF + LDM | 임의(2D) | 고품질, 스케일 일관성, 역문제 해결 |
| **PixNerd** | 2025 | 픽셀 뉴럴 필드 Diffusion | 임의 | VAE 없는 엔드-투-엔드, FID 2.15 |

INFD의 능력과 한계를 완전히 이해하고, 핵심 방법론의 잠재적 확장과 개선을 탐색하기 위한 추가 연구가 필요하다. 그럼에도 불구하고 이 논문의 결과는 Diffusion 모델에서 뉴럴 필드의 활용이 이미지 생성 커뮤니티에 유망한 탐구 영역임을 시사한다.

---

## 📚 참고 자료 및 출처

1. **Chen, Y., Wang, O., Zhang, R., Shechtman, E., Wang, X., & Gharbi, M. (2024)**. *Image Neural Field Diffusion Models*. CVPR 2024 (Highlight). Pages 8007–8017.
   - 프로젝트 페이지: https://yinboc.github.io/infd/
   - arXiv: https://arxiv.org/abs/2406.07480
   - CVF: https://openaccess.thecvf.com/content/CVPR2024/papers/Chen_Image_Neural_Field_Diffusion_Models_CVPR_2024_paper.pdf
   - Adobe Research: https://research.adobe.com/publication/image-neural-field-diffusion-models/
   - GitHub: https://github.com/yinboc/infd
   - AI Models: https://www.aimodels.fyi/papers/arxiv/image-neural-field-diffusion-models
   - ResearchGate: https://www.researchgate.net/publication/381318974_Image_Neural_Field_Diffusion_Models

2. **Chen, Y., Liu, S., et al. (2021)**. *Learning Continuous Image Representation with Local Implicit Image Function (LIIF)*. CVPR 2021.
   - https://openaccess.thecvf.com/content/CVPR2021/papers/Chen_Learning_Continuous_Image_Representation_With_Local_Implicit_Image_Function_CVPR_2021_paper.pdf

3. **Kim, S.W. et al. (2023)**. *NeuralField-LDM: Scene Generation with Hierarchical Latent Diffusion Models*. CVPR 2023.
   - https://research.nvidia.com/labs/toronto-ai/NFLDM/

4. **PixNerd: Pixel Neural Field Diffusion** (2025). arXiv:2507.23268.
   - https://arxiv.org/html/2507.23268v1

5. **Rombach, R. et al. (2022)**. *High-Resolution Image Synthesis with Latent Diffusion Models*. CVPR 2022. (Stable Diffusion)

---

> ⚠️ **정확도 참고**: 본 답변은 공식 프로젝트 페이지, arXiv 원문 HTML, CVF 공식 논문 PDF, ResearchGate 등 1차 출처를 기반으로 작성되었습니다. 논문 본문의 상세 수치(FID 등 정량 지표)는 원문 PDF를 직접 확인하시길 권장드립니다.
