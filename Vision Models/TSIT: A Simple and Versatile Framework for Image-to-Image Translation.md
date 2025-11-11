# TSIT: A Simple and Versatile Framework for Image-to-Image Translation

### 1. 핵심 주장 및 주요 기여

**TSIT (Two-Stream Image-to-image Translation)**는 **단순하면서도 범용적인 이미지 변환 프레임워크**로, 기존의 태스크 특화 방식의 한계를 극복하는 것을 목표로 합니다.[1]

#### 핵심 주장

TSIT의 가장 중요한 주장은 **정규화층(normalization layers)의 중요성**입니다. 논문은 콘텐츠와 스타일을 다중 스케일 피처 레벨에서 분리하고, 일관된 피처 변환 방식을 통해 이들을 효과적으로 융합할 수 있다고 제시합니다.[1]

#### 주요 기여

1. **범용적 프레임워크 제안**: 감독 학습(semantic image synthesis)과 비감독 학습(arbitrary style transfer) 모두에 적용 가능한 단일 프레임워크 개발
2. **다중 스케일 정규화 기법**: FADE(Feature Adaptive Denormalization)와 FAdaIN(Feature Adaptive Instance Normalization) 제시
3. **대칭 구조 설계**: 콘텐츠 스트림과 스타일 스트림을 대칭적으로 설계하여 효과적인 정보 융합 달성
4. **순환 일관성 제약 제거**: 복잡한 추가 제약 없이 표준 손실함수만으로도 경쟁력 있는 성능 달성

***

### 2. 문제 정의, 제안 방법 및 모델 구조

#### 2.1 문제 정의

기존 이미지 변환 방법들의 주요 문제점:[1]

- **비감독 방식의 한계**: 순환 일관성(cycle consistency), 의미 특징(semantic features) 등 추가 제약 필요
- **감독 방식의 한계**: 높은 해상도 생성, 공간적 일관성 유지를 위해 특화된 구조 필요
- **범용성 부족**: 서로 다른 특성의 태스크에 맞춰 구조를 다시 설계해야 함

TSIT는 이러한 문제들을 해결하기 위해 **멀티 스케일 피처 레벨에서 콘텐츠와 스타일을 모두 고려**하되, **표준 손실함수만으로도 동작하는 깔끔한 방법**을 추구합니다.

#### 2.2 제안 방법 및 수식

**TSIT의 핵심 메커니즘은 두 가지 피처 변환 기법으로 구성됩니다.**[1]

##### (1) Feature Adaptive Denormalization (FADE)

콘텐츠 스트림에서 추출한 다중 스케일 피처 표현 $$f^c_i$$를 사용하여 생성 네트워크의 정규화된 피처를 변환합니다:[1]

$$
\gamma^{l,h,w}_i \cdot \frac{z^{n,l,h,w}_i - \mu^l_i}{\sigma^l_i} + \beta^{l,h,w}_i
$$

여기서:
- $$\mu^l_i = \frac{1}{NH_iW_i}\sum_{n,h,w}z^{n,l,h,w}_i$$ (채널별 평균)
- $$\sigma^l_i = \sqrt{\frac{1}{NH_iW_i}\sum_{n,h,w}(z^{n,l,h,w}_i)^2 - (\mu^l_i)^2}$$ (표준편차)
- $$\gamma^{l,h,w}_i, \beta^{l,h,w}_i$$: 콘텐츠 피처에서 학습된 스케일과 바이어스 파라미터[1]

FADE는 **원소별 정규화 해제(element-wise denormalization)**를 수행하며, 피처 레벨에서의 의미 정보 보존이 SPADE 같은 이전 방법들보다 효과적입니다.[1]

##### (2) Feature Adaptive Instance Normalization (FAdaIN)

스타일 스트림에서 추출한 다중 스케일 피처 $$f^s_i$$를 사용하여 스타일 정보를 적응적으로 주입합니다:[1]

$$
\text{FAdaIN}(z_i, f^s_i) = \sigma(f^s_i)\left(\frac{z_i - \mu(z_i)}{\sigma(z_i)}\right) + \mu(f^s_i)
$$

여기서 $$\mu$$와 $$\sigma$$는 각각 평균과 표준편차입니다.[1]

이 기법은 적응적 인스턴스 정규화(AdaIN)를 일반화하여, 다중 스케일 스타일 특징을 효과적으로 인코딩합니다.[1]

#### 2.3 모델 구조

**TSIT 프레임워크는 4개의 주요 구성요소로 이루어져 있습니다:**[1]

1. **콘텐츠 스트림 (Content Stream, CS)**
   - 표준 잔차 블록 기반의 대칭 구조
   - 다운샘플링을 통해 다중 스케일 피처 추출
   - 피처 채널: 64 → 128 → 256 → 512 → 1024 → 1024 → 1024

2. **스타일 스트림 (Style Stream, SS)**
   - 콘텐츠 스트림과 동일한 대칭 구조
   - 스타일 이미지의 다중 스케일 표현 학습

3. **생성 네트워크 (Generator, G)**
   - 콘텐츠/스타일 스트림과 정확히 역구조
   - FADE 잔차 블록으로 구성
   - FAdaIN 모듈을 각 FADE 블록 이전에 적용
   - 가우시안 분포에서 샘플링한 노이즈 맵을 입력으로 사용

4. **차별자 (Discriminators)**
   - 다중 스케일 패치 기반 차별자
   - 3개의 스케일에서 작동하여 다양한 해상도 지원[1]

**구조의 핵심 특징**:
- **Coarse-to-fine 생성**: 고수준 잠재 코드에서 저수준 이미지 표현으로 점진적 정제
- **대칭성**: 스트림과 생성 네트워크의 대칭 구조로 의미 추상화 수준 일치
- **피처 레벨 주입**: 이미지 레벨이 아닌 피처 레벨에서의 적응적 정보 주입[1]

#### 2.4 목적함수

생성 네트워크의 손실함수:[1]

$$
L_G = -\mathbb{E}[D(g)] + \lambda_P L_P(g, x_c) + \lambda_{FM} L_{FM}(g, x_s)
$$

차별 네트워크의 손실함수:[1]

$$
L_D = -\mathbb{E}[\min(-1 + D(x_s), 0)] - \mathbb{E}[\min(-1 - D(g), 0)]
$$

여기서:
- $$g = G(z_0, x_c, x_s)$$: 생성된 이미지
- $$L_P$$: VGG-19 기반 지각 손실(perceptual loss)
- $$L_{FM}$$: 다중 스케일 차별자의 피처 매칭 손실
- $$\lambda_P, \lambda_{FM}$$: 손실 가중치[1]

***

### 3. 성능 향상 분석

#### 3.1 정량적 평가

**임의 스타일 변환 (Arbitrary Style Transfer):**[1]

| 태스크 | 방법 | FID ↓ | IS ↑ |
|------|------|-------|------|
| Summer→Winter | MUNIT | 118.225 | 2.537 |
| | DMIT | 87.969 | 2.884 |
| | **TSIT** | **80.138** | **2.996** |
| Day→Night | MUNIT | 110.011 | 2.185 |
| | DMIT | 83.898 | 2.156 |
| | **TSIT** | **79.697** | **2.203** |
| Photo→Art | MUNIT | 167.314 | 3.961 |
| | DMIT | 166.933 | 3.871 |
| | **TSIT** | **165.561** | **4.020** |

**의미 이미지 합성 (Semantic Image Synthesis):**[1]

| 데이터셋 | 메트릭 | CRN | SIMS | pix2pixHD | SPADE | CC-FPSE | **TSIT** |
|---------|--------|-----|------|-----------|-------|---------|----------|
| Cityscapes | mIoU ↑ | 52.4 | 47.2 | 58.3 | 62.3 | 65.5 | **65.9** |
| | accu ↑ | 77.1 | 75.5 | 81.4 | 81.9 | 82.3 | **82.7** |
| | FID ↓ | 104.7 | 49.7 | 95.0 | 71.8 | 54.3 | **59.2** |
| ADE20K | mIoU ↑ | 22.4 | N/A | 20.3 | 38.5 | 43.7 | **38.6** |
| | accu ↑ | 68.8 | N/A | 69.2 | 79.9 | 82.9 | **80.8** |
| | FID ↓ | 73.3 | N/A | 81.8 | 33.9 | 31.7 | **31.6** |

#### 3.2 정성적 성능

- **Yosemite 계절 변환**: MUNIT보다 명확하고 의미론적으로 더 정확한 결과
- **BDD100K 낮과 밤 변환**: 더욱 사실적인 샘플과 자연스러운 색상 생성
- **의미 이미지 합성**: 다양한 환경(실외, 실내, 거리)에서 더욱 견고한 성능[1]

#### 3.3 성능 향상의 근본 원인

**1. 다중 스케일 정규화의 효과성**
- 다양한 의미 추상화 수준에서 구조와 스타일 정보 포착
- Coarse-to-fine 방식으로 세부 정보 점진적 정제[1]

**2. 콘텐츠와 스타일의 효과적 분리**
- FADE: 공간적 구조 보존에 집중
- FAdaIN: 다중 스케일 스타일 특징 적응적 주입
- 이 두 기법의 조합이 인공물(artifacts) 감소[1]

**3. 다중 모달 합성 가능성**
- 단일 모델로 임의 스타일 제어
- 노이즈 입력을 통한 생성 다양성 확보[1]

---

### 4. 모델 한계 및 일반화 성능 분석

#### 4.1 명시적 한계

**1. Cityscapes ADE20K 성능 차이**[1]
- Cityscapes (구조화된 도시 장면): mIoU 65.9% (우수)
- ADE20K (복잡한 실내/야외): mIoU 38.6% (약화)
- **원인**: 복잡한 장면 의미론에서 FADE의 공간 정보 보존 한계

**2. Photo→Art 성능 정체**[1]
- 다른 태스크 대비 상대적으로 적은 성능 향상
- DMIT와 유사한 수준의 성능 유지

#### 4.2 일반화 성능 분석

**교차 검증 실험 (Cross Validation)에서의 한계 극복:**[1]

**MUNIT (비감독 방법)을 감독 문제에 적용:**
- 의미 이미지 합성 완전 실패
- 원인: 쌍이 없는 데이터 학습에 특화된 구조

**SPADE (감독 방법)를 비감독 문제에 적용:**
- 매우 강한 인공물 및 색상 왜곡
- 원인: 픽셀 레벨 마스크 리사이징의 부적절성

**TSIT의 우수성:**
- 두 역방향 모두에서 현저히 우수한 결과
- **핵심 이유**: 피처 레벨에서 콘텐츠와 스타일 학습[1]

#### 4.3 절제 연구 (Ablation Study)를 통한 일반화 성능 확인

| 모듈 | FID ↓ | IS ↑ | 의미론적 영향 |
|------|-------|------|----------|
| 전체 모델 | **85.876** | **2.934** | - |
| CS 제거 | 89.429 | 2.851 | 의미 구조 약화 (-3.1% FID) |
| SS 제거 | 86.263 | 2.734 | 다중 모달 불가능 (-0.5% FID) |
| FADE 제거 | 86.463 | 2.881 | 콘텐츠 누설 (-0.7% FID) |
| FAdaIN 제거 | 89.795 | 2.890 | 스타일 과잉 강화 (-3.1% FID) |

**분석**: 모든 구성 요소가 일반화에 필수적이며, 특히 두 스트림의 기여도가 중요함을 시사[1]

***

### 5. 최신 연구 발전과 미래 연구 방향

#### 5.1 TSIT 이후 발전된 연구 동향

**1. 확산 모델(Diffusion Models) 기반 이미지 변환의 부상**[2][3][4][5][6][7]

최근 이미지 변환 분야는 GAN 기반에서 **확산 모델 기반**으로 패러다임 변화:

- **DMT (Diffusion Model Translator, 2025)**: 경량 번역기를 통한 효율적 확산 기반 이미지 변환[8]
- **MirrorDiffusion (2024)**: 제로 샷 이미지 변환을 위한 확산 모델 활용[5]
- **Conditional Wavelet Diffusion Models (2024)**: 고해상도 3D 의료 이미지 합성[3]

**장점**: GAN 기반보다 훨씬 우수한 생성 품질과 다양성, 더 안정적인 학습[9][3]

**2. 비전 트랜스포머 (Vision Transformer) 통합**[10][11]

- **ViT-ClarityNet (2025)**: 수중 이미지 개선에 ViT 적용[10]
- **ViT-SGAN (2025)**: 텍스처 합성에 ViT와 공간 GAN 결합[11]
- **효과**: 더 나은 장거리 의존성(long-range dependency) 포착으로 일반화 성능 향상[11][10]

**3. 제로 샷 도메인 적응 (Zero-Shot Domain Adaptation)**[12][13][14][15]

- **SIDA (2025)**: 합성 이미지를 활용한 제로 샷 도메인 적응[12]
- **ZoDi (2024)**: 확산 모델 기반 제로 샷 도메인 적응[14]
- **PODA (2023)**: 프롬프트 기반 제로 샷 도메인 적응[15]

**TSIT와의 비교**:
- TSIT: 쌍 데이터 필요 (감독) 또는 비감독 설정
- 최신 방법: 텍스트 프롬프트나 합성 이미지만으로 도메인 적응 가능[14][12]

#### 5.2 TSIT의 영향 및 기여도

**학술적 영향**:
- 정규화층의 중요성을 명시적으로 강조
- FADE/FAdaIN 기법이 후속 연구의 기초 제공
- 범용 프레임워크 설계에 대한 새로운 관점 제시[1]

**한계**:
- GAN 기반 접근의 고질적 문제 (모드 붕괴, 학습 불안정성) 여전히 존재
- 확산 모델의 부상으로 GAN 기반 연구 감소[9][3]

#### 5.3 TSIT 기반 미래 연구 시 고려사항

**1. 확산 모델로의 마이그레이션**

TSIT의 핵심 아이디어를 확산 모델로 적응:
- FADE/FAdaIN의 컨셉을 확산의 조건부 생성 과정에 통합
- 피처 레벨 적응이 확산의 잠재 공간(latent space)에서 더욱 효과적[6][16][17]

**예시**: 확산 기반 TSIT 변형은 더 우수한 일반화와 안정성을 제공할 수 있음[18][6]

**2. 트랜스포머 기반 아키텍처**

- Vision Transformer를 두 스트림에 통합
- 전역 컨텍스트 포착 능력으로 복잡한 장면 의미론 개선
- 특히 ADE20K 같은 복잡한 데이터셋 성능 향상 가능[19][10][11]

**3. 멀티 모달 조건부 생성**

- TSIT의 다중 모달 능력 확장
- 텍스트 프롬프트 + 참조 이미지 + 의미 마스크 결합
- 제로 샷 도메인 적응 능력 추가[13][12][14]

**4. 메모리 효율성 및 확장성**

현재 TSIT:
- V100 GPU에서 약 10-32GB 메모리 소비[1]
- 더 큰 해상도나 배치 크기 확장 제한

개선 방향:
- 정규화 기법의 경량화
- 모듈식 설계로 부분 업데이트 가능하게 변경
- 최신 경량 생성 모델 기법 적용[20][8]

**5. 구성 일반화 (Compositional Generalization)**

- 새로운 콘텐츠-스타일 조합에 대한 외삽(extrapolation) 성능
- 길이 일반화: 학습 데이터보다 더 많은 객체 생성
- 지역성(locality) 메커니즘 통합[21]

***

### 결론

TSIT는 **정규화층의 역할을 재조명**하고, **다중 스케일 피처 레벨에서 콘텐츠와 스타일을 효과적으로 분리 및 융합**하여 범용적인 이미지 변환을 달성한 중요한 연구입니다. 절제 연구를 통해 각 구성 요소의 일반화 기여도를 명확히 했으며, 교차 검증 실험으로 기존 태스크 특화 방법들의 한계를 극복했습니다.[1]

그러나 **최근의 확산 모델과 비전 트랜스포머의 부상**은 TSIT와 같은 GAN 기반 접근의 향후 개선 방향을 제시합니다. 향후 연구는 TSIT의 핵심 개념(특히 FADE/FAdaIN)을 **확산 모델 기반 프레임워크와 트랜스포머 아키텍처에 통합**함으로써, 더욱 우수한 일반화 성능과 생성 안정성을 달성할 수 있을 것으로 예상됩니다.[3][21][6][9]

***

## 참고 문헌

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/196e5878-8783-4395-817a-4d6516b834a9/2007.12072v2.pdf)
[2](https://arxiv.org/html/2404.09633)
[3](https://arxiv.org/abs/2411.17203)
[4](https://arxiv.org/pdf/2308.13767.pdf)
[5](https://arxiv.org/pdf/2401.03221.pdf)
[6](https://arxiv.org/html/2502.00307v1)
[7](https://arxiv.org/html/2407.03006)
[8](https://arxiv.org/abs/2502.00307)
[9](https://arxiv.org/abs/2506.17324)
[10](https://www.nature.com/articles/s41598-025-91212-8)
[11](https://arxiv.org/abs/2502.01842)
[12](https://arxiv.org/abs/2507.18632)
[13](https://proceedings.iclr.cc/paper_files/paper/2024/file/dbb8193ad7e6fcbc7bb62ed9ee835110-Paper-Conference.pdf)
[14](https://arxiv.org/abs/2403.13652)
[15](https://openaccess.thecvf.com/content/ICCV2023/papers/Fahes_PODA_Prompt-driven_Zero-shot_Domain_Adaptation_ICCV_2023_paper.pdf)
[16](https://arxiv.org/html/2408.00998v1)
[17](https://snu.elsevierpure.com/en/publications/diffusion-based-image-to-image-translation-bynoise-correction-via/)
[18](https://arxiv.org/html/2401.09742v2)
[19](https://openaccess.thecvf.com/content/WACV2023/papers/Lin_Vision_Transformer_for_NeRF-Based_View_Synthesis_From_a_Single_Input_WACV_2023_paper.pdf)
[20](https://github.com/THU-LYJ-Lab/dmt)
[21](https://arxiv.org/abs/2509.16447)
[22](https://www.worldwidejournals.com/international-journal-of-scientific-research-(IJSR)/fileview/utilizing-anterior-tibial-translation-sign-on-magnetic-resonance-imaging-for-differential-diagnosis-of-partial-and-complete-anterior-cruciate-ligament-tears_May_2024_7731715151_1408684.pdf)
[23](https://biss.pensoft.net/article/135629/)
[24](https://eurasia-art.ru/art/article/view/1063)
[25](https://jamanetwork.com/journals/jamacardiology/fullarticle/2817468)
[26](https://ieeexplore.ieee.org/document/10447455/)
[27](https://jamanetwork.com/journals/jamaneurology/fullarticle/2815043)
[28](https://aacrjournals.org/mct/article/23/6_Supplement/B022/745667/Abstract-B022-TEAD-inhibition-overcomes-YAP-TAZ)
[29](https://journals.openedition.org/signata/5290)
[30](https://aacrjournals.org/cancerres/article/84/6_Supplement/232/737736/Abstract-232-Multi-dimensional-patient-derived)
[31](https://www.researchprotocols.org/2024/1/e56562)
[32](https://arxiv.org/pdf/2303.16280.pdf)
[33](https://arxiv.org/pdf/2111.05826v1.pdf)
[34](https://arxiv.org/abs/1805.11145)
[35](http://arxiv.org/pdf/2205.12952.pdf)
[36](https://arxiv.org/pdf/1701.02676.pdf)
[37](http://arxiv.org/pdf/2403.20142.pdf)
[38](https://arxiv.org/pdf/2203.08382.pdf)
[39](https://ieeexplore.ieee.org/document/10159430/)
[40](https://openaccess.thecvf.com/content/ICCV2023/papers/Cheng_General_Image-to-Image_Translation_with_One-Shot_Image_Guidance_ICCV_2023_paper.pdf)
[41](https://openaccess.thecvf.com/content/WACV2024/papers/Song_StyleGAN-Fusion_Diffusion_Guided_Domain_Adaptation_of_Image_Generators_WACV_2024_paper.pdf)
[42](https://www.techscience.com/cmc/v77n1/54482)
[43](https://www.jait.us/articles/2024/JAIT-V15N9-1019.pdf)
[44](https://www.semanticscholar.org/paper/da0bc8aff42754f8969484e80f399b06beb63ffb)
[45](https://ieeexplore.ieee.org/document/11147651/)
[46](https://link.springer.com/10.1007/s00261-025-05164-8)
[47](https://ieeexplore.ieee.org/document/10876546/)
[48](https://www.semanticscholar.org/paper/1a65219f0d3852b55d1fadf58e1ca75c1090805e)
[49](http://pubs.rsna.org/doi/10.1148/radiol.243423)
[50](https://aacrjournals.org/cancerres/article/85/8_Supplement_1/3410/757644/Abstract-3410-Fc-engineering-of-CD9-antibodies)
[51](https://arxiv.org/html/2409.00654)
[52](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123560392.pdf)
[53](https://openaccess.thecvf.com/content/CVPR2024/papers/Cai_Condition-Aware_Neural_Network_for_Controlled_Image_Generation_CVPR_2024_paper.pdf)
[54](https://arxiv.org/html/2409.04757v1)
[55](https://arxiv.org/html/2505.16001v1)
