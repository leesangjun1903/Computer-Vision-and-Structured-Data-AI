# Object-Centric Slot Diffusion

### 1. 핵심 주장과 주요 기여
**Object-Centric Slot Diffusion (LSD)**는 Latent Diffusion 모델을 객체 중심 학습에 처음으로 통합한 획기적 접근법이다. 논문의 핵심 주장은 기존의 약용량 mixture decoder나 autoregressive transformer decoder와 달리, diffusion 모델의 높은 표현력이 복잡한 자연 장면에서 object-centric 표현 학습을 크게 향상시킬 수 있다는 것이다.[1]

**주요 기여 사항**:

1. **첫 번째 object-centric learning 모델**: Conventional slot decoder를 latent diffusion model로 대체한 최초의 시도[1]
2. **첫 번째 unsupervised compositional conditional diffusion**: 텍스트 같은 supervised annotation 없이 시각적 개념을 추출하여 구성적 생성을 실현[1]
3. **FFHQ 적용의 돌파**: 고해상도(256×256) 자연 이미지 데이터셋에 object-centric 모델을 처음으로 성공적으로 적용[1]
4. **Pre-trained diffusion model 활용**: 기존에 모르던 object-centric representation을 실제 이미지에서 학습 가능함을 보여줌[1]

***

### 2. 해결하고자 하는 문제
Object-centric learning의 근본적인 도전 과제는 **복잡한 자연 장면 이미지로의 확장**이다. 기존 접근 방식의 문제점:[1]

- **Mixture decoder의 한계**: 약용량(weak-capacity) decoder는 단순 합성 이미지(CLEVR)에서는 효과적이지만, 복잡한 텍스처와 배경의 자연 이미지에서는 성능 저하[1]
- **Transformer decoder의 생성 품질 제한**: Singh et al. (2022)의 SLATE가 제시한 고용량 decoder도 생성 이미지에 흐림(blur)과 왜곡이 발생[1]
- **구성적 생성의 어려움**: 다양한 객체 조합으로부터 고품질 이미지를 생성하는 능력 부족[1]

***

### 3. 제안하는 방법 (수식 포함)
#### 3.1 Object-Centric Encoder: Slot Attention

입력 이미지 $x \in \mathbb{R}^{H \times W \times C}$에서 slot 표현 $S \in \mathbb{R}^{N \times D}$를 추출:

$$A = \text{softmax}_N\left(\frac{q(S) \cdot k(E)^T}{\sqrt{D}}\right) \Rightarrow A_{n,m} = \frac{A_{n,m}}{\sum_{m=1}^{M} A_{n,m}} \Rightarrow u_n = \sum_{m=1}^{M} v(E_m) A_{n,m}$$

여기서:
- $E \in \mathbb{R}^{M \times D_{input}}$: CNN backbone의 출력 특징
- $q, k, v$: 선형 투영 함수 (common dimension $D$로 매핑)
- $A$: $N \times M$ 주의 가중치 (각 입력 특징을 슬롯에 할당)
- $u_n$: 각 슬롯에 대한 attention readout

Slot은 RNN을 통해 반복적으로 정제됨: $s_n = f^{RNN}_\phi(s_n, u_n)$[1]

#### 3.2 Latent Slot Diffusion Decoder

**Pre-trained Auto-Encoder**:

$$z_0 = f^{AE}_\phi(x), \quad \hat{x} = g^{AE}_\theta(z_0)$$

여기서 $z_0 \in \mathbb{R}^{H_{AE} \times W_{AE} \times D_{AE}}$는 저차원 latent 표현이다. OpenImages에 사전학습된 KL-8 VAE 사용[1]

**Slot-Conditioned Diffusion**:

생성 분포를 T-step denoising 과정으로 모델링:

$$p_\theta(z_0|S) = \int p(z_T) \prod_{t=T,...,1} p_\theta(z_{t-1}|z_t, t, S) \, dz_{1:T}$$

여기서 $p(z_T) = \mathcal{N}(0, I)$이고, one-step denoising 분포는:

$$p_\theta(z_{t-1}|z_t, t, S) = \mathcal{N}\left(\frac{1}{\sqrt{\alpha_t}}\left(z_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}} \hat{\epsilon}_t\right), \beta_t I\right)$$

여기서:
- $\hat{\epsilon}\_t = g^{LSD}_\theta(z_t, t, S)$: 신경망이 예측한 노이즈
- $\beta_t$: linearly increasing variance schedule
- $\alpha_t = 1 - \beta_t$
- $\bar{\alpha}\_t = \prod_{i=1}^{t} (1-\beta_i)$

**Training 절차**:

Noisy latent를 생성:
$$z_t = \sqrt{\bar{\alpha}_t} z_0 + \sqrt{1-\bar{\alpha}_t} \epsilon_t, \quad \epsilon_t \sim \mathcal{N}(0, I)$$

네트워크를 MSE loss로 학습:
$$L(\phi, \theta) = ||\hat{\epsilon}_t - \epsilon_t||^2$$

실제 training에서는 uniform하게 noise level $t \in \{1, ..., T\}$를 샘플링[1]

#### 3.3 Denoising Network 아키텍처

UNet 기반 구조에 slot-conditioned transformer 통합:

$$\tilde{h}_l = \text{CNN}_l^\theta([h_{l-1}, h_{\text{skip}(l)}], t)$$
$$h_l = \text{Transformer}_l^\theta(\tilde{h}_l + p_l, \text{cond}=S)$$

**CNN Layers**: 표준 U-Net 구조로 feature map을 down/upsample, skip connection 활용[1]

**Slot-Conditioned Transformer**: 각 층에서 CNN 출력을 flatten하고 positional embedding $p_l$을 추가한 후, transformer 통과:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right) \cdot V$$

여기서:
- $Q = W_Q^{(i)} \cdot (\tilde{h}_l + p_l)$: denoising feature를 query로 변환
- $K = W_K^{(i)} \cdot S$, $V = W_V^{(i)} \cdot S$: slot을 key-value로 변환
- $W_Q^{(i)}, W_K^{(i)}, W_V^{(i)}$: multi-head 인덱스 $i$에 따른 학습 가능한 투영 행렬[1]

***

### 4. 모델 구조
전체 모델 파이프라인은 세 가지 주요 모듈로 구성:

**Encoding**: CNN backbone으로 이미지를 patch-level 특징으로 변환한 후, Slot Attention을 통해 각 slot이 하나의 객체에 할당되도록 학습[1]

**Diffusion Decoding**: 
1. Pre-trained AE로 이미지를 저차원 latent space로 압축
2. Slot 조건에 따라 noisy latent에서 progressively denoising
3. 최종 latent를 원본 해상도로 복원[1]

**Compositional Generation**: K-means clustering을 통해 시각적 개념 library를 구축, 다양한 개념 조합으로 새로운 이미지 생성[1]

***

### 5. 성능 향상 및 비교 분석
#### 5.1 Object Segmentation Performance
**정량적 결과** (mIoU - 높을수록 좋음):

| 데이터셋 | SLATE | SLATE+ | LSD | 개선율 (LSD vs SLATE+) |
|---------|-------|--------|-----|------------------------|
| CLEVRTex | 49.54 | 52.96 | **62.52** | +18.0% |
| MOVi-C | 37.75 | 36.44 | **44.19** | +21.3% |
| MOVi-E | 28.59 | 20.63 | **37.64** | +82.4% |

**핵심 발견**: 
- 복잡도가 증가할수록 LSD의 성능 우위가 극적으로 증가
- 가장 도전적인 MOVi-E에서 SLATE+보다 82.4% 향상 달성
- SLATE+보다 더 정교한 객체 경계 학습, 객체 분할 감소, 배경 분할 개선[1]

#### 5.2 Compositional Generation Quality

**FID 점수** (낮을수록 좋음):

| 데이터셋 | SLATE | SLATE+ | LSD | 개선율 |
|---------|-------|--------|-----|--------|
| CLEVRTex | 105.83 | 69.23 | **29.53** | -257% |
| MOVi-C | 170.83 | 148.27 | **69.12** | -147% |
| MOVi-E | 169.32 | 126.51 | **64.76** | -162% |
| FFHQ | 112.38 | 98.76 | **27.83** | -302% |

**획기적 성과**:
- 모든 데이터셋에서 FID 점수 2-3배 개선
- **FFHQ에 처음 적용**: 고해상도(256×256) 자연 이미지에서 object-centric 생성 가능
- 세밀한 얼굴 특징, 머리 스타일, 의류, 배경을 명확하게 생성[1]

#### 5.3 Downstream Property Prediction

Frozen slot으로부터 객체 속성 예측 능력 평가:

| 속성 | SLATE | SLATE+ | LSD |
|-----|-------|--------|-----|
| Shape Accuracy | 74.24 | 71.63 | **80.23** |
| Material Accuracy | 69.73 | 63.61 | **75.56** |
| Position MSE | 1.28 | 1.26 | **1.13** ↓ |

LSD의 슬롯이 더 높은 의미론적 정보를 캡처함을 입증[1]

***

### 6. 모델의 일반화 성능 향상 가능성 분석
#### 6.1 Scene Complexity에 따른 성능 곡선

논문의 가장 중요한 발견 중 하나는 **scene complexity가 증가할수록 diffusion decoder의 장점이 극대화**된다는 점이다:[1]

- **CLEVR (단순, 텍스처 없음)**: 보조적 성능 저하 관찰 (과적합 문제)
- **CLEVRTex (중간, 복잡한 배경)**: ~13% segmentation 개선
- **MOVi-C/E (복잡한 장면, 많은 객체)**: ~20-82% 개선
- **FFHQ (고해상도 자연 이미지)**: 첫 번째 성공적 적용

이는 diffusion 모델의 높은 표현력이 복잡한 시각적 분포를 더 잘 모델링함을 시사[1]

#### 6.2 CLEVR에서의 과적합 현상과 해결책

**문제**: CLEVR의 단순한 배경에서 background slots가 배경을 memorize하여 정보 혼합 발생[1]

**근본 원인**: 강력한 diffusion decoder가 slot conditioning 없이도 단순 배경을 독립적으로 렌더링 가능[1]

**해결책**: CLEVRTex 데이터(60개 배경 텍스처) 추가로 암시적 memorization 방지:

| 모델 | mBO | mIoU | FID |
|-----|-----|------|-----|
| LSD (CLEVR only) | 38.49 | 37.49 | **16.22** |
| LSD-Mix (CLEVR+CLEVRTex) | **61.09** | **59.76** | 19.45 |

~22% segmentation 개선 달성하면서 generation quality는 유지[1]

#### 6.3 Pre-trained Diffusion Model을 통한 Real-world 확장

**Stable-LSD 변형** (부록 제시):
- Pre-trained DINOv2 인코더 (frozen) + Pre-trained Stable Diffusion 활용
- Slot Attention만 학습 (계산량 크게 감소)
- Classifier-free guidance ($\text{cfg}=1.3$) 활용[1]

**성과**:
- COCO dataset에서 최초의 real-world compositional generation 달성
- Instance-level FG-ARI: 35.02 (DINOSAUR의 34.10과 비교)
- 의미론적 일관성을 유지하면서 시각적 다양성 생성[1]

***

### 7. 주요 한계점
#### 7.1 기술적 한계

1. **Part-whole Ambiguity**: 복잡한 텍스처를 가진 객체의 부분과 전체를 구분하기 어려움. 예: 실제 이미지의 사람을 상반신과 하반신으로 분할[1]

2. **Slot Number Sensitivity**: 적절한 슬롯 수를 미리 결정해야 하며, 실제 객체 수를 모를 때 문제[1]
   - 슬롯 수 < 객체 수: 의미론적 마스크 생성
   - 슬롯 수 > 객체 수: 객체의 과분할

3. **CLEVR 성능 저하**: 매우 단순한 이미지에서 과적합 현상[1]

#### 7.2 계산 및 확장성

- 훈련 시간: 2×RTX 6000 GPU에서 4.5일 소요 (SLATE는 1일, SLATE+는 2.7일)[1]
- 메모리 요구사항: 32GB (SLATE+는 36GB로 비슷)[1]
- 생성 시간: 50장 생성에 50.7초 (SLATE보다 빠름)[1]

#### 7.3 실무 적용 제한

- 다양한 외형의 실제 객체 처리 능력 제약
- 텍스처와 모양의 극단적 다양성에서 성능 저하 가능성
- 개인정보 보호 및 deepfake 생성 우려로 인한 윤리적 고려 필요[1]

***

### 8. 2020년 이후 관련 최신 연구 비교 분석
#### 8.1 진화의 타임라인
**1단계 (2020)**: **Slot Attention** 기초 확립
- Competitive spatial attention을 통한 객체 발견
- Weak spatial broadcast decoder 사용
- 단순 합성 이미지에만 효과적[2]

**2단계 (2022)**: **SLATE** - Transformer decoder 도입
- Autoregressive transformer로 decoder 용량 증대
- Compositional generation 능력 개선
- CLEVRTex, MOVi-E에서 성능 향상

**3단계 (2023)**: **LSD 및 SlotDiffusion** - Diffusion 모델 통합
- 두 논문 모두 diffusion 기반 decoder 제안 (동시 진행, 2023년 모두 발표)[3][4]
- LSD: unsupervised compositional generation, FFHQ 첫 적용
- SlotDiffusion: 비디오 확장, PASCAL VOC/COCO 확장성[3]

**4단계 (2024-2025)**: Guidance 및 Pre-trained Model 활용
- **GLASS** (2024): Semantic/instance guidance로 real-world 성능 향상[5]
- **SlotAdapt** (2025): Pre-trained diffusion model adapter 도입[6]

#### 8.2 LSD vs SlotDiffusion 상세 비교

| 측면 | LSD | SlotDiffusion |
|------|-----|--------------|
| **발표 시기** | 2023년 3월 (arXiv) | 2023년 5월 (NeurIPS) |
| **Auto-encoder** | OpenImages 사전학습 (공유) | 데이터셋별 독립 학습 |
| **주요 초점** | Unsupervised compositional generation | Video dynamics + real-world scaling |
| **FFHQ 적용** | **최초 달성** | 제한적 |
| **Real-world** | Control-LSD 통해 예비 탐색 | PASCAL VOC/COCO 체계적 평가 |
| **기술 혁신** | Slot-conditioned transformer | Video diffusion architecture 적응 |
| **한계점 분석** | Detailed (과적합 분석 포함) | 덜 자세함 |

**종합 평가**: LSD는 개념적 순결성(concept purity)과 일반화 분석에서 우수하며, SlotDiffusion은 확장성과 video 응용에서 강점[4][3]

#### 8.3 Recent Advances (2024-2025)

**GLASS** (2024, ECCV):
- Semantic/instance guidance 메커니즘 추가
- Real-world 이미지에 LSD보다 더 강력한 신호 제공
- 첫 compositional generation of complex realistic scenes 달성[5]

**SlotAdapt** (2025, ICLR):
- Pre-trained diffusion model (Stable Diffusion 등)을 adapter 통해 적응
- Foundation model의 강력한 표현력 활용
- Text-centric 편향 제거[7][6]

**Object-Centric Diffusion for Video Editing** (2024, ECCV):
- Object-centric sampling과 token merging으로 효율성 향상
- Diffusion step을 전경에 집중[8]

**Slot-guided Video Representation Learning** (2025):
- Video에서 slot이 시간 일관성을 유지하면서 학습
- Temporal dynamics 이해[9]

#### 8.4 기술 진화 분석

**Decoder 설계의 진화 경로**:

```
Mixture Decoder (약용량)
    ↓
Spatial Broadcast (약용량 기반)
    ↓
Autoregressive Transformer (고용량) [SLATE]
    ↓
Latent Diffusion (고표현력) [LSD, SlotDiffusion]
    ↓
Guided Latent Diffusion (외부 신호) [GLASS]
    ↓
Adapter-based Pre-trained Diffusion (foundation model) [SlotAdapt]
```

각 단계에서 decoder 용량과 표현력이 단조증가하며, 자연 이미지에 대한 적응성이 향상[1][3][5]

***

### 9. 향후 연구에 미치는 영향 및 고려 사항
#### 9.1 개념적 기여와 영향

1. **Diffusion-based Representation Learning의 개척**
   - 기존 diffusion model은 주로 supervised 생성 모델로 사용
   - LSD는 unsupervised object-centric learning에 diffusion의 수학적 프레임워크를 적용하여 새로운 패러다임 제시[1]
   - 후속 연구들이 이를 따라 diffusion + structured representation 결합 탐색[3][5]

2. **Compositional Generation의 새로운 방향**
   - 기존 방법: text prompt → 텍스트 조건 diffusion
   - LSD의 방법: unsupervised slots → slot 조건 diffusion
   - 이는 annotation 없이도 구성적 생성이 가능함을 보여주며, 자가지도 학습의 새로운 방향 제시[1]

3. **Multi-modal 연결의 가능성**
   - Slot representation을 텍스트, 음성 등 다른 모달리티와 연결하는 기초 제공
   - Stable-LSD 변형에서 classifier-free guidance를 slot에 적용 시도는 이러한 방향의 초석[1]

#### 9.2 기술적 개선 방향

##### 9.2.1 Decoder 아키텍처 최적화

**현황**: Diffusion decoder는 computation 비용이 높음 (4.5일 훈련)

**연구 방향**:
- Masked transformer decoder로 autoregressive 단계 제거 (계산 효율 개선)[10]
- Knowledge distillation으로 diffusion step 축소
- Hybrid decoder: 단순 배경은 빠른 방법, 복잡한 객체는 diffusion[8]

##### 9.2.2 일반화 성능 강화

**Part-whole Ambiguity 해결**:
- Hierarchical slot structure (여러 granularity level)
- 의미론적 가이드(semantic guidance) 통합 [GLASS 방향][5]
- 자기지도 신호(self-supervision) 활용

**Slot Number 자동 결정**:
- Dynamic slot allocation 메커니즘
- 정보이론 기반 최적 슬롯 수 학습
- Open-set object discovery

##### 9.2.3 Real-world Scaling

**Pre-trained Model 활용 강화** [SlotAdapt 방향]:[6]
- DINO, CLIP 등 강력한 vision encoder와 결합
- Foundation model의 semantic 이해 활용
- Domain adaptation 기법 통합

**자동 주석(auto-annotation) 파이프라인**:
- Self-supervised video encoder로 temporal consistency 강화
- Contrastive learning으로 slot clustering 개선
- Interactive learning으로 human feedback 통합

#### 9.3 응용 분야별 고려 사항

##### 9.3.1 비디오 및 동적 장면

**이슈**: 
- Temporal consistency 유지하면서 slot-based decomposition
- 동적 객체의 track 유지

**해결책**:
- Temporal slot refinement (현재의 공간 attention → 시공간 attention)[9]
- Dynamics model과의 결합 [SlotDiffusion 방향][3]
- Video prediction 통한 consistency 강화

##### 9.3.2 3D 장면 이해

**기회**:
- Slot representation을 3D coordinate로 확장
- DORSal처럼 NeRF + slot 결합[11]
- Multi-view consistent object discovery

##### 9.3.3 로봇 제어 및 계획

**응용**:
- Object-centric policy learning [SPOT, Lan-o3dp][12][13]
- Manipulation에서 충돌 회피 [language-guided object-centric diffusion][12]
- Slot-based action primitives

#### 9.4 이론적 분석 필요 영역

1. **Diffusion Model의 구성적 특성**: 왜 diffusion이 object-centric learning에 효과적인지 수학적 분석[14]

2. **Slot Representation의 의미성**: Slot이 어떤 방식으로 semantic 정보를 캡처하는지 분석[1][5]

3. **일반화 경계(Generalization Bound)**: 복잡도 함수, VC 차원 등을 통한 theoretical 분석

4. **Decoder Expressiveness의 정량화**: Mixture decoder vs Transformer vs Diffusion의 capacity 비교

#### 9.5 윤리 및 안전성 고려사항

**LSD 논문에서 언급한 주의사항**:

1. **이미지 조작 및 개인정보**
   - Deepfake 생성 가능성
   - Identity theft 위험
   - 얼굴 교체 기술의 부정적 활용[1]

2. **편향(Bias) 및 공정성**
   - K-means clustering의 uniform sampling이 데이터 편향 반영 가능
   - 인종, 성별, 나이 등 민감한 특성에서 misrepresentation 가능[1]

3. **규제 요구사항**
   - Strict ethical guidelines 필요
   - 법적 규제 프레임워크 마련
   - 사용 사례 제한 (의료, 금융, 보안 등)[1]

***

### 결론
**Object-Centric Slot Diffusion (LSD)**는 object-centric learning과 diffusion model의 결합이라는 단순하지만 강력한 아이디어로 이 분야에 혁신을 가져왔다. 복잡도 증가에 따른 성능 향상의 명확한 증거, FFHQ 데이터셋으로의 확장, 그리고 unsupervised compositional generation의 실현은 모두 이후 연구의 새로운 표준을 제시했다.[1]

2023년 이후의 진화(SlotDiffusion, GLASS, SlotAdapt)를 보면, LSD가 개척한 방향이 학계에서 얼마나 중요하게 평가되고 있는지를 알 수 있다. 향후 연구는 다음 세 가지 축에서 진행될 것으로 예상된다: **(1) 더욱 강력한 pretrained model과의 통합**, **(2) 실제 세계 적용을 위한 일반화 성능 강화**, **(3) 윤리적 프레임워크 구축**[5][6][7]

특히 foundation model 시대에 LSD의 insight를 어떻게 활용할 것인가가 향후 개년간의 주요 연구 질문이 될 것이다.

***

### 참고문헌

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/07bdee68-6e67-40f0-8c3e-5c9309be0f55/2303.10834v5.pdf)
[2](https://papers.neurips.cc/paper_files/paper/2020/file/8511df98c02ab60aea1b2356c013bc0f-Paper.pdf)
[3](https://arxiv.org/abs/2305.11281)
[4](https://arxiv.org/abs/2303.10834)
[5](https://ieeexplore.ieee.org/document/11093494/)
[6](https://arxiv.org/html/2501.15878)
[7](https://openreview.net/forum?id=kZvor5aaz7)
[8](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/07396.pdf)
[9](https://arxiv.org/html/2508.01345)
[10](https://openreview.net/pdf/161f6aeb8ec39f34952d96122738b14f936d8711.pdf)
[11](https://arxiv.org/abs/2306.08068)
[12](https://ieeexplore.ieee.org/document/11127231/)
[13](https://arxiv.org/html/2411.00965v1)
[14](https://proceedings.neurips.cc/paper_files/paper/2023/file/9d0f188c7947eacb0c07f709576824f6-Paper-Conference.pdf)
[15](https://ieeexplore.ieee.org/document/10678525/)
[16](https://arxiv.org/abs/2310.19145)
[17](https://arxiv.org/abs/2407.00451)
[18](https://www.semanticscholar.org/paper/945a899a93c03eb63be5e3197e318c077473cef9)
[19](https://ieeexplore.ieee.org/document/10744565/)
[20](https://arxiv.org/html/2305.11281)
[21](https://arxiv.org/pdf/2403.11208.pdf)
[22](https://arxiv.org/pdf/2306.08068.pdf)
[23](https://arxiv.org/html/2403.17827v2)
[24](https://arxiv.org/html/2407.00451v1)
[25](http://arxiv.org/pdf/2411.18660.pdf)
[26](https://proceedings.mlr.press/v202/du23a/du23a.pdf)
[27](https://arxiv.org/html/2507.04920v1)
[28](https://openaccess.thecvf.com/content/CVPR2024/papers/Kakogeorgiou_SPOT_Self-Training_with_Patch-Order_Permutation_for_Object-Centric_Learning_with_Autoregressive_CVPR_2024_paper.pdf)
[29](https://kimjy99.github.io/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0/lsd/)
[30](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136770426.pdf)
[31](https://ostin.tistory.com/387)
[32](https://arxiv.org/pdf/2507.04920.pdf)
[33](https://openaccess.thecvf.com/content/ICCV2023/papers/Liu_Unsupervised_Compositional_Concepts_Discovery_with_Text-to-Image_Generative_Models_ICCV_2023_paper.pdf)
[34](https://arxiv.org/html/2511.02225v1)
[35](https://arxiv.org/html/2405.20180v1)
[36](https://arxiv.org/pdf/2406.19298.pdf)
[37](https://arxiv.org/html/2507.23755)
[38](https://arxiv.org/abs/2408.09792)
[39](https://arxiv.org/html/2501.15878v3)
[40](https://paperswithcode.com/paper/slotdiffusion-object-centric-generative-1)
[41](https://aiflower.tistory.com/49)
[42](https://arxiv.org/abs/2206.01714)
