
# Wavelet Latent Diffusion (Wala): Billion-Parameter 3D Generative Model with Compact Wavelet Encodings

> **논문 정보**
> - 제목: *Wavelet Latent Diffusion (WaLa): Billion-Parameter 3D Generative Model with Compact Wavelet Encodings*
> - 저자: Aditya Sanghi, Aliasghar Khani, Pradyumna Reddy, Arianna Rampini, Derek Cheung, Kamal Rahimi Malekshan, Kanika Madan, Hooman Shayani (Autodesk AI Lab)
> - arXiv: [2411.08017](https://arxiv.org/abs/2411.08017) (2024년 11월 12일)
> - 코드: [GitHub - AutodeskAILab/WaLa](https://github.com/AutodeskAILab/WaLa)
> - 프로젝트 페이지: [autodeskailab.github.io/WaLaProject](https://autodeskailab.github.io/WaLaProject)

---

## 1. 핵심 주장 및 주요 기여 요약

### 🎯 핵심 주장

대규모 3D 생성 모델은 막대한 계산 자원을 필요로 하면서도 고해상도에서 세밀한 디테일과 복잡한 기하 구조를 충분히 포착하지 못하는 경우가 많다. 이 논문은 그 원인이 현재 표현 방식의 **비효율성**, 즉 생성 모델을 효과적으로 학습하기 위한 **압축성(compactness)의 부족**에 있다고 주장한다.

이를 해결하기 위해 웨이블릿(wavelet) 기반의 컴팩트 잠재 인코딩(compact latent encodings)으로 3D 형태를 인코딩하는 **Wavelet Latent Diffusion(WaLa)**을 제안한다. 구체적으로, $256^3$ 크기의 Signed Distance Field(SDF)를 $12^3 \times 4$ 잠재 그리드로 압축하여 **2,427배의 압축비**를 달성하며, 이 고수준 압축을 통해 추론 시간을 늘리지 않고도 대규모 생성 네트워크를 효율적으로 학습한다.

### 🏆 주요 기여 (5가지)

| 기여 | 내용 |
|------|------|
| **① 압축 표현** | $256^3$ SDF → $12^3 \times 4$ 잠재 그리드, 2,427× 압축 |
| **② 대규모 모델** | 약 10억 파라미터(1B) 조건부·비조건부 모델 |
| **③ 다중 모달 지원** | 텍스트, 스케치, 단일/다중 뷰 이미지, 포인트 클라우드, 저해상도 복셀, 깊이 맵 |
| **④ 빠른 추론** | 조건에 따라 2~4초 내 생성 |
| **⑤ 오픈소스** | 역대 최대 규모의 사전학습 3D 생성 모델 공개 |

조건부·비조건부 모델 모두 약 10억 파라미터를 포함하며 $256^3$ 해상도에서 고품질 3D 형태를 생성한다. WaLa는 조건에 따라 2~4초 내 형태를 생성하며, 여러 데이터셋에서 최신 성능(state-of-the-art)을 달성하고 생성 품질, 다양성, 계산 효율성에서 유의미한 개선을 보였다. 또한 코드와 가장 큰 사전학습 3D 생성 모델을 오픈소스로 공개하였다.

---

## 2. 해결하고자 하는 문제, 제안하는 방법, 모델 구조, 성능 및 한계

### 2.1 🔍 해결하고자 하는 문제

기존의 3D 생성 모델에서 입력 변수의 수를 줄이기 위해 더 컴팩트한 표현을 도입하려는 시도가 있었으나, 이러한 표현들은 불규칙하거나 이산적 특성을 지녀 신경망으로 모델링하기 어렵고, 이미지나 자연어 데이터에 비해 여전히 크기가 상당하여 모델 파라미터의 효율적 확장을 어렵게 한다.

기존의 웨이블릿 기반 표현(Neural Wavelet, UDiFF, wavelet-tree 등)은 웨이블릿 변환과 역변환을 활용하여 웨이블릿 공간과 고해상도 TSDF 표현 사이를 원활히 변환하며 데이터 압축과 대규모 3D 데이터셋의 효율적 처리를 가능하게 한다. 그러나 **대규모 생성 모델로 확장할 때 웨이블릿 기반 표현 자체도 여전히 상당히 크다는 한계**가 있다.

---

### 2.2 ⚙️ 제안하는 방법 (수식 포함)

WaLa는 **2단계(Two-Stage) 프레임워크**로 구성된다.

#### **Stage 1: VQ-VAE 기반 압축 (웨이블릿 → 잠재 그리드)**

WaLa 프레임워크는 오토인코더를 학습하여 웨이블릿 표현을 더욱 압축하고 정보 손실을 최소화한다. 3D 웨이블릿 표현을 합성곱 기반 VQ-VAE로 압축하여 $256^3$ TSDF를 $12^3 \times 4$ 그리드로 축소하며, GSO 데이터셋에서 **IoU 0.978**(97.8%)을 유지하면서 **2,427×** 압축을 달성한다.

**VQ-VAE 목적함수:**

$$\mathcal{L}_{VQ} = \mathcal{L}_{recon} + \|\text{sg}[z_e] - e\|_2^2 + \beta \|z_e - \text{sg}[e]\|_2^2$$

- $z_e$: 인코더 출력(연속 잠재 벡터)
- $e$: 코드북 임베딩 벡터
- $\text{sg}[\cdot]$: stop-gradient 연산
- $\beta$: commitment loss 가중치

**압축 과정:**

$$\text{TSDF}_{256^3} \xrightarrow{\text{Wavelet Transform}} W \xrightarrow{\text{VQ-VAE Encoder}} Z \in \mathbb{R}^{12^3 \times 4}$$

**압축비:**

$$\text{Compression Ratio} = \frac{256^3 \times 1}{12^3 \times 4} \approx 2427 \times$$

#### **Stage 2: 잠재 공간에서의 Diffusion 모델 학습**

두 번째 단계에서는 잠재 그리드 $Z_n$ 위에 확산(diffusion) 기반 생성 모델을 학습하며, 이 모델은 조건 벡터 시퀀스에 의해 조건화될 수 있다. 추론 시에는 완전히 노이즈가 섞인 잠재 벡터에서 출발하여, 역방향 확산 과정을 통해 점진적으로 디노이징(denoising)하며 **Classifier-Free Guidance(CFG)**를 활용한다.

**DDPM 기반 Forward Process:**

$$q(z_t | z_0) = \mathcal{N}(z_t; \sqrt{\bar{\alpha}_t} z_0, (1 - \bar{\alpha}_t)I)$$

**Reverse Process (디노이징):**

$$p_\theta(z_{t-1} | z_t, c) = \mathcal{N}(z_{t-1}; \mu_\theta(z_t, t, c), \Sigma_\theta(z_t, t, c))$$

**학습 목적함수 (단순화된 손실):**

$$\mathcal{L}_{diffusion} = \mathbb{E}_{z_0, \epsilon, t} \left[ \| \epsilon - \epsilon_\theta(z_t, t, c) \|^2 \right]$$

**Classifier-Free Guidance (추론 시):**

$$\tilde{\epsilon}_\theta(z_t, t, c) = (1 + w) \epsilon_\theta(z_t, t, c) - w \cdot \epsilon_\theta(z_t, t, \emptyset)$$

- $w$: guidance scale
- $c$: 조건(텍스트, 이미지, 포인트 클라우드 등)
- $\emptyset$: 무조건 입력

**최종 추론 파이프라인:**

$$z_T \sim \mathcal{N}(0, I) \xrightarrow{\text{Denoising}} z_0 \xrightarrow{\text{VQ-VAE Decoder}} W \xrightarrow{\text{Inv. Wavelet}} \text{TSDF} \xrightarrow{\text{Marching Cubes}} \text{Mesh}$$

---

### 2.3 🏗️ 모델 구조

WaLa 네트워크 아키텍처는 **2단계 학습 프로세스**로 구성된다. Stage 1은 오토인코더 학습으로, Wavelet Tree(W) 형태 표현을 컴팩트 잠재 공간으로 압축한다. Stage 2는 조건부/비조건부 확산(diffusion) 학습 단계이다.

학습 데이터는 ModelNet, ShapeNet 등을 포함하여 **19개의 공개 데이터셋에서 수집된 1,000만 개 이상의 3D 형태**로 구성된다.

각 생성 모델은 단일 H100 GPU에서 조건별로 학습되며, 포인트 클라우드를 포함한 6가지 조건에 대해 훈련된다. 합성 스케치 데이터와 단일 깊이 데이터를 사용한 파인튜닝을 통해 추가 조건을 얻는다. 또한 단일 뷰 RGB에 대해 **14억 파라미터(WaLa Large 모델)**를 8개의 H100 GPU로 학습한다.

**지원 입력 모달리티 요약:**

| 모달리티 | 모델명 | 파라미터 수 |
|----------|--------|------------|
| Single-view Image | WaLa-SV-1B | ~1B |
| Point Cloud | WaLa-PC-1B | ~1B |
| Single Depth Map | WaLa-DM1-1B | ~1B |
| Multi-view RGB | WaLa-RGB4-1B | ~1B |
| Multi-view Depth (6) | WaLa-MVDream-DM6 | - |
| Text-to-3D | WaLa-MVDream-RGB4 | - |
| Single-view (Large) | WaLa Large | ~1.4B |

---

### 2.4 📈 성능 향상

VQ-VAE를 통한 3D 웨이블릿 표현 압축으로 $256^3$ TSDF를 $12^3 \times 4$ 그리드로 줄이며, GSO 데이터셋에서 **IoU 97.8%**를 유지하면서 **2,427×** 압축을 달성한다. 그 결과 생성 모델은 로컬 디테일을 모델링할 필요 없이 전역 구조에 집중할 수 있게 된다. 이로써 10억 파라미터의 대규모 3D 생성 모델 학습이 가능해진다.

WaLa는 많은 귀납적 편향(inductive bias) 없이 다양한 입력 모달리티를 통한 제어된 생성을 지원하므로 프레임워크가 유연하고 적응 가능하다. 이를 통해 복잡한 기하학, 그럴듯한 구조, 복잡한 위상(topology), 매끄러운 표면을 갖는 3D 형태를 생성한다.

---

### 2.5 ⚠️ 한계점

논문 및 관련 자료에서 확인 가능한 한계는 다음과 같다:

1. **텍스처 미지원**: WaLa는 텍스트, 스케치, 저해상도 복셀, 포인트 클라우드, 단일 뷰 및 다중 뷰 이미지 등 다양한 입력으로부터 고품질 3D 메쉬를 생성할 수 있으나, 현재 공개된 구현체는 **SDF 기반의 기하(geometry) 생성에 집중**하며 텍스처(texture)나 재질(material) 정보는 직접 생성하지 않는다.

2. **학습 데이터 의존성**: 훈련 데이터는 19개 공개 데이터셋에서 1,000만 개 이상의 3D 형태를 포함하는데, 이처럼 **방대한 학습 데이터와 고성능 GPU**가 필수적으로 요구된다.

3. **표현 방식의 제약**: 웨이블릿 기반 표현은 이미지나 자연어 데이터에 비해 여전히 상대적으로 크기가 크고, 대규모 생성 모델로 확장 시에는 상당히 커질 수 있다는 한계가 남아 있다.

4. **실내외 및 씬(scene) 레벨 생성 미지원**: 모델은 단일 객체(object) 생성에 특화되어 있어, 복잡한 씬 전체를 생성하는 데는 적용이 어렵다.

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 🌐 다중 모달리티를 통한 일반화

WaLa는 압축으로 인한 입력 변수 수의 대폭 감소를 통해 10억 파라미터의 대규모 3D 생성 모델 학습을 가능케 한다. 또한 많은 귀납적 편향을 추가하지 않고 다양한 입력 모달리티를 통한 제어된 생성을 지원함으로써, 단일 뷰 3D 재건 작업 이상으로 유연하고 적응 가능한 프레임워크를 제공한다.

→ 이는 다양한 입력 환경에서도 일관되게 작동하는 **범용 3D 생성 모델**의 방향성을 제시한다.

### 3.2 📦 대규모 학습 데이터 활용

입력 변수의 수를 대폭 줄임으로써, 세밀하고 다양한 형태를 생성하는 **최대 10억 파라미터**의 대규모 3D 생성 모델 학습이 가능해진다.

학습 데이터는 19개 공개 데이터셋에서 **1,000만 개 이상의 3D 형태**를 포함하며, 이 광범위한 데이터 커버리지는 형태 다양성(shape diversity)에서의 일반화 성능을 높이는 핵심 요소이다.

### 3.3 🔄 분리된 압축-생성(Decoupled Compression-Generation)의 장점

잠재 공간으로의 압축과 생성을 **분리(decoupling)**하여, 대규모 생성 모델을 잠재 공간 내에서 효율적으로 확장할 수 있다.

이 접근 방식은 일반화 측면에서 매우 중요하다:
- VQ-VAE 인코더는 새로운 데이터셋에 독립적으로 적응 가능
- 생성 모델은 압축된 범용 잠재 공간에서 동작
- 새로운 도메인 적용 시 인코더 재학습 없이 생성 모델만 파인튜닝 가능

### 3.4 🎯 Zero-shot 가능성

조건부 모델은 포인트 클라우드, 이미지, 저해상도 복셀, 스케치, 텍스트에 이르는 다양한 입력 조건에 적응성을 보여준다. 더불어 텍스트-3D 및 스케치-3D 분야에서 zero-shot 방법들이 주목받고 있어, 보다 유연하고 일반화 가능한 3D 생성의 잠재성을 보여준다.

---

## 4. 미래 연구에 미치는 영향과 고려 사항

### 4.1 🚀 향후 연구에 미치는 영향

#### (1) 3D 생성 모델의 스케일링 법칙(Scaling Law) 정립
입력 변수 수의 대폭 감소는 10억 파라미터 수준의 3D 생성 모델 학습을 가능하게 하며, 세밀하고 다양한 형태 생성으로 이어진다. WaLa는 LLM에서 확인된 스케일링 법칙이 3D 생성 도메인에도 적용될 수 있음을 실증하여, **대규모 3D 파운데이션 모델(Foundation Model)** 연구를 촉진할 것이다.

#### (2) 컴팩트 3D 표현 연구 가속화
WaLa는 컴팩트 표현과 고품질 생성 사이의 간극을 연결함으로써 이 분야의 연구 지평을 넓힌다. 웨이블릿 도메인에서의 압축 + 잠재 확산 모델의 조합은 향후 다양한 3D 표현(NeRF, 3DGS 등)으로 확장될 가능성이 있다.

#### (3) 멀티모달 3D 생성 표준화
WaLa는 스케치, 텍스트, 단일 뷰 이미지, 저해상도 복셀, 포인트 클라우드, 깊이 맵 등 다양한 조건으로부터 형태를 생성하는 새로운 3D 생성 모델을 제안한다. 단일 프레임워크 내에서 다중 모달리티를 지원하는 이 접근법은 멀티모달 3D 이해 및 생성 연구의 표준 벤치마크로 자리잡을 수 있다.

#### (4) 오픈소스 생태계 기여
코드와 현존하는 최대 규모의 사전학습 3D 생성 모델이 다양한 모달리티에 걸쳐 오픈소스로 공개됨으로써, 3D 생성 AI 연구의 진입 장벽을 낮추고 재현 가능한 연구 환경을 마련한다.

---

### 4.2 🔬 향후 연구 시 고려할 점

#### ① 텍스처 및 재질 생성 통합
현재 WaLa는 기하(geometry) 생성에 특화되어 있다. GET3D가 복잡한 위상과 고품질 텍스처를 가진 명시적 텍스처 3D 메쉬를 직접 생성한다는 점과 비교할 때, WaLa 프레임워크에 **텍스처/재질 생성 모듈을 통합**하는 것이 중요한 과제이다.

#### ② 씬 레벨(Scene-Level) 생성으로의 확장
현재 WaLa는 단일 객체 생성에 집중되어 있으나, 실제 응용에서는 씬 전체 생성이 필요하다. 웨이블릿 압축 프레임워크를 씬 표현으로 확장하는 연구가 필요하다.

#### ③ 더 효율적인 표현 탐색
3DShape2VecSet은 벡터 집합(set of vectors) 위에 뉴럴 필드를 인코딩하며, 방사 기저 함수(RBF) 표현과 교차-어텐션 및 자기-어텐션 함수를 활용하여 트랜스포머 처리에 특히 적합한 학습 가능 표현을 설계한다. WaLa의 격자 기반 잠재 표현과 벡터 집합 기반 표현의 장점을 결합하는 방향도 탐색할 만하다.

#### ④ 실시간 생성 및 경량화
WaLa Large 모델(1.4B)은 8개의 H100 GPU를 사용하므로, 모바일·엣지 디바이스를 위한 **지식 증류(Knowledge Distillation)** 또는 **모델 경량화** 연구가 병행되어야 한다.

#### ⑤ 동적 형태(Dynamic Shape) 및 애니메이션 생성
현재 WaLa는 정적(static) 형태 생성에 집중되어 있다. 시간 축을 포함한 4D 생성(동적 형태 및 애니메이션)으로의 확장은 게임, 영화 산업에서 큰 응용 가치를 지닌다.

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

| 모델 | 연도 | 표현 방식 | 입력 조건 | 핵심 특징 | 한계 |
|------|------|-----------|-----------|-----------|------|
| **GET3D** (NVIDIA) | 2022 | DMTet 기반 명시적 메쉬 | 2D 이미지 | 텍스처 포함 직접 메쉬 생성 | 학습 데이터셋 범위 제한적 |
| **3DShape2VecSet** | 2023 | 벡터 집합 기반 뉴럴 필드 | 포인트 클라우드, 이미지, 텍스트 | 트랜스포머 친화적 표현 | 대규모 파라미터 확장 미검증 |
| **LION** | 2022 | 계층적 잠재 포인트 | 비조건부 | 계층적 VAE + 확산 | 해상도 및 다양성 제한 |
| **Shape·E** | 2023 | 암묵적 함수 파라미터 | 텍스트, 이미지 | 빠른 생성 | 세밀한 기하 표현 한계 |
| **WaLa (본 논문)** | 2024 | 웨이블릿 압축 잠재 그리드 | 텍스트/이미지/포인트 클라우드/스케치/복셀/깊이 | **2427× 압축, 1B 파라미터, 2~4초 추론** | 텍스처 미지원, 씬 레벨 미지원 |

오토인코더를 통한 잠재 공간 처리는 포인트 클라우드 및 암묵적 형태와 같은 복잡한 표현의 생성을 가능케 했으며, 포인트 클라우드, 복셀, 점유 함수, 뉴럴 웨이블릿 계수에 대한 직접 확산 학습도 발전을 보였다. WaLa는 컴팩트 표현과 고품질 생성 사이의 간극을 연결하여 이 연구 분야를 새로운 수준으로 끌어올린다.

최신 3D 생성 연구들은 컴팩트 VAE 잠재 공간을 학습한 후 확산 또는 플로우 기반 트랜스포머로 모델링하는 2단계 설계를 따르며, Dora, TripoSG, Hunyuan3D 2.0 같은 대표적 오픈 모델들이 이 패러다임을 따르면서 이미지-to-3D 성능에서 강력한 결과를 보이고 있다.

---

## 📚 참고 자료 및 출처

1. **본 논문 (arXiv)**
   - Sanghi, A. et al. (2024). *Wavelet Latent Diffusion (WaLa): Billion-Parameter 3D Generative Model with Compact Wavelet Encodings*. arXiv:2411.08017. [https://arxiv.org/abs/2411.08017](https://arxiv.org/abs/2411.08017)

2. **Autodesk AI Lab 공식 프로젝트 페이지**
   - [https://autodeskailab.github.io/WaLaProject/](https://autodeskailab.github.io/WaLaProject/)

3. **공식 GitHub 코드**
   - [https://github.com/AutodeskAILab/WaLa](https://github.com/AutodeskAILab/WaLa)

4. **Autodesk Research 공식 출판 페이지**
   - [https://www.research.autodesk.com/publications/wala-billion-parameter-3d-generative-model-compact-wavelet-encodings/](https://www.research.autodesk.com/publications/wala-billion-parameter-3d-generative-model-compact-wavelet-encodings/)

5. **OpenReview (동료 심사)**
   - [https://openreview.net/forum?id=D48jvLN45W](https://openreview.net/forum?id=D48jvLN45W)

6. **Semantic Scholar**
   - [https://www.semanticscholar.org/paper/Wavelet-Latent-Diffusion-(Wala):-Billion-Parameter-Sanghi-Khani/89d689c1b55d35a47ccad0889c6f6f1ad190b770](https://www.semanticscholar.org/paper/Wavelet-Latent-Diffusion-(Wala):-Billion-Parameter-Sanghi-Khani/89d689c1b55d35a47ccad0889c6f6f1ad190b770)

7. **HuggingFace 논문 페이지**
   - [https://huggingface.co/papers/2411.08017](https://huggingface.co/papers/2411.08017)

8. **비교 연구 - 3DShape2VecSet**
   - Zhang, B. et al. (2023). *3DShape2VecSet: A 3D Shape Representation for Neural Fields and Generative Diffusion Models*. ACM TOG. arXiv:2301.11445

9. **비교 연구 - GET3D**
   - Gao, J. et al. (2022). *GET3D: A Generative Model of High Quality 3D Textured Shapes Learned from Images*. NeurIPS 2022. arXiv:2209.11163
