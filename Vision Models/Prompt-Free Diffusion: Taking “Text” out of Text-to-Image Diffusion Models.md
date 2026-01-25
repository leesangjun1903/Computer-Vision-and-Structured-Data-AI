
# Prompt-Free Diffusion: Taking “Text” out of Text-to-Image Diffusion Models

## I. 핵심 주장 및 주요 기여

"Prompt-Free Diffusion: Taking 'Text' out of Text-to-Image Diffusion Models"는 텍스트-이미지 확산(T2I) 모델의 근본적 문제점을 해결하는 혁신적 접근법을 제시한다. 기존 T2I 시스템은 프롬프트 엔지니어링이라는 비효율적 과정에 의존해 왔다. 이 논문의 핵심 주장은 **"사람이 자연언어로 시각적 세부사항을 완전히 전달할 수 없다"**는 근본적 한계에 기반한다. 논문의 저자들은 다음 세 가지 주요 기여를 제시한다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/bb37e6cf-da95-489b-a711-0732307b5aa5/2305.16223v2.pdf)

1. **텍스트 제거 기반 접근법**: CLIP 텍스트 인코더를 대체하여 참조 이미지만으로 고품질 생성 달성
2. **SeeCoder의 재사용성**: 기존 T2I 모델에 추가 학습 없이 적용 가능한 범용 시각 인코더
3. **실무 확장성**: 애니메이션 생성, 가상 피팅 등 다양한 다운스트림 애플리케이션 지원

***

## II. 문제 정의 및 기존 접근의 한계

### A. 프롬프트 기반 T2I의 근본적 문제

기존 T2I 모델들의 주요 문제점은 다음과 같이 분석된다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/bb37e6cf-da95-489b-a711-0732307b5aa5/2305.16223v2.pdf)

| 접근법 | 개인화 품질 | 설치 용이성 | 입력 유연성 |
|--------|-----------|----------|---------|
| **모델 파인튜닝** (DreamBooth) | 완전 개인화 | 어려움 (GPU/데이터 필요) | 프롬프트 품질에 종속 |
| **프롬프트 엔지니어링** | 제한적 | 용이 | 프롬프트 엔지니어링 필수 |
| **ControlNet/T2I-Adapter** | 구조만 제어 | 용이 | 여전히 프롬프트 필요 |
| **Image-Variation** | 의미만 제어 | 도메인별 모델 필요 | 이미지만 사용 가능 |
| **Prompt-Free Diffusion** | 거의 완전 제어 | SeeCoder 교체로 용이 | 이미지만 사용, 프롬프트 불필요 |

### B. CLIP 텍스트 인코더의 한계

논문에서 지적한 CLIP 기반 접근의 제약은 다음과 같다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/bb37e6cf-da95-489b-a711-0732307b5aa5/2305.16223v2.pdf)

1. **저해상도 한계**: ViT 입력이 384×384로 제한
2. **세부 정보 손실**: 텍스처, 객체 디테일 포착 불충분
3. **간접적 학습**: 대조학습 손실함수가 시각적 신호를 직접 처리하지 못함

이러한 문제는 CLIP의 기본 설계에 내재되어 있으며, 최근 연구들이 이를 확인하고 있다. CLIP의 특성 상관관계 문제를 다루는 Unmix-CLIP이나 세밀한 작업에서의 한계를 지적하는 "CLIP See Both Forest and Trees" 연구 등이 이를 입증한다. [arxiv](https://arxiv.org/abs/2502.02977v1)

***

## III. 제안 방법: Prompt-Free Diffusion의 기술적 체계

### A. 전체 아키텍처

Prompt-Free Diffusion은 **세 개의 핵심 모듈**로 구성된다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/bb37e6cf-da95-489b-a711-0732307b5aa5/2305.16223v2.pdf)

```
참조 이미지 ─┐
            ├─→ SeeCoder ─→ 시각 임베딩 (N×C)
선택적 조건 ─┤                      ↓
             └──────────────→ 확산 모델의 Cross-Attention
```

### B. Semantic Context Encoder (SeeCoder)의 상세 구조

#### 1) 백본 인코더 (Backbone Encoder)

**아키텍처**: SWIN-L (Shifted Window Transformer)

SWIN-L을 선택한 이유는: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/bb37e6cf-da95-489b-a711-0732307b5aa5/2305.16223v2.pdf)

$$\text{특성 피라미드} = \{\text{Res3}, \text{Res4}, \text{Res5}\}$$

- 임의 해상도 이미지를 다층 특성 피라미드로 변환
- 이미지 분할(patching)을 통한 효율적 처리
- 다양한 스케일의 시각적 신호 포착

#### 2) 디코더 (Transformer-based Decoder)

디코더는 **다층 특성 통합 메커니즘**을 포함한다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/bb37e6cf-da95-489b-a711-0732307b5aa5/2305.16223v2.pdf)

**구성 요소**:
- 채널 균등화 컨볼루션
- 모든 레벨의 특성 평탄화 및 연결
- 6개의 Multi-Head Self-Attention 모듈

**수식적 표현**:

$$\text{Decoder}(\{F_3, F_4, F_5\}) = \sum_{i=3}^{5} \text{MHA}(\text{Concat}(F_i)) + F_i$$

여기서 $F_i$는 각 해상도 레벨의 특성이며, MHA는 Multi-Head Attention이다.

#### 3) Query Transformer

**구조적 특징**: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/bb37e6cf-da95-489b-a711-0732307b5aa5/2305.16223v2.pdf)

- **글로벌 쿼리**: 4개 (전체 이미지 의미 포착)
- **로컬 쿼리**: 144개 (세부 영역 정보)

**계산 프로세스**:

$$\text{로컬} = \text{CrossAttn}(Q_{\text{local}}, K_{\text{visual}}, V_{\text{visual}})$$

$$\text{글로벌} = \text{SelfAttn}([Q_{\text{global}}, Q_{\text{local}}])$$

$$\text{최종 임베딩} = \text{Concat}([Q_{\text{global}}, Q_{\text{local}}])$$

**SeeCoder-PA (Position-Aware) 변형**:

$$\text{위치 임베딩} = \sin\cos(\text{위치}) + \text{MLP}(\text{위치})$$

구조적 조건화가 없을 때 성능 향상을 위해 2D 공간 임베딩 추가.

### C. 확산 프로세스와 손실함수

#### 확산 과정의 수학적 정의

**정방향 확산 (Forward Process)**: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/bb37e6cf-da95-489b-a711-0732307b5aa5/2305.16223v2.pdf)

$$q(x_T|x_0) = \prod_{t=1}^{T} q(x_t|x_{t-1})$$

$$q(x_t|x_{t-1}) = \mathcal{N}(\sqrt{1-\beta_t}x_{t-1}, \beta_t\mathbf{I})$$

여기서:
- $\beta_t$: 각 타임스텝의 노이즈 스케일
- $T = 1000$: 확산 스텝 수
- $\beta_t$ 스케줄: $8.5 \times 10^{-5}$ → $1.2 \times 10^{-2}$ (선형 증가)

**역방향 확산 (Reverse Process)**: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/bb37e6cf-da95-489b-a711-0732307b5aa5/2305.16223v2.pdf)

$$p_\theta(x_{t-1}|x_t) = \mathcal{N}(\mu_\theta(x_t, t), \sigma_\theta(x_t, t))$$

#### 변분 하한 손실함수

$$\mathcal{L}_{\text{VLB}} = \mathbb{E}_{q(x_0)} \left[-\log p_\theta(x_0) + \mathbb{E}_{q(x_1^T|x_0)} \left[-\log p_\theta(x_0^T) \right] \right]$$

이를 재정렬하면:

$$\mathcal{L} = \sum_{t=1}^{T} \mathbb{E}_{q(x_t|x_0)} \left[\left\|\epsilon - \epsilon_\theta(x_t, t, c_{\text{SeeCoder}})\right\|_2^2 \right]$$

여기서 $c_{\text{SeeCoder}} = \text{SeeCoder}(I_{\text{ref}})$는 참조 이미지로부터 추출한 시각 조건이다.

### D. 학습 전략: Frozen Backbone 방식

**혁신적인 학습 효율성**: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/bb37e6cf-da95-489b-a711-0732307b5aa5/2305.16223v2.pdf)

| 구성요소 | 상태 | 이유 |
|---------|------|------|
| VAE (SD2.0) | Frozen | 이미지 인코딩/디코딩 보존 |
| Diffuser (T2I) | Frozen | 생성 능력 유지 |
| SWIN-L 백본 | Frozen | 사전학습된 특성 추출 유지 |
| Decoder | **학습 가능** | 시각 특성 통합 학습 |
| Query Transformer | **학습 가능** | 최종 임베딩 생성 학습 |

**학습 설정**: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/bb37e6cf-da95-489b-a711-0732307b5aa5/2305.16223v2.pdf)

- **총 반복**: 100,000번
  - 처음 50,000: 학습률 $10^{-4}$
  - 다음 50,000: 학습률 $10^{-5}$
- **배치 크기**: 512 (GPU당 8개)
- **하드웨어**: 16개 A100 GPU (2개 노드)
- **Gradient Accumulation**: 4

**SeeCoder-PA 파인튜닝**:

$$\text{SeeCoder-PA}(\text{checkpoint}_{50k}) \leftarrow \text{20k 추가 스텝}$$

학습률: $5 \times 10^{-5}$

***

## IV. 성능 평가 및 실험 결과

### A. T2I 모델과의 비교

논문에서 수행한 핵심 실험은 ControlNet + Stable Diffusion과의 직접 비교였다. 프롬프트 복잡도의 세 가지 단계에서 평가했을 때: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/bb37e6cf-da95-489b-a711-0732307b5aa5/2305.16223v2.pdf)

**프롬프트 진화 단계**:

1. 기본 프롬프트: "medieval streets, yellow and blue, rain and fog"
2. 의미 + 스타일: 위에 "dystopian", "wide shot", "fantasy" 추가
3. 엔지니어링: 추가로 "realistic", "8k" 등 품질 프롬프트 포함

**결과**: Prompt-Free Diffusion은 2-3 단계 수준의 성능 달성. 특히 색상 시프트 현상이 없어 더 안정적.

### B. Image-Variation 성능 분석

**핵심 발견**: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/bb37e6cf-da95-489b-a711-0732307b5aa5/2305.16223v2.pdf)

| 모델 | In-Domain | Out-of-Domain | 특징 |
|-----|-----------|---------------|------|
| Versatile Diffusion | **최고** | 최고 | 파인튜닝으로 최고 성능 |
| ControlNet Shuffle | 중간 | **최저** | Out-of-domain 약함 |
| ControlNet Reference-Only | **최고** | 중간 | 일관성 불안정 |
| **Prompt-Free Diffusion** | **높음** | **높음** | 균형잡힌 성능 |

특히 텍스처, 색상, 스타일, 배경 보존에서 우수.

### C. SeeCoder의 범용성 검증

**재학습 없는 도메인 적응**: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/bb37e6cf-da95-489b-a711-0732307b5aa5/2305.16223v2.pdf)

논문은 6개의 서로 다른 T2I 모델에서 SeeCoder의 재사용성을 검증했다:

1. **SD1.5** (기본 모델)
2. **OpenJourney-V4** (예술 중심)
3. **Deliberate-V2** (예술 중심)
4. **OAM-V2** (애니메 모델)
5. **Anything-V4** (애니메 모델)
6. **RealisticVision-V2** (사진현실적)

**결과**: 모든 모델에서 **재학습 없이** SeeCoder-PA가 높은 수준의 성능 달성. 이는 범용 컴포넌트의 성공을 입증한다.

***

## V. 일반화 성능 분석 및 한계

### A. 일반화 성능 향상의 핵심 메커니즘

#### 1) Out-of-Domain 강인성

Prompt-Free Diffusion의 가장 주목할 만한 특징은 **in-domain과 out-of-domain 모두에서 일관된 성능**이다. 이는 다음 요인들에 의해 가능하다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/bb37e6cf-da95-489b-a711-0732307b5aa5/2305.16223v2.pdf)

**시각적 신호의 직접 처리**:

$$\text{CLIP} \xrightarrow{\text{대조학습}} \text{제한된 해상도}, \text{간접적 학습}$$

$$\text{SeeCoder} \xrightarrow{\text{직접 처리}} \text{임의 해상도}, \text{계층적 학습}$$

#### 2) 다중 스케일 특성 추출

SWIN-L 백본의 피라미드 구조가 저수준(텍스처)부터 고수준(의미)까지 모두 포착:

$$\text{저수준 정보} \subset \text{Res3} \cup \text{Res4} \cup \text{Res5} \subset \text{고수준 정보}$$

#### 3) 효율적인 파라미터 활용

학습 가능한 파라미터가 Decoder와 Query Transformer에만 집중:

$$\frac{\text{학습 가능 파라미터}}{\text{전체 파라미터}} = \frac{D + Q}{D + Q + B + V} \approx 5-10\%$$

이는 과적합을 감소시켜 일반화 성능을 향상시킨다.

### B. 일반화 성능의 한계

#### 1) Out-of-Domain 성능 갭

논문의 자체 실험 결과에서도 out-of-domain 참조 이미지에서 약간의 성능 저하가 나타났다. 이는 다음 원인에 기인한다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/bb37e6cf-da95-489b-a711-0732307b5aa5/2305.16223v2.pdf)

**사전학습과 분포 불일치**:

$$\mathcal{D}_{\text{학습}} \subset \text{웹 이미지} \quad \text{vs} \quad \mathcal{D}_{\text{테스트}} = \text{임의 이미지}$$

특히 매우 특이한 스타일이나 도메인의 참조 이미지는 SeeCoder가 본 적 없는 시각 패턴을 포함할 수 있다.

#### 2) 도메인별 파인튜닝 필요성

애니메이션 피규어 생성 같은 특화 작업에서 추가 파인튜닝이 필요했다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/bb37e6cf-da95-489b-a711-0732307b5aa5/2305.16223v2.pdf)

$$\text{일반 SeeCoder} + 30k \text{ 스텝 파인튜닝} = \text{도메인별 최적화}$$

#### 3) 참조 이미지 의존성

프롬프트 엔지니어링 문제를 이미지 선택 문제로 치환:

$$\text{프롬프트} \rightarrow \text{참조 이미지}$$

고품질 참조 이미지 선택이 여전히 사용자의 책임이다.

### C. SeeCoder-PA의 역할

구조적 조건화가 없을 때 성능 향상을 위해 도입된 SeeCoder-PA:

$$\text{성능 향상} = f(\text{공간 임베딩}) + f(\text{구조 조건}) + f(\text{정보 보존})$$

이는 공간 정보가 구조적 가이드를 부분적으로 대체할 수 있음을 의미한다.

***

## VI. 최신 관련 연구 비교 (2020년 이후)

### A. ControlNet 계열의 발전

#### ControlNet (2023) [arxiv](https://arxiv.org/abs/2302.05543)

**핵심 혁신**: Zero Convolution을 통한 안정적 학습

$$\text{ControlNet 출력} = F(x; \Theta) + Z(F(x + Z(c; \Theta_{z1}); \Theta_c); \Theta_{z2})$$

- **장점**: 구조적 제어의 정밀성
- **한계**: 여전히 프롬프트 필요, out-of-domain 약함

#### ControlNet-XS (2023) [arxiv](https://arxiv.org/html/2312.06573v1/)

**개선사항**: 피드백 제어 시스템 관점에서 정보 흐름 최적화

$$\text{피드백 신호 대역폭} \uparrow \Rightarrow \text{제어 지연} \downarrow$$

- 353M → 53-117M 파라미터 감소
- 2배 속도 향상
- 유사하거나 더 나은 성능

#### DC-ControlNet (2025) [arxiv](http://arxiv.org/pdf/2502.14779.pdf)

**아이디어**: 다중 조건 제어의 분리

$$\text{DC-ControlNet} = \text{Local Control} + \text{Global Control}$$

- 유연한 다중 조건 제어
- 요소 간 상호작용 제어

### B. 이미지 기반 조건화의 진화

#### Versatile Diffusion (2022) [factory.skku](https://factory.skku.edu/factory/sanhakproject.do?mode=download&articleNo=202542&attachNo=169964)

**특징**: 멀티모달 생성 지원

- 텍스트, 이미지, 변형 모드 통합
- Prompt-Free Diffusion의 직접적 선행 연구
- 다만 도메인별 모델 필요

#### CLIP 기반 접근의 한계와 개선

**CLIP의 문제점**: [arxiv](http://www.arxiv.org/abs/2503.08723)

1. **특성 상관관계**: 의미 있는 차별화 불가능
2. **공간 편향**: 전역 이미지 패턴에만 집중
3. **복합 상호작용**: 속성 결합, 공간 관계 처리 약함

**개선 시도**:

| 방법 | 개선 영역 | 효과 |
|-----|---------|------|
| **Unmix-CLIP** | 특성 분리 | 24.9% 유사도 감소 |
| **Dense Cosine Similarity Maps** | 공간 보존 | 토폴로지 유지 |
| **D&D (Decomposition)** | 세밀한 인식 | 다중 크롭 활용 |

### C. 텍스트-이미지 정렬 개선

#### CoMat (2024) [proceedings.neurips](https://proceedings.neurips.cc/paper_files/paper/2024/file/8b54ecd9823fff6d37e61ece8f87e534-Paper-Conference.pdf)

**문제**: T2I 모델의 텍스트-이미지 미정렬

**해결책**: 개념 일치 및 속성 집중 모듈

- 개념 무시 문제 해결
- 속성 매핑 오류 수정
- SDXL 개선: +1.6점 (TIFA), +7.4점 (SD1.5)

#### SVG-T2I (2024) [arxiv](https://arxiv.org/html/2512.11749v1)

**접근**: 시각 기반 인코딩 파라다임

$$\text{VAE 잠재공간} \rightarrow \text{시각 기반 모듈(VFM) 표현}$$

- 의미론적 구조 유지
- 고해상도 (1024×1024) 생성
- GenEval: 0.75, DPG-Bench: 85.78

### D. Prompt-Free Diffusion의 위치

**경쟁 기술과의 비교**:

| 기술 | 구조 제어 | 의미 제어 | 프롬프트 불필요 | 도메인 적응 | 계산 효율 |
|-----|---------|---------|--------------|---------|---------|
| **ControlNet** | ⭐⭐⭐ | ⭐ | ✗ | ⭐⭐ | ⭐ |
| **Versatile Diffusion** | ⭐ | ⭐⭐ | ⭐⭐ | ✗ | ⭐ |
| **Prompt-Free Diffusion** | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |

***

## VII. 미래 연구 방향 및 시사점

### A. 기술적 개선 영역

#### 1) 더 강력한 시각 인코더

현재 SWIN-L 기반 SeeCoder 성능을 초과할 수 있는 방향:

$$\text{Vision Transformer (ViT)} \text{ or } \text{Mamba} \text{ 기반 백본}$$

고해상도 입력, 더 긴 시퀀스 처리로 세부 정보 보존 향상.

#### 2) 적응형 일반화

Out-of-domain 성능 향상을 위한 메타러닝:

$$\min_\theta \sum_{D \in \{\text{train}, \text{test}\}} \mathcal{L}(D; \theta)$$

#### 3) 구조-의미 상호작용

ControlNet과 SeeCoder의 통합:

$$\text{최종 생성} = f(\text{SeeCoder}(I_{\text{ref}}), \text{ControlNet}(c_{\text{structure}}))$$

### B. 연구 배포 및 실무 고려사항

#### 1) 참조 이미지 큐레이션 도구

사용자가 좋은 참조 이미지를 선택하도록 돕는 도구 필요:

- 자동 스타일 추천
- 품질 평가 메트릭
- 검색 기능

#### 2) 도메인 특화 모델 라이브러리

미리 파인튠 된 SeeCoder:

$$\{\text{SeeCoder}_{\text{anime}}, \text{SeeCoder}_{\text{photorealistic}}, \ldots\}$$

#### 3) 윤리 및 사회적 영향

논문에서 명시한 우려사항: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/bb37e6cf-da95-489b-a711-0732307b5aa5/2305.16223v2.pdf)

- 오픈소스 T2I 모델의 잠재적 편향
- 생성 결과의 책임 있는 사용
- 조작 탐지 기술 개발

***

## VIII. 결론: 학술적 의의와 실무적 가치

### A. 학술적 기여

1. **패러다임 전환**: 텍스트 기반에서 시각 기반으로의 조건화 재정의
2. **효율적 학습**: Frozen backbone을 통한 학습 효율성 극대화
3. **일반화 강인성**: In/out-of-domain 균형잡힌 성능

### B. 실무적 가치

**즉시 응용 분야**:

1. **콘텐츠 제작**: 프롬프트 엔지니어링 비용 감소
2. **도메인 적응**: 새로운 스타일 적응의 용이성
3. **사용자 접근성**: 기술적 배경 없는 사용자도 사용 가능

### C. 한계와 개방된 질문

1. **참조 이미지 선택 문제**: 여전히 사용자의 책임
2. **Out-of-domain 성능 갭**: 완전히 해결되지 않음
3. **도메인 특화 필요성**: 일부 작업은 추가 파인튜닝 필요

***

## 참고 문헌

 Xu, X., Guo, J., Wang, Z., Huang, G., Essa, I., & Shi, H. (2023). Prompt-Free Diffusion: Taking "Text" out of Text-to-Image Diffusion Models. arXiv:2305.16223v2. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/bb37e6cf-da95-489b-a711-0732307b5aa5/2305.16223v2.pdf)

 Chen, H., et al. (2024). SVG-T2I: Scaling up text-to-image latent diffusion model with visual-semantic tokens. [arxiv](https://arxiv.org/html/2512.11749v1)

 Jiang, D., et al. (2024). CoMat: Aligning Text-to-Image Diffusion Model with Image-to-Text Concept Matching. NeurIPS 2024. [proceedings.neurips](https://proceedings.neurips.cc/paper_files/paper/2024/file/8b54ecd9823fff6d37e61ece8f87e534-Paper-Conference.pdf)

 Zhang, L., & Agrawala, M. (2023). Adding Conditional Control to Text-to-Image Diffusion Models. arXiv:2302.05543. [arxiv](https://arxiv.org/abs/2302.05543)

 Liu, Y., et al. (2025). DC-ControlNet: Decoupling Inter- and Intra-Element Conditions in Image Generation. [arxiv](http://arxiv.org/pdf/2502.14779.pdf)

 논문 리뷰: ControlNet 기본 개념, 2023. [kimjy99.github](https://kimjy99.github.io/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0/controlnet/)

 Xu, X., Wang, Z., Zhang, E., Wang, K., & Shi, H. (2022). Versatile Diffusion: Text, Images and Variations All in One Diffusion Model. arXiv:2211.08332. [factory.skku](https://factory.skku.edu/factory/sanhakproject.do?mode=download&articleNo=202542&attachNo=169964)

 ControlNet 구글 코랩 실습, 2025. [blog.cslee.co](https://blog.cslee.co.kr/controlnet-spatial-control-text-to-image-diffusion/)

 Is CLIP ideal? No. Can we fix it? Yes! arXiv:2503.08723. [arxiv](http://www.arxiv.org/abs/2503.08723)

 Zavadski, D., & Feiden, J. (2023). ControlNet-XS: Designing an Efficient and Effective Architecture. arXiv:2312.06573. [arxiv](https://arxiv.org/html/2312.06573v1/)

 Rawelekar, S., et al. (2025). Disentangling CLIP Features for Enhanced Localized Understanding. arXiv:2502.02977. [arxiv](https://arxiv.org/abs/2502.02977v1)

 ControlNet-XS: Rethinking the Control as Feedback-Control Systems. arXiv:2312.06573. [arxiv](https://www.arxiv.org/abs/2312.06573)

 Xue, L., et al. (2025). Helping CLIP See Both the Forest and the Trees. arXiv:2507.03458. [arxiv](https://www.arxiv.org/abs/2507.03458)
