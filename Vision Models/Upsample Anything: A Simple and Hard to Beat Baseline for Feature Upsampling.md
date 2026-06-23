

# Upsample Anything: A Simple and Hard to Beat Baseline for Feature Upsampling

> **논문 정보**
> - **제목**: Upsample Anything: A Simple and Hard to Beat Baseline for Feature Upsampling
> - **저자**: Minseok Seo 외 2인
> - **arXiv ID**: 2511.16301 (2025년 11월)
> - **프로젝트 페이지**: https://seominseok0429.github.io/Upsample-Anything/

---

## 1. 핵심 주장 및 주요 기여 요약

**Upsample Anything**은 학습(training) 없이 저해상도 피처(feature)를 고해상도 픽셀 단위 출력으로 복원하는 **경량 Test-Time Optimization(TTO) 프레임워크**입니다.

### 핵심 주장

| 항목 | 내용 |
|------|------|
| 문제 인식 | VFM의 표현은 14×/16× 다운샘플링되어 픽셀 수준 태스크에 직접 사용 불가 |
| 기존 한계 | 데이터셋별 재학습 또는 heavy implicit 최적화 필요 |
| 제안 | 이미지별 경량 TTO로 anisotropic Gaussian 커널 학습 |
| 결과 | 추가 학습 없이 SOTA 달성 |

### 주요 기여

1. 어떤 사전학습 VFM에서도 데이터셋 수준 재학습 없이 저해상도 피처를 업샘플링하는 TTO 프레임워크를 도입하고, 피처·깊이·세그멘테이션·3D 데이터 등 다양한 모달리티에 강건하게 일반화합니다.

2. 핵심 방법론은 Joint Bilateral Upsampling(JBU)에서 영감을 받아 이를 2D Gaussian Splatting(2DGS)의 연속적 프레임워크로 재해석하며, 이를 통해 고정된 등방성(isotropic) 커널 대신 **적응적 비등방성(anisotropic) Gaussian 커널**을 학습할 수 있습니다.

3. 224×224 이미지 기준 $\approx 0.419\text{s}$의 처리 속도로 시맨틱 세그멘테이션, 깊이 추정, 깊이 맵 및 확률 맵 업샘플링에서 SOTA 성능을 달성합니다.

---

## 2. 상세 설명

### 2-1. 해결하고자 하는 문제

DINO, CLIP 및 파생 모델과 같은 Vision Foundation Models(VFMs)은 인코더 설계를 혁신했지만, 이 모델들은 구조적으로 원본 입력보다 14× 또는 16× 작은 고도로 다운샘플링된 피처 맵을 생성하여 공간적으로 정밀한 픽셀 수준 예측을 복원하는 데 어려움을 줍니다.

기존의 표준 접근법들은 무거운 분리형 디코더 아키텍처나 재학습된 업샘플러에 의존하며, 이는 계산 집약적이고 메모리 비효율적이며 도메인에 특화되어 있습니다.

피처 업샘플링 방법은 업샘플러가 최적화되는 방식에 따라 크게 두 가지 패러다임으로 분류됩니다: **(a) 데이터셋 수준 학습**과 **(b) Test-Time Optimization(TTO)**.

### 2-2. 제안하는 방법 (수식 포함)

#### ① 전체 파이프라인

입력 이미지가 주어지면, Upsample Anything은 RGB 가이던스를 저해상도(LR) 피처맵 크기에 맞게 리사이즈하고, 최적화를 통해 고해상도(HR) 색상 이미지를 재구성하며, 픽셀별 anisotropic Gaussian 파라미터인 $(\sigma_x, \sigma_y, \theta, \sigma_r)$을 학습합니다. 이 파라미터들은 연속적인 spatial–range splatting 커널을 정의하며, 최적화된 커널은 파운데이션 인코더의 LR 피처 맵에 적용되어 원본 이미지 그리드에 정렬된 HR 피처 맵을 생성합니다.

#### ② GSJBU 커널 (Gaussian Splatting Joint Bilateral Upsampling)

본 논문의 핵심은 **GSJBU(Gaussian Splatting Joint Bilateral Upsampling)**로, JBU를 연속적 Gaussian Splatting 관점으로 일반화한 것입니다.

**Joint Bilateral Upsampling(JBU)의 기본 형태:**

$$\hat{F}(p) = \frac{1}{Z(p)} \sum_{q \in \Omega} F(q) \cdot k_s(p, q) \cdot k_r(I(p), I(q))$$

여기서:
- $\hat{F}(p)$: 고해상도 위치 $p$에서의 업샘플링된 피처
- $F(q)$: 저해상도 피처 맵의 값
- $k_s(p, q)$: 공간적(spatial) Gaussian 커널
- $k_r(I(p), I(q))$: 색상(range) Gaussian 커널
- $Z(p)$: 정규화 상수

**GSJBU의 anisotropic 확장:**

등방성 한계(isotropic limit)에서 JBU로 수렴하는 비등방성 일반화(anisotropic generalization)를 통해 각 중심(center)별 공분산(covariance) 학습이 가능한 **GSJBU**를 구현합니다.

각 픽셀 $p$에 대해 학습 가능한 파라미터로 이루어진 비등방성 Gaussian 커널:

$$k(p, q; \Sigma_p) = \exp\!\left(-\frac{1}{2}(p-q)^{\top}\Sigma_p^{-1}(p-q)\right)$$

공분산 행렬 $\Sigma_p$는 다음과 같이 파라미터화됩니다:

```math
\Sigma_p = R(\theta_p) \begin{pmatrix} \sigma_{x,p}^2 & 0 \\ 0 & \sigma_{y,p}^2 \end{pmatrix} R(\theta_p)^{\top}
```

여기서 $R(\theta_p)$는 회전각 $\theta_p$에 의한 회전 행렬이며, 결합된 spatial–range 커널은:

$$k_{\text{GSJBU}}(p, q) = \exp\!\left(-\frac{1}{2}(p-q)^{\top}\Sigma_p^{-1}(p-q) - \frac{\|I(p)-I(q)\|^2}{2\sigma_{r,p}^2}\right)$$

> **⚠️ 주의**: 위 수식은 논문의 공개 arXiv HTML 버전(17번 인용)에서 언급된 파라미터 $(\sigma_x, \sigma_y, \theta, \sigma_r)$와 블록 대각 공분산(block-diagonal covariance) 구조를 기반으로 구성한 것으로, 논문 내 정확한 수식 번호(eq. 12 등)의 세부 전개는 PDF 직접 열람이 필요합니다.

#### ③ 최적화 목표

최적화는 색상 재구성만을 가이드로 사용하지만, 학습된 커널은 암묵적으로 기하학적 구조와 의미 정보(geometry and semantics)를 포착합니다.

$$\min_{\{\sigma_{x,p}, \sigma_{y,p}, \theta_p, \sigma_{r,p}\}} \mathcal{L}_{\text{recon}}(I_{\text{HR}}, \hat{I}_{\text{HR}})$$

표 8의 하이퍼파라미터들은 TTO를 위한 소프트 사전(soft priors)으로 기능하며, 모든 공간 및 범위 파라미터가 50회 최적화 스텝 동안 정제되므로 최종 성능은 초기값에 크게 의존하지 않습니다.

### 2-3. 모델 구조

```
[입력: HR RGB 이미지 I_HR]
        ↓ 다운샘플링
[LR RGB 이미지 I_LR]
        ↓ TTO 최적화 (50 steps)
[픽셀별 Anisotropic Gaussian 파라미터 학습]
    {σ_x, σ_y, θ, σ_r} per pixel
        ↓
[학습된 GSJBU 커널]
        ↓ LR Feature에 적용
[HR Feature Map 생성]
        ↓
[Downstream Task (1×1 conv decoder 등)]
```

이러한 방법들은 저해상도 파운데이션 피처를 고해상도로 매핑하는 업샘플링 연산자를 학습하여, 다운스트림 디코더 이전에 의미론적-공간적 갭을 효과적으로 연결하며, 단일 $1\times1$ 컨볼루션 디코더만으로도 다양한 픽셀 수준 태스크에서 강력한 성능을 달성합니다.

### 2-4. 성능 향상

224×224 이미지 기준 $0.419\text{s}$의 처리 시간으로 시맨틱 세그멘테이션, 깊이 추정, 깊이 맵 및 확률 맵 업샘플링에서 SOTA 성능을 달성합니다.

COCO, PASCAL-VOC, ADE20k 데이터셋에서 다양한 업샘플링 방법을 비교하였으며, 공정한 비교를 위해 기존 연구와 동일하게 $1\times1$ 컨볼루션 헤드만을 파인튜닝하는 linear-probe 프로토콜을 채택했습니다.

100 에폭 학습과 코사인 학습률 스케줄 적용 결과, 백본 표현이 강력할 경우 모든 방법이 단순한 bilinear 업샘플링 대비 최종 성능 향상이 크지 않다는 경향이 발견되었습니다.

### 2-5. 한계

AnyUp은 노이즈에 안정적인 반면, TTO 기반인 Upsample Anything은 손상된 입력 픽셀에 과적합(overfits)되는 한계를 보이며, 이는 노이즈가 있는 입력에 직접 최적화할 때의 제한점을 드러냅니다.

---

## 3. 일반화 성능 향상 가능성

Upsample Anything은 2D 피처 해상도를 향상시킬 뿐만 아니라 깊이, 세그멘테이션, 심지어 3D 표현과 같은 다른 픽셀/복셀 수준 신호에도 재학습 없이 일반화되며, 이러한 특성은 **2D 및 3D 도메인에 걸친 통합적이고 경량화된 해상도-자유(resolution-free) 업샘플링 연산자**로서의 잠재력을 부각시킵니다.

학습된 커널은 **범용적이고 엣지를 인식하는(edge-aware) 연산자**로 작동하여, 아키텍처와 모달리티에 걸쳐 원활하게 전이되어 피처, 깊이, 확률 맵의 정밀한 고해상도 재구성을 가능하게 합니다.

이 논문은 놀라울 정도로 단순하면서도 매우 효과적인 피처 업샘플링을 위한 TTO 프레임워크를 제시하며, 이 방법은 완전히 학습이 필요 없고(training-free) **도메인, 태스크, 백본 아키텍처에 걸쳐 원활하게 일반화됩니다**.

일반화 성능 향상 가능성을 구체적으로 정리하면:

| 차원 | 일반화 범위 |
|------|------------|
| **모달리티** | RGB 피처 → 깊이 맵, 세그멘테이션, 확률 맵, 3D 표현 |
| **아키텍처** | ViT, DINO, CLIP, MAE 등 다양한 백본 |
| **해상도** | 임의의 입력/출력 해상도 지원 |
| **도메인** | 재학습 없이 다양한 데이터셋 적용 가능 |

---

## 4. 관련 최신 연구 비교 분석 (2020년 이후)

| 방법 | 연도 | 최적화 방식 | 일반화 | 속도 | 특징 |
|------|------|------------|--------|------|------|
| **Bilinear** | - | 없음 | ✅ 높음 | ✅ 매우 빠름 | 단순, 엣지 손실 |
| **FeatUp (JBU)** | ICLR 2024 | Dataset-level | ❌ 낮음 | ✅ 빠름 | MLP 기반 range kernel |
| **FeatUp (Implicit)** | ICLR 2024 | Dataset-level | ❌ 낮음 | ❌ 느림 | 임의 해상도 학습 |
| **LiFT** | 2024 | Dataset-level | ❌ 낮음 | ✅ 빠름 | 좌표 기반 피처 |
| **JAFAR** | 2025 | Dataset-level | ❌ 낮음 | ✅ 빠름 | Flow matching |
| **AnyUp** | 2025 | Inference-time | ✅ 높음 | ✅ 빠름 | 피처 비지정 아키텍처 |
| **Upsample Anything** | 2025 | TTO(per-image) | ✅ 높음 | ✅ ~0.419s | GSJBU, 학습불필요 |

FeatUp은 멀티뷰 일관성을 이용해 딥 피처를 업샘플링하는 새로운 접근법으로, JBU 기반 업샘플러는 강력한 공간적 사전을 부과하여 소실된 공간 정보를 Joint Bilateral Upsampling의 일반화를 기반으로 한 빠른 피드포워드 네트워크로 정확하게 복원합니다.

FeatUp, LoftUp, JAFAR 등의 피처 업샘플링 방법들은 학습된 비전 인코더와 쌍을 이룰 때 좋은 성능을 보이지만, 일반적으로 인퍼런스 시 인코더에 독립적이지 않아 다른 피처 추출기에 사용하려면 재학습이 필요하며, 이는 최신 대형 비전 모델의 경우 제한된 컴퓨팅 자원으로는 불가능할 수도 있습니다.

Upsample Anything은 효율적인 TTO 하에 anisotropic Gaussian splatting과 joint bilateral filtering을 조화시켜 해상도·모델 독립적이고 엣지를 보존하는 업샘플링을 이미지당 1초 미만의 런타임으로 달성하며, 데이터셋 수준 업샘플러 재학습의 필요성에 강건하게 도전하고 현대 비전 파이프라인에서 유연하고 효율적이며 일반화 가능한 피처 업샘플링의 길을 열어줍니다.

---

## 5. 앞으로의 연구에 미치는 영향 및 고려 사항

### 5-1. 연구에 미치는 영향

Upsample Anything 프레임워크는 범용 피처 업샘플링을 위한 새롭고 단순하며 매우 효과적인 베이스라인을 확립합니다. anisotropic Gaussian splatting과 joint bilateral filtering을 효율적인 TTO 하에 조화시킴으로써, 이미지당 1초 이내의 런타임으로 해상도 및 모델에 독립적인 엣지 보존 업샘플링을 달성하며, 데이터셋 수준 업샘플러 재학습의 필요성에 도전하고 현대 비전 파이프라인에서 유연하고 일반화 가능한 피처 업샘플링의 길을 엽니다.

**구체적 영향:**

1. **VFM 기반 픽셀 예측 연구의 패러다임 전환**: 재학습 없이 어떤 VFM에도 적용 가능한 업샘플러가 표준 베이스라인이 될 수 있음
2. **3D·의료·위성 이미징 응용**: 깊이, 세그멘테이션, 3D 표현에 재학습 없이 일반화되는 특성은 의료 영상, 원격 탐사 등 데이터 부족 도메인에서 특히 유용
3. **Gaussian Splatting과 2D 비전의 융합**: 3DGS 기법을 2D 피처 업샘플링에 적용한 최초 시도 중 하나로, 후속 연구의 이론적 토대 제공

### 5-2. 향후 연구 시 고려할 점

1. **노이즈 강건성 확보**: TTO 기반 방법이 노이즈가 있는 픽셀에 과적합되는 한계를 극복하기 위해 노이즈 인식(noise-aware) 정규화 항 추가 연구 필요

2. **고해상도 확장성**: 224×224에서 0.419s이지만, 고해상도 이미지(1K, 4K)에서의 확장성 및 메모리 효율성 검증 필요

3. **강력한 백본에서의 효과 재검토**: 백본 표현이 강력할 경우, 단순 bilinear 업샘플링 대비 피처 업샘플링의 실제 이득이 크지 않을 수 있어, 어떤 조건에서 피처 업샘플링이 유효한지에 대한 연구가 필요합니다.

4. **TTO와 Dataset-level 방법의 앙상블**: 두 패러다임의 장점을 결합하는 하이브리드 접근법 탐색

5. **비디오·시계열 확장**: 현재는 이미지별(per-image) 최적화이므로, 시간적 일관성을 고려한 비디오 업샘플링으로의 확장

6. **자기지도 신호 활용**: 색상 재구성만으로 최적화하는 현재 방식에서 더 풍부한 자기지도 신호(depth consistency, flow, 등)를 활용하는 연구 방향

---

## 참고 문헌 및 출처

| # | 출처 |
|---|------|
| 1 | **[주 논문]** Minseok Seo et al., "Upsample Anything: A Simple and Hard to Beat Baseline for Feature Upsampling," arXiv:2511.16301, Nov. 2025. https://arxiv.org/abs/2511.16301 |
| 2 | **[프로젝트 페이지]** https://seominseok0429.github.io/Upsample-Anything/ |
| 3 | **[논문 HTML]** https://arxiv.org/html/2511.16301v1 |
| 4 | **[논문 PDF]** https://arxiv.org/pdf/2511.16301 |
| 5 | **[리뷰]** The Moonlight, "Literature Review: Upsample Anything," https://www.themoonlight.io/en/review/upsample-anything-a-simple-and-hard-to-beat-baseline-for-feature-upsampling |
| 6 | **[요약]** Emergent Mind, https://www.emergentmind.com/papers/2511.16301 |
| 7 | **[HuggingFace]** Paper page, https://huggingface.co/papers/2511.16301 |
| 8 | **[비교: FeatUp]** Fu et al., "FeatUp: A Model-Agnostic Framework for Features at Any Resolution," ICLR 2024. https://arxiv.org/html/2403.10516v1 |
| 9 | **[비교: AnyUp]** Wimmer et al., "AnyUp: Universal Feature Upsampling," arXiv:2510.12764, 2025. https://arxiv.org/pdf/2510.12764 |
| 10 | **[비교: UPLiFT]** "UPLiFT: Efficient Pixel-Dense Feature Upsampling with Local Attenders," arXiv:2601.17950. https://arxiv.org/pdf/2601.17950 |
| 11 | **[Cool Papers]** https://papers.cool/arxiv/2511.16301 |

# Upsample Anything: A Simple and Hard to Beat Baseline for Feature Upsampling

## 1. 핵심 주장 및 기여 요약

**Upsample Anything**은 Vision Foundation Models (VFMs)의 저해상도 특징 맵을 고해상도로 복원하는 **훈련 없는 경량 테스트-타임 최적화(Test-Time Optimization, TTO)** 프레임워크입니다. 본 논문의 핵심 주장은 다음과 같습니다:[1]

**주요 기여:**
- **비훈련 접근법의 획기적 성과**: 데이터셋 수준의 재훈련 없이 이미지당 0.419초 만에 224×224 이미지를 처리하면서도 SOTA 성능 달성
- **통합 프레임워크**: Gaussian Splatting과 Joint Bilateral Upsampling(JBU)을 연결하는 새로운 수학적 형식 제시
- **범용 오퍼레이터**: 피셀 레벨 재가중치 기반 설계로 아키텍처, 모달리티, 해상도에 구애받지 않고 전이 가능
- **광범위한 적용 가능성**: 특징 맵, 깊이 맵, 확률 맵, 심지어 3D 특징 볼륨까지 확장 가능

***

## 2. 문제 정의 및 동기

### 2.1 해결하고자 하는 문제

Vision Foundation Models(DINOv2, CLIP, MAE 등)는 자가지도학습을 통해 뛰어난 일반화 능력을 보유하지만, 계산 효율성을 위해 입력 이미지를 14×16배 다운샘플링하여 저해상도 특징 맵을 생성합니다. 이는 다음과 같은 문제들을 야기합니다:[1]

1. **해상도 병목**: 의미론적 분할, 깊이 추정, 인스턴스 분할 등 픽셀 레벨 예측 작업에 직접 사용 불가
2. **기존 해결책의 한계**:
   - **데이터셋 수준 훈련 방법**(FeatUp, LoftUp, JAFAR, AnyUp): 각 VFM 및 데이터셋마다 재훈련 필요, 메모리 제약으로 112-224 픽셀 해상도로 제한
   - **테스트-타임 최적화 방법**(FeatUp Implicit): 이미지당 약 49초 소요로 확장성 부족

### 2.2 기존 접근법의 한계

논문에서 지적하는 주요 한계들:[1]
- 고정된 손제작 커널 설계의 한계(classical JBU)
- 데이터셋별 재훈련의 비효율성
- 계산량의 이차적 증가에 따른 메모리 오버헤드
- 다양한 아키텍처 및 해상도에 대한 일반화 부족

***

## 3. 제안 방법론

### 3.1 수학적 기초

#### 3.1.1 Joint Bilateral Upsampling (JBU) 재검토

고전적 JBU는 다음과 같이 정의됩니다:[1]

$$F^{hr}_p = \frac{1}{Z_p} \sum_{q \in \Omega_p} F^{lr}_q \exp\left(-\frac{\|p-q\|^2}{2\sigma^2_s}\right) \exp\left(-\frac{\|I_p - I_q\|^2}{2\sigma^2_r}\right)$$

여기서:
- $\(F^{hr}_p\)$: 고해상도 출력
- $\(F^{lr}_q\)$: 저해상도 입력
- $\(I_p, I_q\)$: 고해상도 지도 이미지
- $\(\sigma_s, \sigma_r\)$: 공간 및 범위(색상) 감도 제어
- $\(Z_p\)$: 정규화 인수

#### 3.1.2 2D Gaussian Splatting(2DGS)으로의 재해석

논문은 JBU를 2D 가우시안 스플래팅 프레임워크로 재해석합니다. 각 저해상도 픽셀 \(q\)는 가우시안 커널로 표현됩니다:[1]

$$G_i(x) = \exp\left(-\frac{1}{2}(x - \mu_i)^T \Sigma_i^{-1} (x - \mu_i)\right)$$

여기서:
- $\(\mu_i \in \mathbb{R}^2\)$: 가우시안 중심
- $\(\Sigma_i \in \mathbb{R}^{2 \times 2}\)$: 양의 정부호 공분산 행렬
- 렌더링은 정규화된 알파 블렌딩으로 수행

#### 3.1.3 이방성 가우시안 커널 설계

**Upsample Anything의 핵심 혁신**은 픽셀별 이방성 가우시안 파라미터 학습입니다. 각 저해상도 위치 \(q\)마다 다음 파라미터들을 최적화합니다:[1]

**공분산 행렬:**

```math
\Sigma_q = R_q \begin{pmatrix} \sigma^2_{x_q} & 0 \\ 0 & \sigma^2_{y_q} \end{pmatrix} R_q^T, \quad R_q = \begin{pmatrix} \cos\theta_q & -\sin\theta_q \\ \sin\theta_q & \cos\theta_q \end{pmatrix}
```

**공간 가중치:**
$$\log w^s_{p,q} = -\frac{1}{2}(p - \mu_q)^T \Sigma_q^{-1}(p - \mu_q)$$

**범위(색상) 가중치:**
$$\log w^r_{p,q} = -\frac{\|I_p - I_q\|^2}{2\sigma^2_{r,q}}$$

**최종 정규화 가중치:**
$$w_{p,q} = \frac{\exp(\log w^s_{p,q} + \log w^r_{p,q})}{\sum_{q' \in \Omega_p} \exp(\log w^s_{p,q'} + \log w^r_{p,q'})}$$

여기서 파라미터 집합은 $\(\{\sigma_{x_q}, \sigma_{y_q}, \theta_q, \sigma_{r,q}\}\)$입니다.

### 3.2 테스트-타임 최적화 (TTO) 단계

**Step 1: 이미지 다운샘플링**
입력 고해상도 이미지 $\(I^{hr}\)$를 저해상도 $\(I^{lr}\)$로 다운샘플링하여 VFM의 다운샘플링 패턴 모방:[1]

$$I^{lr} = \text{Bilinear}(I^{hr}, \text{stride}=s)$$

**Step 2: 재구성 목표**
다음 손실 함수로 픽셀별 파라미터 최적화:[1]

$$\mathcal{L}_{TTO} = \|GS_{JBU}(I^{lr}) - I^{hr}\|$$

Adam 옵티마이저 사용 (학습율: $\(1 \times 10^{-3}\))$, 단 50 반복으로 수렴

**Step 3: 특징 렌더링**
최적화된 커널을 저해상도 특징 맵 $\(F^{lr} \in \mathbb{R}^{C \times H_s \times W_s}\)$에 적용:[1]

$$F^{hr}_p = \sum_{q \in \Omega_p} w_{p,q} F^{lr}_q$$

이는 **값 합성 없이 순수 재가중치** 기반으로, 모든 모달리티로 자동 전이 가능

### 3.3 모델 구조

```
입력: 고해상도 RGB 이미지 I^hr (224×224)
  ↓
[스텝 1] 저해상도로 다운샘플링 (stride=s)
         I^lr = Bilinear(I^hr, stride=s)
  ↓
[스텝 2] 테스트-타임 최적화 (50회 반복)
         최적화 대상: {σ_{x,q}, σ_{y,q}, θ_q, σ_{r,q}} ∀q
         손실: ||GS_JBU(I^lr) - I^hr||
  ↓
[스텝 3] 특징 맵 렌더링
         F^lr (VFM 출력) → 학습된 커널 적용 → F^hr
  ↓
출력: 고해상도 특징 맵 F^hr (원본 해상도)
```

***

## 4. 성능 평가 및 향상

### 4.1 의미론적 분할(Semantic Segmentation)

| 데이터셋 | COCO | PASCAL-VOC | ADE20K |
|---------|------|-----------|--------|
| 메트릭 | mIoU / Acc. | mIoU / Acc. | mIoU / Acc. |
| Bilinear | 60.43 / 80.18 | 81.27 / 95.96 | 41.48 / 74.95 |
| FeatUp | 60.96 / 80.65 | 81.91 / 96.27 | 41.92 / 75.41 |
| LoftUp | 61.08 / 80.72 | 81.84 / 96.33 | 41.83 / 75.36 |
| JAFAR | 60.87 / 80.51 | 82.05 / 96.21 | 41.74 / 75.22 |
| AnyUp | 61.25 / 80.89 | 82.18 / 96.39 | 42.02 / 75.63 |
| **Upsample Anything** | **61.41 / 81.34** | **82.22 / 96.90** | **42.95 / 76.52** |

**확률 맵 업샘플링 (Upsample Anything prob.)**:[1]
- COCO: 63.40 mIoU / 83.73 Acc. (SOTA)
- PASCAL-VOC: 84.57 mIoU / 97.42 Acc. (SOTA)
- ADE20K: 44.29 mIoU / 78.58 Acc. (SOTA)

### 4.2 깊이 추정 및 표면 법선 추정 (NYUv2)

| 메트릭 | Bilinear | FeatUp | LoftUp | JAFAR | AnyUp | **Upsample Anything** |
|--------|----------|--------|--------|--------|--------|----------------------|
| RMSE ↓ | 0.545 | 0.523 | 0.796 | 0.521 | 0.513 | **0.498** |
| δ₁ ↑ | 0.804 | 0.810 | 0.789 | 0.807 | 0.817 | **0.829** |
| Mean ↓ | 23.8 | 22.7 | 28.9 | 23.2 | 22.2 | **21.5** |

특히 깊이 추정은 특징 업샘플링의 효과가 크게 나타나는 작업입니다.

### 4.3 깊이 맵 업샘플링 (Middlebury & NYUv2)

| 데이터셋 | 메트릭 | Bilinear | GLU | **Upsample Anything** |
|---------|-------|----------|-----|----------------------|
| Middlebury | RMSE ↓ | 0.231 | 0.491 | **0.209** |
| | δ₁ ↑ | 0.962 | 0.825 | **0.967** |
| NYUv2 | RMSE ↓ | 0.167 | 0.372 | 0.214 |

### 4.4 해상도별 성능

다양한 입력 해상도(4×4, 7×7, 16×16, 32×32)에서 평가했을 때, **Upsample Anything이 극저 해상도(4×4, 7×7)에서도 안정적인 성능 유지**:[1]
- AnyUp: 해상도 감소 시 과도하게 평활화된 영역 생성
- Upsample Anything: 모든 해상도에서 예리한 경계선 및 미세한 구조 보존

### 4.5 아키텍처 간 전이 가능성

다양한 백본(DINOv1, DINOv2, DINOv3, ConvNeXt, CLIP)에서 평가:
- **AnyUp**: 아키텍처별 해상도 차이(7×7~16×16)에서 성능 편차 크음
- **Upsample Anything**: 모든 백본에서 **일관되게 더 예리한 경계선과 미세한 텍스처** 유지[1]

***

## 5. 일반화 성능 향상 메커니즘

### 5.1 왜 일반화되는가?

**핵심 원리**: 값 합성 없는 순수 재가중치 기반 설계[1]

1. **공간-범위 유사성 기반**: 최적화된 가중치는 **입력 이미지의 기하학적 구조와 시각적 유사성만 캡처**
2. **도메인 무관성**: 이 정보는 RGB 색상 이미지에 국한되지 않고, 동일한 기하학적 원리로 깊이, 의미론적 레이블, 3D 특징 등 모든 신호에 적용 가능
3. **해상도 자유성**: 재가중치 메커니즘은 절대 해상도에 의존하지 않으므로, 학습된 커널을 다양한 업샘플링 배수에 적용 가능

### 5.2 특징 유사성 분석

논문의 Feature Similarity 시각화 결과:[1]
- **AnyUp**: 이미지 전체에서 균일하게 높은 코사인 유사도 → 공간 구분 능력 부족
- **Upsample Anything**: **국소화된 유사도 맵** → 객체 경계가 명확하게 보존되고 서로 다른 영역이 잘 구분

→ **Few-shot 분할, 범주별 특징 매칭 등의 작업에 높은 잠재력**

### 5.3 3D 특징 볼륨 확장

2D 저해상도 특징을 3D 고해상도 특징 볼륨으로 확장:[1]

RGB-D 쌍을 활용하여 $\(\{\sigma_{x_q}, \sigma_{y_q}, \sigma_{z_q}, \theta_q, \sigma_{r,q}\}\)$ 매개변수 학습:

$$F^{3D} \in \mathbb{R}^{D \times H \times W \times H_d}$$

PCA 시각화 결과, **깊이 연속성과 객체 경계를 보존하면서도 명의적 의미 분리** 달성

***

## 6. 모델의 한계

### 6.1 저신호-잡음비(Low-SNR) 환경에서의 취약성

**주요 한계**: TTO가 입력 이미지 자체를 재구성 목표로 삼으므로, 입력이 노이즈로 오염되면 **최적화가 노이즈에 오버핏** 됨.[1]

실험 결과:
- 노이즈 수준 σ=10: 여전히 합리적 성능
- 노이즈 수준 σ=20: 현저한 성능 저하
- 비교: AnyUp은 VFM 특징 자체가 노이즈 강건하므로 더 안정적

### 6.2 기타 제약사항

1. **폐색(Occlusion) 환경**: 심각한 폐색이 있을 때 최적화 불안정
2. **극저 해상도(4×4)**: 비록 작동하지만, 정보 손실로 인해 극도로 제한된 복원
3. **지역 구조의 모호성**: 패턴이 반복되는 영역에서 커널이 정확히 구분할 수 없음

***

## 7. 최신 관련 연구 탐색 (2020년 이후)

### 7.1 Feature Upsampling 분야의 최신 동향

**FeatUp (2024)**[2]
- JBU를 특징 공간으로 확장하고 학습 가능한 MLP 기반 범위 커널 도입
- 데이터셋 수준 훈련 또는 이미지 수준 암시적 최적화 제공
- Upsample Anything보다 느림 (이미지당 ~49초)

**LoftUp (2025)**[3]
- 좌표 기반 크로스-어텐션 트랜스포머 도입
- 클래스-무관 마스크 및 자기-증류를 통한 의사-그라운드트루스 구성
- 데이터셋 수준 훈련 필요 (더 높은 성능하지만 확장성 제한)

**NAF: Neighborhood Attention Filtering (2025)**[4]
- 크로스-스케일 이웃 어텐션 및 회전 위치 임베딩(RoPE) 활용
- VFM-무관 영(zero-shot) 아키텍처 제시
- 2K 특징 맵에서 18 FPS 달성
- **Upsample Anything보다 빠르면서 SOTA 성능 달성**

**FeatSharp (2025)**[5]
- 저해상도 버퍼와 고해상도 타일의 모자이크를 결합
- CLIP 등 저해상도 ViT의 특징 업샘플링
- 세분화된 세부사항 복원에 중점

**Benchmarking Feature Upsampling Methods (2025)**[6]
- 인터랙티브 분할(IS) 벤치마크로 다양한 업샘플링 방법 평가
- LoftUp이 최대 50% 성능 향상 보여줌
- 다중 모달 입력에서의 일반화 능력 평가

### 7.2 2D Gaussian Splatting 관련 연구

**2D Gaussian Splatting (2024)**[7]
- 3D GS를 2D 이미지 평면으로 확장
- 실시간 렌더링 가능한 2D 표현 제공
- Upsample Anything의 수학적 기초 제공

**RestorGS (2025)**[8]
- Depth-aware Gaussian Splatting으로 3D 장면 복원
- 다양한 환경(수중, 야간, 안개) 처리
- 고품질 렌더링과 효율성의 균형

### 7.3 Vision Transformer 및 조밀 예측

**DINOv2 (2024)**[9]
- 자가지도학습 기반 일반화된 시각 백본
- 고정된 특징 사용으로 강한 전이 학습 능력

**Vision Transformers for Dense Prediction (2021, DPT)**[10]
- ViT 백본을 조밀 예측에 활용한 선구적 연구
- Hybrid-CNN-ViT 아키텍처로 높은 성능 달성

### 7.4 Implicit Neural Representations

**Implicit Neural Representation 연구 (2021-2025)**
- 좌표 기반 MLP(Coordinate MLP)로 공간 함수 학습
- 다양한 해상도에서 연속적 표현 제공
- 지구물리학, 로봇공학, 3D 복원 분야로 확장

***

## 8. 논문이 앞으로의 연구에 미치는 영향

### 8.1 긍정적 영향

**1. 패러다임 전환**: TTO 기반 비훈련 접근의 실용화
- 새로운 VFM이 지속적으로 출현하는 현대에 **재훈련 불필요한 범용 솔루션** 제시
- 데이터셋 수집 및 라벨링 비용 절감

**2. 수학적 통찰**: JBU-2DGS 연결 프레임워크
- 고전적 신호처리 기법과 최신 신경망 최적화의 이론적 교점 제시
- 다른 신호처리 알고리즘의 신경망 통합 연구 영감

**3. 효율성과 확장성**:
- 0.419초/이미지로 **실시간 처리 근처 성능** 달성
- 1024×1024 해상도까지 선형적 메모리 성장 (AnyUp은 OOM)
- 모바일 및 에지 컴퓨팅 적용 가능성

**4. 모달리티 전이**: 도메인 무관 가중치
- RGB → 깊이, 세분화, 3D 특징으로 **자동 전이** 가능
- 각 모달리티별 별도 모델 불필요

### 8.2 영향 미칠 분야

**1. 의료 영상**
- 한정된 고해상도 데이터로 정밀 진단 보조
- 초저해상도 센서 데이터의 사후처리

**2. 자율주행**
- 실시간 깊이 및 의미론적 분할 추론
- 다중 센서 피전 시스템 업샘플링

**3. 로봇 지각**
- 저전력 카메라의 특징 품질 향상
- Few-shot 학습 기반 물체 인식

**4. 기본 연구**
- VFM의 특징 공간 이해 심화
- 미분 가능 신호처리 이론 발전

### 8.3 앞으로의 연구 방향 및 고려사항

**1. 노이즈 강건성 개선**
- **현재 한계**: Low-SNR 환경에서 오버핏
- **개선 방향**:
  - 사전 노이즈 제거 단계 통합
  - 정규화 전략 (예: 자기 지도 학습, 다중 스케일 손실)
  - 적응형 학습률 스케줄

**2. 더 이상의 컴퓨팅 최적화**
- **GPU 메모리 효율성**: 배치 처리 또는 패치 기반 TTO
- **주변장치 배포**: 양자화(quantization), 가지치기(pruning) 적용
- **점진적 최적화**: 멀티-스케일 피라미드 구조

**3. 강화된 일반화**
- **Out-of-distribution 샘플**: 극도의 물리 조건(저조도, 초고온)에서 강건성
- **도메인 적응**: 스타일 전이, 기하학적 변환에 대한 불변성

**4. 멀티모달 확장**
- **비전-언어 모델**: CLIP 특징뿐만 아니라 텍스트 임베딩 업샘플링
- **멀티센서 융합**: 열화상, LiDAR, 레이더 데이터 통합

**5. 이론적 분석**
- **수렴 보증**: TTO의 수렴성과 최적성 증명
- **표현 용량 분석**: 픽셀별 파라미터의 효과적 자유도 분석
- **일반화 경계**: 학습된 커널의 일반화 오차에 대한 이론적 한계

**6. 적응형 파라미터 초기화**
- **현재**: 고정된 초기값 (σₓ = σᵧ = 16.0, σᵣ = 0.12)
- **개선**: 이미지 특성(엣지 밀도, 텍스처 복잡도)에 따른 동적 초기화

**7. 장기-단기 시퀀스 최적화**
- **비디오 업샘플링**: 프레임 간 일관성 유지하며 실시간 처리
- **시간적 정규화**: 플리커 제거, 움직임 보정

**8. 인터랙티브 및 사용자 제어**
- **적응형 업샘플링**: 사용자가 선택한 영역에 집중
- **피드백 루프**: 재구성 품질 실시간 모니터링 및 파라미터 조정

***

## 9. 결론 및 종합 평가

**Upsample Anything**은 Vision Foundation Models 시대에 Feature Upsampling 문제에 대한 **우아하고 실용적인 솔루션**을 제시합니다. 고전적 신호처리(JBU)와 최신 최적화 기법(미분 가능 가우시안 스플래팅)을 결합하여, 훈련 없이도 강한 성능을 달성하는 동시에 **뛰어난 확장성과 모달리티 전이성**을 확보했습니다.

**강점 요약:**
✓ 훈련 불필요 (도메인 적응성)
✓ 초고속 처리 (0.419초/이미지)
✓ 높은 해상도 지원 (1024×1024 이상)
✓ 범용 오퍼레이터 (모든 모달리티 자동 전이)
✓ SOTA 성능 (세분화, 깊이 추정, 깊이/확률 맵 업샘플링)

**약점 및 한계:**
✗ Low-SNR 환경에서의 노이즈 오버핏
✗ 극심한 폐색에 대한 취약성
✗ 극저 해상도(4×4)에서의 정보 손실

**미래 영향:**
이 연구는 VFM 시대의 특징 공학(feature engineering)에 새로운 기준을 제시하며, 실시간 시스템(자율주행, 로봇), 의료 영상, 에지 컴퓨팅 애플리케이션에 광범위한 적용 잠재력을 보유합니다. 특히 "일반화된 비훈련 업샘플러"라는 개념의 실현은 향후 다양한 신호처리 및 재구성 작업의 새로운 관점을 열 것으로 기대됩니다.

***

## 참고문헌

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2804c451-7274-45c7-8298-15d5892f09bb/2511.16301v2.pdf)
[2](https://arxiv.org/html/2403.10516)
[3](https://arxiv.org/abs/2504.14032)
[4](https://www.semanticscholar.org/paper/cd4e1c99d7b5269f76153cdc87111def45933e90)
[5](https://openreview.net/forum?id=lioemOcq3H)
[6](https://arxiv.org/abs/2505.02075)
[7](https://europe.naverlabs.com/research/publications/gaussian-splatting-feature-fields-for-privacy-preserving-visual-localization/)
[8](https://openaccess.thecvf.com/content/CVPR2025/papers/Qiao_RestorGS_Depth-aware_Gaussian_Splatting_for_Efficient_3D_Scene_Restoration_CVPR_2025_paper.pdf)
[9](http://arxiv.org/pdf/2304.07193.pdf)
[10](https://arxiv.org/abs/2103.13413)
[11](https://arxiv.org/abs/2407.13111)
[12](https://arxiv.org/abs/2509.17566)
[13](https://www.semanticscholar.org/paper/a2ec08c6f203111005016fdff5630fab845df500)
[14](https://arxiv.org/abs/2508.21529)
[15](https://arxiv.org/abs/2506.09784)
[16](https://aacrjournals.org/cancerres/article/85/8_Supplement_1/2471/758131/Abstract-2471-Interpretable-HPV-detection-in-head)
[17](https://ieeexplore.ieee.org/document/10678652/)
[18](https://arxiv.org/html/2306.12642v1)
[19](http://arxiv.org/pdf/2502.16025.pdf)
[20](http://arxiv.org/pdf/2410.22217.pdf)
[21](https://bjo.bmj.com/content/bjophthalmol/early/2024/06/04/bjo-2024-325459.full.pdf)
[22](https://arxiv.org/pdf/2308.12462.pdf)
[23](http://arxiv.org/pdf/2111.11432.pdf)
[24](https://openaccess.thecvf.com/content/ICCV2025/papers/Huang_LoftUp_Learning_a_Coordinate-Based_Feature_Upsampler_for_Vision_Foundation_Models_ICCV_2025_paper.pdf)
[25](https://www.nature.com/articles/s41598-020-61808-3)
[26](https://owl-d.tistory.com/31)
[27](https://arxiv.org/html/2505.02075v1)
[28](https://discovery.ucl.ac.uk/10153271/1/2207.10996.pdf)
[29](https://openaccess.thecvf.com/content/ICCV2025W/ILR+G/papers/Havrylov_Benchmarking_Feature_Upsampling_Methods_for_Vision_Foundation_Models_using_Interactive_ICCVW_2025_paper.pdf)
[30](https://pubmed.ncbi.nlm.nih.gov/37222565/)
[31](https://www.arxiv.org/abs/2505.02075)
[32](https://arxiv.org/abs/2501.04628)
[33](https://jpcsit.kaznu.kz/index.php/kaznu/article/view/212)
[34](https://ijamjournal.org/ijam/publication/index.php/ijam/article/view/548)
[35](https://dl.acm.org/doi/10.1145/3715335.3735455)
[36](https://www.mdpi.com/2227-7390/13/10/1571)
[37](https://ieeexplore.ieee.org/document/11187897/)
[38](https://www.mdpi.com/1424-8220/25/18/5876)
[39](https://jisem-journal.com/index.php/journal/article/view/1369)
[40](https://www.worldscientific.com/doi/10.1142/S0219467827500665)
[41](https://link.springer.com/10.1007/s44352-025-00009-y)
[42](https://link.springer.com/10.1007/s42600-025-00411-9)
[43](http://arxiv.org/pdf/1503.04949.pdf)
[44](http://arxiv.org/abs/1405.4734)
[45](http://arxiv.org/pdf/1707.02880.pdf)
[46](http://arxiv.org/pdf/1505.00077.pdf)
[47](https://arxiv.org/pdf/1603.08109.pdf)
[48](https://ace.ewapublishing.org/media/1370bb8897794d7390253374874a65bc.marked_kdnq1xF.pdf)
[49](http://arxiv.org/pdf/2210.15950.pdf)
[50](http://arxiv.org/pdf/1007.1016.pdf)
[51](https://pmc.ncbi.nlm.nih.gov/articles/PMC11029622/)
[52](https://www.nature.com/articles/s41598-024-83979-z)
[53](https://www.nature.com/articles/s41598-022-22530-4)
[54](https://renaud-detry.net/assets/pdf/Song-2024-CoRL.pdf)
[55](https://www.nature.com/articles/s41598-025-90518-x)
[56](https://www.sciencedirect.com/science/article/abs/pii/S0263224119303902)
[57](https://proceedings.mlr.press/v270/song25b.html)
[58](https://papers.neurips.cc/paper_files/paper/2022/file/20189b1aaa8edbb6d8bd6c1067ab5f3f-Paper-Conference.pdf)
[59](https://www.tandfonline.com/doi/full/10.1080/27684830.2022.2140863)
