
# TextureDreamer: Image-guided Texture Synthesis through Geometry-aware Diffusion

> **논문 정보**
> - **저자**: Yu-Ying Yeh, Jia-Bin Huang, Changil Kim, Lei Xiao, Thu Nguyen-Phuoc, Numair Khan, Cheng Zhang, Manmohan Chandraker, Carl S. Marshall, Zhao Dong 외
> - **발표**: CVPR 2024
> - **arXiv**: [2401.09416](https://arxiv.org/abs/2401.09416) (2024년 1월 17일)
> - **프로젝트 페이지**: [texturedreamer.github.io](https://texturedreamer.github.io/)

---

## 1. 핵심 주장 및 주요 기여 요약

TextureDreamer는 소수의 입력 이미지(3~5장)에서 재조명 가능한 텍스처(relightable textures)를 임의 카테고리의 목표 3D 형상으로 전이하는 새로운 이미지 유도 텍스처 합성 방법입니다.

### 🔑 핵심 주장

기존의 고전적 방법들은 조밀하게 샘플링된 뷰와 정확히 정렬된 기하 구조를 필요로 하고, 학습 기반 방법들은 데이터셋 내 카테고리 특정 형상에 국한됩니다. 반면 TextureDreamer는 소수의 비공식적으로 촬영된 이미지만으로 실세계 환경의 고도로 상세하고 복잡한 텍스처를 임의의 객체로 전이할 수 있으며, 이는 텍스처 생성 과정을 크게 민주화할 가능성이 있습니다.

### 🏆 주요 기여

핵심 아이디어인 **Personalized Geometry-aware Score Distillation (PGSD)**는 텍스처 정보 추출을 위한 개인화 모델링, 세부 외관 합성을 위한 Variational Score Distillation, ControlNet을 이용한 명시적 기하 가이던스를 통합한 것으로, 이러한 통합 및 여러 핵심 수정이 텍스처 품질을 실질적으로 향상시킵니다.

| 기여 | 설명 |
|---|---|
| PGSD 프레임워크 | DreamBooth + VSD + ControlNet 통합 |
| 기하 인식 텍스처 합성 | 노멀 맵 기반 3D 일관성 보장 |
| 재조명 가능 BRDF 최적화 | albedo, metallic, roughness 분리 |
| 카테고리 무관 전이 | 임의 형상에 적용 가능 |

---

## 2. 문제 정의, 제안 방법, 모델 구조, 성능 및 한계

### 2-1. 해결하고자 하는 문제

현실적이고 세밀한 텍스처 생성은 증강/가상 현실, 로보틱스, 엔터테인먼트 등 다양한 분야에서 중요한 위치를 차지합니다. 3D 자산용 텍스처를 제작하는 전통적인 방법은 노동 집약적이고 비용이 많이 들며 일반적으로 전문 아티스트에 의존합니다. 최근에도 많은 이미지 세트가 필요하거나 특정 객체 카테고리에 제한되는 문제들이 장애로 남아 있었습니다.

구체적으로 두 가지 핵심 문제가 있습니다:

1. **과평활화(Over-smoothing) / 과채화(Over-saturation)**: Score Distillation Sampling (SDS)는 텍스트-3D 생성에서 큰 가능성을 보였지만, 과채화, 과평활화, 다양성 부족 문제를 겪습니다.
2. **3D 비일관성**: VSD 손실만으로는 3D 일관성 문제를 완전히 해결할 수 없습니다.

---

### 2-2. 제안하는 방법 (수식 포함)

TextureDreamer의 파이프라인은 두 단계로 구성됩니다.

#### 📌 단계 1: 텍스처 추출 — DreamBooth 파인튜닝

DreamBooth는 소수의 입력 이미지로 사전 학습된 텍스트-이미지 확산 모델을 파인튜닝하는 간단하면서도 효과적인 방법입니다. 특정 텍스트 토큰 "[V]"로 피사체의 외관을 확산 모델 가중치에 저장합니다. DreamBooth는 두 가지 손실 함수로 파인튜닝됩니다: 재구성 손실은 입력 이미지에 대한 표준 확산 디노이징 감독이며, 클래스별 사전 보존 손실은 파인튜닝으로 인한 언어 드리프트 및 다양성 손실을 방지하기 위해 제안됩니다.

**재구성 손실 (Reconstruction Loss)**:

$$\mathcal{L}_{\text{recon}} = \mathbb{E}_{x, t, \epsilon}\left[\|\epsilon - \epsilon_\psi(\mathbf{x}_t, t, c)\|^2\right]$$

여기서 $\mathbf{x}\_t = \alpha_t \mathbf{x} + \sigma_t \epsilon$는 노이즈가 추가된 렌더링 이미지, $\epsilon_\psi$는 파인튜닝된 개인화 확산 모델입니다.

> **중요한 설계 선택**: 클래스별 사전 보존 손실(class-specific prior preservation loss)은 적용하지 않는데, 이는 DreamBooth 파인튜닝 모델이 다른 카테고리로 일반화되기를 원하기 때문입니다.

---

#### 📌 단계 2: 텍스처 합성 — PGSD (Personalized Geometry-aware Score Distillation)

BRDF 필드는 개인화 기하 인식 스코어 증류(PGSD)를 통해 최적화됩니다.

**기반: Variational Score Distillation (VSD)**

VSD는 3D 파라미터를 확률 변수로 취급하고 그 분포를 추론하는, 원칙에 입각한 입자 기반 변분 프레임워크입니다.

**VSD Gradient (기본 형태)**:

$$\nabla_\theta \mathcal{L}_{\text{VSD}} = \mathbb{E}_{t, \epsilon, c}\left[w(t)\left(\epsilon_\phi(\mathbf{x}_t, t, y) - \epsilon_\psi(\mathbf{x}_t, t, y, c)\right) \frac{\partial \mathbf{x}}{\partial \theta}\right]$$

여기서:
- $\theta$: BRDF 필드의 MLP 파라미터
- $\mathbf{x} = g(\theta; c)$: 카메라 포즈 $c$에서 렌더링된 이미지
- $\mathbf{x}_t = \alpha_t \mathbf{x} + \sigma_t \epsilon$: 노이즈 추가된 렌더링 이미지
- $\epsilon_\phi$: 대규모 데이터셋으로 사전학습된 **일반 확산 모델**
- $\epsilon_\psi$: DreamBooth로 파인튜닝된 **개인화 확산 모델**
- $w(t)$: 시간 가중치 함수

**PGSD — ControlNet을 통한 기하 가이던스 적용**:

기하 정보 조건화를 위해 ControlNet 아키텍처를 통해 주어진 메시에서 렌더링된 노멀 맵을 파인튜닝된 확산 모델에 주입합니다.

$$\nabla_\theta \mathcal{L}_{\text{PGSD}} = \mathbb{E}_{t, \epsilon, c}\left[w(t)\left(\epsilon_\phi(\mathbf{x}_t, t, y, \mathbf{n}) - \epsilon_\psi(\mathbf{x}_t, t, y, \mathbf{n}, c)\right) \frac{\partial \mathbf{x}}{\partial \theta}\right]$$

여기서 $\mathbf{n}$은 ControlNet에 입력되는 **노멀 맵**입니다.

두 모델 모두 노멀 맵에 조건화된 ControlNet으로 보강됩니다. 개인화 모델은 소수의 이미지로 파인튜닝되었기 때문에 CFG(Classifier-Free Guidance)로부터 이점을 얻지 못하는 것으로 나타났습니다.

**VSD를 위한 LoRA 학습 목표**:

변분 분포의 스코어를 모델링하기 위해 LoRA로 파라미터화된 추가적인 확산 모델을 학습합니다.

$$\mathcal{L}_{\text{LoRA}} = \mathbb{E}_{t, \epsilon}\left[\|\epsilon - \epsilon_\psi(\mathbf{x}_t, t, y, \mathbf{n}, c)\|^2\right]$$

**SDS와 VSD의 차이**:

SDS에서처럼 상수로 취급하지 않고 3D 파라미터를 확률 변수로 모델링하는 VSD는, SDS가 작은 CFG와 큰 CFG 모두에서 나쁜 샘플을 생성한다는 것을 보여줍니다.

| 항목 | SDS | VSD (TextureDreamer) |
|---|---|---|
| 3D 파라미터 취급 | 상수 | 확률 변수 |
| CFG 의존성 | 큰 CFG 필요 | 일반 CFG (7.5) 작동 |
| 결과 품질 | 과평활화, 과채화 | 고품질, 세밀한 텍스처 |

---

### 2-3. 모델 구조

TextureDreamer는 주어진 메시에 대해 3~5장의 입력 이미지와 유사한 외관의 텍스처를 합성하는 프레임워크입니다. 먼저 DreamBooth 파인튜닝으로 개인화 확산 모델 $\psi$를 얻습니다. 그 후 BRDF 필드 $\mathcal{M}$이 PGSD를 통해 최적화됩니다. 최적화가 완료되면 최적화된 BRDF 필드에서 albedo, metallic, roughness에 해당하는 고해상도 텍스처 맵을 추출합니다.

```
┌──────────────────────────────────────────────────────┐
│              TextureDreamer 파이프라인                 │
├──────────────────────────────────────────────────────┤
│  [입력] 3~5장의 이미지                                │
│       ↓                                              │
│  [Stage 1] DreamBooth 파인튜닝                       │
│   • 재구성 손실으로 텍스처 정보 추출                   │
│   • "[V]" 토큰에 외관 정보 저장                       │
│       ↓                                              │
│  [Stage 2] PGSD 최적화                               │
│   • VSD: 개인화 모델 vs 일반 모델 스코어 차이         │
│   • ControlNet (Normal Map 조건화)                   │
│   • LoRA 업데이트 (변분 분포 모델링)                  │
│   • 카메라 인코더 ρ 업데이트                          │
│       ↓                                              │
│  [출력] BRDF 필드 → Albedo + Metallic + Roughness    │
│         고해상도 텍스처 맵 (재조명 가능)               │
└──────────────────────────────────────────────────────┘
```

---

### 2-4. 성능 향상 (Ablation 분석)

ControlNet이 노멀 맵에 조건화된 결과는 최상의 텍스처-기하 일관성을 가집니다. ControlNet 없이 또는 깊이 기반 ControlNet을 사용한 결과는 텍스처-기하 불일치 문제를 겪습니다. SDS 손실 사용은 흐릿한 텍스처를 생성합니다. LoRA 모듈을 제거하면 개인화 확산 모델에서 기존 텍스처가 제거되는 경향이 있습니다. 전체 방법은 입력 외관과 유사한 정확한 텍스처를 합성할 수 있습니다.

다양한 카테고리의 실제 이미지에 대한 실험에서 TextureDreamer는 임의의 객체에 매우 사실적이고 의미론적으로 의미 있는 텍스처를 성공적으로 전이하여 이전 최첨단 방법의 시각적 품질을 능가합니다.

**비교 대상 방법들**: Latent-Paint, TEXTure와 비교 수행.

**평가 지표**: CLIP 스코어는 소스와 생성된 텍스처 간의 의미론적·시각적 유사성을 측정하고, SIFID는 단일 이미지 내부 패치 통계를 측정하기 위해 제안된 지표입니다.

---

### 2-5. 한계

TextureDreamer는 과거 텍스처 합성 방법의 평활화, 채화, 3D 일관성 문제를 극복하였으나, 현재의 한계에도 불구하고 현실적인 3D 모델링에 대한 접근성을 확장하고 미래 연구 가능성을 가지고 있습니다.

주요 한계를 정리하면 다음과 같습니다:

1. **계산 비용**: DreamBooth 파인튜닝 + PGSD 최적화의 이중 단계로 인한 높은 계산 비용. ProlificDreamer 또한 생성에 수 시간이 걸릴 정도로, 확산 모델 기반 3D 생성은 이미지 생성보다 훨씬 느립니다.
2. **희소 입력 수렴 어려움**: 희소 입력으로 파인튜닝하면 수렴이 어려워집니다.
3. **단일 객체 텍스처 집중**: 동시 연구들은 주로 전체 3D 객체 재구성에 초점을 맞추는 반면, TextureDreamer는 일치하지 않는 기하를 가진 대상 3D 형상으로의 텍스처 전이에 집중합니다.

---

## 3. 모델의 일반화 성능 향상 가능성

### 3-1. 카테고리 무관(Cross-category) 일반화

TextureDreamer는 소수의 입력 이미지(3~5장)에서 다른 카테고리에 속하는 대상 3D 형상으로도 텍스처를 전이할 수 있는 새로운 방법입니다.

TextureDreamer가 일반화 성능을 확보하는 핵심 메커니즘:

**① 클래스별 사전 보존 손실 제거**
논문은 클래스별 사전 보존 손실을 적용하지 않는데, 저자들은 모델이 다른 카테고리로 일반화되기를 원하기 때문입니다.

$$\mathcal{L}_{\text{DreamBooth}} = \mathcal{L}_{\text{recon}} \quad (\text{prior preservation 제거})$$

**② DreamBooth의 카테고리 독립적 외관 학습**
대안적인 텍스트 인버전 방법과 비교하여, DreamBooth는 더 빠르게 수렴하고 복잡한 텍스처 패턴을 더 잘 보존하는 것으로 나타났으며, 이는 더 큰 용량 때문일 수 있습니다.

**③ ControlNet의 기하 인식 조건화로 형상 독립성 보장**
3D 일관성을 보장하기 위해 TextureDreamer는 텍스처 생성 과정을 대상 메시의 기하에 명시적으로 조건화합니다. 이는 사전 학습된 확산 모델에 공간적 조건화 제어를 추가하는 신경망 아키텍처인 ControlNet을 사용하여 달성됩니다. 3D 메시에서 렌더링된 노멀 맵이 ControlNet에 입력되어 기하와 정렬된 텍스처를 생성하도록 확산 프로세스를 안내합니다.

**④ 사전 학습된 대규모 확산 모델의 일반화 활용**
대규모 텍스트-이미지 쌍 데이터셋으로 사전 학습된 이 모델들은 텍스트 프롬프트에서 고품질의 다양한 이미지를 생성하는 능력으로 주목받아 왔습니다. TextureDreamer는 이 모델들을 활용하여 소수의 이미지에서 텍스처 세부 정보를 외삽합니다.

---

### 3-2. 일반화 성능 한계 및 개선 가능성

| 일반화 차원 | 현재 상태 | 개선 방향 |
|---|---|---|
| 카테고리 | 임의 카테고리 가능 | 더 다양한 카테고리 실험 검증 필요 |
| 입력 이미지 수 | 3~5장 | 1~2장으로 줄이는 연구 |
| 기하 복잡도 | 일반 메시 | 매우 복잡한 위상 구조 개선 필요 |
| 재조명 조건 | 제한적 환경 조명 | 더 다양한 조명 환경 지원 |

---

## 4. 관련 최신 연구 비교 분석 (2020년 이후)

### 4-1. 텍스처 합성 방법 비교 표

| 방법 | 연도/학회 | 입력 | 기하 인식 | 카테고리 제한 | 핵심 기술 |
|---|---|---|---|---|---|
| **TEXTure** | SIGGRAPH 2023 | 텍스트 | △ depth | 없음 | 순차적 inpainting |
| **Text2Tex** | ICCV 2023 | 텍스트 | △ depth | 없음 | 다중 뷰 inpainting |
| **TexFusion** | ICCV 2023 | 텍스트 | 없음 | 없음 | SDS-free 3D 일관성 |
| **SceneTex** | CVPR 2024 | 텍스트 | ✅ depth | 실내 장면 | VSD + 멀티해상도 텍스처 필드 |
| **Paint3D** | CVPR 2024 | 텍스트/이미지 | △ UV | 없음 | Lighting-Less 확산 |
| **TextureDreamer** | **CVPR 2024** | **이미지 3~5장** | **✅ Normal Map** | **없음** | **PGSD = DreamBooth + VSD + ControlNet** |

---

### 4-2. 주요 관련 연구 분석

#### 🔵 TEXTure (SIGGRAPH 2023)
Dreambooth3D와 TEXTure는 희소 뷰에서 정보를 새 텍스트 토큰과 파인튜닝된 확산 모델 가중치로 추출하여 개인화된 3D 객체 또는 보지 못한 객체의 텍스처를 생성하는 데 사용합니다. TextureDreamer는 희소 이미지에서 정보를 추출하는 유사한 방법을 사용하지만, 텍스처 생성을 위한 추출된 정보 활용 방식에서 차이가 나며 일관성과 사실감 개선으로 이어집니다.

#### 🔵 Text2Tex (ICCV 2023)
Text2Tex는 주어진 텍스트 프롬프트로부터 3D 메시의 고품질 텍스처를 생성하는 새로운 방법입니다. 사전 학습된 깊이 인식 이미지 확산 모델에 inpainting을 통합하여 여러 관점에서 부분 텍스처를 점진적으로 합성합니다. 뷰 간의 불일치하고 늘어난 아티팩트 누적을 방지하기 위해 렌더링된 뷰를 동적으로 세그먼트합니다.

#### 🔵 SceneTex (CVPR 2024 Highlight)
SceneTex는 텍스처 합성 작업을 스타일과 기하 일관성이 적절히 반영되는 RGB 공간의 최적화 문제로 수식화합니다. 핵심으로 SceneTex는 메시 외관을 암묵적으로 인코딩하기 위한 멀티해상도 텍스처 필드를 제안합니다. 각각의 RGB 렌더링에서 스코어-증류 기반 목적 함수를 통해 목표 텍스처를 최적화합니다. 뷰 간 스타일 일관성을 더욱 확보하기 위해 크로스-어텐션 디코더를 도입합니다.

#### 🔵 ProlificDreamer / VSD (NeurIPS 2023)
VSD는 3D 파라미터를 확률 변수로 취급하고 그 분포를 추론하는 원칙에 입각한 입자 기반 변분 프레임워크입니다. SDS는 단일점 Dirac 분포를 변분 분포로 사용하는 VSD의 특별한 경우임을 보여줍니다. 이로써 SDS에 의한 3D 장면 생성의 제한된 다양성과 충실도를 설명합니다. 심지어 단일 입자로도 VSD는 매개변수적 스코어 모델을 학습할 수 있어 SDS보다 우월한 일반화를 제공할 가능성이 있습니다.

TextureDreamer는 VSD를 직접 채택하여 텍스처 합성에 적용합니다:

$$\mathcal{L}_{\text{PGSD}} = \mathcal{L}_{\text{VSD}} + \mathcal{L}_{\text{ControlNet(Normal)}} + \mathcal{L}_{\text{DreamBooth}}$$

---

## 5. 앞으로의 연구에 미치는 영향 및 고려할 점

### 5-1. 연구에 미치는 영향

**① 소수샷 텍스처 전이의 새 패러다임**
TextureDreamer는 텍스처 생성 과정을 크게 민주화할 준비가 되어 있습니다. 관련 없는 소수의 이미지 세트에서 고품질 텍스처를 생성하는 능력은 훈련된 전문가의 영역을 넘어 더 넓은 청중에게 세부적이고 현실적인 3D 모델링을 더 접근 가능하게 만들 수 있으며, 3D 그래픽 및 콘텐츠 생성 분야에 변혁을 일으킬 가능성이 있습니다.

**② PGSD 프레임워크의 확장성**
Fantasia3D 등 하이브리드 파이프라인이 기하와 외관을 분리하고 BRDF/재질 표현을 도입하는 반면, TextureDreamer와 DreamMat는 기하 및 조명 인식 확산 목적 함수를 통합하여 재조명 가능한 텍스처와 PBR 재질 추정을 개선합니다.

**③ 관련 후속 연구들에의 영향**
- FabricDiffusion은 단일 의류 이미지에서 임의 형상의 3D 의류로 섬유 텍스처를 전이하는 방법으로, 보지 못한 텍스처와 의류 형상으로 일반화합니다.
- MD-ProjTex, TriTex 등 후속 방법들도 유사한 이미지-투-텍스처 패러다임을 따릅니다.

---

### 5-2. 앞으로 연구 시 고려할 점

#### 🔬 기술적 고려사항

1. **계산 효율화**
   - DreamBooth 파인튜닝 + PGSD 최적화의 이중 비용 감소
   - 단일 피드포워드(feed-forward) 추론으로 단순화 연구
   - 2D 확산 모델을 3D 텍스처링에 적용한 현재 방법들은 긴 처리 시간과 시각적 아티팩트로 어려움을 겪고 있습니다.

2. **더 적은 입력 이미지로 일반화**
   - 현재 3~5장 → 1~2장 또는 단일 이미지 대응으로 확장
   - Zero-shot 텍스처 전이 연구

3. **복잡한 기하 구조 대응**
   - 다중뷰 이미지의 일관성을 높이는 데 어려움이 남아 있으며, 이는 fusion 및 baking 단계에서 아티팩트와 이음새를 유발할 수 있습니다.

4. **재조명 가능성 개선**
   - 현재 BRDF 분리(albedo, metallic, roughness)에서 더 복잡한 조명 모델 지원
   - DreamMat과 FlashTex는 기하와 조명에 조건화하도록 이미지 확산 모델을 파인튜닝하여 조명 및 재질 분리를 개선합니다.

5. **다중 스케일 세밀도 개선**
   - 3D 생성 모델의 최근 발전에도 불구하고 광범위한 일반화와 여러 관점에 걸친 스타일 일관성 유지에 상당한 도전이 남아 있습니다.

#### 🔬 응용 분야별 고려사항

| 응용 분야 | 고려 사항 |
|---|---|
| AR/VR | 실시간 텍스처 생성 속도 개선 |
| 게임/영화 | 아티스트 워크플로우와의 통합 |
| 로보틱스 | 실세계 물체의 빠른 텍스처 인식/재구성 |
| 의류/패션 | FabricDiffusion처럼 특화된 섬유 텍스처 모델링 |

---

## 📚 참고자료 및 출처

| # | 자료 | 출처 |
|---|---|---|
| 1 | **TextureDreamer** (arXiv 원문) | https://arxiv.org/abs/2401.09416 |
| 2 | **TextureDreamer** (CVPR 2024 논문) | https://openaccess.thecvf.com/content/CVPR2024/papers/Yeh_TextureDreamer... |
| 3 | **TextureDreamer** 프로젝트 페이지 | https://texturedreamer.github.io/ |
| 4 | **TextureDreamer** (arXiv HTML 전문) | https://arxiv.org/html/2401.09416v1 |
| 5 | **TextureDreamer** (EmergentMind 분석) | https://www.emergentmind.com/papers/2401.09416 |
| 6 | **TextureDreamer** (Moonlight 리뷰) | https://www.themoonlight.io/en/review/texturedreamer-image-guided-texture-synthesis |
| 7 | **TextureDreamer** (Semantic Scholar) | https://www.semanticscholar.org/paper/TextureDreamer.../2bf4651539a636712fe96f1d82b91aded53d6b03 |
| 8 | **ProlificDreamer** (NeurIPS 2023, VSD 제안) | https://ar5iv.labs.arxiv.org/html/2305.16213 |
| 9 | **ProlificDreamer** (OpenReview) | https://openreview.net/forum?id=ppJuFSOAnM |
| 10 | **SceneTex** (CVPR 2024 Highlight) | https://github.com/daveredrum/scenetex |
| 11 | **Text2Tex** (ICCV 2023) | https://github.com/daveredrum/Text2Tex |
| 12 | **Paint3D** (ResearchGate) | https://www.researchgate.net/publication/384237233_Paint3D... |
| 13 | **CVPR 2024 포스터** | https://cvpr.thecvf.com/virtual/2024/poster/29334 |
| 14 | **HuggingFace 논문 페이지** | https://huggingface.co/papers/2401.09416 |
| 15 | **IEEE Xplore** | https://ieeexplore.ieee.org/iel8/10654794/10654797/10655018.pdf |

> ⚠️ **주의사항**: 본 분석은 공개된 논문 초록, 프로젝트 페이지, 관련 리뷰 자료를 기반으로 작성되었습니다. PGSD의 정확한 수식 전문은 arXiv 원문(https://arxiv.org/pdf/2401.09416)을 직접 확인하시기 바랍니다. 일부 세부 수식의 기호 표기는 논문 내용에 기반하여 표준 표기법으로 재구성한 것입니다.
