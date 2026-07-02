# Unsupervised Video Domain Adaptation with Masked Pre-Training and Collaborative Self-Training

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

UNITE(Unsupervised Adaptation with Teacher-Enhanced Learning)는 **비디오 비지도 도메인 적응(VUDA)** 문제를 해결하기 위해 다음 두 가지를 결합한 최초의 접근법입니다:

1. **마스크 기반 자기지도 사전학습(Masked Self-Supervised Pre-training)**: CLIP 이미지 인코더를 교사(teacher)로 활용하여 타겟 도메인 비디오에서 판별적(discriminative) 특징 학습
2. **협력적 자기훈련(Collaborative Self-Training)**: 비디오 학생 모델과 이미지 교사 모델이 협력하여 정제된 의사 레이블(pseudolabel)을 생성

### 주요 기여

| 기여 항목 | 설명 |
|-----------|------|
| UNITE 파이프라인 제안 | 마스크 비디오 모델링 + 자기훈련의 신규 조합 |
| 3개 벤치마크 평가 | Daily-DA, Sports-DA, UCF↔HMDB_full |
| 최초의 VUDA용 마스크 증류 탐색 | 비디오 도메인 적응에서 최초 적용 |
| 광범위한 Ablation 실험 | 각 단계의 기여도 및 설계 선택 분석 |

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

#### 문제 정의

레이블이 있는 소스 도메인 비디오 $\mathcal{D}\_S := \{(\mathbf{x}^S_i, y^S_i)\}\_{i=1}^{N\_S}$와 레이블이 없는 타겟 도메인 비디오 $\mathcal{D}\_T := \{\mathbf{x}^T_i\}\_{i=1}^{N_T}$가 주어졌을 때, 분포 이동(distribution shift) $\mathcal{P}\_S \neq \mathcal{P}\_T$ 환경에서 타겟 도메인 비디오를 정확히 분류하는 함수 $f_\theta$의 파라미터를 학습하는 것입니다.

#### 핵심 문제점들

```
1. 도메인 갭(Domain Gap): 소스/타겟 도메인 간 시각적 분포 차이
2. 레이블 부재: 타겟 도메인에는 레이블 없음
3. 시공간적 모델링: 비디오는 이미지와 달리 시간 축 정보 필요
4. 의사 레이블 품질: 자기훈련 시 노이즈 레이블 문제
```

---

### 2.2 제안 방법 및 수식

UNITE는 **3단계 파이프라인**으로 구성됩니다.

#### 📌 Stage 1: 비지도 타겟 도메인 사전학습 (Unsupervised Target Domain Pre-Training)

UMT(Unmasked Teacher) 목적함수를 사용합니다. 학생 모델 $g_a$는 마스킹된 비디오 $m(\mathbf{x}^T)$를 입력으로 받고, 교사 모델 $g_*$(CLIP ViT-B)는 비마스킹 프레임을 처리합니다.

$$\mathcal{L}_{\text{UMT}} = \mathbb{E}_{\mathbf{x} \sim \mathcal{P}_T} \left[ \frac{1}{|\mathcal{A}|} \sum_{l \in \mathcal{A}} \text{MSE}\left( d^l(\mathbf{z}^l_a), \mathbf{z}^l_* \right) \right]$$

- $\mathbf{z}^l_a$: 학생 모델의 $l$번째 레이어 L2 정규화 표현 (마스킹 입력)
- $\mathbf{z}^l_*$: 교사 모델의 $l$번째 레이어 표현 (가시 패치 위치만)
- $\mathcal{A}$: 정렬 레이어 집합 (마지막 6개 레이어)
- $d^l(\cdot)$: $l$번째 레이어용 선형 사영(projection)
- 마스킹 비율 $r = 0.8$ (어텐션 기반 마스킹)

> **핵심 아이디어**: 교사는 완전한 프레임을, 학생은 80% 마스킹된 프레임을 처리 → 학생이 가시 패치로부터 풍부한 시공간 표현을 학습하도록 강제

#### 📌 Stage 2: 소스 도메인 파인튜닝 (Source Domain Fine-Tuning)

$$\mathcal{L}_{\text{SFT}} = \mathbb{E}_{(\mathbf{x}^S, y^S) \sim \mathcal{P}_S} \left[ \mathcal{L}_{CE}(f_a(\mathbf{x}^S), y^S) \right] \tag{1}$$

- 선형 분류 헤드 $h_a$를 추가: $f_a = h_a(g_a(\cdot))$
- 레이어별 학습률 감쇠(layer-wise LR decay = 0.65) 적용 → Stage 1에서 학습한 타겟 도메인 특징 보존
- 마스킹 없이 소스 비디오에 대해 전체 파인튜닝 수행

#### 📌 Stage 3: 협력적 자기훈련 (Collaborative Self-Training, CST)

**기존 자기훈련(ST) 손실함수:**

$$\mathcal{L}_{\text{ST}} = \mathbb{E}_{(\mathbf{x}^S, y^S) \sim \mathcal{P}_S} \left[ \mathcal{L}_{CE}(f_a(\mathbf{x}^S), y^S) \right] + \lambda \mathbb{E}_{\mathbf{x}^T \sim \mathcal{P}_T} \left[ s(\mathbf{x}^T)\mathcal{L}_{CE}(f_a(\mathbf{x}^T), \hat{y}_a(\mathbf{x}^T)) \right] \tag{2}$$

**MatchOrConf 의사 레이블 생성 방식:**

$$\tilde{y} = \begin{cases} \hat{y}_a & \text{if } \hat{y}_a = \hat{y}_* \\ \hat{y}_a & \text{if } \hat{y}_a \neq \hat{y}_* \text{ and } \text{Conf}(\hat{y}_a) > \gamma \text{ and } \text{Conf}(\hat{y}_*) \leq \gamma \\ \hat{y}_* & \text{if } \hat{y}_a \neq \hat{y}_* \text{ and } \text{Conf}(\hat{y}_a) \leq \gamma \text{ and } \text{Conf}(\hat{y}_*) > \gamma \\ -1 & \text{otherwise} \end{cases} \tag{3}$$

**샘플 선택 마스크:**

$$s(\mathbf{x}^T) = \begin{cases} 1 & \text{if } \tilde{y}(\mathbf{x}^T) \neq -1 \\ 0 & \text{otherwise} \end{cases} \tag{4}$$

**신뢰도 기반 손실 가중치:**

$$q(\mathbf{x}^T) = \text{Conf}(\hat{y}_a(\mathbf{x}^T)) \tag{5}$$

**CST 최종 손실함수:**

$$\mathcal{L}_{\text{CST}} = \mathbb{E}_{(\mathbf{x}^S, y^S) \sim \mathcal{P}_S} \left[ \mathcal{L}_{CE}(f_a(\mathbf{x}^S), y^S) \right] + \lambda \mathbb{E}_{(\mathbf{x}^T) \sim \mathcal{P}_T} \left[ s(\mathbf{x}^T)q(\mathbf{x}^T)\mathcal{L}_{CE}(f_a(m(\mathbf{x}^T)), \tilde{y}(\mathbf{x}^T)) \right] \tag{6}$$

> **핵심**: 타겟 CE 손실을 **마스킹된** 비디오에 적용 → 더 강인한 타겟 도메인 인식 유도

---

### 2.3 모델 구조

```
┌─────────────────────────────────────────────────────────────────┐
│                        UNITE 전체 구조                           │
├─────────────────┬───────────────────┬───────────────────────────┤
│    Stage 1      │     Stage 2       │         Stage 3           │
│  (비지도 사전학습) │  (소스 파인튜닝)  │    (협력적 자기훈련)        │
├─────────────────┼───────────────────┼───────────────────────────┤
│ 입력: 타겟 비디오 │ 입력: 소스 비디오  │ 입력: 소스+타겟 비디오      │
│ (마스킹 80%)    │ (레이블 있음)      │ (타겟: 마스킹 80%)         │
│                 │                   │                           │
│ 교사: CLIP ViT-B│ 학생 $g_a$ + $h_a$│ 교사: CLIP (Zero-Shot)    │
│ (Frozen)        │ CE Loss 최소화     │ 학생: $f_a$               │
│                 │                   │ MatchOrConf 의사레이블     │
│ UMT Loss:       │ Layer-wise LR     │ 마스킹 CE + 소스 CE        │
│ MSE(d(z_a),z_*) │ decay = 0.65      │ 신뢰도 가중치 q 적용        │
└─────────────────┴───────────────────┴───────────────────────────┘

네트워크 아키텍처: ViT-B/16 (87M 파라미터)
- 패치 임베딩: 각 프레임 독립적 처리
- Self-Attention: 공간 + 시간 차원 동시 처리
- 분류: 최종 레이어 패치 표현의 평균 풀링
- 프레임 샘플링: T=8 균일 세그먼트
```

---

### 2.4 성능 향상

#### Daily-DA 결과

| 방법 | 평균 정확도 |
|------|------------|
| DANN | 29.5% |
| TA³N | 28.5% |
| DALL-V (SFVUDA) | 51.4% |
| UNITE w/o CST | 50.8% |
| **UNITE (Ours)** | **59.2%** |

#### Sports-DA 결과

| 방법 | 평균 정확도 |
|------|------------|
| DANN | 73.8% |
| TA³N | 73.7% |
| DALL-V (SFVUDA) | 82.3% |
| UNITE w/o CST | 88.9% |
| **UNITE (Ours)** | **94.0%** |

#### UCF↔HMDB_full 결과

| 방법 | 평균 정확도 |
|------|------------|
| CO²A | 91.8% |
| UDAVT | **94.6%** |
| **UNITE (Ours)** | 93.8% |

> **주목할 점**: UNITE는 U→H에서 95.0%로 SOTA 달성, H→U에서는 UDAVT에 소폭 미치지 못함

#### Ablation Study 핵심 결과 (ARID→HMDB)

| 구성 | H→A | A→H |
|------|-----|-----|
| Source Only | 40.4% | 49.6% |
| + UMT Pre-Training | 43.8% | 51.7% |
| + CST만 | 42.0% | 60.8% |
| **UNITE (둘 다)** | **48.0%** | **67.9%** |

Stage 1과 Stage 3의 시너지 효과: A→H에서 각각 +1.8%, +11.2% → 합산 **+18.3%** (단순 합 이상)

---

### 2.5 한계점

1. **단일 도메인 사전학습의 필요성**: 소스+타겟 혼합 사전학습이 이미지 기반 UDA에서는 효과적이지만 VUDA에서는 타겟만 사용하는 것이 더 효과적 → 이유에 대한 이론적 근거 부족

2. **신뢰도 임계값 $\gamma$의 민감성**: 도메인 이동마다 최적 $\gamma$ 값이 다르며, 자동화된 원칙적 방법 없음

3. **특정 클래스 성능 저하**: 자기훈련 과정에서 잘 수행되는 클래스를 강화하는 반면 어려운 클래스('pick', 'run')는 성능 감소 발생

4. **계산 비용**: 3단계 파이프라인으로 훈련 시간 증가 (Stage 2: ~6시간, Stage 3: ~6시간, Stage 1: ~1.5시간 on 4×NVIDIA A5000)

5. **UCF↔HMDB에서의 한계**: H→U 방향에서 UDAVT 대비 낮은 성능

6. **이미지-비디오 모달리티 갭**: CLIP은 이미지 기반으로 시간적 추론 능력에 한계 존재

---

## 3. 모델 일반화 성능 향상 가능성

### 3.1 일반화를 높이는 핵심 메커니즘

#### (1) 마스킹 기반 불변성 학습

마스크 기반 사전학습은 본질적으로 **입력의 일부가 마스킹되어도 동일한 표현을 생성**하는 불변성(invariance)을 강제합니다:

$$\forall m: g_a(m(\mathbf{x})) \approx g_*({\mathbf{x}}) \quad \text{(의미적으로 유사한 표현)}$$

이러한 불변성은 타겟 도메인의 다양한 시각적 변형에 대한 강인성을 높입니다. 실험에서 마스킹 적용 시 A→H에서 54.6% → 67.9%로 대폭 향상됩니다.

#### (2) 교사-학생 협력으로 인한 다양성 활용

- **CLIP 교사**: 강력한 이미지 의미 이해 (공간적 강점)
- **비디오 학생**: 시공간 패턴 포착 (시간적 강점)
- 두 모델의 **상호보완적 특성**을 MatchOrConf로 결합하여 단일 모델보다 높은 품질의 의사 레이블 생성

```
예시 (ARID→HMDB):
- 'jump' 클래스: 이미지 교사가 더 우수
- 'wave' 클래스: 비디오 학생이 더 우수
→ 협력으로 두 클래스 모두 Stage 2 학생 및 교사 초과 성능 달성
```

#### (3) 자기지도 사전학습의 일반화 우위

논문에서는 지도 Kinetics-400 사전학습 대신 **자기지도 UMT K710 사전학습**을 선택합니다:

> "When the pre-trained network has been trained to classify categories present in the DA dataset, some DA techniques could perform well simply by preserving the capabilities of the pre-trained model despite not generalizing well."

이는 UDA 연구의 신뢰성과 실제 일반화 능력을 구분하는 중요한 선택입니다. Daily-DA와 Sports-DA의 8개 클래스 중 6개가 Kinetics-400에 포함되어 있으므로, 지도 사전학습은 도메인 적응 능력이 아닌 사전 지식을 단순 보존하는 방식으로 동작할 위험이 있습니다.

#### (4) 소스 도메인 손실의 정규화 효과

CST 단계에서 소스 CE 손실을 포함하는 것이 훈련 안정성에 결정적:

| 설정 | H→A | A→H |
|------|-----|-----|
| 타겟 CE만 | 33.7% | 42.1% |
| **타겟 + 소스 CE** | **48.0%** | **67.9%** |

소스 레이블 데이터가 초기에 부정확한 의사 레이블의 영향을 완화하는 정규화 역할을 합니다.

#### (5) 레이어별 학습률 감쇠(Layer-wise LR Decay)

Stage 1에서 학습된 타겟 도메인 특징을 Stage 2에서 보존하기 위해 레이어별 LR 감쇠(0.65)를 적용합니다. 이는 하위 레이어(범용 특징)를 보존하면서 상위 레이어(과제 특화 특징)를 업데이트하는 전략으로 일반화 성능을 높입니다.

### 3.2 일반화 성능의 잠재적 확장 가능성

| 확장 방향 | 현재 UNITE | 개선 가능성 |
|-----------|-----------|------------|
| 더 강력한 교사 모델 | CLIP ViT-B | CLIP ViT-L, InternVideo 등 |
| 멀티소스 도메인 | 단일 소스 | 다중 소스 통합 |
| 클래스 불균형 처리 | MatchOrConf | FlexMatch류 커리큘럼 |
| 임계값 자동화 | 고정 $\gamma=0.1$ | 도메인별 적응적 임계값 |

---

## 4. 향후 연구에 미치는 영향 및 고려 사항

### 4.1 향후 연구에 미치는 영향

#### (A) 마스크 기반 비디오 도메인 적응의 새 패러다임

UNITE는 VUDA에서 **마스크 증류(Masked Distillation)를 처음으로 탐색**함으로써, 이후 연구들이 다음을 탐구할 수 있는 기반을 마련합니다:
- 다양한 마스킹 전략(무작위 마스킹, 튜브 마스킹 등)의 VUDA 효과
- 마스킹 비율의 도메인별 최적화
- 픽셀 수준 vs. 특징 수준 재구성 비교

#### (B) 이미지-비디오 교차 모달리티 학습의 가능성 제시

CLIP을 공간 교사로 활용하여 시공간 학생 모델을 훈련하는 접근법은:
- 대규모 이미지-텍스트 모델의 비디오 이해 전이 연구를 촉진
- 비디오 전용 대규모 사전학습 데이터가 없는 상황에서도 강력한 비디오 표현 학습 가능성 시사

#### (C) 협력적 의사 레이블링 프레임워크

MatchOrConf 방식의 다모델 협력 의사 레이블링은:
- 반지도 학습(Semi-supervised Learning) 분야로 확장 가능
- 다중 교사 모델(Multiple Teachers) 앙상블 연구로 발전 가능

#### (D) 자기지도 초기화의 중요성 재조명

지도 사전학습 대신 자기지도 초기화 사용의 당위성을 실험적으로 보여줌으로써:
- UDA 벤치마크 평가 기준의 재정립 필요성 제기
- 공정한 비교를 위한 사전학습 프로토콜 표준화 논의 촉발

---

### 4.2 향후 연구 시 고려할 사항

#### ⚠️ 기술적 고려사항

**1. 임계값 $\gamma$ 자동화**

현재 $\gamma = 0.1$로 고정되어 있으나 도메인별 최적값이 다릅니다. 적응적 임계값 방법이 필요합니다:

$$\gamma^* = \arg\min_\gamma \mathbb{E}_{\mathbf{x}^T}\left[\mathcal{L}_{val}(f_a(\mathbf{x}^T), \tilde{y}_{\gamma}(\mathbf{x}^T))\right]$$

FlexMatch[68] 방식의 클래스별 커리큘럼 임계값 적용을 고려할 수 있습니다.

**2. 클래스 불균형 문제**

자기훈련이 쉬운 클래스는 강화하고 어려운 클래스는 소홀히 하는 경향이 있습니다. 이를 해결하기 위해:

$$\mathcal{L}_{\text{CST}}^{balanced} = \mathcal{L}_{\text{CST}} + \beta \cdot \mathcal{L}_{div}$$

여기서 $\mathcal{L}_{div}$는 예측 다양성을 장려하는 정규화 항입니다.

**3. 소스-타겟 배치 비율**

현재 20:20 고정 비율 사용. 동적 비율 조정이 성능을 개선할 수 있습니다.

**4. 교사 모델의 시간적 한계 극복**

CLIP은 이미지 기반으로 시간적 추론이 없습니다. 향후 연구에서:
- InternVideo, VideoMAE 등 비디오 기반 교사 모델 활용
- 다중 교사(이미지 + 비디오) 앙상블 방식

**5. 더 긴 비디오 처리**

현재 T=8 프레임으로 제한. 더 긴 시간적 컨텍스트가 필요한 도메인(예: 복잡한 스포츠 동작)에서 성능 한계 가능성이 있습니다.

#### ⚠️ 방법론적 고려사항

**6. 소스-프리 설정으로의 확장**

현재 UNITE는 소스 데이터를 Stage 2와 Stage 3에서 사용합니다. 소스 데이터가 없는 SFVUDA 설정으로의 확장 연구가 필요합니다.

**7. 다중 소스 도메인 처리**

Daily-DA와 같이 여러 소스 도메인이 존재하는 경우, 소스별 가중치 적응 메커니즘이 필요합니다.

**8. 공정한 비교 기준 확립**

논문에서 지적하듯, 지도 Kinetics 사전학습과 자기지도 사전학습의 성능 차이가 크므로 (Daily-DA: 49.6% vs 63.2%), 향후 연구들은 동일한 사전학습 프로토콜 하에서 비교해야 합니다.

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

### 5.1 방법론 계보

```
DANN (2016) → MK-MMD (2015) → TA³N (ICCV 2019)
     ↓ (적대적 정렬 패러다임)
CoMix (NeurIPS 2021) → CO²A (WACV 2022) → UDAVT (ICPR 2022)
     ↓ (대조 학습 패러다임)
ATCoN (ECCV 2022) → EXTERN (2022) → DALL-V (ICCV 2023)
     ↓ (소스프리 + 대규모 모델 패러다임)
UNITE (2023, arXiv 2312.02914) ← 마스크 사전학습 + 협력 자기훈련
```

### 5.2 주요 연구 비교

| 방법 | 연도 | 주요 기법 | Sports-DA Avg | UCF↔HMDB Avg | 한계 |
|------|------|-----------|--------------|--------------|------|
| TA³N | ICCV 2019 | 시공간 적대적 정렬 | 73.7% | 81.5% | 불안정한 훈련 |
| CoMix | NeurIPS 2021 | 대조학습 + 배경 혼합 | - | - | 복잡한 증강 필요 |
| CO²A | WACV 2022 | 6개 손실 동시 최적화 | - | 91.8% | 최적화 복잡성 |
| UDAVT | ICPR 2022 | 정보 병목 + ViT | - | **94.6%** | H→U 제한적 |
| ATCoN | ECCV 2022 | 시간 일관성 (소스프리) | 73.8% | 82.5% | 소스프리 한계 |
| EXTERN | 2022 | 마스킹 + 시간 정규화 | 83.2% | 90.4% | 소스프리 |
| DALL-V | ICCV 2023 | CLIP + 어댑터 (소스프리) | 82.3% | 91.0% | 이미지 모델 한계 |
| **UNITE** | **2023** | **마스크 증류 + CST** | **94.0%** | 93.8% | 3단계 훈련 비용 |

### 5.3 패러다임 변화 분석

#### 패러다임 1: 적대적 정렬 (2016~2020)
- **대표**: DANN, MK-MMD, TA³N
- **원리**: 도메인 판별자를 속이는 방식으로 특징 정렬
- **단점**: 경쟁적 목적함수로 인한 불안정한 훈련

#### 패러다임 2: 대조 학습 기반 (2021~2022)
- **대표**: CoMix, CO²A, UDAVT
- **원리**: 도메인 간 특징 구조의 대조적 정렬
- **단점**: 다수의 손실함수 동시 최적화의 복잡성

#### 패러다임 3: 대규모 사전학습 모델 활용 (2022~현재)
- **대표**: DALL-V, EXTERN, **UNITE**
- **원리**: CLIP 등 대규모 모델의 강력한 표현 활용
- **특징**: 소스프리 또는 마스킹 기반 접근

#### UNITE의 차별화 포인트

```
기존 연구들의 한계:
├── 적대적 방법: 훈련 불안정
├── 대조 학습: 복잡한 다중 손실
├── 소스프리 방법: 소스 정보 미활용
└── 순수 CLIP 방법: 시간적 모델링 부재

UNITE의 해결책:
├── 마스크 증류: 안정적 자기지도 학습
├── 단순한 손실: UMT + CE (2종)
├── UDA 설정: 소스+타겟 모두 활용
└── 교사-학생: 공간+시간 상호보완
```

---

## 참고 자료

**주요 참고 논문 (논문 내 인용 기준):**

1. **[본 논문]** Reddy et al., "Unsupervised Video Domain Adaptation with Masked Pre-Training and Collaborative Self-Training," arXiv:2312.02914v5, 2025. (GitHub: https://github.com/reddyav1/unite)

2. **[UMT]** Li et al., "Unmasked Teacher: Towards Training-Efficient Video Foundation Models," ICCV 2023.

3. **[CLIP]** Radford et al., "Learning Transferable Visual Models From Natural Language Supervision," ICML 2021.

4. **[DALL-V]** Zara et al., "The Unreasonable Effectiveness of Large Language-Vision Models for Source-Free Video Domain Adaptation," ICCV 2023.

5. **[PACMAC]** Prabhu et al., "Adapting self-supervised vision transformers by probing attention-conditioned masking consistency," NeurIPS 2022.

6. **[MIC]** Hoyer et al., "MIC: Masked Image Consistency for Context-Enhanced Domain Adaptation," CVPR 2023.

7. **[MatchOrConf]** Zhang et al., "Rethinking the Role of Pre-Trained Networks in Source-Free Domain Adaptation," CVPR 2023.

8. **[TA³N]** Chen et al., "Temporal Attentive Alignment for Large-Scale Video Domain Adaptation," ICCV 2019.

9. **[UDAVT]** Da Costa et al., "Unsupervised Domain Adaptation for Video Transformers in Action Recognition," ICPR 2022.

10. **[CO²A]** Turrisi da Costa et al., "Dual-Head Contrastive Domain Adaptation for Video Action Recognition," WACV 2022.

11. **[VideoMAE]** Tong et al., "VideoMAE: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training," NeurIPS 2022.

12. **[MAE]** He et al., "Masked Autoencoders Are Scalable Vision Learners," CVPR 2021.

13. **[FlexMatch]** Zhang et al., "Flexmatch: Boosting semi-supervised learning with curriculum pseudo labeling," NeurIPS 2021.

14. **[ATCoN]** Xu et al., "Source-Free Video Domain Adaptation by Learning Temporal Consistency for Action Recognition," ECCV 2022.

15. **[Daily-DA/Sports-DA]** Xu et al., "Multi-Source Video Domain Adaptation With Temporal Attentive Moment Alignment Network," IEEE TCSVT 2023.
