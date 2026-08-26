# Diagnosing Compositional Generalization in Sequential Robot Tasks
---

## 1. Executive Summary (10문장 이내)

이 논문은 순차적 로봇 조작 태스크에서 **조합적 일반화(Compositional Generalization)** 문제를 데이터 커버리지 관점에서 체계적으로 분석한다.  
로봇이 훈련 중 보지 못한 명령어 조합(OOD)을 수행해야 할 때, 가능한 모든 명령어 튜플을 수집하는 것은 조합론적으로 비현실적이다.  
저자들은 일반화 격차(Generalization Gap)를 세 가지 원인으로 분해한다:  
**한계 명령어 이동(Marginal Instruction Shift)**, **명령어-조합적 이동(Instruction-Compositional Shift)**, **문맥-행동 이동(Context-Action Shift)**.  
이 분해를 통해 어떤 훈련 커버리지가 실제로 필요한지 진단할 수 있다.  
실험 결과, 전체 태스크 공간의 1/4에 해당하는 구조화된 직교 태스크 집합이 전체 커버리지와 유사한 OOD 성능을 달성한다.  
희소 훈련 실패의 원인은 저수준 기술 부재가 아니라 **명령어 스티어링(Instruction Steering)** 능력의 부재임을 실험으로 확인한다.  
태스크당 단 1개의 데모 파인튜닝만으로 OOD 성공률이 0.4%에서 54.7%로 급증한다. 의미적으로 종속된 태스크의 경우, 인수 다양성보다 **관계 구조(Relational Structure)** 커버리지가 핵심임을 보인다.  
이 연구는 로봇 데이터 수집 전략을 태스크 확장보다 **의존성 커버리지 우선**으로 전환해야 함을 시사한다.

> 💡 **용어 설명:**
> - **조합적 일반화(Compositional Generalization):** 개별 구성 요소는 알고 있지만, 그 새로운 조합은 학습하지 못한 상황에서도 올바르게 동작하는 능력
> - **OOD(Out-of-Distribution):** 훈련 데이터 분포 밖의 입력에 대한 일반화 능력
> - **명령어 스티어링(Instruction Steering):** 주어진 복합 명령어에 따라 적절한 서브태스크 행동을 선택·유도하는 능력

### 1-1. 연구의 목적과 필요성

| 구분 | 내용 |
|------|------|
| **문제 상황** | 순차적 로봇 조작에서 가능한 모든 명령어 조합에 대한 데이터 수집은 조합론적으로 불가능 (태스크 수가 지수적으로 증가) |
| **기존 접근의 한계** | 계층적 파이프라인 및 end-to-end 정책 모두 훈련-테스트 명령어가 겹칠 때만 성능이 보장됨 |
| **핵심 질문** | 조합적 일반화를 위해 명령어 공간의 어떤 커버리지가 실제로 필요한가? |
| **필요성** | 실제 배포 환경에서 로봇은 훈련 중 보지 못한 명령어 재조합을 수행해야 하므로, 데이터 수집 원칙의 이론적 근거 필요 |

---

## 2. 핵심 주장과 근거 표

| 핵심 주장 | 근거 | 위치 |
|-----------|------|-------|
| 전체 튜플 열거는 불필요하다 | 직교 16-태스크 집합(전체 64개의 1/4)이 78.2% OOD 성공률 달성 | p.6, Figure 2, Figure 3 |
| 희소 훈련 실패는 기술 부재가 아닌 명령어 스티어링 부재 | 파인튜닝 1개 데모/태스크 → OOD 0.4%→54.7% | p.7, Figure 5 |
| 의존적 명령어에는 관계 구조 커버리지가 필요 | 더 많은 (o₁, c₂) 쌍 관측 시 OOD 성공률 및 위반율 개선 | p.7-8, Figure 6 |
| 쌍별 커버리지 수가 OOD 성공률을 예측 | Seen pair count r=0.82 상관관계 | p.5, Figure 2c |
| 직교 설계가 데이터 제한 환경에서 우월 | 16/32 데모 환경에서 랜덤 커버리지 대비 직교 집합 우위 | p.16, Figure 11 |

### 2-1. 상세 분석

#### ① 해결하고자 하는 문제

- **명령어 공간의 조합 폭발 문제:** $n$개의 서브태스크, 각 $m+1$개의 값 → $|\mathcal{L}| = (m+1)^n$ 개 조합
- **핵심 질문:** 어떤 $E_{\text{train}} \subset E_{\text{total}}$이 OOD 일반화에 충분한가?

---

#### ② 제안하는 방법 및 수식

**[수식 1] 훈련 지지집합 (Training Support)**

$$E_{\text{train}} := \text{supp}(p(l)) = \{l \in \mathcal{L}_1 \times \cdots \times \mathcal{L}_n \mid p(l) > 0\}$$

- $l = (l_1, \ldots, l_n)$: 복합 명령어 튜플
- $\mathcal{L}_j = \{0, \ldots, m\}$: $j$번째 서브태스크의 명령어 집합
- $p(l)$: 훈련 데이터의 명령어 분포

> 💡 **supp(지지집합, Support):** 확률분포에서 확률이 0보다 큰 영역, 즉 훈련 데이터에 실제로 등장한 명령어 조합의 집합

**[수식 2] OOD 집합**

$$E_{\text{ood}} := E_{\text{total}} \setminus E_{\text{train}}$$

- $E_{\text{total}} := \text{supp}(q)$: 테스트 시 가능한 모든 명령어 조합
- $q(l)$: 테스트 명령어 분포

**[수식 3] 기대 위험 (Expected Risk)**

$$\mathcal{R}_{\mathcal{D}}(\theta) := \int_{\mathcal{Z} \times \mathcal{A} \times \mathcal{L}} \mathcal{D}(z, a, l) L_\theta(z, a, l) \, dz \, da \, dl$$

- $\mathcal{D} \in \{p, q\}$: 훈련 또는 테스트 분포
- $z \in \mathcal{Z}$: 시간 $t$에서의 문맥 (관측값, 로봇 상태)
- $a \in \mathcal{A}$: 로봇 행동
- $L_\theta(z, a, l)$: 파라미터 $\theta$를 가진 정책의 손실 함수

**[수식 4] 조합적 일반화 격차 (Compositional Generalization Gap)**

$$\Delta_q(\theta) := \mathcal{R}_q(\theta) - \mathcal{R}_p(\theta) = \int_{\mathcal{Z} \times \mathcal{A} \times \mathcal{L}} \left(q(z, a, l) - p(z, a, l)\right) L_\theta(z, a, l) \, dz \, da \, dl$$

**[수식 5] 일반화 격차의 상한 분해 (Proposition 4.1 / A.1)** ← *본 논문의 핵심 이론*

$$|\Delta_q(\theta)| \leq M \left( \|q(l_i) - p(l_i)\|_1 + \mathbb{E}_{l_i \sim p(l_i)}\left[\|q(l_{-i} \mid l_i) - p(l_{-i} \mid l_i)\|_1\right] + \mathbb{E}_{l \sim p(l)}\left[\|q(z, a \mid l) - p(z, a \mid l)\|_1\right] \right)$$

각 기호 설명:

| 기호 | 의미 |
|------|------|
| $M$ | 손실 함수의 균일 상한: $0 \leq L_\theta \leq M$ |
| $l_i$ | $i$번째 서브태스크 명령어 (현재 활성 서브태스크) |
| $l_{-i}$ | $l_i$를 제외한 나머지 서브태스크 명령어들 |
| $\|q(l_i) - p(l_i)\|_1$ | **한계 명령어 이동**: 개별 서브태스크 명령어의 훈련-테스트 분포 차이 |
| $\mathbb{E}\_{l_i \sim p(l_i)}[\|q(l_{-i} \mid l_i) - p(l_{-i} \mid l_i)\|_1]$ | **명령어-조합적 이동**: 알려진 서브태스크들이 새로운 방식으로 조합될 때 발생 |
| $\mathbb{E}_{l \sim p(l)}[\|q(z, a \mid l) - p(z, a \mid l)\|_1]$ | **문맥-행동 이동**: 동일한 전체 명령어에서의 맥락-행동 분포 차이 |
| $\|\mu - \nu\|\_1 = \int_\mathcal{X} \|\mu(x) - \nu(x)\| dx$ | $L^1$ 노름 (두 분포의 총변동 거리) |

> 💡 **$L^1$ 노름(Total Variation Distance):** 두 확률분포 사이의 거리 측정 방법. 값이 클수록 두 분포 차이가 큼

**[수식 6] 단계별 모듈형 정책 (Stage-wise Modular Policy)**

$$\pi_\theta(a \mid z, l) = \pi_{\theta, \sigma(z)}(a \mid z, l_{\sigma(z)})$$

- $\sigma: \mathcal{Z} \to \{1, \ldots, n\}$: 현재 맥락 $z$에서 활성 서브태스크 단계를 결정하는 함수
- 즉, 정책은 현재 활성 서브태스크 명령어 $l_{\sigma(z)}$에만 조건화됨

**[수식 7] 모듈형 정책의 손실 분해**

$$L_\theta(z, a, l) = \sum_{i=1}^n \mathbf{1}\{\sigma(z) = i\} \, \ell_{i,\theta}(z, a, l_i)$$

- $\mathbf{1}\{\sigma(z) = i\}$: 현재 단계 $i$인지를 나타내는 지시 함수
- $\ell_{i,\theta}(z, a, l_i)$: 단계 $i$에서의 손실

**[수식 8] 모듈형 정책의 일반화 격차 상한 (Corollary 4.2)**

$$|\Delta_q^m(\theta)| \leq \sum_{i=1}^n M_i \left( \|q(l_i) - p(l_i)\|_1 + \mathbb{E}_{l_i \sim p}\|q(z, a \mid l_i) - p(z, a \mid l_i)\|_1 \right)$$

- 일반 정책(수식 5) 대비 **명령어-조합적 이동 항이 제거됨** → 더 작은 상한 제공
- $M_i$: 단계 $i$에서의 손실 상한

> 💡 **단계별 모듈형 정책(Stage-wise Modular Policy):** 각 서브태스크 단계를 독립적으로 처리하는 계층적 정책. SayCan, π0.5, Gemini Robotics 등이 이 구조를 채택

---

#### ③ 모델 구조 (Figure 7, Appendix B)

```
입력: 이미지 + 명령어 텍스트
    ↓
[DINOv2 이미지 인코더] (frozen)
    → image tokens (K, V)
    
[명령어 토큰] (각 서브태스크 = 하나의 토큰)
    → query tokens (Q)
    ↓
[Q-Former × N층]
: 명령어 토큰이 Q, 이미지 토큰이 K,V
    → output tokens
    ↓
[Flow-Matching Action Decoder × M층]
: proprioception 정보 + cross-attention
    → 연속 행동 청크 (clean actions a₀)
```

> 💡 **DINOv2:** Meta에서 개발한 자기지도학습(Self-supervised) 기반 시각 인코더로, 레이블 없이 강력한 시각 특징을 학습
>
> 💡 **Q-Former:** BLIP-2에서 도입된 구조로, 고정된 이미지 인코더와 언어 모델을 연결하는 경량 트랜스포머
>
> 💡 **Flow-Matching:** 확률 흐름을 학습하여 행동을 생성하는 생성 모델 계열. Diffusion Policy와 유사하지만 더 효율적인 훈련 가능

---

#### ④ 성능 향상 및 한계

**성능 향상:**

| 실험 | 결과 |
|------|------|
| PP 태스크 포화 지점 | $B = 8$ (전체 16개의 50%) |
| PPP 직교 집합 | 전체의 1/4로 78.2% OOD 성공률 달성 |
| 파인튜닝 효과 | OOD 0.4% → 54.7% (태스크당 1개 데모) |
| 2S-PP 쌍 커버리지 | Cov12 > Cov9 > Cov6 > Cov4 순으로 OOD 향상 |

**한계:**
- 소규모 순차적 조작 태스크에만 실험 제한 (명령어 복잡도 증가 시 기하급수적 비용)
- 동결된 비전 인코더 위에 처음부터 훈련하는 정책만 분석 (사전학습된 VLA 모델 미분석)

---

## 3. 각 주장별 페이지/Figure 번호

| 주장 | 위치 |
|------|------|
| 세 가지 일반화 격차 분해 (이론) | p.3-4, Proposition 4.1, 수식 (5) |
| 모듈형 정책의 이점 및 한계 | p.4, Corollary 4.2, 수식 (6) |
| 전체 튜플 커버리지 불필요 | p.5-6, Figure 2(a)(b), Figure 3 |
| Seen pair count와 OOD 성공률 상관관계 | p.6, Figure 2(c), r=0.82 |
| 직교 집합의 효율성 | p.6, Figure 3, Figure 8(a) |
| 파인튜닝으로 스티어링 능력 획득 | p.7, Figure 5, Figure 9 |
| 의존적 명령어 쌍 커버리지 필요성 | p.7-8, Figure 6, Figure 12, Figure 13 |
| 정책 네트워크 구조 | Appendix B, Figure 7 |

---

## 4. 저자 보고 결과 vs. 나의 해석

### 저자가 직접 보고한 결과

**연구 주제 (저자 직접 기술):**
> "This paper studies compositional generalization through the lens of instruction-space coverage." (p.1)

**방법 (저자 직접 기술):**
- 일반화 격차를 3항으로 분해 (수식 5, p.3-4)
- 직교 배열 설계(Orthogonal Array Design) 적용 (p.6)
- 희소 사전훈련 후 광범위 파인튜닝 실험 (p.7)

**수치 결과 (저자 직접 보고):**
- PP: $B=8$에서 포화, PPP: $B=16$에서 포화 (Figure 2, p.6)
- 직교 집합(16태스크): OOD 78.2% (전체 64태스크 수준) (p.6)
- 파인튜닝: OOD 0.4% → 54.7% (태스크당 1 데모) (p.7)
- Seen pair count 상관계수 $r = 0.82$ (Figure 2c, p.5)

---

### 나의 해석 (저자 보고와 분리)

1. **이론과 실험의 연결 강도:** 수식 (5)는 상한(upper bound)으로, 실제로 각 항의 크기를 직접 측정하지 않는다. 저자들은 이 분해를 실험적 관찰의 사후 해석에 활용하고 있으며, 인과 관계의 직접 증명보다는 상관 관계 분석에 가깝다.

2. **직교 설계의 일반화 가능성:** 직교 설계가 PPP(3-인수 태스크)에서 효과적이지만, 더 많은 인수나 비균등한 인수 공간에서도 동일하게 효과적인지는 논문에서 직접 확인되지 않는다.

3. **파인튜닝 결과의 해석:** OOD가 0.4% → 54.7%로 향상된 것은 인상적이지만, ID 성능이 동시에 감소한다(p.7). 이는 단순한 OOD 개선이 아니라 훈련 분포의 재가중화(distributional reweighting)로 인한 트레이드오프일 수 있다.

4. **$r=0.82$ 상관계수의 한계:** Seen pair count와 OOD 성공률 간의 상관은 설명 변수를 제안하지만, 인수 간 의존성 구조나 데모 수 편향 등 교란 변수가 통제되지 않는다.

---

## 5. 통계적으로 취약한 부분과 비교 불가능한 수치

| 항목 | 문제점 |
|------|--------|
| ⚠️ **$r = 0.82$ (Figure 2c)** | 단일 실험 설정(PPP)에서만 측정된 상관계수. 신뢰구간, p값 미보고. 다른 태스크로의 일반화 불확실 |
| ⚠️ **OOD 0.4% → 54.7%** | 3개 훈련 시드 평균이지만, 각 조건 간 데모 총량이 다름. 절대적 비교 불가 |
| ⚠️ **PP B=8, PPP B=16 포화** | "포화"의 통계적 기준(수렴 기준) 미명시. 오차 막대가 크면 포화 주장이 약해질 수 있음 |
| ⚠️ **2S-PP 결과 (Figure 6)** | ID 성능이 Cov가 낮을수록 높은 역설적 결과를 "denser supervision"으로 설명하나, 이는 사후 해석 |
| ⚠️ **직교 집합 78.2%** | 단일 특정 직교 설계에 대한 결과. 다른 가능한 16-태스크 부분집합과 비교 없음 |
| ⚠️ **Figure 10 over-fitting 주장** | 직교 사전훈련 후 파인튜닝 시 OOD 하락을 과적합으로 귀인하지만, 학습률이나 데이터 불균형 등 대안적 설명 미검토 |
| ❌ **VLA 모델과의 비교 없음** | 현재 SOTA인 π0.5, Gemini Robotics 등과 직접 비교 불가 (다른 설정) |
| ❌ **실세계(Real-world) 실험 없음** | 모든 실험이 시뮬레이터(robomimic) 기반으로, 현실 전이 성능 미검증 |

---

## 6. 문서가 답하지 않는 질문

| # | 미해결 질문 |
|---|------------|
| 1 | 직교 설계가 아닌 다른 조합 설계(예: 라틴 방격, 커버링 어레이)와의 체계적 비교는? |
| 2 | 커버리지 원칙이 4개 이상의 인수(factor)를 가진 태스크로 확장될 때 동일하게 작동하는가? |
| 3 | 분해된 세 항($A_i, B_i, C_i$)의 실제 크기를 실험적으로 측정할 수 있는가? |
| 4 | 사전훈련된 VLA 모델(π0.5, OpenVLA 등)에도 동일한 커버리지 원칙이 적용되는가? |
| 5 | 연속적(continuous) 명령어 공간 또는 자연어 명령어에도 이론이 적용 가능한가? |
| 6 | OOD 성능 향상을 위한 최적 직교 설계를 자동으로 찾는 알고리즘은 무엇인가? |
| 7 | 파인튜닝 후 ID 성능 하락을 방지하면서 OOD를 개선하는 방법은? |
| 8 | 실세계 로봇에서 동일한 결과가 재현되는가? 시뮬레이션-실세계 갭은 얼마나 큰가? |
| 9 | 단계 지시자 $\sigma(z)$를 잘못 학습하거나 오류가 발생할 때의 영향은? |
| 10 | 의미적 의존성이 있는 태스크의 의존성 구조를 사전에 자동으로 발견하는 방법은? |

---

## 7. 가장 중요한 그림 5개 해석

### 🖼️ Figure 2: 전체 튜플 커버리지 불필요 (p.5-6)

**구성:** (a) PP 태스크, (b) PPP 태스크의 훈련 예산 B vs 성공률 곡선, (c) PPP의 Seen pair count vs OOD 성공률 산점도

**해석:**
- (a): PP 태스크에서 $B=8$(전체 16개의 50%)에서 OOD 성공률이 포화. In-distribution(ID)과 OOD 성공률이 수렴하며, 전체 Cartesian 커버리지 불필요함을 직접 시각화
- (b): PPP에서 $B=16$(전체 64개의 25%)에서 유사한 포화 현상. OOD 성공률이 ID를 따라가며, 작은 구조화된 집합의 충분성을 지지
- (c): **가장 중요한 패널.** 각 OOD 태스크에 대해 "Seen pair count"(0~3)와 OOD 성공률 간 $r=0.82$의 강한 양의 상관 확인. 쌍별 커버리지가 OOD 성능의 핵심 예측 변수임을 시사

> ⚠️ 단, 상관계수의 인과 해석에 주의 필요 (교란 변수 가능성)

---

### 🖼️ Figure 3: 직교 집합 vs 전체 집합 (p.6)

**구성:** In-distribution(좌)과 Out-of-distribution(우)에서 데모 수에 따른 성공률 비교

**해석:**
- **저데이터 환경(64~256 데모):** 직교 집합이 전체 집합 대비 OOD 성능에서 명확히 우위. 데이터가 적을 때 구조적으로 다양한 조합에 데모를 집중하는 것이 효과적임을 보임
- **고데이터 환경(512~1024 데모):** 두 접근법의 격차가 줄어들며, 충분한 데이터가 있을 때는 구조가 덜 중요해짐
- **실용적 함의:** 데이터 수집 예산이 제한적인 현실에서 직교 설계가 훨씬 효율적

> 💡 **직교 배열 설계(Orthogonal Array Design):** 실험설계론에서 유래. 모든 인수 쌍의 조합이 균등하게 나타나도록 설계된 최소 실험 집합

---

### 🖼️ Figure 5: 희소 사전훈련 후 파인튜닝 (p.7)

**구성:** 대각선 사전훈련(diag_B4) 후 파인튜닝 데모 수에 따른 ID(좌)와 OOD(우) 성공률

**해석:**
- **"no FT" 지점:** 사전훈련만 했을 때 ID~80% 이상이지만 OOD~0.4%. 희소 커버리지 정책이 서브태스크 기술을 학습하지만 명령어 스티어링에는 실패함을 명확히 보임
- **파인튜닝 효과:** full-FT와 orthogonal-FT 모두 소량의 데모로 OOD를 급격히 개선 (54.7%까지)
- **"trained from scratch" 기준선:** 동일한 총 데모 예산에서 처음부터 훈련하면 훨씬 낮은 OOD 성능 → 사전훈련의 기술 재사용 가치를 직접 입증
- **ID 하락 트레이드오프:** 파인튜닝 후 ID 성공률이 소폭 하락하는 현상은 분포 재가중화 효과로 해석

---

### 🖼️ Figure 6: 2S-PP 의존적 쌍 커버리지 효과 (p.8)

**구성:** 다양한 의존적 쌍 커버리지(Cov4~Full)에서의 성공률(좌)과 같은 컨테이너 위반율(우)

**해석:**
- **성공률 패턴:** Cov4 → Cov6 → Cov9 → Cov12 → Full 순으로 OOD 성공률 단조 증가. 의존적 명령어 쌍 $(o_1, c_2)$의 커버리지가 OOD 성능 핵심 결정 요인
- **위반율 패턴:** OOD에서 쌍 커버리지가 낮을수록 "wrong container" 위반율 높음 → 의존성을 학습하지 못한 결과
- **역설적 ID 패턴:** Cov4에서 ID 성능이 가장 높은 것은 더 집중된 데모 분배(per-pair denser supervision) 때문. 일반화와 개별 태스크 성능 간 트레이드오프 존재
- **이론적 연결:** 이는 수식(5)의 명령어-조합적 이동 항이 의존적 쌍의 커버리지에 민감함을 실험적으로 확인

---

### 🖼️ Figure 7: 정책 네트워크 구조 (Appendix B, p.14)

**구성:** 전체 정책 파이프라인 블록 다이어그램

**해석:**
- **입력 처리:** DINOv2(frozen)로 이미지를 K,V 토큰으로 인코딩. 각 서브태스크 명령어는 별도의 Q 토큰으로 표현 → 다중 명령어를 명시적으로 분리 표현
- **Q-Former 역할:** 명령어 토큰이 이미지 특징을 질의(Query)하여 태스크 관련 시각 정보 추출. N개 레이어 반복
- **Flow-Matching Decoder:** proprioception(고유 수용성 감각) 정보를 추가로 활용하여 노이즈 행동( $a_\varepsilon \sim \mathcal{N}(0, I)$ )을 M번의 denoise 단계를 거쳐 실제 행동($a_0$)으로 변환
- **설계의 시사점:** Q-Former 구조가 자연스럽게 명령어별 주의(attention)를 분리하여, 단계별 모듈성(Stage-wise Modularity)의 귀납 편향(inductive bias)을 소프트하게 구현

> 💡 **Proprioception(고유 수용성 감각):** 로봇 자신의 관절 각도, 속도, 힘 등의 내부 상태 정보. 6D 그리퍼 포즈 포함
>
> 💡 **귀납 편향(Inductive Bias):** 모델 구조 자체가 특정 종류의 해를 선호하도록 유도하는 사전 가정

---

## 8. 결론 및 후속 연구

### 연구자들이 제시한 시사점 (저자 직접 기술)

1. **데이터 수집 원칙:** 태스크 집합을 무분별하게 확장하기보다 **의존성 커버리지 우선**으로 데모 할당
2. **직교 설계의 실용성:** 전체 공간의 1/4로도 충분한 OOD 성능 달성 가능
3. **파인튜닝 전략:** 희소 사전훈련 + 소량 광범위 파인튜닝의 조합이 효율적

### 저자가 제시한 후속 연구 방향

1. **대규모 훈련으로 확장:** 명령어 복잡도가 클 때의 커버리지 원칙 분석
2. **사전훈련 VLA 모델 분석:** π0.5, OpenVLA 등 대형 모델에서 조합적 일반화 기원 진단 및 커버리지 원칙 전이 가능성 평가

---

### 8-1. 모델의 일반화 성능 향상 가능성

본 논문의 분석 틀을 바탕으로 다음과 같은 방향에서 일반화 성능 향상이 가능하다:

**① 데이터 측면**

수식 (5)의 세 항을 직접 최소화하는 전략:

- **항 1 최소화** ($\|q(l_i) - p(l_i)\|_1$): 모든 서브태스크 값의 균등한 마지널 커버리지 보장 (기본 조건)
- **항 2 최소화** ($\mathbb{E}[\|q(l_{-i}|l_i) - p(l_{-i}|l_i)\|_1]$): 직교 배열, 커버링 어레이 등을 이용한 **체계적 쌍별 커버리지** 설계
- **항 3 최소화** ($\mathbb{E}[\|q(z,a|l) - p(z,a|l)\|_1]$): 도메인 랜덤화(domain randomization), MimicGen 스타일 자동 데이터 증강

**② 모델 측면**

$$\pi_\theta(a \mid z, l) = \pi_{\theta, \sigma(z)}(a \mid z, l_{\sigma(z)})$$

수식 (6)의 모듈형 정책 구조가 이론적으로 일반화 격차를 줄이므로:

- **더 정확한 단계 지시자** $\sigma(z)$ 학습: 현재 어떤 서브태스크가 활성인지 더 정확히 인식
- **플래너-컨트롤러 분리 강화:** 고수준 플래너가 의존성 관계를 명시적으로 저수준 컨트롤러에 전달
- **어텐션 마스킹:** 비활성 서브태스크 명령어에 대한 어텐션을 명시적으로 억제

**③ 이론-실험 연결 강화**

$$\Delta_q(\theta) = A_i + B_i + C_i$$

각 항을 실험적으로 측정 가능하게 만드는 **진단 메트릭** 개발이 필요하다. 예를 들어:
- $A_i$: 마지널 shift 측정을 위한 held-out marginal 실험
- $B_i$: 각 쌍별 OOD 성능과 seen pair count의 회귀 분석
- $C_i$: 동일 명령어 하에서의 맥락 변이 실험

---

### 8-2. 2020년 이후 관련 최신 연구 비교 분석

> ⚠️ **주의:** 아래 비교 분석은 논문 내 인용된 문헌들과 공개된 정보를 기반으로 하며, 논문에서 직접 비교하지 않은 부분은 명확히 구분합니다.

| 연구 | 연도 | 핵심 아이디어 | 본 논문과의 관계 |
|------|------|---------------|-----------------|
| **SCAN (Lake & Baroni)** [12,13] | 2018 | 언어 모델의 조합 일반화 실패 진단 | 언어 도메인의 문제를 로보틱스로 확장하는 동기 제공 |
| **VIMA** [11] | 2022 | 멀티모달 프롬프트로 다양한 조작 태스크 | OOD 조합 평가 벤치마크 제공. 본 논문은 데이터 커버리지 측면 분석 |
| **SayCan** [1] | 2022 | 언어 모델 + 어포던스 함수의 계층적 구조 | Stage-wise Modular Policy의 실제 구현 선례 |
| **MimicGen** [17] | 2023 | 인간 데모에서 자동 서브태스크 합성으로 데이터 확장 | 데이터 효율성 관점 공유. 본 논문은 커버리지 구조 설계에 집중 |
| **ClevrSkills** [15] | 2024 | 로보틱스 조합 일반화 벤치마크 | 본 논문과 동일 문제 평가. 본 논문은 이론적 분해 추가 |
| **π0.5** [2] | 2025 | 오픈-월드 일반화 VLA 모델 | 본 논문이 제안하는 분석 틀을 적용해야 할 다음 대상 |
| **Gao et al.** [18] | 2024 | 환경 인수 다양화를 통한 효율적 데이터 수집 | 본 논문과 가장 유사한 문제의식. 본 논문은 명령어 인수에 집중하고 이론 추가 |
| **RoboHiMan** [16] | 2025 | 장기 조작에서 조합 일반화 계층적 평가 | 본 논문의 분석 원칙을 더 복잡한 설정에 적용할 수 있는 벤치마크 |
| **Interleave-VLA** [5] | 2025 | 교차 이미지-텍스트 명령어로 다양한 조작 | 명령어 공간 표현 다양화. 본 논문의 커버리지 원칙 적용 가능성 |

**해당 논문이 앞으로의 연구에 미치는 영향:**

1. **데이터 수집 패러다임 전환:** "더 많은 데이터"가 아닌 "더 구조적인 데이터"로의 전환 근거 제공. 대규모 로봇 데이터셋(Open-X Embodiment 등) 설계 원칙에 영향 가능

2. **VLA 연구 방향:** π0.5, Gemini Robotics 등 대형 VLA 모델이 왜 특정 OOD 조합에 실패하는지 진단하는 이론적 틀 제공

3. **커리큘럼 학습:** 어떤 순서로 태스크를 노출해야 하는지(curriculum)에 대한 이론적 근거가 될 수 있음

**앞으로 연구 시 고려할 점:**

| 고려사항 | 내용 |
|----------|------|
| **실세계 검증** | 시뮬레이션에서 도출된 원칙이 현실 로봇에서도 성립하는지 검증 필수 |
| **자동 의존성 발견** | 어떤 명령어 인수 쌍이 의존적인지를 사전에 자동 식별하는 방법 개발 |
| **연속 명령어 공간** | 이산 명령어에 특화된 이론을 자연어나 연속 임베딩 공간으로 확장 |
| **VLA 사전훈련 효과** | 대규모 웹 데이터로 사전훈련된 모델에서 커버리지 원칙이 어떻게 변하는지 |
| **교란 변수 통제** | Seen pair count의 인과 효과를 더 엄밀히 검증하는 실험 설계 필요 |
| **계산 비용** | 직교 설계 계산 자체의 비용 및 대규모 인수 공간에서의 확장성 |

---

## 📚 참고문헌 (논문 내 인용 문헌)

1. Ahn et al., "Do as I Can, Not as I Say," arXiv:2204.01691, 2022 (SayCan)
2. Physical Intelligence et al., "π0.5: A Vision-Language-Action Model," arXiv:2504.16054, 2025
3. Gemini Robotics Team, "Gemini Robotics 1.5," arXiv:2510.03342, 2025
4. Octo Model Team, "Octo: An Open-Source Generalist Robot Policy," arXiv:2405.12213, 2024
5. Fan et al., "Interleave-VLA," arXiv:2505.02152, 2025
6. Zitkovich et al., "RT-2: Vision-Language-Action Models," CoRL 2023
7. Kim et al., "OpenVLA," arXiv:2406.09246, 2024
8. Vaswani et al., "Attention is All You Need," NeurIPS 2017
9. Pertsch et al., "FAST: Efficient Action Tokenization," arXiv:2501.09747, 2025
10. Chi et al., "Diffusion Policy," IJRR 2023
11. Jiang et al., "VIMA: General Robot Manipulation," arXiv:2210.03094, 2022
12, 13. Lake & Baroni, "Generalization without Systematicity," ICML 2018
14. Yagcioglu et al., "Sequential Compositional Generalization," arXiv:2404.12013, 2024
15. Haresh et al., "ClevrSkills," NeurIPS 2024
16. Chen et al., "RoboHiMan," arXiv:2510.13149, 2025
17. Mandlekar et al., "MimicGen," arXiv:2310.17596, 2023
18. Gao et al., "Efficient Data Collection for Robotic Manipulation," arXiv:2403.05110, 2024
19. Kacker et al., "Taguchi's Orthogonal Arrays," J. NIST 1991
20. Hedayat et al., "Orthogonal Arrays: Theory and Applications," Springer 2012
21. Kuhn et al., "Introduction to Combinatorial Testing," CRC Press 2013
22. Bacon et al., "The Option-Critic Architecture," AAAI 2017
23. Nachum et al., "Data-Efficient Hierarchical RL," NeurIPS 2018
24. Garrett et al., "Integrated Task and Motion Planning," Annual Review 2021
25-26. (SayCan 관련 후속 연구들)
27. Mandlekar et al., "What Matters in Learning from Offline Demonstrations," CoRL 2021 (robomimic)
28. Oquab et al., "DINOv2," TMLR 2024
29. Li et al., "BLIP-2," ICML 2023

**논문 원문:** Wang et al., "Diagnosing Compositional Generalization in Sequential Robot Tasks," arXiv:2607.29687v1, 2026
**프로젝트 페이지:** https://yixiaowang7.github.io/Diagnosing_Compositional_Generalization_Robot_Page/
