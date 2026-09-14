# Boosting Out-of-Distribution Image Detection With Epistemic Uncertainty

> **참고 문헌:**
> - Oh, D., Ji, D., Kwon, O., & Hyun, Y. (2022). Boosting Out-of-Distribution Image Detection With Epistemic Uncertainty. *IEEE Access*, 10, 109289–109298. DOI: 10.1109/ACCESS.2022.3213667
> - 본 답변은 제공된 논문 PDF 원문에만 근거하며, 확인되지 않은 내용은 명시적으로 표기합니다.

---

## 1. Executive Summary (10문장 이내)

현대 딥러닝 모델은 학습 분포 외의 샘플(OOD)에 대해서도 과도하게 높은 신뢰도(Confidence)를 출력하는 문제가 있다.  
이 논문은 이를 해결하기 위해 **UA-FGSM(Uncertainty-based Additive Fast Gradient Sign Method)** 을 제안한다.  
핵심 아이디어는 사전 학습된 모델이 내재적으로 보유한 **인식론적 불확실성(Epistemic Uncertainty)** 을 MC Dropout으로 추출하여 입력 이미지에 역방향 perturbation을 가하는 것이다.  
In-distribution 샘플은 perturbation 후 더 과신뢰(over-confident)해지고, OOD 샘플은 덜 과신뢰해지도록 유도하여 두 분포 간 분리를 극대화한다.  
여러 MC Dropout 시행에서 얻은 gradient를 누적(additive)하는 방식으로 탐지 성능을 추가로 향상시킨다.  
이 방법은 재학습이나 앙상블 없이 사전 학습 모델만으로 동작하는 단순 전처리 기법이다.  
MobileNet-v2, ShuffleNet-v2, DenseNet-BC100 등 다양한 아키텍처와 CIFAR-10/100, TinyImageNet, LSUN, iSUN, 노이즈 데이터셋에서 기존 Odin 대비 개선된 성능을 보였다.  
특히 정확도 손실 없이 OOD 탐지 성능을 향상시키는 것이 주요 장점이다.  
알고리즘 민감도 테스트에서 10개의 다른 랜덤 시드에 대해 동일한 결과를 보여 안정성을 입증하였다.  
계산 비용 측면에서도 기존 Odin(45초/1,000샘플)과 거의 동등(48초/1,000샘플)하게 효율적이다.

---

### 1-1. 연구의 목적과 필요성

**목적:** 사전 학습된 딥러닝 모델에서 OOD 샘플을 효과적으로 탐지하는 전처리 알고리즘 개발

**필요성:**
- 자율주행차, 음성 비서 등 안전-핵심(safety-critical) 시스템에서 미지의 입력에 대한 즉각적 거부 피드백이 요구됨 (p.1, Abstract)
- 기존 딥러닝 모델은 OOD 샘플에도 높은 신뢰도를 부여하는 **과신뢰(overconfidence) 문제** 가 있음
- 기존 Odin [2] 방법은 ε(perturbation 크기) 튜닝이 민감하며 정확도 저하 위험이 존재함

> 💡 **OOD(Out-of-Distribution):** 모델이 학습한 데이터 분포와 다른 분포에서 온 입력 샘플. 예: 고양이 분류기에 자동차 이미지 입력.

---

## 2. 핵심 주장과 근거 표

| 핵심 주장 | 근거 | 위치 |
|---|---|---|
| OOD 샘플의 epistemic uncertainty가 in-distribution보다 높다 | MC Dropout 적용 시 OOD gradient가 다양한 클래스로 분산됨 | p.2, p.7 (Fig. 6) |
| 역방향 perturbation이 in-distribution을 더 과신뢰하게 만든다 | Fig. 3의 클래스 확률 분포 시각화 | p.5, Fig. 3 |
| Gradient 누적(additive)이 정확도 유지에 기여한다 | MC 누적 시 gradient가 평균으로 수렴(averaging out) | p.3, Fig. 2; p.7, Fig. 5 |
| 재학습 없이 성능 향상 가능 | 전처리만으로 Odin 대비 대부분 메트릭 향상 | p.6, Table 2 |
| 계산 효율이 I-FGSM 대비 뛰어나다 | UA-FGSM 48s vs I-FGSM 410s (1,000샘플 기준) | p.8 |

---

### 2-1. 해결 문제, 제안 방법, 모델 구조, 성능 향상 및 한계

#### 🔴 해결하고자 하는 문제

딥러닝 분류기는 학습 분포 외의 입력(OOD)에 대해서도 특정 클래스에 높은 softmax 확률을 출력한다. 이를 **과신뢰 문제(overconfidence problem)** 라 하며, 안전-핵심 응용에서 위험하다. 기존 Odin [2]은 perturbation 파라미터 ε에 매우 민감하고 정확도를 저하시킬 수 있다.

---

#### 🟡 제안하는 방법 (UA-FGSM)

**Step 1: 기존 FGSM (Goodfellow et al., 2015) [6]**

$$\mathbf{x}^{adv} = \mathbf{x} + \varepsilon \cdot \text{sign}(\nabla_{\mathbf{x}} J(\mathbf{x}, y)) $$

| 기호 | 설명 |
|---|---|
| $\mathbf{x}$ | 원본 입력 이미지 |
| $\mathbf{x}^{adv}$ | perturbation이 적용된 적대적 예시 |
| $\varepsilon$ | perturbation 크기 조절 파라미터 |
| $J(\mathbf{x}, y)$ | 올바른 레이블 $y$에 대한 비용 함수 |
| $\nabla_{\mathbf{x}}$ | 입력 $\mathbf{x}$에 대한 비용 함수의 기울기 |

> 💡 **FGSM(Fast Gradient Sign Method):** 입력 이미지에 손실 함수의 기울기 부호 방향으로 작은 노이즈를 추가해 모델을 속이는 적대적 공격 방법.

**Step 2: 반복 FGSM (I-FGSM, Kurakin et al.) [7]**

$$\mathbf{x}_0^{adv} = \mathbf{x}, \quad \mathbf{x}_{t+1}^{adv} = \mathbf{x}_t^{adv} + \alpha \cdot \text{sign}(\nabla_{\mathbf{x}} J(\mathbf{x}_t, y)) $$

| 기호 | 설명 |
|---|---|
| $\alpha$ | 각 반복에서의 스텝 크기 |
| $t$ | 반복 횟수 인덱스 |

**Step 3: 제안 방법 UA-FGSM (핵심 수식)**

$$\mathbf{x}^{adv} = \mathbf{x} - \varepsilon \cdot \sum_{mc} \text{sign}(\nabla_{\mathbf{x}_{mc}} J(\mathbf{x}_{mc}, y)) $$

| 기호 | 설명 |
|---|---|
| $\mathbf{x}_{mc}$ | 동일한 이미지 $\mathbf{x}$를 $mc$번 복사하여 mini-batch 축으로 연결한 변수 |
| $mc$ | MC Dropout 시행 횟수 (dropout trial) |
| $\nabla_{\mathbf{x}_{mc}}$ | backward MC Dropout을 사용한 $\mathbf{x}_{mc}$에 대한 기울기 |
| $-$ (음수 부호) | **역방향 perturbation**: 기존 FGSM과 반대 방향으로 적용 |
| $\sum_{mc}$ | $mc$번의 stochastic gradient를 누적(additive) |

> 💡 **MC Dropout(Monte Carlo Dropout):** 추론(inference) 시에도 Dropout을 활성화하여 여러 번 forward pass를 수행함으로써 모델의 불확실성을 추정하는 기법. Gal & Ghahramani (2016) [19]에 의해 Bayesian 근사로 해석됨.

> 💡 **Epistemic Uncertainty(인식론적 불확실성):** 데이터 부족이나 지식 한계로 인한 모델 자체의 불확실성. 더 많은 데이터로 줄일 수 있음. OOD 샘플에서 특히 높게 나타남.

**Step 4: U(x) — Epistemic Uncertainty 분석 지표**

$$U(\mathbf{x}) = \frac{1}{N-1} \sum_{i \neq y} [f_y(\mathbf{x}) - f_i(\mathbf{x})] $$

| 기호 | 설명 |
|---|---|
| $N$ | 클래스 수 |
| $f_y(\mathbf{x})$ | 타겟 클래스 $y$의 logit 값 |
| $f_i(\mathbf{x})$ | 나머지 클래스 $i$의 logit 값 |
| $U(\mathbf{x})$ | 타겟 클래스 logit과 나머지 클래스 logit 간 평균 거리 (신뢰도 지표) |

> 💡 **Logit:** Softmax 함수 적용 전의 원시 출력값. 값이 클수록 해당 클래스에 대한 모델의 확신이 강함.

---

#### 🟢 모델 구조 (Fig. 1 기반)

```
입력 이미지 x
    │
    ├─[1차 Forward]─→ 클래스 확률 P → argmax → 예측 레이블 y
    │
    ├─[Backward w/ MC Dropout]─→ gradient 누적 → sign 합산
    │
    └─[x - ε·ΣSign(gradient)]─→ x^adv
                                    │
                               [2차 Forward]─→ 신뢰도 점수 C = max(P')
                                    │
                               OOD Detector (임계값 비교)
```

- 총 **2회 Forward pass + 1회 Backward pass** 만으로 구성 (재학습 불필요)
- MC Dropout은 각 네트워크 기본 빌딩 블록 끝에 적용
- 파라미터: $\varepsilon = 0.0014$, $mc = 2$, dropout ratio = 1% (Table 2)

---

#### 🔵 성능 향상

(Table 2, p.6 기준 — **저자 직접 보고값**)

- 대부분의 아키텍처·데이터셋 조합에서 Odin 대비 **TNR, Detection Accuracy, AUROC, AUPR 향상**
- 예: ShuffleNet-v2 + CIFAR-10 + Uniform noise → TNR: Odin 6.0% → **UA-FGSM 49.7%** (대폭 향상)
- 계산 시간: Odin 45s ≈ UA-FGSM 48s << I-FGSM 410s (1,000 샘플, GTX 1080 기준)

---

#### 🔴 한계

- **특정 조합에서 TNR = 0%**: MobileNet-v2 + CIFAR-100 (in) + Gaussian noise (OOD) 조합에서 Detection Accuracy는 87.7%이나 TNR = 0 (Table 2, p.6; Further Work, p.9)
- **파라미터 최적값이 아키텍처마다 다름**: mc와 dropout ratio의 최적값이 네트워크 구조에 따라 달라짐 (p.7)
- **소규모 이미지 데이터셋만 평가**: CIFAR-10/100, 32×32 이미지에 국한
- **Gaussian noise에 대한 과신뢰 현상 미해결**: 별도 추가 연구 필요 (p.9)
- **Random seed 10개 사용**: 통계적 검증이 제한적

---

## 3. 각 주장의 페이지/Figure/Table 번호

| 주장 | 위치 |
|---|---|
| OOD의 epistemic uncertainty > in-distribution | p.2 (본문), Fig. 6 (p.7) |
| UA-FGSM 수식 제안 | p.3, 수식 (3) |
| 역방향 perturbation의 attention 효과 | p.3, Fig. 3 (p.5) |
| gradient 누적의 averaging-out 효과 | p.3, Fig. 2 (p.4) |
| 파라미터 효과 분석 | p.6–7, Fig. 5 |
| Odin 대비 성능 비교 | p.6, Table 2; p.5, Fig. 4 |
| U(x) 분포 분석 | p.7, 수식 (4), Fig. 6 |
| additive way 효과 | p.8, Fig. 7 |
| 계산 시간 비교 | p.8 (본문) |
| TNR=0 한계 | p.9, Table 2 |

---

## 4. 저자 직접 보고 결과 vs. 해석 분리

### 📌 저자가 직접 보고한 결과

| 항목 | 내용 |
|---|---|
| 성능 (Table 2) | 대부분 조합에서 Odin 대비 모든 메트릭 향상 |
| 계산 시간 | UA-FGSM: 48s, Odin: 45s, I-FGSM: 410s (1,000 샘플) |
| 파라미터 설정 | $\varepsilon=0.0014$, $mc=2$, dropout ratio=1% |
| 한계 사례 | MobileNet-v2 + CIFAR-100 + Gaussian → TNR=0 |
| 민감도 | 10개 랜덤 시드에서 동일 결과 |
| ShuffleNet-v2 + CIFAR-100 + LSUN ROC | Fig. 4a에서 큰 차이 시각적 확인 |

### 🔍 검토자(본 분석)의 해석

- **긍정적 해석:** 재학습 불필요, 계산 효율, 정확도 유지 조합은 실용적 가치가 높음. 특히 mini-batch 병렬 MC Dropout 구현은 영리한 엔지니어링 선택임
- **비판적 해석:**
  - 평가 데이터셋이 모두 32×32 소형 이미지에 국한되어 **실제 고해상도 환경 일반화 여부 불명확**
  - $\varepsilon=0.0014$를 Odin에서 그대로 가져온 점은 **자체 파라미터 최적화 미수행**을 의미 (p.6)
  - 10개 랜덤 시드 동일 결과는 dropout ratio=1%의 극소값 때문으로 저자가 스스로 인정 → **통계적 다양성 검증의 의미가 약함**
  - Gaussian noise에서 TNR=0인 현상은 단순 전처리 방법의 구조적 한계를 시사

---

## 5. ⚠️ 통계적으로 취약한 부분 및 비교 불가능한 수치

| 항목 | 문제점 |
|---|---|
| **랜덤 시드 10개** | Dropout ratio=1%로 인해 결과가 동일 → 통계적 변동성 검증 의미 없음 (p.8) |
| **iSUN 샘플 수 불일치** | iSUN은 8,925개, 나머지 OOD는 10,000개 → 직접 비교 시 주의 필요 (p.4) |
| **$\varepsilon$ 파라미터 고정** | Odin 최적값 그대로 사용 → UA-FGSM에 최적화된 값이 아닐 수 있음 (p.6) |
| **DenseNet-BC100 + CIFAR-100 + Gaussian TNR** | Ours: 14.5% (Hendrycks 0.0%, Odin 0.2%) — 개선이 있으나 절대값 매우 낮음 (Table 2) |
| **ShuffleNet-v2 + CIFAR-100 대부분 조합** | TNR이 47% 이하로 낮아 실용적 의미 제한적 (Table 2) |
| **비교 대상이 Odin만** | Mahalanobis [5], GAN 기반 [3] 등과의 직접 비교 없음 → 범용 SOTA 주장 불가 |
| **하드웨어 환경 단일** | GTX 1080 단일 GPU 환경만 보고 → 일반화된 계산 시간 주장 취약 |

---

## 6. 논문이 답하지 않는 질문

1. **대형·고해상도 이미지(ImageNet 224×224 등)에서의 성능은?** — 실험이 32×32 이미지에 국한
2. **Gaussian noise에서 TNR=0 현상의 근본 원인과 해결책은?** — "추후 연구" 언급만 있음 (p.9)
3. **Dropout이 없는 아키텍처(ViT, Transformer 계열)에 적용 가능한가?** — 미언급
4. **Temperature scaling (Odin의 핵심 구성요소)과의 조합 효과는?** — UA-FGSM이 Odin의 temperature scaling을 포함하는지 명확하지 않음
5. **Mahalanobis 거리 기반 방법 [5]이나 Energy-based OOD [Liu et al., 2020] 대비 성능은?** — 비교 없음
6. **Aleatoric uncertainty(우연적 불확실성)와의 구분 및 활용 가능성은?** — epistemic uncertainty만 다룸
7. **OOD 샘플의 종류(semantic shift vs. covariate shift)에 따른 성능 차이는?**
8. **실제 배포 환경에서 임계값(threshold) 설정 기준은?** — 평가 메트릭만 제시

> 💡 **Aleatoric Uncertainty(우연적 불확실성):** 데이터 자체의 노이즈나 내재적 불확실성. 더 많은 데이터를 수집해도 줄일 수 없음. Epistemic uncertainty와 대비되는 개념.

---

## 7. 가장 중요한 그림 5개 해석

### 📊 Figure 1 (p.3) — UA-FGSM 전체 구조

**해석:** 전체 파이프라인을 시각화. 입력 이미지 X에서 1차 forward로 예측 레이블 y를 얻고, backward MC Dropout으로 gradient를 계산해 $X^{adv}$를 생성한 뒤 2차 forward로 최종 신뢰도 점수 C를 산출한다. **재학습 없이 2회 forward + 1회 backward만으로 작동**하는 경량성이 핵심 메시지. CNN은 공유(Shared)되어 추가 메모리 오버헤드가 최소화됨.

---

### 📊 Figure 2 (p.4) — Uncertainty-based Additive Gradient 시각화

**해석:** In-distribution 샘플(개)과 OOD 샘플(판다)의 gradient 패턴 변화를 $mc=1,3,5$로 비교.

- **In-distribution:** mc가 증가해도 gradient의 텍스처 구조(물체 윤곽 등)가 유지됨 → perturbation이 의미 있는 방향으로 집중
- **OOD:** mc가 증가하면 gradient가 빠르게 평균으로 수렴(averaging out) → 무의미한 perturbation

이 차이가 두 분포 간 신뢰도 격차를 만드는 핵심 메커니즘이며, 동시에 정확도 저하를 방지하는 이유이기도 함.

---

### 📊 Figure 4 (p.5) — Odin vs UA-FGSM 비교

**해석:**
- **(a) ROC 곡선:** ShuffleNet-v2 + CIFAR-100(in) + LSUN(OOD). 붉은 선(UA-FGSM)이 파란 선(Odin) 위로 크게 올라가 있어 AUROC가 명확히 향상됨.
- **(b) 신뢰도 점수 분포:** UA-FGSM 적용 후 in-distribution(빨간 점선)이 오른쪽으로, OOD(파란 점선)가 왼쪽으로 이동 → 두 분포가 더 잘 분리됨. 이것이 탐지 성능 향상의 직관적 증거.

---

### 📊 Figure 5 (p.7) — 파라미터 효과 분석

**해석:** mc(x축)와 dropout ratio(범례)에 따른 TNR, AUROC, 정확도 변화.

- **TNR/AUROC:** mc 증가에 따라 처음에는 성능이 Odin 이하이다가, mc≥2부터 급격히 향상. Dropout ratio가 높을수록 빠르게 수렴.
- **정확도 drop (c, f):** mc가 증가해도 정확도는 거의 0% 수준 유지 → **탐지 성능↑ + 정확도 유지** 동시 달성의 핵심 근거.
- 최적 mc는 아키텍처마다 다름 (MobileNet vs ShuffleNet의 최적점 위치 상이).

---

### 📊 Figure 6 (p.7) — U(x) 분포와 Epistemic Uncertainty 효과

**해석:** 수식 (4)의 $U(\mathbf{x})$ 값 분포를 in/out-distribution × Odin/UA-FGSM으로 4개 곡선 비교.

- **In-distribution (빨간색):** UA-FGSM 적용 후 분포가 오른쪽으로 이동 → $U(\mathbf{x})$값 증가 → 더 과신뢰해짐 (UA-FGSM의 의도대로 작동)
- **OOD + 낮은 U(x) 영역 (U < 0.00015):** UA-FGSM 적용 후 분포가 왼쪽으로 이동 → 덜 과신뢰해짐 → **epistemic uncertainty가 OOD 샘플의 gradient를 분산시키는 메커니즘의 직접 증거**

---

## 8. 결론, 시사점, 후속 연구

### 8-1. 저자 제시 시사점

- UA-FGSM은 재학습·앙상블 없이 사전 학습 모델에 바로 적용 가능한 실용적 OOD 탐지 기법
- MC Dropout의 역방향 누적 gradient가 in-distribution과 OOD의 신뢰도 격차를 증폭시킴
- 정확도 저하 없이 탐지 성능 향상이 가능함을 다양한 아키텍처로 검증

### 저자 제시 후속 연구

- Gaussian noise에 대한 TNR=0 현상의 원인 분석 및 개선 (p.9, Further Work)
- 알고리즘 강건성(robustness) 향상

---

### 8-1. 모델의 일반화 성능 향상 가능성

**현재 논문의 일반화 제약:**
- 32×32 소형 이미지와 3개 아키텍처(MobileNet-v2, ShuffleNet-v2, DenseNet-BC100)에만 검증
- Dropout 레이어를 내재적으로 활용하므로, **Dropout-free 아키텍처(ViT, ResNet without dropout 등)에 직접 적용 불가**

**일반화 가능성과 방향:**

1. **다른 불확실성 추정 방법과의 결합:** MC Dropout 대신 Deep Ensemble, Variational Dropout, Spectral Normalization 등과 결합 시 Dropout-free 모델에도 확장 가능
2. **고해상도·대형 데이터셋 검증:** ImageNet-1K 규모 실험을 통한 일반화 검증 필요
3. **Semantic OOD vs Covariate shift 분리 평가:** 현재는 두 유형을 구분하지 않음
4. **도메인 적응(Domain Adaptation)과의 연계:** 불확실성 기반 perturbation을 few-shot, zero-shot 시나리오에 확장

---

### 8-2. 2020년 이후 관련 최신 연구 비교 분석

> ⚠️ **주의:** 아래 비교는 본 PDF 원문에 직접 인용된 내용이 아닌, 공개된 AI/ML 연구 동향에 기반한 분석입니다. 각 논문의 정확한 수치는 해당 원문을 직접 확인하시기 바랍니다.

| 연구 | 방법 | 특징 | UA-FGSM 대비 |
|---|---|---|---|
| **Energy-based OOD** (Liu et al., NeurIPS 2020) | Energy score를 OOD 지표로 사용 | Softmax 대체, 이론적 근거 강함 | 에너지 함수가 epistemic uncertainty보다 더 calibrated된 지표일 수 있음 |
| **REACT** (Sun et al., NeurIPS 2021) | Feature activation 클리핑 | 재학습 불필요, 간단한 후처리 | UA-FGSM과 유사한 전처리 방식이나 gradient 미사용 |
| **VIM** (Wang et al., CVPR 2022) | Virtual-logit Matching | 특징 공간 활용, 강한 성능 | 더 복잡한 계산 필요 |
| **KNN-based OOD** (Sun et al., ICML 2022) | 특징 공간 K-최근접 이웃 | 비모수적 접근, 재학습 불필요 | 메모리 요구량 높음 |
| **DICE** (Sun & Li, ECCV 2022) | 가중치 희소화(sparsification) | 단순하고 효과적 | 재학습 불필요, UA-FGSM과 직교적 접근 |

**UA-FGSM이 앞으로의 연구에 미치는 영향:**

1. **Uncertainty + Perturbation의 조합 패러다임 제시:** epistemic uncertainty와 input perturbation의 결합은 후속 연구의 설계 방향을 제시
2. **전처리(pre-processing) 방식의 재조명:** 재학습 없는 방법의 실용성 증명
3. **MC Dropout의 OOD 탐지 활용 확장:** 기존에는 예측 불확실성 추정에만 쓰이던 MC Dropout을 OOD 탐지용 gradient 계산에 활용

**앞으로 연구 시 고려할 점:**

1. **Transformer/ViT 아키텍처 호환성:** Attention 기반 모델에서 MC Dropout 대안 필요
2. **Large Language Model(LLM)로의 확장:** 텍스트 도메인 OOD 탐지에 유사 아이디어 적용 가능성
3. **Calibration과의 관계:** OOD 탐지 성능과 신뢰도 보정(calibration) 간 trade-off 분석 필요
4. **적대적 공격에 대한 취약성:** perturbation 기반 방법이 adversarial attack에 노출될 위험성 검토 (Carlini & Wagner [18] 관점)
5. **Benchmark 표준화:** OpenOOD 등 표준화된 벤치마크와의 비교 필요

---

> **최종 참고 자료 목록:**
> 1. Oh et al. (2022). "Boosting Out-of-Distribution Image Detection With Epistemic Uncertainty." *IEEE Access*, 10, 109289–109298.
> 2. Liang et al. (2018). "Enhancing the reliability of out-of-distribution image detection in neural networks." *ICLR 2018*. (논문 내 [2] 인용)
> 3. Gal & Ghahramani (2016). "Dropout as a Bayesian approximation." *ICML 2016*. (논문 내 [19] 인용)
> 4. Goodfellow et al. (2015). "Explaining and harnessing adversarial examples." *ICLR 2015*. (논문 내 [6] 인용)
> 5. Kendall & Gal (2017). "What uncertainties do we need in Bayesian deep learning for computer vision." *NeurIPS 2017*. (논문 내 [15] 인용)
> 6. Liu et al. (2020). "Energy-based out-of-distribution detection." *NeurIPS 2020*. (**⚠️ 논문 원문에 직접 인용되지 않음 — 비교 분석 목적으로 참조**)
