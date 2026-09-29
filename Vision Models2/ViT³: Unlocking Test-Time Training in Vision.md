# ViT³: Unlocking Test-Time Training in Vision

**검토 기준:** 첨부하신 **arXiv:2512.01643v2, 2026년 4월 20일판**을 분석했습니다. 최초 공개본은 2025년 12월 1일이며, 아래 페이지 번호는 **첨부 PDF의 물리적 페이지 1–13쪽**을 기준으로 합니다. 본문은 1–8쪽, 참고문헌은 9–11쪽, 부록은 12–13쪽입니다. 외부 연구는 2026년 9월 29일에 확인한 1차 자료를 사용했습니다. :chatgpt-content-reference{index="0"}

아래에서 **[저자 보고]**는 논문에 제시된 결과, **[해석]**은 그 결과에 대한 제 판단, **[미검증]**은 논문이 실험하거나 입증하지 않은 사항을 뜻합니다. 성능 수치는 독립 재현 결과가 아니라 **저자 보고치**이며, 성능 차이는 해당 표의 수치로부터 계산했습니다.

---

## 1. Executive summary

ViT³는 시각 모델에서 **Test-Time Training, TTT를 효율적인 토큰 간 정보 교환 연산으로 설계하는 방법**을 체계적으로 연구한 논문입니다〔1–2쪽, Figure 1〕. :chatgpt-content-reference{index="1"}  
연구의 필요성은 Softmax attention의 이차 계산 복잡도를 줄이면서도, 기존 선형 attention의 제한된 표현력을 보완할 실용적인 설계 지침이 부족하다는 데 있습니다〔1–2쪽〕. :chatgpt-content-reference{index="2"}  
핵심 방법은 각 입력에서 얻은 key–value 쌍으로 작은 내부 신경망을 업데이트한 뒤, 업데이트된 신경망에 query를 입력해 출력을 계산하는 것입니다〔3쪽, 식 (5)〕. :chatgpt-content-reference{index="3"}  
저자들은 손실함수의 미분 특성, 전체 토큰을 이용한 한 번의 업데이트, 학습률, 내부 모델의 폭과 깊이, 합성곱 사용에 관한 여섯 가지 경험적 지침을 제시합니다〔4–6쪽, Tables 1–4·6〕. :chatgpt-content-reference{index="4"}  
이를 결합한 ViT³-S는 ImageNet-1K에서 81.6%를 기록해 DeiT-S의 79.8%보다 1.8%포인트 높지만, 파라미터와 연산량도 조금 더 큽니다〔6쪽, Table 7〕. :chatgpt-content-reference{index="5"}  
분할에서는 H-ViT³-B가 51.7 mIoU로 VMamba-B의 51.0을 넘지만 TransNeXt-B의 53.0에는 미치지 못하므로, 모든 Transformer를 능가했다는 결론은 부적절합니다〔8쪽, Table 9〕. :chatgpt-content-reference{index="6"}  
고해상도 효율성 실험에서는 $1248\times1248$ 입력에서 ViT³-T가 DeiT-T보다 처리량 4.6배, GPU 메모리 사용량 90.3% 감소를 보고합니다〔8쪽, Figure 4〕. :chatgpt-content-reference{index="7"}  
**[해석] 가장 확실한 기여는 “시각 TTT의 설계 원리와 여러 과제에서의 경쟁력”이며, 분포 변화에 대한 일반화 향상이나 테스트 시 학습량을 늘릴수록 성능이 좋아진다는 주장은 아직 입증되지 않았습니다**〔평가 범위: 7–8쪽, Tables 5·7–10〕. :chatgpt-content-reference{index="8"} :chatgpt-content-reference{index="9"}

> **용어:** ‘토큰’은 이미지를 작은 패치나 특징 벡터로 나눈 처리 단위이고, ‘선형 복잡도’는 다른 조건을 고정했을 때 토큰 수에 비례해 계산량이 증가한다는 뜻입니다. 이 논문의 TTT는 정답 라벨 없이 **입력별 내부 모듈을 업데이트하는 연산**으로, 전체 모델을 테스트 데이터에 다시 학습시키는 절차와 구분해야 합니다.

---

## 2. 연구 목적과 핵심 주장·근거

### 2.1 연구가 해결하려는 문제

**[저자 보고]** 연구 질문은 단순히 “attention을 더 빠르게 만들 수 있는가?”가 아니라, **“효율적이면서 표현력이 높은 시각 TTT 내부 모듈을 어떤 원칙으로 설계해야 하는가?”**입니다. 연구 범위를 내부 학습 설정과 내부 모델 구조라는 두 축으로 나누고, DeiT-S의 attention을 TTT로 교체한 기준 모델에서 ImageNet-1K 300-epoch 실험을 수행합니다〔2–3쪽, Section 4〕. :chatgpt-content-reference{index="10"} :chatgpt-content-reference{index="11"}

> **용어:** ‘내부 학습’은 한 입력의 토큰들을 이용한 작은 모델의 업데이트이고, ‘외부 학습’은 실제 학습 데이터와 과제 손실을 이용한 전체 네트워크 학습입니다. ‘절제 실험, ablation’은 구성요소를 바꾸거나 제거해 그 영향을 비교하는 실험입니다.

### 2.2 여섯 가지 설계 주장

표의 정확도는 ImageNet-1K Top-1이며, **%포인트는 정확도끼리의 차이**를 의미합니다.

| 핵심 주장 | 저자가 직접 보고한 근거 | 해석 및 적용 범위 | 원문 위치 |
|---|---|---|---|
| **① 내부 손실의 혼합 2차 미분이 중요하다.** | MSE 79.2%, 내적 손실 78.9%, MAE 76.5%. MAE는 관련 혼합 미분이 거의 모든 점에서 0이다. | 내부 업데이트를 거쳐 value 투영행렬에 전달되는 외부 학습 신호가 중요하다는 설명과 일치한다. 모든 TTT 구조에 적용되는 보편적 불가능성 정리는 아니다. | 4쪽, 식 (6), Table 1. :chatgpt-content-reference{index="12"} |
| **② 전체 토큰으로 한 번 업데이트하는 설정이 좋은 효율–성능 절충이다.** | 전체 배치: 78.9%, 1,315 FPS. 토큰을 네 배치로 나누면 78.1%, 1,101 FPS. 전체 배치 3회는 79.2%, 787 FPS. | 한 번이 정확도 최고라는 뜻은 아니다. 여러 번의 업데이트가 주는 작은 향상과 큰 처리량 감소를 함께 고려한 선택이다. | 4쪽, Table 2. :chatgpt-content-reference{index="13"} |
| **③ 내부 학습률 1.0이 효과적이다.** | 학습률 0.1은 77.5%, 1.0과 2.0은 각각 78.9%. 5.0·10.0에서는 학습 발산이 나타난다. | 1.0이 유일한 최적값은 아니다. 손실 정규화와 특징 크기를 포함한 해당 설정에서의 실용적 선택이다. | 4쪽, Table 3. :chatgpt-content-reference{index="14"} |
| **④ 내부 MLP의 폭을 늘리면 성능이 개선된다.** | 은닉 폭 $d\rightarrow4d$에서 78.9%→79.6%. FLOPs는 4.58G→5.62G, FPS는 1,315→836. | 표현력 확대의 효과와 계산비용 증가가 함께 나타난다. 동일 계산예산에서 외부 모델 확대보다 낫다는 실험은 아니다. | 5쪽, Table 4, Remark 4. :chatgpt-content-reference{index="15"} :chatgpt-content-reference{index="16"} |
| **⑤ 내부 모델을 깊게 만드는 것은 현재 설정에서 오히려 어렵다.** | 선형 1층 79.1%, 2층 MLP 78.9%, 3층 MLP 77.5%. 깊은 모델은 학습 손실도 높다. | 과적합보다 최적화·과소적합 문제가 있다는 해석을 지지한다. 다만 원인을 특정한 직접 증명은 아니다. | 5–6쪽, Figure 3, Tables 4·6. :chatgpt-content-reference{index="17"} :chatgpt-content-reference{index="18"} |
| **⑥ 합성곱 내부 모델이 시각 과제에 적합하다.** | 일반 $3\times3$ 합성곱 79.9%, depthwise 합성곱 80.1%. 기준 2층 MLP는 78.9%. | 이미지 전체에서 얻은 정보로 지역 필터를 바꾸는 구조가 유용하다는 근거다. 분포 변화 강건성까지 입증한 결과는 아니다. | 5–7쪽, Table 4, Insight 6·Remark 6. :chatgpt-content-reference{index="19"} :chatgpt-content-reference{index="20"} :chatgpt-content-reference{index="21"} |

> **용어:** MLP는 여러 완전연결층을 쌓은 신경망이며, 폭은 층의 특징 차원, 깊이는 층 수입니다. FLOPs는 계산에 필요한 부동소수점 연산량, FPS는 초당 처리 이미지 수입니다. 두 지표는 메모리 접근과 병렬화 효율 때문에 반드시 비례하지 않습니다.

---

## 3. 제안 방법: 수식, 학습 과정, 모델 구조

### 3.1 Attention을 ‘입력으로 구성되는 내부 모델’로 해석한다

입력 특징을 $x\in\mathbb{R}^{N\times C}$라 하면 다음과 같이 query, key, value를 만듭니다.

$$
Q=xW_Q,\qquad K=xW_K,\qquad V=xW_V.
$$

여기서 $N$은 토큰 수, $C$는 입력 특징 차원, $d$는 한 head의 차원이며,

$$
W_Q,W_K,W_V\in\mathbb{R}^{C\times d},
\qquad
Q,K,V\in\mathbb{R}^{N\times d}
$$

입니다〔3쪽, 식 (1)〕. :chatgpt-content-reference{index="22"}

> **용어:** Query는 “어떤 정보를 가져올 것인가”, key는 “어떤 정보와 연결되는가”, value는 “전달할 내용”에 대응하는 특징입니다. Head는 서로 다른 투영을 사용해 정보를 처리하는 병렬 처리 갈래입니다.

#### Softmax attention

원문 식 (2)는 다음과 같습니다.

$$
O=\text{Softmax}(QK^\top)V
  =\text{Softmax}(QW_1)W_2,
\qquad
W_1=K^\top,\quad W_2=V.
$$

$O\in\mathbb{R}^{N\times d}$는 출력이고, $\top$는 전치입니다. 원문은 통상적인 $1/\sqrt d$ 계수를 $Q,K$의 스케일에 흡수할 수 있다며 생략합니다.

**[저자 보고]** 이 식은 Softmax attention을 **은닉 폭이 $N$인 2층 MLP 형태**로 볼 수 있음을 보여줍니다. 다만 이는 대수적 재표현이지, 실제로 별도의 MLP 학습을 수행한다는 뜻은 아닙니다〔3쪽, 식 (1)–(2)〕. :chatgpt-content-reference{index="23"}

#### 선형 attention

선형 attention에서는 별도로 $Q=\phi(xW_Q)$, $K=\phi(xW_K)$를 사용하여 다음 계산 재배열을 가능하게 합니다.

$$
O_i=
\frac{
Q_i\left(\sum_{j=1}^{N}K_j^\top V_j\right)
}{
Q_i\left(\sum_{j=1}^{N}K_j^\top\right)
}.
$$

$\phi$는 특징 변환 함수이고, $i,j$는 토큰 인덱스입니다. 분모의 정규화를 잠시 제외하면,

$$
O=Q(K^\top V)=QW,\qquad W=K^\top V
$$

가 됩니다. 즉 $K,V$의 정보를 $d\times d$ 행렬로 모아 사용합니다〔3쪽, 식 (3)–(4)〕. :chatgpt-content-reference{index="24"}

> **용어:** 이때 ‘커널’은 두 특징의 관계를 계산하는 함수적 표현입니다. 뒤에서 나오는 ‘합성곱 커널’, 즉 작은 공간 필터와는 다른 의미입니다.

**[해석]** ViT³의 출발점은 “고정된 형태의 $K^\top V$만 사용할 필요가 있는가?”라는 질문입니다. 내부 모델을 비선형 신경망이나 합성곱으로 확장하고, 그 파라미터를 입력에 맞게 구성하자는 접근입니다.

---

### 3.2 내부 업데이트와 외부 학습

원문 식 (5)의 일반적인 내부 학습은 다음과 같습니다.

$$
\widehat V_B=F_{W_t}(K_B),
$$

```math
W_{t+1}
=
W_t-\eta\nabla_{W_t}
\mathcal L(\widehat V_B,V_B),
```

$$
O=F_{W^*}(Q).
$$

기호의 의미는 다음과 같습니다.

| 기호 | 의미 |
|---|---|
| $F_W$ | 파라미터 $W$를 갖는 내부 모델 |
| $K_B,V_B$ | 한 입력에서 선택한 $B$개 key–value 토큰 |
| $\widehat V_B$ | key를 입력해 내부 모델이 예측한 value 특징 |
| $\mathcal L$ | 내부 학습 손실 |
| $\eta$ | 내부 학습률 |
| $t$ | 내부 업데이트 단계 |
| $W_0$ | 외부 학습을 통해 학습되는 내부 모델의 초기값 |
| $W^*$ | 해당 입력에서 내부 업데이트를 마친 파라미터 |

**중요한 구분:** $V_B$는 정답 클래스나 정답 마스크가 아니라, **입력을 value 투영행렬에 통과시켜 얻은 특징**입니다. 또한 $B$는 이미지 수가 아니라 **한 입력 내부의 토큰 배치 크기**입니다〔3쪽〕. :chatgpt-content-reference{index="25"}

**[저자 보고]** 외부 학습에서는 내부 업데이트 자체를 미분 가능한 계산 과정으로 펼쳐서 전체 네트워크와 함께 학습합니다. 최종 ViT³는 $B=N$, $\eta=1.0$, 내부 업데이트 1회를 사용합니다〔3쪽 및 7쪽, Section 5〕. :chatgpt-content-reference{index="26"} :chatgpt-content-reference{index="27"}

이를 설명용 목적함수로 묶어 쓰면 다음과 같습니다. **이 식은 원문의 학습 설명을 정리한 것이며, 원문에 번호가 붙어 제시된 식은 아닙니다.**

$$
\min_{\theta}
\mathbb E_{(x,y)\sim D_{\text{train}}}
\left[
\mathcal L_{\text{task}}
\left(f_\theta(x;W_\theta^*(x)),y\right)
\right].
$$

$\theta$는 전체 네트워크 파라미터, $D_{\text{train}}$은 외부 학습 데이터, $y$는 과제 정답, $\mathcal L_{\text{task}}$는 분류·검출 등 실제 과제의 손실입니다. 핵심은 **내부 학습이 잘된다는 사실 자체보다, 그 업데이트가 최종 과제에 유용하도록 외부 학습된다는 점**입니다.

> **용어:** ‘메타학습’은 학습 절차나 빠르게 적응할 초기값까지 학습하는 접근입니다. ‘업데이트를 펼쳐 미분한다’는 것은 업데이트 식도 네트워크의 계산 그래프에 포함한다는 뜻입니다.

---

### 3.3 손실함수: 왜 MAE가 특히 불리한가?

부록의 표기를 간단히 정리하기 위해 다음을 정의하겠습니다.

$$
\alpha=\frac{1}{B\sqrt d},
\qquad
e_{ij}=\widehat V_{ij}-V_{ij},
\qquad
S=\alpha\sum_{i=1}^{B}\sum_{j=1}^{d}e_{ij}^{2}.
$$

$i$는 토큰, $j$는 특징 좌표입니다. 동일 좌표에 대한 혼합 2차 미분을

```math
D_{ij}
=
\frac{\partial^2\mathcal L}
{\partial V_{ij}\,\partial\widehat V_{ij}}
```

로 쓰면 부록의 결과를 다음처럼 정리할 수 있습니다.

| 손실 | 원문 정의와 동등한 수식 | $D_{ij}$ | 원문 위치 |
|---|---|---|---|
| 내적 손실 | $\mathcal L=-\alpha\sum_i\widehat V_iV_i^\top$ | $-\alpha$ | 12쪽, 식 (8)–(9). :chatgpt-content-reference{index="28"} |
| MSE | $\mathcal L=S/2$ | $-\alpha$ | 12쪽, 식 (10)–(11). :chatgpt-content-reference{index="29"} |
| RMSE | $\mathcal L=\sqrt S$ | $-\alpha/\sqrt S+\alpha^2e_{ij}^2/S^{3/2}$, $S>0$ | 12쪽, 식 (12)–(13). :chatgpt-content-reference{index="30"} |
| MAE | $\mathcal L=\alpha\sum_{i,j}\lvert e_{ij}\rvert$ | $0$, $e_{ij}\neq0$ | 12쪽, 식 (14)–(15). :chatgpt-content-reference{index="31"} |
| Smooth L1 | $\mathcal L=\alpha\sum_{i,j}\rho(e_{ij})$ | $-\alpha$ if $\lvert e_{ij}\rvert<1$; $0$ if $\lvert e_{ij}\rvert>1$ | 12–13쪽, 식 (16)–(17). :chatgpt-content-reference{index="32"} |

Smooth L1의 함수 $\rho$는 다음과 같습니다.

$$
\rho(e)=
\begin{cases}
\frac12e^2,&\lvert e\rvert<1,\\
\lvert e\rvert-\frac12,&\text{그 외}.
\end{cases}
$$

경계점에서의 미분 가능성은 별도로 다뤄야 하며, RMSE에는 서로 다른 좌표 사이의 혼합 미분 항도 존재합니다. 위 표는 원문 부록처럼 **동일 좌표 항**을 표시한 것입니다.

> **용어:** MSE는 오차 제곱, RMSE는 제곱오차 합의 제곱근, MAE는 절댓값 오차에 기반합니다. ‘혼합 2차 미분’은 여기서 목표 특징 $V$의 변화가 예측 특징에 대한 학습 신호를 얼마나 바꾸는지 나타냅니다.

원문 식 (6)의 핵심을 벡터화한 Jacobian 표기로 쓰면 다음과 같습니다.

```math
\frac{\partial G}{\partial\theta_V}
=
J_{\widehat v,w}^{\top}
H_{\widehat v,v}
J_{v,\theta_V},
\qquad
G=\nabla_w\mathcal L.
```

여기서 $w,\widehat v,v,\theta_V$는 각각 $W,\widehat V_B,V_B,W_V$를 벡터로 펼친 것이고, $J_{a,b}=\partial a/\partial b$는 Jacobian, $H_{\widehat v,v}$는 손실의 혼합 미분 행렬입니다. 이 식은 내부 업데이트의 해당 경로에서 $K,w$를 고정해 보는 원문 분석에 대응합니다〔4쪽, 식 (6)〕. :chatgpt-content-reference{index="33"}

> **용어:** Jacobian은 여러 출력 각각이 여러 입력 각각에 얼마나 민감한지를 모은 미분 행렬입니다.

**[저자 보고]** MAE에서는 중간의 혼합 미분이 거의 모든 점에서 0이 되어, 내부 업데이트를 통한 $W_V$의 학습 신호가 사라지는 문제가 발생합니다.

**[해석]** 여기서 얻을 교훈은 “일반적인 예측 문제에서 좋은 손실이 TTT에도 좋다”가 아니라, **외부 학습이 내부 업데이트를 어떻게 미분하는지까지 고려해야 한다**는 것입니다. 또한 MSE가 79.2%로 정확도는 가장 높지만, 최종 모델은 내적 손실을 사용하므로 이를 “정확도 최적 손실의 선택”보다는 **효율과 구현 단순성을 포함한 절충**으로 읽는 것이 타당합니다〔4쪽, Table 1; 7쪽, Section 5〕. :chatgpt-content-reference{index="34"} :chatgpt-content-reference{index="35"}

**추가 수학적 주의:** 내적 손실은 별도 크기 제약이 없으면 아래로 유계인 재구성 오차가 아닙니다. 따라서 내적 손실 감소를 곧바로 “ $V$를 더 정확하게 복원했다”는 뜻으로 해석하면 안 됩니다. 또한 원문의 정규화는 $1/(B\sqrt d)$이므로, 이를 다른 평균 방식으로 바꾸면 학습률 1.0의 의미도 달라집니다.

---

### 3.4 내부 모델: 단순화한 게이트와 depthwise 합성곱

최종 모델은 두 종류의 내부 모듈을 결합합니다.

```math
F_1(z)
=
(zW_a)\odot\text{SiLU}(zW_b),
```

```math
F_2(z)
=
\text{DWConv}_{3\times3}(z).
```

$z$는 내부 모델 입력이며, $W_a,W_b\in\mathbb{R}^{d\times d}$는 서로 다른 학습 가능한 행렬입니다. $\odot$는 원소별 곱이고,

$$
\text{SiLU}(u)=u\,\text{Sigmoid}(u)
$$

입니다. 각 TTT 블록에서 **한 head만 $F_2$를 사용하고 나머지 head는 $F_1$을 사용**합니다〔7쪽, Section 5; 13쪽, Section 9〕. :chatgpt-content-reference{index="36"} :chatgpt-content-reference{index="37"}

> **용어:** ‘게이트’는 한 특징으로 다른 특징의 통과 정도를 조절하는 장치입니다. Depthwise 합성곱은 채널마다 별도의 작은 공간 필터를 적용하므로, 일반 합성곱보다 채널 혼합이 제한되고 가볍습니다.

**왜 지역 합성곱이 전역 정보를 이용할 수 있는가?**

**[저자 보고]** 합성곱의 적용 범위는 지역적이지만, 그 필터의 업데이트에는 전체 이미지의 key–value 정보가 사용됩니다. 따라서 **전역 정보는 업데이트된 필터에, 지역 정보는 합성곱의 공간 이웃에 반영된다**는 설명입니다〔6–7쪽, Insight 6·Remark 6〕. :chatgpt-content-reference{index="38"} :chatgpt-content-reference{index="39"}

이를 더 명확하게 보이는 다음 식은 **원문의 식 (5)·(8)에서 유도한 설명용 식**입니다. 편향을 생략한 선형 depthwise 합성곱과 한 번의 전체 배치 업데이트를 가정하면,

```math
W_{\delta,c}^{*}
=
W_{\delta,c}^{0}
+
\frac{\eta}{N\sqrt d}
\sum_{i=1}^{N}
K_{i+\delta,c}V_{i,c},
```

```math
O_{i,c}
=
\sum_{\delta\in\Delta}
W_{\delta,c}^{*}Q_{i+\delta,c}.
```

$i$는 공간 위치, $c$는 채널, $\Delta$는 $3\times3$ 필터의 아홉 위치 오프셋, $\delta$는 그중 하나입니다. 경계 위치는 사용하는 패딩 규칙에 따릅니다.

첫 번째 식의 $\sum_i$가 보여주듯, **특정 위치의 출력에 쓰이는 필터가 이미지 전체의 통계에 의해 달라집니다.** 다만 이것이 모든 토큰 쌍의 정보를 손실 없이 보존한다는 뜻은 아닙니다.

---

### 3.5 전체 모델 구조

Figure 2의 구조는 Transformer의 정규화·잔차연결·FFN을 유지하고 attention 자리를 TTT로 바꾼 형태입니다.

```math
x_l'
=
x_{l-1}
+
\text{TTT}_l\!\left(\text{Norm}(x_{l-1})\right),
```

```math
x_l
=
x_l'
+
\text{FFN}_l\!\left(\text{Norm}(x_l')\right).
```

$l$은 블록 번호, $x_{l-1}$는 입력, $x_l'$는 TTT 이후 특징, $x_l$은 블록 출력입니다. 위 식은 Figure 2를 수식으로 표현한 것입니다〔3쪽, Figure 2〕. :chatgpt-content-reference{index="40"}

> **용어:** 정규화는 특징의 크기와 분포를 조정하는 연산이고, 잔차연결은 입력을 출력에 더하는 연결입니다. FFN은 주로 각 토큰의 채널 특징을 변환하는 신경망입니다.

| 모델군 | 구조 | 세부 설정 |
|---|---|---|
| **ViT³-T/S/B** | 비계층형, 패치 크기 16, 모두 12블록 | 특징 차원 192/384/768, head 수 6/6/12 |
| **H-ViT³-T** | 4단계 계층형 | 단계별 차원 64/128/320/512, 블록 수 1/3/9/4 |
| **H-ViT³-S** | 4단계 계층형 | 단계별 차원 64/128/320/512, 블록 수 2/6/18/8 |
| **H-ViT³-B** | 4단계 계층형 | 단계별 차원 96/192/448/640, 블록 수 2/6/18/8 |
| **DiT³-S/B** | 이미지 생성용 DiT의 attention 교체 | 모두 12블록, 차원 384/768, head 수 6/12, 패치 설정 8·4·2 |

위 설정은 13쪽 Tables 11–13에 제시됩니다. 위치 정보에는 conditional positional encoding을 사용합니다. 따라서 최종 모델의 성능을 **TTT 업데이트 하나만의 효과로 모두 귀속할 수는 없습니다.** :chatgpt-content-reference{index="41"}

> **용어:** ‘계층형’은 단계가 진행되면서 공간 해상도를 줄이고 특징 차원을 늘리는 구조입니다. Conditional positional encoding은 입력 특징에 조건화된 방식으로 위치 정보를 부여합니다.

### 3.6 선형 복잡도의 정확한 의미

**[저자 보고]** 내부 모델이 토큰 수에 대해 선형 복잡도이고 내부 업데이트 횟수가 고정되어 있으면, TTT도 시간·메모리 복잡도 $O(N)$을 갖습니다〔3쪽〕. :chatgpt-content-reference{index="42"}

**[해석]** 여기에는 특징 차원, 내부 모델 크기, 업데이트 횟수를 고정한다는 조건이 있습니다. 또한 작은 내부 모듈이라도 key에 대한 순전파, 내부 역전파, query에 대한 순전파가 필요합니다. 저자들은 일반적인 내부 모듈의 한 번 업데이트를 약 **4회 순전파 상당의 연산량**으로 설명하며, 이는 전체 ViT³가 일반 모델보다 정확히 4배 비싸다는 뜻은 아닙니다〔5쪽, Remark 4〕. :chatgpt-content-reference{index="43"}

---

## 4. 성능 결과: 개선된 부분과 남은 격차

### 4.1 학습·평가 조건

분류 학습은 ImageNet-1K에서 처음부터 300 epochs, AdamW, 외부 배치 크기 4,096, 초기 학습률 $4\times10^{-3}$, weight decay 0.05를 사용합니다. RandAugment, Mixup, CutMix, random erasing을 적용하며, MESA 추가 결과도 별도로 보고합니다〔7쪽, Section 5.1〕. :chatgpt-content-reference{index="44"}

> **용어:** 위 학습률은 전체 모델의 **외부 학습률**로, 내부 학습률 1.0과 다릅니다. MESA는 논문에서 추가 적용한 과적합 완화 학습 전략이며, 표의 $\ddagger$가 적용 여부를 표시합니다〔6쪽, Table 5〕. :chatgpt-content-reference{index="45"}

### 4.2 대표 정량 결과

| 과제·조건 | ViT³ 계열 | 비교 대상 | 수치 차이와 해석 | 원문 |
|---|---:|---:|---|---|
| ImageNet, 비계층형 Tiny | ViT³-T **76.5%** | DeiT-T 72.2% | **+4.3%p**. 표에서 둘 다 6M·1.2G로 보고되지만 세부 구성까지 동일하지는 않다. | 6쪽, Table 7. :chatgpt-content-reference{index="46"} |
| ImageNet, 비계층형 Small | ViT³-S **81.6%** | DeiT-S 79.8% | **+1.8%p**. 24M·4.8G 대 22M·4.6G로 자원이 조금 더 많다. | 6쪽, Table 7. :chatgpt-content-reference{index="47"} |
| ImageNet, 계층형 Small, MESA 없음 | H-ViT³-S **84.4%** | VMamba-S 83.6% | **+0.8%p**. 54M·8.8G 대 50M·8.7G. | 6쪽, Table 5. :chatgpt-content-reference{index="48"} |
| ImageNet, Base, 양쪽 MESA 적용 | H-ViT³-B $^\ddagger$ **85.5%** | MILA-B $^\ddagger$ 85.3% | **+0.2%p**. 작은 차이이며 통계적 유의성은 알 수 없다. | 6쪽, Table 5. :chatgpt-content-reference{index="49"} |
| COCO, Mask R-CNN 1×, Base | **50.0 AP $^b$ /44.6 AP $^m$** | VMamba-B 49.2/43.9 | **+0.8/+0.7**. FLOPs는 510G 대 485G. | 7쪽, Table 8. :chatgpt-content-reference{index="50"} |
| COCO, Mask R-CNN 3×, Small | **50.5/45.0** | MILA-S 50.5/44.9 | 상자 검출은 동률, 마스크는 **+0.1**. 일관된 큰 우위라고 보기 어렵다. | 7쪽, Table 8. :chatgpt-content-reference{index="51"} |
| ADE20K, Base | **51.7 mIoU** | VMamba-B 51.0 / TransNeXt-B 53.0 | VMamba보다 **+0.7**, TransNeXt보다 **−1.3**. | 8쪽, Table 9. :chatgpt-content-reference{index="52"} |
| ImageNet 생성, Small/2 | DiT³-S/2 **FID 62.65** | DiT-S/2 68.40 | FID **5.75 감소**. 파라미터·FLOPs는 증가한다. | 8쪽, Table 10. :chatgpt-content-reference{index="53"} |
| ImageNet 생성, Base/2 | DiT³-B/2 **FID 39.31** | DiT-B/2 43.47 | FID **4.16 감소**. 134M·23.35G 대 130M·23.01G. | 8쪽, Table 10. :chatgpt-content-reference{index="54"} |

> **용어:** Top-1은 가장 높은 점수를 준 클래스가 정답인 비율입니다. AP는 검출의 정밀도–재현율 관계를 요약한 점수이며, $b$는 상자, $m$은 마스크입니다. mIoU는 클래스별 예측 영역과 정답 영역의 겹침 비율을 평균한 값입니다. FID는 생성 이미지와 실제 이미지의 특징 분포 차이를 평가하며 **낮을수록 좋습니다**.

**[저자 보고]** 생성 실험의 여섯 가지 대응 설정 모두에서 DiT³의 FID가 DiT보다 낮습니다. 그러나 기준 DiT의 IS·precision·recall은 표에서 빠져 있으므로, **그 세 지표까지 개선되었다고 말할 수는 없습니다**〔8쪽, Table 10〕. :chatgpt-content-reference{index="55"}

**[해석]** 결과의 가장 균형 잡힌 표현은 “여러 선형 복잡도 모델에 대해 경쟁력이 있고, 일부 강한 Transformer와의 격차를 줄였다”입니다. “모든 모델·규모·과제에서 우수하다”는 요약은 표의 동률과 열세 사례를 지워버립니다.

---

## 5. 가장 중요한 그림 세 개와 해석

### Figure 1 — 논문의 개념적 기여〔1쪽〕

!:chatgpt-content-reference{index="106"}[ViT³ 논문 Figure 1 발췌](sandbox:/mnt/data/vit3_figure_1.png)

**[저자 보고]** Softmax attention은 입력 토큰 수만큼 커지는 내부 표현, 선형 attention은 $K^\top V$로 구성한 선형 상태, TTT는 학습 업데이트로 구성한 내부 신경망으로 대비됩니다. :chatgpt-content-reference{index="56"}

**[해석]** 이 그림의 핵심은 attention을 없앤다는 선언보다, **입력 정보를 저장하고 이용하는 내부 모델의 설계 공간을 넓힌다**는 데 있습니다. 빨간 역방향 화살표는 테스트 시에도 내부 업데이트 계산이 필요함을 보여줍니다. 다만 그림의 ‘압축·기억’ 설명은 작동 원리를 이해하는 관점이지, 모든 정보를 잘 기억한다는 보장은 아닙니다.

### Figure 3 — 깊이 확장의 병목〔5쪽〕

!:chatgpt-content-reference{index="107"}[ViT³ 논문 Figure 3 발췌](sandbox:/mnt/data/vit3_figure_3.png)

**[저자 보고]** 내부 모델이 깊어질수록 왼쪽의 학습 손실이 높고, 오른쪽의 평가 정확도도 낮아집니다. 저자들은 이를 최적화 어려움으로 해석합니다. :chatgpt-content-reference{index="57"}

**[해석]** 학습 데이터에서도 더 잘 맞추지 못하므로, 단순한 “모델이 커져 과적합됐다”는 설명보다 **과소적합 또는 최적화 실패**가 더 설득력 있습니다. 그러나 이 그림은 외부 초기값 학습의 문제와 내부 업데이트의 문제를 분리하지 않으며, 기울기 폭발·소실을 직접 측정한 그림도 아닙니다〔관련 가설: 6쪽, Remark 5〕. :chatgpt-content-reference{index="58"}

> **용어:** 과적합은 학습 데이터에는 잘 맞지만 새로운 데이터에서 성능이 나쁜 상태이고, 과소적합은 학습 데이터의 패턴조차 충분히 학습하지 못한 상태입니다.

### Figure 4 — 고해상도 계산 효율성〔8쪽〕

!:chatgpt-content-reference{index="108"}[ViT³ 논문 Figure 4 발췌](sandbox:/mnt/data/vit3_figure_4.png)

**[저자 보고]** 해상도가 커질수록 처리량과 메모리 사용량에서 ViT³-T의 상대적 이점이 커집니다. $1248^2$ 해상도, 6,084토큰에서 4.6배 처리량과 90.3% 메모리 감소를 보고합니다. 왼쪽 세로축은 로그 척도입니다. :chatgpt-content-reference{index="59"}

**[해석]** 이 그림은 **계산적 확장성**의 근거이지, 고해상도에서 정확도까지 유지된다는 근거가 아닙니다. 또한 비교 대상은 해당 구현의 DeiT-T이며, 최적화된 모든 Softmax attention 구현에 대한 같은 비율의 우위를 보장하지 않습니다.

---

## 6. 통계적으로 취약한 부분과 비교할 수 없는 수치

### 6.1 결과를 읽을 때 붙여야 할 경고

| 표시 | 취약점·비교 문제 | 허용되는 결론과 허용되지 않는 결론 |
|---|---|---|
| **[통계주의] 반복 실험 정보 부족** | 주요 성능표에 seed별 결과, 표준편차, 신뢰구간이 제시되지 않는다. | 0.1–0.3%p 차이를 관찰했다고 말할 수 있으나, 통계적으로 유의한 개선이라고 단정할 수 없다. 이는 “한 번만 실험했다”는 뜻은 아니다〔Tables 1–10〕. :chatgpt-content-reference{index="60"} :chatgpt-content-reference{index="61"} |
| **[통계주의] 발산 전 최고값** | Table 2의 4회 업데이트 57.0%와 Table 3의 일부 큰 학습률 결과는 발산 전 최고 정확도다. | 정상 수렴한 최종 모델의 성능과 같은 종류의 숫자로 취급하면 안 된다〔4쪽, Tables 2–3〕. :chatgpt-content-reference{index="62"} |
| **[조건차이] MESA** | H-ViT³-B는 84.9%, MESA 적용 시 85.5%다. | 추가 0.6%p를 TTT 구조만의 기여로 돌릴 수 없다〔6쪽, Table 5〕. :chatgpt-content-reference{index="63"} |
| **[조건차이] 모델 크기·학습법** | 동일한 T/S/B 이름이라도 파라미터·FLOPs·세부 학습법이 완전히 같지 않다. | 모델 수준의 성능–비용 비교는 가능하지만, 차이 전체를 토큰 혼합 연산 하나의 인과효과로 볼 수 없다〔6쪽, Tables 5·7〕. :chatgpt-content-reference{index="64"} :chatgpt-content-reference{index="65"} |
| **[직접비교불가] 서로 다른 학습 일정** | COCO 1×와 3×는 서로 다른 학습 조건이다. | 두 구간의 수치를 섞어서 모델 순위를 정하면 안 된다〔7쪽, Table 8〕. :chatgpt-content-reference{index="66"} |
| **[정보부족] 생성 평가의 세부 조건** | 첨부 문서는 FID-50K·해상도를 제시하지만, 비교에 필요한 전체 학습 예산과 샘플링 설정을 충분히 열거하지 않는다. | Table 10의 저자 보고 비교를 소개할 수는 있으나, 외부 논문의 다른 FID와 곧바로 순위를 매길 수 없다〔8쪽, Section 5.4〕. :chatgpt-content-reference{index="67"} |
| **[조건차이] 효율성 측정** | Figure 4는 RTX3090의 특정 모델 비교이며, 정밀도·배치 크기·최적화 커널 사용 여부 등 재현에 필요한 상세 조건이 충분하지 않다. | 4.6배·90.3%를 모든 하드웨어와 attention 구현에 일반화할 수 없다〔8쪽〕. :chatgpt-content-reference{index="68"} |
| **[인과근거 제한] 단일 기준 모델 중심 설계 탐색** | 여섯 지침의 핵심 탐색은 DeiT-S 기반 ImageNet 실험이다. | 내부 최적 설정이 검출·분할·생성·다른 데이터 규모에서도 동일하다는 결론은 추가 실험이 필요하다〔3쪽, Section 4〕. :chatgpt-content-reference{index="69"} |

> **용어:** 신뢰구간은 추정치의 통계적 불확실성을 나타내는 구간입니다. 작은 성능 차이를 비교할 때는 두 모델이 **같은 이미지에서 어떤 예측을 했는지**를 이용하는 대응 비교와, 학습 seed에 따른 변동을 함께 확인해야 합니다.

### 6.2 Softmax attention의 메모리를 무조건 $O(N^2)$라고 하면 안 된다

**[외부 연구]** *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*는 큰 attention 행렬을 GPU 메모리에 통째로 저장하지 않는 방식으로 **정확한 Softmax attention의 메모리 사용을 토큰 수에 대해 선형으로 줄일 수 있음**을 보여줍니다. 계산량의 이차 증가와 메모리의 이차 증가는 구분해야 합니다. 따라서 Figure 4의 메모리 이점을 평가하려면 FlashAttention 계열과 같은 최적화된 기준선을 포함해야 합니다. :chatgpt-content-reference{index="70"}

### 6.3 원문 내부의 참고문헌 표기 불일치

첨부본에는 **Table 8의 “PolaFormer-T [20]”와 참고문헌 [20]의 FLatten Transformer**, **Table 9의 “FasterViT-2 [25]”와 참고문헌 [25]의 Neighborhood Attention Transformer**가 일치하지 않는 부분이 있습니다. 여기서는 표의 값을 임의로 수정하지 않았으며, 해당 행의 비교 출처를 재현하려면 원 저작을 다시 대조해야 합니다〔7–9쪽〕. :chatgpt-content-reference{index="71"} :chatgpt-content-reference{index="72"} :chatgpt-content-reference{index="73"} :chatgpt-content-reference{index="74"}

---

## 7. 일반화 성능 향상 가능성: 무엇이 입증되었는가?

### 7.1 서로 다른 종류의 ‘일반화’를 구분해야 한다

| 일반화의 의미 | 이 논문의 근거 | 판단 |
|---|---|---|
| 학습하지 않은 같은 데이터셋의 이미지에서 좋은 성능 | ImageNet 검증 정확도 | **직접 평가됨** |
| 여러 시각 과제에 같은 설계가 유용함 | 분류·검출·분할·생성 결과 | **아키텍처의 범용성에 대한 근거가 있음** |
| 더 긴 토큰·고해상도 입력을 처리함 | Figure 4의 처리량·메모리 | **계산 효율은 평가됨; 정확도 외삽은 별도** |
| 노이즈·날씨·스타일·센서 변화에서도 잘 작동함 | 해당 평가가 보고되지 않음 | **미검증** |
| 새로운 범주·도메인에서 추가 학습 없이 잘 작동함 | 해당 평가가 보고되지 않음 | **미검증** |
| 연속적으로 변하는 테스트 환경에 적응함 | 지속적인 테스트 스트림 평가가 없음 | **미검증** |

이는 원문의 실제 평가 범위에 근거한 구분입니다〔7–8쪽, Sections 5.1–5.5〕. :chatgpt-content-reference{index="75"} :chatgpt-content-reference{index="76"}

> **용어:** ‘분포 변화’는 학습 때와 테스트 때의 데이터 특성이 달라지는 현상입니다. OOD는 학습 분포 밖의 데이터를 뜻합니다. ‘제로샷’은 해당 과제나 범주에 대한 별도 학습 없이 수행하는 설정입니다.

### 7.2 일반화 개선을 기대할 수 있는 이유 — 그러나 아직 가설이다

**[해석 1: 입력별 조건화]** 내부 파라미터가 입력에 따라 달라지므로, 고정 필터보다 이미지별 문맥에 맞는 처리를 할 가능성이 있습니다. 다만 입력 변화가 유용한 문맥이 아니라 잡음이라면, 업데이트가 잡음을 증폭할 수도 있습니다. 원문의 입력별 업데이트는 이러한 가능성을 제공하지만, 어느 쪽이 우세한지는 OOD 실험으로 확인해야 합니다〔3쪽, 식 (5)〕. :chatgpt-content-reference{index="77"}

**[해석 2: 지역성과 전역성의 결합]** 합성곱의 지역적 구조는 가까운 픽셀·특징 사이의 관계를 활용하고, 입력 전체에서 얻은 필터 업데이트는 전역 문맥을 반영합니다. 이는 유용한 시각적 귀납 편향일 가능성이 있으나, 특정 스타일이나 질감에 더 의존하게 될 가능성도 배제할 수 없습니다〔6–7쪽, Insight 6〕. :chatgpt-content-reference{index="78"} :chatgpt-content-reference{index="79"}

> **용어:** ‘귀납 편향’은 제한된 데이터에서 학습하기 위해 모델이 구조적으로 선호하는 규칙입니다. 합성곱의 “가까운 위치를 같은 필터로 처리한다”는 성질이 한 예입니다.

**[해석 3: 계산 절약의 간접 효과]** 같은 자원으로 더 높은 해상도나 더 넓은 문맥을 볼 수 있다면 일반화에 유리할 수 있습니다. 그러나 **계산비용 감소 → 일반화 향상**은 자동으로 성립하는 인과관계가 아니며, 정확도·강건성까지 동일 자원에서 비교해야 합니다〔8쪽, Figure 4〕. :chatgpt-content-reference{index="80"}

### 7.3 핵심 이론적 질문: 내부 손실과 실제 과제 손실이 정렬되는가?

다음은 **ViT³가 증명한 정리가 아니라**, 일반화 연구를 설계하기 위한 1차 근사입니다.

```math
R_{\text{target}}(w-\eta g_{\text{inner}})
\approx
R_{\text{target}}(w)
-
\eta
\left\langle
\nabla_wR_{\text{target}}(w),
g_{\text{inner}}
\right\rangle.
```

$R_{\text{target}}$는 목표 환경에서의 미분 가능한 과제 손실의 기댓값, $w$는 내부 파라미터, $g_{\text{inner}}$는 내부 손실의 기울기, $\langle\cdot,\cdot\rangle$는 내적입니다. 충분히 작은 업데이트의 1차 근사에서는 두 기울기가 같은 방향을 향할 때 목표 손실 감소를 기대할 수 있습니다.

**[해석]** ViT³의 혼합 미분 분석은 “외부 학습 신호가 전달되는가?”를 다룹니다. 위 질문은 “그 업데이트가 새로운 환경에서도 실제 과제를 개선하는가?”를 다룹니다. **전자는 후자를 보장하지 않습니다.** 또한 실제 설정의 $\eta=1.0$에서 고차항을 무시할 수 있는지는 별도 검증이 필요합니다.

---

## 8. 2020년 이후 관련 연구 비교 및 참고자료

아래는 이 답변의 외부 비교에 사용한 **논문 전체 제목과 1차 출처**입니다. 서로 다른 데이터·과제의 수치를 하나의 순위로 합치지 않고, **목적·업데이트 대상·일반화 근거**를 비교했습니다.

### 8.1 분포 변화에 적응하는 TTT와 ViT³의 차이

| 연구·출처 | 해당 저자의 핵심 결과·방법 | ViT³와 비교한 해석 |
|---|---|---|
| **Sun et al., ICML 2020 — *Test-Time Training with Self-Supervision for Generalization under Distribution Shifts*** | 테스트 입력에 대한 자기지도 과제로 모델을 조정해 분포 변화에 대응한다. :chatgpt-content-reference{index="81"} | 이름은 같지만 연구 목표가 다르다. 이 연구는 분포 변화 일반화가 중심이고, ViT³는 효율적인 내부 시퀀스 연산의 설계가 중심이다. |
| **Wang et al., ICLR 2021 — *Tent: Fully Test-Time Adaptation by Entropy Minimization*** | 예측 엔트로피를 줄이도록 정규화 통계와 채널별 affine 파라미터를 조정한다. :chatgpt-content-reference{index="82"} | ViT³의 목표는 예측 확신 증가가 아니라 내부 key–value 특징에 따른 연산 구성이다. Tent의 강건성 결과를 ViT³의 근거로 가져올 수 없다. |
| **Gandelsman et al., NeurIPS 2022 — *Test-Time Training with Masked Autoencoders*** | 테스트 이미지의 가려진 부분을 복원하는 자기지도 학습으로 여러 분포 변화 벤치마크에서 개선을 보고한다. :chatgpt-content-reference{index="83"} | ViT³가 일반화 향상을 주장하려면 참고해야 할 직접적인 평가 계열이다. 픽셀 복원 목표와 내부 value 특징 목표는 동일하지 않다. |

> **용어:** ‘자기지도’는 사람이 붙인 정답 대신 입력 자체로 학습 목표를 구성하는 방식입니다. 엔트로피는 예측의 불확실성을 나타내지만, 이를 줄였다고 반드시 정답에 가까워지는 것은 아닙니다. Masked autoencoder는 가린 입력 부분을 복원하도록 학습하는 모델입니다.

### 8.2 효율적 attention·시각 backbone 연구와의 관계

| 연구·출처 | 해당 저자의 핵심 방법 | ViT³와 비교한 해석 |
|---|---|---|
| **Katharopoulos et al., ICML 2020 — *Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention*** | 특징 변환과 행렬곱의 결합법칙으로 토큰 수에 대한 선형 계산을 구현한다. :chatgpt-content-reference{index="84"} | ViT³의 직접적인 출발점이다. ViT³는 선형 상태 구성법을 더 유연한 내부 모듈과 업데이트로 확장한다. |
| **Schlag et al., ICML 2021 — *Linear Transformers Are Secretly Fast Weight Programmers*** | 선형 attention과 빠르게 갱신되는 가중치 메모리의 관계를 보이고, 기존 연결을 수정하는 delta-rule 계열 업데이트를 제안한다. :chatgpt-content-reference{index="85"} | 입력에 따라 가중치를 구성한다는 관점은 ViT³ 이전부터 존재했다. ViT³의 차별점은 시각 설계 공간의 체계적 실험이다. |
| **Dao et al., NeurIPS 2022 — *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*** | 정확한 attention의 메모리 접근을 최적화한다. :chatgpt-content-reference{index="86"} | 표현 방식을 바꾸는 ViT³와 구현을 최적화하는 접근은 구분된다. 실제 효율성 비교에서는 반드시 중요한 기준선이다. |
| **Han et al., ICCV 2023 — *FLatten Transformer: Vision Transformer using Focused Linear Attention*** | 선형 attention의 집중 능력과 특징 다양성 문제를 분석하고 특징 변환·rank 복원 모듈을 제안한다. :chatgpt-content-reference{index="87"} | 표현력과 지역 정보 보완이라는 문제의식이 겹친다. ViT³는 이를 내부 모델과 내부 학습의 설계 문제로 다룬다. |
| **Liu et al., 2024 — *VMamba: Visual State Space Model*** | 여러 방향의 2D selective scan으로 시각 정보를 수집하는 선형 시간 backbone을 제안한다. :chatgpt-content-reference{index="88"} | ViT³의 전체 배치 업데이트는 순차적 스캔과 다른 정보 집계 방식이다. 둘 중 무엇이 좋은지는 입력 구조와 과제에 따라 평가해야 한다. |
| **Han et al., NeurIPS 2024 — *Demystify Mamba in Vision: A Linear Attention Perspective*** | Mamba와 선형 attention의 차이를 분석하고 MILA를 제안한다. :chatgpt-content-reference{index="89"} | ViT³와 같은 저자 계열의 설계 분석 흐름이다. 공정한 비교에서는 게이트·블록 구조·위치 정보의 기여도 맞춰야 한다. |
| **Shi, CVPR 2024 — *TransNeXt: Robust Foveal Visual Perception for Vision Transformers*** | 전역·지역 정보 교환과 convolutional GLU를 결합하고 강건성 평가도 보고한다. :chatgpt-content-reference{index="90"} | ViT³가 실제로 뒤처지는 비교 대상이기도 하다. 표준 정확도뿐 아니라 OOD 평가까지 비교해야 일반화 우위를 논할 수 있다. |

> **용어:** ‘Fast weights’는 입력이나 문맥에 따라 빠르게 바뀌는 가중치입니다. ‘상태공간 모델’은 이전 정보를 내부 상태에 누적해 처리하는 모델이며, selective scan은 입력에 따라 그 갱신 방식을 달리하는 연산입니다. ‘Rank’는 행렬이 표현할 수 있는 독립적인 방향의 수입니다.

### 8.3 현대 TTT 레이어와 2025–2026년 연구

| 연구·출처 | 해당 저자의 핵심 방법·결과 | ViT³와 비교한 해석 |
|---|---|---|
| **Sun et al., 2024 공개·ICML 2025 — *Learning to (Learn at Test Time): RNNs with Expressive Hidden States*** | 은닉 상태를 선형 모델이나 MLP로 두고 자기지도 업데이트로 갱신하는 TTT 레이어를 제안한다. :chatgpt-content-reference{index="91"} | ViT³의 직접적 토대다. 언어에서 얻은 순차·미니배치 설계를 시각 데이터에 그대로 옮기지 않고 검토했다는 것이 ViT³의 기여다. |
| **Dalal et al., 2025 — *One-Minute Video Generation with Test-Time Training*** | 사전학습 Transformer에 TTT 레이어를 추가해 긴 영상의 일관성을 개선한다. :chatgpt-content-reference{index="92"} | 긴 시각 문맥에 대한 가능성을 보여주지만, 특정 영상 데이터와 평가의 결과를 ViT³의 이미지 분류 일반화로 환산할 수 없다. |
| **Zhang et al., 2025 공개 — *Test-Time Training Done Right*** | LaCT는 매우 큰 토큰 묶음으로 업데이트하여 하드웨어 활용과 비선형 상태 확장을 개선한다. :chatgpt-content-reference{index="93"} | ViT³의 전체 배치 선호와 방향이 맞닿아 있다. 다만 큰 문맥·다른 과제에서의 결과이므로 “큰 배치가 모든 시각 TTT에 최적”이라는 증거는 아니다. |
| **Liu et al., 2026 — *Test-Time Training with KV Binding Is Secretly Linear Attention*** | ViT³를 포함한 TTT를 확장된 선형 attention 형태로 재해석하고 단순한 기억·검색 설명을 재검토한다〔해당 논문 6쪽, Section 5.4〕. :chatgpt-content-reference{index="94"} | ViT³의 후속 영향이 실제로 확인되는 사례다. 다만 복잡한 게이트·특징 변환을 포함한 재표현을 단순한 $QK^\top V$와 동일시하면 안 된다. |
| **Feng et al., 2026 — *In-Place Test-Time Training*** | 기존 LLM의 MLP 마지막 투영을 빠른 가중치로 활용하고, 내부 목표를 다음 토큰 예측 과제에 정렬한다. :chatgpt-content-reference{index="95"} | ViT³ 후속 연구에서도 “내부 목표가 최종 과제에 맞는가?”를 우선 검토할 필요성을 시사한다. 시각 일반화 성능을 입증한 결과는 아니다. |

### 8.4 특히 주목할 2026년의 재해석

**[외부 저자 보고]** *Test-Time Training with KV Binding Is Secretly Linear Attention*의 Table 1은 ViTTT 기준 79.34%, 내부 gradient ascent 변형 79.61%를 보고합니다. 그러나 부록의 분류 실험은 **ViTTT-B를 60 epochs 학습한 설정**으로, ViT³ 원문의 300-epoch 결과와 직접 비교할 수 없습니다〔해당 논문 4쪽 Table 1, 12쪽 Appendix A〕. :chatgpt-content-reference{index="96"}

**[해석]** 이 결과는 “이미 학습된 ViT³에서 테스트 시 업데이트 방향만 뒤집어도 안전하다”는 뜻이 아닙니다. 더 중요한 시사점은 **내부 손실을 잘 줄이는 것과 최종 과제 성능을 개선하는 것을 동일시하지 말아야 한다**는 것입니다. 이 재해석이 ViT³의 벤치마크 결과를 무효화하지는 않지만, “성능이 좋은 이유는 key–value를 더 정확히 기억해서다”라는 설명에는 추가적인 인과 검증이 필요합니다.

---

## 9. 문서가 답하지 않는 질문

아래는 원문 결과로 답을 확정할 수 없는 질문들입니다.

| 남은 질문 | 필요한 추가 근거 |
|---|---|
| **입력별 업데이트 자체가 얼마나 기여하는가?** | 동일 구조의 업데이트 없는 모델, 입력과 무관한 업데이트, key–value 대응을 섞은 대조군 |
| **노이즈·스타일·자연적 분포 변화에서 더 강건한가?** | ImageNet-C/R/A 등에서 동일 학습·계산예산 비교 |
| **토큰 수가 늘어도 정확도와 정보 보존 능력이 유지되는가?** | 해상도별 정확도, 작은 객체·복잡한 장면·장거리 관계 평가 |
| **왜 깊은 내부 모델의 최적화가 어려운가?** | 내부·외부 기울기 크기, 초기값 민감도, 곡률·안정성 측정과 분리 실험 |
| **내부 손실 개선이 최종 성능 개선을 예측하는가?** | 내부 손실과 외부 과제 손실·정확도 사이의 관계 분석 |
| **테스트 때만 업데이트 횟수를 늘리는 것이 유익한가?** | 학습 시 업데이트 횟수와 테스트 시 횟수를 분리한 실험 |
| **연속적인 환경 변화에서 상태를 유지해야 하는가, 초기화해야 하는가?** | 순서가 있는 테스트 스트림, 상태 초기화 정책, 장기 성능 저하 평가 |
| **최적화된 Softmax attention 대비 실제 이점은 어느 정도인가?** | 동일 하드웨어·정밀도·배치·커널 최적화 조건의 시간·메모리·정확도 비교 |

이 질문들은 원문의 제한된 설계 탐색 범위와 평가 범위에서 도출한 것입니다. 저자도 내부 optimizer, 내부 데이터 증강, Transformer 내부 모델 등을 탐색하지 않았다고 명시합니다〔12쪽, Section 7〕. :chatgpt-content-reference{index="97"}

---

## 10. 결론과 후속 연구 방향

### 10.1 저자들이 제시한 시사점과 후속 방향

**[저자 보고]** 저자들은 ViT³를 완결된 최종 해법보다는 **시각 TTT를 위한 강한 기준 모델과 설계 지침**으로 제시합니다. 후속 방향은 시각 데이터에 맞는 미니배치 구성, 가볍고 표현력 높은 내부 모델, 깊은 내부 모델의 최적화, 내부 optimizer·데이터 증강·새로운 내부 구조 탐색입니다〔4쪽 Remark 2, 5–6쪽 Remarks 4–5, 12쪽 Section 7〕. :chatgpt-content-reference{index="98"} :chatgpt-content-reference{index="99"} :chatgpt-content-reference{index="100"} :chatgpt-content-reference{index="101"}

이는 구체적인 일정이나 이미 진행 중인 프로젝트를 밝힌 실행 계획이라기보다, **논문에서 제안한 연구 방향**입니다.

### 10.2 추가 제안: 일반화 개선을 검증하는 연구의 우선순위

#### A. 먼저 ‘구조 효과’와 ‘적응 효과’를 분리해야 한다

**[제안]** 같은 backbone·위치 정보·파라미터·학습 예산에서 내부 업데이트 유무를 비교하는 것이 우선입니다. 이때 **업데이트 없는 구조를 다시 학습한 대조군**과 **학습된 모델에서 업데이트만 제거하는 진단 실험**은 다른 질문에 답하므로 둘 다 필요합니다.

전자는 적응 연산 없이도 같은 성능을 배울 수 있는지, 후자는 학습된 모델이 실제로 업데이트에 의존하는지를 확인합니다.

#### B. 일반화는 평균 정확도뿐 아니라 ‘적응 실패’까지 평가해야 한다

**[제안]** 표준 검증셋과 분포 변화 평가를 함께 수행하고, 평균 성능뿐 아니라 **업데이트 때문에 오히려 틀리게 된 입력의 비율**, 변화가 심한 환경의 최악 성능, 예측 확신의 신뢰성도 측정해야 합니다. 테스트 환경의 정답 라벨로 내부 학습률이나 업데이트 횟수를 선택하지 않도록 별도의 개발 절차도 필요합니다.

여러 seed의 재학습 결과와 이미지별 대응 비교를 제공하면, 작은 향상이 반복 가능한지 판단할 수 있습니다.

#### C. 내부 손실은 ‘미분 가능성’과 ‘과제 정렬’을 함께 만족해야 한다

**[제안]** 혼합 미분이 0이 되지 않는다는 조건만으로 손실 선택을 끝내지 말아야 합니다. 새로운 환경에서 내부 업데이트가 실제 과제 손실을 줄이는지, 잡음을 따라가는지, 특징 크기를 과도하게 키우는지까지 살펴봐야 합니다.

이를 위해 서로 다른 변환을 적용한 이미지의 일관성, 부드러운 강건 손실, 업데이트 크기 제한 등을 비교할 수 있습니다. **이들이 ViT³의 일반화를 개선한다는 것은 현재 결과가 아니라 검증할 가설**입니다.

#### D. 깊이 확장은 일반적인 잔차연결보다 내부 학습 특화 설계가 필요하다

**[제안]** 원문에서는 단순 잔차연결과 초기화 변경의 효과가 제한적이었습니다. 따라서 단순히 층을 더 쌓는 대신, 일부 가중치만 업데이트하는 구조, 내부 단계별 크기 제어, 학습된 업데이트 규칙 등을 검토하는 편이 더 타당합니다〔6쪽, Table 6〕. :chatgpt-content-reference{index="102"}

중요한 비교 기준은 “더 큰 내부 모델” 자체가 아니라 **같은 전체 연산 예산에서 얻는 정확도와 일반화**입니다.

#### E. 고해상도 효율성을 고해상도 일반화로 연결하는 실험이 필요하다

**[제안]** Figure 4를 해상도별 정확도·검출 성능·분할 성능으로 확장해야 합니다. 훈련보다 큰 해상도, 객체 수 증가, 공간 배치 변화에 대해 성능을 측정하면, 내부 상태가 더 긴 문맥을 실제로 유용하게 요약하는지 확인할 수 있습니다.

상태 크기를 토큰 수에 따라 키우는 방법도 연구할 수 있지만, 그 경우에는 **더 이상 동일한 조건의 $O(N)$ 주장으로 묶지 말고**, 상태 크기까지 포함한 계산 복잡도를 다시 제시해야 합니다.

### 최종 판단

**ViT³의 가장 중요한 기여는 “테스트 시 학습을 하면 일반화가 좋아진다”는 증명이 아니라, “시각 TTT의 내부 학습과 내부 구조를 어떻게 설계해야 경쟁력 있는 모델이 되는가”에 대한 실증적 지침입니다.** 여러 시각 과제에서의 성능과 고해상도 효율성은 의미 있는 결과지만, 분포 변화 강건성·지속 적응·테스트 시 계산량 확대의 이익은 별도로 검증해야 합니다〔8쪽 결론, 12쪽 한계〕. :chatgpt-content-reference{index="103"} :chatgpt-content-reference{index="104"}

**[해석]** 앞으로 가장 가치 있는 연구는 단순히 내부 모델을 더 크게 만드는 연구보다, **“어떤 입력 변화에서, 어떤 업데이트가, 왜 실제 과제의 일반화를 개선하거나 악화시키는가”를 통제 실험과 이론으로 밝히는 연구**라고 판단합니다.
