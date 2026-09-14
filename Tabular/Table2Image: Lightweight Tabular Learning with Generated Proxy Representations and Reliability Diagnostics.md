# Table2Image: Lightweight Tabular Learning with Generated Proxy Representations and Reliability Diagnostics

분석 대상은 첨부하신 **v3 논문 _“Table2Image: Lightweight Tabular Learning with Generated Proxy Representations and Reliability Diagnostics”_**입니다. 첨부 PDF는 제목과 저자, 그리고 `arXiv:2412.06265v3 [cs.LG] 20 Aug 2026`을 명시하고 있습니다.  

한 가지 중요한 버전 주의점이 있습니다. 웹에서 현재 검색되는 arXiv 랜딩 페이지 일부는 아직 **v2의 옛 제목 _“Table2Image: Interpretable Tabular Data Classification with Realistic Image Transformations”_**과 2025-01-23 수정일을 노출하지만, SciRate·Academus 등의 최신 인덱스는 **v3의 새 제목과 내용**을 인식합니다. 따라서 아래의 수식·페이지·Figure/Table 분석은 **사용자가 첨부한 2026년 v3 PDF를 최우선 원문**으로 삼고, 웹 검색은 관련 연구 비교와 공개 코드 검증에 사용했습니다. ([arXiv][1])

---

# 1. Executive Summary — 10문장 이내

1. **Table2Image(T2I)**는 표형 데이터를 곧바로 분류하는 대신, 먼저 학습 가능한 생성 경로를 통해 $28\times28$의 **structured proxy representation**으로 변환한 뒤 작은 CNN으로 분류하는 경량 표형 딥러닝 모델이다. 
2. 저자들의 핵심 아이디어는 이미지 자체가 필요한 것이 아니라, 표형 입력 $x$와 클래스 정보를 구조화된 중간 공간으로 보내는 **보조 표현 학습(inductive bias)**이 일반화를 도울 수 있다는 것이다.
3. Proxy target은 FashionMNIST/MNIST를 사용하는 **external mode**와, Gaussian class template + periodic pattern + sample-specific variation으로 만드는 **internal mode** 두 가지가 있다. 
4. 추가 변형인 **T2I-VIF**는 $VIF_i=1/(1-R_i^2)$를 이용하여 다중공선성이 큰 feature의 초기 가중치를 작게 설정하지만, 이후 학습에서는 이 제약을 고정하지 않는다. 
5. 외부 proxy target을 사용했을 때 OpenML-CC18에서 T2I-VIF는 평균 ACC/AUC $0.879/0.922$, TabZilla에서 $0.862/0.905$를 기록하여 여러 대형 neural baseline과 경쟁적인 성능을 보였다. 
6. 내부 proxy를 사용한 더 엄격한 비교에서는 T2I가 평균 ACC/AUC $0.876/0.936$으로 parameter-matched MLP의 $0.868/0.934$보다 좋았지만, 그 차이는 **크지 않다**. 
7. 이 논문의 중요한 두 번째 기여는 irrelevant features, noisy labels, shortcut decorrelation이라는 세 종류의 corruption을 severity별로 평가하면서 **APS/WSP뿐 아니라 DPI·LCI·SAI라는 prediction-instability 진단**을 함께 제안한 점이다. 
8. Ablation에서 성능 저하가 가장 큰 경우는 generator 자체를 제거한 경우였기 때문에, 실험적으로는 **proxy의 특정 시각적 모양보다 learned generation pathway 자체**가 더 중요한 것으로 보인다. 
9. 다만 T2I의 label-corruption instability는 오히려 높은 편이고, 자연 발생 distribution shift, regression, calibration, 실제 추론 latency 등은 검증하지 않아 “robust/generalizable model”이라는 주장은 아직 제한적이다. 
10. 따라서 이 연구의 가장 중요한 의미는 “tabular→image” 자체보다는 **작은 모델에 구조화된 stochastic auxiliary representation을 삽입하고, 평균 성능과 prediction stability를 함께 평가하는 연구 방향**을 제시했다는 데 있다.

### 용어 설명

**Proxy representation**: 실제 관측 대상의 의미 자체를 복원하려는 표현이 아니라, 최종 예측을 잘하도록 중간에 만들어 놓는 대리 표현입니다.

**Inductive bias**: 학습 전에 모델 구조에 넣어 둔 “좋은 답은 이런 형태일 가능성이 높다”는 가정입니다.

**Generalization**: 학습 데이터가 아니라 보지 못한 데이터에서 성능을 유지하는 능력입니다.

---

# 1-1. 연구의 목적과 필요성

논문이 출발하는 문제는 표형 데이터에서 세 가지 목표가 보통 **동시에 최적화되지 않는다**는 것입니다.

$$
\boxed{
\text{Predictive Performance}
+
\text{Parameter Efficiency}
+
\text{Reliability}
}
$$

최근 tabular deep learning은 Transformer, prior-data fitted network, ensemble 등으로 성능을 높여 왔지만 파라미터 수·사전학습 비용·추론 복잡성이 증가했습니다. 반대로 XGBoost, LightGBM 같은 GBDT는 여전히 높은 효율성과 강력한 성능 때문에 실무의 기본 선택지로 남아 있습니다. 논문은 DL의 end-to-end 표현 학습 장점을 유지하면서 모델을 작게 만들고, 동시에 **불완전한 학습 신호에 대한 신뢰성까지 평가**하려 합니다. 

또 하나의 문제는 기존 robustness 연구가 보통 label noise, feature corruption, shortcut, adversarial attack 등을 **서로 다른 데이터셋·severity·평가지표로 따로 연구**했다는 것입니다. 그래서 “모델 A가 noise에는 강하고 shortcut에는 약한가?” 같은 구조적 비교가 어렵습니다. 저자들은 이를 하나의 severity-controlled protocol로 통합합니다. 

### 용어 설명

**Severity-controlled evaluation**: 오염을 “있다/없다”만 비교하는 것이 아니라 5%, 10%, 15%, 20%처럼 강도를 단계적으로 증가시켜 성능 곡선을 보는 평가입니다.

**Shortcut learning**: 진짜 원인이 아닌데 학습 데이터에서 우연히 정답과 강하게 연결된 변수를 모델이 이용하는 현상입니다. 배포 환경에서 그 관계가 사라지면 성능이 급락할 수 있습니다.

---

# 2. 핵심 주장과 근거

| 핵심 주장                                                           | 논문이 제시한 근거                                            | 위치                          | 제 평가                                                           |
| --------------------------------------------------------------- | ----------------------------------------------------- | --------------------------- | -------------------------------------------------------------- |
| Structured proxy generation이 경쟁력 있는 tabular representation을 만든다 | External T2I/T2I-VIF가 OpenML·TabZilla에서 높은 평균 ACC/AUC | Table I, p.6                | **부분적으로 강함**. 평균값은 좋지만 강한 baseline과 pairwise significance는 제한적 |
| 작은 파라미터로 강한 성능-효율 trade-off를 얻는다                                | T2I 0.628M vs TuneTables 25.8M, TabM 37.9M            | Fig.2 p.7, Table IX p.11    | **맞지만 상대적 표현 필요**. FT-Transformer 0.136M, MLP 0.007M보다 T2I가 큼  |
| VIF initialization이 도움이 된다                                      | T2I-VIF $0.879/0.922$ vs T2I $0.877/0.920$            | Table I p.6, Table XIV p.13 | **효과는 매우 작음**                                                  |
| 원 표현과 VIF 표현을 같이 유지해야 한다                                        | T2I-DIR ACC 0.609, T2I-MUL 0.871, T2I-VIF 0.879       | Table XIV p.13              | **강한 ablation 근거**                                             |
| Generator pathway가 성능의 핵심 요소다                                   | Generator-free가 clean/robustness에서 가장 일관되게 하락         | Table VI p.9                | **이 논문의 가장 설득력 있는 ablation**                                   |
| T2I는 corruption에 강하다                                            | IF/NL/SD에서 높은 APS·WSP                                 | Table III, Fig.3 p.7        | **절대 성능은 좋음**, 하지만 relative degradation이 항상 느린 것은 아님           |
| 높은 성능과 안정성은 같은 개념이 아니다                                          | T2I는 낮은 SAI지만 높은 LCI                                  | Table V p.8                 | **매우 의미 있는 결과**                                                |
| multiple proxy mapping이 single prototype보다 좋다                   | 3-class subset에서 ACC $0.8876$ vs $0.8483$             | Table XII p.13              | **해당 subset에서는 강함**, 전체 클래스 수에 대한 검증은 아님                       |
| T2I pathway는 일반 encoder wrapper가 될 수 있다                         | MLP에서는 약간 향상                                          | Table VII p.9               | **일반적인 wrapper라는 주장은 약함**                                      |
| Transformer에도 적용 가능하다                                           | FT-Transformer ACC $0.865\to0.788$                    | Table VII p.9               | 오히려 **반증에 가까움**                                                |

Table VI에서 full T2I의 clean ACC는 $0.8760$이지만 generator-free는 $0.8607$로 감소합니다. 반면 proxy permutation은 $0.8747$, reconstruction-loss 제거는 $0.8740$으로 변화가 훨씬 작습니다. 

이 결과에서 저는 논문의 가장 중요한 메커니즘을 다음처럼 해석합니다.

$$
\boxed{
\text{특정 이미지 패턴 자체}
<
\text{learned stochastic bottleneck / generation pathway}
}
$$

즉 “Gaussian blob과 stripe가 특별히 좋은 representation이어서 성능이 좋다”는 결론보다, **tabular embedding을 별도의 stochastic high-dimensional representation으로 다시 재구성하도록 강제하는 학습 과정**이 정규화·표현 분리 효과를 주었을 가능성이 더 큽니다.

---

# 2-1. 해결하려는 문제 → 방법 → 모델 구조

## 2-1-1. 기본 문제 설정

입력 tabular sample을

$$
x\in\mathbb{R}^{N},
$$

class label을

$$
y\in\{0,\ldots,C-1\}
$$

라 둡니다.

$N$은 feature 개수, $C$는 class 수입니다.

일반적인 분류기는

$$
x \longrightarrow f(x)\longrightarrow \hat y
$$

이지만 Table2Image는

$$
\boxed{
x
\longrightarrow
P(x)
\longrightarrow
G(P(x),r)
\longrightarrow
\text{CNN}
\longrightarrow
\hat y
}
$$

라는 중간 생성 단계를 추가합니다. 이 전체 구조는 **Figure 1, p.3**에 제시됩니다. 

---

## 2-1-2. Tabular feature processor

입력 $x$는 작은 MLP로 먼저 처리됩니다.

```math
P(x)
=
\text{ReLU}
\left(
FC_2\left(
\text{ReLU}(FC_1(x))
\right)
\right).
```

$FC_1$은

$$
N\rightarrow N+4
$$

차원으로 확장하고, $FC_2$는 다시

$$
N+4\rightarrow N
$$

으로 줄입니다. 

### 기호

* $x$: 원 tabular feature vector
* $N$: feature 수
* $FC_k$: fully-connected layer
* $\text{ReLU}(a)=\max(0,a)$
* $P(x)$: 학습된 tabular embedding

### 용어 설명

**Embedding**은 원 feature를 예측에 더 적합한 좌표계로 변환한 벡터입니다.

---

# 2-1-3. External proxy target

External mode에서는 FashionMNIST, 필요하면 MNIST까지 사용합니다.

각 $(x,y)$에 대해 같은 class에 대응시킨 외부 이미지 중 하나를 임의로 선택하여

$$
p_x^*=i_x
$$

를 proxy target으로 사용합니다.

중요한 점은 한 class를 하나의 고정 그림에 연결하지 않고

$$
x_1\to i^{(1)}_y,\qquad
x_2\to i^{(2)}_y,\qquad
x_3\to i^{(3)}_y
$$

처럼 **one-to-many randomized mapping**을 사용한다는 점입니다. 

Table XII에서 3-class 데이터셋에 대해 multiple mapping은

$$
ACC=0.8876,\qquad AUC=0.9506
$$

이고 single mapping은

$$
ACC=0.8483,\qquad AUC=0.9305
$$

였습니다. 

### 해석

고정 prototype 하나를 맞추게 하면 같은 class의 모든 sample이 지나치게 같은 representation으로 수렴할 수 있습니다. 여러 target을 사용하면 class identity는 유지하면서 intra-class diversity가 생기는 일종의 **representation regularization**으로 작용할 수 있습니다.

---

# 2-1-4. Internal proxy target

외부 이미지를 쓰지 않을 경우 저자는

```math
\boxed{
q(x,y)
=
(1-\alpha)T_y+\alpha V(x)
}
```

를 사용합니다.

기본값은

$$
\alpha=0.35.
$$

여기서:

* $T_y$: class $y$에 공통인 template
* $V(x)$: sample $x$에 따라 달라지는 variation map
* $\alpha$: class template와 instance variation의 혼합 비율

입니다. 

따라서

$$
\alpha\to0
$$

이면 class prototype에 가까워지고,

$$
\alpha\to1
$$

이면 instance variation 비중이 커집니다.

---

## Gaussian class structure

Class $y$의 Gaussian 중심은 원 위에 배치합니다.

```math
c_1^{(y)}
=
0.45\cos\left(\frac{2\pi y}{C}\right),
```

```math
c_2^{(y)}
=
0.45\sin\left(\frac{2\pi y}{C}\right).
```

pixel coordinate를 $(u_{ij},v_{ij})\in[-1,1]^2$라 하면,

```math
G_y(i,j)
=
\exp
\left[
-\frac{
(u_{ij}-c_1^{(y)})^2+
(v_{ij}-c_2^{(y)})^2
}{0.18}
\right].
```

즉 각 class가 서로 다른 2D 위치에 밝은 blob을 가집니다. 

### 용어 설명

**Gaussian blob**은 중심에서 가장 크고 멀어질수록 지수적으로 작아지는 둥근 밝기 패턴입니다. 여기서는 “class의 대략적인 공간 위치”를 표현합니다.

---

## Periodic class structure

Class 수가 커지면 Gaussian들이 겹칠 수 있으므로 periodic pattern을 더합니다.

```math
P_y(i,j)
=
\frac12
\left[
\sin\left((y+1)\pi\psi_y(u_{ij},v_{ij})\right)+1
\right],
```

여기서

```math
\psi_y(u,v)
=
\begin{cases}
u, & y\bmod4=0,\\
v, & y\bmod4=1,\\
u+v, & y\bmod4=2,\\
\sqrt{u^2+v^2}, & y\bmod4=3.
\end{cases}
```

입니다.

따라서 class에 따라 horizontal, vertical, diagonal, radial pattern이 반복되고, $(y+1)\pi$가 주파수도 바꿉니다. 

최종 template은

```math
\boxed{
T_y
=
\text{Norm}
\left(
0.55G_y+0.45P_y
\right)
}
```

입니다.

$\text{Norm}$은 $[0,1]$ min-max normalization입니다.

---

## Sample-specific map

논문의 prose를 수식으로 압축하여 쓰면 $V(x)$는 대략

```math
V(x)
=
\text{AvgPool}_{3\times3}
\left[
\text{MinMax}
\left(
\tanh(
\text{Tile}_{784}(x)
)
\right)
\right].
```

이 식은 **제가 논문의 순차 설명을 하나의 composition으로 적은 것**이며, 논문 자체는 feature를 784개까지 반복 → $\tanh$ → min-max normalization → $3\times3$ average pooling 순으로 설명합니다. 

---

# 2-1-5. Stochastic proxy generator

매 forward pass마다

$$
r\sim U(0,1)^{28\times28}
$$

의 random noise를 새로 생성합니다.

이를 flatten하고 $P(x)$와 연결합니다.

```math
z
=
\text{ReLU}
\left[
FC_4
\left(
\text{ReLU}
\left(
FC_3(
\text{flatten}(r)\oplus P(x)
)
\right)
\right)
\right].
```

$\oplus$는 concatenation입니다.

Generator는 다시

```math
\boxed{
G(P(x),r)
=
\text{reshape}
\left[
\text{sigmoid}
\left(
FC_6
\left(
\text{ReLU}
\left(
FC_5(z\oplus P(x))
\right)
\right)
\right)
\right]
}
```

으로 $28\times28$ proxy representation을 만듭니다. 

### 왜 random noise를 넣는가?

동일한 $x$에 대해 항상 완전히 같은 intermediate representation을 만드는 대신,

$$
G(P(x),r_1)\neq G(P(x),r_2)
$$

가 가능해집니다.

이는 일종의 stochastic regularization으로 해석할 수 있습니다.

> 단, 논문은 “randomness 자체가 일반화 향상의 원인”임을 분리한 noise-free ablation까지 제공하지는 않습니다. 따라서 이 메커니즘 해석은 **합리적 가설이지 논문이 직접 증명한 사실은 아닙니다.**

### 용어 설명

**Stochastic representation**: 같은 입력에도 내부 random variable 때문에 조금씩 다른 representation이 생성되는 구조입니다.

---

# 2-1-6. CNN classifier

생성된 $28\times28$ representation은 작은 CNN으로 들어갑니다.

```math
z'
=
\text{MaxPool}_2
\left(
\text{ReLU}
\left[
Conv_2
\left(
\text{MaxPool}_1
(
\text{ReLU}(Conv_1(G))
)
\right)
\right]
\right),
```

그 뒤

```math
\hat y
=
FC_8
\left[
\text{Dropout}
\left(
\text{ReLU}
\left(
FC_7(\text{flatten}(z'))
\right)
\right)
\right].
```

즉 전체 구조는

$$
\boxed{
\text{Tabular}
\rightarrow
\text{MLP}
\rightarrow
\text{Generator}
\rightarrow
28\times28
\rightarrow
\text{CNN}
\rightarrow
\text{Class}
}
$$

입니다. 

---

# 2-1-7. Loss function

전체 손실은

```math
\boxed{
L_{\text{total}}
=
L_{\text{cls}}
+
\lambda_{\text{recon}}L_{\text{recon}}
}
```

입니다.

분류 손실:

```math
L_{\text{cls}}
=
\text{CE}
\left(
\text{CNN}(G(P(x),r)),y
\right).
```

Proxy reconstruction:

```math
L_{\text{recon}}
=
\left\|
G(P(x),r)-p_x^*
\right\|_2^2.
```

$p_x^*$는 external mode에서는 image target, internal mode에서는 $q(x,y)$입니다.  

### 의미

$L_{\text{cls}}$는

> “정답 class를 맞혀라.”

이고,

$L_{\text{recon}}$은

> “그 과정에서 중간 representation도 구조화된 proxy 공간에 가까워져라.”

입니다.

즉 이것은 일종의 **multi-objective representation shaping**입니다.

---

# 2-1-8. VIF initialization

Feature $X_i$를 나머지 feature로 회귀한 $R_i^2$에 대해

```math
\boxed{
VIF_i
=
\frac{1}{1-R_i^2}
}
```

를 정의합니다.

$R_i^2\rightarrow1$이면

$$
VIF_i\rightarrow\infty
$$

이므로 해당 feature는 다른 feature로 거의 설명된다는 뜻입니다.

T2I-VIF의 초기 weight는

```math
\boxed{
w_{ij}
=
\frac{1}{VIF_i}
}
```

입니다. 

따라서 다중공선성이 큰 변수는 처음에 작은 영향력을 갖습니다.

중요한 점은 이것이 regularization constraint가 아니라 **initial condition**이라는 것입니다.

```math
w_{ij}^{(0)}
=
\frac1{VIF_i}
\quad\text{이지만}\quad
w_{ij}^{(t)}
\text{는 이후 자유롭게 학습된다.}
```

최종 feature representation은

$$
P(x)\oplus P_{\text{VIF}}(x)
$$

입니다.

### 용어 설명

**Multicollinearity(다중공선성)**: 여러 입력 feature가 서로 거의 선형 결합 관계에 있어 개별 변수의 효과를 구분하기 어려운 현상입니다.

**VIF**: 어떤 feature가 다른 feature들에 얼마나 중복되어 있는지를 나타내는 전통적인 회귀 진단량입니다.

---

# 2-1-9. Reliability test ① Irrelevant Features

원 feature 수가 $N$일 때

$$
\text{round}(\rho_rN)
$$

개의 Gaussian random feature를 추가합니다.

$$
R_{\text{train}},R_{\text{test}}
\overset{\text{i.i.d.}}{\sim}
\mathcal N(0,1),
$$

```math
\tilde X
=
X\parallel R.
```

Severity는

$$
\rho_r\in\{0.05,0.10,0.15,0.20\}.
$$

Train/test noise를 독립적으로 뽑기 때문에 train의 random pattern을 암기해도 test에서는 도움이 되지 않습니다. 

---

# 2-1-10. Reliability test ② Noisy Labels

$$
m_i\sim\text{Bernoulli}(\rho_n)
$$

이고

```math
\tilde y_i
=
\begin{cases}
y_i, & m_i=0,\\
(y_i+\delta_i)\bmod C,&m_i=1,
\end{cases}
```

$$
\delta_i\sim\text{Uniform}\{1,\ldots,C-1\}.
$$

따라서 noisy sample은 반드시 다른 class로 바뀝니다.

$$
\rho_n\in\{0.05,0.10,0.15,0.20\}.
$$

Test label은 clean 상태로 유지됩니다. 

---

# 2-1-11. Reliability test ③ Shortcut Decorrelation

Training에서는 synthetic shortcut이 label과 확률 $\rho_s$로 일치합니다.

$$
m_i\sim\text{Bernoulli}(\rho_s),
$$

$$
h_i=
\begin{cases}
y_i,&m_i=1,\\
(y_i+\delta_i)\bmod C,&m_i=0.
\end{cases}
$$

shortcut은

```math
s_i^{\text{train}}
=
a_s\frac{h_i}{C-1}.
```

하지만 test에서는

$$
h_i^{\text{test}}
\sim
\text{Uniform}\{0,\ldots,C-1\}
$$

로 만들어 label correlation을 제거합니다. 

즉:

$$
P_{\text{train}}(s,y)
\neq
P_{\text{test}}(s,y).
$$

이것이 shortcut reliance를 측정합니다.

---

# 2-1-12. 새로운 instability diagnostic

같은 severity 조건에서 두 realization $w,\ell$의 확률 예측을

$$
p_i^{(w)},p_i^{(\ell)}\in[0,1]^C
$$

라 하겠습니다.

예측 class가 다르면:

$$
d^{(w,\ell)}(i)=1.
$$

예측 class는 같지만 확률분포가 다르면:

```math
d^{(w,\ell)}(i)
=
\frac12
\left\|
p_i^{(w)}-p_i^{(\ell)}
\right\|_1.
```

즉

$$
d^{(w,\ell)}(i)=
\begin{cases}
1,&
\hat y_i^{(w)}\neq\hat y_i^{(\ell)},\\[4pt]
\frac12\|p_i^{(w)}-p_i^{(\ell)}\|_1,
&
\hat y_i^{(w)}=\hat y_i^{(\ell)}.
\end{cases}
$$

입니다.  

$K$ realizations을 모두 비교하면

```math
I
=
\frac1{n_{\text{test}}}
\sum_{i=1}^{n_{\text{test}}}
\frac{2}{K(K-1)}
\sum_{1\le w<\ell\le K}
d^{(w,\ell)}(i).
```

세 scenario에 대해 이를 각각

* DPI: Decision-aware Prediction Instability
* LCI: Label Corruption Instability
* SAI: Shortcut Assignment Instability

라고 부릅니다.

모두

$$
0\le I\le1
$$

이며 **작을수록 안정적**입니다.

### 매우 중요한 아이디어

두 모델이 동일한 평균 accuracy를 가져도

$$
\text{Model A: }
\hat y^{(1)}\simeq\hat y^{(2)}\simeq\hat y^{(3)}
$$

일 수도 있고,

$$
\text{Model B: }
\hat y^{(1)},\hat y^{(2)},\hat y^{(3)}
\text{가 크게 요동}
$$

할 수도 있습니다.

기존 accuracy만 보면 이 차이가 보이지 않습니다.

---

# 2-1-13. APS, WSP, $\tau$-degradation

Severity $\rho$에서 성능을 $m(\rho)$라 하면

```math
\boxed{
APS_m
=
\frac{1}{\rho_{\max}}
\int_0^{\rho_{\max}}m(\rho)\,d\rho
}
```

입니다. 

즉 corruption curve의 평균 높이입니다.

Worst Severity Performance:

$$
WSP=m(\rho_{\max}).
$$

또한 성능이 clean보다 $\tau$만큼 떨어지는 최초 severity를

```math
\boxed{
\rho_\tau^{(m)}
=
100
\frac{
\min\{\rho\in\mathcal P:
m(\rho)\le m_{\text{clean}}-\tau\}
}
{\rho_{\max}}
}
```

로 정의합니다.

큰 값일수록 더 심한 corruption까지 견딘다는 의미입니다. 

---

# 3. 주장별 Figure/Table/Page 대응

| 주장                                 | 근거 위치                             |
| ---------------------------------- | --------------------------------- |
| 전체 T2I 구조                          | **Figure 1, p.3**                 |
| External/Internal proxy target     | Section III-A, **pp.2–4**         |
| VIF initialization                 | Section III-B, **p.4**            |
| IF/NL/SD corruption protocol       | Section III-C, **pp.4–5**         |
| DPI/LCI/SAI                        | Section III-D, **p.5**, Appendix  |
| External clean performance         | **Table I, p.6**                  |
| Internal clean performance         | **Table II, p.6**                 |
| Parameter efficiency               | **Figure 2, p.7; Table IX, p.11** |
| Robustness APS/WSP                 | **Table III, Figure 3, p.7**      |
| $\rho_\tau$                        | **Table IV, p.7; Table X, p.12**  |
| Dataset-level T2I vs MLP           | **Figure 4, p.8**                 |
| Instability                        | **Table V, p.8**                  |
| Generator ablation                 | **Table VI, p.9**                 |
| Encoder compatibility              | **Table VII, p.9**                |
| Generated external proxy images    | **Figure 5, p.9**                 |
| Internal proxy evolution           | **Figure 6, p.9**                 |
| Statistical pairwise tests         | **Table VIII, p.11**              |
| Multiple vs single mapping         | **Table XII, p.13**               |
| DeepInsight/HACNet/IGTD comparison | **Table XIII, p.13**              |
| VIF variants                       | **Table XIV, p.13**               |
| Train performance under corruption | **Table XV, p.13**                |

---

# 4. 연구 방법과 결과: 저자 보고 vs 제 해석

## 4-1. Clean predictive performance

### 저자가 직접 보고한 결과

External proxy target:

**OpenML-CC18**

$$
\text{T2I-VIF}: ACC=0.879,\quad AUC=0.922
$$

$$
\text{T2I}: ACC=0.877,\quad AUC=0.920.
$$

TabZilla:

$$
\text{T2I-VIF}: ACC=0.862,\quad AUC=0.905
$$

$$
\text{T2I}: ACC=0.857,\quad AUC=0.906.
$$



Internal proxy에서는:

$$
\text{T2I}: 0.876/0.936,
$$

$$
\text{MLP}:0.868/0.934,
$$

$$
\text{FT-Transformer}:0.865/0.934.
$$



### 제 해석

External target 결과는 상당히 좋지만, internal target 결과를 함께 보는 것이 모델 자체의 성능을 평가하는 데 더 중요합니다.

Internal proxy 기준으로 MLP 대비 ACC 차이는

$$
0.876-0.868=0.008
$$

즉 **0.8 percentage point 정도**입니다.

따라서

> “기존 모델보다 압도적으로 높은 성능”

보다는

> **“작은 추가 구조로 MLP보다 일관된 소폭 개선을 얻었다.”**

가 더 정확합니다.

---

# 4-2. Parameter efficiency

### 저자 보고

Figure 2의 예시 조건 $N=78$, binary classification에서:

| 모델             | Trainable parameters |
| -------------- | -------------------: |
| MLP            |               0.007M |
| MLP-PLR        |               0.009M |
| FT-Transformer |               0.136M |
| **T2I**        |           **0.628M** |
| TuneTables     |                25.8M |
| TabM           |                37.9M |



### 제 해석

“lightweight”는 **절대적인 의미가 아닙니다**.

T2I는:

$$
\frac{0.628}{0.007}\approx90
$$

배의 MLP parameter를 가지고,

FT-Transformer의 약

$$
\frac{0.628}{0.136}\approx4.6
$$

배입니다.

반대로 TuneTables 대비:

$$
\frac{25.8}{0.628}\approx41
$$

배 작습니다.

따라서 정확한 표현은:

> **대규모 neural tabular model보다 compact하지만 가장 작은 tabular neural model은 아니다.**

입니다.

또 parameter count만으로 latency나 FLOPs를 알 수 없습니다. T2I에는 generator와 $28\times28$ CNN이 있기 때문입니다.

---

# 4-3. Robustness

Table III의 대표값은 다음과 같습니다.

### Irrelevant Features

$$
APS_{\text{ACC}}:
\quad
T2I=0.863,\quad MLP=0.857.
$$

### Noisy Labels

$$
APS_{\text{ACC}}:
\quad
T2I=0.846,\quad MLP=0.844.
$$

### Shortcut Decorrelation

$$
APS_{\text{ACC}}:
\quad
T2I=0.853,\quad
MLP=0.849,\quad
XGBoost=0.797.
$$

그리고 worst-severity SD에서는

$$
WSP_{\text{ACC}}:
\quad
T2I=0.721,\quad
XGBoost=0.460.
$$

입니다. 

### 제 해석

가장 흥미로운 결과는 **shortcut decorrelation**입니다.

하지만 MLP와 비교하면

$$
0.853-0.849=0.004
$$

이므로 평균 APS 차이는 매우 작습니다.

따라서 이 결과는

> “T2I가 모든 corruption에 압도적으로 robust하다”

보다

> “T2I가 높은 clean baseline을 대체로 유지하며, 특히 일부 shortcut setting에서 tree/transformer보다 유리하다”

로 읽는 것이 안전합니다.

---

# 4-4. Stability

Table V에서:

| Model          |     DPI ↓ |     LCI ↓ |     SAI ↓ |
| -------------- | --------: | --------: | --------: |
| SVM            |     0.081 | **0.163** |     0.288 |
| XGBoost        | **0.081** |     0.182 |     0.309 |
| MLP            |     0.127 |     0.257 |     0.197 |
| TabTransformer |     0.138 |     0.236 |     0.267 |
| T2I            | **0.114** | **0.291** | **0.190** |

 

### 저자 해석

T2I는 SAI가 가장 낮고 DPI도 좋은 편이지만 LCI는 높습니다.

### 제 해석

이 결과는 오히려 논문의 가장 가치 있는 부분 중 하나입니다.

$$
\boxed{
\text{Robust Accuracy}
\not\Rightarrow
\text{Stable Prediction}
}
$$

T2I는 noisy-label dataset을 평균적으로 잘 맞히더라도, **어떤 label들이 noise로 선택되었느냐에 따라 최종 prediction이 상당히 변한다**는 뜻입니다.

따라서 일반화 연구에서는:

$$
\text{mean performance}
+
\text{variance / instability}
$$

를 동시에 봐야 합니다.

---

# 4-5. Ablation 결과

Table VI:

| Variant               |  Clean ACC | IF APS ACC | NL APS ACC | SD APS ACC |
| --------------------- | ---------: | ---------: | ---------: | ---------: |
| Full                  | **0.8760** | **0.8626** | **0.8458** | **0.8529** |
| Proxy permuted        |     0.8747 |     0.8626 |     0.8440 |     0.8523 |
| no $L_{\text{recon}}$ |     0.8740 |     0.8626 |     0.8439 |     0.8516 |
| Generator-free        | **0.8607** | **0.8495** | **0.8427** | **0.8438** |



### 가장 중요한 해석

Proxy permutation이 거의 영향을 주지 않습니다.

$$
0.8760\rightarrow0.8747.
$$

$L_{\text{recon}}$을 빼도

$$
0.8760\rightarrow0.8740.
$$

하지만 generator 자체를 제거하면

$$
0.8760\rightarrow0.8607.
$$

따라서 실험은 다음 가설을 더 강하게 지지합니다.

$$
\boxed{
\text{proxy target의 정확한 spatial geometry}
\text{보다}
\text{generation pathway의 표현 변환}
\text{이 더 중요하다}
}
$$

이는 향후 연구에 매우 중요한 단서입니다.

---

# 4-6. Encoder generalization

Table VII:

$$MLP_{100percent}:0.868\to0.869,$$

$$
MLP_{50percent}:0.868\to0.872,
$$

$$
MLP_{25percent}:0.867\to0.871.
$$

반면 FT-Transformer:

$$
ACC:0.865\to0.788,
$$

$$
AUC:0.934\to0.895.
$$



따라서 현 단계에서 Table2Image pathway는 **architecture-agnostic general-purpose wrapper라고 볼 수 없습니다**.

오히려

$$
\text{MLP-like encoder}
\Rightarrow \text{compatible}
$$

$$
\text{Transformer encoder}
\Rightarrow \text{currently incompatible}
$$

라는 증거입니다.

---

# 5. 통계적으로 취약하거나 직접 비교하기 어려운 부분

## 5-1. “Friedman test 유의” ≠ “T2I가 모든 baseline보다 유의하게 좋다”

OpenML-CC18에서 저자는 Friedman test에 대해

$$
p<0.001
$$

을 보고합니다. 하지만 Friedman test는

> “여러 모델 가운데 적어도 하나의 성능 차이가 있다”

는 것을 의미할 뿐,

> “T2I가 각 경쟁 모델보다 유의하게 낫다”

를 의미하지 않습니다. 

실제로 Table VIII의 Holm-corrected pairwise Wilcoxon 결과를 보면:

OpenML ACC:

$$
p_{\text{Holm}}(\text{T2I vs TuneTables})=0.091,
$$

$$
p_{\text{Holm}}(\text{T2I vs CatBoost})=0.073,
$$

$$
p_{\text{Holm}}(\text{T2I vs LightGBM})=0.046.
$$

즉 conventional $0.05$ 수준에서 유의한 것은 이 비교 중 LightGBM뿐입니다. 

더구나 TuneTables와의 median difference CI는

$$
[-0.009,0.024]
$$

로 0을 포함합니다.

따라서 **평균 순위가 좋다는 결과와 강한 baseline에 대한 통계적 superiority는 구별해야 합니다.**

---

# 5-2. Kendall's $W$도 압도적 차이를 뜻하지 않는다

논문은 OpenML에서

$$
W_{\text{ACC}}=0.297,
\qquad
W_{\text{AUC}}=0.316
$$

을 보고합니다. 

이는 모델 순위 차이가 존재한다는 근거지만 $W\approx1$에 가까운 “매우 일관된 강한 효과”는 아닙니다.

즉 $p$-value와 effect size를 분리해서 봐야 합니다.

### 용어 설명

**Effect size**는 “차이가 존재하는가?”보다 “그 차이가 얼마나 큰가?”를 측정합니다.

---

# 5-3. Internal proxy의 실제 개선폭이 작음

T2I vs parameter-matched MLP:

```math
0.876-0.868
=
0.008.
```

16 dataset 중 accuracy 우위도 10/16입니다. 

통계적으로 유의한 pairwise 차이도 TabTransformer, XGBoost, SVM에 대해서 보고되며, 가까운 MLP나 FT-Transformer에 대해서 superiority가 증명된 것은 아닙니다.

따라서 이것은 **incremental improvement**로 보는 것이 타당합니다.

---

# 5-4. External proxy 비교는 동일 정보 조건이 아님

External T2I는 FashionMNIST/MNIST의 class-conditioned image를 추가 auxiliary signal로 사용합니다.

반면 XGBoost나 ordinary MLP는 그런 외부 proxy dataset을 사용하지 않습니다.

따라서 Table I의 성능을

> “동일한 정보만 사용했을 때 T2I architecture가 우월하다”

라고 해석하면 과도합니다.

저자도 robustness 비교에서는 이 문제를 피하기 위해 **internal proxy target으로 전환**합니다. 

이 설계는 적절한 결정입니다.

---

# 5-5. DeepInsight·IGTD 결과는 서로 다른 subset에서 측정됨

논문 본문에서:

* HACNet: full OpenML subset
* DeepInsight: 최소 sample 조건을 만족하는 subset
* IGTD: 또 다른 compatible subset

을 사용합니다. 

따라서

$$
0.888\text{ vs }0.803
$$

과

$$
0.995\text{ vs }0.971
$$

을 하나의 동일 benchmark leaderboard처럼 비교하면 안 됩니다.

**서로 다른 dataset subset의 평균은 직접 비교 불가능합니다.**

---

# 5-6. “Lightweight”는 parameter 수만 측정

논문은 trainable parameter를 주로 사용하지만 다음은 제시되지 않습니다.

$$
\text{training time},
\quad
\text{inference latency},
\quad
\text{FLOPs},
\quad
\text{peak memory},
\quad
\text{energy}.
$$

따라서

$$
\text{parameter-efficient}
$$

는 지지되지만

$$
\text{compute-efficient}
$$

까지 자동으로 결론낼 수 없습니다.

---

# 5-7. Robustness significance 검증 부족

Table III의 차이는 특히 T2I와 MLP 사이에 매우 작습니다.

예:

$$
APS_{\text{NL,ACC}}:
0.846\text{ vs }0.844.
$$

그러나 본문 Table III에는 이러한 robustness 차이에 대한 paired CI나 Holm-corrected significance가 제공되지 않습니다. 

따라서 이 수치 차이가 dataset-level variability를 고려해 얼마나 확실한지 판단하기 어렵습니다.

---

# 5-8. $K=3$ instability realization

본 분석은 기본적으로

$$
K=3
$$

realization을 사용합니다.

저자들은 5개 대표 dataset에서

$$
K\in\{3,5,10\}
$$

을 비교했고, T2I의 $\Delta_{\max}$가 최대 약 $0.003$이라 보고합니다. 

좋은 sensitivity analysis이지만 **16개 전체 dataset에서 $K=10$을 검증한 것은 아닙니다.**

---

# 5-9. Synthetic corruption → real-world shift 일반화 불확실

논문의 corruption은 잘 통제된 synthetic mechanism입니다.

하지만 실제 deployment에서는

$$
P_{\text{train}}(X,Y)
\rightarrow
P_{\text{deploy}}(X,Y)
$$

의 변화가 훨씬 복잡합니다.

WhyShift 연구는 실제 tabular data에서 특히

$$
P(Y\mid X)
$$

자체가 변하는 shift가 흔하다는 점을 보여줍니다. ([NeurIPS Proceedings][2])

TableShift 역시 15개의 실제 distribution-shift task에서 ID와 OOD 성능 사이의 trade-off와 label-distribution shift의 중요성을 보여줍니다. ([NeurIPS Proceedings][3])

따라서 T2I가 synthetic SD에서 강하다는 것이 자연 발생 shift에서도 강하다는 보장은 없습니다.

---

# 6. 논문이 답하지 않는 질문

### Q1. 왜 $28\times28$이어야 하는가?

MNIST/FashionMNIST 호환성 때문인 것으로 보이지만,

$$
16\times16,\quad32\times32,\quad 1\times d
$$

등에 대한 sensitivity가 없습니다.

**답변:** 향후 spatial resolution을 hyperparameter로 두고 parameter/accuracy/robustness frontier를 비교해야 합니다.

---

### Q2. 왜 $\alpha=0.35$인가?

$$
q=(1-\alpha)T_y+\alpha V(x)
$$

에서 핵심 hyperparameter인데 systematic sensitivity curve가 제시되지 않습니다.

**추천 연구:**

$$
\alpha\in\{0,0.1,\ldots,1\}
$$

에 대해 clean ACC, APS, DPI/LCI/SAI를 동시에 측정해야 합니다.

---

### Q3. Gaussian + periodic pattern이 정말 필요한가?

Ablation 결과 permutation 영향이 작습니다.

**따라서 제 답은:** 현재 근거만 보면 **특정 geometry가 필수라는 증거는 약합니다.**

더 중요한 것은

$$
G(P(x),r)
$$

이라는 learned generation bottleneck일 가능성이 큽니다.

---

### Q4. Random noise가 얼마나 중요한가?

noise를 제거한

$$
G(P(x))
$$

와 비교하지 않았습니다.

따라서 stochasticity의 causal contribution은 미확인입니다.

---

### Q5. Regression에서도 가능한가?

논문은 classification only입니다. 저자도 이를 limitation으로 명시합니다. 

Regression에서는 class template $T_y$가 존재하지 않기 때문에 별도 설계가 필요합니다.

가능한 방향은 continuous target embedding:

```math
T(y)
=
\sum_{k=1}^{K}
\phi_k(y)B_k
```

처럼 $y$를 continuous basis map으로 바꾸는 것입니다.

---

### Q6. Missing values와 categorical variables에 얼마나 robust한가?

논문 본문은 이 문제를 핵심 평가 대상으로 두지 않습니다.

실제 tabular deployment에서는 매우 중요합니다.

---

### Q7. Calibration은 좋은가?

AUC와 ACC가 높다고

$$
P(Y=c\mid X)
$$

의 확률이 calibrated된 것은 아닙니다.

ECE, Brier score, NLL이 없습니다.

---

### Q8. Dataset size가 매우 작을 때도 유리한가?

별도의 sample-efficiency curve가 없습니다.

TabPFN 같은 foundation model과 비교하면 이 부분이 특히 중요합니다.

2025년 Nature의 TabPFN 연구는 작은 tabular dataset에서 pretrained foundation approach의 강력한 성능을 보고합니다. ([Nature][4])

---

### Q9. 대규모 dataset에서도 lightweight인가?

28×28 생성 단계는 sample마다 발생합니다.

따라서

$$
n\rightarrow10^6
$$

규모에서 training throughput을 직접 검증해야 합니다.

---

### Q10. VIF가 nonlinear redundancy에도 작동하는가?

아닙니다.

```math
VIF_i
=
\frac1{1-R_i^2}
```

는 기본적으로 **linear collinearity** 측정입니다.

예를 들어

$$
X_2=X_1^2
$$

와 같은 강한 nonlinear redundancy는 VIF가 충분히 포착하지 못할 수 있습니다.

---

# 7. 가장 중요한 Figure 5개

## Figure 1 — 전체 Table2Image architecture, p.3

가장 중요한 그림입니다.

흐름은

$$
x
\rightarrow
P(x)
\rightarrow
\{r,P(x)\}
\rightarrow
z
\rightarrow
G
\rightarrow
28\times28
\rightarrow
CNN
\rightarrow
\hat y
$$

입니다.

VIF branch도 별도로

$$
P(x)\oplus P_{\text{VIF}}(x)
$$

형태로 결합됩니다. 

### 핵심 해석

이 모델은 “tabular feature를 이미지로 배치”하는 기존 방식보다 **image-shaped latent representation 자체를 end-to-end 학습**한다는 점이 핵심입니다.

---

# Figure 2 — 성능 vs parameter count, p.7

$x$축은 log parameter count, $y$축은 OpenML 평균 ACC입니다.

T2I는 TuneTables/TabM보다 훨씬 왼쪽에 있으면서 성능이 높습니다. 

그러나 MLP와 FT-Transformer보다 오른쪽입니다.

따라서 그림의 올바른 해석은:

$$
\boxed{
\text{T2I = Pareto-like compromise}
}
$$

이지 “가장 작은 모델”이 아닙니다.

### 용어 설명

**Pareto frontier**: 어떤 한 기준을 더 개선하려면 다른 기준을 희생해야 하는 효율적인 후보 집합입니다.

---

# Figure 3 — Corruption severity curve, p.7

IF, NL, SD의 severity가 증가하면서 ACC/AUC가 어떻게 감소하는지 보여줍니다. 

### 핵심

* IF: 비교적 완만
* NL: 점진적 하락
* SD: 고 predictivity에서 급격한 성능 붕괴

특히 SD에서 XGBoost의 급격한 감소가 눈에 띄고 T2I/T2I-VIF가 상대적으로 높은 곡선을 유지합니다.

즉 단일 clean score보다

$$
m(\rho)
$$

라는 **성능 곡선 전체를 봐야 한다**는 논문의 메시지를 가장 잘 보여줍니다.

---

# Figure 4 — Clean gain vs corrupted gain, p.8

$x$축:

```math
\Delta ACC_{\text{clean}}
=
ACC_{\text{T2I}}-ACC_{\text{MLP}}
```

$y$축:

```math
\Delta APS_{\text{ACC}}
=
APS_{\text{T2I}}-APS_{\text{MLP}}.
```



Upper-right에 있으면 clean과 corruption 모두 T2I가 우위입니다.

논문에서 이 영역의 dataset 수는:

$$
IF:9/16,
$$

$$
NL:7/16,
$$

$$
SD:8/16.
$$

즉 평균 하나만 보면 보이지 않는 **dataset heterogeneity**가 드러납니다.

---

# Figure 6 — Internal proxy가 학습되는 과정, p.9

Figure 6은 제가 이 논문에서 메커니즘을 이해하는 데 두 번째로 중요한 그림이라고 봅니다.

왼쪽부터:

1. Gaussian blob
2. periodic structure
3. $V(x)$
4. final target
5. generated image
6. epoch별 생성 과정

을 보여줍니다. 

처음 random noise에 가까운 representation이 학습되면서 class-specific structure에 접근합니다.

하지만 완벽한 이미지 재현은 목표가 아닙니다.

$$
\text{photorealism}
\neq
\text{objective}.
$$

목표는

$$
\boxed{
\text{classification-friendly latent geometry}
}
$$

입니다.

---

# 8. 결론 및 저자들이 제시한 후속 연구

저자들은 Table2Image가:

* structured proxy representation을 사용하는 lightweight tabular architecture,
* VIF-informed initialization,
* performance degradation과 realization instability를 함께 측정하는 reliability protocol

을 제시했다고 결론짓습니다. 

명시적 한계는 세 가지입니다.

$$
\boxed{
\text{classification only}
}
$$

$$
\boxed{
\text{synthetic controlled corruption only}
}
$$

$$
\boxed{
\text{external proxy mode: }C\le20
}
$$

저자들이 제시한 future work는:

$$
\text{broader tasks}
+
\text{natural shifts}
+
\text{more flexible proxy representations}
$$

입니다. 

---

# 8-1. 모델의 일반화 성능 향상 가능성

이 논문의 일반화 가능성을 연구 관점에서 분해하면 다음과 같습니다.

## A. Proxy generation을 regularizer로 이해해야 한다

Ablation상 가장 중요한 것은 generator입니다.

따라서 future model은 고정 $28\times28$ 이미지보다

```math
z_{\text{proxy}}
=
G_\theta(P_\phi(x),\epsilon)
```

라는 일반적인 stochastic latent generator로 재해석하는 것이 좋습니다.

예를 들어:

$$
z_{\text{proxy}}\in\mathbb R^d
$$

로 만들어 CNN 자체를 제거할 수도 있습니다.

이렇게 하면 “이미지를 만드는 것”이 아니라

> **보조 구조를 갖는 latent-space regularization**

이라는 더 일반화 가능한 방법이 됩니다.

---

## B. Multi-view consistency 추가

현재 동일 $x$에서

$$
G(x,r_1),G(x,r_2)
$$

가 달라질 수 있습니다.

두 representation이 동일 label geometry를 보존하도록

```math
L_{\text{cons}}
=
\left\|
h(G(x,r_1))-h(G(x,r_2))
\right\|_2^2
```

를 추가할 수 있습니다.

최종 목적함수:

```math
L
=
L_{\text{cls}}
+
\lambda_rL_{\text{recon}}
+
\lambda_cL_{\text{cons}}.
```

이 방법은 instability 감소에 특히 도움이 될 가능성이 있습니다.

---

## C. LCI를 직접 최적화

T2I의 약점은 높은 LCI입니다.

따라서 noisy-label robustness 연구에서는 단순 CE보다 symmetric/generalized loss를 적용하거나,

```math
L_{\text{stability}}
=
\mathbb E_{w,\ell}
\left[
\|p^{(w)}-p^{(\ell)}\|_1
\right]
```

을 넣을 수 있습니다.

목표는

$$
\max \text{Accuracy}
\quad\text{subject to}\quad
\text{LCI}\le\epsilon
$$

형태로 보는 것이 좋습니다.

---

## D. VIF → nonlinear redundancy-aware initialization

현재 VIF를 일반화하면

```math
w_i^{(0)}
=
f(\text{redundancy}_i)
```

라고 볼 수 있습니다.

향후에는 다음을 쓸 수 있습니다.

```math
\text{redundancy}_i
=
I(X_i;X_{-i})
```

또는 conditional mutual information/HSIC 기반 측정입니다.

즉

```math
w_i^{(0)}
=
\frac1{1+\gamma I(X_i;X_{-i})}
```

같은 초기화를 연구할 수 있습니다.

### 용어 설명

**Mutual information**은 두 변수가 선형뿐 아니라 일반적인 비선형 의존 관계를 얼마나 공유하는지 나타내는 정보이론량입니다.

---

## E. Natural distribution shift 학습

TableShift와 WhyShift 결과를 고려하면 future T2I는 적어도 다음 세 shift를 분리해야 합니다.

### Covariate shift

$$
P_{\text{train}}(X)
\neq
P_{\text{test}}(X)
$$

but

$$
P(Y\mid X)
\approx\text{same}.
$$

### Conditional shift

$$
P_{\text{train}}(Y\mid X)
\neq
P_{\text{test}}(Y\mid X).
$$

### Label shift

$$
P_{\text{train}}(Y)
\neq
P_{\text{test}}(Y).
$$

WhyShift는 실제 tabular dataset에서 특히 $Y|X$ shift가 중요하다고 보고합니다. ([NeurIPS Proceedings][2])

---

## F. Reliability-aware model selection

현재 hyperparameter 선택 목표가 validation ACC 중심이라면 앞으로는

```math
J
=
ACC
-\lambda_1 DPI
-\lambda_2 LCI
-\lambda_3 SAI
-\lambda_4\log(\#\text{Params})
```

와 같은 다목적 objective도 가능합니다.

이는 이 논문의 가장 자연스러운 확장 중 하나입니다.

---

# 8-2. 2020년 이후 최신 연구와 비교

| 연구                 |        연도 | 핵심 아이디어                                                  | T2I와 관계                                            |                       |
| ------------------ | --------: | -------------------------------------------------------- | -------------------------------------------------- | --------------------- |
| **VIME**           |      2020 | mask estimation + value reconstruction self-supervision  | 구조화된 auxiliary objective라는 점에서 선행 아이디어             |                       |
| **TabTransformer** |   2020/21 | categorical contextual embedding + self-attention        | representation learning 방식이 Transformer 기반         |                       |
| **FT-Transformer** |      2021 | 강력한 Transformer tabular baseline                         | T2I가 비교하는 주요 neural baseline                       |                       |
| **SCARF**          |      2022 | random feature corruption + contrastive learning         | stochastic augmentation으로 robust representation 학습 |                       |
| **TabPFN**         | 2023→2025 | synthetic prior data에 사전학습된 foundation model             | 작은 데이터 generalization에 매우 강하지만 큰 사전학습 모델           |                       |
| **TabZilla**       |      2023 | 176 datasets에서 NN/GBDT 비교                                | T2I benchmark 선택의 핵심 배경                            |                       |
| **TableShift**     |      2023 | 자연 발생 tabular distribution shift                         | T2I synthetic robustness의 다음 검증 단계                 |                       |
| **WhyShift**       |      2023 | shift 원인을 $P(X),P(Y \mid X)$ 관점으로 분해                                        | 단순 corruption보다 더 현실적 |
| **HACNet**         |      2024 | end-to-end table→image converter + CNN                   | T2I와 가장 직접적인 architecture 선행연구                     |                       |
| **TabularBench**   |      2024 | constraint-aware adversarial robustness benchmark        | T2I reliability protocol보다 공격 관점이 강함               |                       |
| **RealMLP**        |      2024 | 강하게 pre-tuned된 simple MLP                                | “복잡한 모델이 꼭 필요한가?”라는 강한 반론 baseline                 |                       |
| **TabM**           |      2025 | parameter-efficient MLP ensembling                       | performance/efficiency 관점에서 매우 강한 최신 baseline      |                       |
| **TabFSBench**     |      2025 | feature shift benchmark                                  | T2I IF보다 훨씬 다양한 feature-shift 평가                   |                       |
| **Table2Image v3** |      2026 | proxy generation + multi-failure reliability diagnostics | representation + robustness + compactness를 통합 평가   |                       |

---

## VIME와의 차이

VIME는 corrupted input에서 mask와 원 feature를 복원하면서 tabular dependency를 학습합니다. NeurIPS 2020의 대표적 tabular self-supervised method입니다. ([NeurIPS Papers][5])

VIME:

$$
x
\rightarrow
\tilde x
\rightarrow
\text{recover }x
$$

T2I:

$$
x
\rightarrow
\text{generated proxy }q
\rightarrow
\text{classification}.
$$

즉 T2I는 원 데이터를 복원하기보다 **새로운 auxiliary geometry를 만든다**는 점이 다릅니다.

---

## SCARF와의 차이

SCARF는 random feature corruption으로 두 view를 만들고 contrastive alignment를 수행하며 label noise/semi-supervised 환경에서도 성능 개선을 보고했습니다. ([ML Anthology][6])

SCARF:

$$
x
\rightarrow(x,\tilde x)
\rightarrow
\text{contrastive invariance}
$$

T2I:

$$
x
\rightarrow G(x,r)
\rightarrow
\text{proxy reconstruction + classification}.
$$

따라서 future T2I에서 SCARF-style contrastive consistency를 결합하는 것이 상당히 자연스럽습니다.

---

## FT-Transformer와의 관계

2021년의 FT-Transformer 연구는 tabular DL에 대해 strong ResNet/Transformer baseline과 GBDT를 공정하게 비교했고, 모든 데이터에서 한 방법이 보편적으로 우월하지는 않음을 강조했습니다. ([arXiv][7])

Table2Image에서도 이 교훈은 그대로 중요합니다.

특히 T2I wrapper를 FT-Transformer에 붙였을 때 성능이 크게 악화되므로 “T2I transformation은 보편적인 representation enhancer”라고 일반화해서는 안 됩니다.

---

## TabZilla와의 관계

TabZilla 연구는 176개 dataset, 19개 알고리즘을 대규모로 비교하고 NN과 GBDT의 차이가 많은 dataset에서 작거나 hyperparameter tuning보다 덜 중요할 수 있음을 보여줬습니다. ([NeurIPS Proceedings][8])

이는 T2I의 작은 평균 개선을 해석할 때 매우 중요합니다.

$$
0.8\%
$$

의 average accuracy 차이가 새로운 architecture 자체 덕분인지, tuning/preprocessing 차이인지 더욱 엄격히 통제할 필요가 있습니다.

---

## HACNet과의 관계

HACNet 역시 end-to-end table-to-image converter와 CNN을 연결합니다. Hard attention과 class template를 사용한다는 점이 특징입니다. ([ScienceDirect][9])

T2I의 차별점은:

$$
\text{single representative template}
\rightarrow
\text{instance-varying/randomized proxy}
$$

라는 점입니다.

논문의 multiple-mapping ablation은 이 차별점에 상당한 경험적 근거를 제공합니다.

---

## RealMLP와의 관계

NeurIPS 2024의 RealMLP 연구는 복잡한 architecture 없이도 strong preprocessing과 meta-tuned defaults를 사용하면 GBDT와 경쟁적인 time-accuracy trade-off를 얻을 수 있음을 보였습니다. ([NeurIPS Proceedings][10])

따라서 future T2I 연구에서는 반드시

$$
\text{T2I}
\quad\text{vs}\quad
\text{strongly tuned RealMLP}
$$

을 비교해야 합니다.

그렇지 않으면 proxy pathway의 이점과 “MLP training recipe”의 이점을 구분하기 어렵습니다.

---

## TabM과의 관계

ICLR 2025의 TabM은 BatchEnsemble 계열의 parameter-efficient ensembling을 통해 단순한 MLP 계열에서도 매우 강한 tabular 성능을 얻습니다. ([ICLR Proceedings][11])

T2I와 TabM의 철학은 흥미롭게 대비됩니다.

$$
\text{T2I}:
\text{representation diversity}
$$

$$
\text{TabM}:
\text{prediction/model diversity}
$$

두 방식을 결합하면

$$
\{
G(x,r_k)
\}_{k=1}^{K}
\rightarrow
\{\hat y_k\}_{k=1}^{K}
\rightarrow
\frac1K\sum_k\hat y_k
$$

같은 모델을 만들 수 있습니다.

이는 제가 가장 유망하다고 보는 후속 연구 중 하나입니다.

---

## TabPFN과의 관계

TabPFN 계열은 많은 synthetic task에서 사전학습한 prior-data fitted network라는 전혀 다른 전략입니다. 2025년 Nature 논문은 small-data tabular problem에서 강력한 성능을 보고했습니다. ([Nature][4])

T2I의 장점은 별도의 대규모 prior training 없이 dataset별로 직접 학습할 수 있다는 것이고, TabPFN의 강점은 사전학습된 prior를 활용한 few-shot/small-data generalization입니다.

따라서 앞으로는 특히

$$
n < 1{,}000,\quad
1{,}000 < n < 10{,}000,\quad
n > 10{,}000
$$

처럼 sample size별로 비교해야 합니다.

---

## TableShift / WhyShift / TabFSBench와의 관계

TableShift는 15개 real-world shift task를 사용하며 ID와 OOD accuracy 사이에 강한 관계가 있음을 보였습니다. ([NeurIPS Proceedings][3])

WhyShift는 자연 발생 tabular shift에서 $Y|X$ 변화가 자주 발생한다고 보고했습니다. ([NeurIPS Proceedings][2])

ICML 2025의 TabFSBench는 네 가지 feature shift scenario를 이용하여 많은 모델이 open-environment feature shift에 취약하다고 보고합니다. ([Proceedings of Machine Learning Research][12])

따라서 Table2Image의 다음 단계는 synthetic IF/NL/SD를 더 만드는 것이 아니라

$$
\boxed{
\text{TableShift}
+
\text{WhyShift}
+
\text{TabFSBench}
}
$$

위에서 동일 DPI/LCI/SAI 철학을 적용해 보는 것입니다.

---

# 9. 앞으로 연구한다면 가장 중요한 실험

제가 이 논문을 후속 연구한다면 다음 순서가 가장 합리적입니다.

### ① Proxy geometry 제거 실험

$$
q_{\text{random}},
\quad
q_{\text{orthogonal}},
\quad
q_{\text{learned}},
\quad
q_{\text{Gaussian-periodic}}
$$

을 동일 generator에서 비교합니다.

목적:

> 정말 handcrafted proxy geometry가 중요한가?

---

### ② Generator stochasticity ablation

$$
G(P(x),r)
\quad\text{vs}\quad
G(P(x)).
$$

목적:

> random latent noise가 일반화에 기여하는가?

---

### ③ Latent dimension sweep

이미지를 고정하지 않고

$$
d_{\text{proxy}}\in\{16,32,64,128,256,784\}
$$

를 비교합니다.

28×28이 본질적인지 검증할 수 있습니다.

---

### ④ Natural shift benchmark

TableShift·WhyShift·TabFSBench에서

$$
ACC,\quad AUC,\quad APS,\quad
\text{instability}
$$

를 동시에 측정합니다.

---

### ⑤ Accuracy-stability Pareto optimization

$$
\min_\theta
\left[
L_{\text{cls}}
+
\lambda I_{\text{prediction}}
\right].
$$

단순 정확도 최대화가 아니라 “높은 정확도 + 낮은 instability”를 직접 학습합니다.

---

### ⑥ Strong baseline update

반드시 다음과 비교해야 합니다.

$$
\text{RealMLP},
\quad
\text{TabM},
\quad
\text{TabPFN},
\quad
\text{CatBoost/XGBoost}.
$$

2026년 시점에서는 단순 MLP나 FT-Transformer만으로 강한 baseline을 대표하기 어렵습니다.

---

# 10. 종합 연구자 평가

제가 이 논문을 한 문장으로 평가한다면:

> **Table2Image의 가장 중요한 기여는 “tabular data를 이미지로 바꾸었다”는 데 있지 않고, 작은 모델 내부에 stochastic structured proxy-generation bottleneck을 삽입하고 그 효과를 성능·corruption robustness·realization stability의 세 축에서 동시에 진단했다는 데 있습니다.**

성능 측면에서는 분명 경쟁력이 있지만 “압도적인 SOTA”라는 표현은 적절하지 않습니다. External proxy 결과는 매우 높지만 외부 class-conditioned image를 사용한다는 조건 차이가 있고, internal proxy 기준 MLP 대비 clean 개선은 약 $0.8$ percentage point입니다.

반면 방법론적으로 더 흥미로운 결과는:

$$
\boxed{
\text{generator removal}
\gg
\text{proxy permutation/removing reconstruction loss}
}
$$

이라는 ablation입니다.

이는 미래 연구가 **“어떤 그림을 만들 것인가?”보다 “왜 stochastic generation bottleneck이 tabular generalization을 돕는가?”**를 이론적으로 밝히는 방향으로 가야 함을 시사합니다.

또한 T2I의 높은 LCI는 매우 중요합니다. 높은 평균 accuracy와 낮은 prediction instability는 동일한 목표가 아니므로 앞으로 tabular 모델 평가는

$$
\boxed{
\text{Accuracy}
+
\text{OOD Robustness}
+
\text{Prediction Stability}
+
\text{Calibration}
+
\text{Computational Cost}
}
$$

로 확장하는 것이 타당합니다.

---

# 11. 참고한 논문·사이트 및 자료

**주 분석 원문**

1. **Lee et al., *Table2Image: Lightweight Tabular Learning with Generated Proxy Representations and Reliability Diagnostics*, arXiv:2412.06265v3, 2026** — 사용자가 첨부한 PDF. 
2. **Academus — *Table2Image: Lightweight Tabular Learning with Generated Proxy Representations and Reliability Diagnostics*** — v3 웹 인덱스. ([Academ.us][13])
3. **SciRate — arXiv:2412.06265v3 entry** — v3 title/author metadata 확인. ([Scirate][14])
4. **arXiv — *Table2Image: Interpretable Tabular Data Classification with Realistic Image Transformations*** — 현재 검색 캐시에 남아 있는 이전 v2 metadata 확인용. ([arXiv][1])
5. **GitHub — duneag2/table2image** — 저자 공개 구현 repository 및 VIF 실행 코드 존재 확인. ([GitHub][15])

**2020년 이후 관련 연구 비교**
6. **Yoon et al., *VIME: Extending the Success of Self- and Semi-supervised Learning to Tabular Domain*, NeurIPS 2020.** ([NeurIPS Papers][5])
7. **Huang et al., *TabTransformer: Tabular Data Modeling Using Contextual Embeddings*, 2020.** ([arXiv][16])
8. **Gorishniy et al., *Revisiting Deep Learning Models for Tabular Data*, NeurIPS 2021.** ([NeurIPS Proceedings][17])
9. **Bahri et al., *SCARF: Self-Supervised Contrastive Learning Using Random Feature Corruption*, ICLR 2022.** ([ML Anthology][6])
10. **McElfresh et al., *When Do Neural Nets Outperform Boosted Trees on Tabular Data?*, NeurIPS 2023 / TabZilla.** ([NeurIPS Proceedings][8])
11. **Gardner et al., *Benchmarking Distribution Shift in Tabular Data with TableShift*, NeurIPS 2023.** ([NeurIPS Proceedings][3])
12. **Liu et al., *On the Need for a Language Describing Distribution Shifts: Illustrations on Tabular Datasets*, NeurIPS 2023 / WhyShift.** ([NeurIPS Proceedings][2])
13. **Matsuda et al., *HACNet: End-to-end learning of interpretable table-to-image converter and convolutional neural network*, Knowledge-Based Systems, 2024.** ([ScienceDirect][9])
14. **Simonetto et al., *TabularBench: Benchmarking Adversarial Robustness for Tabular Deep Learning in Real-world Use-cases*, NeurIPS 2024.** ([NeurIPS Papers][18])
15. **Holzmüller et al., *Better by Default: Strong Pre-Tuned MLPs and Boosted Trees on Tabular Data*, NeurIPS 2024.** ([NeurIPS Proceedings][10])
16. **Gorishniy et al., *TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling*, ICLR 2025.** ([ICLR Proceedings][11])
17. **Hollmann et al., *Accurate predictions on small data with a tabular foundation model*, Nature, 2025.** ([Nature][4])
18. **Cheng et al., *TabFSBench: Tabular Benchmark for Feature Shifts in Open Environments*, ICML 2025.** ([Proceedings of Machine Learning Research][12])

종합하면, **Table2Image v3는 “경량 tabular architecture”보다 “proxy-generation 기반 representation regularization + reliability diagnostics” 논문으로 읽을 때 연구 가치가 가장 분명합니다.** 특히 다음 논문에서는 “Gaussian/periodic 이미지를 왜 쓰는가?”보다 **generator 자체가 어떤 regularization·information bottleneck·margin 효과를 만들어 일반화 성능을 높이는지 이론적으로 규명하는 것**이 가장 중요한 후속 과제라고 판단합니다.

[1]: https://arxiv.org/abs/2412.06265 "[2412.06265] Table2Image: Interpretable Tabular Data Classification with Realistic Image Transformations"
[2]: https://proceedings.neurips.cc/paper_files/paper/2023/hash/a134eaebd55b7406ff29cd75d5f1a622-Abstract-Datasets_and_Benchmarks.html?utm_source=chatgpt.com "On the Need for a Language Describing Distribution Shifts: Illustrations on Tabular Datasets"
[3]: https://proceedings.neurips.cc/paper_files/paper/2023/hash/a76a757ed479a1e6a5f8134bea492f83-Abstract-Datasets_and_Benchmarks.html?utm_source=chatgpt.com "Benchmarking Distribution Shift in Tabular Data with TableShift"
[4]: https://www.nature.com/articles/s41586-024-08328-6?utm_source=chatgpt.com "Accurate predictions on small data with a tabular foundation model | Nature"
[5]: https://papers.neurips.cc/paper_files/paper/2020/hash/7d97667a3e056acab9aaf653807b4a03-Abstract.html?utm_source=chatgpt.com "VIME: Extending the Success of Self- and Semi-supervised Learning to Tabular Domain"
[6]: https://mlanthology.org/iclr/2022/bahri2022iclr-scarf/?utm_source=chatgpt.com "Scarf: Self-Supervised Contrastive Learning Using Random Feature Corruption | ML Anthology"
[7]: https://arxiv.org/abs/2106.11959?utm_source=chatgpt.com "Revisiting Deep Learning Models for Tabular Data"
[8]: https://proceedings.neurips.cc/paper_files/paper/2023/hash/f06d5ebd4ff40b40dd97e30cee632123-Abstract.html?utm_source=chatgpt.com "When Do Neural Nets Outperform Boosted Trees on Tabular Data?"
[9]: https://www.sciencedirect.com/science/article/pii/S0950705123010419?utm_source=chatgpt.com "HACNet: End-to-end learning of interpretable table-to-image converter and convolutional neural network - ScienceDirect"
[10]: https://proceedings.neurips.cc/paper_files/paper/2024/hash/2ee1c87245956e3eaa71aaba5f5753eb-Abstract.html?utm_source=chatgpt.com "Better by default: Strong pre-tuned MLPs and boosted trees on tabular data"
[11]: https://proceedings.iclr.cc/paper_files/paper/2025/hash/c1ba41c694834aeef91ae161711d4939-Abstract-Conference.html?utm_source=chatgpt.com "TabM: Advancing tabular deep learning with parameter-efficient ensembling"
[12]: https://proceedings.mlr.press/v267/cheng25e.html?utm_source=chatgpt.com "TabFSBench: Tabular Benchmark for Feature Shifts in Open Environments"
[13]: https://academ.us/article/2412.06265/?utm_source=chatgpt.com "[2412.06265] Table2Image: Lightweight Tabular Learning with Generated Proxy Representations and Reliability Diagnostics - Academus scientific article reader"
[14]: https://scirate.com/search?q=au%3AKwak_I+in%3Acs&utm_source=chatgpt.com "au:Kwak_I in:cs - SciRate Search"
[15]: https://github.com/duneag2/table2image "GitHub - duneag2/table2image · GitHub"
[16]: https://arxiv.org/abs/2012.06678?utm_source=chatgpt.com "TabTransformer: Tabular Data Modeling Using Contextual Embeddings"
[17]: https://proceedings.neurips.cc/paper_files/paper/2021/file/9d86d83f925f2149e9edb0ac3b49229c-Paper.pdf?utm_source=chatgpt.com "Revisiting Deep Learning Models for Tabular Data"
[18]: https://papers.nips.cc/paper_files/paper/2024/hash/8f5defaa8dc728c3e4aaee59ac8ca942-Abstract-Datasets_and_Benchmarks_Track.html?utm_source=chatgpt.com "TabularBench: Benchmarking Adversarial Robustness for Tabular Deep Learning in Real-world Use-cases"

# v2 vs v3
두 버전을 나란히 읽으면, **v3는 v2의 단순한 보강판이 아니라 논문의 중심 질문 자체를 바꾼 대규모 재설계**에 가깝습니다.

v2의 핵심 질문이

$$
\boxed{\text{“표 데이터를 현실적인 이미지로 바꾸면 성능과 해석가능성을 함께 얻을 수 있는가?”}}
$$

였다면, v3의 질문은

$$
\boxed{\text{“구조화된 proxy representation을 생성하는 경량 모델이 성능·효율·신뢰성을 함께 확보할 수 있는가?”}}
$$

로 바뀌었습니다. 특히 **“realistic image”와 “interpretability/DualSHAP”가 중심에서 빠지고, “proxy representation”과 “reliability diagnostics”가 그 자리를 차지한 것**이 가장 중요한 변화입니다.

웹상 버전 정보에는 약간의 비동기성이 있습니다. arXiv의 일반 abstract 페이지는 아직 v2를 “2025-01-23 마지막 수정”으로 표시하지만, 동일 arXiv HTML의 새 본문과 SciRate·Academus는 v3의 새 제목과 내용을 노출합니다. 첨부된 v3 PDF 자체에는 `arXiv:2412.06265v3 ... 20 Aug 2026`이 명시되어 있습니다. ([arXiv][1]) 

---

# 1. 한눈에 보는 핵심 차이

| 구분               | v2: *Interpretable…Realistic Image Transformations*          | v3: *Lightweight…Proxy Representations and Reliability Diagnostics*        | 중요도           |
| ---------------- | ------------------------------------------------------------ | -------------------------------------------------------------------------- | ------------- |
| 연구의 중심           | 성능 + **해석가능성** + lightweight                                 | 성능 + parameter efficiency + **reliability/robustness**                     | **매우 큼**      |
| 생성 결과의 의미        | “realistic/diverse image”                                    | “structured proxy representation”                                          | **개념적 재정의**   |
| 생성 target        | FashionMNIST/MNIST 외부 이미지                                    | External image + **새 internal proxy**                                      | **핵심 추가**     |
| 이미지 의미론          | 현실적 시각 representation 강조                                     | internal mode는 **image semantics를 의도하지 않음**                                | **주장의 방향 전환** |
| 생성부 명칭           | MLP-based **autoencoder**                                    | learned **proxy-generation pathway / generator**                           | 중요한 정리        |
| Loss             | $L_{\text{recon}}+L_{\text{cls}}$                            | $L_{\text{cls}}+\lambda_{\text{recon}}L_{\text{recon}}$                    | 개선            |
| Interpretability | **DualSHAP**가 주요 기여                                          | 사실상 삭제                                                                     | **가장 큰 삭제**   |
| VIF              | 중요 기여, robustness/stability 향상을 강하게 주장                       | 방식은 거의 유지, 주장은 더 신중                                                        | 주장 완화         |
| Reliability 평가   | 사실상 없음                                                       | IF / NL / SD + APS/WSP/ $\rho_\tau$ + DPI/LCI/SAI                           | **대규모 신설**    |
| 통계검정             | 평균·win count 중심                                              | Friedman, Kendall's $W$, Holm-Wilcoxon, CI                                 | **엄밀성 증가**    |
| Ablation         | single/multiple mapping, VIF variants                        | 기존 ablation + generator-free, no recon, permutation, encoder compatibility | **원인분석 강화**   |
| 클래스 제한           | 사실상 $C\le20$                                                 | external은 그대로, **internal은 클래스 수 제한 없음**                                   | 일반화 범위 확대     |
| 핵심 성능 수치         | OpenML/TabZilla 결과                                           | external 결과가 사실상 동일; 신규 internal/robustness 결과 추가                          | 중요            |
| Future work      | better mapping, text/audio, multimodal, DualSHAP assumptions | broader tasks, natural shifts, flexible proxy representations              | 연구방향 변경       |
| 저자               | 7명                                                           | 8명, Julia Stoyanovich 추가                                                   | 메타데이터 변화      |

---

# 2. 가장 큰 변화: “이미지 변환 논문”에서 “representation-learning 논문”으로

## v2의 논리

v2는 표 데이터가 “compressed latent-space-like representation”이고, 이미지는 보다 explicit한 “uncompressed representation”이라는 직관을 출발점으로 둡니다. 그래서 Table2Image가 표 데이터를 **현실적이고 해석 가능한 이미지**로 변환하면 CNN이 패턴을 더 잘 추출할 수 있다고 주장합니다. 제목부터 *Realistic Image Transformations*이고, abstract에서도 “realistic and diverse image representations”가 핵심입니다. 

v2의 파이프라인은 개념적으로

$$
x
\rightarrow
P(x)
\rightarrow
AE(P(x),r)
\approx
M(x)
\rightarrow
CNN
\rightarrow
\hat y
$$

입니다.

여기서 $M(x)$는 FashionMNIST/MNIST에서 **정답 class $y$에 맞추어 무작위 선택한 실제 이미지**입니다. v2는 이를 “현실적인 이미지”를 생성한다는 관점에서 설명합니다. 

---

## v3의 논리

v3는 이 주장을 훨씬 더 일반적인 representation-learning 문제로 재구성합니다.

$$
x
\rightarrow
P(x)
\rightarrow
G(P(x),r)
\rightarrow
\text{structured proxy}
\rightarrow
CNN
\rightarrow
\hat y.
$$

생성된 2D representation이 실제 물체처럼 보여야 한다는 요구를 없애고,

> downstream classification에 유용한 **structured auxiliary representation**이면 충분하다

는 방향으로 갑니다. v3 abstract와 introduction 역시 “realistic image”보다 **learned generation pathway, structured proxy representation, reliability**를 전면에 둡니다. 

더 결정적으로 v3는 internal proxy를 설명하면서 이를 명시적으로

> image semantics를 전달하기 위한 것이 아니라 structured auxiliary learning signal을 제공하기 위한 것

이라고 설명합니다. 

그리고 생성 결과 분석에서도

> photorealistic reconstruction이 목적이 아니다

라는 점을 명시합니다. 

### 제 해석

이 변화는 단순한 용어 변경 이상의 의미가 있습니다.

v2:

$$
\text{“tabular}\rightarrow\text{realistic image”}
$$

v3:

$$
\boxed{
\text{tabular}
\rightarrow
\text{useful structured latent proxy}
}
$$

즉 v3는 **“왜 굳이 실제 이미지여야 하는가?”라는 v2의 가장 취약한 이론적 질문에서 상당 부분 벗어났습니다.**

연구적으로는 v3의 framing이 더 강합니다.

---

# 3. “Autoencoder”라는 표현도 사실상 수정됐다

v2는 생성 경로를 명시적으로 **MLP-based autoencoder**라고 부릅니다. 

하지만 전통적인 autoencoder는 보통

$$
x\xrightarrow{Encoder}z\xrightarrow{Decoder}\hat x
$$

처럼 원 입력 $x$를 재구성합니다.

v2 Table2Image는 실제로는

$$
(x,r)\rightarrow AE(x,r)\approx M(x)
$$

이고 $M(x)$는 **원래 $x$가 아니라 동일 label을 가진 FashionMNIST/MNIST image**입니다.

따라서 엄밀하게 보면 conventional autoencoder라기보다는 **conditional proxy generator**에 가깝습니다.

v3는 이를 더 적절하게

* proxy representation generator
* learned generation pathway
* generator $G$

라고 부릅니다. 

### 제 평가

이 변경은 방법 자체를 크게 바꾼 것은 아니지만 **개념적 정확성을 크게 높였습니다.**

---

# 4. 가장 큰 방법론적 추가: Internal Proxy Target

v2에는 사실상 external FashionMNIST/MNIST mapping 하나만 있습니다.

$$
x,y
\longrightarrow
i_x,\qquad i_x\sim I_y.
$$

그 결과 전체 framework가

$$
C\le20
$$

이라는 외부 image-class 수에 묶입니다. v2는 $10$ classes 이하에서는 FashionMNIST, 그 이상에서는 MNIST를 합쳐 최대 20 class까지 지원하며 $C>20$은 future work로 둡니다. 

---

## v3에서는 두 모드로 분리

### External mode

기존 v2 방법을 거의 그대로 유지합니다.

$$
p_x^*=i_x.
$$

### Internal mode — 새로 추가

```math
\boxed{
q(x,y)
=
(1-\alpha)T_y+\alpha V(x),
\qquad
\alpha=0.35
}
```

입니다. 

$T_y$는 class template이고 $V(x)$는 instance-specific variation입니다.

또

```math
T_y
=
\text{Norm}
\left(
0.55G_y+0.45P_y
\right)
```

이며 $G_y$는 Gaussian component, $P_y$는 horizontal/vertical/diagonal/radial periodic component입니다. 

---

## 왜 이 추가가 중요한가

v3는 robustness 실험에서 **external proxy를 일부러 사용하지 않습니다.**

저자들은 external image가 “externally sourced, class-conditioned supervision”을 제공하기 때문에 architecture 자체의 robustness와 혼동될 수 있다고 명시합니다. 그래서 robustness는 internal target으로만 평가합니다. 

이것은 v2에 존재하던 중요한 confound를 의식한 설계 변경입니다.

### 다만 중요한 주의점

internal target도

$$
q(x,y)
$$

이므로 **training label $y$를 이용해 target을 생성합니다.**

즉 v3가 제거한 것은

$$
\text{external class-conditioned image information}
$$

이지

$$
\text{class-conditioned auxiliary supervision}
$$

그 자체는 아닙니다.

논문도 $y$는 training-time proxy construction에만 쓰고 inference에서는 generator에 들어가지 않는다고 명시합니다. 

---

# 5. 클래스 수 제한 문제가 부분적으로 해결됨

이 차이는 꽤 중요합니다.

v2:

$$
C > 20
$$

이면 FashionMNIST+MNIST mapping 자체가 불가능합니다.

v3 external mode는 동일한 제한을 유지하지만 internal proxy는

$$
\boxed{\text{class-number restriction 없음}}
$$

이라고 명시합니다. 

따라서 v3에서는 Table2Image라는 **architecture 자체**와 FashionMNIST/MNIST를 이용한 **특정 proxy construction 방식**을 분리할 수 있게 됐습니다.

이는 일반화 가능성 측면에서 v3가 훨씬 좋은 설계입니다.

---

# 6. Loss function도 미묘하지만 중요한 변화

## v2

```math
\boxed{
L_{\text{total}}
=
L_{\text{recon}}
+
L_{\text{cls}}
}
```

입니다. 

즉 두 loss의 상대적 중요도가 사실상 $1:1$로 고정됩니다.

---

## v3

```math
\boxed{
L_{\text{total}}
=
L_{\text{cls}}
+
\lambda_{\text{recon}}L_{\text{recon}}
}
```

로 바뀝니다. 

이는

$$
\lambda_{\text{recon}}
$$

으로 classification과 proxy reconstruction의 역할을 분리할 수 있게 한 것입니다.

### 의미

v2는 implicit하게

$$
\lambda_{\text{recon}}=1
$$

인 special case라고 볼 수 있습니다.

v3가 더 일반적인 objective입니다.

---

# 7. v2의 핵심 기여였던 DualSHAP가 v3에서 완전히 중심에서 빠졌다

이것이 아마 **논문 정체성 측면에서 가장 큰 삭제**입니다.

v2 contribution 4번은 “Enhanced interpretability”이고, 별도의 Section 3.3 **DualSHAP**, Figure 3, Figure 4, Appendix A–C가 여기에 할당되어 있습니다. 

DualSHAP는

* tabular SHAP $\phi_{\text{tab}}$
* image SHAP $\phi_{\text{img}}$
* image→tabular reconstruction $X_{\text{recon}}$
* tabular→image representation $I_{\text{recon}}$

을 사용합니다.

그리고 두 확률변수처럼 취급하는 $S,T$를

$$
S\sim \mathcal N(\mu_s,\sigma_s^2),
\qquad
T\sim \mathcal N(\mu_t,\sigma_t^2)
$$

로 가정합니다. 

최종적으로

```math
P
=
\frac{S\times\phi_{\text{tab}}}
{X_{\text{recon}}},
\qquad
Q
=
\frac{T\times\phi_{\text{img}}}
{I_{\text{recon}}}
```

의 discrepancy를

```math
L_{\text{DualSHAP}}
=
L_{\text{MSE}}
+
L_{\text{KLD}}
+
L_{\text{MMD}}
```

로 줄입니다. 

---

# 8. DualSHAP 삭제는 왜 중요한가

v2 Appendix A에는 상당히 중요한 문장이 있습니다.

저자들은 Bayes theorem에서 영감을 받았지만, 해당 항들을 실제 probability distribution으로 다루는 것이 아니라 **그 의미만 차용한다고 명시**합니다. 

즉 엄밀한 Bayesian derivation이라기보다는 heuristic construction입니다.

또 $S,T$가 normal distribution이라고 가정합니다. 

v2 스스로도 conclusion에서 이 normality assumption을 limitation으로 인정합니다. 

그리고 interpretability 검증은 주로

* 10회 반복 feature importance의 SD,
* shuffled column과 original column explanation의 MSE,
* DualSHAP optimization의 MSE/KLD/MMD

를 봅니다. 

### 여기에는 중요한 통계적 문제가 있습니다

낮은

$$
L_{\text{MSE}},L_{\text{KLD}},L_{\text{MMD}}
$$

는

> **P와 Q를 서로 비슷하게 학습시켰다**

는 증거이지,

> **최종 설명이 모델의 실제 causal/functional behavior를 정확하게 설명한다**

는 evidence는 아닙니다.

즉 **optimization success와 explanation faithfulness가 동일하지 않습니다.**

### 제 해석

저자들이 v3에서 DualSHAP를 제거한 이유를 논문은 명시하지 않습니다.

따라서 “왜 제거했는가”를 사실처럼 말할 수는 없습니다.

다만 연구 설계만 보면 v3는 **검증이 어려운 interpretability claim을 제거하고, 직접 측정 가능한 robustness/stability claim으로 연구 범위를 재설정한 것**으로 해석할 수 있습니다.

이 변화는 학술적 엄밀성 측면에서는 긍정적이라고 봅니다.

---

# 9. Interpretability 자리에 Reliability Diagnostics가 들어왔다

v2의 후반부가 DualSHAP이었다면 v3의 핵심 신규 contribution은 사실상 이 부분입니다.

v3는 세 종류의 failure mode를 정의합니다.

$$
\boxed{
\text{IF}
+
\text{NL}
+
\text{SD}
}
$$

즉,

$$
\text{Irrelevant Features},
\quad
\text{Noisy Labels},
\quad
\text{Shortcut Decorrelation}.
$$



이는 각각

$$
\text{input-side distraction},
$$

$$
\text{supervision corruption},
$$

$$
\text{train-test association shift}
$$

를 대표합니다.

---

## 그리고 성능 하나가 아니라 세 층으로 평가합니다

v3:

$$
\text{clean performance}
$$

*

$$
\text{performance degradation across severity}
$$

*

$$
\text{realization-level instability}.
$$



성능 기반 metric은

$$
APS,
\qquad
WSP,
\qquad
\rho_\tau
$$

이고, stability metric은

$$
DPI,\quad LCI,\quad SAI
$$

입니다.

이건 v2에 전혀 없던 평가 축입니다.

---

# 10. 특히 “동일한 corruption 수준에서도 모델 예측이 흔들리는가?”를 새로 측정한다

두 corruption realization $w,\ell$에 대해

```math
d^{(w,\ell)}(i)
=
\begin{cases}
1,
&
\hat y_i^{(w)}
\neq
\hat y_i^{(\ell)},\\[4pt]
\dfrac12
\left\|
p_i^{(w)}
-
p_i^{(\ell)}
\right\|_1,
&
\hat y_i^{(w)}
=
\hat y_i^{(\ell)}.
\end{cases}
```

로 정의합니다. 

그 뒤 각각

* DPI: irrelevant-feature realization
* LCI: noisy-label realization
* SAI: shortcut assignment realization

으로 통합합니다. 

### 이것이 v2와 질적으로 다른 이유

v2가 설명 안정성을 물었다면,

$$
\text{“feature importance가 안정적인가?”}
$$

v3는 모델 prediction 자체의 안정성을 묻습니다.

$$
\boxed{
\text{“같은 수준의 corruption을 다시 샘플링해도 예측이 안정적인가?”}
}
$$

후자가 deployment reliability와 훨씬 직접적으로 연결됩니다.

---

# 11. 실제로 v3는 “성능이 좋다 = 안정적이다”라는 가정을 깨뜨린다

Table V에서 T2I는

$$
DPI=0.114,
\qquad
LCI=0.291,
\qquad
SAI=0.190.
$$

입니다.

SAI는 가장 좋고 DPI도 좋은 편이지만 **LCI는 비교 모델 중 오히려 높습니다.** 

즉

$$
\boxed{
\text{high robustness performance}
\not\Rightarrow
\text{low prediction instability}
}
$$

라는 훨씬 미묘한 결론을 냅니다.

v2의 “robust/reliable solution”이라는 비교적 광범위한 표현보다 v3의 reliability 주장이 훨씬 측정 가능하고 반증 가능합니다.

---

# 12. VIF 방법 자체는 거의 그대로지만, “주장”은 크게 온건해졌다

## v2

v2 abstract는 VIF initialization이 모델의

* stability,
* robustness,
* performance

를 높인다고 상당히 강하게 서술합니다. 

방법은

```math
VIF_i
=
\frac{1}{1-R_i^2},
\qquad
w_{ij}^{(0)}
=
\frac1{VIF_i}
```

이고,

$$
P(x)\oplus P_{\text{VIF}}(x)
$$

를 사용합니다. 

---

## v3

수학적 방법은 사실상 같습니다.

그러나 설명은

> highly collinear features가 training 시작 시 작은 weight를 갖게 하는 **collinearity-aware starting point**

로 훨씬 제한적으로 바뀝니다.

그리고 중요한 문장을 추가합니다.

$$
\boxed{
\text{VIF determines where optimization starts,
not where it can end.}
}
$$



### 이 수정이 타당한 이유

v3 신규 robustness 결과를 보면 VIF가 항상 T2I보다 좋은 것이 아닙니다.

예를 들어 noisy labels:

$$
APS_{\text{ACC}}:
\quad
T2I=0.846,
\quad
T2I\text{-VIF}=0.844.
$$

Worst severity:

$$
T2I=0.822,
\quad
T2I\text{-VIF}=0.815.
$$



즉 v3에서는

$$
\text{VIF}
\Rightarrow
\text{universally greater robustness}
$$

라고 말하기 어렵다는 것이 드러났습니다.

그 결과 v3의 표현이 더 신중해진 것은 데이터와 잘 맞습니다.

---

# 13. Clean benchmark 결과는 놀랍게도 거의 그대로다

이 부분은 반드시 구분해서 봐야 합니다.

## v2 OpenML-CC18

$$
T2I:
ACC=0.8766,\quad AUC=0.9202
$$

$$
T2I\text{-VIF}:
ACC=0.8787,\quad AUC=0.9219.
$$



## v3

$$
T2I:
0.877/0.920
$$

$$
T2I\text{-VIF}:
0.879/0.922.
$$



숫자를 반올림하면 **정확히 동일합니다.**

TabZilla도 마찬가지입니다.

v2:

$$
T2I=0.8567/0.9059,
\qquad
T2I\text{-VIF}=0.8624/0.9053
$$



v3:

$$
T2I=0.857/0.906,
\qquad
T2I\text{-VIF}=0.862/0.905.
$$



### 결론

적어도 이 aggregate external-proxy benchmark 결과는

$$
\boxed{\text{v2 결과가 v3에 그대로 유지된 것으로 보입니다.}}
$$

즉 v3의 새로운 실험적 가치는 기존 clean leaderboard 수치가 아니라 **internal proxies + robustness + stability + ablation + statistics**에 있습니다.

---

# 14. v3에서는 성능 주장의 표현도 훨씬 보수적이다

v2 abstract는 대략

> superior accuracy, AUC, interpretability

라는 강한 표현을 사용합니다. 

반면 v3 abstract는

> competitive clean predictive performance

와

> favorable balance

를 강조합니다. 

이것은 상당히 중요한 수정입니다.

왜냐하면 v3가 정식 통계검정을 추가하면서, strong baseline에 대해 모든 pairwise comparison이 유의한 것은 아니라는 사실도 드러나기 때문입니다.

---

# 15. 통계적 엄밀성이 크게 높아졌다

v2 main results는 주로

```math
\text{Avg ACC},
\quad
\text{Avg AUC},
\quad
\#\text{Wins}
```

을 사용합니다.

v2 PDF에서 Friedman이나 Wilcoxon 기반 비교는 찾을 수 없고, 실험 설명은 3회 반복 평균을 보고합니다. v2는 8:2 split, batch 64, 100 epochs, 세 번 반복이라고 명시합니다. 

v3는 여기에 정식 across-dataset statistics를 추가합니다.

OpenML의 경우:

$$
p<0.001,
\qquad
W_{\text{ACC}}=0.297,
\qquad
W_{\text{AUC}}=0.316
$$

의 Friedman/Kendall's $W$를 보고합니다. 

그리고 Appendix Table VIII에는

* Holm-corrected Wilcoxon signed-rank,
* median paired difference,
* $95%$ confidence interval

까지 추가했습니다. 

### 결과적으로 오히려 주장이 정교해짐

예를 들어 T2I vs TuneTables OpenML ACC의 Holm-adjusted 값은

$$
p_{\text{Holm}}=0.091
$$

이고 CI 역시 0을 포함합니다.

따라서 v3는 “모든 strong baseline보다 유의하게 우월”하다는 식의 주장을 하지 않고 **competitive**라고 표현하는 것이 더 정확합니다.

이 점에서 v3는 v2보다 통계적으로 성숙합니다.

---

# 16. v3가 새로 추가한 가장 중요한 ablation: “무엇이 실제로 효과를 만드는가?”

v2의 주요 ablation은 크게

1. multiple mapping vs single mapping,
2. VIF initialization variants

였습니다.

예를 들어 multiple mapping은

$$
ACC:0.8876
$$

vs single mapping

$$
0.8483
$$

이었습니다. 

이 결과는 v3에서도 그대로 Table XII로 유지됩니다.

---

## v3 신규 ablation

v3는 internal proxy를 사용하여 세 가지를 제거합니다.

$$
\text{Proxy permutation}
$$

$$
\text{without }L_{\text{recon}}
$$

$$
\text{Generator-free}.
$$

Table VI의 clean ACC:

$$
\text{Full}=0.8760
$$

$$
\text{Permuted}=0.8747
$$

$$
\text{w/o reconstruction}=0.8740
$$

$$
\boxed{
\text{Generator-free}=0.8607
}
$$

입니다. 

### 이 결과는 v2의 원래 narrative를 상당히 수정합니다

v2의 설명은 거의

$$
\text{realistic image target}
\rightarrow
\text{better representation}
\rightarrow
\text{better prediction}
$$

이었습니다.

하지만 v3 결과는 오히려

$$
\boxed{
\text{specific proxy geometry/reconstruction} < \text{learned generation pathway itself}
}
$$

를 지지합니다.

Proxy를 permutation해도 성능 변화가 작고 $L_{\text{recon}}$을 제거해도 차이가 작은데 **generator 전체를 없애면 가장 크게 떨어집니다.**

따라서 v3가 제목을 “image transformation”에서 “generated proxy representations”으로 바꾼 것은 실험 결과와도 잘 맞습니다.

---

# 17. 새로운 encoder-generalization 실험도 v2에는 없었다

v3는 T2I pathway를 MLP와 FT-Transformer에 붙여봅니다.

MLP 50% capacity:

$$
ACC:
0.868\rightarrow0.872.
$$

MLP 25%:

$$
0.867\rightarrow0.871.
$$

하지만 FT-Transformer는

$$
ACC:
0.865\rightarrow0.788
$$

$$
AUC:
0.934\rightarrow0.895.
$$



### 이것 역시 v3의 장점

v2에서는 architecture가 성공한 사례 중심이었지만 v3는

> T2I pathway가 모든 encoder에서 작동하지 않는다

는 **negative result에 가까운 결과도 공개**합니다.

따라서 v3의 일반화 주장은 오히려 더 제한적이지만, 연구적으로는 더 신뢰할 만합니다.

---

# 18. Figure 구성 변화만 봐도 논문의 정체성이 완전히 달라졌다

| v2                | 역할                            | v3    | 역할                                                  |
| ----------------- | ----------------------------- | ----- | --------------------------------------------------- |
| Fig.1             | Table2Image autoencoder→CNN   | Fig.1 | proxy generator + CNN + VIF 통합                      |
| Fig.2             | VIF initialization            | Fig.2 | parameter-performance trade-off                     |
| Fig.3             | **DualSHAP**                  | Fig.3 | **corruption severity curves**                      |
| Fig.4             | interpretability example      | Fig.4 | T2I vs MLP clean/APS dataset comparison             |
| Fig.5             | Pixel Unshuffle               | Fig.5 | generated external proxy examples                   |
| Appendix H images | realistic/prototypical images | Fig.6 | internal target construction + learning progression |

v2의 시각적 중심은

$$
\text{interpretability}
$$

였고, v3의 시각적 중심은

$$
\boxed{
\text{robustness + mechanism + representation}
}
$$

입니다.

---

# 19. 기존 실험 중 상당수는 삭제된 것이 아니라 “부록으로 이동”했다

이 점도 중요합니다.

v2 Table 6의 VIF variants:

$$
T2I=0.8766,
\quad
T2I\text{-VIF}=0.8787,
\quad
T2I\text{-DIR}=0.6089,
\quad
T2I\text{-MUL}=0.8711
$$

은 v3에서 사실상 같은 결과로 Table XIV에 남습니다. 

DeepInsight/HACNet/IGTD 비교도 v2 Appendix Table 7/8에서 v3 Table XIII으로 이동했습니다.  

즉 v3는 기존 Table2Image evidence를 버린 것이 아니라,

$$
\text{기존 clean/image evidence}
\rightarrow
\text{appendix/supporting evidence}
$$

로 낮추고,

$$
\text{reliability/mechanistic analysis}
\rightarrow
\text{main contribution}
$$

로 올렸습니다.

---

# 20. “Realistic image”라는 주장도 훨씬 약해졌다

v2에서는 생성된 이미지가

* visually realistic,
* class-typical,
* interpretable

하다는 narrative가 중요했습니다. 실제로 v2는 autoencoder가 average/prototypical image를 만든다고 설명합니다. 

v3 Figure 5에서는 external target일 때 여전히 의류/신발 class별 pattern을 보여주지만, internal target에서는 Gaussian/periodic pattern을 그대로 사용합니다. 그리고 논문은 이것이 photorealistic해야 할 이유가 없다고 명시합니다. 

### 연구적으로는 중요한 철학 변화입니다

v2:

$$
\text{visual realism}
\approx
\text{useful representation}
$$

이라는 암묵적 관계.

v3:

$$
\boxed{
\text{visual realism is unnecessary}
}
$$

$$
\boxed{
\text{structured auxiliary representation is sufficient}
}
$$

입니다.

제가 보기에는 이것이 Table2Image를 훨씬 더 일반적인 ML 방법으로 만드는 변화입니다.

---

# 21. Limitations/Future Work도 완전히 바뀐다

## v2

주요 limitation은:

$$
\text{no regression}
$$

그리고 DualSHAP에서

$$
S,T\sim \mathcal N
$$

가정입니다.

Future work는

* statistical property를 활용한 더 정확한 tabular-image mapping,
* random mapping 탈피,
* text/audio transformation,
* multimodal architecture

입니다. 

---

## v3

명시적 limitation은

$$
\text{classification only}
$$

$$
\text{controlled synthetic corruption only}
$$

$$
\text{external proxy mode }C\le20
$$

입니다.

Future work는

$$
\text{broader tasks}
+
\text{naturally occurring shifts}
+
\text{more flexible proxy representations}.
$$

즉 미래 방향 역시

$$
\text{multimodality/interpretability}
$$

에서

$$
\boxed{
\text{generalization/reliability/representation learning}
}
$$

으로 이동했습니다.

---

# 22. 저자 구성도 변경됐다

v2는 7명입니다.

Seungeun Lee, Il-Youp Kwak, Kihwan Lee, Subin Bae, Sangjun Lee, Seulbin Lee, Seungsang Oh. 

v3에서는 **Julia Stoyanovich가 새로 추가되어 8명**이 되고, NYU affiliation이 추가됩니다. Seungeun Lee 역시 v3에서는 NYU affiliation으로 기재됩니다. 

저자순서도

v2:

$$
\text{Lee, Kwak, Lee, Bae, Lee, Lee, Oh}
$$

에서 v3:

$$
\text{Lee, Lee, Bae, Lee, Lee, Stoyanovich, Kwak, Oh}
$$

로 크게 바뀝니다.

이 변화의 이유는 논문에 설명되어 있지 않으므로 그 이상은 추정하지 않는 것이 맞습니다.

---

# 23. 실험 infrastructure도 변경

v2는 실험을

* 8:2 train-test,
* batch size 64,
* 100 epochs,
* three repetitions,
* NVIDIA V100

으로 설명합니다. 

v3도

* 8:2 split,
* batch size 64,
* 100 epochs,
* three runs

은 유지하지만 hardware는

* Intel Xeon Platinum 8592+
* NVIDIA A100 80GB

로 바뀝니다. 

따라서 새로운 reliability 실험은 최소한 기존 v2와 동일한 hardware 그대로 재실행한 것은 아닙니다.

---

# 24. 코드 공개 측면에서 중요한 불일치가 하나 있다

v3 논문은 코드가 GitHub에 공개되어 있다고 명시합니다. 

그런데 현재 공개 GitHub README를 확인하면 여전히 제목을

> “A Lightweight Framework for Enhanced Tabular Data Classification with Image Transformation”

으로 표시하고,

* `run.py`
* `run_vif.py`
* “interpretation directory for DualSHAP”

를 안내합니다. 즉 README는 아직 **v2 구조와 DualSHAP를 상당히 강하게 반영**합니다. ([GitHub][2])

### 따라서

현재 README만 보고는 v3의

* internal proxy target,
* IF/NL/SD protocol,
* DPI/LCI/SAI,
* generator ablation

을 그대로 재현하는 경로가 명확하지 않습니다.

이것은 **v3 재현성 측면에서 확인해야 할 사항**입니다.

다만 README에 없다고 해서 repository 내부 어디에도 코드가 없다고 단정할 수는 없습니다. 제가 확인한 범위에서 정확하게 말하면 **현재 공개 README가 v3 기능을 문서화하지 않고 있다**는 것입니다.

---

# 25. 무엇이 그대로 유지됐는가

완전히 새 논문은 아닙니다. 핵심 architecture의 backbone은 상당 부분 동일합니다.

두 버전 모두

```math
P(x)
=
\text{ReLU}
\left(
FC_2(
\text{ReLU}(FC_1(x))
)
\right)
```

를 사용하며,

random noise

$$
r\in\mathbb R^{28\times28}
$$

를 $P(x)$와 concatenate합니다.

생성 경로의 핵심도 사실상 같습니다.

```math
z
=
\text{ReLU}\left(
FC_4\left(
\text{ReLU}\left(
FC_3(\text{flatten}(r)\oplus P(x))
\right)
\right)
\right),
```

```math
G(P(x),r)
=
\text{reshape}
\left[
\sigma
\left(
FC_6(
\text{ReLU}(FC_5(z\oplus P(x)))
)
\right)
\right].
```

v3의 generator architecture는 v2 AE decoder의 재발명이라기보다 **동일한 핵심 구조를 더 일반적인 representation-learning language로 재정의한 것**에 가깝습니다. 

VIF initialization도 수식적으로 거의 동일합니다.

---

# 26. 논문 주장 변화의 “계보”

| v2 주장                               | v3에서의 상태                                   | 제 평가                                       |
| ----------------------------------- | ------------------------------------------ | ------------------------------------------ |
| “realistic image가 핵심”               | **약화/사실상 폐기**                              | 좋은 수정                                      |
| “이미지가 interpretable하다”              | DualSHAP 포함 전체 interpretability 축 삭제       | 주장 범위 축소                                   |
| “VIF가 robustness/stability를 향상”     | 초기화 역할로 더 제한적 설명                           | 더 정확                                       |
| “T2I가 superior”                     | “competitive/favorable balance”            | 통계 결과와 더 잘 부합                              |
| “random multi-mapping이 single보다 좋다” | 유지                                         | 여전히 근거 있음                                  |
| “lightweight”                       | 유지                                         | large neural baselines 대비라는 조건 필요          |
| “image transformation architecture” | “proxy-generation pathway”                 | 훨씬 일반적                                     |
| robust/reliable solution            | 실제 corruption/stability metric으로 **새로 검증** | 큰 개선                                       |
| $C\le20$                            | internal proxy에서 해결                        | 일반화성 개선                                    |
| multimodal 확장                       | 후퇴                                         | reliability/natural shift가 future work로 이동 |

---

# 27. v3가 학술적으로 더 강해진 부분

제가 두 버전을 reviewer 관점에서 비교하면 v3가 명백히 강해진 부분은 네 가지입니다.

첫째, **claim–evidence alignment**가 좋아졌습니다. v2의 “realistic”이나 DualSHAP의 “interpretability”는 객관적 검증이 쉽지 않았습니다. 반면 v3의 IF/NL/SD와 DPI/LCI/SAI는 정의가 명시적이고 재현 가능한 quantitative claim입니다.

둘째, **외부 이미지에 대한 의존성을 줄였습니다.** internal proxy는 external dataset 없이 같은 architecture를 평가할 수 있고 $C>20$ 문제도 해결합니다. 

셋째, **mechanism ablation이 생겼습니다.** generator 자체가 가장 중요한지, reconstruction target의 geometry가 중요한지를 분리합니다. 결과상 generator-free degradation이 가장 큽니다. 

넷째, **통계적 표현이 신중해졌습니다.** “superior”보다 “competitive”, 단순 평균보다 Friedman/Kendall/Wilcoxon/CI가 추가됐습니다.

---

# 28. 반대로 v3에서 잃은 것도 있다

v3가 무조건 모든 측면에서 상위 버전인 것은 아닙니다.

v2는 최소한 명시적으로

$$
\text{“왜 이 모델의 prediction을 그렇게 했는가?”}
$$

라는 explainability 문제에 답하려고 시도했습니다.

v3는 그 축을 거의 완전히 내려놓았습니다.

따라서 연구 목적이 의료·금융처럼 **feature-level explainability가 필수**라면,

$$
\boxed{
v3\text{가 v2의 interpretability 기능을 개선한 것이 아니라,
그 문제를 연구 범위에서 제외한 것}
}
$$

에 가깝습니다.

즉 reliability는 올라갔지만 explainability evidence는 사라졌습니다.

---

# 29. 가장 중요한 비판적 변화: v3 결과는 사실 v2의 “realistic image 가설”을 약화시킨다

이것이 두 버전을 함께 읽었을 때 가장 흥미로운 결론입니다.

v2의 핵심 narrative는 현실적인 FashionMNIST/MNIST representation이 중요한 것이었습니다.

그런데 v3에서는:

$$
\text{hand-crafted internal pattern}
$$

도 잘 작동하고,

proxy permutation의 효과는 매우 작으며,

$$
L_{\text{recon}}
$$

을 제거해도 효과가 작고,

generator 자체를 제거할 때 가장 크게 악화됩니다. 

따라서 데이터를 종합하면,

$$
\boxed{
\text{Realistic image semantics}
\text{가 성능의 핵심 원인이라는 evidence는 약해졌다.}
}
$$

오히려

$$
\boxed{
\text{learned stochastic generation pathway}
}
$$

라는 inductive bias가 핵심일 가능성이 커졌습니다.

이것은 v2와 v3 사이에서 단순히 설명이 바뀐 것이 아니라 **모델이 작동하는 이유에 대한 저자들의 실험적 이해가 바뀐 것으로 볼 수 있는 부분**입니다.

---

# 30. 두 버전을 연구 논문으로 평가하면

제 평가는 다음과 같습니다.

### v2의 정체성

$$
\boxed{
\text{Tabular-to-Image}
+
\text{VIF}
+
\text{DualSHAP}
}
$$

한 논문에 성능, image transformation, 통계적 feature engineering, interpretability, multimodality까지 많은 주제를 담습니다.

장점은 아이디어가 넓고 시각적으로 직관적이라는 것입니다.

하지만 이 때문에 **한 논문이 증명해야 하는 것이 너무 많고**, 특히 “realism→performance”, “DualSHAP→faithful interpretation”의 인과적 근거가 상대적으로 약합니다.

### v3의 정체성

$$
\boxed{
\text{Proxy-generation architecture}
+
\text{parameter efficiency}
+
\text{reliability characterization}
}
$$

으로 훨씬 집중됩니다.

제 판단으로는 **v3가 더 좁지만 훨씬 강한 논문**입니다.

특히

$$
\text{representation mechanism}
\rightarrow
\text{ablation}
\rightarrow
\text{controlled corruption}
\rightarrow
\text{prediction instability}
$$

라는 논리 구조가 명확해졌습니다.

---

# 최종 결론

두 버전의 관계를 한 문장으로 요약하면,

$$
\boxed{
\text{v2는 “표를 현실적 이미지로 변환하는 해석가능 모델”이었고,
v3는 “구조화된 proxy를 생성하여 일반화와 신뢰성을 연구하는 경량 representation-learning 모델”이다.}
}
$$

따라서 **v3는 v2에 robustness experiment 몇 개를 추가한 revision이 아닙니다.** 핵심 연구 질문, 주요 contribution, 방법을 해석하는 방식, evaluation philosophy, limitation, future work까지 바뀌었습니다.

특히 연구자로서 가장 주목해야 할 변화는 다음 관계입니다.

$$
\boxed{
\text{Realistic Image Transformation}
\;\longrightarrow\;
\text{Structured Proxy Representation}
}
$$

$$
\boxed{
\text{DualSHAP Interpretability}
\;\longrightarrow\;
\text{Reliability Diagnostics}
}
$$

$$
\boxed{
\text{“무엇처럼 보이는가?”}
\;\longrightarrow\;
\text{“왜 일반화되고 얼마나 안정적인가?”}
}
$$

그리고 v3의 ablation 결과를 함께 보면, 향후 연구의 핵심도 **더 realistic한 image를 만드는 것보다는 generation pathway가 왜 일반화에 도움이 되는지, 어떤 proxy geometry·stochasticity·regularization이 최적인지를 규명하는 것**으로 이동하는 것이 타당합니다.

웹에서 추가 확인한 주요 자료는 **arXiv:2412.06265의 v2 submission history 및 HTML 본문**, **Academus의 v3 *Table2Image: Lightweight Tabular Learning with Generated Proxy Representations and Reliability Diagnostics***, **SciRate의 v3 metadata**, 그리고 **저자 공식 GitHub `duneag2/table2image`**입니다. arXiv의 일반 metadata가 아직 v2를 노출하는 반면 다른 인덱스와 첨부 v3 PDF는 새 제목을 반영한다는 점은 버전 추적 시 주의해야 합니다. ([arXiv][1])

[1]: https://arxiv.org/abs/2412.06265 "[2412.06265] Table2Image: Interpretable Tabular Data Classification with Realistic Image Transformations"
[2]: https://github.com/duneag2/table2image "GitHub - duneag2/table2image · GitHub"

