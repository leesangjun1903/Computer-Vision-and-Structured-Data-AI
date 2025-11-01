# CatBoost: Unbiased Boosting with Categorical Features

**핵심 주장 및 주요 기여**  
CatBoost는 기존 그래디언트 부스팅 방식이 가진 **타깃 누수(target leakage)** 문제를 해결하기 위해 두 가지 핵심 기법을 제안한다. 첫째, **Ordered Boosting**을 도입하여 부스팅 단계에서 발생하는 예측 분포 편향(prediction shift)을 제거하고, 둘째, **Ordered Target Statistics**를 이용해 범주형 변수 처리 시 생기는 타깃 누수를 방지한다. 이 두 기법의 결합으로 CatBoost는 XGBoost, LightGBM 대비 다양한 데이터셋에서 일관된 성능 향상을 보인다.[1]

## 1. 해결하고자 하는 문제  
기존 그래디언트 부스팅은 학습 과정에서  
- 모델 업데이트 과정에서 현재 모델이 사용한 훈련 타깃을 다시 참조함으로써 발생하는 **예측 분포 편향(prediction shift)**  
- 범주별 통계(target statistics)를 산출할 때 동일 레코드의 타깃을 포함하여 계산하는 **타깃 누수**  
를 겪는다. 이로 인해 훈련 시와 테스트 시 모델이 보는 분포가 달라져 일반화 성능이 저하된다.[1]

## 2. 제안 방법

### 2.1 Ordered Target Statistics  
각 훈련 샘플 $$(x_k, y_k) $$에 대해 무작위 순열 $$\sigma$$를 정의하고, 해당 샘플의 범주형 값에 대한 타깃 통계를  

$$
\(TS_{k,i}=\frac{\sum _{j=1}^{k-1}[x_{\sigma (j),i}=x_{\sigma (k),i}]\cdot y_{\sigma (j)}+\alpha \cdot \text{prior}}{\sum _{j=1}^{k-1}[x_{\sigma (j),i}=x_{\sigma (k),i}]+\alpha }\)
$$

로 계산한다. 여기서 

$\(TS_{k,i}\)$: 순열 상 \(k\)번째 샘플의 \(i\)번째 범주형 특징에 대한 인코딩된 값 (타겟 통계량).

$\(\sigma \)$: 학습 데이터셋의 임의의 순열.

$\(k\)$: 현재 처리 중인 샘플의 순서 인덱스.

$\(j\)$: 현재 샘플 \(k\) 이전에 있는 샘플들의 인덱스 $(\(j<k\))$.

$\([x_{\sigma (j),i}=x_{\sigma (k),i}]\)$: 지시 함수(indicator function)이다.  

순열 상 $\(j\)$ 번째 샘플의 $\(i\)$ 번째 특징 범주가 $\(k\)$ 번째 샘플의 $\(i\)$ 번째 특징 범주와 같으면 1, 아니면 0의 값을 가진다. 

$\(y_{\sigma (j)}\)$: 순열 상 $\(j\)$ 번째 샘플의 실제 타겟 값.

$$prior$$는 사전에 정의된 상수값 (일반적으로 전체 데이터셋의 타겟 평균 또는 특정 상수), $$\alpha$$는 스무딩 파라미터로, 희귀한 범주(rare categories)의 분산을 줄이는 데 도움을 준다.  
이 방식은 훈련 샘플의 자기 자신을 제외하고 이전 순서의 샘플만 사용하므로 타깃 누수를 제거한다.[1]

##
CatBoost의 **Ordered Target Statistics (Ordered TS)**는 타겟 유출(target leakage) 및 예측 편향(prediction shift)을 방지하기 위해 고안된 범주형 특징(Categorical Feature) 인코딩 방법입니다.  
데이터에 인위적인 순서(permutation)를 부여하고, 각 샘플의 TS 계산 시 해당 샘플보다 순서상 앞에 있는 데이터만 활용합니다.

핵심 아이디어는 각 데이터 포인트의 목표 값을 사용 하지 않고 목표 통계를 계산하여 직접적인 누출을 방지하는 것입니다. 이는 데이터에 특정 순서를 부여함으로써 가능합니다.

#### 원리
기존의 타겟 인코딩 방식은 전체 데이터셋의 타겟 평균을 사용하여 과적합(overfitting)을 유발할 수 있습니다.  
Ordered TS는 온라인 학습(online learning) 개념에서 영감을 받아, 각 데이터 포인트의 타겟 통계량을 계산할 때 해당 데이터 포인트 이전에 등장한 데이터만 사용합니다.

#### 과정
- 학습 데이터셋에 대한 임의의 순열(random permutation, $\(\sigma$ \))을 생성합니다.

<details>
<summary>임의의 순열(random permutation, $(\sigma$ ))</summary>

CatBoost의 Ordered Target Statistics 수식에서 $\(\sigma \)$ 는 학습 데이터셋에 대한 **임의의 순열(random permutation)**을 의미하며, 이는 데이터 샘플의 순서를 무작위로 재배열하는 함수를 수학적으로 표현한 것입니다.

학습 데이터셋의 샘플 개수를 $\(N\)$ 이라고 할 때, 이 샘플들은 인덱스 집합 $\(I=\{1,2,\dots ,N\}\)$ 으로 표현할 수 있습니다.  
$\(\sigma \)$ 는 인덱스 집합 $\(I\)$ 에서 $\(I\)$ 로의 일대일 대응 함수입니다.  

$\(\sigma :I\rightarrow I\)$

이는 원래 데이터셋의 인덱스를 새로운 순서의 인덱스로 매핑하는 것을 의미합니다.  

예를 들어, $\(N=4\)$인 데이터셋의 인덱스 집합은 $\(\{1,2,3,4\}\)$  이고, 이 집합의 가능한 순열 중 하나인 $\(\sigma =\{3,1,4,2\}\)$ 는 다음과 같이 해석할 수 있습니다.  

원래 4번째 샘플이 순열된 데이터에서는 3번째 순서에 위치합니다 $(\(\sigma (3)=4\))$.  
원래 2번째 샘플이 순열된 데이터에서는 4번째 순서에 위치합니다 $(\(\sigma (4)=2\))$.  
원래 3번째 샘플이 순열된 데이터에서는 1번째 순서에 위치합니다 $(\(\sigma (1)=3\))$.  
원래 1번째 샘플이 순열된 데이터에서는 2번째 순서에 위치합니다 $(\(\sigma (2)=1\))$. 

따라서 CatBoost는 이 임의의 순열 $\(\sigma \)$를 사용하여 데이터에 가상적인 순서를 부여하고, 이 순서대로 데이터 샘플을 처리하여 타겟 통계량을 계산합니다.  
수식에서 $\(x_{\sigma (j)}\)$는 순열 $\(\sigma \)$에 의해 재배열된 데이터셋의 $\(j\)$번째 샘플을 의미합니다. 

</details>

- 순열된 순서대로 각 샘플 \(k\)에 접근합니다.
- 샘플 \(k\)의 특정 범주형 특징 \(i\)에 대한 타겟 통계량 $\(TS_{k,i}\)$ 를 계산할 때, 순열 상에서 그 이전에 위치한 샘플들(\(j<k\))만을 사용합니다. 

<img width="639" height="517" alt="스크린샷 2025-11-01 오전 11 33 16" src="https://github.com/user-attachments/assets/60a61f49-be16-4721-a454-2099f56a2074" />

#### 주요 특징
- 타겟 유출 방지: 현재 샘플의 타겟 값을 포함하여 미래의 어떤 정보도 현재 샘플의 인코딩에 사용하지 않으므로 타겟 유출이 근본적으로 방지됩니다.
- 과적합 감소: 통계적으로 편향되지 않은(unbiased) 방식으로 통계량을 계산하여 모델의 과적합을 줄이고 일반화 성능을 향상시킵니다.
- 여러 순열 사용: 단일 순열로 인한 편향을 줄이기 위해, CatBoost는 학습 과정에서 여러 개의 무작위 순열을 생성하고 이들을 사용하여 타겟 통계량을 계산하거나 트리를 빌드합니다.

### 2.2 Ordered Boosting  
부스팅 매 스텝에서 잔차(residual)를 계산할 때, 각 샘플을 제외하고 학습된 **supporting model**을 이용한다. 무작위 순열 $$\sigma$$에 따라  
- 샘플 $$i$$의 잔차 계산에는 순서상 이전 $$i-1$$개의 샘플로 학습된 모델 $$M_{i-1}$$ 사용  
- 전체 훈련 샘플에 대해 $$n$$개의 supporting model을 유지  
를 기본 개념으로 한다. 이론적으로는 $$n$$개의 모델을 학습해야 하지만, 실용화를 위해 로그 단위로 축소하여 유지하며 계산 복잡도를 표준 GBDT와 동일한 $$O(n)$$ 수준으로 유지한다.[1]

<img width="1024" height="513" alt="image" src="https://github.com/user-attachments/assets/52f17e0b-bcec-467f-a4b5-31b59b7942cb" />

> 이 시각화는 CatBoost가 잔차를 계산하고 모델을 업데이트하는 방식을 보여줍니다. 샘플 xᵢ에 대해 모델은 이전 데이터 포인트만 사용하여 예측합니다.

##
CatBoost의 **순서형 부스팅(Ordered Boosting)**은 기존 그레이디언트 부스팅 알고리즘에서 발생하는 **예측 편향(prediction shift)**과 타깃 정보 누수(target leakage) 문제를 해결하여 과적합을 방지하고 모델의 일반화 성능을 향상시키는 핵심 기술입니다. 

기존 부스팅 방식에서는 현재 모델이 학습하는 잔차(residual, 오차) 계산 시, 모델이 학습하려는 데이터 포인트 자체의 타겟 값 정보가 암묵적으로 포함될 수 있습니다.  
이는 마치 시험 문제를 풀 때 정답을 미리 살짝 훔쳐보는 것과 같아서, 모델이 훈련 데이터에 과적합되고 실제 새로운 데이터에 대한 예측 성능이 떨어지는 '예측 편향(prediction shift)'을 유발합니다.

순서형 부스팅은 이 문제를 해결하기 위해 다음과 같은 독창적인 접근 방식을 사용합니다.

- 데이터 순서 섞기 (Permutation): 학습 데이터를 무작위로 여러 번 섞습니다 (순열 생성).
- 순차적 학습: 각 데이터 순열 내에서, 모델은 데이터를 순차적으로 처리합니다. 특정 데이터 포인트의 잔차를 계산하거나 새로운 트리를 학습할 때, 해당 데이터 포인트보다 순열 상 앞에 위치한 데이터 포인트들의 정보만 사용합니다.

표준 Gradient Boosting Decision Tree 모델에서 \(m\)번째 트리를 학습할 때, $\(i\)$ 번째 데이터 포인트 $\(x_{i}\)$ 에 대한 잔차 $\(r_{m-1}(x_{i})\)$ 는 이전 모델 $\(F_{m-1}\)$을 사용하여 다음과 같이 계산됩니다: 

```math
r_{m-1}(x_{i})=-\frac{\partial L(y_{i},F_{m-1}(x_{i}))}{\partial F_{m-1}(x_{i})}
```

여기서 $\(L\)$ 은 손실 함수(loss function), $\(y_{i}\)$ 는 실제 타겟 값입니다. 문제는 $\(F_{m-1}(x_{i})\)$가 $\(y_{i}\)$의 정보를 이미 포함할 수 있다는 점입니다.

CatBoost의 Ordered Boosting에서는 이 문제를 해결하기 위해, 데이터의 특정 순열 $\(\sigma \)$ 내에서 $\(k\)$번째 데이터 포인트 $\(x_{\sigma (k)}\)$에 대한 잔차를 계산할 때, 이전에 나타난 데이터 $\(x_{\sigma (1)},\dots ,x_{\sigma (k-1)}\)$만을 사용하여 학습된 모델 $\(F_{m-1}^{\sigma (k-1)}\)$을 사용합니다: 

```math
r_{m-1}(x_{\sigma (k)})=-\frac{\partial L(y_{\sigma (k)},F_{m-1}^{\sigma (k-1)}(x_{\sigma (k)}))}{\partial F_{m-1}^{\sigma (k-1)}(x_{\sigma (k)})}
```

여기서 $\(F_{m-1}^{\sigma (k-1)}\)$은 순열 $\(\sigma \)$에서 $\(k\)$번째 데이터 포인트 이전에 있는 모든 데이터를 사용해 학습된 모델입니다.  
이 모델은 $\(y_{\sigma (k)}\)$에 대한 어떠한 정보도 포함하지 않으므로, 잔차 추정치가 비편향적(unbiased)이 되어 과적합을 줄이고 모델의 일반화 성능을 향상시킵니다. 

- 독립성 유지: 이렇게 하면 현재 학습 중인 데이터 포인트의 타겟 값이 모델 학습에 직접적인 영향을 주지 않으므로, 타겟 유출이 방지되고 그레이디언트 추정치가 편향되지 않습니다.
- 앙상블: 여러 순열을 통해 학습된 모델들의 결과를 결합하여 최종 예측 모델을 완성합니다. 

쉽게 비유하자면, 모델에게 데이터를 시간 순서대로 보여주면서, 미래의 정보를 보지 못하게 하여 공정한 학습을 유도하는 방식입니다.


결론적으로, Ordered Target Statistics는 범주형 변수를 효율적으로 처리하여 모델에 입력할 수 있도록 돕는 기술이고, Ordered Boosting은 그렇게 처리된 데이터를 포함하여 모델 전체를 학습하는 과정에서 발생하는 편향을 근본적으로 제거하는 기술입니다.

CatBoost는 이 두 가지 '순서형' 기법을 모두 사용하여, 일반적인 그레이디언트 부스팅 알고리즘에 비해 과적합이 적고 안정적인 성능을 제공합니다.

## 3. 모델 구조  
- **Base Learner**: 균형 잡힌 *Oblivious Decision Tree*(모든 레벨에서 동일 분할 기준)  
- **Boosting Modes**:  
  - *Plain*: 기존 GBDT에 Ordered TS만 도입  
  - *Ordered*: Ordered Boosting + Ordered TS  
- **범주형 조합(feature combinations)**: Greedy 방식으로 두 개 이상의 범주형 피처 조합하여 높은 차수 의존성 포착.

## 4. 성능 향상 및 한계  

### 4.1 성능 향상  
CatBoost는 XGBoost, LightGBM 대비  
- 성능 지표(logloss, zero-one loss)에서 모든 벤치마크 데이터셋에서 우수한 성능 달성  
- 특히 소규모 데이터셋에서 Ordered 모드가 Plain 모드 대비 더 큰 이득을 보이며 일반화 성능이 개선됨  
- 범주형 처리를 위한 Ordered TS가 Holdout, Leave-one-out 방식 대비 가장 높은 성능 제공

### 4.2 한계 및 고려 사항  
- **계산 복잡도**: Ordered Boosting 지원 모델 수 증가로 Plain 모드 대비 약 1.7배 느린 학습 속도 발생  
- **분산 증가**: 초기 순열 단계에서 샘플이 적은 경우 TS와 예측 분산이 커질 수 있어 여러 순열 사용 필요  
- **메모리 요구**: supporting model 유지로 메모리 소비 증가  

## 5. 일반화 성능 향상 가능성  
Ordered Boosting과 Ordered TS는 **훈련-테스트 분포 차이를 최소화**하여 과적합 위험을 줄인다. 특히 소규모 데이터셋과 고카디널리티 범주형 피처 환경에서 일반화 성능이 크게 개선될 수 있다.

## 6. 향후 연구 영향 및 고려할 점  
- **다른 모델과의 결합**: Transformer, 딥러닝 모델 등과 부스팅 기법 하이브리드 연구 가능  
- **분산 학습 최적화**: supporting model 유지 비용 절감을 위한 효율적 분산/병렬 학습 알고리즘 개발  
- **자동 하이퍼파라미터 최적화**: 순열 수(s), 조합 차수(cmax) 등 민감도 분석 및 자동화  
- **타깃 누수 제거 일반화**: Ordered 원칙을 다른 학습 알고리즘의 타깃 통계, 교차 검증 단계에 적용  

CatBoost의 Ordered Boosting과 Ordered TS는 부스팅 계열 모델의 **타깃 누수 문제**를 근본적으로 해결하며, 범주형 피처 처리 및 일반화 성능 개선에 있어 중요한 연구 방향으로 자리잡았다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/aea9e55a-97d7-4122-8325-dc3e0930fdd9/1706.09516v5.pdf)

Ordered Target Statistics (Ordered TS) : 
https://apxml.com/courses/mastering-gradient-boosting-algorithms/chapter-6-catboost-gradient-boosting/catboost-ordered-ts

Why CatBoost Works So Well: The Engineering Behind the Magic :
https://towardsdatascience.com/catboost-inner-workings-and-optimizations/
