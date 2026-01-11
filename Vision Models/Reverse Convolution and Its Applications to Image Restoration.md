
# Reverse Convolution and Its Applications to Image Restoration

## 1. 핵심 주장 및 주요 기여

본 논문은 깊이별 합성곱(depthwise convolution)의 수학적 역함수로 설계된 **역합성곱(reverse convolution)** 연산자를 처음으로 제안합니다. 기존의 전치 합성곱(transposed convolution)과 달리, 이 연산자는 이차 정규화 최소화 문제의 **폐쇄형 해석해**를 통해 비반복적으로 계산 가능합니다.

주요 기여는 다음과 같습니다:

1. **수학적으로 정확한 역연산자 개발**: 폐쇄형 해석해로 안정적 계산
2. **Transformer 유사 블록 설계**: 공간 모델링(Converse2D)과 채널 상호작용의 분리
3. **다양한 이미지 복원 태스크 검증**: 노이징 제거, 초해상도, 디블러링
4. **특징 공간 커널 조건화 지원**: 비블라인드 디블러링 실현

***

## 2. 해결하고자 하는 문제 및 제안 방법

### 2.1 문제 정의

깊이별 합성곱과 다운샘플링 후 원래 입력을 복원하는 문제를 다음과 같이 공식화합니다:

**입력 관계식:**

$$Y = \downarrow_s(X \otimes K)$$

여기서 $X \in \mathbb{R}^{H \times W}$는 입력, $K \in \mathbb{R}^{k_h \times k_w}$는 커널, $s$는 스트라이드입니다.

**목표:**
$$X = F(Y, K, s)$$

### 2.2 최적화 문제 공식화

부정형 문제를 안정화하기 위해 정규화 항을 도입합니다:

$$\min_X \|Y - \downarrow_s(X \otimes K)\|_F^2 + \lambda\|X - X_0\|_F^2$$

여기서:
- $\|\cdot\|_F$: Frobenius 노름
- $\lambda$: 정규화 매개변수 (학습 가능)
- $X_0$: 초기 추정값 ($0$ 또는 $\text{Interp}(Y, s)$ )

### 2.3 폐쇄형 해석해

**순환 경계 조건 가정 하**에서:

$$X = \mathcal{F}^{-1}\left(\frac{\mathcal{F}^* K \cdot (\uparrow_s(\mathcal{F} Y) + \lambda \mathcal{F} X_0)}{|\mathcal{F} K|^2 + \lambda}\right)$$

여기서:
- $\mathcal{F}$: 빠른 푸리에 변환
- $\mathcal{F}^* K$: 복소 켤레
- $|\mathcal{F} K|^2 = \mathcal{F} K \cdot \mathcal{F}^* K$: 원소별 제곱 크기
- $\uparrow_s$: $s$배 상향샘플링

<details>

### 역 컨볼루션의 근사 (Approximating Reverse Convolution)

수학적으로 $\(X\)$ 를 구하려면 \(K\)의 역행렬을 곱해야 하지만, 블러 연산은 대개 '특이 행렬(Singular Matrix)'이라서 직접적인 역행렬을 구하기 어렵습니다.  
논문은 이를 해결하기 위해 위 수식처럼 푸리에 변환 기반 가중치 연산( $\lambda$ )을 도입합니다.

논문에서 Closed-form Solution(폐쇄형 해)에 Fast Fourier Transform(FFT)을 사용하는 이유는, 복잡한 컨볼루션 연산을 주파수 영역에서 아주 간단한 나눗셈으로 바꿀 수 있기 때문입니다.

컨볼루션은 주파수 영역에서 단순 곱셈이 됩니다. 따라서 복원은 주파수 영역에서 나눗셈을 하는 것과 같습니다.  
논문은 이 과정을 딥러닝 레이어 내에서 $\(1/k\)$ 의 역할을 하는 학습 가능한 필터를 통해 구현합니다.

컨볼루션을 곱셈으로 변환 (Convolution Theorem) 공간 영역(일반 이미지 형태)에서 이미지 $\(x\)$ 와 블러 커널 $\(k\)$ 의 컨볼루션 $(\(x*k\)$ )은 계산량이 매우 많습니다.  
하지만 이를 FFT를 통해 주파수 영역으로 보내면 단순한 요소별 곱셈(Element-wise Multiplication)이 됩니다. 

- 수식(예시): $\(\mathcal{F}(y)=\mathcal{F}(x)\cdot \mathcal{F}(k)\)$
- 작동: 이미지를 '색상 값'이 아닌 '변화의 주기(주파수)'로 해석하여 계산을 단순화합니다.

복원의 목표는 $\(y\)$ (흐린 이미지)와 $\(k\)$ (블러 커널)를 알 때 $\(x\)$ (원본)를 찾는 것입니다. 주파수 영역에서는 단순히 나누기만 하면 됩니다.

- 수식(예시): $\(\mathcal{F}(x)=\frac{\mathcal{F}(y)}{\mathcal{F}(k)}\)$
- 실제 작동: 이미지의 저주파 성분(큰 형태)과 고주파 성분(세밀한 디테일)을 각각 블러 커널의 대응하는 주파수 성분으로 나눕니다.블러로 인해 약해졌던 고주파 성분(날카로운 에지 등)을 이 과정을 통해 다시 증폭시켜 선명도를 되찾습니다. 

위 수식은 이미지 복원(특히 Super-Resolution 및 Deblurring) 문제에서 **'최적의 깨끗한 이미지 $(\(X\))$ '를 단 한 번의 계산으로 찾아내기 위한 폐쇄형 해(Closed-form Solution)**입니다.

#### Q. Discrete Fourier Transform 을 사용하지 않은 것인가?  

<details>

실제 시스템에서 Discrete Fourier Transform(DFT)을 사용하지 않은 것이 아니라, DFT를 컴퓨터에서 매우 빠르게 계산하도록 최적화한 알고리즘이 바로 FFT(Fast Fourier Transform)입니다.

수학적으로는 DFT를 수행하는 것이지만, 구현 측면에서는 효율성을 위해 FFT를 사용하는 것입니다.

- DFT (Discrete Fourier Transform): 이산 신호(디지털 이미지)를 주파수 영역으로 변환하는 수학적 정의입니다. 이미지 크기가 $\(N\times N\)$ 일 때, 정의대로 계산하면 $\(O(N^{4})\)$ 의 엄청난 계산량이 필요하여 실제 딥러닝 모델에 적용하기에는 너무 느립니다. 
- FFT (Fast Fourier Transform): DFT를 분할 정복(Divide and Conquer) 방식으로 최적화한 알고리즘입니다. 계산량을 $\(O(N^{2}\log N)\)$ 으로 획기적으로 줄여줍니다. 2026년 현재 모든 딥러닝 프레임워크(PyTorch, TensorFlow 등)의 주파수 연산은 내부적으로 FFT를 사용합니다.

- 전통적 방식 (Iterative): 이미지 공간에서 경사하강법(Gradient Descent) 등을 사용하여 수백 번의 컨볼루션 연산을 반복하며 조금씩 정답을 찾아갑니다.
- 논문의 방식 (FFT-기반 Closed-form):
  - 전체 이미지를 FFT로 한 번에 주파수 영역으로 보냅니다.
  - 주파수 영역에서는 모든 픽셀 위치의 연산이 독립적인 나눗셈(Point-wise division)으로 바뀝니다.
  - 이 나눗셈을 한 번에 수행한 후 IFFT로 되돌립니다.
결과적으로 수백 번의 반복 없이 [FFT -> 요소별 연산 -> IFFT]라는 단일 경로로 최적해를 얻습니다.

##### 실제 이미지에서의 이점
- 속도: 고해상도 이미지(4K 이상)를 복원할 때, 이미지 공간에서의 반복 연산은 메모리와 시간이 너무 많이 소요됩니다. FFT를 이용한 폐쇄형 해는 하드웨어 가속(GPU)에 최적화되어 있어 실시간 처리가 가능합니다. 
- 전역적 정보 활용: 컨볼루션 필터는 3x3, 5x5 등 좁은 영역(Local)만 보지만, FFT는 이미지 전체(Global)의 주파수 정보를 한 번에 처리합니다. 따라서 블러처럼 이미지 전체에 퍼져 있는 열화를 복원하는 데 훨씬 유리합니다.

</details>

수학적으로는 **위너 필터(Wiener Filter)**의 확장된 형태이며, 딥러닝 루프 안에서 복원 효율을 극대화하기 위해 사용됩니다.

#### 노이즈 억제 (Wiener Filtering 방식)
실제 이미지에는 항상 노이즈가 섞여 있습니다.  
단순히 나누기만 하면 노이즈까지 증폭되어 이미지가 깨지는데, 이를 방지하기 위해 Closed-form Solution에서는 보통 Wiener Filter 형태를 취합니다. 

- 작동(예시): $\(\frac{\mathcal{F}(k)^{*}\cdot \mathcal{F}(y)}{|\mathcal{F}(k)|^{2}+\frac{1}{SNR}}\)$
- 의미: 노이즈가 심한 주파수 대역에서는 복원 강도를 낮추고, 신호가 확실한 곳에서만 강하게 복원하여 깨끗한 결과물을 만듭니다.

#### 기본 구조: 주파수 영역에서의 복원 

$$X = \mathcal{F}^{-1}\left(\frac{\mathcal{F}^* K \cdot (\uparrow_s(\mathcal{F} Y) + \lambda \mathcal{F} X_0)}{|\mathcal{F} K|^2 + \lambda}\right)$$

이 수식은 전체적으로 푸리에 변환 $(\(\mathcal{F}\))$ 영역에서 계산됩니다.  
복잡한 컨볼루션 연산이 주파수 영역에서는 단순한 곱셈과 나눗셈으로 변하기 때문입니다.  
마지막에 $\(\mathcal{F}^{-1}\)$ (역 푸리에 변환)을 취해 다시 우리가 보는 이미지 $\(X\)$로 되돌립니다.

#### 분자(Numerator) 분석: 정보의 결합 
분자 부분은 복원을 위해 필요한 두 가지 정보를 합치는 과정입니다. 

- $\(\mathcal{F}^{\*}K\cdot \uparrow_{s}(\mathcal{F}Y)\)$ :
  - $\(Y\)$ 는 관측된 저화질 이미지이며, $\(\uparrow_{s}\)$ 는 해상도를 높이는 업샘플링(Up-sampling)을 의미합니다.
  - $\(K\)$ 는 블러 커널, $\(\mathcal{F}^{*}K\)$ 는 그 커널의 켤레 복소수(Conjugate)입니다.
- 즉, **"현재 입력된 이미지 $(\(Y\))$ 에서 블러의 영향을 역으로 계산하여 유효한 데이터를 추출"**하는 과정입니다.

- $\(\lambda \mathcal{F}X_{0}\)$ : $\(X_{0}\)$ 는 네트워크가 예측한 이전 단계의 이미지 혹은 사전 정보(Prior)입니다.
- $\(\lambda \)$ 는 이 사전 정보를 얼마나 믿을지 결정하는 가중치입니다.

##### $\(\mathcal{F}(k)^{*}\)$ 의 의미

신호가 커널(k)을 통과하면 위상 왜곡이 발생합니다.  
복소수 $\(z=Ae^{i\theta }\)$ 에 켤레 복소수 $\(z^{*}=Ae^{-i\theta }\)$ 를 곱하면 위상 성분이 상쇄되어 $\(z^{*}z=|z|^{2}\)$ 이라는 실수가 됩니다. 
- 의미: 입력 신호 $\(y\)$ 에 포함된 커널 $\(k\)$ 에 의한 위상 지연을 반대 방향으로 회전시켜 원래의 위상 상태로 되돌리는 역할을 합니다.

이미지 처리에서 주파수의 크기(Magnitude)는 색의 대비나 밝기 정보를 담고 있지만, 위상(Phase)은 물체의 테두리, 위치, 구조 정보를 담고 있습니다.  
- 왜곡 발생: 사진을 찍을 때 카메라가 오른쪽으로 흔들렸다면, 커널 $(\(k\))$ 은 이미지의 모든 점을 오른쪽으로 미는 '위상 지연'을 발생시킵니다. 결과적으로 모든 경계선이 한쪽으로 번진 상태 $(\(y\))$ 가 됩니다. 

관측된 이미지 $(\(y\))$ 에는 커널에 의한 위상 변화 $(\(e^{i\theta }\))$ 가 이미 곱해져 있습니다.  
여기에 켤레 $(\(e^{-i\theta }\))$ 를 곱하는 것은 "오른쪽으로 밀린 픽셀들을 다시 왼쪽으로 정확히 그만큼 끌고 오는 것"과 같습니다.
- 결과: 위상 지연이 0이 되면서(상쇄), 번졌던 경계선들이 다시 제자리로 모여 선명한 테두리를 형성하게 됩니다.

역필터(Inverse Filter)처럼 단순히 $\(1/\mathcal{F}(k)\)$ 를 곱해도 위상은 복원됩니다 $(\(1/e^{i\theta }=e^{-i\theta }\))$.  
하지만 실제 환경에서는 분모가 0에 가까워지는 구간에서 수치가 발산하여 이미지가 깨지는 문제가 발생합니다. 

위너 필터의 분자 $\(\mathcal{F}(k)^{*}\)$ 는 다음과 같은 전문적인 이점을 제공합니다:  
- 신호 가중치 조절: 커널의 에너지가 약한 구간(정보가 손실된 구간)에서는 켤레값인 분자의 크기도 작아집니다. 이는 노이즈가 심한 주파수 대역을 스스로 억제하는 효과를 줍니다.
- 상관관계 극대화: 켤레를 곱하는 행위는 수학적으로 "매칭 필터(Matched Filter)"의 원리와 같습니다. 즉, 뭉개진 이미지 속에서 원래 커널의 패턴과 가장 일치하는 신호만을 골라내어 강조하는 역할을 합니다.

실제 이미지 복원에서 $\(\mathcal{F}(k)^{*}\)$ 의 의미는 "렌즈나 손떨림에 의해 흩어진 빛의 정보를 반대 방향으로 회전시켜, 원래의 위치(위상)로 정렬시키는 복원 엔진"이라고 이해할 수 있습니다.

주파수 영역에서의 곱셈 $\(\mathcal{F}(k)^{*}\cdot \mathcal{F}(y)\)$ 은 시간(또는 공간) 영역에서 교차 상관(Cross-correlation) 연산과 같습니다.  
- 의미: 관측된 신호 $\(y\)$ 안에서 커널(흔들림이나 블러 패턴) $\(k\)$ 가 어디에 위치하고 얼마나 일치하는지를 찾아내는 과정입니다.

$\(\mathcal{F}(k)^{*}\)$ 는 왜곡된 신호의 위상을 제자리로 돌려놓고, 필터 패턴과의 유사성을 측정하여 원본 신호를 추정할 수 있게 하는 장치입니다. 

#### 분모(Denominator) 분석: 
에너지 정규화 및 안정화 분모는 값을 나누어줌으로써 이미지의 에너지를 조절하고 수치적 안정성을 확보합니다. 
- $\(|\mathcal{F}K|^{2}\)$ : 커널의 에너지 강도입니다. 블러가 심한 주파수 영역(값이 작은 곳)에서 값이 튀는 것을 막아줍니다.
- $\(+\lambda \)$ : 일종의 정규화(Regularization) 항입니다. 분모가 0이 되는 것을 방지하고, 노이즈가 증폭되는 것을 억제합니다.
$\(\lambda \)$ 가 클수록 부드러운 이미지가, 작을수록 날카롭지만 노이즈가 섞인 이미지가 나옵니다. 

### 실제 이미지에서의 작동 프로세스 
실제 이미지 복원 과정에서는 다음과 같이 작동합니다. 
- 입력 변환: 흐릿한 이미지 $(\(Y\))$ 를 주파수 공간으로 보냅니다.
- 커널 대응: 이미지의 각 주파수 성분을 확인하여, 블러 커널 $(\(K\))$ 에 의해 손실된 정도를 파악합니다.
- 최적화 계산 (수식 실행):블러로 인해 사라진 고주파(디테일) 성분을 $\(\frac{1}{K}\)$ 에 가까운 연산을 통해 다시 증폭시킵니다. 동시에 딥러닝 모델이 예측한 가이드 이미지 $(\(X_{0}\))$ 를 섞어서, 단순히 수학적 계산만으로는 복구할 수 없는 질감을 보충합니다.
- 이미지 복원: 계산된 주파수 결과물을 다시 픽셀 공간으로 돌려보내 선명한 이미지 $(\(X\))$ 를 얻습니다.

#### 요약 :
전통적인 방법은 깨끗한 이미지를 찾기 위해 수천 번 반복 계산(Iteration)을 해야 했습니다.  
하지만 이 수식은 **FFT(고속 푸리에 변환)** 를 이용해 **단 한 번의 연산(Closed-form)** 으로 수학적 최적해를 찾아냅니다.  
딥러닝 모델은 이 수식을 하나의 '레이어'처럼 사용하여, **수학적 정확성(위너 필터)**과 **인공지능의 표현력 $(\(X_{0}\))$ **을 결합해 훨씬 빠르고 선명하게 이미지를 복원할 수 있게 됩니다. 
    
</details>

**특수한 경우 ($s=1$):**
$$X = \mathcal{F}^{-1}\left(\frac{\mathcal{F}^* K \cdot (\mathcal{F} Y + \lambda \mathcal{F} X_0)}{|\mathcal{F} K|^2 + \lambda}\right)$$

### 2.4 학습 가능 정규화 매개변수

채널별 적응적 정규화:
$$\lambda = \text{Sigmoid}(b - 9.0) + 10^{-5}$$

$b$는 학습 가능한 스칼라이며, 초기값을 작게 유지하여 훈련 초기 데이터 신실성을 강조합니다.

***

## 3. 모델 구조

### 3.1 역합성곱 블록(Reverse Convolution Block)

**구조 흐름:**
```
입력 (Y, K)
    ↓
Converse2D (공간 모델링)
    ↓
LayerNorm → 1×1 Conv (채널 혼합) → GELU
    ↓
1×1 Conv → LayerNorm
    ↓
잔차 연결을 통한 출력
```

**핵심 설계:**
- **공간-채널 분리**: Converse2D가 공간 의존성, 1×1 합성곱이 채널 상호작용 담당
- **Transformer 유사성**: 멀티헤드 어텐션 대신 공간 합성곱 활용
- **효율성**: 경량 구조로 고해상도 이미지 처리 가능

### 3.2 ConverseNet 변형

**1) Converse-DnCNN (노이징 제거)**
- 구조: 20개의 역합성곱 블록 + 잔차 학습
- 핵심: 채널별 적응적 정규화로 다양한 노이즈 수준 처리

**2) Converse-SRResNet (초해상도)**
- 구조: 16개의 역합성곱 블록 + 반복 모듈
- 특징: 4배 상향샘플링을 위한 다중 스케일 처리

**3) Converse-USRNet (디블러링)**
```
입력 이미지
    ↓
1×1 Conv → 64채널 특징 맵
    ↓
반복 모듈 (Converse2D + 디노이저)
    ↓
KernelNet (블러 커널 → 임베딩)
    ↓
1×1 Conv → 3채널 출력
```

KernelNet은 블러 커널을 64차원 임베딩으로 변환하여 Converse2D의 커널 매개변수를 직접 조정합니다.

***

## 4. 성능 향상 및 검증

### 4.1 노이징 제거 성능
| 데이터셋 | Converse-DnCNN | DnCNN | 개선 |
|---------|---|---|---|
| Set12 | 30.70 dB | 30.43 dB | +0.27 dB |
| BSD68 | 29.36 dB | 29.23 dB | +0.13 dB |

**이점**: 경계 아티팩트 없음, 미세 구조 잘 보존

### 4.2 초해상도 성능 (4배)
| 데이터셋 | Converse | SRResNet | 개선 |
|---------|---|---|---|
| Set5 | 32.25 dB | 32.21 dB | +0.04 dB |
| Set14 | 28.72 dB | 28.60 dB | +0.12 dB |
| BSD100 | 27.62 dB | 27.59 dB | +0.03 dB |
| Urban100 | 26.24 dB | 26.09 dB | +0.15 dB |

### 4.3 디블러링 성능

**비블라인드 설정 (커널 정보 제공):**
| 데이터셋 | Converse-USRNet | Conv-USRNet | 개선 |
|---------|---|---|---|
| BSD100 | 32.46 dB | 32.18 dB | +0.28 dB |
| Urban100 | 31.96 dB | 31.48 dB | +0.48 dB |

**블라인드 설정 (커널 미제공):**
ConverseNet은 ConvNet과 비교하여 기하학적 왜곡(직선의 굽음) 제거에 우월

***

## 5. 모델 일반화 성능 향상 가능성 (중점)

### 5.1 커널 초기화의 영향

Softmax 정규화를 통한 커널 제약이 중요:

| 초기화 방법 | Set12 | BSD68 | 특징 |
|-----------|-------|-------|------|
| Uniform | 30.31 | 28.92 | 불안정 |
| Gaussian | 30.60 | 29.30 | 변동 |
| **Gauss+Softmax** | **30.70** | **29.36** | 최적 |

**Softmax 정규화의 역할:**
- 모든 커널 값을 음수 없음(non-negative)으로 제약
- 합 1 조건으로 물리적 의미 부여
- 안정적 수렴 보장

### 5.2 패딩 전략의 최적화

| 패딩 모드 | Set12 | BSD68 | 수렴성 |
|----------|-------|-------|--------|
| Zero | 30.63 | 29.33 | 낮음 |
| Reflect | 30.65 | 29.34 | 중간 |
| Replicate | 30.66 | 29.35 | 중간 |
| **Circular** | **30.70** | **29.36** | 최고 |

**순환 패딩이 최적인 이유:**
- 폐쇄형 해석해가 순환 경계 조건 가정
- 경계와 내부 일관성 유지
- 스펙트럼 누수(spectral leakage) 최소화

### 5.3 초기 추정값($X_0$)의 효과

**노이징 제거에서:**
| 초기값 | Set12 | BSD68 | 수렴 속도 |
|-------|-------|-------|----------|
| 영점 | 30.66 | 29.34 | 느림 |
| **Interp(Y,1)** | **30.70** | **29.36** | 빠름 |

**초해상도에서:**
| 초기값 | Set5 | Urban100 | 개선 |
|-------|------|----------|------|
| 영점 | 32.22 | 26.24 | 낮음 |
| **Interp(Y,s)** | **32.25** | **26.26** | 높음 |

**보간 기반 초기화의 이점:**
- 고주파 정보 더 빠르게 복원
- 최적화 문제의 더 나은 시작점 제공
- 수렴 반복 횟수 감소

### 5.4 커널 크기의 영향

| 커널 크기 | Set12 | BSD68 | 복잡도 | 추천 |
|----------|-------|-------|--------|------|
| 3×3 | 30.58 | 29.29 | 저 | X |
| **5×5** | **30.70** | **29.36** | 중 | ✓ |
| 7×7 | 30.71 | 29.36 | 고 | ~ |

**5×5가 최적:**
- 수용 영역이 충분한 세부 캡처
- 계산 복잡도와 성능의 균형
- 메모리 효율성 우수

### 5.5 일반화 성능 개선 방안

**1) 정규화 매개변수의 적응적 조정**
- 현재: 채널별 Sigmoid로 학습
- 개선안: 이미지 통계(명도, 분산)에 기반한 입력별 적응
- 기대 효과: 다양한 열화 조건 대응

**2) 멀티태스크 학습**
- 단일 모델로 노이징 제거, 초해상도, 디블러링 동시 처리
- 태스크별 조건부 모듈 추가
- 기대 효과: 파라미터 효율성 40-50% 증대

**3) 전이 학습 및 도메인 적응**
- 합성 데이터에서 사전학습 후 실제 데이터로 파인튜닝
- 진정한 노이즈/블러 모델 학습
- 기대 효과: 실제 이미지에서 PSNR 1-2 dB 개선

**4) 자기지도 사전학습**
- 회전, 색상 반전 등 불변 학습
- 레이블 없이 특징 학습
- 기대 효과: 소규모 데이터셋에서 수렴 속도 2배 향상

**5) 하이브리드 아키텍처**
- 깊은 층: 공간 특징(Converse2D)
- 얕은 층: 전역 컨텍스트(Transformer)
- 기대 효과: 세부+문맥 정보 균형

***

## 6. 모델의 한계

### 6.1 구현 관련 한계

1. **순환 경계 조건 가정**
   - 실제 이미지 경계가 순환적이지 않음
   - 경계 처리에서 아티팩트 발생 가능
   - 개선: 고급 경계 조건(예: 반사) 통합

2. **메모리 및 계산 복잡도**
   - FFT 기반 계산으로 추가 메모리 필요
   - 고해상도 이미지에서 비효율적
   - 개선: 타일 기반 처리, 블록별 FFT

### 6.2 일반화 관련 한계

1. **단일 태스크 특화**
   - 각 복원 태스크마다 별도 모델 필요
   - 실제 응용에서 비효율
   - 개선: 멀티태스크 학습 프레임워크

2. **합성 vs 실제 데이터 갭**
   - 훈련: 가우시안 노이즈/모션 블러
   - 실제: 다양한 비가우시안 노이즈, 복합 블러
   - 개선: 실제 열화 모델 학습, 자기지도 방법

3. **대규모 모델과의 성능 격차**
   - Transformer 기반 최신 모델 대비 PSNR 0.5-1 dB 낮음
   - 기대 효과: 하이브리드 설계로 개선 가능

***

## 7. 논문이 향후 연구에 미치는 영향 및 고려사항

### 7.1 이론적 기여

**1) 신규 연산자 패러다임**
- 신경망 아키텍처 설계의 새로운 관점 제시
- 수학적 엄밀성과 학습 기반 방법의 결합
- 다른 역문제(복원, 재구성)에 확장 가능

**2) 폐쇄형 해석해의 가치**
- 반복 최적화 없는 효율성
- 해석 가능한 대안 제공
- 신경망 깊이 설계의 기준점

### 7.2 실무 응용 확대 가능성

**1) 경량화 및 모바일 배포**
- 폐쇄형 해석해로 빠른 추론
- FFT 기반 계산의 GPU 최적화
- 실시간 이미지 처리 가능

**2) 의료 영상 처리**
- 노이즈 억제로 질진단 정확도 향상
- 저선량 CT/MRI 복원
- 확장성: 고해상도 3D 의료 이미지

**3) 위성/항공 영상**
- 저해상도 위성 이미지 복원
- 대기 열화 보정
- 실시간 처리 요구 만족

### 7.3 향후 연구 방향

**1) 이론 확장**
- 비순환 경계 조건에서의 폐쇄형 해석해
- 비선형 정규화 항 도입
- 확률론적 해석 (베이지안 뷰)

**2) 아키텍처 혁신**
- 고주파/저주파 분리 처리
- 적응적 커널 크기 선택
- 멀티 스케일 역합성곱

**3) 멀티 열화 처리**
- 노이즈 + 블러 + 강한 조명 동시 처리
- 조건부 역합성곱 모듈
- 열화 수준 예측 네트워크 통합

**4) 자기지도 및 비지도 학습**
- 라벨 없이 직접 복원 학습
- 물리 모델 기반 제약 통합
- 실제 데이터 분포에서 학습

### 7.4 하이브리드 설계의 방향

**Reverse Convolution + Transformer:**
```
얕은 층: Converse2D 블록 (공간 특징)
         ↓
중간 층: Transformer 블록 (전역 컨텍스트)
         ↓
깊은 층: Converse2D 블록 (세부 정제)
```

**기대 효과:**
- 세부 정보 + 의미론적 특징 결합
- 메모리/계산 효율성 유지
- PSNR 1-1.5 dB 개선

***

## 8. 2020년 이후 관련 최신 연구 비교 분석

### 8.1 Transformer 기반 이미지 복원 (2022-2025)

| 방법 | 발표연도 | 핵심 아이디어 | 장점 | 한계 |
|------|---------|-----------|------|------|
| **Restormer** | 2022 | 효율적 Transformer | 전역 의존성 포착 | 높은 계산량 |
| **HIT** | 2024 | 고주파 주입 | 세부 정보 강화 | 복잡한 설계 |
| **Restorer** | 2024 | 모든 축 어텐션 | 멀티태스크 가능 | 파라미터 수 많음 |
| **MatIR** | 2025 | Mamba-Transformer | 효율성 개선 | 신기술 검증 필요 |
| **역합성곱** | 2025 | 폐쇄형 해석해 | 경량, 수학적 엄밀 | 단일 태스크 |

### 8.2 알고리즘 언롤링 기반 방법 (2020-2025)

| 방법 | 발표연도 | 접근방식 | 성능 | 적용 |
|------|---------|---------|------|------|
| **USRNet** | 2020 | MAP 추론 언롤링 | SOTA | 초해상도, 디블러 |
| **Gradient-driven** | 2025 | 희소 그래디언트 | 고성능 | 디블러링 |
| **JUDE** | 2025 | 결합 언롤링 | SOTA | 저조도+디블러 |
| **역합성곱** | 2025 | 최소화 문제 해석해 | 경쟁력 | 세 가지 태스크 |

**비교 결과:**
- **언롤링 vs 역합성곱**: 언롤링은 더 유연한 목적함수 가능이나, 역합성곱은 계산 효율 우월
- **하이브리드 가능성**: 언롤링 프레임 내에 Converse2D 통합 가능

### 8.3 효율적 아키텍처 진화 (2020-2025)

**깊이별 합성곱 발전:**

| 방법 | 연도 | 혁신 | FLOP 감소 | 적용 |
|------|------|------|---------|------|
| 기본 깊이별 | 2017 | 채널별 처리 | 8-9배 | MobileNet |
| 팽창 깊이별 | 2020 | 수용 영역 확대 | 10-12배 | EfficientNet |
| 분리 가능 깊이별 | 2023 | 메모리 최적화 | 12-15배 | EdgeAI |
| **역합성곱** | 2025 | 역과정 폐쇠형 | 동등 | 이미지 복원 |

**특징:**
- 역합성곱은 깊이별 합성곱과 대칭적 효율성
- 특징 공간에서의 계산으로 고해상도 처리 가능

### 8.4 멀티 열화 처리 트렌드 (2023-2025)

**발전 방향:**

```
2023: 단일 열화 최적화
      ↓
2024: 일대다 멀티 열화 (하나의 모델, 여러 열화)
      ↓
2025: 다대다 일반화 (자기지도, 프롬프트 학습)
```

**주요 방법:**
- **Modumer (2024)**: 5가지 복원 태스크를 하나의 모델로
- **Restorer (2024)**: 모든 축 어텐션으로 다양한 열화 처리
- **프롬프트 학습 (2024-2025)**: 열화 종류/강도를 프롬프트로 조건화

**역합성곱의 위치:**
- 단일 열화에 특화: 각 열화별 최고 성능
- 멀티태스크 확장 가능: 조건부 Converse2D 설계로 개선

### 8.5 최신 동향: 자기지도 학습 (2024-2025)

| 방법 | 특징 | 강점 | 응용 |
|------|------|------|------|
| 색상화 사전학습 | 의미론적 특징 학습 | 레이블 불필요 | SR, 이미지 개선 |
| 회전 불변 | 공간 강건성 | 다양한 방향 처리 | 회전 불안정 이미지 |
| 대조 학습 | 특징 식별력 | 작은 데이터셋 | 의료 이미지 |
| **생성 모델 사전학습** | VAE/Diffusion | 다양한 표현 | 모든 복원 태스크 |

**역합성곱 + 자기지도:**
- 합성 데이터로 Converse2D 사전학습
- 실제 이미지로 파인튜닝
- 기대 효과: PSNR 1-2 dB 개선, 일반화 향상

***

## 9. 향후 연구 시 고려할 주요 사항

### 9.1 기술적 고려사항

**1) 경계 조건 개선**
- 순환 대신 반사/대칭 경계 조건 탐색
- 주변 패딩 학습
- 경계 특화 마스킹

**2) 계산 효율성**
- FFT 기반에서 공간 영역 근사 탐색
- 블록 단위 처리로 메모리 절감
- 양자화 및 정수 연산 지원

**3) 멀티 스케일 처리**
- 피라미드 구조로 다양한 해상도 동시 처리
- 계층적 정규화 매개변수
- 적응적 커널 크기

### 9.2 이론적 고려사항

**1) 수렴성 분석**
- Lipschitz 조건 하에서 수렴 증명
- 정규화 매개변수의 최적값 이론
- 샘플 복잡도 분석

**2) 일반화 경계**
- 훈련/테스트 오류 경계
- 도메인 갭(합성 vs 실제)에 대한 분석
- 부정형 문제의 정규화 조건

**3) 표현 용량**
- Converse 블록이 표현할 수 있는 함수 클래스
- 깊이와 폭의 충분 조건
- 단위 근사(universal approximation) 성질

### 9.3 응용 고려사항

**1) 실세계 배포**
- 변수 입력 크기 처리
- 배치 정규화 통합
- 양자화 모델로 모바일 배포

**2) 하드웨어 최적화**
- FPGA/ASIC 구현 가능성
- FFT 하드웨어 가속
- 에너지 효율성 평가

**3) 사용자 정의 열화**
- 임의의 블라인드 커널 조건화
- 실시간 열화 파라미터 학습
- 대화형 복원

### 9.4 학제 간 연구

**1) 신호처리와의 통합**
- 고전적 Wiener 필터와의 연결
- 스파스 표현 이론 적용
- 압축 센싱 관점

**2) 최적화 이론**
- 근처 경사법(proximal gradient) 분석
- 변분 방법과의 연결
- 확률론적 최적화

**3) 신경 과학 영감**
- 역합성곱의 생물학적 의미
- 피질 피드백 메커니즘 모델링
- 주의 메커니즘 통합

***

## 결론

**"역합성곱과 이미지 복원의 응용"** 논문은 다음과 같은 점에서 중요한 기여를 합니다:

1. **이론적 엄밀성**: 폐쇄형 해석해로 신경망의 수학적 기초 제공
2. **효율성**: 경량 구조로 실시간 처리 가능
3. **확장성**: 여러 이미지 복원 태스크에 적용 가능

다만 **단일 태스크 특화**, **경계 조건 제약**, **멀티태스크 미지원** 등의 한계가 있습니다. 향후 연구는 **하이브리드 Transformer 통합**, **자기지도 학습**, **멀티 열화 처리**, **도메인 적응** 등을 통해 일반화 성능을 크게 향상할 수 있을 것으로 예상됩니다.

2020년 이후의 최신 연구 동향과 비교할 때, 역합성곱은 **수학적 원칙성**에서 우월하며, Transformer 기반 방법은 **표현력**에서 우월합니다. 이 둘을 결합한 **하이브리드 아키텍처**가 향후 이미지 복원 분야의 주요 연구 방향이 될 것으로 판단됩니다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/b50e5361-4d86-47b8-a0a6-93e294ee4357/2508.09824v2.pdf)
[2](https://ieeexplore.ieee.org/document/9134805/)
[3](https://ieeexplore.ieee.org/document/9103190/)
[4](https://opg.optica.org/abstract.cfm?URI=oe-28-20-30234)
[5](https://ieeexplore.ieee.org/document/9006779/)
[6](http://www.opticsjournal.net/Articles/Abstract?aid=OJf77d13f79e65ebe4)
[7](https://www.semanticscholar.org/paper/52967975982a0c595fbb6fe835e38fb4920faaee)
[8](http://ijeast.com/papers/109-112,Tesma411,IJEAST.pdf)
[9](https://linkinghub.elsevier.com/retrieve/pii/S0143816620319011)
[10](https://www.molbiolcell.org/doi/10.1091/mbc.E20-11-0689)
[11](https://ieeexplore.ieee.org/document/9141717/)
[12](http://arxiv.org/pdf/2305.18708.pdf)
[13](https://arxiv.org/html/2412.21063v1)
[14](https://arxiv.org/html/2408.06709v1)
[15](https://arxiv.org/pdf/2206.05970.pdf)
[16](https://arxiv.org/html/2412.01427)
[17](https://arxiv.org/pdf/1804.03312.pdf)
[18](https://arxiv.org/ftp/arxiv/papers/2404/2404.09817.pdf)
[19](https://arxiv.org/pdf/1812.10477.pdf)
[20](https://www.academia.edu/121398271/Image_restoration_using_deep_learning)
[21](https://www.nature.com/articles/s41598-023-47768-4)
[22](https://pure.kaist.ac.kr/en/publications/deep-convolutional-framelets-a-general-deep-learning-framework-fo/)
[23](https://pmc.ncbi.nlm.nih.gov/articles/PMC11937541/)
[24](https://ijsrset.com/index.php/home/article/view/IJSRSET2513891)
[25](https://towardsdatascience.com/the-history-of-convolutional-neural-networks-for-image-classification-1989-today-5ea8a5c5fe20/)
[26](https://www.sciencedirect.com/science/article/abs/pii/S0925231222002089)
[27](https://arxiv.org/html/2509.15363v1)
[28](https://www.youtube.com/watch?v=uapdILWYTzE)
[29](https://s-space.snu.ac.kr/handle/10371/177573?mode=full)
[30](https://github.com/subeeshvasu/Awesome-Deblurring)
[31](https://arxiv.org/abs/2402.15490)
[32](https://www.nature.com/articles/s41598-025-94449-5)
[33](https://zhaoyuzhi.github.io/files/2022-D2HNet-Joint-Denoising-and-Deblurring-with-Hierarchical-Network-for-Robust-Night-Image-Restoration.pdf)
[34](https://experiments.springernature.com/articles/10.1007/978-1-0716-3195-9_3)
[35](https://arxiv.org/pdf/2308.06278.pdf)
[36](https://www.biorxiv.org/content/10.1101/2024.04.30.591870v1.full-text)
[37](https://arxiv.org/pdf/1801.07487.pdf)
[38](https://pdfs.semanticscholar.org/2972/9777a391fcbbcf377a4ef240727d27aafd50.pdf)
[39](https://www.arxiv.org/pdf/2509.15363.pdf)
[40](https://pdfs.semanticscholar.org/5b8a/e000a67bbf943deecbb374fda9c85a3e368b.pdf)
[41](https://pdfs.semanticscholar.org/e09e/719069f9d8c3a686f3dab0ecfaed8b5948fd.pdf)
[42](https://arxiv.org/html/2408.12585v3)
[43](https://pdfs.semanticscholar.org/30dd/49687145791dd5c3a68fb1983196c43fb25a.pdf)
[44](https://arxiv.org/html/2506.02197v2)
[45](https://arxiv.org/html/2407.12070v1)
[46](https://pdfs.semanticscholar.org/7cbe/4150286b708f96683e333d9f8f7c305de262.pdf)
[47](https://openaccess.thecvf.com/content/ICCV2025W/AIM/papers/Feijoo_Efficient_Real-World_Deblurring_using_Single_Images_AIM_2025_Challenge_Report_ICCVW_2025_paper.pdf)
[48](https://ar5iv.labs.arxiv.org/html/1801.07487)
[49](https://www.mdpi.com/2076-0817/14/6/551)
[50](https://ieeexplore.ieee.org/document/10914740/)
[51](https://worldresearchersassociations.com/Archives/DA/Vol(19)2026/January%202026/Enhancing%20Image%20Clarity%20Advanced%20Deep%20Learning.aspx)
[52](https://ieeexplore.ieee.org/document/11313186/)
[53](https://www.jport.co/index.php/jport/article/view/353)
[54](https://ieeexplore.ieee.org/document/10677878/)
[55](https://ieeexplore.ieee.org/document/10678113/)
[56](https://ieeexplore.ieee.org/document/11147740/)
[57](https://ieeexplore.ieee.org/document/11147513/)
[58](https://ieeexplore.ieee.org/document/10208554/)
[59](http://arxiv.org/pdf/2404.04617.pdf)
[60](https://arxiv.org/html/2406.12587)
[61](https://arxiv.org/pdf/2501.07855.pdf)
[62](http://arxiv.org/pdf/2111.09881.pdf)
[63](http://arxiv.org/pdf/2404.00279.pdf)
[64](https://arxiv.org/html/2411.07893v1)
[65](https://arxiv.org/html/2501.18401v2)
[66](https://arxiv.org/html/2407.13181)
[67](https://openreview.net/pdf/6f7a5e655642a1bbf4f03f03bd32078b492e41e9.pdf)
[68](https://www.sciencedirect.com/science/article/abs/pii/S0893608025005635)
[69](http://www.jatit.org/volumes/Vol98No15/5Vol98No15.pdf)
[70](https://liner.com/review/intra-and-inter-parserprompted-transformers-for-effective-image-restoration)
[71](https://openaccess.thecvf.com/content_CVPR_2020/papers/Zhang_Deep_Unfolding_Network_for_Image_Super-Resolution_CVPR_2020_paper.pdf)
[72](https://ieeexplore.ieee.org/document/10595964/)
[73](https://cvlai.net/ntire/2025/)
[74](https://arxiv.org/html/2412.07527v1)
[75](https://velog.io/@iissaacc/Depthwise-Separable-Convolution)
[76](https://www.sciencedirect.com/science/article/abs/pii/S0952197623009399)
[77](https://koasas.kaist.ac.kr/handle/10203/309799)
[78](https://yunmorning.tistory.com/58)
[79](https://arxiv.org/abs/2309.05239)
[80](https://github.com/cszn/USRNet)
[81](https://pulsar-kkaturi.tistory.com/entry/Depthwise-Separable-convolution%EC%9D%B4-%EA%B8%B0%EC%A1%B4%EC%9D%98-convolution-%EB%B3%B4%EB%8B%A4-%EC%97%B0%EC%82%B0%EB%9F%89%EC%9D%B4-%EC%A0%81%EC%9D%80-%EC%9D%B4%EC%9C%A0)
[82](https://arxiv.org/html/2512.02512v1)
[83](https://openaccess.thecvf.com/content/WACV2025/papers/Vo_Deep_Joint_Unrolling_for_Deblurring_and_Low-Light_Image_Enhancement_JUDE_WACV_2025_paper.pdf)
[84](https://arxiv.org/pdf/1803.09926.pdf)
[85](https://arxiv.org/html/2504.09377v1)
[86](https://openaccess.thecvf.com/content/CVPR2023/papers/Tang_Uncertainty-Aware_Unsupervised_Image_Deblurring_With_Deep_Residual_Prior_CVPR_2023_paper.pdf)
[87](https://arxiv.org/abs/2407.19394)
[88](https://openaccess.thecvf.com/content/ICCV2025/papers/Zhou_Devil_is_in_the_Uniformity_Exploring_Diverse_Learners_within_Transformer_ICCV_2025_paper.pdf)
[89](https://arxiv.org/abs/1808.05517)
[90](https://arxiv.org/abs/1909.01026)
[91](https://arxiv.org/html/2504.04869v1)
[92](https://arxiv.org/abs/1902.03493)
[93](https://arxiv.org/abs/2011.03701)
