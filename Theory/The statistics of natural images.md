# The statistics of natural images

### 핵심 개요

Daniel L. Ruderman의 "The Statistics of Natural Images"(1994)는 영상 통계학 분야의 기초 논문으로, 자연 이미지가 무작위 이미지와 근본적으로 다르게 구조화되어 있음을 수학적으로 입증했습니다. 이 논문은 이미지 압축, 생물시각 처리, 신경망 설계의 이론적 토대를 제공하며, 특히 스케일 불변성(scale invariance)과 계층적 분산 정규화(hierarchical variance normalization)라는 두 가지 핵심 대칭성을 발견했습니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/8b3ca3ea-1fb3-4b43-9d55-6fe6fdbbd0ae/Ruderman-statistics.pdf)

***

## 1. 해결하고자 하는 문제와 동기

Ruderman이 직면한 근본적 질문은 "자연 이미지의 구조를 어떻게 정량화할 것인가?"였습니다. 당시 이미지 처리 연구는 두 가지 중요한 간극에 직면해 있었습니다:

**첫째, 이미지 압축의 한계**: 기존 알고리즘(예측 코딩, JPEG)은 경험적으로 8×8 픽셀 블록당 약 1/8 압축률을 달성했으나, 이것이 이론적 최대치인지 알 수 없었습니다. 이는 자연 이미지의 진정한 엔트로피(정보 이론적 하한)를 모르기 때문이었습니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/8b3ca3ea-1fb3-4b43-9d55-6fe6fdbbd0ae/Ruderman-statistics.pdf)

**둘째, 생물시각 시스템의 설계 원리**: 포유류 망막과 대뇌피질이 왜 특정 구조를 가지는지 설명할 이론적 틀이 부재했습니다. Atick과 Redlich(1992), Laughlin(1981) 등의 선행 연구는 효율적인 신경 부호화가 자연 이미지 통계에 최적화되어 있다고 제안했으나, 그 통계가 무엇인지 체계적으로 규명되지 않았습니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/8b3ca3ea-1fb3-4b43-9d55-6fe6fdbbd0ae/Ruderman-statistics.pdf)

따라서 Ruderman의 연구 목표는: **(1) 자연 이미지와 무작위 이미지의 통계적 차이 정량화, (2) 이미지에 내재된 간단한 불변성 원리 발견, (3) 이를 통한 시각 처리 시스템의 최적 설계 원리 도출**이었습니다.

***

## 2. 제안하는 방법과 수식 (상세)

### 2.1 데이터 수집 및 전처리

뉴저지 숲에서 수집한 이미지는 다음과 같이 처리됩니다:

**로그-콘트라스트 변환:**
$$z(x) = \ln\frac{I(x)}{I_0}$$

여기서 $I(x)$는 측정된 휘도이고, $I_0$는 각 이미지마다 $\sum_x z(x) = 0$이 되도록 선택됩니다. 이 로그 변환은 시각 시스템의 적응 특성을 반영합니다(평균 조명에 무관). [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/8b3ca3ea-1fb3-4b43-9d55-6fe6fdbbd0ae/Ruderman-statistics.pdf)

### 2.2 전력 스펙트럼 분석

**원점에서의 전력 스펙트럼:**
$$\langle \tilde{z}_m(k_1, k_2) \rangle = \frac{1}{2\pi}|k|^{-\gamma}$$

여기서 $k = \sqrt{k_1^2 + k_2^2}$는 공간 주파수의 크기이고, $\gamma = 0.19 \pm 0.01$입니다. 이 멱법칙 스케일링은 이미지의 스케일 불변성을 나타냅니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/8b3ca3ea-1fb3-4b43-9d55-6fe6fdbbd0ae/Ruderman-statistics.pdf)

**스케일 불변성의 수학적 표현:**
$$Q(az) = a^u Q(z)$$

여기서 $Q(z)$는 임의의 앙상블 통계이고, $u$는 보편 지수(universal exponent)입니다. 이는 이미지를 스케일 $a$만큼 확대/축소해도 통계적 형태가 변하지 않음을 의미합니다(재정규화 후). [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/8b3ca3ea-1fb3-4b43-9d55-6fe6fdbbd0ae/Ruderman-statistics.pdf)

### 2.3 코어스 그레이닝과 히스토그램 스케일링

**블록 평균화:**
$$\varphi_N(m,n) = \frac{1}{N^2}\sum_{i,j=1}^{N}\varphi(m+i, n+j)$$

**정규화된 히스토그램:**
$$P(m_N) \text{ vs } P(m_N/\sigma_N)$$

놀랍게도, 크기 $N = 1, 2, 4, 8, 16, 32$에서 정규화된 분포가 동일한 형태를 유지합니다 (중심극한정리 위반). 이는 픽셀들이 독립적이 아니라 강하게 상관되어 있음을 의미합니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/8b3ca3ea-1fb3-4b43-9d55-6fe6fdbbd0ae/Ruderman-statistics.pdf)

### 2.4 그래디언트 분포 분석

**이산 그래디언트 크기:**
$$G_N(m,n) = \sqrt{[\varphi(m+1,n)-\varphi(m,n)]^2 + [\varphi(m,n+1)-\varphi(m,n)]^2}$$

**핵심 발견**: 이 분포는 선형 필터링으로 제거될 수 없는 강한 비-가우시안 특성을 보입니다. 특히 매우 긴 지수 꼬리를 가집니다:

$$P(G_N) \sim e^{-\alpha G_N}$$

반면 가우시안 이미지는 Rayleigh 분포를 따릅니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/8b3ca3ea-1fb3-4b43-9d55-6fe6fdbbd0ae/Ruderman-statistics.pdf)

### 2.5 상호정보와 예측 가능성

**두 픽셀 간 상호정보:**
$$I_d = \sum_{z_1, z_2} p(z_1, z_2) \log\frac{p(z_1, z_2)}{p(z_1)p(z_2)}$$

**멱법칙 스케일링:**
$$I_d \propto d^{-e}$$

여기서 $e \approx 0.84$입니다. 수평 분리 대비 수직 분리에서 더 강한 상관(의존도 최소값)을 보이는데, 이는 숲의 수직 트리 구조 때문입니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/8b3ca3ea-1fb3-4b43-9d55-6fe6fdbbd0ae/Ruderman-statistics.pdf)

**선형 예측:**
$$\hat{z}(z_i) = m_d \cdot z_i + b_d$$

최근접 이웃 픽셀만으로도 RMS 예측 오류가 전체 픽셀 변동의 70%이고, 8개 이웃을 사용하면 50%로 감소합니다. 이는 상당한 공간적 중복을 시사합니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/8b3ca3ea-1fb3-4b43-9d55-6fe6fdbbd0ae/Ruderman-statistics.pdf)

### 2.6 분산 정규화 절차 (혁신적 기여)

**1단계: 분산 정규화**
$$\varphi'(x) = \frac{\varphi(x) - \mu_x}{\sigma_x}$$

여기서:
- $\mu_x = \frac{1}{N^2}\sum_{(m,n) \in \text{N×N 윈도우}}  \varphi(m,n)$
- $\sigma_x = \sqrt{\frac{1}{N^2}\sum ([\varphi(m,n)-\mu_x]^2)}$
- $N = 5$일 때 최적 (쿠르토시스 최소화) [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/8b3ca3ea-1fb3-4b43-9d55-6fe6fdbbd0ae/Ruderman-statistics.pdf)

**핵심 결과**: 정규화된 이미지 $\varphi'(x)$의 히스토그램은 거의 가우시안이 되고, 그래디언트는 Rayleigh 분포를 따릅니다.

**2단계: 분산 이미지 생성**
$$U(x) = \sigma_x$$

정규화되지 않은 분산 이미지의 로그-콘트라스트:
$$w(x) = \ln\frac{U(x)}{U_0}$$

놀랍게도 $w(x)$의 통계는 원본 $\varphi(x)$의 통계와 동일합니다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/8b3ca3ea-1fb3-4b43-9d55-6fe6fdbbd0ae/Ruderman-statistics.pdf)

$$\text{Cov}(w_1, w_2) \approx \text{Cov}(\varphi_1, \varphi_2)$$

**3단계: 재귀 적용**
$\varphi'$에서 시작하여 11×11 블록으로 다시 절차를 수행하면, 다시 정규화된 신호 $\varphi''$를 얻고, 이로부터 추출한 분산 $w_2$는 다시 원본과 유사한 통계를 보입니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/8b3ca3ea-1fb3-4b43-9d55-6fe6fdbbd0ae/Ruderman-statistics.pdf)

**수학적 의미**: 자연 이미지는 **계층적 분산 불변성(hierarchical variance invariance)**을 만족합니다:
$$p(\varphi) = p(\log \sigma_N) = p(\log \sigma_{N}^{(2)}) = \cdots$$

이는 선형 변환으로는 절대 제거될 수 없는 비선형 대칭성입니다.

### 2.7 망막 정보 용량

**광수용체 배열 모델**: 육각형 격자 배치 (황반부의 실제 배열)

**픽셀당 정보량:**
$$I_{\text{receptor}} = \log_2\left(1 + \frac{S_k}{a^2 \cdot N_k} \cdot \text{SNR}\right)$$

여기서 $a$는 픽셀 간격이고, $N_k$는 노이즈 전력입니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/8b3ca3ea-1fb3-4b43-9d55-6fe6fdbbd0ae/Ruderman-statistics.pdf)

**결과**: SNR 범위 1~1000에서 픽셀당 수 비트로, 공간적 중복도가 2배 이상입니다.

***

## 3. 모델 구조와 구성 요소

### 3.1 통계적 프레임워크

Ruderman의 접근법은 **통계 신호 처리의 전형적 패러다임**을 따릅니다:

**Bayes 추정의 사후 분포:**
$$p(\varphi|y) = \frac{p(y|\varphi)p(\varphi)}{p(y)}$$

여기서 $p(\varphi)$는 이미지 앙상블(자연 이미지 분포)이고, $p(y|\varphi)$는 센서 모델(광학, 양자 노이즈)입니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/8b3ca3ea-1fb3-4b43-9d55-6fe6fdbbd0ae/Ruderman-statistics.pdf) 

**저SNR 확장(Low SNR Expansion)**: 
$$\hat{\varphi}(k) = \frac{S_k(k)}{S_k(k) + N_k} \tilde{y}(k)$$

이는 Wiener 필터 형태로, 자연 이미지의 전력 스펙트럼이 최적 필터 설계의 핵심 입력입니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/8b3ca3ea-1fb3-4b43-9d55-6fe6fdbbd0ae/Ruderman-statistics.pdf)

### 3.2 계층적 처리 구조

```
입력 이미지 φ(x)
    ↓
[로그-콘트라스트 변환]
    ↓
φ'(x) = [φ(x) - μ(x)] / σ(x)  ← 정규화 (N=5)
    ↓
[거의 가우시안 분포 획득]
    ↓
σ(x) → w(x) = ln[σ(x)/σ_0]
    ↓
[w의 통계 ≈ φ의 통계]
    ↓
재귀: w에 동일 절차 적용 (N=11)
    ↓
φ''(x) + w_2(x) [다시 원본 통계 복제]
```

### 3.3 모델의 설계 기준

Ruderman이 암묵적으로 제시하는 최적 설계 원리:

**원리 1: 중복 제거**
자연 이미지의 ~90% 정보가 압축 가능 → 신경망은 불필요한 중복 제거 필요

**원리 2: 비-가우시안 특성 보존**
선형 필터링만으로는 비-가우시안 꼬리 제거 불가 → 비선형 게인 제어(정규화) 필수

**원리 3: 계층적 구조**
분산의 분산이 분포와 동일 → 다중 스케일 처리 아키텍처가 자연스러움

***

## 4. 성능 향상 결과

### 4.1 압축 예측

**이론적 상한(상한치)**: 
- 로우SNR 영역: 전력 스펙트럼만으로 충분
- 고SNR 영역: 더 높은 차수 상관 함수 필요

**실증 결과**:
- 실제 TV 신호 압축: 3 bits/픽셀 최소 중복도 (Kretzmer 1952)
- Ruderman의 숲 이미지: 코드북 기반 분석으로 10% 근처 중복 가능성 시사

### 4.2 신경 반응 예측

**대비 응답 곡선 예측**:
Laughlin의 최대 정보 전송 기준과 Ruderman의 대비 히스토그램 결합:
$$R(c) = R_{\max} \cdot \frac{c^n}{c^n + c_50^n}$$

이 S자 곡선은 파리 LMC 세포의 실제 응답과 정량적으로 일치합니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/8b3ca3ea-1fb3-4b43-9d55-6fe6fdbbd0ae/Ruderman-statistics.pdf)

### 4.3 신경 레이아웃 설계

**황반부 광수용체 간격**:
Ruderman의 분석은 Nyquist 샘플링 기준이 자연 이미지 통계와 어떻게 최적이 되는지 설명:
- 육각형 격자의 공간 해상도 ≈ 이미지의 특성 공간 주파수
- 이는 포유류 황반부의 실제 해부학적 배치와 일치 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/8b3ca3ea-1fb3-4b43-9d55-6fe6fdbbd0ae/Ruderman-statistics.pdf)

### 4.4 정량적 메트릭

| 메트릭 | 값 | 함의 |
|--------|-----|------|
| 전력 스펙트럼 지수 ($\gamma$) | 0.19±0.01 | 정규화된 로그-콘트라스트 |
| 스케일 범위 | 2.5 orders (주파수) | 인간 시각 범위 포함 |
| 비이웃 중복도 | ~10% | 근처 픽셀 예측 가능 |
| 정규화 후 쿠르토시스 | ~3 (가우시안) | 거의 정규 분포 달성 |
| 재귀 가능 깊이 | 2단계 | 데이터 제한으로 3단계 불가 |

***

## 5. 모델의 한계

### 5.1 데이터 한계

**규모**: 45개(15mm) + 25개(80mm) 이미지는 256² = 65,536차원 공간에서 매우 작은 표본입니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/8b3ca3ea-1fb3-4b43-9d55-6fe6fdbbd0ae/Ruderman-statistics.pdf)

**이론적 표본 크기**: $4 \times 4$ 이미지만 해도 $256^{16} \approx 10^{38}$ 가능한 이미지 중 수집해야 분포를 적절히 표현할 수 있습니다. 따라서 완전한 확률 분포 $P(\varphi)$를 구성할 수 없습니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/8b3ca3ea-1fb3-4b43-9d55-6fe6fdbbd0ae/Ruderman-statistics.pdf)

**해결책**: Ruderman은 **최소 충분 통계(minimal sufficient statistics)** 접근으로 몇 가지 상관 함수와 불변성만 분석합니다.

### 5.2 환경 특이성

**숲의 특성**: 뉴저지 봄의 숲은 수직 방향성이 강함(나무). 따라서:
- 수평 대비 수직 상관이 더 강함 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/8b3ca3ea-1fb3-4b43-9d55-6fe6fdbbd0ae/Ruderman-statistics.pdf)
- 해변 이미지는 다른 전력 스펙트럼 지수 ($\gamma \approx 0.3$) 표시 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/8b3ca3ea-1fb3-4b43-9d55-6fe6fdbbd0ae/Ruderman-statistics.pdf)
- 도시, 실내, 다른 기후의 통계 미포함

**일반화 위험**: 모든 자연 환경이 동일한 통계를 가지지 않습니다.

### 5.3 재귀 절차의 미완성

**데이터 부족**: 재귀 절차는 이론적으로 무한히 반복 가능하지만, 3단계는 불가능합니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/8b3ca3ea-1fb3-4b43-9d55-6fe6fdbbd0ae/Ruderman-statistics.pdf)

**최적성 미증명**: 왜 5×5, 11×11 블록이 최적인지 이론적 증명이 없습니다. 원형 윈도우나 가중 평균이 더 나을 수 있습니다.

**수렴성**: 재귀의 고정점(fixed point)이 존재하는지, 그것이 유일한지 증명되지 않았습니다.

### 5.4 고차 통계의 해석 부족

**2점 상관만 사용**: 전력 스펙트럼은 2점 상관 함수만 포함합니다:
$$S_k = \mathcal{F}\{\langle \varphi(x)\varphi(x+d) \rangle\}$$

**3점, 4점 상관**: Bispectrum, Trispectrum 등 고차 통계는 분석하지 않습니다. 이들은 비선형 위상 관계를 캡처하며, 자연 이미지의 엣지 구조에 중요할 수 있습니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/8b3ca3ea-1fb3-4b43-9d55-6fe6fdbbd0ae/Ruderman-statistics.pdf)

### 5.5 시간 영역 확장 부재

**정적 이미지만**: 논문은 움직이는 장면(동영상)의 시간적 상관을 분석하지 않습니다. 시간 스케일의 불변성도 존재할 수 있습니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/8b3ca3ea-1fb3-4b43-9d55-6fe6fdbbd0ae/Ruderman-statistics.pdf)

***

## 6. 모델의 일반화 성능 향상 가능성

### 6.1 이론적 토대

#### 스케일 불변성의 강력함
$$Q(\alpha z) = \alpha^u Q(z)$$

이 대칭성은 카메라 초점거리, 객체 거리, 조명 조건의 변화에 대한 불변성을 제공합니다. 따라서 학습된 특징이 여러 환경에서 전이 가능합니다.

**응용**: 전이 학습에서 ImageNet 사전학습 모델이 의료 이미지, 위성 이미지 등 다양한 도메인으로 일반화되는 이유가 여기에 있습니다. [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC11048359/)

#### 계층적 분산 정규화의 모듈성
$$\sigma_{\text{multi-scale}} \text{는} \varphi \text{와 동일 통계}$$

이는 신경망의 각 계층에서 독립적으로 정규화할 수 있음을 의미합니다. 각 계층이 이미지의 다른 수준의 구조를 처리할 때:
- 계층 1: 저수준 텍스처 (픽셀 수준 분산)
- 계층 2: 중간 수준 객체 경계 (지역적 분산)
- 계층 3: 고수준 장면 구조 (전역 분산)

모두 동일한 통계 구조를 가지므로, 같은 비선형 연산(정규화)이 모든 계층에서 작동합니다.

### 6.2 신경망 설계에의 응용

#### Batch Normalization으로의 진화
Ioffe & Szegedy(2015)의 배치 정규화:
$$\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}$$

이는 Ruderman의 분산 정규화의 직접적 현대화입니다. 배치는 미니 앙상블 역할을 합니다.

#### Divisive Normalization의 신경생물학적 타당성

2021년 PNAS 논문 "Divisive Normalization Unifies Disparate Response..."는 Ruderman의 분산 정규화가 사실 전시각 피질(V1~V3)의 일관된 신경 계산이라고 증명했습니다: [pnas](https://www.pnas.org/doi/10.1073/pnas.2108713118)

$$R(x) = \frac{R_0(x)}{\alpha + \sum_j R_0(y_j)}$$

여기서 $R_0(x)$는 선형 응답(여과)이고, 분모는 주변 신경(surround suppression)을 나타냅니다.

**일반화 효과**: 이 정규화가 표준화되면:
- **Shift invariance 개선**: 조명 변화에 강건
- **Contrast sensitivity**: 상대적 대비만 인코딩 (절대값 무관)
- **Feature disentanglement**: 독립적인 특징 학습 [sciencedirect](https://www.sciencedirect.com/science/article/pii/S0167865523002209)

### 6.3 깊은 신경망에서의 성능

#### ImageNet 사전학습의 통계적 토대

ImageNet으로 학습된 CNN은 사실상 자연 이미지의 통계를 암묵적으로 학습합니다. [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC11048359/)

**계층별 특징**:
- **조기 계층** (conv1, conv2): Gabor 필터 같은 지향성 엣지 (강한 그래디언트 분포와 일치)
- **중간 계층**: 질감, 모양 조합 (계층적 구조)
- **후기 계층**: 의미 속성 (모든 자연 환경에서 공유)

따라서 ImageNet 사전학습이 도메인 간 전이를 잘 수행합니다. [nature](https://www.nature.com/articles/s41598-023-33887-5)

#### Transfer Learning 성능 메트릭

최근 연구(2024)에 따르면: [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC11048359/)

| 시나리오 | 1 epoch 정확도 | 20 epoch 정확도 |
|---------|--------------|--------------|
| Transfer Learning | >80% | >90% |
| Training from Scratch | <30% | ~75% |

**해석**: 자연 이미지 통계 기반 특징이 초기부터 매우 강력합니다. 이는 Ruderman의 스케일 불변성이 다양한 객체와 배경을 빠르게 학습하게 하는 효과입니다.

### 6.4 특정 작업에서의 일반화 향상 전략

#### 1. 다중 스케일 정규화 (Multi-Scale Normalization)
기존 배치 정규화는 단일 스케일의 통계만 정규화합니다:
$$\hat{x}_i = \gamma \cdot \frac{x_i - \mu_B}{\sqrt{\sigma_B^2}} + \beta$$

**개선안**: Ruderman의 계층적 접근을 따르면:

$$\hat{x}^{(l)} = \frac{x^{(l)} - \mu_B^{(l)}}{\sqrt{\sigma_B^{(l)2} + \epsilon}}, \quad l = 1,2,\ldots,L$$

각 계층에서 독립적 정규화함으로써 국소 변동성(local variance)을 포착합니다.

**효과**: 2023년 Hierarchical Image Transformation (HIT-MiLF) 논문은 이 접근으로 이상 탐지에서 13% 성능 향상을 보였습니다. [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC9861680/)

#### 2. 자연 장면 통계 기반 정규화
일반적인 정규화 상수 대신, 자연 이미지의 수집된 통계를 사용:

$$\hat{x}\_i = \gamma \cdot \frac{x_i - \mu_{\text{natural}}}{\sqrt{\sigma_{\text{natural}}^2 + \epsilon}} + \beta$$

여기서 $\mu_{\text{natural}}, \sigma_{\text{natural}}$는 큰 자연 이미지 데이터베이스에서 계산합니다.

**타당성**: 2020년 논문 "Low-level Image Statistics in Natural Scenes Influence Visual Perception"은 장면 복잡도(contrast energy, spatial coherence)가 인지 성능을 직접 조절함을 보였습니다. [nature](https://www.nature.com/articles/s41598-020-67661-8)

#### 3. 도메인 외(Out-of-Distribution) 강건성
자연 이미지 통계에 대한 사전이 얼마나 강한지:

**실험**: 스크린 콘텐츠 이미지(SCI) 품질 평가 [arxiv](https://arxiv.org/html/2209.05321v4)
- 기존 딥러닝 NR-IQA: 교차 데이터셋에서 PLCC 0.52
- 자연 장면 통계 기반 방법(DFSS-IQA): PLCC 0.79

**원인**: 자연 이미지 통계의 보편성이 매우 강하여, 컴퓨터 생성 이미지(SCI)와의 차이를 감지하는 데 도움이 됩니다.

### 6.5 최신 연구와의 연결고리

#### 계층적 VAE (Hierarchical Variational Autoencoder)
2023년 논문 "Hierarchical VAEs provide a normative account of motion perception"은 다음을 보였습니다: [biorxiv](https://www.biorxiv.org/content/10.1101/2023.09.27.559646v2.full)

$$\varphi^{(l)} = \mu^{(l)} + z^{(l)} \odot \sigma^{(l)}$$

여기서 각 계층 $l$의 잠재 변수 $z^{(l)}$는 독립적이고, $\sigma^{(l)}$는 계층별 불확실성입니다.

**Ruderman과의 연결**: 이는 정확히 계층적 분산 정규화의 확률적 해석입니다. VAE가 학습한 다중 스케일 분산이 Ruderman의 이론과 동일한 구조를 가집니다.

#### Scale-Invariant Wavelets과 Multiscale Analysis
2022년 "On Scale-Invariant Properties in Natural Images" 논문은 Ruderman의 발견을 확장하여: [arxiv](https://arxiv.org/pdf/2201.13312.pdf)
- Gabor 웨이블릿의 다중 스케일 응답이 멱법칙 구조를 따름
- 동적 모델로 이 멱법칙 전력 스펙트럼 재현 가능

**함의**: 신경망의 다중 감수야 구조는 자연 이미지의 멱법칙 스케일 구조에 최적화되어 있습니다.

***

## 7. 2020년 이후 관련 최신 연구 비교 분석

### 7.1 Scale Invariance 연구의 발전

#### Ruderman (1994): 기초 발견
- **발견**: $S(k) \propto k^{-\gamma}$, $\gamma = 0.19$
- **방법**: 전력 스펙트럼 분석
- **한계**: 단일 환경, 제한된 데이터

#### Cohen et al. (1975), Deriugin (1956): 선행 언급
- 텔레비전 신호에서 멱법칙 관찰
- 응용 목적 (부호화)만 명시

#### Burton & Moorhead (1987), Field (1987): 재발견
- 무색 이미지 재확인
- 색 이미지로 확장

#### 최신: Scale-Wise Convolution (2019) [arxiv](https://arxiv.org/pdf/1912.09028.pdf)
$$y_{s,l} = \sum_j w_{s,l,j} * x_j$$

여기서 $s$는 스케일 인덱스입니다.

**발전**: 단순 스펙트럼 분석 → CNN에 내재된 계산으로 변화
- 깊이별 스케일 처리
- 멀티스케일 테스팅보다 효율적

### 7.2 Hierarchical Model의 진화

#### Ruderman (1994): 재귀적 분산 정규화
- 2단계까지만 검증
- 이론적 수렴성 미증명

#### Saremi et al. (2013): Hierarchical Ising Model [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC3581899/)
$$\mathcal{H} = -J \sum_{<i,j>} s_i s_j - \sum_i h_i s_i$$

각 강도 계층을 이중화 변수로 매핑하여, 임계점 근처의 계층이 장거리 상관을 생성함을 증명했습니다. [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC3581899/)

**의의**: 
- Ruderman의 직관(계층적 구조)을 물리학 프레임워크(Ising 모델)로 정당화
- 스케일 불변성의 물리적 기원 규명 (2차 상전이)

#### 최신: Nested Subspace Networks (2025) [arxiv](https://arxiv.org/html/2509.17874v1)
$$\mathbf{W}^{(r)} = \mathbf{U}^{(r)} \mathbf{V}^{(r)\top}, \quad r = 1, 2, \ldots, R$$

저순위 행렬의 계층을 동시에 최적화하여:
- 단일 모델에서 다양한 계산 비용 처리
- 중요도 순서(계층구조) 자동 학습

**연결고리**: Ruderman의 재귀적 분산이 신경망의 계층적 저순위 구조로 구체화됨.

### 7.3 Divisive Normalization의 신경과학적 확인

#### Ruderman (1994): 분산 정규화의 발견적 소개
- 수학적 우아함
- 신경생물학적 기반 미명시

#### Carandini & Heeger (2012): Divisive Normalization 리뷰
- V1, MT, MSTd 등에서 실증
- 정규화 메커니즘의 신경화학적 토대 제시

#### Aqil et al. (2021): 시각 피질 전체에서의 일관된 계산 [pnas](https://www.pnas.org/doi/10.1073/pnas.2108713118)

```math
R(x)=\frac{R_{0}(x)^{p}}{d^{p}+\sum _{j}R_{0}(y_{j})^{p}}
```

여기서 $p = 2$이고, 정규화 상수 $d$가 계층마다 변합니다.

**발견**:
- V1: 강한 억제 ($d$ 작음)
- V2, V3: 중간 억제
- IT: 약한 억제 ($d$ 큼)

**함의**: 정규화의 강도가 처리 수준에 따라 조정됨 (Ruderman의 N=5→11→... 발전과 유사)

#### Burg et al. (2021): V1에서 DN의 학습 [journals.plos](https://journals.plos.org/ploscompbiol/article?id=10.1371%2Fjournal.pcbi.1009028)
$$R(x) = \frac{g \cdot x}{1 + \sum_j c_j |R_j|}$$

신경 데이터에서 직접 학습 가능하며, 정규화 계수 $c_j$가 방향 특이성을 반영합니다. [journals.plos](https://journals.plos.org/ploscompbiol/article?id=10.1371%2Fjournal.pcbi.1009028)

**신경망 응용**: CNN에 같은 구조 도입 시 일반화 성능 향상. [arxiv](https://arxiv.org/html/2407.17829v1)

### 7.4 Transfer Learning과 Generalization

#### 전통적 Transfer Learning의 한계
- ImageNet → Medical: 도메인 시프트로 인한 성능 저하
- Domain-specific pretraining 필요

#### Real-World Feature Transfer Learning (2024) [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC11048359/)
**핵심 발견**: ImageNet 사전학습 모델의 특징은 **자연 이미지 통계를 암묵적으로 학습**하였으며, 이는 의료 이미지에도 적용 가능합니다.

| 모델 | ImageNet 정확도 | 의료 X선 (0 epoch) | 의료 X선 (20 epoch) |
|-----|----------------|-----------------|------------------|
| ResNeXt50 | 79.3% | 78% | 96% |
| From Scratch | - | 12% | 75% |

**해석**: Ruderman의 스케일 불변성이 자연 이미지(ImageNet)와 의료 이미지 간 특징 전이를 가능하게 합니다.

#### Deep Transfer Learning Strategy (2023) [nature](https://www.nature.com/articles/s41598-023-33887-5)
하위 합성곱 계층(domain-specific) vs 상위 완전연결 계층(general)의 재훈련 전략:

| 계층 | 역할 | 재훈련 필요도 |
|-----|------|------------|
| Conv1-3 | 저수준 도메인 특징 | 높음 |
| Conv4-5 | 중수준 일반 특징 | 중간 |
| Dense | 고수준 의미 | 낮음 |

**Ruderman 해석**: 각 계층의 통계가 계층적으로 구조화되어 있으므로, 깊을수록 일반화 가능성이 높습니다 (계층적 분산 정규화 원리).

### 7.5 Natural Scene Statistics의 응용 심화

#### Low-level Statistics와 Perception (2020) [nature](https://www.nature.com/articles/s41598-020-67661-8)
$$\text{SC} = \sqrt{\text{Variance}(\text{Contrast})}$$

이 지표로 장면 복잡도를 정량화하여 인지 반응 시간 예측:
$$\text{RT} = a + b \cdot \text{SC}^2$$

**의의**: Ruderman의 콘트라스트 히스토그램이 실제 인지 성능을 결정합니다.

#### Scene Category와 Brain Representation (2013) [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC5464350/)
뇌 fMRI에서 장면 카테고리(산, 숲, 도시)를 예측하는 데, 객체 공출현 통계가 가장 강한 예측자입니다. [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC5464350/)

**원리**: Ruderman이 밝혀낸 자연 이미지의 통계적 구조가 뇌의 신경 표현에 직접 반영됩니다.

#### Deep Feature Statistics IQA (2022) [arxiv](https://arxiv.org/html/2209.05321v4)
자연 vs 스크린 콘텐츠 이미지의 품질 평가:

$$\mathcal{L}_{\text{quality}} = \text{MMD}(F_{\text{input}}, \mathcal{N}(0, I))$$

여기서 특징 분포를 정규 분포와 비교합니다.

**성능**: 
- 동일 데이터셋: PLCC 0.95
- 교차 데이터셋: PLCC 0.79 (기존 방법 대비 +27%)

**원인**: 자연 이미지 통계의 보편성이 매우 강함.

***

## 8. 앞으로의 연구에 미치는 영향과 고려 사항

### 8.1 이론적 기초 확립의 필요성

#### 스케일 불변성의 물리적 기원
**현재**: 경험적 관찰만 존재
$$S(k) \propto k^{-\gamma}$$

**필요한 이론**:
1. **Fractal Dimension**: 자연 풍경이 fractal 특성을 가지는가?
   $$D_H = \frac{\ln N}{\ln(1/r)}, \quad \text{where } D_H \in $$ [ijmcs.co](http://ijmcs.co.uk/details&cid=3)
   
   Mandelbrot은 해안선이 fractal임을 제시했으나, 2D 자연 이미지의 fractal 차원은 미결정입니다.

2. **객체 크기 분포의 멱법칙**: 
   $$P(\text{object size} = s) \propto s^{-\alpha}$$
   
   이것이 전력 스펙트럼의 $k^{-\gamma}$로 매핑되는 과정을 수학적으로 증명해야 합니다. [sciencedirect](https://www.sciencedirect.com/science/article/pii/S0042698997000084)

3. **Critical Phenomena와의 연결**: 
   Saremi et al.(2013)이 Ising 모델의 임계점과 연결했으나, 실제 자연 이미지가 왜 임계 상태 근처인지 설명 필요:
   
   $$\xi(T) \sim |T - T_c|^{-\nu}, \quad \nu = 1 \text{ (2D)}$$
   
   임계점에서만 스케일 불변성이 정확히 성립합니다.

#### 분산 정규화의 최적성
**현재**: 휴리스틱 발견
$$\varphi'(x) = \frac{\varphi(x) - \mu_x}{\sigma_x}$$

**필요한 이론**:
1. **정보 이론적 최적성**:
   $$I(\varphi'; \text{다른 패치})$$
   vs
   $$I(\varphi; \text{다른 패치})$$
   
   정규화가 정보 전송(mutual information)을 최대화하는가?

2. **최적 블록 크기의 결정**:
   현재는 쿠르토시스 최소화로 N=5, N=11 선택.
   
   $$N_{\text{opt}} = \arg\max_N \text{Information Transfer}$$
   
   또는 이론적 유도 필요.

3. **수렴성과 고정점**:
   $$\varphi^{(\infty)} = \lim_{n \to \infty} \text{NormalizationOp}^n(\varphi)$$
   
   이것이 특정 분포(예: 표준 정규 분포)로 수렴하는가?

### 8.2 실무적 과제

#### 다양한 환경의 통계 수집

**현재 데이터**: 뉴저지 봄 숲, 45+25 이미지

**필요**:
- **환경별**: 도시, 해변, 사막, 산악, 극지, 동굴, 실내, 수중
- **계절별**: 봄, 여름, 가을, 겨울 (조명, 식생 변화)
- **시간대별**: 일출, 정오, 황혼, 야간 (색 온도, 그림자)
- **날씨별**: 맑음, 흐림, 비, 눈 (산란, 스펙큘러 반사)

**예상 발견**:
- 모든 환경이 동일한 $\gamma \approx 0.19$를 가지는가?
- 방향 이방성(anisotropy)이 환경마다 다른가?
- 계절에 따른 "온도(temperature)" 매개변수 변화?

**타당성**: 기존 van Hateren 데이터베이스(16,000 이미지)를 활용하되, 메타데이터(장소, 시간, 조건) 추가 수집.

#### 신경망 아키텍처에의 통합

**현재 상태**:
- Batch Normalization: 미니배치 통계 사용
- Instance Normalization: 이미지별 통계 사용
- Group Normalization: 채널 그룹별 통계

**개선안**: Natural Image Statistics-Aware Normalization (NISAN)

$$\hat{x}\_{i,c} = \gamma_{c} \frac{x_{i,c} - \mu_{\text{natural},c}}{\sqrt{\sigma_{\text{natural},c}^2 + \epsilon}} + \beta_{c}$$

여기서:
- $\mu_{\text{natural},c}$: 자연 이미지의 채널 $c$의 평균
- $\sigma_{\text{natural},c}$: 자연 이미지 데이터베이스에서 계산

**예상 효과**:
- 도메인 외 강건성 향상
- 적응형 정규화 강도 (계층마다 다를 수 있음)

#### 계층적 정규화의 실무 구현

**이상적 구조** (Ruderman 이론 기반):
```
Input image φ
  ↓ Conv1 + BN (σ₁ 인코딩) + Pool
  ↓ Conv2 + BN (σ₂ 인코딩) + Pool
  ↓ Conv3 + BN (σ₃ 인코딩) + Pool
  ...
각 계층의 통계는 계층별로 독립적인 분산 불변성을 만족해야 함
```

**검증 방법**:
1. 각 계층 활성화의 로그 분포 측정
2. 다중 스케일 그래디언트 히스토그램 생성
3. Ruderman 기준(지수 꼬리, 스케일 불변성)과 비교

**성공 지표**: 계층이 깊어져도 히스토그램 형태가 유지되면 ✓

### 8.3 연구 시 고려할 근본적 문제들

#### 1. Generalization의 원천
**질문**: 왜 자연 이미지 통계에 기반한 설계가 새로운 도메인(의료, 위성 이미지)에서도 일반화되는가?

**Ruderman의 설명**:
- 스케일 불변성 덕분에 객체 거리 무관
- 분산 정규화의 비선형성이 조명 변화 흡수

**더 깊은 답**: 
- **자연 선택**: 생물 시각 시스템은 자연 환경에 최적화 → 이것이 모든 자연 이미지의 기초 구조
- **정보 기하학**: Riemannian 기하학에서 자연 이미지의 매니폴드가 낮은 곡률?

#### 2. 비-가우시안 특성의 정보 역할
**현재 이해**: 
- 긴 꼬리 = 드문 하지만 중요한 특징 (에지, 코너)
- 선형 필터링으로 제거 불가능

**미해결**: 
- 정확히 어떤 정보가 꼬리에 인코딩되는가?
- 이를 신경망이 어떻게 활용하는가?

**가설**: 고대비 영역(에지)의 초과 표현이 객체 인식의 핵심. 정규화는 이러한 드문 사건을 증폭시킵니다.

#### 3. 시간 영역의 확장
**미제 문제**: 동영상의 시간 통계

**예상 구조**:
$$S(k_x, k_y, k_t) \propto (k_x^2 + k_y^2 + k_t^2)^{-\gamma/2}?$$

또는 이방성:
$$S(k_x, k_y, k_t) \propto (k_{\perp}^2 + k_t^2)^{-\gamma_{\perp}} \cdot g(k_x/k_y)$$

**응용**: 동영상 압축(MPEG), 3D CNN 설계에 직접 영향

#### 4. 물리적 제약과의 연결
**질문**: 자연 이미지의 통계가 물리 법칙(조명, 기하학)에 의해 결정되는가?

**가능한 틀**:
$$I(x) = \rho(x) \cdot L(x) \cdot \cos\theta(x)$$

여기서:
- $\rho$: 표면 반사율 (느리게 변함)
- $L$: 조명 (매우 변함) 
- $\cos\theta$: 표면 법선 (중간 정도)

Ruderman의 로그-콘트라스트는 이러한 곱셈 분해를 **덧셈 분해**로 변환:
$$z = \ln\rho + \ln L + \ln\cos\theta$$

이것이 왜 자연 이미지 통계를 "정규화"하는가?

### 8.4 새로운 평가 지표의 제안

#### Ruderman Compliance Index (RCI)
신경망이 Ruderman의 자연 이미지 통계를 얼마나 잘 반영하는가 측정:

$$\text{RCI} = \alpha \cdot \text{ScaleInvarianceScore} + \beta \cdot \text{VarianceNormalizationScore} + \gamma \cdot \text{NonGaussianScore}$$

여기서:
- **ScaleInvarianceScore**: 다중 스케일 입력에서 활성화 분포의 불변성
- **VarianceNormalizationScore**: 계층별 분산이 입력과 독립적인 정도
- **NonGaussianScore**: 활성화 분포의 첨도(kurtosis) 얼마나 Gaussian에서 벗어나는지

**용도**: 신경망이 자연 이미지에 얼마나 적합하게 구조화되었는지 정량화.

#### Domain Generalization Taxonomy
Ruderman의 불변성을 기반으로 도메인 외 일반화 실패를 분류:

| 실패 원인 | 분류 | 해결책 |
|---------|------|-------|
| 스케일 변화 | Scale Invariance 위반 | 다중 스케일 데이터 증강 |
| 조명 변화 | Variance Normalization 부족 | 적응형 정규화 추가 |
| 배경 변화 | 비-가우시안 특성 미포착 | 고대비 특징 강조 |

***

## 결론

Ruderman의 "The Statistics of Natural Images"는 단순한 데이터 분석을 넘어 **시각 정보 처리의 기초 원리를 규명한 역사적 논문**입니다. 스케일 불변성과 계층적 분산 정규화 불변성은 30년이 지난 오늘날에도 신경망 설계, 신경과학, 정보 이론의 핵심 개념으로 작용합니다.

**핵심 기여**:
1. **자연 이미지의 정량화**: 무작위성과 구조의 명확한 경계 설정
2. **멀티스케일 불변성**: CNN의 계층적 구조를 이론적으로 정당화
3. **비선형 정규화의 필요성**: Batch Normalization, Divisive Normalization의 선구자 역할

**2020년 이후 진화**:
- ImageNet 사전학습이 자연 이미지 통계의 학습임을 실증적으로 확인
- Divisive Normalization이 전시각 피질의 일관된 신경 계산임을 신경생물학적으로 증명
- Deep Learning의 일반화 성능이 Ruderman의 불변성에서 비롯됨을 규명

**미래 과제**:
자연 이미지 통계의 물리적 기원, 시간 영역 확장, 최적화 이론의 보강이 필요하며, 이는 고급 시각 기술과 신경과학적 이해의 새로운 경계를 열 것입니다.

***

### 참고 자료 (주요 인용)

<span style="display:none">[^1_22][^1_23][^1_24][^1_25][^1_26][^1_27][^1_28][^1_29][^1_30][^1_31][^1_32][^1_33][^1_34][^1_35][^1_36][^1_37][^1_38][^1_39][^1_40][^1_41][^1_42][^1_43][^1_44][^1_45][^1_46][^1_47][^1_48][^1_49][^1_50][^1_51][^1_52][^1_53][^1_54][^1_55][^1_56][^1_57][^1_58][^1_59][^1_60][^1_61][^1_62][^1_63][^1_64][^1_65][^1_66][^1_67][^1_68][^1_69][^1_70][^1_71][^1_72][^1_73][^1_74][^1_75][^1_76][^1_77][^1_78][^1_79][^1_80][^1_81][^1_82][^1_83][^1_84][^1_85][^1_86][^1_87][^1_88][^1_89]</span>

<div align="center">⁂</div>

[^1_1]: Ruderman-statistics.pdf

[^1_2]: https://cs.uwaterloo.ca/~mannr/cs886-w10/Ruderman-statistics.pdf

[^1_3]: https://pmc.ncbi.nlm.nih.gov/articles/PMC11048359/

[^1_4]: https://www.pnas.org/doi/10.1073/pnas.2108713118

[^1_5]: https://www.sciencedirect.com/science/article/pii/S0167865523002209

[^1_6]: https://www.nature.com/articles/s41598-023-33887-5

[^1_7]: https://pmc.ncbi.nlm.nih.gov/articles/PMC9861680/

[^1_8]: https://www.nature.com/articles/s41598-020-67661-8

[^1_9]: https://arxiv.org/html/2209.05321v4

[^1_10]: https://www.biorxiv.org/content/10.1101/2023.09.27.559646v2.full

[^1_11]: https://arxiv.org/pdf/2201.13312.pdf

[^1_12]: https://arxiv.org/pdf/1912.09028.pdf

[^1_13]: https://pmc.ncbi.nlm.nih.gov/articles/PMC3581899/

[^1_14]: https://www.pnas.org/doi/10.1073/pnas.1222618110

[^1_15]: https://arxiv.org/html/2509.17874v1

[^1_16]: https://journals.plos.org/ploscompbiol/article?id=10.1371%2Fjournal.pcbi.1009028

[^1_17]: https://arxiv.org/html/2407.17829v1

[^1_18]: https://pmc.ncbi.nlm.nih.gov/articles/PMC5464350/

[^1_19]: http://ijmcs.co.uk/details\&cid=3

[^1_20]: https://www.sciencedirect.com/science/article/pii/S0042698997000084

[^1_21]: https://arxiv.org/html/2406.08924v1

[^1_22]: https://oarjst.com/node/710

[^1_23]: https://www.sciendo.com/article/10.2478/fcds-2020-0009

[^1_24]: https://ietresearch.onlinelibrary.wiley.com/doi/10.1049/trit.2018.0017

[^1_25]: https://www.semanticscholar.org/paper/673a2a9c49a99b2e7a867176e4b417cff361a036

[^1_26]: https://onlinelibrary.wiley.com/doi/10.1111/j.1440-1843.2011.02020.x

[^1_27]: http://arxiv.org/pdf/1801.06302.pdf

[^1_28]: https://harvest.aps.org/v2/journals/articles/10.1103/PhysRevLett.73.814/fulltext

[^1_29]: https://pmc.ncbi.nlm.nih.gov/articles/PMC5144165/

[^1_30]: https://pmc.ncbi.nlm.nih.gov/articles/PMC3865819/

[^1_31]: https://people.csail.mit.edu/danielzoran/zoranweiss09.pdf

[^1_32]: https://par.nsf.gov/servlets/purl/10110617

[^1_33]: https://www.nature.com/articles/s41467-025-62086-1

[^1_34]: https://proceedings.neurips.cc/paper_files/paper/2022/file/5b4a459db23e6db9be2a128380953d96-Paper-Conference.pdf

[^1_35]: http://theis.io/publications/GerhardTheisBethge_Review.pdf

[^1_36]: https://www.tandfonline.com/doi/full/10.1080/18824889.2025.2567085

[^1_37]: https://www.nature.com/articles/s41598-025-22560-8

[^1_38]: https://ece.uwaterloo.ca/~z70wang/publications/SPM11.pdf

[^1_39]: https://www.sciencedirect.com/science/article/pii/S2590005623000309

[^1_40]: https://www.sciencedirect.com/science/article/pii/S277306462200010X

[^1_41]: https://www.sciencedirect.com/science/article/abs/pii/S014976341630598X

[^1_42]: https://arxiv.org/html/2507.16406v1

[^1_43]: https://arxiv.org/pdf/2509.12406.pdf

[^1_44]: https://arxiv.org/abs/2209.05321

[^1_45]: https://arxiv.org/html/2512.24385v2

[^1_46]: https://arxiv.org/html/2510.03598v1

[^1_47]: https://arxiv.org/html/2502.12600v1

[^1_48]: https://arxiv.org/html/2503.14012v1

[^1_49]: https://arxiv.org/html/2510.08449

[^1_50]: https://www.biorxiv.org/lookup/external-ref?access_num=10.7554%2FeLife.54347\&link_type=DOI

[^1_51]: https://arxiv.org/pdf/2510.23825.pdf

[^1_52]: https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0324504

[^1_53]: https://ar5iv.labs.arxiv.org/html/2305.15134

[^1_54]: https://www.semanticscholar.org/paper/f938236698917d3ad4808bccad7835c7bb8cc24e

[^1_55]: https://link.springer.com/10.3103/S0005105524700055

[^1_56]: https://qims.amegroups.com/article/view/125583/html

[^1_57]: https://www.mdpi.com/2227-7390/8/12/2260

[^1_58]: https://onlinelibrary.wiley.com/doi/10.1002/asi.20744

[^1_59]: https://www.acpjournals.org/doi/10.7326/0003-4819-118-9-199305010-00010

[^1_60]: https://www.semanticscholar.org/paper/c0d429effffe3df1ee8e9cceb7c19d54e18fe727

[^1_61]: http://www.tandfonline.com/doi/abs/10.1080/02331934.2015.1027530

[^1_62]: https://www.semanticscholar.org/paper/09ce3c253a0a2151070e12ee19361dd7734de376

[^1_63]: https://journals.sagepub.com/doi/10.1068/p2909ed

[^1_64]: https://pmc.ncbi.nlm.nih.gov/articles/PMC3554546/

[^1_65]: https://pmc.ncbi.nlm.nih.gov/articles/PMC2964243/

[^1_66]: https://pmc.ncbi.nlm.nih.gov/articles/PMC9133799/

[^1_67]: https://pmc.ncbi.nlm.nih.gov/articles/PMC6618700/

[^1_68]: https://pmc.ncbi.nlm.nih.gov/articles/PMC11888429/

[^1_69]: https://inc.ucsd.edu/mplab/users/marni/Igert/Ruderman_1994.pdf

[^1_70]: https://pubmed.ncbi.nlm.nih.gov/22031877/

[^1_71]: https://www.semanticscholar.org/paper/Statistics-of-Natural-Images:-Scaling-in-the-Woods-Ruderman-Bialek/4ea293ac6d42d09ccb9ffab7bd72dcf6102c3eab

[^1_72]: https://www.sciencedirect.com/science/article/abs/pii/S0952197623017864

[^1_73]: https://scholar.google.com/citations?user=6u1XiRQAAAAJ\&hl=en

[^1_74]: https://www.nature.com/articles/s41467-020-15630-0

[^1_75]: https://consensus.app/search/techniques-for-improving-the-generalization-of-dee/GCO-5inMT-egVPK9J3KS-g/

[^1_76]: https://openreview.net/pdf/68ead7ea1c298990cb74843e5bd8d87806b7cd4b.pdf

[^1_77]: https://ieeexplore.ieee.org/iel8/6287639/11323511/11316642.pdf

[^1_78]: https://arxiv.org/html/2404.13182v6

[^1_79]: https://arxiv.org/pdf/2506.04129.pdf

[^1_80]: https://arxiv.org/html/2508.13866v2

[^1_81]: https://ar5iv.labs.arxiv.org/html/1906.08246

[^1_82]: https://arxiv.org/pdf/2410.01086.pdf

[^1_83]: https://arxiv.org/html/2511.21715v1

[^1_84]: https://arxiv.org/html/2407.07816v1

[^1_85]: https://arxiv.org/html/2504.10201v2

[^1_86]: https://arxiv.org/pdf/2410.04038.pdf

[^1_87]: https://arxiv.org/pdf/2412.08893.pdf

[^1_88]: https://journals.plos.org/ploscompbiol/article?id=10.1371%2Fjournal.pcbi.1011667

[^1_89]: https://arxiv.org/pdf/2507.12590.pdf
