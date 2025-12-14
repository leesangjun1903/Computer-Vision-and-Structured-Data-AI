# SPADE : Sub-Image Anomaly Detection with Deep Pyramid Correspondences

---

## 1. 논문의 핵심 주장과 주요 기여

### 1.1 핵심 주장
**SPADE (Semantic Pyramid Anomaly Detection)** 방법은 **사전 학습된 깊은 신경망의 특징(Feature)과 k-NN(k-Nearest Neighbors) 기반 다중 해상도 피라미드 대응(Correspondence) 기법을 결합하여 이미지 내 이상 영역을 픽셀 수준에서 정확하게 검출하고 분할할 수 있다**는 것을 입증합니다.

### 1.2 주요 기여
1. **훈련 불필요(Training-Free) 방식**: 사전 학습된 ResNet 특징을 활용하므로 별도의 모델 훈련 단계가 거의 필요 없습니다.
2. **픽셀 수준의 정확한 이상 분할**: 일반 이미지만으로 학습하여 이상 영역의 정확한 위치를 파악합니다.
3. **다층 특징 피라미드 활용**: 서로 다른 해상도의 특징을 결합하여 지역적(Local) 세부 정보와 전역적(Global) 맥락을 모두 고려합니다.
4. **최첨단 성능**: MVTec AD와 ShanghaiTech Campus 데이터셋에서 이전 방법들을 크게 상회하는 성능을 달성합니다.

---

## 2. 해결하고자 하는 문제

### 2.1 배경 및 동기
이상 탐지(Anomaly Detection)는 제조업의 품질 관리, 보안 감시, 의료 영상 분석 등 다양한 분야에서 핵심적인 역할을 수행합니다. 그러나 이상 탐지는 다음과 같은 근본적인 어려움을 갖고 있습니다:

- **예측 불가능성**: 이상의 형태가 훈련 단계에서 관찰되지 않으므로 다양한 형태의 이상을 모두 대처하기 어렵습니다.
- **제한된 이상 샘플**: 정상 데이터는 충분하지만, 이상 데이터는 매우 적습니다(불균형 문제).
- **위치 파악의 어려움**: 단순히 "이 이미지가 이상이다"라는 판정뿐 아니라 "어디가 이상인가?"를 정확하게 파악해야 합니다.

### 2.2 기존 방법의 한계
논문이 지적하는 기존 방법들의 한계:

| 방법 | 한계 |
|-----|------|
| **재구성 기반(Reconstruction-based)** | 손실 함수 선택에 민감하며, 정상 데이터도 잘 재구성할 수 있어 이상을 구분하기 어려움 |
| **분포 기반(Distribution-based)** | 고차원 데이터에서 확률밀도함수 추정이 어렵고 샘플 복잡도가 높음 |
| **k-NN 이미지 수준 분류** | 이미지 전체에 대한 정상/이상 판정은 가능하지만, 어느 부분이 이상인지 분할하지 못함 |
| **자체 감독(Self-supervised) 방법** | 작은 데이터셋에서 학습한 특징이 사전 학습된 특징(ImageNet)보다 성능이 낮음 |

### 2.3 SPADE의 해결책
SPADE는 다음 문제들을 해결합니다:

1. **정렬 기반 접근**: 이상 이미지와 정상 이미지들 간의 픽셀 수준 대응을 찾아 이상 위치를 파악합니다.
2. **다중 정상 이미지 활용**: 단일 정상 이미지와의 비교 대신 K개의 최근접 정상 이미지들을 모두 고려하여 강건성을 높입니다.
3. **특징 피라미드의 활용**: 다양한 해상도의 특징을 결합하여 미세한 이상도 탐지할 수 있습니다.

---

## 3. 제안하는 방법 및 수식

### 3.1 방법의 3단계 구조

#### **1단계: 특징 추출(Feature Extraction)**
$$f_i = F(x_i) \quad \text{(식 1)}$$

여기서:
- $F$: 사전 학습된 ResNet 특징 추출기 (ImageNet 학습)
- $x_i$: 입력 이미지
- $f_i$: 전역 풀링 후 추출된 특징 벡터 (2048차원)

**설명**: ResNet의 마지막 합성곱 계층을 전역 평균 풀링하여 이미지 수준의 특징을 얻습니다.

#### **2단계: k-최근접 이웃 검색(k-NN Retrieval)**
$$d(y) = \frac{1}{K} \sum_{f \in N_K(f_y)} \|f - f_y\|_2 \quad \text{(식 2)}$$

여기서:
- $y$: 검사 대상 이미지
- $N_K(f_y)$: 테스트 이미지의 특징 $f_y$에서 K개의 최근접 정상 이미지 특징
- $d(y)$: 이미지 수준의 이상 점수

**설명**: 유클리드 거리를 기반으로 훈련 세트에서 K개의 가장 유사한 정상 이미지를 찾습니다. K개 이미지까지의 평균 거리가 임계값 $\tau$를 초과하면 이미지를 이상으로 분류합니다.

#### **3단계: 피라미드 대응을 통한 픽셀 수준 이상 탐지(Sub-image Detection)**
$$d(y, p) = \frac{1}{\kappa} \sum_{f \in N_\kappa(F(y,p))} \|f - F(y,p)\|_2 \quad \text{(식 3)}$$

여기서:
- $p$: 이미지 내 픽셀 위치
- $F(y,p)$: 위치 $p$에서 추출된 특징
- $G = \{F(x_1,p)|p \in P\} \cup \{F(x_2,p)|p \in P\} \cup \cdots \cup \{F(x_K,p)|p \in P\}$: K개의 최근접 정상 이미지들의 모든 픽셀 위치에서의 특징으로 구성된 갤러리
- $N_\kappa(F(y,p))$: 갤러리에서 $F(y,p)$의 $\kappa$개 최근접 특징
- 픽셀은 $d(y,p) > \theta$ (임계값)일 때 이상으로 판정됩니다.

**설명**: 각 픽셀 위치에서의 특징과 K개의 정상 이미지에서 같은 위치의 특징들 간의 최소 거리를 계산합니다. 거리가 크다는 것은 정상 이미지에서 대응하는 특징을 찾을 수 없다는 의미이므로 이상입니다.

### 3.2 특징 피라미드 매칭(Feature Pyramid Matching)

SPADE의 핵심 혁신은 **다층 특징 피라미드 활용**입니다:

$$\text{Combined Feature} = \text{Concat}[F_{\text{block1}}(56 \times 56), F_{\text{block2}}(28 \times 28), F_{\text{block3}}(14 \times 14)]$$

**장점**:
- **얕은 층** (56×56): 높은 공간 해상도의 지역 정보 제공
- **중간 층** (28×28): 지역 정보와 의미적 정보의 균형
- **깊은 층** (14×14): 전역 맥락과 의미론적 정보 제공

이 다층 접근으로 이미지 정렬은 **명시적 기하학적 정렬 없이** 특징 대응을 통해 수행됩니다.

### 3.3 계산 복잡도 최적화

이미지 수준의 특징은 2048차원으로 낮은 복잡도의 k-NN 검색이 가능합니다. 픽셀 수준의 k-NN 검색은 비용이 높지만, 다음 전략으로 제어합니다:

1. 대부분의 이미지가 정상이므로 픽셀 수준 검색은 소수의 이상 이미지에만 적용
2. 픽셀 수준 검색은 K개의 최근접 정상 이미지로만 제한
3. 실험에서 K=50(MVTec), K=1(STC) 사용

---

## 4. 모델 구조

### 4.1 전체 아키텍처

```
입력 이미지
    ↓
[1단계] 사전학습 ResNet50 특징 추출
    ├─ 전역 풀링 → 2048차원 특징 (이미지 수준)
    └─ 중간 활성화 → 3개 해상도 특징 (픽셀 수준)
    ↓
[2단계] k-NN 이미지 검색
    ├─ 훈련 세트에서 K개 최근접 정상 이미지 검색
    ├─ 거리 임계값 τ와 비교
    ├─ 이미지 정상/이상 판정
    └─ 이상일 경우만 다음 단계 진행
    ↓
[3단계] 피라미드 대응 기반 픽셀 분할
    ├─ K개 정상 이미지의 특징 갤러리 구성
    ├─ 각 픽셀의 대응 탐색 (κ-NN)
    ├─ 픽셀 수준 이상 점수 계산
    └─ 가우시안 필터로 평활화
    ↓
이상 분할 맵 (Anomaly Segmentation Mask)
```

### 4.2 핵심 컴포넌트

| 컴포넌트 | 역할 | 특징 |
|---------|------|------|
| **사전학습 ResNet50** | 특징 추출 | ImageNet 학습, 고품질 특징 |
| **다층 특징** | 다해상도 정보 제공 | Block1, 2, 3의 활성화 결합 |
| **k-NN 갤러리** | 정상 패턴 저장 | 훈련 단계의 특징 저장소 |
| **거리 메트릭** | 유클리드 거리 | 계산 효율성과 해석성 우수 |
| **후처리** | 평활화 | σ=4 가우시안 필터 |

---

## 5. 성능 향상 및 한계

### 5.1 성능 향상 결과

#### **MVTec AD 데이터셋 (이미지 수준 검출)**

| 방법 | Geom | GANomaly | AEL2 | ITAE | **SPADE** |
|------|------|---------|------|------|----------|
| 평균 ROCAUC (%) | 67.2 | 76.2 | 75.4 | 83.9 | **85.5** |

**개선율**: ITAE 대비 +1.6 p.p.

#### **MVTec AD 데이터셋 (픽셀 수준 ROCAUC)**

SPADE는 다음 방법들을 능가했습니다:
- AESSIM: 87% → **96.0%** (+9.0 p.p.)
- AEL2: 82% → **96.0%** (+14.0 p.p.)
- Student-Teacher (Bergmann 외, 2019): 85.7% → **96.0%** (+10.3 p.p.)

**특히 강점**:
- Carpet: 97.5%, Grid: 93.7%, Leather: 97.6%, Hazelnut: 99.1%, Screw: 98.9%, Zipper: 96.5%

#### **MVTec AD 데이터셋 (픽셀 수준 PRO 메트릭)**

| 방법 | Student | 1-NN | OC-SVM | ℓ2-AE | VAE | **SPADE** |
|------|---------|------|--------|-------|-----|----------|
| PRO Score (%) | 85.7 | 64 | 47.9 | 79 | 63.9 | **91.7** |

PRO 메트릭은 큰 이상에 편향된 ROCAUC의 한계를 보완합니다.

#### **ShanghaiTech Campus 데이터셋**

| 메트릭 | AEL2 | AESSIM | CAVGA-Ru | **SPADE** |
|--------|------|--------|----------|----------|
| 픽셀 ROCAUC (%) | 74 | 76 | 85 | **89.9** |

**개선율**: CAVGA-Ru 대비 +4.9 p.p.

### 5.2 절제 연구(Ablation Study) - 피라미드 해상도의 영향

피라미드 수준별 성능 (MVTec AD, PRO 메트릭):

| 사용 층 | (14×14) | (28×28) | (56×56) | 모두 결합 |
|--------|---------|---------|---------|----------|
| 평균 PRO (%) | 89.38 | 89.60 | 87.74 | **91.7** |

**분석**:
- **14×14만 사용**: 저해상도로 세부 정보 손실
- **56×56만 사용**: 높은 해상도지만 맥락 정보 부족
- **모두 결합**: 최적 성능 달성 (+2.1 p.p. 향상)

### 5.3 k-NN 검색의 중요성

| 설정 | Grid 클래스 | 평균 |
|------|----------|------|
| 무작위 K개 선택 | 73.2% | 89.2% |
| **k-NN으로 선택** | **86.3%** | **91.4%** |

**개선율**: Grid 클래스 +13.1 p.p., 평균 +0.2 p.p.

### 5.4 한계와 제약사항

#### **1. 계산 복잡도**
- k-NN 기반이므로 대규모 훈련 세트에서 느릴 수 있음
- 픽셀 수준 k-NN은 이미지 수준보다 훨씬 비용이 높음
- **완화책**: 
  - KDTree, LSH 등의 효율적 k-NN 구현 가능
  - 대부분의 이미지는 이미지 수준에서 정상으로 판정되어 픽셀 검색 불필요

#### **2. 특징 의존성**
- 성능이 사전학습된 ResNet50의 품질에 크게 의존
- ImageNet과 매우 다른 도메인에서는 특징이 최적이 아닐 수 있음
- **그러나**: 실험 결과 ImageNet과 매우 다른 MVTec AD에서도 강한 성능 달성

#### **3. 임계값 선택**
- 이미지 수준 임계값 $\tau$, 픽셀 수준 임계값 $\theta$를 데이터 기반으로 설정 필요
- 원논문에서는 구체적인 임계값 설정 방법이 명시적으로 제시되지 않음

#### **4. 다양성 제한**
- 정상 훈련 세트의 다양성이 부족하면 성능 저하
- 매우 새로운 형태의 이상은 탐지 어려울 수 있음

---

## 6. 모델의 일반화 성능 향상 가능성

### 6.1 강점과 일반화 우점

#### **1. 도메인 일반화**
- **사전학습된 특징 사용**: ImageNet 학습으로 다양한 도메인에 일반화
- **비훈련 특성**: 도메인별 재훈련 불필요, 즉시 배포 가능
- **증거**: 산업용(MVTec) 데이터와 감시(STC) 데이터 모두에서 최첨단 성능

#### **2. 작은 데이터셋 처리**
- 정상 이미지만 필요 (이상 이미지 불필요)
- 방법이 간단하여 과적합 위험 적음
- 표 7에서 Grid 클래스 실험: 10개의 k-NN 선택으로 충분한 성능

#### **3. 다양한 이미지 크기 대응**
- 특징 추출만 하면 다양한 해상도 처리 가능
- MVTec: 256×256, STC: 다양한 크기 모두 처리

### 6.2 일반화 성능 향상의 제약사항

#### **1. 특징 병목(Feature Bottleneck)**
ResNet50의 특징은 고정되어 있으므로:
- 매우 특이한 도메인에서는 제한적
- 예: 의료 영상, 적외선 영상 등 자연 이미지와 매우 다른 도메인

#### **2. K값의 민감성**
- MVTec: K=50 (충분한 훈련 데이터)
- STC: K=1 (제한된 데이터와 계산 제약)
- 최적의 K값은 데이터셋별로 다름

#### **3. 정상 데이터의 대표성**
- 훈련 세트가 정상 이미지의 다양성을 충분히 포함해야 함
- 예: 제품 촬영 각도, 조명 조건 등의 변화

### 6.3 향상 가능성 분석

#### **A. 특징 미세 조정(Feature Fine-tuning)**
**제안**: 사전학습된 ResNet50을 작은 학습률로 미세 조정
- **예상 효과**: 도메인 특화 특징으로 성능 향상
- **비용**: 약간의 훈련 시간 추가 필요
- **원논문의 언급**: "미래 작업에서 특징 미세 조정을 통한 추가 개선 예상"

#### **B. 앙상블 방식(Ensemble Approach)**
**제안**: 여러 백본(ResNet18, ResNet50, DenseNet 등)의 특징 결합
- **예상 효과**: 더욱 강건한 특징 표현
- **참고**: PatchCore에서 앙상블 사용으로 추가 성능 향상 보고

#### **C. 적응형 임계값(Adaptive Thresholding)**
**제안**: 각 클래스/도메인별로 임계값 동적 조정
- **예상 효과**: 다양한 도메인에서 일관된 성능
- **방법**: 검증 세트를 활용한 임계값 최적화

#### **D. 메모리 뱅크 개선**
**제안**: 코어셋(Coreset) 선택 개선
- **현재**: k-NN 갤러리가 모든 훈련 특징 포함
- **개선 방향**: 대표성 있는 부분집합만 선택 (PatchCore의 접근)
- **효과**: 계산 속도 향상 + 메모리 절감

### 6.4 최신 연구와의 비교를 통한 일반화 성능 평가

2020년 이후의 관련 최신 방법들과 비교:

| 방법 | 발표 | 방식 | MVTec ROCAUC | 특징 |
|------|------|------|-------------|------|
| **SPADE** | 2020 | k-NN + 피라미드 | 96.0% (PRO) | 훈련 불필요 |
| **PaDiM** | 2020 | 가우시안 분포 | 95.1% (ROCAUC) | 다층 특징의 분포 모델링 |
| **PatchCore** | 2021 | 메모리뱅크 + 코어셋 | 99.6% (ROCAUC) | 최적 메모리 선택 |
| **Reverse Distillation** | 2022 | 교사-학생 증류 | 94.2% (ROCAUC) | 훈련 필요 |
| **SimpleNet** | 2023 | 간단한 네트워크 | 96.0%+ (ROCAUC) | 초고속 추론 |
| **FR-PatchCore** | 2024 | PatchCore 개선 | 98.5%+ (ROCAUC) | 향상된 일반화 |

**분석**:
- **PatchCore의 등장(2021)**: 메모리 뱅크 최적화로 99% 초과 달성
- **SPADE의 위치**: 훈련 불필요하면서도 96% 수준의 강력한 성능
- **최신 경향**: 
  - 메모리 최적화 (PatchCore → FR-PatchCore)
  - 초고속 추론 (SimpleNet)
  - 일반화 성능 향상 (FR-PatchCore)

---

## 7. 논문이 앞으로의 연구에 미치는 영향

### 7.1 학문적 영향

#### **1. k-NN 기반 이상 탐지의 부활**
**영향**: 
- 기존에 효과적이지 않다고 여겨진 k-NN이 사전학습된 특징과 결합하면 매우 강력함을 증명
- 간단한 방법이 복잡한 딥러닝 방법(GAN, VAE)을 능가 가능함을 보여줌

**이후 연구들**:
- **PatchCore (2021)**: k-NN 갤러리의 코어셋 최적화
- **PaDiM (2020)**: k-NN과 가우시안 분포 결합
- **AnomalousPatchCore (2024)**: 특징 추출기 미세 조정 추가

#### **2. 특징 피라미드의 중요성 강조**
**영향**: 
- 다층 특징의 효과적인 결합 방식 제시
- 얕은 층(고해상도)과 깊은 층(고맥락)의 균형 중요성 입증

**이후 적용**:
- 거의 모든 최신 방법들이 다층 특징 활용
- Feature Pyramid Network(FPN) 개념의 이상 탐지 분야 확산

#### **3. 훈련 불필요(Training-Free) 이상 탐지의 가능성**
**영향**:
- 사전학습 특징만으로도 충분히 강력한 성능 가능
- 새로운 도메인에 즉시 적용 가능한 방법론 제시

**이후 경향**:
- Zero-shot, Few-shot 이상 탐지 연구 활성화
- 기초 모델(Foundation Models) 활용 연구 증가

### 7.2 실무적 영향

#### **1. 산업용 품질 검사 시스템**
**현황**:
- SPADE는 훈련 불필요하므로 새로운 제품에 즉시 배포 가능
- 실시간 처리 가능한 속도 (MVTec 기준 약 200ms/이미지)

**확산**:
- 제조업의 온라인 검사 시스템 도입 증가
- PatchCore, SimpleNet 등으로 더욱 정교한 시스템 개발

#### **2. 설명 가능성(Explainability) 강조**
**SPADE의 기여**:
- k-NN 기반이므로 "어떤 정상 이미지와 다른가?"를 시각적으로 설명 가능
- 신뢰성 있는 시스템 구축에 유리

**결과**:
- SHAP, LIME 등 설명 가능성 도구와의 결합 연구
- 의료, 안보 등 투명성이 중요한 분야의 도입 확대

#### **3. 빠른 배포와 낮은 비용**
**SPADE의 장점**:
- 별도의 GPU 훈련 불필요
- 사전학습 모델만으로 충분
- 개발 시간 단축

**산업 적용**:
- 중소 제조업체도 적용 가능한 저비용 솔루션
- 클라우드 기반 서비스로의 확산

### 7.3 방법론적 영향

#### **1. 대응(Correspondence) 기반 접근의 확산**
**SPADE의 기여**:
- 이미지 정렬 대신 특징 수준의 대응으로 강건한 이상 탐지 가능

**이후 연구**:
- 광학 흐름(Optical Flow) 기반 이상 탐지
- 동적 시간 워핑(DTW) 기반 시계열 이상 탐지

#### **2. 다중 스케일 표현 학습**
**SPADE의 시사**:
- 단일 해상도보다 다중 해상도 표현의 우월성

**적용 분야**:
- 의료 영상 분석
- 원격 감지(Remote Sensing) 이상 탐지
- 비디오 이상 탐지

---

## 8. 앞으로의 연구 시 고려할 점

### 8.1 도메인 적응 및 일반화

#### **1. 도메인 시프트 문제**
**과제**: 훈련과 테스트 도메인이 다를 때 성능 저하

**해결 방향**:
- **특징 적응**: 사전학습 모델의 선택적 미세 조정
- **다중 도메인 학습**: 여러 도메인의 정상 이미지로 일반화된 특징 학습
- **도메인 정규화**: 입력 정규화, 스타일 정규화 등 도메인 불변성 강화

**참고**: FR-PatchCore(2024)에서 일반화 성능 향상에 주력

#### **2. 소수 샘플(Few-Shot) 학습**
**과제**: 훈련 정상 이미지가 매우 적을 때의 성능

**해결 방향**:
- **메타 학습**: 다양한 도메인의 데이터로 메타 특징 학습
- **데이터 증강**: 정상 이미지의 의미론적 증강 (회전, 색상 변환 등)
- **프로토타입 학습**: 소수 샘플로부터 클래스 프로토타입 추출

### 8.2 계산 효율성

#### **1. 빠른 k-NN 검색**
**과제**: 대규모 데이터셋에서의 k-NN 탐색 속도

**해결 방향**:
- **근사 k-NN**: LSH(Locality-Sensitive Hashing), HNSW(Hierarchical Navigable Small World)
- **코어셋 최적화**: PatchCore의 메모리 뱅크 최적화 방식 발전
- **계층적 검색**: 주요 특징과 부분 특징의 계층적 k-NN

#### **2. 메모리 효율**
**과제**: 갤러리 특징 저장으로 인한 메모리 사용

**해결 방향**:
- **특징 압축**: 양자화, 해싱을 통한 특징 표현 축소
- **선택적 저장**: 대표성 있는 특징만 선택적 저장
- **온라인 학습**: 스트리밍 데이터에 대한 적응적 갤러리 업데이트

### 8.3 특징 학습 개선

#### **1. 도메인 특화 특징**
**과제**: ImageNet 특징이 최적이 아닐 수 있음

**해결 방향**:
- **자기 감독 학습**: 회전 예측, 색상 예측 등으로 도메인 내 특징 학습
- **대조 학습**: 정상 이미지의 다양한 변환으로부터 특징 대조
- **다중 모달 특징**: 텍스트, 시간 정보 등과 결합

#### **2. 이상 특화 특징**
**과제**: 정상 데이터만으로는 이상을 최적으로 분리하기 어려움

**해결 방향**:
- **합성 이상 생성**: GAN으로 다양한 이상 패턴 생성 후 학습
- **대조 학습**: 정상-이상 쌍의 대조 학습
- **약한 감독**: 제한적 이상 샘플로부터의 학습

### 8.4 멀티 모달 및 고급 기능

#### **1. 비디오/시계열 이상 탐지**
**확장 가능성**: 
- 연속 프레임의 특징 시퀀스에 적용
- 시공간(Spatio-temporal) 피라미드 구성

#### **2. 설명 가능한 이상 탐지**
**방향**:
- SHAP, LIME 등과의 결합
- 어떤 특징이 이상을 유발했는지 해석
- 사람-AI 협력 인터페이스

#### **3. 이상 심각도 평가**
**현황**: 현재는 이진 분류(정상/이상)

**개선**:
- 다단계 심각도 분류 (경미/중간/심각)
- 결함의 면적, 깊이 등 정량적 평가

### 8.5 벤치마크 및 평가

#### **1. 새로운 평가 메트릭**
**제안**:
- **일반화 메트릭**: 도메인 간 전이 성능 평가
- **효율성 메트릭**: 속도, 메모리, 에너지 효율
- **신뢰도 메트릭**: 신뢰 구간, 불확실성 정량화

#### **2. 표준화된 벤치마크**
**현황**: MVTec AD (산업), STC (감시)

**필요성**:
- 의료 이미지, 위성 이미지 등 다양한 도메인 벤치마크
- 도메인 시프트에 대한 표준 평가 프로토콜

---

## 9. 2020년 이후 관련 최신 연구 비교 분석

### 9.1 주요 최신 방법들의 특징 비교

#### **표 1: 최신 이상 탐지 방법 비교 (2020-2024)**

| 방법 | 발표 | 핵심 기술 | 훈련 필요 | MVTec ROCAUC | 연산 속도 | 일반화 |
|------|------|---------|---------|-------------|---------|------|
| **SPADE** | 2020 | k-NN + 피라미드 | ✗ | 96.0% (PRO 91.7%) | 빠름 | 중상 |
| **PaDiM** | 2020 | 가우시안 분포 | ✗ | 95.1% | 빠름 | 중상 |
| **PatchCore** | 2021 | 메모리뱅크 + 코어셋 | ✗ | **99.6%** | 중간 | 상 |
| **Reverse Distillation** | 2022 | 교사-학생 증류 | ✓ | 94.2% | 느림 | 중 |
| **SimpleNet** | 2023 | 간단한 네트워크 | ✓ | 96.0%+ | **가장빠름** | 중상 |
| **FR-PatchCore** | 2024 | PatchCore + 특징적응 | ✗ | **98.5%+** | 중간 | **상** |

### 9.2 방법별 상세 분석

#### **A. PaDiM (Patch Distribution Modeling, 2020)**

**핵심 아이디어**:
- 각 패치 위치에서의 특징을 다변량 정규분포로 모델링
- 마할라노비스 거리(Mahalanobis Distance) 사용

**장점**:
- 특징의 상관관계 고려
- 고정적 계산 시간 O(1)
- SPADE와 유사한 시점(2020)에 발표되며 병렬적 발전

**차이점 (SPADE 대비)**:
- SPADE: k-NN 거리 (비모수)
- PaDiM: 가우시안 분포 (모수)

**성능**: MVTec ROCAUC 95.1% (SPADE: 96.0%)

#### **B. PatchCore (Towards Total Recall, 2021)**

**핵심 아이디어**:
- 메모리 뱅크의 코어셋(Coreset) 최적화
- 최소-최대 시설 위치(Minimax Facility Location) 문제로 공식화

**혁신**:
- SPADE와 PaDiM의 모든 특징 사용 vs. PatchCore는 1%만 선택
- 메모리 사용 대폭 절감, 속도 향상

**성능**: MVTec ROCAUC **99.6%** (가장 높음)

**한계**:
- 코어셋 선택 알고리즘이 복잡
- 메모리 vs 성능의 트레이드오프 필요

**영향**: 이후 대부분의 논문이 PatchCore를 기준으로 비교

#### **C. Reverse Distillation (2022)**

**핵심 아이디어**:
- 교사 인코더-학생 디코더의 이상 증류
- 정상 패턴 복원 실패를 이상 신호로 사용

**특징**:
- 유일하게 훈련 기반 방법
- 이방향(Reverse) 증류로 이상에 더 민감

**장점**:
- 훈련 가능한 매개변수로 도메인 적응
- 직관적인 개념

**단점**:
- 훈련 시간 필요 (수 시간)
- SPADE, PaDiM보다 약간 낮은 성능

#### **D. SimpleNet (2023)**

**핵심 아이디어**:
- 매우 간단한 4-모듈 구조
- 고효율성과 정확성의 최적화

**특징**:
- 가장 빠른 추론 속도 (500+ FPS)
- MVTec AD에서 96% 이상 성능
- 실시간 처리 가능

**혁신**:
- 복잡성과 성능의 새로운 균형점 제시
- 엣지 디바이스 배포 가능

#### **E. FR-PatchCore (2024)**

**핵심 아이디어**:
- PatchCore에 특징 미세 조정 추가
- 동적 임계값 결정 방법 제시

**개선사항**:
- MVTec ROCAUC 98.5%+ (PatchCore 99.6%과 경쟁)
- **도메인 일반화 성능 대폭 향상** (다른 도메인으로 전이)
- 임계값 선택의 과학적 방법 제시

**의의**:
- PatchCore 이후 가장 중요한 개선
- 일반화 성능의 중요성 강조

### 9.3 발전 경로 분석

```
2020년: 기반 마련
├─ SPADE: k-NN + 피라미드 (이 논문)
└─ PaDiM: 가우시안 분포 + 다층 특징

2021년: 메모리 최적화
└─ PatchCore: 코어셋 기반 메모리뱅크

2022년: 훈련 기반 접근
└─ Reverse Distillation: 교사-학생 증류

2023년: 효율성 극대화
└─ SimpleNet: 초고속 추론

2024년: 일반화 성능 향상
└─ FR-PatchCore: 특징 적응 + 임계값 최적화
```

### 9.4 각 방법의 적용 상황별 추천

| 상황 | 추천 방법 | 이유 |
|------|---------|------|
| **높은 정확도 필요** | PatchCore | 최고 성능 (99.6%) |
| **빠른 배포** | SPADE/PaDiM | 훈련 불필요, 즉시 적용 |
| **도메인 적응** | FR-PatchCore | 일반화 성능 우수 |
| **실시간 처리** | SimpleNet | 가장 빠른 속도 |
| **설명 가능성** | SPADE | k-NN 기반으로 명확한 해석 |
| **메모리 제약** | SimpleNet | 가장 간단한 구조 |
| **유연한 훈련** | Reverse Distillation | 도메인 특화 학습 가능 |

### 9.5 최신 연구의 공통 트렌드

#### **1. 특징 품질의 중요성**
- 모든 최신 방법이 사전학습 특징(ImageNet ResNet) 활용
- 특징 공학(Feature Engineering)의 중요성 입증

#### **2. 메모리 효율성**
- SPADE: 모든 특징 저장
- PatchCore: 1% 코어셋 (혁신)
- SimpleNet: 최소 메모리 구조

#### **3. 훈련 불필요 vs. 훈련 가능의 균형**
- 훈련 불필요: SPADE, PaDiM, PatchCore (빠른 배포)
- 훈련 가능: Reverse Distillation, FR-PatchCore (도메인 적응)

#### **4. 일반화 성능의 중시**
- 초기(2020-2021): 높은 정확도 추구
- 최신(2023-2024): 도메인 일반화 강조
- FR-PatchCore: "일반화" 성능을 주제로

#### **5. 간소화의 추세**
- 초기: 복잡한 GAN, VAE 기반
- 최신: 단순한 k-NN, 특징 비교로 귀환
- "Occam의 면도날": 단순한 방법이 최고 성능

---

## 10. 결론 및 시사점

### 10.1 SPADE의 학문적 기여

SPADE는 이상 탐지 분야에서 다음을 입증했습니다:

1. **사전학습 특징의 강력함**: ImageNet 특징이 매우 이질적인 도메인(MVTec, STC)에서도 충분히 강력
2. **단순한 방법의 우수성**: 복잡한 생성 모델보다 k-NN이 더 효과적 가능
3. **다층 특징의 필요성**: 다해상도 특징의 결합이 최적의 성능 제공
4. **픽셀 수준 이상 분할의 가능성**: 훈련 없이도 정확한 이상 위치 파악 가능

### 10.2 실무적 영향

- **빠른 배포**: 훈련 불필요하여 새로운 응용에 즉시 적용
- **낮은 비용**: 사전학습 모델만으로 충분
- **투명성**: k-NN 기반으로 설명 가능
- **확장성**: 다양한 도메인에 일반화 가능

### 10.3 2020년 이후의 발전 평가

SPADE 이후 4년의 발전:

| 측면 | SPADE (2020) | 최신 (2024) | 진전 |
|------|-------------|-----------|------|
| 정확도 | 96.0% | 99.6% | +3.6% |
| 속도 | 중간 | 초고속 | 크게 개선 |
| 일반화 | 중상 | 우수 | 크게 개선 |
| 메모리 | 다량 | 최소 | 크게 개선 |

### 10.4 향후 연구의 방향

1. **도메인 적응 강화**: 다양한 도메인에 최적화된 특징 학습
2. **효율성 극대화**: PatchCore 이상의 메모리/속도 최적화
3. **자동 임계값 결정**: 데이터 기반의 동적 임계값 설정
4. **멀티 모달 확장**: 이미지 외 센서 데이터 포함
5. **실시간 온라인 학습**: 새로운 정상 패턴의 동적 적응

### 10.5 최종 평가

**SPADE (2020)**는:
- 이상 탐지 분야의 중추적 논문 (1000+ 인용)
- 이후 4년의 연구 방향 설정
- 훈련 불필요한 이상 탐지의 가능성 증명
- 현재도 여전히 강력한 기준 방법

PatchCore (2021)가 더 높은 성능을 달성했지만, SPADE는 **단순함과 효율성** 측면에서 여전히 우수하며, 많은 실무 응용에서 선택되는 방법입니다.

---

## 참고문헌 (논문 내 참고문헌 주요 항목)

### SPADE 관련
- Cohen, N., & Hoshen, Y. (2020). Sub-Image Anomaly Detection with Deep Pyramid Correspondences. arXiv:2005.02357v3.

### 최신 비교 방법들
- Defard, T., et al. (2020). PaDiM: a Patch Distribution Modeling Framework for Anomaly Detection and Localization.
- Roth, K., et al. (2021). Towards Total Recall in Industrial Anomaly Detection (PatchCore).
- Deng, H., & Li, X. (2022). Anomaly Detection via Reverse Distillation From One-Class Embedding.
- Liu, Z., et al. (2023). SimpleNet: A Simple Network for Image Anomaly Detection and Localization.
- Jiménez-Cabello, A. A., et al. (2024). FR-PatchCore: An Industrial Anomaly Detection Method for Improving Generalization.

### 벤치마크 데이터셋
- Bergmann, P., et al. (2019). MVTec AD – A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection.
- Luo, W., et al. (2017). Revisiting a Sparse Coding Based Anomaly Detection in a Stacked Framework (STC dataset).

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/02cf3bf6-c072-43c2-84c3-e4285ee47515/2005.02357v3.pdf)
[2](https://arxiv.org/abs/2011.08785)
[3](http://arxiv.org/pdf/2106.08265.pdf)
[4](https://openaccess.thecvf.com/content/CVPR2022/papers/Deng_Anomaly_Detection_via_Reverse_Distillation_From_One-Class_Embedding_CVPR_2022_paper.pdf)
[5](https://openaccess.thecvf.com/content/CVPR2023/papers/Liu_SimpleNet_A_Simple_Network_for_Image_Anomaly_Detection_and_Localization_CVPR_2023_paper.pdf)
[6](https://pmc.ncbi.nlm.nih.gov/articles/PMC10934034/)
[7](https://www.nature.com/articles/s41467-024-54043-1)
[8](https://academic.oup.com/sleep/article/doi/10.1093/sleep/zsae184/7729858)
[9](http://pubs.rsna.org/doi/10.1148/ryai.240067)
[10](http://pubs.rsna.org/doi/10.1148/radiol.233197)
[11](https://academic.oup.com/europace/article/doi/10.1093/europace/euae102.583/7681449)
[12](http://pubs.rsna.org/doi/10.1148/ryai.240017)
[13](https://bmjopen.bmj.com/lookup/doi/10.1136/bmjopen-2023-077366)
[14](http://pubs.rsna.org/doi/10.1148/radiol.231938)
[15](https://link.springer.com/10.1007/s11430-024-1396-x)
[16](https://onlinelibrary.wiley.com/doi/10.1002/spy2.347)
[17](https://arxiv.org/pdf/2201.07284.pdf)
[18](http://arxiv.org/pdf/2106.05410v2.pdf)
[19](https://arxiv.org/html/2503.13195v1)
[20](https://arxiv.org/pdf/1901.03407.pdf)
[21](https://arxiv.org/abs/2409.08521)
[22](https://arxiv.org/pdf/2502.18601.pdf)
[23](https://arxiv.org/pdf/2305.16114.pdf)
[24](http://arxiv.org/pdf/2306.12703.pdf)
[25](https://www.sciencedirect.com/science/article/abs/pii/S0262885623001919)
[26](https://pmc.ncbi.nlm.nih.gov/articles/PMC4106157/)
[27](https://www.sciencedirect.com/science/article/abs/pii/S0098135423004301)
[28](https://openaccess.thecvf.com/content/CVPR2021/papers/Wang_Glancing_at_the_Patch_Anomaly_Localization_With_Global_and_Local_CVPR_2021_paper.pdf)
[29](https://www.atlantis-press.com/article/25897526.pdf)
[30](https://www.sciencedirect.com/science/article/pii/S2666827023000233)
[31](https://openaccess.thecvf.com/content/WACV2022/papers/Tsai_Multi-Scale_Patch-Based_Representation_Learning_for_Image_Anomaly_Detection_and_Segmentation_WACV_2022_paper.pdf)
[32](https://openaccess.thecvf.com/content/WACV2024W/ASTAD/papers/Nizan_K-NNN_Nearest_Neighbors_of_Neighbors_for_Anomaly_Detection_WACVW_2024_paper.pdf)
[33](https://arxiv.org/abs/2211.05244)
[34](https://arxiv.org/html/2505.22762v1)
[35](https://pubmed.ncbi.nlm.nih.gov/25105164/)
[36](https://arxiv.org/html/2507.01924v1)
[37](https://arxiv.org/html/2509.18354v1)
[38](https://arxiv.org/pdf/2002.10445.pdf)
[39](https://arxiv.org/html/2511.11165v2)
[40](https://arxiv.org/html/2510.07927v1)
[41](https://arxiv.org/html/2211.05244v3)
[42](https://dualitytech.com/blog/anomaly-detection-k-nearest-neighbors/)
[43](https://dl.acm.org/doi/10.1145/3691338)
[44](https://dl.acm.org/doi/10.1145/3465631.3465927)
[45](https://arxiv.org/abs/2206.05876)
[46](http://biorxiv.org/lookup/doi/10.1101/2022.08.15.504032)
[47](https://link.springer.com/10.1007/978-3-031-04826-5_1)
[48](https://openaccess.cms-conferences.org/publications/book/978-1-958651-43-8/article/978-1-958651-43-8_12)
[49](https://revistaft.com.br/the-role-of-ai-in-enhancing-identity-and-access-management-systems/)
[50](https://www.semanticscholar.org/paper/f6e8faf8461309fd4924568d142921c5dd06c86b)
[51](https://www.semanticscholar.org/paper/878cc4086f06c0e803a034d142d38d2c3f424be5)
[52](https://ijareeie.com/upload/2022/april/4_Machine.pdf)
[53](https://journals.sagepub.com/doi/10.1177/10935266221086454)
[54](https://arxiv.org/html/2409.20353)
[55](https://arxiv.org/html/2408.15113)
[56](https://arxiv.org/abs/2307.10792)
[57](https://arxiv.org/html/2501.09579v1)
[58](https://www.mdpi.com/1424-8220/24/5/1368/pdf?version=1708441132)
[59](https://arxiv.org/html/2407.06519v1)
[60](https://dataroots.io/blog/anomaly-detection-in-images-using-patchcore)
[61](https://openaccess.thecvf.com/content/CVPR2022/papers/Roth_Towards_Total_Recall_in_Industrial_Anomaly_Detection_CVPR_2022_paper.pdf)
[62](https://www.youtube.com/watch?v=ZI3TKNxTur4)
[63](https://openaccess.thecvf.com/content/CVPR2023/papers/Tien_Revisiting_Reverse_Distillation_for_Anomaly_Detection_CVPR_2023_paper.pdf)
[64](https://arxiv.org/abs/2106.08265)
[65](https://velog.io/@maseully_hoit/PaDiM-a-Patch-Distribution-Modeling-Framework-for-Anomaly-Detection-and-Localization)
[66](https://arxiv.org/html/2412.07579v1)
[67](https://www.mathworks.com/help/vision/ref/patchcoreanomalydetector.html)
[68](https://arxiv.org/pdf/2011.08785.pdf)
[69](https://arxiv.org/pdf/2503.13828.pdf)
[70](https://arxiv.org/pdf/2508.16034.pdf)
[71](https://ar5iv.labs.arxiv.org/html/2011.08785)
[72](https://arxiv.org/html/2408.03143v2)
[73](https://arxiv.org/html/2408.15113v1)
[74](https://anomalib.readthedocs.io/en/latest/markdown/guides/reference/models/image/padim.html)
[75](https://www.mathworks.com/help/vision/ug/detect-pcb-defects-using-patchcore-detector.html)
