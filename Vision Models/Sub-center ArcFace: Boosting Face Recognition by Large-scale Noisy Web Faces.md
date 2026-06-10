# Sub-center ArcFace: Boosting Face Recognition by Large-scale Noisy Web Faces

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

기존 ArcFace를 포함한 마진 기반 얼굴 인식 방법들은 **대규모 노이즈 레이블 데이터에 취약**하며, 고품질 학습 데이터셋 구축에 막대한 인적 비용이 소요된다. 본 논문은 각 클래스에 $K$개의 서브센터(sub-center)를 도입하여 **인트라클래스 제약을 완화**함으로써 노이즈 강건성을 획득하면서도 최종적으로 우수한 식별 성능을 달성할 수 있음을 주장한다.

### 주요 기여 (3가지)

| 기여 항목 | 설명 |
|-----------|------|
| **Sub-class 도입** | ArcFace에 $K$개 서브센터를 도입, 대규모 실세계 노이즈 데이터에서 일관된 성능 향상 |
| **자동 노이즈 정제** | 비지배 서브센터(non-dominant sub-center) 제거 및 고신뢰도 노이즈 샘플 드롭으로 수동 정제 데이터 수준 달성 |
| **대규모 확장성** | 병렬 툴킷 기반 구현으로 Celeb500K 등 대규모 원시 웹 얼굴 데이터에서 IJB-B, IJB-C, MegaFace, FRVT SOTA 달성 |

---

## 2. 상세 분석

### 2-1. 해결하고자 하는 문제

**문제 1: 대규모 웹 크롤링 데이터의 노이즈**

- 인터넷 수집 데이터셋(MS1MV0 등)의 노이즈 비율: 약 $47.1\% \sim 54.4\%$
- 노이즈 유형:
  - **Open-set noise**: 학습 레이블 집합 밖의 실제 정체성을 가진 얼굴이 잘못 레이블됨
  - **Close-set noise**: 학습 레이블 집합 내의 다른 정체성으로 잘못 레이블됨

**문제 2: 수동 데이터 정제의 비현실성**

- 고품질 데이터셋(MS1MV3) 구축: 50명의 주석자가 1개월 연속 작업 필요
- 이는 대규모 데이터 활용을 위한 근본적 병목

**문제 3: 기존 노이즈 대응 방법의 한계**

- 샘플 재가중(NT, NR): 초기 모델 성능에 의존적, 다수의 수작업 하이퍼파라미터 필요
- Co-mining: 쌍둥이 네트워크 동시 학습 필요 → 대규모 모델에 비실용적

---

### 2-2. 제안 방법 및 수식

#### 기존 ArcFace 손실 함수

$$\ell_{\text{ArcFace}} = -\log \frac{e^{s\cos(\theta_{y_i}+m)}}{e^{s\cos(\theta_{y_i}+m)} + \sum_{j=1, j\neq y_i}^{N} e^{s\cos\theta_j}} \tag{1}$$

- $\theta_j$: 임베딩 특징 $\mathbf{x}_i \in \mathbb{R}^{512\times1}$와 $j$번째 클래스 센터 $W_j \in \mathbb{R}^{512\times1}$ 사이의 각도
- $m = 0.5$: 각도 마진 파라미터
- $s = 64$: 특징 재스케일 파라미터
- $\ell_2$ 정규화 후 $\theta_j = \arccos(W_j^T \mathbf{x}_i)$

ArcFace는 **단 하나의 양성 센터**로 모든 샘플을 당기므로, 노이즈 샘플이 대규모 잘못된 손실값을 생성하여 학습을 저해한다.

#### 제안: Sub-center ArcFace 손실 함수

$$\ell_{\text{ArcFace}_{\text{subcenter}}} = -\log \frac{e^{s\cos(\theta_{i,y_i}+m)}}{e^{s\cos(\theta_{i,y_i}+m)} + \sum_{j=1, j\neq y_i}^{N} e^{s\cos\theta_{i,j}}} \tag{2}$$

여기서:

$$\theta_{i,j} = \arccos\left(\max_k \left(W_{jk}^T \mathbf{x}_i\right)\right), \quad k \in \{1, \cdots, K\}$$

**핵심 아이디어**: 각 클래스 $j$에 대해 $K$개의 서브센터 $W_{j1}, W_{j2}, \ldots, W_{jK}$를 정의하고, 샘플이 $K$개 양성 서브센터 중 **어느 하나에라도** 가까우면 낮은 손실을 받도록 설계.

#### 구현 파이프라인 (Fig. 2 기반)

```
입력: x_i ∈ R^{512×1}
  ↓ ℓ₂ 정규화
서브센터: W ∈ R^{N×K×512}
  ↓ 행렬 곱 (W^T x_i)
서브클래스별 코사인 유사도: S ∈ R^{N×K}
  ↓ Max Pooling (차원축소)
클래스별 코사인 유사도: S' ∈ R^{N×1}
  ↓ ArcFace 동일 절차 (arccos → margin 추가 → softmax → cross-entropy)
손실값
```

#### 비교 전략 분석 (Table 1)

| 제약 조건 | 서브센터? | 엄격성 | Open-set 강건성 | Close-set 강건성 |
|-----------|-----------|--------|-----------------|-----------------|
| $\text{Min(inter)} - \text{Min(intra)} \geq m$ | ✓ | +++ | ++ | + |
| $\text{Max(inter)} - \text{Min(intra)} \geq m$ | ✓ | + | ++ | ++ |
| $\text{Min(inter)} - \text{Max(intra)} \geq m$ | - | ++++ | - | - |
| $\text{Max(inter)} - \text{Max(intra)} \geq m$ | - | ++ | + | - |

> **논문 선택**: 전략 (1) — Max Pooling을 통한 가장 가까운 양성 서브센터 배정. 이 전략이 Open-set 노이즈에 가장 실용적인 균형을 제공.

#### 비지배 서브센터 제거 절차

훈련이 충분히 진행된 후:

1. **1단계**: Sub-center ArcFace ($K=3$)로 초기 모델 학습
2. **2단계**: 각 클래스에서 지배 서브센터(샘플 다수 배정) 이외의 비지배 서브센터 식별
3. **3단계**: 비지배 서브센터 소속 샘플 중 지배 서브센터와의 각도 $> 75°$인 고신뢰도 노이즈 샘플 제거
4. **4단계**: 자동 정제된 데이터셋으로 $K=1$ 모델 재학습 ($K=3 \downarrow 1$)

---

### 2-3. 모델 구조

- **백본**: ResNet-50, ResNet-100 (512-D 임베딩)
- **얼굴 검출/정렬**: RetinaFace (ResNet-50)
- **서브센터 행렬**: $W \in \mathbb{R}^{N \times K \times 512}$ (기존 $\mathbb{R}^{N \times 512}$ 대비 $K$배 확장)
- **입력**: $112 \times 112$ 정규화된 얼굴 크롭
- **학습 프레임워크**: MXNet, 8× NVIDIA Tesla P40 (24GB)
- **배치 크기**: 512, momentum=0.9, weight decay= $5\times10^{-4}$

---

### 2-4. 성능 향상 결과

#### IJB-B / IJB-C (TAR@FAR=1e-4, ResNet-100)

| 설정 | IJB-B | IJB-C |
|------|-------|-------|
| MS1MV0, $K=1$ (ArcFace) | 87.91% | 90.42% |
| **MS1MV0, $K=3\downarrow1$ (제안)** | **94.94%** | **96.28%** |
| MS1MV3, $K=1$ (수동 정제) | 95.25% | 96.61% |
| **Celeb500K, $K=3\downarrow1$ (제안)** | **95.75%** | **96.96%** |

→ 노이즈 데이터(MS1MV0)에서 수동 정제 데이터(MS1MV3) 수준 성능 달성, 대규모 데이터(Celeb500K)로 SOTA 초과

#### MegaFace (Rank-1 식별 정확도)

| 설정 | Id | Ver |
|------|----|-----|
| MS1MV0, $K=1$ | 96.52% | 96.75% |
| **MS1MV0, $K=3\downarrow1$** | **98.16%** | **98.36%** |
| MS1MV3, $K=1$ | 98.40% | 98.51% |
| **Celeb500K, $K=3\downarrow1$** | **98.78%** | **98.69%** |

#### FRVT 결과 (Wild Track)
- FNMR@FMR≤1e-5 = 0.0303 → **전체 3위** (수백 개 산업 제출물 중)

---

### 2-5. 한계점

1. **Close-set 노이즈 취약성**: 50% Close-set 노이즈 환경에서 Sub-center ArcFace의 성능($72.04\%$)이 ArcFace($75.80\%$)보다 낮음. 인터클래스 제약이 오히려 더 엄격하게 작용하기 때문
2. **K 값의 민감성**: $K$가 너무 크면($K=10$) 인트라클래스 컴팩트성이 급격히 저하됨 ($93.72\% \rightarrow 67.94\%$)
3. **각도 임계값의 수동 설정**: 노이즈 드롭 임계값($>75°$)이 데이터셋에 따라 최적값이 달라질 수 있음
4. **2단계 학습의 계산 비용**: 초기 모델 학습 후 재학습(from scratch)이 필요하여 전체 학습 시간 증가
5. **Hard sample과 Noisy sample 혼재**: 비지배 서브클래스에 일부 정상 Hard sample(4.28%)도 포함되어 잠재적 정보 손실 발생

---

## 3. 모델의 일반화 성능 향상 가능성 (핵심 분석)

### 3-1. 일반화 성능 향상의 원리

#### (A) 자동 데이터 분포 분리에 의한 정규화 효과

Sub-center ArcFace는 각 클래스 내에 **다중 모드 분포(multi-modal distribution)**를 명시적으로 허용한다. 이는 단일 가우시안 가정의 한계를 극복하며, 다음과 같은 일반화 이점을 제공한다:

$$\text{Sub-center: } P(\mathbf{x} | c) \approx \sum_{k=1}^{K} \pi_k \cdot \mathcal{N}(\mathbf{x}; \mu_{ck}, \Sigma_{ck})$$

각 정체성이 정면, 측면, 가려진 얼굴 등 다양한 모달리티를 가짐을 자연스럽게 모델링하여, 테스트 시 다양한 외관 변화에 강건한 표현을 학습한다.

#### (B) 노이즈 감소를 통한 결정 경계 정교화

MS1MV0에서 $K=3$ 적용 시:
- 지배 서브클래스 내 노이즈 비율: $38.47\% \rightarrow 12.40\%$로 **3분의 1 수준**으로 감소
- 이는 학습 과정에서 결정 경계(decision boundary)가 노이즈에 의해 왜곡되는 정도를 줄임

#### (C) 대용량 미정제 데이터 활용 가능성

| 데이터셋 | 크기 | 노이즈 | IJB-C TAR@1e-4 |
|----------|------|--------|----------------|
| MS1MV3 (정제) | 5.1M / 93K ID | ~0% | 96.50% |
| MS1MV0 (미정제) | 10M / 100K ID | ~50% | 95.92%* |
| Celeb500K (미정제) | 대규모 | 높음 | **96.91%*** |

* $K=3\downarrow1$ 적용

→ **데이터 규모가 정제보다 더 큰 일반화 이득을 제공**할 수 있음을 시사. 이는 실용적 일반화 측면에서 매우 중요한 발견이다.

#### (D) 다양한 평가 벤치마크에서의 일관된 향상

LFW(제한된 환경), CFP-FP(정면-측면), AgeDB-30(나이 변화), IJB-B/C(비제한 환경), MegaFace(대규모 산란), FRVT(실환경)에 걸친 일관된 성능 향상은 특정 도메인 편향 없는 **범용 일반화**를 증거한다.

### 3-2. 일반화 한계와 조건

- **전제 조건**: Open-set 노이즈가 지배적인 환경에서 최대 효과. Close-set 노이즈가 50% 이상인 극단적 환경에서는 일반화 능력이 제한됨
- **네트워크 용량 의존성**: 충분한 discriminative power가 확보된 후 노이즈 드롭 단계가 효과적이므로, 모델 크기와 일반화 성능이 연동됨

---

## 4. 앞으로의 연구에 미치는 영향 및 고려할 점

### 4-1. 연구 영향

#### (A) 노이즈 레이블 학습(Learning with Noisy Labels) 방향 전환

기존의 **샘플 재가중(sample re-weighting)** 패러다임에서 **표현 공간 재설계(representation space redesign)** 패러다임으로의 전환을 촉발한다. 손실 함수 차원에서 노이즈를 다루는 것이 후처리보다 근본적으로 효율적임을 보였다.

#### (B) 마진 기반 손실 함수 설계의 새로운 방향

단일 센터 가정(single prototype assumption)을 다중 센터로 확장하는 아이디어는 얼굴 인식 외에 다음 분야로 확장될 수 있다:
- 보행자 재식별(Person Re-ID)
- 세립 분류(Fine-grained Classification)
- 의료 영상 진단 (동일 질병의 다양한 표현형 처리)

#### (C) 자동 데이터 정제 파이프라인 표준화

수동 정제 없이 원시 웹 크롤링 데이터로 고성능 모델을 구축하는 실용적 파이프라인을 제시하여, 데이터 수집-학습 사이클의 비용을 획기적으로 낮춤.

#### (D) 대용량 데이터 시대의 얼굴 인식 패러다임

"데이터 품질 vs. 데이터 양" 트레이드오프에서 **품질 보장 없이 양을 확장**하는 방향이 실용적임을 실증적으로 증명.

---

### 4-2. 향후 연구 시 고려할 점

#### (A) Close-set 노이즈 강건성 강화

현재 방법은 Close-set 노이즈에 상대적으로 취약하다. 향후 연구는 다음을 고려해야 한다:
- **그래프 기반 레이블 정정**: 유사 클래스 간의 그래프 구조를 활용한 Close-set 노이즈 탐지
- **대조 학습(Contrastive Learning)과의 결합**: SimCLR, MoCo 등과 결합하여 Close-set 노이즈에서의 표현 학습 개선

#### (B) K 값의 자동 결정

현재 $K$는 수동으로 설정($K=3$)된다. 클래스별로 최적 $K$가 다를 수 있으므로:
- **적응적 서브센터 수 결정**: 클래스 내 분산을 기반으로 동적으로 $K$ 조절
- **베이지안 비모수 방법**: Dirichlet Process 기반 자동 클러스터 수 결정

#### (C) 노이즈 임계값의 데이터 적응적 설정

현재 $75°$ 고정 임계값은 데이터셋 의존적이다:
- **GMM 기반 자동 임계값**: 각도 분포를 Gaussian Mixture Model로 피팅하여 자동으로 분리점 탐색
- **Curriculum 방식**: 학습 초기에는 관대한 임계값, 후기에는 엄격한 임계값 적용

#### (D) 경량화 및 효율성

- 서브센터 행렬 $W \in \mathbb{R}^{N \times K \times 512}$는 메모리를 $K$배 증가시킴 → Knowledge Distillation으로 서브센터 정보를 단일 센터로 압축하는 연구 필요
- 모바일 환경에서의 적용 가능성 탐색

#### (E) 다른 도메인으로의 확장 시 유의사항

- 얼굴 인식에서의 "다양한 자세/조명" = 도메인에서의 "자연스러운 다중 모드" → 도메인마다 다중 모드의 성격이 다름
- 의료 영상 등에서는 노이즈와 하드 샘플(드문 증상)의 구분이 더 복잡함

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

> **주의**: 아래 연구들은 논문 PDF에 수록되지 않은 2020년 이후 연구로, 제 학습 데이터 기반 지식을 활용합니다. 직접 확인되지 않은 수치는 명시하겠습니다.

### 5-1. 노이즈 레이블 얼굴 인식 관련 후속 연구

#### (A) ElasticFace (CVPR 2022)
- **논문**: "ElasticFace: Elastic Margin Loss for Deep Face Recognition" (Boutros et al., CVPR 2022)
- **핵심**: 고정 마진($m$) 대신 학습 중 무작위 마진 샘플링으로 더 유연한 결정 경계 형성
- **Sub-center ArcFace와의 관계**: 서브센터 아이디어와 결합 가능한 상호보완적 접근

#### (B) AdaFace (CVPR 2022)
- **논문**: "AdaFace: Quality Adaptive Margin for Face Recognition" (Kim et al., CVPR 2022)
- **핵심**: 이미지 품질에 따라 마진을 적응적으로 조절 — 저품질(=잠재적 노이즈) 샘플에는 작은 마진, 고품질 샘플에는 큰 마진
- **Sub-center ArcFace와의 비교**: Sub-center는 서브클래스 구조로 노이즈를 격리하는 반면, AdaFace는 품질 기반 마진 조절로 접근 — 두 방법은 상호보완적

#### (C) NPT-Loss (ECCV 2022 계열)
- 노이즈 내성 표현 학습을 위한 다양한 후속 연구들이 Sub-center ArcFace의 아이디어를 기반으로 확장

#### (D) PartialFC (ICCV 2021)
- **논문**: "Killing Two Birds with One Stone: Efficient and Robust Training of Face Recognition CNNs by Partial FC" (An et al., ICCV 2021)
- **핵심**: Sub-center ArcFace와 결합하여 대규모 클래스(수백만)에서 부분적 클래스만 활성화하여 메모리 효율 대폭 향상
- **중요성**: Sub-center ArcFace의 확장성 문제(메모리)를 해결하는 직접적 후속 연구

### 5-2. 비교 요약표

| 방법 | 노이즈 대응 전략 | 메모리 효율 | Close-set 강건성 | 적응적 마진 |
|------|-----------------|-------------|-----------------|-------------|
| ArcFace (2019) | ✗ | ✓ | ✗ | ✗ |
| **Sub-center ArcFace (2020)** | **✓ (서브클래스)** | **△** | **△** | **✗** |
| AdaFace (2022) | ✓ (품질 기반) | ✓ | ✓ | ✓ |
| ElasticFace (2022) | △ | ✓ | △ | ✓ (마진 탄성화) |
| PartialFC + Sub-center (2021) | ✓ | ✓✓ | △ | ✗ |

---

## 참고 자료

1. **[주 논문]** Deng, J., Guo, J., Liu, T., Gong, M., Zafeiriou, S.: "Sub-center ArcFace: Boosting Face Recognition by Large-scale Noisy Web Faces." ECCV 2020. (제공된 PDF)

2. **[기반 논문]** Deng, J., Guo, J., Xue, N., Zafeiriou, S.: "ArcFace: Additive Angular Margin Loss for Deep Face Recognition." CVPR 2019.

3. **[비교 방법]** Wang, X., Wang, S., Wang, J., Shi, H., Mei, T.: "Co-mining: Deep Face Recognition with Noisy Labels." ICCV 2019.

4. **[비교 방법]** Qian, Q., Shang, L., Sun, B., Hu, J., Li, H., Jin, R.: "SoftTriple Loss: Deep Metric Learning Without Triplet Sampling." ICCV 2019.

5. **[후속 연구]** An, X., et al.: "Killing Two Birds with One Stone: Efficient and Robust Training of Face Recognition CNNs by Partial FC." ICCV 2021.

6. **[후속 연구]** Kim, M., et al.: "AdaFace: Quality Adaptive Margin for Face Recognition." CVPR 2022.

7. **[후속 연구]** Boutros, F., et al.: "ElasticFace: Elastic Margin Loss for Deep Face Recognition." CVPR 2022.

8. **[데이터셋]** Wang, F., et al.: "The Devil of Face Recognition is in the Noise." ECCV 2018.

---

> **정확도 관련 고지**: 2020년 이후 최신 연구(Section 5) 중 AdaFace, ElasticFace, PartialFC의 일반적 특성과 방향성은 제 학습 데이터에 기반하며, 구체적 수치는 원 논문을 통해 검증하시기 바랍니다. 주 논문(PDF) 기반 분석(Section 1-4)은 제공된 원문에 직접 근거합니다.
