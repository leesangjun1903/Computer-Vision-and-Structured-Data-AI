
# Towards Total Recall in Industrial Anomaly Detection

## 1. 논문의 핵심 주장과 주요 기여

### 1.1 핵심 주장
**PatchCore**는 산업용 이상 탐지에서 "완전한 회상(Total Recall)"을 달성하기 위한 방법이다. 이 논문의 핵심 주장은 세 가지이다:

첫째, **명목 정보의 최대화**: 테스트 시에 사용 가능한 정상 데이터 정보를 최대한 활용해야 한다. 기존 방법들은 고수준의 특성만 사용하여 정상 패턴에 대한 정보를 손실한다.

둘째, **ImageNet 편향 감소**: 사전 학습된 네트워크의 깊은 특성은 자연 이미지 분류에 최적화되어 있어 산업 이상 탐지 작업과의 불일치가 발생한다. 중간 수준의 특성을 활용하여 이 편향을 감소시킬 수 있다.

셋째, **추론 속도 유지**: 메모리 뱅크 방식은 계산 비용이 크기 때문에 Coreset 부분 샘플링을 통해 효율성을 유지해야 한다.

### 1.2 주요 기여도
1. **MVTec AD 벤치마크에서 AUROC 99.6% 달성**, 이전 최고 성능(PaDiM 97.9%)과 비교하여 오류를 **57% 이상 감소**
2. **국소 인식 패치 특성(Locally Aware Patch Features)** 도입으로 공간적 맥락 보존
3. **Greedy Coreset 부분 샘플링**으로 메모리 뱅크를 **100배 감소**시키면서도 성능 유지
4. **상태-최고 이상 검출 및 지역화** 성능 달성

***

## 2. 해결하고자 하는 문제와 제안 방법

### 2.1 문제 정의: Cold-Start 이상 탐지

**산업 환경에서의 과제**:
- 정상(비결함) 이미지만으로 모델 학습 가능해야 함 (비용 및 실용성)
- 미세한 스크래치부터 대규모 구조적 결함까지 다양한 이상 탐지
- 정렬되지 않은 이미지 처리 필요
- 빠른 추론 속도 요구

### 2.2 기존 방법의 한계
- **SPADE**: 메모리 뱅크의 모든 특성을 비교하여 추론 시간 오래 걸림
- **PaDiM**: 이미지 정렬 가정 필요, 고정된 패치 위치별 Mahalanobis 거리만 사용

### 2.3 PatchCore의 제안 방법

#### 2.3.1 국소 인식 패치 특성 (Locally Aware Patch Features)

중간 수준의 특성 맵 $$\phi_{i,j}$$에서 패치 수준 특성을 추출한다:

$$\phi_{i,j}\left(N_{(h,w)}^p\right) = f_{\text{agg}}\left(\{\phi_{i,j}(a,b) | (a,b) \in N_{(h,w)}^p\}\right)$$

여기서:
- $$N_{(h,w)}^p = \{(a,b) | a \in [h-\lfloor p/2 \rfloor, ..., h+\lfloor p/2 \rfloor], b \in [w-\lfloor p/2 \rfloor, ..., w+\lfloor p/2 \rfloor]\}$$는 위치 $$(h,w)$$ 의 이웃 집합
- $$f_{\text{agg}}$$는 적응형 평균 풀링(Adaptive Average Pooling)

패치 특성 수집:

```math
P_{s,p}(\phi_{i,j}) = \{\phi_{i,j}(N_{(h,w)}^p) | h, w \mod s = 0, h < h^*, w < w^*, h,w \in \mathbb{N}\}
```

메모리 뱅크 구성:

$$M = \bigcup_{x_i \in X_N} P_{s,p}(\phi_j(x_i))$$

#### 2.3.2 Coreset 기반 메모리 뱅크 축소

메모리 뱅크의 크기를 줄이기 위해 **Minimax Facility Location Coreset Selection**을 사용한다:

$$M_C^* = \arg\min_{M_C \subset M} \max_{m \in M} \min_{n \in M_C} \|m - n\|_2$$

이는 NP-Hard 문제이므로, 탐욕적 근사를 사용한다(Algorithm 1):

```
for i in [0, ..., l-1]:
    m_i := arg max_{m ∈ M-M_C} min_{n ∈ M_C} ∥ψ(m) - ψ(n)∥_2
    M_C := M_C ∪ {m_i}
```

여기서 $$\psi: \mathbb{R}^d \to \mathbb{R}^{d'}$$ ($$d' < d$$)는 Johnson-Lindenstrauss 정리에 기반한 랜덤 선형 사영이다.

#### 2.3.3 이상 탐지 점수 계산

**이미지 수준 이상 점수**:

```math
m_{\text{test},*}, m^* = \arg\max_{m_{\text{test}} \in P(x_{\text{test}})} \arg\min_{m \in M} \|m_{\text{test}} - m\|_2
```

기본 점수:

```math
s^* = \|m_{\text{test},*} - m^*\|_2
```

이웃 패치들의 거리를 고려한 재가중치 점수:

```math
s = \left[1 - \frac{\exp \|m_{\text{test},*} - m^*\|_2}{\sum_{m \in N_b(m^*)} \exp \|m_{\text{test},*} - m\|_2}\right] \cdot s^*
```

여기서 $$N_b(m^\*)$$는 메모리 뱅크에서 $$m^*$$의 $$b$$-근처 패치 특성들이다.

### 2.4 모델 구조

PatchCore의 전체 파이프라인:

1. **학습 단계**:
   - 정상 이미지들로부터 국소 인식 패치 특성 추출
   - 모든 패치 특성을 메모리 뱅크 $$M$$에 저장
   - Greedy Coreset 부분 샘플링으로 $$M_C$$ 생성

2. **추론 단계**:
   - 테스트 이미지의 패치 특성 추출
   - 각 패치에 대해 메모리 뱅크에서 가장 가까운 정상 패치 찾기
   - 이상 점수 계산 및 이미지 분류
   - 픽셀 수준 이상 지역화 (쌍선형 보간 및 가우시안 평활)

***

## 3. 성능 향상 및 한계점 분석

### 3.1 주요 성능 지표 (MVTec AD 벤치마크)

| 메트릭 | SPADE | PaDiM | DifferNet | PatchCore-25% | PatchCore-10% | PatchCore-1% |
|--------|-------|-------|-----------|---------------|---------------|--------------|
| **Image AUROC ↑** | 85.5% | 95.3% | 94.9% | 99.1% | 99.0% | 99.0% |
| **Pixel AUROC ↑** | 96.0% | 97.5% | - | 98.1% | 98.1% | 98.0% |
| **PRO Score ↑** | 91.7% | 92.1% | - | 93.4% | 93.5% | 93.1% |
| **오류 감소 ↓** | 14.5% | 4.7% | 5.1% | 0.9% | 1.0% | 1.0% |
| **추론 시간** | 0.66s | 0.19s | - | 0.6s | 0.22s | 0.17s |

### 3.2 저샷(Low-shot) 학습 성능

PatchCore는 **샘플 효율성에서 탁월**:
- **1샷** (정상 이미지 1개): AUROC 83.4% (SPADE 71.6%, PaDiM 76.1%)
- **5샷**: AUROC 90.8% (SPADE 75.2%, PaDiM 81.0%)
- **20샷**: AUROC 95.8% (SPADE 79.6%, PaDiM 86.5%)

이는 기존 최고 성능 방법(DifferNet)을 20%의 훈련 데이터로 매칭할 수 있음을 의미한다.

### 3.3 절제 연구(Ablation Studies)

#### 3.3.1 국소 인식 특성의 중요성
- 이웃 크기 $$p=3$$에서 최적 성능
- $$p=1$$ (이웃 없음): 성능 저하
- $$p=5$$ 이상: 과도한 매끄러움으로 세부사항 손실

#### 3.3.2 네트워크 계층의 선택
- 계층 2만 사용: AUROC 98.7% (이미 SOTA 수준)
- 계층 2+3 (기본값): AUROC 99.1% (최고 성능)
- 계층 3+4: ImageNet 편향 증가로 성능 저하

#### 3.3.3 Coreset 부분 샘플링의 효과
- 50%-10% 범위에서: **오히려 성능 향상** (중복 제거)
- 무샘플링 vs 1% 샘플링: 100배 메모리 감소, 성능 거의 동일
- 메모리 뱅크 활용도: 무샘플링 30% → 1% Coreset 95%

### 3.4 성능 향상 주요 요인

1. **중간 수준 특성 활용** (이미지 수준 특성 대비)
   - 공간 해상도 유지
   - ImageNet 편향 감소
   - 더 많은 명목 정보 보존

2. **국소 이웃 집계**
   - 수용 영역 확대
   - 공간적 변동에 강건
   - 해상도 손실 없음

3. **Coreset 부분 샘플링**
   - 중복 특성 제거
   - 추론 시간 대폭 단축
   - 메모리 사용량 최소화

### 3.5 한계점 및 실패 사례 분석

#### 3.5.1 거짓 양성 (False Positives: 19건)
- **높은 명목 분산 오류 (8건)**: 정상 변동이 큰 영역에서 이상으로 오탐
- **라벨 모호성 (8건)**: 경계선상의 변화 (표지판 변화, 배경 색상 등)
- **배경 변동 (3건)**: 조명 변화, 텍스처 변동 등

#### 3.5.2 거짓 음성 (False Negatives: 23건)
- **충분하지 않은 가중치 (13건)**: 이상 검출은 하나 점수가 낮음
- **고해상도 필요 (6건)**: 미세한 결함은 저해상도에서 감지 불가
- **완전히 놓친 이상 (2건)**: 드물고 매우 희미한 이상
- **이미지 전처리 오류 (1건)**: 크롭으로 인한 이상 영역 제거 (1건)

#### 3.5.3 원인별 분석
1. **적응 부족**: 사전 학습된 특성에만 의존하므로 도메인 특정 특성 못 포착
2. **정렬 의존성**: 높은 성능을 위해 어느 정도의 정렬 필요 (PaDiM보다는 덜함)
3. **고해상도 효율성**: 224×224로 제한되면 미세 결함 감지 어려움
4. **경계 케이스**: 명목과 이상의 경계가 불명확한 경우 처리 어려움

***

## 4. 모델의 일반화 성능 향상 가능성

### 4.1 도메인 외 성능 평가

#### 4.1.1 MTD (Magnetic Tile Defects) 데이터셋
- **PatchCore-10**: AUROC 97.9%
- **이전 SOTA (DifferNet)**: 97.7%
- 평가: 높은 변동성 데이터셋에서도 안정적 성능

#### 4.1.2 mSTC (미니 ShanghaiTech Campus)
- **픽셀 AUROC**: 91.8% (PaDiM 91.2%)
- 평가: 자연 장면 데이터셋으로의 전이 학습 성공
- 주의: 계층 3+4로 교환하여 깊은 특성 활용 (도메인 특성 반영)

### 4.2 일반화 성능 향상 방향

#### 4.2.1 기초 모델 활용 (Foundation Models)
**AnomalyDINO (2024-2025)** 사례:
- **DINOv2 기반**: 자기지도 학습으로 사전 학습
- **One-shot 성능**: AUROC 96.6% (기존 PatchCore 93.1% vs)
- **Few-shot 우수성**: 매우 적은 데이터로도 높은 성능
- **장점**: 
  - ImageNet 편향 감소
  - 더 일반화된 특성 표현
  - 도메인 전이 성능 향상

#### 4.2.2 적응 메커니즘 통합
**정규화 흐름 + PatchCore**:
$$p_{\text{model}}(x) = \sum_i w_i \cdot \mathcal{N}(f_{\text{flow}(i)}(x); 0, I)$$

- 학습 가능한 변환으로 명목 분포 학습
- 특성 공간의 분포 불일치 감소

#### 4.2.3 학생-교사 증류 (Student-Teacher Distillation)
**원리**:

$$\mathcal{L} = \mathcal{L}_{\text{distill}} + \lambda \mathcal{L}_{\text{AD}}$$

- 교사 네트워크: 사전 학습된 원본 특성
- 학생 네트워크: 역방향 아키텍처로 정상만 학습
- 이점: 이상 특성에 대한 민감도 향상

#### 4.2.4 합성 이상 활용 (Synthetic Anomalies)
**RealNet (2024) 접근**:

1. **SDAS (Strength-Controllable Diffusion Anomaly Synthesis)**:

$$x_{\text{anom}} = \text{diffusion}_\text{reverse}(x_{\text{normal}}, t, \lambda_{\text{strength}})$$

2. **특성 선택 (AFS - Anomaly-aware Feature Selection)**:

$$\mathcal{F}_{\text{selected}} = \arg\max_{\mathcal{F}} \text{MI}(\mathcal{F}; y_{\text{anom}})$$

- MI: 상호 정보량
- 이점: 더 판별력 있는 특성 부분집합 사용

#### 4.2.5 다중 특성 계층 활용
**Wavelet-Enhanced PaDiM 예시**:
- 단순 패치 특성 대신 웨이블릿 변환 특성 사용
- 성능: Image AUC 99.32%, Pixel AUC 92.10%
- 이점: 다양한 주파수 대역에서 특성 정보 포착

#### 4.2.6 멀티모달 특성 결합
**2D 이미지 + 3D 포인트 클라우드**:

$$\text{Anomaly Score} = \alpha \cdot s_{\text{visual}} + (1-\alpha) \cdot s_{\text{geometric}}$$

- MVTec 3D-AD: Image AUROC 97.2% 달성
- 이점: 표면 기하학적 정보로 미세 결함 감지 향상

#### 4.2.7 고해상도 처리
**Divide-and-Conquer Tiled Ensemble**:
- 고해상도 이미지를 타일로 분할
- 각 타일에서 이상 탐지 수행
- 앙상블로 최종 결정
- 메모리 효율적이면서도 세부 정보 보존

### 4.3 벤치마크에서의 일반화 성능 추이

| 방법 | MVTec AD | MTD | mSTC | 도메인 특정화 필요성 |
|------|----------|-----|------|--------|
| PatchCore | 99.1% | 97.9% | 91.8% | 계층 선택 |
| AnomalyDINO | 99.5% | - | - | 최소화 |
| RealNet | 99.3+% | - | - | 합성 데이터 필요 |
| SACD | 99.2% | - | - | 낮음 |
| PatchEAD | 98.5-99.2% | - | 92.1% | 프롬프트 엔지니어링 |

***

## 5. 2020년 이후 관련 최신 연구 비교 분석

### 5.1 주요 연구 간 계보(Genealogy)

```
┌─ SPADE (2020)
│  └─ 메모리 뱅크 방식 도입
│
└─ PaDiM (2020)
   └─ Mahalanobis 거리 기반 패치 모델링
   
      ↓ (종합)
   
   PatchCore (2021)
   ├─ 국소 인식 패치 특성
   ├─ Coreset 부분 샘플링
   └─ SOTA 성능 달성
   
      ↓ (개선 방향)
   
   ├─ FR-PatchCore (2024)
   │  └─ 피라미드 풀링으로 패치 연결성 강화
   │
   ├─ AnomalyDINO (2024)
   │  └─ 기초 모델 (DINOv2) 활용
   │
   ├─ RealNet (2024)
   │  └─ 합성 이상 + 특성 선택
   │
   ├─ SACD (2025)
   │  └─ 합성 이상 + 역방향 증류
   │
   └─ PatchEAD (2025)
      └─ 기초 모델 호환 통합 프레임워크
```

### 5.2 성능 비교 종합표

| 연도 | 방법 | Image AUROC | Pixel AUROC | 핵심 기술 | 주요 한계 |
|------|------|-------------|-------------|---------|---------|
| 2020 | SPADE | 85.5% | 96.0% | 메모리 뱅크 | 높은 계산 비용 |
| 2020 | PaDiM | 95.3% | 97.5% | 분포 모델링 | 정렬 의존성 |
| 2021 | DifferNet | 94.9% | - | 정규화 흐름 | 구조적 이상 약함 |
| **2021** | **PatchCore** | **99.1%** | **98.1%** | **Coreset 샘플링** | **도메인 적응 부족** |
| 2022 | Student (Distill) | 85.7% | - | 증류 | - |
| 2023 | ComAD | - | - | 컴포넌트 인식 | - |
| 2024 | FR-PatchCore | 98.81% | - | 강화된 연결성 | - |
| 2024 | RealNet | 99.3+% | 98.2+% | 합성 이상 | 합성 데이터 필요 |
| 2024 | AnomalyDINO | 99.5% | - | DINOv2 기반 | - |
| 2025 | SACD | 99.2% | 98.5% | 대조 증류 | - |
| 2025 | PatchEAD | 98.5-99.2% | 97.5-98.5% | 기초 모델 통합 | 프롬프트 조정 |

### 5.3 기술적 혁신 방향의 변화

#### Phase 1 (2018-2020): 자동인코더 중심
- 재구성 오류 기반 이상 탐지
- 한계: 정상 데이터 정규화 어려움

#### Phase 2 (2020-2021): 사전 학습 특성 활용
- **SPADE, PaDiM, PatchCore**: 이미지넷 특성 재사용
- 장점: 훈련 필요 없음, 빠른 배포
- 핵심 진화: 메모리 뱅크 → 분포 모델링 → Coreset 최적화

#### Phase 3 (2021-2023): 적응 메커니즘
- 정규화 흐름 (DifferNet, NormFlow)
- 학생-교사 증류 (Knowledge Distillation)
- 목표: 도메인 특정 분포 학습

#### Phase 4 (2023-2025): 기초 모델과 합성 데이터
- **DINOv2, CLIP** 등 멀티모달 기초 모델
- 합성 이상 생성 (확산 모델 기반)
- 특성 선택 및 강화 (차원 압축, 주의 메커니즘)
- 목표: 도메인 외 일반화 및 적응 효율성

### 5.4 각 방법별 특성 및 적용 시기

#### SPADE (2020): 메모리 뱅크의 시작
```
장점: 직관적, 단순
단점: 계산 비용 높음 (모든 특성 비교)
적용: 소규모 데이터, 빠른 프로토타이핑
```

#### PaDiM (2020): 통계적 모델링
```
장점: 가우시안 분포로 정상 모델링, 정렬 효율성
단점: 고정 패치 위치, 이미지 정렬 필수
적용: 정렬된 공업 제품, 중간 규모 데이터
```

#### PatchCore (2021): 실용적 최적화 ⭐
```
장점: 
  - SOTA 성능 (AUROC 99.1%)
  - 저샷 학습 우수
  - 추론 속도 빠름
  - 정렬 의존도 낮음
단점: 도메인 특정화 필요 (깊이 선택)
적용: 현재 산업 표준, 다양한 제품 검사
```

#### RealNet (2024): 합성 데이터 활용
```
장점: 
  - 구조적 이상 탐지 개선
  - 합성 이상으로 성능 부스트
  - 특성 선택으로 효율성 향상
단점: 합성 데이터 생성 오버헤드, 파라미터 증가
적용: 불균형 데이터셋, 희귀 이상 학습
```

#### AnomalyDINO (2024): 기초 모델 전환 🔥
```
장점:
  - One-shot에서 96.6% AUROC
  - 훈련 완전히 불필요
  - 도메인 전이 우수
  - 간단한 구현
단점: DINOv2 모델 의존성, 새로운 기초 모델 등장 시 재검토 필요
적용: 빠른 배포, 극도로 제한된 데이터
```

#### PatchEAD (2025): 통합 프레임워크
```
장점:
  - 다양한 기초 모델 호환
  - 훈련 필요 없음
  - Few-shot & Zero-shot 성능
  - 일반화 강함
단점: 기초 모델 선택 영향 큼, 시각 프롬프트 엔지니어링
적용: 다양한 산업 시나리오, 빠른 적응
```

### 5.5 성능 포화 현상과 평가 지표의 한계

최근 연구의 중요한 발견:
- **AUROC, PRO 지표의 천정 효과**: 99% 이상 성능이 표준화되면서 지표의 구분력 상실
- **AUPIMO (2024)** 제안: 속도와 정확도를 동시에 고려하는 새로운 메트릭

```
기존: AUROC = 99.1% vs 99.0% → 구분 불가
신규: AUPIMO = 속도(ms) × 정확도를 고려한 복합 지표
```

***

## 6. 앞으로의 연구에 미치는 영향 및 향후 연구 고려사항

### 6.1 PatchCore의 학술적 영향

#### 6.1.1 패러다임 전환
PatchCore는 산업 이상 탐지 분야에서 **새로운 표준을 설정**했다:

1. **메모리 기반에서 최적화 중심으로**
   - 기존: "더 많은 정보를 저장하자" (SPADE)
   - PatchCore: "필요한 정보를 효율적으로 선택하자" (Coreset)
   - 영향: 후속 연구들이 특성 선택, 차원 압축에 집중

2. **특성 계층의 재평가**
   - 기존 인식: 깊을수록 좋다
   - PatchCore 발견: 중간 계층이 최적 (이미지넷 편향 vs 추상화의 균형)
   - 영향: 아키텍처 선택에 대한 새로운 관점

3. **저샷 학습의 중요성 부각**
   - 산업에서 현실적인 관심: 몇 개의 정상 이미지로 시작
   - PatchCore 성과: 5개 이미지로 91% AUROC 달성
   - 영향: Few-shot/Zero-shot 학습으로 연구 방향 전환

#### 6.1.2 인용 및 확산
- **Github 별**: 3,000+ (주요 산업 연구 프로젝트들이 기본 코드로 사용)
- **후속 개선 논문**: 20+ (FR-PatchCore, 웨이블릿-PaDiM 등)
- **상업 도구**: Amazon SageMaker 등에 PatchCore 통합

### 6.2 미래 연구 방향 및 고려사항

#### 6.2.1 도메인 적응 강화 필수
**현재 문제**: 
- 특성 계층 선택이 수동
- 각 데이터셋에 따라 (j=2 vs j=3+4) 재조정 필요

**미래 방향**:

```math
j^* = \arg\max_j \text{AutoML}(\text{검증\_세트}, j)
```

- 자동 백본 검색 (NAS - Neural Architecture Search)
- 자체 감독 적응 (Self-Supervised Adaptation)

#### 6.2.2 실제 환경 강건성
**현재 한계**: MVTec AD는 통제된 환경
- 조명 변화
- 각도 변화
- 손상된 레이블 (오염된 훈련 데이터)

**미래 해결책**:
- **노이즈 강건성**: 메타 학습으로 오염된 데이터 처리
- **적응적 임계값**: 환경에 따라 자동 조정
$$\text{threshold} = \mu_{\text{normal}} + k \cdot \sigma_{\text{normal}}$$

#### 6.2.3 논리적 이상 탐지 (Logical Anomalies)
**미해결 문제**: 
- PatchCore는 "표면 결함"에 최적화
- 누락된 부품, 부정확한 조립 등 구조적 이상 미흡

**미래 방향**:
- **컴포넌트 인식 감지** (ComAD 2023):
$$s_{\text{logical}} = \sum_i w_i \cdot \text{ComponentAD}(c_i)$$

- **그래프 신경망** 기반 부품 관계 모델링
- **VLM (Vision Language Model)** 활용: "부품 B가 부품 A 위에 있어야 함"

#### 6.2.4 계산 효율성과 배포
**현재 달성**: PatchCore-1%로 0.17초/이미지
**향후 목표**: 엣지 디바이스 배포 (모바일, 로봇)

**기술**:
- 모델 양자화 (INT8 실수)
- 지식 증류 (가벼운 학생 모델)
- 원-샷 양자화 (재훈련 불필요)

#### 6.2.5 멀티모달 및 3D 통합
**추세**: 2D 이미지만으로는 한계
```
RGB 이미지 + 깊이 (Depth) + 법선 (Normal)
    ↓
멀티모달 특성 융합
    ↓
미세 표면 결함 + 기하학적 이상
```

**구체적 구현**:
$$f_{\text{fusion}} = \text{Attention}(f_{\text{RGB}}, f_{\text{depth}}, f_{\text{normal}})$$

#### 6.2.6 설명 가능성 (Explainability)
**산업 요구**: "왜 이것을 이상으로 판단했는가?"

**미래 방법**:
- Attention 맵 시각화 (어느 패치가 결정에 영향?)
- 특성 중요도 분석 (어느 특성 채널이 중요?)
- 가장 유사한 정상 패치 표시

```python
# 예시 구현
top_k_normal_patches = search_knn(m_test, M, k=5)
# 이상 이미지에서 "이 부분이 정상 패치들과 다릅니다"를 시각화
visualize_difference(m_test, top_k_normal_patches)
```

#### 6.2.7 연속 학습 (Continual Learning)
**현재**: 정적 메모리 뱅크
**미래**: 새로운 정상 변동에 점진적 적응

```math
M_t = (1-\alpha) M_{t-1} + \alpha \cdot \text{new\_patches}
```

- 동적 임계값 학습
- 개념 드리프트(Concept Drift) 처리

#### 6.2.8 이상 심각도 분류
**현재**: 이상 / 정상 이진 분류
**미래**: 결함 등급 분류 (미세 / 중등 / 심각)

$$s_{\text{severity}} = f_{\theta}(\max_{distance} \text{ in anomaly region})$$

### 6.3 산업 적용 시 고려사항

#### 6.3.1 배포 체크리스트

| 항목 | 검토사항 | PatchCore 기준 | 개선 필요 |
|------|---------|---|---|
| **성능** | AUROC > 95% | ✅ 99.1% | - |
| **속도** | < 200ms/이미지 | ✅ 170ms | - |
| **메모리** | < 1GB | ✅ 500MB (1%) | - |
| **저샷** | 5개 이미지로 90%+ | ✅ 90.8% | - |
| **도메인 적응** | 추가 훈련 불필요 | ⚠️ 부분 (깊이 선택) | 자동화 필요 |
| **실제 환경** | 오염된 데이터 | ❌ 15% 성능 저하 | 강건성 개선 |
| **논리적 이상** | 부품 누락 감지 | ❌ 미흡 | ComAD 고려 |
| **설명성** | 왜 이상인가? | ⚠️ 패치 기반만 | Attention 추가 |

#### 6.3.2 하이브리드 접근 추천
```
┌─────────────────────────────────────┐
│  실제 산업 배포 (2025 기준)         │
├─────────────────────────────────────┤
│                                     │
│  기본: PatchCore (빠르고 효율적)    │
│   ↓                                  │
│  + DINOv2 특성 (일반화 강화)        │
│   ↓                                  │
│  + 합성 이상 (드문 결함 학습)       │
│   ↓                                  │
│  + 컴포넌트 인식 (논리적 이상)      │
│   ↓                                  │
│  + 강건성 검증 (노이즈 데이터)      │
│   ↓                                  │
│  최종: 하이브리드 엔상블 모델       │
└─────────────────────────────────────┘
```

#### 6.3.3 비용-성능 트레이드오프

| 상황 | 추천 방법 | 이유 |
|------|---------|------|
| **빠른 배포 (< 1주)** | PatchCore | 훈련 불필요, SOTA 성능 |
| **극도 제한 샘플 (1-5개)** | AnomalyDINO | One-shot AUROC 96.6% |
| **고정밀 요구** | RealNet | 99.3% AUROC, 합성 이상 활용 |
| **다양한 제품** | PatchEAD | 기초 모델 호환, 재사용 가능 |
| **논리적 이상** | ComAD + PatchCore | 하이브리드 접근 |

***

## 요약

**"Towards Total Recall in Industrial Anomaly Detection"**은 2021년 발표되었음에도 여전히 산업 표준으로 사용되는 근본적인 이유는:

1. **극단적 성능**: AUROC 99.1%로 실질적으로 문제를 해결
2. **실용성**: 0.17초/이미지로 실시간 처리 가능
3. **간단성**: 사전 학습된 모델만 필요, 훈련 불필요
4. **적응성**: 저샷 학습으로 새 제품에 빠르게 적응

2024-2025 최신 연구들은 PatchCore를 기반으로 **도메인 외 일반화**(AnomalyDINO), **합성 데이터 활용**(RealNet), **논리적 이상**(ComAD) 등으로 확장하고 있다. 그러나 **정상 데이터만으로 이상을 탐지**하는 근본적인 아이디어와 **Coreset 기반 최적화**는 여전히 혁신적이다.

미래의 성공적인 산업 이상 탐지는 **PatchCore의 효율성 + 기초 모델의 일반화력 + 합성 이상의 다양성 + 컴포넌트 인식의 논리성**을 결합한 하이브리드 접근이 될 것으로 예상된다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/64441467-a421-42f6-b9fb-331f856c03f6/2106.08265v2.pdf)
[2](https://linkinghub.elsevier.com/retrieve/pii/S016636152400126X)
[3](https://ieeexplore.ieee.org/document/11035252/)
[4](https://ieeexplore.ieee.org/document/11003961/)
[5](https://linkinghub.elsevier.com/retrieve/pii/S1568494620304786)
[6](https://arxiv.org/abs/2509.25856)
[7](https://ieeexplore.ieee.org/document/11147474/)
[8](https://ieeexplore.ieee.org/document/10820339/)
[9](https://ieeexplore.ieee.org/document/10972598/)
[10](https://www.mdpi.com/2079-9292/14/8/1613)
[11](https://www.mdpi.com/1424-8220/25/12/3721)
[12](https://arxiv.org/pdf/2503.04997.pdf)
[13](https://arxiv.org/html/2501.11310v1)
[14](https://arxiv.org/pdf/2401.16402.pdf)
[15](http://arxiv.org/pdf/2404.17925.pdf)
[16](https://arxiv.org/pdf/2207.10298.pdf)
[17](https://arxiv.org/abs/2305.08509)
[18](http://arxiv.org/pdf/2503.01569.pdf)
[19](https://arxiv.org/html/2502.05761v1)
[20](https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2025.1554196/abstract)
[21](https://www.manuscriptlink.com/society/kics/media?key=kics%2Fconference%2Ficaiic2024%2F1570978404.pdf)
[22](https://www.sciencedirect.com/science/article/abs/pii/S0164121223003096)
[23](https://www.ijsat.org/papers/2025/2/6154.pdf)
[24](https://pmc.ncbi.nlm.nih.gov/articles/PMC10934034/)
[25](https://pmc.ncbi.nlm.nih.gov/articles/PMC10300695/)
[26](https://pmc.ncbi.nlm.nih.gov/articles/PMC12349016/)
[27](https://arxiv.org/html/2509.25856v1)
[28](https://arxiv.org/pdf/2404.17931.pdf)
[29](https://github.com/M-3LAB/awesome-industrial-anomaly-detection)
[30](https://pdfs.semanticscholar.org/c033/2c33bb2ccee4adb28eca55ca6a534d154555.pdf)
[31](https://openaccess.thecvf.com/content/ACCV2020/papers/Yi_Patch_SVDD_Patch-level_SVDD_for_Anomaly_Detection_and_Segmentation_ACCV_2020_paper.pdf)
[32](https://arxiv.org/pdf/2409.19892.pdf)
[33](https://openaccess.thecvf.com/content/WACV2022/papers/Tsai_Multi-Scale_Patch-Based_Representation_Learning_for_Image_Anomaly_Detection_and_Segmentation_WACV_2022_paper.pdf)
[34](https://arxiv.org/html/2508.19060v1)
[35](https://pdfs.semanticscholar.org/33b6/3380d52f4d0ddd00b1c1e1a28c21cea16b46.pdf)
[36](https://arxiv.org/html/2511.02541v1)
[37](https://arxiv.org/html/2507.22659v1)
[38](https://dataroots.io/blog/anomaly-detection-in-images-using-patchcore)
[39](https://arxiv.org/abs/2501.11310)
[40](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/13456/3052541/Patch-distribution-modeling-framework-adaptive-cosine-estimator-PaDiM-ACE-for/10.1117/12.3052541.full)
[41](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/13164/3018802/High-precision-anomaly-detection-based-on-pre-trained-features-enhanced/10.1117/12.3018802.full)
[42](https://www.jstage.jst.go.jp/article/ieejeiss/144/9/144_886/_article/-char/ja/)
[43](https://ieeexplore.ieee.org/document/10818308/)
[44](https://ieeexplore.ieee.org/document/10457237/)
[45](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/13072/3023184/Insights-of-anomaly-detection--How-does-polluted-training-data/10.1117/12.3023184.full)
[46](https://ieeexplore.ieee.org/document/10944159/)
[47](http://link.springer.com/10.1007/978-3-030-68799-1_35)
[48](https://ieeexplore.ieee.org/document/10658311/)
[49](https://ieeexplore.ieee.org/document/10655216/)
[50](https://arxiv.org/pdf/2011.08785.pdf)
[51](https://www.mdpi.com/2076-3417/13/23/12655/pdf?version=1700837405)
[52](http://arxiv.org/pdf/2405.14325.pdf)
[53](http://arxiv.org/pdf/2408.04817.pdf)
[54](https://arxiv.org/pdf/2307.08059.pdf)
[55](https://arxiv.org/html/2401.01984v5)
[56](https://arxiv.org/html/2403.04932v2)
[57](http://arxiv.org/pdf/2405.14529.pdf)
[58](https://www.sciencedirect.com/science/article/abs/pii/S0952197623005742)
[59](https://dippingtodeepening.tistory.com/111)
[60](https://liu.diva-portal.org/smash/get/diva2:1766718/FULLTEXT01.pdf)
[61](https://paperreading.club/page?id=192599)
[62](https://www.sciencedirect.com/science/article/abs/pii/S0957417425030647)
[63](https://wandb.ai/s23998/Anomaly%20Detection/reports/PaDiM-performance-on-all-datasets--VmlldzoxMDI5MDU3MA)
[64](https://www.sciencedirect.com/science/article/abs/pii/S0166361523000519)
[65](https://chips.it.kr/posts/paper-review-Towards-Total-Recall-in-Industrial-Anomaly-Detection/)
[66](https://ffighting.net/deep-learning-paper-review/anomaly-detection/padim/)
[67](https://arxiv.org/pdf/2508.16034.pdf)
[68](https://arxiv.org/html/2507.13378v1)
[69](https://openaccess.thecvf.com/content/ICCV2021/papers/Hou_Divide-and-Assemble_Learning_Block-Wise_Memory_for_Unsupervised_Anomaly_Detection_ICCV_2021_paper.pdf)
[70](https://openaccess.thecvf.com/content/WACV2021/papers/Rudolph_Same_Same_but_DifferNet_Semi-Supervised_Defect_Detection_With_Normalizing_Flows_WACV_2021_paper.pdf)
[71](https://arxiv.org/html/2403.12362v1)
[72](https://arxiv.org/html/2508.16034v1)
[73](https://openaccess.thecvf.com/content/WACV2024/papers/Vieira_e_Silva_Attention_Modules_Improve_Image-Level_Anomaly_Detection_for_Industrial_Inspection_A_WACV_2024_paper.pdf)
[74](https://ar5iv.labs.arxiv.org/html/2106.08265)
[75](https://arxiv.org/abs/2504.08049)
