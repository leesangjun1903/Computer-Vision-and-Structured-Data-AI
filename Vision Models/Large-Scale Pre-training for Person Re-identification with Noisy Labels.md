# Large-Scale Pre-training for Person Re-identification with Noisy Labels

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

이 논문의 핵심 주장은 **비디오 트랙렛(tracklet)에서 자동으로 생성된 노이즈 레이블(noisy label)을 활용한 대규모 사전 학습(pre-training)이 Person Re-ID 표현 학습에 효과적**이라는 것입니다. 기존의 ImageNet 기반 사전 학습이나 비지도 사전 학습보다 더 우수한 Re-ID 특화 표현을 학습할 수 있음을 실험적으로 증명합니다.

### 주요 기여

| 기여 항목 | 내용 |
|-----------|------|
| **LUPerson-NL 데이터셋 구축** | 약 10M 이미지, 430K 노이즈 레이블 ID, 21K 씬(scene) 포함. 인간 레이블링 없이 구축된 최대 규모의 Re-ID 데이터셋 |
| **PNL 프레임워크 제안** | 지도 분류 학습 + 프로토타입 기반 대조 학습 + 레이블 가이드 대조 학습을 통합한 새로운 사전 학습 프레임워크 |
| **SOTA 달성** | CUHK03, Market1501, DukeMTMC, MSMT17 등 주요 벤치마크에서 기존 방법 대비 우수한 성능 달성 |
| **Few-shot/Small-scale 우수성** | 데이터가 제한된 환경에서 더욱 큰 성능 향상을 보임 |

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

Person Re-ID 분야는 다음과 같은 문제들을 지니고 있습니다:

1. **도메인 갭(Domain Gap)**: ImageNet으로 사전 학습된 모델은 일반 이미지 기반이므로, 사람 중심의 Re-ID 이미지와 도메인 차이가 큼
2. **레이블 획득 비용**: 고품질 Re-ID 레이블은 다중 카메라 뷰를 확인해야 하므로 비용이 매우 큼
3. **데이터 규모 한계**: 기존 Re-ID 데이터셋(Market1501: 32K, DukeMTMC: 36K)은 다른 비전 태스크 대비 매우 소규모
4. **노이즈 레이블 처리**: 트랙렛 기반 자동 레이블에는 필연적으로 두 가지 유형의 노이즈가 포함됨:
   - **Noise-I**: 같은 사람이 서로 다른 ID로 분류됨 (tracklet 분리)
   - **Noise-II**: 서로 다른 사람이 같은 ID로 분류됨 (tracklet 혼합)

### 2.2 LUPerson-NL 데이터셋 구축

FairMOT 다중 객체 추적 알고리즘을 사용하여 기존 LUPerson의 원시 비디오에서 트랙렛을 추출하고, 각 트랙렛에 고유 ID를 부여합니다.

**필터링 전략**:
- 200 프레임 미만 등장하는 ID 제거
- 20 프레임당 1장 샘플링으로 중복 이미지 감소
- Human pose estimation으로 불완전한 바운딩 박스 필터링

**결과 통계**:

| 항목 | 수치 |
|------|------|
| 전체 이미지 수 | 10,683,716 |
| ID 수 | ~433,997 |
| 씬(scene) 수 | 21,697 |
| 레이블 유형 | 노이즈 포함 자동 레이블 |

### 2.3 PNL 프레임워크 (Pre-training with Noisy Labels)

#### 전체 구조

PNL은 Siamese 네트워크 구조를 기반으로 하며, 세 가지 학습 모듈로 구성됩니다.

입력 이미지 $\boldsymbol{x}_i$에 대해 두 가지 랜덤 증강 $(T, T')$을 적용하여 $(\tilde{\boldsymbol{x}}_i, \tilde{\boldsymbol{x}}'_i)$를 생성합니다:
- $\tilde{\boldsymbol{x}}_i$: 인코더 $E_q$ → 쿼리 특징 $\boldsymbol{q}_i$
- $\tilde{\boldsymbol{x}}'_i$: 모멘텀 인코더 $E_k$ → 키 특징 $\boldsymbol{k}_i$

$E_k$의 가중치는 $E_q$의 지수 이동 평균(EMA)으로 업데이트됩니다.

#### 모듈 1: 지도 분류 학습 (Supervised Classification)

레이블이 교정된 $\hat{y}_i$를 기반으로 분류 손실을 계산합니다:

$$\mathcal{L}^i_{ce} = -\log(\boldsymbol{p}_i[\hat{y}_i]) \tag{1}$$

여기서 $\boldsymbol{p}_i \in \mathbb{R}^K$는 인코더 $E_q$의 출력에 분류기를 적용한 확률 벡터이며, $K$는 전체 ID 클래스 수입니다.

#### 모듈 2: 레이블 교정 (Label Rectification with Prototypes)

프로토타입 딕셔너리 $\{c_1, c_2, \ldots, c_K\}$를 유지하며, 각 프로토타입 $c_k \in \mathbb{R}^d$는 클래스별 특징 중심(centroid)을 나타냅니다.

**프로토타입 유사도 점수**:

$$s_i^k = \frac{\exp(\boldsymbol{q}_i \cdot \boldsymbol{c}_k / \tau)}{\sum_{k=1}^K \exp(\boldsymbol{q}_i \cdot \boldsymbol{c}_k / \tau)} \tag{2}$$

**레이블 교정**:

$$\boldsymbol{l}_i = \frac{1}{2}(\boldsymbol{p}_i + \boldsymbol{s}_i)$$

$$\hat{y}_i = \begin{cases} \arg\max_j \boldsymbol{l}_i^j & \text{if } \max_j \boldsymbol{l}_i^j > T \\ y_i & \text{otherwise} \end{cases} \tag{3}$$

- $\boldsymbol{l}_i$: 분류 확률 $\boldsymbol{p}_i$와 프로토타입 유사도 $\boldsymbol{s}_i$의 평균으로 계산된 소프트 의사 레이블
- $T$: 임계값 (논문에서 $T = 0.8$로 설정)
- 최고 점수가 $T$를 초과하면 교정된 레이블 사용, 그렇지 않으면 원본 레이블 유지

#### 모듈 3: 프로토타입 기반 대조 학습 (Prototype-Based Contrastive Learning)

각 샘플이 자신이 속한 프로토타입에 가까워지도록 학습:

$$\mathcal{L}^i_{pro} = -\log\frac{\exp(\boldsymbol{q}_i \cdot \boldsymbol{c}_{\hat{y}_i} / \tau)}{\sum_{j=1}^{K} \exp(\boldsymbol{q}_i \cdot \boldsymbol{c}_j / \tau)} \tag{4}$$

프로토타입 모멘텀 업데이트:

$$\boldsymbol{c}_{\hat{y}_i} = m\boldsymbol{c}_{\hat{y}_i} + (1-m)\boldsymbol{q}_i \tag{5}$$

여기서 $m = 0.999$로 설정됩니다.

#### 모듈 4: 레이블 가이드 대조 학습 (Label-Guided Contrastive Learning)

기존 인스턴스 수준 대조 학습:

$$\mathcal{L}^i_{ic} = -\log\frac{\exp(\boldsymbol{q}_i \cdot \boldsymbol{k}^+_i / \tau)}{\exp(\boldsymbol{q}_i \cdot \boldsymbol{k}^+_i / \tau) + \sum_{j=1}^{M} \exp(\boldsymbol{q}_i \cdot \boldsymbol{k}^-_j / \tau)} \tag{6}$$

기존 방법의 한계(같은 사람이더라도 다른 인스턴스로 취급)를 극복하기 위해, 교정된 레이블을 큐에 함께 기록하여 양성/음성 쌍을 더 정확하게 구분합니다.

**레이블 가이드 대조 손실**:

$$\mathcal{L}^i_{lgc} = \frac{-1}{|\mathcal{P}(i)|} \log \frac{\sum_{\boldsymbol{k}^+ \in \mathcal{P}(i)} \exp\left(\frac{\boldsymbol{q}_i \cdot \boldsymbol{k}^+}{\tau}\right)}{\sum_{\boldsymbol{k}^+ \in \mathcal{P}(i)} \exp\left(\frac{\boldsymbol{q}_i \cdot \boldsymbol{k}^+}{\tau}\right) + \sum_{\boldsymbol{k}^- \in \mathcal{N}(i)} \exp\left(\frac{\boldsymbol{q}_i \cdot \boldsymbol{k}^-}{\tau}\right)} \tag{7}$$

양성/음성 집합 정의:

$$\mathcal{P}(i) = \{\boldsymbol{k}_{j_t} | \hat{y}_{j_t} = \hat{y}_i, \forall (\boldsymbol{k}_{j_t}, \hat{y}_{j_t}) \in \mathcal{Q}\} \cup \{\boldsymbol{k}_i\}$$

$$\mathcal{N}(i) = \{\boldsymbol{k}_{j_t} | \hat{y}_{j_t} \neq \hat{y}_i, \forall (\boldsymbol{k}_{j_t}, \hat{y}_{j_t}) \in \mathcal{Q}\} \tag{8}$$

큐 크기 $M = 65,536$으로 설정하여 양성 샘플 출현 빈도를 높입니다.

#### 최종 손실 함수

$$\mathcal{L}^i = \mathcal{L}^i_{ce} + \lambda_{pro}\mathcal{L}^i_{pro} + \lambda_{lgc}\mathcal{L}^i_{lgc} \tag{9}$$

$\lambda_{pro} = \lambda_{lgc} = 1$로 설정합니다.

### 2.4 성능 향상

#### 지도 Re-ID 방법에 대한 개선 (mAP/cmc1)

| 데이터셋 | 방법 | IN sup. | LUP unsup. | LUPnl pnl. (Ours) |
|----------|------|---------|------------|-------------------|
| CUHK03 | MGN | 70.5/71.2 | 74.7/75.4 | **80.4/80.9** |
| Market1501 | MGN | 87.5/95.1 | 91.0/96.4 | **91.9/96.6** |
| DukeMTMC | MGN | 79.4/89.0 | 82.1/91.0 | **84.3/92.0** |
| MSMT17 | MGN | 63.7/85.1 | 65.7/85.5 | **68.0/86.0** |

#### Ablation Study (MSMT17, small-scale 설정)

| # | ce | ic | pro | lgc | 20% | 40% | 100% |
|---|----|----|-----|-----|-----|-----|------|
| 1 | ✓ | | | | 32.0/56.1 | 45.0/69.5 | 62.7/83.0 |
| 2 | | ✓ | | | 34.5/59.5 | 47.9/72.6 | 65.3/84.0 |
| 3 | ✓ | ✓ | | | 37.6/62.6 | 49.6/73.5 | 66.5/84.7 |
| 7 | ✓ | | ✓ | ✓ | **39.6/63.7** | **51.9/75.0** | **68.0/86.0** |

> **주목할 점**: 노이즈 레이블을 직접 사용하는 분류 손실($\mathcal{L}\_{ce}$, row 1)보다 레이블 없이 인스턴스 대조 학습($\mathcal{L}_{ic}$, row 2)이 더 높은 성능을 보입니다. 이는 노이즈 레이블이 교정 없이 직접 사용될 경우 표현 학습을 방해함을 시사합니다.

### 2.5 한계점

1. **계산 비용**: 8×V100 GPU에서 90 에폭 학습이 필요하여 상당한 컴퓨팅 자원 요구
2. **노이즈 레이블의 완전한 교정 불가**: 현재 추적 알고리즘의 기술적 한계로 인해 완벽한 레이블 정확도 달성 불가
3. **동일 도메인 의존성**: LUPerson-NL이 주로 스트리트뷰 비디오 기반이므로, 완전히 다른 도메인(예: 실내 CCTV)에서의 일반화 가능성은 추가 검증 필요
4. **임계값 민감성**: $T$ 값이 너무 작거나 크면 레이블 교정 성능이 크게 저하됨
5. **UDA(Unsupervised Domain Adaptation) 일부 태스크에서 LUPerson 비지도 모델에 소폭 뒤처짐**: Market1501의 USL 설정에서 LUP unsup. (76.2)가 LUPnl pnl. (75.6)보다 높음

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 Small-scale 및 Few-shot 설정에서의 우수성

논문에서 일반화 성능의 핵심 증거는 **데이터가 적을수록 더 큰 성능 향상**을 보인다는 것입니다.

**Market1501 Small-scale 설정 (MGN 기준, mAP)**:

| 사용 데이터 비율 | IN sup. | LUP unsup. | LUPnl pnl. | 향상폭 (vs LUP unsup.) |
|-----------------|---------|------------|------------|----------------------|
| 10% | 53.1 | 64.6 | **72.4** | +7.8% |
| 30% | 75.2 | 81.9 | **85.2** | +3.3% |
| 100% | 87.5 | 91.0 | **91.9** | +0.9% |

**Market1501 Few-shot 설정 (MGN 기준, mAP)**:

| 사용 이미지 비율 | IN sup. | LUP unsup. | LUPnl pnl. | 향상폭 (vs LUP unsup.) |
|-----------------|---------|------------|------------|----------------------|
| 10% | 21.1 | 26.4 | **42.0** | +15.6% |
| 30% | 68.1 | 78.3 | **83.7** | +5.4% |
| 100% | 87.5 | 91.0 | **91.9** | +0.9% |

이는 PNL이 학습한 표현이 **더 풍부하고 일반화된 Re-ID 특징**을 담고 있음을 의미합니다. 특히 10% few-shot 설정에서 mAP 42.0(75명의 ID, 1,170장의 이미지만으로 750명 ID, 19,281장 이미지로 구성된 테스트셋 평가)은 실용적 관점에서 매우 중요합니다.

### 3.2 일반화 성능 향상의 메커니즘

**1. 대규모 다양성 데이터**:
- 21K 씬에서 수집된 10M 이미지는 다양한 조명, 각도, 배경을 포함하며, 이는 다운스트림 데이터셋에서의 분포 외(out-of-distribution) 샘플에 대한 로버스트성을 높임
- LUPerson-NL vs SYSU30K 비교(MSMT17: 66.1 vs 55.2)에서 단순히 이미지가 많은 것보다 **다양성**이 중요함을 확인

**2. 레이블 교정을 통한 표현 품질 향상**:
- 노이즈 레이블을 교정하면서 동시에 유사한 특징을 가진 샘플들을 같은 클래스로 클러스터링
- Noise-I 교정: 동일인의 서로 다른 트랙렛을 하나로 통합 → ID 일관성 있는 특징 학습
- Noise-II 교정: 서로 다른 사람의 혼합 트랙렛을 분리 → ID 변별력 있는 특징 학습

**3. 레이블 가이드 대조 학습의 역할**:
- 기존 인스턴스 대조 학습은 같은 사람이라도 다른 인스턴스면 밀어내는 문제가 있음
- $\mathcal{L}_{lgc}$는 동일 ID의 모든 인스턴스를 양성으로 묶어 **ID 레벨의 의미론적 일관성** 확보
- 이는 다운스트림 태스크에서 새로운 ID에 대한 일반화 성능을 직접적으로 향상시킴

**4. 사전 학습 데이터 규모의 영향**:

| 사전 학습 데이터 비율 | MSMT17 mAP |
|---------------------|------------|
| 10% | 57.4 |
| 30% | 62.2 |
| 100% | **68.0** |

데이터 규모가 증가할수록 일반화 성능이 지속적으로 향상되며, 이는 스케일링의 가능성을 보여줍니다.

**5. 더 강력한 백본에서의 확장성**:

| 백본 | MSMT17 mAP |
|------|------------|
| ResNet-50 | 68.0 |
| ResNet-101 | 70.8 |
| ResNet-152 | **71.6** |

더 강력한 백본을 사용할수록 성능이 지속적으로 향상되며, PNL이 학습한 표현의 품질이 백본 용량에 따라 확장됨을 보여줍니다.

---

## 4. 앞으로의 연구에 미치는 영향 및 고려사항

### 4.1 연구에 미치는 영향

**1. Re-ID 사전 학습 패러다임 전환**:
- ImageNet 사전 학습 → Re-ID 도메인 특화 대규모 사전 학습으로의 전환을 가속화
- 노이즈 레이블도 적절히 처리하면 효과적인 지도 신호가 될 수 있음을 증명
- 비용 없는 약지도(weakly supervised) 레이블 생성 파이프라인의 가능성 제시

**2. 노이즈 레이블 학습(Learning with Noisy Labels) 분야**:
- 프로토타입 기반 레이블 교정과 대조 학습의 결합이라는 새로운 접근법 제시
- Re-ID 외 다른 fine-grained 인식 태스크(차량 Re-ID, 동물 Re-ID 등)에 적용 가능

**3. 데이터셋 구축 방법론**:
- 인간 레이블링 없이 대규모 도메인 특화 데이터셋 구축 방법론 제시
- 비디오 공개 데이터를 활용한 자동 레이블링 파이프라인의 표준화

**4. 실용적 Re-ID 시스템**:
- Few-shot/small-scale 설정에서의 뛰어난 성능은 실제 현장 배치(deployment) 비용 절감에 기여
- 새로운 환경에서 소수의 레이블 데이터만으로도 높은 성능 달성 가능

### 4.2 향후 연구 시 고려해야 할 점

**1. 노이즈 비율 분석 및 적응적 임계값**:
- 현재 고정된 임계값 $T = 0.8$을 사용하지만, 데이터셋의 노이즈 비율에 따라 적응적으로 조정하는 방법 연구 필요
- 노이즈 비율 추정 및 동적 임계값 조정(curriculum learning 결합 등) 고려

**2. 더 정교한 노이즈 모델링**:
- Noise-I과 Noise-II를 구별하여 각각에 최적화된 교정 전략 개발
- Noise-II(ID 혼합)는 단순 프로토타입 유사도만으로 교정하기 어려울 수 있으므로, 외관 다양성 기반 클러스터링 방법 결합 고려

**3. 비디오 시퀀스 정보의 활용**:
- 현재 방법은 정적 이미지 기반이지만, 트랙렛 내의 시퀀스 정보(temporal dynamics)를 직접 활용하는 비디오 기반 표현 학습 가능성 탐색
- Transformer 기반 시퀀스 모델(Video Transformer)과의 결합

**4. 트랜스포머(Transformer) 기반 백본으로의 확장**:
- 현재 ResNet 백본 중심이지만, ViT(Vision Transformer) 등 최신 백본과 PNL의 결합 연구 필요
- 특히 대규모 Vision Foundation Model과의 결합 시 성능 향상 가능성

**5. 크로스 도메인 일반화**:
- LUPerson-NL이 스트리트뷰 중심이므로, 실내 환경, 적외선 카메라, 낮은 해상도 등 다양한 도메인에서의 일반화 성능 검증 필요
- 도메인 일반화(Domain Generalization) 설정에서의 평가 추가

**6. 프라이버시 및 윤리적 고려사항**:
- 대규모 스트리트뷰 비디오에서의 개인 정보 수집 및 사용에 관한 규정 준수 필요
- 얼굴 블러링, 개인 식별 정보 제거 등 프라이버시 보호 메커니즘 통합

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

### 5.1 대규모 사전 학습 관련 연구

| 연구 | 방법 | 특징 | PNL 대비 |
|------|------|------|----------|
| **LUPerson** (Fu et al., CVPR 2021) [12] | MoCo 기반 비지도 사전 학습 | 레이블 없이 LUPerson 데이터셋 활용 | PNL의 기반 연구; PNL이 mAP에서 일관되게 우수 |
| **DINO** (Caron et al., ICCV 2021) | Self-distillation with no labels | ViT 기반 자기 지식 증류 | Re-ID 특화 사전 학습 없음; 일반 비전 표현 |
| **MoCo v3** (Chen et al., ICCV 2021) | Momentum Contrast + ViT | ViT와 대조 학습 결합 | Re-ID 도메인 특화 없음 |
| **SupCon** (Khosla et al., NeurIPS 2020) [26] | 지도 대조 학습 | 레이블을 활용한 대조 학습 | 노이즈 레이블 처리 없음; PNL이 MSMT17에서 68.0 > 66.5 달성 |

### 5.2 Person Re-ID 특화 연구

| 연구 | 방법 | 특징 | PNL과의 관계 |
|------|------|------|-------------|
| **SpCL** (Ge et al., NeurIPS 2020) [15] | 하이브리드 메모리 + 자기 페이스 클러스터링 | 비지도 도메인 적응 | PNL 사전 학습 모델이 SpCL의 성능을 추가로 향상시킴 |
| **ISP** (Zhu et al., ECCV 2020) [63] | Identity-guided 파싱 | 부분 특징 학습 | MSMT17: ISP 결과 없음; PNL+MGN이 전반적 우위 |
| **TransReID** (He et al., ICCV 2021) | Vision Transformer + Re-ID | ViT 기반 Re-ID | PNL과 상호보완적; ViT 백본에 PNL 적용 가능성 있음 |
| **CLIP-ReID** (Li et al., AAAI 2023) | CLIP 기반 Re-ID | 비전-언어 사전 학습 활용 | 대규모 언어-비전 쌍을 활용한 다른 접근법 |

### 5.3 노이즈 레이블 학습 관련 연구

| 연구 | 방법 | 특징 |
|------|------|------|
| **DivideMix** (Li et al., ICLR 2020) | GMM 기반 샘플 분류 + MixUp | 노이즈/클린 샘플 구분 | PNL보다 복잡한 노이즈 모델링이지만 Re-ID 특화 아님 |
| **CORES** (Cheng et al., CVPR 2021) | 일관성 기반 정규화 | 특징 일관성을 활용한 노이즈 처리 | PNL의 프로토타입 교정과 유사한 철학 |

### 5.4 주요 트렌드와 PNL의 위치

```
비지도 학습          약지도 학습(PNL)       완전 지도 학습
(LUPerson)    →→→   (LUPerson-NL)    →→→  (Human-labeled)
낮은 비용            중간 비용               높은 비용
낮은 성능            높은 성능               최고 성능
```

PNL은 **비용 대비 성능 효율성**이 가장 높은 영역을 점유하며, 특히 실용적 배포 환경에서 그 가치가 큽니다.

---

## 참고자료

- **주 논문**: Dengpan Fu, Dongdong Chen, Hao Yang, Jianmin Bao, Lu Yuan, Lei Zhang, Houqiang Li, Fang Wen, Dong Chen. "Large-Scale Pre-training for Person Re-identification with Noisy Labels." arXiv:2203.16533v2.
- **코드 저장소**: https://github.com/DengpanFu/LUPerson-NL
- **참조 논문 (논문 내 인용)**:
  - [12] Fu et al., "Unsupervised pre-training for person re-identification," CVPR 2021 (LUPerson)
  - [15] Ge et al., "Self-paced contrastive learning with hybrid memory," NeurIPS 2020 (SpCL)
  - [19] He et al., "Momentum contrast for unsupervised visual representation learning," CVPR 2020 (MoCo)
  - [26] Khosla et al., "Supervised contrastive learning," NeurIPS 2020 (SupCon)
  - [51] Wang et al., "Learning discriminative features with multiple granularities," ACM MM 2018 (MGN)
  - [56] Zhang et al., "FairMOT: On the fairness of detection and re-identification," arXiv 2020

> **주의**: 2020년 이후 최신 연구 비교 분석 부분(TransReID, CLIP-ReID 등)은 해당 논문 원문에 직접 언급되지 않은 내용을 포함하므로, 해당 논문들의 원문을 직접 확인하시기를 권장합니다. 비교 분석은 각 논문의 공개된 정보를 기반으로 작성하였습니다.
