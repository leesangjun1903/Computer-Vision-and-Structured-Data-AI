# Background-Aware Pooling and Noise-Aware Loss for Weakly-Supervised Semantic Segmentation

---

## 1. 핵심 주장과 주요 기여 요약

### 핵심 주장

이 논문(이하 BANA)은 **바운딩 박스(bounding box) 어노테이션**만을 사용하는 약지도 의미론적 분할(Weakly-Supervised Semantic Segmentation, WSSS)에서 두 가지 핵심 문제를 해결합니다:

1. **고품질 의사 레이블(pseudo label) 생성**: 바운딩 박스 내부의 전경(foreground)과 배경(background)을 효과적으로 분리
2. **노이즈에 강인한 학습**: 생성된 의사 레이블의 노이즈(특히 객체 경계 부분)에 덜 민감한 손실 함수 설계

### 주요 기여

| 기여 | 설명 |
|------|------|
| **BAP (Background-Aware Pooling)** | 배경이 이미지 내에서 지각적으로 일관성을 갖는다는 prior를 활용, 주의 맵(attention map)으로 전경 특징을 집중적으로 집계하는 새로운 풀링 방법 |
| **NAL (Noise-Aware Loss)** | CNN 특징과 분류기 가중치 간 거리를 활용한 신뢰도 맵(confidence map)으로 잘못된 레이블의 영향을 완화하는 손실 함수 |
| **SOTA 달성** | PASCAL VOC 2012에서 약지도 및 반지도 의미론적 분할 모두에서 당시 최고 성능 달성 |

---

## 2. 해결 문제, 제안 방법, 모델 구조, 성능 및 한계

### 2.1 해결하고자 하는 문제

바운딩 박스 기반 WSSS의 두 가지 근본적인 어려움:

- **문제 1**: 바운딩 박스는 전경과 배경이 혼재하며 객체 경계를 명시하지 않음 → 고품질 의사 레이블 생성이 어려움
- **문제 2**: 생성된 의사 레이블, 특히 객체 경계 부근에 노이즈가 존재 → CNN 학습 시 성능 저하

기존 방법들(GrabCut, MCG, DenseCRF)은 외부 분할 도구에 의존하거나 정확도가 낮은 경계를 생성하는 한계가 있었습니다.

---

### 2.2 제안하는 방법 (수식 포함)

#### Stage 1: BAP를 이용한 이미지 분류 네트워크

**핵심 아이디어**: 배경 영역은 이미지 내에서 지각적으로 일관성이 있다는 prior를 이용하여, 바운딩 박스 외부(확실한 배경)의 특징을 쿼리로 삼아 박스 내부의 배경 영역을 검색(retrieval)합니다.

**Step 1: 쿼리 추출**

특징 맵 $f$를 $N \times N$ 격자로 나누고, 바운딩 박스 외부(확실한 배경)에 해당하는 격자 셀 $G(j)$의 특징을 가중 평균으로 집계:

$$q_j = \frac{\sum_{\mathbf{p} \in G(j)} M(\mathbf{p}) f(\mathbf{p})}{\sum_{\mathbf{p} \in G(j)} M(\mathbf{p})} \tag{1}$$

여기서 $M(\mathbf{p}) = 1$이면 위치 $\mathbf{p}$가 바운딩 박스 외부(확실한 배경)임을 의미합니다.

**Step 2: 배경 주의 맵(Attention Map) 생성**

쿼리 $q_j$와 바운딩 박스 내부 특징 간의 코사인 유사도로 배경 가능성을 계산:

$$A_j(\mathbf{p}) = \begin{cases} \text{ReLU}\left(\frac{f(\mathbf{p})}{\|f(\mathbf{p})\|} \cdot \frac{q_j}{\|q_j\|}\right) & , \mathbf{p} \in \mathcal{B} \\ 1 & , \mathbf{p} \notin \mathcal{B} \end{cases} \tag{3}$$

$$A(\mathbf{p}) = \frac{1}{J} \sum_j A_j(\mathbf{p}) \tag{2}$$

$A(\mathbf{p})$가 1에 가까울수록 해당 픽셀이 배경일 가능성이 높습니다.

**Step 3: BAP로 전경 특징 집계**

각 바운딩 박스 $B_i$에 대해 전경 가중 평균 풀링:

$$r_i = \frac{\sum_{\mathbf{p} \in B_i} (1 - A(\mathbf{p})) f(\mathbf{p})}{\sum_{\mathbf{p} \in B_i} (1 - A(\mathbf{p}))} \tag{4}$$

> **참고**: $A = 0$이면 BAP는 GAP(Global Average Pooling)과 동일해집니다.

---

#### Stage 2: 의사 레이블 생성

두 가지 상호보완적인 의사 레이블을 생성합니다.

**레이블 1: $Y_\text{crf}$ (DenseCRF 기반)**

각 클래스 $c$에 대한 CAM:

$$\text{CAM}_c(\mathbf{p}) = \text{ReLU}(f(\mathbf{p}) \cdot w_c) \tag{6}$$

객체 클래스의 unary term:

$$u_c(\mathbf{p}) = \begin{cases} \frac{\text{CAM}_c(\mathbf{p})}{\max_{\mathbf{p}}(\text{CAM}_c(\mathbf{p}))} & , \mathbf{p} \in \mathcal{B}_c \\ 0 & , \mathbf{p} \notin \mathcal{B}_c \end{cases} \tag{5}$$

배경 클래스의 unary term (주의 맵 직접 활용):

$$u_0(\mathbf{p}) = A(\mathbf{p}) \tag{7}$$

$u_c$와 $u_0$을 DenseCRF에 입력하여 $Y_\text{crf}$ 생성.

**레이블 2: $Y_\text{ret}$ (Retrieval 기반)**

$Y_\text{crf}$로 클래스별 프로토타입 특징 추출:

$$q_c = \frac{1}{|\mathcal{Q}_c|} \sum_{\mathbf{p} \in \mathcal{Q}_c} f(\mathbf{p}) \tag{8}$$

각 클래스와의 상관 맵 계산:

$$C_c(\mathbf{p}) = \frac{f(\mathbf{p})}{\|f(\mathbf{p})\|} \cdot \frac{q_c}{\|q_c\|} \tag{9}$$

$C_c$에 argmax를 적용하여 $Y_\text{ret}$ 생성.

---

#### Stage 3: NAL을 이용한 의미론적 분할 학습

DeepLab을 $Y_\text{crf}$와 $Y_\text{ret}$로 학습합니다.

**신뢰 영역 $\mathcal{S}$**: $Y_\text{crf}$와 $Y_\text{ret}$가 동일한 레이블을 제공하는 영역 → 표준 크로스 엔트로피 손실:

$$\mathcal{L}_\text{ce} = -\frac{1}{\sum_c |\mathcal{S}_c|} \sum_c \sum_{\mathbf{p} \in \mathcal{S}_c} \log H_c(\mathbf{p}) \tag{10}$$

**불신뢰 영역 $\sim\mathcal{S}$**: $Y_\text{crf}$와 $Y_\text{ret}$가 다른 레이블을 제공하는 영역 → NAL 적용

CNN 특징과 분류기 가중치 간의 상관 맵:

$$D_c(\mathbf{p}) = 1 + \left(\frac{\phi(\mathbf{p})}{\|\phi(\mathbf{p})\|} \cdot \frac{W_c}{\|W_c\|}\right) \tag{11}$$

신뢰도 맵:

$$\sigma(\mathbf{p}) = \left(\frac{D_{c^*}(\mathbf{p})}{\max_c(D_c(\mathbf{p}))}\right)^\gamma \tag{12}$$

여기서 $c^* = Y_\text{crf}(\mathbf{p})$이고, $\gamma \geq 1$은 damping 파라미터입니다.

신뢰도 가중 크로스 엔트로피:

$$\mathcal{L}_\text{wce} = -\frac{1}{\sum_c \sum_{\mathbf{p} \in \sim\mathcal{S}_c} \sigma(\mathbf{p})} \sum_c \sum_{\mathbf{p} \in \sim\mathcal{S}_c} \sigma(\mathbf{p}) \log H_c(\mathbf{p}) \tag{13}$$

**최종 NAL**:

$$\mathcal{L} = \mathcal{L}_\text{ce} + \lambda \mathcal{L}_\text{wce} \tag{14}$$

논문에서는 $\gamma = 7$, $\lambda = 0.1$로 설정하였습니다.

---

### 2.3 모델 구조

```
[입력 이미지 + 바운딩 박스]
        ↓
[특징 추출기 (VGG-16 기반, AffinityNet 구조)]
        ↓
[BAP: 배경 쿼리 → 주의 맵 → 전경 특징 집계]
        ↓
[(L+1)-way Softmax 분류기 (코사인 유사도 기반)]
        ↓
[CAM + 배경 주의 맵 → DenseCRF → Ycrf]
                              ↓
[프로토타입 특징 retrieval → Yret]
        ↓
[DeepLab (V1: VGG-16 / V2: ResNet-101) + NAL]
        ↓
[최종 의미론적 분할 결과]
```

---

### 2.4 성능 향상

**PASCAL VOC 2012 의사 레이블 품질 (mIoU)**:

| 방법 | train | val |
|------|-------|-----|
| GrabCut | 65.7 | 66.1 |
| WSSL | 69.7 | 71.1 |
| GAP (제안 방법 베이스라인) | 75.5 | 76.1 |
| BAP: $Y_\text{crf}$ | **78.7** | **79.2** |

**DeepLab-V1 (VGG-16) 분할 결과 (mIoU)**:

| 방법 | val | test |
|------|-----|------|
| SDI | 65.7 | 67.5 |
| BCM | 66.8 | - |
| Ours w/ NAL | **68.1** | **69.4** |

**DeepLab-V2 (ResNet-101) 분할 결과 (mIoU)**:

| 방법 | val | test |
|------|-----|------|
| SDI† | 74.2 | - |
| Box2Seg | 76.4 | - |
| Ours† w/ NAL | **74.6** | **76.1** |

---

### 2.5 한계

1. **객체 경계 부정확성**: DenseCRF 기반 $Y_\text{crf}$는 저수준 색상/텍스처 정보에 의존하므로 경계가 부정확할 수 있음
2. **저해상도 특징 맵**: $Y_\text{ret}$는 저해상도 특징 맵 $f$를 사용하므로 단독으로는 $Y_\text{crf}$보다 성능이 낮음 (train mIoU: 70.8 vs 78.7)
3. **바운딩 박스 어노테이션 필요**: 이미지 레이블보다는 강한 감독 신호를 필요로 함
4. **하이퍼파라미터 민감성**: DenseCRF 파라미터, 격자 크기 $N$, $\gamma$, $\lambda$ 등 다수의 하이퍼파라미터 튜닝 필요
5. **Box2Seg 대비 성능**: UPerNet을 사용하는 Box2Seg(76.4 val)에 비해 낮은 성능 (단, 더 간단한 네트워크 사용)
6. **Mask-RCNN 적용 제한**: 이진 크로스 엔트로피 손실 구조상 NAL의 상관 맵($D_c$)을 직접 적용하기 어려움

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 Cross-Dataset 일반화: VOC → COCO

논문의 핵심 일반화 실험은 **VOC-to-COCO** 크로스 데이터셋 평가입니다:

| 방법 | AP | $\text{AP}_{50}$ | $\text{AP}_{75}$ | $\text{AP}_S$ | $\text{AP}_M$ | $\text{AP}_L$ |
|------|----|------|----|---|---|---|
| VOC-to-COCO ($Y_\text{crf}$) | 11.7 | 28.7 | 8.0 | 3.0 | 15.0 | 27.1 |
| COCO-to-COCO ($Y_\text{crf}$) | 17.2 | 40.5 | 12.5 | 5.9 | 20.4 | 32.2 |

**핵심 분석**: VOC로만 훈련된 모델이 COCO의 훈련 샘플을 전혀 사용하지 않았음에도 합리적인 의사 레이블을 생성합니다. 이는 배경 주의 맵 $u_0$이 클래스에 무관하게(class-agnostic) $1 - u_0$를 전경 주의 맵으로 활용할 수 있기 때문입니다.

### 3.2 일반화 성능의 근거

**1. 클래스 비의존적(Class-Agnostic) 전경 탐지**

BAP의 배경 주의 맵 $A(\mathbf{p})$는 특정 클래스 정보 없이도 전경/배경을 분리할 수 있습니다. 훈련 시 보지 못한 클래스에 대해서도 $1 - u_0$를 전경 주의 맵으로 사용 가능합니다.

**2. 비파라메트릭 검색(Nonparametric Retrieval)**

$Y_\text{ret}$는 클래스 프로토타입과 특징 맵 간의 코사인 유사도를 이용합니다. 이 방식은 새로운 클래스에 대한 프로토타입만 있다면 확장 가능합니다:

$$C_c(\mathbf{p}) = \frac{f(\mathbf{p})}{\|f(\mathbf{p})\|} \cdot \frac{q_c}{\|q_c\|}$$

**3. 두 의사 레이블의 상호보완성**

- $Y_\text{crf}$: 대형 객체에 유리 (DenseCRF의 저수준 특징 활용)
- $Y_\text{ret}$: 소형 객체에 유리 (고수준 의미 특징 활용)

이 상호보완성은 다양한 스케일의 객체에 대한 일반화를 지원합니다.

**4. 코사인 유사도 기반 분류기**

$$H_c(\mathbf{p}) = \frac{e^{\tau \frac{\phi(\mathbf{p})}{\|\phi(\mathbf{p})\|} \cdot \frac{W_c}{\|W_c\|}}}{\sum_i e^{\tau \frac{\phi(\mathbf{p})}{\|\phi(\mathbf{p})\|} \cdot \frac{W_i}{\|W_i\|}}}$$

특징과 가중치를 하이퍼스피어 위에 놓음으로써 클래스 내 변화(intra-class variation)에 더 강인하고, 이는 일반화 성능 향상에 기여합니다.

**5. NAL의 적응적 노이즈 처리**

$\sigma(\mathbf{p})$는 훈련 중에 모델의 확신도에 따라 동적으로 조정됩니다. 새로운 데이터에서도 모델이 자체적으로 신뢰할 수 있는 레이블을 선별하는 메커니즘을 제공합니다.

### 3.3 일반화의 한계

- 완전히 새로운 도메인(예: 의료 이미지, 위성 이미지)으로의 전이는 검증되지 않음
- 배경의 일관성(perceptual consistency) 가정이 성립하지 않는 복잡한 장면에서는 성능 저하 가능

---

## 4. 앞으로의 연구에 미치는 영향과 고려할 점

### 4.1 연구에 미치는 영향

**1. 배경 Prior의 명시적 활용**

배경 영역의 일관성을 prior로 활용하는 아이디어는 이후 연구에서 더 정교한 배경 모델링 방법으로 발전될 수 있습니다. 예를 들어, 트랜스포머 기반의 자기 주의(self-attention) 메커니즘과 결합하면 더 풍부한 배경 표현이 가능합니다.

**2. 두 단계 의사 레이블 생성 패러다임**

상호보완적인 두 가지 의사 레이블($Y_\text{crf}$와 $Y_\text{ret}$)을 생성하고 불일치 영역에 적응적 손실을 적용하는 패러다임은 반지도학습 및 약지도학습 전반에 적용 가능한 일반적인 프레임워크를 제시합니다.

**3. 신뢰도 기반 손실 함수**

NAL의 신뢰도 맵 $\sigma(\mathbf{p})$은 특징 공간에서의 거리를 활용하는 메타-학습(meta-learning) 및 지식 증류(knowledge distillation) 연구와 연결될 수 있습니다.

**4. 크로스 데이터셋 일반화 벤치마크**

VOC → COCO 크로스 데이터셋 평가 프로토콜은 향후 WSSS 연구의 일반화 능력 평가 기준으로 활용될 수 있습니다.

### 4.2 2020년 이후 관련 최신 연구 비교 분석

아래 비교는 논문에 직접 인용된 연구 및 논문 내용에서 확인 가능한 정보에 한정합니다. **2021년 이후의 구체적 수치는 본 논문 PDF에 포함되지 않으므로, 논문 내에 언급된 방법들과의 비교만 제시합니다.**

| 방법 | 연도 | 감독 신호 | 특이사항 | val mIoU (DeepLabV2) |
|------|------|----------|----------|----------------------|
| SDI | CVPR 2017 | Box | GrabCut + MCG | 74.2 |
| BCM | CVPR 2019 | Box | Filling rate constraint | 70.2 |
| Box2Seg | ECCV 2020 | Box | UPerNet (FPN+PPM) | **76.4** |
| **BANA (Ours)** | arXiv 2021 | Box | BAP + NAL | 74.6 |

**Box2Seg와의 비교**:
- Box2Seg는 attention weighted loss와 UPerNet 아키텍처를 사용하여 BANA보다 높은 성능(76.4 vs 74.6)을 달성하나, FPN, PPM, 3개의 디코더를 요구하는 복잡한 구조입니다.
- BANA는 표준 DeepLab 프레임워크를 유지하면서도 경쟁력 있는 성능을 달성합니다.

> **⚠️ 주의**: 2021년 이후 발표된 최신 논문들(예: CVPR 2022, NeurIPS 2022 등)과의 정량적 비교는 본 PDF에 포함된 정보를 벗어나므로, 정확한 수치를 제공하지 않습니다. 최신 비교를 위해서는 Papers With Code (https://paperswithcode.com/task/weakly-supervised-semantic-segmentation)를 참조하시길 권장합니다.

### 4.3 앞으로 연구 시 고려할 점

**1. 트랜스포머 기반 아키텍처와의 결합**

현재 BAP는 CNN 특징 맵을 기반으로 하지만, Vision Transformer(ViT)의 자기 주의 맵은 더 풍부한 전경/배경 분리 신호를 제공할 수 있습니다. DINO와 같은 자기지도 ViT의 주의 헤드를 배경 주의 맵으로 활용하는 연구가 유망합니다.

**2. 더 약한 감독 신호로의 확장**

바운딩 박스 대신 이미지 레벨 레이블과 BAP 아이디어를 결합하는 연구 방향이 가능합니다. 이미지 내 배경 일관성 prior는 레이블 형태에 독립적으로 적용 가능합니다.

**3. 적응적 하이퍼파라미터 설계**

현재 $\gamma$, $\lambda$, DenseCRF 파라미터 등이 수동으로 설정됩니다. 메타 학습이나 Bayesian 최적화를 통해 이를 자동화하면 더 넓은 데이터셋에 대한 일반화가 가능합니다.

**4. 실시간 응용을 위한 경량화**

DenseCRF는 CPU에서 0.4초를 소요하며, 실시간 응용을 위해 학습 가능한 CRF 또는 DenseCRF를 대체하는 경량 경계 정제 모듈 연구가 필요합니다.

**5. 도메인 적응(Domain Adaptation)과의 결합**

VOC → COCO 크로스 데이터셋 실험에서 성능 차이(COCO-to-COCO가 더 우수)가 존재합니다. 도메인 적응 기법을 결합하면 더 강한 일반화 성능을 기대할 수 있습니다.

**6. 밀집 예측 과제로의 확장**

BAP의 배경 prior는 깊이 추정(depth estimation), 광학 흐름(optical flow) 등 다른 밀집 예측 과제에도 적용 가능성이 있습니다.

---

## 참고 자료

- **주요 논문**: Youngmin Oh, Beomjun Kim, Bumsub Ham. "Background-Aware Pooling and Noise-Aware Loss for Weakly-Supervised Semantic Segmentation." arXiv:2104.00905v1, 2021. (제공된 PDF)
- **프로젝트 페이지**: https://cvlab.yonsei.ac.kr/projects/BANA
- 논문 내 참조 문헌:
  - Chen et al., "DeepLab," IEEE Trans. PAMI, 2018
  - Ahn & Kwak, "AffinityNet," CVPR 2018
  - Khoreva et al., "SDI," CVPR 2017
  - Song et al., "BCM," CVPR 2019
  - Kulharia et al., "Box2Seg," ECCV 2020
  - Papandreou et al., "WSSL," ICCV 2015
  - Krähenbühl & Koltun, "DenseCRF," NeurIPS 2011
  - Reed et al., "Bootstrapping," ICLR 2015
