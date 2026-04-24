
# RobustSAM: Segment Anything Robustly on Degraded Images

> **논문 정보**
> - **제목**: RobustSAM: Segment Anything Robustly on Degraded Images
> - **저자**: Wei-Ting Chen, Yu-Jiet Vong, Sy-Yen Kuo, Sizhuo Ma, Jian Wang
> - **학회**: CVPR 2024 (Highlight)
> - **arXiv**: 2406.09627

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

SAM(Segment Anything Model)은 강력한 제로샷(zero-shot) 세그멘테이션 능력과 유연한 프롬프팅 시스템으로 이미지 분할 분야에서 혁신적인 모델로 평가받는다. 그러나 품질이 저하된(degraded) 이미지에서는 성능이 크게 저하된다는 한계가 존재한다. 이를 해결하고자 RobustSAM을 제안하며, SAM의 프롬프트 가능성(promptability)과 제로샷 일반화 능력을 유지하면서 저화질 이미지에서의 성능을 개선하는 것이 핵심 목표이다.

### 주요 기여 요약

| 기여 항목 | 내용 |
|---|---|
| **핵심 모듈** | AOTG, AMFG, ROT 모듈 설계 |
| **학습 전략** | Clear-Degraded 이미지 쌍 기반 일관성 손실 학습 |
| **신규 데이터셋** | Robust-Seg (688K 이미지) 구축 |
| **하류 작업 개선** | Dehazing, Deblurring 등 향상 |
| **효율성** | 8개 GPU, 30시간 이내 학습 가능 |

RobustSAM의 핵심 기여는 **Anti-Degradation Output Token Generation (AOTG)**과 **Anti-Degradation Mask Feature Generation (AMFG)** 모듈로, 저화질 이미지에서 원본 SAM이 깨끗한 이미지로부터 추출하는 것과 정렬된(aligned) 열화 불변(degradation-invariant) 정보를 추출한다.

---

## 2. 해결하고자 하는 문제, 제안 방법, 모델 구조, 성능 향상 및 한계

### 2-1. 해결하고자 하는 문제

이미지 복원(image restoration) 기법들이 이미지 품질을 어느 정도 향상시킬 수 있지만, 선택된 복원 기법이 반드시 세그멘테이션 성능을 향상시킨다는 보장은 없다. 대부분의 이미지 복원 알고리즘은 인간의 시각적 인식을 위해 최적화되어 있어, SAM과 같은 세그멘테이션 모델의 특정 요구사항과 맞지 않는다.

저하된 이미지로 SAM을 직접 파인튜닝하는 대안도 존재하지만, SAM 디코더를 직접 조정하거나 새로운 디코더 모듈을 통합하면 제로샷 태스크에서 모델의 일반화 능력이 심각하게 저하될 수 있다. 더 나아가, 저하된 이미지로 SAM을 무작정 파인튜닝하면 **catastrophic forgetting**—원본의 깨끗한 이미지에서 학습한 지식을 실수로 상실하는 현상—이 발생할 수 있다.

---

### 2-2. 제안하는 방법

#### (1) 학습 전략: Clear-Degraded Pair 기반 학습

훈련 시 깨끗한 이미지는 원본 SAM 모듈(고정)을 통과하여 깨끗한 장면의 피처를 생성한다. 이후 깨끗한 입력의 증강으로 생성된 저화질 이미지가 RobustSAM을 통과하여 저화질 시나리오에 대한 피처를 생성한다. 이 피처들은 Anti-Degradation 모듈을 통해 정제되어 깨끗한 장면의 피처와 일관성이 유지된다. 이 방법론은 세그멘테이션 손실로 지원되어 깨끗한 이미지와 저화질 이미지 모두에서 정확한 세그멘테이션 결과를 달성한다.

이는 15가지 유형의 합성 열화 증강(synthetic degradation augmentation)을 통해 깨끗한-저화질 이미지 쌍을 생성함으로써 달성된다. 그런 다음 다양한 손실 함수가 적용되어 깨끗한 피처와 저화질 피처 간의 일관성, 그리고 예측 세그멘테이션과 정답 간의 일관성을 강제한다.

#### (2) 손실 함수 (Loss Functions)

전체 손실 함수는 세 가지 구성 요소를 포함한다:
**Mask Feature Consistency Loss ($\mathcal{L}_{MFC}$)**: AMFG에서 정제된 피처가 깨끗한 이미지의 피처와 정렬되도록 보장한다.
**Token Consistency Loss ($\mathcal{L}_{TC}$)**: 정제된 출력 토큰과 깨끗한 이미지 처리 시 원본 SAM의 출력 토큰 간의 정렬을 강화한다.

전체 훈련 목적 함수는 다음과 같이 구성됩니다:

$$
\mathcal{L}_{total} = \mathcal{L}_{seg} + \lambda_1 \mathcal{L}_{MFC} + \lambda_2 \mathcal{L}_{TC}
$$

여기서:
- $\mathcal{L}_{seg}$: 세그멘테이션 마스크 예측 손실 (Focal Loss + Dice Loss 조합)
- $\mathcal{L}_{MFC}$: AMFG 출력 피처와 SAM의 클리어 이미지 피처 간 일관성 손실
- $\mathcal{L}_{TC}$: AOTG 출력 토큰과 원본 SAM 출력 토큰 간 일관성 손실
- $\lambda_1, \lambda_2$: 각 손실의 가중치 하이퍼파라미터

피처 일관성 손실의 일반적 형태:

$$
\mathcal{L}_{MFC} = \left\| F_{robust}^{mask} - F_{clear}^{mask} \right\|_2^2
$$

$$
\mathcal{L}_{TC} = \left\| T_{robust} - T_{clear} \right\|_2^2
$$

---

### 2-3. 모델 구조

#### 원본 SAM 구조

SAM은 이미지 인코더, 프롬프트 인코더, 마스크 디코더의 세 가지 핵심 구성 요소를 포함한다. 이미지 인코더는 Vision Transformer(ViT)를 사용하여 입력 이미지를 처리한다.

#### RobustSAM 추가 구성 요소

RobustSAM 훈련 시 원본 사전 훈련된 SAM 모델의 파라미터는 고정되며, 제안된 RobustSAM 구성 요소만 학습된다: **Robust Output Token (ROT)**, **Anti-Degradation Output Token Generation (AOTG)** 모듈, **Anti-Degradation Mask Feature Generation (AMFG)** 모듈, 그리고 최종 강건한 마스크를 생성하는 **3-layer MLP**.

##### (a) AMFG 모듈 (Anti-Degradation Mask Feature Generation)

입력 피처는 먼저 Instance Normalization(IN)으로 처리된다. IN의 목적은 이미지 열화와 관련된 변동을 표준화하는 것이다.

Squeeze-and-Excitation 모듈의 출력 피처는 푸리에 변환(Fourier Transform)을 사용하여 위상(phase) 성분과 진폭(amplitude) 성분으로 변환된다. 이후 진폭 성분에 1×1 컨볼루션(제로 패딩, stride 1)을 적용하여 열화 요소를 제거한다. 다음으로 역 푸리에 변환(Inverse Fourier Transform)을 수행하여 정제된 피처를 원래의 공간 표현으로 복원한다. 마지막으로 2×2 커널, stride 2의 두 개의 전치 컨볼루션 레이어 조합이 차원 정렬 및 AMFG 모듈의 최종 출력 피처 생성에 사용된다.

AMFG의 Fourier Degradation Suppression 과정:

$$
\mathcal{F}_{amp}, \mathcal{F}_{phase} = \text{FFT}(f_{SE})
$$

$$
\hat{\mathcal{F}}_{amp} = \text{Conv}_{1\times1}(\mathcal{F}_{amp})
$$

$$
f_{refined} = \text{IFFT}(\hat{\mathcal{F}}_{amp}, \mathcal{F}_{phase})
$$

$$
f_{AMFG} = \text{TransposeConv}(f_{refined})
$$

##### (b) AOTG 모듈 (Anti-Degradation Output Token Generation)

AOTG 모듈은 두 개의 IN(Instance Normalization) 레이어와 MLP 네트워크로 구성된다. 원본 강건 출력 토큰이 먼저 IN 레이어를 통과하여 열화 관련 세부 정보에 민감한 정보를 필터링한다. 이후 MLP가 강건 출력 토큰의 차원을 조정하는 데 사용된다.

$$
T_{AOTG} = \text{MLP}(\text{IN}(\text{IN}(T_{ROT})))
$$

##### (c) Robust Output Token (ROT)

원본 SAM 프레임워크와 달리, 출력 토큰을 파인튜닝하여 **Robust Output Token (ROT)**이라 명명한다.

#### 전체 RobustSAM 모델 구조 요약

```
[입력: 저화질 이미지]
       ↓
[ViT Image Encoder (고정)]
       ↓
  ┌────────────────────────────────┐
  │  [AMFG Module]                  │
  │   IN → BN → SE → FFT →         │
  │   Conv(1×1) → IFFT →           │
  │   TransposeConv                 │
  └────────────────────────────────┘
       ↓ (Anti-Degradation Mask Features)
  ┌────────────────────────────────┐
  │  [AOTG Module]                  │
  │   ROT → IN → IN → MLP          │
  └────────────────────────────────┘
       ↓ (Anti-Degradation Output Token)
[Transformer Mask Decoder (수정)]
       ↓
[3-layer MLP]
       ↓
[출력: Robust Segmentation Mask]
```

---

### 2-4. Robust-Seg 데이터셋

Robust-Seg는 7개의 기존 데이터셋에서 꼼꼼하게 어노테이션된 43,000개의 이미지를 결합한다. 각 이미지에 15가지 유형의 신중하게 모델링된 합성 열화가 적용되어 Robust-Seg의 688,000개 이미지로 구성된 포괄적인 컬렉션이 만들어진다. 이 광범위한 데이터셋은 이미지 세그멘테이션의 경계를 확장하고 미래 연구를 위한 귀중한 자원이 되는 것을 목표로 한다.

데이터셋에 포함된 15가지 열화 유형:

| 열화 카테고리 | 세부 유형 |
|---|---|
| 노이즈 관련 | Gaussian noise, Speckle noise, Salt & Pepper noise |
| 블러 관련 | Gaussian blur, Motion blur, Defocus blur |
| 날씨/환경 | Haze, Rain, Snow, Fog |
| 저화질 | Low-light, Low resolution |
| 압축/기타 | JPEG compression artifacts, etc. |

> ⚠️ 정확한 15가지 목록은 논문 원문에서 확인을 권장합니다.

---

### 2-5. 성능 향상

SAM-B와 RobustSAM-B의 BDD-100k+LIS 데이터셋 비교에서 IoU: 0.3003 → 0.3317, PA: 0.8826 → 0.8972로 향상되었으며, COCO에서도 AP가 0.4589 → 0.4710으로 개선되었다.

Ablation Study 결과, 각 모듈을 추가할수록 성능이 일관되게 향상됨을 보인다: AMFG만 추가 시 IoU 0.3455, AMFG-F 추가 시 0.3535, AMFG-F+AOTG 추가 시 0.3651, 모든 모듈(ALL) 추가 시 0.3717로 각 모듈이 RobustSAM의 성능을 향상시킨다.

벤치마킹에 사용된 핵심 데이터셋은 MSRA10K, LVIS, NDD20, STREETS, FSS-1000, COCO, BDD-100k, LIS를 포함하며, 이 모두는 합성 및 실제 열화를 포함한다. MSRA10K 데이터셋에서 15가지 합성 열화 이미지에 대해 RobustSAM은 다른 모델을 크게 앞선다. NDD20, STREETS, FSS-1000, COCO 같은 미공개 데이터셋에 대한 제로샷 세그멘테이션에서도 탁월한 성능을 보인다.

RobustSAM의 추가 모듈은 효율적으로 훈련될 수 있다. 수백 개의 GPU를 요구하는 원본 SAM과 달리, RobustSAM은 8개의 A100에서 30시간 이내에 훈련이 가능하다. 이는 RobustSAM의 접근성을 나타내며, 다양한 응용 시나리오에 통합될 준비가 되어 있다.

---

### 2-6. 한계

1. **합성 열화에 기반한 학습**: 비록 합성 열화로 지도 학습을 받지만, RobustSAM은 실제 이미지에도 잘 일반화된다고 논문은 주장한다. 그러나 합성 열화와 실제 열화 사이의 도메인 갭이 완전히 해소되었다고 보기 어렵다.

2. **SAM2와의 통합 부재**: 본 연구는 SAM(v1) 기반으로, SAM2(2024, 영상 세그멘테이션 지원)와의 통합이나 비교는 포함되지 않는다.

3. **고정된 열화 유형 수**: 15가지 유형의 열화만 커버하므로, 사전에 정의되지 않은 새로운 유형의 열화에 대해서는 추가적인 연구가 필요하다.

4. **실시간 처리 한계**: RobustSAM은 다양한 백본에서 SAM보다 일관되게 우수한 성능을 보이지만, 추가 모듈로 인한 경미한 연산 부담 증가가 있다.

---

## 3. 모델의 일반화 성능 향상 가능성

### 3-1. 제로샷 일반화의 핵심 메커니즘

두 가지 핵심 모듈(Anti-Degradation Token Generation Module, Anti-Degradation Mask Feature Generation Module)은 원본 SAM이 깨끗한 이미지에서 추출한 쌍 피처와의 일관성 손실로 지도되며, 열화 불변 세그멘테이션 피처를 추출하도록 설계되었다. 또한 SAM의 원본 출력 토큰을 파인튜닝하여 강건한 세그멘테이션 방식에 적응시켰다. SAM의 원본 모듈을 훈련 중 동결(freeze)함으로써, 저화질 이미지 처리 능력을 향상시키면서 제로샷 세그멘테이션에서의 효과성을 보존한다.

### 3-2. 일반화 성능 향상의 핵심 설계 원칙

**① Catastrophic Forgetting 방지**

저화질 이미지로 SAM을 무작정 파인튜닝하면 네트워크가 원본 깨끗한 이미지에서 학습한 지식을 실수로 상실하는 **catastrophic forgetting**이 발생할 수 있다. RobustSAM은 이를 방지하면서 저화질 이미지 처리에 강건성을 달성한다.

**② 인스턴스 정규화(IN)를 통한 열화 표준화**

Anti-Degradation Mask Feature Generation 및 Output Token Generation 모듈은 열화 효과를 정규화하면서 필수 콘텐츠를 보존함으로써 이미지 품질과 세그멘테이션을 크게 향상시킨다. Instance 및 Batch Normalization을 활용하여 다양한 조건에서 콘텐츠를 안정화한다. Fourier Degradation Suppression 모듈이 추가로 구조적 무결성 유지에 집중하며 열화를 격리한다.

**③ 다양한 ViT 백본에서의 확장성**

접미사 -B, -L, -H는 각각 ViT-B(Base), ViT-L(Large), ViT-H(Huge) 버전에 해당하며, Vision Transformer 아키텍처의 다양한 규모와 복잡도를 나타낸다. 이를 통해 다양한 스케일의 모델에서 일관된 일반화 성능이 검증되었다.

**④ 하류 작업(Downstream Tasks)으로의 일반화**

더 나아가, 이 방법은 단일 이미지 디헤이징(dehazing) 및 디블러링(deblurring)과 같은 SAM 기반 하류 작업의 성능을 효과적으로 향상시키는 것으로 나타났다.

---

## 4. 앞으로의 연구에 미치는 영향과 연구 시 고려할 점

### 4-1. 앞으로의 연구에 미치는 영향

**① Foundation Model 강건성(Robustness) 연구 촉진**

RobustSAM의 우월성은 디헤이징, 디블러링과 같은 SAM 기반 태스크 향상으로까지 확장되며, 저화질 조건에서의 이미지 처리에 대한 신뢰할 수 있는 도구로서의 가치를 확인한다. 그 성능은 제로샷 세그멘테이션에서의 강건성에 대한 새로운 기준을 제시하며, 미래 연구를 위한 유망한 방향을 제시한다.

**② Plug-in 방식의 모듈 설계 패러다임 확립**

이 방법은 미미한 파라미터 증가와 연산 요구 사항만으로 사전 훈련된 SAM 모델을 활용한다. 이는 대규모 Foundation Model을 재훈련 없이 특정 도메인에 적응시키는 **Adapter/Plugin 패러다임**의 효율적 사례로, 이후 연구에서 참조할 수 있는 중요한 설계 원칙을 제공한다.

**③ SAM2 및 차세대 세그멘테이션 모델로의 확장 가능성**

SAM2(Ravi et al., Meta, 2024)는 영상 세그멘테이션까지 확장한 모델로, 이전 세대 모델들(SAM 포함)은 여전히 위장된 물체 탐지, 의료 이미지 세그멘테이션, 세포 이미지 세그멘테이션, 그림자 탐지와 같은 세밀한 저수준 세그멘테이션 과제와 씨름하고 있다. RobustSAM의 접근법은 SAM2 및 후속 모델에도 적용 가능한 방향성을 제시한다.

**④ 의료 영상, 자율주행 등 응용 도메인에 직접적 파급 효과**

자율주행이나 의료 영상 등 이미지 세그멘테이션의 많은 실제 응용 분야는 종종 완벽하지 않은 이미지 품질을 다루어야 하는데, 이는 중요한 발전을 의미한다.

---

### 4-2. 앞으로 연구 시 고려할 점

**① 실제(Real-world) 열화와 합성(Synthetic) 열화 간 도메인 갭 해소**

현재 RobustSAM은 15가지 합성 열화로 훈련되지만, 실제 환경에서의 열화는 이보다 훨씬 복잡하고 복합적이다. 향후 연구에서는 **비지도 도메인 적응(Unsupervised Domain Adaptation)** 또는 **실제 열화 데이터를 포함한 혼합 훈련 전략**을 고려해야 한다.

**② 더 다양한 열화 유형으로의 확장**

열화 증강 방식에는 15가지 유형의 열화와 항등 매핑(identity mapping)이 포함된다. 이는 깨끗한 이미지가 원래의 품질을 유지하여 비저하 시나리오에서의 성능 저하를 방지하기 위함이다. 향후에는 이 15가지를 넘어서 더 넓은 범위의 열화 유형(예: 모션 블러 + 저조도 복합 열화)을 다루는 연구가 필요하다.

**③ SAM2 기반으로의 확장 연구**

비디오 세그멘테이션 시나리오에서도 열화 이미지 문제는 동일하게 존재한다. SAM2의 메모리 어텐션 메커니즘과 RobustSAM의 Anti-Degradation 모듈을 결합하는 방향의 연구가 기대된다.

**④ 경량화(Lightweight) 및 엣지 디바이스 배포**

제안된 모델은 열화 시나리오에서 미미한 연산 부담 증가만으로 성능을 효과적으로 향상시킨다. 그러나 MobileSAM, EfficientSAM 등과의 통합을 통해 모바일·엣지 환경에서의 실시간 배포 가능성을 더욱 연구해야 한다.

**⑤ 강건한 프롬프트 엔지니어링과의 결합**

저화질 이미지에서는 프롬프트(포인트, 바운딩 박스) 자체도 부정확해질 수 있다. 열화된 이미지에 대한 **강건한 자동 프롬프트 생성(Auto-prompt generation)** 연구가 병행될 필요가 있다.

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

| 연구 | 학회/연도 | 핵심 접근법 | RobustSAM과의 관계 |
|---|---|---|---|
| **SAM** (Kirillov et al.) | ICCV 2023 | 11B 마스크 기반 대규모 사전훈련, ViT 인코더+프롬프트 시스템 | RobustSAM의 기반 모델; 저화질 이미지에서 성능 저하 문제 존재 |
| **SAM2** (Ravi et al., Meta) | ICLR 2025 | 영상 세그멘테이션으로 확장, 메모리 어텐션 도입 | SAM1 기반 RobustSAM의 후속 확장 대상 |
| **SAM-Adapter** (Chen et al.) | 2023 | Adapter 모듈로 SAM을 위장 객체·그림자·의료 도메인에 적응 | 특정 도메인 적응에 초점; RobustSAM은 열화 이미지 강건성에 초점 |
| **HQ-SAM** (Ke et al.) | NeurIPS 2023 | 고품질 세그멘테이션 토큰 도입, 세밀한 마스크 개선 | 깨끗한 이미지 품질 향상에 집중; RobustSAM과 상호 보완적 |
| **MobileSAM** (Zhang et al.) | 2023 | Knowledge Distillation으로 경량화, 실시간 추론 | 효율성 중심; RobustSAM은 강건성 중심 |
| **Restormer** (Zamir et al.) | CVPR 2022 | 고해상도 이미지 복원을 위한 Transformer 구조 | 복원 선처리 후 SAM 적용 파이프라인의 대안이나, 세그멘테이션 최적화 아님 |
| **All-in-One Restoration** (Li et al.) | CVPR 2022 | 단일 모델로 다양한 열화 복원 | 복원 파이프라인 접근; RobustSAM은 End-to-End 세그멘테이션 접근 |

> 이러한 복원 방법들은 이미지 품질을 향상시킬 수 있지만, 선택된 복원 기법이 세그멘테이션을 개선한다는 보장은 없다. 대부분의 이미지 복원 알고리즘은 인간의 시각적 인식이 아닌 세그멘테이션 모델의 특정 요구사항에 최적화되어 있지 않기 때문이다.

---

## 📚 참고 자료 (출처)

1. **논문 원문 (CVPR 2024 Open Access)**
   - Chen, Wei-Ting et al. "RobustSAM: Segment Anything Robustly on Degraded Images." *CVPR 2024*, pp. 4081–4091.
   - URL: https://openaccess.thecvf.com/content/CVPR2024/papers/Chen_RobustSAM_Segment_Anything_Robustly_on_Degraded_Images_CVPR_2024_paper.pdf

2. **arXiv 프리프린트**
   - arXiv:2406.09627, https://arxiv.org/abs/2406.09627

3. **공식 프로젝트 페이지**
   - https://robustsam.github.io/

4. **공식 GitHub 저장소**
   - https://github.com/robustsam/RobustSAM

5. **IEEE Xplore 공식 출판**
   - https://ieeexplore.ieee.org/document/10657905/

6. **CVPR 2024 포스터 페이지**
   - https://cvpr.thecvf.com/virtual/2024/poster/29230

7. **Labellerr 블로그 분석**
   - "How RobustSAM Helps With Blurry/Degraded Image Segmentation", https://www.labellerr.com/blog/robustsam-image-segmentation-for-degraded-images/

8. **The Moonlight 리뷰**
   - "[Literature Review] RobustSAM: Segment Anything Robustly on Degraded Images", https://www.themoonlight.io/en/review/robustsam-segment-anything-robustly-on-degraded-images

9. **ResearchGate**
   - https://www.researchgate.net/publication/381470707_RobustSAM_Segment_Anything_Robustly_on_Degraded_Images

10. **arXiv HTML 전문**
    - https://arxiv.org/html/2406.09627 / https://arxiv.org/html/2406.09627v1

> ⚠️ **정확도 안내**: 손실 함수 수식의 구체적인 가중치($\lambda_1, \lambda_2$) 값과 15가지 열화 유형의 정확한 전체 목록은 논문 원문 PDF에서 직접 확인하시기 바랍니다. 본 답변은 공개된 논문 원문, arXiv, 공식 프로젝트 페이지, 보충 자료(supplemental)를 기반으로 작성되었습니다.
