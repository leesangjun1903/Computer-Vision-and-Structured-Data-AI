# Depth Anything: Unleashing the Power of Large-Scale Unlabeled Data

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

**Depth Anything**은 대규모 레이블 없는(unlabeled) 단안(monocular) 이미지를 활용하여 **어떠한 이미지에서도 강건한 깊이 추정**이 가능한 파운데이션 모델을 구축할 수 있다는 것을 주장한다. 새로운 아키텍처를 고안하는 대신, **데이터 스케일업(data scaling-up)**과 두 가지 간단하지만 효과적인 전략으로 일반화 성능을 극대화한다.

### 주요 기여

| 기여 항목 | 내용 |
|-----------|------|
| 데이터 엔진 설계 | 약 62M 비레이블 이미지 자동 수집·주석 |
| 도전적 최적화 목표 | 강한 데이터 증강(CutMix, 색상 왜곡)으로 학생 모델 강제 학습 |
| 시맨틱 사전 보존 | DINOv2의 풍부한 의미론적 표현을 특징 정렬 손실로 계승 |
| 최고 성능(SOTA) | Zero-shot 깊이 추정에서 MiDaS v3.1 능가, 파인튜닝 후 ZoeDepth 능가 |

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

Monocular Depth Estimation(MDE)은 로보틱스, 자율주행, VR 등 광범위한 분야에 활용되지만, 기존 방법들은 다음 문제를 갖는다:

- **레이블 데이터 구축의 어려움**: LiDAR, 스테레오 매칭, SfM 등은 비용·시간이 많이 소요됨
- **일반화 한계**: 기존 모델(MiDaS 포함)은 데이터 커버리지가 제한적이어서 보지 못한 도메인에서 성능 저하
- **단순 self-training의 한계**: 충분한 레이블 이미지가 있을 때, 단순히 pseudo label을 추가해도 성능 향상이 없음

---

### 2.2 제안하는 방법 (수식 포함)

#### 전체 파이프라인

$$\mathcal{D}^l = \{(x_i, d_i)\}_{i=1}^M, \quad \mathcal{D}^u = \{u_i\}_{i=1}^N$$

- 교사 모델 $T$를 레이블 데이터 $\mathcal{D}^l$로 학습
- $T$를 이용해 $\mathcal{D}^u$에 pseudo label 생성
- 학생 모델 $S$를 $\mathcal{D}^l \cup \hat{\mathcal{D}}^u$로 학습

---

#### (1) 레이블 이미지 학습: Affine-Invariant Loss

다양한 데이터셋의 깊이 스케일 불일치 문제를 해결하기 위해 **affine-invariant loss**를 사용한다.

$$\mathcal{L}_l = \frac{1}{HW}\sum_{i=1}^{HW}\rho(d_i^*, d_i)$$

여기서 $\rho$는 affine-invariant mean absolute error:

```math
\rho(d_i^*, d_i) = |\hat{d}_i^* - \hat{d}_i|
```

정규화 방식:

$$\hat{d}_i = \frac{d_i - t(d)}{s(d)}$$

$$t(d) = \text{median}(d), \quad s(d) = \frac{1}{HW}\sum_{i=1}^{HW}|d_i - t(d)|$$

- $t(d)$: 번역(translation) 정렬 (median)
- $s(d)$: 스케일 정렬 (mean absolute deviation)

---

#### (2) 비레이블 이미지 활용: 강한 섭동(Perturbation) + CutMix

pseudo labeled set 생성:

$$\hat{\mathcal{D}}^u = \{(u_i, T(u_i)) \mid u_i \in \mathcal{D}^u\}_{i=1}^N$$

단순 pseudo label 추가는 효과 없음 → **강한 섭동으로 도전적 최적화 목표** 부여

**CutMix** 적용: 두 비레이블 이미지 $u_a$, $u_b$를 합성

$$u_{ab} = u_a \odot M + u_b \odot (1 - M)$$

- $M$: 직사각형 영역이 1인 이진 마스크

손실 함수:

$$\mathcal{L}_u^M = \rho\big(S(u_{ab}) \odot M,\; T(u_a) \odot M\big)$$

$$\mathcal{L}_u^{1-M} = \rho\big(S(u_{ab}) \odot (1-M),\; T(u_b) \odot (1-M)\big)$$

가중 평균:

$$\mathcal{L}_u = \frac{\sum M}{HW}\mathcal{L}_u^M + \frac{\sum(1-M)}{HW}\mathcal{L}_u^{1-M}$$

> **핵심**: 교사 모델에는 **clean 이미지**를 입력, 학생 모델에는 **강하게 왜곡된 이미지**를 입력하여 불일치를 통해 추가 시각적 지식 습득을 강제함

---

#### (3) 시맨틱 보조 감독: Feature Alignment Loss

이산 클래스 공간(semantic segmentation)으로의 디코딩은 정보 손실이 큼 → **DINOv2의 연속 feature 공간**을 직접 정렬

$$\mathcal{L}_{feat} = 1 - \frac{1}{HW}\sum_{i=1}^{HW}\cos(f_i, f'_i)$$

- $f_i$: 학생 깊이 모델 $S$의 feature
- $f'_i$: **동결된(frozen)** DINOv2 인코더의 feature
- $\cos(\cdot, \cdot)$: 코사인 유사도

**Tolerance Margin $\alpha$ 도입**: DINOv2는 같은 객체의 전면/후면 feature가 유사하지만, 깊이 추정에서는 다른 값이어야 함 → 코사인 유사도가 $\alpha = 0.85$ 초과 시 해당 픽셀은 $\mathcal{L}_{feat}$ 계산에서 제외

---

#### 최종 손실 함수

$$\mathcal{L} = \mathcal{L}_l + \mathcal{L}_u + \mathcal{L}_{feat}$$

---

### 2.3 모델 구조

| 구성 요소 | 상세 |
|-----------|------|
| **인코더** | DINOv2 pre-trained ViT (ViT-S/B/L) |
| **디코더** | DPT (Dense Prediction Transformer) |
| **파라미터 수** | ViT-S: 24.8M / ViT-B: 97.5M / ViT-L: 335.3M |
| **훈련 해상도** | 518×518 (patch size 14에 맞춤) |
| **학습률** | 인코더: 5e-6 / 디코더: 5e-5 (10× 큰 값) |
| **옵티마이저** | AdamW + linear decay |

#### 2단계 학습 전략

```
Stage 1: Teacher 모델 T → 1.5M 레이블 이미지로 20 epoch 학습
Stage 2: Student 모델 S → (1.5M 레이블 + 62M pseudo labeled) 
         배치 비율 labeled:unlabeled = 1:2
```

---

### 2.4 성능 향상

#### Zero-Shot 상대 깊이 추정 (Table 2 기반)

| 방법 | 인코더 | KITTI AbsRel↓ | KITTI δ₁↑ | NYUv2 AbsRel↓ | NYUv2 δ₁↑ |
|------|--------|--------------|-----------|--------------|-----------|
| MiDaS v3.1 | ViT-L | 0.127 | 0.850 | 0.048 | 0.980 |
| **Depth Anything** | ViT-S | 0.080 | 0.936 | 0.053 | 0.972 |
| **Depth Anything** | ViT-B | 0.080 | 0.939 | 0.046 | 0.979 |
| **Depth Anything** | **ViT-L** | **0.076** | **0.947** | **0.043** | **0.981** |

> MiDaS는 KITTI/NYUv2 훈련 이미지를 사용(non-zero-shot)함에도 불구하고 Depth Anything이 우세

#### 파인튜닝 후 메트릭 깊이 추정 (NYUv2, Table 3)

| 방법 | δ₁↑ | AbsRel↓ | RMSE↓ |
|------|-----|---------|-------|
| VPD | 0.964 | 0.069 | 0.254 |
| **Ours** | **0.984** | **0.056** | **0.206** |

#### 시맨틱 분할 전이 성능 (Table 7, Cityscapes)

| 방법 | mIoU (s.s.) | mIoU (m.s.) |
|------|-------------|-------------|
| OneFormer (ConvNeXt-XL) | 83.6 | 84.6 |
| **Ours (ViT-L)** | **84.8** | **86.2** |

---

### 2.5 한계

논문에서 명시적으로 언급한 한계:

1. **모델 크기 제한**: 최대 ViT-Large로만 실험. ViT-Giant 확장 미실시
2. **훈련 해상도 제한**: 518×518은 실제 응용에 불충분할 수 있음 (700+ 또는 1000+ 해상도 필요)
3. **메트릭 깊이 한계**: 모델이 제공하는 것은 기본적으로 상대 깊이(relative depth)이며, 절대 메트릭 깊이는 파인튜닝이 필요
4. **pseudo label 노이즈**: 교사 모델의 오류가 학생 모델에 전파될 수 있음
5. **계산 비용**: 62M 이미지를 처리하는 데이터 엔진 운영에 막대한 연산 자원 필요

---

## 3. 모델의 일반화 성능 향상 관련 심층 분석

### 3.1 데이터 커버리지 확장을 통한 일반화

$$\text{Generalization Error} \propto \frac{1}{\text{Data Coverage}}$$

논문의 핵심 통찰은 **일반화 오차(generalization error)를 줄이는 가장 효과적인 방법이 데이터 커버리지 확장**이라는 것이다. 62M 비레이블 이미지는 8개의 대규모 공개 데이터셋(BDD100K, SA-1B, ImageNet-21K, LSUN, Objects365, Open Images V7, Places365, Google Landmarks)에서 수집되어 실내·실외·다양한 기상 조건·원거리 장면 등을 망라한다.

### 3.2 강한 섭동을 통한 불변 표현 학습

일반화 능력의 핵심은 **invariant representation** 학습이다:

- **색상 강한 왜곡**: Color jittering + Gaussian blurring → 조명 변화, 안개, 저조도 환경에 강건
- **CutMix 공간 왜곡**: 두 이미지를 합성해 맥락 이해를 강제 → 부분적 장면에서도 올바른 깊이 예측 유도

**Ablation Study 결과 (Table 9)**:

| $\mathcal{L}_l$ | $\mathcal{L}_u$ | $\mathcal{S}$ | $\mathcal{L}_{feat}$ | KITTI | NYU | Mean AbsRel↓ |
|---|---|---|---|---|---|---|
| ✓ | | | | 0.085 | 0.053 | 0.180 |
| ✓ | ✓ | | | 0.085 | 0.054 | ~0.180 |
| ✓ | ✓ | ✓ | | 0.081 | 0.048 | ~0.176 |
| ✓ | ✓ | ✓ | ✓ | **0.076** | **0.043** | **~0.170** |

→ 단순 pseudo label 추가(행 2)는 효과 없으나, 강한 섭동($\mathcal{S}$)과 $\mathcal{L}_{feat}$ 조합이 핵심

### 3.3 시맨틱 사전 보존을 통한 장면 이해 강화

DINOv2의 시맨틱 표현을 계승함으로써:
- 객체 경계, 재질, 카테고리 정보를 깊이 추정에 활용
- **tolerance margin $\alpha = 0.85$**: 깊이 추정의 pixel-discriminative 특성과 시맨틱의 region-uniform 특성 사이의 균형 유지

### 3.4 데이터 다양성의 중요성 (Table 6 분석)

놀랍게도 HRWSI(20K 이미지)가 단일 데이터셋 중 가장 강한 zero-shot 일반화를 보임 → **데이터 양보다 다양성이 일반화에 더 중요**함을 시사

---

## 4. 향후 연구에 미치는 영향 및 고려사항

### 4.1 향후 연구에 미치는 영향

#### (1) 파운데이션 모델 패러다임의 MDE 확장
Depth Anything은 MDE 분야에서도 GPT, SAM과 같은 **파운데이션 모델 패러다임**이 유효함을 증명했다. 이는 3D 재구성, 광학 흐름 등 다른 기하학적 비전 태스크에도 같은 접근이 적용될 수 있음을 시사한다.

#### (2) 비용 효율적 semi-supervised 학습의 새 방향
"충분한 레이블 데이터가 있을 때 unlabeled 데이터가 도움이 안 된다"는 기존 통념을 깨고, **강한 섭동**을 통해 효과적으로 활용할 수 있음을 보였다. 이는 객체 탐지, 세그멘테이션 등에도 적용 가능한 원리다.

#### (3) 크로스 태스크 지식 증류
시맨틱 feature alignment 전략은 **태스크 간 지식 전이**의 새로운 방식을 제안한다. 이산 레이블 대신 연속 feature space를 활용하는 접근은 향후 다중 태스크 모델 설계에 영향을 줄 것이다.

#### (4) Universal Encoder의 가능성
본 논문에서 훈련된 인코더는 MDE와 시맨틱 분할 모두에서 우수한 성능을 보이며 **범용 인코더**의 가능성을 열었다. Cityscapes에서 86.2 mIoU, ADE20K에서 59.4 mIoU로 기존 SOTA 초과.

#### (5) ControlNet 등 생성 모델과의 결합
더 정확한 깊이 맵은 Depth-conditioned ControlNet의 품질을 높이며, **이미지/비디오 생성 및 편집 분야**에도 직접적 영향을 미친다.

---

### 4.2 향후 연구 시 고려할 점

#### 기술적 고려사항

| 고려사항 | 설명 |
|----------|------|
| **모델 스케일 확장** | ViT-Giant 기반 교사 모델로 더 정확한 pseudo label 생성 가능 |
| **고해상도 훈련** | 518×518 → 1000+ 해상도로 재훈련 시 경계 세부 묘사 향상 |
| **메트릭 깊이 통합** | Scale-Invariant 학습과 절대 스케일 복원을 동시에 학습하는 프레임워크 |
| **시간적 일관성** | 비디오 시퀀스에서 깊이 시간적 일관성 보장 메커니즘 미흡 |
| **동적 객체 처리** | 움직이는 객체에 대한 깊이 추정 정확도 향상 필요 |

#### 데이터 관련 고려사항

- **pseudo label 품질 검증**: 교사 모델 오류의 학생 모델 전파 방지를 위한 신뢰도 기반 필터링
- **데이터 불균형**: 특정 도메인(예: 의료 영상, 수중 환경)에 대한 커버리지 부족
- **프라이버시 및 저작권**: 대규모 인터넷 이미지 수집 시 법적·윤리적 고려 필요

#### 평가 방법론 관련 고려사항

- **공정한 zero-shot 평가**: MiDaS가 KITTI/NYUv2 훈련 데이터를 사용한 것처럼, 평가 프로토콜의 엄밀한 통일 필요
- **실제 환경 벤치마크**: 실험실 데이터셋을 넘어 실제 자율주행, 로봇 환경에서의 평가 필요

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

| 논문 | 연도 | 핵심 방법 | 일반화 | 메트릭 깊이 | 비고 |
|------|------|-----------|--------|------------|------|
| **MiDaS v3.1** (Birkl et al.) | 2023 | Affine-invariant loss, 혼합 데이터셋 | ✅ (ViT-L) | ❌ | 12개 데이터셋 사용, 코드 미공개 |
| **ZoeDepth** (Bhat et al.) | 2023 | Relative→Metric 파인튜닝, bin 분류기 | 부분적 | ✅ | MiDaS 인코더 기반 |
| **Metric3D** (Yin et al.) | 2023 | 카메라 정규화로 스케일 모호성 해결 | ✅ | ✅ | 절대 메트릭 깊이 직접 예측 |
| **DPT** (Ranftl et al.) | 2021 | ViT 기반 dense prediction | 부분적 | ❌ | Transformer 깊이 추정 도입 |
| **AdaBins** (Bhat et al.) | 2021 | 적응형 bin 분류 | ❌ (domain-specific) | ✅ | NYUv2/KITTI 전용 |
| **VPD** (Zhao et al.) | 2023 | 텍스트-이미지 diffusion 활용 | ❌ | ✅ | NYUv2 SOTA (당시) |
| **NDDepth** (Shao et al.) | 2023 | 법선-거리 보조 정보 활용 | ❌ | ✅ | KITTI 특화 |
| **Depth Anything** (Yang et al.) | 2024 | 62M unlabeled + 강한 섭동 + DINOv2 alignment | ✅✅ | ✅ (파인튜닝) | 본 논문 |

### 핵심 차별점

```
MiDaS: 레이블 데이터 혼합 → 제한적 커버리지
ZoeDepth: 상대→절대 깊이 변환 → 일반화 약함  
Metric3D: 카메라 파라미터 정규화 → 메트릭 깊이 직접 예측
Depth Anything: 비레이블 데이터 62M + 강한 섭동 → 일반화 극대화
```

> **중요**: Depth Anything 이후 발표된 **Depth Anything V2** (Yang et al., 2024)는 합성 데이터(synthetic data)를 중간 단계로 활용하여 pseudo label 품질을 더욱 향상시켰다. 본 답변은 제공된 논문(V1, arXiv:2401.10891v2) 기준으로 작성되었으며, V2에 대한 상세 내용은 해당 논문을 별도 참조 바란다.

---

## 참고자료

**주요 참고 문헌 (본 논문 내 인용 기준)**

1. **Depth Anything** (본 논문): Yang, L. et al. "Depth Anything: Unleashing the Power of Large-Scale Unlabeled Data." arXiv:2401.10891v2, 2024.
2. **MiDaS v3.1**: Birkl, R. et al. "MiDaS v3.1 – A Model Zoo for Robust Monocular Relative Depth Estimation." arXiv:2307.14460, 2023.
3. **ZoeDepth**: Bhat, S.F. et al. "ZoeDepth: Zero-shot Transfer by Combining Relative and Metric Depth." arXiv:2302.12288, 2023.
4. **DINOv2**: Oquab, M. et al. "DINOv2: Learning Robust Visual Features without Supervision." TMLR, 2023.
5. **DPT**: Ranftl, R. et al. "Vision Transformers for Dense Prediction." ICCV, 2021.
6. **MiDaS (원본)**: Ranftl, R. et al. "Towards Robust Monocular Depth Estimation: Mixing Datasets for Zero-Shot Cross-Dataset Transfer." TPAMI, 2020.
7. **Metric3D**: Yin, W. et al. "Metric3D: Towards Zero-shot Metric 3D Prediction from a Single Image." ICCV, 2023.
8. **VPD**: Zhao, W. et al. "Unleashing Text-to-Image Diffusion Models for Visual Perception." ICCV, 2023.
9. **CutMix**: Yun, S. et al. "CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features." ICCV, 2019.
10. **FixMatch**: Sohn, K. et al. "FixMatch: Simplifying Semi-Supervised Learning with Consistency and Confidence." NeurIPS, 2020.
11. **SAM (Segment Anything)**: Kirillov, A. et al. "Segment Anything." ICCV, 2023.
12. **AdaBins**: Bhat, S.F. et al. "AdaBins: Depth Estimation Using Adaptive Bins." CVPR, 2021.
13. **Foundation Models**: Bommasani, R. et al. "On the Opportunities and Risks of Foundation Models." arXiv:2108.07258, 2021.
