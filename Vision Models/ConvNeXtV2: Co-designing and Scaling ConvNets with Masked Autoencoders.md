# ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders

---

## 1. 핵심 주장과 주요 기여 요약

### 핵심 주장
ConvNeXt V2는 **자기지도 학습(self-supervised learning) 프레임워크와 네트워크 아키텍처를 공동 설계(co-design)** 해야만 순수 합성곱 신경망(pure ConvNet)에서도 마스크 기반 사전학습이 효과적으로 작동한다는 것을 입증한다. 단순히 기존 ConvNeXt에 MAE를 적용하면 성능이 제한적이지만, 아키텍처와 학습 프레임워크를 함께 개선하면 Transformer 기반 모델에 필적하거나 이를 능가하는 성능을 달성할 수 있다.

### 주요 기여 (4가지)
1. **FCMAE (Fully Convolutional Masked AutoEncoder)**: 희소 합성곱(sparse convolution)을 활용하여 ConvNet에서도 마스크 기반 사전학습이 가능하도록 한 완전 합성곱 프레임워크
2. **GRN (Global Response Normalization)**: 채널 간 특징 경쟁을 촉진하여 feature collapse 문제를 해결하는 새로운 정규화 레이어
3. **Co-design 패러다임**: 학습 프레임워크와 아키텍처 개선이 시너지를 이루어야 함을 실증적으로 입증
4. **광범위한 모델 스케일링**: 3.7M (Atto) ~ 650M (Huge) 파라미터에 이르는 8종 모델 제공, 모든 크기에서 일관된 성능 향상 달성

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

#### 문제 1: MAE의 ConvNet 비호환성
MAE(Masked Autoencoder)는 Transformer의 시퀀스 처리 특성에 최적화된 비대칭 인코더-디코더 설계를 사용한다. 이 설계에서 인코더는 **가시 패치(visible patches)만** 입력으로 받아 연산 비용을 절감하는데, 밀집 슬라이딩 윈도우(dense sliding window) 연산을 사용하는 표준 ConvNet에는 이 방식이 직접 적용되기 어렵다. 기존 접근법(BEiT, SimMIM 등)에서 학습 가능한 마스크 토큰을 입력에 삽입하는 방식은:
- 사전학습 효율 저하
- 학습/테스트 시 불일치(train-test inconsistency) 발생
- 마스크 영역으로부터의 정보 누출(information leakage)

#### 문제 2: Feature Collapse
ConvNeXt V1을 FCMAE로 사전학습하면 MLP 확장 레이어에서 **feature collapse** 현상이 발생한다. 이는 많은 채널이 죽은(dead) 또는 포화(saturated) 상태가 되어 채널 간 활성화가 중복되는 현상이다. 이로 인해 자기지도 사전학습의 이점이 지도 학습 대비 제한적이 된다.

#### 문제 3: 아키텍처-학습 프레임워크 분리 설계의 한계
기존 연구는 지도 학습용으로 설계된 아키텍처를 자기지도 학습에 그대로 재사용하는 관행이 있었으나, 이는 최적 성능 달성에 한계가 있다.

---

### 2.2 제안하는 방법

#### (A) FCMAE 프레임워크

**마스킹 전략**: 0.6의 마스킹 비율로 $32 \times 32$ 패치 중 60%를 무작위 제거. 계층적 구조의 마지막 스테이지에서 마스크를 생성하고 최고 해상도까지 재귀적으로 업샘플링.

**인코더 설계**: 마스크된 이미지를 2D 희소 배열로 간주하고, 표준 합성곱을 **서브매니폴드 희소 합성곱(submanifold sparse convolution)**으로 대체하여 가시 픽셀에만 연산 수행. 파인튜닝 시에는 표준 밀집 합성곱으로 복원 가능.

**디코더 설계**: 단일 ConvNeXt 블록을 사용한 경량 디코더. Transformer 디코더 대비 1.7배 속도 향상을 달성하면서 동등한 파인튜닝 정확도 유지.

**재구성 목표**: 패치 단위 정규화된 원본 이미지와 재구성 이미지 간의 **MSE(Mean Squared Error)** 손실을 마스크된 패치에만 적용:

$$\mathcal{L} = \frac{1}{|\mathcal{M}|} \sum_{p \in \mathcal{M}} \| \hat{x}_p - \text{normalize}(x_p) \|_2^2$$

여기서 $\mathcal{M}$은 마스크된 패치 집합, $\hat{x}_p$는 재구성된 패치, $x_p$는 원본 패치이다.

#### (B) Global Response Normalization (GRN)

GRN은 세 단계로 구성된다:

**Step 1. 전역 특징 집계 (Global Feature Aggregation)**

입력 특징 $X \in \mathbb{R}^{H \times W \times C}$에서 각 채널 $X_i \in \mathbb{R}^{H \times W}$의 공간적 통계를 L2-norm으로 집계:

$$\mathcal{G}(X) := X \in \mathbb{R}^{H \times W \times C} \rightarrow g_x \in \mathbb{R}^C$$

$$\mathcal{G}(X)_i = \|X_i\|_2$$

이는 $g_x = \{\|X_1\|, \|X_2\|, \ldots, \|X_C\|\} \in \mathbb{R}^C$를 산출한다.

**Step 2. 특징 정규화 (Feature Normalization)**

집계된 값에 분할 정규화(divisive normalization)를 적용하여 채널 간 상대적 중요도를 계산:

$$\mathcal{N}(\|X_i\|) := \|X_i\| \in \mathbb{R} \rightarrow \frac{\|X_i\|}{\sum_{j=1,\ldots,C} \|X_j\|} \in \mathbb{R}$$

이 단계는 채널 간 **상호 억제(mutual inhibition)**를 통해 특징 경쟁을 유도한다.

**Step 3. 특징 보정 (Feature Calibration)**

정규화 점수를 사용하여 원래 입력 응답을 보정:

$$X_i = X_i * \mathcal{N}(\mathcal{G}(X)_i) \in \mathbb{R}^{H \times W}$$

**최종 GRN 블록** (학습 가능 파라미터 $\gamma$, $\beta$ 및 잔차 연결 포함):

$$X_i = \gamma * X_i * \mathcal{N}(\mathcal{G}(X)_i) + \beta + X_i$$

$\gamma$와 $\beta$는 0으로 초기화되어 학습 초기에는 항등 함수로 동작하며 점진적으로 적응한다.

**PyTorch 의사코드** (Algorithm 1):
```python
gx = torch.norm(X, p=2, dim=(1,2), keepdim=True)
nx = gx / (gx.mean(dim=-1, keepdim=True) + 1e-6)
return gamma * (X * nx) + beta + X
```

#### (C) 특징 코사인 거리 분석

특징 다양성의 정량적 평가를 위해 활성화 텐서 $X \in \mathbb{R}^{H \times W \times C}$에서 채널 간 평균 쌍별 코사인 거리를 계산:

$$d = \frac{1}{C^2} \sum_{i}^{C} \sum_{j}^{C} \frac{1 - \cos(X_i, X_j)}{2}$$

높은 거리 값은 더 다양한 특징을, 낮은 값은 특징 중복을 나타낸다.

---

### 2.3 모델 구조

#### ConvNeXt V2 블록 설계

ConvNeXt V1 블록 대비 변경 사항:
1. **GRN 레이어 추가**: 차원 확장 MLP 레이어(GELU 활성화 후)에 GRN을 삽입
2. **LayerScale 제거**: GRN 적용 시 LayerScale이 불필요해짐

블록 구조:
```
Input (96-d)
  → Depthwise Conv 7×7, 96
  → LayerNorm
  → Pointwise Conv 1×1, 384 (차원 확장)
  → GELU + GRN          ← 신규 추가
  → Pointwise Conv 1×1, 96 (차원 축소)
  → Residual Connection
```

#### 모델 패밀리 구성

| 모델 | 채널 (C) | 블록 (B) | 파라미터 |
|------|---------|---------|---------|
| Atto | 40 | (2,2,6,2) | 3.7M |
| Femto | 48 | (2,2,6,2) | 5.2M |
| Pico | 64 | (2,2,6,2) | 9.1M |
| Nano | 80 | (2,2,8,2) | 15.6M |
| Tiny | 96 | (3,3,9,3) | 28.6M |
| Base | 128 | (3,3,27,3) | 89M |
| Large | 192 | (3,3,27,3) | 198M |
| Huge | 352 | (3,3,27,3) | 659M |

---

### 2.4 성능 향상

#### ImageNet-1K 분류

| 모델 | V1 Supervised | V2 + FCMAE | 개선 |
|------|--------------|------------|-----|
| Atto (3.7M) | 75.7% | 76.7% | +1.0% |
| Nano (15.6M) | 80.8% | 81.9% | +1.1% |
| Base (89M) | 83.8% | 84.9% | +1.1% |
| Large (198M) | 84.3% | 85.8% | +1.5% |
| Huge (659M) | — | 86.3% | — |

#### Co-design 시너지 효과 (Table 3)

| 설정 | Base | Large |
|------|------|-------|
| V1 + Supervised | 83.8 | 84.3 |
| V1 + FCMAE | 83.7 | 84.4 |
| V2 + Supervised | 84.3 (+0.5) | 84.5 (+0.2) |
| V2 + FCMAE | **84.6 (+0.8)** | **85.6 (+1.3)** |

FCMAE만 적용하거나 GRN만 적용하면 개선이 제한적이지만, **두 가지를 함께 적용하면 시너지가 극대화**된다. 특히 Large 모델에서 +1.3%라는 큰 폭의 개선이 나타난다.

#### ImageNet-22K 중간 파인튜닝 후 최종 결과

ConvNeXt V2-Huge ($512^2$): **88.9% top-1 accuracy** — 공개 데이터만 사용한 모델 중 최고 성능 달성.

#### COCO 객체 검출/인스턴스 분할 (Mask R-CNN)

| 모델 | $\text{AP}^{\text{box}}$ | $\text{AP}^{\text{mask}}$ |
|------|-------------------------|--------------------------|
| V1-B Supervised | 50.3 | 44.9 |
| V2-B FCMAE | **52.9** | **46.6** |
| V2-H FCMAE | **55.7** | **48.9** |

#### ADE20K 의미론적 분할 (UPerNet)

| 모델 | mIoU |
|------|------|
| V1-B Supervised | 49.9 |
| V2-B FCMAE | 52.1 |
| V2-H FCMAE | 55.0 |
| V2-H FCMAE + 22K ft ($640^2$) | **57.0** |

---

### 2.5 한계

1. **Huge 모델 규모에서 ViT 대비 약간 열위**: ViT-H + MAE (1600 epoch)가 86.9%인 반면, ConvNeXt V2-H + FCMAE (1600 epoch)는 86.3%. 이는 거대 ViT 모델이 자기지도 사전학습에서 더 큰 이점을 얻을 수 있음을 시사.

2. **희소 합성곱 라이브러리 최적화 부족**: MinkowskiEngine 등 희소 합성곱 라이브러리가 현대 하드웨어에 최적화되지 않아, 이론적 효율 대비 실제 속도 향상이 제한적 (평균 1.3× throughput 향상, 2× 메모리 절감).

3. **Base/Large 규모에서 SimMIM + Swin 대비 제한적 우위**: Segmentation에서 Base 모델 규모에서 Swin-B + SimMIM (52.8 mIoU) 대비 ConvNeXt V2-B + FCMAE (52.1 mIoU)가 약간 열위.

4. **사전학습 에폭 효율성**: MAE는 1600 에폭에서 큰 이점을 보이지만, FCMAE는 800→1600 에폭 증가 시 개선 폭이 상대적으로 작음.

5. **GRN의 사전학습/파인튜닝 동시 요구**: GRN을 파인튜닝 시에만 추가하거나 제거하면 성능이 크게 하락(84.6 → 80.6 또는 78.8)하여 유연성이 제한적.

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 Feature Collapse 해결을 통한 일반화

ConvNeXt V1의 FCMAE 사전학습에서 발견된 feature collapse는 채널 간 중복 활성화로 인해 **표현력이 제한**되는 현상이다. GRN은 채널 간 **상호 억제(lateral inhibition)**를 통해 이 문제를 해결한다.

코사인 거리 분석(Figure 4)에서:
- **V1 FCMAE**: 깊은 레이어에서 코사인 거리가 급격히 감소 → feature collapse
- **V2 FCMAE**: 전체 레이어에서 일관되게 높은 코사인 거리 유지 → MAE 사전학습된 ViT와 유사한 학습 행동

이는 GRN이 **특징 다양성(feature diversity)**을 유지하여 모델이 더 풍부한 표현을 학습할 수 있게 함을 의미한다.

### 3.2 Class Selectivity Index 분석

ConvNeXt V2의 FCMAE 사전학습 가중치에 대한 class selectivity index 분석(Figure 7)은 일반화 성능과 직접 관련된 중요한 발견을 제공한다:

- **V1 (unimodal 분포)**: 깊은 레이어에서 클래스 특화 특징 위주
- **V2 (bimodal 분포)**: 깊은 레이어에서도 **클래스 비의존적(class-generic) 특징**을 더 많이 포함

Morcos et al. (2018)의 연구에 따르면 클래스 비의존적 특징은 더 **전이 가능(transferrable)**하므로, 이는 ConvNeXt V2가 다운스트림 태스크에서 더 나은 일반화 성능을 보이는 이유를 설명한다.

### 3.3 다운스트림 전이 학습 일반화

실험 결과는 ConvNeXt V2의 우수한 일반화 성능을 다방면에서 입증한다:

- **ImageNet 분류**: 모든 8개 모델 크기에서 일관된 개선 (3.7M ~ 659M)
- **COCO 검출/분할**: V1 대비 $\text{AP}^{\text{box}}$ 최대 +5.4 개선 (Huge 모델)
- **ADE20K 분할**: mIoU 최대 +5.1 개선 (Base 모델, 49.9→55.0 Huge)

### 3.4 광범위한 모델 스케일링에서의 일관성

Figure 1에서 보듯이, Atto(3.7M)부터 Huge(659M)까지 **모든 모델 크기에서 일관되게** 지도 학습 대비 개선을 보인다는 것은, 제안된 방법이 특정 모델 크기에 의존하지 않는 **일반적인(general) 접근법**임을 시사한다. 이는 마스크 이미지 모델링의 이점이 이렇게 넓은 모델 스펙트럼에서 입증된 최초의 사례이다.

### 3.5 대조 학습 대비 우수한 일반화

MoCo V3(대조 학습)과의 비교에서 FCMAE가 더 나은 표현 품질을 보여준다:

| 방법 | ImageNet 정확도 |
|------|-------------|
| Supervised (300ep) | 84.3% |
| MoCo V3 | 83.7% |
| FCMAE | **84.9%** |

---

## 4. 향후 연구에 미치는 영향과 고려사항

### 4.1 연구에 미치는 영향

#### (1) Co-design 패러다임의 확립
이 논문은 **아키텍처와 학습 프레임워크의 공동 설계**가 필수적이라는 중요한 메시지를 전달한다. 기존에는 아키텍처를 고정하고 학습 방법만 변경하거나, 그 반대의 접근이 일반적이었다. ConvNeXt V2는 두 요소의 시너지 효과를 실증적으로 보여주어, 향후 연구에서 이 접근법이 표준이 될 가능성을 제시한다.

#### (2) 순수 ConvNet의 재조명
Transformer가 지배적인 상황에서, 순수 ConvNet도 적절한 사전학습 프레임워크와 결합하면 경쟁력 있는 성능을 달성할 수 있음을 보여준다. 이는 ConvNet의 효율적인 추론 특성(하드웨어 친화성, 낮은 지연시간)과 결합하여 실용적 가치가 크다.

#### (3) 자기지도 학습의 아키텍처 인식 필요성
Feature collapse와 같은 아키텍처 특화 문제를 발견하고 해결한 사례로서, 향후 다른 아키텍처에 자기지도 학습을 적용할 때도 유사한 분석이 필요함을 시사한다.

#### (4) 효율적 모델 계열 제공
3.7M ~ 659M의 넓은 범위에서 사전학습된 모델을 공개함으로써, 에지 디바이스부터 대규모 서버까지 다양한 배포 환경에 활용 가능하다.

### 4.2 앞으로 연구 시 고려할 점

1. **더 큰 규모에서의 스케일링**: Huge 규모에서 ViT-H + MAE 대비 성능 격차가 존재하므로, 10억 파라미터 이상의 ConvNet 스케일링 연구 필요

2. **희소 합성곱 라이브러리 최적화**: 현재 라이브러리의 하드웨어 최적화 부족으로 인한 효율성 제한을 극복할 수 있는 구현 개선 필요

3. **다양한 도메인 적용**: 의료 영상, 위성 이미지, 비디오 등 다양한 도메인에서의 전이 학습 성능 검증 필요

4. **GRN의 이론적 기반 강화**: GRN이 feature collapse를 해결하는 메커니즘에 대한 더 깊은 이론적 분석 필요

5. **마스킹 전략의 고도화**: 현재 무작위 마스킹을 사용하지만, 의미론적(semantic) 마스킹이나 적응형(adaptive) 마스킹 전략의 탐색 필요

6. **멀티모달 확장**: 텍스트-이미지 등 멀티모달 사전학습에서의 ConvNeXt V2 활용 가능성 탐색

7. **Linear probing 성능**: 논문이 end-to-end 파인튜닝에 초점을 맞추었으므로, linear probing이나 few-shot 학습에서의 성능 검증 필요

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

### 5.1 마스크 이미지 모델링 (Masked Image Modeling) 계열

| 연구 | 연도 | 아키텍처 | 핵심 특징 | IN-1K 정확도 (Base급) |
|------|------|---------|---------|---------------------|
| **BEiT** (Bao et al.) | 2022 (ICLR) | ViT | 이산 토큰 예측, dVAE 토크나이저 | 83.2% (ViT-B) |
| **MAE** (He et al.) | 2022 (CVPR) | ViT | 픽셀 재구성, 비대칭 인코더-디코더, 75% 마스킹 | 83.6% (ViT-B) |
| **SimMIM** (Xie et al.) | 2022 (CVPR) | Swin Transformer | 단순 마스킹 + 픽셀 예측, 계층적 아키텍처 | 84.0% (Swin-B) |
| **MCMAE** (Gao et al.) | 2022 (NeurIPS) | Hybrid (Conv+Trans) | 합성곱 블록을 입력 토크나이저로 사용 | — |
| **ConvNeXt V2** (본 논문) | 2023 | ConvNeXt (pure Conv) | 희소 합성곱 + GRN, 완전 합성곱 프레임워크 | **84.9%** (ConvNeXt V2-B) |

**분석**: ConvNeXt V2는 Base급 모델에서 MAE (+1.3%), BEiT (+1.7%), SimMIM (+0.9%)를 모두 상회한다. 특히 순수 합성곱 아키텍처로 이를 달성한 것이 의의가 있다. Large급에서는 MAE의 ViT-L (307M, 85.9%)과 ConvNeXt V2-L (198M, 85.8%)이 거의 동등하지만, ConvNeXt V2가 **36% 적은 파라미터**로 이를 달성한다.

### 5.2 아키텍처 설계 비교

| 연구 | 연도 | 유형 | 핵심 혁신 | ConvNeXt V2와의 관계 |
|------|------|------|---------|---------------------|
| **ConvNeXt V1** (Liu et al.) | 2022 (CVPR) | Conv | ResNet→Transformer 설계 요소 체계적 도입 | V2의 기반 아키텍처, GRN 추가 |
| **Swin Transformer V2** (Liu et al.) | 2022 (CVPR) | Transformer | Log-CPB, 잔차 후 정규화 | Huge 모델에서 V2에 열위 (85.7 vs 85.8) |
| **MaxViT** (Tu et al.) | 2022 (ECCV) | Hybrid | 다축 어텐션, 블록/그리드 어텐션 교차 | IN-22K ft에서 88.7% vs V2 88.9% |
| **CoAtNet** (Dai et al.) | 2021 (NeurIPS) | Hybrid | Conv + Attention 단계적 결합 | V2가 적은 FLOPS로 상회 |
| **EfficientNet V2** (Tan & Le) | 2021 (ICML) | Conv | 점진적 학습, Fused-MBConv | 87.3% vs V2 88.9% |

### 5.3 자기지도 학습 방법론 비교

| 방법론 | 유형 | ConvNet 호환성 | 특징 |
|--------|------|-------------|------|
| **MoCo V3** (Chen et al., 2021) | 대조 학습 | 높음 | ConvNet V2-B에서 83.7% (FCMAE 84.9%) |
| **DINO/DINOv2** (Caron et al., 2021/2023) | 자기증류 | ViT 위주 | Self-distillation, 최근 DINOv2는 대규모 큐레이션 데이터 사용 |
| **MAE** (He et al., 2022) | 마스크 재구성 | ViT 전용 설계 | ConvNet에 직접 적용 불가 → FCMAE가 해결 |
| **data2vec** (Baevski et al., 2022) | 자기증류 + 마스킹 | 멀티모달 | 학생-교사 프레임워크, 다양한 모달리티 지원 |
| **I-JEPA** (Assran et al., 2023) | JEPA | ViT | 픽셀이 아닌 잠재 공간에서 예측, MAE와 다른 접근 |
| **FCMAE** (본 논문) | 마스크 재구성 | ConvNet 전용 설계 | 희소 합성곱으로 ConvNet 호환 해결 |

**분석**: FCMAE는 MAE의 핵심 아이디어를 ConvNet에 성공적으로 적용한 최초의 프레임워크이다. 대조 학습(MoCo V3) 대비 end-to-end 파인튜닝에서 우수하며, 이는 마스크 이미지 모델링이 대조 학습보다 나은 표현을 학습한다는 최근 경향과 일치한다.

### 5.4 정규화 기법 비교

| 기법 | 정규화 범위 | GRN과의 차이 | FCMAE + ConvNeXt 성능 |
|------|---------|-----------|---------------------|
| **Batch Norm** (Ioffe & Szegedy, 2015) | 배치-공간 | 마스크 입력에 부적합, 글로벌 채널 경쟁 없음 | 80.5% |
| **Layer Norm** (Ba et al., 2016) | 채널 | 암시적 경쟁만 제공, 명시적 채널 대비 없음 | 83.8% |
| **Local Response Norm** (Krizhevsky et al., 2012) | 인접 채널 | 글로벌 컨텍스트 없음, 로컬 대비만 | 83.2% |
| **SE-Net** (Hu et al., 2018) | 전역-채널 | 추가 MLP 파라미터 필요 (+20M), 채널 게이팅 | 84.4% |
| **CBAM** (Woo et al., 2018) | 전역-공간+채널 | 추가 파라미터 필요 (+20M) | 84.5% |
| **GRN** (본 논문) | 전역-채널 | 파라미터 오버헤드 없음, L2-norm 기반 분할 정규화 | **84.6%** |

**분석**: GRN은 가장 높은 성능을 달성하면서도 **추가 파라미터가 거의 없다** ($\gamma$, $\beta$만 추가). SE-Net과 CBAM은 20M의 추가 파라미터가 필요하지만, GRN은 3줄의 코드로 구현 가능하며 더 나은 성능을 보인다. 이는 채널 경쟁을 위해 복잡한 게이팅 메커니즘보다 **간결한 정규화**가 더 효과적임을 시사한다.

### 5.5 최신 후속 연구 동향 (2023년 이후)

ConvNeXt V2 이후 등장한 관련 연구들:

- **InternImage** (Wang et al., 2023): Deformable convolution 기반 대규모 ConvNet. ConvNeXt V2와 유사하게 ConvNet의 가능성을 확장하지만, 변형 가능 합성곱(deformable conv)을 사용하는 점에서 차별화.
- **DINOv2** (Oquab et al., 2023): 대규모 큐레이션 데이터로 self-supervised ViT 학습. ConvNet보다는 ViT에 초점, 대규모 데이터 사용.
- **EVA-02** (Fang et al., 2023): MIM 사전학습 기반 ViT 확장. CLIP 특징 증류 활용으로 ConvNeXt V2와 다른 접근.
- **FastViT** (Vasu et al., 2023): RepMixer로 효율적 하이브리드 설계. ConvNeXt V2의 경량 모델(Atto, Femto 등)과 경쟁적 포지셔닝.

---

## 참고자료

1. Woo, S., Debnath, S., Hu, R., Chen, X., Liu, Z., Kweon, I.S., & Xie, S. (2023). "ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders." *arXiv:2301.00808*. (본 논문)
2. Liu, Z., Mao, H., Wu, C.Y., Feichtenhofer, C., Darrell, T., & Xie, S. (2022). "A ConvNet for the 2020s." In *CVPR 2022*.
3. He, K., Chen, X., Xie, S., Li, Y., Dollár, P., & Girshick, R. (2022). "Masked Autoencoders Are Scalable Vision Learners." In *CVPR 2022*.
4. Xie, Z., Zhang, Z., Cao, Y., Lin, Y., Bao, J., Yao, Z., Dai, Q., & Hu, H. (2022). "SimMIM: A Simple Framework for Masked Image Modeling." In *CVPR 2022*.
5. Bao, H., Dong, L., & Wei, F. (2022). "BEiT: BERT Pre-Training of Image Transformers." In *ICLR 2022*.
6. Dosovitskiy, A. et al. (2021). "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale." In *ICLR 2021*.
7. Hu, J., Shen, L., & Sun, G. (2018). "Squeeze-and-Excitation Networks." In *CVPR 2018*.
8. Woo, S., Park, J., Lee, J.Y., & Kweon, I.S. (2018). "CBAM: Convolutional Block Attention Module." In *ECCV 2018*.
9. Krizhevsky, A., Sutskever, I., & Hinton, G.E. (2012). "ImageNet Classification with Deep Convolutional Neural Networks." In *NeurIPS 2012*.
10. Chen, X., Xie, S., & He, K. (2021). "An Empirical Study of Training Self-Supervised Vision Transformers." In *ICCV 2021*.
11. Morcos, A.S., Barrett, D.G.T., Rabinowitz, N.C., & Botvinick, M. (2018). "On the Importance of Single Directions for Generalization." In *ICLR 2018*.
12. Choy, C., Gwak, J., & Savarese, S. (2019). "4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural Networks." In *CVPR 2019*.
13. Graham, B. & van der Maaten, L. (2017). "Submanifold Sparse Convolutional Networks." *arXiv:1706.01307*.
14. Tu, Z. et al. (2022). "MaxViT: Multi-Axis Vision Transformer." In *ECCV 2022*.
15. Dai, Z. et al. (2021). "CoAtNet: Marrying Convolution and Attention for All Data Sizes." In *NeurIPS 2021*.
16. Tan, M. & Le, Q. (2021). "EfficientNetV2: Smaller Models and Faster Training." In *ICML 2021*.
17. Gao, P. et al. (2022). "MCMAE: Masked Convolution Meets Masked Autoencoders." In *NeurIPS 2022*.
18. GitHub Repository: https://github.com/facebookresearch/ConvNeXt-V2
