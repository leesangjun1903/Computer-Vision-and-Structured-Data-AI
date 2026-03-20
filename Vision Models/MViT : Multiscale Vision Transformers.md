# Multiscale Vision Transformers

---

## 1. 핵심 주장 및 주요 기여 요약

**Multiscale Vision Transformers (MViT)**는 컴퓨터 비전의 근본 원리인 **다중 스케일 특징 계층(multiscale feature hierarchy)**을 Transformer 아키텍처에 결합한 모델이다. 핵심 주장과 기여는 다음과 같다:

1. **다중 스케일 피라미드 구조의 Transformer 도입**: 기존 Vision Transformer(ViT)가 전체 네트워크에서 동일한 채널 차원과 공간 해상도를 유지하는 것과 달리, MViT는 여러 **채널-해상도 스케일 스테이지(scale stage)**를 두어 초기 레이어에서는 높은 공간 해상도/낮은 채널 차원으로 단순한 저수준 시각 정보를 처리하고, 깊은 레이어에서는 낮은 공간 해상도/높은 채널 차원으로 복잡한 고수준 의미를 모델링한다.

2. **Multi Head Pooling Attention (MHPA)**: Query, Key, Value 텐서를 독립적으로 풀링하여 시퀀스 길이를 점진적으로 줄이는 새로운 어텐션 메커니즘을 제안하여, 연산량과 메모리를 극적으로 절감한다.

3. **외부 사전학습 없이 우수한 성능**: ImageNet-21K 등 대규모 외부 데이터 사전학습 없이(from scratch) 학습하면서도, 동시대 비디오 Vision Transformer(VTN, TimeSformer, ViViT)를 능가하며, 이들 모델 대비 **5~10배 적은 연산량과 파라미터**로 동등 이상의 정확도를 달성한다.

4. **강한 시간적(temporal) 모델링 능력**: 프레임 셔플링 실험에서 MViT는 정확도가 크게 하락(-7.1%)하여 시간 정보를 효과적으로 활용함을 입증한 반면, 기존 ViT는 거의 하락 없음(-0.1%)으로 시간 정보를 무시함을 보인다.

5. **이미지 인식으로의 확장**: 비디오 모델에서 시간 차원만 제거하여 ImageNet 이미지 분류에서도 DeiT 대비 우수한 성능을 달성한다.

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

기존 Vision Transformer(ViT)는 입력을 고정 크기 패치로 분할한 후, **전체 네트워크에서 동일한 채널 차원 $D$와 공간 해상도를 유지**한다. 이는 두 가지 핵심적 한계를 초래한다:

- **시각 신호의 밀집(dense) 특성을 효과적으로 활용하지 못함**: 초기 레이어에서 고해상도 정보를 처리하면서도 불필요하게 높은 채널 차원을 사용하여 연산 낭비가 발생한다.
- **비디오의 시공간(spatiotemporal) 정보 모델링 실패**: 기존 ViT를 비디오에 적용하면 시간 정보를 사실상 무시하며, 이를 보완하려면 ImageNet-21K 같은 대규모 외부 사전학습에 의존해야 한다.
- **이차(quadratic) 어텐션 복잡도**: 전체 시퀀스 길이에 대한 어텐션 연산이 $O(L^2)$로 매우 비효율적이다.

### 2.2 제안하는 방법

#### (A) Multi Head Pooling Attention (MHPA)

입력 텐서 $X \in \mathbb{R}^{L \times D}$에서 선형 사영을 통해 Query, Key, Value를 생성한다:

$$\hat{Q} = XW_Q, \quad \hat{K} = XW_K, \quad \hat{V} = XW_V$$

여기서 $W_Q, W_K, W_V \in \mathbb{R}^{D \times D}$이다.

**풀링 연산자** $\mathcal{P}(\cdot; \Theta)$를 정의한다. $\Theta := (\mathbf{k}, \mathbf{s}, \mathbf{p})$로, 풀링 커널 $\mathbf{k} = k_T \times k_H \times k_W$, 스트라이드 $\mathbf{s} = s_T \times s_H \times s_W$, 패딩 $\mathbf{p} = p_T \times p_H \times p_W$를 사용한다. 입력 시퀀스 길이 $\mathbf{L} = T \times H \times W$는 다음과 같이 축소된다:

$$\tilde{\mathbf{L}} = \left\lfloor \frac{\mathbf{L} + 2\mathbf{p} - \mathbf{k}}{\mathbf{s}} \right\rfloor + 1$$

이 수식은 각 좌표(T, H, W)별로 독립 적용된다.

풀링된 Q, K, V는 다음과 같다:

$$Q = \mathcal{P}(\hat{Q}; \Theta_Q), \quad K = \mathcal{P}(\hat{K}; \Theta_K), \quad V = \mathcal{P}(\hat{V}; \Theta_V)$$

**Pooling Attention**은 다음과 같이 계산된다:

$$\text{PA}(\cdot) = \text{Softmax}\!\left(\frac{\mathcal{P}(Q; \Theta_Q) \cdot \mathcal{P}(K; \Theta_K)^T}{\sqrt{d}}\right) \mathcal{P}(V; \Theta_V)$$

여기서 $\sqrt{d}$는 내적 행렬의 행별 정규화를 위한 스케일링이다. 이 연산의 출력 시퀀스 길이는 Query 풀링 스트라이드에 의해 $s_T^Q \cdot s_H^Q \cdot s_W^Q$ 만큼 축소된다.

**시퀀스 길이 축소 인자**:

$$f_j = s_T^j \cdot s_H^j \cdot s_W^j, \quad \forall\, j \in \{Q, K, V\}$$

**런타임 복잡도** (헤드당):

$$O\!\left(\frac{THWD}{h}\left(D + \frac{THW}{f_Q f_K}\right)\right)$$

**메모리 복잡도**:

$$O\!\left(THW \cdot h \left(\frac{D}{h} + \frac{THW}{f_Q f_K}\right)\right)$$

#### (B) Multiscale Transformer 네트워크 구조

각 Transformer 블록의 연산은 다음과 같다:

$$X_1 = \text{MHPA}(\text{LN}(X)) + X$$

$$\text{Block}(X) = \text{MLP}(\text{LN}(X_1)) + X_1$$

MViT의 핵심 설계 원리:

| 설계 요소 | 설명 |
|---|---|
| **Scale Stage** | 동일 해상도/채널 차원의 Transformer 블록 집합 |
| **채널 확장** | 공간 해상도 $4\times$ 축소 시 채널 $2\times$ 확장 (예: $2D \times \frac{T}{s_T} \times \frac{H}{8} \times \frac{W}{8} \to 4D \times \frac{T}{s_T} \times \frac{H}{16} \times \frac{W}{16}$) |
| **Query 풀링** | 각 스테이지 첫 블록에서만 $s^Q > 1$ 적용하여 해상도 감소 |
| **K,V 풀링** | 모든 블록에서 adaptive하게 적용하여 연산 효율 확보 |
| **Skip Connection** | 차원 불일치 시 query 풀링을 residual path에도 적용; 채널 확장 시 layer-normalized linear 사용 |

**구체적 모델 인스턴스 (MViT-B)**:
- **cube₁**: 입력을 $3 \times 7 \times 7$ 크기의 시공간 큐브로 사영, 채널 $D=96$
- **초기 시퀀스 길이**: $8 \times 56 \times 56 + 1 = 25{,}089$
- **최종 시퀀스 길이**: $8 \times 7 \times 7 + 1 = 393$
- **4개 스케일 스테이지**: 각각 $[1, 2, 11, 2]$ 블록

대비, **ViT-B**는 고정 시퀀스 길이 $8 \times 14 \times 14 + 1 = 1{,}569$, 고정 채널 $D = 768$.

### 2.3 성능 향상

#### 비디오 인식 (Kinetics-400)

| 모델 | 사전학습 | Top-1 (%) | FLOPs × views | Param (M) |
|---|---|---|---|---|
| ViT-B (baseline) | 없음 | 68.5 | 180×1×5 | 87.2 |
| ViT-B (baseline) | IN-21K | 79.3 | 180×1×5 | 87.2 |
| ViT-L-ViViT | IN-21K | 81.3 | 3992×3×4 | 310.8 |
| **MViT-B, 16×4** | **없음** | **78.4** | **70.5×1×5** | **36.6** |
| **MViT-B, 64×3** | **없음** | **81.2** | **455×3×3** | **36.6** |

- MViT-B는 ViT-B 대비 **+9.9% 정확도**, **2.6× 적은 FLOPs**, **2.4× 적은 파라미터**
- ViViT-L 대비 **6.8× 적은 FLOPs**, **8.5× 적은 파라미터**로 동등 정확도 달성, 외부 사전학습 불필요

#### 이미지 분류 (ImageNet-1K)

| 모델 | Top-1 (%) | FLOPs (G) | Param (M) |
|---|---|---|---|
| DeiT-B | 81.8 | 17.6 | 86.6 |
| **MViT-B-16** | **83.0** | **7.8** | **37.0** |
| DeiT-B ↑384² | 83.1 | 55.5 | 87.0 |
| **MViT-B-24-wide-320²** | **84.8** | **32.7** | **72.9** |

- MViT-B-16은 DeiT-B 대비 **+1.2% 정확도**, **2.3× 적은 FLOPs/파라미터**

#### 전이 학습 성능

- **SSv2** (temporal modeling): MViT-B-24 = 68.7% (K600 사전학습)
- **Charades**: MViT-B-24 = 47.7 mAP
- **AVA v2.2** (action detection): MViT-B-24 = 28.7 mAP

### 2.4 한계

1. **절대적 연산 효율에서 경량 ConvNet에 미치지 못함**: X3D-S, X3D-M 등 경량 CNN 모델은 multiply-add 연산 측면에서 여전히 더 효율적 (Figure A.4 log-scale 참조)
2. **학습 레시피의 아키텍처 종속성**: MViT 학습 레시피를 ConvNet(SlowFast)에 적용하면 오히려 성능이 하락 (Table A.2: SlowFast R101 78.0% → 61.6%)
3. **K,V 풀링 하이퍼파라미터 민감성**: 풀링 스트라이드의 적응적(adaptive) 설정이 필수적이며, 부적절한 설정 시 정확도가 크게 하락 (Table 14: 비적응적 2×4×4에서 74.8%)
4. **대규모 입력에서의 메모리 제약**: 초기 스테이지의 높은 공간 해상도로 인해 여전히 메모리 소비가 상당함
5. **단일 태스크 검증**: 주로 분류/검출 태스크에 한정되어, 세그멘테이션, 생성 등 다른 비전 태스크에서의 일반화는 검증되지 않음

---

## 3. 모델의 일반화 성능 향상 가능성

MViT의 설계에는 일반화 성능 향상에 기여하는 여러 메커니즘이 내재되어 있다:

### 3.1 다중 스케일 구조의 귀납적 편향 (Inductive Bias)

MViT는 ConvNet의 다중 스케일 피라미드(e.g., ResNet의 stage 구조)와 유사한 강력한 **귀납적 편향**을 Transformer에 도입한다. 이는 시각 신호의 본질적 특성 — 초기에는 밀집된 저수준 정보, 후반에는 추상적 고수준 정보 — 에 부합하여, **제한된 데이터에서도 효과적으로 학습**할 수 있게 한다.

이 점은 **외부 사전학습 없이 from scratch 학습**으로도 ImageNet-21K 사전학습 모델을 능가하는 결과에서 입증된다. 기존 ViT 기반 비디오 모델(VTN, TimeSformer, ViViT)은 외부 사전학습 없이는 학습이 실패하거나 성능이 극도로 저하된다.

### 3.2 시간 정보의 효과적 모델링

프레임 셔플링 실험 (Table 9)이 이를 명확히 보여준다:

| 모델 | 정상 Acc (%) | 셔플링 Acc (%) | 하락폭 |
|---|---|---|---|
| MViT-B | 77.2 | 70.1 | **-7.1** |
| ViT-B | 68.5 | 68.4 | -0.1 |

ViT-B는 시간 정보를 사실상 무시하는 "bag-of-frames" 분류를 수행하는 반면, MViT는 다중 스케일 풀링을 통해 시공간 정보를 효과적으로 캡처한다. 이는 SSv2(temporal modeling 벤치마크)에서의 우수한 성능(68.7%)으로도 확인된다.

### 3.3 Attention Distance의 다양성

Figure A.6의 질적 분석에서:
- **초기화 시**: MViT의 attention distance 동적 범위가 ViT 대비 **약 4배** 넓음
- **수렴 후**: ViT는 깊이에 따라 단조 증가하는 attention distance를 보이지만, MViT는 **비단조적(non-monotonic)** 패턴으로, 깊은 레이어에서도 서로 다른 헤드가 여전히 다른 특징(로컬/글로벌)에 집중
- 이는 MViT의 각 헤드가 **중복 연산 없이 더 효율적으로 계산 자원을 활용**함을 시사

### 3.4 다중 태스크/데이터셋 전이 성능

MViT는 단일 아키텍처로 다양한 성격의 태스크에서 일관된 성능을 보인다:
- **Kinetics** (appearance-heavy), **SSv2** (temporal-heavy), **Charades** (long-range), **AVA** (spatiotemporal detection), **ImageNet** (image classification)

비디오 모델의 시간 차원만 제거하여 이미지 분류에서도 우수한 성능을 달성한 점은 **아키텍처의 근본적 일반성**을 보여준다.

### 3.5 Adaptive K,V Pooling

적응적 풀링은 스테이지의 해상도에 비례하여 K,V 텐서의 스트라이드를 조정하여, 모든 블록에서 K,V 텐서의 스케일을 일관되게 유지한다. 이는 과도한 정보 손실을 방지하면서 연산 효율을 확보하는 정규화 효과를 제공한다 (Table 14: adaptive 1×8×8에서 77.2% vs. non-adaptive 2×4×4에서 74.8%).

---

## 4. 향후 연구에 미치는 영향 및 고려사항

### 4.1 학술적 영향

MViT는 이후 비전 트랜스포머 연구에 **패러다임적 영향**을 미쳤다:

1. **다중 스케일 트랜스포머의 표준화**: MViT 이후 Swin Transformer, PVT 등 다중 스케일/계층적 구조가 비전 트랜스포머의 표준 설계 원리로 자리잡았다.
2. **사전학습 의존도 감소**: from scratch 학습의 가능성을 입증하여, 대규모 사전학습 데이터 없이도 경쟁력 있는 모델 구축이 가능함을 보였다.
3. **비디오 이해의 트랜스포머 기반 접근 발전**: 시공간 정보의 효과적 모델링 방법론을 제시하여, 이후 비디오 트랜스포머 연구의 기반이 되었다.

### 4.2 앞으로 연구 시 고려할 점

1. **더 효율적인 어텐션 메커니즘**: MHPA가 연산량을 크게 줄이지만, 여전히 초기 고해상도 스테이지에서의 연산 비용이 높다. Window attention(Swin), linear attention 등과의 결합이 유망하다.

2. **자기지도학습(Self-supervised Learning)과의 결합**: MViT의 다중 스케일 구조가 MAE, DINO 등 자기지도학습 프레임워크에서 어떤 이점을 제공하는지 탐구 필요 (실제로 MViTv2에서 일부 검증됨).

3. **다운스트림 밀집 예측 태스크**: 객체 검출, 세그멘테이션 등에서의 FPN(Feature Pyramid Network)과의 자연스러운 결합 가능성 탐구.

4. **학습 레시피의 아키텍처 범용성**: MViT 레시피가 ConvNet에서 실패하는 문제(Table A.2)는, 아키텍처별 최적 학습 전략의 차이를 시사하며, 범용적 학습 레시피 개발의 필요성을 제기한다.

5. **스케일링 법칙(Scaling Law)**: MViT의 깊이/너비/해상도 스케일링에 대한 체계적 연구가 필요하다.

6. **풀링 전략의 학습**: 현재 풀링 커널/스트라이드는 수동 설계이며, NAS(Neural Architecture Search) 또는 학습 가능한 풀링 전략이 성능을 더 향상시킬 수 있다.

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

| 연구 | 발표 시기 | 핵심 차별점 | MViT와의 관계 |
|---|---|---|---|
| **Swin Transformer** (Liu et al., 2021) | ICCV 2021 | Shifted window 기반 local attention으로 선형 복잡도 달성; 계층적 특징 맵 | MViT와 유사한 다중 스케일 설계이나, 풀링 대신 window partition 사용. 더 일반적인 밀집 예측 태스크에 적합 |
| **PVT (Pyramid Vision Transformer)** (Wang et al., 2021) | ICCV 2021 | Spatial Reduction Attention으로 다중 스케일 특징 추출 | MViT의 K,V 풀링과 유사한 공간 축소 전략; 검출/세그멘테이션에 초점 |
| **MViTv2** (Li et al., 2022) | CVPR 2022 | Decomposed relative position encoding, residual pooling connection 도입 | MViT의 직접적 후속작; 위치 인코딩과 풀링 연결 개선으로 성능 향상 |
| **Video Swin Transformer** (Liu et al., 2022) | CVPR 2022 | 3D shifted window를 비디오에 적용 | MViT의 비디오 시공간 모델링과 경쟁; window 기반 접근 |
| **ViViT** (Arnab et al., 2021) | ICCV 2021 | 시공간 분해(factorization) 어텐션 | 대규모 사전학습 필요; MViT가 from scratch에서 동등 성능 |
| **TimeSformer** (Bertasius et al., 2021) | ICML 2021 | Divided space-time attention | MViT 대비 5×+ 높은 FLOPs; IN-21K 사전학습 필수 |
| **MAE (Masked Autoencoders)** (He et al., 2022) | CVPR 2022 | 마스킹 기반 자기지도학습 | MViT를 백본으로 MAE 적용 시 시너지 가능 (VideoMAE에서 검증) |
| **Hiera** (Ryali et al., 2023) | ICML 2023 | MAE 사전학습으로 MViT에서 불필요한 요소 제거 (간소화) | MViT의 직계 후속으로, MAE와 결합하여 아키텍처 단순화 및 속도 개선 |

### 핵심 비교 분석

**MViT vs. Swin Transformer**: 두 모델 모두 다중 스케일 계층 구조를 채택하지만 접근이 다르다. MViT는 **풀링 기반 어텐션**으로 글로벌 수용장을 유지하면서 해상도를 줄이고, Swin은 **윈도우 기반 로컬 어텐션**으로 선형 복잡도를 달성한다. Swin은 밀집 예측 태스크에서 더 널리 채택되었으나, MViT의 글로벌 어텐션은 비디오의 장거리 시간 의존성 모델링에 유리하다.

**MViT vs. MViTv2**: MViTv2는 원본 MViT의 절대 위치 인코딩을 **분해된 상대 위치 인코딩(decomposed relative positional encoding)**으로 대체하고, **잔차 풀링 연결(residual pooling connection)**을 도입하여 성능과 일반화를 개선했다. 이는 MViT의 한계였던 위치 인코딩의 해상도 종속성 문제를 해결한다.

**MViT vs. Hiera**: Hiera는 MViT 구조에서 MAE 사전학습을 통해 **불필요한 모듈(상대 위치 인코딩, conv 풀링 등)을 제거**하고도 동등 이상의 성능을 달성하여, MViT 설계의 일부가 충분한 사전학습 하에서는 불필요할 수 있음을 시사한다.

---

## 참고자료

1. **Fan, H., Xiong, B., Mangalam, K., Li, Y., Yan, Z., Malik, J., & Feichtenhofer, C.** (2021). *Multiscale Vision Transformers*. arXiv:2104.11227. (본 논문)
2. **Liu, Z., Lin, Y., Cao, Y., et al.** (2021). *Swin Transformer: Hierarchical Vision Transformer using Shifted Windows*. ICCV 2021. arXiv:2103.14030.
3. **Wang, W., Xie, E., Li, X., et al.** (2021). *Pyramid Vision Transformer: A Versatile Backbone for Dense Prediction without Convolutions*. ICCV 2021. arXiv:2102.12122.
4. **Li, Y., Wu, C.-Y., Fan, H., Mangalam, K., Xiong, B., Malik, J., & Feichtenhofer, C.** (2022). *MViTv2: Improved Multiscale Vision Transformers for Classification and Detection*. CVPR 2022. arXiv:2112.01526.
5. **Liu, Z., Ning, J., Cao, Y., et al.** (2022). *Video Swin Transformer*. CVPR 2022. arXiv:2106.13230.
6. **Arnab, A., Dehghani, M., Heigold, G., Sun, C., Lučić, M., & Schmid, C.** (2021). *ViViT: A Video Vision Transformer*. ICCV 2021. arXiv:2103.15691.
7. **Bertasius, G., Wang, H., & Torresani, L.** (2021). *Is Space-Time Attention All You Need for Video Understanding?* ICML 2021. arXiv:2102.05095.
8. **He, K., Chen, X., Xie, S., Li, Y., Dollár, P., & Girshick, R.** (2022). *Masked Autoencoders Are Scalable Vision Learners*. CVPR 2022. arXiv:2111.06377.
9. **Ryali, C., Hu, Y.-T., Bolya, D., Wei, C., Fan, H., Huang, P.-Y., Misra, I., & Feichtenhofer, C.** (2023). *Hiera: A Hierarchical Vision Transformer without the Bells-and-Whistles*. ICML 2023. arXiv:2306.00989.
10. **Dosovitskiy, A., Beyer, L., Kolesnikov, A., et al.** (2020). *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale*. ICLR 2021. arXiv:2010.11929.
11. **Touvron, H., Cord, M., Douze, M., et al.** (2021). *Training data-efficient image transformers & distillation through attention (DeiT)*. ICML 2021. arXiv:2012.12877.
12. **Vaswani, A., Shazeer, N., Parmar, N., et al.** (2017). *Attention Is All You Need*. NeurIPS 2017. arXiv:1706.03762.
