# Reversible Vision Transformers

---

## 1. 핵심 주장과 주요 기여 요약

**Reversible Vision Transformers** (Mangalam et al., 2023)는 Vision Transformer(ViT)와 Multiscale Vision Transformer(MViT)를 **가역적(reversible) 구조**로 재설계하여, GPU 메모리 사용량을 모델 깊이로부터 분리(decouple)하는 **메모리 효율적 아키텍처**를 제안한다.

### 주요 기여 3가지:

1. **Rev-ViT 및 Rev-MViT 제안**: ViT와 MViT를 가역적 변환 기반의 두 개의 잔차 스트림(two-residual-stream) 구조로 적응시켜, 중간 활성화(activation)를 저장하지 않고도 역방향 전파 시 재계산할 수 있게 함.
2. **가역 구조의 강한 내재적 정규화(inherent regularization) 발견**: 가역 트랜스포머는 비가역 모델 대비 더 강한 정규화 특성을 가지므로, 더 가벼운 데이터 증강(augmentation) 레시피가 필요함을 발견.
3. **다양한 태스크에서의 광범위한 벤치마킹**: 이미지 분류, 객체 탐지, 비디오 분류에서 성능 손실 없이 최대 **15.5배** 메모리 절감, 최대 **2.3배** 학습 처리량(throughput) 향상을 달성.

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

트랜스포머 모델의 학습 시 **GPU 메모리 사용량이 모델 깊이 $D$에 선형적으로 증가**하는 문제가 핵심이다. 역전파(back-propagation) 알고리즘은 기울기 계산을 위해 순방향 전파 시의 모든 중간 활성화를 저장해야 하므로, 피크 메모리 사용량이 네트워크 깊이에 비례하여 증가한다.

특히:
- GPU 가속기의 연산 성능(FLOPs)은 2년마다 약 $\sim 3.1\times$ 증가하지만, 메모리 대역폭은 $\sim 1.4\times$만 증가하여 **"메모리 벽(memory wall)"** 문제가 심화됨.
- 비디오 인식 등 메모리 집약적 도메인에서는 배치 크기 1로도 학습이 제한되는 상황이 발생.

### 2.2 제안하는 방법 (수식 포함)

#### 2.2.1 가역 변환 (Reversible Transformation)

입력 텐서 $\mathbf{I}$를 두 개의 $d$차원 텐서 $[\mathbf{I}_1; \mathbf{I}_2]$로 분할한다. 임의의 미분 가능한 함수 $F(\cdot): \mathbb{R}^d \to \mathbb{R}^d$와 $G(\cdot): \mathbb{R}^d \to \mathbb{R}^d$를 사용하여, 두 변환 $T_1$과 $T_2$의 합성 $T = T_2 \circ T_1$을 다음과 같이 정의한다:

```math
\mathbf{I} = \begin{bmatrix} \mathbf{I}_1 \\ \mathbf{I}_2 \end{bmatrix} \xrightarrow{T} \begin{bmatrix} \mathbf{O}_1 \\ \mathbf{O}_2 \end{bmatrix} = \begin{bmatrix} \mathbf{I}_1 + G(\mathbf{I}_2 + F(\mathbf{I}_1)) \\ \mathbf{I}_2 + F(\mathbf{I}_1) \end{bmatrix} = \mathbf{O} \tag{1}
```

이 변환 $T$는 역변환 $T' = T_1' \circ T_2'$를 허용하여 $T'(T(\mathbf{I})) = \mathbf{I}$를 만족한다. 역변환은 $F$와 $G$를 정확히 한 번씩만 호출하므로 순방향 변환과 **동일한 계산 비용**을 가진다.

#### 2.2.2 기존 네트워크가 활성화 캐싱을 필요로 하는 이유

역전파 알고리즘에서 계산 그래프 노드 $\mathcal{M}$에 대한 기울기는 다음과 같이 계산된다:

$$
\frac{d\mathcal{L}}{d\mathcal{M}} = \sum_{\mathcal{N}_j} \left( \frac{\partial f_j}{\partial \mathcal{M}} \right)^T \frac{d\mathcal{L}}{d\mathcal{N}_j}
$$

가장 단순한 신경망 층 $f(X) = W^T X$의 경우:

$$
\frac{d\mathcal{L}}{dW} = \left( \frac{d\mathcal{L}}{dY} \right) X^T, \qquad \frac{d\mathcal{L}}{dX} = W \frac{d\mathcal{L}}{dY}
$$

따라서 가중치에 대한 기울기 계산 시 **중간 활성화 $X$가 반드시 필요**하며, 이를 GPU 메모리에 캐싱해야 한다. 네트워크 깊이 $D$에 대해 피크 메모리가 **선형적으로** 증가하는 근본 원인이 된다.

#### 2.2.3 가역 변환을 통한 캐싱 없는 학습

가역 변환 $T$로 구성된 네트워크는 출력으로부터 입력을 재계산할 수 있으므로, 중간 활성화를 저장할 필요가 없다. 이를 통해 **활성화 메모리 증가가 모델 깊이와 독립적**이 된다.

단, **등차원 제약(Equidimensional Constraint)**이 존재한다: 함수 $F$와 $G$의 입력/출력 차원이 동일해야 하므로, 변환 $T$ 하에서 특징 차원이 일정해야 한다. ViT는 전 층에 걸쳐 일정한 특징 차원을 유지하므로 이 제약을 자연스럽게 만족한다.

### 2.3 모델 구조

#### Rev-ViT (Reversible Vision Transformer)

- **Two-Residual-Stream 구조**: 입력 $\mathbf{I}_1$과 $\mathbf{I}_2$가 각각 자신의 잔차 스트림을 유지하며, 함수 $F$(Multi-Head Attention)와 $G$(MLP)를 통해 정보를 교환.
- **시작(Initiation)**: 패치화(patchification) 출력을 $\mathbf{I}_1$과 $\mathbf{I}_2$ 양쪽에 동일하게 전송 (채널 분할이 아닌 복제 방식).
- **종료(Termination)**: Layer Normalization 후 연결(concatenation)하여 최종 분류기 헤드에 입력.
- **잔차 연결 재구성(Reconfiguring Residual Connections)**: MLP 및 Attention 서브블록 내부의 잔차 연결(internal skip connections)을 **제거**. 이는 가역 변환 $T$ 자체에 이미 내재된 스킵 연결이 존재하기 때문이며, 내부 스킵 연결은 깊은 모델에서 학습 수렴 불안정성을 초래함.

| 모델 | 구성 | FLOPs | 파라미터 | 메모리/이미지 | Top-1 정확도 |
|------|------|-------|---------|-------------|------------|
| Rev-ViT-S | $[F: \text{MHA}(384), G: \text{MLP}(1536)] \times 12$ | 4.6G | 22M | 8.8MB | 79.9% |
| Rev-ViT-B | $[F: \text{MHA}(768), G: \text{MLP}(3072)] \times 12$ | 17.6G | 87M | 17.0MB | 81.8% |
| Rev-ViT-L | $[F: \text{MHA}(1024), G: \text{MLP}(4096)] \times 24$ | 61.6G | 305M | 22.6MB | 81.4% |

#### Rev-MViT (Reversible Multiscale Vision Transformer)

MViT의 다중 스케일 특징 계층을 가역적으로 구현하기 위해 두 종류의 블록을 사용:

1. **Stage-Transition Block** (Figure 2b):
   - **Lateral Connections**: $\mathbf{I}_1$과 $\mathbf{I}_2$를 융합하여 해상도 다운샘플링 및 채널 업샘플링 수행.
   - **Feature Upsampling 재배치**: 채널 업샘플링을 기존 MLP 블록이 아닌 Pooling Attention 서브블록 내부의 Q/K/V 선형 층 이후로 이동. 이를 통해 (A) 모든 차원 변경이 하나의 블록 내에서 동기화되어 등차원 제약을 만족하고, (B) 이전 MLP 및 풀링 층의 불필요한 연산을 절약.

2. **Stage-Preserving Block** (Figure 2c):
   - 입출력 차원을 보존하는 가역 블록으로, 네트워크 연산과 메모리 사용의 **대부분**을 차지.
   - Multi-Head Pooling Attention을 포함하지만, K/V에 대한 풀링이 시퀀스 길이만 변경하고 출력 차원은 보존하므로 등차원 제약을 만족.

### 2.4 성능 분석

#### 이미지 분류 (ImageNet-1K)

| 모델 | 정확도 | 메모리(MB/img) | 메모리 절감 | 최대 배치 크기 | 배치 크기 증가 |
|------|-------|-------------|---------|-----------|-----------|
| ViT-S → Rev-ViT-S | 79.9 → 79.9 | 66.5 → 8.8 | **7.5×** | 207 → 1232 | **5.9×** |
| ViT-B → Rev-ViT-B | 81.8 → 81.8 | 129.7 → 17.0 | **7.6×** | 95 → 602 | **6.3×** |
| ViT-L → Rev-ViT-L | 81.5 → 81.4 | 349.3 → 22.6 | **15.5×** | 26 → 341 | **13.1×** |
| MViT-B → Rev-MViT-B | 82.8 → 82.5 | 153.6 → 66.8 | **2.3×** | 89 → 157 | **1.8×** |

핵심 관찰: **모델이 깊어질수록 메모리 절감 효과가 극대화**됨 (ViT-S: 7.5× → ViT-L: 15.5×).

#### 비디오 분류 (Kinetics-400/600)

- Kinetics-400: Rev-MViT-B-16 (78.5% top-1)이 MViT-B-16 (78.4%)과 정확도를 유지하면서 메모리 50% 절감, 배치 크기 2× 증가.
- Kinetics-600: Rev-MViT-B-24 (83.7%)가 MViT-B-24 (83.8%)와 거의 동일한 성능에서 메모리 62.7% 절감, 배치 크기 3.5× 증가.

#### 객체 탐지 (MS-COCO)

- Rev-MViT-B: $AP^{box}$ 48.0 / $AP^{mask}$ 43.5 (vs. MViT-B: 48.2 / 43.9)로 메모리 **1.7배** 절감.

#### 학습 처리량 (Throughput)

- 깊은 모델(80 layers)에서 Rev-MViT가 MViT 대비 최대 **2.3×** 높은 처리량을 달성.
- 작은 모델(12 layers)에서는 활성화 재계산 오버헤드로 인해 처리량이 약간 낮지만(98.5 vs. 86.0 imgs/s), 모델이 깊어지고 해상도가 높아질수록 이 오버헤드가 메모리 대역폭 절감 이점에 의해 상쇄됨.

### 2.5 한계

1. **활성화 재계산의 추가 연산 비용**: 역방향 전파 시 활성화를 재계산해야 하므로 추가적인 FLOPs가 소요됨. 작고 얕은 모델에서는 오히려 처리량 감소를 초래할 수 있음.
2. **등차원 제약**: 가역 변환의 특성상 함수 $F$와 $G$의 입출력 차원이 동일해야 하므로, 해상도/채널 변경이 필요한 계층적 구조(MViT)에서 Stage-Transition 블록은 가역적으로 만들 수 없어 메모리 절감 효과가 제한됨 (Rev-MViT의 절감폭이 Rev-ViT보다 작은 이유).
3. **학습 레시피 재설계 필요**: 가역 구조의 강한 내재적 정규화로 인해 기존 학습 레시피를 그대로 사용하면 최적 성능에 도달하지 못하며, augmentation 강도, stochastic depth, weight decay 등을 모델별로 재조정해야 함.
4. **내부 잔차 연결 제거에 따른 설계 제약**: 깊은 모델에서의 학습 안정성을 위해 서브블록 내부 잔차 연결을 제거해야 하므로, 기존 ViT/MViT 블록과 구조적으로 상이해짐.
5. **나이브 적용 시 깊은 모델에서 실패**: 기존 가역 구조를 비전 트랜스포머에 단순 적용하면 8블록 이상에서 학습 수렴 불안정성이 발생하며, 내부 잔차 연결 재구성이 필수적임 (Figure 3a).

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 강한 내재적 정규화 (Stronger Inherent Regularization)

이 논문의 가장 흥미로운 발견 중 하나는 **가역 구조가 비가역 구조보다 더 강한 내재적 정규화 효과**를 가진다는 것이다.

Table 5에서 보듯이, 동일한 FLOPs/파라미터 조건에서 Rev-ViT-B의 학습 레시피를 단계적으로 개선하는 과정은 다음과 같다:

| 학습 개선 | Train Acc | Top-1 ImageNet Acc |
|---------|-----------|-------------------|
| Naïve Rev-ViT-B | 15.3 | 12.1 |
| + 잔차 연결 재구성 | 82.1 | 77.2 |
| + Repeated Augmentation | 84.9 | 80.6 |
| + 가벼운 Augmentation magnitude | 93.2 | 81.0 |
| + 강한 Stochastic Depth | 92.0 | 81.4 |
| + 높은 Weight Decay | 91.0 | **81.8** |

핵심 관찰:
- **Augmentation magnitude를 9에서 7로 줄이면** 학습 정확도가 84.9→93.2로 크게 상승하면서 테스트 정확도도 80.6→81.0으로 향상됨. 이는 가역 구조가 이미 충분한 정규화를 제공하고 있어, 과도한 augmentation이 오히려 학습을 방해함을 시사.
- **Weight decay를 높이면** 학습 정확도가 92.0→91.0으로 적절히 감소하면서 테스트 정확도가 81.4→81.8로 향상됨. 이는 가역 구조의 정규화와 weight decay가 시너지를 발휘하여 **일반화 갭(generalization gap)을 효과적으로 줄임**을 보여줌.

### 3.2 일반화 관점에서의 구조적 분석

가역 구조의 강한 정규화는 다음 요인에서 비롯되는 것으로 추정된다:

1. **내부 잔차 연결 제거**: 서브블록 내부의 직접적 스킵 연결 대신, 가역 변환 $T$의 구조적 스킵 연결만을 통해 기울기가 전파됨. 이는 정보 흐름의 경로를 제한하여 암묵적 정규화 효과를 발생시킴.

2. **Two-residual-stream 구조**: 두 스트림 간의 상호 의존적 정보 교환 구조가 각 스트림의 표현 학습을 제약하여, 과적합을 억제하는 효과를 가질 수 있음.

3. **활성화 재계산에 따른 수치적 차이**: 부동소수점 연산의 비결합성(non-associativity)으로 인해 재계산된 활성화가 캐싱된 활성화와 미세하게 다를 수 있으며, 이것이 일종의 노이즈 주입 역할을 할 수 있음. 다만, Figure 3a에서 캐싱 유/무에 따른 성능 차이가 거의 없음을 보여, 이 효과는 미미한 것으로 판단됨.

### 3.3 Lateral Fusion과 일반화

Table 6의 Lateral Fusion 전략 분석은 일반화 성능과 모델 용량 사이의 trade-off를 명확히 보여준다:

- **2×-MLP**: 학습 정확도 80.2%, Top-1 81.8% → 적절한 용량 확장
- **4×-MLP**: 학습 정확도 80.4%, Top-1 82.3% → 일반화 갭 확대, 과적합 경향
- **Concatenation**: 학습 정확도 79.1%, Top-1 82.0% → 보수적 접근

이는 가역 구조에서 **lateral connection의 설계가 일반화 성능에 직접적 영향**을 미치며, 과도한 용량 확장은 오히려 일반화를 해칠 수 있음을 시사한다.

### 3.4 일반화 향상 가능성을 위한 방향

- **더 깊은 모델 스케일링**: 가역 구조는 메모리 제약 없이 모델 깊이를 확장할 수 있으므로, 깊이를 통한 표현력 향상이 가능하며, 내재적 정규화 덕분에 과적합 위험이 완화됨.
- **배치 크기 증가**: 메모리 절감으로 더 큰 배치 크기로 학습할 수 있어, 배치 정규화 효과 개선 및 학습 안정성 향상에 기여.
- **다운스트림 태스크 전이**: 강한 내재적 정규화는 사전 학습된 표현의 전이 학습 시에도 과적합을 억제하여 일반화 성능 향상에 기여할 가능성이 있음.

---

## 4. 향후 연구에 미치는 영향과 고려할 점

### 4.1 연구에 미치는 영향

1. **메모리 효율적 대규모 모델 학습 패러다임**: 이 논문은 가역 구조가 비전 트랜스포머의 깊이 확장에 실질적 해법이 됨을 입증. 향후 수십~수백 층 깊이의 비전 트랜스포머 개발에 직접적 기반을 제공.

2. **메모리 집약적 도메인의 backbone 혁신**: 비디오 인식, 3D 의료 영상, point cloud 처리 등 메모리가 극도로 제한되는 도메인에서 가역 backbone의 활용이 확대될 것으로 예상.

3. **학습 레시피 설계의 새로운 관점**: 아키텍처의 구조적 특성(가역성)이 정규화에 미치는 영향을 발견함으로써, 향후 새로운 아키텍처 설계 시 학습 레시피의 공동 최적화 필요성을 강조.

4. **후속 연구 촉진**: Chen et al. [77]이 Reversible Swin Transformers를 temporal action localization에 적용하고, Zhu [80]가 활성화 재계산을 기울기 계산과 병렬화하는 기법을 제안하는 등, 이미 후속 연구가 활발히 진행 중.

### 4.2 향후 연구 시 고려할 점

1. **Numerical Stability**: 매우 깊은 가역 모델에서 순차적 역변환 시 부동소수점 오차가 누적될 수 있으므로, 수치적 안정성 검증이 필수적.

2. **Mixed Precision Training과의 상호작용**: FP16/BF16 혼합 정밀도 학습에서 가역 재계산의 수치적 정확도가 학습 안정성에 미치는 영향 분석 필요.

3. **Self-Supervised Learning과의 결합**: MAE, DINO 등 자기지도 학습 패러다임에서 가역 구조의 정규화 특성이 어떤 영향을 미치는지 탐구 필요.

4. **등차원 제약 극복**: 현재 Stage-Transition 블록은 완전 가역이 아니므로, 차원 변경을 허용하는 새로운 가역 변환 설계가 필요.

5. **추론 시 메모리 효율**: 이 논문은 학습 시 메모리 효율에 초점을 맞추지만, 추론 시에는 중간 활성화 저장이 불필요하므로 가역 구조의 추론 메모리 절감 효과는 제한적일 수 있음.

6. **Attention 메커니즘의 다양화**: FlashAttention 등 메모리 효율적 attention과 가역 구조의 시너지 효과 탐구.

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

| 연구 | 연도 | 핵심 접근법 | Rev-ViT과의 비교 |
|------|------|-----------|---------------|
| **Reformer** (Kitaev et al.) [38] | 2020 | LSH attention + reversible layers for NLP | 시퀀스 길이에 초점, 깊이 확장 미고려. 비전 태스크 벤치마크 없음 |
| **Swin Transformer** (Liu et al.) [48] | 2021 | Shifted window attention | 메모리 효율 고려 없음. Rev-ViT-L이 유사 정확도에서 15.5× 메모리 절감 |
| **MViT** (Fan et al.) [18] | 2021 | Multi-scale pooling attention | Rev-MViT의 기반 모델. 메모리 4.5× 절감 가능 |
| **ViViT** (Arnab et al.) [1] | 2021 | Factorized video ViT | 3992 GFLOPs, 310M params. Rev-MViT는 훨씬 효율적 (223 GFLOPs, 51.8M params) |
| **TimeSformer** (Bertasius et al.) [3] | 2021 | Divided space-time attention | 1703 GFLOPs, 121.4M params. 메모리 효율 미고려 |
| **DeiT** (Touvron et al.) [63] | 2021 | Knowledge distillation for ViT | Rev-ViT의 ViT-S/B 레시피 기반. 가역 구조와 직교적 기법으로 결합 가능 |
| **CSWin** (Dong et al.) [14] | 2021 | Cross-shaped window attention | 4.3 GFLOPs에서 82.7% top-1. 가역 적용 시 추가 메모리 절감 가능 |
| **Re²TAL** (Chen et al.) [77] | 2022 | Reversible Swin for TAL | Rev-ViT의 설계를 Swin에 적용, 시간적 행동 국소화에서 end-to-end 학습 가능 |
| **Speeding up Rev-ViT** (Zhu) [80] | 2022 | Staggered activation recomputation | 활성화 재계산과 기울기 계산의 비동기 병렬화로 학습 속도 추가 향상 |

### 주요 비교 분석 요약

**메모리 효율 관점**: 기존 ViT, Swin, MViT 등은 모두 메모리가 깊이에 선형 비례하여 증가하지만, Rev-ViT/Rev-MViT는 이를 **상수적(constant)으로** 유지. 이는 gradient checkpointing이나 model parallelism 없이 달성된 구조적 해결책이라는 점에서 차별화됨.

**성능 유지 관점**: Reformer [38]가 NLP에서 reversible 구조를 적용했지만 비전 태스크에서의 검증이 없었고, 이 논문이 비전 분야 최초로 가역 트랜스포머의 성능 유지를 입증.

**확장성 관점**: 깊이 80층까지 확장 시 2.3× 처리량 향상은 gradient checkpointing(일반적으로 33% 추가 연산 비용)보다 우수한 효율을 보여주며, 특히 memory-bound 환경에서의 장점이 극대화됨.

---

## 참고자료

1. **Mangalam, K., Fan, H., Li, Y., Wu, C.-Y., Xiong, B., Feichtenhofer, C., & Malik, J.** (2023). *Reversible Vision Transformers*. arXiv:2302.04869. (본 분석의 주요 논문)
2. **Gomez, A. N., Ren, M., Urtasun, R., & Grosse, R. B.** (2017). *The Reversible Residual Network: Backpropagation Without Storing Activations*. NeurIPS.
3. **Dosovitskiy, A., et al.** (2021). *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale*. ICLR.
4. **Fan, H., et al.** (2021). *Multiscale Vision Transformers*. ICCV.
5. **Kitaev, N., Kaiser, Ł., & Levskaya, A.** (2020). *Reformer: The Efficient Transformer*. arXiv:2001.04451.
6. **Liu, Z., et al.** (2021). *Swin Transformer: Hierarchical Vision Transformer Using Shifted Windows*. ICCV.
7. **Dinh, L., Krueger, D., & Bengio, Y.** (2014). *NICE: Non-linear Independent Components Estimation*. arXiv:1410.8516.
8. **Gholami, A., et al.** (2021). *AI and Memory Wall*. RiseLab Medium Post.
9. **Zhao, C., Liu, S., Mangalam, K., & Ghanem, B.** (2022). *Re²TAL: Rewiring Pretrained Video Backbones for Reversible Temporal Action Localization*. arXiv:2211.14053.
10. **Zhu, T.** (2022). *Speeding up Reversible Vision Transformers*. http://bit.ly/3J6Q0Cb.
11. **Touvron, H., et al.** (2021). *Training Data-Efficient Image Transformers & Distillation Through Attention*. ICML.
12. **GitHub Repository**: https://github.com/facebookresearch/slowfast 및 https://github.com/karttikeya/minREV
