# EfficientViT: Memory Efficient Vision Transformer with Cascaded Group Attention

---

## 1. 핵심 주장 및 주요 기여 요약

**핵심 주장:** Vision Transformer(ViT)의 추론 속도는 FLOPs나 파라미터 수보다 **메모리 접근 비효율성(memory-bound operations)**에 의해 주로 제한되며, 이를 체계적으로 해결하면 정확도를 유지하면서 실시간 추론이 가능한 고속 ViT를 설계할 수 있다.

**주요 기여:**
1. **체계적 속도 병목 분석:** 메모리 접근(memory access), 연산 중복(computation redundancy), 파라미터 사용(parameter usage) 세 관점에서 ViT의 추론 속도 저하 요인을 실증적으로 분석하고 설계 지침을 도출.
2. **Sandwich Layout 블록:** 메모리 비효율적인 MHSA 레이어를 최소화하고 FFN 레이어를 늘린 새로운 블록 구조 제안.
3. **Cascaded Group Attention (CGA):** 어텐션 헤드 간 중복성을 제거하고 다양성을 높이는 새로운 어텐션 모듈 제안.
4. **파라미터 재분배(Parameter Reallocation):** Taylor structured pruning 기반 분석을 통해 Q, K, V, FFN 채널을 최적으로 재분배.
5. **SOTA 대비 우수한 속도-정확도 트레이드오프:** EfficientViT-M5는 MobileNetV3-Large 대비 정확도 1.9%↑, GPU 처리량 40.4%↑, CPU 처리량 45.2%↑ 달성.

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

기존 ViT 모델들은 높은 성능에도 불구하고 **실시간 추론에 부적합한 무거운 연산 비용**을 수반한다. 기존 경량화 연구들은 FLOPs나 파라미터 수 감소에 집중하지만, 이 지표들은 **실제 추론 처리량(wall-clock throughput)과 상관관계가 낮다.** 예를 들어, MobileViT-XS(700M FLOPs)는 DeiT-T(1,220M FLOPs)보다 Nvidia V100 GPU에서 오히려 느리게 동작한다.

논문은 세 가지 핵심 병목을 식별한다:

1. **메모리 비효율성:** MHSA의 텐서 리셰이핑(reshape), 요소별 연산(element-wise addition), 정규화(normalization) 등이 메모리 바운드 연산으로서 전체 런타임의 상당 부분을 차지한다 (Swin-T에서 ~44.9%, DeiT-T에서 ~32.2%).
2. **연산 중복:** 다수의 어텐션 헤드가 유사한 선형 프로젝션을 학습하여 어텐션 맵 간 높은 코사인 유사도를 보임 (특히 후반 블록에서 심화).
3. **파라미터 비효율:** Q, K, V에 동일 차원을 할당하고, FFN 확장비를 4로 설정하는 NLP 트랜스포머의 관행이 경량 ViT에서는 비효율적.

### 2.2 제안하는 방법 (수식 포함)

#### (A) Sandwich Layout Block

MHSA 레이어의 비율을 줄이고 FFN 레이어를 늘린 샌드위치 구조를 제안한다. 하나의 self-attention 레이어 $\Phi_i^A$를 $N$개의 FFN 레이어 $\Phi_i^F$로 감싸는 구조:

$$X_{i+1} = \prod_{N} \Phi_i^F \left( \Phi_i^A \left( \prod_{N} \Phi_i^F(X_i) \right) \right) $$

여기서 $X_i$는 $i$번째 블록의 입력 특징이며, $\prod_N$은 $N$개의 FFN을 순차적으로 적용하는 것을 의미한다. 실험적으로 MHSA 비율이 20%–40%일 때 최적의 정확도-속도 트레이드오프가 달성됨을 확인했다 (기존 ViT는 50% 사용).

각 FFN 앞에는 **Depthwise Convolution (DWConv)** 기반 토큰 상호작용 레이어를 추가하여 로컬 구조 정보의 귀납적 편향(inductive bias)을 도입한다.

#### (B) Cascaded Group Attention (CGA)

기존 MHSA가 모든 헤드에 동일한 전체 특징을 입력하는 것과 달리, CGA는 입력 특징을 헤드 수만큼 분할(split)하여 각 헤드에 서로 다른 부분을 공급한다:

$$\widetilde{X}_{ij} = \text{Attn}(X_{ij} W_{ij}^Q,\; X_{ij} W_{ij}^K,\; X_{ij} W_{ij}^V) $$

$$\widetilde{X}_{i+1} = \text{Concat}[\widetilde{X}_{ij}]_{j=1:h} \cdot W_i^P$$

여기서:
- $X_i = [X_{i1}, X_{i2}, \ldots, X_{ih}]$: 입력 특징의 채널 방향 분할
- $j$번째 헤드는 $j$번째 분할 $X_{ij}$에 대해 self-attention 수행
- $W_{ij}^Q, W_{ij}^K, W_{ij}^V$: 각 헤드의 프로젝션 행렬
- $W_i^P$: 출력 프로젝션 행렬

**캐스케이드(cascade)** 메커니즘을 통해 이전 헤드의 출력을 다음 헤드의 입력에 더하여 점진적으로 특징 표현을 정제한다:

$$X'_{ij} = X_{ij} + \widetilde{X}_{i(j-1)}, \quad 1 < j \leq h $$

여기서 $X'\_{ij}$가 $X_{ij}$를 대체하여 $j$번째 헤드의 새로운 입력으로 사용된다.

**CGA의 이점:**
- 입출력 채널이 $h$배 감소하여 FLOPs와 파라미터가 $h$배 절약됨
- 캐스케이드 구조로 네트워크 깊이가 효과적으로 증가하여 모델 용량 향상 (추가 파라미터 없이)
- 어텐션 맵의 다양성 증가 (코사인 유사도 감소 실증)

#### (C) Parameter Reallocation

Taylor structured pruning 분석 결과에 기반한 파라미터 재분배 전략:
- **Q, K 차원 축소:** 각 헤드당 작은 채널 차원 사용
- **V 차원 확장:** 입력 임베딩 차원과 동일한 크기 유지 (V가 성능에 중요)
- **FFN 확장비 축소:** 기존 4에서 2로 감소 (FFN에 상당한 파라미터 중복 존재)

### 2.3 모델 구조

EfficientViT는 **3단계(stage) 계층적 구조**를 채택한다:

| 구성 요소 | 설명 |
|----------|------|
| **입력 임베딩** | Overlapping Patch Embedding (16×16 패치 → $C_1$ 차원 토큰) |
| **Stage 1~3** | EfficientViT 빌딩 블록 × $L_i$ (각 스테이지 후 2× 해상도 축소) |
| **서브샘플링** | Inverted residual block 기반 EfficientViT Subsample 블록 |
| **정규화** | BatchNorm (LN 대비 추론 시 linear/conv에 fold 가능) |
| **활성화 함수** | ReLU (GELU/HardSwish 대비 빠르고 배포 호환성 높음) |
| **출력** | Average Pooling + Classifier |

**모델 변형 (M0~M5):**

| Model | $\{C_1, C_2, C_3\}$ | $\{L_1, L_2, L_3\}$ | $\{H_1, H_2, H_3\}$ |
|-------|----------------------|----------------------|----------------------|
| M0 | {64, 128, 192} | {1, 2, 3} | {4, 4, 4} |
| M5 | {192, 288, 384} | {1, 3, 4} | {3, 3, 4} |

### 2.4 성능 향상

**ImageNet-1K 주요 결과:**
- **EfficientViT-M5:** 77.1% Top-1 정확도, GPU 처리량 10,621 images/s
  - vs. MobileNetV3-Large: +1.9% 정확도, +40.4% GPU 속도, +45.2% CPU 속도
  - vs. EfficientNet-B0: 동등 정확도(77.1%), 2.3× GPU 속도, 1.9× CPU 속도
- **EfficientViT-M2:** 70.8% Top-1 정확도
  - vs. MobileViT-XXS: +1.8% 정확도, 5.8× GPU 속도, 3.7× CPU 속도, 7.4× ONNX 속도
- **고해상도 파인튜닝:** M5↑512에서 80.8% Top-1 정확도 달성

**Ablation 결과 (100 epoch 기준, M4):**

| 변경 사항 | Top-1 변화 |
|----------|-----------|
| Sandwich → Swin 블록 | -3.0% |
| CGA → MHSA | -1.1% |
| Cascade 제거 | -1.5% |
| QKV 재분배 제거 | -1.4% |
| FFN ratio 2→4 | -1.5% |

### 2.5 한계

1. **모델 크기:** 샌드위치 레이아웃의 추가 FFN으로 인해 MobileNetV3 등 SOTA efficient CNN 대비 파라미터 수가 다소 큼 (M5: 12.4M vs. MobileNetV3-Large: 5.4M).
2. **수동 설계:** 모델이 도출된 가이드라인에 따라 수동으로 설계되었으며, NAS(Neural Architecture Search)를 활용하지 않음.
3. **ONNX 성능:** 리셰이핑 연산이 ONNX에서 느려 일부 설정에서 MobileNetV3 대비 약간 느린 ONNX 처리량 보임.
4. **세밀한 분류 한계:** Stanford Cars 등 로컬 디테일이 중요한 데이터셋에서 CNN 대비 약간 열세.

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 논문에서 제시하는 일반화 관련 근거

#### (A) 다운스트림 태스크 전이 성능
EfficientViT-M5는 다양한 다운스트림 분류 데이터셋에서 우수한 전이 학습 능력을 입증하였다:

| 데이터셋 | EfficientViT-M5 | MobileNetV3 | ViT-S/16 |
|---------|-----------------|-------------|----------|
| CIFAR-10 | **98.0** | 97.6 | 97.6 |
| CIFAR-100 | **86.4** | 85.5 | 85.7 |
| Flowers | **97.1** | 97.0 | 86.4 |
| Pets | **92.0** | 90.1 | 90.4 |

CNN과 대형 ViT 모두를 능가하는 전이 성능을 보여 **범용적 특징 표현 학습 능력**이 우수함을 시사한다.

#### (B) 객체 탐지 전이 성능
COCO val2017에서 RetinaNet 기반 실험 결과, EfficientViT-M4는 MobileNetV2 대비 +4.4% AP를 달성하며, SPOS 대비 18.1% 적은 FLOPs로 +2.0% AP 향상을 보여 **탐지 태스크에 대한 일반화 능력**을 입증.

#### (C) 고해상도 파인튜닝
M5↑384(79.8%)와 M5↑512(80.8%)의 결과는 모델이 다양한 입력 해상도에 대해 **강건하게 확장 가능**함을 보여준다.

#### (D) 1,000 에폭 학습 + 지식 증류
장기 학습 및 지식 증류 적용 시 EfficientViT-M4는 ImageNet-ReaL에서 LeViT-128S 대비 +1.0%의 향상을 보여, **학습 스케줄 확장에 대한 강한 수용력**을 갖추고 있음.

### 3.2 일반화 성능 향상의 구조적 요인

1. **CGA의 어텐션 다양성:** 각 헤드에 서로 다른 특징 분할을 입력함으로써 다양한 패턴을 학습하여, 단일 태스크에 과적합되지 않는 **다양한 표현**을 획득한다.

2. **DWConv 기반 토큰 상호작용:** 로컬 구조 정보의 귀납적 편향을 도입하여, 순수 self-attention의 데이터 의존적 학습 한계를 보완하고 **제한된 데이터에서의 일반화**를 돕는다.

3. **BatchNorm 사용:** LayerNorm 대비 미니배치 통계에 기반한 정규화 효과를 제공하여 일반화에 기여할 수 있다. 단, 이로 인해 배치 크기에 대한 민감성이 증가할 가능성이 있다.

4. **캐스케이드 구조의 암묵적 깊이 증가:** 추가 파라미터 없이 네트워크의 효과적 깊이를 증가시켜, 보다 복잡한 특징 계층을 학습할 수 있어 일반화 능력 향상에 기여한다.

### 3.3 일반화 성능 향상을 위한 미래 방향

- **NAS 통합:** 수동 설계를 자동 검색으로 대체하면 특정 태스크/하드웨어에 최적화된 구조를 탐색할 수 있어 일반화 성능 추가 향상 가능.
- **자기지도 학습(Self-supervised Learning):** MAE, DINO 등의 사전학습 기법과 결합하면 라벨이 부족한 도메인에서의 일반화가 개선될 수 있음.
- **다양한 데이터 증강 및 정규화 기법:** Knowledge distillation(이미 일부 실험됨)과 함께 CutMix, RandAugment 등의 추가 적용 가능.

---

## 4. 앞으로의 연구에 미치는 영향 및 고려할 점

### 4.1 연구에 미치는 영향

1. **메모리 효율성 중심의 설계 패러다임 전환:** 이 논문은 FLOPs/파라미터 수가 아닌 **실제 추론 처리량(throughput)**을 최적화 목표로 삼아야 한다는 관점을 강화하였다. 이는 이후 EfficientFormer, FastViT, EfficientViT (MIT 버전) 등 후속 연구들의 설계 철학에 직접적 영향을 미쳤다.

2. **어텐션 헤드 중복성 문제의 정량적 분석:** 코사인 유사도 기반으로 헤드 간 중복성을 측정하고 이를 구조적으로 해결한 접근은, 효율적 어텐션 설계의 새로운 방향을 제시하였다.

3. **그룹 어텐션의 ViT 적용:** CNN에서의 그룹 컨볼루션 아이디어를 self-attention에 적용한 CGA는 후속 연구에서 **하이브리드 효율성 설계**의 영감을 제공한다.

4. **파라미터 재분배 분석:** Taylor pruning을 통한 Q, K, V, FFN 중요도 분석은 경량 트랜스포머 설계에서 파라미터 배분의 경험적 지침을 제공하며, 이후 연구에서 널리 참조될 수 있는 방법론이다.

### 4.2 앞으로 연구 시 고려할 점

1. **하드웨어 특화 최적화:** GPU, CPU, 모바일 칩셋, NPU 등 다양한 하드웨어에서의 성능 평가가 필수적이며, 특정 하드웨어의 메모리 계층에 맞춘 설계가 중요하다.

2. **모델 크기 vs. 속도 트레이드오프:** EfficientViT는 높은 처리량에도 불구하고 파라미터 수가 다소 크다. 엣지 디바이스에서는 모델 크기 자체가 제약이 될 수 있으므로, 메모리 풋프린트와 속도를 동시에 최적화하는 연구가 필요하다.

3. **동적 해상도/토큰 처리:** 입력 해상도나 토큰 수에 따라 연산량을 동적으로 조절하는 적응적 설계를 통해 추가적인 효율성 향상이 가능하다.

4. **대규모 사전학습과의 결합:** 이 논문은 ImageNet-1K 학습에 초점을 맞추고 있으나, ImageNet-21K, JFT 등 대규모 데이터셋이나 자기지도 학습과의 결합 효과를 탐구할 필요가 있다.

5. **Downstream 태스크 확장:** Semantic segmentation, video understanding, 3D vision 등 보다 다양한 다운스트림 태스크에서의 검증이 필요하다.

6. **Activation/Normalization 선택의 배포 환경 고려:** ReLU와 BatchNorm의 선택은 추론 효율에는 유리하지만, 학습 안정성이나 소규모 배치에서의 성능 저하 가능성을 고려해야 한다.

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

| 연구 | 발표년도/학회 | 핵심 아이디어 | EfficientViT와의 비교 |
|------|-----------|-----------|-------------------|
| **DeiT** [Touvron et al.] | 2021/ICML | 지식 증류 기반 데이터 효율적 ViT 학습 | EfficientViT는 DeiT-T 대비 훨씬 높은 처리량을 달성하면서 유사 정확도 유지. DeiT는 속도 최적화가 아닌 학습 효율에 초점 |
| **Swin Transformer** [Liu et al.] | 2021/ICCV | Shifted window 기반 계층적 ViT | EfficientViT의 분석 기반선으로 활용됨. Swin-T 대비 EfficientViT-M5는 4.1% 정확도 열세이나 12.3× CPU 속도 우위 |
| **MobileViT** [Mehta & Rastegari] | 2021/ICLR | CNN-Transformer 하이브리드 모바일 모델 | EfficientViT-M2가 MobileViT-XXS 대비 +1.8% 정확도, 5.8× GPU 속도 |
| **MobileViTV2** [Mehta & Rastegari] | 2022/arXiv | Separable self-attention으로 선형 복잡도 달성 | EfficientViT-M2가 MobileViTV2-0.5 대비 유사 정확도에서 3.4× GPU 처리량 |
| **EfficientFormer** [Li et al.] | 2022/NeurIPS | MobileNet 속도급 ViT, 4D 특징 기반 MHSA 선택적 사용 | EfficientFormer-L1(79.2%)은 정확도 우위이나 EfficientViT-M5가 2.4× GPU 처리량 우위 |
| **EdgeViT** [Pan et al.] | 2022/ECCV | 로컬-글로벌 어텐션 분리, sparse attention | EfficientViT-M4가 EdgeViT-XXS와 유사 정확도에서 4.4× GPU 속도, 3.7× ONNX 속도 |
| **MobileOne** [Vasu et al.] | 2022/arXiv | Re-parameterization 기반 1ms 모바일 백본 | MobileOne-S1(75.9%)과 EfficientViT-M5(77.1%) 비교 시 EfficientViT가 +1.2% 정확도, 1.6× GPU 속도 |
| **FastViT** [Vasu et al.] | 2023/ICCV | RepMixer + structural reparameterization | FastViT는 모바일 하드웨어에서 추가적 최적화를 제공하지만, EfficientViT의 메모리 효율성 분석과 보완적 |
| **EfficientViT (MIT)** [Cai et al.] | 2023/ICCV | Linear attention 기반 세그멘테이션 특화 EfficientViT | 이름은 같지만 별개 연구. MIT 버전은 세그멘테이션에 초점, 본 논문은 분류/탐지에서의 속도-정확도 트레이드오프에 초점 |
| **FlashAttention** [Dao et al.] | 2022/NeurIPS | IO-aware exact attention, 메모리 접근 최적화 | EfficientViT의 메모리 효율 분석과 상호 보완적. FlashAttention은 소프트웨어 수준, EfficientViT는 아키텍처 수준 최적화 |
| **MetaFormer / PoolFormer** [Yu et al.] | 2022/CVPR | Token mixer의 종류보다 전체 구조(MetaFormer)가 중요 | PoolFormer-12S는 EfficientViT-M5와 유사 정확도이나 3.0× GPU 속도 열세 |
| **LeViT** [Graham et al.] | 2021/ICCV | CNN 스타일 추론을 위한 ViT, 하드웨어 친화적 설계 | EfficientViT-M4가 LeViT-128S 대비 +0.5% ImageNet, +1.0% ImageNet-ReaL, 34.2% ONNX 속도 향상 |
| **TinyViT** [Wu et al.] | 2022/ECCV | 대규모 모델로부터의 사전학습 증류 | 대규모 사전학습 데이터 활용 시 추가 성능 향상 가능, EfficientViT와 결합 가능한 접근 |

### 주요 트렌드 및 비교 분석

1. **FLOPs vs. Throughput 괴리 인식:** EfficientViT, EfficientFormer, MobileOne 등 2022년 이후 연구들은 FLOPs가 아닌 실제 추론 속도를 핵심 지표로 채택하는 추세이다.

2. **CNN-Transformer 하이브리드:** MobileViT, EdgeViT, Mobile-Former 등은 CNN과 Transformer를 결합하지만, EfficientViT는 순수 Transformer 기반이면서 DWConv을 보조적으로 사용하여 더 높은 처리량을 달성한다.

3. **어텐션 효율화 접근의 다양화:**
   - **Sparse/Linear attention:** Reformer, Performer, MobileViTV2 → 정확도 손실 가능성
   - **그룹 어텐션:** EfficientViT의 CGA → 정확도 유지하면서 중복 제거
   - **IO-aware 최적화:** FlashAttention → 소프트웨어 수준 최적화, 아키텍처와 직교적

4. **Re-parameterization 기법:** MobileOne, FastViT 등은 학습 시 다중 분기를 추론 시 단일 분기로 융합하여 속도를 높이는데, EfficientViT는 이를 사용하지 않아 추가적 속도 향상의 여지가 있다.

---

## 참고자료

1. **Xinyu Liu, Houwen Peng, Ningxin Zheng, Yuqing Yang, Han Hu, Yixuan Yuan.** "EfficientViT: Memory Efficient Vision Transformer with Cascaded Group Attention." *arXiv:2305.07027*, 2023. (본 논문)
2. **Hugo Touvron et al.** "Training data-efficient image transformers & distillation through attention." *ICML*, 2021. (DeiT)
3. **Ze Liu et al.** "Swin Transformer: Hierarchical vision transformer using shifted windows." *ICCV*, 2021.
4. **Sachin Mehta and Mohammad Rastegari.** "MobileViT: Light-weight, general-purpose, and mobile-friendly vision transformer." *ICLR*, 2021.
5. **Sachin Mehta and Mohammad Rastegari.** "Separable self-attention for mobile vision transformers." *arXiv:2206.02680*, 2022. (MobileViTV2)
6. **Yanyu Li et al.** "EfficientFormer: Vision Transformers at MobileNet Speed." *NeurIPS*, 2022.
7. **Junting Pan et al.** "EdgeViTs: Competing Light-weight CNNs on Mobile Devices with Vision Transformers." *ECCV*, 2022.
8. **Pavan Kumar Anasosalu Vasu et al.** "An Improved One Millisecond Mobile Backbone." *arXiv:2206.04040*, 2022. (MobileOne)
9. **Tri Dao et al.** "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness." *NeurIPS*, 2022.
10. **Benjamin Graham et al.** "LeViT: a Vision Transformer in ConvNet's Clothing for Faster Inference." *ICCV*, 2021.
11. **Weihao Yu et al.** "MetaFormer is Actually What You Need for Vision." *CVPR*, 2022. (PoolFormer)
12. **Andrew Howard et al.** "Searching for MobileNetV3." *ICCV*, 2019.
13. **Kan Wu et al.** "TinyViT: Fast Pretraining Distillation for Small Vision Transformers." *ECCV*, 2022.
14. **Alexey Dosovitskiy et al.** "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale." *ICLR*, 2021. (ViT)
15. **Ashish Vaswani et al.** "Attention Is All You Need." *NeurIPS*, 2017. (원본 Transformer)
16. **Pavlo Molchanov et al.** "Importance Estimation for Neural Network Pruning." *CVPR*, 2019. (Taylor structured pruning)
17. **Pavan Kumar Anasosalu Vasu et al.** "FastViT: A Fast Hybrid Vision Transformer using Structural Reparameterization." *ICCV*, 2023.
18. **Han Cai, Junyan Li, Muyan Hu, Chuang Gan, Song Han.** "EfficientViT: Lightweight Multi-Scale Attention for High-Resolution Dense Prediction." *ICCV*, 2023. (MIT 버전 EfficientViT)
