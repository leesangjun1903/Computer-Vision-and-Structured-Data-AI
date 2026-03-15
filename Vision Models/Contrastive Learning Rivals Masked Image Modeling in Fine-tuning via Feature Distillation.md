# Contrastive Learning Rivals Masked Image Modeling in Fine-tuning via Feature Distillation

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장
Masked Image Modeling(MIM)이 fine-tuning에서 우수한 성능을 보이는 이유는 학습된 표현의 **"최적화 친화성(optimization friendliness)"** 때문이며, 이 속성은 간단한 **Feature Distillation(FD)** 후처리를 통해 다른 사전학습 방법(contrastive learning, CLIP, image classification 등)에도 부여할 수 있다. 이를 통해 contrastive learning 기반 방법들이 MIM과 동등한 fine-tuning 성능을 달성할 수 있다.

### 주요 기여
1. **범용적 Feature Distillation 프레임워크 제안**: 사전학습된 모델의 표현을 fine-tuning에 더 적합한 새로운 표현으로 변환하는 간단하고 효과적인 후처리 방법
2. **MIM 우수 성능의 원인 분석**: attention distance, attention map, loss landscape 등 진단 도구를 통해 MIM의 fine-tuning 우위가 "최적화 친화성"에서 기인함을 규명
3. **SOTA 달성**: CLIP ViT-L에서 ImageNet-1K 89.0% top-1 accuracy, SwinV2-G에서 ADE20K 61.4 mIoU / COCO 64.2 mAP 달성
4. **연구 방향 제시**: 최적화 친화성은 후처리로 해결 가능하므로, 향후 연구는 표현의 **일반성(generality)과 확장성(scalability)**에 집중해야 함을 제안

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

MIM(BEiT, MAE, SimMIM 등)은 fine-tuning 평가에서 탁월한 성능을 보이지만, 다른 유력한 사전학습 방법들—instance contrastive learning(DINO, MoCo), visual-language alignment(CLIP), image classification(DeiT)—은 fine-tuning 성능에서 열세를 보인다. 이 논문은 다음 질문에 답한다:

> *"MIM이 fine-tuning에서 훨씬 우수한 이유는 무엇이며, 다른 사전학습 방법에도 동일한 핵심 요소를 추가할 수 있는가?"*

### 2.2 제안하는 방법: Feature Distillation (FD)

#### 전체 구조

사전학습된 모델을 **teacher**, 새로 초기화된 동일 구조의 모델을 **student**로 설정하는 teacher-student 프레임워크이다. Teacher의 출력 feature map을 student가 모방하도록 학습한다.

#### 핵심 설계 요소

**(1) Feature Map 기반 증류 (logit 대신)**

Logit이 아닌 출력 feature map 전체를 증류 대상으로 사용하여, classification head가 없는 모델(CLIP 등)에도 적용 가능하고 더 높은 fine-tuning 정확도를 달성한다.

**(2) Teacher Feature Whitening**

서로 다른 사전학습 모델들의 feature 크기 스케일 차이 문제를 해결하기 위해, teacher의 출력 feature map에 비모수적 layer normalization(scaling/bias 없이)을 적용한다:

$$\text{whiten}(\mathbf{t}) = \frac{\mathbf{t} - \mu(\mathbf{t})}{\sigma(\mathbf{t})}$$

**(3) Smooth $\ell_1$ Loss**

Student와 teacher feature map 사이의 증류 손실 함수:

$$\mathcal{L}_{\text{distill}}(\mathbf{s}, \mathbf{t}) = \begin{cases} \frac{1}{2}(g(\mathbf{s}) - \text{whiten}(\mathbf{t}))^2 / \beta, & |g(\mathbf{s}) - \text{whiten}(\mathbf{t})| \leq \beta \\ |g(\mathbf{s}) - \text{whiten}(\mathbf{t})| - \frac{1}{2}\beta, & \text{otherwise} \end{cases}$$

여기서:
- $\beta = 2.0$ (기본값)
- $\mathbf{s}$, $\mathbf{t}$: 각각 student, teacher의 출력 feature vector
- $g(\cdot)$: $1 \times 1$ convolution layer (차원 맞춤용 projector)

**(4) Shared Relative Position Bias (RPB)**

Student 네트워크에서 모든 레이어가 동일한 relative position bias 행렬을 공유하는 설정을 사용한다. 이는 attention head의 다양성을 증가시키고 fine-tuning 성능을 향상시킨다.

**(5) Asymmetric Drop Path Rates**

Teacher에는 drop path를 적용하지 않고(정확한 teacher signal 유지), student에만 0.1~0.3의 drop path rate를 적용하여 overfitting을 방지한다.

### 2.3 모델 구조

- **Teacher**: 사전학습된 모델 (DINO, EsViT, CLIP, DeiT, MAE 등)의 가중치를 고정(frozen)
- **Student**: Teacher와 동일한 아키텍처(ViT-B, ViT-L, Swin-B 등)를 처음부터(from scratch) 학습
- Student 위에 $1 \times 1$ convolution projector를 추가하여 teacher-student 간 feature dimension 차이를 허용
- Student에 shared RPB를 적용 (원래 ViT의 APE 대신)
- ImageNet-1K 학습 이미지 1.28M으로 300 epoch 학습

### 2.4 성능 향상

| Method | Backbone | IN-1K (f.t.) | ADE20K (mIoU) |
|--------|----------|:------------:|:-------------:|
| DINO → FD-DINO | ViT-B | 82.8 → **83.8** (+1.0) | 46.2 → **47.7** (+1.5) |
| EsViT → FD-EsViT | Swin-B | 83.9 → **85.1** (+1.2) | 47.3 → **48.9** (+1.6) |
| CLIP → FD-CLIP | ViT-B | 82.9 → **84.9** (+2.0) | 49.5 → **52.8** (+3.3) |
| CLIP → FD-CLIP | ViT-L | 86.1 → **87.7** (+1.6) | 53.5 → **55.7** (+2.2) |
| DeiT → FD-DeiT | ViT-B | 81.8 → **83.0** (+1.2) | 47.0 → **48.0** (+1.0) |
| SwinV2-G → FD-SwinV2-G | SwinV2-G | 89.2 → **89.4** (+0.2) | 59.9 → **61.4** (+1.5) |

- FD-CLIP ViT-L은 ImageNet-22K intermediate fine-tuning과 $336^2$ 해상도로 **89.0%** top-1 accuracy 달성
- FD-SwinV2-G: COCO **64.2 mAP** (당시 새로운 기록)

### 2.5 한계

1. **MIM에 대한 추가 이득이 미미**: MAE에 FD를 적용하면 +0.2%만 향상 (83.6% → 83.8%). MIM은 이미 최적화 친화적 표현을 학습하므로 FD의 기여가 제한적
2. **추가 학습 비용**: 300 epoch의 추가 증류 학습이 필요 (CLIP의 경우 약 +3%의 추가 비용)
3. **Linear evaluation 성능 저하 가능성**: DINO의 경우 linear probe 성능이 78.2 → 76.1로 하락 (fine-tuning은 향상되지만 linear separability는 감소할 수 있음)
4. **표현의 일반성/확장성 자체를 개선하지 않음**: FD는 최적화 친화성만 개선하며, 표현이 인코딩하는 지식의 질이나 양을 근본적으로 변화시키지 않음

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 최적화 친화성을 통한 일반화

논문에서 사용한 4가지 진단 도구로 확인된 "최적화 친화적" 속성들은 일반화와 직접 연결된다:

**(1) Attention Head 다양성 증가**
- FD 전: 깊은 레이어에서 서로 다른 attention head들의 attention distance가 좁은 범위에 밀집 → 유사한 시각 단서만 포착하여 모델 용량 낭비
- FD 후: attention distance가 골고루 분포 → 다양한 스케일의 시각 패턴을 동시에 포착하여 다양한 downstream task에 적응 가능

**(2) 상대 위치 의존도 증가 (Translational Invariance)**
- FD 후 attention map에서 **대각선(diagonal) 패턴** 증가, **열(column) 패턴** 감소
- 대각선 패턴: 상대적 위치 관계 인코딩 → 이동 불변성(translational invariance) 향상
- 열 패턴: 절대적 위치에 의존 → 해상도/크기 변화에 취약
- 이는 object detection, semantic segmentation 등 다양한 공간적 태스크로의 전이(transfer)를 용이하게 함

**(3) Loss Landscape 평탄화**
- FD 후 loss landscape이 더 평탄(flat) → Gaussian noise perturbation에 대한 robustness 향상
- 평탄한 loss landscape는 일반화 성능 향상의 간접적 지표 (Li et al., 2017)

### 3.2 다양한 사전학습 방법에 대한 범용 적용 가능성

FD는 다음 모든 범주의 사전학습 모델에 효과적:
- **Self-supervised contrastive**: DINO, EsViT
- **Vision-language**: CLIP
- **Supervised classification**: DeiT
- **대규모 모델**: SwinV2-G (30억 파라미터)

이는 FD가 사전학습 방법에 독립적인(agnostic) 일반적 후처리 도구로 활용 가능함을 시사한다.

### 3.3 일반성(Generality)과 확장성(Scalability) 분리의 함의

논문의 핵심 통찰: 표현 학습의 두 가지 바람직한 속성—**(1) 최적화 친화성**과 **(2) 확장 가능하고 일반화 가능한 지식 인코딩**—을 분리(decouple)할 수 있다.

- 최적화 친화성은 FD로 후처리 가능 → 사전학습 시에는 확장성과 일반성에 집중 가능
- CLIP처럼 대규모 데이터와 대형 모델에 확장 가능한 방법이, FD를 통해 fine-tuning 성능까지 확보 가능
- MIM은 확장성 문제 존재: 더 큰 데이터에서의 이점이 불확실 (El-Nouby et al., 2021)

---

## 4. 연구 영향 및 향후 연구 시 고려할 점

### 4.1 앞으로의 연구에 미치는 영향

1. **사전학습 방법 선택의 재고**: MIM의 fine-tuning 우위가 "최적화 친화성"이라는 부차적 요소에 기인한다면, 사전학습 방법을 선택할 때 fine-tuning 성능만으로 평가하는 것은 부적절할 수 있음
2. **Contrastive Learning의 부활 가능성**: FD를 통해 contrastive 방법들의 fine-tuning 열세가 해소되므로, 이들의 우수한 linear evaluation, few-shot, zero-shot 성능과 결합하면 더 매력적인 선택지가 될 수 있음
3. **대규모 Vision-Language 모델 활용 촉진**: CLIP 등 대규모 모델의 fine-tuning 성능을 간단히 향상시킬 수 있어, foundation model의 downstream 적용이 용이해짐
4. **표현 학습 연구의 초점 전환**: 최적화 친화성에서 벗어나 **데이터 확장성(scaling law)**, **task 일반성**, **멀티모달 통합** 등 보다 근본적인 문제에 집중할 수 있는 기반 마련

### 4.2 향후 연구 시 고려할 점

1. **FD의 한계 극복**: 
   - Linear evaluation 성능 저하 문제 해결
   - MIM에 대한 추가 이득 확보 방안 탐색
   - Feature map 전체 대비 더 효율적인 증류 타겟 탐색

2. **확장성 연구**: 
   - Contrastive learning의 모델 크기 및 데이터 크기에 대한 scaling law 연구
   - FD가 초대규모 모델(수십~수백억 파라미터)에서도 유효한지 검증

3. **방법 통합**: 
   - MIM + Contrastive + Vision-Language 학습의 결합 가능성
   - FD를 사전학습 과정에 통합하여 별도 후처리 없이 최적화 친화적 표현 학습

4. **이론적 기반 강화**: 
   - "최적화 친화성"의 엄밀한 정의와 정량적 측정 지표 개발
   - 평탄한 loss landscape와 일반화 성능 간의 인과적 관계 규명

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

| 연구 | 연도 | 접근 방식 | 핵심 차이점 | FD 논문과의 관계 |
|------|------|----------|------------|--------------|
| **MoCo v3** (Chen et al.) | 2021 | Instance contrastive + ViT | ViT에 contrastive learning 적용, 안정적 학습 기법 제안 | FD의 teacher 모델 후보; FD가 MoCo의 fine-tuning 성능을 개선할 가능성 |
| **DINO** (Caron et al.) | 2021 | Self-distillation + ViT | Teacher-student 구조로 자기 지도 학습, 우수한 linear probe 성능 | FD의 직접적 대상; FD-DINO로 fine-tuning +1.0% 향상 |
| **MAE** (He et al.) | 2021 | Masked autoencoder | 75% masking ratio로 효율적 MIM, 우수한 fine-tuning | FD 적용 시 +0.2%만 향상 → MIM이 이미 최적화 친화적임을 증명 |
| **BEiT** (Bao et al.) | 2021 | Tokenizer 기반 MIM | dVAE tokenizer로 시각 토큰 예측 | FD 논문의 MIM 비교 대상 |
| **SimMIM** (Xie et al.) | 2022 | Simple MIM framework | 단순 구조로 MIM 효과 재현, Swin Transformer에도 적용 | FD 논문에서 직접 비교 |
| **CLIP** (Radford et al.) | 2021 | Vision-language contrastive | 대규모 이미지-텍스트 쌍으로 학습, zero-shot 능력 | FD-CLIP으로 fine-tuning 성능 대폭 향상 (+2.0%) |
| **BEiT v2** (Peng et al.) | 2022 | VQ-KD tokenizer + MIM | 더 강력한 visual tokenizer 사용 | FD와 유사하게 teacher 모델의 지식을 활용하지만, MIM 프레임워크 내에서 동작 |
| **EVA** (Fang et al.) | 2022 | MIM + CLIP feature reconstruction | CLIP feature를 MIM의 reconstruction target으로 사용 | FD와 유사한 동기: CLIP의 의미적 지식 + MIM의 최적화 친화성 결합 |
| **DINOv2** (Ousterhout et al.) | 2023 | Self-supervised at scale | 대규모 curated 데이터로 ViT 학습, linear/fine-tuning 모두 우수 | FD의 통찰을 반영하여 self-supervised 방법의 확장성에 집중 |
| **iBOT** (Zhou et al.) | 2022 | MIM + self-distillation | MIM과 DINO-style self-distillation 결합 | MIM과 contrastive의 결합으로, FD가 제시한 "방법 통합" 방향과 유사 |

### 핵심 비교 포인트

- **EVA** (Fang et al., 2022)는 FD와 가장 유사한 접근으로, CLIP의 feature를 MAE의 reconstruction target으로 사용한다. 이는 사전학습 단계에서 FD의 아이디어를 통합한 것으로 볼 수 있으며, FD가 후처리인 반면 EVA는 사전학습에 직접 적용한다는 차이가 있다.

- **DINOv2**는 FD 논문의 제안대로 self-supervised 방법의 확장성에 집중하여, 대규모 curated 데이터와 대형 모델로 linear evaluation과 fine-tuning 모두에서 우수한 성능을 달성했다.

- **iBOT**은 MIM과 contrastive learning의 장점을 하나의 프레임워크로 결합하여, FD 없이도 양쪽의 이점을 동시에 획득하려는 시도이다.

---

## 참고자료

1. Wei, Y., Hu, H., Xie, Z., Zhang, Z., Cao, Y., Bao, J., Chen, D., & Guo, B. (2022). "Contrastive Learning Rivals Masked Image Modeling in Fine-tuning via Feature Distillation." *arXiv preprint arXiv:2205.14141v3*.
2. He, K., Chen, X., Xie, S., Li, Y., Dollár, P., & Girshick, R. (2021). "Masked Autoencoders Are Scalable Vision Learners." *arXiv:2111.06377*.
3. Caron, M., Touvron, H., Misra, I., Jégou, H., Mairal, J., Bojanowski, P., & Joulin, A. (2021). "Emerging Properties in Self-Supervised Vision Transformers." *arXiv:2104.14294* (DINO).
4. Radford, A. et al. (2021). "Learning Transferable Visual Models From Natural Language Supervision." (CLIP)
5. Xie, Z. et al. (2022). "SimMIM: A Simple Framework for Masked Image Modeling." *CVPR*.
6. Bao, H., Dong, L., & Wei, F. (2021). "BEiT: BERT Pre-Training of Image Transformers." *arXiv:2106.08254*.
7. Li, H., Xu, Z., Taylor, G., Studer, C., & Goldstein, T. (2017). "Visualizing the Loss Landscape of Neural Nets."
8. Dosovitskiy, A. et al. (2021). "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale." *ICLR*. (ViT)
9. El-Nouby, A. et al. (2021). "Are Large-Scale Datasets Necessary for Self-Supervised Pre-training?" *arXiv:2112.10740*.
10. Fang, Y. et al. (2022). "EVA: Exploring the Limits of Masked Visual Representation Learning at Scale." *arXiv:2211.07636*.
11. Oquab, M. et al. (2023). "DINOv2: Learning Robust Visual Features without Supervision." *arXiv:2304.07193*.
12. Zhou, J. et al. (2022). "iBOT: Image BERT Pre-Training with Online Tokenizer." *ICLR*.
13. Liu, Z. et al. (2021). "Swin Transformer V2: Scaling Up Capacity and Resolution." *arXiv:2111.09883*.
14. Hinton, G., Vinyals, O., & Dean, J. (2015). "Distilling the Knowledge in a Neural Network."
15. Touvron, H. et al. (2021). "Training Data-Efficient Image Transformers & Distillation Through Attention." *ICML*. (DeiT)
