# Proper Reuse of Image Classification Features Improves Object Detection

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장
이미지 분류(ImageNet, JFT-300M) 태스크에서 사전 학습된 백본(backbone) 네트워크의 가중치를 객체 탐지(object detection) 학습 시 **완전히 동결(freeze)하는 극단적 지식 보존 전략**이, 기존의 미세 조정(fine-tuning)이나 처음부터 학습(training from scratch)보다 **일관되게 우수한 성능**을 달성할 수 있음을 보인다. 단, 이 이점은 **나머지 탐지기 구성요소(detector components)가 충분한 용량(capacity)을 가질 때** 실현된다.

### 주요 기여
1. **모순 해결**: Sun et al. [47]의 "대규모 사전 학습이 유용하다"는 주장과 He et al. [14]의 "긴 학습에서 사전 학습 이점이 사라진다"는 상반된 관찰을 통합적으로 설명
2. **백본 동결의 체계적 검증**: 다양한 백본(ResNet-50/101, EfficientNet-B7), 탐지기 구조(FPN, NAS-FPN, Cascade), 데이터셋(MSCOCO, LVIS)에 걸친 광범위한 실험
3. **자원 절약**: 학습 시 메모리와 FLOPs를 대폭 절감하면서도 성능 향상 달성
4. **Long-tail 클래스에서의 뚜렷한 개선**: 희귀 클래스(rare classes) 탐지 성능에서 가장 큰 이점
5. **Residual Adapter를 통한 추가 개선 가능성** 제시

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

객체 탐지에서의 전이 학습(transfer learning)에는 두 가지 상반된 관찰이 존재했다:

- **관찰 A** (Sun et al., 2017): 대규모 분류 데이터셋(JFT-300M)에서 사전 학습하면 탐지 성능이 향상된다.
- **관찰 B** (He et al., 2019): 충분히 긴 학습 스케줄에서는 사전 학습과 처음부터 학습의 성능 차이가 사라진다.

이 논문은 이 모순의 원인을 규명하고, 사전 학습된 표현의 **올바른 재사용 방법**을 제시하고자 한다. 핵심 통찰은 다음과 같다:

> 긴 학습 스케줄 동안 미세 조정(fine-tuning)이 백본 가중치를 사전 학습된 초기화로부터 점점 멀어지게 하여, 분류 태스크에서 학습된 유용한 표현이 훼손된다.

### 2.2 제안하는 방법

#### 2.2.1 백본 동결 (Backbone Freezing)

제안 방법은 극히 단순하다. 분류 태스크에서 사전 학습된 백본 네트워크 $f_\theta$의 파라미터 $\theta$를 객체 탐지 학습 동안 **완전히 고정**하고, 나머지 탐지기 구성요소만 학습한다.

전체 모델의 파라미터를 백본 파라미터 $\theta_{\text{backbone}}$과 탐지기 파라미터 $\theta_{\text{det}}$로 분리하면:

$$\theta = \{\theta_{\text{backbone}},\; \theta_{\text{det}}\}$$

기존 미세 조정 방식의 학습 업데이트:

$$\theta_{\text{backbone}}^{(t+1)} = \theta_{\text{backbone}}^{(t)} - \eta \nabla_{\theta_{\text{backbone}}} \mathcal{L}_{\text{det}}$$

$$\theta_{\text{det}}^{(t+1)} = \theta_{\text{det}}^{(t)} - \eta \nabla_{\theta_{\text{det}}} \mathcal{L}_{\text{det}}$$

제안 방법 (백본 동결):

$$\theta_{\text{backbone}}^{(t+1)} = \theta_{\text{backbone}}^{(t)} = \theta_{\text{backbone}}^{\text{pretrained}} \quad (\text{frozen, no gradient update})$$

$$\theta_{\text{det}}^{(t+1)} = \theta_{\text{det}}^{(t)} - \eta \nabla_{\theta_{\text{det}}} \mathcal{L}_{\text{det}}$$

여기서 $\mathcal{L}_{\text{det}}$는 객체 탐지 손실 함수, $\eta$는 학습률이다.

#### 2.2.2 탐지기 용량(Capacity)의 중요성

백본 동결의 이점은 나머지 **학습 가능한 탐지기 구성요소의 용량**에 의해 제한된다. 전체 학습 가능 파라미터 비율을 다음과 같이 정의할 수 있다:

$$r_{\text{trainable}} = \frac{|\theta_{\text{det}}|}{|\theta_{\text{backbone}}| + |\theta_{\text{det}}|}$$

논문의 실험 결과(Table 3)에 따르면:

| 탐지 모델 | 전체 파라미터(M) | 학습 가능 비율 $r_{\text{trainable}}$ | $\Delta$ mAP |
|---|---|---|---|
| FPN | 83.5 | 26.1% | −6.5 |
| NAS-FPN | 102.6 | 39.9% | +0.1 |
| Cascade + NAS-FPN | 132.9 | 53.6% | +0.7 |

학습 가능 파라미터 비율이 약 **40% 이상**일 때 백본 동결이 이점을 발휘한다.

#### 2.2.3 Residual Adapter를 통한 추가 적응

백본 동결이 최적 전략임을 주장하는 것이 아니라, 더 나은 지식 보존 방법이 존재할 수 있음을 보이기 위해 **Residual Adapter** [40, 41]를 도입한다. Residual adapter는 동결된 백본 레이어 $l$의 출력 $h_l$에 경량의 학습 가능한 변환 $A_l$을 추가한다:

$$h_l' = h_l + A_l(h_l)$$

여기서 $A_l$은 소수의 추가 파라미터로 구성된 어댑터 모듈이다. 이를 통해 사전 학습된 표현을 보존하면서도 탐지 태스크에 적응할 수 있는 유연성을 부여한다.

#### 2.2.4 데이터 증강과의 상보성

Large Scale Jittering (LSJ)과 Copy-and-Paste 증강을 모든 실험에 사용하며, 백본이 동결된 상태에서 이러한 증강 기법은 **탐지기 구성요소만을 개선**하는 역할을 한다. 즉, 백본 동결과 데이터 증강은 **상호 보완적(complementary)**이다.

### 2.3 모델 구조

논문은 특정 새로운 아키텍처를 제안하는 것이 아니라, 기존 탐지 프레임워크에 백본 동결 전략을 적용한다. 사용된 모델 구조(Figure 2)는:

```
Image → [Backbone (Frozen)] → [Decoder/FPN] → [RPN] → [Detection Head] → Post-processing
```

**주요 구성요소:**
- **백본**: ResNet-50/101, EfficientNet-B7 (ImageNet 또는 JFT-300M에서 사전 학습 후 동결)
- **디코더/Feature Pyramid**: FPN [28], NAS-FPN
- **탐지 헤드**: Fast-RCNN, Cascade R-CNN [3], Mask-RCNN
- **RPN**: Region Proposal Network [42]

탐지기 구성요소의 용량을 점진적으로 증가시키며 (FPN → NAS-FPN → NAS-FPN + Cascade) 백본 동결의 효과를 검증한다.

### 2.4 성능 향상

#### MSCOCO 결과 (Table 1, Table 4)

| 모델 | 사전학습 | 전략 | mAP | AP@50 |
|---|---|---|---|---|
| ResNet-101 + NAS-FPN + Cascade | JFT-300M | Fine-tune | 51.1 | 69.0 |
| ResNet-101 + NAS-FPN + Cascade | JFT-300M | **Freeze** | **52.8 (+1.7)** | **71.1 (+2.1)** |
| ResNet-101 + NAS-FPN + Cascade | JFT-300M | **Freeze + Res. Adapter** | **53.6 (+2.5)** | **72.0 (+3.0)** |
| EfficientNet-B7 + NAS-FPN (1280) | ImageNet | Fine-tune + Copy-Paste | 55.9 | 47.2 |
| EfficientNet-B7 + NAS-FPN (1280) | ImageNet | **Freeze + Copy-Paste** | **57.0 (+1.1)** | **48.7 (+1.5)** |

#### LVIS 결과 (Table 4, Table 5)

LVIS 데이터셋에서 더 큰 성능 향상을 보임 (long-tail 분포):

| 설정 | Box mAP | Mask mAP |
|---|---|---|
| EfficientNet-B7 + Copy-Paste (baseline) | 41.6 | 38.1 |
| + Freeze backbone | **43.1 (+1.5)** | **39.9 (+1.8)** |

특히 **희귀 클래스(rare)**에서 가장 큰 개선:
- Box mAP $_r$: +1.7 (31.5 → 33.2)
- Mask mAP $_r$: +1.5 (32.1 → 33.6)

#### 자원 절약 (Figure 1 우측)

백본 동결 시 학습 단계당 FLOPs가 대폭 감소:
- ResNet-101 + NAS-FPN + Cascade: 약 **5000B → 3500B FLOPs** (약 30% 절감)하면서 성능은 향상

### 2.5 한계

1. **탐지기 용량 의존성**: FPN과 같이 용량이 작은 탐지기에서는 백본 동결이 오히려 성능을 저하시킴 (최대 −6.5 mAP)
2. **최적 전략이 아닐 수 있음**: 저자들도 백본 동결이 최적이라고 주장하지 않으며, residual adapter 등 더 나은 방법이 존재할 수 있음을 인정
3. **백본 아키텍처 제한**: CNN 기반 백본(ResNet, EfficientNet)에서만 검증됨. Vision Transformer 기반 백본에서의 검증이 부족
4. **사전 학습 데이터 접근성**: JFT-300M은 공개되지 않은 데이터셋으로, 재현성에 제약이 있음
5. **소형 객체에 대한 제한적 개선**: Table 6에서 소형 객체(mAP $_s$ )에 대한 개선은 상대적으로 작음 (+0.2~+0.7)

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 일반화 관점에서의 핵심 메커니즘

이 논문의 가장 중요한 발견은 사전 학습된 표현이 **탐지 태스크 자체에도 유용한 일반적 특징(general features)**을 포함하고 있으며, 이러한 특징이 미세 조정 과정에서 **취약하게 소실된다(brittle to fine-tuning)**는 것이다.

이를 수식으로 표현하면, 사전 학습된 백본의 특징 공간 $\mathcal{F}\_{\text{pretrained}}$와 미세 조정 후의 특징 공간 $\mathcal{F}_{\text{finetuned}}$의 관계:

$$\mathcal{F}_{\text{finetuned}} = \mathcal{F}_{\text{pretrained}} - \Delta\mathcal{F}_{\text{beneficial}} + \Delta\mathcal{F}_{\text{task-specific}}$$

여기서 $\Delta\mathcal{F}\_{\text{beneficial}}$은 미세 조정 중 손실되는 일반적 특징이고, $\Delta\mathcal{F}\_{\text{task-specific}}$은 탐지 태스크에 특화된 특징이다. 백본 동결은 $\Delta\mathcal{F}_{\text{beneficial}} = 0$을 보장하여 일반적 특징을 보존한다.

### 3.2 Catastrophic Forgetting과의 유사성

저자들은 이 현상이 **치명적 망각(catastrophic forgetting)** [11]과 유사하다고 지적한다. 그러나 전통적 의미의 치명적 망각(이전 태스크의 성능 저하)과는 다르게, 여기서는 미세 조정이 **상위 태스크뿐 아니라 하위 태스크(detection) 자체에도 유용한 지식을 손상**시킨다는 점에서 차이가 있다.

### 3.3 어노테이션 수에 따른 일반화 성능 분석 (Figure 4, 5)

클래스별 mAP를 학습 어노테이션 수의 함수로 분석한 결과:

- **어노테이션이 많은 클래스**: 동결, 미세 조정, 처음부터 학습 간 성능 차이가 미미
- **어노테이션이 적은 클래스**: 백본 동결이 점점 더 큰 이점을 보임

이는 사전 학습된 표현이 **소수의 예시만으로는 학습하기 어려운 범용적 시각 특징**을 포함하고 있음을 시사한다. 수학적으로, 클래스 $c$에 대한 어노테이션 수 $n_c$와 백본 동결로 인한 성능 변화 $\Delta\text{mAP}_c$ 사이의 관계:

$$\Delta\text{mAP}_c = \text{mAP}_c^{\text{freeze}} - \text{mAP}_c^{\text{finetune}} \propto \frac{1}{n_c}$$

즉, 어노테이션이 적을수록 동결의 이점이 커진다.

### 3.4 데이터셋 규모와 일반화

사전 학습 데이터셋의 규모가 커질수록 일반화 이점이 증가함을 확인:

$$\text{mAP}(\text{JFT-300M, freeze}) > \text{mAP}(\text{ImageNet, freeze})$$

ImageNet과 JFT-300M 동결 모델 간 일관된 +0.9 mAP 차이가 다양한 아키텍처에서 관찰됨 (Table 1).

### 3.5 탐지기 용량과 일반화의 상호작용

일반화 성능 향상은 탐지기의 **탐지 특화 용량(detection-specific capacity)**이 충분할 때만 실현된다. 이는 동결된 백본이 제공하는 범용 특징 위에 탐지 특화 변환을 학습할 수 있는 충분한 파라미터가 필요하기 때문이다:

$$\text{Performance} = g(\underbrace{f_{\theta_{\text{backbone}}}(x)}_{\text{frozen general features}},\; \underbrace{\theta_{\text{det}}}_{\text{learnable detection capacity}})$$

$|\theta_{\text{det}}|$가 충분히 클 때만 $g$가 범용 특징을 탐지 태스크에 효과적으로 적응시킬 수 있다.

---

## 4. 향후 연구에 미치는 영향 및 고려사항

### 4.1 연구에 미치는 영향

1. **전이 학습 패러다임의 재고**: "사전 학습 → 미세 조정"이라는 표준적 파이프라인에 대한 근본적 재검토를 촉발. 특히 **지식 보존(knowledge preservation)**의 중요성을 강조

2. **탐지기 아키텍처 설계 방향 전환**: 백본 동결을 전제로 한 **탐지기 구성요소의 용량 설계**가 새로운 연구 방향으로 부상. NAS(Neural Architecture Search) 시 백본 동결을 가정한 최적 탐지기 구조 탐색 가능

3. **자원 효율적 학습**: 컴퓨팅 자원이 제한된 연구자들에게 동등 이상의 성능을 달성할 수 있는 실용적 방법을 제공

4. **Long-tail 객체 인식**: 희귀 클래스 탐지에서의 뚜렷한 개선은 실세계 응용(자율주행, 의료영상 등)에서의 활용 가능성을 확장

5. **Parameter-Efficient Transfer Learning (PETL)과의 연결**: Residual adapter 실험은 NLP 분야의 adapter-based 방법론이 비전 태스크에도 적용 가능함을 시사

### 4.2 향후 연구 시 고려할 점

1. **Vision Transformer 기반 백본에서의 검증**: 이 논문은 CNN 기반 백본만 다룸. ViT, Swin Transformer 등에서의 동결 효과 검증이 필요

2. **부분 동결(Partial Freezing) 전략 탐색**: 전체 백본이 아닌 특정 레이어만 동결하는 세밀한 전략의 효과 연구

3. **자기지도 학습(Self-supervised Learning) 사전 학습과의 결합**: DINO, MAE 등 자기지도 방식으로 학습된 백본의 동결 효과 검증

4. **동결 + Adapter의 최적 설계**: Residual adapter 외에도 LoRA, Prompt Tuning 등 다양한 PETL 기법의 적용 가능성

5. **학습 에폭 수 최적화**: 동결된 백본에서 탐지기 수렴에 필요한 최소 에폭 수에 대한 연구

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

### 5.1 주요 비교 연구

| 연구 | 연도 | 핵심 접근 | 본 논문과의 관계 |
|---|---|---|---|
| **DINO** (Caron et al.) | 2021 | 자기지도 ViT 사전 학습 | 더 강력한 사전 학습 표현 → 동결 시 더 큰 이점 기대 |
| **MAE** (He et al.) | 2022 | Masked Autoencoder 사전 학습 | 미세 조정 중심 설계이나, 동결 전략과의 비교 필요 |
| **Swin Transformer** (Liu et al.) | 2021 | 계층적 ViT 기반 탐지 | Table 4에서 비교. ImageNet-22K 사전 학습으로 57.1 mAP 달성 |
| **ViTDet** (Li et al.) | 2022 | Plain ViT backbone for detection | 사전 학습된 ViT를 탐지에 적용; 동결 전략 미탐구 |
| **EVA** (Fang et al.) | 2023 | 대규모 사전 학습 ViT | 초대규모 사전 학습 표현의 동결 재사용 가능성 시사 |
| **Co-DETR** (Zong et al.) | 2023 | Collaborative hybrid detection | 탐지 특화 구조의 용량 확장; 본 논문의 "용량이 핵심" 관찰과 일치 |
| **Frozen CLIP** (various) | 2022-2023 | CLIP 특징 동결 후 탐지 적용 | 본 논문의 동결 전략을 VLM으로 확장한 자연스러운 후속 |

### 5.2 핵심 비교 분석

#### (a) Foundation Model 시대의 동결 전략

본 논문의 핵심 통찰은 2022-2023년 Foundation Model 시대에 더욱 중요해졌다. CLIP, DINO v2 등의 대규모 사전 학습 모델에서 학습된 표현을 **동결하고 재사용**하는 접근이 점점 더 보편화되고 있다. 이는 본 논문이 CNN 시대에 제시한 통찰의 자연스러운 확장이다.

#### (b) Parameter-Efficient Fine-Tuning (PEFT)의 부상

본 논문의 residual adapter 실험(Table 7, 17, 18)은 이후 LoRA (Hu et al., 2022), AdaptFormer (Chen et al., 2022) 등의 PEFT 기법이 비전 분야에서도 활발히 연구되는 흐름과 직접적으로 연결된다. 본 논문은 이러한 방향의 **초기 실증적 근거**를 제공했다고 볼 수 있다.

#### (c) 탐지기 용량 vs. 백본 크기의 트레이드오프

최신 탐지 모델들(DINO detector, Co-DETR 등)은 점점 더 큰 탐지 헤드와 디코더를 사용하는 추세이다. 이는 본 논문의 핵심 관찰—"탐지기 구성요소의 충분한 용량이 사전 학습 표현의 이점을 실현하는 데 필수적"—과 정확히 일치하는 방향이다.

### 5.3 한계와 미래 방향에 대한 최신 관점

본 논문의 관찰이 **Transformer 기반 아키텍처**에서 어떻게 변화하는지는 아직 충분히 탐구되지 않았다. ViT의 self-attention 메커니즘이 CNN의 convolution 연산과 근본적으로 다르기 때문에, 동결 전략의 효과가 달라질 수 있다. 특히:

- ViT는 전역적(global) 수용 영역을 가지므로, 탐지에 필요한 공간적 정보가 이미 사전 학습 과정에서 더 잘 인코딩될 수 있다
- 반면, ViT의 positional encoding이 해상도 변화에 민감하여 동결 시 문제가 될 수 있다

---

## 참고자료

1. Vasconcelos, C., Birodkar, V., & Dumoulin, V. (2022). "Proper Reuse of Image Classification Features Improves Object Detection." *CVPR 2022*. (본 논문)
2. Sun, C., Shrivastava, A., Singh, S., & Gupta, A. (2017). "Revisiting unreasonable effectiveness of data." *ICCV*. [참조 47]
3. He, K., Girshick, R., & Dollár, P. (2019). "Rethinking imagenet pre-training." *ICCV*. [참조 14]
4. Ghiasi, G., et al. (2020). "Simple copy-paste is a strong data augmentation method for instance segmentation." *arXiv:2012.07177*. [참조 12]
5. Du, X., et al. (2021). "Simple training strategies and model scaling for object detection." *arXiv:2107.00057*. [참조 7, 8, 9]
6. Rebuffi, S.-A., Bilen, H., & Vedaldi, A. (2017). "Learning multiple visual domains with residual adapters." *arXiv:1705.08045*. [참조 40]
7. Liu, Z., et al. (2021). "Swin Transformer: Hierarchical vision transformer using shifted windows." *arXiv:2103.14030*. [참조 31]
8. Caron, M., et al. (2021). "Emerging Properties in Self-Supervised Vision Transformers (DINO)." *ICCV 2021*.
9. He, K., et al. (2022). "Masked Autoencoders Are Scalable Vision Learners." *CVPR 2022*.
10. Hu, E. J., et al. (2022). "LoRA: Low-Rank Adaptation of Large Language Models." *ICLR 2022*.
11. Li, Y., et al. (2022). "Exploring Plain Vision Transformer Backbones for Object Detection (ViTDet)." *ECCV 2022*.
12. Fang, Y., et al. (2023). "EVA: Exploring the Limits of Masked Visual Representation Learning at Scale." *CVPR 2023*.
13. Zong, Z., et al. (2023). "DETRs with Collaborative Hybrid Assignments Training (Co-DETR)." *ICCV 2023*.
14. French, R. M. (1999). "Catastrophic forgetting in connectionist networks." *Trends in Cognitive Sciences*. [참조 11]
15. Lin, T.-Y., et al. (2017). "Feature Pyramid Networks for Object Detection." *CVPR*. [참조 28]
16. Ren, S., et al. (2015). "Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks." *NeurIPS*. [참조 42]
17. Cai, Z. & Vasconcelos, N. (2018). "Cascade R-CNN: Delving into High Quality Object Detection." *CVPR*. [참조 3]
