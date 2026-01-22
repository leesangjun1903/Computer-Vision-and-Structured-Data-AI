
# Pyramid Adversarial Training Improves ViT Performance
## 1. 논문 핵심 주장 및 주요 기여
**논문명**: "Pyramid Adversarial Training Improves ViT Performance"
**저자**: Charles Herrmann, Kyle Sargent, Lu Jiang, Ramin Zabih, Huiwen Chang, Ce Liu, Dilip Krishnan, Deqing Sun (Google Research)
**발표**: CVPR 2022 (arXiv:2111.15121v2)

본 논문의 핵심 주장은 전통적인 adversarial training의 정확도-견고성 간 trade-off를 **깨뜨릴 수 있다**는 것입니다. 구체적으로, Vision Transformer(ViT) 아키텍처에 특화된 Pyramid Adversarial Training(PyramidAT)을 제안하여 깨끗한 이미지(clean accuracy)와 분포 이동에 대한 견고성(out-of-distribution robustness) 모두를 동시에 향상시킵니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/830bad09-be3b-4f75-82c4-bfb8c812b126/2111.15121v2.pdf)

**주요 기여:**

1. **ViT에 대한 첫 adversarial training 성공**: ImageNet 정확도와 7개 OOD 견고성 메트릭 모두에서 동시 개선을 달성한 최초의 연구 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/830bad09-be3b-4f75-82c4-bfb8c812b126/2111.15121v2.pdf)
2. **Pyramid Adversarial Training**: 다중 스케일 구조화된 섭동 생성 기법
3. **Matched Dropout과 Stochastic Depth**: 깨끗한 샘플과 적대적 샘플에 대한 정규화 동기화
4. **SOTA 성능**: ImageNet-C (41.42 mCE), ImageNet-R (53.92%), ImageNet-Sketch (41.04%) [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/830bad09-be3b-4f75-82c4-bfb8c812b126/2111.15121v2.pdf)

***

## 2. 해결하고자 하는 문제
### 2.1 문제 정의
기존 adversarial training은 두 가지 심각한 한계가 있습니다:

**문제 1: Accuracy-Robustness Trade-off**
표준 adversarial training(Eq. 2)은 최악의 경우 성능을 최적화하지만, 깨끗한 정확도를 감소시킵니다. 이는 다음 식에서 비롯됩니다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/830bad09-be3b-4f75-82c4-bfb8c812b126/2111.15121v2.pdf)

$$\mathbb{E}_{(x,y) \sim \mathcal{D}} \Big[\max_{\delta \in \mathcal{P}} L(\theta, \tilde{x} + \delta, y) + f(\theta) \Big]$$

**문제 2: Pixel-wise Adversarial Training의 한계**
픽셀 단위 공격은 지나치게 유연하여 객체 구조를 파괴합니다. 따라서 높은 ε 값에서 clean accuracy가 급격히 저하됩니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/830bad09-be3b-4f75-82c4-bfb8c812b126/2111.15121v2.pdf)

**문제 3: ViT에 Batch Norm 부재**
CNN의 AdvProp 기법은 split batch norm을 사용하지만, ViT는 batch normalization이 없어 직접 적용 불가능합니다. [peerj](https://peerj.com/articles/cs-1197)

### 2.2 CNN과 ViT의 근본적 차이
ViT는 약한 귀납적 편향(weak inductive bias)과 높은 모델 용량을 가지므로, 강한 데이터 증강과 정규화에 크게 의존합니다. 기존 CNN 기반 robustness 방법이 ViT에 직접 적용되지 않는 이유가 여기에 있습니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/830bad09-be3b-4f75-82c4-bfb8c812b126/2111.15121v2.pdf)

***

## 3. 제안 방법: 수식 중심 설명
### 3.1 Baseline Training Loss (Equation 1)
$$\mathbb{E}_{(x,y) \sim \mathcal{D}} \Big[L(\theta, \tilde{x}, y) + f(\theta) \Big]$$

여기서:
- $\tilde{x}$: RandAug 같은 표준 데이터 증강이 적용된 이미지
- $L(\theta, x, y)$: 크로스엔트로피 손실
- $f(\theta)$: 가중치 정규화 (weight decay)

### 3.2 Mixed Adversarial Training (Equation 3)
본 논문의 기반이 되는 혼합 adversarial training:

$$\mathbb{E}_{(x,y) \sim \mathcal{D}} \Big[L(\theta, \tilde{x}, y) + \lambda \max_{\delta \in \mathcal{P}} L(\theta, \tilde{x} + \delta, y) + f(\theta) \Big]$$

이는 깨끗한 이미지 손실과 적대적 이미지 손실을 모두 최소화합니다.
- $\lambda$: 적대적 손실의 가중치
- $\delta$: 적대적 섭동
- $\mathcal{P}$: 섭동 분포

### 3.3 **Pyramid Adversarial Training (Equation 4) - 핵심 기여**
$$x^{a} = C_{\mathcal{B}_1}\Big( \tilde{x} + \sum_{s \in S} m_s \cdot C_{\mathcal{B}_{\epsilon_s}}(\delta_s)\Big)$$

여기서:
- $S = \{32, 16, 1\}$: 스케일 집합 (패치 크기)
- $m_s = \{20, 10, 1\}$: 각 스케일의 승수 (coarse scale에 더 큰 가중치)
- $\delta_s$: s×s 패치 단위로 공유되는 학습된 섭동
- $C_{\mathcal{B}_{\epsilon_s}}$: Lp norm clipping (ε_s = 6/255)
- $C_{\mathcal{B}_1}$: 이미지 범위  유지 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/830bad09-be3b-4f75-82c4-bfb8c812b126/2111.15121v2.pdf)

**핵심 직관**: 거친 스케일에서는 큰 구조화된 변화(예: 밝기), 미세 스케일에서는 작고 유연한 변화를 허용합니다. 이는 객체 구조를 보존하면서 강한 정규화 효과를 제공합니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/830bad09-be3b-4f75-82c4-bfb8c812b126/2111.15121v2.pdf)

### 3.4 Matched Dropout and Stochastic Depth (Equation 5)
$$\mathbb{E}_{(x,y) \sim \mathcal{D}} \Big[L(\mathcal{M}(\theta), \tilde{x}, y) + \lambda \max_{\delta \in \mathcal{P}} L(\theta, x^{a}, y) + f(\theta) \Big]$$

여기서 $\mathcal{M}(\theta)$는 동일한 dropout 마스크와 stochastic depth 구성을 깨끗한 샘플과 적대적 샘플 모두에 적용합니다.

**핵심 통찰**: 기존 방식은 깨끗한 분기와 적대적 분기에 다른 regularization을 적용하므로, 적대적 분기가 더 많은 네트워크를 업데이트하여 trade-off가 발생합니다. Matched dropout은 이를 해결합니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/830bad09-be3b-4f75-82c4-bfb8c812b126/2111.15121v2.pdf)

***

## 4. 모델 구조 및 구현 세부사항
### 4.1 주요 아키텍처
| 항목 | 설정 |
|------|------|
| **주 모델** | ViT-B/16 (86M 파라미터) |
| **비교 대상** | ViT-Ti/16, ResNet-50/101/200, MLP-Mixer, Discrete ViT |
| **배치 크기** | 4096 |
| **옵티마이저** | AdamW |
| **학습률 스케줄** | 코사인 감쇠 (0.001 magnitude) + 선형 워밍업 (10k steps) |

### 4.2 데이터 증강 및 정규화
**기본 설정 (RandAug 기반):**
- RandAug: (2, 15) - 2개 변환, 크기 15
- Dropout: 확률 0.1
- Stochastic Depth: 확률 0.1

### 4.3 Pyramid Attack 세부 설정
| 파라미터 | 값 | 설명 |
|----------|-----|------|
| Levels | 3 | 3-레벨 피라미드 (coarse-to-fine) |
| Scales (S) |  [arxiv](https://arxiv.org/abs/2110.07858) | 패치 크기 |
| Multiplicative (m_s) |  [irfanessa.gatech](https://www.irfanessa.gatech.edu/discrete-representations-strengthen-vision-transformer-robustness/) | 각 레벨의 강도 |
| Epsilon (ε_s) | 6/255 | 모든 레벨의 클리핑 반경 |
| Optimizer | PGD (SGD) | 5 스텝 |
| Learning Rate | 1/255 | - |

### 4.4 설정 근거
- **3-레벨 피라미드**: Ablation study에서 최적 (2-레벨, 4-레벨보다 우수) [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/830bad09-be3b-4f75-82c4-bfb8c812b126/2111.15121v2.pdf)
- **Coarse scale에 큰 가중치**: 구조화된 대역폭 낮은 섭동 선호
- **Random label PGD**: 표준 타겟 라벨 기반 PGD보다 label leaking 감소 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/830bad09-be3b-4f75-82c4-bfb8c812b126/2111.15121v2.pdf)

***

## 5. 성능 향상 및 실험 결과
### 5.1 ViT-B/16 ImageNet-1K 주요 결과
| 메트릭 | Baseline | PixelAT | PyramidAT | 개선폭 |
|--------|----------|---------|-----------|--------|
| **ImageNet (Clean)** | 79.92% | 80.42% | 81.71% | **+1.82%** ↑ |
| **ImageNet-ReaL** | 85.14% | 85.78% | 86.82% | +1.68% |
| **ImageNet-A** | 17.48% | 19.15% | 22.99% | **+5.51%** ↑ |
| **ImageNet-C (mCE)** | 52.46 | 47.68 | 44.99 | **-7.47** ↓ |
| **ImageNet-Rendition** | 38.24% | 45.39% | 47.66% | **+9.42%** ↑ |
| **ImageNet-Sketch** | 29.08% | 34.40% | 36.77% | **+7.69%** ↑ |
| **Stylized ImageNet** | 11.02% | 18.28% | 19.14% | **+8.12%** ↑ |

**주목할 점**: PyramidAT은 PixelAT과 달리 모든 메트릭에서 Baseline을 상회합니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/830bad09-be3b-4f75-82c4-bfb8c812b126/2111.15121v2.pdf)

### 5.2 SOTA 벤치마크 결과
ImageNet-C에서 PyramidAT는 기존의 모든 방법을 능가합니다:
- AdvProp (2020): 52.90 mCE
- Robust ViT (2021): 46.80 mCE  
- Discrete ViT (2022): 46.20 mCE
- **PyramidAT (2022): 41.42 mCE** ✓ 새로운 SOTA

### 5.3 ImageNet-21K 사전학습 결과
| 메트릭 | PixelAT | PyramidAT |
|--------|---------|-----------|
| ImageNet (512×512) | 84.82% | 85.35% |
| ImageNet-A | 57.39% | 62.44% |
| ImageNet-C (mCE) | 43.31 | 40.85 |
| ImageNet-Rendition | 53.35% | 56.15% |

추가 데이터로 훈련 시에도 일관된 개선 유지. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/830bad09-be3b-4f75-82c4-bfb8c812b126/2111.15121v2.pdf)

***

## 6. 모델의 일반화 성능 향상 가능성: 심층 분석
### 6.1 Attention Pattern 분석
PyramidAT 모델은 기하학적으로 다른 주의 패턴을 학습합니다:

**Baseline 모델**: 무작위 주의 분포, 객체와 배경 모두에 관심
**PixelAT 모델**: **중심 편향** - 객체에 공격적으로 집중하되, 부분 객체만 포착 (세분화 분류에 취약)
**PyramidAT 모델**: **전역 관점** - 객체 + 관련 배경 맥락을 균형있게 처리 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/830bad09-be3b-4f75-82c4-bfb8c812b126/2111.15121v2.pdf)

이는 OOD 성능 향상의 핵심 메커니즘입니다. 객체만 보는 것이 아니라 **맥락 정보**를 함께 학습하므로, 새로운 분포의 데이터에서도 강건합니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/830bad09-be3b-4f75-82c4-bfb8c812b126/2111.15121v2.pdf)

### 6.2 주파수 영역 분석
PyramidAT의 학습된 섭동을 푸리에 스펙트럼에서 분석하면: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/830bad09-be3b-4f75-82c4-bfb8c812b126/2111.15121v2.pdf)

- **Random Pixel**: 모든 주파수에 균등 분포
- **Adversarial Pixel**: 저주파 집중 (texture 기반)
- **Adversarial Pyramid**: **더 강한 저주파 편향** + 구조화된 고주파

**의미**: Pyramid 모델은 저주파 섭동에 더 강건하며, 이는 자연의 분포 이동(조명, 색상, 스타일 변화)과 일치합니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/830bad09-be3b-4f75-82c4-bfb8c812b126/2111.15121v2.pdf)

### 6.3 생성 메커니즘: Shape Bias 강화
PyramidAT 모델의 적대적 섭동을 시각화하면:

- Pixel level: 원본 이미지 구조 유지 (강아지의 다리, 등 등)
- PixelAT: 텍스처 수준 노이즈 (구조 손실)
- PyramidAT: **형태 정보 강조** (세부 구조 보존)

이는 최근 연구(Geirhos et al., 2019)의 발견 - shape bias가 강한 모델이 texture bias가 강한 모델보다 OOD 견고하다는 것과 일치합니다. PyramidAT은 자동으로 shape 표현을 학습합니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/830bad09-be3b-4f75-82c4-bfb8c812b126/2111.15121v2.pdf)

### 6.4 네트워크 용량 의존성
**중요 발견**: PyramidAT의 일반화 성능은 네트워크 용량과 데이터 증강 강도에 상호작용이 있습니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/830bad09-be3b-4f75-82c4-bfb8c812b126/2111.15121v2.pdf)

| 모델 | RandAug=0.1 | RandAug=0.4 |
|------|-------------|-------------|
| ViT-Ti/16 (Low Capacity) | PyramidAT 우수 | PyramidAT 우수 |
| ViT-B/16 (High Capacity) | PyramidAT 우수 | PyramidAT 우수 |
| ResNet-50 | AdvProp(분할 BN) 필요 | PyramidAT 가능 |

**결론**: PyramidAT은 low-capacity 모델에서도 작동하며, PixelAT과 달리 증강 강도 튜닝에 덜 민감합니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/830bad09-be3b-4f75-82c4-bfb8c812b126/2111.15121v2.pdf)

***

## 7. 논문의 한계
### 7.1 계산 비용
**주요 한계**: 훈련 시간이 **7배 증가**합니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/830bad09-be3b-4f75-82c4-bfb8c812b126/2111.15121v2.pdf)
- K-step PGD 공격 = K회 forward + K회 backward pass
- 추론 시간은 변화 없음

**해결책**: Universal PyramidAT (Poursaeed et al., 2023)은 단일 universalpattern을 모든 이미지에 공유하여 **70% 비용 감소**를 달성했습니다. [arxiv](https://arxiv.org/abs/2312.16339)

### 7.2 하이퍼파라미터 민감성
- Pyramid scales (S) 선택
- Multiplicative factors (m_s) 조정
- Epsilon (ε_s) 클리핑 값

이들 파라미터는 데이터셋과 모델에 따라 재조정 필요합니다. 본 논문은 ImageNet에 최적화된 설정을 제시합니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/830bad09-be3b-4f75-82c4-bfb8c812b126/2111.15121v2.pdf)

### 7.3 아키텍처 의존성
**Discrete ViT**에서는 PyramidAT 이득이 PixelAT 대비 덜 일관됩니다. 이는 discrete representation 자체가 이미 robustness를 제공하기 때문으로 추정됩니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/830bad09-be3b-4f75-82c4-bfb8c812b126/2111.15121v2.pdf)

**MLP-Mixer**에서는 PixelAT이 clean accuracy 저하를 보이지만, PyramidAT은 개선을 보입니다. 그러나 gain이 ViT 대비 작습니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/830bad09-be3b-4f75-82c4-bfb8c812b126/2111.15121v2.pdf)

### 7.4 Label Leaking 이슈
표준 타겟 라벨 기반 adversarial training은 ViT에서 "label leaking" 발생 - 네트워크가 공격을 예측하고 적대적 이미지에서 더 잘 수행합니다. 본 논문은 random label PGD로 이를 해결합니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/830bad09-be3b-4f75-82c4-bfb8c812b126/2111.15121v2.pdf)

***

## 8. 2020년 이후 관련 최신 연구 비교 분석
### 8.1 주요 경쟁 방법들
#### **AdvProp (Xie et al., 2020)** [peerj](https://peerj.com/articles/cs-1197)
- **방법**: Split batch normalization으로 clean/adversarial 통계 분리
- **대상**: EfficientNet, ResNet (CNN)
- **성과**: CNN에서 clean + robust accuracy 동시 개선 가능 증명
- **한계**: ViT에는 batch norm 부재로 직접 적용 불가
- **비교**: PyramidAT은 ViT 특화, AdvProp과 유사 철학 다른 구현

#### **Robust ViT (Mao et al., 2021)** [arxiv](https://arxiv.org/abs/2308.02533)
- **방법**: Adversarial pre-training + 강한 증강
- **성과**: ImageNet-C에서 46.80 mCE
- **한계**: PyramidAT (41.42 mCE)에 비해 4.38 mCE 뒤짐

#### **Discrete ViT / Enhance Visual via Discrete (Mao et al., 2022)** [arxiv](https://arxiv.org/html/2505.20872v1)
- **방법**: Vector-quantized encoder로 discrete tokens 생성
- **원리**: Discrete tokens는 작은 섭동에 invariant하며, 전역 정보 학습 유도
- **성과**: 
  - ImageNet-C (no extra data): 46.20 mCE
  - ImageNet-C (ImageNet-21K): 38.74 mCE
- **장점**: PyramidAT과 직교 (조합 가능)
- **비교**: PyramidAT이 별도 인코더 없이 pure adversarial training으로 유사 성과 달성

#### **Universal PyramidAT (Poursaeed et al., 2023)** [arxiv](https://arxiv.org/abs/2312.16339)
- **방법**: 데이터셋 전체에 공유하는 단일 pyramid pattern 학습
- **성과**: 
  - 계산 비용 70% 감소
  - 대부분의 이점 유지
  - **최초로 universal adversarial training이 clean accuracy 개선 증명**
- **관계**: PyramidAT의 직접적 후속 연구 (효율성 개선)

#### **MIMIR: Mutual Information-based Adversarial Robustness (Ma et al., 2023)** [arxiv](https://arxiv.org/abs/2312.04960)
- **방법**: 자기지도 사전학습(masked image modeling)에 상호정보 페널티 적용
- **성과**: ImageNet-1K에서 SOTA AT 성능
- **차이**: PyramidAT은 감독학습, MIMIR은 자기지도 + AT

### 8.2 OOD 일반화 방법론 비교
| 방법 | 연도 | 핵심 아이디어 | 주요 성과 | 제한사항 |
|------|------|-------------|---------|---------|
| **PyramidAT** | 2022 | 다중 스케일 구조화 섭동 | ImageNet-C: 41.42 mCE | 7배 훈련 시간 |
| **Targeted Aug** (Gao et al., 2023) | 2023 | Spurious 변수만 증강 | WILDS: +3.2%, +14.4% | 도메인 지식 필요 |
| **Universal PyramidAT** | 2023 | 공유 패턴 | 70% 비용 감소 | 약간의 성능 손실 |
| **Discrete ViT** | 2022 | 양자화 토큰 | ImageNet-C: 46.20 mCE | 추가 인코더 필요 |
| **DecAug** (Bai et al., 2022) | 2022 | 특징 분해 + 맥락 증강 | PACS/VLCS에서 우수 | 특정 도메인에 최적화 |
| **Robust ViT** | 2021 | 강한 사전학습 + 증강 | ImageNet-C: 46.80 mCE | 대규모 데이터 필요 |

### 8.3 성능 벤치마크 비교 (2020-2024)
**ImageNet-C (mCE, 낮을수록 좋음):**
```
2020: AdvProp (CNN) ..................... 52.90
2021: Robust ViT ........................ 46.80
2022: Discrete ViT ...................... 46.20
2022: PyramidAT ......................... 41.42 ← SOTA
2022: PyramidAT + 21K ................... 36.80 ← SOTA + Extra Data
```

**ImageNet-Sketch (Top-1 Accuracy):**
```
2021: Robust ViT ........................ 36.00%
2022: Discrete ViT ...................... 39.10%
2022: PyramidAT ......................... 41.04% ← SOTA
2022: PyramidAT + 21K ................... 46.03% ← SOTA + Extra Data
```

### 8.4 최신 관련 연구 방향 (2023-2025)
#### **SAFER (2025): Sharpness Aware Fine-tuning** [arxiv](http://arxiv.org/pdf/2501.01529.pdf)
- 레이어별 선택적 파인튜닝으로 clean + adversarial 정확도 개선
- 개선: ~5%, 최대 20%
- ViT 특화 접근

#### **Towards Robust OOD (Bai et al., 2024)** [arxiv](https://arxiv.org/abs/2410.21313)
- NAS(Neural Architecture Search)로 OOD 최적 아키텍처 탐색
- DecAug: 특징 분해를 통한 spurious correlation 제거
- PyramidAT과 직교적으로 결합 가능

#### **SATA: Spatial Autocorrelation Token Analysis (2024)** [arxiv](http://arxiv.org/pdf/2409.19850.pdf)
- 토큰 분석을 통한 빠른 robustness 향상
- 훈련 시간 대폭 감소 (PyramidAT 대비 이점)
- 성능은 PyramidAT 미달

***

## 9. 논문이 앞으로의 연구에 미치는 영향
### 9.1 패러다임 전환: Accuracy-Robustness Trade-off 극복
PyramidAT의 성공은 다음을 증명했습니다:
1. **Transformer 아키텍처에 특화된 adversarial training 설계가 필요** [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/830bad09-be3b-4f75-82c4-bfb8c812b126/2111.15121v2.pdf)
2. **Trade-off는 절대적이 아니라 구현 문제** (AdvProp이 CNN에서 그랬듯이)
3. **구조화된 섭동이 픽셀 단위보다 효과적**

이는 이후 Universal PyramidAT, MIMIR, SAFER 등의 연구를 촉발했습니다. [arxiv](https://arxiv.org/abs/2312.16339)

### 9.2 Matched Regularization의 중요성 발견
본 논문의 matched dropout/stochastic depth 발견은:
- **Adversarial training에서 train-test mismatch 개념 제시** [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/830bad09-be3b-4f75-82c4-bfb8c812b126/2111.15121v2.pdf)
- 이후 다양한 regularization 전략에 영향 (예: SAFER의 layer-selective fine-tuning)
- ViT 훈련 방법론 전반에 시사점 제공

### 9.3 Attention Pattern 기반 해석
PyramidAT이 "전역 주의"를 유도한다는 발견:
- Shape bias 강화 메커니즘 규명 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/830bad09-be3b-4f75-82c4-bfb8c812b126/2111.15121v2.pdf)
- 이후 ViT의 형태 인식 능력 강화 연구 활성화
- 시각 주의 메커니즘 이해 심화

### 9.4 멀티스케일 설계의 일반화
Pyramid 구조는 단순하지만 강력한 설계 원칙:
- Vision, NLP, 3D 다양한 도메인에 적용 시도
- Coarse-to-fine 다중 해상도 처리의 이점 재확인
- 하이브리드 아키텍처 설계에 영향

***

## 10. 앞으로의 연구 시 고려할 점
### 10.1 비효율성 극복
**현안**: 7배 훈련 시간 증가
**해결 방향**:
1. **효율적 공격 생성**: Universal PyramidAT (70% 비용 감소) 방향 지속 [arxiv](https://arxiv.org/abs/2312.16339)
2. **조기 정지 기법**: 수렴 후반부 adversarial training 강도 조절
3. **혼합 정밀도**: FP16 활용으로 메모리/속도 개선

**추천**: 대규모 모델(ViT-22B 같은)의 경우 Universal 접근이 필수 [openreview](https://openreview.net/pdf?id=Lhyy8H75KA)

### 10.2 데이터셋 특화 설정
현재 설정 (S=, m_s=)은 ImageNet에 최적화: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/830bad09-be3b-4f75-82c4-bfb8c812b126/2111.15121v2.pdf)

**고려할 사항**:
- **소규모 데이터셋**: Pyramid 강도 조절 필요 (overfitting 위험)
- **도메인 특화**: 의료영상, 위성영상 등에서 재최적화 필요
- **고해상도 이미지**: 패치 크기 조정 (예: 512×512)

### 10.3 아키텍처 호환성 연구
PyramidAT 성능이 아키텍처마다 다름: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/830bad09-be3b-4f75-82c4-bfb8c812b126/2111.15121v2.pdf)
- ViT: 우수 ✓
- Discrete ViT: 중간 (discrete 이점과 충돌)
- MLP-Mixer: 약함
- ResNet: 우수 (split BN과 결합)

**향후 연구**: 각 아키텍처의 특성에 맞춘 변형 개발 필요

### 10.4 조합 가능성 탐색
PyramidAT은 다른 기법과 직교적:

**시도할 가치 있는 조합**:
1. **PyramidAT + Discrete Tokens**: 더 강한 OOD 견고성
2. **PyramidAT + 자기지도 사전학습 (MAE)**: MIMIR 방향
3. **PyramidAT + Shape Bias 강화**: Stylized ImageNet에서 추가 이득
4. **PyramidAT + Domain Adaptation**: 도메인 전이 학습 개선

### 10.5 이론적 이해 심화
현재 PyramidAT의 성공은 **경험적**: 왜 작동하는지 이론이 부족합니다.

**필요한 연구**:
1. **안정성 분석**: Adversarial training의 일반화 경계
2. **주파수 분석**: Pyramid 섭동의 주파수 특성 정식화
3. **기하학적 해석**: 다중 스케일 섭동가 손실 경관에 미치는 영향
4. **통계적 보증**: OOD 성능의 이론적 보장 제공

***

## 결론
**Pyramid Adversarial Training은 Vision Transformer 시대의 robustness 문제에 대한 우아한 해결책입니다.** 

다중 스케일 구조화 섭동과 정규화 동기화라는 간단한 아이디어로:
- ✓ ImageNet 정확도 +1.82% 개선
- ✓ 7개 OOD 벤치마크 모두 개선
- ✓ 3개 새로운 SOTA 달성 (ImageNet-C, R, Sketch) [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/830bad09-be3b-4f75-82c4-bfb8c812b126/2111.15121v2.pdf)

그러나 **7배 훈련 시간, 하이퍼파라미터 민감성, 아키텍처 의존성** 등의 한계가 있으며, 이들은 Universal PyramidAT, MIMIR, SAFER 등 후속 연구로 점진적으로 해결되고 있습니다.

**향후 중요한 연구 방향**:
1. 효율성-성능 trade-off 최적화
2. 이론적 기반 강화  
3. 멀티모달/다중 도메인으로 확대
4. 더 큰 모델 (ViT-22B+)에 적용

PyramidAT은 단순한 방법이지만, Vision Transformer의 견고성 문제를 보는 새로운 관점을 제시한 의미있는 연구입니다. [arxiv](https://arxiv.org/abs/2312.16339)

***

## 참고문헌 및 인용
<span style="display:none">[^1_16][^1_17][^1_18][^1_19][^1_20][^1_21][^1_22][^1_23][^1_24][^1_25][^1_26][^1_27][^1_28][^1_29][^1_30][^1_31][^1_32][^1_33][^1_34][^1_35][^1_36][^1_37][^1_38][^1_39][^1_40][^1_41][^1_42][^1_43][^1_44][^1_45][^1_46][^1_47][^1_48][^1_49][^1_50][^1_51][^1_52][^1_53][^1_54][^1_55][^1_56][^1_57][^1_58][^1_59][^1_60][^1_61][^1_62][^1_63][^1_64][^1_65][^1_66][^1_67][^1_68][^1_69][^1_70][^1_71][^1_72][^1_73][^1_74][^1_75][^1_76][^1_77][^1_78][^1_79][^1_80][^1_81][^1_82][^1_83][^1_84][^1_85][^1_86]</span>

<div align="center">⁂</div>

[^1_1]: 2111.15121v2.pdf

[^1_2]: https://peerj.com/articles/cs-1197

[^1_3]: https://arxiv.org/abs/2110.07858

[^1_4]: https://arxiv.org/pdf/2309.02031.pdf

[^1_5]: https://www.irfanessa.gatech.edu/discrete-representations-strengthen-vision-transformer-robustness/

[^1_6]: https://arxiv.org/abs/2209.15076

[^1_7]: https://arxiv.org/abs/2312.16339

[^1_8]: https://arxiv.org/abs/2308.02533

[^1_9]: https://arxiv.org/html/2505.20872v1

[^1_10]: https://arxiv.org/abs/2312.04960

[^1_11]: http://arxiv.org/pdf/2501.01529.pdf

[^1_12]: https://arxiv.org/abs/2410.21313

[^1_13]: http://arxiv.org/abs/2410.21313

[^1_14]: http://arxiv.org/pdf/2409.19850.pdf

[^1_15]: https://openreview.net/pdf?id=Lhyy8H75KA

[^1_16]: https://www.mdpi.com/2073-445X/12/10/1926

[^1_17]: https://ieeexplore.ieee.org/document/10178853/

[^1_18]: https://arxiv.org/abs/2302.05442

[^1_19]: https://ieeexplore.ieee.org/document/10278410/

[^1_20]: https://ojs.aaai.org/index.php/AAAI/article/view/20103

[^1_21]: https://arxiv.org/abs/2203.06649

[^1_22]: https://arxiv.org/abs/2210.06983

[^1_23]: https://arxiv.org/html/2407.15385v1

[^1_24]: https://arxiv.org/pdf/2106.01548.pdf

[^1_25]: http://arxiv.org/pdf/2402.11301.pdf

[^1_26]: https://arxiv.org/pdf/2409.03901v2.pdf

[^1_27]: https://pmc.ncbi.nlm.nih.gov/articles/PMC10280230/

[^1_28]: https://arxiv.org/pdf/2112.09747.pdf

[^1_29]: https://www.sciencedirect.com/science/article/abs/pii/S016786552400223X

[^1_30]: https://openreview.net/pdf?id=Bcg0It4i1g

[^1_31]: https://daiqi1989.github.io/assets/pdf/TMM_final_Robustness.pdf

[^1_32]: http://proceedings.mlr.press/v139/yi21a/yi21a.pdf

[^1_33]: https://www.thejournal.club/c/paper/274765/

[^1_34]: https://openaccess.thecvf.com/content/ICCV2021/papers/Poursaeed_Robustness_and_Generalization_via_Generative_Adversarial_Training_ICCV_2021_paper.pdf

[^1_35]: https://openaccess.thecvf.com/content/ICCV2021/papers/Hendrycks_The_Many_Faces_of_Robustness_A_Critical_Analysis_of_Out-of-Distribution_ICCV_2021_paper.pdf

[^1_36]: https://viso.ai/deep-learning/vision-transformer-vit/

[^1_37]: https://aclanthology.org/2023.findings-acl.496.pdf

[^1_38]: https://paperswithcode.com/paper/towards-robust-out-of-distribution-1

[^1_39]: https://openaccess.thecvf.com/content/ICCV2023/papers/Zhu_Improving_Generalization_of_Adversarial_Training_via_Robust_Critical_Fine-Tuning_ICCV_2023_paper.pdf

[^1_40]: https://www.themoonlight.io/en/review/towards-robust-out-of-distribution-generalization-data-augmentation-and-neural-architecture-search-approaches

[^1_41]: https://arxiv.org/html/2506.19591v1

[^1_42]: https://arxiv.org/pdf/2309.12593.pdf

[^1_43]: https://arxiv.org/abs/2505.10223

[^1_44]: https://arxiv.org/pdf/2503.02891.pdf

[^1_45]: https://arxiv.org/abs/2210.00960

[^1_46]: https://arxiv.org/abs/2006.16241

[^1_47]: https://arxiv.org/abs/1906.06032

[^1_48]: https://www.arxiv.org/pdf/2503.14751.pdf

[^1_49]: https://arxiv.org/html/2506.18516v1

[^1_50]: https://arxiv.org/html/2505.12317v1

[^1_51]: https://link.springer.com/10.1007/s40273-022-01198-8

[^1_52]: https://www.tandfonline.com/doi/full/10.1080/00207160.2024.2363467

[^1_53]: https://www.tandfonline.com/doi/full/10.1080/14697688.2021.2013621

[^1_54]: https://journals.sagepub.com/doi/full/10.1080/25726838.2022.2084233

[^1_55]: https://pubs.geoscienceworld.org/bssa/article/113/2/524/619845/Earthquake-Phase-Association-with-Graph-Neural

[^1_56]: https://link.springer.com/10.1007/s00431-022-04673-8

[^1_57]: https://arxiv.org/abs/2211.09981

[^1_58]: https://ieeexplore.ieee.org/document/9992966/

[^1_59]: https://pubs.aip.org/cha/article/33/2/023128/2876208/Transition-to-hyperchaos-and-rare-large-intensity

[^1_60]: https://arxiv.org/abs/2111.10493

[^1_61]: https://arxiv.org/pdf/2105.10497v1)%3C%22.pdf

[^1_62]: http://arxiv.org/pdf/2204.12143.pdf

[^1_63]: https://arxiv.org/pdf/2302.10468.pdf

[^1_64]: https://openaccess.thecvf.com/content/CVPR2022/papers/Herrmann_Pyramid_Adversarial_Training_Improves_ViT_Performance_CVPR_2022_paper.pdf

[^1_65]: https://omidpoursaeed.github.io/publication/uni_at/

[^1_66]: https://www.nature.com/articles/s41598-024-72254-w

[^1_67]: https://openreview.net/pdf/248e8be4fabe9aa0349e4e24290dd07abecaa424.pdf

[^1_68]: https://arxiv.org/html/2308.09372v2

[^1_69]: https://papers.neurips.cc/paper_files/paper/2022/file/31928aa24124da335bec23f5e1f91a46-Paper-Conference.pdf

[^1_70]: https://openreview.net/forum?id=HGYh7BFA6J

[^1_71]: https://dl.acm.org/doi/full/10.1145/3729167

[^1_72]: https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136730307.pdf

[^1_73]: https://www.sciencedirect.com/science/article/abs/pii/S1077314223001807

[^1_74]: https://kimjy99.github.io/논문리뷰/pyramidat/

[^1_75]: https://liner.com/review/vision-transformers-are-robust-learners

[^1_76]: https://arxiv.org/html/2507.00754v1

[^1_77]: https://arxiv.org/pdf/2210.08457.pdf

[^1_78]: https://openaccess.thecvf.com/content_ICCV_2019/papers/Liu_Universal_Adversarial_Perturbation_via_Prior_Driven_Uncertainty_Approximation_ICCV_2019_paper.pdf

[^1_79]: https://arxiv.org/html/2312.16339v1

[^1_80]: https://arxiv.org/html/2510.04794v1

[^1_81]: https://arxiv.org/abs/2105.07581

[^1_82]: https://arxiv.org/pdf/2410.15042.pdf

[^1_83]: https://arxiv.org/html/2308.09372v3

[^1_84]: https://arxiv.org/pdf/2208.00906.pdf

[^1_85]: https://www.semanticscholar.org/paper/Universal-Adversarial-Training-with-Class-Wise-Benz-Zhang/f8bbfb98406e15747a84d7cf2872949a10274085

[^1_86]: https://arxiv.org/html/2406.06136v1
