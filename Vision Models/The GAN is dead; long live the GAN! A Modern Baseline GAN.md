# The GAN is dead; long live the GAN! A Modern Baseline GAN

## 핵심 주장 및 주요 기여

"The GAN is dead; long live the GAN! A Modern Baseline GAN" 논문은 GAN이 훈련하기 어렵다는 널리 퍼진 주장에 대해 반박하며, 수학적으로 안정적인 손실 함수와 현대적 신경망 구조를 결합한 **R3GAN**을 제안합니다. 이 논문의 핵심 기여는 다음과 같습니다.[1]

**첫째, 정규화된 상대론적 페어링 GAN 손실(RpGAN + R1 + R2)**을 도출하여 기존의 ad-hoc 트릭 없이도 모드 드로핑(mode dropping)과 비수렴(non-convergence) 문제를 해결합니다. 저자들은 이 손실이 국소적 수렴 보장(local convergence guarantees)을 갖는다는 것을 수학적으로 증명합니다.[1]

**둘째, StyleGAN2 구조를 단순화하고 현대화**하여 모든 트릭을 제거하고 ResNet, ConvNeXt 같은 최신 아키텍처를 도입합니다. 결과적으로 매개변수 수가 유사하면서도 더 나은 성능을 달성하는 미니멀한 기준 모델을 제시합니다.[1]

## 해결하고자 하는 문제

GAN 분야의 두 가지 근본적인 문제를 다룹니다.

**1) 훈련 불안정성과 수렴 실패**: 전통적 GAN 손실 함수 \(L(\theta, \psi) = \mathbb{E}_{z \sim p_z} [f(D_\psi(G_\theta(z)))] + \mathbb{E}_{x \sim p_D} [f(-D_\psi(x))]\)는 미니배치에서 실제 데이터와 가짜 데이터를 단순히 분리하려고 하기 때문에, 생성기는 모든 가짜 샘플을 하나의 결정 경계 너머로 밀어낸다는 퇴화된 해를 선호합니다. 이는 모드 드로핑으로 이어집니다.[1]

**2) 아키텍처의 낙후성**: StyleGAN2는 여전히 2015년 DCGAN의 기본 구조를 기반으로 하며, 미니배치 표준편차 트릭, 동일한 학습률, 노이즈 주입, 경로 길이 정규화 등 많은 ad-hoc 트릭에 의존합니다. 반면 확산 모델(diffusion models)은 Vision Transformer, U-Net 등 현대 아키텍처를 채용하고 있습니다.[1]

## 제안하는 방법 및 수식

### 상대론적 페어링 GAN (RpGAN)

상대론적 손실 함수는 실제 데이터와 가짜 데이터를 결합하여 평가합니다:

$$L(\theta, \psi) = \mathbb{E}\_{z \sim p_z, x \sim p_D} [f(D_\psi(G_\theta(z)) - D_\psi(x))]$$[1]

이 접근 방식은 각 실제 샘플 근처에 결정 경계를 유지하기 때문에, Sun et al.이 보인 것처럼 RpGAN의 손실 함수 랜드스케이프에는 모드 드로핑 해에 해당하는 국소 최소값이 존재하지 않습니다.[1]

### 그래디언트 페널티 (R1, R2)

안정성을 위해 제로-센터링 그래디언트 페널티를 적용합니다:

$$R_1(\psi) = \frac{\gamma}{2}\mathbb{E}_{x \sim p_D}\left[\|\nabla_x D_\psi\|^2\right]$$

$$R_2(\theta, \psi) = \frac{\gamma}{2}\mathbb{E}\_{x \sim p_\theta}\left[\|\nabla_x D_\psi\|^2\right]$$[1]

**핵심 통찰**: R1과 R2는 판별기가 실제 데이터와 가짜 데이터 모두에서 영점 근처의 기울기를 가지도록 강제하여, 생성기가 최적 상태에서 벗어나지 않도록 합니다.[1]

### Proposition II (정규화된 RpGAN의 국소 수렴)

저자들은 다음을 증명합니다:

**정칙화된 RpGAN (R1 또는 R2 포함)는 국소 수렴한다.** 증명은 정규화된 그래디언트 벡터 필드의 야코비안(Jacobian) 고유값을 분석하여 모든 고유값이 음의 실수부를 가짐을 보입니다. 이는 충분히 작은 학습률에 대해 $\theta^\*$ , $\psi^*$ 근처에서 수렴성을 보장합니다.[1]

## 모델 구조

### 설정 진행 로드맵 (Config A-E)

저자들은 StyleGAN2 (Config A)에서 시작하여 단계적으로 개선합니다:

| 구성 | FID↓ | 주요 변경 사항 |
|------|------|---|
| Config A: StyleGAN2 | 7.516 | 베이스라인 |
| Config B: 트릭 제거 | 12.46 | z 정규화, 미니배치 표준편차, 동등한 학습률 등 제거 |
| Config C: 우수한 손실 | 11.65 | RpGAN + R2 추가 |
| Config D: ConvNeXt-ify (1단계) | 7.507 | ResNet 기본 구조 + GroupedConv |
| Config E: ConvNeXt-ify (2단계) | 7.045 | 반전 병목 아키텍처[1] |

### 네트워크 아키텍처 설계 원칙

**포함된 요소:**
- Fix-up 초기화: \(\text{scale} = L^{-0.25}\) (L은 잔차 블록 수)[1]
- 1-3-1 병목 ResNet 아키텍처
- 선형 이중 샘플링 보간
- Leaky ReLU 활성화
- 정규화 층 제거

**각 해상도 스테이지 구조:**
```
Transition Layer: 이중 선형 리샘플링 + 1×1 Conv
Residual Block: Conv1×1 → LeakyReLU → Conv3×3 → LeakyReLU → Conv1×1
```

### Config E의 고급 설계

**집단화 컨볼루션 (Grouped Convolution)**: 그룹 크기를 16으로 설정하여 깊이별 컨볼루션의 비효율성을 피하고 병목 압축 비율을 2로 감소시킵니다.[1]

**반전 병목 (Inverted Bottleneck)**: 줄기(stem) 너비와 병목 너비를 반전시켜 집단화 컨볼루션 용량을 2배로 증가시킵니다.[1]

## 성능 향상

### StackedMNIST 모드 커버리지 (1000개 모드)

| 모델 | 모드 수↑ | D_KL ↓ |
|------|---------|--------|
| StyleGAN2 | 940 | 0.42 |
| RpGAN + R1 | 실패 | 실패 |
| RpGAN + R1 + R2 | **1000** | **0.029** |[1] |

이 실험은 R1 단독으로는 불충분하며, R2를 추가해야 글로벌 수렴이 달성됨을 보여줍니다.[1]

### 주요 벤치마크 성능

**FFHQ-256** (고해상도 얼굴):[1]
- StyleGAN2: FID 3.78
- **R3GAN (Config E): FID 2.75** ⬇️ 27% 개선

**CIFAR-10** (자연 이미지):[1]
- StyleGAN2 + ADA: FID 2.42
- **R3GAN: FID 1.96** ⬇️ 19% 개선

**ImageNet-64**:[1]
- BigGAN-deep: FID 2.09
- **R3GAN: FID 2.09** (더 적은 매개변수)

추론은 **1번의 함수 평가(NFE)**만 필요하므로, 수십 또는 수백 번의 NFE가 필요한 확산 모델보다 훨씬 빠릅니다.[1]

## 일반화 성능 향상

### 일반화 메커니즘

**1) 모드 커버리지의 개선**: RpGAN + R1 + R2 조합은 StyleGAN의 미니배치 표준편차 트릭보다 훨씬 효과적으로 모드 드로핑을 제거합니다. StackedMNIST에서 StyleGAN의 857-881개 모드에서 1000개로 증가했습니다.[1]

**2) 그래디언트 페널티의 정규화 효과**: 논문은 Roth et al.의 분석을 참고하여, R1과 R2가 실제 데이터 분포 \(p_D\)와 생성된 분포 \(p_\theta\)를 가우시안 노이즈로 부드럽게 만든다고 설명합니다. 이는 판별기의 과적합을 감소시킵니다.[1]

**3) 최신 아키텍처의 표현력**: 현대적 ResNet 설계와 grouped convolution은 더 효율적인 특성 표현을 가능하게 하여, 제한된 매개변수 내에서 더 다양한 모드를 학습할 수 있습니다.[1]

### 회상률 (Recall) 메트릭

| 데이터셋 | R3GAN 회상률 | 비교 대상 |
|---------|-----------|---------|
| CIFAR-10 | 0.57 | StyleGAN-XL: 0.47 |
| FFHQ-256 | 0.49 | StyleGAN2: 0.43 |
| ImageNet-32 | 0.63 | 확산 모델: ~0.63 |[1] |

## 한계

### 명시적 한계 사항

**1) 기능 제한**: 저자들은 이 모델이 미니멀리즘을 우선시하기 때문에 이미지 편집이나 제어 가능한 생성 같은 downstream 응용에 부적합하다고 명시합니다. 특히 StyleGAN의 스타일 주입 기능과 잠재 공간 조작 특성이 제거되었습니다.[1]

**2) 아키텍처 개선의 부분적 적용**: 적응형 정규화(예: FiLM), 다중 헤드 자기 주의(multi-headed self-attention) 등 증명된 기법들이 의도적으로 제외되었습니다.[1]

**3) 스케일링 검증 부족**: 64×64 ImageNet 이상의 고해상도 또는 대규모 텍스트-이미지 생성 작업에서의 확장성이 아직 검증되지 않았습니다.[1]

**4) 통계적 유의성**: 각 실험이 수일에서 수주일 소요되기 때문에 오류 막대나 신뢰 구간이 제공되지 않았습니다.[1]

### 이론적 한계

**1) 국소 수렴 분석**: 증명된 수렴 보장은 국소 수렴에만 적용되며, 훈련 초기에 $\theta, \psi$ 가 $\theta^\*, \psi^*$ 에 충분히 가깝다는 가정은 현실적이지 않습니다. 논문은 이를 R1, R2 모두를 적용함으로써 실무적으로 해결합니다.[1]

**2) 정규화 트릭-오프**: 정규화 층을 제외한 설계는 분산 폭발을 방지하기 위해 Fix-up 초기화에 의존하므로 다른 설정에서의 견고성이 불명확합니다.[1]

## 최신 연구 동향과 미래 방향

### 현재 산업 동향 (2025년)

**1) 일반화 성능의 중심화**: CHAIN이라는 최신 방법은 Batch Normalization의 장점을 GAN 훈련에 적용하여 리피치츠 연속성 제약을 통해 일반화를 향상시키고 있습니다. 이는 R3GAN의 그래디언트 페널티 접근과 보완적입니다.[2]

**2) 모드 커버리지의 확대된 관심**: 최근 연구는 GAN이 diffusion 모델만큼 mode coverage를 달성할 수 있음을 보여주고 있습니다. R3GAN의 1000개 모드 커버리지는 이 추세를 강력하게 지원합니다.[3]

**3) 훈련 동역학의 제어 이론 접근**: Brownian Motion Controller와 같은 제어 이론 기반 방법이 GAN 안정성 개선에 사용되고 있으며, 이는 R1+R2의 정규화 아이디어와 유사한 맥락입니다.[4]

### GAN vs 확산 모델 경쟁

**장점 유지**:
- R3GAN은 **단일 함수 평가(NFE=1)**로 샘플 생성 (확산 모델: 79-1000 NFE)[1]
- 훈련 및 추론 비용에서 확산 모델보다 훨씬 효율적

**개선 필요 영역**:
- 확산 모델의 안정성과 일반화는 여전히 우수합니다
- 텍스트-이미지, 비디오 생성 등 복합 태스크에서 확산 모델이 선호됨[5]

### 미래 연구 시 고려할 점

**1) 대규모 확장성**: 저자들이 명시하지 않은 영역인 1024×1024 이상 해상도나 10억 개 이상의 매개변수 모델에 대한 검증이 필요합니다.[1]

**2) 멀티모달 생성**: 텍스트-이미지, 오디오-이미지 같은 조건부 생성 태스크에 R3GAN 아키텍처를 적용할 때, R1+R2 정규화가 여전히 효과적인지 검토가 필요합니다.

**3) 정규화 층의 재검토**: 최신 정규화 기법(LayerNorm, InstanceNorm 변형)이 현대적 GAN 아키텍처에 통합될 가능성이 있습니다.[6]

**4) 하이브리드 접근**: 확산-GAN 하이브리드처럼, R3GAN의 효율성과 확산 모델의 안정성을 결합하는 연구가 진행 중입니다.[7]

**5) 도메인 특화 모델**: R3GAN의 미니멀한 구조는 의료 영상, 과학 시뮬레이션 같은 특정 도메인에 맞춤형 모델을 개발하기 위한 기초로 활용될 수 있습니다.[8]

### 논문의 학계 영향

이 논문이 앞으로의 GAN 연구에 미칠 영향:

**이론적 기여**:
- 상대론적 페어링 GAN에 대한 첫 번째 수렴 증명은 GAN 최적화 이론의 공백을 채웁니다.
- 아키텍처와 손실 함수의 독립적 개선 가능성을 보여줍니다.

**실용적 영향**:
- StyleGAN2 이후 8년 만에 GAN의 실질적 개선을 제시합니다.
- 단순함이 효과성과 모순되지 않음을 입증합니다.

**커뮤니티 규범 변화**:
- GAN 연구에서 "더 많은 트릭 = 더 나은 성능"이라는 편견을 도전합니다.
- 원칙에 기반한 설계가 최신 아키텍처 차용만큼 중요함을 강조합니다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/a93c4aa4-b269-410e-8b41-6bdc5a4d67de/2501.05441v1.pdf)
[2](https://arxiv.org/pdf/2411.16567.pdf)
[3](https://arxiv.org/pdf/1902.03984.pdf)
[4](https://arxiv.org/html/2411.03999)
[5](http://arxiv.org/pdf/2102.07074v2.pdf)
[6](https://arxiv.org/abs/2111.01007)
[7](https://arxiv.org/html/2501.05441v1)
[8](http://arxiv.org/pdf/2203.11242.pdf)
[9](https://arxiv.org/pdf/2104.09630.pdf)
[10](https://www.artificialintelligencepub.com/journals/jairi/jairi-aid1004.php)
[11](https://www.sapien.io/blog/gans-vs-diffusion-models-a-comparative-analysis)
[12](https://www.aicerts.ai/news/generative-ai-trends-2025-enterprise-tech/)
[13](https://arxiv.org/abs/2404.00521)
[14](https://shieldbase.ai/blog/diffusion-models-vs-gans)
[15](https://digital.nemko.com/news/generative-ai-trends-2025-how-enterprises-scale-dependably)
[16](https://www.sciencedirect.com/science/article/abs/pii/S0951832024008627)
[17](https://vasundhara.io/blogs/diffusion-models-vs-gans-who-is-winningg-thhe-ai-image-race-in-2025)
[18](https://www.decodingdiscontinuity.com/p/tech-predictions-for-2025-scaling)
[19](https://www.sciencedirect.com/science/article/abs/pii/S0378778822004182)
[20](http://arxiv.org/pdf/1803.00657.pdf)
[21](https://arxiv.org/abs/1907.00109)
[22](https://arxiv.org/html/2306.10468)
[23](https://arxiv.org/pdf/1909.13188.pdf)
[24](http://arxiv.org/pdf/1801.04406.pdf)
[25](https://arxiv.org/abs/1910.00927)
[26](https://arxiv.org/abs/1807.00734)
[27](https://personalpages.surrey.ac.uk/w.wang/papers/Pu%20et%20al_ICANN_2023.pdf)
[28](https://www.sciencepublishinggroup.com/article/10.11648/j.ajnna.20251102.11)
[29](https://arxiv.org/pdf/1910.06922.pdf)
[30](https://huggingface.co/learn/computer-vision-course/unit2/cnns/convnext)
[31](https://www.scitepress.org/Papers/2024/129379/129379.pdf)
[32](https://cs.brown.edu/people/ycheng79/csci1952qs23/Top_Project_1_Nick%20Huang_Jayden%20Yi_Convergence%20of%20Relativistic%20GANs%20With%20Zero-Centered%20Gradient%20Penalties.pdf)
[33](https://openreview.net/pdf/9197a4074900f763675c3c896d7c4ca3402a3c55.pdf)
[34](https://openaccess.thecvf.com/content/CVPR2024/papers/Ni_CHAIN_Enhancing_Generalization_in_Data-Efficient_GANs_via_lipsCHitz_continuity_constrAIned_CVPR_2024_paper.pdf)
