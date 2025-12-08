# Semantic Image Synthesis via Diffusion Models

### 1. 논문의 핵심 주장과 주요 기여[1]

**Semantic Diffusion Model (SDM)**은 Denoising Diffusion Probabilistic Models (DDPMs)을 기반으로 의미론적 이미지 합성(semantic image synthesis) 문제를 해결하는 혁신적 프레임워크입니다. 이 논문의 핵심 주장은 기존의 GAN 기반 방식이 복잡한 장면에서 높은 충실도(fidelity)와 다양성(diversity)을 동시에 달성하지 못한다는 점을 지적하며, 확산 모델의 반복적 정제(progressive refinement) 능력이 이를 극복할 수 있다는 것입니다.[1]

주요 기여는 다음과 같습니다:[1]

1. **구조적 개선**: 기존 조건부 확산 모델과 달리, 의미론적 배치(semantic layout)와 노이즈가 있는 이미지를 독립적으로 처리하는 새로운 네트워크 구조 제안
2. **다층 공간 적응 정규화(Multi-layer Spatially-Adaptive Normalization)**: SPADE 연산자를 활용한 디코더 수준의 의미론적 정보 주입
3. **Classifier-free Guidance 도입**: 조건부/비조건부 모델의 예측 차이를 활용한 샘플링 개선
4. **광범위한 실험**: 네 개의 벤치마크 데이터셋(Cityscapes, ADE20K, CelebAMask-HQ, COCO-Stuff)에서 최첨단 성능 달성

***

### 2. 해결 문제, 제안 방법, 모델 구조, 성능 향상[1]

#### 2.1 해결하고자 하는 문제

의미론적 이미지 합성은 의미 맵(semantic label map)을 입력으로 하여 사실적이고 다양한 이미지를 생성하는 작업입니다. 기존 GAN 기반 방법들의 한계:[1]

- **충실도와 다양성의 부조화**: 높은 충실도를 추구하면 다양성이 감소하고 그 반대도 성립
- **복잡한 장면에서의 성능 저하**: 미세한 디테일(예: 먼 자동차, 신호등)을 정확히 생성하기 어려움
- **모드 붕괴(mode collapse)** 문제

#### 2.2 제안하는 방법과 수식

**조건부 DDPM 기초 이론**:[1]

$$p_\theta(y_{0:T}|x) = p(y_T) \prod_{t=1}^{T} p_\theta(y_{t-1}|y_t, x)$$

$$p_\theta(y_{t-1}|y_t, x) = \mathcal{N}(y_{t-1}; \mu_\theta(y_t, x, t), \Sigma_\theta(y_t, x, t))$$

순방향 과정(forward process):

$$q(y_t|y_{t-1}) = \mathcal{N}(y_t; \sqrt{1-\beta_t}y_{t-1}, \beta_t I)$$

$$q(y_t|y_0) = \mathcal{N}(y_t; \sqrt{\alpha_t}y_0, (1-\alpha_t)I)$$

여기서 \(\alpha_t := \prod_{s=1}^{t}(1-\beta_s)\)

손실 함수(denoising objective):

$$\mathcal{L}_{t-1} = \mathbb{E}_{y_0, \epsilon}[\gamma_t \|\epsilon - \epsilon_\theta(\sqrt{\alpha_t}y_0 + \sqrt{1-\alpha_t}\epsilon, x, t)\|^2]$$

**Classifier-free Guidance**:[1]

기본 가이던스 원리:

$$\hat{\epsilon}_\theta(y_t|x) = \epsilon_\theta(y_t|x) + s \cdot (\epsilon_\theta(y_t|x) - \epsilon_\theta(y_t|\emptyset))$$

여기서 \(s\)는 가이던스 스케일(guidance scale), \(\emptyset\)는 null 라벨(모두 영)

**SPADE 기반 정규화**:[1]

$$f^{i+1} = \gamma^i(x) \cdot \text{Norm}(f^i) + \beta^i(x)$$

여기서 \(\gamma^i(x), \beta^i(x)\)는 의미 맵에서 학습된 공간 적응 가중치와 편향

#### 2.3 모델 구조[1]

**인코더 부분 (Encoder)**:
- 노이즈가 있는 이미지를 입력으로 받음
- 시간단계 \(t\)에 따른 계수 조정: \(f^{i+1} = w(t) \cdot f^i + b(t)\)
- 자기-주의(self-attention) 블록 포함 (32×32, 16×16, 8×8 해상도)

**자기-주의 블록 메커니즘**:[1]

$$f(x) = W_f x, \quad g(x) = W_g x, \quad h(x) = W_h x$$

$$M(u,v) = \frac{f(x_u)^\top g(x_v)}{\|f(x_u)\| \|g(x_v)\|}$$

$$y_u = x_u + W_v \sum_v \text{softmax}_v(\alpha M(u,v)) \cdot h(x_v)$$

**디코더 부분 (Decoder)**:
- 의미 맵을 다층 SPADE 연산자를 통해 주입
- 인코더의 특징맵과 스킵 연결(skip connection)
- SDDResblock (Semantic Diffusion Decoder Resblock) 사용

#### 2.4 성능 향상[1]

**정량적 성능**:

| 데이터셋 | FID↓ | LPIPS↑ | mIoU↑ |
|---------|------|--------|-------|
| CelebAMask-HQ | 18.8 | 0.422 | 77.0 |
| Cityscapes | 42.1 | 0.362 | 77.5 |
| ADE20K | 27.5 | 0.524 | 39.2 |
| COCO-Stuff | 15.9 | 0.647 | 40.2 |

- **FID(Fidelité Inception Distance)**: 생성 이미지의 화질을 평가하는 지표로, 낮을수록 좋음
- **LPIPS(Learned Perceptual Image Patch Similarity)**: 다양성을 평가하는 지표로, 높을수록 좋음
- **mIoU(mean Intersection-over-Union)**: 의미론적 일관성 평가, 높을수록 좋음

**사용자 연구 결과**:[1]
- SPADE 대비: 76.5~94.0%의 사용자 선호도
- INADE 대비: 75.5~93.5%의 사용자 선호도
- OASIS 대비: 80.0~84.0%의 사용자 선호도

**주요 개선 요인**:

1. **구조적 분리의 효과**: 의미 맵을 디코더에만 주입함으로써 기존 방식 대비 +5.3 mIoU 향상 (ADE20K)
2. **Classifier-free Guidance의 효과**: FID 개선 및 mIoU 향상 (+9.5 mIoU)
3. **미세 디테일 생성**: 기존 GAN 기반 방법들과 달리 구조적으로 올바른 세부사항 생성

***

### 3. 모델의 일반화 성능 향상 가능성[2][3][4][1]

#### 3.1 논문 내 일반화 성능 분석[1]

**다중 데이터셋 일반화**:
논문은 4개의 서로 다른 벤치마크 데이터셋에서 일관되게 우수한 성능을 달성했습니다:[1]

- **도시 장면 (Cityscapes)**: 교통 장면의 복잡한 기하학적 구조
- **자연 장면 (ADE20K, COCO-Stuff)**: 다양한 실내/실외 환경
- **얼굴 합성 (CelebAMask-HQ)**: 특정 도메인(얼굴) 내 다양성

**해석 가능성 기반 일반화**:
의미론적 일관성(mIoU)이 높다는 것은 모델이 의미 맵의 고수준 의미를 올바르게 이해하고 유지한다는 의미로, 이는 **도메인 간 전이 가능성**을 시사합니다.[1]

#### 3.2 최근 연구 트렌드와 일반화 성능 향상 방안[3][4][5][6][2]

**1. 사전 학습 모델 활용 (Transfer Learning)**[5][2]

2024년 최근 연구는 **ControlNet** 기반 접근을 강조합니다:[2][1]
- Stable Diffusion의 대규모 사전 학습 능력 활용
- 특정 작업에 맞춘 세밀한 조정(fine-tuning)

LoRA(Low-Rank Adaptation)를 통한 경량 적응:
$$W' = W + \alpha \frac{B A}{r}$$

여기서 \(A, B\)는 저차원 행렬, \(r\)은 저차 수

**2. 도메인 적응 (Domain Adaptation)**[7][8][9]

최근 연구(2023-2025)에서 제시하는 방향:[8][7]

- **Multi-source Domain Adaptation**: 여러 도메인의 데이터를 동시에 활용
- **Online Domain Adaptation**: 배포 단계에서의 연속적 적응
- **Cross-domain Consistency**: 도메인 간 의미론적 일관성 유지

**3. Few-shot Learning과 일반화**[4][6][3][5]

**메타-학습(Meta-Learning) 기반 접근**:[4]

$$\min_\theta \sum_{i=1}^{N} \mathcal{L}(f_\theta(\mathcal{T}_i), y_i)$$

여기서 \(\mathcal{T}_i\)는 개별 작업(task)

최근 "Diffusion-FSCIL" 연구(2025)는 확산 모델의 다중 스케일 표현을 활용하여 몇 가지 샘플만으로도 새로운 클래스 학습 가능함을 보였습니다.[6]

**4. Vision Transformer(ViT) 기반 확산 모델**[10][11]

DiffiT(Diffusion Vision Transformers, 2023)는 시간 의존 다중 헤드 자기-주의(TMSA) 메커니즘 도입:[11]

- 각 시간 단계별로 다른 주의 가중치 학습
- 공간적 관계를 더 정확히 포착

이는 **일반화 성능을 향상**시킬 수 있습니다.

#### 3.3 SDM의 일반화 한계와 개선 방향

**논문에서 지적한 한계**:[1]

> "우리의 의미 확산 모델은 데이터셋에서 처음부터 훈련된 픽셀 레벨 확산 모델에 기반하므로, 우리의 접근 방식은 데이터셋에서 좋은 성능을 보이지만 자연 이미지에 대해 일반화되지 않습니다."[1]

**개선 방안**:

1. **SDM-LoRA 실험**:[1]
   - 사전 학습된 Stable Diffusion에 LoRA 모듈 추가
   - 의미 맵을 공간 적응 방식으로 주입
   - 결과: 기본 SDM과 비교하여 유망한 성과 시사

2. **고급 공간 적응 기법 통합**:[1]
   - INADE, CLADE, CC-FPSE 등 개선된 정규화 방식
   - GAN 기반 방법에서 증명된 기법들을 확산 모델에 적용

***

### 4. 앞으로의 연구에 미치는 영향과 고려사항

#### 4.1 학술적 영향[12][2][1]

**패러다임 전환**:
이 논문은 **의미론적 이미지 합성 분야에서 GAN 중심에서 확산 모델 중심으로의 전환**을 촉발했습니다. 2022년 발표 이후:[13][2][1]

- ControlNet (2023): 텍스트-이미지 확산에 조건부 제어 추가[14]
- SatDM (2023): 위성 이미지 합성에 적용[15]
- SCDM (2024): 노이즈가 있는 라벨에 대한 견고성 개선[2]

#### 4.2 응용 분야에 미치는 영향[16][17][1]

**1. 자율 주행 시뮬레이션**:
- 복잡한 도시 장면의 현실적 생성
- 데이터 증강을 통한 모델 강화

**2. 의료 영상 합성**:
- 최근 Report2CT (MICCAI 2025): 방사선 보고서에서 3D CT 생성[16]
- 다중 텍스트 인코더를 통한 임상 문맥 정확도 향상

**3. 얼굴 편집 및 조작**:
- 의미 맵 편집을 통한 대화형 이미지 조작 (그림 13)[1]
- 원본 이미지 문맥 보존

#### 4.3 앞으로 연구 시 고려할 점

**1. 계산 효율성 개선**[1]

현재 문제점:
- 1000개의 디노이징 스텝으로 인한 느린 샘플링 (V100에서 약 60초/이미지)

최근 개선 방안:
- **DPM-Solver 적용**: 100 스텝으로 감소 가능 (17.5초/이미지)[1]
- **일반 ODE 기반 고속 샘플러** 연구 (2024-2025)[18]

**2. 대규모 사전 학습 모델 활용**[12][2][1]

최근 트렌드(2023-2025):
- Stable Diffusion, CLIP 등 기반 모델 활용
- 매개변수 효율적 적응(parameter-efficient adaptation)
- 다중 모달리티 학습

**3. 견고성 및 신뢰성 향상**[2]

최근 연구:
- **SCDM (Stochastic Conditional Diffusion)**: 노이즈가 있는 의미 맵에 대한 견고성[2]
- 적대적 로버스트성(adversarial robustness) 검증 필요

**4. 도메인 적응 및 일반화**[3][7][8]

향후 연구 방향:
- **Zero-shot 및 Few-shot 일반화**: 학습되지 않은 새로운 의미 클래스 처리
- **도메인 시프트 대응**: 시간적 변화(날씨, 조명) 또는 도메인 간 변화 처리

**수식**: Cross-domain consistency loss로 볼 수 있는 개념:

$$\mathcal{L}_{cross} = \mathbb{E}_{x_s, x_t} [\|\text{Seg}(y_s) - \text{Seg}(y_t)\|^2]$$

여기서 \(y_s, y_t\)는 소스/타겟 도메인의 생성 이미지

**5. 이론적 기초 강화**[12]

현재 갭(Gap):
- 무조건 확산 모델에 비해 조건부 확산 모델의 이론적 이해 부족
- 가이던스 전략의 최적성 보장 미흡

향후 필요성:
- 조건부 점수 함수 \(\nabla \log p_t(x|y)\)의 수렴 보장
- 가이던스 강도의 이론적 최적값 도출

**6. 실시간 애플리케이션**[1]

Controlled Generation(의미 기반 편집):
- 재페인팅 알고리즘 개선
- 사용자 상호작용 최적화

***

### 5. 2020년 이후 관련 최신 연구 탐색[13][15][5][6][11][3][4][12][2][1]

#### 5.1 핵심 발전 시간표

| 연도 | 방법 | 주요 기여 | 상태 |
|------|------|---------|------|
| 2020 | DDPM (Ho et al.) | 확산 모델 기초 확립 | 기초 이론 |
| 2021 | Classifier-free Guidance | 무분류기 가이던스 제안 | 광범위 채택 |
| 2022 | **SDM (이 논문)** | **의미 합성에 확산 모델 적용** | **주요 전환점** |
| 2023 | ControlNet[14] | 다양한 조건에 대한 제어 | 실제 응용 |
| 2023 | DiffiT[11] | ViT 기반 확산 모델 | 아키텍처 혁신 |
| 2023 | SatDM[15] | 위성 이미지 합성 | 도메인 특화 |
| 2024 | SCDM[2] | 노이즈 견고성 개선 | 신뢰성 강화 |
| 2024 | DIAGen[3] | Few-shot 학습 강화 | 일반화 개선 |
| 2024-2025 | Diffusion-FSCIL[6] | 점진적 클래스 학습 | 연속 학습 |

#### 5.2 주요 최신 연구 방향 (2024-2025)[5][6][3][12][2]

**1. 조건부 확산의 이론적 발전**[12]

Nature Scientific Reports (2024)에 게재된 종합 리뷰:[12]
- 연속시간 확산 프로세스의 수학적 형식화
- 조건부 점수 함수의 수렴성 보장
- 검은 상자 최적화와의 연결

**2. Few-shot 학습과의 융합**[6][3][4][5]

**Meta-DM** (2023):[4]
- 데이터 증강 모듈로서의 확산 모델 활용
- 기존 Few-shot 방법과 결합 가능

**Diffusion-FSCIL** (2025):[6]
- 점진적 클래스 학습에서 확산 모델의 다중 스케일 표현 활용
- 혼재된 클래스 간 구분(class boundary) 유지

**3. 도메인 적응**[7][8][5][2]

**Model Diffusion** (2025):[5]
- 학습 이론적 보증이 있는 전이 학습
- 한계 위험(PAC-Bayes 경계) 제공

**4. 빠른 샘플링**[18][1]

**원스텝 확산**:[18]
- 지식 증류(knowledge distillation)를 통한 가속
- 샘플링 시간을 초 단위로 단축

#### 5.3 멀티모달 확장[19][20][14]

**ArtAug (2024)**:[19]
- 텍스트-이미지 모델 간 상호작용
- 자기 수정(self-corrective) 레이저닝 적용

**의미 기반 스타일 제어 (2025)**:[20]
- 다중 참조 이미지에서 스타일 추출
- Scene Graph를 통한 정밀 제어

***

### 결론

Semantic Diffusion Model은 **의미론적 이미지 합성 분야에 혁신적인 변화를 가져온 논문**입니다. 기존 GAN 기반 방식의 충실도-다양성 트레이드오프 문제를 확산 모델의 반복적 정제 능력으로 극복했으며, 다층 공간 적응 정규화와 classifier-free guidance를 통해 의미론적 일관성과 시각적 품질을 동시에 달성했습니다.

**향후 연구의 핵심 과제**는:
1. **계산 효율성**: 대규모 실시간 응용을 위한 샘플링 가속화
2. **일반화 성능**: 사전 학습 모델 활용과 도메인 적응
3. **이론적 기초**: 조건부 확산 모델의 수렴성 및 최적성 보장
4. **신뢰성**: 노이즈 견고성 및 안전성 검증

2023년 ControlNet 이후 현재까지의 발전은 **조건부 제어의 정밀성 향상**과 **다양한 도메인으로의 확장**에 초점을 맞추고 있으며, 이는 SDM이 제시한 방향을 계승하고 더욱 정교화하는 과정입니다. 특히 2024-2025년의 최근 연구들은 **대규모 사전 학습 모델의 활용**과 **이론적 보장의 제공**이 차세대 확산 기반 합성 모델의 핵심이 될 것임을 시사합니다.

***

### 참고자료

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/36ffd05e-fe90-4918-a02f-0c2e967b46a3/2207.00050v3.pdf)
[2](https://arxiv.org/pdf/2402.16506.pdf)
[3](https://link.springer.com/10.1007/978-3-031-85181-0_10)
[4](https://arxiv.org/pdf/2305.08092.pdf)
[5](https://arxiv.org/pdf/2502.06970.pdf)
[6](https://arxiv.org/html/2503.23402v1)
[7](https://papers.nips.cc/paper/8949-multi-source-domain-adaptation-for-semantic-segmentation)
[8](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136940125.pdf)
[9](https://openaccess.thecvf.com/content/WACV2023/papers/Cho_Domain_Adaptive_Video_Semantic_Segmentation_via_Cross-Domain_Moving_Object_Mixing_WACV_2023_paper.pdf)
[10](https://www.v7labs.com/blog/vision-transformer-guide)
[11](https://arxiv.org/abs/2312.02139)
[12](https://academic.oup.com/nsr/article/11/12/nwae348/7810289)
[13](https://arxiv.org/html/2207.00050v3)
[14](https://openaccess.thecvf.com/content/ACCV2024/papers/Gao_PSG-Adapter_Controllable_Planning_Scene_Graph_for_Improving_Text-to-Image_Diffusion_ACCV_2024_paper.pdf)
[15](https://arxiv.org/pdf/2309.16812.pdf)
[16](https://arxiv.org/abs/2509.14780)
[17](https://dl.acm.org/doi/10.1145/3746266.3762155)
[18](https://arxiv.org/abs/2410.12557v1)
[19](https://arxiv.org/html/2412.12888v1)
[20](https://www.nature.com/articles/s41598-025-28715-x)
[21](https://www.semanticscholar.org/paper/945a899a93c03eb63be5e3197e318c077473cef9)
[22](https://dl.acm.org/doi/10.1145/3707292.3707367)
[23](https://link.springer.com/10.1007/s10489-025-06673-1)
[24](https://www.semanticscholar.org/paper/6c708659768e470f63d06f791ff8420e7ff0feac)
[25](https://ieeexplore.ieee.org/document/11225950/)
[26](https://ieeexplore.ieee.org/document/11141031/)
[27](https://www.mdpi.com/2227-9059/13/12/2862)
[28](https://iopscience.iop.org/article/10.1149/MA2025-031244mtgabs)
[29](http://arxiv.org/pdf/2306.04321.pdf)
[30](http://arxiv.org/pdf/2309.14303.pdf)
[31](https://arxiv.org/pdf/2112.05744v3.pdf)
[32](https://arxiv.org/pdf/2312.03048.pdf)
[33](https://www.sciencedirect.com/science/article/abs/pii/S0957417424025120)
[34](https://towardsdatascience.com/six-ways-to-control-style-and-content-in-diffusion-models/)
[35](https://openreview.net/pdf/671acc9490cd564242d9a886119fb2c4c1a47b7a.pdf)
[36](https://kimjy99.github.io/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0/semantic-diffusion-model/)
[37](https://www.ijcai.org/proceedings/2025/0728.pdf)
[38](https://arxiv.org/html/2403.04279v1)
[39](https://www.semanticscholar.org/paper/afaf5efb17a33bb5c38b06d5b2da649d0af95f15)
[40](https://ieeexplore.ieee.org/document/10558817/)
[41](https://ieeexplore.ieee.org/document/10504869/)
[42](http://biorxiv.org/lookup/doi/10.1101/2024.04.25.591050)
[43](https://ieeexplore.ieee.org/document/10716619/)
[44](https://ieeexplore.ieee.org/document/10863102/)
[45](https://ieeexplore.ieee.org/document/10698826/)
[46](https://arxiv.org/abs/2411.01168)
[47](https://link.springer.com/10.1007/s00521-024-09645-7)
[48](http://arxiv.org/pdf/2308.03047.pdf)
[49](https://arxiv.org/pdf/2305.18455.pdf)
[50](https://arxiv.org/pdf/2110.09446.pdf)
[51](http://arxiv.org/pdf/2410.10663.pdf)
[52](https://www.merl.com/publications/docs/TR2025-025.pdf)
[53](https://proceedings.neurips.cc/paper_files/paper/2024/file/b25222d2d405e0768d218e7fc90070b2-Paper-Conference.pdf)
[54](https://pmc.ncbi.nlm.nih.gov/articles/PMC10892187/)
[55](https://arxiv.org/html/2502.06970v1)
[56](https://papers.nips.cc/paper_files/paper/2024/file/f782860c2a5d8f675b0066522b8c2cf2-Paper-Conference.pdf)
