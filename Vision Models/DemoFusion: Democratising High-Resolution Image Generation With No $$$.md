
# DemoFusion: Democratising High-Resolution Image Generation With No $$$
**Ruoyi Du, Dongliang Chang, Timothy Hospedales, Yi-Zhe Song, Zhanyu Ma**
**CVPR 2024** | arXiv:2311.16973

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 문제 인식

고해상도 이미지 생성은 GenAI에서 막대한 잠재력을 가지고 있지만, 훈련에 필요한 막대한 자본 투자로 인해 소수의 대형 기업에 집중되고 유료 서비스 뒤에 숨겨져 있다.

### 핵심 주장

이 논문은 광범위한 청중이 접근할 수 있도록 유지하면서 고해상도 생성의 경계를 넓혀 고해상도 GenAI를 민주화하고자 한다. 기존 Latent Diffusion Models(LDMs)이 더 높은 해상도의 이미지 생성을 위한 잠재력을 이미 보유하고 있음을 입증한다.

### 주요 기여

DemoFusion 프레임워크는 오픈소스 GenAI 모델을 원활하게 확장하며, **Progressive Upscaling**, **Skip Residual**, **Dilated Sampling** 메커니즘을 사용하여 더 높은 해상도의 이미지를 생성한다.

SDXL은 최대 $1024^2$ 해상도의 이미지를 합성할 수 있는 반면, DemoFusion은 파인튜닝이나 과도한 메모리 사용 없이 SDXL을 4×, 16×, 심지어 더 높은 해상도로 확장한다. 생성된 모든 이미지는 단일 RTX 3090 GPU로 생성된다.

---

## 2. 논문 상세 분석

### 2-1. 해결하고자 하는 문제

#### (1) 직접 고해상도 추론의 실패

텍스트-이미지 LDM은 훈련 과정에서 많은 크롭된 사진을 접한다. 이 크롭된 사진들은 훈련 세트에 본래 존재하거나 데이터 증강을 위해 의도적으로 크롭된다. 결과적으로 SDXL 같은 모델은 때때로 객체의 국소적인 부분에 집중된 출력을 생성한다.

#### (2) 기존 패치 기반 방법의 한계

기존 오픈소스 LDM은 이미 충분한 사전 지식을 포함하고 있어 고해상도 이미지를 생성할 수 있지만, 이를 활용하기 위해 여러 결과를 융합해야 한다.

직접 SDXL 추론은 고해상도에서 실패하고, MultiDiffusion은 반복적이고 의미론적으로 일관성 없는 콘텐츠를 생성하며, ScaleCrafter는 반복적인 국소 패턴 및 객체의 여러 팔다리 등으로 전체 이미지 품질을 저하시킨다.

---

### 2-2. 제안하는 방법 (수식 포함)

#### 기반 이론: Latent Diffusion Model (LDM)

LDM의 Forward Process (Diffusion):

$$q(\mathbf{z}_t | \mathbf{z}_{t-1}) = \mathcal{N}\left(\mathbf{z}_t; \sqrt{1-\beta_t}\,\mathbf{z}_{t-1},\; \beta_t \mathbf{I}\right)$$

LDM의 Reverse Process (Denoising):

$$p_\theta(\mathbf{z}_{t-1} | \mathbf{z}_t) = \mathcal{N}\left(\mathbf{z}_{t-1}; \boldsymbol{\mu}_\theta(\mathbf{z}_t, t),\; \boldsymbol{\Sigma}_\theta(\mathbf{z}_t, t)\right)$$

여기서 $\beta_t$는 noise schedule, $\boldsymbol{\mu}_\theta$는 학습된 denoising 네트워크의 평균 예측값이다.

---

#### (1) Progressive Upscaling (점진적 업스케일링)

DemoFusion은 저해상도 생성 결과를 noise inversion을 통해 고해상도를 위한 초기화로 삼아 "upsample-diffuse-denoise" 루프를 수행한다.

이를 수식으로 표현하면, 해상도 단계를 $s = 1, 2, \ldots, S$로 정의하고 각 단계에서:

$$\mathbf{z}^{(s)}_T = \text{NoiseInversion}\left(\text{Upsample}(\hat{\mathbf{x}}^{(s-1)})\right)$$

$$\hat{\mathbf{x}}^{(s)} = \text{Denoise}\left(\mathbf{z}^{(s)}_T\right)$$

각 단계에서 이전 단계의 저해상도 결과를 업샘플링한 뒤 노이즈를 추가하여 새로운 denoising 과정의 시작점으로 사용한다.

---

#### (2) Skip Residual (스킵 잔차)

루프 내에서, 이전 diffusion 프로세스의 해당 타임스텝에서 noise-inverted 표현이 글로벌 가이던스를 위한 skip-residual로 작동한다.

Skip Residual 기법은 중간 noise-inverted 표현을 skip connection으로 활용하여 고해상도와 저해상도 이미지 사이의 전역적 일관성을 유지한다.

코사인 스케줄에 기반한 Skip Residual 블렌딩 수식:

$$\mathbf{z}^{(\text{HR})}_t \leftarrow (1 - \alpha_1(t)) \cdot \mathbf{z}^{(\text{HR})}_t + \alpha_1(t) \cdot \mathbf{z}^{(\text{LR,inv})}_t$$

여기서 $\alpha_1(t)$는 코사인 스케줄에 따라 감소하는 가중치:

$$\alpha_1(t) = \cos\left(\frac{\pi \cdot t}{2T}\right)^{c_1}$$

$c_1$은 `cosine_scale_1` 하이퍼파라미터이다. cosine_scale_1은 skip-residual의 감소율을 제어하며, 값이 작을수록 저해상도 결과와의 일관성이 높아지지만 업샘플링 노이즈가 두드러질 수 있다.

---

#### (3) Dilated Sampling (팽창 샘플링)

MultiDiffusion의 로컬 디노이징 경로를 개선하기 위해 dilated sampling을 도입하여 전역 디노이징 경로를 확립하고 더 전역적으로 일관된 콘텐츠 생성을 촉진한다.

Dilated sampling에서는 이미지의 모든 픽셀을 순서대로 선택하는 대신, 두 번째 또는 세 번째 픽셀을 선택하여 이미지의 희소하지만 더 넓은 뷰를 만든다. 이 방식으로 더 적은 단계로 더 넓은 영역을 커버하여 더 넓은 맥락을 제공한다. 이미지의 국소적 세부 사항보다 더 많은 전역 정보를 수집하기 위한 것이다. 이는 전역적 맥락을 생성하는 데 도움이 된다.

팽창 인수 $d$를 가진 뷰 $v$에서의 디노이징 예측값 융합:

$$\hat{\mathbf{z}}_t = \frac{\sum_{v} W_v \cdot \hat{\mathbf{z}}^{(v)}_t}{\sum_{v} W_v}$$

Gaussian 필터 $W_v$는 시간에 따라 적응적으로 감소:

$$W_v(t) = \mathcal{G}(\sigma(t)), \quad \sigma(t) = \sigma_0 \cdot \cos\left(\frac{\pi \cdot t}{2T}\right)^{c_3}$$

여기서 $c_3$은 `cosine_scale_3` 파라미터이다. sigma는 Gaussian 필터의 표준값이며, 값이 클수록 dilated sampling의 전역 가이던스를 촉진하지만 과도한 스무딩의 위험이 있다.

---

#### (4) MultiDiffusion 기반 로컬-글로벌 융합

최종 잠재 표현은 로컬 패치 추론과 글로벌 dilated 추론의 결합:

$$\hat{\mathbf{z}}_t = \alpha_2(t) \cdot \hat{\mathbf{z}}^{(\text{dilated})}_t + (1-\alpha_2(t)) \cdot \hat{\mathbf{z}}^{(\text{local})}_t$$

$$\alpha_2(t) = \cos\left(\frac{\pi \cdot t}{2T}\right)^{c_2}$$

cosine_scale_2는 dilated sampling의 감소율을 제어하며, 값이 작을수록 반복 문제를 더 잘 해결하지만 거친(grainy) 이미지를 초래할 수 있다.

---

### 2-3. 모델 구조

DemoFusion의 전체 파이프라인은 다음과 같다:

```
[Phase 1] 표준 해상도 생성 (SDXL → 1024×1024)
    ↓ Noise Inversion
[Phase 2] 2× 해상도 (2048×2048)
    → Skip Residual로 전역 일관성 유지
    → Dilated Sampling으로 전역 경로 생성
    → Local Patch (MultiDiffusion) 병합
    ↓ Noise Inversion
[Phase N] 4×, 16× ... 해상도 반복
```

DemoFusion의 튜닝 불필요 특성은 많은 LDM 기반 응용 프로그램과의 원활한 통합을 가능하게 한다. 예를 들어 DemoFusion과 ControlNet을 결합하면 제어 가능한 고해상도 생성을 달성할 수 있다.

DemoFusion은 점진적 방식으로 작동하므로, Phase 1의 출력을 실제 이미지 표현으로 대체하여 실제 이미지의 업스케일링을 달성할 수 있다. 그러나 출력이 기반 LDM의 잠재 데이터 분포 쪽으로 기울어지는 경향이 있어 "슈퍼 해상도"라는 용어를 사용하지 않는다.

---

### 2-4. 성능 향상

정성적으로 DemoFusion은 직접 SDXL 추론, MultiDiffusion, ScaleCrafter를 크게 능가하며, 더 선명한 털 텍스처, 더 상세한 눈, 더 풍부한 숲 식물 등 풍부한 국소 세부 사항과 강한 전역 의미론적 일관성을 달성한다.

1024×2048과 같은 낮은 해상도에서 ASGDiffusion이 최고의 FID 및 IS 점수를 보여주지만, 3072×3072와 같은 더 높은 해상도에서 DemoFusion이 FID 및 IS 지표 모두에서 ASGDiffusion을 능가한다.

---

### 2-5. 한계점

#### (a) 추론 시간 문제
DemoFusion은 우수한 이미지 품질에도 불구하고, 특히 더 높은 해상도에서 다른 방법들에 비해 더 긴 추론 시간을 요구한다(예: 4096×4096에 25분 vs SDXL+BSRGAN의 1분). 이 트레이드오프는 점진적 업스케일링과 MultiDiffusion 스타일 추론으로 인해 발생하지만, 메모리 비용은 소비자급 GPU에 적합할 만큼 낮다.

예를 들어 4 GPU를 사용할 경우 ASGDiffusion은 2048×2048 해상도에서 DemoFusion보다 13.4배 빠르게 작동한다(14초 대 188초).

#### (b) 반복(Repetition) 문제
DemoFusion은 작은 반복 우주인을 생성하는 경향이 있으며, 이미지 해상도가 높아질수록 반복 빈도가 증가하여 이미지 품질이 크게 저하된다.

#### (c) 비디오 생성으로의 확장성 한계
DemoFusion은 비디오 생성에서 완전히 예상치 못한 동작을 보인다. Dilated Sampling 메커니즘은 프레임 전체에 이상한 패턴을 만들고 Skip Residual 작동으로 전체 비디오가 흐려진다.

#### (d) LDM 분포 의존성
AccDiffusion의 분석에 따르면, 고해상도 이미지의 충실도는 사전 훈련된 확산 모델에 의존하며, 크롭된 이미지에 대한 LDM의 사전 지식에 의존하기 때문에 선명한 클로즈업 이미지 생성에서 국소적으로 비합리적인 콘텐츠를 생성할 수 있다.

---

## 3. 모델의 일반화 성능 향상 가능성

### 3-1. 다른 LDM으로의 확장성

DemoFusion이 Stable Diffusion 1.5, Stable Diffusion 2.1 등 다른 LDM과 통합될 때의 성능이 검증되었다.

DemoFusion의 튜닝 불필요 특성은 많은 LDM 기반 응용 프로그램과의 원활한 통합을 가능하게 하며, ControlNet과 결합하여 제어 가능한 고해상도 생성을 달성할 수 있다.

### 3-2. Image-to-Image 태스크로의 일반화

DemoFusion은 점진적 방식으로 작동하므로, Phase 1의 출력을 실제 이미지 표현으로 대체하여 실제 이미지의 업스케일링을 달성할 수 있다.

튜닝이 필요 없는 프레임워크로서 DemoFusion의 Image2Image 기능은 SDXL의 훈련 데이터 분포와 강하게 연관되어 있어 상당한 편향을 보일 수 있으며, 정확한 프롬프트가 성능을 크게 향상시킨다.

### 3-3. 다운스트림 태스크 일반화 가능성

LDM의 탁월한 일반화 능력은 제어 가능한 생성 및 편집 가능한 생성 등의 후속 연구뿐만 아니라, 텍스트-비디오, 텍스트-3D, 텍스트-아바타, 텍스트-인간 등 수많은 다운스트림 생성 태스크에 광범위하게 적용되고 있다.

### 3-4. 일반화의 핵심 메커니즘

기존 LDM 훈련 데이터에 이미 존재하는 크롭된 이미지에 대한 Prior Knowledge를 활용한다는 점이 일반화의 핵심이다:

$$\text{Generalization} \leftarrow \underbrace{\text{LDM Prior}}_{\text{크롭 이미지 지식}} + \underbrace{\text{DemoFusion}}_{\text{추론 시 융합 전략}}$$

DemoFusion은 기존 오픈소스 모델과의 원활한 플러그 앤 플레이 호환성을 통해 LDM의 잠재력을 활용하며, 메모리 요구량과 계산 효율성의 균형을 유지하면서 해상도 능력을 증폭한다.

---

## 4. 관련 최신 연구 비교 분석 (2020년 이후)

| 방법 | 연도 | 학회 | 핵심 메커니즘 | 주요 장점 | 한계 |
|------|------|------|--------------|---------|------|
| **MultiDiffusion** | 2023 | ICML | 겹치는 패치 융합 | 이음새 없는 이미지 | 반복 및 의미론적 비일관성 |
| **ScaleCrafter** | 2024 | ICLR | 재팽창 합성곱 수용야 조정 | 텍스처 세부 사항 우수 | 국소 반복 문제 잔존 |
| **DemoFusion** | 2024 | CVPR | Progressive Upscaling + Skip Residual + Dilated Sampling | 전역 의미 일관성, 소비자 GPU 작동 | 추론 시간 ↑, 소규모 반복 |
| **HiDiffusion** | 2024 | - | RAU-Net 특징 맵 크기 조정 | 추론 속도 40-60% 향상 | 구조 수정으로 성능 저하 가능 |
| **AccDiffusion** | 2024 | - | 패치 내용 인식 프롬프트 | 반복 없는 고해상도 | DemoFusion 파이프라인 의존 |
| **FreeScale** | 2024 | - | Scale Fusion (주파수 융합) | 8K 해상도 최초 달성 | - |

#### ScaleCrafter 분석

ScaleCrafter는 확산 모델의 U-Net 구조적 구성 요소를 검토하여 핵심 원인이 합성곱 커널의 제한된 인식 야임을 확인하고, 추론 중 합성곱 인식 야를 동적으로 조정할 수 있는 간단하지만 효과적인 재팽창을 제안한다.

#### HiDiffusion 분석

HiDiffusion은 새로운 관점을 탐색하여 U-Net의 특징 맵을 조사하며, 생성된 이미지가 구조적으로 깊은 블록의 특징 맵과 높은 상관관계가 있고 깊은 블록에서 특징 중복이 발생함을 발견한다. 고도로 중복된 특징이 합성 방향을 안내하여 객체 중복을 초래한다.

#### FreeScale 분석

DemoFusion은 로컬 패치를 사용하여 로컬 집중을 강화함으로써 국소 반복을 완화하지만, 이 국소 패치 작동은 소규모 객체 반복을 전역적으로 가져온다. 양쪽 전략의 장점을 결합하기 위해 FreeScale은 다른 수용 스케일에서 정보를 융합하여 국소 및 전역 세부 사항의 균형 잡힌 향상을 달성하는 Scale Fusion을 설계한다.

#### AccDiffusion 분석

AccDiffusion은 DemoFusion의 파이프라인을 따르며 고해상도 이미지 생성 과정에서 패치 내용 인식 프롬프트를 사용한다. 또한 AccDiffusion은 윈도우 상호작용으로 dilated sampling을 강화한다.

AccDiffusion은 더 정확한 패치 내용 인식 프롬프트와 상호작용이 있는 dilated sampling으로 도입된 더 정확한 전역 정보로 인해 DemoFusion을 능가하며, 특히 고해상도 생성 시나리오에서 두드러진다.

---

## 5. 향후 연구에 미치는 영향 및 고려할 점

### 5-1. 연구에 미치는 영향

#### (a) 튜닝 불필요 패러다임의 확립
DemoFusion은 비용이 많이 드는 훈련 없이 LDM의 해상도 능력을 향상시키는 새로운 프레임워크로, Progressive Upscaling, Skip Residual, Dilated Sampling 세 가지 혁신적 기법을 사용하여 4096 픽셀 이미지 해상도를 달성한다. RTX 3090과 같은 접근 가능한 GPU에서 작동하여 고해상도 이미지 생성의 범위를 더 넓은 청중에게 확장한다.

#### (b) 후속 연구의 기반
AccDiffusion은 DemoFusion의 파이프라인을 따르며 추가적인 개선을 통해 상태-최고 성능을 달성하였다. 이는 DemoFusion이 후속 연구의 강력한 기준선(baseline)으로 기능하고 있음을 의미한다.

#### (c) 민주화 측면에서의 영향
DemoFusion은 고해상도 이미지 생성의 민주화를 향한 중요한 발전으로, 기존 오픈소스 모델과의 원활한 플러그 앤 플레이 호환성으로 LDM의 잠재력을 활용한다. 이는 고해상도 이미지 합성에 대한 보다 공평한 접근과 혁신의 길을 열어준다.

### 5-2. 향후 연구 시 고려할 점

#### (a) 추론 속도 개선
고해상도 이미지 생성은 더 많은 실행 시간을 필요로 한다. 이는 부분적으로 점진적 업스케일링이 더 많은 패스를 요구하기 때문이지만, 주로 시간이 해상도에 따라 지수적으로 증가하여 최고 해상도 패스가 비용을 지배하기 때문이다. 멀티 GPU 병렬화나 효율적인 어텐션 메커니즘 도입이 중요하다.

#### (b) 반복 아티팩트 해결
MultiDiffusion, ScaleCrafter, DemoFusion은 고해상도 생성에서 패턴 반복 문제를 해결하지 못한다. 이 문제를 구조적으로 해결하는 것이 핵심 연구 과제이다.

#### (c) 비디오 및 3D 생성으로의 확장
DemoFusion은 비디오 생성에서 완전히 예상치 못한 동작을 보이며, Dilated Sampling 메커니즘은 프레임 전체에 이상한 패턴을 만든다. 시간적 일관성을 보장하는 새로운 메커니즘 설계가 필요하다.

#### (d) Transformer 기반 확산 모델로의 적용
현재 DemoFusion은 U-Net 기반 LDM에 최적화되어 있다. 훈련 없는 기준선 방법들은 U-Net 아키텍처에 최적화되어 SDXL에서 평가된다. DiT(Diffusion Transformer) 계열 모델(FLUX 등)로의 확장을 위한 아키텍처 독립적 방법론 개발이 중요하다.

#### (e) LDM 사전 지식의 한계 극복
SR(슈퍼 해상도) 모델은 저해상도 이미지를 더 선명하고 명확하게 보이도록 향상시킬 수 있지만, 네이티브 고해상도 이미지에 고유한 복잡한 국소 세부 사항을 제공하는 데는 부족하다. 이 구분은 고해상도 생성과 슈퍼 해상도 사이의 근본적인 차이를 강조한다.

#### (f) 평가 메트릭의 한계 고려
FID, KID, IS 등의 메트릭은 생성된 이미지와 실제 이미지 사이를 계산하지만, 이 메트릭은 일반적으로 이미지를 $299^2$ 픽셀로 리사이징해야 하여 세밀한 세부 사항의 평가를 제한한다. 고해상도 생성에 특화된 새로운 평가 지표 개발이 필요하다.

---

## 📚 참고 자료 (출처)

1. **DemoFusion 원논문 (arXiv)**: Du, R., Chang, D., Hospedales, T., Song, Y.-Z., & Ma, Z. (2023). *DemoFusion: Democratising High-Resolution Image Generation With No $$$*. arXiv:2311.16973. https://arxiv.org/abs/2311.16973

2. **DemoFusion CVPR 2024**: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024, pp. 6159–6168. https://openaccess.thecvf.com/content/CVPR2024/papers/Du_DemoFusion_Democratising_High-Resolution_Image_Generation_With_No__CVPR_2024_paper.pdf

3. **DemoFusion 공식 프로젝트 페이지**: https://ruoyidu.github.io/demofusion/demofusion.html

4. **DemoFusion GitHub**: https://github.com/PRIS-CV/DemoFusion

5. **Semantic Scholar**: https://www.semanticscholar.org/paper/DemoFusion:-Democratising-High-Resolution-Image-No-Du-Chang/af908ee2d0580b15a3814a3b302af4b61bbfdb2e

6. **University of Edinburgh Research Explorer**: https://www.research.ed.ac.uk/en/publications/demofusion-democratising-high-resolution-image-generation-with-no/

7. **ScaleCrafter (ICLR 2024)**: He, Y. et al. *ScaleCrafter: Tuning-free Higher-Resolution Visual Generation with Diffusion Models*. https://openreview.net/forum?id=u48tHG5f66

8. **HiDiffusion**: Zhang, et al. *HiDiffusion: Unlocking Higher-Resolution Creativity and Efficiency in Pretrained Diffusion Models*. arXiv:2311.17528. https://arxiv.org/html/2311.17528v2

9. **AccDiffusion**: *AccDiffusion: An Accurate Method for Higher-Resolution Image Generation*. arXiv:2407.10738. https://arxiv.org/html/2407.10738

10. **FreeScale**: *FreeScale: Unleashing the Resolution of Diffusion Models via Tuning-Free Scale Fusion*. arXiv:2412.09626. https://arxiv.org/html/2412.09626v1

11. **ASGDiffusion**: *ASGDiffusion: Parallel High-Resolution Generation with Asynchronous Structure Guidance*. arXiv:2412.06163. https://arxiv.org/html/2412.06163v1

12. **DiffuseHigh**: *DiffuseHigh: Training-free Progressive High-Resolution Image Synthesis through Structure Guidance*. arXiv:2406.18459. https://arxiv.org/html/2406.18459v5

13. **Dilated Diffusion 개념 설명 (Medium)**: https://medium.com/@ulalaparis/dilated-difusion-concept-from-demofusion-e32a7b5d09d6

14. **Liner Quick Review**: https://liner.com/review/demofusion-democratising-highresolution-image-generation-with-no

15. **Emergent Mind Summary**: https://www.emergentmind.com/papers/2311.16973
