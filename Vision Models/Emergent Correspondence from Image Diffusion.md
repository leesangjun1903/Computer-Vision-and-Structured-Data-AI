
# Emergent Correspondence from Image Diffusion (DIFT)

> **논문 정보**
> - **제목**: Emergent Correspondence from Image Diffusion
> - **저자**: Luming Tang\*, Menglin Jia\*, Qianqian Wang\*, Cheng Perng Phoo, Bharath Hariharan (Cornell University)
> - **학회**: NeurIPS 2023
> - **arXiv**: [2306.03881](https://arxiv.org/abs/2306.03881)
> - **프로젝트 페이지**: https://diffusionfeatures.github.io

---

## 1. 📌 핵심 주장 및 주요 기여 요약

이 논문은 이미지 확산(diffusion) 모델에서 어떠한 명시적 감독(supervision) 없이도 이미지 간 대응(correspondence)이 자연스럽게 발현(emerge)됨을 보여줍니다.

저자들은 확산 네트워크로부터 이 암묵적 지식을 이미지 특징으로 추출하는 간단한 전략, 즉 **DIFT(DIffusion FeaTures)**를 제안하고, 이를 실제 이미지 간의 대응 관계 확립에 활용합니다.

### 주요 기여 요약

| 기여 | 내용 |
|---|---|
| ✅ 새로운 발견 | 명시적 학습 없이 확산 모델에서 correspondence가 출현 |
| ✅ 방법론 제안 | DIFT: 확산 U-Net의 중간 활성화(activation)를 특징 맵으로 추출 |
| ✅ 태스크 범위 | 의미적(semantic), 기하학적(geometric), 시간적(temporal) 대응 |
| ✅ 성능 | 추가 fine-tuning 없이 약지도(weakly-supervised) 및 자기지도(self-supervised) 방법을 상회 |

---

## 2. 🔬 상세 분석

### 2-1. 해결하고자 하는 문제

이미지 간 대응 관계를 찾는 것은 컴퓨터 비전의 근본적인 문제이며, 3D 재건, 객체 추적, 비디오 분할, 이미지 편집, 이미지-이미지 변환 등 수많은 응용 분야에 필수적입니다.

인간은 서로 다른 시점, 관절 변형, 조명 변화뿐 아니라 완전히 다른 카테고리(고양이와 말) 혹은 다른 모달리티(사진과 만화) 사이에서도 쉽게 대응점을 찾아낼 수 있습니다.

기존 방법들의 문제:
- **지도 학습 방법**: 대규모 레이블 데이터셋 의존, 일반화 한계
- **약지도 방법(weakly-supervised)**: 이미지 수준의 레이블 필요
- **자기지도 방법(DINO, OpenCLIP 등)**: 정밀한 픽셀 수준 대응에 한계

---

### 2-2. 제안하는 방법 (DIFT) 및 수식

#### 🔷 DDPM 순전파(Forward Process)

확산 모델의 순전파 과정은 다음과 같습니다:

$$q(\mathbf{x}_t | \mathbf{x}_0) = \mathcal{N}(\mathbf{x}_t;\, \sqrt{\bar{\alpha}_t}\,\mathbf{x}_0,\, (1-\bar{\alpha}_t)\mathbf{I})$$

여기서:
- $\mathbf{x}_0$: 원본 이미지
- $\mathbf{x}_t$: 타임스텝 $t$에서의 노이즈 이미지
- $\bar{\alpha}\_t = \prod_{s=1}^{t}(1 - \beta_s)$: 누적 노이즈 스케일링 인자

#### 🔷 DIFT 추출 전략

U-Net은 **노이즈 제거(denoising)**를 위해 훈련되었으므로 노이즈가 추가된 이미지에서 동작합니다. 이 문제를 해결하기 위해 저자들은 입력 이미지에 노이즈를 추가하여 순전파 과정을 시뮬레이션한 후 U-Net에 통과시켜 특징 맵을 추출합니다. 이를 **DIFT(DIffusion FeaTures)**라 부르며, 코사인 거리 기반의 최근접 이웃(nearest neighbor) 탐색으로 두 이미지 간 매칭 픽셀 위치를 찾습니다.

수식으로 표현하면:

$$\mathbf{x}_t = \sqrt{\bar{\alpha}_t}\,\mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t}\,\boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$

$$\text{DIFT}(\mathbf{x}_0, t) = f_\ell\!\left(\mathbf{x}_t, t\right)$$

여기서 $f_\ell(\cdot)$은 U-Net의 $\ell$번째 중간 레이어의 활성화 출력입니다.

#### 🔷 대응점 탐색 (Nearest Neighbor Matching)

소스 이미지의 픽셀 $p$에 대응하는 타겟 이미지 픽셀 $\hat{q}$를 코사인 유사도로 탐색:

$$\hat{q} = \arg\max_{q'} \frac{\text{DIFT}(\mathbf{x}^A, t)[p] \cdot \text{DIFT}(\mathbf{x}^B, t)[q']}{\|\text{DIFT}(\mathbf{x}^A, t)[p]\| \cdot \|\text{DIFT}(\mathbf{x}^B, t)[q']\|}$$

#### 🔷 앙상블(Ensembling) 전략

안정성을 높이기 위해 같은 타임스텝에서 서로 다른 랜덤 시드로 여러 번 노이즈를 추가하고, 배치로 U-Net을 통과시킨 뒤 결과 특징 맵을 평균 내어 최종 표현을 형성하는 앙상블 기법을 사용합니다.

$$\widehat{\text{DIFT}}(\mathbf{x}_0, t) = \frac{1}{K}\sum_{k=1}^{K} f_\ell\!\left(\mathbf{x}_t^{(k)}, t\right), \quad \mathbf{x}_t^{(k)} = \sqrt{\bar{\alpha}_t}\,\mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t}\,\boldsymbol{\epsilon}_k$$

---

### 2-3. 모델 구조

저자들은 두 가지 확산 모델(Stable Diffusion, ADM)을 사용하여 DIFT를 평가하며, 의미적 대응, 기하학적 대응, 시간적 대응의 세 가지 시각적 대응 태스크 그룹에 대해 실험을 수행합니다.

DIFT는 주로 U-Net의 **디코더(decoder) 피처**에서 발현되는 대응 현상을 활용합니다.

#### 타임스텝 $t$ 및 레이어 선택 전략

특징 추출에 있어 핵심 고려사항은 타임스텝 $t$와 네트워크 레이어의 선택입니다. 직관적으로 더 큰 $t$와 더 앞선 네트워크 레이어는 의미적으로 풍부한 특징을 제공하며, 더 작은 $t$와 뒤쪽 레이어는 저수준 세부 정보에 집중합니다. 최적의 $t$와 레이어 선택은 특정 대응 태스크에 따라 다릅니다.

실험적으로 의미적 태스크에는 $t=101$ 또는 $t=261$, 기하학적 태스크에는 $t=41$ 또는 $t=0$을 사용합니다.

| 모델 | 사용처 | 특징 |
|---|---|---|
| **Stable Diffusion (SD)** | 의미적 대응 주력 | LAION 기반 대규모 텍스트-이미지 쌍으로 학습 |
| **ADM (Ablated Diffusion Model)** | 비지도 설정 검증 | ImageNet에서 레이블 없이 학습 |

---

### 2-4. 성능 향상

추가적인 fine-tuning이나 태스크 특정 데이터/어노테이션 없이, DIFT는 의미적·기하학적·시간적 대응 식별에서 약지도 방법과 경쟁력 있는 off-the-shelf 특징들을 모두 능가합니다. 특히 의미적 대응에서 DIFT(Stable Diffusion 기반)는 SPair-71k 벤치마크에서 DINO를 19포인트, OpenCLIP을 14포인트 능가합니다. 심지어 18개 카테고리 중 9개에서 최신 지도학습 기반 방법도 능가합니다.

DIFT는 가려짐(occlusion), 복잡한 장면, 시점 변화, 자세 변형, 인스턴스 수준 외형 변화 조건에서도 더 나은 대응을 식별하며, 자기지도 학습 대응 특징(DIFT_sd vs. OpenCLIP; DIFT_adm vs. DINO)보다 14 PCK 포인트 이상 우수합니다.

---

### 2-5. 한계점

1. **타임스텝 및 레이어 민감도**: 태스크마다 최적의 $t$와 레이어가 다르며 수동 튜닝이 필요합니다.

2. **노이즈 의존성**: 확산 모델은 노이즈가 추가된 입력 이미지를 필요로 하는데, 이는 이미지의 정보를 파괴하고 각 태스크마다 튜닝해야 하는 노이즈 레벨 하이퍼파라미터를 도입합니다.

3. **속도 문제**: CleanDIFT(후속 연구)는 DIFT 대비 8배 속도 향상을 제공하는데, 이는 역으로 기존 DIFT의 앙상블 기반 특징 추출이 상당한 계산 비용을 요구함을 의미합니다.

4. **단일 디스크립터 부재**: DIFT는 확산 모델에서 대응 패턴이 자연스럽게 나타날 수 있음을 보여주는 진전이지만, 여전히 서로 다른 태스크에 대해 별개의 특징 디스크립터를 사용하므로 매칭 유형이 알려지지 않은 경우 활용도를 제한합니다. 또한 비지도 방식으로 학습된 대응은 완전 지도 학습 방법의 매칭 정확도에는 미치지 못합니다.

---

## 3. 🌐 모델의 일반화 성능 향상 가능성

DIFT는 fine-tuning 또는 correspondence supervision 없이, 인스턴스·카테고리·도메인을 가로질러(예: 오리에서 펭귄, 사진에서 유화까지) 의미적 대응을 확립할 수 있습니다.

이러한 일반화 능력의 근원과 향상 가능성은 다음과 같이 정리됩니다:

### 3-1. 일반화 능력의 원천

- **대규모 비지도 사전학습**: Stable Diffusion은 수십억 쌍의 텍스트-이미지 데이터로 학습되어, 다양한 시각적 개념과 구조를 내재적으로 파악합니다.
- **디노이징 목표의 부산물**: 디노이징 과정에서 모델이 이미지의 의미적 구조를 이해해야 하므로, 자연스럽게 풍부한 표현이 학습됩니다.
- DIFT는 확산 특징에 인상적인 의미적 대응이 내재됨을 보여주며, 이 전략은 이미지 확산 모델에서 비디오 확산 모델까지 다양한 아키텍처에 걸쳐 효과적입니다.

### 3-2. 일반화 향상을 위한 향후 방향

| 전략 | 설명 |
|---|---|
| **더 나은 확산 모델 학습** | 더욱 신중하게 훈련된 확산 모델을 구축하려는 노력을 장려합니다. |
| **멀티스케일 특징 통합** | 다양한 레이어·타임스텝의 특징을 집계해 태스크 의존성 감소 |
| **클린 이미지 특징 추출** | 노이즈 없이 깨끗한 이미지를 입력으로 특징 추출기를 fine-tune하여 의미적 대응 매칭, 단안 깊이 추정, 의미적 분할, 분류 등 다양한 다운스트림 태스크에서 성능을 일관되게 향상시킵니다(CleanDIFT). |
| **의료/3D 도메인 확장** | DIFT를 의료 도메인으로 확장한 MedDIFT는 사전 훈련된 3D 의료 확산 모델(MAISI)에서 멀티스케일 의미적 디스크립터를 추출하여 3D 의료 이미지의 복셀 대응 관계를 확립합니다. |

---

## 4. 📊 2020년 이후 관련 최신 연구 비교 분석

| 연구 | 연도 | 방법 | 주요 특징 | DIFT와의 차이 |
|---|---|---|---|---|
| **DINO** (Caron et al.) | 2021 | Self-supervised ViT | Self-distillation으로 시각적 특징 학습 | DIFT에 비해 SPair-71k에서 19점 낮은 성능 |
| **OpenCLIP** | 2022 | 대규모 텍스트-이미지 대조 학습 | 범용 비전-언어 특징 | DIFT에 비해 SPair-71k에서 14점 낮은 성능 |
| **DIFT (본 논문)** | 2023 | 확산 U-Net 중간 활성화 추출 | 비지도, 태스크 특정 fine-tuning 불필요 | — |
| **Diffusion Hyperfeatures** | 2023 | 다중 타임스텝·레이어 집계 + 경량 MLP | 타임스텝과 레이어에 걸쳐 다양한 특징 맵을 추출하고 경량 신경망으로 집계하여 의미적 대응을 수행합니다. | 추가 학습 MLP 필요 |
| **A Tale of Two Features** | 2023 | Stable Diffusion + DINO 융합 | Stable Diffusion과 DINO 특징을 결합 | 두 모델의 앙상블 필요 |
| **CleanDIFT** | 2024 | 클린 이미지 기반 확산 특징 추출 | 노이즈 없이 깨끗한 이미지에서 동작하며, 기존 확산 특징 추출 방식은 이미지에 노이즈를 추가해야 하고 각 태스크마다 타임스텝을 튜닝해야 합니다. | 노이즈 불필요, 8배 속도 개선 |
| **MATCHA** | 2025 | 단일 디스크립터 + 명시적 지도 학습 | MATCHA는 기하학적·의미적·시간적 매칭을 위한 단일 특징 디스크립터를 학습하며, 명시적 지도 학습을 통합하면서 대규모 파운데이션 모델의 풍부한 지식을 활용합니다. | 모든 태스크에 단일 디스크립터 사용 |
| **MedDIFT** | 2024 | 3D 의료 확산 모델로 DIFT 확장 | DIFT를 의료 도메인으로 확장, 3D 의료 이미지에서 복셀 대응 관계 확립 | 의료·3D 도메인 특화 |

---

## 5. 🔭 향후 연구에 미치는 영향 및 고려사항

### 5-1. 향후 연구에 미치는 영향

1. **확산 모델의 재해석**: 이 연구는 이미지 확산에서 발현된 대응을 더 잘 활용하는 방법과 확산 모델을 **자기지도 학습기(self-supervised learner)**로 재고(rethink)하는 미래 연구에 영감을 줄 것을 기대합니다.

2. **비지도 표현 학습 패러다임 전환**: 생성 모델이 단순히 이미지를 합성하는 것 이상으로, 시각적 이해를 위한 풍부한 표현을 내재하고 있음을 입증함으로써 생성-인식 통합 연구를 촉진합니다.

3. **다운스트림 태스크 확장**: 이 전략은 이미지 확산 모델에서 비디오 확산 모델까지 다양한 아키텍처에 효과적이며, 대응 추출, 이미지/비디오 편집 등의 다양한 응용으로 확장됩니다.

4. **의료 영상 및 3D 분야**: MedDIFT는 국소 강도 기반 등록과 풍부하게 학습된 특징 사이의 간극을 연결하며, 국소 및 전역 해부학적 의미를 모두 포착합니다.

### 5-2. 향후 연구 시 고려할 점

| 고려사항 | 세부 내용 |
|---|---|
| **하이퍼파라미터 최적화** | 타임스텝 $t$와 레이어 선택의 자동화 또는 태스크 독립적 방법 연구 필요 |
| **계산 효율성** | 앙상블 기반 특징 추출의 계산 비용 절감 (CleanDIFT 방향) |
| **단일 디스크립터** | 다양한 대응 태스크를 하나의 디스크립터로 처리하는 통합 방법론 연구 |
| **지도학습과의 결합** | 파운데이션 모델의 지식과 명시적 지도 학습을 결합하는 것이 정확하고 일반화된 매칭의 핵심임을 실험적으로 확인할 필요가 있습니다. |
| **도메인 특화 사전학습** | 의료, 위성 이미지 등 특수 도메인에서 확산 모델을 사전학습하여 DIFT 성능 극대화 |
| **노이즈 없는 특징 추출** | 클린 이미지 기반 특징 추출 방법으로 정보 손실 최소화 |
| **이론적 근거 마련** | 왜 디노이징 학습이 대응 표현을 유발하는지에 대한 이론적 분석 심화 |

---

## 📚 참고 자료 (출처)

1. **arXiv 원문**: Tang et al., "Emergent Correspondence from Image Diffusion," arXiv:2306.03881 — https://arxiv.org/abs/2306.03881
2. **NeurIPS 2023 공식 논문**: Proceedings of the 37th International Conference on Neural Information Processing Systems — https://proceedings.neurips.cc/paper_files/paper/2023/file/0503f5dce343a1d06d16ba103dd52db1-Paper-Conference.pdf
3. **프로젝트 페이지**: https://diffusionfeatures.github.io
4. **OpenReview (NeurIPS 2023 리뷰)**: https://openreview.net/forum?id=ypOiXjdfnU
5. **ACM DL**: https://dl.acm.org/doi/10.5555/3666122.3666190
6. **ResearchGate**: https://www.researchgate.net/publication/371347585
7. **Hugging Face Papers**: https://huggingface.co/papers/2306.03881
8. **NSF PAGES**: https://par.nsf.gov/biblio/10492161
9. **CleanDIFT (CVPR 2025)**: Stracke et al., "CleanDIFT: Diffusion Features without Noise" — https://arxiv.org/abs/2412.03439
10. **MATCHA (2025)**: "Towards Matching Anything" — https://arxiv.org/abs/2501.14945
11. **MedDIFT (2024)**: "Multi-Scale Diffusion-Based Correspondence in 3D Medical Imaging" — https://arxiv.org/abs/2512.05571
12. **ML with a Honk 블로그 (DIFT 해설)**: https://mlhonk.substack.com/p/36-dift-diffusion-features
