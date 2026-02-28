# IMAGEBIND: One Embedding Space To Bind Them All

**출처:** Girdhar, R., El-Nouby, A., Liu, Z., Singh, M., Alwala, K.V., Joulin, A., & Misra, I. (2023). *IMAGEBIND: One Embedding Space To Bind Them All.* arXiv:2305.05665v2 [cs.CV]. FAIR, Meta AI.

---

## 1. 핵심 주장과 주요 기여 요약

### 핵심 주장
IMAGEBIND는 **이미지(image)를 중심축(anchor)**으로 활용하여, 6가지 모달리티(이미지, 텍스트, 오디오, 깊이, 열화상, IMU)를 **하나의 공유 임베딩 공간(joint embedding space)**으로 통합할 수 있음을 보인다. 핵심적으로, **모든 모달리티 쌍의 페어링 데이터가 필요하지 않으며**, 각 모달리티를 이미지와만 정렬(align)하면 **모달리티 간 창발적 정렬(emergent alignment)**이 자동으로 발생한다는 점을 실증한다.

### 주요 기여
1. **이미지 중심 바인딩(Image-centric Binding):** 이미지-페어 데이터만으로 6개 모달리티를 단일 임베딩 공간에 통합하는 방법론 제시
2. **창발적 제로샷 능력(Emergent Zero-shot Capabilities):** 직접 학습하지 않은 모달리티 쌍(예: 오디오↔텍스트)에 대해서도 제로샷 분류 및 검색 성능 달성
3. **기존 모델의 무재학습 업그레이드:** CLIP 기반 모델(Detic, DALL-E 2)을 재학습 없이 오디오 기반으로 전환 가능
4. **임베딩 산술(Embedding Arithmetic):** 서로 다른 모달리티의 임베딩을 더하여 의미적 합성(compositional semantics) 가능
5. **모달리티 간 최첨단 성능:** 오디오, 깊이 등에서 전문 지도학습 모델에 필적하거나 능가하는 창발적 제로샷 성능 달성

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

기존 멀티모달 학습은 특정 모달리티 쌍(예: (이미지, 텍스트) 또는 (비디오, 오디오))에 국한되어 학습되었으며, 학습된 임베딩은 **해당 쌍의 태스크에만 적용 가능**하다는 근본적 한계가 있었다. 예를 들어:

- CLIP [60]의 (이미지, 텍스트) 임베딩은 오디오 태스크에 직접 활용 불가
- AudioCLIP [27]의 (오디오, 텍스트) 임베딩은 깊이(depth) 태스크에 적용 불가

모든 모달리티가 동시에 존재하는 대규모 데이터셋을 수집하는 것은 현실적으로 불가능하기에, **제한된 페어 데이터만으로 통합 임베딩 공간을 학습하는 방법**이 필요했다.

### 2.2 제안하는 방법

#### 핵심 아이디어: 이미지를 통한 바인딩(Binding via Images)

이미지는 다양한 감각 경험을 연결하는 자연스러운 매개체 역할을 한다. IMAGEBIND는 각 모달리티 $\mathcal{M}$을 이미지 $\mathcal{I}$와 정렬하는 방식으로, 모든 모달리티를 간접적으로 연결한다.

#### 학습 손실 함수 (InfoNCE Loss)

모달리티 쌍 $(\mathcal{I}, \mathcal{M})$에 대해, 이미지 $\mathbf{I}_i$와 대응하는 모달리티 관측 $\mathbf{M}_i$를 각각 인코더 $f$, $g$를 통해 정규화된 임베딩으로 변환한다:

$$\mathbf{q}_i = f(\mathbf{I}_i), \quad \mathbf{k}_i = g(\mathbf{M}_i)$$

이 임베딩들은 InfoNCE 손실로 최적화된다:

$$L_{\mathcal{I}, \mathcal{M}} = -\log \frac{\exp(\mathbf{q}_i^\top \mathbf{k}_i / \tau)}{\exp(\mathbf{q}_i^\top \mathbf{k}_i / \tau) + \sum_{j \neq i} \exp(\mathbf{q}_i^\top \mathbf{k}_j / \tau)}$$

여기서:
- $\tau$: 소프트맥스 분포의 평활도를 조절하는 스칼라 온도(temperature)
- $j \neq i$: 미니배치 내 비관련 샘플(negatives)

실제 학습에는 대칭 손실을 사용한다:

$$L = L_{\mathcal{I}, \mathcal{M}} + L_{\mathcal{M}, \mathcal{I}}$$

#### 창발적 정렬 메커니즘

$(\mathcal{I}, \mathcal{M}_1)$과 $(\mathcal{I}, \mathcal{M}_2)$ 쌍으로만 학습하더라도, 이미지 임베딩을 공유 앵커로 활용함으로써 $(\mathcal{M}_1, \mathcal{M}_2)$ 간의 정렬이 **자동으로 발생**한다. 예를 들어, (이미지, 텍스트)와 (이미지, 오디오)로 학습하면, (텍스트, 오디오) 간의 정렬이 창발적으로 나타난다.

### 2.3 모델 구조

| 구성 요소 | 세부 사항 |
|----------|----------|
| **이미지/비디오 인코더** | Vision Transformer (ViT-H, 630M params), OpenCLIP에서 사전학습, 학습 중 동결(frozen) |
| **텍스트 인코더** | CLIP 텍스트 인코더 (302M params), 학습 중 동결 |
| **오디오 인코더** | ViT-B, 2초 오디오 → 128 mel-spectrogram bins → 2D 신호로 변환, patch size 16, stride 10 |
| **깊이(Depth) 인코더** | ViT-S, 1채널 이미지(disparity map)로 처리 |
| **열화상(Thermal) 인코더** | ViT-B, 1채널 이미지로 처리 |
| **IMU 인코더** | 6층 Transformer (512 dim, 8 heads), 1D 컨볼루션(커널 8)으로 프로젝션 |
| **프로젝션 헤드** | 각 인코더에 모달리티별 선형 프로젝션(linear projection) → $d$차원 정규화 임베딩 |

**학습 데이터:**
- (이미지, 텍스트): 대규모 웹 데이터 (OpenCLIP의 사전학습 활용)
- (비디오, 오디오): Audioset (~2M 비디오)
- (이미지, 깊이): SUN RGB-D (~5K 쌍, 50× 복제)
- (이미지, 열화상): LLVIP (~12K 쌍, 25× 복제)
- (비디오, IMU): Ego4D (~510K 클립)

### 2.4 성능 향상

#### 창발적 제로샷 분류 (Table 2)

| 벤치마크 | IMAGEBIND | Text Paired 기준 | 절대 SOTA |
|----------|-----------|-----------------|----------|
| ESC (오디오) | **66.9%** | 68.6% (AudioCLIP†) | 97.0% |
| NYU-D (깊이) | **54.0%** | 41.9%* | 76.7% |
| SUN-D (깊이) | **35.1%** | 25.4%* | 64.9% |
| LLVIP (열화상) | **63.4%** | - | - |
| Ego4D (IMU) | **25.0%** | - | - |

†AudioCLIP은 학습 시 AudioSet 클래스명을 사용하여 엄밀히 "zero-shot"이 아님
*OpenCLIP ViT-H를 깊이를 그레이스케일로 직접 사용

#### 창발적 제로샷 오디오 검색 (Table 3)

| 방법 | Clotho R@1 | AudioCaps R@1 |
|------|------------|---------------|
| AVFIC (오디오-텍스트 학습) | 3.0 | 8.7 |
| **IMAGEBIND** (오디오-텍스트 미학습) | **6.0** | **9.3** |

#### Few-shot 분류 (Figure 3)

- 오디오(ESC): 자기지도 AudioMAE 대비 **~40% 정확도 향상** (≤4-shot)
- 깊이(SUN-D): MultiMAE 대비 **전 설정에서 유의미한 개선**

### 2.5 한계

1. **전문 모델 대비 성능 격차:** 특정 다운스트림 태스크에 최적화된 전문 모델(supervised specialists)보다 절대 성능이 낮음 (예: ESC에서 66.9% vs. SOTA 97.0%)
2. **데이터 의존성:** 소규모 페어 데이터(SUN RGB-D ~5K, LLVIP ~12K)에 의존하여 학습이 제한적이며, 데이터 복제(50×)로 보완
3. **범용 임베딩의 태스크 적응 한계:** 특정 다운스트림 태스크(검출, 세그먼테이션 등)를 위한 적응(adaptation) 연구가 부족
4. **모달리티 간 관계의 간접성:** 이미지를 거치는 간접 정렬로 인해 일부 모달리티 쌍의 정렬이 최적이 아닐 수 있음
5. **윤리적 고려:** 웹 데이터 기반 사전학습 모델의 편향(bias)을 상속하며, 의도치 않은 모달리티 간 연관성이 생성될 수 있음

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 이미지 인코더 스케일링 효과

IMAGEBIND의 일반화 성능은 **이미지 인코더의 강도에 비례하여 향상**된다 (Figure 6). 이는 이미지 임베딩이 모든 모달리티 정렬의 앵커 역할을 하기 때문이다:

| 이미지 인코더 | NYU-D (깊이) | ESC (오디오) | LLVIP (열화상) | Ego4D (IMU) |
|------------|-------------|-------------|-------------|-------------|
| ViT-B | ~44% | ~62% | ~59% | ~18% |
| ViT-L | ~48% | ~63% | ~61% | ~22% |
| ViT-H | ~50% | ~64% | ~62% | ~24% |

깊이와 오디오에서 ViT-H는 ViT-B 대비 각각 **~7%**, **~4%**의 성능 향상을 보여, **더 강력한 시각 표현이 비시각적 모달리티의 인식 성능까지 개선**할 수 있음을 실증한다.

### 3.2 일반화를 강화하는 설계 결정

**온도 파라미터:** 고정 온도($\tau$)가 학습 가능한(learnable) 온도보다 우수 (Table 5a). 모달리티별 최적 온도가 상이함:
- 깊이/열화상/IMU: 높은 온도 ($\tau = 0.2 \sim 1.0$)
- 오디오: 낮은 온도 ($\tau = 0.05$)

**공간/시간 정렬:** 이미지-깊이 간 공간 정렬된 크롭과 비디오-오디오 간 시간 정렬된 샘플이 일반화에 핵심적:
- 공간 비정렬 → 정렬: SUN-D 16.0% → 26.7% (**+10.7%**)
- 시간 비정렬 → 정렬: ESC 55.7% → 56.7%

**데이터 증강의 모달리티별 차별화:**
- 깊이: 강한 증강(RandAugment + RandErase) → 성능 향상
- 오디오: 기본 증강이 최적 (강한 증강 시 ESC 56.7% → 22.6%로 급락)

### 3.3 비전 모델 평가 도구로서의 일반화

IMAGEBIND를 사용하여 사전학습된 비전 모델의 멀티모달 일반화 능력을 평가할 수 있음 (Table 8):

| 비전 모델 | IN1K | VGGS | ESC | SUN-D | NYU-D |
|----------|------|------|-----|-------|-------|
| DINO (자기지도) | 64.4 | **17.2** | **44.7** | **26.8** | **48.8** |
| DeiT (지도학습) | **74.4** | 9.6 | 25.0 | 25.2 | 48.0 |

ImageNet 성능(IN1K)이 높은 DeiT가 오히려 멀티모달 창발적 제로샷에서는 DINO보다 낮은 성능을 보여, **순수 비전 성능과 멀티모달 일반화가 다른 속성**임을 시사한다.

### 3.4 일반화 향상을 위한 미래 방향

1. **추가 정렬 데이터 활용:** 이미지-페어뿐 아니라 텍스트-페어 데이터나 모달리티 간 직접 페어링(예: 오디오-IMU) 추가
2. **태스크 적응(Task Adaptation):** 범용 임베딩을 특정 태스크(검출, 세그먼테이션)에 효율적으로 적응시키는 방법 연구
3. **더 큰 스케일의 페어 데이터:** 깊이, 열화상 등의 소규모 데이터셋 확대
4. **새로운 벤치마크:** 창발적 제로샷 능력을 체계적으로 평가할 벤치마크 개발

---

## 4. 향후 연구에 미치는 영향과 고려할 점

### 4.1 연구 영향

1. **멀티모달 학습의 패러다임 전환:** 모든 모달리티 쌍의 데이터가 필요하다는 기존 가정을 깨고, 단일 앵커 모달리티(이미지)를 통한 간접 정렬이라는 새로운 패러다임을 제시
2. **기존 모델의 확장 가능성:** CLIP 기반 생태계(검출, 생성, 세그먼테이션 등)를 새로운 모달리티로 **무재학습 확장**하는 실용적 경로 제공
3. **기초 모델(Foundation Model) 연구 촉진:** 단일 임베딩 공간에서 다양한 모달리티를 통합하는 "기초 모델"의 실현 가능성 입증
4. **자기지도 학습의 가치 재조명:** 자연적으로 발생하는 페어 데이터(비디오-오디오, 이미지-깊이 등)만으로도 강력한 크로스모달 표현 학습 가능

### 4.2 향후 연구 시 고려할 점

1. **앵커 모달리티 의존성:** 이미지가 앵커로 적합하지 않은 모달리티(예: 촉각, 후각)에 대한 확장 방법 연구 필요
2. **창발적 정렬의 이론적 이해:** 왜 간접 정렬이 작동하는지에 대한 이론적 분석 부족 — 임베딩 공간의 기하학적 구조에 대한 심층 연구 필요
3. **스케일링 법칙:** 모달리티 수, 데이터 규모, 인코더 크기에 따른 성능 변화의 체계적 스케일링 법칙 도출 필요
4. **편향과 공정성:** 웹 데이터 기반 사전학습 모델의 편향이 모든 모달리티로 전파될 위험성에 대한 체계적 연구 필요
5. **실시간 응용:** 다중 모달리티 인코더의 추론 비용과 실시간 응용 간의 트레이드오프 연구

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

| 연구 | 연도 | 모달리티 | 핵심 특징 | IMAGEBIND과의 차이 |
|------|------|----------|----------|-------------------|
| **CLIP** [60] (Radford et al.) | 2021 | 이미지, 텍스트 | 대규모 이미지-텍스트 대조 학습, 제로샷 분류 | 2개 모달리티에 제한; IMAGEBIND는 이를 6개로 확장 |
| **ALIGN** [31] (Jia et al.) | 2021 | 이미지, 텍스트 | 노이즈 텍스트 기반 대규모 학습 | CLIP과 유사한 2개 모달리티 제한 |
| **AudioCLIP** [27] (Guzhov et al.) | 2021 | 이미지, 텍스트, 오디오 | CLIP에 오디오 모달리티 추가, 3개 모달리티 명시적 페어링 필요 | 모든 모달리티 쌍의 명시적 페어링 필요; IMAGEBIND는 이미지-페어만 필요 |
| **CoCa** [82] (Yu et al.) | 2022 | 이미지, 텍스트 | 대조 학습 + 이미지 캡셔닝 목적 함수 | IN1K SOTA 91.0% 달성하나 2개 모달리티에 제한 |
| **Flamingo** [1] (Alayrac et al.) | 2022 | 이미지, 텍스트 | 인터리브된 이미지-텍스트 입력, few-shot 학습 | 생성적 접근; IMAGEBIND는 판별적 임베딩 학습 |
| **AVFIC** [51] (Nagrani et al.) | 2022 | 비디오, 오디오, 텍스트 | 약하게 레이블된 비디오-오디오-캡션 데이터셋 | 3개 모달리티에 약한 텍스트 지도; IMAGEBIND는 텍스트 없이도 오디오 제로샷 가능 |
| **MultiMAE** [4] (Bachmann et al.) | 2022 | 이미지, 깊이, 세그먼테이션 | 멀티태스크 마스크 오토인코더 | 재구성 기반; IMAGEBIND는 대조 학습 기반으로 few-shot에서 더 우수 |
| **OmniMAE** [20] (Girdhar et al.) | 2023 | 이미지, 비디오 | 단일 모델로 이미지와 비디오 마스크 사전학습 | 2개 시각 모달리티에 제한; IMAGEBIND는 비시각 모달리티까지 포함 |
| **OpenCLIP** [11, 30] (Cherti et al.) | 2023 | 이미지, 텍스트 | CLIP의 재현 가능한 스케일링 법칙 | IMAGEBIND의 이미지/텍스트 인코더 초기화에 활용됨 |
| **LiT** [84] (Zhai et al.) | 2022 | 이미지, 텍스트 | 동결된 이미지 인코더로 대조 학습 파인튜닝 | IMAGEBIND도 동결된 이미지/텍스트 인코더 사용; LiT은 2개 모달리티에 제한 |

### 주요 비교 관점

**1. 모달리티 확장성:**
CLIP, ALIGN, CoCa 등은 2개 모달리티에 제한되며, AudioCLIP은 3개로 확장하나 모든 쌍의 명시적 데이터가 필요하다. IMAGEBIND는 $N$개 모달리티에 대해 $N-1$개의 이미지-페어 데이터만 필요하여, 모달리티 수에 따른 페어 데이터 요구량이 $O(N)$으로 선형적이다 (기존 접근법은 $O(N^2)$ ).

**2. 학습 효율성:**
IMAGEBIND는 이미지/텍스트 인코더를 동결한 채 새 모달리티 인코더만 학습하므로, 전체 모델을 처음부터 학습하는 것보다 훨씬 효율적이다. 이는 LiT [84]의 동결 전략과 유사하되, 더 많은 모달리티로 확장한 것이다.

**3. 창발적 능력(Emergent Capabilities):**
IMAGEBIND의 가장 독창적인 기여는 직접 학습하지 않은 모달리티 쌍에서의 제로샷 능력이며, 이는 다국어 기계번역에서 관찰되는 zero-shot translation [33, 40]과 유사한 현상이다. CLIP, AudioCLIP 등은 이러한 창발적 능력을 보이지 않는다.

**4. 실용적 확장성:**
IMAGEBIND의 임베딩이 CLIP 공간과 호환됨으로써, DALL-E 2, Detic 등 기존 CLIP 기반 모델을 **재학습 없이** 새 모달리티로 확장할 수 있다는 실용적 가치가 크다. 이는 다른 멀티모달 모델에서는 제공하지 않는 고유한 장점이다.

---

### 참고 자료

- [60] Radford, A., et al. (2021). *Learning Transferable Visual Models From Natural Language Supervision.* ICML.
- [31] Jia, C., et al. (2021). *Scaling Up Visual and Vision-Language Representation Learning With Noisy Text Supervision.* ICML.
- [27] Guzhov, A., et al. (2021). *AudioCLIP: Extending CLIP to Image, Text and Audio.* arXiv:2106.13043.
- [82] Yu, J., et al. (2022). *CoCa: Contrastive Captioners are Image-Text Foundation Models.* TMLR.
- [1] Alayrac, J.-B., et al. (2022). *Flamingo: A Visual Language Model for Few-Shot Learning.* NeurIPS.
- [51] Nagrani, A., et al. (2022). *Learning Audio-Video Modalities from Image Captions.* ECCV.
- [4] Bachmann, R., et al. (2022). *MultiMAE: Multi-modal Multi-task Masked Autoencoders.* ECCV.
- [54] van den Oord, A., et al. (2018). *Representation Learning with Contrastive Predictive Coding.* NeurIPS.
- [84] Zhai, X., et al. (2022). *LiT: Zero-Shot Transfer with Locked-Image Text Tuning.* CVPR.
- [11] Cherti, M., et al. (2023). *Reproducible Scaling Laws for Contrastive Language-Image Learning.* CVPR.
- [20] Girdhar, R., et al. (2023). *OmniMAE: Single Model Masked Pretraining on Images and Videos.* CVPR.
