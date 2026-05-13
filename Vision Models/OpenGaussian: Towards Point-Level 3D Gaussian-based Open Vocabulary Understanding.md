
# OpenGaussian: Towards Point-Level 3D Gaussian-based Open Vocabulary Understanding

> **논문 정보**
> - **제목**: OpenGaussian: Towards Point-Level 3D Gaussian-based Open Vocabulary Understanding
> - **저자**: Yanmin Wu, Jiarui Meng, Haijie Li, Chenming Wu, Yahao Shi, Xinhua Cheng, Chen Zhao, Haocheng Feng, Errui Ding, Jingdong Wang, Jian Zhang
> - **학회**: NeurIPS 2024 (Advances in Neural Information Processing Systems, pp.19114–19138)
> - **arXiv**: [2406.02058](https://arxiv.org/abs/2406.02058) (v1: 2024.06.04 / v2: 2024.12.06)
> - **프로젝트 페이지**: https://3d-aigc.github.io/OpenGaussian
> - **GitHub**: https://github.com/yanmin-wu/OpenGaussian

---

## 1. 핵심 주장과 주요 기여 (간결 요약)

### 🎯 핵심 주장

이 논문은 3D Gaussian Splatting(3DGS) 기반의 **3D 포인트 레벨 오픈 보케블러리 이해** 능력을 지닌 OpenGaussian 방법을 제안한다. 핵심 동기는 기존 3DGS 기반 오픈 보케블러리 방법들이 주로 **2D 픽셀 수준의 파싱**에 집중해 왔다는 관찰에서 비롯된다.

이는 기존 3DGS 방법들이 장면을 2D 픽셀 수준에서 해석하는 데 그쳤던 것과 달리, OpenGaussian은 자연어를 이용해 **3D 공간의 개별 포인트 수준**에서 장면을 이해하고 상호작용하는 것을 가능하게 한다는 것을 의미한다.

---

### 📌 주요 기여 (3가지 핵심 기술 기여)

논문의 기술적 기여는 다음과 같이 요약된다:
1. SAM의 부울(boolean) 마스크와 **인트라-마스크 스무딩 손실(intra-mask smoothing loss)** 및 **인터-마스크 대조 손실(inter-mask contrastive loss)**을 활용하여, 크로스-프레임 연계 없이 3D 일관성을 가진 포인트 레벨 인스턴스 피처를 학습;
2. 인스턴스 피처를 거칠게-섬세하게(coarse-to-fine) 이산화하는 **2단계 코드북(two-level codebook)** 도입;
3. IoU 및 피처 거리 기반으로 CLIP 피처를 3D 포인트에 연계하는 **인스턴스 레벨 2D-3D 연계 방법(instance-level 2D-3D association method)** 제안.

OpenGaussian은 피처 차원 압축 또는 양자화를 위한 추가 네트워크 없이, 원본 CLIP 피처의 오픈 보케블러리 능력을 그대로 계승한다.

---

## 2. 상세 설명

### 2.1 해결하고자 하는 문제

기존 3DGS 기반 오픈 보케블러리 방법들은 주로 2D 픽셀 수준 파싱에 집중해 왔으며, **피처 표현력 부족(weak feature expressiveness)** 과 **부정확한 2D-3D 피처 연관(inaccurate 2D-3D feature associations)** 문제로 인해 3D 포인트 레벨 태스크에서 어려움을 겪고 있다.

이 두 가지 문제를 구체적으로 분해하면:

**문제 1: 약한 피처 표현력**

수백만 개의 Gaussian 포인트에 대해 고차원 언어 피처를 학습하는 것은 계산적으로 매우 까다롭기 때문에, 기존 방법들은 피처의 고유성과 표현력을 희생하는 **차원 축소 기법**에 의존하게 된다.

**문제 2: 부정확한 2D-3D 대응**

3DGS의 알파-블렌딩(alpha-blending) 렌더링 과정이 2D 픽셀과 3D 포인트 사이의 직접적인 일대일 관계 수립을 어렵게 만들어, 정확한 3D 해석을 방해한다.

---

### 2.2 제안하는 방법 (수식 포함)

OpenGaussian의 파이프라인은 **3단계**로 구성된다.

#### ① Stage 1: SAM 기반 3D 일관성 인스턴스 피처 학습

크로스-프레임 연계 없이 SAM 마스크를 사용하여 3D 일관성 있는 인스턴스 피처를 학습한다. 이 피처는 **오브젝트 내부 일관성(intra-object consistency)** 과 **오브젝트 간 구별성(inter-object distinction)** 을 동시에 나타낸다.

학습에는 두 가지 손실 함수가 사용된다:

**인트라-마스크 스무딩 손실 (Intra-mask Smoothing Loss):**

SAM 마스크 $B_i$와 렌더링된 피처 맵 $M$이 주어졌을 때, 마스크 내의 평균 피처 $\bar{M}_i$는 다음과 같이 계산된다:

$$\bar{M}_i = \frac{B_i \cdot M}{\sum B_i}$$

스무딩 손실 $\mathcal{L}_s$는 다음과 같이 정의된다:

$$\mathcal{L}_s = \sum_{i=1}^{m} \sum_{h=1}^{H} \sum_{w=1}^{W} B_{i,h,w} \cdot \| M_{:,h,w} - \bar{M}_i \|_2^2$$

여기서 $H$와 $W$는 이미지 차원이고, $m$은 SAM 마스크의 수이다.

**인터-마스크 대조 손실 (Inter-mask Contrastive Loss):**

이 손실은 서로 다른 인스턴스 간의 피처 다양성을 촉진하기 위해 서로 다른 SAM 마스크의 평균 피처 간 거리를 최대화한다. 대조 손실 $\mathcal{L}_c$는 다음과 같이 정의된다:

$$\mathcal{L}_c = \frac{1}{m(m-1)} \sum_{i=1}^{m} \sum_{j=1, j \neq i}^{m} \| \bar{M}_i - \bar{M}_j \|_2^2$$

이 두 손실을 최적화함으로써, 같은 오브젝트에 속하는 Gaussian 포인트들이 유사하고 일관된 피처를 학습하는 동시에, 서로 다른 오브젝트의 피처는 구별되게 유지된다.

**총 학습 손실:**

$$\mathcal{L}_{total} = \mathcal{L}_s - \lambda \mathcal{L}_c$$

여기서 $\lambda$는 두 손실 사이의 균형을 조절하는 하이퍼파라미터이다.

---

#### ② Stage 2: 거칠게-섬세하게 코드북 (Coarse-to-Fine Codebook)

인스턴스 피처를 거칠게-섬세하게 이산화하는 2단계 코드북을 제안한다. **거친 레벨(coarse level)**에서는 3D 포인트의 위치 정보를 고려하여 위치 기반 클러스터링을 수행하고, **섬세한 레벨(fine level)**에서 이를 정제한다.

- **코어스 레벨**: 3D 공간 좌표 $(x, y, z)$를 기반으로 공간적으로 근접한 포인트들을 묶는 위치 기반 클러스터링 수행
- **파인 레벨**: 코어스 레벨 클러스터를 더 세밀하게 분리

각 Gaussian 포인트에는 해당 서브-클러스터를 나타내는 파인 레벨 인덱스 $I_{fine}$이 할당된다. 이후 **의사 피처 손실(pseudo feature loss)**을 도입하여 피처 학습을 더욱 정제한다. 1단계에서 학습된 인스턴스 피처가 양자화된 피처를 감독하는 의사 정답(pseudo ground truth)으로 사용된다.

---

#### ③ Stage 3: 인스턴스 레벨 2D-3D 피처 연계

OpenGaussian은 3D Gaussian 포인트를 CLIP 언어 피처와 연계시키기 위해, 피처 압축을 위한 추가 네트워크나 장면별(scene-specific) 인코더-디코더가 필요 없으며, 복잡한 깊이 기반 가림(occlusion) 테스트도 수행하지 않는다.

구체적으로는:

단일 3D 인스턴스의 피처가 "단일 인스턴스 맵(single-instance map)"으로 렌더링된다. 현재 뷰의 각 SAM 마스크와의 **IoU(Intersection over Union)**를 계산한다. 연계를 정제하기 위해 SAM 마스크에 의사 정답 피처를 채워 "피처-채운 마스크(feature-filled masks)"를 구성한다. "단일 인스턴스 맵"과 각 "피처-채운 마스크" 간의 피처 거리를 계산한다. **IoU와 피처 유사도의 조합이 가장 높은 SAM 마스크**가 해당 3D 인스턴스와 연계되고, 해당 마스크의 CLIP 이미지 피처가 3D 인스턴스의 Gaussian 포인트들에 연계된다.

멀티뷰 피처도 통합하여 강건성을 향상시킨다. 이 방법은 훈련 없이 3D 포인트를 풍부한 CLIP 언어 피처에 효과적으로 연결한다.

---

### 2.3 모델 구조

아래는 OpenGaussian의 전체 프레임워크이다:

```
입력 이미지 (다중 뷰)
      │
      ▼
[SAM 마스크 생성] ──────────────────────────────────────────┐
      │                                                    │
      ▼                                                    │
[Stage 1: 3D 인스턴스 피처 학습]                           │
 - Intra-mask Smoothing Loss (L_s)                         │
 - Inter-mask Contrastive Loss (L_c)                       │
 → 3DGS에 인스턴스 피처 임베딩                             │
      │                                                    │
      ▼                                                    │
[Stage 2: 2-Level Codebook 이산화]                         │
 - Coarse-level: 3D 위치 기반 클러스터링                   │
 - Fine-level: 피처 기반 서브-클러스터링                   │
 - Pseudo Feature Loss                                     │
 → 3D 인스턴스 클러스터 형성                               │
      │                                                    │
      ▼                                                    ▼
[Stage 3: 인스턴스-레벨 2D-3D 피처 연계] ←── CLIP 이미지 피처
 - Single-instance map 렌더링
 - IoU + 피처 거리 기반 SAM 마스크 매칭
 - 멀티뷰 CLIP 피처 통합
 → 각 3D Gaussian 포인트에 512-dim CLIP 피처 연계
      │
      ▼
[오픈 보케블러리 텍스트/클릭 쿼리]
 - CLIP 텍스트 피처 추출
 - 코사인 유사도 계산
 - 관련 3D Gaussian 선택 및 렌더링
```

구체적으로: (a) SAM 부울 마스크를 사용하여 3DGS의 3D 일관성 있는 인스턴스 피처를 학습; (b) 코어스-to-파인으로 인스턴스 피처를 이산화하는 2-레벨 코드북 제안; (c) 훈련 없이 2D CLIP 피처를 3D 포인트에 연계하는 인스턴스-레벨 3D-2D 피처 연계 방법.

---

### 2.4 추론 방법

오픈 보케블러리 텍스트 쿼리가 주어지면, CLIP을 이용해 텍스트 피처를 추출하고 해당 피처와 각 Gaussian의 언어 피처 간의 **코사인 유사도(cosine similarity)**를 계산한다. 그런 다음 관련성이 높은 3D 포인트들을 선택하고 3DGS 파이프라인을 사용해 멀티뷰 이미지로 렌더링한다.

수식으로 표현하면:

$$s_i = \frac{\mathbf{f}_{text} \cdot \mathbf{f}_{3D,i}}{\|\mathbf{f}_{text}\| \cdot \|\mathbf{f}_{3D,i}\|}$$

여기서 $s_i$는 $i$번째 Gaussian 포인트의 텍스트 쿼리에 대한 유사도 점수이다.

---

### 2.5 성능 향상

오픈 보케블러리 기반 3D 오브젝트 선택, 3D 포인트 클라우드 이해, 클릭 기반 3D 오브젝트 선택, 어블레이션 연구를 포함한 광범위한 실험을 통해 제안 방법의 효과가 입증되었다.

주요 성능 결과:

OpenGaussian은 텍스트 쿼리에 해당하는 3D 오브젝트를 정확히 식별하는 데 있어 LangSplat 및 LEGaussians보다 뛰어난 성능을 보인다.

텍스트 기반 3D 공간에서의 오브젝트 선택에서 기존 방법들을 능가하는 정확도를 달성하고, ScanNet 데이터셋에서의 의미적 분할 태스크에서 기존 방법들 대비 현저히 향상된 성능을 보이며, 2D 이미지의 단일 클릭만으로도 완전한 3D 오브젝트를 분할하는 인터랙티비티를 보여준다.

OpenGaussian은 각 Gaussian에 **512차원 CLIP 피처**를 연계하며, 이를 통해 LangSplat 및 LEGaussians와 달리 차원 축소 없이 고차원의 풍부한 언어 피처를 유지한다.

**평가 데이터셋:**
ScanNet에서 19, 15, 10개 카테고리에 대한 텍스트 가이드 분할 성능을 평가한다.

---

### 2.6 한계점

논문은 오클루전(occlusion), 센서 노이즈 등 실제 환경에서 Gaussian 표현이 어떻게 동작하는지에 대한 상세한 분석을 제공하지 않으며, 이러한 더 현실적인 시나리오에서의 접근법의 강건성 이해를 위한 추가 연구가 필요하다.

추가로 다음과 같은 한계들을 구조적으로 정리할 수 있다:

1. **씬별 학습(per-scene training) 의존성**: 3DGS 기반이므로 새로운 장면마다 별도의 최적화가 필요하여 일반화에 제약이 있음
2. **정적 장면 가정**: 동적 오브젝트나 시간에 따른 변화에 대한 대응이 제한적
3. **SAM 품질 의존성**: 인스턴스 피처 학습이 SAM 마스크의 품질에 직접적으로 영향을 받음
4. **계산 비용**: 코드 최적화 및 더 적합한 하이퍼파라미터 사용으로 최신 평가 지표는 논문에 보고된 것보다 높을 수 있으나, 훈련 비용 자체는 여전히 상당함

---

## 3. 모델의 일반화 성능 향상 가능성

OpenGaussian의 일반화 관련 설계 및 가능성을 심층 분석한다.

### 3.1 현재 구조에서 일반화를 지원하는 요소들

**① 훈련 불필요한(training-free) CLIP 연계**

인스턴스-레벨 3D-2D 피처 연계 방법은 **훈련 없이** 2D CLIP 피처를 3D 포인트에 연계할 수 있다.

이는 다양한 도메인의 새로운 장면에 CLIP의 제로-샷(zero-shot) 능력을 그대로 전달할 수 있음을 의미한다.

**② 고차원 CLIP 피처 유지**

OpenGaussian은 오브젝트 간 및 오브젝트 내에서 모두 3D 포인트 수준의 **독특하고 일관된 피처**를 학습하며, 고차원의 손실 없는 CLIP 피처를 3D Gaussian 포인트와 연계하여 오픈 보케블러리 3D 장면 이해를 가능하게 한다.

**③ 크로스-프레임 연계 불필요**

SAM 부울 마스크를 뷰-독립적(view-independent)으로 활용하여 3DGS의 3D 일관성 인스턴스 피처를 학습한다.

이는 카메라 궤적이나 뷰 분포에 덜 민감하게 만들어 다양한 촬영 조건에 대한 강건성을 향상시킨다.

**④ 클릭 기반 인터랙션으로 도메인 확장성**

단일 2D 이미지 클릭만으로 완전한 3D 오브젝트를 분할하는 기능은 레이블이 없는 새로운 도메인에서도 직관적인 상호작용을 통해 작동할 수 있음을 시사한다.

### 3.2 일반화 성능을 더 높일 수 있는 연구 방향

| 방향 | 설명 |
|------|------|
| **Generalizable 3DGS** | PixelSplat, MVSplat 등과 결합하여 단일 패스 추론으로 일반화 |
| **Foundation Model 통합** | DINOv2, GPT-4V 등 대형 비전-언어 모델과 연계 |
| **동적 장면 확장** | 4D-GS, SpacetimeGaussian 등과 결합하여 시간축 일반화 |
| **도메인 적응** | 실내→실외, 합성→실제 등 도메인 간 피처 정렬 |
| **멀티모달 쿼리** | 텍스트뿐 아니라 이미지, 오디오 기반 쿼리 지원 확장 |

---

## 4. 관련 최신 연구 비교 분석 (2020년 이후)

| 방법 | 연도 | 베이스 | 언어 임베딩 | 이해 수준 | 주요 특징 |
|------|------|--------|-------------|-----------|-----------|
| **NeRF-based** | | | | | |
| LERF | 2023 | NeRF | CLIP | 3D 볼륨 | 언어 필드 직접 학습 |
| OV-NeRF | 2024 | NeRF | CLIP+VLM | 3D 볼륨 | 오픈 보케블러리 시맨틱 |
| **3DGS-based** | | | | | |
| LangSplat | 2024 | 3DGS | CLIP (저차원) | **2D 픽셀** | 씬-와이즈 오토인코더 |
| LEGaussians | 2024 | 3DGS | CLIP | **2D 픽셀** | 실시간 렌더링 |
| Feature 3DGS | 2024 | 3DGS | CLIP/DINO | **2D 픽셀** | 피처 필드 증류 |
| Semantic Gaussians | 2024 | 3DGS | 2D 사전학습 | 3D 점 | 추가 훈련 불필요 |
| **OpenGaussian (본 논문)** | **2024** | **3DGS** | **CLIP (고차원)** | **3D 포인트** | **2단계 코드북 + IoU 연계** |
| **Point Cloud-based** | | | | | |
| OpenScene | 2023 | 포인트 클라우드 | CLIP | 3D 포인트 | 태스크-비독립 학습 |
| OpenMask3D | 2023 | 포인트 클라우드 | CLIP | 3D 인스턴스 | 멀티뷰 CLIP 집약 |

OpenMask3D는 멀티뷰 CLIP 임베딩 융합을 통해 마스크별 피처를 집약하는 제로-샷 오픈 보케블러리 3D 인스턴스 분할 접근법으로, 특히 롱-테일 분포에서 다른 오픈 보케블러리 방법들보다 뛰어난 성능을 보인다.

OpenScene은 3D 장면 포인트에 대해 CLIP 피처 공간에 텍스트 및 이미지 픽셀과 공동으로 임베딩된 밀집 피처를 예측하는 모델을 제안하며, 태스크-비독립(task-agnostic) 학습과 오픈 보케블러리 쿼리를 가능하게 한다.

**OpenGaussian의 차별점**: 기존 3DGS 기반 방법(LangSplat, LEGaussians)이 2D 픽셀 레벨에 머물고 차원 축소로 인해 피처 품질이 저하되는 반면, OpenGaussian은 512차원 CLIP 피처를 그대로 유지하면서 직접 3D 포인트에 연계한다는 점에서 근본적인 패러다임 전환을 제시한다.

---

## 5. 앞으로의 연구에 미치는 영향 및 고려할 점

### 5.1 연구에 미치는 영향

**① 3D 장면 이해의 패러다임 전환**

기존의 2D 픽셀-레벨 이해와 달리 언어-임베딩 3D Gaussian이 2D 피처 맵으로 렌더링되는 방식에서, **3D 공간에서 직접 텍스트 쿼리에 관련된 Gaussian을 선택하는** 3D 포인트-레벨 이해로의 전환을 명확히 제시한다. 이는 후속 연구의 기본 벤치마크가 될 것이다.

**② 로보틱스 및 구현 AI에 대한 기여**

OpenGaussian 기반으로 재구성된 장면에서 오브젝트의 제거, 삽입, 색상 수정 등 **장면 편집 기능**을 구현할 수 있다. 이는 로봇의 오브젝트 조작, AR/VR 장면 편집 등 실용적인 응용으로 직결된다.

저자들이 시사하듯이, 이 분야의 지속적인 연구는 로보틱스, 증강 현실, 자율 주행 분야에서 중요한 발전을 이끌 것이다.

**③ 오픈 보케블러리의 lossless한 피처 전달 패러다임**

기존 방법들의 차원 축소에 의한 피처 손실 문제를 해결하여, 추가 훈련 없이 512차원 원본 CLIP 피처를 직접 활용하는 패러다임을 확립한다.

---

### 5.2 앞으로 연구 시 고려할 점

| 고려 사항 | 구체적 내용 |
|-----------|-------------|
| **동적 장면 확장** | 현재 정적 장면에 제한됨 → 4D-GS 등과 통합 필요 |
| **씬별 최적화 탈피** | 새 장면마다 재훈련 필요 → Generalizable NeRF/GS와 통합하여 진정한 일반화 달성 |
| **SAM 품질 의존성** | 마스크 품질이 피처 학습에 직접 영향 → 더 강건한 기반 모델(SAM 2 등) 활용 고려 |
| **계산 효율성** | 대규모 장면에서의 메모리/시간 비용 최적화 |
| **Part-level 이해** | 오브젝트 수준을 넘어 파츠(parts) 수준의 세밀한 이해로 확장 |
| **멀티모달 쿼리** | 텍스트 외 이미지, 오디오, 제스처 기반 쿼리 지원 |
| **실세계 강건성** | 오클루전, 센서 노이즈 등 실제 환경 도전 과제에 대한 강건성 분석 및 개선 |
| **평가 표준화** | 평가가 일관된 설정을 따르나, 비교 방법들의 공식 지표와 일치하지 않을 수 있음 → 통일된 벤치마크 수립 필요 |

InstanceGaussian과 같은 후속 연구는 외형과 의미적 피처를 공동으로 학습하며 인스턴스를 적응적으로 집약하는 방향으로 발전하여, 카테고리-비독립 오픈 보케블러리 3D 포인트-레벨 분할에서 최첨단 성능을 달성하고 있다. 이는 OpenGaussian이 제시한 방향이 후속 연구에 실질적으로 영향을 미치고 있음을 보여준다.

---

## 📚 참고 자료 (출처)

1. **arXiv 원문** (v1: 2024.06.04, v2: 2024.12.06): https://arxiv.org/abs/2406.02058
2. **NeurIPS 2024 공식 논문**: Wu, Y. et al., "OpenGaussian: Towards Point-Level 3D Gaussian-based Open Vocabulary Understanding," *Proceedings of the Advances in Neural Information Processing Systems (NeurIPS)*, pp.19114–19138, 2024. https://proceedings.neurips.cc/paper_files/paper/2024/hash/21f7b745f73ce0d1f9bcea7f40b1388e-Abstract-Conference.html
3. **NeurIPS 2024 논문 PDF**: https://proceedings.neurips.cc/paper_files/paper/2024/file/21f7b745f73ce0d1f9bcea7f40b1388e-Paper-Conference.pdf
4. **프로젝트 페이지**: https://3d-aigc.github.io/OpenGaussian/
5. **GitHub 공식 코드**: https://github.com/yanmin-wu/OpenGaussian
6. **OpenReview (NeurIPS 2024 리뷰)**: https://openreview.net/forum?id=3NAEowLh7Q
7. **Semantic Scholar**: https://www.semanticscholar.org/paper/OpenGaussian:-Towards-Point-Level-3D-Gaussian-based-Wu-Meng/ed1643e6ac201a38767ae24ee07aeda6837068ea
8. **ACM DL (NeurIPS 2024)**: https://dl.acm.org/doi/10.5555/3737916.3738520
9. **Moonlight Literature Review**: https://www.themoonlight.io/en/review/opengaussian-towards-point-level-3d-gaussian-based-open-vocabulary-understanding
10. **VILLA 연구실 페이지**: https://villa.jianzhang.tech/publication/100092/

> ⚠️ **정확도 참고**: 핵심 손실 함수의 수식($\mathcal{L}_s$, $\mathcal{L}_c$)은 Moonlight 리뷰 및 공식 HTML 논문에서 인용된 것으로, 정식 PDF 논문과의 세부 표기가 다를 수 있습니다. 정확한 수식 확인을 위해서는 반드시 공식 NeurIPS 2024 PDF 원문을 참조하시길 권장합니다.
