
# Material Magic Wand: Material-Aware Grouping of 3D Parts in Untextured Meshes

> **논문 정보**
> - **저자**: Umangi Jain (University of Toronto), Vladimir Kim, Matheus Gadelha, Igor Gilitschenski (Equal advising), Zhiqin Chen (Adobe Research)
> - **발표**: CVPR 2026
> - **arXiv**: [2603.17370](https://arxiv.org/abs/2603.17370) (2026년 3월 18일)
> - **프로젝트 페이지**: https://umangi-jain.github.io/material-magic-wand

---

## 1. 핵심 주장 및 주요 기여 요약

### 🎯 핵심 주장

이 논문은 **텍스처가 없는(untextured) 3D 메쉬에서 재질 인식 기반 파트 그룹화(material-aware part grouping)** 라는 새로운 문제를 제안합니다.

Photoshop의 Magic Wand 도구처럼, 사용자가 하나의 파트를 클릭하면 동일한 재질을 공유하는 파트들을 자동으로 선택할 수 있게 합니다.

### 📌 주요 기여

| 기여 항목 | 설명 |
|---|---|
| 새로운 문제 정의 | Untextured 3D 메쉬에서의 material-aware part grouping 문제 최초 제안 |
| 모델 제안 | Material-aware embedding을 생성하는 Part Encoder |
| 손실 함수 | Supervised Contrastive Loss 기반 학습 |
| 벤치마크 | 100개 형상, 241개 파트 쿼리로 구성된 큐레이션 데이터셋 |
| 인터랙티브 도구 | 3D 에셋 제작 워크플로우에 직접 통합 가능한 실용적 도구 |

기존에는 사전 분할된 파트들을 재질 일관성 기준으로 그룹화하는 방법이나 벤치마크가 존재하지 않았습니다.

---

## 2. 상세 설명

### 2-1. 해결하고자 하는 문제

솔방울의 비늘이나 건물의 창문과 같이 많은 실제 형상은 동일한 재질을 공유하지만 기하학적 변형을 가진 반복 구조를 포함합니다. 이러한 메쉬에 재질을 할당할 때 반복된 파트들은 일일이 수동으로 식별하고 선택해야 하는 번거롭고 시간 소모적인 작업이 필요합니다.

이 논문은 이미 세분화(fine-grained)된 파트로 분해된 untextured 3D 메쉬와 쿼리 파트를 입력으로 받아, 동일한 재질을 공유할 가능성이 높은 다른 파트들을 검색하는 것을 목표로 합니다. 기하학적 변형 하에서 재질 일관성 파트를 그룹화하는 문제는 아직까지 거의 다루어지지 않았습니다.

재질 분할(material segmentation) 기존 방법들은 텍스처에 의존하여 재질 특성의 힌트를 얻고, 이미지 기반 파운데이션 세그멘테이션 모델을 활용하기 때문에 해상도 한계로 인해 작은 파트 처리에 어려움을 겪습니다.

---

### 2-2. 제안하는 방법 (수식 포함)

#### 🔑 핵심 아이디어: Material-Aware Embedding

핵심 아이디어는 각 파트를 재질 유사도를 인코딩하는 임베딩 공간으로 투영하여, 쿼리 파트의 임베딩과 가까운 임베딩을 검색함으로써 파트 그룹화를 수행하는 것입니다. 고전적 기하 서술자(geometric descriptor)나 DINO, SigLIP과 같은 최신 이미지 임베딩 모델들은 재질 이해 능력이 부족합니다. 따라서 대규모 3D 형상 컬렉션에서 재질 인식 임베딩을 학습하는 Part Encoder 모델을 설계합니다.

#### 📐 Supervised Contrastive Loss

Part Encoder는 로컬 기하학(local geometry)과 글로벌 컨텍스트(global context)를 모두 고려하여 각 3D 파트에 대한 재질 인식 임베딩을 생성합니다. 모델은 재질이 일관된 파트들의 임베딩을 가깝게, 다른 재질의 파트들의 임베딩을 멀리 배치하는 Supervised Contrastive Loss로 학습합니다.

Supervised Contrastive Loss의 일반적 형식은 다음과 같습니다:

$$\mathcal{L}_{\text{SupCon}} = \sum_{i \in I} \frac{-1}{|P(i)|} \sum_{p \in P(i)} \log \frac{\exp(\mathbf{z}_i \cdot \mathbf{z}_p / \tau)}{\sum_{a \in A(i)} \exp(\mathbf{z}_i \cdot \mathbf{z}_a / \tau)}$$

여기서:
- $i$: 앵커(anchor) 파트 인덱스
- $P(i)$: 배치 내에서 $i$와 동일 재질을 공유하는 파트(positive) 집합
- $A(i)$: $i$를 제외한 모든 파트(anchor 제외) 집합
- $\mathbf{z}_i$: 파트 $i$의 정규화된 임베딩 벡터
- $\tau$: temperature 하이퍼파라미터

파트 그룹화는 다음과 같이 임베딩 유사도 기반 검색으로 수행됩니다:

$$\hat{G}(q) = \{p_j \mid \text{sim}(\mathbf{z}_q, \mathbf{z}_{p_j}) \geq \lambda\}$$

여기서:
- $q$: 쿼리 파트
- $\mathbf{z}_q$: 쿼리 파트의 임베딩
- $\lambda$: 검색 임계값(threshold)
- $\text{sim}(\cdot, \cdot)$: 코사인 유사도

임계값 $\lambda$는 검색 집합을 엄격하게 일치하는 파트에서 더 넓은 구조적으로 관련된 영역으로 점진적으로 확장합니다. 예를 들어 침대 예시에서, 검색은 동일한 베개에서 기하학적으로 유사한 베개로, 그리고 모든 베개로 확장됩니다.

---

### 2-3. 모델 구조

인코더는 각 3D 파트에 대해 여러 이미지를 입력으로 받으며, 이 이미지들은 로컬 기하학과 글로벌 컨텍스트를 모두 포착하기 위한 특정 구성(configuration)으로 렌더링됩니다.

모델은 **DINOv2 small backbone**을 기반으로 파인튜닝(supervised fine-tuning)하여 학습됩니다.

모델 구조 요약:

```
[3D Part (Untextured Mesh)]
        ↓
[Multi-view Rendering (local + global context)]
        ↓
[DINOv2-small Backbone (fine-tuned)]
        ↓
[Material-Aware Embedding z ∈ R^d]
        ↓
[Cosine Similarity Retrieval (threshold λ)]
        ↓
[Part Grouping Result]
```

이 Part Encoder는 로컬 기하학과 글로벌 컨텍스트를 모두 반영하여 각 3D 파트에 대한 재질 인식 임베딩을 생성합니다.

---

### 2-4. 성능 향상

Material Magic Wand는 100개 메쉬, 241개 파트 레벨 쿼리로 구성된 큐레이션 벤치마크에서 기하학 기반 히스토그램 매칭, SigLIP, DINO와 같은 비전 파운데이션 모델 임베딩, PartField와 비교 평가되었으며, 모든 검색 및 그룹화 지표에서 최고 성능을 달성했습니다.

그룹화 F1 스코어에서 **+16.6%** 의 성능 향상을 달성하였으며, Precision-Recall 곡선에서 전체 Recall 범위에 걸쳐 지속적으로 높은 Precision을 유지합니다.

정성적으로도 기하학적으로나 맥락적으로 쿼리와 유사한 구성 요소를 검색하는 반면, 기준선(baseline)들은 구조적으로 관련된 파트를 누락하거나 시각적으로는 유사하지만 맥락적으로 부정확한 파트를 포함하는 경우가 많았습니다.

기하학 기반 Histogram Matching은 Recall이 증가할수록 Precision이 급격히 하락하여 거의 중복 파트에만 효과적입니다. PartField의 낮은 성능은 재질 일관 임베딩 학습이 아닌 계층적 파트 분할이라는 다른 학습 목적에서 기인합니다.

---

### 2-5. 한계

재질 인식 그룹화에서는 장면의 다양한 해석에 따라 파트들을 다양한 방식으로 그룹화할 수 있는 모호성(ambiguity)이 자연스럽게 발생합니다. 평가에서 이러한 모호성으로 인한 노이즈를 줄이기 위해 100개 형상의 벤치마크를 제안하고 241개의 파트 그룹을 정의합니다.

주요 한계를 정리하면:

1. **벤치마크 규모**: 100개 형상, 241개 쿼리로 구성되어 있어 다양한 도메인에 대한 일반화 평가가 제한적
2. **선처리 의존성**: 메쉬가 이미 fine-grained 파트로 분해되어 있다는 전제를 가지므로, 파트 분할 자체는 별도로 필요
3. **모호성 문제**: 재질 할당의 주관성(ambiguity)으로 인해 정답 ground truth 설정이 어렵고, 다양한 해석이 존재할 수 있음
4. **학습 데이터 도메인**: Objaverse 기반 학습으로 인한 도메인 편향 가능성

---

## 3. 모델의 일반화 성능 향상 가능성

Objaverse를 넘어선 일반화 평가를 위해, **TexVerse** 데이터셋의 메쉬에서 정성적 테스트를 수행했습니다. 테스트에는 아티스트가 제작한(스캔되지 않은) 메쉬 중 연결 요소로 분해 가능하고, 동일 재질을 공유하지만 동일하지 않은 파트를 포함하는 메쉬가 사용되었습니다.

Objaverse 테스트 벤치와 유사한 경향을 보였습니다. 기준 모델들은 집의 동일하지 않은 창문 프레임, 보트의 수직 마스트, 계단 아래층 발판, 또는 지붕 통나무 등 유효한 인스턴스를 놓치거나, 집의 벽돌, 보트의 밧줄, 계단 옆판 또는 오두막의 지붕처럼 관련 없는 구조를 검색하는 경우가 많았습니다. 반면 제안된 방법은 기하학적·구조적 변형에 걸쳐 재질 일관 파트를 더 신뢰성 있게 검색합니다.

Material Magic Wand는 메쉬 내의 기하학적으로나 맥락적으로 유사한 구성 요소를 연결하는 방법을 학습하는 재질 인식 파트 그룹화 프레임워크로, 대규모 데이터에 대한 대조 학습(contrastive training)을 통해 강건한 검색 및 그룹화가 가능합니다.

일반화 성능 향상을 위한 주요 메커니즘:

| 요소 | 일반화에 기여하는 방식 |
|---|---|
| **DINOv2 Backbone** | 대규모 이미지 사전학습으로 강력한 시각적 특징 추출 |
| **Multi-view Rendering** | 단일 시점이 아닌 다중 뷰를 활용하여 3D 구조 이해 향상 |
| **Global Context 인코딩** | 단순 로컬 형상이 아닌 전체 메쉬 맥락을 반영 |
| **Contrastive Learning** | 재질 레이블이 아닌 유사도 기반 학습으로 새로운 재질에도 적용 가능 |
| **다중 쿼리 지원** | 추가 쿼리 파트는 더 완전한 원하는 구성 요소 집합을 포착하여 검색을 개선합니다. |

---

## 4. 향후 연구에 미치는 영향 및 고려할 점

### 🔭 향후 연구에 미치는 영향

**① 새로운 연구 과제 정립**

기하학적 변형 하에서 재질 일관 파트 그룹화 문제는 기존에 거의 다루어지지 않은 문제입니다. 이로써 3D 자산 제작 파이프라인에서의 재질 인식 파트 그룹화라는 새로운 연구 방향을 개척했습니다.

**② 3D 콘텐츠 제작 워크플로우 혁신**

이 도구는 단 하나의 인터랙션으로 수백 개의 파트에 재질을 할당할 수 있게 하여, 디자이너의 재질 할당 과정을 크게 가속화할 수 있습니다.

**③ 관련 분야에의 확장 가능성**

기존 3D 파트 분할 방법들이 기하학을 의미론적 구성요소로 분할하는 것을 목표로 하는 반면, 이 연구는 초기 분해가 아닌 상위 수준의 그룹화를 다루는 새로운 연구 방향을 제시합니다.

### ⚠️ 앞으로 연구 시 고려할 점

1. **파트 분할 의존성 해소**: 현재 모델은 fine-grained part segmentation이 이미 완료된 메쉬를 전제로 하므로, 파트 분할과 재질 그룹화를 엔드-투-엔드(end-to-end)로 통합하는 연구가 필요

2. **텍스처 정보와의 융합**: SAMa와 같은 관련 연구처럼 2D 이미지 특징과 3D 기하학 정보를 결합하여 성능을 더 향상시킬 수 있는 멀티모달 접근이 가능

3. **대규모 벤치마크 확장**: 현재 100개 형상, 241개 쿼리로 구성된 벤치마크는 규모가 작아 더 다양하고 대규모의 벤치마크 구축이 필요

4. **모호성 처리**: 동일한 파트에 여러 재질 해석이 가능한 경우를 처리하기 위한 불확실성 모델링(uncertainty modeling) 도입 고려

5. **실시간 응용**: 고도로 반사성이 있거나 투명한 재질에 대한 성능 저하 문제 개선 및 실시간 애플리케이션을 위한 처리 속도 향상이 필요

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

| 연구 | 접근 방식 | 입력 | 주요 특징 | 한계 |
|---|---|---|---|---|
| **Material Magic Wand** (2026, CVPR) | Contrastive Learning + Multi-view Rendering | Untextured 3D Mesh | 재질 인식 파트 그룹화, 인터랙티브 선택 | 파트 분할 전처리 필요 |
| **SAMa** (2024, arXiv) | SAM 기반 3D Material Selection | Textured 3D (Mesh, NeRF, 3DGS) | 멀티뷰 비디오 객체 선택 모델 응용 | 반사성·투명 재질 취약 |
| **PartField** (2025) | Contrastive Learning + Clustering | 3D Mesh | 계층적 파트 분할 특징 필드 학습 | 재질 일관성 이해 부재 |
| **PartSAM** (2025) | Native 3D Training | 3D Point Cloud/Mesh | SAM 기반 프롬프터블 파트 분할 | 내부 구조 이해 제한 |

비교 실험에서 DINO와 PartField는 임계값이 증가할수록 관련 없는 구성 요소를 검색하기 시작합니다. PartField는 또한 매우 낮은 임계값에서도 대부분의 파트를 그룹화하는 등 세밀한 제어가 부족합니다.

PartField는 특징 공간에서의 클러스터링을 통해 분할 결과를 얻음으로써 인상적인 성능을 달성했지만, 클러스터링 기반 분할 방식은 사용자 중심의 제어 가능성이 부족하고, 클러스터 수를 신중하게 튜닝하지 않으면 파편화된 파트 결과를 낳는 경우가 많습니다.

---

## 📚 참고 자료

1. **arXiv 논문 원문**: Jain et al., "Material Magic Wand: Material-Aware Grouping of 3D Parts in Untextured Meshes", arXiv:2603.17370, March 2026. https://arxiv.org/abs/2603.17370
2. **arXiv HTML 전문**: https://arxiv.org/html/2603.17370
3. **프로젝트 페이지**: https://umangi-jain.github.io/material-magic-wand/
4. **논문 리뷰 (CVPR 2026)**: https://kimjy99.github.io/논문리뷰/material-magic-wand/
5. **관련 연구 - SAMa**: "SAMa: Material-aware 3D Selection and Segmentation", arXiv:2411.19322. https://arxiv.org/pdf/2411.19322
6. **관련 연구 - PartSAM**: "PartSAM: A Scalable Promptable Part Segmentation Model Trained on Native 3D Data", arXiv:2509.21965. https://arxiv.org/pdf/2509.21965
7. **관련 연구 - P3-SAM**: "P3-SAM: Native 3D Part Segmentation", arXiv:2509.06784. https://arxiv.org/html/2509.06784v3

> ⚠️ **정확도 안내**: 본 답변은 공개된 arXiv 논문 원문, HTML 전문, 프로젝트 페이지 등을 기반으로 작성되었습니다. 수식(특히 Supervised Contrastive Loss)은 논문에서 직접 명시된 공식 표기가 아닌 일반적인 Supervised Contrastive Learning 공식(Khosla et al., 2020)을 참조하여 표기한 것이며, 논문의 세부 구현 수식과 완전히 동일하지 않을 수 있습니다. 정확한 수식과 세부 구현은 논문 원문을 직접 확인하시기 바랍니다.
