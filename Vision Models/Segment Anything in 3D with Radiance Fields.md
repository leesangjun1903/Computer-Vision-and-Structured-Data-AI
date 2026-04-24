
# Segment Anything in 3D with Radiance Fields

---

## 1. 핵심 주장 및 주요 기여 요약

이 논문은 **SA3D**라는 프레임워크를 제안하며, 2D SAM(Segment Anything Model)을 Radiance Field를 구조적 사전 정보(structural prior)로 활용하여 3D 객체 분할로 일반화한다.

핵심 아이디어는 3D 지각 능력이 없는 2D 기반 모델(SAM)에 3D 표현 모델을 통해 3D 지각 능력을 부여하는 것이며, 3D 기반 모델을 처음부터 새로 구축할 필요가 없다는 것이다.

### 주요 기여 요약

| 기여 | 설명 |
|------|------|
| **SA3D 프레임워크** | SAM과 NeRF를 통합한 최초의 one-shot 3D 분할 프레임워크 |
| **Mask Inverse Rendering** | SAM의 2D 마스크를 NeRF 밀도 분포 기반으로 3D 공간에 투영 |
| **Cross-view Self-Prompting** | 자동 프롬프트 생성으로 다시 SAM 입력을 생성하는 반복 구조 |
| **범용성** | 다양한 Radiance Field에 추가 수정 없이 적용 가능 |
| **효율성** | 초 단위의 3D 분할 달성 (최신 버전 기준) |

이 논문은 NeurIPS 2023 및 IJCV 2025에 게재되었다.

---

## 2. 문제 정의, 제안 방법, 모델 구조, 성능 및 한계

### 2.1 해결하고자 하는 문제

3D 분할은 2D 분할에 비해 레이블 데이터의 희소성과 높은 계산 복잡성 때문에 SAM과 유사한 분할 기반 모델을 설계하기 어렵다.

3D 데이터의 접근성 부족과 수집·주석의 높은 비용으로 인해, SAM을 3D로 확장하는 것은 도전적이지만 가치 있는 연구 방향이다.

3D에서의 비용이 많이 드는 데이터 수집·주석 절차를 복제하는 대신, Radiance Field를 다시 뷰 2D 이미지를 3D 공간에 연결하는 저렴하고 바로 사용할 수 있는 사전 정보로 활용하는 효율적인 해결책을 설계한다.

---

### 2.2 제안 방법 및 수식

#### 전체 파이프라인

주어진 NeRF 모델에 대해 SA3D는 단일 렌더링 뷰에서의 one-shot 수동 프롬프팅만으로 임의의 목표 객체의 3D 분할 결과를 얻을 수 있다.

SA3D는 다양한 뷰에서 두 단계를 반복적으로 수행하여 복셀 그리드로 구성된 3D 마스크를 완성한다. 각 라운드의 첫 번째 단계는 **Mask Inverse Rendering**으로, SAM이 얻은 이전 2D 분할 마스크를 NeRF의 밀도 기반 역 렌더링을 통해 3D 마스크에 투영한다. 두 번째 단계는 **Cross-view Self-Prompting**으로, NeRF를 사용해 3D 마스크를 다른 뷰에서 2D로 렌더링하고, 렌더링된 마스크에서 자동으로 포인트 프롬프트를 생성하여 SAM에 다시 입력한다. 이 과정은 필요한 모든 뷰가 샘플링될 때까지 반복된다.

---

#### NeRF 기초 수식

NeRF는 아래 볼륨 렌더링(Volume Rendering) 방정식을 통해 2D 이미지를 렌더링한다:

$$
\hat{C}(\mathbf{r}) = \int_{t_n}^{t_f} T(t)\, \sigma(\mathbf{r}(t))\, \mathbf{c}(\mathbf{r}(t), \mathbf{d})\, dt
$$

여기서:
- $T(t) = \exp\left(-\int_{t_n}^{t} \sigma(\mathbf{r}(s))\, ds\right)$ : 누적 투과율(Transmittance)
- $\sigma(\mathbf{r}(t))$ : 위치 $\mathbf{r}(t)$에서의 밀도(density)
- $\mathbf{c}(\mathbf{r}(t), \mathbf{d})$ : 위치 및 시선 방향 $\mathbf{d}$에 따른 색상
- $\mathbf{r}(t) = \mathbf{o} + t\mathbf{d}$ : 카메라 원점 $\mathbf{o}$에서 방향 $\mathbf{d}$로 진행하는 광선

---

#### Mask Inverse Rendering 수식

SAM이 생성한 마스크 $M_\text{SAM}(\mathbf{r}) = 1$일 때, $\mathbb{1}_V(\mathbf{r}(t))$는 3D 마스크 그리드의 꼭짓점 값을 삼선형 보간(trilinear interpolation)하여 계산된다. Mask Inverse Rendering의 목표는 이 값 $V(\mathbf{r}(t))$를 최대화하는 것이다.

3D 마스크 렌더링 함수:

$$
M(\mathbf{r}) = \int_{t_n}^{t_f} T(t)\, \sigma(\mathbf{r}(t))\, \mathbb{1}_V(\mathbf{r}(t))\, dt
$$

마스크 투영 손실 (Mask Projection Loss):

$$
\mathcal{L}_\text{proj} = -M_\text{SAM}(\mathbf{r}) \cdot M(\mathbf{r})
$$

이는 경사 하강법(gradient descent)으로 최적화되며, 실제로는 NeRF와 SAM의 분할 결과가 정확하지 않을 수 있으므로 음의 정제 항(negative refinement term)을 손실에 추가하여 3D 마스크 그리드를 최적화한다.

전체 손실 함수:

$$
\mathcal{L} = -M_\text{SAM}(\mathbf{r}) \cdot M(\mathbf{r}) + \lambda \cdot M_\text{neg}(\mathbf{r}) \cdot M(\mathbf{r})
$$

여기서 $\lambda$는 음의 정제 항의 가중치이며, $M_\text{neg}$는 배경(음성) 마스크이다.

---

#### Cross-view Self-Prompting

Cross-view self-prompting은 새로운 뷰에서 렌더링된 2D 마스크로부터 신뢰할 수 있는 프롬프트를 자동으로 추출하여 SAM 디코더의 입력으로 사용한다. 이 교번(alternated) 과정은 정확한 3D 마스크를 얻을 때까지 반복적으로 실행된다.

---

### 2.3 모델 구조

```
[사용자 입력: 단일 뷰 포인트 프롬프트]
          ↓
    [SAM 인코더/디코더]
          ↓ 2D 마스크 생성
  [Mask Inverse Rendering]
    - NeRF 밀도 분포 활용
    - 3D Voxel Grid 업데이트
          ↓
  [Cross-view Self-Prompting]
    - 새 뷰에서 2D 마스크 렌더링
    - 자동 포인트 프롬프트 추출
          ↓ SAM에 재입력
    ↑────────────────┘ 반복
          ↓
   [최종 3D 마스크 출력]
```

SA3D는 추가적인 재설계 없이도 다양한 Radiance Field에 효과적으로 적응할 수 있다.

---

### 2.4 성능

SA3D는 MVSeg에 비해 대부분의 장면에서 성능을 초과하며, 특히 Truck 장면에서 +5.6 mIoU, Lego 장면에서 +17.3 mIoU의 향상을 보인다. "단일 뷰" 모델과 비교하면 +17.8 mIoU의 큰 성능 향상이 달성된다.

9개 데이터셋에 걸친 광범위한 실험은 방법의 우수성을 입증하며, 제한된 감독 뷰로 경쟁력 있는 분할 품질을 달성한다. 특히 렌더링(추론) 시간을 90% 감소시키면서 평균 mIoU를 최대 3.5%까지 향상시킨다.

2024년 4월에는 3D Gaussian Splatting(3D-GS) 버전의 SA3D가 공개되었으며, 이를 통해 3D 분할을 수 초 내에 달성할 수 있게 되었다.

---

### 2.5 한계

SA3D는 파노픽 분할(panoptic segmentation)에 한계가 있다.

SA3D는 단일 뷰 프롬프팅 전략으로 인해 가려진(occluded) 식물 영역을 놓치는 문제가 있다.

SA3D는 단순한 객체 중심 장면(Kitchen)에서는 잘 작동하지만, 가려짐이 있는 장면(Garden)에서는 단일 뷰 프롬프트가 여러 시점에서 목표 영역을 포착하는 데 실패하는 어려움이 있다.

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 다양한 2D 분할 모델 통합 능력

SAM 외에도 네 가지 다른 프롬프트 기반 2D 분할 모델을 프레임워크에 통합하여 SA3D의 일반화 능력을 검증하였다. NVOS 데이터셋 평가 결과, 다양한 2D 분할 모델에서 SA3D는 안정적인 성능을 보인다. SEEM과 통합했을 경우 SAM보다 다소 낮은 성능을 보이지만, 참조 이미지와 같은 더 많은 종류의 입력 프롬프트를 지원할 수 있다. 이 결과는 SA3D가 다양한 분할 모델과 통합되어 다양한 3D 분할 작업을 지원할 잠재력을 나타낸다.

### 3.2 다양한 Radiance Field와의 호환성

SA3D를 바닐라 NeRF에도 적응시켜 일반화 가능성을 보여주었으며, LLFF 데이터셋에서 시각화 결과를 제시하였다. SA3D는 바닐라 NeRF와 함께 추가적인 수정 없이도 우수한 성능을 보인다.

SA3D with 3D-GS는 전방향(forward-facing) 및 360° 장면 모두에서 최적의 효율성을 달성한다.

### 3.3 2D → 3D 일반화 방법론으로서의 잠재력

SA3D는 NeRF나 다른 3D 구조적 사전 정보를 활용하는 것이 2D 비전 기반 모델을 3D로 끌어올리는 자원 효율적인 방법임을 시사하며, 기반 모델이 자기 프롬프팅(self-prompting) 능력을 가지고 있어야 한다는 조건을 제시한다. 이 방법론은 3D 데이터 수집 비용이 많이 들기 때문에 많은 자원을 절약할 수 있다.

SA3D는 3D에서 무엇이든 분할하는 효율적인 도구를 제공할 뿐만 아니라, 2D 기반 모델을 3D 공간으로 확장하는 일반적인 방법론을 제시한다. 유일한 전제조건은 여러 뷰에 걸쳐 프롬프트 기반 분할을 안정적으로 수행할 수 있는 능력이며, 이것이 미래 2D 기반 모델의 일반적인 속성이 되기를 기대한다.

NeRF는 SAM의 분할 품질을 향상시키며, SA3D는 SAM의 분할 오류를 제거하고 구멍과 가장자리와 같은 세부 사항을 효과적으로 포착할 수 있다. SAM과 같은 2D 인식 모델은 시점에 민감한 경향이 있는 반면, NeRF는 3D 모델링과 보완적인 인식 지원 능력을 제공한다.

---

## 4. 최신 관련 연구 비교 분석 (2020년 이후)

| 논문/방법 | 연도/학회 | 3D 표현 | 핵심 방법 | 특징 |
|-----------|----------|---------|----------|------|
| **NeRF** | 2020 ECCV | Implicit MLP | 볼륨 렌더링 | 3D 장면 표현의 기반 |
| **SA3D** (본 논문) | 2023 NeurIPS | NeRF + Voxel Grid | Mask Inverse Rendering + Cross-view Self-Prompting | One-shot 3D 분할 |
| **SAGA** | 2023→AAAI-25 | 3D-GS | Scale-gated affinity feature, Contrastive Learning | 4ms 실시간 분할 |
| **SAI3D** | 2023 | Point Cloud | Geometric primitives + SAM | Zero-shot 3D 인스턴스 분할 |
| **SAGD** | 2024 | 3D-GS | Gaussian Decomposition | 경계 정밀도 향상 |

SAGA(Segment Any 3D GAussians)는 3D Gaussian Splatting 기반의 고효율 3D 프롬프트 분할 방법으로, 2D 시각 프롬프트가 주어지면 4ms 이내에 해당 3D 목표를 분할할 수 있다.

SA3D와 비교하면, SAGA는 훨씬 짧은 시간 안에 더 높은 품질의 3D 자산을 얻을 수 있다.

SAI3D는 SAM에서 파생된 기하학적 사전 정보와 의미론적 단서를 시너지적으로 활용하는 새로운 제로샷 3D 인스턴스 분할 접근법이다. 3D 장면을 기하학적 프리미티브로 분할하고, 이를 다시 SAM 마스크와 일관된 3D 인스턴스 분할로 병합한다. 계층적 영역 성장 알고리즘과 동적 임계값 메커니즘을 설계하여 세밀한 3D 장면 파싱의 견고성을 향상시킨다.

CoSSegGaussians는 DINO의 의미 특징과 3D 가우시안의 공간 기하를 특징 융합 네트워크와 다중 스케일 집계를 통해 결합하였다. SAGA는 SAM의 2D 분할과 3D 가우시안 포인트 클라우드를 대조 학습을 통한 다중 레벨 특징 임베딩으로 결합하였다. Click-Gaussian은 이중 레벨 특징 필드와 Global Feature Guidance Learning을 사용하여 뷰 간 일관된 3D 가우시안 의미 표현을 구성하였다.

---

## 5. 앞으로의 연구에 미치는 영향과 고려할 점

### 5.1 연구에 미치는 영향

**① 2D → 3D 일반화 패러다임 제시**

이 연구는 비전 기반 모델을 2D에서 3D로 끌어올리는 자원 효율적인 방법론을 제시한다.

**② 3D 분할 연구의 촉매제**

3D 프롬프트 분할은 3D 데이터의 희소성과 높은 주석 비용으로 인해 상대적으로 미탐구 상태였으나, 이를 해결하기 위해 많은 연구들이 Radiance Field를 사용하여 SAM의 2D 분할 능력을 3D로 확장하는 방법을 제안하였으며 주목할 만한 성과를 거두고 있다.

**③ 3D-GS 기반 후속 연구로의 확장**

SAGA는 3D-GS에서 프롬프트 가능한 분할을 다루는 최초의 방법 중 하나로, 그 단순성과 효율성이 이 분야의 미래 발전을 위한 길을 열었다.

---

### 5.2 향후 연구 시 고려할 점

**① 가려짐(Occlusion) 문제 해결**

SA3D는 단일 뷰 프롬프팅 전략으로 인해 가려진 영역을 놓치는 문제가 있어, 멀티뷰 단서를 3D로 융합하는 방법이 필요하다.

**② 파노픽 분할 및 자동화**

향후 연구는 분할 모델의 도움으로 Radiance Field의 학습된 3D 기하를 개선하고, 자동 파노픽 분할(automatic panoptic segmentation)을 지원하는 방향을 포함할 수 있다.

**③ 3D 인식 능력의 기반 모델 통합**

2D 기반 모델에 3D 인식 능력을 강화하는 연구(예: 2D 사전 학습에 3D 인식 손실 주입)가 필요하다.

**④ 동적 장면 및 대규모 장면 처리**

NeRF 방법은 훈련 및 렌더링 중 계산 비용과 높은 데이터 요구량이 있으며, 동적 장면 및 실시간 응용 프로그램에 대한 처리 능력이 제한적이므로 추가적인 발전이 필요하다.

**⑤ 3D-GS에서의 경계 정밀도 문제**

가우시안 분포 간의 부정확한 분리는 인접 객체의 모호성 해소를 복잡하게 만들어, 분할 정밀도를 낮추고 결과 불일치를 증가시키며, 새로운 시점과 스케일에서 모델의 일반화 능력을 제한한다.

**⑥ SAM2와의 통합 가능성**

SAM 2(Segment Anything in Images and Videos)와 같은 최신 기반 모델과의 통합으로 시간적 일관성 및 비디오 장면 분할 성능을 향상시킬 수 있다.

---

## 📚 참고자료 및 출처

| 번호 | 자료 |
|------|------|
| 1 | **arXiv 논문 원문**: Cen et al., "Segment Anything in 3D with Radiance Fields", arXiv:2304.12308, 2023/2024 — https://arxiv.org/abs/2304.12308 |
| 2 | **NeurIPS 2023 공식 게재 버전**: "Segment Anything in 3D with NeRFs" — https://papers.nips.cc/paper_files/paper/2023/file/525d24400247f884c3419b0b7b1c4829-Paper-Conference.pdf |
| 3 | **IJCV 2025 확장 버전**: Springer Nature Link — https://link.springer.com/article/10.1007/s11263-025-02421-7 |
| 4 | **프로젝트 페이지**: https://jumpat.github.io/SA3D/ |
| 5 | **GitHub 코드**: https://github.com/Jumpat/SegmentAnythingin3D |
| 6 | **OpenReview (NeurIPS)**: https://openreview.net/forum?id=2NkGfA66Ne |
| 7 | **ar5iv 렌더링 버전**: https://ar5iv.labs.arxiv.org/html/2304.12308 |
| 8 | **SAGA 논문**: Cen et al., "Segment Any 3D Gaussians", arXiv:2312.00860, AAAI 2025 — https://arxiv.org/abs/2312.00860 |
| 9 | **SAI3D 논문**: "SAI3D: Segment Any Instance in 3D Scenes", arXiv:2312.11557 — https://arxiv.org/html/2312.11557v2 |
| 10 | **DivAS 논문**: "DivAS: Interactive 3D Segmentation of NeRFs via Depth-Weighted Voxel Aggregation", arXiv:2601.04860 — https://arxiv.org/html/2601.04860 |
| 11 | **Semantic Scholar**: https://www.semanticscholar.org/paper/7d61252826c0aea5b2403af4879bf1fb834d60cb |
| 12 | **NeurIPS Virtual Poster**: https://neurips.cc/virtual/2023/poster/72957 |
