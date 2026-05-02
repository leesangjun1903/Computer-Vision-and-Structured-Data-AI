
# Grounding Image Matching in 3D with MASt3R

> **논문 정보**
> - **제목**: Grounding Image Matching in 3D with MASt3R
> - **저자**: Vincent Leroy, Yohann Cabon, Jérôme Revaud (NAVER Labs Europe)
> - **발표**: ECCV 2024
> - **arXiv**: [2406.09756](https://arxiv.org/abs/2406.09756)

---

## 1. 🔑 핵심 주장 및 주요 기여 요약

이미지 매칭은 3D 비전의 핵심 구성요소이다. 그런데 매칭은 본질적으로 카메라 포즈와 장면 기하학과 연결된 3D 문제임에도 불구하고, 통상적으로 2D 문제로 다루어져 왔다.

논문은 세 가지 핵심 기여를 주장한다. 첫째, DUSt3R 프레임워크 위에 구축된 3D-인식 매칭 접근법인 MASt3R을 제안한다. MASt3R은 고도로 정확하고 강인한 매칭을 가능케 하는 로컬 피처 맵을 출력한다.

둘째, 고해상도 이미지에서도 작동 가능한 빠른 매칭 알고리즘과 결합된 **coarse-to-fine 매칭 스킴**을 제안한다.

셋째, 매칭을 수십 배 가속화할 뿐만 아니라 이론적 보장까지 갖추고 결과도 개선하는 **고속 상호(reciprocal) 매칭 스킴**을 도입한다. 광범위한 실험에서 MASt3R은 다수의 매칭 태스크에서 SOTA를 크게 능가하며, 특히 Map-free 로컬라이제이션 데이터셋에서 VCRE AUC 기준 30% (절대적 향상) 개선을 달성한다.

---

## 2. 🔍 상세 분석

### 2.1 해결하고자 하는 문제

DUSt3R은 매칭에 활용될 수 있으나, 시점 변화에는 매우 강인하면서도 정밀도가 상대적으로 떨어진다. 이 결함을 해결하기 위해 두 번째 헤드를 붙여 dense 로컬 피처 맵을 회귀하고, InfoNCE loss로 학습하는 방법을 제안한다.

또한 dense 매칭의 이차 복잡도(quadratic complexity) 문제를 해결해야 했으며, 이는 주의 깊게 처리하지 않으면 하위 응용에서 매우 느려진다.

---

### 2.2 제안 방법 (수식 포함)

#### 📌 배경: DUSt3R의 Pointmap 회귀

DUSt3R 아키텍처에서 두 뷰 $(I_1, I_2)$는 공유 ViT 인코더로 Siamese 방식으로 인코딩된다. 결과 토큰 표현 $F_1$과 $F_2$는 cross-attention으로 정보를 교환하는 두 Transformer 디코더에 전달된다. 최종적으로 두 개의 회귀 헤드가 두 개의 포인트맵과 신뢰도 맵을 출력하며, 두 포인트맵은 첫 번째 이미지 $I_1$의 동일한 좌표 프레임으로 표현된다.

DUSt3R의 **신뢰도 기반 회귀 손실(Confidence-Aware Regression Loss)**은 다음과 같이 정의된다:

$$\mathcal{L}_{\text{conf}} = \sum_{i} C_i \cdot \| X_i^{\text{pred}} - X_i^{\text{gt}} \|_2 - \alpha \log C_i$$

여기서 $C_i$는 픽셀 $i$의 예측 신뢰도, $X_i^{\text{pred}}$는 예측된 3D 포인트, $X_i^{\text{gt}}$는 정답 3D 포인트, $\alpha$는 균형 파라미터이다.

---

#### 📌 MASt3R의 핵심: Matching Head + InfoNCE Loss

DUSt3R이 스케일 불변 회귀 손실을 사용하는 것과 달리, MASt3R은 **InfoNCE loss**라 불리는 cross-entropy 손실의 변형을 사용하여 더 나은 픽셀 대응 관계를 확립하면서 기본적으로 메트릭 포인트 맵을 출력한다.

MASt3R에서 이미지 1의 픽셀 $i$와 이미지 2의 픽셀 $j$는 동일한 정답 3D 점에 대응할 때 true match로 간주된다. 즉, 하나의 이미지 내의 각 로컬 디스크립터는 다른 이미지의 단일 디스크립터에만 매칭된다. 네트워크는 InfoNCE loss를 이용해 비매칭 피처 디스크립터에 패널티를 부과하면서 이러한 디스크립터를 학습하도록 훈련되며, 이는 DUSt3R에서 사용된 단순한 3D 회귀 손실보다 훨씬 효과적이다. 이를 통해 MASt3R은 서브픽셀 수준의 정확도로 세밀한 디테일을 학습할 수 있다.

**InfoNCE Matching Loss**는 다음과 같이 정식화된다:

$$\mathcal{L}_{\text{match}} = -\frac{1}{|\mathcal{M}|} \sum_{(i,j) \in \mathcal{M}} \log \frac{\exp\!\left(\mathbf{d}_1^{(i)} \cdot \mathbf{d}_2^{(j)} / \tau\right)}{\sum_{k} \exp\!\left(\mathbf{d}_1^{(i)} \cdot \mathbf{d}_2^{(k)} / \tau\right)}$$

여기서 $\mathbf{d}_1^{(i)} \in \mathbb{R}^d$는 이미지 1의 픽셀 $i$에 대한 단위 정규화된 로컬 피처 벡터, $\mathbf{d}_2^{(j)}$는 이미지 2의 대응 픽셀 피처, $\mathcal{M}$은 정답 대응 쌍의 집합, $\tau$는 temperature 파라미터이다.

또한 스케일 불변성이 반드시 바람직하지는 않으며, map-free 시각적 로컬라이제이션 같은 사용 사례는 메트릭 스케일 예측을 필요로 한다. 따라서 회귀 손실을 수정하여 정답 포인트맵이 메트릭인 경우 예측 포인트맵에 대한 정규화를 무시한다.

**최종 학습 목표(Total Loss)**는 두 손실의 합으로 구성된다:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{conf}} + \beta \cdot \mathcal{L}_{\text{match}}$$

피처 매칭 헤드의 출력 차원은 $d = 24$이며, 신뢰도 손실 가중치는 $\alpha = 0.2$, 매칭 손실 가중치는 $\beta = 1$로 설정하여 속도와 정확도의 균형을 맞춘다.

---

#### 📌 Fast Reciprocal Matching

놀랍게도 중간 수준의 서브샘플링에서 성능이 크게 향상된다. $k = 3000$을 사용하면 매칭을 64배 가속하면서도 성능을 크게 개선할 수 있다.

고속 상호 매칭은 이미지 1에서 이미지 2로의 최근접 이웃(NN) 검색과 반대 방향 검색을 반복적으로 수행하여 **상호 대응(reciprocal correspondence)**만을 선택하는 방식이다:

$$\mathcal{M} = \{(i, j) \mid \text{NN}_2(\mathbf{d}_1^{(i)}) = j \;\wedge\; \text{NN}_1(\mathbf{d}_2^{(j)}) = i\}$$

---

### 2.3 모델 구조

두 이미지 $I_1$과 $I_2$는 가중치 공유 ViT 인코더 $E$에 의해 Siamese 방식으로 처리된다. 디코더는 두 피처 집합을 공동으로 정제하며, 자기-어텐션(self-attention)과 교차-어텐션(cross-attention)을 교차 적용하여 디코더는 입력 간 정보를 교환함으로써 상대적인 시점과 씬의 전역 3D 구조를 모두 포착한다.

```
Input: (I₁, I₂)
         ↓
  [ViT Encoder (shared)]
  F₁ = Enc(I₁), F₂ = Enc(I₂)
         ↓
  [Cross-Attention Decoder (Transformer)]
  Self-Attn ↔ Cross-Attn (F₁ ↔ F₂)
         ↓
  ┌──────────────────────┐
  │  Head 1: Pointmap    │ → X₁, X₂ (3D 포인트맵) + C₁, C₂ (신뢰도)
  │  Head 2: Feature Map │ → D₁, D₂ ∈ R^{H×W×d} (로컬 피처)
  └──────────────────────┘
         ↓
  Fast Reciprocal Matching → 픽셀 대응 M
         ↓
  Coarse-to-Fine Refinement → 고해상도 대응
```

구현에서는 인코더(ViT-Large), 디코더(ViT-Base), 헤드(Dense Prediction Transformer)를 사용하며, 지원 해상도는 512×384, 512×336, 512×288, 512×256, 512×160이다.

DUSt3R과 MASt3R은 모두 동일 팀이 개발한 **CroCo(Cross-View Completion)**라는 공통 사전학습 전략 위에 구축되어 있으며, CroCo는 이 모델들이 기하학 비전 태스크를 위한 강력한 기반 아키텍처가 되도록 하는 핵심 요소이다.

---

### 2.4 성능 향상

MASt3R은 모든 SOTA 접근법을 큰 차이로 능가하며, VCRE AUC 93% 이상을 달성한다. 이는 두 번째로 좋은 방법인 LoFTR+KBR의 63.4% AUC 대비 **30% 절대적 향상**이다. 마찬가지로, 중앙값 변환 오차(translation error)는 약 2m에서 **36cm로 크게 감소**한다.

MASt3R은 단일 통합 비전 트랜스포머로 이미지 매칭과 dense 재구성을 수행하며 LoFTR, SuperGlue 같은 태스크 특화 모델을 능가한다. MASt3R이 일반화를 잘 하는 이유는 모든 것을 2D가 아닌 **3D 관점에서 처리**하기 때문이다. 부산물로, 매우 도전적인 벤치마크에서 인상적인 메트릭을 달성하는 zero-shot 설정에서의 단안 메트릭 깊이와 같은 다중 3D 하위 태스크를 해결할 수 있다.

| 방법 | Map-free VCRE AUC | 중앙값 Translation Error |
|------|------------------|------------------------|
| LoFTR+KBR | 63.4% | ~2m |
| **MASt3R** | **>93%** | **36cm** |

---

### 2.5 한계점

MASt3R은 최대 크기 512픽셀의 이미지만 처리한다. 더 큰 이미지는 학습에 훨씬 더 많은 컴퓨팅 파워가 필요하며, ViT는 더 큰 테스트 시 해상도에 일반화되지 않는다.

결과적으로 고해상도 이미지(예: 100만 픽셀)는 매칭을 위해 다운스케일되어야 하며, 이후 결과적인 대응이 원래 이미지 해상도로 업스케일된다. 이는 일부 성능 손실을 야기할 수 있으며, 때로는 로컬라이제이션 정확도나 재구성 품질에서 상당한 저하를 초래하기에 충분하다.

신경망 기반이므로 DUSt3R과 MASt3R은 다른 도메인의 신경망과 유사한 문제를 공유한다. 인상적인 zero-shot 성능에도 불구하고, 이 접근법들은 수중 이미지, 항공 이미지, 또는 매우 넓거나 좁은 렌즈 이미지처럼 학습 데이터와 실질적으로 다른 환경에서는 효과적으로 작동하기 위해 재학습이 필요할 수 있다. 전통적 MVS 접근법은 이런 변화에 더 강인할 수 있다.

또한, A40 GPU 기준 이미지 쌍당 198.16ms의 레이턴시로, ViT 인코더-디코더와 Fast Reciprocal NN 매칭 단계로 인한 계산 오버헤드로 인해 **추론 속도가 병목**이 된다.

또한 Niantic 데이터셋에서 MASt3R의 성능이 Map-free Niantic 리더보드 대비 비교적 낮은데, MASt3R이 map-free 재로컬라이제이션을 위해 설계·학습된 반면, 다른 태스크에서는 카메라 고유 파라미터가 다양한 여러 프레임 간 매칭을 포함하기 때문이다.

또한 비싼 2단계 글로벌 최적화가 모든 pairwise 재구성을 동일한 좌표 시스템에 정렬하는 데 필요하며, DUSt3R과 MASt3R은 모든 뷰가 소규모 영역에 집중된 객체 중심 DTU 데이터에서만 평가된다. 대규모 장면 재구성에서의 성능 검증이 부족하다.

---

## 3. 🌐 모델 일반화 성능 향상 가능성

### 3.1 일반화 강점의 근거

3D Foundation Model(3DFM)은 깊이, 형태, 포즈 등 장면의 공간적·구조적 속성을 포착하도록 학습된 대규모 비전 모델이다. 외형 중심 모델과 달리, 3DFM은 2D 및 3D 지도 학습을 사용하여 기하학 인식 표현을 학습하므로 태스크 전반에 걸쳐 강력한 일반화를 가능케 한다.

매칭은 GT 카메라에서 오는 에피폴라 제약 조건을 활용하지 않는 one-versus-all 전략으로 수행되었으며, 이는 모든 기존 MVS 접근법과 극명한 대조를 이룬다. MASt3R은 특히 정밀하고 강인하여, 날카롭고 밀도 있는 디테일을 제공한다.

DUSt3R은 합성 데이터와 실제 데이터가 동등한 비율로 구성된 대규모 데이터 코퍼스로 사전학습된다. 합성 데이터는 완벽한 기하학적 사전 지식을 제공하고, 실제 데이터는 실제 세계 시나리오에 맞게 분포를 조정한다. 이는 실내외 장면을 모두 포함하는 약 850만 이미지 쌍이다.

### 3.2 일반화 한계 및 가능성

DUSt3R, MASt3R, VGGT 등 최근 개발된 3D 재구성 기반 모델은 매우 희소한 이미지 겹침 처리 능력과 일반화 가능성으로 상당한 주목을 받고 있다. 항공 이미지에서의 MASt3R 평가는 중요한데, 이 모델들이 극도로 낮은 이미지 겹침, 스테레오 가림(occlusion), 텍스처 없는 영역을 처리할 가능성을 갖고 있기 때문이다. 고도로 중복적인 컬렉션의 경우 극도로 희소화된 이미지 세트를 사용하여 3D 재구성을 가속할 수 있다.

도메인 특화 학습 없이도 MASt3R은 DTU 데이터셋에서 경쟁적 성능을 달성하며, DUSt3R을 능가하고 최고 방법에 근접한다.

일반화 향상을 위한 잠재적 방향으로는:

1. **데이터 다양화**: 합성 데이터가 완벽한 기하학적 사전 지식을 제공하고 실제 데이터가 실제 세계 시나리오에 맞게 분포를 조정하는 방식처럼, 수중, 의료, 항공 도메인 데이터를 혼합하면 일반화가 향상될 수 있다.

2. **도메인 적응(Domain Adaptation)**: 특정 도메인에 DUSt3R/MASt3R을 파인튜닝하면 성능을 크게 향상시킬 수 있다는 것이 실험적으로 확인되었다.

3. **Splatt3R 사례**: Splatt3R은 ScanNet++ 데이터셋으로 학습하고 보정되지 않은 실제 이미지에 대한 우수한 일반화를 보여준다. 이처럼 MASt3R 백본을 활용한 확장은 광범위한 도메인 일반화를 달성할 수 있다.

---

## 4. 🔬 2020년 이후 관련 연구 비교 분석

| 방법 | 발표연도 | 방식 | 강점 | 약점 |
|------|---------|------|------|------|
| **SuperGlue** | CVPR 2020 | Sparse + GNN | 빠른 추론 | 극단적 시점 변화에 취약 |
| **LoFTR** | CVPR 2021 | Detector-Free, Dense | 텍스처 없는 영역 강인 | 과도한 일반화 취약 |
| **DUSt3R** | CVPR 2024 | Pointmap 회귀, 3D 기반 | 극단 시점 강인 | 정밀도 제한 |
| **RoMa** | CVPR 2024 | DINOv2 기반 Dense | 다양한 도메인 강인 | 계산 비용 高 |
| **MASt3R** | ECCV 2024 | 3D+Dense Feature | 정밀+강인, 메트릭 깊이 | 속도, 고해상도 한계 |
| **MASt3R-SfM** | ICCV 2025 | MASt3R+SfM 통합 | End-to-End SfM | 대규모 장면 메모리 이슈 |

LoFTR, DKM, RoMa, SuperGlue와 같은 딥러닝 기반 dense 매칭 기법은 Transformer 기반 아키텍처를 통해 전역 피처 추론을 활용한다. 이 방법들은 큰 시점 및 조명 변화에서 강인성을 개선하여 벤치마크에서 SOTA 성능을 달성한다. 그러나 dense 매칭은 높은 계산 비용을 유발하여 실시간 응용에는 적합하지 않다.

DUSt3R은 픽셀 대응을 위한 3D 포인트맵 사용의 선구자로, 극단적 시점 변화에 대한 우수한 복원력을 보여준다. MASt3R은 3D 구조와 함께 로컬 피처를 학습하는 Transformer 기반 매칭 헤드를 통합함으로써 이 접근법을 확장하여 더 정밀한 매칭을 가능케 한다.

MASt3R 기반 모델에 인코딩된 강력한 사전 지식 덕분에 모션이 없는 경우도 처리할 수 있으며, RANSAC에 전혀 의존하지 않는다.

---

## 5. 🔮 미래 연구 영향 및 고려사항

### 5.1 앞으로의 연구에 미치는 영향

**① 3D 기반 매칭 패러다임의 전환**

실용적으로 MASt3R은 강인성과 정확도 덕분에 시각적 로컬라이제이션, 내비게이션, 로보틱스, 사진측량 분야에서 새로운 가능성을 열고 있다. 이론적으로는 컴퓨터 비전 태스크에서 보다 전체론적이고 기하학 인식 접근법으로의 전환을 나타낸다.

**② 후속 연구 파급효과**

MASt3R은 이미지 매칭을 3D 태스크로 재정의하고 매칭을 수십 배 가속하는 빠른 상호 매칭 스킴을 도입했으며, DUSt3R과 MASt3R은 짧은 기간 내에 250개 이상의 인용을 축적하여 커뮤니티에서의 영향력을 입증했다.

**③ SLAM 및 SfM 분야 확장**

MASt3R-SLAM은 2뷰 3D 재구성 및 매칭 사전 지식인 MASt3R에서 bottom-up 방식으로 설계된 실시간 단안 dense SLAM 시스템을 제시한다.

SfM의 전통적 해결책은 오류를 전파하고 이미지 중첩이 충분하지 않거나 모션이 너무 적을 때 실패하는 복잡한 최소 솔버 파이프라인으로 구성된다. MASt3R-SfM은 이 핵심 이슈들을 해결하지 못하는 최근 방법들의 한계를 실증적으로 보여주며 MASt3R 기반 SfM을 제안한다.

**④ Gaussian Splatting과의 융합**

MASt3R의 정신에 따라, 아키텍처에 대한 단순한 수정과 잘 선택된 학습 손실만으로도 강력한 novel view synthesis 성능을 달성하기에 충분하다는 것이 입증되었다.

---

### 5.2 앞으로의 연구 시 고려해야 할 점

#### ① 고해상도 이미지 처리
MASt3R은 최대 크기 512픽셀의 이미지만 처리하는 제약이 있으며, 향후 연구에서는 ViT의 해상도 스케일링 문제를 해결하는 방향(예: Hierarchical ViT, FlexiViT)이 필요하다.

#### ② 계산 효율화
A40 GPU에서 이미지 쌍당 198.16ms의 레이턴시를 보이며, ViT 인코더-디코더와 FastNN 매칭 단계로 인한 계산 오버헤드가 병목으로 남아 있다. 이를 해결하기 위한 **모델 경량화, 양자화, 지식 증류** 연구가 요구된다.

#### ③ 멀티뷰 확장 및 시간적 일관성
DUSt3R과 MASt3R은 한 번에 이미지 쌍만 처리하며, 2개 이상의 뷰를 다룰 때 조합적으로 많은 오류 가능성이 있는 pairwise 재구성과 이어지는 비싼 글로벌 최적화가 필요하다. **MV-DUSt3R, Spann3R, SLAM3R**처럼 다중 뷰를 단일 패스로 처리하는 방향의 연구가 요구된다.

#### ④ 도메인 일반화 검증
다양한 컴퓨터 비전 벤치마크에서 테스트되었음에도 불구하고, 항공 측량 블록 같은 도메인에서의 잠재력은 여전히 탐구되지 않았다. 수중, 의료, 위성 이미지 등 미탐색 도메인에 대한 일반화 평가 및 파인튜닝 연구가 필요하다.

#### ⑤ 동적 장면 및 반사 표면 처리
매칭은 다양한 텍스처나 재질에도 강인하며 Lambertian 가정 위반, 즉 표면의 반사에도 강인하다는 긍정적인 측면이 있지만, 움직이는 객체가 있는 동적 장면에서의 한계는 여전히 과제로 남아 있다.

---

## 📚 참고 문헌 및 출처

1. **Leroy, V., Cabon, Y., & Revaud, J. (2024).** *Grounding Image Matching in 3D with MASt3R.* ECCV 2024. [arXiv:2406.09756](https://arxiv.org/abs/2406.09756)

2. **ECCV 2024 공식 논문 PDF**: https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/09080.pdf

3. **Springer Nature Link (ECCV 2024)**: https://link.springer.com/chapter/10.1007/978-3-031-73220-1_5

4. **Semantic Scholar**: https://www.semanticscholar.org/paper/Grounding-Image-Matching-in-3D-with-MASt3R-Leroy-Cabon/4f997c404e97087194d2df538deb82b2c5428c1e

5. **GitHub 공식 구현 (NAVER)**: https://github.com/naver/mast3r

6. **Duisterhof et al. (2024).** *MASt3R-SfM: a Fully-Integrated Solution for Unconstrained Structure-from-Motion.* [arXiv:2409.19152](https://arxiv.org/abs/2409.19152)

7. **LearnOpenCV - MASt3R & MASt3R-SfM 분석**: https://learnopencv.com/mast3r-sfm-grounding-image-matching-3d/

8. **LearnOpenCV - DUSt3R 분석**: https://learnopencv.com/dust3r-geometric-3d-vision/

9. **NAVER Labs Blog**: https://europe.naverlabs.com/blog/3d-reconstruction-models-made-easy/

10. **Mismatched (2024).** *Evaluating the Limits of Image Matching Approaches and Benchmarks.* [arXiv:2408.16445](https://arxiv.org/abs/2408.16445)

11. **Speedy MASt3R (2025).** [arXiv:2503.10017](https://arxiv.org/html/2503.10017v1)

12. **SegMASt3R (NeurIPS 2025).** [arXiv:2510.05051](https://arxiv.org/html/2510.05051)

13. **MV-DUSt3R+ (2024).** [arXiv:2412.06974](https://arxiv.org/pdf/2412.06974)

14. **Splatt3R Project Page**: https://splatt3r.active.vision/

15. **Tandfonline - DUSt3R/MASt3R/VGGT 항공 측량 평가**: https://www.tandfonline.com/doi/full/10.1080/10095020.2025.2597491

16. **Emergent Mind - MASt3R 요약**: https://www.emergentmind.com/papers/2406.09756

17. **Awesome-DUSt3R GitHub**: https://github.com/ruili3/awesome-dust3r
