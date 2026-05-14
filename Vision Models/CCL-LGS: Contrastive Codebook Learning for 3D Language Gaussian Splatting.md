
# CCL-LGS: Contrastive Codebook Learning for 3D Language Gaussian Splatting

> **⚠️ 정확도 안내:** 본 논문은 ICCV 2025에 게재된 논문(arXiv:2505.20469)으로, 공개된 abstract, HTML 버전, 프로젝트 페이지를 기반으로 작성되었습니다. 논문 내부의 세부 수식(특히 loss function의 구체적인 파라미터 값 등)은 직접 PDF 전문을 파싱하지 못한 관계로 일반적인 contrastive learning 공식 형태로 기술하되, 불확실한 부분은 명시합니다.

---

## 1. 핵심 주장과 주요 기여 요약

### 📌 배경

3D 재구성 기술과 비전-언어 모델의 최근 발전은 로보틱스, 자율주행, 가상/증강현실 등에 필수적인 3D 의미론적 이해(3D semantic understanding) 분야에서 큰 진전을 이끌었다.

### 📌 핵심 문제 (Core Problem)

그러나 2D prior에 의존하는 방법들은 치명적인 문제를 안고 있다. **폐색(occlusion), 이미지 블러, 시점 종속 변화(view-dependent variation)**에 의해 유발되는 **크로스 뷰 시맨틱 불일치(cross-view semantic inconsistency)** 가 발생하며, 이 불일치가 프로젝션 기반 감독(projection supervision)을 통해 전파되면 3D Gaussian 시맨틱 필드의 품질이 저하되고 렌더링 결과물에 아티팩트(artifact)가 발생한다.

### 📌 핵심 주장 (Core Claims)

본 논문에서는 **CCL-LGS**라는 뷰 일관성 있는 3D 의미론적 재구성을 위한 새로운 프레임워크를 제안한다. 핵심 혁신은 특별히 설계된 **3단계 파이프라인**을 통해 뷰 일관적인 시맨틱 감독을 구축하는 것으로, 구체적으로 SAM을 이용한 정확한 인스턴스 마스크 추출 → 제로샷 트래킹을 통한 크로스 뷰 대응 정렬 → CCL 모듈을 통한 시맨틱 증류의 순서로 이루어진다.

### 📌 주요 기여 (Main Contributions)

| 기여 항목 | 설명 |
|---|---|
| **CCL 모듈** | Contrastive Codebook Learning으로 intra-class compactness + inter-class distinctiveness 강화 |
| **Zero-shot Tracker 기반 Mask Association** | SAM 생성 2D 마스크의 크로스 뷰 대응 신뢰성 향상 |
| **듀얼 스케일 시맨틱 추출** | 멀티뷰 이미지에서 두 단계 의미 특징 추출 |
| **SOTA 달성** | 오픈 어휘 의미 분할(open-vocabulary semantic segmentation) 벤치마크에서 SOTA 달성 |

---

## 2. 논문 상세 분석

### 2-1. 해결하고자 하는 문제

3D Gaussian Splatting(3DGS)과 비전-언어 모델의 발전으로 3D 의미론적 이해에 큰 진전이 있었지만, 3D 표현을 2D로 투영해 감독하는 방법들은 **폐색, 이미지 블러, 시점 종속 변화**에 의한 크로스 뷰 시맨틱 불일치 문제를 겪는다. 이러한 불일치는 3D 시맨틱 필드에 전파되어 품질을 저하시키고 렌더링 아티팩트를 유발한다.

기존 방법(LangSplat 등)의 한계:
직접 CLIP을 불완전한 마스크에 적용하면 일관성 없는 시맨틱 특징이 생성되어 3D 재구성에 아티팩트가 발생한다. 이를 해결하기 위해 compact하고 distinctive한 특징을 만드는 전용 CCL 모듈이 제안된다.

---

### 2-2. 제안 방법 및 모델 구조

#### 🏗️ 전체 파이프라인 (3단계)

먼저 멀티뷰 이미지에서 두 단계의 시맨틱 특징을 추출하고, 그 다음 마스크 연관(mask association)과 대조적 코드북 학습(contrastive codebook learning)을 수행하여 특징을 정제하며, 마지막으로 이 시맨틱 정보를 3D Gaussian 시맨틱 필드에 통합한다.

#### 🔷 Stage 1: 듀얼 스케일 시맨틱 특징 추출

정확하고 경계를 잘 인식하는 시맨틱 특징을 얻기 위해, **SAM ViT-H**를 사용하며 각 이미지에 균일한 32×32 포인트 프롬프트 그리드를 적용한다.

#### 🔷 Stage 2: 마스크 연관 및 CCL 모듈

제로샷 트래커를 사용하여 SAM이 생성한 2D 마스크들을 정렬하고, 해당 카테고리를 신뢰성 있게 식별한다. 그 다음 CLIP을 사용하여 뷰 전반에 걸친 강건한 시맨틱 인코딩을 추출한다. 최종적으로 CCL 모듈이 **intra-class compactness(클래스 내 응집성)**와 **inter-class distinctiveness(클래스 간 변별력)**를 강제하여 판별적 시맨틱 특징을 추출한다.

CCL 모듈은 대조적 메트릭 학습(contrastive metric learning)을 도입하여 클래스 내 특징 응집성을 강화하는 동시에 클래스 간 특징 변별성을 유지한다. 이 설계는 불완전하거나 노이즈가 있는 마스크로 인해 발생하는 시맨틱 모호성을 효과적으로 완화한다. 기존 방식과 달리, 이 프레임워크는 뷰 전반에 걸친 신뢰할 수 있는 시맨틱 대응을 구축하면서 카테고리별 변별성을 보존한다.

#### 🔷 CCL 손실 함수 (수식)

CCL 모듈은 코드북 $\mathcal{C} = \{c_k\}_{k=1}^{K}$ (K개의 코드워드)를 학습하며, "Pull Loss"(intra-class)와 "Push Loss"(inter-class)로 구성된 대조 손실을 사용합니다.

**Pull Loss (intra-class compactness):**

$$\mathcal{L}_{\text{pull}} = \frac{1}{N} \sum_{i=1}^{N} \left\| \mathbf{f}_i - \mathbf{c}_{k(i)} \right\|_2^2$$

여기서 $\mathbf{f}\_i$는 $i$번째 마스크 특징, $\mathbf{c}_{k(i)}$는 해당 카테고리 $k$의 코드워드이다.

**Push Loss (inter-class distinctiveness):**

$$\mathcal{L}_{\text{push}} = \frac{1}{K(K-1)} \sum_{j \neq k} \max\left(0,\, m - \left\| \mathbf{c}_k - \mathbf{c}_j \right\|_2 \right)^2$$

여기서 $m$은 margin 하이퍼파라미터이며, 서로 다른 클래스 코드워드 간 최소 거리를 보장한다.

**전체 CCL 손실:**

$$\mathcal{L}_{\text{CCL}} = \mathcal{L}_{\text{pull}} + \lambda \mathcal{L}_{\text{push}}$$

> ⚠️ **주의:** 위 수식은 논문에서 명시된 contrastive codebook learning의 pull/push 구조에 기반하여 표준적인 형태로 기술하였습니다. Ablation 연구를 통해 pull loss와 push loss 모두 특징의 일관성과 변별성을 향상시키는 데 결정적인 역할을 한다는 것이 확인되었다. 정확한 수식 표기는 원문 PDF를 직접 확인하시기 바랍니다.

#### 🔷 Stage 3: 3D Gaussian 시맨틱 필드 통합

멀티뷰 이미지에서 두 단계 시맨틱 특징을 추출한 후, 마스크 연관 및 대조적 코드북 학습으로 이 특징들을 정리하고 정제하며, 마지막으로 시맨틱 정보를 3D Gaussian 시맨틱 필드에 통합한다.

각 3D Gaussian $g_i$에는 시맨틱 특징 벡터 $\mathbf{s}_i$가 할당되며, 알파 블렌딩(alpha-blending)을 통한 렌더링으로 2D 시맨틱 맵을 생성하고, 이를 CCL로 정제된 감독 신호와 비교하여 학습된다.

**Gaussian 렌더링 (Semantic Feature Rendering):**

$$\hat{\mathbf{S}} = \sum_{i \in \mathcal{N}} \mathbf{s}_i \alpha_i \prod_{j < i}(1 - \alpha_j)$$

여기서 $\alpha_i$는 각 Gaussian의 불투명도(opacity), $\mathbf{s}_i$는 시맨틱 특징 벡터이다.

---

### 2-3. 성능 향상

CCL-LGS와 LangSplat의 세 가지 어려운 시나리오(폐색, 이미지 블러, 시점 종속 변화)에 대한 정량적 비교 결과는 CCL-LGS 방법이 이러한 도전들을 처리하는 데 있어 더 큰 강건성과 충실도를 보여줌을 명확히 입증한다.

벤치마크 데이터셋에서의 광범위한 실험은 본 방법이 오픈 어휘 의미 분할(open-vocabulary semantic segmentation) 작업에서 **SOTA 성능**을 달성함을 보여준다.

3D-OVS 데이터셋에서는 CCL-LGS가 SOTA와 비슷한 성능(평균 mIoU 95.2 vs 3D VL-GS의 96.9)을 보이는데, 저자들은 3D VL-GS의 데이터 증강이 이 단순한 데이터셋에서 더 유리하게 작용하기 때문이라고 설명한다. 정성적 결과에서는 CCL-LGS가 베이스라인 대비 더 정확하고 일관성 있는 분할 맵을 생성함을 보여준다.

효율성 분석에서는 성능과 리소스 사용 사이의 합리적인 균형을 보여준다.

---

### 2-4. 한계 (Limitations)

한계는 SAM 및 SAM2의 고유한 역량에서 비롯되며, 불완전한 마스크가 여전히 결과에 영향을 미친다. 향후 연구에서는 더 강건한 결과를 위해 마스크 품질을 개선할 것이다.

방법의 성능은 본질적으로 기반이 되는 비전 파운데이션 모델(SAM, CLIP)의 품질에 묶여 있다.

---

## 3. 모델의 일반화 성능 향상 가능성

### 3-1. 제로샷 트래커의 일반화 기여

CCL-LGS는 멀티뷰 시맨틱 큐를 통합함으로써 뷰 일관적인 시맨틱 감독을 강화하는 새로운 프레임워크이다. 제로샷 트래커를 사용한 마스크 연관은 특정 장면에 종속되지 않으며, 다양한 장면 유형에 적용 가능하다.

### 3-2. Open-Vocabulary 쿼리 지원

CCL-LGS는 정확하고 효율적인 오픈 어휘 쿼리를 가능하게 하는 3D 언어 필드를 구축하는 새로운 방법으로, 기존에 간과되어 온 문제인 "불완전한 마스크에 직접 CLIP을 적용하면 비일관적인 시맨틱 특징이 생성된다"는 점을 직접 해결한다. CLIP 기반의 오픈 어휘 특성 덕분에, 학습 시 보지 못한 카테고리에도 언어 쿼리가 가능하다.

### 3-3. CCL 모듈의 범용적 일반화 효과

CCL-LGS는 뷰 일관적인 시맨틱 감독을 통합하여 3D Gaussian 시맨틱 필드 재구성을 가능하게 한다. 또한 CCL 모듈은 noisy하거나 불완전한 마스크 상황에서도 intra-class compactness와 inter-class distinctiveness를 강화하여 시맨틱 모호성을 해결하고 강건한 시맨틱 표현을 가능하게 한다.

이는 다음과 같은 일반화 시나리오에서 효과적이다:

| 도전 시나리오 | CCL-LGS의 대응 |
|---|---|
| **폐색(Occlusion)** | 제로샷 트래커로 가려진 객체의 멀티뷰 대응 추적 |
| **이미지 블러** | 크로스 뷰 집계로 단일뷰 블러 영향 완화 |
| **시점 종속 변화** | 코드북 기반 특징 정규화로 뷰 의존적 변화 흡수 |
| **노이즈 마스크** | CCL의 pull/push loss로 노이즈 특징 보정 |

### 3-4. 파운데이션 모델 의존성의 양면성

CCL-LGS는 SAM과 CLIP이라는 강력한 파운데이션 모델을 활용하므로, 이 모델들이 업데이트될수록 CCL-LGS의 성능도 자동으로 향상될 가능성이 높다. 그러나 방법의 성능은 본질적으로 SAM, CLIP 등 기반 비전 파운데이션 모델의 품질에 종속된다는 한계도 있다.

---

## 4. 관련 최신 연구 비교 분석 (2020년 이후)

### 4-1. 핵심 관련 연구 흐름

| 연도 | 방법 | 특징 | 한계 |
|---|---|---|---|
| 2020 | **NeRF** | 암묵적 신경망 기반 3D 표현 | 학습/렌더링 속도 느림 |
| 2023 | **LERF** | NeRF에 CLIP 언어 임베딩 증류 | 느린 추론 속도, per-scene 최적화 필요 |
| 2023 | **LangSplat** (CVPR 2024) | 3DGS에 CLIP 언어 특징 임베딩, SAM 마스크 활용 | 크로스 뷰 시맨틱 불일치 문제 미해결 |
| 2023 | **LEGaussians** | 2D 코드북 기반 언어 Gaussian | 2D 코드북의 3D prior 부재 |
| 2024 | **3D VL-GS** | 데이터 증강 기반 시맨틱 향상 | 단순 장면에서는 효과적이나 복잡한 장면에서는 제한적 |
| **2025** | **CCL-LGS** (ICCV 2025) | CCL 모듈로 크로스 뷰 일관성 확보 | SAM/CLIP 품질에 종속 |
| 2025 | **LangSplatV2** (NeurIPS 2025) | 450+ FPS, 전역 3D 코드북, MLP 제거 | 학습 비용 증가 |

### 4-2. LangSplat vs. CCL-LGS

LangSplat은 기존 NeRF 기반 방법들과 달리 3D Gaussian 컬렉션을 활용하여 CLIP에서 추출된 언어 특징을 인코딩함으로써 언어 필드를 표현한다. 그러나 LangSplat은 크로스 뷰 시맨틱 불일치 문제를 명시적으로 해결하지 않는다.

기존 방법들이 직접 CLIP을 불완전한 마스크에 적용하는 것과 달리, CCL-LGS 프레임워크는 시맨틱 충돌을 명시적으로 해결하면서 카테고리 변별성을 보존한다.

### 4-3. LEGaussians vs. CCL-LGS

LEGaussians는 2D 이미지에서 특징을 양자화하여 2D 코드북을 먼저 학습한 후, 코드북 인덱스를 예측하는 3D 모델을 학습한다. 반면 LangSplatV2는 모든 3D Gaussian 포인트가 공유하는 전역 3D 코드북을 학습한다. CCL-LGS의 코드북은 단순 압축이 목적이 아닌, 시맨틱 충돌 해소와 뷰 일관성 확보에 초점을 맞추는 점에서 차별화된다.

### 4-4. LangSplatV2와의 관계

LangSplatV2는 476.2 FPS에서 고차원 특징 스플래팅, 384.6 FPS에서 3D 오픈 어휘 텍스트 쿼리를 달성하며, LangSplat 대비 각각 42배, 47배 속도 향상과 개선된 쿼리 정확도를 제공한다.

LangSplatV2는 **속도와 효율성**, CCL-LGS는 **뷰 일관성과 로버스트니스**에 각각 초점을 맞춘 상호 보완적 연구로 볼 수 있다.

---

## 5. 앞으로의 연구에 미치는 영향과 고려점

### 5-1. 앞으로의 연구에 미치는 영향

#### 🔹 3D 시맨틱 필드 품질의 새로운 기준 제시
CCL-LGS는 정확하고 효율적인 오픈 어휘 쿼리를 가능하게 하는 3D 언어 필드를 구축하는 새로운 방법을 제시하며, 크로스 뷰 불일치라는 기존의 간과된 문제를 정면으로 다룸으로써, 후속 연구들이 시맨틱 일관성을 평가 지표로 더 적극적으로 고려하도록 유도할 것이다.

#### 🔹 Contrastive Codebook 패러다임 확장 가능성
CCL의 pull/push 기반 코드북 학습 아이디어는 3D 언어 필드 이외에도 포인트 클라우드 이해, 4D 동적 장면 이해, 멀티모달 학습 등 다양한 분야로 확장될 수 있다.

#### 🔹 로보틱스·자율주행 응용 촉진
3D 의미론적 이해는 로보틱스, 자율주행, 가상/증강현실에 핵심적인 역량이다. 폐색이 빈번한 실세계 환경에서의 강건한 시맨틱 필드 구축은 이 분야들에 직접적인 기여를 한다.

#### 🔹 파운데이션 모델 활용의 표준화
제로샷 트래커(SAM2)와 CLIP을 조합하는 모듈러 파이프라인은, 새로운 파운데이션 모델이 등장할 때 각 모듈을 교체하기 쉬운 구조를 제시하여 연구 커뮤니티의 빠른 발전에 기여할 것이다.

---

### 5-2. 향후 연구 시 고려할 점

| 고려 사항 | 설명 |
|---|---|
| **마스크 품질 개선** | SAM/SAM2의 한계로 불완전한 마스크가 여전히 결과에 영향을 미치므로 더 강건한 마스크 생성 또는 마스크 없는 방법 연구 필요 |
| **실시간 처리** | LangSplatV2처럼 CCL-LGS의 속도를 실시간 수준으로 끌어올리는 연구 필요 |
| **4D/동적 장면** | 정적 장면 가정에서 벗어나, 시간적으로 변화하는 동적 장면으로의 확장 |
| **Sparse-view 일반화** | 3D 의미 필드 학습은 자율 항법, AR/VR, 로보틱스에 중요한데, 기존 방법들은 희소 뷰 조건에서 어려움을 겪는다. 적은 뷰로도 일관성 있는 시맨틱을 학습하는 방법 연구가 필요하다 |
| **멀티모달 확장** | 텍스트 외에도 오디오, 깊이 정보 등 추가 모달리티와의 통합 |
| **파운데이션 모델 종속성 탈피** | SAM/CLIP에 종속되지 않는 자체 시맨틱 추출 방법 연구 |
| **데이터셋 다양성** | 실내 장면 위주에서 실외 대규모 환경으로의 평가 확장 |

---

## 📚 참고 자료 및 출처

| 자료 | 정보 |
|---|---|
| **CCL-LGS 논문 (arXiv)** | arXiv:2505.20469, Lei Tian et al., 2025 — https://arxiv.org/abs/2505.20469 |
| **CCL-LGS ICCV 2025 공식 페이지** | https://openaccess.thecvf.com/content/ICCV2025/html/Tian_CCL-LGS... (ICCV 2025, pp. 9855–9864) |
| **CCL-LGS 프로젝트 페이지** | https://epsilontl.github.io/CCL-LGS/ |
| **Semantic Scholar** | https://www.semanticscholar.org/paper/CCL-LGS... |
| **Moonlight Literature Review** | https://www.themoonlight.io/en/review/ccl-lgs-... |
| **LangSplat (CVPR 2024)** | arXiv:2312.16084, Qin et al. — https://arxiv.org/abs/2312.16084 |
| **LangSplatV2 (NeurIPS 2025)** | arXiv:2507.07136, Li et al. — https://arxiv.org/abs/2507.07136 |
| **ICCV 2025 Poster** | https://iccv.thecvf.com/virtual/2025/poster/290 |
| **ResearchGate** | https://www.researchgate.net/publication/392133501 |
