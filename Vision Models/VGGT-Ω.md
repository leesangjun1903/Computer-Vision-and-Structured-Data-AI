
# VGGT- $\Omega$ 

> **논문 정보**
> - **제목**: VGGT- $\Omega$
> - **저자**: Jianyuan Wang, Minghao Chen, Shangzhan Zhang, Nikita Karaev, Johannes Schönberger, Patrick Labatut, Piotr Bojanowski, David Novotny, Andrea Vedaldi, Christian Rupprecht
> - **소속**: University of Oxford (Visual Geometry Group), Meta AI
> - **발표**: CVPR 2026 **Oral**
> - **arXiv**: [2605.15195](https://arxiv.org/abs/2605.15195)

---

## 1. 🔑 핵심 주장과 주요 기여 (Summary)

VGGT- $\Omega$는 정적 및 동적 장면 모두에서 정확도, 효율성, 능력 면에서 최신 기술 수준을 크게 향상시키는 feed-forward 3D 재구성 모델입니다.

### 핵심 주장 (3가지)

| # | 핵심 주장 |
|---|-----------|
| 1 | **스케일링 법칙 검증**: Feed-forward 3D 재구성 모델의 품질이 모델 및 데이터 크기에 따라 **예측 가능하게(predictably) 향상**된다 |
| 2 | **효율적 아키텍처**: Register Attention 도입으로 메모리를 획기적으로 절감하면서 대규모 학습 가능 |
| 3 | **공간 이해의 일반화**: 학습된 기하 표현이 단순한 재구성을 넘어 VLA(Vision-Language-Action) 모델 등 downstream task에도 전이 가능 |

### 주요 기여 요약

전례 없는 규모에서 모델을 훈련하기 위해, 훈련 효율성을 개선하는 아키텍처 변경, 동적 장면을 지원하는 고품질 데이터 어노테이션 파이프라인, 그리고 자기지도 학습(self-supervised learning) 프로토콜을 도입합니다.

---

## 2. 🧩 해결하고자 하는 문제 / 제안 방법 / 모델 구조 / 성능 및 한계

### 2-1. 해결하고자 하는 문제

기존 모델들은 스케일링 장벽에 부딪혔습니다. 더 큰 모델을 훈련하려면 과도한 GPU 메모리가 필요했고, 이는 모델 용량과 훈련 데이터 크기 모두를 제한했습니다. VGGT는 비용이 큰 고해상도 컨볼루션과 모든 프레임에 걸친 전역 어텐션(global attention)을 사용했기 때문에 15배 더 큰 데이터셋이나 대규모 비레이블 비디오를 활용하는 것이 불가능했습니다. 연구 분야는 더 큰 모델 + 더 많은 데이터 = 더 나은 재구성(스케일링 법칙 가설)을 알고 있었지만, 아키텍처가 스케일되지 않았기 때문에 이를 검증할 수 없었습니다.

구체적으로는 두 가지 병목이 존재했습니다:

1. **메모리 병목**: VGGT의 All-to-all 전역 어텐션 → $O(N^2)$ 복잡도 (N = 전체 토큰 수)
2. **동적 장면 한계**: 기존 모델이 정적 장면에만 집중하여 동적 객체 처리 불가

---

### 2-2. 제안하는 방법 (수식 포함)

#### ① Register Attention (핵심 혁신)

VGGT의 아키텍처를 멀티태스크 감독을 사용하는 단일 밀집 예측 헤드로 단순화하고 비용이 큰 고해상도 컨볼루션 레이어를 제거합니다. 또한 레지스터(registers)를 사용하여 장면 정보를 간결한 표현으로 집약하고, 프레임 간 정보 교환을 이 레지스터들로 제한하는 register attention을 도입하여, 전역 어텐션을 부분적으로 대체합니다.

기존 VGGT의 전역 어텐션과 VGGT- $\Omega$의 Register Attention을 비교하면:

**[기존 VGGT - 전역 어텐션]**

모든 프레임의 토큰이 서로 직접 어텐션을 수행합니다.

$$\text{GlobalAttn}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\!\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}}\right)\mathbf{V}$$

여기서 $\mathbf{Q}, \mathbf{K}, \mathbf{V} \in \mathbb{R}^{(N \cdot F) \times d}$ ($F$=프레임 수, $N$=프레임당 토큰 수)로, 복잡도가 $O((NF)^2)$입니다.

**[VGGT- $\Omega$ - Register Attention]**

소수의 학습 가능한 레지스터 토큰 $\mathbf{r} = \{r_1, r_2, \ldots, r_K\}$ ($K \ll N \cdot F$)를 도입하여, 프레임 토큰들은 이 레지스터를 통해서만 간접적으로 정보를 교환합니다:

$$\text{RegisterAttn}(\mathbf{x}_i, \mathbf{r}) = \text{Attn}(\mathbf{x}_i \rightarrow \mathbf{r}) \oplus \text{Attn}(\mathbf{r} \rightarrow \mathbf{x}_j)$$

이 구조에서 통신 복잡도는 $O(N \cdot F \cdot K)$로 감소합니다 (hub-and-spoke 방식). $K$가 고정 상수이므로 사실상 선형 스케일링입니다.

#### ② 멀티태스크 손실 함수

VGGT의 아키텍처를 여러 밀집 예측 헤드를 손실 주도(loss-driven) 멀티태스크 학습으로 대체하고, 불안정한 DPT 블록을 제거하며, scene token을 통한 더 효율적인 전역 어텐션을 도입하여 크게 단순화합니다.

멀티태스크 손실은 단일 헤드에서 다음과 같이 구성됩니다:

$$\mathcal{L}_{\text{total}} = \lambda_{\text{depth}} \mathcal{L}_{\text{depth}} + \lambda_{\text{cam}} \mathcal{L}_{\text{cam}} + \lambda_{\text{feat}} \mathcal{L}_{\text{feat}}$$

여기서 각 손실 항은 깊이 예측, 카메라 포즈 추정, 기하 특징에 해당하며, $\lambda$ 값들은 손실 주도 가중치 조정을 통해 학습됩니다.

#### ③ 자기지도 학습 (Self-Supervised Learning)

이를 통해 VGGT- $\Omega$는 언어 모델이 비레이블 텍스트를 사용하는 것처럼 대규모 비레이블 데이터셋을 활용할 수 있습니다.

광도 일관성(photometric consistency)을 기반으로 한 자기지도 학습을 통해, 비레이블 비디오에서도 기하 구조를 학습합니다:

$$\mathcal{L}_{\text{SSL}} = \sum_{(i,j)} \left\| I_i - \text{warp}(I_j, D_j, \mathbf{P}_{ij}) \right\|_1$$

여기서 $I_i, I_j$는 인접 프레임, $D_j$는 예측된 깊이 맵, $\mathbf{P}_{ij}$는 상대적 카메라 변환입니다.

---

### 2-3. 모델 구조

1B 파라미터 체크포인트, 텍스트 정렬 변형(text-aligned variant), Hugging Face 데모가 포함된 릴리즈로, arXiv 논문 2605.15195와 연결되어 있습니다.

```
입력: RGB 이미지 시퀀스 {I_1, ..., I_F}
         ↓
[DINOv2 기반 이미지 토크나이저]
         ↓
[Frame-wise Self-Attention] (프레임 내부)
         ↓
[Register Attention] (프레임 간 → Register 토큰 경유)
         ↓
[단일 Dense Prediction Head (Multi-task)]
         ↓
출력: 깊이 맵 D, 카메라 포즈 P, 기하 피처 F
```

이 모델은 카메라 외인수(extrinsics), 내인수(intrinsics), 깊이, 포인트 클라우드 재구성을 소규모 이미지 스택이나 비디오 프레임에서 필요로 하는 워크플로를 목표로 합니다.

256 해상도 체크포인트는 `predictions["text_alignment_embedding"]`을 노출하여, 언어 조건부 검색이나 레이블링과 기하를 연결하는 실험에 유용합니다.

---

### 2-4. 성능 향상

이 변경 사항들은 이전 작업보다 20배 많은 지도 데이터와 100배 많은 비지도 데이터로 VGGT- $\Omega$를 효율적으로 훈련할 수 있게 하면서, VGGT 메모리의 30%만 필요하고 추론 속도는 1.6배 빠릅니다.

결과적으로 VGGT- $\Omega$는 Sintel 데이터셋에서 카메라 추정 정확도를 77% 향상시키는 등, 정적 및 동적 장면 모두에서 다양한 벤치마크에 걸쳐 3D 재구성의 새로운 최신 기술 수준을 달성합니다.

| 지표 | VGGT (기존) | VGGT- $\Omega$ |
|------|------------|----------------|
| GPU 메모리 | 100% | **~30%** |
| 추론 속도 | 1.0× | **1.6×** 빠름 |
| 지도 학습 데이터 | 1× | **20×** |
| 비지도 데이터 | 1× | **100×** |
| Sintel 카메라 추정 | 기준선 | **+77% 향상** |

---

### 2-5. 한계

VRAM 사용량이 입력 프레임 수와 함께 빠르게 증가하여, 대규모 멀티뷰 배치에서는 소형 GPU를 사용할 수 없습니다.

이 레포지토리는 명시적 전역 최적화 또는 검증된 포토그래메트리 제어가 필요할 때 고전적인 SfM 스택을 대체하지 않습니다. 텍스트 정렬 변형은 256 해상도로 낮기 때문에, 언어 정렬보다 기하 충실도(geometry fidelity)가 더 중요한 경우 적합하지 않습니다.

고해상도 이미지와 대규모 세트에서는 한계가 있으며, 이미지 수가 많아질수록 포즈 신뢰도가 저하됩니다.

---

## 3. 🌐 모델의 일반화 성능 향상 가능성

이 부분은 VGGT- $\Omega$의 가장 중요한 이론적 기여 중 하나입니다.

### 3-1. 스케일링 법칙을 통한 일반화

이 모델들의 정확도와 견고성이 모델 용량과 데이터 크기에 따라 예측 가능하게 스케일된다는 것을 추가로 입증합니다.

이는 NLP에서 GPT 계열 모델들이 보인 **스케일링 법칙(Scaling Law)**과 유사한 패턴이 3D 재구성 분야에서도 성립함을 최초로 체계적으로 증명한 것입니다:

$$\text{Reconstruction Quality} \propto f\!\left(\log(\text{Model Size}),\, \log(\text{Data Size})\right)$$

### 3-2. Register 토큰의 전이 가능성

학습된 레지스터들이 비전-언어-행동(Vision-Language-Action) 모델을 개선하고 언어와의 정렬을 지원할 수 있음을 보여주며, 이는 재구성이 공간적 이해를 위한 강력하고 스케일 가능한 프록시 태스크(proxy task)가 될 수 있음을 시사합니다.

이는 중요한 시사점을 가집니다:

- **재구성 → 사전 학습 태스크로서의 가능성**: 재구성을 통해 학습된 표현(representation)이 로봇 조작, VQA, 장면 이해 등에 직접 활용될 수 있음
- **언어 정렬**: `text_alignment_embedding`을 통해 언어와 기하 구조 사이의 시맨틱 연결 가능

### 3-3. 자기지도 학습을 통한 도메인 일반화

VGGT- $\Omega$ 이전에는 feed-forward 재구성이 소규모 지도 학습 훈련에 갇혀 있었지만, 이후에는 대규모 데이터셋과 비레이블 비디오로 스케일업되어, 재구성이 언어 모델과 유사한 스케일링 법칙을 따름을 증명합니다.

이 문제는 재구성이 구현형 AI(embodied AI)를 위한 사전 훈련 태스크로 부상하는 중요한 시점에 있으므로, 그것이 스케일됨을 증명하는 것은 단순히 더 나은 깊이 맵 이상의 의미가 있습니다.

### 3-4. 정적 → 동적 장면으로의 일반화

VGGT- $\Omega$는 정적 및 동적 장면 모두에서 재구성 정확도, 효율성, 능력을 크게 향상시킵니다. 이 모델을 전례 없는 규모에서 훈련할 수 있도록, 훈련 효율성을 향상시키는 아키텍처 변경, 동적 장면을 지원하는 고품질 데이터 어노테이션 파이프라인, 그리고 자기지도 학습 프로토콜을 도입합니다.

---

## 4. 🔮 미래 연구에 미치는 영향 및 고려사항

### 4-1. 연구에 미치는 영향

#### (1) 3D 재구성의 패러다임 전환 확인
이 모델들은 단일 또는 스테레오 이미지에서 포인트 클라우드를 직접 예측하는 엔드-투-엔드 패러다임을 따르며, 이는 희소 재구성에 이어 밀집 재구성을 수행하는 전통적인 두 단계 프로세스를 우회하고 폐색(occlusion)에 대한 견고성을 향상시킵니다.

#### (2) 구현형 AI(Embodied AI)를 위한 공간 기반 모델 가능성
VGGT- $\Omega$는 register attention으로 메모리 효율적인 feed-forward 3D 재구성을 동적 장면으로 스케일링하여 대규모 공간 기반 모델링(spatial foundation modeling)을 가능하게 합니다. 또한 학습된 기하 표현이 재구성 이상으로 전이되는지를 연구합니다.

#### (3) 자기지도 사전학습의 확장
비레이블 비디오를 대규모로 활용할 수 있음을 증명함으로써, 향후 인터넷 규모의 비디오 데이터를 활용한 3D 사전학습 연구를 촉진합니다.

### 4-2. 향후 연구 시 고려할 점

| 분야 | 고려사항 |
|------|---------|
| **아키텍처** | Register 수($K$)의 최적화 — 너무 적으면 정보 병목, 너무 많으면 메모리 증가 |
| **데이터** | 동적 장면 어노테이션 파이프라인의 확장 및 도메인 다양성 확보 |
| **평가** | 항공 이미지, 의료 영상 등 Out-of-Distribution 도메인에서의 일반화 체계적 평가 필요 |
| **언어 통합** | 기하-언어 정렬 품질 향상 및 고해상도 지원 (현재 256px 제한) |
| **추론 효율** | 대규모 프레임(500+) 처리 시 VRAM 요구량 추가 절감 연구 |
| **다운스트림** | VLA, SLAM, NeRF/Gaussian Splatting과의 통합 연구 |

---

## 5. 📊 2020년 이후 관련 최신 연구 비교 분석

DUSt3R, MASt3R, VGGT와 같이 최근 개발된 3D 재구성을 위한 기반 모델들은 매우 희소한 이미지 오버랩을 처리하는 능력과 일반화 능력으로 상당한 주목을 받고 있습니다.

| 모델 | 발표 | 핵심 방식 | 동적 장면 | 후처리 필요 | 주요 한계 |
|------|------|----------|----------|------------|----------|
| **DUSt3R** | CVPR 2024 | 스테레오 쌍 → 포인트맵 직접 예측, Transformer | ❌ | 전역 최적화 필요 | 쌍 기반 처리로 확장성 제한 |
| **MASt3R** | 2024 | DUSt3R + 매칭 특화 | ❌ | 전역 최적화 필요 | 고해상도 한계 |
| **MonST3R** | ICLR 2025 | DUSt3R를 동적 비디오에 fine-tune | ✅ (제한적) | 필요 | DUSt3R 기반 쌍 구조의 한계 상속 |
| **VGGT** | CVPR 2025 Best Paper | 단일 feed-forward pass로 전체 멀티뷰 처리 | ❌ | ❌ (선택적) | 전역 어텐션 메모리 폭발 |
| **VGGT- $\Omega$ ** | **CVPR 2026 Oral** | Register Attention + 자기지도 + 동적 지원 | ✅ | ❌ | 고해상도/대규모 프레임 VRAM |

VGGT는 DUSt3R가 사용하는 비용이 큰 반복 후최적화를 제거하는 feed-forward 신경망으로 파이프라인을 추가 발전시킵니다. 결과적으로 VGGT는 속도와 품질 모두에서 DUSt3R와 MASt3R를 능가할 수 있습니다.

MonST3R는 비디오 시퀀스에서 DUSt3R를 파인튜닝하는 반면, D2USt3R는 4D 포인트맵 표현과 교차 프레임 어텐션 메커니즘을 통한 명시적 시간적 모델링을 도입하여 프레임 간 움직이는 객체들 사이의 대응 관계 설정을 개선합니다.

DUSt3R 기반 접근법들이 동적 콘텐츠에서 진전을 보이지만, 이들 모두 DUSt3R의 쌍 기반 처리 프레임워크에 의해 제약됩니다. 이 점에서 VGGT- $\Omega$의 전역적 멀티뷰 처리 방식이 근본적인 차별점을 가집니다.

---

## 📚 참고 자료 및 출처

| # | 제목 | 출처 |
|---|------|------|
| 1 | **VGGT- $\Omega$ 공식 프로젝트 페이지** | https://vggt-omega.github.io/ |
| 2 | **arXiv 논문 (2605.15195)** | https://arxiv.org/abs/2605.15195 |
| 3 | **CVPR 2026 Oral 발표 페이지** | https://cvpr.thecvf.com/virtual/2026/oral/40381 |
| 4 | **CVPR 2026 Poster 페이지** | https://cvpr.thecvf.com/virtual/2026/poster/39730 |
| 5 | **GitHub 공식 레포지토리** | https://github.com/facebookresearch/vggt-omega |
| 6 | **Hugging Face 모델 카드** | https://huggingface.co/facebook/VGGT-Omega |
| 7 | **Oxford VGG 연구 페이지** | https://www.robots.ox.ac.uk/~vedaldi/research/2026/vggt-omega/ |
| 8 | **VGGT 원본 논문 (CVPR 2025 Best Paper)** | https://github.com/facebookresearch/vggt |
| 9 | **DUSt3R/MASt3R/VGGT 비교 평가 논문** | https://arxiv.org/abs/2507.14798 |
| 10 | **Review of Feed-forward 3D Reconstruction: From DUSt3R to VGGT** | https://www.coscipress.com/journal/JAICS/article/07e56774e440bb36c2a79d1f7a1ab815 |
| 11 | **VGGT- $\Omega$ 분석 블로그 (Alan Hou)** | https://alanhou.org/blog/arxiv-vggt-/ |
| 12 | **ToolHunter VGGT- $\Omega$ 리뷰** | https://toolhunter.cc/tools/vggt-omega |
| 13 | **ResearchGate - VGGT- $\Omega$ 요약** | https://www.researchgate.net/publication/404891398_VGGT-O |
| 14 | **PAGE-4D 논문 (관련 연구)** | https://arxiv.org/pdf/2510.17568 |
| 15 | **VGGT ArXiv 전문 PDF (CVPR 2025)** | https://arxiv.org/pdf/2503.11651 |

> ⚠️ **주의사항**: 본 분석은 공개된 arXiv 초록, 프로젝트 페이지, 공식 GitHub, CVPR 페이지를 기반으로 작성되었습니다. 논문 전문(full paper)에만 존재하는 세부 수식(예: 정확한 레지스터 수 $K$, 정밀한 손실 가중치 $\lambda$ 값 등)은 공개 정보 범위 내에서만 기술하였으며, 일부 수식은 공개된 설명을 바탕으로 재구성한 것임을 명확히 밝힙니다.
