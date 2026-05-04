
# Fast3R: Towards 3D Reconstruction of 1000+ Images in One Forward Pass

> **논문 정보**
> - **저자**: Jianing Yang, Alexander Sax, Kevin J. Liang, Mikael Henaff, Hao Tang, Ang Cao, Joyce Chai, Franziska Meier, Matt Feiszli (Meta FAIR & University of Michigan)
> - **학회**: CVPR 2025 (pages 21924–21935)
> - **arXiv**: [2501.13928](https://arxiv.org/abs/2501.13928) (v1: 2025.01.23, v2: 2025.03.19)
> - **공식 코드**: [github.com/facebookresearch/fast3r](https://github.com/facebookresearch/fast3r)
> - **프로젝트 페이지**: [fast3r-3d.github.io](https://fast3r-3d.github.io/)

---

## 1. 핵심 주장 및 주요 기여 요약

멀티뷰 3D 재구성(Multi-view 3D reconstruction)은 컴퓨터 비전의 핵심 난제로, DUSt3R과 같은 현재 선도적 방법들은 근본적으로 **쌍(pairwise) 방식**으로 이미지를 처리하며, 다중 뷰를 재구성하기 위해 비용이 큰 전역 정렬(global alignment) 절차를 필요로 한다.

이에 대응하여 본 논문은 **Fast3R(Fast 3D Reconstruction)**을 제안한다. 이는 DUSt3R의 새로운 멀티뷰 일반화로서, 많은 뷰를 병렬로 처리하여 효율적이고 확장 가능한 3D 재구성을 달성한다. Fast3R의 **Transformer 기반 아키텍처**는 N개의 이미지를 단일 순전파(single forward pass)로 처리하여 반복적 정렬의 필요성을 우회하며, 추론 속도 향상과 오류 누적 감소를 통해 SOTA 성능을 입증한다.

### 🔑 주요 기여 요약

| 기여 항목 | 내용 |
|---|---|
| **Pairwise 병목 제거** | 쌍 기반 처리 → 전체 N개 뷰 동시 처리 |
| **단일 순전파 재구성** | 1000+ 이미지를 1회 forward pass로 처리 |
| **오류 누적 제거** | 글로벌 정렬 단계 완전 제거 |
| **확장 가능한 인퍼런스** | >250 FPS, 단일 A100에서 1500장 처리 |
| **4D 확장** | 아키텍처 변경 없이 동적 장면 재구성 가능 |

---

## 2. 해결하고자 하는 문제 / 제안 방법 / 모델 구조 / 성능 및 한계

### 2-1. 해결하고자 하는 문제

거의 모든 현대 3D 재구성 접근법은 전통적인 **다중뷰 기하학(MVG) 파이프라인**에 기반한다. MVG 기반 방법들은 먼저 이미지 쌍 간의 대응 픽셀을 식별한 후 카메라 모델과 사영 다중뷰 기하학을 사용하여 3D 포인트로 리프팅한다. 이 과정은 순차적 단계(피처 추출 → 쌍 간 이미지 대응 → 삼각측량 → 전역 번들 정렬)로 진행된다. 그러나 파이프라인 방식은 오류 누적에 취약하며, 순차적 특성이 병렬화를 막아 속도와 확장성을 제한한다.

Fast3R은 대부분의 기존 3D 재구성 방법에서 오래 지속된 **2뷰 아키텍처 설계에서 탈피**하여 모든 뷰를 함께 처리한다. 그 결과 시간과 메모리를 많이 소비하는 전통적인 뷰 선택 및 전역 정렬 단계가 제거되고, 이 모든 것이 단일 통합 이미지-to-3D 모델 안에서 **엔드-투-엔드 학습 가능**해져 속도와 메모리가 극적으로 향상된다.

---

### 2-2. 제안하는 방법 (수식 포함)

#### (1) 포인트맵(Pointmap) 예측

입력 이미지 $\mathbf{I} \in \mathbb{R}^{N \times H \times W \times 3}$이 주어졌을 때, Fast3R은 대응하는 포인트맵 $\mathbf{X} \in \mathbb{R}^{N \times H \times W \times 3}$를 예측함으로써 장면의 3D 구조를 재구성한다. 포인트맵은 이미지의 픽셀에 의해 인덱싱된 3D 위치의 집합이다.

$$\mathbf{I} \in \mathbb{R}^{N \times H \times W \times 3} \xrightarrow{\text{Fast3R}} \mathbf{X} \in \mathbb{R}^{N \times H \times W \times 3}$$

#### (2) 로컬/글로벌 포인트맵 및 신뢰도 맵 예측

Fast3R은 별도의 DPT-L 디코더 헤드를 사용하여 토큰을 로컬 및 글로벌 포인트맵 $(\mathbf{X}_L, \mathbf{X}_G)$와 신뢰도 맵 $(\Sigma_L, \Sigma_G)$로 매핑한다.

$$\hat{\mathbf{X}}_L^{(i)}, \hat{\mathbf{X}}_G^{(i)}, \hat{\Sigma}_L^{(i)}, \hat{\Sigma}_G^{(i)} = \text{Head}(f_i), \quad \forall i \in \{1, \ldots, N\}$$

여기서:
- $\hat{\mathbf{X}}_L^{(i)}$: $i$번째 뷰의 **로컬** 좌표계 포인트맵
- $\hat{\mathbf{X}}_G^{(i)}$: $i$번째 뷰의 **글로벌** 좌표계 포인트맵
- $\hat{\Sigma}_L^{(i)}, \hat{\Sigma}_G^{(i)}$: 로컬/글로벌 신뢰도 맵

#### (3) 학습 손실 함수 (DUSt3R 기반 Confidence-Weighted Regression Loss)

DUSt3R 패러다임을 따라 신뢰도 가중 포인트맵 회귀 손실을 사용한다:

$$\mathcal{L} = \sum_{i=1}^{N} \sum_{p} \left[ \frac{1}{\Sigma_L^{(i,p)}} \| \hat{\mathbf{X}}_L^{(i,p)} - \mathbf{X}_L^{*(i,p)} \|_2 + \log \Sigma_L^{(i,p)} + \frac{1}{\Sigma_G^{(i,p)}} \| \hat{\mathbf{X}}_G^{(i,p)} - \mathbf{X}_G^{*(i,p)} \|_2 + \log \Sigma_G^{(i,p)} \right]$$

여기서 $\mathbf{X}^*$는 GT(ground-truth) 포인트맵이고, $\Sigma$는 예측 신뢰도를 나타낸다.

#### (4) 이미지 인덱스 위치 임베딩 (Train-Short, Test-Long)

**무작위화된 위치 인덱스 임베딩(Randomized Positional Index Embedding)** 전략을 통해, Fast3R은 20개 뷰로 훈련하면서 추론 시 1000+개 뷰로 일반화한다. 이는 LLM 스타일 PE 방식에서 흔히 발생하는 외삽(extrapolation) 문제를 회피한다.

$$\text{PE}(i) = \text{Embed}(\tilde{i}), \quad \tilde{i} \sim \text{Uniform}\{1, \ldots, N_{\max}\}$$

훈련 시 이미지 인덱스 임베딩을 샘플링하지 않으면, 훈련 때보다 많은 뷰로 테스트할 때 회귀 손실이 급증(orange)한다. 이 임베딩 전략은 훈련 시 본 것보다 더 많은 뷰에서도 비교 가능한 성능을 발휘한다.

---

### 2-3. 모델 구조

Fast3R 모델은 1000+개의 이미지를 단일 순전파로 처리하여 3D 포인트 클라우드를 생성하고 카메라 포즈를 추정하는 **인코더-디코더-헤드 아키텍처**를 사용한다. 모델은 세 가지 주요 컴포넌트로 구성된다: (1) 인코더(입력 이미지 → 피처 표현), (2) 디코더(다중 뷰 피처 → 일관된 표현), (3) 헤드(디코딩된 피처 → 3D 포인트 클라우드 + 신뢰도 점수)

#### 상세 아키텍처

```
Input: N images (I₁, I₂, ..., Iₙ)
     ↓
[1] ViT Encoder (CroCo pretrained, 공유 가중치)
    - 각 이미지를 패치 임베딩 + RoPE 위치 인코딩으로 인코딩
     ↓
[2] Fusion Transformer (핵심 모듈)
    - 전체 뷰의 패치 토큰을 연결(concatenation)
    - All-to-All Self-Attention 수행
     ↓
[3] DPT-L Pointmap Head (뷰별 별도 헤드)
    - Local Pointmap (X_L) + Confidence Map (Σ_L)
    - Global Pointmap (X_G) + Confidence Map (Σ_G)
     ↓
Output: 3D Pointmaps + Camera Poses (PnP 사용)
```

Fast3R의 대부분의 계산은 **퓨전 트랜스포머(Fusion Transformer)**에서 이루어지며, 이는 ViT-B 또는 BERT와 유사한 12레이어 트랜스포머를 사용한다(그러나 확장 가능하다). 이 퓨전 트랜스포머는 모든 뷰에서 인코딩된 이미지 패치를 연결(concatenate)하여 입력받아 **all-to-all 자기 주의(self-attention)**를 수행한다. 이 연산은 Fast3R에 쌍 정보만으로는 얻을 수 없는 **전체 문맥(full context)**을 제공한다.

Fast3R은 핵심적으로 뷰 간 정보를 융합하는 대형 Transformer를 사용하며, 효율적이고 확장 가능한 처리를 위해 다음과 같은 LLM 훈련/추론 기술을 활용한다: **FlashAttention 2.0**(메모리 효율적 어텐션 계산), **DeepSpeed ZeRO-2**(분산 훈련 최적화), **위치 임베딩 보간(Positional Embedding Interpolation)**("train short, test long"), **Tensor Parallelism**(멀티 GPU 가속 추론)

---

### 2-4. 성능 향상

CO3Dv2에서 Fast3R은 포즈 추정에서 15도 이내 정확도 **99.7%**를 달성하였으며, 이는 전역 정렬을 사용하는 DUSt3R 대비 **14배 이상의 오류 감소**이다.

Fast3R은 초당 251.1 FPS($108 \times 224 \times 224$ 이미지 기준)를 달성하며 단일 A100에서 1500개의 뷰를 1회 통과로 처리한다. DUSt3R은 32개 뷰 이후 Out-of-Memory(OOM) 상태가 된다.

실험 결과는 DUSt3R 대비 최대 320배, MASt3R 대비 최대 1000배 더 빠른 추론 속도로 카메라 포즈 추정에서 SOTA 또는 경쟁력 있는 성능을 보였으며, 훈련 및 추론 시 더 많은 뷰를 사용할수록 명확한 확장성 이점이 있다.

3D 재구성 품질에서도 7-Scenes, DTU, NRGBD 기준으로 DUSt3R 및 Spann3R 대비 경쟁력 있거나 더 나은 3D 품질을 보이면서도 **300배 이상의 처리량**을 달성한다.

#### 성능 비교 요약표

| 방법 | 처리 방식 | CO3Dv2 정확도(@15°) | 최대 처리 뷰 수 | 속도 |
|---|---|---|---|---|
| DUSt3R (CVPR 2024) | Pairwise + Global Align | ~71% | ~32 (OOM) | 기준 |
| MASt3R (2024) | Pairwise + 매칭 헤드 | 높음 | ~32 (OOM) | 느림 |
| Spann3R (2024) | 순차적 + 공간 메모리 | 낮음 | 많음 | 중간 |
| MV-DUSt3R+ (CVPR 2025) | 멀티뷰 디코더 | 높음 | ~24 | 빠름 |
| **Fast3R (CVPR 2025)** | **All-to-All Transformer** | **99.7%** | **1500+** | **~251 FPS** |

---

### 2-5. 한계점

현재 확장의 제한 요인은 **데이터의 정확도와 양**일 수 있다.

저자들은 데이터 품질과 장면 크기와 관련된 한계를 논의하며, 동적 장면과 합성 데이터 증강을 포함한 향후 연구 방향을 제시한다.

- **대규모 야외/무한 장면**: 매우 큰 장면에서의 글로벌 일관성 유지 어려움
- **메모리 이차 복잡도**: All-to-All Attention은 뷰 수에 대해 $O(N^2)$ 복잡도를 가짐
- **동적 장면**: 기본 모델은 정적 장면 가정에 기반 (파인튜닝으로 일부 해결)
- **라이선스 제약**: 코드와 모델은 **FAIR NC Research License** 하에 제공되어 상업적 활용이 제한됨

---

## 3. 모델의 일반화 성능 향상 가능성

### 3-1. Train-Short, Test-Long 전략

Fast3R 모델 아키텍처는 추론 시 1000개 이상의 이미지로 확장 가능하도록 설계되었으며, 훈련 시에는 이미지 마스킹을 사용하여 훨씬 적은 수로 훈련한다.

카메라 포즈 위치파악 및 재구성 작업에서 모델은 점진적으로 더 큰 뷰 세트로 훈련할 때 성능이 향상되며, 뷰별 정확도는 추론 시 더 많은 뷰를 사용할수록 더 향상되고, **훈련 시 본 것보다 훨씬 많은 뷰로 일반화**할 수 있다.

### 3-2. 모델 크기 스케일링

퓨전 트랜스포머의 크기를 ViT-Base, ViT-Large, ViT-Huge의 세 가지로 실험한 결과, **더 큰 모델 크기가 카메라 포즈 추정 및 3D 재구성을 포함한 3D 작업에 지속적으로 이점**을 가져다줌을 보여준다.

### 3-3. 데이터 스케일링

4가지 다른 데이터 스케일을 사용한 실험에서 **Fast3R은 더 많은 데이터로부터 지속적인 이익**을 얻음을 보여준다.

Fast3R의 성능은 모델 크기와 데이터 크기가 증가함에 따라 확장되어, 대규모 3D 재구성의 흥미로운 미래를 제시한다.

### 3-4. 도메인 일반화: 동적 장면(4D) 확장

동적 데이터(PointOdyssey, TartanAir)로 동일한 Fast3R 아키텍처를 엔드-투-엔드로 파인튜닝하면, 포인트맵 회귀 목표와 아키텍처를 수정하지 않고도 **4D 재구성**에도 작동함을 보여준다. 중요하게도, 이 방법은 훨씬 빠르게 유지되어 실시간 응용의 가능성을 열어준다.

### 3-5. Transformer 스케일링 법칙의 상속

Fast3R은 단일 순전파에서 공통 참조 프레임의 모든 픽셀에 대한 3D 위치를 예측하는 트랜스포머이다. 전체 SfM 파이프라인을 엔드-투-엔드로 훈련된 범용 아키텍처로 대체함으로써, Fast3R과 유사한 접근법들은 **더 나은 데이터와 증가된 파라미터로 일관되게 향상되는 트랜스포머의 일반적인 스케일링 규칙**으로부터 이점을 얻어야 한다. Fast3R은 전역 어텐션을 사용하므로 기존 시스템의 병목으로 인한 두 가지 인위적인 확장 한계를 피한다.

---

## 4. 앞으로의 연구에 미치는 영향 및 고려할 점

### 4-1. 앞으로의 연구에 미치는 영향

#### ① 3D 재구성의 패러다임 전환
Fast3R은 멀티뷰 3D 재구성에서의 **쌍 기반 병목을 타파**한다. DUSt3R을 기반으로, 1000+개의 뷰를 단일 순전파로 처리하며 포즈가 없는(unposed) 비순서 RGB 이미지로부터 밀집 3D 포인트맵을 직접 회귀하는 Transformer 기반 아키텍처를 도입한다.

#### ② 다운스트림 태스크 가속
Fast3R 출력의 다운스트림 Novel View Synthesis 태스크에 대한 활용 가능성을 정성적으로 보여주며, **InstantSplat 파이프라인**을 채택한 Gaussian Splatting 생성을 시각화한다.

#### ③ 실시간 3D 응용의 길 열기
Fast3R의 역량은 벤치마크를 훨씬 뛰어넘는다. 증강 현실에서는 실시간 3D 모델링이 가상 환경과 상호작용 방식을 재정의할 수 있으며, 로보틱스에서도 카메라 포즈 추정의 정확도가 더 나은 내비게이션과 물체 조작을 보장한다.

#### ④ DUSt3R 패밀리 발전에 영향
DUSt3R과 MASt3R은 뛰어난 일반화를 보이는 기하학적 3D 비전을 위한 유망한 기초 모델로 부상했다. 이러한 연구 발전에 영감을 받아 커뮤니티는 **VGGT, Fast3R, Spann3R, MonST3R** 등 여러 후속 연구를 발전시켰다.

---

### 4-2. 앞으로 연구 시 고려할 점

#### ① 메모리 및 계산 복잡도 개선
All-to-All Attention의 $O(N^2)$ 복잡도는 뷰 수 증가에 따른 근본적 한계다. **선형 어텐션(Linear Attention), Sparse Attention, 또는 계층적 어텐션** 방식을 도입하여 더 효율적인 확장 방식을 탐구해야 한다:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V \quad [\text{복잡도: } O(N^2 d)]$$

$$\downarrow \quad \text{Linear Attention 또는 Sparse Attention으로 대체}$$

#### ② 대규모 야외 장면으로의 일반화
현재 학습 데이터는 주로 실내 및 객체 중심 장면에 집중되어 있다. 항공 사진, 대규모 도시 재구성 등 **대규모 야외 시나리오**로의 일반화를 위한 데이터셋 확장과 스케일 일관성 처리 방법이 필요하다.

#### ③ 동적/비정적 장면으로의 확장
MonST3R은 DUSt3R을 기반으로 동적 장면 재구성을 다루는 동시 연구이나, DUSt3R과 마찬가지로 쌍 기반 아키텍처를 가정하고 광학 흐름(optical flow) 예측을 위한 별도 모델을 사용한다. Fast3R의 아키텍처가 4D로 확장 가능함을 입증했지만, 빠르게 움직이는 물체나 비강체(non-rigid) 변형이 있는 환경에서의 견고성 연구가 더 필요하다.

#### ④ 데이터 품질 및 다양성
현재 확장의 제한 요인은 데이터의 정확도와 양일 수 있다. 향후 연구에서는 **합성 데이터 증강**, 더 다양한 카메라 타입(어안렌즈, 파노라마 등), 및 약지도(weakly supervised) 학습 방법을 통한 데이터 효율성 개선이 중요하다.

#### ⑤ 번들 조정과의 결합
필수적이지는 않지만, 추론 시 번들 조정(bundle adjustment)을 사용하면 Fast3R의 성능을 향상시킬 수 있다. **Gaussian Splatting을 이용한 번들 조정(GS-BA)** 예시를 통해, InstantSplat을 사용하여 포인트 클라우드로부터 초기화를 활용해 가우시안 집합을 최적화하고, 위치와 포즈를 업데이트하여 재투영 오류를 최소화한다.

#### ⑥ 라이선스 및 접근성 문제
현재 코드와 모델은 FAIR NC Research License 하에 제공되어 상업적 응용에 제한이 있다. 오픈 라이선스 모델 개발이나 경량화(distillation) 연구를 통해 더 넓은 접근성을 확보하는 방향도 중요한 연구 과제이다.

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

**DUSt3R (CVPR 2024)**: 임의의 이미지 컬렉션에 대한 밀집 비제약 스테레오 3D 재구성의 새로운 패러다임을 도입했다. 카메라 보정 및 뷰포인트 포즈에 대한 사전 정보 없이 작동하며, 쌍 재구성 문제를 포인트맵 회귀로 변환하여 일반 카메라 모델의 강한 제약을 완화한다.

**Spann3R (2024)**: DUSt3R 패러다임에 기반하여, 장면이나 카메라 파라미터에 대한 사전 지식 없이 이미지에서 직접 포인트맵을 회귀하는 Transformer 기반 아키텍처를 사용한다. 각 이미지 쌍의 로컬 좌표 프레임으로 표현된 포인트맵을 예측하는 DUSt3R와 달리, Spann3R은 **전역 좌표계**에서 뷰별 포인트맵을 예측하여 최적화 기반 전역 정렬이 필요 없다. 핵심 아이디어는 이전의 모든 관련 3D 정보를 추적하는 외부 공간 메모리(spatial memory)를 관리하는 것이다.

**MASt3R (2024)**: DUSt3R에서 영감을 받은 여러 후속 연구 중 하나로, 각 디코더 출력에 **로컬 피처 헤드**를 추가하여 매칭 능력을 강화한다.

**MV-DUSt3R+ (CVPR 2025 Oral)**: 더 많은 뷰 처리, 오류 감소, 추론 시간 향상을 위해 **단일 단계 피드포워드 네트워크**를 제안한다. 핵심은 참조 뷰를 고려하면서 임의 수의 뷰에 걸쳐 정보를 교환하는 멀티뷰 디코더 블록이다. MV-DUSt3R+는 참조 뷰 선택에 강건하도록 다른 참조 뷰 선택 간 정보를 융합하는 크로스 참조 뷰 블록을 사용한다.

| 방법 | 연도 | 핵심 아이디어 | 최대 뷰 수 | 전역 정렬 필요 여부 |
|---|---|---|---|---|
| DUSt3R | CVPR 2024 | Pairwise Pointmap Regression | ~32 | ✅ 필요 |
| Spann3R | 2024 | 공간 메모리 기반 순차 처리 | 많음 | ❌ 불필요 |
| MASt3R | 2024 | DUSt3R + 로컬 피처 헤드 | ~32 | ✅ 필요 |
| MonST3R | 2024 | 동적 장면 특화 | ~32 | ✅ 필요 |
| MV-DUSt3R+ | CVPR 2025 | 멀티뷰 디코더 블록 | ~24 | ❌ 불필요 |
| **Fast3R** | **CVPR 2025** | **All-to-All Fusion Transformer** | **1500+** | **❌ 불필요** |

---

## 📚 참고 자료

1. **Fast3R 논문 (arXiv)**: Jianing Yang et al., "Fast3R: Towards 3D Reconstruction of 1000+ Images in One Forward Pass," arXiv:2501.13928, 2025. — https://arxiv.org/abs/2501.13928
2. **Fast3R 논문 (CVPR 2025 Open Access)**: https://openaccess.thecvf.com/content/CVPR2025/papers/Yang_Fast3R_Towards_3D_Reconstruction_of_1000_Images_in_One_Forward_CVPR_2025_paper.pdf
3. **Fast3R 프로젝트 페이지**: https://fast3r-3d.github.io/
4. **Fast3R GitHub (Meta FAIR)**: https://github.com/facebookresearch/fast3r
5. **Fast3R HTML 논문 전문 (arXiv v1)**: https://arxiv.org/html/2501.13928v1
6. **Fast3R HTML 논문 전문 (arXiv v2)**: https://arxiv.org/html/2501.13928v2
7. **Fast3R Hugging Face 논문 페이지**: https://huggingface.co/papers/2501.13928
8. **Fast3R CVPR 2025 Open Access (IEEE Xplore)**: https://ieeexplore.ieee.org/document/11093261
9. **Fast3R 모델 아키텍처 분석 (DeepWiki)**: https://deepwiki.com/facebookresearch/fast3r/2-model-architecture
10. **Fast3R 소개 (OpenCV.org, CVPR 2025)**: https://opencv.org/fast3r/
11. **Fast3R 분석 블로그 (Disruptive Concepts)**: https://disruptive-concepts.com/2025/01/how-fast3r-is-shaping-the-new-frontier-of-3d-imaging/
12. **DUSt3R 논문 (CVPR 2024)**: Wang et al., "DUSt3R: Geometric 3D Vision Made Easy," CVPR 2024. — https://europe.naverlabs.com/research/publications/dust3r-geometric-3d-vision-made-easy/
13. **MV-DUSt3R+ 논문 (CVPR 2025 Oral)**: Tang et al., "MV-DUSt3R+: Single-Stage Scene Reconstruction from Sparse Views In 2 Seconds," CVPR 2025. — https://mv-dust3rp.github.io/
14. **MASt3R 상세 분석 (LearnOpenCV)**: https://learnopencv.com/mast3r-sfm-grounding-image-matching-3d/
15. **Awesome-DUSt3R 리포지토리**: https://github.com/ruili3/awesome-dust3r
16. **ScienceStack Fast3R 요약**: https://www.sciencestack.ai/paper/2501.13928v2
