# FastVGGT: Training-Free Acceleration of Visual Geometry Transformer

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

FastVGGT는 3D 비전 기반 모델인 VGGT(Visual Geometry Grounded Transformer)의 **추론 시간 비효율 문제를 학습 없이(Training-Free)** 해결하는 프레임워크입니다. 핵심 주장은 다음과 같습니다:

> "VGGT의 Global Attention에서 발생하는 **토큰 붕괴(Token Collapse) 현상**이 계산 중복을 야기하며, 이를 **전략적 토큰 병합(Token Merging)**으로 제거하면 4배 속도 향상과 동시에 오류 누적 완화가 가능하다."

### 주요 기여 (3가지)

| 기여 | 설명 |
|------|------|
| ① 병목 분석 | VGGT의 Global Attention이 주요 추론 병목임을 프로파일링으로 규명 |
| ② 최초 적용 | 3D 피드-포워드 기하 모델에 Token Merging을 **최초로** 도입 |
| ③ 성능 검증 | 1000장 입력 기준 4× 속도 향상, 재구성 품질 유지, 오류 누적 완화 |

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

VGGT는 두 가지 핵심 병목을 가집니다:

**병목 1: 이차 시간 복잡도**

Flash-Attention은 메모리 복잡도를 줄이지만 시간 복잡도는 여전히:

$$\mathcal{O}(N^2 \cdot d)$$

여기서 $N$은 전체 토큰 수, $d$는 특징 차원입니다. 프레임 수 증가 시 Global Attention 비용이 폭발적으로 증가합니다 (Figure 2: 200 프레임에서 Global Attention이 26,152ms).

**병목 2: 오류 누적(Error Accumulation)**

장시간 시퀀스에서 토큰 공간이 확장될수록 미세한 부정확도가 증폭되어 **예측 드리프트(Prediction Drift)**가 발생합니다.

**관찰된 Token Collapse 현상:**

Figure 3에서 다양한 블록과 토큰에 걸쳐 attention 패턴이 **높은 유사성**을 보임 → Global Attention에 상당한 중복 계산이 존재함을 의미합니다.

---

### 2.2 제안하는 방법 (수식 포함)

#### 핵심 아이디어: 3D 특화 Token Merging

토큰을 세 범주로 분류합니다:
- **Salient Token**: 각 프레임의 가장 독특한 특징 보유 (약 10%)
- **Destination (Dst) Token**: 대표 앵커 역할
- **Source (Src) Token**: 중복 정보를 담은 병합 대상

#### Step 1: Reference Token Selection (참조 토큰 선택)

VGGT는 첫 번째 프레임을 **월드 좌표계(World Coordinate System)**로 정의합니다. 따라서 첫 프레임의 모든 토큰을 Dst 토큰으로 지정합니다:

$$\mathcal{D}_{\text{ref}} = \{\text{all tokens from frame } f_1\}$$

#### Step 2: Salient Token Selection (현저 토큰 선택)

효율성을 위해 고정 보폭 샘플링(fixed-stride sampling)으로 각 프레임에서 약 10%의 토큰을 선택합니다:

$$\mathcal{S} = \{t_i \mid i \equiv 0 \pmod{K}\}, \quad |\mathcal{S}| \approx 0.1 \cdot N_{\text{frame}}$$

상위-k 노름 기반 선택도 가능하지만, 계산 비용이 시퀀스 길이에 비례하므로 고정 보폭이 기본값입니다.

#### Step 3: Region-Based Random Sampling (균일 토큰 샘플링)

각 프레임 내 토큰을 2D 그리드로 배열하고, 각 셀 내에서 Dst와 Src를 균일하게 배분합니다. 이는 ToMeSD [1]에서 영감을 얻었습니다:

$$\mathcal{D}_{\text{frame}} \leftarrow \text{stride-}K \text{ sampling within each grid cell}$$
$$\mathcal{P}_{\text{src}} = \mathcal{T}_{\text{all}} \setminus (\mathcal{D}_{\text{ref}} \cup \mathcal{S} \cup \mathcal{D}_{\text{frame}})$$

#### Step 4: Token Merging Procedure (토큰 병합)

각 Src 토큰 $x_s \in \text{src}$와 모든 Dst 토큰 $x_d \in \text{dst}$ 간의 **코사인 유사도** 계산:

$$\text{sim}(x_s, x_d) = \frac{x_s \cdot x_d}{\|x_s\| \|x_d\|}$$

가장 유사한 Dst 토큰으로 병합 (평균 풀링):

$$x'_d = \frac{x_d + x_s}{2}$$

이를 통해 Global Attention의 실질적 시퀀스 길이가 줄어들어:

$$\mathcal{O}(N^2 \cdot d) \rightarrow \mathcal{O}\!\left(\left(\frac{N}{r}\right)^2 \cdot d\right)$$

여기서 $r$은 병합 비율(merging ratio)입니다.

#### Step 5: Token Unmerging Procedure (토큰 역병합)

밀집 3D 재구성은 **토큰별 출력(per-token output)**을 필요로 합니다. 두 토큰 $x_1, x_2$가 병합된 경우:

$$x^*_{1,2} = \frac{x_1 + x_2}{2}$$

역병합 시 단순 복제(replication):

$$x'_1 = x^*_{1,2}, \quad x'_2 = x^*_{1,2}$$

이를 통해 디코더는 원래 해상도의 밀집 예측을 수행할 수 있습니다.

---

### 2.3 모델 구조

```
입력 이미지 시퀀스 (N장)
        ↓
   Tokenization (28×37 패치 그리드, 카메라 토큰, 레지스터 토큰)
        ↓
┌─────────────────────────────────────────────────────┐
│ Token Partitioning (5단계)                           │
│  Step 1: 첫 프레임 → Dst Token (참조 좌표계 보존)     │
│  Step 2: Salient Token 추출 (~10%, 병합 제외)        │
│  Step 3: Region-based 랜덤 샘플링 (균일 분포)         │
│  Step 4: Src → Dst 병합 (코사인 유사도 기반)          │
└─────────────────────────────────────────────────────┘
        ↓
   Global Attention (L=24블록, 압축된 시퀀스)
        ↓
   Token Unmerging (원래 해상도 복원)
        ↓
   Frame Attention (프레임 내 지역 정보 복원)
        ↓
   Camera Head / Depth Head (카메라 파라미터, 깊이맵)
```

**VGGT\* (VRAM-Efficient 변형):**
- 원본 VGGT는 24블록 중간 결과를 모두 저장 → 300프레임 이상에서 OOM 발생
- VGGT\*는 실제로 필요한 레이어 4, 11, 17, 23의 출력만 유지 → 1000프레임 처리 가능

---

### 2.4 성능 향상

#### 3D 재구성 (ScanNet-50, Chamfer Distance ↓)

| Method | 1000프레임 CD | 1000프레임 Time | 100프레임 CD | 100프레임 Time |
|--------|--------------|----------------|-------------|----------------|
| $\pi^3$ | OOM | OOM | OOM | OOM |
| StreamVGGT | OOM | OOM | OOM | OOM |
| Fast3R | 0.684 | 397.8s | 0.723 | 4.8s |
| CUT3R | 0.786 | 34.8s | 0.767 | 3.6s |
| VGGT\* | 0.471 | 724.6s | 0.423 | 9.1s |
| **FastVGGT** | **0.425** | **180.7s** | **0.426** | **5.4s** |

**핵심 결과:** 1000프레임에서 VGGT\* 대비 **4× 속도 향상** (724.6s → 180.7s), CD는 오히려 개선 (0.471 → 0.425, 오류 누적 완화 효과)

#### 카메라 포즈 추정 (ScanNet-50)

| 프레임 수 | ATE (Base/Ours) | ARE (Base/Ours) | RPE-rot (Base/Ours) |
|---------|-----------------|-----------------|---------------------|
| 1000 | 0.196 / **0.164** | 4.636 / **3.860** | 0.997 / **0.667** |
| 500 | 0.174 / **0.145** | 4.190 / **3.591** | 0.963 / **0.627** |
| 300 | 0.145 / **0.142** | 3.689 / **3.554** | 0.786 / 0.801 |
| 100 | 0.140 / 0.141 | 3.625 / **3.512** | 1.224 / 1.262 |

장시간 시퀀스에서 **포즈 추정 정확도가 기준선보다 향상**됩니다 (오류 누적 억제).

#### Ablation Study (Token Partitioning 전략)

| 방법 | Uniform | Reference | Salient | CD ↓ | ATE ↓ |
|------|---------|-----------|---------|------|-------|
| (a) | - | - | - | 0.947 | 0.842 |
| (b) | ✓ | - | - | 0.637 | 0.722 |
| (c) | ✓ | ✓ | - | 0.431 | 0.149 |
| (d) | ✓ | ✓ | ✓ | **0.411** | **0.141** |

각 전략이 단계적으로 중요한 기여를 함을 확인할 수 있습니다.

---

### 2.5 한계점

논문에서 명시적으로 언급된 한계와 분석을 통해 도출된 한계:

1. **단방향 의존성:** 첫 번째 프레임을 세계 좌표계로 고정하므로, 첫 프레임 품질에 전체 성능이 크게 의존합니다.
2. **단순 병합 함수:** 현재 Src → Dst 병합이 단순 평균 ($x'_d = (x_d + x_s)/2$)으로, 특징 중요도를 고려하지 않습니다.
3. **역병합의 근사 오류:** 역병합 시 단순 복제를 사용하므로, 개별 토큰의 미세한 지역 정보가 손실될 수 있습니다.
4. **실내 환경 편향:** 평가가 ScanNet, 7-Scenes, NRGBD 등 실내 데이터셋에 집중되어 있어 실외/대규모 장면에서의 일반화가 불확실합니다.
5. **단일 GPU 평가:** NVIDIA A800 80GB 단일 GPU 기준이므로, 다양한 하드웨어 환경에서의 성능 검증이 필요합니다.
6. **고정 병합 비율:** 현재 90% 고정 병합 비율을 사용하며, 장면 복잡도에 따른 적응적 조절이 없습니다.

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 Training-Free 특성에서 비롯된 일반화 이점

FastVGGT는 학습 없이(Training-Free) 적용되므로 **VGGT의 학습된 표현 능력을 그대로 유지**합니다. 이는 다음을 의미합니다:

$$\text{FastVGGT}(\mathcal{D}_{\text{new}}) \approx \text{VGGT}(\mathcal{D}_{\text{new}}) \quad \forall \mathcal{D}_{\text{new}} \notin \mathcal{D}_{\text{train}}$$

즉, 새로운 데이터셋에 대한 재학습 없이도 VGGT 수준의 일반화 성능을 유지합니다.

### 3.2 오류 누적 완화의 일반화 효과

**장시간 시퀀스에서의 일반화:** VGGT의 token collapse 현상이 오류를 누적시키는 반면, FastVGGT의 토큰 병합은:

- **Salient Token 보존**: 고유한 특징 토큰을 유지하여 다양한 시점에서의 매칭 정확도 유지
- **Reference Token 고정**: 일관된 공간 참조 좌표계 유지로 드리프트 억제

이를 통해 학습 데이터에 없는 **더 긴 시퀀스나 새로운 환경**에서도 안정적인 성능을 기대할 수 있습니다.

카메라 포즈 추정에서 1000프레임 기준:
$$\Delta \text{ATE} = 0.196 - 0.164 = 0.032 \quad (\text{약 16.3\% 개선})$$
$$\Delta \text{RPE-rot} = 0.997 - 0.667 = 0.330 \quad (\text{약 33.1\% 개선})$$

### 3.3 다중 도메인 검증을 통한 일반화 근거

세 가지 이질적 데이터셋에서 일관된 성능:

| 데이터셋 | 특성 | FastVGGT 결과 |
|---------|------|--------------|
| ScanNet-50 | 대규모 실내, 다양한 장면 | CD 0.425 (vs VGGT\* 0.471) |
| 7 Scenes | 소규모 실내, 밀집 시퀀스 | Acc 0.018 (≈ VGGT\* 0.019) |
| NRGBD | RGB-D 실내 | NC 0.730 (vs VGGT\* 0.727) |

이 일관성은 특정 도메인에 과적합되지 않았음을 시사합니다.

### 3.4 일반화 성능 향상을 위한 미래 방향성

1. **적응적 병합 비율(Adaptive Merging Ratio):**
   장면 복잡도에 따라 동적으로 $r$을 조절하는 메커니즘:
   
   $$r^* = \arg\min_r \mathbb{E}[\mathcal{L}_{\text{recon}} | r, \mathcal{C}]$$
   
   여기서 $\mathcal{C}$는 장면 복잡도 지표입니다.

2. **도메인 불변 현저 토큰 선택:** 현재 고정 보폭 샘플링을 학습 없이도 장면에 적응 가능한 방식으로 발전시킬 수 있습니다.

3. **실외/대규모 장면 적용:** 현재 실내 데이터셋 중심의 평가를 실외 및 도시 규모 데이터셋으로 확장합니다.

---

## 4. 최신 연구 비교 분석 (2020년 이후)

### 4.1 Feed-Forward 3D 재구성 계보

```
DUSt3R (CVPR 2024)
    ↓ 두 뷰 → 점 지도 직접 회귀
MASt3R (ECCV 2024)
    ↓ 신뢰도 가중 손실, 메트릭 스케일
VGGT (CVPR 2025) ← 1.2B 파라미터, 카메라/깊이/포인트 통합
    ├── FastVGGT (이 논문, 2025) ← 추론 가속화
    ├── VGGT-Long (arXiv 2025) ← 오류 누적 해결 (청킹 방식)
    └── StreamVGGT (arXiv 2025) ← 스트리밍 처리
```

### 4.2 방법론 비교 분석

| 방법 | 기반 | 장점 | 단점 | 1000프레임 처리 |
|------|------|------|------|----------------|
| **DUSt3R** (2024) | 2이미지 쌍 비교 | 캘리브레이션 불필요 | 전역 최적화 필요 | 불가 |
| **MASt3R** (2024) | DUSt3R 확장 | 메트릭 스케일 추정 | 쌍별 처리 한계 | 불가 |
| **Fast3R** (CVPR 2025) | 분산 추론 | 1000장 처리 가능 | CD 0.684 (낮은 품질) | 397.8s |
| **CUT3R** (CVPR 2025) | 지속 상태 모델 | 빠른 추론 | CD 0.786 (낮은 품질) | 34.8s |
| **VGGT-Long** (2025) | VGGT 청킹 | 오류 누적 해결 | 속도 저하 심각 | - |
| **VGGT\*** (2025) | VGGT | 높은 품질 | OOM 문제 | 724.6s |
| **FastVGGT** (이 논문) | VGGT | 품질+속도 균형 | 실외 미검증 | **180.7s** |

### 4.3 Token Merging 관련 연구 비교

| 방법 | 도메인 | 방식 | 3D 적용 | 특징 |
|------|--------|------|---------|------|
| **ToMe** (2022) [2] | 2D ViT | Bipartite 매칭 | ✗ | 학습 불필요 |
| **ToMeSD** (2023) [1] | 확산 모델 | 영역 기반 랜덤 | ✗ | FastVGGT의 직접 영감 |
| **PuMer** (2023) [3] | VLM | 가지치기+병합 | ✗ | 멀티모달 |
| **vid-TLDR** (2024) [5] | 비디오 | 시공간 | 부분 | 시간적 중복 활용 |
| **TokenLearner** (2021) [18] | 비디오 | MLP 선택 | ✗ | 학습 필요 |
| **FastVGGT** (이 논문) | **3D** | **3D 특화** | **✓** | **최초 3D 적용** |

**FastVGGT의 차별점:** 기존 Token Merging이 2D/비디오 도메인에 한정된 반면, FastVGGT는 **다중 뷰 교차 대응(cross-view correspondence)** 을 보존하는 3D 특화 전략을 최초로 제안합니다.

---

## 5. 향후 연구 영향 및 고려사항

### 5.1 앞으로의 연구에 미치는 영향

**① 3D 비전 확장성의 패러다임 전환**

FastVGGT는 Token Merging이 3D 기하 추론 모델에서도 유효함을 입증했습니다. 이는 향후 더 큰 규모의 3D 기반 모델 개발 시 **추론 효율화 전략의 표준 기법**이 될 가능성이 높습니다.

**② 오류 누적 연구의 새로운 관점**

기존 VGGT-Long이 청킹(chunking) 방식으로 오류 누적을 해결했다면, FastVGGT는 **토큰 중복 제거를 통해 근본적으로 오류 전파를 억제**할 수 있음을 보였습니다. 이는 시계열 모델의 오류 누적 문제에 대한 새로운 접근법을 제시합니다.

**③ Training-Free 최적화의 확산**

다른 3D 기반 모델(DUSt3R, MASt3R, Fast3R 등)에도 유사한 Training-Free 가속화 기법을 적용하는 연구를 자극할 것입니다.

**④ 메모리 최적화(VGGT\*)의 독립적 기여**

VGGT\*의 중간 결과 선택적 저장 기법은 다른 대형 트랜스포머 모델의 VRAM 최적화에도 직접 적용 가능한 아이디어입니다.

### 5.2 향후 연구 시 고려할 점

**🔴 단기 과제 (즉시 개선 가능)**

1. **가중 병합 도입:**
   단순 평균 대신 특징 중요도를 반영한 가중 병합:
   
```math
x'_d = \frac{w_d \cdot x_d + w_s \cdot x_s}{w_d + w_s}, \quad w = f(\|x\|, \text{attn\_score})
```

2. **적응적 Salient Token 선택:**
   고정 보폭 대신 장면 복잡도에 적응하는 동적 선택 전략 개발

3. **실외 및 대규모 장면 검증:**
   MegaDepth, Tanks and Temples 등 실외 데이터셋에서의 성능 평가

**🟡 중기 과제 (방법론 확장)**

4. **다른 3D 모델로의 일반화:**
   FastVGGT의 방법론을 DUSt3R, Fast3R, FLARE 등에 적용하는 체계적 연구

5. **계층적 Token Merging:**
   모든 Global Attention 블록에 동일한 비율을 적용하는 대신, 블록 깊이에 따라 차등 적용하는 전략

6. **동영상 스트리밍 적용:**
   실시간 로보틱스나 자율주행에서의 온라인 스트리밍 처리를 위한 확장

**🟢 장기 과제 (근본적 연구)**

7. **Token Collapse의 이론적 분석:**
   VGGT의 Global Attention에서 발생하는 token collapse를 정보 이론적으로 분석하고, 최적 병합 전략을 유도하는 이론 연구

8. **학습 기반 병합과의 융합:**
   Training-Free와 학습 기반(예: TokenLearner) 방식을 결합하여 더 정밀한 중요도 추정 실현

9. **불균일 시점 분포 처리:**
   현재 균일 샘플링 기반인 Salient Token 선택을 시점 중복도에 따라 적응적으로 조절하는 연구

10. **멀티모달 3D 이해로의 확장:**
    언어-3D 모델(LLM + 3D Vision)에서의 Token Merging 적용 가능성 탐색

---

## 참고 자료

**주요 논문 (본 논문 내 인용 기준):**

1. **FastVGGT (본 논문):** You Shen et al., "FastVGGT: Training-Free Acceleration of Visual Geometry Transformer," arXiv:2509.02560v2, 2025.
2. **VGGT:** Jianyuan Wang et al., "VGGT: Visual Geometry Grounded Transformer," CVPR 2025, pp. 5294–5306.
3. **ToMeSD:** Daniel Bolya & Judy Hoffman, "Token Merging for Fast Stable Diffusion," CVPR 2023, pp. 4599–4603.
4. **ToMe:** Daniel Bolya et al., "Token Merging: Your ViT But Faster," arXiv:2210.09461, 2022.
5. **DUSt3R:** Shuzhe Wang et al., "DUSt3R: Geometric 3D Vision Made Easy," CVPR 2024, pp. 20697–20709.
6. **MASt3R:** Vincent Leroy et al., "Grounding Image Matching in 3D with MASt3R," ECCV 2024, pp. 71–91.
7. **Fast3R:** Jianing Yang et al., "Fast3R: Towards 3D Reconstruction of 1000+ Images in One Forward Pass," CVPR 2025, pp. 21924–21935.
8. **CUT3R:** Qianqian Wang et al., "Continuous 3D Perception Model with Persistent State," CVPR 2025, pp. 10510–10522.
9. **VGGT-Long:** Kai Deng et al., "VGGT-Long: Chunk It, Loop It, Align It," arXiv:2507.16443, 2025.
10. **FlashAttention-2:** Tri Dao, "FlashAttention-2: Faster Attention with Better Parallelism," arXiv:2307.08691, 2023.
11. **ViT:** Alexey Dosovitskiy et al., "An Image is Worth 16×16 Words," arXiv:2010.11929, 2020.
12. **DINO:** Mathilde Caron et al., "Emerging Properties in Self-Supervised Vision Transformers," ICCV 2021, pp. 9650–9660.
13. **DINOv2:** Maxime Oquab et al., "DINOv2: Learning Robust Visual Features without Supervision," arXiv:2304.07193, 2023.
14. **TokenLearner:** Michael Ryoo et al., "TokenLearner: Adaptive Space-Time Tokenization for Videos," NeurIPS 2021, pp. 12786–12797.
15. **StreamVGGT:** Dong Zhuo et al., "Streaming 4D Visual Geometry Transformer," arXiv:2507.11539, 2025.
16. **$\pi^3$:** Yifan Wang et al., "Scalable Permutation-Equivariant Visual Geometry Learning," arXiv:2507.13347, 2025.

> **⚠️ 정확도 주의:** 본 답변은 제공된 논문 PDF(arXiv:2509.02560v2)를 기반으로 작성되었습니다. 논문에 명시되지 않은 내용(예: 특정 한계의 저자 명시 여부)은 분석적 추론임을 밝힙니다.
