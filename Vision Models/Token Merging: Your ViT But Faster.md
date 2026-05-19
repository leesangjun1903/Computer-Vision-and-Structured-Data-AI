
# Token Merging (ToMe): Your ViT But Faster

> **논문 정보**
> - **제목:** Token Merging: Your ViT But Faster
> - **저자:** Daniel Bolya, Cheng-Yang Fu, Xiaoliang Dai, Peizhao Zhang, Christoph Feichtenhofer, Judy Hoffman
> - **발표:** ICLR 2023
> - **arXiv:** [2210.09461](https://arxiv.org/abs/2210.09461)
> - **공식 GitHub:** [facebookresearch/ToMe](https://github.com/facebookresearch/ToMe)

---

## 1. 핵심 주장 및 주요 기여 (Executive Summary)

ToMe(Token Merging)는 기존 ViT 모델의 처리량(throughput)을 **재학습 없이도** 향상시킬 수 있는 간단한 방법이며, 가지치기(pruning)만큼 빠르면서도 더 높은 정확도를 유지하는 **경량 매칭 알고리즘**을 사용해 트랜스포머 내 유사한 토큰들을 점진적으로 결합한다.

**주요 기여 요약:**

| 기여 항목 | 설명 |
|---|---|
| ✅ 학습 불필요 추론 가속 | 기존 모델에 플러그인 방식으로 적용 |
| ✅ Bipartite Soft Matching | 새로운 고속 병렬 토큰 매칭 알고리즘 제안 |
| ✅ Proportional Attention | 병합된 토큰 크기를 어텐션에 반영 |
| ✅ 멀티모달 적용 | 이미지, 비디오, 오디오 전 영역 적용 가능 |
| ✅ 학습 속도 향상 | MAE fine-tuning에서 최대 2× 학습 가속 |

off-the-shelf 방식으로 ViT-L @ 512 및 ViT-H @ 518 모델에서 이미지 처리량을 **2×**, 비디오에서는 ViT-L의 처리량을 **2.2×** 향상시키며, 각각의 경우 정확도 손실은 **0.2~0.3%에 불과**하다.

---

## 2. 해결하고자 하는 문제

### 2.1 배경 및 문제 정의

트랜스포머 모델은 오디오 처리부터 이미지 인식까지 다양한 도메인에서 광범위하게 사용되고 있으나, 해당 모델들은 매우 거대하여 학습 및 실행이 어렵다.

이를 해결하기 위해 **가지치기(Pruning)** 가 도입되었으나, 이는 토큰에 대해 계산된 메트릭에 기반해 덜 중요한 토큰을 단순히 **제거**한다.

기존 방법들의 핵심 한계는: 추가 파라미터 필요, **학습 속도 향상에 미적용**, 입력 콘텐츠에 따라 다른 수의 토큰을 가지치기하여 **배치 추론 불가** 등이었다.

ToMe는 유사도에 기반해 토큰을 병합하여 객체의 일부를 암묵적으로 그룹화하는데, 이는 단순히 배경 토큰만 제거하는 가지치기와 대조적이며, 전경(foreground) 중복 토큰까지 병합함으로써 더 많은 토큰 감소를 달성할 수 있다.

---

## 3. 제안하는 방법 (수식 포함)

### 3.1 전체 전략: 점진적 토큰 감소

각 트랜스포머 블록에서 레이어당 $r$개의 토큰을 병합하며, $r$은 비율이 아닌 토큰의 **절대 개수**이다. 네트워크의 $L$개 블록 전체에 걸쳐 총 $rL$개의 토큰이 점진적으로 병합되며, $r$을 조절함으로써 **속도-정확도 트레이드오프**를 제어할 수 있다.

$$\text{총 병합 토큰 수} = r \times L$$

여기서:
- $r$: 레이어당 병합 토큰 수 (하이퍼파라미터)
- $L$: 트랜스포머 블록(레이어) 수

### 3.2 토큰 유사도 측정

Keys $(K)$는 이미 각 토큰의 정보를 요약하고 있으므로, 각 토큰의 키 벡터 간 **코사인 유사도** 등 내적 기반 유사도 메트릭을 사용하여 어떤 토큰이 유사한 정보를 담고 있는지 판단한다.

$$\text{sim}(K_i, K_j) = \frac{K_i \cdot K_j}{\|K_i\| \|K_j\|}$$

클래스 토큰(CLS token)과 증류 토큰(distillation token)은 병합에서 제외되며, 각 토큰의 keys $K$ (모든 헤드에서 평균화)를 이용하여 토큰 유사도를 정의한다.

### 3.3 Bipartite Soft Matching (핵심 알고리즘)

BSM에서 이미지 패치를 표현하는 토큰들은 집합 $\mathcal{A}$와 $\mathcal{B}$로 분리되고, 두 집합 간 쌍별 코사인 유사도를 계산하여 가장 유사도가 높은 상위 $r$쌍의 토큰을 병합한다.

**알고리즘 절차:**

```math
\mathcal{A}, \mathcal{B} \leftarrow \text{partition}(\text{Tokens})
```

```math
S_{ij} = \text{cosine\_sim}(K_i^{\mathcal{A}},\ K_j^{\mathcal{B}}) \quad \forall i \in \mathcal{A},\ j \in \mathcal{B}
```

```math
\text{Merge top-}r\text{ pairs: } \{(i^*, j^*)\} = \underset{(i,j)}{\text{argtop-}r}\ S_{ij}
```

```math
\tilde{T}_k = \text{mean}(T_{i^*},\ T_{j^*}) \quad \text{(feature averaging)}
```

이 bipartite matching은 greedy 방식의 정확도와 pruning의 속도를 동시에 가지며, $r$에 대해 **상수 시간(constant runtime)** 으로 동작한다.

### 3.4 Proportional Attention (비례 어텐션)

토큰이 병합되면 더 이상 하나의 입력 패치를 나타내지 않는다. 이는 소프트맥스 어텐션의 결과에 영향을 줄 수 있는데, 동일한 키를 가진 두 토큰을 병합하면 해당 키의 소프트맥스 가중치가 감소하게 된다.

이를 해결하기 위해 **비례 어텐션(Proportional Attention)** 을 도입한다:

$$A = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d}} + \log(s)\right)$$

여기서 $s$는 각 토큰이 대표하는 패치의 수를 담은 행 벡터(row vector)이다.

이를 통해 병합된 토큰이 더 많은 패치를 대표할수록 어텐션 스코어에서 더 높은 가중치를 자연스럽게 부여받게 된다.

### 3.5 학습 시 병합 적용

학습 시, 토큰 병합은 **풀링 연산(pooling operation)으로 처리**되며, 병합된 토큰에 대해 평균 풀링을 사용한 것처럼 역전파(backprop)가 이루어진다.

### 3.6 모델 구조 내 삽입 위치

토큰 병합은 각 트랜스포머 블록의 **어텐션(attention) 브랜치와 MLP 브랜치 사이**에 적용된다.

**구조 다이어그램 (의사코드):**

```
Input Tokens
     ↓
[Self-Attention] → Q, K, V 추출
     ↓
[Token Merging Module] ← 어텐션과 MLP 사이에 삽입
  ① K로 코사인 유사도 계산
  ② Bipartite Soft Matching으로 r쌍 선택
  ③ 선택된 토큰 평균 병합
  ④ 토큰 크기 s 업데이트
     ↓
[MLP / FFN]
     ↓
Output (토큰 수 감소됨)
```

---

## 4. 성능 향상

### 4.1 이미지 분류 (ImageNet)

학습 적용 시, ViT-H에서 2× 처리량에서의 오류율을 **0.4%**, ViT-L은 **0.6%**, ViT-B는 **1.7%** 수준으로 낮출 수 있다.

### 4.2 비디오·오디오

ToMe는 학습 중에도 쉽게 적용할 수 있으며, MAE 비디오 파인튜닝에서 학습 속도를 최대 2×까지 향상시킨다. 학습 적용 시 정확도 손실이 더욱 최소화되어, 오디오에서 ViT-B의 처리량을 2× 향상시키면서 **mAP 손실은 0.4%에 그친다.**

### 4.3 토큰 가지치기 방법과의 비교

A-ViT, DynamicViT, SP-ViT 등 토큰 가지치기 방법들과 비교 시, ToMe는 Gumbel Softmax 등의 gradient tricks, 추가 파라미터, 혹은 특수 학습 기법 없이도 기존의 훨씬 복잡한 토큰 가지치기 방법들의 성능에 필적하거나 처리량을 능가한다.

더불어, 대부분의 토큰 가지치기 방법은 학습 중 패딩 토큰 또는 어텐션 마스킹을 사용해야 하므로 가지치기의 이점이 상쇄되는 반면, ToMe는 이러한 문제 없이 DeiT에서 **1.5× 학습 속도 향상**을 보인다.

### 4.4 MAE와의 시너지

ToMe는 MAE의 고유한 문제점을 암묵적으로 해결한다. MAE는 사전학습 시 토큰을 제거해 epoch 시간이 ~4× 빠르지만, 일반 파인튜닝은 모든 토큰을 사용해 이 이점이 사라진다. ToMe는 이 문제를 해결하여 대형 모델에서 무시할 만한 정확도 손실로 **~2× 빠른 epoch**을 가능하게 한다.

이는 Token Merging을 활용하면 이전에는 불가능했던 **더 큰 모델도 학습할 수 있음을 시사**한다.

---

## 5. 모델의 일반화 성능 향상 가능성

### 5.1 정성적 관찰: 부분 분할(Part Segmentation) 효과

ToMe 적용 결과, 토큰 병합이 **부분 분할(part segmentation)** 과 유사한 패턴을 보임을 발견했다. 예를 들어, 허스키 이미지에서는 다리, 몸통, 얼굴에 각각 다른 토큰이 생성되고, 원숭이 이미지에서는 손, 몸, 얼굴, 눈, 입이 각각의 토큰으로 구분된다.

이는 ToMe가 **의미 있는 시각적 구조를 자연스럽게 학습하는 능력**을 갖출 수 있음을 의미한다.

### 5.2 멀티모달 일반화

ToMe는 이미지, 비디오, 오디오 모달리티 전반에 걸쳐 효과적임이 입증되었으며, 최소한의 정확도 손실로 상당한 처리량 향상과 가속화된 학습 시간을 제공한다.

### 5.3 다양한 ViT 아키텍처 호환성

AugReg와 SWAG 같은 대규모 지도/약지도 사전학습 ViT 모델과, MAE 자기지도 사전학습 방법 모두에 ToMe를 적용한 비교 실험이 수행되었으며, 모든 방식에서 일관된 가속 효과가 확인되었다.

### 5.4 다른 방법과의 결합 가능성

ToMe는 Dynamic Tuning(DyT) 같은 파라미터 효율화 기법과 결합되었을 때, 정확도를 유지하면서 처리량을 추가로 향상시키는 것이 확인되었다.

### 5.5 일관성(Consistency) 분석

가지치기 기반 방법의 감소 패턴은 백본 크기를 변경할 때 **비일관적**인 반면, ToMe를 포함한 병합 기반 방법들은 $r > 25\%$ 조건에서 백본 크기에 상관없이 **일관된** 감소 패턴을 보인다. 이는 ToMe의 더 우수한 일반화 가능성을 의미한다.

---

## 6. 한계점 (Limitations)

### 6.1 콘텐츠 독립적 병합

ToMe는 이미지 콘텐츠와 무관하게 고정된 수($rL$)의 토큰을 감소시킨다. 이는 단순하고 배치 처리에 유리하지만, 복잡한 이미지에서는 과도한 정보 손실이 발생할 수 있다.

### 6.2 공간 정보 활용 부족

병합된 토큰은 반드시 인접한 입력 영역을 대표하지 않으며, ToMe가 가지는 유일한 공간 신호는 **위치 인코딩(position encodings)에서만** 나온다.

이를 개선하기 위해 최근 후속 연구인 **ToSA(Token Merging with Spatial Awareness)** 가 제안되었다.
ToMe는 각 ViT 레이어에서 유사한 의미적 시각 특징을 공유하는 토큰을 병합하지만, 이러한 방법들은 어텐션 스코어나 시각적 특징 유사도에만 의존한다는 한계가 있다.

### 6.3 TensorRT 비호환

ToMe는 NVIDIA의 TensorRT 라이브러리에서 지원되지 않는 함수를 포함하고 있어 TensorRT로 변환할 수 없다는 실용적 배포 한계가 있다.

### 6.4 비전-언어 도메인 확장의 어려움

텍스트 입력과의 문맥화(contextualisation) 부족은 ToMe가 비전-언어(Vision-Language) 도메인으로 확장되는 데 주요 한계로 작용한다.

---

## 7. 2020년 이후 관련 최신 연구 비교 분석

| 방법 | 년도 | 방식 | 재학습 필요 | 병렬화 | 핵심 아이디어 |
|---|---|---|---|---|---|
| **DynamicViT** | 2021 | Pruning | ✅ 필요 | ⚠️ 제한적 | MLP로 중요도 점수 예측 후 가지치기 |
| **EViT** | 2022 | Pruning+Fusion | ✅ 필요 | ⚠️ | 클래스 어텐션으로 중요 토큰 선별 |
| **ToMe** | 2022 | **Merging** | ❌ 불필요 | ✅ 완전 병렬 | Bipartite Soft Matching |
| **VidToMe** | 2023 | Merging | ❌ | ✅ | 비디오 프레임 간 토큰 병합 |
| **ToSA** | 2025 | Merging+ | ❌ | ✅ | 공간 정보 추가 고려 |
| **SP-ViT(Spectrum)** | 2024 | Merging | ❌ | ✅ | 스펙트럼 보존 토큰 병합 |

ToMe는 Bipartite Soft Matching(BSM) 알고리즘을 도입하여 간결하고 효과적인 유사 토큰 병합으로 주목받았으며, 이후 ToFu, Pumer, LTPM, DiffRate 등의 연구들이 비전 및 언어 도메인에서 다양한 방식으로 BSM을 확장했다.

비디오 편집 분야로의 확장인 VidToMe에서는 원래 ToMe 알고리즘이 토큰 값을 평균화하여 병합하는데, 이로 인해 생성된 비디오에서 다양성과 무작위성이 부족해지는 문제가 발견되었다.

---

## 8. 앞으로의 연구에 미치는 영향 및 고려할 점

### 8.1 연구에 미치는 영향

**① 플러그인 방식의 효율화 패러다임 확립**

이전의 학습 필요 방법들과 달리, 실용성 향상을 위해 추가 학습이 필요 없이 토큰 수를 줄이고 ViT와 VLM의 효율성을 향상시킬 수 있는 **plug-and-play 모듈**로서의 토큰 감소 방법이 여러 후속 연구에서 탐구되고 있다.

**② 대규모 모델 학습의 가능성 확장**

ToMe의 ~2× 빠른 epoch 달성은 이전에는 불가능했던 **더 큰 모델 학습 가능성**을 제시하며, 이는 스케일 업 연구의 새로운 방향을 열었다.

**③ 비전-언어 모델(VLM) 효율화 연구 자극**

ViT-L/14를 비주얼 인코더로 사용하면 해상도 336px의 이미지에서 576개의 토큰이 생성되며, 이는 표준 LLM의 2048 토큰 입력 한계에서 처리가 어려워지는 문제를 낳는다. ToMe의 토큰 감소 아이디어는 이를 해결하는 VLM 효율화 연구에 직접적인 영향을 미쳤다.

---

### 8.2 앞으로 연구 시 고려할 점

**① 콘텐츠 적응적 병합 비율 (Content-Aware Scheduling)**

현재 ToMe는 이미지 복잡도와 무관하게 고정된 $r$을 사용한다. 복잡한 이미지에는 낮은 $r$, 단순한 이미지에는 높은 $r$을 동적으로 적용하는 연구가 필요하다.

**② 공간 일관성 보존**

Bipartite Soft Matching은 ViT의 각 레이어 내에서 토큰을 병합하는 효율적인 방법이지만, 어텐션 블록의 Key feature 코사인 유사도에 기반하므로 순수하게 시각적 특징의 유사도에 의존한다. 위치 및 공간 정보를 함께 고려하는 병합 기준 설계가 중요한 연구 방향이다.

**③ 다운스트림 태스크(Detection/Segmentation) 적용 확장**

ToMe는 주로 분류 태스크를 기준으로 평가되었다. 객체 탐지, 인스턴스 분할 등 **위치 정보가 중요한 태스크**에서의 적합성 연구가 필요하다.

**④ VLM / LLM과의 통합**

정보 손실 없이 배경 토큰을 제거하고 덜 중요한 전경 토큰을 병합하는 이 접근법의 다양성은 학습 및 비학습 시나리오 모두로 확장되어 연산 및 메모리 사용량을 크게 줄인다. LLaVA, GPT-4V와 같은 대규모 VLM에 적용하여 입력 토큰 수를 줄이는 방향이 매우 유망하다.

**⑤ 하드웨어 최적화 관점**

ToMe는 TensorRT 비호환 문제가 있으므로, 엣지 디바이스 및 실제 배포 환경을 위한 **하드웨어 친화적 병합 알고리즘** 설계가 중요한 실용적 과제이다.

**⑥ 자기지도 학습(Self-Supervised Learning)과의 결합**

MAE는 사전학습 시 토큰 제거로 약 4× 빠른 epoch을 달성하지만, 일반 파인튜닝에서는 이 이점이 사라진다. ToMe와 다양한 SSL 방법론의 결합을 통한 효율적 사전학습 + 파인튜닝 파이프라인 구축이 중요한 연구 주제이다.

---

## 📚 참고 자료 출처

1. **arXiv 원문**: Bolya et al., "Token Merging: Your ViT But Faster," arXiv:2210.09461, ICLR 2023. https://arxiv.org/abs/2210.09461
2. **ICLR 2023 OpenReview PDF**: https://openreview.net/pdf?id=JroZRaRw7Eu
3. **arXiv PDF (Full paper)**: https://arxiv.org/pdf/2210.09461
4. **공식 GitHub (facebookresearch/ToMe)**: https://github.com/facebookresearch/ToMe
5. **MYRIAD Blog Review**: Token Merging: Your ViT But Faster — CREATIS MYRIAD. https://creatis-myriad.github.io/2022/10/24/Token-Merging-Your-ViT-But-Faster.html
6. **MarkTechPost Summary**: "Meta AI Researchers Propose Token Merging (ToMe)..." https://www.marktechpost.com/2022/11/10/meta-ai-researchers-propose-token-merging-tome-to-make-vision-transformers-run-faster/
7. **Liner Quick Review**: https://liner.com/review/token-merging-your-vit-but-faster
8. **ResearchGate PDF**: https://www.researchgate.net/publication/364431397_Token_Merging_Your_ViT_But_Faster
9. **ToSA (후속 연구)**: "ToSA: Token Merging with Spatial Awareness," arXiv:2506.20066. https://arxiv.org/html/2506.20066v1
10. **VidToMe**: "VidToMe: Video Token Merging for Zero-Shot Video Editing," arXiv:2312.10656. https://arxiv.org/html/2312.10656v2
11. **Spectrum-Preserving Token Merging**: arXiv:2405.16148. https://arxiv.org/pdf/2405.16148
12. **Token Merging with Class Importance Score (TomeCIS)**: Kwang-Soo Seol et al., Hanyang University. http://esoc.hanyang.ac.kr/publications/2023/sks_Token_Merging_with_Class_Importance_Score%20final_ver.pdf
13. **Which Tokens to Use? (ICCVW 2023)**: Haurum et al., ICCV Workshop 2023.
14. **Dynamic Tuning Towards Parameter and Inference Efficiency (NeurIPS 2024)**: https://proceedings.neurips.cc/paper_files/paper/2024/file/d0241a0fb1fc9be477bdfde5e0da276a-Paper-Conference.pdf
15. **Pruning and Merging of Tokens for Efficient VL Models — Medium**: https://medium.com/@vedantpalit10/pruning-and-merging-of-tokens-for-efficient-vl-models-a-review-5fa833a0c7e6
16. **Automatic Pruning Rate Adjustment (Applied Intelligence, Springer, 2025)**: https://link.springer.com/article/10.1007/s10489-025-06265-z

# Token Merging: Your ViT But Faster

---

## 1. 핵심 주장과 주요 기여 요약

### 핵심 주장

**Token Merging (ToMe)**는 기존 Vision Transformer (ViT) 모델을 **재학습 없이** 처리 속도를 향상시킬 수 있는 간단하고 범용적인 방법이다. 토큰을 제거(pruning)하는 대신 **유사한 토큰들을 병합(merging)** 함으로써, 정보 손실을 최소화하면서 속도를 높인다.

### 주요 기여

| 기여 | 내용 |
|------|------|
| **학습 불필요** | 기존 ViT 모델에 off-the-shelf 적용 가능 |
| **이분 소프트 매칭** | 빠르고 정확한 신규 토큰 병합 알고리즘 제안 |
| **비례 어텐션** | 병합 후 토큰 크기를 반영한 소프트맥스 보정 |
| **다중 모달리티** | 이미지, 비디오, 오디오 모두에 코드 변경 없이 적용 |
| **훈련 가속** | 학습 시 적용 시 MAE fine-tuning 속도 최대 2× 향상 |

---

## 2. 해결하고자 하는 문제, 제안 방법, 모델 구조, 성능, 한계

### 2.1 해결하고자 하는 문제

ViT는 강력한 성능을 가지지만 **연산 비용이 높다**. 기존 토큰 감소 방법들(token pruning)은 다음 문제를 가진다:

- 정보 손실로 인한 정확도 저하
- 재학습 필수 (일부는 추가 파라미터 필요)
- 훈련 속도 향상 불가능 (훈련 시 마스킹 필요)
- 동적 토큰 수로 인해 배치 추론 불가능

### 2.2 제안하는 방법

#### 2.2.1 전략 (Strategy)

각 트랜스포머 블록에서 $r$개의 토큰을 병합하여, $L$개 레이어에 걸쳐 총 $rL$개 토큰을 점진적으로 줄인다. $r$은 비율이 아닌 **개수**이며, 입력 내용과 무관하게 고정된다(→ 배치 추론 가능).

#### 2.2.2 토큰 유사도 (Token Similarity)

중간 feature 공간이 과잉 파라미터화되어 있어 feature 값 그대로 유사도를 계산하면 노이즈에 취약하다. 대신, QKV self-attention의 **Key 행렬** $K$를 활용하여 코사인 유사도를 계산한다:

$$\text{similarity}(i, j) = \frac{K_i \cdot K_j}{\|K_i\| \|K_j\|}$$

#### 2.2.3 이분 소프트 매칭 (Bipartite Soft Matching)

알고리즘:

1. 토큰을 두 집합 $\mathbb{A}$, $\mathbb{B}$로 분할 (교번 방식)
2. $\mathbb{A}$의 각 토큰에서 $\mathbb{B}$ 내 가장 유사한 토큰으로 엣지 연결
3. 유사도 상위 $r$개의 엣지만 유지
4. 연결된 토큰 쌍을 병합 (가중 평균)
5. 두 집합을 다시 연결(concatenate)

이 알고리즘은 이분 그래프를 형성하여 연결 요소(connected component) 탐색이 $O(1)$이며, **$r$에 대해 상수 시간 복잡도**를 가진다.

#### 2.2.4 비례 어텐션 (Proportional Attention)

토큰이 병합되면 하나의 토큰이 여러 패치를 대표한다. 이를 소프트맥스 어텐션에 반영하기 위해:

$$\boldsymbol{A} = \text{softmax}\!\left(\frac{\boldsymbol{Q}\boldsymbol{K}^\top}{\sqrt{d}} + \log \boldsymbol{s}\right) $$

여기서 $\boldsymbol{s}$는 각 토큰이 대표하는 패치 수(토큰 크기)를 담은 행 벡터이다. $\log \boldsymbol{s}$를 더함으로써, 마치 해당 키를 $s$번 복사한 것과 동일한 효과를 낸다.

병합 시 가중 평균:

$$\text{merged token} = \frac{\sum_{i \in \text{cluster}} s_i \cdot x_i}{\sum_{i \in \text{cluster}} s_i}$$

#### 2.2.5 병합 스케줄 (Merging Schedule)

$$\text{Constant Schedule: } x \text{ per layer} \quad \rightarrow \quad r_x $$

$$\text{Decreasing Schedule: } 2x \to 0 \text{ per layer} \quad \rightarrow \quad r_x \searrow $$

두 스케줄 모두 총 $rL$개의 토큰을 제거하지만, decreasing schedule은 초반에 많이 제거하여 더 높은 처리량을 달성한다.

#### 2.2.6 학습 중 병합 (Training with Merging)

토큰 병합을 **평균 풀링**으로 취급하여 역전파를 통해 학습한다. Gumbel-softmax 등의 gradient trick이 필요 없으며, 기존 ViT 학습 레시피를 그대로 사용할 수 있다.

### 2.3 모델 구조

```
Input Tokens
     ↓
[Attention Module]
     ↓
[ToMe: Bipartite Soft Matching] ← K matrix에서 유사도 계산
     ↓
[MLP Module]
     ↓
Output (reduced tokens)
```

- ToMe는 **어텐션과 MLP 사이**에 삽입됨 (블록 시작 부분에 삽입하는 기존 연구와 차별화)
- 이 위치 선택이 정확도 향상에 기여 (어텐션 후 K 행렬 활용 가능)

### 2.4 성능 향상

#### 이미지 (ImageNet-1k)

| 모델 | 설정 | 처리량 향상 | 정확도 하락 |
|------|------|------------|------------|
| ViT-L/16 @ 512 (SWAG) | off-the-shelf | $2\times$ | $0.3\%$ |
| ViT-H/14 @ 518 (SWAG) | off-the-shelf | $2\times$ | $0.3\%$ |
| ViT-H/14 (MAE) | 학습 적용 | $2\times$ | $0.4\%$ |
| ViT-L/16 (MAE) | 학습 적용 | $2\times$ | $0.6\%$ |

#### 비디오 (Kinetics-400)

| 모델 | 설정 | 처리량 향상 | 정확도 하락 |
|------|------|------------|------------|
| ViT-L (Spatiotemporal MAE) | $r=65$, constant | $2.2\times$ | $0.2\%$ |
| ViT-L (MAE) | $r=65$, constant | 훈련 시간 $0.5\times$ | 무시 가능 |

#### 오디오 (AudioSet-2M)

| 모델 | 설정 | 처리량 향상 | mAP 하락 |
|------|------|------------|----------|
| ViT-B (Audio MAE) | $r=40$, 학습 적용 | $\approx 2\times$ | $0.4\%$ |

#### 토큰 프루닝 대비 비교 (ViT-S, DeiT 기준)

| 방법 | 정확도 | 처리량(im/s) | 훈련 속도 |
|------|--------|------------|----------|
| DeiT-S (baseline) | 79.8% | 930 | $1\times$ |
| DynamicViT | 79.3% | 1505 | $1\times$ |
| SP-ViT | 79.3% | — | $1\times$ |
| **ToMe DeiT** $r_{13}\rightarrow$ | **79.4%** | **1552** | **$1.5\times$** |

### 2.5 한계

1. **소형 모델에서 정확도 하락 큼**: ViT-B, S, Ti 등 작은 모델은 $2\times$ 속도에서 4~5% 정확도 하락 발생
2. **레이어 깊이 의존성**: 더 깊은(larger) 모델에서 효과가 크고, 얕은 모델에서는 효과 감소
3. **콘텐츠 무관 고정 병합 수**: 동적 방법보다 정확도 측면에서 이론적으로 불리
4. **조밀한 예측 태스크 미검증**: 분류 태스크 위주로 검증; 객체 탐지, 세그멘테이션 등은 미래 과제
5. **프리트레이닝과 정합성**: MAE 모델은 proportional attention 없이도 잘 작동하지만, 지도학습 모델은 필요 → 프리트레이닝 방식에 따른 최적 설정이 다름
6. **멀티클립 보상 효과**: 비디오에서 여러 클립 평가가 정보 손실을 일부 보상할 수 있어, 실제 단일 추론 성능은 다를 수 있음

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 멀티모달 일반화

ToMe의 가장 강력한 일반화 특성은 **코드 변경 없이** 이미지, 비디오, 오디오에 모두 적용 가능하다는 점이다. 이는 모달리티 특화 귀납적 편향(inductive bias)이 없기 때문이다.

```math
\text{ToMe} \xrightarrow{\text{동일 코드}} \begin{cases} \text{Image (ImageNet-1k)} \\ \text{Video (Kinetics-400)} \\ \text{Audio (AudioSet-2M)} \end{cases}
```

### 3.2 학습 없이도 일반화 가능

**Off-the-shelf 적용**: 재학습 없이 기존 모델에 바로 적용 가능하다는 것은, ToMe가 모델 가중치에 내재된 표현 능력에 의존하며 **추가적인 도메인 특화 학습 없이** 일반화됨을 시사한다.

### 3.3 대형 모델에서의 일반화 향상

실험 결과, **모델이 크고 깊을수록** ToMe의 정확도 하락이 작다:

$$\text{ViT-Ti: } \Delta\text{acc} \approx -4\% \quad \xrightarrow{\text{모델 크기 증가}} \quad \text{ViT-H: } \Delta\text{acc} \approx -0.3\% \quad (\text{at } 2\times)$$

저자들은 이 현상을 "깊은 모델일수록 레이어당 feature 변화가 점진적이어서 병합의 영향이 줄어든다"고 설명한다. 이는 **스케일링 법칙(scaling law)과 ToMe의 시너지** 가능성을 시사한다.

### 3.4 Re-evaluation을 통한 일반화

흥미로운 발견: $r=5$로 학습된 ViT-L 모델을 $r=0$(병합 없음)으로 재평가하면 기존 baseline보다 **정확도가 향상**된다(85.7% → 85.8%). 이는 ToMe 학습이 일종의 **정규화(regularization)** 효과를 가지며, 모델의 일반화 성능을 향상시킬 수 있음을 의미한다.

$$\mathcal{M}_{r=5}^{\text{train}} \xrightarrow{r=0 \text{ 재평가}} \text{acc: } 85.8\% > 85.7\% = \mathcal{M}_{r=0}^{\text{baseline}}$$

### 3.5 시각적 일반화: 부분 분할 및 객체 추적

ToMe의 토큰 병합은 **부분 분할(part segmentation)**과 유사한 패턴을 자연스럽게 학습한다. 이미지에서는 객체의 의미론적 부분들이 같은 토큰으로 병합되고, 비디오에서는 동일 객체가 여러 프레임에 걸쳐 추적된다. 이는 ToMe가 의미론적 표현의 일반화를 촉진할 수 있음을 시사한다.

### 3.6 MAE와의 시너지를 통한 일반화

MAE는 프리트레이닝 시 토큰을 마스킹하여 제거하는데, ToMe의 병합 방식이 이와 유사한 효과를 내어 **fine-tuning 시 일반화 성능**을 향상시킬 수 있다. 특히 fine-tuning 시 원래 레시피를 그대로 사용할 수 있다는 점은 ToMe가 학습 동역학을 크게 변형하지 않음을 보여준다.

---

## 4. 최신 연구 비교 분석 (2020년 이후)

### 4.1 토큰 프루닝 계열

| 논문 | 방법 | 장점 | 단점 vs ToMe |
|------|------|------|-------------|
| **DynamicViT** (Rao et al., NeurIPS 2021) | 어텐션 기반 동적 프루닝 | 높은 정확도 | 재학습 필요, 배치 추론 불가, 훈련 속도 향상 없음 |
| **A-ViT** (Yin et al., CVPR 2022) | 적응적 토큰 프루닝 | 입력 적응적 | DeiT 체크포인트 fine-tuning 필요, 배치 불가 |
| **SP-ViT** (Kong et al., ECCV 2022) | 소프트 토큰 프루닝 | 높은 정확도 | 추가 파라미터, 배치 불가 |
| **EViT** (Liang et al., ICLR 2022) | 비중요 토큰 집약 | 정보 보존 | 재학습 필요 |
| **ToMe** (Bolya et al., ICLR 2023) | 이분 소프트 매칭 병합 | **학습 불필요, 배치 가능, 훈련 가속** | 소형 모델 정확도 하락 |

### 4.2 효율적 ViT 아키텍처 계열

| 논문 | 방법 | vs ToMe |
|------|------|---------|
| **Swin Transformer** (Liu et al., ICCV 2021) | 시프트 윈도우 어텐션 | 도메인 특화 설계 필요, 재학습 필요 |
| **MViTv2** (Li et al., CVPR 2022) | 멀티스케일 풀링 | 비디오 특화, 재학습 필요 |
| **LeViT** (Graham et al., ICCV 2021) | Conv 모듈 혼합 | 비ViT 아키텍처 |
| **ToMe** | 플러그인 병합 | **기존 ViT 재사용, 코드 수정 최소화** |

### 4.3 토큰 병합/풀링 계열

| 논문 | 방법 | vs ToMe |
|------|------|---------|
| **Token Pooling** (Marin et al., 2021) | k-means 클러스터링 | 느림(순차적), 학습 없이 10~40% 정확도 하락 |
| **GroupViT** (Xu et al., CVPR 2022) | 크로스 어텐션 그루핑 | 효율성 목적 아님, 세그멘테이션 특화 |
| **TokenLearner** (Ryoo et al., NeurIPS 2021) | MLP 기반 토큰 감소 | 추가 파라미터 필요 |
| **ToMe** | 이분 소프트 매칭 | **추가 파라미터 없음, 병렬화 가능, off-the-shelf** |

### 4.4 이후 ToMe의 영향을 받은/관련된 연구 방향

ToMe 발표(2022~2023) 이후, 다음과 같은 연구 방향들이 활발히 진행되었다:

- **ToMe for Stable Diffusion**: ToMe 개념을 디퓨전 모델의 U-Net에 적용하여 생성 속도 향상 시도 (Bolya & Hoffman, 2023)
- **동적 병합 스케줄 최적화**: 입력 콘텐츠에 따른 적응적 병합 전략 연구
- **LLM 토큰 감소**: 언어 모델에서의 유사 아이디어 적용

---

## 5. 향후 연구에 미치는 영향과 고려 사항

### 5.1 향후 연구에 미치는 영향

#### 5.1.1 효율적 ViT 연구 패러다임 전환

ToMe는 기존의 **"더 작은 아키텍처 설계"** 패러다임에서 **"기존 대형 모델의 효율적 배포"** 패러다임으로의 전환을 촉진한다. 재학습 없이 대형 모델을 효율화할 수 있다는 점은 산업 현장에서의 활용성을 크게 높인다.

#### 5.1.2 자연적 계층 모델로서의 ViT

저자들이 언급하듯, ToMe는 ViT를 Swin이나 MViT와 유사한 **자연적 계층 구조** 모델로 변환한다. 이는 기존 계층적 아키텍처와 순수 ViT 사이의 간극을 메우는 새로운 관점을 제공한다.

#### 5.1.3 대규모 모델 훈련 가속

훈련 시 최대 $2\times$ 속도 향상은, 이전에는 불가능했던 **더 큰 모델의 훈련**을 가능하게 한다. 이는 스케일링 연구에 직접적인 영향을 미친다.

#### 5.1.4 멀티모달 AI 연구

코드 변경 없이 이미지, 비디오, 오디오에 적용 가능하다는 점은, **멀티모달 파운데이션 모델**의 효율화에 직접 적용 가능하며, 향후 멀티모달 연구의 중요한 기반이 될 수 있다.

### 5.2 향후 연구 시 고려할 점

#### 5.2.1 조밀 예측 태스크로의 확장

현재 ToMe는 분류 태스크에서 검증되었다. **객체 탐지, 인스턴스 세그멘테이션, 깊이 추정** 등 공간 정보가 중요한 태스크에서는 토큰 병합이 위치 정보를 손상시킬 수 있다. 병합 후에도 공간 정보를 보존하는 방법이 필요하다.

#### 5.2.2 비병합 토큰의 선택적 보호

현재 알고리즘은 유사한 토큰을 병합하지만, **의미론적으로 중요한 토큰**(예: 희귀 객체, 작은 세부 사항)을 보호하는 메커니즘이 없다. 중요도 점수와 유사도를 결합하는 방법을 고려할 수 있다.

#### 5.2.3 적응적 $r$ 값 선택

현재 $r$은 고정값이다. **입력 복잡도에 따른 동적 $r$ 선택** (단, 배치 추론 가능성 유지)은 정확도-속도 트레이드오프를 개선할 수 있다. 예를 들어, 학습 가능한 경량 predictor를 통해 레이어별 $r$을 결정하는 방법이 있다.

#### 5.2.4 소형 모델에서의 정확도 하락 문제

ViT-Ti/S/B 등 소형 모델에서는 정확도 하락이 크다. 이를 해결하기 위해:

- ToMe를 고려한 **처음부터의 학습(training from scratch)** 전략
- **Knowledge Distillation**과의 결합
- **더 정교한 병합 기준** (단순 코사인 유사도 외 의미론적 정보 활용)

등을 고려할 수 있다.

#### 5.2.5 LLM/디퓨전 모델로의 확장

ToMe의 핵심 아이디어는 어텐션 기반 모델 전반에 적용 가능하다. **Large Language Model (LLM)** 에서의 KV-cache 최적화나 **Diffusion Model**의 U-Net 가속에 적용하는 연구가 진행되고 있으며, 이 방향에서의 이론적 분석이 필요하다.

#### 5.2.6 병합의 이론적 이해

왜 Key 행렬 기반 코사인 유사도가 가장 효과적인지, 병합이 왜 정규화 효과를 가지는지에 대한 **이론적 분석**이 부족하다. 정보 이론적 관점에서의 분석이 향후 연구의 방향을 제시할 수 있다.

#### 5.2.7 하드웨어 최적화와의 결합

ToMe는 현재 소프트웨어 레벨의 최적화이다. **FlashAttention, Sparse Attention** 등 하드웨어 친화적 최적화와의 결합을 통해 추가적인 속도 향상이 가능할 것이다.

---

## 참고 자료

**주요 논문 (직접 인용)**

1. **Bolya, D., Fu, C.-Y., Dai, X., Zhang, P., Feichtenhofer, C., & Hoffman, J. (2023).** "Token Merging: Your ViT But Faster." *ICLR 2023.* arXiv:2210.09461v3.

**논문 내 참조 문헌 (주요)**

2. Dosovitskiy, A., et al. (2020). "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale." *ICLR 2021.*
3. He, K., et al. (2022). "Masked Autoencoders Are Scalable Vision Learners." *CVPR 2022.*
4. Rao, Y., et al. (2021). "DynamicViT: Efficient Vision Transformers with Dynamic Token Sparsification." *NeurIPS 2021.*
5. Yin, H., et al. (2022). "A-ViT: Adaptive Tokens for Efficient Vision Transformer." *CVPR 2022.*
6. Kong, Z., et al. (2022). "SPViT: Enabling Faster Vision Transformers via Soft Token Pruning." *ECCV 2022.*
7. Liang, Y., et al. (2022). "Not All Patches Are What You Need: Expediting Vision Transformers via Token Reorganizations." *ICLR 2022.*
8. Liu, Z., et al. (2021). "Swin Transformer: Hierarchical Vision Transformer Using Shifted Windows." *ICCV 2021.*
9. Li, Y., et al. (2022). "MViTv2: Improved Multiscale Vision Transformers for Classification and Detection." *CVPR 2022.*
10. Feichtenhofer, C., et al. (2022). "Masked Autoencoders as Spatiotemporal Learners." *NeurIPS 2022.*
11. Huang, P.-Y., et al. (2022). "Masked Autoencoders That Listen." *NeurIPS 2022.*
12. Marin, D., et al. (2021). "Token Pooling in Vision Transformers." arXiv:2110.03860.
13. Singh, M., et al. (2022). "Revisiting Weakly Supervised Pre-Training of Visual Perception Models." *CVPR 2022.*
14. Steiner, A., et al. (2022). "How to Train Your ViT? Data, Augmentation, and Regularization in Vision Transformers." *TMLR 2022.*
15. Vaswani, A., et al. (2017). "Attention Is All You Need." *NeurIPS 2017.*

**공개 코드**
- GitHub: [facebookresearch/ToMe](http://github.com/facebookresearch/ToMe)
