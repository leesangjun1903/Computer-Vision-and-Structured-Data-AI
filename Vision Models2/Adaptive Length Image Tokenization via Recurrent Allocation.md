# Adaptive Length Image Tokenization via Recurrent Allocation (ALIT)

> **참고 문헌**: Shivam Duggal, Phillip Isola, Antonio Torralba, William T. Freeman, *"Adaptive Length Image Tokenization via Recurrent Allocation"*, arXiv:2411.02393 (ICLR 2025 accepted). 참고 사이트: arXiv abstract/HTML (arxiv.org/abs/2411.02393, arxiv.org/html/2411.02393v1), OpenReview (openreview.net/forum?id=mb2ryuZ3wz), ICLR 2025 Proceedings PDF (proceedings.iclr.cc), GitHub 공식 저장소(github.com/shivamduggal4/adaptive-length-tokenizer), Semantic Scholar, ResearchGate, HuggingFace Papers, Liner Quick Review, TheMoonlight Literature Review, DeepLearn.org, 그리고 후속연구 비교를 위한 KARL(arxiv 2507.07995), FlexTok(arxiv 2502.13967), DOVE(arxiv 2506.03643), ALTo(arxiv 2505.16495), EmergentMind topic pages.

---

## 1. Executive Summary (10문장 이내)

현재의 비전 시스템은 정보 내용과 관계없이 이미지에 고정 길이 표현을 할당하는데, 이는 엔트로피·맥락·친숙도에 따라 표현 용량을 달리 배분하는 인간 지능 및 대형 언어모델과 대조적이다.  
이에 저자들은 2D 이미지에 대한 가변 길이 토큰 표현을 학습하는 접근법을 제안하며, 인코더-디코더 아키텍처가 여러 차례의 반복적 롤아웃(recurrent rollout)을 통해 2D 이미지 토큰을 재귀적으로 처리해 1D 잠재 토큰으로 증류(distill)한다.  
각 반복은 2D 토큰을 정제하고, 기존 1D 잠재 토큰을 업데이트하며, 새로운 토큰을 추가해 표현 용량을 적응적으로 증가시킨다.  
이를 통해 이미지를 32개에서 256개 사이의 가변적인 토큰 수로 압축할 수 있다.  
이 방법은 **ALIT(Adaptive Length Image Tokenizer)**로 명명되었으며, 자기지도 이미지 재구성 목표로 학습된다.  
표현 용량이 커짐에 따라 각 재귀적 업데이트는 잠재 토큰이 국소 영역에 특화·주목하도록 이끌어 물체/부분 발견(object/part discovery)의 가능성을 시사한다.  
저자들은 VQGAN(Esser et al., 2020) 및 고정 길이 1D 토크나이저 TiTok(Yu et al., 2024)과 비교해 유사한 재구성 지표(L1 손실, FID)와 ImageNet-1K 선형 프로빙 성능을 보이면서도 이미지별로 유연한 토큰 수를 허용함을 입증했다.  
동적 정지(dynamic halting) 기법을 통해 잘 재구성된 토큰은 추가 처리에서 제외되어, 모델이 아직 덜 증류된 영역에 계산 자원을 집중할 수 있다.  
예컨대 8층 네트워크인 ALIT-S는 32~256개의 가변 토큰을 학습하면서도, 32개 고정 토큰을 위해 24층 인코더/디코더를 사용하는 TiTok-L-32와 대등한 재구성 품질을 달성한다.  
이 연구는 이미지 토큰화에서 "고정 길이"라는 암묵적 전제를 깨고, 콘텐츠 기반 적응형 압축이라는 새로운 연구 흐름(ElasticTok, FlexTok, KARL, DOVE 등)의 초석이 되었다.

### 1-1. 연구의 목적과 필요성

기존 비전 토크나이저(VQGAN, ViT, MAE 등)는 이미지의 2D 공간적 귀납적 편향(inductive bias)에 크게 의존하여 2D 패치를 토큰으로 취급한다. 이는 이미지의 복잡도, 친숙도, 다운스트림 과제 요구사항과 무관하게 항상 동일한 개수의 토큰을 사용하는 비효율을 낳는다. 저자들은 1D 토큰화가 패치-토큰 제약을 극복하지만, 더 보편적인 토크나이저는 콘텐츠 엔트로피·친숙도 등에 기반해 가변 토큰을 적응적으로 할당하는 것이라고 지적한다. 이러한 문제의식에서 입력 시각 관찰(visual observation)을 점진적으로 증가하는 1D 잠재 토큰으로 자동회귀적으로 증류하는 방식으로 적응형/가변 길이 표현 학습 문제를 다루고, 이를 ALIT로 명명했다.

---

## 2. 핵심 주장과 근거 (표 정리)

| # | 핵심 주장 | 근거 (논문 인용) | 위치 |
|---|---|---|---|
| 1 | 고정 길이 토큰화는 정보량과 무관하게 동일 자원을 소모해 비효율적 | "Current vision systems typically assign fixed-length representations to images... This contrasts with human intelligence... which allocate varying representational capacities based on entropy, context and familiarity." | Abstract, 서론 |
| 2 | 재귀적 롤아웃으로 2D→1D 토큰 증류, 반복마다 토큰 추가 | "recurrent computing involves recursively distilling an input image or 2D image tokens into 1D latent tokens through a shared encoder-decoder architecture until each image token has been sufficiently processed... At each iteration... we provide additional computational resources in the form of new learnable latent tokens" | 본문(Method), arXiv HTML |
| 3 | 32~256개의 가변 토큰 수로 압축 가능 | "This enables compression of images into a variable number of tokens, ranging from 32 to 256." | Abstract |
| 4 | VQGAN·TiTok 대비 유사 재구성 성능 + 토큰 유연성 확보 | "comparable reconstruction metrics (L1 loss and FID) and linear probing results on ImageNet-1K, relative to the 2D VQGAN tokenizer... and the fixed-latent 1D tokenizer, Titok... while also allowing for flexible token counts per image" | Sec. 4 (결과) |
| 5 | 재귀적 처리로 토큰 특화(specialization) → 물체/부분 발견 가능성 | "provide additional computational resources in the form of new learnable latent tokens, enabling the model to learn adaptive and variable-length representations across different iterations... each recurrent update leads to the latent tokens specializing and attending to localized regions, hinting at object / part discovery" | Fig. 7, 8, Appendix Fig. 15, 16 (ICLR판) / Fig. 8, 9, Appendix Fig. 13, 16 (arXiv v1판) |
| 6 | 동적 정지(halting)로 계산 자원을 미완성 영역에 집중 | "dynamic halting, where well-reconstructed tokens are masked from further processing, allowing the model to focus computational resources on less-distilled regions" | 본문(Method) |
| 7 | 더 적은 층수로도 대형 고정형 모델과 대등한 성능 | "ALIT-S, with an 8-layer network, can learn variable tokens (32 to 256) and achieve reconstruction quality on par with Titok-L-32, which uses a 24-layer encoder/decoder for a fixed 32 tokens" | Table (본문 실험) |
| 8 | 다양한 데이터셋(ImageNet, COCO, Places, 예술 데이터셋, 인터넷 이미지)에서 검증 | "comparable reconstruction metrics (L1 loss and FID) on multiple datasets (IN, COCO, Places, Art-dataset and even randomly selected internet images Fig. 20) and linear probing results on ImageNet-1K" | Fig. 20 |
| 9 | 저해상도(적은 토큰)에서 FID 손실 존재 인정 | "conjecture that multiple factors could enable further improving the slight FID loss (at low tokens) when training an amortized architecture for variable-length" | 결론/한계 |

---

## 2-1. 문제, 방법(수식 포함), 모델 구조, 성능, 한계 상세 설명

### (1) 해결하고자 하는 문제
VAE, VQGAN, ViT와 같은 대표적인 토크나이저들은 이미지의 2D 공간적 귀납 편향에 강하게 의존하여 2D 패치를 토큰으로 취급하는데, 이는 토크나이저 구조를 이미지 해상도/그리드 구조에 강하게 결합시킨다. 또한 기존 인코더-디코더 방식, 특히 트랜스포머 기반 방법들은 고정 패치 수준 토큰화에 제한되어 적응형 표현과 다양한 이미지에 대한 효율적 압축을 방해한다. 즉, 배경이 단순한 이미지와 복잡한 장면이 담긴 이미지에 동일한 토큰 예산이 할당되는 것이 근본적 비효율이다.

### (2) 제안 방법 — 개념적 수식화
*※ 아래 수식은 검색으로 확인된 논문의 서술("반복적으로 2D 토큰을 1D 잠재 토큰으로 증류하고, 매 반복마다 새 토큰을 추가한다")을 바탕으로 제가 이해하기 쉽게 재구성한 것이며, 원 논문에 실린 정확한 수식 번호/기호와 100% 동일하다고 확언할 수는 없습니다 (원문 수식 전체를 직접 확인하지 못했음을 밝힙니다).*

입력 이미지를 $X \in \mathbb{R}^{H \times W \times 3}$ 라 하고, 2D 이미지 인코더 $E_{2D}$ 가 이를 패치 토큰 집합 $\{p_1, \dots, p_N\}$ 으로 변환한다고 하자.

$$
P^{(0)} = E_{2D}(X), \quad P^{(0)} = \{p_1^{(0)}, \dots, p_N^{(0)}\}
$$

반복 $t = 1, \dots, T$ 에 대해, 이전 잠재 토큰 집합 $Z^{(t-1)} = \{z_1, \dots, z_{k_{t-1}}\}$ 에 새로운 학습 가능한 토큰 $\Delta Z^{(t)}$ 을 추가한다:

$$
Z'^{(t)} = Z^{(t-1)} \cup \Delta Z^{(t)}, \qquad k_t = k_{t-1} + |\Delta Z^{(t)}|
$$

증류(distillation)는 2D 토큰과 잠재 토큰 간의 교차어텐션(cross-attention) 및 잠재 토큰 간 자기어텐션(self-attention)을 반복하는 공유 인코더-디코더로 수행된다:

$$
Z^{(t)} = f_\theta\big(P^{(t-1)}, Z'^{(t)}\big), \qquad P^{(t)} = g_\theta\big(P^{(t-1)}, Z^{(t)}\big)
$$

디코더 $D_\theta$ 는 잠재 토큰만으로 이미지를 재구성한다:

$$
\hat{X}^{(t)} = D_\theta(Z^{(t)})
$$

학습 목적함수는 각 반복 단계에서 재구성 손실(L1) 및 (전체 미세조정 단계에서) 적대적/지각 손실을 합산한 형태로 구성된다:

$$
\mathcal{L} = \sum_{t=1}^{T} \lambda_t \Big( \|X - \hat{X}^{(t)}\|_1 + \mathcal{L}_{\text{GAN}}(X, \hat{X}^{(t)}) \Big)
$$

- $X$: 원본 이미지, $\hat{X}^{(t)}$: 반복 $t$ 시점의 재구성 이미지
- $P^{(t)}$: 반복 $t$ 시점의 정제된 2D 토큰 집합
- $Z^{(t)}$: 반복 $t$ 시점의 누적 1D 잠재 토큰 집합 (개수 $k_t$)
- $\Delta Z^{(t)}$: 반복 $t$에서 새로 추가되는 학습 가능한 토큰(추가 "메모리")
- $f_\theta, g_\theta, D_\theta$: 공유 파라미터를 사용하는 인코더/디코더 함수
- $\lambda_t$: 반복 단계별 손실 가중치
- $T$: 총 롤아웃(반복) 횟수 — 동적 정지(dynamic halting) 시 이미지별로 가변적

이 정식화는 "recurrent computing involves recursively distilling an input image or 2D image tokens into 1D latent tokens through a shared encoder-decoder architecture until each image token has been sufficiently processed/distilled into the latent tokens. At each iteration... we provide additional computational resources in the form of new learnable latent tokens"라는 저자들의 서술을 반영한 것입니다.

**동적 정지(Dynamic Halting)**: 저자들은 영역별로 추가 정제가 필요한 부분에 대한 향상된 증류를 위해 선택적 동적 정지(dynamic halting) 기능도 제공한다고 밝혔다. 이는 잘 재구성된 토큰을 이후 처리에서 마스킹(masking)하여 모델이 아직 덜 정제된 영역에 계산 자원을 집중하도록 하는 방식으로, 이미지 내에서도 공간적으로 불균일한 자원 배분을 가능하게 한다.

### (3) 모델 구조
아키텍처는 2D 이미지 토큰을 재귀적으로 처리해 1D 잠재 토큰으로 증류하는 인코더-디코더 구조이며, 인코더는 2D 시각 데이터를 잠재 공간으로 압축하면서 토큰 간 자기어텐션을 수행해 관련 특징을 포착한다. 이 구조는 RIN(Recurrent Interface Networks, Jabri et al. 2022)류의 "인터페이스-잠재 분리 + 교차어텐션 라우팅" 철학과 유사하며, 다만 ALIT는 잠재 토큰 개수 자체를 반복마다 늘려간다는 점이 차별점이다. GitHub 저장소에 따르면 ALIT는 두 단계(잠재 증류 사전학습, GAN 손실을 포함한 전체 미세조정)로 학습되며, 1단계에서는 이미지 인코더/디코더를 고정한 채 잠재-증류 인코더/디코더 모듈만 학습한다.

### (4) 성능 향상
- VQGAN, TiTok과 비교해 유사한 L1/FID 재구성 지표와 ImageNet-1K 선형 프로빙 성능을 보이면서도 이미지별 가변 토큰 수를 허용 (Sec. 4)
- 더 얕은 네트워크(ALIT-S, 8층)로 훨씬 깊은 고정형 모델(TiTok-L-32, 24층)과 대등한 재구성 품질 달성 — 파라미터/연산 효율성 측면의 강점
- ALIT는 고정 길이 베이스라인 대비 우수한 L1 재구성 손실을 보이며, 특히 복잡도가 다른 이미지들에서 적응형 토큰 할당의 이점이 뚜렷하다. 고복잡도 이미지는 적은 토큰으로도 전역 정렬을 유지하며, 저복잡도 이미지는 최소 토큰으로 충분히 재구성된다. (Liner 리뷰 요약)
- 반복적 토큰 처리 및 반복마다 증가하는 표현 용량이 토큰 특화의 징후를 보이며, 물체/부분 발견의 가능성을 드러낸다.

### (5) 한계
- 저자들 스스로 적은 토큰 수에서의 FID 손실이 다소 존재함을 인정하며, 가변 길이용 상각(amortized) 아키텍처 학습 시 이를 개선할 여러 요인이 있을 것이라 추측한다.
- 저자들은 논문의 Fig. 9가 더 큰 모델 크기·더 긴 학습·더 큰 데이터셋으로 확장했을 때의 잠재력을 보여주지만, 학계 수준의 컴퓨팅 자원 제약으로 실제 대규모 확장 실험은 수행하지 못했다고 명시한다. 이는 스케일업 가능성이 검증되지 않은 채 시사에 그친 부분입니다.
- 고정 길이 토크나이저가 특정 학습 트레이드오프 덕분에 지각 품질(FID) 면에서 더 나을 수 있다는 트레이드오프도 지적된다.

---

## 3. 페이지/Figure/Table 표시

- 서론/문제제기: Abstract, 서론부
- 방법론(재귀 롤아웃, 동적 정지): 본문 Method 섹션
- 물체/부분 발견 관련: **Fig. 7, 8, Appendix Fig. 15, 16** (ICLR 2025 최종판 기준) / **Fig. 8, 9, Appendix Fig. 13, 16** (arXiv v1 초기판 기준) — 버전 간 그림 번호 불일치 존재 (§5 참고)
- 엔트로피-압축 가설 검증: **Fig. 3** (Out-of-Distribution People 데이터셋 대상 압축 대 정보 엔트로피 가설)
- 다양한 데이터셋(ImageNet, COCO, Places, Art, 인터넷 이미지) 재구성 검증: **Fig. 20**
- 스케일업 잠재력 시연: **Fig. 9** (GitHub README에서 언급)
- 아키텍처/학습 단계 설명: 본문 Sec. 3~4, GitHub README "Training" 섹션

---

## 4. 저자 보고 결과 vs. 저의 해석 분리

| 구분 | 내용 |
|---|---|
| **저자 보고 (원문 근거)** | VQGAN, TiTok 대비 비교 가능한 L1/FID 및 선형 프로빙 성능; 토큰 특화 및 물체/부분 발견 가능성; 32~256 토큰 범위의 압축; 저토큰 구간에서의 FID 손실 인정 |
| **저의 해석/추론** | (1) 위 수식화는 원문 수식 번호와 정확히 일치하지 않을 수 있는 개념적 재구성입니다. (2) "ALIT-S가 TiTok-L-32와 대등하다"는 비교는 서로 다른 모델 크기·훈련 예산 간 비교이므로, 엄밀한 통제 실험이 아니라 "효율성 증거"로 해석하는 것이 타당합니다. (3) 후속 논문(DOVE, KARL)에서 ALIT를 베이스라인으로 재현했을 때 성능이 열세로 나온 것은, 원 저자의 결과가 아니라 제3자의 재현 실험 결과이므로 별도로 취급해야 합니다 — 예: "Table 2: FID scores (↓) across the ImageNet100, COCO, and WIT datasets... Our method consistently outperforms ALIT across all token lengths" (이는 DOVE 논문의 자체 비교이며 학습 세팅 차이가 있을 수 있음을 유의해야 합니다). |

---

## 5. 통계적으로 취약한 부분 / 비교 불가능한 수치

1. **모델 크기 불일치 비교**: ALIT-S(8층)와 TiTok-L-32(24층)의 비교는 파라미터 수·연산량이 다른 모델 간 비교로, "동일 조건 통제 실험"이 아니라는 점에서 통계적 엄밀성이 떨어집니다.
2. **데이터셋별 FID 격차의 일반화 불가**: "64~256 토큰 사이 FID 격차는 분포 내(in-distribution)인 ImageNet-100 검증셋에서 가장 작고(7.92), 분포 외(out-of-distribution) 성격이 강한 COCO에서 더 크다"는 결과는 특정 두 데이터셋만을 대상으로 하며, 다른 도메인(의료 영상, 위성 사진 등)에 대한 일반화는 검증되지 않았습니다.
3. **제3자 재현 결과의 신뢰성**: DOVE 논문의 FID 비교 표(ALIT 22.31~8.06, DOVE 18.91~7.73 등)는 원저자가 아닌 후속 연구팀이 재구현/재훈련한 수치일 가능성이 높아, 원 논문이 보고한 수치와 직접 비교하기 어렵습니다(학습 스텝 수, 데이터 전처리, GAN 손실 가중치 등이 다를 수 있음).
4. **스케일업 주장의 미검증**: Fig. 9가 스케일업 잠재력을 보여주지만, 실제 대규모 확장 실험은 컴퓨팅 자원 제약으로 수행되지 않았다는 점에서, "확장하면 더 좋아질 것"이라는 주장은 외삽(extrapolation)에 가깝고 직접적 통계적 근거가 부족합니다.
5. **시드/반복 실험 정보 부재**: 검색된 자료에서 FID/L1 수치에 대한 표준편차, 다중 시드 반복 실험 여부는 확인되지 않았습니다 — 즉 보고된 단일 수치들의 통계적 변동성은 불명확합니다.

---

## 6. 문서(검색 결과)가 답하지 않는 질문들

- 정확한 반복 횟수(T)의 평균/최댓값 및 동적 정지 임계값(threshold)의 구체적 수치는 무엇인가?
- 추론 시 재귀 반복으로 인한 실제 지연시간(latency) 오버헤드는 고정형 토크나이저 대비 어느 정도인가?
- 코드북 크기, 양자화(quantization) 방식의 세부 하이퍼파라미터는?
- 동영상·3D·멀티모달(텍스트-이미지 결합) 데이터로의 확장 가능성에 대한 실험적 근거는?
- 다운스트림 생성 모델(diffusion, autoregressive generation)과 결합했을 때의 생성 품질(gFID 등)은? (본 논문은 재구성/분류 중심이며, 생성 태스크로의 확장은 후속 연구(FlexTok, KARL 등)에서 다뤄짐)
- 물체/부분 발견이 "정성적 관찰"을 넘어 정량적 세그멘테이션 벤치마크로 검증되었는가?
- 학습에 사용된 총 GPU 시간, 탄소 발자국 등 자원 소모 정보는?

---

## 7. 가장 중요한 그림 5개 해석

1. **Fig. 3 (압축 vs. 정보 엔트로피 가설, Out-of-Distribution People 데이터셋)**: "Compression vs. Information Entropy Hypothesis on the Out-of-Distribution People-..." — 분포 외 데이터에서도 이미지의 정보 엔트로피(복잡도)와 필요한 토큰 수 사이의 상관관계가 유지되는지를 검증하는 핵심 그림으로, ALIT의 "적응적 할당이 실제로 콘텐츠 복잡도를 반영한다"는 중심 주장의 실증적 근거입니다.

2. **Fig. 7~9 계열 (토큰 특화·물체/부분 발견)**: "each recurrent update leads to the latent tokens specializing and attending to localized regions, hinting at object / part discovery (see, Fig. 7, Fig. 8, Appendix Fig. 15 and Fig. 16)" — 재귀 반복이 진행될수록 각 잠재 토큰이 이미지의 특정 국소 영역(예: 얼굴, 배경, 특정 물체 부분)에 어텐션을 집중시키는 시각화로, "적응형 메모리가 창발적 의미 구조를 학습한다"는 주장의 시각적 근거입니다. (단, 버전별로 그림 번호가 상이함에 유의)

3. **Fig. 9 (스케일업 잠재력)**: GitHub README에 따르면 "논문의 Fig. 9는 적응형 토크나이저를 더 큰 모델 크기, 더 긴 학습, 더 큰 데이터셋으로 확장하는 것의 위력을 명확히 보여준다"고 설명되어 있으나, 저자들은 학계 수준의 컴퓨팅 자원으로는 이를 직접 수행하지 못했다고 밝힙니다. 즉 이 그림은 "잠재력의 시사"로만 해석해야 하며, 완전한 검증으로 받아들여서는 안 됩니다.

4. **Fig. 20 (다양한 도메인 재구성 비교 — ImageNet, COCO, Places, Art, 인터넷 임의 이미지)**: "comparable reconstruction metrics (L1 loss and FID) on multiple datasets (IN, COCO, Places, Art-dataset and even randomly selected internet images Fig. 20)" — 모델이 특정 학습 분포에 과적합되지 않고 다양한 도메인에서도 일관되게 작동함을 보여주려는 일반화 검증용 그림입니다.

5. **L1 재구성 손실 히스토그램(ImageNet-100)**: 고정 길이 베이스라인 대비 ALIT의 L1 재구성 손실 우위를 보여주는 히스토그램으로, 고복잡도 이미지는 적은 토큰으로도 전역 정렬을 유지하고 저복잡도 이미지는 최소 토큰으로 충분히 재구성됨을 보여준다 — 적응형 토큰 배분이 실제 픽셀 수준 정확도에 미치는 영향을 정량적으로 뒷받침하는 그림입니다.

---

## 8. 결론: 시사점, 후속 연구 계획, 추가 방향

### 저자들이 직접 제시한 시사점 및 계획
저자들은 가변 표현과 누적 데이터셋 표현(cumulative dataset representations)을 도입함으로써, 효과적 표현의 핵심 요소인 '요구되는 용량이 이미지의 복잡도·친숙도·다운스트림 모델/과제에 따라 달라진다'는 점을 강조한다고 밝히며, "재귀적 처리와 적응 메모리 — 잠재 토큰을 반복적으로 정제하고 추가하는 방식 — 는 향후 더 긴 롤아웃을 통한 스트리밍 데이터용 더 긴 표현으로 가는 문을 연다"고 후속 연구 방향을 제시합니다.

### 8-1. 모델의 일반화 성능 향상 가능성
ALIT의 일반화 잠재력은 다음 세 가지 근거에서 확인됩니다.
1. ImageNet 외에도 COCO, Places, 예술 작품 데이터셋, 무작위 인터넷 이미지에서도 유사한 재구성 성능을 보였다는 점은 도메인 간 강건성(robustness)을 시사합니다.
2. 다만 FID 성능 격차가 분포 내(ImageNet-100)보다 분포 외(COCO)에서 더 크다는 사실은, 완전한 분포 불변성은 아직 확보되지 않았음을 보여줍니다 — 이는 향후 도메인 적응(domain adaptation) 기법이나 더 다양한 사전학습 데이터로 개선될 여지가 있는 지점입니다.
3. **저의 해석**: 재귀적 "필요한 만큼만 계산"하는 구조 자체가 각 인스턴스의 난이도에 맞춰 적응하므로, 원리적으로는 분포 이동(distribution shift)에 대해 고정 길이 모델보다 더 유연하게 대응할 잠재력이 있습니다. 그러나 이는 저자들이 인정한 컴퓨팅 자원 제약으로 대규모·다양한 데이터에서의 스케일업이 검증되지 않았다는 한계와 맞물려, "잠재력은 있으나 실증되지 않은 가설"로 보는 것이 정확합니다.

### 8-2. 2020년 이후 관련 최신 연구 비교 분석 및 향후 고려사항

| 연구 | 핵심 아이디어 | ALIT와의 관계 |
|---|---|---|
| **Perceiver** (Jaegle et al., 2021) | 2D 이미지 토큰을 특정 패치에 묶이지 않는 1D 잠재 토큰으로 증류, 모달리티 비특정적 트랜스포머를 지향 | ALIT의 1D 잠재 토큰 개념의 직접적 전신 |
| **RIN** (Jabri et al., 2022) | 데이터 차원과 분리된 잠재 토큰에 계산을 집중시키고, 교차어텐션으로 데이터-잠재 간 정보를 라우팅하는 적응형 계산 아키텍처 | ALIT의 인코더-디코더 라우팅 구조와 유사한 철학 공유 |
| **AdaTape** (Xue et al.) | 고정 길이 토크나이저(RIN, TiTok)와 달리, 테이프 토큰의 일회성 선택을 허용 | ALIT은 반복적/누적적 할당이라는 점에서 차별화 |
| **ElasticTok** (Yan et al., 2024) | Matryoshka 표현과 유사하게, 한 번에 고정된 최대 크기 표현을 학습한 뒤 이 표현의 부분집합을 샘플링하는 마스크를 탐색 | ALIT과 동시기 발표된 대안적 가변 길이 접근법 |
| **FlexTok** (2025) | 기존 방법들은 고정된 토큰 수를 사용해 이미지 고유 복잡도에 적응할 수 없다는 문제의식 하에, 2D 이미지를 가변 길이의 순서화된 1D 토큰 시퀀스로 투영. ALIT 등 다른 1D 토크나이저 대비 단일 모델에서 1개 토큰까지 고도로 의미론적이고 순서화된 방식으로 토큰화 가능 | 직접적 후속·경쟁 연구로, ALIT를 명시적 베이스라인으로 비교(Fig. 4) |
| **DOVE** (2025, "Images are Worth Variable Length of Representations") | 모든 토큰 길이에서 ALIT을 일관되게 능가하며, VQGAN·TiTok과도 비슷하거나 더 나은 결과를 여러 길이에서 달성 | ALIT의 성능을 뛰어넘는 것을 목표로 한 직접적 벤치마크 경쟁자 |
| **KARL** (Duggal et al., 2025, 동일 제1저자) | Matryoshka류가 여러 디코더 패스를 요구하는 데 반해, ALIT과 같은 재귀적 접근은 반복적 인코더-디코더 루프로 재구성 품질을 충족시키는 반면, KARL은 콜모고로프 복잡도(Kolmogorov Complexity)에 기반한 단일 패스 인코더를 사용해 반복적 정제를 회피한다. KARL의 정지 메커니즘은 최소 기술 길이(minimum description length)에서 영감을 받아 단일 패스 토큰화로 4~8배 적은 패스로 경쟁력 있는 FID/LPIPS를 달성한다. | **ALIT의 직접적 후속 연구**로, ALIT의 핵심 한계(반복적 추론 비용)를 정면으로 해결하려는 시도 |
| **ALTo** (2025) | "ALIT discretizes the images into flexible-length tokens by recurrent distillation until the reconstruction quality is good or the maximum iterations are met"이라 기술하며, 이를 세그멘테이션 마스크 생성용 적응형 토크나이저로 확장 | ALIT의 아이디어를 마스크 생성 등 새로운 응용 영역으로 확장 |

**논문이 향후 연구에 미치는 영향**: ALIT는 "이미지 토큰화도 콘텐츠 적응적이어야 한다"는 프레임을 명확히 제시함으로써, 이후 2년간 ElasticTok, FlexTok, DOVE, KARL, ALTo, GPSToken, STAT 등 매우 다양한 적응형 토크나이저 연구를 촉발한 것으로 보입니다. 특히 KARL과 같은 후속 연구는 ALIT의 접근을 콜모고로프 복잡도·최소 기술 길이의 신경망적 대리(neural surrogate)로 재해석하며 알고리즘 정보이론과의 접점을 만들었습니다.

**향후 연구 시 고려할 점**:
1. **추론 효율성**: ALIT의 반복적 롤아웃 구조는 재구성 품질 대비 추론 비용(다중 패스)이 크다는 점이 KARL 등 후속 연구에서 지적된 핵심 한계이므로, 단일 패스/저지연 적응형 토큰화 설계가 중요한 고려사항입니다.
2. **공정한 벤치마킹**: 서로 다른 연구팀이 각기 다른 학습 설정으로 ALIT을 재구현·비교하고 있어(§5 참조), 향후 연구에서는 통일된 학습 프로토콜(동일 데이터, 동일 GAN 손실 가중치, 동일 학습 스텝)을 사용한 비교가 필요합니다.
3. **일반화 검증의 확대**: 이미지 도메인을 넘어 동영상·3D·멀티모달 데이터로의 확장, 그리고 실제 스케일업 실험(대형 모델·대형 데이터셋)을 통한 검증이 후속 과제로 남아 있습니다.
4. **정성적 관찰의 정량화**: 물체/부분 발견과 같은 창발적 현상에 대해 세그멘테이션 IoU 등 정량적 벤치마크를 도입해 검증할 필요가 있습니다.
