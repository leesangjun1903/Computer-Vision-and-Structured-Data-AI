# Back to Basics: Let Denoising Generative Models Denoise

**저자**: Tianhong Li, Kaiming He (MIT) | **논문번호**: arXiv:2511.13720 | **학회**: CVPR 2026 게재 예정

---

## Executive Summary (10문장 이내)

오늘날의 디노이징 확산 모델은 실제로 깨끗한 이미지를 직접 예측하지 않으며, 신경망은 노이즈 또는 노이즈가 섞인 양(quantity)을 예측하도록 학습된다.  
이 논문은 깨끗한 데이터(clean data)를 예측하는 것과 노이즈화된 양을 예측하는 것이 근본적으로 다르다고 주장하며, 매니폴드 가정(manifold assumption)에 따르면 자연 데이터는 저차원 매니폴드 위에 존재하지만 노이즈화된 양은 그렇지 않다고 설명한다.  
이 가정을 바탕으로, 저자들은 깨끗한 데이터를 직접 예측하는 모델을 제안하며, 이는 겉보기에 용량(capacity)이 부족해 보이는 네트워크도 매우 고차원 공간에서 효과적으로 작동할 수 있게 한다.  
이를 검증하기 위해 토크나이저나 사전학습, 추가 손실함수 없이 픽셀 패치에 대해 작동하는 단순한 대형 패치 Transformer가 강력한 생성 모델이 될 수 있음을 보이며, 이를 "Just image Transformers"(JiT)라 명명한다.  
JiT는 패치 크기 16과 32를 사용해 ImageNet 256×256, 512×512 해상도에서 경쟁력 있는 결과를 보고하며, 이 조건에서는 고차원 노이즈화된 양을 예측하는 방식이 파국적으로 실패할 수 있다.  
이 연구는 확산 모델 설계에서 오랫동안 당연시되어온 "무엇을 예측할 것인가"라는 질문을 원점에서 재검토하는 시도다.

---

### 1-1. 연구의 목적과 필요성

확산 모델의 원조 연구는 네트워크가 정규분포의 평균과 표준편차 같은 파라미터를 예측하는 역방향 확률과정을 학습하도록 제안했으며, 이후 DDPM에 의해 대중화되었다.  
이 과정에서 ε-prediction(노이즈 예측) 방식이 표준으로 자리 잡았고, 이후 flow matching/rectified flow 계열에서는 v-prediction(속도 예측)이 널리 쓰이게 되었다.  
그러나 x-prediction이 손실 가중치를 적절히 재구성하면 ε- 및 v-prediction과 밀접하게 연관되어 있다는 이유로, 정작 네트워크가 "무엇을 예측해야 하는가"라는 근본 질문에는 상대적으로 관심이 적었다는 것이 이 논문의 문제의식이다.  
저자들은 매니폴드 가정을 통해 이 선택이 결코 사소하지 않으며, 특히 픽셀 공간처럼 초고차원 데이터를 다룰 때는 결정적 차이를 만든다는 점을 규명하고자 한다.  
이는 잠재공간(latent space) 확산 모델이 VAE 토크나이저에 의존해야 했던 관행에서 벗어나, 원본 픽셀에서 직접 작동하는 "자기완결적(self-contained)" 생성 모델 패러다임을 모색하려는 실용적 필요성과도 맞닿아 있다.

---

## 2. 핵심 주장과 근거 요약표

| # | 핵심 주장 | 근거 | 위치 |
|---|---|---|---|
| 1 | 매니폴드 가정: 깨끗한 이미지는 저차원 매니폴드 위(on-manifold), 노이즈/속도는 고차원 전역에 분포(off-manifold) | 깨끗한 이미지 x는 온-매니폴드로 모델링될 수 있지만, 노이즈 ϵ이나 플로우 속도 v(예: v=x−ϵ)는 본질적으로 오프-매니폴드다 | Fig. 1 |
| 2 | x-prediction만이 초고차원 patch에서 안정적으로 작동 | 패치가 매우 고차원(수백~수천 차원)임에도 x-prediction을 사용한 모델은 쉽게 강력한 결과를 낼 수 있는 반면, ε- 및 v-prediction은 파국적으로 실패한다 | Table 2, p.4-5 |
| 3 | 네트워크 폭(width)이 패치 차원보다 작아도 무방하며, 병목(bottleneck)이 오히려 유리 | 병목 임베딩이 일반적으로 유익하며, 오히려 축소하는 병목을 도입하는 것이 성능을 개선한다 | Fig. 4, Sec. 4 |
| 4 | 토이(toy) 실험으로 가정을 직접 검증 | 2D 패턴을 512차원 공간에 숨긴 실험에서, 오직 깨끗한 이미지 예측기만 차원이 커져도 패턴을 복원하며 노이즈/혼합 예측은 실패한다 | Fig. 2, Sec. 3.3 |
| 5 | 대규모 ImageNet 실험에서 경쟁력 있는 FID 달성 | JiT-G/16이 FID 1.82를 달성해 DiT-XL/2(2.27), SiT-XL/2(2.06)보다 우수 | Table 13 |
| 6 | 해상도 확장 시에도 연산량이 거의 증가하지 않음 | 해상도에 비례한 패치 크기를 사용해 서로 다른 해상도에서도 시퀀스 길이가 동일하게 유지되며, 패치당 차원은 3072 또는 12288까지 커질 수 있으나 일반적 모델은 이만큼의 은닉 유닛을 갖지 않는다 | Table 8, p.8 |

---

## 2-1. 문제, 방법(수식), 모델 구조, 성능, 한계 상세 설명

### (1) 해결하고자 하는 문제

확산 모델은 "denoise"라는 이름과 달리 실제로는 깨끗한 이미지를 직접 예측하지 않고, 노이즈 또는 노이즈화된 양을 예측한다. 저자들은 이 관행이 왜 옳은지, 혹은 언제 실패하는지를 검토한 연구가 부족했다는 문제의식에서 출발한다.

### (2) 제안하는 방법 — 수식

**① 확산/플로우의 기본 노이즈화 과정** (일반적으로 통용되는 표준 정식화를 재구성한 것이며, 개념 이해를 돕기 위함)

$$x_t = \alpha_t x_0 + \sigma_t \epsilon,\qquad \epsilon \sim \mathcal{N}(0, I)$$

- $x_0$: 깨끗한 원본 데이터(이미지 패치)
- $x_t$: 시간(노이즈 레벨) $t$에서 노이즈가 섞인 데이터
- $\alpha_t, \sigma_t$: 노이즈 스케줄 계수
- $\epsilon$: 표준 가우시안 노이즈

**② 세 가지 예측 대상(prediction target)**

$$v = x - \epsilon$$

- $x$: x-prediction (깨끗한 데이터 직접 예측)
- $\epsilon$: ϵ-prediction (노이즈 예측)
- $v$: v-prediction (플로우 속도 예측, $v=x-\epsilon$로 정의)

**③ 토이 실험의 데이터 생성 수식**

관측 데이터는 열-직교(column-orthogonal) 행렬 $P \in \mathbb{R}^{D\times d}$($P^\top P = I_{d\times d}$)를 이용해 $\mathbf{x} = P\hat{\mathbf{x}} \in \mathbb{R}^D$로 합성되며, 여기서 저차원 실제 데이터는 $\hat{\mathbf{x}} \in \mathbb{R}^d$이다. 행렬 $P$는 모델에 알려지지 않아 이는 모델 입장에서 D차원 생성 문제가 된다.

- $D$: 관측(고차원) 공간 차원
- $d$: 실제 데이터의 내재적(intrinsic) 저차원 ($d<D$)
- $P$: 무작위로 생성되어 고정된 임베딩 행렬

이 설정에서 256차원 은닉 유닛을 가진 5개 층의 ReLU MLP를 생성기(generator)로 학습시켜 결과를 시각화한다(Fig. 2).

**④ 손실공간(loss space) × 예측공간(network output space) 조합**

손실 공간과 예측 공간의 9가지 가능한 조합을 Table 1에 정리했다. 타겟(target)은 네트워크 출력이 무엇을 나타내는지를 결정하고, 손실 공간은 어떤 선형 재매개변수화(reparameterization) 이후에 예측 오차를 측정할지를 결정한다.

### (3) 모델 구조 — JiT (Just image Transformers)

이미지를 $p\times p$ 크기 패치들로 나누어 $\frac{H}{p}\times\frac{W}{p}$ 길이의 시퀀스를 만들고, 각 패치는 $p\times p\times 3$차원 벡터가 된다. 이 시퀀스는 선형 임베딩 투영, 위치 임베딩 추가, Transformer 블록 스택을 거치며, 출력층은 각 토큰을 다시 $p\times p\times3$차원 패치로 되돌리는 선형 예측기다. (Fig. 3)

세부 아키텍처 요소: 조건화(conditioning)는 adaLN-Zero를 사용하며, SwiGLU, RMSNorm, RoPE, qk-norm, in-context class token(기본값 32개, 원본 ViT의 단일 class token보다 우수) 등 일반적 개선 기법이 층층이 적용된다.

**병목(bottleneck) 임베딩**: 원시 패치는 768차원(16×16×3)이며, 두 개의 순차적 선형 계층으로 임베딩되되 중간에 병목 차원 $d'$($d'<768$)을 둔다. 병목 임베딩은 일반적으로 유익하며, x-prediction 모델은 32나 16처럼 공격적인 병목에서도 준수하게 작동한다.

**해상도 확장 전략**: 해상도에 비례해 패치 크기를 증가시킴으로써(256²는 JiT/16, 512²는 JiT/32, 1024²는 JiT/64) 일정한 시퀀스 길이와 연산 비용을 유지하며 강한 확장성을 보인다.

### (4) 성능 향상 (저자 보고 수치)

| 조건 | 결과 | 출처 |
|---|---|---|
| ImageNet 256×256, JiT-B/16, 768차원 패치 | 오직 x-prediction만 잘 작동하며, 3가지 손실 모두에서 성공한다. 패치는 768차원으로 JiT-B의 은닉 크기 768과 동일하다 | Table 2(a) |
| ImageNet 64×64, JiT-B/4 (48차원 패치, 저차원) | 저차원 설정에서는 예측 대상 간 성능 격차가 미미하다 | Table 2(b) |
| 병목 차원 영향 | JiT-B/16(패치 768차원)을 이용해 FID 대 병목 차원 $d'$을 도식화했으며, 병목 차원을 16까지 줄여도 파국적 실패가 없고, 오히려 32~512 범위에서 품질이 개선된다 | Fig. 4 |
| ImageNet 256×256 최종 성능 | JiT-G/16: FID 1.82, IS 292.6, Precision 0.79, Recall 0.62 (DiT-XL/2 FID 2.27, SiT-XL/2 FID 2.06 대비 우수) | Table 13 |
| ImageNet 512×512 | JiT-G의 512 해상도 FID(1.78)가 오히려 256 해상도(1.82)보다 낮다 | Table 6 |
| 해상도 간 전이(cross-resolution) | 512 모델 이미지를 256으로 다운샘플링하면 FID@256=1.84로, 256 전문 모델(1.82)과 경쟁력 있는 반면, 256 모델 이미지를 512로 업샘플링하면 FID@512=2.45로 512 전문 모델(1.78)보다 눈에 띄게 나쁘다 | Table 12 |

### (5) 한계

- JiT-B/16은 Recall이 0.50으로 다른 모델(0.57~0.67)보다 낮아, 소형 모델에서는 다양성(diversity) 손실 가능성이 있다(Table 13).
- 거대 모델의 성능은 상당 부분 과적합(overfitting) 여부에 좌우되며, 512 해상도 디노이징이 더 어려운 과제이기 때문에 과적합에 덜 취약하다는 설명은 사후적(post-hoc) 해석에 가깝다.
- 병목이 왜 도움이 되는지에 대한 이론적 증명은 제시되지 않고 경험적 관찰(Fig. 4)에 그친다.

---

## 3. 주장별 페이지/Figure/Table 표시

| 주장 | 위치 |
|---|---|
| 매니폴드 가정 도식 | Fig. 1 |
| 토이 실험(저차원 데이터 매립) | Fig. 2, Sec. 3.3 |
| 9가지 조합 표 | Table 1 |
| JiT-B/16, JiT-B/4 FID 비교 | Table 2(a),(b) |
| JiT 아키텍처 도식 | Fig. 3 |
| 병목 차원 vs FID | Fig. 4 |
| 모델 크기별 확장성(Table 6) | Table 6, Sec. 4 |
| ImageNet 256/512 uncurated 샘플 | Fig. 5, Fig. 8~11 |
| 해상도 간 전이 실험 | Table 12 |
| Precision/Recall 비교 | Table 13, Appendix B.5 |
| FLOPs/파라미터 비교 | Table 8 |

---

## 4. 저자 보고 결과 vs. 해석 분리

| 구분 | 내용 |
|---|---|
| **저자 직접 보고 (사실)** | "JiT-B/16에서는 오직 x-prediction만 잘 작동하며, 세 가지 손실 모두에서 성공한다"; JiT-G/16 FID 1.82; "병목 차원을 16까지 줄여도 파국적 실패가 발생하지 않는다" |
| **본 요약자의 해석** | 이 결과들은 "네트워크 폭이 데이터 차원과 비례해야 한다"는 통념에 반하는 실증 사례로 읽을 수 있으며, 이는 향후 트랜스포머 기반 생성모델 설계 시 임베딩 차원 축소를 적극 고려할 근거가 될 수 있다는 점에서 실무적 함의가 크다. 다만 이 결론이 이미지 도메인, 특히 자연 사진(ImageNet)에 국한된 것인지, 다른 모달리티(음성·비디오·3D)에도 일반화되는지는 저자들이 직접 검증하지 않았다. |

---

## 5. 통계적으로 취약한 부분 / 비교 불가능한 수치

- **에폭(epoch) 수 불일치**: Table 2 소규모 실험은 200 epoch, CFG 적용 조건에서 수행되었다는 점을 명시하지만, Table 13의 SOTA 비교(JiT-G/16 vs DiT-XL/2, SiT-XL/2, RAE)는 각 방법의 훈련 epoch·데이터 파이프라인·CFG 튜닝 강도가 동일하다는 보장이 없어 엄밀한 통제 비교(controlled comparison)라 보기 어렵다.
- **CFG 값 차이**: JiT-G/16의 보고된 FID 1.82는 CFG 값 2.2에서 얻어진 것인데, 비교 대상 모델들의 CFG 값이 동일하게 최적화되었는지 불명확해 직접적 수치 비교의 공정성에 한계가 있다.
- **작은 표본의 토이 실험**: 단일 무작위 투영 행렬 $P$와 5-layer MLP 하나로 얻은 결과이며, 반복 실험에 따른 분산이나 신뢰구간이 보고되지 않는다.
- **Recall 지표의 모델 간 편차**: JiT-B/16의 Recall(0.50)은 RAE(0.67)와 큰 차이를 보이는데, 이는 모델 크기 차이에 기인한 것인지 x-prediction 고유의 특성인지 분리되지 않는다.
- **해상도 전이 비교(Table 12)**: 다운/업샘플링을 통한 FID 비교는 표준 평가 프로토콜과 다른 후처리(resizing) 단계를 포함하므로, 순수한 "전문가 모델 대 전이 모델" FID로 단순 비교하기엔 주의가 필요하다.

---

## 6. 문서가 답하지 않는 질문

1. 병목 임베딩이 성능을 높이는 **이론적 메커니즘**은 무엇인가? (경험적 관찰만 제시됨)
2. ImageNet 이외의 데이터셋(예: 텍스트-이미지, 비디오, 오디오)에서도 x-prediction 우위가 유지되는가?
3. JiT의 **추론 속도/샘플링 지연시간**은 잠재공간(latent) 확산 모델 대비 어떠한가?
4. 매니폴드 차원 $d$를 실제 데이터에서 사전에 알 수 없을 때, 병목 차원 $d'$을 어떻게 선택해야 하는지에 대한 원칙은?
5. JiT-G보다 더 큰 스케일(초거대 모델)에서도 x-prediction의 이점이 유지되는가, 아니면 스케일이 커지면 격차가 좁혀지는가?
6. 텍스트 조건부(text-to-image) 생성과 같은 조건화가 복잡한 태스크에서도 결론이 유지되는가?

---

## 7. 가장 중요한 그림 5개 해석

1. **Figure 1 (매니폴드 가정 도식)**: 자연 이미지가 고차원 픽셀 공간 내 저차원 매니폴드 위에 존재한다는 가정을 시각화하며, 깨끗한 이미지는 온-매니폴드이지만 노이즈나 플로우 속도는 오프-매니폴드임을 보여, x-prediction과 ε/v-prediction 학습이 근본적으로 다름을 강조한다. 논문 전체 논증의 개념적 출발점.

2. **Figure 2 (토이 실험)**: 2D 패턴을 512차원 상자에 숨기는 실험에서, 공간이 커질수록 오직 깨끗한 이미지 예측기만 패턴을 복원하고 노이즈·혼합 예측은 실패한다는 결과를 시각적으로 증명해, 매니폴드 가정을 정성적으로 뒷받침하는 핵심 근거다.

3. **Figure 3 (JiT 아키텍처)**: 플레인 ViT를 픽셀 패치에 적용해 x-prediction을 수행하는 구조를 보여주는 것으로, 이 논문 제안 방법의 단순성(토크나이저·사전학습 불필요)을 시각적으로 요약한다.

4. **Figure 4 (병목 차원 vs FID)**: JiT-B/16 기준 병목 차원 $d'$에 따른 FID 변화를 나타내며, 16차원까지 줄여도 실패하지 않고 32~512 범위에서 오히려 품질이 개선됨을 보여, "고차원 입력엔 넓은 네트워크가 필요하다"는 통념을 반박하는 가장 반직관적인 결과다.

5. **Figure 5 (JiT-G/16 생성 샘플)**: JiT-G/16 모델의 생성 샘플을 보여주며, ImageNet 클래스 전반에 걸쳐 고품질·다양한 이미지 합성이 가능함을 정성적으로 입증한다.

---

## 8. 결론: 시사점 및 후속 연구

저자들은 네트워크가 매니폴드의 기본으로 돌아가도록 함으로써, 원시 자연 데이터에 대한 Transformer 기반 확산의 자기완결적(self-contained) 패러다임을 추구한다는 연구 방향을 제시한다. 이는 VAE 토크나이저 의존을 없애고, 노이즈 예측이라는 관행적 선택을 재검토하도록 촉구하는 시사점을 갖는다.

### 8-1. 모델의 일반화 성능 향상 가능성

- **아키텍처-태스크 분리를 통한 확장성**: Transformer 설계를 태스크와 분리함으로써 확장성(scalability) 잠재력을 활용하는 것이 핵심 목표이며, 실제로 이 접근법은 스케일링의 이점을 누린다(Table 6). 이는 모델 크기가 커질수록 x-prediction 기반 JiT가 다양한 해상도·데이터 분포에 더 잘 일반화될 가능성을 시사한다.
- **해상도 간 전이 가능성**: 다운샘플링을 통한 해상도 전이가 전문가 모델과 경�쟁력 있는 FID를 보인다는 점은, 단일 모델이 여러 해상도에 걸쳐 일반화될 잠재력을 시사하지만, 역방향(저해상도→고해상도 업샘플링)은 성능 저하가 뚜렷해 완전한 해상도 불변성(invariance)에는 한계가 있다.
- **병목 구조의 일반화 함의**: 병목이 성능을 개선한다는 발견은, 데이터의 내재적 저차원 구조를 강제로 학습하도록 유도하는 일종의 정규화(regularization) 효과로 해석될 수 있어, 향후 다른 도메인(비디오, 3D)에서도 유사한 압축 기반 일반화 전략이 유효할 수 있다는 가설로 이어질 수 있다.

### 8-2. 2020년 이후 관련 최신 연구 비교 분석 및 향후 연구 시 고려사항

- **역사적 맥락**: 2020년 DDPM(Denoising Diffusion Probabilistic Models)이 노이즈 예측 방식을 대중화한 이후, flow matching(2023)과 rectified flow(2023) 계열이 v-prediction을 도입하며 발전해왔다. 이 논문은 이러한 5년여 흐름 속에서 "예측 대상" 선택 자체를 재검토한다는 점에서 계보적으로 중요한 위치에 있다.
- **후속 연구에서의 채택**: 이미 여러 2025~2026년 연구가 JiT를 기반/비교 대상으로 채택하고 있다. 예를 들어 한 적대적 플로우 모델 연구는 공식 사전학습된 JiT-H/16을 생성기(G)의 출발점으로 사용하며 x-prediction 결과를 v로 변환해 판별기에 전달했고, 아웃라이어 토큰 연구는 속도 손실을 사용한 데이터 예측 방식이 JiT에서 효과적임이 입증되었다고 언급한다. 또한 마스크드 비트 모델링(BAR) 연구는 JiT를 최신 SOTA 비교군 중 하나로 포함시키고 있다.
- **RAE(Representation Autoencoders)와의 비교**: Table 13에서 RAE 기반 방법(FID 1.13)이 JiT-G/16(FID 1.82)보다 우수한 FID를 기록하는데, 이는 사전학습된 표현(pre-trained representation)을 활용하는 최신 흐름이 "토크나이저 없는 순수 픽셀 접근"보다 여전히 강력할 수 있음을 시사하며, 향후 연구는 x-prediction의 이점과 강력한 표현 학습을 결합하는 하이브리드 방향을 고려할 필요가 있다.
- **향후 연구 시 고려할 점**:
  1. CFG 값, 훈련 epoch 등 하이퍼파라미터를 통제한 공정한 벤치마크 프로토콜 확립
  2. 병목 설계의 이론적 근거(정보이론적 관점 등) 규명
  3. 이미지 이외 모달리티(비디오, 3D, 오디오)로의 확장 검증
  4. RAE류의 표현 학습 기반 방법과의 결합 가능성 탐구
  5. 초거대 스케일에서 x-prediction 우위가 유지되는지에 대한 스케일링 법칙(scaling law) 연구

---

## 참고 문헌 / 출처 목록

1. Tianhong Li, Kaiming He, "Back to Basics: Let Denoising Generative Models Denoise", arXiv:2511.13720 (CVPR 2026) — https://arxiv.org/abs/2511.13720, https://arxiv.org/pdf/2511.13720, https://arxiv.org/html/2511.13720v1
2. CVPR 2026 Open Access PDF — https://openaccess.thecvf.com/content/CVPR2026/papers/Li_Back_to_Basics_Let_Denoising_Generative_Models_Denoise_CVPR_2026_paper.pdf
3. "[Literature Review] Back to Basics: Let Denoising Generative Models Denoise", Moonlight — https://www.themoonlight.io/en/review/back-to-basics-let-denoising-generative-models-denoise
4. alphaXiv, "Back to Basics: Let Denoising Generative Models Denoise" — https://www.alphaxiv.org/abs/2511.13720, https://www.alphaxiv.org/resources/2511.13720
5. ResearchGate PDF — https://www.researchgate.net/publication/397713136_Back_to_Basics_Let_Denoising_Generative_Models_Denoise
6. Hugging Face Papers page — https://huggingface.co/papers/2511.13720
7. Emergent Mind, "x-Prediction in Diffusion Models" — https://www.emergentmind.com/papers/2511.13720
8. Medium, "We've Been Overcomplicating Diffusion: Why 'Just Image Transformers' Are All You Need" — https://medium.com/@tarunbevara10/weve-been-overcomplicating-diffusion-why-just-image-transformers-are-all-you-need-9254b70dd6a6
9. Paper Notes, "[Paper Note] Back to Basics: Let Denoising Generative Models Denoise" (CVPR2026) — https://en.papernotes.org/CVPR2026/image_generation/back_to_basics_let_denoising_generative_models_denoise/
10. GitHub AkihikoWatanabe Paper Notes Issue #3950 — https://github.com/AkihikoWatanabe/paper_notes/issues/3950
11. "Continuous Adversarial Flow Models", arXiv:2604.11521 — https://arxiv.org/pdf/2604.11521
12. "On Variance Reduction in Learning Mean Flows", arXiv:2605.09235 — https://arxiv.org/pdf/2605.09235
13. "Autoregressive Image Generation with Masked Bit Modeling", arXiv:2602.09024 — https://arxiv.org/pdf/2602.09024
14. "Taming Outlier Tokens in Diffusion Transformers", arXiv:2605.05206 — https://arxiv.org/pdf/2605.05206
