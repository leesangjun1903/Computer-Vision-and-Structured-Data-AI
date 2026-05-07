# BlockDance: Reuse Structurally Similar Spatio-Temporal Features to Accelerate Diffusion Transformers

## 1. 핵심 주장 및 주요 기여 요약

**BlockDance**(Zhang et al., 2025, arXiv:2503.15927)는 Diffusion Transformer(DiT)의 추론 속도를 가속하기 위한 **학습 불필요(training-free)** 접근으로, 인접 시간 단계(adjacent time steps) 사이의 **구조적으로 가장 유사한 시공간 특징(Structurally Similar Spatio-Temporal, STSS)** 을 식별하여 캐싱·재사용한다.

핵심 기여는 다음 세 가지로 요약된다.

첫째, DiT 내부에서 블록 깊이별·시간단계별 특징 유사도가 어떻게 분포하는지를 정량적으로 분석하여, **얕은/중간 블록(0~20)** 이 구조적 정보를 담당하며 **노이즈 제거 후반부**에서 인접 단계간 변화가 매우 작다는 점을 밝혔다. 둘째, 이 관찰을 바탕으로 **BlockDance**라는 plug-and-play 가속 알고리즘을 제안하여 DiT-XL/2(37.4%), PixArt-α(25.4%), Open-Sora(34.8%)에서 25~50% 가속을 달성하면서도 원본과의 일관성(SSIM)을 유지했다. 셋째, 콘텐츠의 구조적 복잡도에 따라 재사용 빈도가 달라져야 한다는 통찰을 반영해 **강화학습 기반 의사결정 네트워크 BlockDance-Ada**를 제안, 인스턴스별로 캐시/재사용 정책을 동적으로 할당하여 동일 가속비에서 더 높은 품질을 얻었다.

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

DiT는 U-Net 기반 확산모델보다 시각적 품질, scaling law 적합성, 멀티모달 통합 측면에서 우수하지만, **반복적 노이즈 제거 과정** 때문에 추론 속도가 느려 실시간 응용이 어렵다. 기존 가속 패러다임은 두 갈래로 나뉜다.

- **샘플링 단계 수 감소**: 빠른 샘플러(DDIM, DPM-Solver) 또는 단계 증류(step distillation)
- **단계당 연산량 감소**: 모델 가지치기·증류·중복 계산 제거(DeepCache, TGATE 등)

저자들은 후자에 주목하되, **DiT 특화 중복성 분석이 부재**하다는 한계를 지적한다. 기존 feature-reuse 기법(DeepCache, FORA 등)은 "낮은 유사도 특징까지 무차별적으로 재사용"하여 구조 왜곡과 prompt-image 정렬 붕괴를 초래한다. 따라서 본 논문은 **"DiT 내에서 어떤 특징이, 어느 시간 구간에서 진정으로 재사용 가능한가?"** 를 정량적으로 규명하는 것이 핵심 연구 문제이다.

### 2.2 핵심 관찰

PCA 시각화와 인접 단계 간 코사인 유사도 분석을 통해 두 가지 사실이 도출된다.

1. 노이즈 제거 초반에는 거친 구조(인간의 자세, 객체의 형태)가 생성되고, 후반에는 텍스처와 디테일이 채워진다.
2. **얕은/중간 블록**(PixArt-α 기준 0~20번 블록)은 후반부에서 인접 단계 간 변화가 거의 없는 반면, **깊은 블록**(21~27)은 텍스처 변화가 활발하다.

따라서 "구조가 안정화된 이후 → 얕은/중간 블록의 출력"이 곧 STSS 특징이며, 이 영역에 자원을 재투입하는 것은 비용 대비 효익이 거의 없다는 결론에 이른다.

### 2.3 제안하는 방법 (수식 포함)

#### (1) Diffusion 기본 정식화

순방향 노이즈 추가 과정의 사후확률은 다음과 같다.

$$q(z_t \mid z_0) = \mathcal{N}\!\left(z_t;\; \sqrt{\bar{\alpha}_t}\, z_0,\; (1-\bar{\alpha}_t)\,\mathbf{I}\right)$$

여기서 $\bar{\alpha}\_t = \prod_{i=0}^{t}\alpha_i = \prod_{i=0}^{t}(1-\beta_i)$이고 $\beta_i \in (0,1)$은 noise schedule이다. 학습된 $\epsilon_\theta(z_t, t)$로 DDIM 역방향 샘플링을 수행한다.

$$z_{t-1} = \sqrt{\alpha_{t-1}}\!\left(\frac{z_t - \sqrt{1-\alpha_t}\,\epsilon_\theta(z_t, t)}{\sqrt{\alpha_t}}\right) + \sqrt{1-\alpha_{t-1}-\sigma_t^2}\cdot\epsilon_\theta(z_t, t) + \sigma_t \epsilon_t$$

#### (2) BlockDance: 학습 불필요 가속

각 단계는 **cache step**과 **reuse step**으로 나뉜다. cache step에서는 표준 forward를 수행하고 $i$번째 블록의 출력 $F_t^{i}$를 저장한다. 이어지는 reuse step에서는 다음과 같이 처리한다.

$$F_{t-1}^{i} \;\leftarrow\; F_{t}^{i} \quad (\text{즉, 첫 } i \text{개 블록 계산을 건너뜀})$$

이후 $(i+1)$번째 블록부터 끝까지만 재계산한다. 논문의 실험 설정은 $i = 20$, 재사용 구간은 전체 노이즈 제거의 후반 60%(PixArt-α 기준 40%~95%)이다. 후반 60%를 $N$ 단계 그룹으로 균등 분할하여 첫 단계는 cache, 나머지 $N-1$ 단계는 reuse로 사용한다. 이 전략을 **BlockDance-N** 으로 명명한다.

#### (3) BlockDance-Ada: 인스턴스 적응형 가속

단순/복잡 컨텐츠마다 재사용 가능한 유사 특징의 분포가 달라지는 점을 고려해, $s$개 전체 단계 중 초반 $\rho$ 단계를 cache로 고정한 뒤, 남은 $s-\rho$ 단계의 cache/reuse 결정을 학습한다. 결정 네트워크 $f_d$ (파라미터 $w$)는 중간 잠재변수 $z_\rho$와 텍스트 임베딩 $c$를 입력으로 받는다.

$$\mathbf{m} = \mathrm{sigmoid}(f_d(z_\rho, c;\, w)) \in [0,1]^{s-\rho}$$

이를 베르누이 분포의 정책으로 정의한다.

$$\pi^f(\mathbf{u} \mid z_\rho, c) = \prod_{t=1}^{s-\rho} m_t^{u_t}\,(1-m_t)^{1-u_t}, \quad \mathbf{u}\in\{0,1\}^{s-\rho}$$

여기서 $u_t = 1$은 cache, $u_t = 0$은 reuse이다. 보상은 품질 보상 $Q(\mathbf{u}) = f_q(x)$(품질 보상모델 활용)와 연산 보상으로 구성된다.

$$C(\mathbf{u}) = 1 - \frac{\sum_{t=1}^{s-\rho} u_t}{s-\rho}$$

$$R(\mathbf{u}) = C(\mathbf{u}) + \lambda\, Q(\mathbf{u})$$

논문에서는 $\lambda = 2$를 사용한다. 의사결정이 비미분 이산 결정이므로 정책 경사(policy gradient)로 학습한다.

$$\nabla_w \mathcal{L} = \mathbb{E}\!\left[ R(\mathbf{u}) \,\nabla_w \log \pi^f(\mathbf{u}\mid z_\rho, c) \right]$$

미니배치 크기 $B$에 대해 다음과 같이 근사한다.

$$\nabla_w \mathcal{L} \approx \frac{1}{B}\sum_{i=1}^{B} R(\mathbf{u}^i)\,\nabla_w \log \pi^f(\mathbf{u}^i \mid z_\rho^i, c^i)$$

### 2.4 모델 구조

- **백본 DiT**는 그대로 사용한다(추가 학습 불필요).
- **BlockDance**는 추론 스케줄러에 캐시 큐 하나를 덧붙이는 형태의 wrapper이다.
- **BlockDance-Ada의 결정 네트워크**는 3개의 transformer 블록 + MLP로 구성된 약 0.08B 파라미터의 경량 모듈이다. PixArt-α 학습 데이터 10K 부분집합에서 100 epoch, 배치 16, Adam(lr= $10^{-5}$ )으로 학습된다.

### 2.5 성능 향상

ImageNet/COCO/MSR-VTT 벤치마크에서 ToMe, DeepCache, TGATE, PixArt-LCM 등을 비교 기준으로 삼은 주요 결과는 다음과 같다.

- **DiT-XL/2 (ImageNet)**: BlockDance(N=2)에서 37.4% 가속, FID 15.70, SSIM 0.98로 DeepCache보다 baseline 일관성이 월등히 높음.
- **PixArt-α (COCO)**: BlockDance(N=2)에서 25.4% 가속, SSIM 0.89로 DeepCache의 0.60을 큰 폭으로 상회.
- **Open-Sora (MSR-VTT)**: BlockDance(N=2)에서 34.8% 가속, FVD 550으로 DeepCache(942)보다 영상 품질이 훨씬 우수.
- **BlockDance-Ada**: PixArt-α에서 30.6% 가속하면서도 BlockDance(N=2)와 거의 동일한 품질 지표 달성, 동일 가속비에서 BlockDance-N을 능가.

ablation에서는 (a) **재사용 구간**(후반 40~95% 적용 시 품질 손실 최소), (b) **블록 깊이**(얕은/중간 블록 재사용 시 품질 유지), (c) **재사용 빈도 N**(N 증가에 따라 속도-품질 trade-off 발생)이 모두 정성·정량적으로 검증되었다.

### 2.6 한계

- 논문이 명시한 핵심 한계는 **매우 짧은 추론(예: 1~4 step)** 환경에서는 인접 단계 간 유사도가 작아 BlockDance의 효익이 줄어든다는 점이다(LCM·Turbo 류 모델과의 결합이 어려움).
- 블록 인덱스 $i=20$, 재사용 구간 40~95% 등의 하이퍼파라미터가 PixArt-α에 맞춰 사전 분석된 값으로, **다른 DiT 변종**에서는 재탐색이 필요하다(DiT-XL/2와 Open-Sora는 25~95%로 다르게 설정됨).
- BlockDance-Ada는 보상 모델 $f_q$의 품질에 종속되며, 보상 해킹(reward hacking) 가능성에 대한 분석은 제시되지 않았다.
- 정량 결과가 주로 PixArt-α/DiT-XL/2/Open-Sora에 한정되어, FLUX·SD3·Hunyuan-DiT 같은 최신 대규모 DiT에서의 효과는 추가 검증이 필요하다.

---

## 3. 모델의 일반화 성능 향상 가능성 (중점 분석)

논문에서 "일반화"라는 용어를 명시적으로 학습 일반화 의미로 사용하지는 않았으나, **방법론의 적용 범위·전이성**의 관점에서 일반화 가능성은 다음과 같이 분석할 수 있다.

**(가) 아키텍처 일반화.** BlockDance는 DiT의 "블록을 거듭 통과하는 순차적 inference" 구조에 의존할 뿐, cross-attention의 존재 여부에 무관하다. 논문은 TGATE가 SD3나 FLUX처럼 cross-attention이 없는 모델에 적용 불가하다는 점을 짚으며, 본 방법이 더 넓은 DiT 변종에 적용 가능함을 강조한다. 즉 PixArt-α, DiT-XL/2, Open-Sora 외에도 단일 스트림 transformer(MMDiT, FLUX, Hunyuan-DiT, Latte 등)로의 확장이 자연스럽다.

**(나) 태스크 일반화.** 클래스 조건 이미지 생성(ImageNet), 텍스트-이미지(COCO), 텍스트-비디오(MSR-VTT) 세 가지 태스크에서 일관된 이득을 보여, "구조 안정화 후 얕은 블록의 정체"라는 현상이 **태스크-불변(task-invariant) 속성**임을 시사한다. 이는 멀티모달 합성, 이미지 편집, controllable generation 등으로의 전이 가능성을 뒷받침한다.

**(다) 콘텐츠 다양성에 대한 적응적 일반화.** Figure 6에서 보였듯, 단순한 장면일수록 유사 특징이 많고 복잡한 장면일수록 적다. BlockDance-Ada는 강화학습으로 이 분포를 **"인스턴스 단위로 학습"** 하기 때문에 학습된 정책이 unseen prompt에 대해서도 적응적으로 cache 빈도를 조절할 수 있다. 다만 학습 데이터가 PixArt-α의 10K subset에 한정되었고 도메인 시프트(예: 의료 영상, 위성 사진) 시 결정 네트워크가 동일 효과를 낼지는 실증되지 않았다.

**(라) 단계 수에 대한 일반화의 한계.** 본 논문이 명시한 한계와 직결되는 부분이다. **Few-step distilled model**(LCM, SDXL-Lightning, Turbo)에서는 인접 단계간 유사성이 약화되어, BlockDance가 학습한 "구조-텍스처 분리" 가설이 깨진다. 이는 일반화의 분명한 경계로, BlockDance-Ada가 1-4 step에서도 의미 있는 정책을 학습할 수 있는지는 후속 연구의 과제이다.

**(마) 일반화 강화를 위한 가능한 방향.** ① 결정 네트워크 입력에 블록별 통계량을 추가 주입해 "어디서 캐싱이 안전한가"를 더 세밀히 추정, ② Cross-model 전이 학습(한 DiT에서 학습한 정책을 다른 DiT로 fine-tune 없이 옮기는 zero-shot 정책 전이), ③ 구조 안정화 시점 자체를 모델이 자가 진단하도록 만드는 메타-학습 프레임워크 등이 제안 가능하다. 최근 DiCache(arXiv:2508.17356)는 "online probe profiling"을 통해 모델이 스스로 캐싱 시점을 결정하는 방향을 보였는데, 이러한 자가 결정 메커니즘과 BlockDance-Ada의 RL 정책이 결합되면 더 강한 일반화를 기대할 수 있다.

---

## 4. 향후 연구에의 영향 및 고려사항

**(가) DiT 가속의 새로운 분석 축 정립.** 본 논문은 "DiT 내부 블록의 시간적 유사도 분포"를 명시적으로 분리해 분석함으로써, 단순 reuse(FORA)나 잔차 reuse(Δ-DiT) 수준에서 더 나아간 **"블록-구간(block-stage) 양차원 분석"** 패러다임을 제시한다. 이후 ProCache(arXiv:2512.17298) 등에서도 깊은 블록의 오류 누적이 더 크다는 점을 보고하는 등, BlockDance의 분석 축이 후속 연구에 채택되고 있다.

**(나) 학습 가능한 캐시 정책의 표준화 가능성.** Learning-to-Cache(L2C)와 BlockDance-Ada는 모두 "캐시 결정의 학습"을 시도하지만, BlockDance-Ada는 RL을 통해 인스턴스별 정책을 학습한 거의 최초의 사례이다. 이 흐름은 SpeCa(speculative caching)·DiCache 같은 후속 연구의 적응형 캐싱 트렌드와 직접적으로 연결된다.

**(다) 향후 연구에서 고려할 점.**

1. **Few-step 모델과의 호환.** Distillation 후의 4-step·1-step 모델에서 유사도 가설이 깨지므로, 단계가 적을 때 적용 가능한 가속 메커니즘(예: 토큰 단위 캐시 ToCa, 어텐션 압축 DiTFastAttn)과의 결합이 필요하다.
2. **양자화/Pruning과의 직교성.** 본 논문은 가속의 한 축(중복 계산 제거)만 다룬다. INT8/FP8 양자화나 sparse attention과 stack 시 성능 손실이 부가적이지 않은지 검증해야 한다.
3. **결정 네트워크의 일반화·전이성.** 새 prompt 도메인이나 새 백본에서 정책 재학습 비용이 가속 이득을 상쇄하지 않도록, **few-shot 전이** 또는 **prompt-conditional cache scheduling**을 고려해야 한다.
4. **품질 평가의 신뢰성.** SSIM/CLIP/IR/Pickscore가 인지적 일관성을 충분히 반영하지 못하는 경우가 있어, 사람 평가 또는 시계열·구조 일관성에 특화된 지표(VBench의 temporal consistency 등)와 함께 보고해야 한다.
5. **에너지·메모리 트레이드오프.** 캐시 메모리(PixArt-α 18MB, Open-Sora 72MB)가 onset latency나 모바일·엣지 환경에서 실질적 비용이 될 수 있으므로, 메모리 제약 하의 성능 보고가 필요하다.

---

## 5. 2020년 이후 관련 최신 연구와의 비교 분석

| 연도 | 방법 | 대상 아키텍처 | 핵심 아이디어 | 학습 필요 | BlockDance와의 차이 |
|------|------|--------------|-------------|---------|------------------|
| 2023 | DeepCache (Ma et al., CVPR'24) | U-Net | skip connection을 활용해 high-level feature 캐싱·재사용 | No | U-Net 의존; DiT엔 부적합. 유사도 무차별 reuse |
| 2023 | Block Caching (Wimbauer et al.) | U-Net | 블록 출력 재사용 | No | DiT 부적합 |
| 2024 | TGATE (Zhang et al.) | U-Net + DiT | cross-attention 출력이 수렴함을 이용해 캐싱 | No | cross-attn 없는 DiT(SD3·FLUX)에 적용 불가 |
| 2024 | Δ-DiT (Chen et al.) | DiT | feature 자체가 아닌 **편차(delta)** 캐싱; 초기엔 후방 블록, 후기엔 전방 블록 가속 | No | BlockDance와 가장 유사한 통찰. 다만 정적 스케줄, RL 적응 정책 부재 |
| 2024 | FORA (Selvaraju et al.) | DiT | 일정 간격으로 모든 layer 캐싱 | No | 정적·전역적 → 품질 손실 큼 |
| 2024 | L2C (Learning-to-Cache) | DiT | layer별 skip 라우터 학습 | Yes | 추론 스텝 수가 바뀌면 재학습 필요; BlockDance-Ada는 prompt 의존 정책 |
| 2024 | ToCa (Token Caching) | DiT | **토큰 단위** 점수에 기반한 selective cache | No | block 단위가 아닌 fine-grained token 단위 |
| 2024 | PAB (Pyramid Attention Broadcast) | Video DiT | attention head별 update 빈도 조정 | No | 비디오 특화, 어텐션 단위 |
| 2024 | FasterCache | Video DiT | CFG·시간 캐시 결합 | No | BlockDance와 보완적 |
| **2025** | **BlockDance / BlockDance-Ada** | **DiT (image+video)** | **STSS feature 식별 + RL 적응형 cache 정책** | **No / Yes(Ada)** | **블록 깊이 × 시간 구간의 정밀 분석을 RL 정책과 결합** |
| 2025 | TaylorSeer | DiT | "cache-then-forecast"로 미래 feature 예측 | No | 예측 검증 부재; BlockDance는 retrospective reuse |
| 2025 | SpeCa | DiT | speculative caching + 검증 | No | 속도 향상 폭 더 크나 검증 비용 발생 |
| 2025 | DiCache | DiT | 모델이 자체적으로 캐싱 시점 결정 | No | online profiling으로 동적 결정; BlockDance-Ada의 자가 결정 버전 |
| 2025+ | ProCache | DiT | 깊은 블록의 오차 누적을 보정하는 selective recompute | No | BlockDance의 분석을 정량 오차 관점에서 확장 |

종합하면, BlockDance는 **(i) Δ-DiT처럼 DiT의 블록 깊이 의존 특성을 활용하면서**, **(ii) FORA의 단순 정적 스케줄을 RL 기반 동적 스케줄로 대체**하고, **(iii) DeepCache가 가진 U-Net 의존성을 제거**한 위치에 자리한다. 2025년 이후 연구는 BlockDance의 "block × stage" 분석을 더 세밀화(ProCache), 자가 결정화(DiCache), 예측·검증화(SpeCa, TaylorSeer)하는 방향으로 진화하고 있다.

---

## 참고자료 출처

본 답변은 다음 자료를 직접 검토·인용하였다.

1. **(주 논문)** Hui Zhang, Tingwei Gao, Jie Shao, Zuxuan Wu. "BlockDance: Reuse Structurally Similar Spatio-Temporal Features to Accelerate Diffusion Transformers." arXiv:2503.15927v1, 2025. (사용자 업로드 PDF 직접 분석)
2. Ma, Fang, Wang. "DeepCache: Accelerating Diffusion Models for Free." arXiv:2312.00858, CVPR 2024.
3. "Faster Diffusion via Temporal Attention Decomposition" (TGATE), arXiv:2404.02747, 2024.
4. Selvaraju, Ding, Chen, Zharkov, Liang. "FORA: Fast-Forward Caching in Diffusion Transformer Acceleration." arXiv:2407.01425, 2024.
5. Chen et al. "Δ-DiT: A Training-Free Acceleration Method Tailored for Diffusion Transformers." arXiv:2406.01125, 2024.
6. "Accelerating Diffusion Transformers with Token-wise Feature Caching" (ToCa), arXiv:2410.05317, 2024.
7. "DiCache: Let Diffusion Model Determine Its Own Cache", arXiv:2508.17356, 2025.
8. "ProCache: Constraint-Aware Feature Caching with Selective Computation for Diffusion Transformer Acceleration", arXiv:2512.17298, 2025+.
9. "SpeCa: Accelerating Diffusion Transformers with Speculative Feature Caching", arXiv:2509.11628, 2025.
10. "Token Caching for Diffusion Transformer Acceleration", arXiv:2409.18523.

**확신 수준에 관한 주의사항.** 본 논문에 직접 명시된 수식·수치(가속비, FID, SSIM 등)와 방법론은 PDF 원문을 그대로 인용하여 정확성을 보장하나, "일반화 가능성" 항목 중 후속 연구와의 융합 시나리오나 도메인 시프트 관련 추론은 본 논문에 직접 실험적 근거가 없는 **분석적 추론**임을 밝혀둔다. 또한 "2020년 이후 비교 표"의 일부 후속 연구(SpeCa, DiCache, ProCache, TaylorSeer)는 BlockDance 발표 이후 등장한 최신 작업으로, 본 논문이 직접 비교 대상으로 삼은 것은 ToMe, DeepCache, TGATE, PixArt-LCM에 한정된다는 점에 유의하시기 바란다.
