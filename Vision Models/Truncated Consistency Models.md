# Truncated Consistency Models
---

## 1. 핵심 주장과 주요 기여 (요약)

**핵심 주장.** 표준 Consistency Model(CM)은 PF ODE 궤적 위의 *모든* 시점 $t \in [0,T]$에서 데이터 끝점으로 매핑하는 것을 동시에 학습하므로, 작은 $t$의 *denoising* 과제와 큰 $t$의 *generation* 과제가 한 네트워크 용량을 다투게 된다. 저자들은 학습이 진행될수록 모델이 작은 $t$에서의 denoising 능력을 *희생*해 큰 $t$의 생성 품질을 끌어올린다는 trade-off를 실험적으로 관찰했고(Fig. 2), 이 암묵적 trade-off를 *명시적으로* 통제하면 1-step/2-step 생성 품질이 크게 좋아진다고 주장한다.

**주요 기여(논문 요약).**
- denoising vs. generation 사이의 학습 자원 배분 trade-off를 정량적으로 드러냄(`dFID_t` 정의 및 시간별 변화 추적).
- 학습 시간 영역을 $[t', T]$로 *잘라내고*, 자명해(상수해)로의 붕괴를 막는 새 파라미터화(Eq. 5)와 2단계 학습 절차를 제안.
- 동일·또는 더 작은 네트워크로 iCT, iCT-deep, ECM 대비 CIFAR-10·ImageNet 64×64에서 1-step/2-step FID 모두 개선. ImageNet에서 Stage 1이 발산하는 구간에서도 Stage 2(TCM)는 안정적으로 개선됨(Fig. 1b, Fig. 7).

---

## 2. 자세한 설명

### 2.1 해결하려는 문제

표준 CM 학습 목표는 인접 시점 $(t, t-\Delta_t)$의 출력 차이를 줄이는 것이다(Eq. 4):

$$
\mathcal{L}_{\mathrm{CT}}(f_\theta, f_{\theta^-}) := \mathbb{E}_{t \sim \psi_t,\, x \sim p_{\mathrm{data}},\, \epsilon \sim \mathcal{N}(0,I)} \left[ \frac{\omega(t)}{\Delta_t}\, d\!\left(f_\theta(x + t\epsilon, t),\, f_{\theta^-}(x + (t-\Delta_t)\epsilon, t-\Delta_t)\right) \right]
$$

논문은 학습이 진행될수록 작은 $t$ (예: $t=0.2$)의 $\mathrm{dFID}_t$가 *증가*하고, 큰 $t$의 $\mathrm{FID}$가 감소하는 현상을 보고한다(Fig. 2). 즉 모델이 한정된 용량을 자발적으로 생성 쪽으로 옮기지만, 그 과정이 통제되지 않아 **(i) 학습이 불안정**하고 **(ii) 생성 성능의 상한이 낮다.**

### 2.2 제안 방법

**(a) 잘라낸 시간 영역과 새 파라미터화.** 시간 분포 $\psi_t$의 지지대를 $[t', T]$로 제한하면 자명해 $f_\theta = \mathrm{const}$가 손실의 최소해가 되어 붕괴할 수 있다(예: $F_\theta(x,t) = -c_{\mathrm{skip}}(t)x/c_{\mathrm{out}}(t)$이면 $f_\theta = 0$). 이를 막기 위해 사전학습된 1단계 모델 $f_{\theta_0}$를 *경계조건*으로 사용하는 새 파라미터화를 도입한다(Eq. 5):

$$
f^{\mathrm{trunc}}_{\theta,\theta_0^-}(x,t) \;=\; f_\theta(x,t)\cdot \mathbf{1}\{t \ge t'\} \;+\; f_{\theta_0^-}(x,t)\cdot \mathbf{1}\{t < t'\}
$$

샘플링 시에는 $t \ge t'$만 사용하므로 $f_\theta$만 있으면 된다.

**(b) 시간 분할과 손실 분해.** $S_{t'} := \{t : t' \le t \le t' + \Delta_t\}$, $S_{t'}^{-} := (t' + \Delta_t,\, T]$로 나누고, 시간 분포를 디랙 델타와 연속 분포의 혼합으로 정의해 $\int_{S_{t'}}\psi_t(t)\,dt \ge \lambda_b > 0$을 *상시* 보장한다(Eq. 7):

$$
\psi_t(t) \;=\; \lambda_b\, \delta(t - t') \;+\; (1-\lambda_b)\, \bar{\psi}_t(t)
$$

이를 Eq. 6에 대입하고 $\Delta_t \to 0$ 극한을 취하면(부록 E의 Taylor 전개) 손실은 두 항으로 깔끔하게 갈라진다(Eq. 8–9):

$$
\mathcal{L}_{B}(f_\theta, f_{\theta_0^-}) \;\approx\; \frac{\omega(t')}{\Delta_{t'}}\, d\!\left(f_\theta(x+t'\epsilon, t'),\; f_{\theta_0^-}(x+(t'-\Delta_{t'})\epsilon, t'-\Delta_{t'})\right)
$$

$$
\mathcal{L}_{C}(f_\theta, f_{\theta^-}) \;=\; \mathbb{E}_{\bar{\psi}_t}\!\left[\frac{\omega(t)}{\Delta_t}\, d\!\left(f_\theta(x+t\epsilon, t),\; f_{\theta^-}(x+(t-\Delta_t)\epsilon, t-\Delta_t)\right)\right]
$$

최종 학습 목표(Eq. 10):

$$
\mathcal{L}_{\mathrm{TCM}} \;=\; w_b\, \mathcal{L}_{B}(f_\theta, f_{\theta_0^-}) \;+\; \mathcal{L}_{C}(f_\theta, f_{\theta^-}), \qquad w_b = \frac{\lambda_b}{1-\lambda_b}
$$

직관: $\mathcal{L}\_B$는 Stage 1 모델이 잘 학습한 $t = t'$ 지점의 매핑을 *계승*시키고, $\mathcal{L}\_C$는 그 경계조건을 $t = T$까지 *전파*한다. 결과적으로 $f_\theta(x_T, T) \approx x_0$가 강제된다.

**(c) 2단계 학습 절차(Algorithm 1).**
1) Stage 1: 표준 CM 목적함수로 $\theta_0$를 수렴까지 학습.  
2) Stage 2: $\theta \leftarrow \theta_0$로 초기화 후, 미니배치를 비율 $\rho \in (0,1)$로 분할해 $N_B = \lfloor B\rho \rfloor$개는 $\mathcal{L}_B$ 추정, 나머지는 $\mathcal{L}_C$ 추정. 논문 기본값은 $t' = 1$, $\rho = 0.25$, $w_b = 0.1$, $\bar{\psi}_t$는 log-Student- $t$ ($\sigma=0.2,\ \nu=0.01$).

### 2.3 모델 구조

CIFAR-10은 EDM의 DDPM++ (≈55.7M), ImageNet 64×64는 EDM2-S(280M) 및 EDM2-XL(≈1.1B)을 백본으로 쓰며, EDM 파라미터화

$$
f_\theta(x, t) = c_{\mathrm{out}}(t) F_\theta(x,t) + c_{\mathrm{skip}}(t) x, \quad c_{\mathrm{out}}(t) = \frac{t\sigma_{\mathrm{data}}}{\sqrt{\sigma_{\mathrm{data}}^2 + t^2}},\ c_{\mathrm{skip}}(t) = \frac{\sigma_{\mathrm{data}}^2}{\sigma_{\mathrm{data}}^2 + t^2}
$$

를 그대로 사용한다(Appendix F). 손실은 Pseudo-Huber $d(x,y) = \sqrt{\|x-y\|_2^2 + c^2} - c$ ($c$는 데이터셋별 상이).

### 2.4 성능 향상

논문 Table 1·2 기준 주요 결과(FID, 작을수록 좋음):

| 데이터셋 | 모델 | NFE | FID | # params (M) |
|---|---|---|---|---|
| CIFAR-10 | iCT | 1 | 2.83 | 56.4 |
| CIFAR-10 | iCT-deep | 1 | 2.51 | 112 |
| CIFAR-10 | **TCM (ours)** | 1 | **2.46** | 55.7 |
| CIFAR-10 | **TCM (ours)** | 2 | **2.05** | 55.7 |
| ImageNet 64×64 | iCT | 1 | 4.02 | 296 |
| ImageNet 64×64 | iCT-deep | 1 | 3.25 | 592 |
| ImageNet 64×64 | TCM (EDM2-S) | 1 | **2.88** | 280 |
| ImageNet 64×64 | TCM (EDM2-XL) | 1 | **2.20** | 1119 |

핵심 관찰: TCM의 **1-step FID가 iCT의 2-step FID와 동급**(예: CIFAR-10 1-step 2.46 vs. iCT 2-step 2.46), 그리고 **2-step TCM(2.05)이 35-step EDM(1.97)에 근접**한다. ImageNet에서 Stage 1은 150K iter 부근에서 발산하지만 Stage 2는 그 시점에서 시작해 2.83 → 2.46까지 안정적으로 개선되었다(Fig. 1b, Fig. 7).

### 2.5 한계

- 추가 학습 단계가 필요해 *iter당 시간 +18%, 메모리 +15%* (ImageNet 64×64, EDM2-S 기준; Appendix A).
- 파라미터화상 forward pass가 표준 CM의 2회에서 *3회*로 증가.
- 1-step 품질은 여전히 *수십~수백 step 디퓨전 모델*과 차이가 있다(저자들도 이 점을 명시).
- Hyper-parameter $t'$, $w_b$, $\rho$, $\bar{\psi}_t$ 형태(특히 log-Student- $t$의 $\sigma, \nu$) 등 튜닝 의존성이 있다(Sec. 4.4).
- 3단계 이상으로 일반화해도 추가 이득이 없었고, $[0, t')$에서 $f_{\theta_0}$를 fine-tune하면 오히려 FID가 나빠짐(Appendix C). 즉 *왜* 2단계가 ‘딱 맞는지*는 부분적으로만 설명되어 있다.

---

## 3. 모델의 일반화 성능 향상 가능성 (요청 사항 중점)

이 논문이 *일반화* 측면에서 시사하는 바는 두 갈래로 읽을 수 있습니다.

**(1) 실험적 일반화 신호.** 같은 알고리즘이 (i) CIFAR-10(unconditional), (ii) ImageNet 64×64(class-conditional, EDM2-S/XL), (iii) COYO 텍스트-이미지(SD 1.5 초기화, MSCOCO zero-shot FID 18.32 → 15.58, Table 5/Fig. 6) 세 가지 *서로 다른 도메인·해상도·조건*에서 일관되게 Stage 1 대비 FID를 낮춥니다. 이는 ‘denoising-generation 자원 충돌’이 데이터셋 특이 현상이 아니라 *consistency 학습 자체*에 내재된 경향임을 시사합니다.

**(2) 일반화 메커니즘 가설.** 저자들은 Fig. 7에서 Stage 1의 그래디언트 노름이 자주 spike(>100)를 일으키며 발산으로 이어지지만 Stage 2는 부드럽다는 점을 보고하며, “시간대별로 편향된 그래디언트 노름의 영향을 덜 받아” 안정적이라고 *가설*을 제시합니다. 이는 학습이 끝까지 수렴할 확률 자체가 높아진다는 뜻이고, 일반화 성능과 직결되는 *학습 안정성* 향상의 근거가 됩니다.

**(3) 한계와 주의점.** 다만 (i) 일반화 향상의 직접 측정(예: held-out distribution shift, OOD 평가)은 논문에 없고 FID/zero-shot FID 위주이며, (ii) text-to-image는 batch size 512의 “quick validation”이라고 저자가 명시(Appendix B)하고 있어 *대규모 학습에서의 일반화 행태*는 추가 검증이 필요합니다. (iii) $t'$의 ‘이상적 위치’는 Fig. 2의 dFID 거동을 보고 사람이 정한 휴리스틱($t' \in [0.8, 1.5]$)이라, 다른 노이즈 스케줄/도메인에서 *자동 결정*하는 절차가 일반화에 핵심이 될 것입니다.

요컨대 **‘용량을 생성 쪽으로 명시적으로 재배분’이라는 아이디어 자체는 도메인-불가지론적**이지만, 그 이득이 공식적으로 OOD 일반화로 옮겨가는지는 논문이 답하지 않은 열린 문제로 남겨둡니다.

---

## 4. 향후 연구에 미치는 영향과 고려할 점

**영향(이미 가시화된 것).**
- 시간축을 *조각*해 task-specialization을 거는 발상은 Multistep Consistency Models(Heek et al., 2024)·Phased Consistency(Wang et al., 2024)와 같은 “구간 분할 학습” 흐름과 결을 같이하며, TCM은 그 중에서도 *경계조건을 명시적으로 설계*해 1-step 성능을 끌어올린 사례로 자리매김합니다.Heek et al. (2024)는 PF ODE 궤적을 여러 구간으로 나누어 consistency training 목표를 단순화하는 multistep consistency models를 제안한 바 있어, 이 두 줄기는 향후 통합·비교 연구의 좋은 축이 될 수 있습니다.
- “teacher-as-boundary”는 *부분 시간대만* 사전학습 모델로 대체하는 형태의 stage-wise 학습 일반에 응용 가능합니다(예: 잠재 디퓨전, 비디오, 음성).

**향후 연구 시 고려할 점.**
1. **$t'$ 자동 결정.** 데이터/네트워크/노이즈 스케줄별로 dFID 거동을 모니터링해 transition 지점을 동적으로 잡는 방법(예: validation dFID 곡선의 변곡점 추적).
2. **3단계 이상에서 이득이 없는 이유의 이론화.** 저자들은 “2단계 이후에는 task가 모두 generation에 가까워서”라고 가설만 제시(Sec. 4.4). 이를 PF ODE의 곡률·sample complexity 관점에서 정형화할 필요.
3. **일반화·다양성 측정 강화.** FID 외에 precision/recall, OOD, mode coverage, classifier-free guidance robustness 등 다축 평가가 빠져 있어 후속 연구의 여지가 큼.
4. **메모리·시간 오버헤드 완화.** Stage 2에서 $f_{\theta_0}$를 메모리에 상주시키는 비용이 크며(특히 텍스트-이미지), distillation/quantization으로 경량화 여지.
5. **다른 가속 패러다임과의 결합.** Shortcut Models는 step-size $d$를 입력으로 받아 *단일 네트워크·단일 학습 단계*로 1-step을 달성하는 노선이고,shortcut models는 현재 노이즈 수준뿐 아니라 desired step size도 입력으로 받아 생성 과정에서 앞으로 건너뛸 수 있게 하며, 단일 네트워크·단일 학습 단계로 고품질 1-step·다-step 샘플을 생성합니다. TCM의 ‘경계조건+시간 truncation’과 shortcut의 ‘step-size 조건화’를 결합하면 학습 단순성과 품질을 동시에 잡을 가능성이 있습니다.
6. **flow-matching 계열로의 이식.** MeanFlow가 flow의 평균 속도를 학습해 1-step 성능을 크게 끌어올렸다는 보고가 있고,최근 도입된 MeanFlow 프레임워크는 보다 안정적인 학습과 더 나은 classifier-free guidance 통합을 가능하게 해, 처음부터 학습한 few-step과 multi-step 디퓨전 모델 사이의 격차를 크게 좁혔다 후속 분석에서 MeanFlow의 학습 목표는 trajectory flow matching과 trajectory consistency 두 성분으로 분해되며, 두 성분이 학습 중 강한 음의 상관을 보여 불안정성을 유발한다는 사실이 밝혀졌습니다. TCM의 ‘denoising vs. generation’ trade-off와 구조적으로 유사한 충돌이라, *경계 항을 명시적으로 두는 TCM식 분리*를 MeanFlow에 이식할 수 있는지는 매우 자연스러운 후속 질문입니다.

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

(연도/소속 정보는 인용한 외부 자료 기준이며, 검증 가능한 사실만 적었습니다. FID 수치는 비교 데이터셋·해상도·NFE·학습 데이터가 다르면 직접 비교가 불공정하므로 “접근 방식의 차이” 위주로 정리합니다.)

| 방법 (연도) | 학습 패러다임 | 목적 함수의 시간 영역 | 단계 수 | 핵심 차별점 vs. TCM |
|---|---|---|---|---|
| **DDPM** (Ho et al., 2020) | score/ε 학습 | $[0,T]$ 전 구간 | 수백~수천 | TCM의 *베이스라인*. 디퓨전 자체. |
| **EDM** (Karras et al., 2022) | preconditioning + 향상된 노이즈 스케줄 | $[0,T]$ | 35 내외 | TCM이 백본·파라미터화로 채택. |
| **Progressive Distillation** (Salimans & Ho, 2022) | 단계 수를 절반씩 줄이는 distillation | $[0,T]$ | 학습 단계마다 절반 | 다단계 학생 학습. TCM은 단계 ‘수’가 아닌 ‘시간대’를 자름. |
| **Consistency Models** (Song et al., 2023) | self-consistency, 경계조건 $f(x_0,0)=x_0$ | $[0,T]$ 전 구간 | 1–2 | TCM이 직접 개선 대상. |
| **iCT / iCT-deep** (Song & Dhariwal, 2023) | CT 학습 기법 개선(스케줄, 손실, 큰 모델) | $[0,T]$ | 1–2 | TCM이 동일·작은 네트워크로 능가(논문 Table 1·2). |
| **CTM** (Kim et al., 2023, ICLR 2024) | 임의 $(s,t)$ 사이 매핑 + GAN | $[0,T]$ 임의 구간 | 1–수 | CTM은 PF ODE 궤적의 임의 두 시점 사이를 traverse하도록 일반화하고, adversarial training과 denoising score matching을 결합해 CIFAR-10 FID 1.73, ImageNet 64×64 FID 1.92를 달성. TCM은 GAN을 쓰지 않고 시간대 자르기만으로 개선. |
| **Multistep CM** (Heek et al., 2024) | 궤적을 *세그먼트*로 나눠 각 구간 CM | 분할 구간 | 4/8/16 | 단일 단계 제약을 완화해 4·8·16 NFE로 표준 디퓨전과 격차 축소. TCM은 1–2 step 유지. |
| **ECM (Consistency Models Made Easy)** (Geng et al., 2024) | 사전학습 디퓨전을 CM으로 fine-tune | $[0,T]$ | 1–2 | 2024년 기준 CIFAR-10 SOTA CM 학습이 8 GPU로 1주가 걸린다는 비효율 문제를 지적하고, 디퓨전을 CM의 특수 케이스로 보고 사전학습 디퓨전에서 fine-tuning하는 효율적 학습 방식을 제안. TCM은 ECM 위에 *2단계 truncation*을 더한 것으로 볼 수 있음(논문 Sec. 4.1). |
| **SCott** (Liu et al., 2024) | SDE solver + adversarial loss 결합 distillation | $[0,T]$ | 2 | MSCOCO에서 2-step FID 22.1로 1-step InstaFlow(23.4)를 능가하고 4-step UFOGen에 필적, 고해상도 다양성에서 최대 16% 향상. TCM은 GAN/SDE solver 없이 simulation-free. |
| **Shortcut Models** (Frans et al., 2024) | step-size $d$를 조건으로 받는 단일 네트워크 | $[0,T]$ | 1·다 | 단일 네트워크·단일 학습 단계로 1-step과 다-step 모두 고품질을 달성, consistency models·reflow 대비 우수. TCM과 달리 2단계 학습 불필요. |
| **MeanFlow** (Geng et al., 2025) | 평균 속도(flow의 적분)를 직접 학습 | $[0,T]$ | 1 | MeanFlow는 처음부터 학습한 few-step 모델과 multi-step 디퓨전의 격차를 크게 좁혔다. 목적함수가 trajectory flow matching과 trajectory consistency 두 성분으로 분해되며, 두 성분이 학습 중 강한 음의 상관을 보여 최적화 충돌을 유발—TCM이 본 trade-off와 유사한 구조적 긴장. |
| **AlphaFlow** (2025) | MeanFlow의 그래디언트 충돌 분석·완화 | $[0,T]$ | few-step | MeanFlow의 학습 메커니즘이 충분히 이해되지 않아 추가 진전을 막고 있다는 문제의식에서 출발, MeanFlow 목표의 분해를 통해 강한 음의 상관을 드러내고 그래디언트 충돌을 해결하는 더 효율적 최적화 기법을 설계. TCM도 ‘충돌 분리’ 계열로 묶을 수 있음. |
| **Duality Models (DUMO)** (2025) | 동일 입력에서 velocity와 flow-map을 동시 예측 | $[0,T]$ | 1 | 기존 ‘하나의 입력, 하나의 출력’ 구조가 학습 예산 분할(예: MeanFlow는 약 75%를 multi-step 목적에 할당)을 강제해 few-step 생성을 덜 학습시키는 trade-off를 지적, ‘하나의 입력, 두 출력’으로 두 신호를 동시 예측해 백본 유지하며 0.5% 미만 오버헤드로 해결. TCM과 ‘두 task의 분리’라는 문제의식을 공유. |

**관통하는 흐름.** 2023년의 CM 이후, “두 시점 사이의 무엇(매핑/속도/평균 속도/flow map)을 학습할까”와 “시간축·단계 수·입력 조건 중 어디를 자르거나 조건으로 줄까”를 두고 분기가 일어났습니다. **TCM의 위치는 “시간축을 잘라 task를 분리하되 사전학습 모델로 경계조건을 강제”**라는, 비교적 적은 변경으로 *기존 CM 파이프라인을 개선*하는 실용 노선입니다. 반대로 MeanFlow/Shortcut/DUMO는 *목적함수·아키텍처 자체*를 재설계하는 노선입니다. 두 흐름은 상호배타적이지 않아 결합 연구가 자연스러운 다음 단계입니다.

---

## 정확도에 대한 단서

- 논문 본문(수식·표·실험·하이퍼파라미터)은 제공된 PDF에서 직접 확인했습니다.
- ICLR 2025 채택 사실, 공식 코드, MeanFlow/Shortcut/CTM/AlphaFlow/DUMO/SCott/ECM의 *접근 방식 설명*은 아래 참고자료에서 확인했습니다.
- **각 외부 모델의 정확한 FID 수치 비교는 평가 프로토콜 차이가 커 단정하지 않았습니다.** 필요하시면 특정 데이터셋·NFE 조건을 지정해 주시면 다시 정리하겠습니다.

---

## 참고자료

본문 (논문 자체)
- Lee, Xu, Geffner, Fanti, Kreis, Vahdat, Nie. *Truncated Consistency Models*. ICLR 2025. arXiv:2410.14895. 제공된 PDF 본문 및 Appendix A–G.
- 프로젝트 페이지: https://truncated-cm.github.io/
- OpenReview (ICLR 2025 Poster): https://openreview.net/forum?id=ZYDEJEvCbv
- 공식 코드: NVlabs/TCM (https://github.com/NVlabs/TCM)

비교 대상 외부 자료
- Song et al. *Consistency Models*. ICML 2023. https://arxiv.org/pdf/2303.01469
- Geng et al. *Consistency Models Made Easy* (ECM). 2024. https://arxiv.org/pdf/2406.14548 ; OpenReview: https://openreview.net/forum?id=xQVxo9dSID
- Kim et al. *Consistency Trajectory Models* (CTM). ICLR 2024. https://consistencytrajectorymodel.github.io/CTM/
- Heek, Hoogeboom, Salimans. *Multistep Consistency Models*. 2024. https://arxiv.org/pdf/2403.06807
- Liu et al. *SCott: Accelerating Diffusion Models with Stochastic Consistency Distillation*. 2024. https://arxiv.org/abs/2403.01505
- Frans, Hafner, Levine, Abbeel. *One Step Diffusion via Shortcut Models*. 2024. https://arxiv.org/pdf/2410.12557
- Geng et al. *Mean Flows for One-step Generative Modeling* (MeanFlow). 2025. https://arxiv.org/pdf/2505.13447
- *AlphaFlow: Understanding and Improving MeanFlow Models*. 2025. https://arxiv.org/pdf/2510.20771
- Sun, Shang, Lin, Shen. *Duality Models* (DUMO). https://arxiv.org/pdf/2602.17682
