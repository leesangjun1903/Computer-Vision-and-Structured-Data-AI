# Position: AI & ML Deepfake Research is Misaligned with AI Generated Non-Consensual Intimate Imagery (AIG-NCII)

> **참고 자료:**
> - Li Qiwei, Wells Lucas Santo, Sarita Schoenebeck, Eric Gilbert. "Position: AI/ML Deepfake Research is Misaligned with AI-Generated Non-Consensual Intimate Imagery (AIG-NCII)." *Proceedings of the 43rd International Conference on Machine Learning (ICML)*, Seoul, South Korea. PMLR 306, 2026.
> - 논문 내 인용 문헌 전체 (본문 참조)

---

## 1. 핵심 주장 및 주요 기여 요약

### 1.1 핵심 주장

이 포지션 페이퍼의 핵심 주장은 다음의 구조적 불일치(structural misalignment)로 집약됩니다:

$$\underbrace{\text{AI/ML 딥페이크 연구}}_{\text{시청자 중심(viewer-centric) 인식론적 피해}} \perp \underbrace{\text{AIG-NCII 현실}}_{\text{피해자 중심(subject-centric) 존엄 피해}}$$

즉, 현재 연구는 **"이 이미지는 가짜인가?"** 라는 질문에 집중하지만, AIG-NCII 피해는 **"이 이미지는 동의 없이 만들어졌는가?"** 라는 질문에 의해 결정됩니다. 두 축은 서로 직교(orthogonal)합니다.

### 1.2 주요 기여 (3가지)

| 기여 | 내용 |
|------|------|
| **① 연구 동기와 실제 피해의 괴리 발굴** | 2020–2025년 상위 인용 논문 39편 분석 → AIG-NCII를 기술적으로 다룬 논문 **0편** |
| **② 진위 판별 도구의 한계 분석** | 탐지·출처추적·워터마킹 패러다임이 AIG-NCII에 부적절하며, 경우에 따라 피해를 악화시킴을 논증 |
| **③ 재정렬을 위한 구체적 권고안 제시** | 위협 모델 갱신, AI 안전 연구 범위 확장, 윤리적 파트너십 등 9가지 권고안(R1–R9) 제시 |

---

## 2. 해결하고자 하는 문제, 제안 방법, 모델 구조, 성능 및 한계

### 2.1 해결하고자 하는 문제

#### 문제의 규모

- 딥페이크 영상의 약 **98%**가 포르노그래픽 성격을 띰 (Security Hero, 2023)
- Grok AI 생성 이미지 중 **51.7%**가 AIG-NCII로 분류 (Bouchaud, 2026)
- 반면, 기술적 방어 논문 중 AIG-NCII를 **기술적으로 구현한 논문 = 0%**

#### 두 가지 피해 유형의 정의

논문은 피해를 다음과 같이 구분합니다:

$$\text{Viewer-centric harm} \cap \text{Subject-centric harm} \neq \emptyset$$

하지만 이 둘을 해결하는 **메커니즘은 서로 독립적**입니다:

- **시청자 중심 인식론적 피해 (Viewer-centric epistemic harm)**: 허위 정보, 사기, 선거 조작 등 시청자가 속아서 발생하는 피해
- **피해자 중심 존엄 피해 (Subject-centric dignity harm)**: 동의 없는 신원 보존(non-consensual identity preservation) 및 비동의적 변형(non-consensual modification)으로 인한 피해 — **이미지의 진위와 무관하게 발생**

#### 직교성의 핵심 논리

현재 탐지 패러다임이 측정하는 축($\text{Authentic} \leftrightarrow \text{Synthetic}$)과 안전성을 결정하는 축($\text{Consensual} \leftrightarrow \text{Non-consensual}$)은 서로 직교합니다:

$$\text{Safety} \not\propto \text{Authenticity}$$

이를 표로 나타내면:

|  | **Safe** | **Harmful** |
|---|---|---|
| **Synthetic** | AI를 이용한 예술적 자기표현 | **AIG-NCII** |
| **Authentic** | 동의된 포르노그래피 | 전통적 NCII |

이 직교성으로 인해, 현재 진위 판별 도구는 AIG-NCII를 다루는 데 있어 **범주 오류(category error)**를 범합니다.

### 2.2 제안하는 방법 (수식 포함)

이 논문은 포지션 페이퍼이므로 새로운 모델을 직접 제안하지는 않습니다. 대신, 기존 패러다임의 한계를 분석하고 대안적 방향을 제시합니다. 기존 기술들의 핵심 수식 및 원리를 정리하면:

#### (A) 탐지 (Detection) 패러다임

현재 탐지 모델은 다음의 이진 분류 문제로 정식화됩니다:

$$f_\theta: \mathcal{X} \rightarrow \{0, 1\}, \quad \text{where } 0 = \text{authentic}, 1 = \text{synthetic}$$

**DIRE (Wang et al., 2023)**의 핵심 원리는 확산 모델로 생성된 이미지가 사전 학습된 확산 프로세스로 역산(inversion)될 때 재구성 오류가 낮다는 관찰에 기반합니다:

$$\mathcal{L}_{\text{recon}} = \|x - \hat{x}\|^2, \quad \hat{x} = \text{DDIM inversion}(x)$$

$$\text{score}(x) = \mathcal{L}_{\text{recon}}(x), \quad \text{synthetic if } \mathcal{L}_{\text{recon}} < \tau$$

**CLIP 기반 범용 탐지기 (Ojha et al., 2023)**:

$$f_\theta(x) = \sigma(W \cdot \phi_{\text{CLIP}}(x) + b)$$

여기서 $\phi_{\text{CLIP}}$은 고정된 CLIP 인코더, $W$는 학습 가능한 선형 분류기입니다.

#### (B) 출처 추적 (Provenance) 패러다임

C2PA 기술 표준에 기반한 방법으로, 미디어 자산에 암호화 서명을 결합합니다:

$$\text{Manifest} = \text{Sign}_{sk}(H(x) \| \text{metadata})$$

$$\text{Verify}(x, \text{Manifest}) = \begin{cases} \text{authentic} & \text{if } \text{Verify}_{pk}(\text{Manifest}) = H(x) \\ \text{tampered} & \text{otherwise} \end{cases}$$

#### (C) 워터마킹 (Watermarking) 패러다임

잠재 워터마킹 (Fernandez et al., 2023):

$$z_w = z + \delta_w, \quad \|\delta_w\|_\infty < \epsilon$$

여기서 $z$는 잠재 공간(latent space) 벡터, $\delta_w$는 사람 눈에 보이지 않는 워터마크 신호입니다.

샘플링 기반 워터마킹 (Tree-Ring, Wen et al., 2023):

$$z_T \leftarrow z_T \odot M_w, \quad M_w \text{: ring-shaped pattern in Fourier domain}$$

$$\text{Detect}(x) = \text{DDIM inversion}(x) \odot M_w^{-1}$$

### 2.3 논문이 지적하는 모델 구조적 한계

#### 진위 판별의 구조적 실패

현재 모든 탐지 모델은 다음 가정 하에 작동합니다:

$$\text{Harm}(x) \approx \mathbf{1}[x \in \mathcal{X}_{\text{synthetic}}]$$

그러나 AIG-NCII의 실제 피해 함수는:

$$\text{Harm}_{\text{AIG-NCII}}(x) = \mathbf{1}[x \in \mathcal{X}_{\text{non-consensual}}], \quad \perp \mathbf{1}[x \in \mathcal{X}_{\text{synthetic}}]$$

#### 기술적 생태계의 발전 경로

| 시기 | 기술 | 핵심 아키텍처 |
|------|------|--------------|
| 2017–2022 | 페이스 스와핑 | 오토인코더, DeepFaceLab |
| ~2022 | 이미지 변환 | Pix2Pix conditional GAN |
| 2022–현재 | 텍스트-투-이미지 생성 | Stable Diffusion (LDM) |
| 현재 | 개인화 미세조정 | DreamBooth, LoRA |

**DreamBooth 파인튜닝 (Ruiz et al., 2023):**

$$\mathcal{L} = \mathbb{E}_{x, c, \epsilon, t}\left[\|\epsilon - \epsilon_\theta(z_t, t, c)\|^2\right] + \lambda \mathbb{E}_{x_{pr}, c_{pr}, \epsilon', t'}\left[\|\epsilon' - \epsilon_\theta(z_{t'}', t', c_{pr})\|^2\right]$$

여기서 첫 번째 항은 대상 인물의 개인화 손실, 두 번째 항($\lambda$로 조정)은 사전 보존(prior preservation) 손실입니다. 이 기술은 **소수의 참조 사진만으로** 특정 인물의 모습을 생성 모델에 "주입"할 수 있어 AIG-NCII의 핵심 도구가 됩니다.

**LoRA (Hu et al., 2022):**

$$W = W_0 + \Delta W = W_0 + BA, \quad B \in \mathbb{R}^{d \times r}, A \in \mathbb{R}^{r \times k}, r \ll \min(d, k)$$

낮은 랭크 행렬 분해를 통해 소수의 파라미터만 학습하여 효율적인 개인화를 가능하게 합니다.

### 2.4 성능 향상 및 기술적 가능성

논문은 새로운 알고리즘을 제안하지 않지만, 다음과 같은 방향에서 성능 향상 가능성을 제시합니다:

#### 적대적 면역화 (Adversarial Immunization)

$$\delta^* = \arg\max_{\|\delta\|_p \leq \epsilon} \mathcal{L}_{\text{inpaint}}(f_\theta(x + \delta))$$

사람 눈에 보이지 않는 perturbation $\delta$를 이미지에 추가하여 생성 모델의 인페인팅(nudification) 기능을 방해합니다.

관련 연구들:
- **Anti-DreamBooth (Van Le et al., 2023)**: 신원 학습 자체를 방해
- **Glaze (Shan et al., 2023)**: 스타일 모방 방지
- **AdvPaint (Jeon et al., 2025)**: 어텐션 구조 교란을 통한 인페인팅 방지
- **BlurGuard (Kim et al., 2026)**: 강건성 향상
- **DiffVax (Ozden et al., 2025)**: 최적화 없는 면역화

### 2.5 한계

1. **적대적 방어의 취약성**: 압축, 크롭, 필터 등 단순 변환에도 방어 효과가 사라짐 (Hönig et al., 2025)
2. **탐지 모델의 범용성 부족**: 새로운 생성 아키텍처 등장 시 재학습 필요
3. **법적/플랫폼 차원의 한계**: 규제가 생성 기술의 진화 속도를 따라가지 못함
4. **윤리적 데이터 문제**: AIG-NCII 방어 연구를 위한 안전한 데이터셋 구축이 극히 어려움
5. **"Whac-a-mole" 문제**: 한 플랫폼에서 차단되면 다른 플랫폼으로 이동 (CivitAI → HuggingFace)

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 현재 탐지 모델의 일반화 실패 원인

일반화 실패의 핵심 원인은 **분포 이동(distribution shift)**입니다:

$$\mathcal{D}_{\text{train}} = \{(x_i, y_i)\}_{i=1}^N \sim p_{\text{GAN}}, \quad \mathcal{D}_{\text{test}} \sim p_{\text{diffusion}} \neq p_{\text{GAN}}$$

GAN 특화 탐지기는 diffusion 모델 이미지에 일반화되지 않으며, 특정 diffusion 모델에 특화된 탐지기도 새로운 아키텍처(예: FLUX)에 일반화되지 않습니다.

### 3.2 CLIP 기반 일반화 전략 (Ojha et al., 2023)

논문이 주목하는 가장 유망한 일반화 접근법은 CLIP 특징 공간의 활용입니다:

$$\text{UnivFD}(x) = \sigma\left(W \cdot \phi_{\text{CLIP}}(x)\right)$$

**일반화 원리**: CLIP은 대규모 자연어-이미지 쌍으로 학습되어 의미론적 패턴을 포착합니다. 생성 모델이 만드는 "합성 의미론적 패턴(synthetic semantic patterns)"은 아키텍처와 무관하게 일관된 특성을 보입니다.

$$\mathcal{L}_{\text{generalize}} = \mathbb{E}_{g \sim \mathcal{G}} \mathbb{E}_{x \sim p_g}[\ell(f(x), y)]$$

여기서 $\mathcal{G}$는 모든 생성 모델의 집합입니다.

### 3.3 AIG-NCII 맥락에서의 일반화 방향 재정의

논문은 기존 탐지 일반화 목표를 비판하고, **AIG-NCII에 적합한 새로운 일반화 목표**를 제시합니다:

#### 기존 일반화 목표 (인식론적 목표):

$$\max_\theta \sum_{g \in \mathcal{G}} \text{AUC}\left(f_\theta, \mathcal{D}_g\right)$$

#### 논문이 제안하는 일반화 목표 (존엄 보호 목표):

$$\min_\theta P(\text{identity retained} | \text{attack attempted})$$

즉, 성공적인 방어는 **탐지 정확도 극대화**가 아니라 **신원 재현 가능성의 최소화**로 정의됩니다.

### 3.4 적대적 면역화의 일반화 가능성

**강건한 perturbation 설계**가 일반화의 핵심입니다:

$$\delta^* = \arg\min_{\|\delta\|_\infty \leq \epsilon} \max_{\theta' \in \Theta} P(\text{identity preserved} | x + \delta, f_{\theta'})$$

이 min-max 문제는 특정 모델에 과적합되지 않는 perturbation을 학습합니다. 그러나 현실에서는:

- **적대적 공격의 이전 가능성(transferability)** 부족: 특정 모델에서 계산된 perturbation이 다른 모델에서 효과가 없을 수 있음
- **압축/변환에 대한 취약성**: JPEG 압축, 리사이징 등에 의해 perturbation이 파괴됨

**BlurGuard (Kim et al., 2026)**와 **DiffVax (Ozden et al., 2025)**는 이 취약성을 줄이려는 시도로, 논문이 이를 미래 연구의 방향으로 제시합니다.

### 3.5 언어-가이드 대조 학습 (Wu et al., 2025)

$$\mathcal{L}_{\text{contrastive}} = -\log \frac{\exp(\text{sim}(\phi_v(x), \phi_l(t^+))/\tau)}{\sum_{j} \exp(\text{sim}(\phi_v(x), \phi_l(t^j))/\tau)}$$

텍스트 설명을 활용해 의미론적 수준에서 진짜/가짜를 구분하는 접근법으로, 새로운 생성 모델에 대한 일반화 가능성을 높입니다.

---

## 4. 미래 연구에 미치는 영향 및 고려 사항

### 4.1 미래 연구에 미치는 영향

#### 4.1.1 연구 어젠다의 패러다임 전환

이 논문은 딥페이크 연구 커뮤니티에 **알고리즘 공정성(Algorithmic Fairness)** 분야의 전환과 유사한 패러다임 변화를 촉구합니다:

$$\text{과거}: \text{Gender Shades (Buolamwini , Gebru, 2018)} \rightarrow \text{FAccT, AIES 등장}$$

$$\text{현재}: \text{본 논문} \rightarrow \text{AIG-NCII 전문 연구 영역 형성 가능성}$$

#### 4.1.2 AI 안전(AI Safety) 정의의 확장

현재 AI 안전은 주로 존재론적 위험(existential risk)에 집중합니다. 본 논문은 이를 다음과 같이 확장할 것을 요구합니다:

$$\text{AI Safety}_{\text{new}} = \text{AI Safety}_{\text{old}} \cup \underbrace{\text{AIG-NCII 방지}}_{\text{현재적 폭력}}$$

#### 4.1.3 위협 모델(Threat Model)의 재설계

기존 위협 모델:

$$\text{Adversary}: \text{anonymous attacker}, \text{Goal}: \text{deceive viewer}$$

AIG-NCII 위협 모델:

$$\text{Adversary}: \text{ex-partner/acquaintance}, \text{Goal}: \text{violate subject's dignity}$$
$$\text{Capability}: \text{few-shot photos} + \text{LoRA/DreamBooth}$$

이 새로운 위협 모델은 컴퓨터 보안의 **친밀 파트너 폭력(IPV) 연구** (Chatterjee et al., 2018; Havron et al., 2019)에서 영감을 얻어야 한다고 논문은 주장합니다.

#### 4.1.4 평가 지표의 혁신

현재 지표:

$$\text{Metric}_{\text{current}} = \text{AUC, F1, Accuracy}$$

새롭게 필요한 지표:

$$\text{Metric}_{\text{new}} = P(\text{abuse generated successfully}), \quad \text{Harm reduction rate}$$

Cretu et al. (2025)의 CSAM 필터 평가 방법(실제 유해 콘텐츠 없이 윤리적 프록시 사용)이 참고 모델입니다.

### 4.2 2020년 이후 관련 최신 연구 비교 분석

| 연구 | 연도 | 접근법 | AIG-NCII 관련성 | 한계 |
|------|------|--------|----------------|------|
| Frank et al. | 2020 | 주파수 분석 (GAN 아티팩트) | ❌ | GAN 특화, 확산 모델에 비적용 |
| Ojha et al. | 2023 | CLIP 기반 범용 탐지 | ❌ (언급 없음) | 진위 판별에만 집중 |
| Wang et al. (DIRE) | 2023 | 확산 재구성 오류 | ❌ | 인식론적 목표에만 집중 |
| Van Le et al. (Anti-DreamBooth) | 2023 | 적대적 면역화 | ✅ (부분적) | 취약성, 실사용 어려움 |
| Shan et al. (Glaze) | 2023 | 스타일 모방 방지 perturbation | ✅ (부분적) | 압축에 취약 |
| Liu et al. | 2024 | 위조 인식 적응형 트랜스포머 | ❌ | 인식론적 목표 |
| Sun et al. (DiffusionFake) | 2024 | Stable Diffusion 가이드 일반화 | ❌ | 탐지 일반화에 집중 |
| Jeon et al. (AdvPaint) | 2025 | 어텐션 교란 기반 방어 | ✅ | 아직 취약성 존재 |
| Gibson et al. | 2025 | AI 누드화 앱 생태계 분석 | ✅ | 기술적 방어 없음, 분석에 그침 |
| Kim et al. (BlurGuard) | 2026 | 강건한 이미지 보호 | ✅ | 최신, 범용성 미검증 |

### 4.3 향후 연구 시 고려해야 할 점

#### ① 위협 모델의 명시적 설계

$$\text{모든 AIG-NCII 방어 연구} \Rightarrow \text{주체 중심 위협 모델 명시 필수}$$

- 가해자 프로파일: 소수의 참조 이미지 + 파라미터 효율적 미세조정 도구
- 피해 시나리오: 신원 보존 vs. 비동의적 변형

#### ② 안전한 데이터셋 구축

- 실제 AIG-NCII 데이터 수집은 비윤리적 → **합성 프록시 데이터셋** 필요
- 누드 이미지를 학습 데이터로 사용하는 것의 윤리적 검토 필수 (Cintaqia et al., 2025)

#### ③ 개방형 모델 릴리즈 정책 재검토

$$\text{Open Science Value} \text{ vs. } \text{Dual-use Risk}$$

고충실도 신원 보존 모델은 **게이트 접근(gated access)** 또는 **연구자 전용** 공개로 제한해야 합니다.

#### ④ 학제 간 협업 필수화

$$\text{기술 연구자} + \text{성폭력 예방 전문가} + \text{피해자 지원 단체} + \text{정책 입안자}$$

기술 설계 단계부터 통합해야 하며, 사후 검증으로 그쳐서는 안 됩니다.

#### ⑤ 오류의 비대칭 비용 고려

$$\text{FP (합법 콘텐츠 차단)}: \text{일시적, 되돌릴 수 있음}$$
$$\text{FN (AIG-NCII 통과)}: \text{영구적, 돌이킬 수 없음}$$

$$\therefore \text{Loss}_{\text{asymmetric}} = w_{\text{FN}} \cdot \text{FN} + w_{\text{FP}} \cdot \text{FP}, \quad w_{\text{FN}} \gg w_{\text{FP}}$$

#### ⑥ 연구자 보호

- 민감한 콘텐츠를 검토하는 연구자의 **이차 외상 스트레스(secondary traumatic stress)** 방지 프로토콜 마련 (Williamson et al., 2020)

#### ⑦ 피해자 다양성 존중

$$\exists \text{ no one-size-fits-all solution}$$

- 어떤 피해자는 콘텐츠 삭제를 원하고, 어떤 피해자는 법적 조치를 원함
- 복원적 정의(restorative justice) 프레임워크 참조

---

## 결론 요약

이 논문은 기술 연구 자체가 AIG-NCII 문제의 **직접적 기여자**임을 지적하고, 연구 커뮤니티가 **책임 있는 대응 주체**로 나서야 한다고 주장합니다. 핵심 메시지를 한 문장으로 요약하면:

> *"AI 안전의 정의는 진실만이 아니라 사람의 존엄성도 보호해야 한다."*
> *"The research community bears a responsibility to ensure that our definitions of AI safety protect not only truth, but also the dignity of people."*
