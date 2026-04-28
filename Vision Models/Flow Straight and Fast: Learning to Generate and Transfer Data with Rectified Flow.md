
# Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow

---

## 1. 핵심 주장 및 주요 기여 요약

이 논문은 **Rectified Flow**를 제안합니다. 이는 두 개의 경험적 분포 $\pi_0$과 $\pi_1$ 사이의 수송을 위한 (신경망 기반) ODE 모델을 학습하는 놀랍도록 단순한 접근법으로, 생성 모델링과 도메인 전이에 대한 통합된 해결책을 제공합니다.

### 주요 기여 4가지

| 기여 | 내용 |
|------|------|
| 통합 프레임워크 | 생성 모델링과 도메인 전이를 하나의 ODE 수송 문제로 통합 |
| 직선 경로 학습 | 최소제곱 최적화로 직선에 가까운 flow 학습 |
| Reflow 절차 | 반복적 재정류를 통한 경로 직선화 및 단일 스텝 생성 |
| 이론적 보장 | 볼록 수송 비용의 단조 감소 증명 |

Rectified Flow의 핵심 아이디어는 ODE가 $\pi_0$과 $\pi_1$에서 추출된 점들을 연결하는 **직선 경로**를 최대한 따르도록 학습하는 것으로, 이는 표준 지도 학습 이상의 추가 파라미터 없이 대규모 모델로 쉽게 확장 가능한 **비선형 최소제곱 최적화**를 통해 달성됩니다.

데이터로부터 Rectified Flow를 학습하는 절차(정류, rectification)는 $\pi_0$과 $\pi_1$의 임의의 결합을 **볼록 수송 비용이 단조 감소**하는 새로운 결정론적 결합으로 변환합니다. 또한 재귀적으로 정류를 적용하면 경로가 점점 더 직선에 가까워지는 flow의 수열을 얻을 수 있어, 추론 단계에서 거친 시간 이산화로도 정확하게 시뮬레이션할 수 있습니다.

---

## 2. 해결하고자 하는 문제, 제안 방법(수식 포함), 모델 구조, 성능, 한계

### 2.1 해결하고자 하는 문제

이 연구는 비지도 학습의 두 가지 주요 문제를 다룹니다: (1) **생성 모델링** (노이즈에서 데이터 생성), (2) **도메인 전이** (두 분포 간 매핑). 기존 연구들은 수백 개의 스텝을 반복 호출해야 하는 비효율적인 ODE/SDE 모델과, 이 두 작업을 별도로 처리하는 단절된 접근법을 가지고 있었습니다. 이 논문은 이 두 작업을 하나의 수송 매핑 문제로 통합하여 고품질 연속 모델과 빠른 단일 스텝 모델 간의 격차를 해소하고자 합니다.

---

### 2.2 제안하는 방법 (수식 포함)

#### ① 선형 보간 및 학습 목표

두 분포 $\pi_0$ (노이즈)와 $\pi_1$ (데이터)에서 쌍 $(X_0, X_1)$을 독립적으로 샘플링한 후, 선형 보간 경로를 구성합니다:

$$X_t = t X_1 + (1-t) X_0, \quad t \in [0,1]$$

이 보간 $\{X_t\}$는 "앵커-브릿지" 방식으로 생성되는 확률 과정입니다. 구성에 의해 $X_0$과 $X_1$의 주변 분포는 보간 과정을 통해 목표 분포 $\pi_0$과 $\pi_1$에 일치합니다. 그러나 $\{X_t\}$는 $\dot{Z}_t = v_t(Z_t)$와 같은 인과적 ODE 프로세스가 아닙니다.

#### ② 핵심 학습 목적함수 (비선형 최소제곱)

속도장 $v(x, t)$를 다음의 최소제곱 최적화로 학습합니다:

$$\min_v \int_0^1 \mathbb{E}\left[\|X_1 - X_0 - v(X_t, t)\|^2\right] dt$$

이 일반화는 여전히 주변 보존 특성(marginal preserving property)을 유지합니다. 즉, 모든 $t \in [0,1]$에 대해 $\mathrm{Law}(X_t) = \mathrm{Law}(Z_t)$가 성립합니다.

#### ③ ODE 시뮬레이션

학습된 속도장으로 ODE를 정의합니다:

$$\frac{dZ_t}{dt} = v(Z_t, t), \quad Z_0 \sim \pi_0$$

Euler 방법으로 이산화하면:

$$Z_{t+\Delta t} = Z_t + \Delta t \cdot v(Z_t, t)$$

직선에 가까운 경로를 가진 flow는 수치 시뮬레이션에서 시간 이산화 오류가 작다는 핵심 계산 이점을 가집니다. 실제로 ODE $dZ_t = v(Z_t,t)dt$의 경로가 완벽하게 직선이라면, 이 ODE는 단일 Euler 스텝으로 정확하게 풀 수 있어 ODE/SDE 모델의 느린 추론이라는 병목을 해결합니다.

#### ④ Reflow 절차 (반복적 직선화)

재귀적으로 $\texttt{Rectify}(\cdot)$ 절차를 적용하여 다음 수열을 생성합니다:
$$\texttt{Reflow:} \quad \{Z_t^{k+1}\} = \texttt{Rectify}(\texttt{Interp}(Z_0^k, Z_1^k))$$
여기서 $\{Z_t^k\}$는 $k$-번째 Rectified Flow (k-rectified flow)라고 부릅니다.

#### ⑤ 직선성 측정 지표

Flow $Z$의 직선성을 다음 지표로 측정합니다:
$$S(\boldsymbol{Z}) = \int_0^1 \mathbb{E}\left[\|Z_1 - Z_0 - v(Z_t, t)\|^2\right] dt$$
$S(\boldsymbol{Z}) = 0$이면 완벽한 직선 경로에 해당합니다. $K$번의 reflow 후 $\min_{k \leq K} S(\boldsymbol{Z}^k) = O(1/K)$임이 증명됩니다.

#### ⑥ Distillation (증류)

이 방법은 증류도 지원합니다. 학습된 flow를 직접 매핑으로 근사하여 모델을 단일 스텝 생성기로 변환하고, ODE 솔버 반복 없이 추론을 수행할 수 있습니다. Reflow와 결합하면 최소한의 계산 비용으로 고품질 생성이 가능합니다.

---

### 2.3 모델 구조

이 방법은 순수 ODE 기반이며, 개념적으로 더 단순하고 실제 추론 시간에서 SDE 기반 방법보다 빠릅니다.

- **백본**: Score SDE와 동일한 U-Net 아키텍처 계열 사용 (DDPM++ 등)
- **학습**: 표준 지도 학습과 동일한 파이프라인, 추가 파라미터 불필요
- **입력**: $(Z_t, t)$ → **출력**: 속도 예측 $v(Z_t, t)$

Rectified Flow는 GAN의 학습 불안정성, MLE 방법의 계산 불가능한 가능도, 기타 접근법의 미묘한 문제들을 회피하는 단순하고 확장 가능한 비제약 최소제곱 최적화 절차로 학습됩니다.

---

### 2.4 성능 향상

CIFAR-10에서 정량적으로 단일 스텝 빠른 확산/흐름 모델 중 **FID 4.85**, **recall 0.51**의 최첨단 결과를 달성했습니다.

Reflow와 증류를 적용한 distilled 2-rectified flow는 단일 Euler 스텝만으로 FID 4.85를 달성하며 기존 단일 스텝 생성 모델을 능가하고, 극단적인 계산 제약 하에서도 높은 품질을 유지합니다.

Reflow 연산을 통해 ODE 궤적을 반복적으로 직선화하여 결국 단일 스텝 생성을 달성하며, GAN보다 높은 다양성과 빠른 확산 모델보다 우수한 FID를 기록합니다.

| 모델 | NFE | FID (CIFAR-10) | Recall |
|------|-----|----------------|--------|
| DDPM (Ho et al.) | 1000 | ~3.17 | 0.57 |
| 1-Rectified Flow | 127 | 6.18 | 0.50 |
| 2-Rectified Flow (Euler 1-step) | 1 | 12.21 | - |
| **Distill 2-Rectified Flow** | **1** | **4.85** | **0.51** |

실증 연구에서 Rectified Flow는 이미지 생성, 이미지-이미지 변환, 도메인 적응에서 뛰어난 성능을 보여줍니다.

---

### 2.5 한계점

Rectified Flow는 반복적 증류를 통해 궤적을 재배선하고 직선화하는 방식을 사용하므로, **여러 라운드의 학습이 필요하고 누적 오류**가 발생할 수 있습니다.

반복적 fake-pair("ghost coupling") reflow 학습은 모델이 실제 데이터에서 멀어지게 할 수 있으며, 단일/소수 스텝 체제에서 품질이 저하될 수 있습니다. 최근의 balanced 및 conic 전략은 실제 쌍 앵커링을 통해 이를 해결합니다.

원래의 Rectified Flow는 비용 무관(cost-agnostic)하므로, 사용자가 지정한 비용에 대한 최적 결합을 항상 제공하지는 않습니다.

- **고차원 데이터**: Reflow를 위해 대규모 합성 쌍 생성에 높은 메모리·저장 비용 필요
- **다중 학습 라운드**: 각 Reflow가 새로운 모델 학습을 필요로 함

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 이론적 일반화 근거

정류 절차는 $\pi_0$과 $\pi_1$의 임의의 결합을 **모든 볼록 비용 함수 $c$에 대해 동시에 볼록 수송 비용이 단조 감소**하는 새로운 결정론적 결합으로 변환함이 증명됩니다.

주변 보존(marginal preservation): ODE 샘플링 포인트들은 신경망 오류와 무관하게 구성에 의해 올바른 시간 주변분포를 가집니다.

### 3.2 다양한 도메인으로의 일반화

Rectified Flow는 $X_0$과 $X_1$의 임의의 매끄러운 보간 과정 $X_t$로 일반화될 수 있습니다. 이 경우 속도장은 시간 $t$에서 $z$를 지나는 모든 $X_t$ 궤적의 기울기 $\dot{X}_t$의 기댓값으로 구성됩니다:
$$\min_v \int_0^1 \mathbb{E}\left[\|\dot{X}_t - v(X_t, t)\|^2\right] dt$$
이 일반화는 여전히 주변 보존 특성을 유지합니다.

재귀적 정류 및 아키텍처별 적응을 통한 발전이 모델 충실도를 향상시키고 이미지, 오디오, 텍스트, 유체 시뮬레이션에 걸쳐 응용 범위를 넓히고 있습니다.

### 3.3 확장성 및 일반화의 실증적 증거

FlowTS와 같은 후속 연구는 Rectified Flow를 활용하여 **무조건부 생성에서 조건부 생성으로 재학습 없이** 원활하게 적응할 수 있음을 보여주며, 효율적인 실제 배포를 가능하게 합니다.

유한 차원에서의 Rectified Flow의 이점들(수송 비용 감소 및 직선화 효과)은 **무한 차원 Hilbert 공간**에서도 동일하게 성립하며, 이는 주변 보존 특성에서 직접 도출됩니다.

---

## 4. 후속 연구에 미치는 영향 및 앞으로 고려할 점

### 4.1 연구 영향

InstaFlow는 연속 시간 확산 모델과 단일 스텝 생성 모델 사이의 격차를 크게 줄이며, 알고리즘 혁신에 영감을 주고 3D 생성 등 다운스트림 작업에 이점을 가져다줍니다.

실증 연구들은 RFM 방법이 vanilla 확산 또는 flow 모델에 비해 **필요한 추론 스텝을 10~100배 줄이면서도** 품질 저하가 최소화됨을 일관되게 보여줍니다.

### 4.2 2020년 이후 관련 최신 연구 비교 분석

| 연구 | 연도 | 핵심 기여 | Rectified Flow와의 관계 |
|------|------|-----------|------------------------|
| **DDPM** (Ho et al.) | 2020 | SDE 기반 확산 모델 | 비교 베이스라인; RF는 더 빠른 ODE 대안 |
| **Score SDE** (Song et al.) | 2020 | Score-matching + SDE | RF는 이를 ODE 특수 케이스로 포함 |
| **Flow Matching** (Lipman et al.) | 2022 | 조건부 flow 학습 | RF와 동시에 제안; 상호 보완적 |
| **Stochastic Interpolants** (Albergo & Vanden-Eijnden) | 2022 | 보간 기반 flow | RF와 동시 제안 |
| **InstaFlow** (Liu et al.) | 2024 | RF를 Stable Diffusion에 적용 | RF의 직접 응용; 1-step text-to-image |
| **Consistency Models** (Song et al.) | 2023 | 단일 스텝 증류 | RF Reflow와 유사한 teacher-student 패러다임 |
| **OT-CFM** (Tong et al.) | 2023 | 미니배치 최적 수송 결합 | RF의 결합 전략을 OT로 개선 |
| **Stable Diffusion 3 / SD3** (Esser et al.) | 2024 | Rectified Flow Transformer 스케일업 | RF를 대규모 생성 모델의 핵심 기반으로 채택 |
| **Rectified Diffusion** (Wang et al.) | 2024 | 직선성이 필수 조건이 아님을 주장 | RF의 핵심 가정을 재검토하고 일반화 |
| **Consistency Flow Matching** | 2024 | 속도 일관성으로 직선 flow 정의 | RF의 직선화를 더 유연하게 개선 |

Rectified Flow는 특정 궤적을 가진 Flow Matching으로 볼 수 있습니다. Rectified Flow는 반복적 증류를 통해 궤적을 재배선하고 직선화하는 방식을 제안하지만, **여러 라운드의 학습이 필요하고 누적 오류가 발생할 수 있습니다.**

Rectified Diffusion(2024)은 정류의 성공이 주로 사전 학습된 확산 모델을 사용하여 매칭된 노이즈-샘플 쌍을 얻고 이를 재학습에 사용하는 데 있으며, 직선성 자체가 필수 학습 목표가 아닐 수 있음을 주장합니다.

최신 연구에서는 엄격한 기하학적 "직선성"이 모든 경우에 필요하거나 최적인 것은 아니며, ODE 경로를 따른 신경망의 1차 일관성(pathwise self-consistency)을 강제하는 것이 빠르고 정확한 샘플링을 위한 근본적 기준임이 밝혀졌습니다.

### 4.3 향후 연구 시 고려할 점

향후 발전은 개선된 솔버 적응 방식, 고차원에서의 직선성과 주변성의 추가 분석, 향상된 의미론적 분리, 그리고 다중 모달, 계층적, 플러그-앤-플레이 생성 작업으로의 확장에 집중될 것으로 예상됩니다.

Rectified Flow 모델을 더 광범위한 생성 모델링 프레임워크와 통합하고, 더 복잡한 비용 함수 환경을 포함하며, 반지도 학습 또는 구조화된 데이터 도메인과 같은 설정으로의 확장을 탐색하는 것이 제안됩니다.

계층적 정형화(hierarchical formulation)는 다중 ODE를 결합하여 속도와 가속도 등 고차 미분을 모델링함으로써 궤적 곡률과 함수 평가 횟수를 더욱 줄이는 방향으로 발전하고 있습니다.

---

## 📚 참고 자료 (출처)

1. **원논문**: Liu, X., Gong, C., & Liu, Q. (2022/2023). *Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow.* ICLR 2023 Spotlight. arXiv:2209.03003. https://arxiv.org/abs/2209.03003

2. **공식 구현**: GitHub - gnobitab/RectifiedFlow. https://github.com/gnobitab/RectifiedFlow

3. **공식 리뷰 페이지**: OpenReview - Flow Straight and Fast (ICLR 2023). https://openreview.net/forum?id=XVjTT1nw5z

4. **공식 소개 웹사이트**: Rectified Flow — Introduction. https://www.cs.utexas.edu/~lqiang/rectflow/html/intro.html

5. **블로그 소개**: Rectified Flow: Straight is Fast. https://rectifiedflow.github.io/blog/2024/intro/

6. **NSF Public Access Repository**: Flow Straight and Fast. https://par.nsf.gov/biblio/10440561

7. **InstaFlow 논문**: Liu, X. et al. (2024). *InstaFlow: One Step is Enough for High-Quality Diffusion-Based Text-to-Image Generation.* ICLR 2024. arXiv:2309.06380.

8. **Rectified Diffusion 논문**: Wang et al. (2024). *Rectified Diffusion: Straightness Is Not Your Need in Rectified Flow.* OpenReview / arXiv:2410.07303.

9. **Consistency Flow Matching**: Yang et al. (2024). *Consistency Flow Matching: Defining Straight Flows with Velocity Consistency.* arXiv. https://openreview.net/pdf/...

10. **Semantic Scholar 관련 논문 목록**: https://www.semanticscholar.org/paper/Flow-Straight-and-Fast

11. **Emergent Mind - Rectified Flow 분석**: https://www.emergentmind.com/topics/rectified-flow-models

12. **FlowTS 논문**: arXiv:2411.07506 (2025). *FlowTS: Time Series Generation via Rectified Flow.*

13. **Functional Rectified Flow 논문**: arXiv:2509.10384 (2025). *Flow Straight and Fast in Hilbert Space.*

14. **개인 블로그 분석**: Dong-Keon Kim (2026). *Rectified Flow: Scalable Generative Modeling via Neural ODE Transport.* Medium.

15. **GitHub Awesome Flow Matching**: https://github.com/dongzhuoyao/awesome-flow-matching

# Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow

### 1. 핵심 주장과 주요 기여

"Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow" 논문은 **Rectified Flow (직선화 흐름)**이라는 간단하면서도 효과적인 접근법을 제시합니다. 이 방법의 핵심 주장은 신경 상미분방정식(Neural ODE)을 학습할 때, **두 분포 사이를 연결하는 경로가 가능한 한 직선에 가까워야 한다**는 것입니다.[1]

**주요 기여는 다음과 같습니다:**

1. **직선 경로 학습을 통한 통일된 프레임워크**: 생성 모델링과 도메인 전환이라는 서로 다른 작업을 하나의 프레임워크로 해결[1]

2. **확장 가능한 최소자승 최적화**: 표준 감독 학습을 넘어서는 추가 매개변수 없이 간단한 비선형 최소자승 문제로 해결[1]

3. **수송 비용 감소**: 단계적 직선화(reflow)를 통해 모든 볼록 비용 함수에 대해 수송 비용이 감소함을 이론적으로 증명[1]

4. **빠른 추론**: 직선 경로는 이산화 없이 정확히 시뮬레이션할 수 있어 단일 오일러 스텝으로도 고품질 결과 생성 가능[1]

### 2. 문제 정의 및 제안 방법

#### 핵심 문제: 수송 매핑 문제

논문이 해결하려는 문제는 다음과 같이 정의됩니다:[1]

**두 분포 $$X_0 \sim \pi_0$$, $$X_1 \sim \pi_1$$의 관찰값이 주어질 때, 수송 함수 $$T: \mathbb{R}^d \to \mathbb{R}^d$$를 찾아 $$Z_1 := T(Z_0) \sim \pi_1$$ (단, $$Z_0 \sim \pi_0$$)을 만족하도록 하는 것**

기존 방법들의 문제점:[1]
- **GAN**: 학습 불안정성과 모드 붕괴
- **VAE**: 계산 복잡성으로 인한 제약
- **확산 모델**: 추론 비용이 매우 높음 (수천 단계 필요)

#### 제안된 방법: Rectified Flow

**기본 아이디어**: 선형 보간 $$X_t = tX_1 + (1-t)X_0$$의 직선 경로를 따르도록 ODE 드리프트(drift) 함수를 학습[1]

**최적화 문제** (식 1):
$$\min_v \int_0^1 \mathbb{E}\left[\|(X_1 - X_0) - v(X_t, t)\|^2\right] dt,$$
여기서 $$X_t = tX_1 + (1-t)X_0$$[1]

**ODE 흐름**:
$$dZ_t = v(Z_t, t)dt$$[1]

#### 알고리즘 1: Rectified Flow 주요 단계[1]

1. **훈련**: 다음 목적함수 최소화

$$\hat{\theta} = \arg\min_{\theta} \mathbb{E}\left[\|X_1 - X_0 - v_\theta(tX_1 + (1-t)X_0, t)\|^2\right], \quad t \sim \text{Uniform}([0,1])$$

2. **샘플링**: $$Z_0 \sim \pi_0$$에서 시작하여 ODE 풀기
   - 정방향: $$dZ\_t = v_\theta(Z_t, t)dt$$
   - 역방향: $$d\tilde{X}\_t = -v_\theta(\tilde{X}_t, t)dt$$

3. **선택사항 - Reflow**: $$Z^{k+1} = \text{RectFlow}((Z^k_0, Z^k_1))$$로 재귀적으로 흐름 직선화

4. **선택사항 - Distillation**: 마지막 단계에서 흐름을 신경망 $$\hat{T}(z_0) = z_0 + v(z_0, 0)$$으로 증류[1]

### 3. 모델 구조와 이론적 기초

#### 최적 드리프트 속도 필드

문제 (1)의 정확한 최솟값은 다음과 같습니다:[1]

$$v_X(x, t) = \mathbb{E}[X_1 - X_0 | X_t = x]$$

조건부 밀도가 존재할 때, 이를 다음과 같이 표현할 수 있습니다:[1]

$$v_X(z, t) = \mathbb{E}\left[\frac{X_1 - z}{1-t}\eta_t(X_1, z)\right],$$

여기서 $$\eta_t(X_1, z) = \frac{\rho\left(\frac{z-tX_1}{1-t}\mid X_1\right)}{\mathbb{E}[\rho\left(\frac{z-tX_1}{1-t}\mid X_1\right)]}$$[1]

#### 핵심 이론 정리

**정리 3.3 - 한계 보존 성질**:[1]
직선화된 흐름 $$dZ_t = v_X(Z_t, t)dt$$는 모든 시간 $$t \in [0,1]$$에서 한계 법칙을 보존합니다:

$$\text{Law}(Z_t) = \text{Law}(X_t), \quad \forall t \in [0,1]$$

**증명 개요**: 연속 방정식(continuity equation)을 이용하여 $$Z_t$$와 $$X_t$$가 동일한 드리프트 속도장으로부터 생성되므로 동일한 한계 법칙을 가집니다.[1]

**정리 3.5 - 볼록 수송 비용 감소**:[1]
모든 볼록 함수 $$c: \mathbb{R}^d \to \mathbb{R}$$에 대해:

$$\mathbb{E}[c(Z_1 - Z_0)] \leq \mathbb{E}[c(X_1 - X_0)]$$

**증명**: Jensen의 부등식을 이용하면,

$$\mathbb{E}[c(Z_1 - Z_0)] = \mathbb{E}\left[c\left(\int_0^1 v_X(Z_t, t)dt\right)\right] \leq \mathbb{E}\left[\int_0^1 c(v_X(Z_t, t))dt\right]$$

$$\text{Law}(Z_t) = \text{Law}(X_t)$$이고 $$v_X(Z_t, t) = \mathbb{E}[X_1 - X_0 | X_t]$$이므로, 다시 한번 Jensen 부등식을 적용하면 원하는 결과를 얻습니다[1].

**정리 3.7 - 직선화 수렴**: $$k$$번째 직선화된 흐름 $$Z^k$$에 대해, 직선성(straightness) 척도

$$S(Z) = \int_0^1 \mathbb{E}\left[\|(Z_1 - Z_0) - \dot{Z}_t\|^2\right] dt$$

는 다음을 만족합니다:[1]

$$\min_{k \leq K} S(Z^k) = \mathcal{O}(1/K)$$

이는 반복적 직선화가 $$O(1/K)$$ 속도로 직선화를 진행함을 보여줍니다.[1]

#### 비선형 확장 및 기존 방법과의 연결

더 일반적인 보간 과정 $$X_t = \alpha_t X_1 + \beta_t X_0$$ (단, $$\alpha_0 = \beta_1 = 1$$, $$\alpha_1 = \beta_0 = 0$$)을 사용할 수 있습니다:[1]
$$\min_v \int_0^1 \mathbb{E}\left[w_t \|v(X_t, t) - \dot{X}_t\|^2\right] dt$$

**중요한 발견**: 확률 흐름 ODE(PF-ODE)와 DDIM은 모두 이 프레임워크의 특수한 경우로 볼 수 있습니다:[1]

- **VP ODE**: $$\alpha_t = \exp\left(-\frac{1}{4}a(1-t)^2 - \frac{1}{2}b(1-t)\right)$$, $$\beta_t = \sqrt{1 - \alpha_t^2}$$
- **Rectified Flow**: $$\alpha_t = t$$, $$\beta_t = 1-t$$ (직선 경로)

Rectified Flow가 우수한 이유:[1]
1. **직선 경로**: $$\alpha_t = t, \beta_t = 1-t$$로 인해 진정한 직선 궤적 생성
2. **균등 속도**: VP ODE와 달리 시간에 따라 균일한 속도 진행
3. **초기 분포 자유도**: 초기 분포 $$\pi_0$$을 임의로 선택 가능

### 4. 성능 향상 분석

#### 실험 결과 1: CIFAR-10 비조건부 이미지 생성[1]

| 방법 | NFE | FID ↓ | Recall ↑ | 비고 |
|------|-----|-------|----------|------|
| 1-Rectified Flow | 127 | 2.58 | 0.57 | 최고 성능 |
| 2-Rectified Flow | 110 | 3.36 | 0.54 | 함수 평가 감소 |
| VP ODE | 140 | 3.93 | 0.51 | - |
| sub-VP ODE | 146 | 3.16 | 0.55 | - |
| VP SDE | 2000 | 2.55 | 0.58 | 높은 계산 비용 |

**단계별 성능** (Figure 8a):[1]
- **1-Step (Distilled)**: FID 4.85, Recall 0.51 (기존 ODE/GAN 최고 성능)
- **2-Step**: FID 12.21 (Reflow 후)
- **3-Step**: FID 8.15

직선화 효과 검증 (Figure 9):[1]
- Reflow 후 straightness measure가 현저히 감소
- 학습 단계 증가에 따라 FID와 Recall 지속적 개선

#### 실험 결과 2: 이미지-이미지 변환[1]

논문은 고양이 ↔ 야생동물, MetFace ↔ CelebA 등의 도메인 간 전환을 시연합니다. 주요 특징:[1]
- N=1 (단일 스텝): 양호한 결과 달성
- N=100: 고품질 다양한 이미지 생성
- 이전 diffusion 기반 방법 대비 추론 속도 대폭 향상

#### 실험 결과 3: 도메인 적응[1]

| 데이터셋 | 기존 최고 | Rectified Flow |
|---------|---------|----------------|
| OfficeHome | 68.7% ± 0.3% | **69.2% ± 0.5%** |
| DomainNet | 41.5% ± 0.2% | 41.4% ± 0.1% |

도메인 표현 공간에서 Rectified Flow를 적용하여 도메인 시프트 완화.[1]

### 5. 일반화 성능 향상 가능성

#### 이론적 보장

**Straightness와 샘플링 오류의 관계**:[1]
직선 경로는 이산화 오류를 최소화합니다. 경로가 완벽하게 직선일 때:
$$Z_1 = Z_0 + v(Z_0, 0) \times 1$$
로 정확히 계산 가능합니다.[1]

**Reflow의 수렴성**: 정리 3.7에서 $$k$$ 번 Reflow 수행 시, 직선성이 $$O(1/k)$$ 속도로 개선되므로, 충분한 Reflow 후 단계별 오류가 거의 무시할 수 있는 수준이 됩니다.[1]

#### 일반화 개선 메커니즘

1. **Deterministic Coupling**: Rectified Flow는 확정적 결합(deterministic coupling)을 학습하므로, 노이즈 기반 방법보다 데이터 간 더 명확한 대응을 학습[1]

2. **모든 Convex Cost에 대한 Pareto 개선**: 특정 비용 함수에만 최적화되지 않고, 모든 볼록 비용에 대해 동시에 개선되므로 다양한 작업에 적응 가능[1]

3. **Flow Crossing 제거**: 선형 보간의 교차점을 제거함으로써 더 간결한 데이터 매니폴드 구조를 학습[1]

### 6. 한계 및 제약사항

#### 이론적 한계

1. **고차원에서의 최적성 부재**:[1]
   - 1차원에서는 직선 결합이 모든 볼록 비용에 대해 동시 최적
   - 고차원($$d \geq 2$$)에서는 특정 비용 함수에 대해 최적이 아닐 수 있음
   - 따라서 특정 작업에 맞춘 수정 필요

2. **Reflow의 오류 누적**:[1]
   - 과도한 Reflow(예: k>3)는 속도장 추정 오류가 누적될 수 있음
   - 실제로는 1-2회 Reflow 권장

3. **조건부 밀도 요구**:[1]
   - 속도장이 잘 정의되려면 조건부 밀도 $$\rho(x_0|x_1)$$이 존재해야 함
   - 조건부 밀도가 불연속인 경우, 가우시안 노이즈 추가 필요

#### 실용적 한계

1. **저차원 비모수 추정의 한계**:[1]
   - 커널 기반 추정(식 5): 대역폭 선택이 중요하며, 계산 복잡도 O(n²)
   - 실제로는 신경망 사용하되, 스무싱 정규화(L2 페널티) 필요

2. **특정 작업에서의 최적성 미확보**:[1]
   - 이차 비용 최적 수송을 위해서는 속도장을 gradient field로 제약해야 함
   - 회전 성분 제거로 인한 추가 복잡성

3. **추론 단계 감소의 한계**:[1]
   - 매우 큰 단계 크기(N<5)에서는 여전히 정확도 저하
   - Distillation 시에만 진정한 1-step 모델 달성

### 7. 앞으로의 연구 영향 및 고려사항

#### 최신 연구 기반 영향 분석 (2024-2025)

**1. Rectified Flow의 산업 적용 확대**[2][3]

최신 텍스트-이미지 생성 모델들이 Rectified Flow를 채택하고 있습니다. 예를 들어:[3]
- **FLUX 모델**: Rectified Flow 기반 Diffusion Transformer로 고해상도 이미지 생성에서 우수한 성능
- **OpenSora**: Rectified Flow와 Transformer 결합으로 비디오 생성 혁신
- 확산 모델 대비 **수십 배 빠른 추론 속도** 달성[3]

**2. 계층적 구조 확장 연구**[4]

2025년 3월 발표된 "Towards Hierarchical Rectified Flow" 논문에서:[4]
- 다중 ODE를 계층적으로 결합하는 구조 제시
- 교차하는 경로를 허용하여 고전적 Rectified Flow보다 **더 직선적인 궤적** 달성
- 신경 함수 평가(NFE) 감소로 추론 효율성 극대화

**3. Flow Matching과의 통합**[5]

"Diff2Flow" 프레임워크 (2025년 6월):[5]
- 사전 학습된 Diffusion 모델의 지식을 Flow Matching으로 효율적 전이
- 타임스텝 재조정, 보간 정렬, 속도장 변환을 통한 seamless 변환
- PEFT(Parameter-Efficient Finetuning) 환경에서도 높은 성능 유지
- **결론**: Rectified Flow 기반 메서드가 기존 Diffusion 생태계 통합의 핵심

**4. 시계열 생성 분야 확장**[2]

"FlowTS: Time Series Generation via Rectified Flow" (2025년 2월):[2]
- Rectified Flow의 직선 경로 특성을 시계열에 적용
- 고차원 시계열에서 확산 모델 대비 **효율성 향상**
- 적응형 배치 크기 조정으로 가변 길이 시계열 처리 개선

**5. 역변환 및 편집 기술 개발**[6]

"Taming Rectified Flow for Inversion and Editing" (2024년 11월):[6]
- RF-Solver: 고차 테일러 전개를 이용한 정확한 ODE 역변환
- FLUX, OpenSora 같은 최신 모델의 편집 능력 향상
- ODE 구간 내 정확한 위치 제어 가능

**6. 이론적 수렴성 분석 강화**[7]

"2-Rectifications are Enough for Straight Flows" (2025년 2월):[7]
- Wasserstein 거리 기반 수렴성 증명
- **2번의 reflow만으로도 충분히 직선적 흐름 달성**
- 계산 비용 대비 성능 최적 지점 제시

#### 앞으로 연구 시 고려사항

**1. 고차원 최적 수송과의 연계**

직선 결합이 특정 비용(특히 이차 비용)에 대해 최적이 아니라는 한계를 극복하기 위해:
- 제약된 최적화(gradient field 제약) 연구 진행 중
- Multi-scale optimal transport와의 통합 모색 필요

**2. 조건부 생성(Conditional Generation) 확장**

현재는 비조건부 생성에 중심이지만:
- 클래스/텍스트 조건 통합의 효율적 방법 개발 필요
- Cross-attention 메커니즘과의 최적 결합 전략 연구 필요

**3. Reflow 최적성 판정**

언제 Reflow를 멈춰야 하는지 자동 판정:
- Straightness 메트릭과 실제 성능 간의 정량적 관계 규명 필요
- Adaptive reflow 전략 개발 진행 중[8]

**4. 도메인 특화 응용**

각 도메인에 맞는 보간(interpolation) 전략:
- 의료 영상, 분자 구조 등에서 비유클리드 기하학 적용
- Manifold 기반 Rectified Flow 개발 필요

**5. 신경망 설계 최적화**

보다 효율적인 속도장 매개변수화:
- 구조화된 정규화 기법(현재는 L2 페널티만 사용)
- 계층적 구조와의 결합으로 추가 효율성 달성 가능[4]

**6. 다중 작업 통합 학습**

생성 + 전환 + 적응을 통일된 프레임워크에서:
- 다중 분포 쌍에 대한 공유 속도장 학습
- 메타 러닝 관점의 Reflow 적응 전략

***

Rectified Flow는 생성 모델 분야에서 **단순성, 이론적 우수성, 실무 효율성**의 완벽한 결합을 제시합니다. 직선 경로 학습이라는 명확한 목표와 $$O(1/K)$$ 수렴 보장은 이론 연구자들에게 매력적이며, 단일 스텝에 가까운 추론 속도는 산업 응용에 획기적입니다. 최근 2024-2025년 연구 동향을 보면, Rectified Flow는 FLUX, OpenSora 같은 최첨단 모델의 백본 기술로 자리 잡았으며, Flow Matching 통합, 시계열 확장, 계층적 개선 등 지속적인 혁신이 진행 중입니다. 향후 연구는 고차원 최적성, 조건부 생성 효율화, 도메인 특화 적용에 집중될 것으로으로 예상됩니다.[5][12][3][4][1]

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/12a7958c-e568-4cc6-8533-ed7b97a6aedb/2209.03003v1.pdf)
[2](https://arxiv.org/html/2411.07506)
[3](http://arxiv.org/pdf/2403.03206.pdf)
[4](https://arxiv.org/html/2502.17436v2)
[5](https://www.themoonlight.io/ko/review/diff2flow-training-flow-matching-models-via-diffusion-model-alignment)
[6](https://arxiv.org/html/2411.04746)
[7](http://arxiv.org/pdf/2410.14949.pdf)
[8](http://arxiv.org/pdf/2405.20320v2.pdf)
[9](http://arxiv.org/pdf/2209.03003v1.pdf)
[10](https://arxiv.org/html/2406.03293)
[11](https://mvje.tistory.com/289)
[12](https://modulabs.co.kr/labs/265)
[13](https://kimjy99.github.io/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0/rectified-flow/)
[14](https://blog.outta.ai/177)
[15](https://dmqa.korea.ac.kr/activity/seminar/486)
[16](https://ostin.tistory.com/225)
[17](https://stibee.com/api/v1.0/emails/share/nwWQtd_IFF_-LGnpU9_Oi_au5ACPsNM)
[18](https://bayesian-bacteria.tistory.com/4)
[19](https://kimjy99.github.io/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0/stable-diffusion-3/)
