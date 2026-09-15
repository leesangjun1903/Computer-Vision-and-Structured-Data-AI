# One-step Latent-free Image Generation with Pixel Mean Flows

> **논문 정보**
> - **제목**: One-step Latent-free Image Generation with Pixel Mean Flows
> - **저자**: Yiyang Lu, Susie Lu, Qiao Sun, Hanhong Zhao, Zhicheng Jiang, Xianbang Wang, Tianhong Li, Zhengyang Geng, Kaiming He (MIT)
> - **arXiv**: [arXiv:2601.22158](https://arxiv.org/abs/2601.22158) (2026년 1월 29일)
> - **코드**: [GitHub – Lyy-iiis/pMF](https://github.com/Lyy-iiis/pMF)

---

## 1. Executive Summary (10문장 이내)

현대 diffusion/flow 기반 이미지 생성 모델은 두 가지 핵심 특성을 공유한다: ① 다중 단계 샘플링(multi-step sampling), ② 잠재 공간(latent space) 내에서의 동작.  
최근의 연구들은 각각의 한계를 개별적으로 극복하는 데 괄목할 진전을 이루어, 잠재 공간 없이 단일 단계로 생성 가능한 방향을 열었다.  
이 논문은 그 목표를 향한 한 걸음 더 나아가 **"pixel MeanFlow" (pMF)**를 제안하며, 핵심 원칙으로 네트워크의 출력 공간과 손실(loss) 공간을 분리하여 설계한다.  
네트워크의 예측 목표(target)는 저차원 이미지 매니폴드(manifold) 위에 놓이도록 설계되며($x$-prediction), 손실은 속도(velocity) 공간에서 MeanFlow를 통해 정의된다.  
이미지 매니폴드와 평균 속도 필드 사이의 간단한 변환을 도입한다.  
MIT 연구진이 개발한 pMF는 ImageNet 256×256에서 FID 2.22를 달성하였고, 1024×1024로의 확장 가능성도 입증했다.  
pMF는 ODE 기반 flow matching의 MeanFlow 공식을 활용하여 픽셀 공간에서 직접 단일 신경망 평가로 가우시안 노이즈를 이미지로 매핑하는 프레임워크이다.  
저자들은 고해상도에서 $x$-prediction이 결정적이며, 지각적 손실(perceptual loss)을 추가하여 품질을 더욱 향상시켰고, 옵티마이저·시간 샘플링·고해상도 확장에 대한 상세한 ablation 실험을 제공했다.  
pMF는 단일 신경망이 노이즈에서 고품질 이미지 픽셀로 직접 매핑하는 완전한 end-to-end 생성 모델링을 향한 실질적 진전을 나타낸다.  
저자들은 이 연구가 diffusion/flow 기반 생성 모델의 경계를 더욱 확장하길 희망한다.

---

### 1-1. 연구의 목적과 필요성

잠재 확산 모델(Latent Diffusion Models)은 고품질 이미지 생성에서 뛰어난 성능을 보이지만, end-to-end 모델링의 이점을 잃는다. 이들은 이미지 인코딩 과정에서 정보를 버리고, 별도로 훈련된 디코더가 필요하며, 원시 데이터(raw data)와는 다른 보조 분포를 모델링하게 된다.

현대의 이미지 생성 모델들은 전형적으로 다중 단계 샘플링과 잠재 공간 내 동작에 의존하며, 이는 복잡성과 계산 오버헤드를 가중시킨다. pMF는 이 두 가지 병목을 동시에 제거함으로써, **단일 순전파(single forward pass)**만으로 픽셀 수준의 고품질 이미지를 생성하는 것을 목표로 한다.

> 📌 **용어 설명**
> - **Latent Space (잠재 공간)**: 원본 이미지를 압축한 저차원의 은닉 표현 공간. VAE(Variational Autoencoder) 등을 이용해 인코딩/디코딩.
> - **Multi-step Sampling**: 노이즈에서 이미지를 복원하기 위해 수십~수백 번의 반복적 역방향 계산을 수행하는 방식 (DDPM의 경우 1000 step).
> - **End-to-End Modeling**: 중간 단계(인코더·디코더·반복 샘플링) 없이 입력에서 출력을 직접 학습하는 방식.

---

## 2. 핵심 주장과 근거 표

| 구분 | 핵심 주장 | 근거/실험 결과 | 위치 |
|------|-----------|----------------|------|
| **문제 제기** | 기존 모델은 다중 단계 샘플링 + 잠재 공간 의존으로 비효율적 | 현존 최고 모델(LDM, DiT 등)의 구조적 분석 | Abstract, Sec.1 |
| **핵심 설계** | 출력 공간(image manifold)과 손실 공간(velocity space)의 분리 | $x$-prediction → $u_\theta$ 변환 구조 | Sec.3 |
| **매니폴드 가설** | $x_\theta$는 저차원 이미지 매니폴드에 근사하여 학습이 용이 | toy experiment 및 ablation (Table 2) | Sec.4, Fig.2 |
| **지각적 손실** | 픽셀 공간에서 LPIPS 손실이 FID를 9.56→3.53으로 대폭 개선 | Ablation Table 3 | Sec.5.3 |
| **옵티마이저** | Muon 옵티마이저가 Adam 대비 수렴 속도·FID 모두 우수 | Ablation (Table 4) | Sec.5.4 |
| **시간 샘플링** | 전체 $(r,t)$ 좌표평면 샘플링이 필수, $r=t$만으로는 성능 붕괴 | Ablation (Table 2) | Sec.5.2 |
| **확장성** | 모델 크기·훈련 에폭 증가 모두에서 성능 향상 | Table 5 (B→L→H 모델) | Sec.6.2 |
| **최종 성능** | ImageNet 256×256: FID 2.22, 512×512: FID 2.48, 1024×1024: FID 4.58 | Table 6, 7 | Sec.6.3 |
| **비교 우위** | 동일 카테고리(one-step, latent-free) 내 유일한 경쟁 모델(EPG, FID 8.82) 대비 압도적 우위 | Table 6 | Sec.6.3 |
| **계산 효율** | StyleGAN-XL 대비 FLOPs 5.8배 절감하면서 유사한 FID 달성 | Table 6 | Sec.6.3 |

---

### 2-1. 상세 설명

#### ① 해결하고자 하는 문제

Pixel MeanFlow는 기존 diffusion 및 flow 기반 모델의 두 가지 주요 한계를 해결한다: 다중 단계 샘플링 알고리즘에 대한 의존성, 그리고 픽셀이 아닌 잠재 공간에서의 동작. 방법론의 핵심 목적은 반복적인 절차 없이, 잠재 변수 표현을 위한 디코더 의존 없이 표준 가우시안 노이즈로부터 이미지를 생성하는 모델을 개발하는 것이다.

#### ② 제안하는 방법 (수식 포함)

pMF의 핵심 아이디어는 신경망 $\text{net}_\theta$가 "노이즈 제거된 이미지 유사 수량(denoised image-like quantity)"을 직접 출력하도록 설계하면서, 훈련 손실은 MeanFlow 프레임워크를 사용하여 속도 공간에서 정의하는 것이다. pMF는 새로운 필드 $x(z_t, r, t)$를 평균 속도 필드 $u$로부터 도출한다:

$$x(z_t, r, t) \triangleq z_t - t \cdot u(z_t, r, t)$$

> **기호 설명**
> - $z_t$: 시각 $t$에서의 노이즈가 섞인 중간 상태 (noisy intermediate state)
> - $t \in [0, 1]$: 시간 변수 ($t=1$: 순수 노이즈, $t=0$: 깨끗한 이미지)
> - $r \in [0, t]$: 평균 속도 구간을 정의하는 보조 시간 변수 (auxiliary time)
> - $u(z_t, r, t)$: $[r, t]$ 구간에서의 **평균 속도 필드(average velocity field)**
> - $x(z_t, r, t)$: 네트워크가 예측할 타겟인 **이미지 유사 필드(image-like field)**

이에 따라 네트워크 출력 $x_\theta$로부터 평균 속도 $u_\theta$를 역산한다:

$$u_\theta(z_t, r, t) = \frac{1}{t}\left(z_t - x_\theta(z_t, r, t)\right)$$

> **기호 설명**
> - $x_\theta(z_t, r, t)$: 네트워크가 직접 출력하는 이미지 유사 예측값 ($x$-prediction)
> - $u_\theta$: $x_\theta$로부터 유도된 평균 속도 추정값 → MeanFlow 손실 계산에 사용

이 $x(z_t, r, t)$ 필드가 학습이 용이한 이유는, 평균 속도 필드 $u$가 고차원적이고 노이즈가 많은 반면, $x$는 저차원 이미지 매니폴드 위에 근사하여 위치하기 때문이라고 논문은 주장한다.

특히 $r=t$일 때, $u(z_t, t, t) = v(z_t, t)$ (순간 속도)가 되어, $x(z_t, t, t) = z_t - t \cdot v(z_t, t)$가 되며, 이는 알려진 x-prediction 기반의 노이즈 제거 이미지 타겟(JiT의 타겟)과 정확히 일치한다.

**MeanFlow 손실(iMF 기반 v-loss):**

$$\mathcal{L}_{v} = \mathbb{E}_{z_t, r, t}\left[\left\| u_\theta(z_t, r, t) - u^*(z_t, r, t) \right\|^2\right]$$

> **기호 설명**
> - $u^*(z_t, r, t)$: Ground-truth 평균 속도 (iMF 공식에 의해 정의)
> - $\mathbb{E}$: 시간 변수 $(r, t)$ 및 상태 $z_t$에 대한 기댓값

**지각적 손실(Perceptual Loss) 통합:**

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{v} + \lambda \cdot \mathcal{L}_{\text{LPIPS}}(x_\theta, x_0)$$

> **기호 설명**
> - $\mathcal{L}_{\text{LPIPS}}$: LPIPS(Learned Perceptual Image Patch Similarity) 손실 — 인간의 지각적 유사도를 모사하는 손실
> - $x_0$: Ground-truth 원본 이미지 (clean image)
> - $\lambda$: LPIPS 손실의 가중치 하이퍼파라미터
> - 이 통합이 가능한 이유: $x_\theta$가 **픽셀 공간의 이미지를 직접 예측**하므로, 이미지-공간 손실 적용이 자연스럽게 가능

> 📌 **용어 설명**
> - **MeanFlow**: 단일 단계 샘플링을 가능하게 하는 flow matching 방식. 순간 속도(instantaneous velocity) 대신 구간 $[r, t]$의 **평균 속도(average velocity)**를 학습 목표로 삼는다.
> - **x-prediction**: 노이즈가 섞인 입력 $z_t$로부터 깨끗한 이미지 $x_0$를 직접 예측하는 parameterization 방식 (v-prediction, $\epsilon$-prediction의 대안).
> - **ODE (Ordinary Differential Equation)**: 확률 미분방정식(SDE) 없이 결정론적(deterministic)으로 노이즈-이미지 경로를 정의하는 방정식. 단일 step 생성에 유리.
> - **LPIPS (Learned Perceptual Image Patch Similarity)**: VGG, AlexNet 등 사전 훈련된 CNN 특징 공간에서 두 이미지 간 거리를 측정하는 지각적 유사도 지표.

#### ③ 모델 구조

pMF는 대규모 패치(large-patch)를 사용하는 Vision Transformer(ViT) 구조를 채택하며, ConvNet 기반의 GAN 방식보다 FLOPs 효율이 높다. 예를 들어, StyleGAN-XL은 순전파 한 번에 1574 GFLOPs가 필요하여 pMF-H/16보다 5.8배 많은 연산량을 소비한다.

모델 스케일(Model Scale) 세부 사항 (ImageNet 256×256, 1-NFE FID):

| 모델 | Depth | Width | 파라미터 수 | GFLOPs | FID (160ep) | FID (320ep) |
|------|-------|-------|------------|--------|------------|------------|
| pMF-B/16 | 16 | 768 | 119M | 34 | 3.53 | 3.12 |
| pMF-L/16 | 32 | 1024 | 411M | 117 | 2.85 | 2.52 |
| pMF-H/16 | 48 | 1280 | 956M | 271 | 2.57 | 2.29 |

고해상도 생성 시 pMF는 패치 크기를 공격적으로 증가시키는 방법으로 효율적인 계산 비용을 유지하면서 512×512, 심지어 1024×1024까지 효과적으로 확장된다.

> 📌 **용어 설명**
> - **ViT (Vision Transformer)**: 이미지를 패치(patch)로 분할하고 Transformer 구조로 처리하는 모델. 이미지 생성에서 DiT 등의 형태로 응용됨.
> - **NFE (Number of Function Evaluations)**: 모델 순전파 호출 횟수. 1-NFE = 단 1회 순전파로 이미지 생성.
> - **GFLOPs**: 초당 10억 부동소수점 연산 수. 모델의 계산 복잡도를 나타내는 지표.

#### ④ 성능 향상 및 한계

**성능 향상:**

pMF는 ImageNet 256×256에서 FID 2.22를 달성하여, 기존 one-step 픽셀 공간 방법(EPG-L/16의 FID 8.82)을 크게 능가하고 선도적인 GAN(StyleGAN-XL FID 2.30)과 경쟁력 있는 수치를 달성했다.

지각적 손실의 효과: LPIPS(ConvNeXt-V2 백본 사용)를 적용하면 FID가 9.56에서 3.53으로 극적으로 개선된다.

옵티마이저 선택: Muon 옵티마이저는 단일 단계 생성에서 Adam 대비 빠른 수렴과 실질적으로 더 나은 FID를 제공한다.

**한계:**

512×512 같은 매우 높은 해상도에서는 패치 크기를 공격적으로 확대해야 하며(예: 32×32), 성능은 우수하지만 여전히 다중 단계 잠재 모델(최고 ~1.1 FID)에는 미치지 못한다.

one-step이면서 latent-free인 기존 방법이 거의 없어 직접 비교 가능한 baseline이 매우 제한적이며, distillation 없이 처음부터 학습한 방법들만 비교 대상으로 고려된다.

> 📌 **용어 설명**
> - **FID (Fréchet Inception Distance)**: 생성 이미지와 실제 이미지 사이의 통계적 거리. 낮을수록 고품질.
> - **Muon 옵티마이저**: Momentum + Newton 방법을 결합한 최근 등장한 2차 최적화 기법. 고차원 딥러닝 훈련에서 Adam 대비 안정성과 수렴 속도를 개선.
> - **Distillation**: 대형 사전 학습 모델의 지식을 소형 모델 또는 단계 수가 적은 모델로 전이하는 기법. pMF는 이를 사용하지 않고 scratch부터 학습.

---

## 3. 각 주장에 페이지/Figure/Table 번호 표시

| 주장 | 위치 |
|------|------|
| 기존 모델의 두 가지 한계 (multi-step + latent) | Abstract, p.1 / Sec.1 Introduction |
| 출력 공간/손실 공간 분리 핵심 원칙 | Sec.3 (Method), p.3-4 |
| $x(z_t, r, t) \triangleq z_t - t \cdot u(z_t, r, t)$ 변환 | Sec.3, Eq.(주요 수식) |
| 매니폴드 가설 검증 (toy experiment) | Sec.4 / Fig.2 |
| 지각적 손실 도입 효과 | Sec.5.3 / **Table 3** (FID: 9.56 → 3.53) |
| 시간 샘플링 전략 필요성 | Sec.5.2 / **Table 2** |
| Muon 옵티마이저 우위 | Sec.5.4 / **Table 4** |
| 모델 확장성 검증 | Sec.6.2 / **Table 5**, **Fig.4** |
| 256×256 ImageNet 최종 비교 | Sec.6.3 / **Table 6** (FID 2.22) |
| 512×512 ImageNet 최종 비교 | Sec.6.3 / **Table 7** (FID 2.48) |
| 1024×1024 확장 시연 | Appendix B / **Fig.** (FID 4.58) |
| StyleGAN-XL 대비 FLOPs 효율 비교 | Sec.6.3 / **Table 6** (5.8× 절감) |

---

## 4. 연구 주제, 방법, 결과: 저자 보고 vs. 나의 해석 분리

### 4-A. 저자가 직접 보고한 결과

저자들은 현대 diffusion/flow 기반 모델의 두 핵심 특성인 다중 단계 샘플링과 잠재 공간 동작을 지적하고, 이를 동시에 제거하는 pMF를 제안한다. 네트워크 예측 목표는 저차원 이미지 매니폴드($x$-prediction), 손실은 속도 공간의 MeanFlow로 정의하며, 이미지 매니폴드와 평균 속도 필드 사이의 단순 변환을 도입한다. 실험에서 pMF는 ImageNet 256×256에서 FID 2.22, 512×512에서 FID 2.48을 달성한다.

측정 결과, 픽셀 공간에서 속도 필드를 직접 회귀(regression)하면 성능이 파국적으로 저하되어, 제안된 $x$-prediction 접근법의 중요성이 확인된다.

### 4-B. 나의 해석 (검토자 시각)

pMF의 핵심 기여는 단순히 "잠재 공간을 제거"한 데 있지 않고, **예측 공간(image manifold)과 감독 공간(velocity space)의 분리**라는 설계 원칙에 있다. 이는 픽셀 공간의 고차원성 문제를 우회하는 영리한 해법이다. 그러나 이 분리가 왜 정확히 작동하는지에 대한 이론적 보장은 "일반화된 매니폴드 가설(generalized manifold hypothesis)"이라는 경험적 직관에 머무르고 있어, 엄밀한 수학적 증명이 결여되어 있다. 또한 FID만을 주요 지표로 사용하여 다양성(diversity), 텍스트 정렬(text alignment), 사람의 지각적 평가 등에 대한 검토는 제한적이다.

---

## 5. 통계적으로 취약한 부분 및 비교 불가능한 수치

> ⚠️ 표시: **[통계 취약]** = 통계적 신뢰도 문제, **[비교 불가]** = 공정한 비교 어려움

| 항목 | 내용 | 위치 |
|------|------|------|
| **[비교 불가]** 단일 카테고리 | one-step, latent-free diffusion/flow 카테고리에서 유일한 경쟁 모델은 EPG(FID 8.82)뿐이어서, 직접 비교 가능한 동등한 방법이 거의 없다. | Table 6 |
| **[비교 불가]** Multi-step 모델과의 비교 | few-step 및 latent 기반 방법과의 비교는 카테고리가 달라 공정한 비교라 볼 수 없으며, 참조 목적으로만 제시된다. | Table 6, 7 |
| **[통계 취약]** FID 수치의 계산 환경 의존성 | TPU/GPU, JAX/PyTorch 등 계산 환경에 따라 FID/IS 수치에 미세한 차이가 발생할 수 있으며, 논문의 결과는 TPU v5p-64에서 계산된 것이다. | GitHub 주석 |
| **[통계 취약]** 단일 데이터셋 검증 | 모든 실험이 ImageNet에 집중되어 있어, 다른 도메인(의료, 위성 등)에서의 일반화 성능 검증 부재 | Sec.6 전반 |
| **[비교 불가]** Distillation 방법 제외 | distillation을 활용한 방법들은 비교에서 제외하여, Consistency Distillation, SDXL-Lightning 등 강력한 one-step 방법과의 비교가 없다. | Sec.6.3 |
| **[통계 취약]** IS(Inception Score) 미보고 | FID만 주요 지표로 사용, CLIP score나 인간 평가(human evaluation) 결과 부재 | Table 6, 7 |

---

## 6. 논문이 답하지 않는 질문

| # | 미해결 질문 |
|---|-----------|
| 1 | $x$-prediction이 픽셀 공간에서 저차원 매니폴드에 근사한다는 **일반화된 매니폴드 가설**의 엄밀한 수학적 증명은? |
| 2 | 텍스트 조건부(text-conditional) 생성에서도 pMF가 동일한 성능을 발휘하는가? |
| 3 | ImageNet 이외의 데이터셋(COCO, LAION 등)에서의 성능은? |
| 4 | Distillation 방법(SDXL-Lightning, LCM 등)과의 공정한 비교는? |
| 5 | 1024×1024에서 FID 4.58이 보고되었으나, 이에 대한 상세한 ablation 및 구조 분석은 제공되는가? |
| 6 | 최적 패치 크기($p=32$) 선택의 이론적 근거와 자동 탐색 방법은? |
| 7 | Muon 옵티마이저가 왜 Muon에서 특히 유리한지에 대한 이론적 분석은? |
| 8 | 생성된 이미지의 다양성(diversity) 지표(Recall, Coverage 등)는 어느 수준인가? |
| 9 | 모델이 훈련 이미지를 기억(memorization)하는 문제는 없는가? |
| 10 | 비디오 생성, 3D 생성 등 다른 모달리티로의 확장 가능성은? |

---

## 7. 가장 중요한 그림 5개 해석

> ⚠️ 논문 내 Figure에 대한 직접 접근이 불가하여, 검색된 정보를 기반으로 내용 해석을 제공합니다. Figure 번호는 논문 원문 기준입니다.

### **Fig. 1 — 방법론 개요도 (Method Overview)**

이 그림은 pMF의 핵심 설계 원칙을 시각화한다. 신경망이 노이즈 입력으로부터 노이즈 제거된 이미지 유사 필드를 예측하도록 훈련되며, 출력은 이미지 매니폴드 위 또는 근방에 위치한다. 이 분리가 핵심이다: 네트워크 예측은 데이터 매니폴드를 타겟으로 하고, 감독(supervision)은 물리적으로 의미있는 속도 공간에서 이루어진다. 한 장의 다이어그램으로 출력 공간($x$-manifold)과 손실 공간(velocity space)이 어떻게 변환을 통해 연결되는지를 직관적으로 보여주는 것으로 해석된다.

---

### **Fig. 2 — Toy Experiment (매니폴드 가설 검증)**

$x(z_t, r, t)$로 유도된 필드는 평균 속도 필드 $u$가 고차원적이고 노이즈가 많은 것과 달리, 저차원 이미지 매니폴드에 근사하여 위치하기 때문에 신경망이 학습하기 더 용이하다. 이 그림은 저차원 합성 데이터(toy data)에서 $u$-prediction과 $x$-prediction의 학습 난이도 차이를 시각적으로 보여줌으로써, pMF 설계의 근본적 동기를 실험적으로 지지한다.

---

### **Fig. 3 — 학습 곡선 비교 (Training Curve: Optimizer Ablation)**

Muon 옵티마이저가 Adam 대비 단일 단계 생성에서 빠른 수렴과 실질적으로 더 나은 FID를 제공한다. 이 그림은 동일 에폭에서 Muon 사용 시 FID가 지속적으로 낮게 유지됨을 보이며, 옵티마이저 선택이 단순한 하이퍼파라미터 차원이 아닌 핵심 설계 요소임을 보여준다.

---

### **Fig. 4 — 생성 이미지 품질 시각화 (Qualitative Samples)**

pMF는 모델 크기와 훈련 에폭 모두에서 확장에 따른 성능 향상을 보인다. 정성적 예제가 Fig.4와 Appendix B에서 제공된다. 단일 순전파로 생성된 256×256 및 512×512 이미지들이 세밀한 텍스처와 구조적 일관성을 갖추고 있음을 시각적으로 확인할 수 있으며, end-to-end 픽셀 공간 생성의 품질 수준을 직접 증명한다.

---

### **Fig. 5 — 해상도 확장 시연 (1024×1024 Generation)**

pMF는 높은 해상도에서의 강한 확장성을 보여주며, 1024×1024에서 FID 4.58을 달성하면서 효율적인 계산을 유지한다. 이 그림은 픽셀 공간에서 직접 초고해상도 이미지를 생성하는 가능성을 증명하며, 단순히 패치 크기 증가라는 간단한 전략만으로도 고해상도 확장이 가능함을 보여준다. 단, 1024 해상도에서 세밀한 품질은 잠재 기반 모델 대비 열위임을 함께 시사한다.

---

## 8. 결론: 시사점, 후속 연구, 추가 방향

### 8-A. 저자가 제시한 시사점 및 후속 연구 계획

pMF는 256×256에서 FID 2.22, 512×512에서 FID 2.48을 달성하여 기존 one-step latent-free 방법 및 다수의 multi-step/latent 모델과 경쟁하면서 훨씬 낮은 계산 오버헤드를 보인다. 이 논문은 pMF가 단일 신경망이 노이즈에서 고충실도 이미지 픽셀로 직접 매핑하는 진정한 end-to-end 생성 모델링을 향한 실질적 진전을 나타낸다고 주장한다.

향후 연구로는 학습된 연속 시간 샘플러(learned continuous-time sampler), 더 효율적인 매니폴드 파라미터화, 적대적/지각적 하이브리드 훈련, 또는 경량 잠재 디코더 등을 통해 FID를 2.0 미만으로 낮추는 방향이 제안된다.

### 8-1. 모델의 일반화 성능 향상 가능성

현재 pMF의 일반화 한계와 개선 방향을 구체적으로 분석하면:

| 일반화 차원 | 현재 상태 | 개선 방향 |
|------------|----------|----------|
| **도메인** | ImageNet 클래스 조건부 생성만 검증 | 텍스트 조건부, 의료 이미지, 위성 데이터 등으로 확장 |
| **해상도** | 256/512/1024 검증 (고해상도에서 FID 열화) | 패치 크기 자동 탐색, 다중 스케일 학습 |
| **모달리티** | 정지 이미지만 | 비디오, 3D, 음성 스펙트로그램 등 |
| **조건부 신호** | 클래스 레이블 | 텍스트(CLIP), 깊이맵, 스케치 등 다양한 조건 |
| **도메인 적응** | 없음 | LoRA, 파인튜닝 기반 적응 |

특히 저자들이 고해상도에서 $x$-prediction의 중요성과 perceptual loss의 필요성을 강조하는 점을 감안할 때, 다양한 도메인에서의 일반화를 위해서는 도메인 특화 지각적 손실(domain-specific perceptual loss)의 설계가 핵심 과제가 될 것이다.

---

### 8-2. 2020년 이후 관련 최신 연구 비교 분석

#### 주요 관련 연구 계보

```
[DDPM, 2020] → [LDM/Stable Diffusion, 2022] → [DiT, 2023]
     ↓                    ↓                          ↓
[Flow Matching, 2023] → [Rectified Flow, 2023] → [SiT, 2024]
     ↓
[MeanFlow (NeurIPS 2025)] → [iMF, 2025] + [JiT(x-pred), 2025]
                                    ↓
                            [pMF, Jan 2026]  ←→  [CrossFlow, 2026]
```

#### 핵심 비교 분석

| 모델 | 연도 | 방식 | 단계수 | 잠재 공간 | FID (IN-256) | 특징 |
|------|------|------|--------|----------|-------------|------|
| LDM-4-G | 2022 | Diffusion | multi | ✅ | 3.60 | Stable Diffusion 원조 |
| DiT-XL/2 | 2023 | Diffusion | 250 | ✅ | 2.27 | Transformer 기반 확산 |
| SiT-XL/2 | 2024 | Flow | 250 | ✅ | 2.06 | Stochastic Interpolant |
| StyleGAN-XL | 2022 | GAN | 1 | ❌ | 2.30 | 1-step, pixel, 고FLOPs |
| EPG | 2026 | Flow | 1 | ❌ | 8.82 | 1-step, pixel, self-supervised |
| **pMF (이 논문)** | **2026** | **Flow** | **1** | **❌** | **2.22** | **1-step, pixel, ViT** |
| CrossFlow | 2026 | Flow | 1 | ❌ | 1.62 | 1-step, encoder 활용 |
| PixelFlow | 2025 | Flow | multi | ❌ | 1.98 | multi-step, pixel |

CrossFlow-XL은 FID 1.62를 달성하며 DiT-XL/2(250 step, FID 2.27)과 경쟁적이고, one-step 잠재 공간 방법인 iMF-XL/2(FID 1.72)를 능가하면서도 별도의 디코더 없이 추론이 가능하다. CrossFlow는 별도의 인코더를 활용하는 반면, pMF는 완전히 encoder-free라는 점에서 더 순수한 end-to-end 접근법이다.

#### pMF가 앞으로의 연구에 미치는 영향

1. **End-to-End 패러다임 전환 가속화**: pMF는 VAE 없는 고품질 이미지 생성이 가능함을 실증하여, 잠재 공간 의존성에 대한 재고를 촉구한다.

2. **x-prediction의 재발견**: JiT 논문에서 제안된 x-prediction의 중심 아이디어가 픽셀 공간의 학습 가능성에 결정적이라는 점이 pMF에 의해 실증적으로 확인된다.

3. **MeanFlow 생태계 확장**: pMF는 improved MeanFlow(iMF) 프레임워크에 직접 기반하며, 픽셀 공간에서의 고품질 one-step 이미지 생성을 가능하게 하도록 이를 적응시킨다. 이 생태계는 비디오, 3D 등 다른 도메인으로 빠르게 확산될 것으로 예상된다.

4. **지각적 손실의 재발견**: 픽셀 공간 생성에서 LPIPS 손실의 극적인 효과(FID 9.56→3.53)는 향후 연구에서 지각적 손실을 기본 구성요소로 포함시키는 트렌드를 강화할 것이다.

#### 앞으로 연구 시 고려할 점

| 고려 사항 | 상세 설명 |
|----------|----------|
| **이론적 토대 강화** | 일반화된 매니폴드 가설의 수학적 증명 또는 더 엄밀한 경험적 검증 필요 |
| **다양한 평가 지표 도입** | FID 단일 지표의 한계를 보완하기 위해 IS, Recall, CLIP Score, 인간 평가 포함 권장 |
| **공정 비교 설계** | 학습 데이터, 계산 예산, 평가 환경을 통일한 비교 필요 |
| **텍스트 조건부 확장** | 클래스 조건부를 넘어 텍스트-이미지 생성으로의 확장 검증 필수 |
| **메모리 효율** | 픽셀 공간에서의 대규모 훈련은 잠재 공간 대비 메모리 부담이 크므로 효율화 연구 필요 |
| **고해상도 품질 격차 해소** | 1024×1024에서 잠재 기반 모델 대비 FID 격차가 크므로, 계층적 생성 또는 초해상도 결합 연구 권장 |
| **Distillation 통합** | 처음부터 학습하는 대신, 사전 학습된 멀티스텝 모델로부터의 distillation과 결합하여 성능 상한선 탐색 |

---

## 📚 참고 자료 및 출처

| # | 제목 / 출처 | URL |
|---|------------|-----|
| 1 | **[논문 원문]** One-step Latent-free Image Generation with Pixel Mean Flows | https://arxiv.org/abs/2601.22158 |
| 2 | **[공식 코드]** GitHub – Lyy-iiis/pMF | https://github.com/Lyy-iiis/pMF |
| 3 | **[논문 PDF]** arXiv PDF | https://arxiv.org/pdf/2601.22158 |
| 4 | **[요약]** AlphaXiv Overview | https://www.alphaxiv.org/overview/2601.22158 |
| 5 | **[리뷰]** Moonlight Literature Review | https://www.themoonlight.io/en/review/one-step-latent-free-image-generation-with-pixel-mean-flows |
| 6 | **[해설]** EmergentMind – Pixel MeanFlow | https://www.emergentmind.com/papers/2601.22158 |
| 7 | **[해설]** QuantumZeitgeist 분석 | https://quantumzeitgeist.com/results-pixel-meanflow-achieves-one-step/ |
| 8 | **[HuggingFace 논문 페이지]** | https://huggingface.co/papers/2601.22158 |
| 9 | **[비교 논문]** CrossFlow: One-Step Generation Across Latent and Pixel Spaces | https://arxiv.org/pdf/2606.19970 |
| 10 | **[비교 논문]** PixelFlow: Pixel-Space Generative Models with Flow | https://arxiv.org/pdf/2504.07963 |
| 11 | **[비교 논문]** FREPix: Frequency-Heterogeneous Flow Matching (pMF 인용) | https://arxiv.org/pdf/2605.06421 |
| 12 | **[관련 연구]** Improved Mean Flows (iMF) – arXiv:2512.02012 | arXiv:2512.02012 |
| 13 | **[관련 연구]** Back to Basics: JiT (x-prediction) – arXiv:2511.13720 | arXiv:2511.13720 |
| 14 | **[관련 연구]** Mean Flows (NeurIPS 2025) – arXiv, Geng et al. | NeurIPS 2025 |
| 15 | **[ScienceStack 요약]** | https://www.sciencestack.ai/paper/2601.22158 |

---

> ⚠️ **정확도 주의**: 이 답변은 공개된 검색 결과(arXiv abstract, GitHub README, 제3자 리뷰)를 기반으로 작성되었습니다. 논문의 내부 세부 구조(Figure 세부 내용, 전체 수식 목록 등)는 PDF 전문에 직접 접근하지 않아 일부 내용이 제한적일 수 있습니다. **정확한 수식 및 Figure 해석은 반드시 원문 PDF([arXiv:2601.22158](https://arxiv.org/pdf/2601.22158))를 직접 확인하시길 권장합니다.**
