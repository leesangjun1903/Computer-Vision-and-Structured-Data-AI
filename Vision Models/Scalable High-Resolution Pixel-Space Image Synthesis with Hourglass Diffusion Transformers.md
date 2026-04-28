
# Scalable High-Resolution Pixel-Space Image Synthesis with Hourglass Diffusion Transformers

> **논문 정보**
> - **제목**: Scalable High-Resolution Pixel-Space Image Synthesis with Hourglass Diffusion Transformers
> - **저자**: Katherine Crowson, Stefan Andreas Baumann, Alex Birch, Tanishq Mathew Abraham, Daniel Z. Kaplan, Enrico Shippole
> - **발표**: ICML 2024 (Proceedings of the 41st International Conference on Machine Learning, pp. 9550–9575)
> - **arXiv**: 2401.11605 (2024년 1월 제출)
> - **코드**: https://github.com/crowsonkb/k-diffusion

---

## 1. 핵심 주장 및 주요 기여 요약

HDiT는 픽셀 수에 대해 **선형 스케일링(linear scaling)**을 보이는 이미지 생성 모델로, $1024 \times 1024$ 해상도까지 **픽셀 공간(pixel-space)에서 직접** 훈련을 지원합니다. Transformer 아키텍처를 기반으로 하며, 수십억 개의 파라미터로 확장 가능한 것으로 알려진 Transformer의 확장성과 합성곱 U-Net의 효율성 사이의 간극을 메웁니다.

HDiT는 멀티스케일 아키텍처, 잠재 오토인코더(latent autoencoder), 자기-조건화(self-conditioning)와 같은 일반적인 고해상도 훈련 기법 없이도 성공적으로 훈련됩니다.

### 주요 기여 요약표

| 기여 항목 | 설명 |
|---|---|
| 선형 계산 복잡도 | 픽셀 수에 대해 $O(N)$ 스케일링 달성 |
| 픽셀 직접 합성 | 잠재 인코더 없이 $1024\times1024$ 생성 |
| Hourglass 구조 | U-Net의 계층성 + Transformer 확장성 결합 |
| Soft-Min-SNR | 부드러운 손실 가중 전략으로 FID 향상 |
| FFHQ SOTA | 확산 모델 중 FFHQ- $1024^2$ 에서 최고 성능 달성 |

---

## 2. 해결하고자 하는 문제, 제안 방법(수식 포함), 모델 구조, 성능 및 한계

---

### 2-1. 해결하고자 하는 문제

Transformer 어텐션 메커니즘의 **이차 계산 복잡도(quadratic computational complexity)**가 픽셀 공간에서의 고해상도 합성을 사실상 불가능하게 만들었으며, 이 때문에 잠재 표현(latent representation)이 일반적으로 작동 해상도를 줄이는 데 사용되어 왔습니다.

확산 모델은 이미지 생성에서 탁월하지만, 고해상도 이미지에 적용 시 상당한 훈련 복잡성이 발생하며, 종종 추가 모델이 필요하거나 품질이 저하됩니다. 잠재 확산이나 캐스케이드 초해상도와 같은 현재의 고해상도 합성 방법은 세부 표현 실패로 샘플 품질을 제한합니다.

구체적으로, 기존 접근법들의 문제점은 다음과 같습니다:

고해상도 생성은 여러 단계로 분리하는 방식이 주류였습니다. 캐스케이드 초해상도(Cascaded Super-Resolution)는 초기에 저해상도 이미지를 목표로 한 뒤, 일련의 초해상도 모델을 통해 스케일업합니다. 잠재 확산(Latent Diffusion)은 공간적으로 다운샘플된 잠재 표현을 대상으로 하여, 합성곱 디코더를 통해 고해상도 픽셀 이미지로 디코딩합니다.

이러한 방법들은 **VAE 병목에 의한 세부 정보 손실** 문제가 있습니다. 표준 VAE(Rombach et al., 2022)를 사용할 경우 세부 정보 손실이 발생하며, 이 VAE는 비교 대상인 기준 DiT 아키텍처에서도 사용됩니다.

---

### 2-2. 제안하는 방법 및 수식

#### (A) 확산 모델의 기본 훈련 목표

HDiT는 EDM(Elucidated Diffusion Model, Karras et al., 2022) 프레임워크를 기반으로 하는 **연속 시간 확산 공식**을 사용합니다.

**노이즈 스케줄**: 노이즈 수준 $\sigma(t)$에서 입력 $\mathbf{x}$에 가우시안 노이즈 $\epsilon$을 추가합니다.

$$\mathbf{x}_t = \mathbf{x}_0 + \sigma(t) \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0, \mathbf{I})$$

**디노이징 목표**: 모델 $D_\theta$는 노이즈가 추가된 입력 $\mathbf{x}_t$로부터 원본 $\mathbf{x}_0$을 복원합니다.

$$\mathcal{L} = \mathbb{E}_{\mathbf{x}_0, \epsilon, t} \left[ w(\sigma(t)) \cdot \| D_\theta(\mathbf{x}_t; \sigma(t), \mathbf{c}) - \mathbf{x}_0 \|^2 \right]$$

여기서 $w(\sigma(t))$는 **손실 가중 함수**, $\mathbf{c}$는 조건 신호(클래스 레이블 등)입니다.

#### (B) Soft-Min-SNR 손실 가중 전략

HDiT는 **Soft-Min-SNR 손실 가중 체계**를 사용하며, 이는 Min-SNR의 부드러운 버전으로, 픽셀 공간 확산에서 모델 수렴과 FID 점수를 향상시킵니다.

기존 Min-SNR 가중치는 다음과 같이 정의됩니다:

$$w_{\text{min-SNR}}(\sigma) = \min\left(\text{SNR}(\sigma),\ \gamma\right), \quad \text{SNR}(\sigma) = \frac{1}{\sigma^2}$$

Soft-Min-SNR의 손실 가중치를 Min-SNR과 비교하면 전환 구간에서 훨씬 부드러운 특성을 보이며, ablation 연구에서 이 가중치 체계가 모델의 FID 점수를 향상시킵니다.

**Soft-Min-SNR 가중치**:

$$w_{\text{soft-min-SNR}}(\sigma) = \text{softmin}\left(\text{SNR}(\sigma),\ \gamma\right) = \frac{\text{SNR}(\sigma) \cdot \gamma}{\text{SNR}(\sigma) + \gamma}$$

이는 Min-SNR의 하드 클리핑(hard clipping)을 소프트 전환(soft transition)으로 대체하여 훈련 안정성을 높입니다.

#### (C) 어텐션 메커니즘: Scaled Cosine Attention

HDiT의 어텐션 메커니즘은 Scaled Cosine Attention의 변형을 사용합니다. $\tau$는 학습된 스케일이 훈련 중 크게 변하도록 만들어, 지수화 전에 100의 최댓값으로 클램핑하여 훈련 불안정을 방지합니다.

$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\tau \cdot \frac{\mathbf{Q}\mathbf{K}^T}{\|\mathbf{Q}\| \|\mathbf{K}\|}\right) \mathbf{V}$$

여기서 $\tau$는 학습 가능한 온도 파라미터(temperature parameter)입니다.

#### (D) Rotary Positional Embedding (2D RoPE)

HDiT는 2D 축 방향 Rotary Positional Embedding(RoPE)과 GEGLU를 피드포워드 네트워크에 통합하여, 수렴을 향상시키고 패치 아티팩트를 완화합니다.

2D RoPE는 각 공간 위치 $(x, y)$에 대해 각도 기반 회전 행렬을 적용합니다:

$$\mathbf{q}_{\text{rot}}^{(2i)} = q^{(2i)} \cos\theta_x - q^{(2i+1)} \sin\theta_x$$

$$\mathbf{q}_{\text{rot}}^{(2i+1)} = q^{(2i)} \sin\theta_x + q^{(2i+1)} \cos\theta_x$$

유사하게 $y$ 방향에도 적용하여 2D 위치 정보를 효과적으로 인코딩합니다.

---

### 2-3. 모델 구조

시퀀스는 Transformer의 인코더 레벨을 따라 내려가면서 짧아지고, 중간에서 가장 짧은 표현에 도달한 다음 디코더 레벨을 따라 올라가면서 다시 확장됩니다. 스킵 연결(skip connection)이 확장 단계 근처에서 고해상도 정보를 다시 도입합니다. Hourglass 구조는 합성곱 레이어 없이 U-Net과 유사합니다.

#### 전체 아키텍처 개요

```
입력 (픽셀 이미지: H × W × 3)
         ↓ 패치화 (patch_size p×p)
         ↓
[레벨 0: 고해상도 - Neighborhood Attention (지역)]
         ↓ 다운샘플
[레벨 1: 중간 해상도 - Neighborhood Attention (지역)]
         ↓ 다운샘플
[레벨 2: 저해상도 - Global Self-Attention (전역)]
         ↑ 업샘플 + Skip
[레벨 1: 중간 해상도]
         ↑ 업샘플 + Skip
[레벨 0: 고해상도]
         ↓
출력 (노이즈 예측)
```

Hourglass 아키텍처는 다중 레벨 계층 구조를 채택하며, 각 레벨은 전역 표현이 형성될 때까지 점점 더 거친 해상도로 데이터를 처리한 후, 대칭적인 디코딩 과정이 뒤따릅니다. 이 접근 방식은 U-Net 아키텍처의 다운샘플링 및 업샘플링 경로와 유사하지만, 순수하게 Transformer 블록만으로 구성됩니다.

**해상도별 어텐션 전략**:

HDiT는 고해상도에서는 지역화된 어텐션(localized attention)을 사용하고, 저해상도에서는 전역 어텐션(global attention)을 사용하여 계산 효율성과 표현 능력 간의 균형을 최적화합니다.

**스킵 병합 메커니즘**:

HDiT는 계층 구조에서 **학습 가능한 선형 보간(learnable linear interpolation) 스킵 병합 메커니즘**을 통합하여, 덧셈(additive) 및 연결(concatenation) 기반 스킵 방식보다 개선된 정보 흐름을 보입니다.

수식으로 표현하면:

$$\mathbf{h}_\text{merged} = \text{lerp}(\mathbf{h}_\text{encoder}, \mathbf{h}_\text{decoder},\ \alpha) = (1 - \alpha) \cdot \mathbf{h}_\text{encoder} + \alpha \cdot \mathbf{h}_\text{decoder}$$

여기서 $\alpha$는 학습 가능한 보간 가중치입니다.

**조건 주입**:

모든 HDiT 블록은 노이즈 레벨과 조건 신호(매핑 네트워크를 통해 공동으로 임베딩)를 추가 입력으로 받습니다.

**해상도 확장 규칙**:

목표 해상도가 두 배가 될 때마다, Neighborhood Attention 블록이 하나 추가됩니다.

이를 통해 계산 복잡도는 다음과 같이 분석됩니다:

$$\text{복잡도} \approx O(N \cdot k^2) + O(M^2)$$

여기서 $N$은 총 픽셀 수, $k$는 지역 어텐션 윈도우 크기, $M$은 최저 해상도의 토큰 수(상수)입니다. 이로써 전체적으로 $O(N)$에 근접하는 **선형 스케일링**이 달성됩니다.

**매핑 네트워크(Mapping Network)**:

조건 임베딩은 별도의 MLP 기반 매핑 네트워크에 의해 처리되어 AdaLN(Adaptive Layer Normalization) 파라미터 $\gamma, \beta$를 생성합니다:

$$\mathbf{c}_\text{mapped} = \text{MLP}([\text{emb}(\sigma),\ \text{emb}(\mathbf{y})])$$

$$\hat{\mathbf{h}} = \gamma(\mathbf{c}_\text{mapped}) \cdot \text{LayerNorm}(\mathbf{h}) + \beta(\mathbf{c}_\text{mapped})$$

---

### 2-4. 성능 향상

HDiT는 ImageNet $256^2$에서 기존 모델들과 경쟁력 있는 성능을 보이며, FFHQ- $1024^2$에서 확산 모델 중 **새로운 최고 성능(state-of-the-art)**을 달성합니다.

**계산 비용 비교**:

메가픽셀 해상도에서 HDiT 모델은 유사한 크기의 표준 확산 트랜스포머(DiT) 대비 계산 비용의 **1% 미만**을 소요합니다.

**ImageNet $256^2$ 성능 비교** (FID 기준, 낮을수록 좋음):

HDiT(557M 파라미터)는 FID 6.92, IS 135.2를 달성하여, LDM-4(400M 파라미터, FID 10.56)와 DiT-XL/2(675M 파라미터, FID 9.62)보다 우수한 성능을 보입니다. 그러나 MaskDiT/2(736M 파라미터, FID 5.69)와 같은 더 발전된 잠재 확산 모델은 여전히 HDiT를 능가합니다.

**FFHQ- $1024^2$ 성능**:

HDiT는 대칭적인 특징을 가진 얼굴 생성에서 탁월하며, NCSN++과 같은 다른 확산 모델이 뚜렷한 비대칭성을 보이는 것과 대조됩니다. 또한 HDiT는 가용 해상도를 효과적으로 활용하여 세부 정보가 선명한 이미지를 생성하며, NCSN++ 모델이 종종 흐릿한 샘플을 산출하는 것과 비교됩니다.

**수치적 요약 표** (ImageNet $256^2$, CFG 없음):

| 모델 | 파라미터 | FID ↓ | IS ↑ | 공간 |
|---|---|---|---|---|
| ADM | 554M | 10.94 | 101.0 | Pixel |
| DiT-XL/2 | 675M | 9.62 | 121.5 | Latent |
| LDM-4 | 400M | 10.56 | 209.5 | Latent |
| **HDiT** | **557M** | **6.92** | **135.2** | **Pixel** |
| MaskDiT/2 | 736M | 5.69 | 178.0 | Latent |
| RIN | 410M | 4.51 | 161.0 | Pixel |
| Simple Diffusion | 2B | 2.77 | 211.8 | Pixel |

이 연구는 HDiT의 성능이 ImageNet 과제에 대한 하이퍼파라미터 튜닝 없이도, 일부 최상위 픽셀 공간 모델(2B 파라미터)에 비해 훨씬 적은 파라미터(557M)로 달성됨을 강조합니다. 이는 HDiT의 효율성과 확장 가능성을 나타내며, 추가 최적화 및 스케일링을 통해 픽셀 공간 이미지 합성에서 현재 최고 성능과의 격차를 좁힐 수 있음을 시사합니다.

---

### 2-5. 한계점

보다 발전된 잠재 확산 모델들(예: MaskDiT/2)은 여전히 HDiT를 능가하며, HDiT가 픽셀 공간에서 강력한 성능을 보이지만 잠재 기반 접근 방식은 더 큰 모델과 특정 최적화를 통해 더 우수한 메트릭을 달성할 수 있습니다.

자기-조건화(self-conditioning)를 샘플링에 통합한 모델(RIN, FID 4.51)이나 실질적으로 더 큰 모델(Simple Diffusion 2B 파라미터, FID 2.77; VDM++ 2B 파라미터, FID 2.40)에는 미치지 못합니다.

FID 지표 측면에서 HDiT는 HiT, StyleSwin과 같은 고해상도 트랜스포머 GAN과 경쟁력이 있지만, StyleGAN-XL과 같은 최첨단 GAN의 FID에는 도달하지 못합니다.

**추가적으로 파악되는 한계**:
- 텍스트 조건 생성(text-to-image)에 대한 검증이 논문 내에서 부족함
- 자기-조건화(self-conditioning) 미사용으로 인한 잠재적 품질 제한
- 잠재 공간 모델 대비 절대적 FID 성능 격차 잔존

---

## 3. 모델의 일반화 성능 향상 가능성

---

### 3-1. 구조적 일반화 강점

클래스 조건부 과제에서 HDiT는 **분류기 없는 가이던스(classifier-free guidance) 없이도** 경쟁력 있는 성능을 보여주며, 이는 강건한 일반화 능력을 나타냅니다.

Shorten-factor dropout이 경량의 정규화기(regularizer)로 작동하여 일반화를 개선한다는 것이 관련 Hourglass Transformer 연구에서 밝혀졌습니다.

스킵 연결은 세부 정보 복원에 필수적이며, 멀티스케일 Hourglass 어텐션은 캐스케이드 또는 순수 피라미드 방식보다 이미지 합성에서 더 나은 구조적 충실도를 달성합니다.

### 3-2. 도메인 다양성과 일반화

HDiT는 다양한 분야로의 적용이 이루어지고 있으며, DiffLocks (CVPR 2025), CryoFM (ICLR 2025), Posterior-Mean Rectified Flow (ICLR 2025), LiDAR 데이터 생성 (ICRA 2025) 등에서 활용되었습니다. 이는 HDiT의 아키텍처가 특정 이미지 도메인에 국한되지 않고 **의료 영상, 3D, 물리 시뮬레이션** 등 다양한 도메인에 일반화됨을 보여줍니다.

이 접근 방식은 복잡한 멀티스케일 아키텍처의 필요성을 제거하여, **비디오 생성 및 초해상도** 등의 응용 분야에서 새로운 가능성을 열어줍니다.

### 3-3. 스케일링에 따른 일반화 잠재력

픽셀 공간에서 전역+지역 블록을 가진 계층적 Hourglass 구조(HDiT)는 $O(N)$ 스케일링을 복원하여, **메가픽셀 해상도에서도 훈련을 가능**하게 합니다.

HDiT는 고품질 생성을 위한 추가적인 트릭이 필요 없으며, 따라서 이러한 기법들을 추가하면 생성 샘플 품질이 더 향상될 것으로 기대됩니다.

### 3-4. 일반화 성능의 핵심 요인 분석

| 요인 | 메커니즘 | 일반화 기여 |
|---|---|---|
| Hourglass 계층 구조 | 다해상도 처리 | 다양한 스케일의 패턴 학습 |
| 지역+전역 어텐션 혼합 | 효율적 장거리 의존성 포착 | 이미지 구조 이해 향상 |
| 2D RoPE | 위치 불변성 강화 | 다양한 해상도/비율 일반화 |
| Soft-Min-SNR | 부드러운 노이즈 레벨 가중치 | 훈련 안정성 및 수렴 개선 |
| 스킵 연결 lerp | 세부 정보 보존 | 고주파 특징 일반화 |

---

## 4. 앞으로의 연구에 미치는 영향 및 고려 사항

---

### 4-1. 연구에 미치는 영향

**① 픽셀 공간 확산의 실용화 가능성 증명**

HDiT는 픽셀 공간에서 직접 확산 모델을 사용하여 고품질 고해상도 이미지 생성의 도전에 맞서며, 픽셀 공간에서의 확장 가능한 고품질 이미지 합성을 위한 새로운 가능성을 열었습니다.

**② 후속 픽셀 공간 연구의 기반**

PixelDiT는 완전히 Transformer 기반의 엔드-투-엔드 확산 생성 모델로, 전통적인 이단계 잠재 확산 아키텍처에서 벗어나며, 잠재 공간 모델의 충실도 및 유연성 한계를 해결합니다. HDiT의 영향을 받은 이 후속 연구에서 PixelDiT-XL은 FID = 1.61을 달성하여 이전 픽셀 공간 SOTA를 능가하고 선도적 잠재 확산 모델과의 격차를 좁혔습니다.

**③ 다양한 모달리티로의 확장 근거 제공**

확산 모델의 성공은 정적 이미지를 넘어 비디오 및 오디오 등 다양한 모달리티로 확장되었으며, 이러한 최근의 성공은 훈련 안정성, 확장성, 다양한 샘플 생성에 기인합니다.

**④ 아키텍처 설계 원칙 제시**

HDiT는 선형 스케일링, 적응적 정규화, 지역/전역 어텐션 분해를 갖춘 픽셀 공간 확산 백본으로서, 이전 고해상도 생성 모델과 동등하거나 능가하는 성능을 훨씬 적은 계산 비용으로 달성합니다.

---

### 4-2. 향후 연구 시 고려할 점

**① 잠재 공간 모델과의 하이브리드 접근**

잠재 확산 모델(LDM)의 훈련은 출력 이미지보다 $8 \times 8$ 낮은 공간 해상도의 잠재 공간에서 이루어지므로 고주파 세부 정보가 손실됩니다. 이를 해결하기 위해 후처리 단계에서 픽셀 공간 감독을 추가하는 방향이 유망합니다.

**② 스케일링 법칙(Scaling Laws) 체계화**

HDiT가 픽셀 공간 생성의 강력한 기준을 제공하지만, 추가적인 아키텍처 스케일링이나 고급 샘플링 기법의 통합이 더 나은 결과를 이끌 수 있습니다.

**③ 자기-조건화(Self-Conditioning) 통합 가능성**

자기-조건화를 샘플링에 통합한 모델(RIN, FID 4.51)이 HDiT를 능가하는 점을 고려하면, 향후 HDiT에 자기-조건화를 결합하는 연구가 유망합니다.

**④ 텍스트 조건부 생성으로의 확장**

HDiT의 비디오 생성 등 전문 과제에서의 추가 성능 향상을 위한 잠재적 적응이 앞으로의 연구 방향으로 주목됩니다.

**⑤ 효율적 어텐션 대안 탐색**

GLA(Gated Linear Attention) 기반 모델(DiG)은 $1792 \times 1792$ 해상도에서 Mamba 기반 모델 대비 4.2배, DiT 대비 2.5배의 속도 향상을 보이면서도 FID 손실이 없어, HDiT의 지역 어텐션 부분을 선형 어텐션으로 대체하는 연구도 고려할 만합니다.

**⑥ 평가 지표의 다각화**

FID 지표는 GAN이 생성한 샘플을 확산 모델 샘플보다 편향적으로 선호하는 것으로 알려져 있어, 확산 모델의 인상적인 성능이 과소평가될 수 있습니다. 따라서 FID 외에 Precision/Recall, FD-DINOv2 등 보완적 지표를 함께 사용해야 합니다.

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

잠재 확산(Latent Diffusion)은 압축된 이미지 잠재 표현에서 이미지를 생성하는 최고 성능 방식으로, DiT(Peebles & Xie, 2023), PixArt(Chen et al., 2023) 등이 이 패러다임을 따릅니다.

| 모델 | 연도 | 공간 | 핵심 기법 | 비교 포인트 |
|---|---|---|---|---|
| **DDPM** (Ho et al.) | 2020 | Pixel | U-Net + DDPM | 확산 모델의 기초 |
| **LDM / Stable Diffusion** (Rombach et al.) | 2022 | Latent | VAE + U-Net | 잠재 공간 표준화 |
| **DiT** (Peebles & Xie) | 2023 | Latent | Transformer + AdaLN | Transformer 확산의 기준 |
| **Simple Diffusion** (Hoogeboom et al.) | 2023 | Pixel | U-Net + 고해상도 트릭 | 픽셀 직접 확산, 2B |
| **HDiT** (Crowson et al.) | **2024** | **Pixel** | **Hourglass + 지역/전역 어텐션** | **선형 복잡도, 효율적** |
| **SANA** (ICLR 2025) | 2025 | Latent | 선형 어텐션 DiT | 텍스트-이미지, 효율성 강조 |
| **PixelDiT** | 2025 | **Pixel** | 이중 레벨 DiT | HDiT 후속, FID 1.61 달성 |

PixelDiT 및 관련 아키텍처는 순수 Transformer를 사용하는 엔드-투-엔드 픽셀 공간 확산의 타당성과 경쟁력을 확립하며, 이중 레벨 메커니즘은 잠재 확산 모델에 대한 실행 가능한 대안으로 픽셀 DiT 유사 모델을 자리매김합니다.

HDiT는 이 계보에서 **잠재 인코더 없이 Transformer만으로 고해상도 픽셀 합성을 가능하게 한 최초의 실용적 모델** 중 하나로, 이후 픽셀 공간 연구의 토대를 마련했다는 점에서 의미가 큽니다.

---

## 📚 참고 자료 및 출처

1. **원논문 (arXiv)**: Crowson, K. et al. (2024). *Scalable High-Resolution Pixel-Space Image Synthesis with Hourglass Diffusion Transformers*. arXiv:2401.11605. https://arxiv.org/abs/2401.11605
2. **ICML 2024 공식 게재**: Proceedings of Machine Learning Research, Vol. 235, pp. 9550–9575. https://proceedings.mlr.press/v235/crowson24a.html
3. **공식 프로젝트 페이지**: https://crowsonkb.github.io/hourglass-diffusion-transformers/
4. **HTML 논문 (ar5iv)**: https://ar5iv.labs.arxiv.org/html/2401.11605
5. **ACM Digital Library**: https://dl.acm.org/doi/10.5555/3692070.3692447
6. **OpenReview**: https://openreview.net/forum?id=WRIn2HmtBS
7. **Hugging Face Papers**: https://huggingface.co/papers/2401.11605
8. **Semantic Scholar**: https://www.semanticscholar.org/paper/9b91b3031ea159e4964d18b2ce703168660ecf46
9. **Liner.com Quick Review**: https://liner.com/review/scalable-highresolution-pixelspace-image-synthesis-with-hourglass-diffusion-transformers
10. **Emergent Mind**: https://www.emergentmind.com/papers/2401.11605
11. **GitHub (코드)**: https://github.com/crowsonkb/k-diffusion
12. **Stability AI Research Blog**: https://stability.ai/research/hourglass-diffusion-transformer-high-resolution-image-synthesis
13. **관련 비교 연구 (PixelDiT)**: https://www.emergentmind.com/topics/pixeldit
