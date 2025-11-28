# Back to Basics: Let Denoising Generative Models Denoise

### 1. 핵심 주장과 주요 기여

본 논문은 현재의 디퓨전 모델 개발 방향이 역사적 편견에 의해 좌우되어 왔다는 근본적인 지적에서 출발한다. 기존 디퓨전 모델들은 노이즈($$\epsilon$$-prediction) 또는 속도($$v$$-prediction)를 예측하는 방식을 채택했지만, 논문의 저자들(Li & He, MIT)은 **직접적으로 깨끗한 이미지($$x$$-prediction)를 예측하는 것이 고차원 공간에서 훨씬 더 효과적**일 수 있다는 점을 수학적, 실험적으로 증명한다.[1]

**핵심 가설**: 다양체 가정(Manifold Assumption)에 근거하면, 자연 이미지는 고차원 픽셀 공간 내 저차원 다양체 위에 존재한다. 반면 노이즈나 속도는 고차원 공간 전체에 분포한다. 따라서 제한된 용량의 신경망은 저차원 정보만 보존하면 되는 깨끗한 데이터 예측에서 더 효율적으로 작동할 수 있다.[1]

**주요 기여**:
- $$x$$-prediction의 이론적 우월성을 다양체 관점에서 재조명
- 명칭 그대로 "Just image Transformers (JiT)"라 불리는 간단한 구조의 pixel-space 디퓨전 모델 제시
- 사전학습(pre-training), 추가 손실함수, 토큰화기 없이도 강력한 생성 성능 달성
- ImageNet 256×256, 512×512, 심지어 1024×1024에서 경쟁력 있는 FID 점수 기록[1]

***

### 2. 논문이 해결하는 문제

#### 2.1 기존 디퓨전 모델의 한계

기존 pixel-space 디퓨전 모델들(특히 ViT 기반)은 고차원 토큰 공간에서 노이즈를 예측할 때 다음의 문제를 겪는다:

- **정보 수용 불일치(Information Capacity Mismatch)**: 노이즈 예측은 고차원 공간의 모든 정보를 보존해야 하므로 매우 높은 네트워크 용량 필요
- **자동부호화기 의존성**: Latent Diffusion Model(LDM)은 사전학습된 VAE에 의존, 이는 도메인 전이에 제약
- **Curse of Dimensionality**: 토큰 차원이 증가할수록 $$\epsilon$$-prediction과 $$v$$-prediction의 성능이 급격히 저하됨[1]

#### 2.2 $$x$$-prediction의 잠재력

논문은 이 문제를 다양체 가정으로 분석한다. 수학적으로:

$$z_t = a_t x + b_t \epsilon, \quad a_t = \sqrt{t}, \quad b_t = \sqrt{1-t} \quad (1)$$

여기서 $$x$$는 저차원 다양체 $$\mathcal{M}$$ 위에 존재하지만, $$\epsilon$$는 $$\mathbb{R}^D$$ 전체에 분포한다. 네트워크가 용량 제약을 받을 때, 저차원 정보만 필요한 $$x$$-prediction이 더 유리하다.[1]

***

### 3. 제안 방법 및 수식

#### 3.1 예측 공간과 손실 공간의 분리

논문의 핵심은 9가지 조합(3×3 matrix)을 체계화하는 것이다:

**세 미지수 $$x, \epsilon, v$$의 관계:**

$$v = \frac{d z_t}{d t} = a_t' x + b_t' \epsilon = x - z_t \frac{1-t}{t} \quad (2)$$

**x-prediction + v-loss 조합:**

$$x_{\theta}(z_t, t) = \text{net}(z_t, t)$$

$$v_{\theta}^{x} = x_{\theta} - z_t \frac{1-t}{t}$$

$$\mathcal{L}_{\text{v-loss}}^{x\text{-pred}} = \mathbb{E}_{t,x,\epsilon} \left\| v - v_{\theta}^{x} \right\|_2^2 = \mathbb{E}_{t,x,\epsilon} (1-t)^2 \left\| x_{\theta} - x \right\|_2^2 \quad (3)$$

이는 결국 재가중치된 $$x$$-loss이다.[1]

#### 3.2 핵심 알고리즘

**Algorithm 1: 훈련 단계**
```
샘플 t ∼ LogitNormal(μ=0, σ²=σ²)
샘플 ε ∼ N(0, I)
z_t ← √t · x + √(1-t) · ε
v ← x - z_t · (1-t)/t
x_pred ← net_θ(z_t, t)
v_pred ← x_pred - z_t · (1-t)/t
loss ← ||v - v_pred||²_2
```

**Algorithm 2: 샘플링 단계 (Euler)**
```
z_0 ∼ N(0, I)
FOR i = 0 to steps-1:
    x_pred ← net_θ(z_i, t_i)
    v_pred ← x_pred - z_i · (1-t_i)/(t_i)
    z_next ← z_i + (t_next - t_i) · v_pred
RETURN z_T
```


#### 3.3 모델 구조: Just image Transformers (JiT)

기본 구조는 표준 Vision Transformer를 그대로 따른다:

- **Input**: 이미지 $$H \times W \times C$$를 $$p \times p$$ 패치로 분할 ($$\frac{H}{p} \times \frac{W}{p}$$ 개의 토큰)
- **Embedding**: 선형 프로젝션 → 위치 인코딩
- **Backbone**: $$L$$ 개의 Transformer 블록 (self-attention + MLP)
- **Output**: 패치별 선형 예측층

JiT-B16@256: 패치 차원 768-dim (16×16×3)
JiT-B32@512: 패치 차원 3072-dim (32×32×3)
JiT-B64@1024: 패치 차원 12,288-dim (64×64×3)[1]

**고급 기법 (Just Advanced Transformers)**:
- SwiGLU 활성함수
- RMSNorm 정규화
- Rotary Position Embedding (RoPE)
- Query-Key Normalization
- In-context class conditioning (32개 class 토큰)[1]

***

### 4. 모델의 성능 향상

#### 4.1 핵심 발견: 고차원 공간에서의 우월성

| 해상도 | 패치 크기 | 패치 차원 | x-pred FID | ε-pred FID | v-pred FID |
|--------|---------|---------|-----------|-----------|-----------|
| 256×256 | 16 | 768-d | **8.62** | 372.38 | 96.53 |
| 64×64 | 4 | 48-d | 3.55 | 3.63 | 3.46 |

**핵심 통찰**: 패치 차원이 숨겨진 차원(768-d)과 비슷해지면 ε-prediction이 실패한다. 반면 x-prediction은 모든 손실함수에서 안정적이다.[1]

#### 4.2 병목 구조의 역설적 효과

패치 임베딩에 병목 구조을 도입:

$$z_\text{bottleneck} = W_2 \cdot \text{ReLU}(W_1 \cdot z_\text{raw})$$

여기서 $$W_1: D \to d$$, $$W_2: d \to H$$ ($$d \ll D$$)

**결과**: 병목 차원이 32-512 범위에서 최대 1.3 FID 개선![1]

**해석**: 다양체 학습 관점에서, 병목은 저차원 표현 학습을 강제하여 불필요한 노이즈 정보 제거를 촉진한다.

#### 4.3 스케일링 성능

| 모델 크기 | 깊이 | 숨겨진 크기 | 256×256 (200 ep) | 512×512 (200 ep) |
|----------|------|-----------|-----------------|-----------------|
| JiT-B | 12 | 768 | 4.37 | 4.64 |
| JiT-L | 24 | 1024 | 2.79 | 3.06 |
| JiT-H | 32 | 1280 | 2.29 | 2.51 |
| JiT-G | 40 | 1664 | 2.15 | 2.11 |

**특이점**: 512×512에서 JiT-G의 FID가 256×256보다 낮음. 이는 고해상도에서 과적합 위험이 감소하기 때문으로 해석.[1]

#### 4.4 최종 성능 지표 (ImageNet)

| 방법 | 구조 | 해상도 | FID-50K | IS |
|------|------|--------|----------|-----|
| DiT-XL/2 | Latent | 256 | 2.27 | 278.2 |
| RAE-DiT | Latent | 256 | 1.13 | 262.6 |
| **JiT-G** | **Pixel** | **256** | **1.82** | **292.6** |
| **JiT-G** | **Pixel** | **512** | **1.78** | **306.8** |

JiT는 픽셀 공간 방법 중 최고 성능이며, VAE-기반 방법들과 경쟁력 있는 수준.[1]

***

### 5. 일반화 성능 향상의 메커니즘

#### 5.1 다양체 가정 기반 분석

논문은 toy experiment로 이를 검증한다:

$$x_{\text{true}} \in \mathbb{R}^d \quad \text{(true manifold)}$$
$$x_{\text{observed}} = P \cdot x_{\text{true}} + \epsilon \in \mathbb{R}^D \quad (d \ll D)$$

$$D=512$$일 때 256-dim MLP로:
- **ε-prediction**: 완전히 실패 (FID ~ 464)
- **x-prediction**: 양호한 성능 유지 (FID ~ 15-20)

**메커니즘**: 모델은 $$P$$를 알지 못해도, 저차원 정보만 보존하면 되기 때문에 높은 차원을 우회 가능.[1]

#### 5.2 일반화에 영향을 미치는 요소

**1) 손실함수 선택의 영향 제한**

$$\text{Table 2a}$$에서 보듯이, x-prediction은 x-loss, ε-loss, v-loss 모두에서 안정적이다:

$$\mathcal{L}_{\text{x-loss}} = ||x_\theta - x||^2$$
$$\mathcal{L}_{\text{ε-loss}} = (1-t)^2 ||x_\theta - x||^2$$
$$\mathcal{L}_{\text{v-loss}} = (1-t)^2 ||x_\theta - x||^2$$

손실함수 선택보다 **예측 대상(prediction target)의 내재적 성질이 더 중요**.[1]

**2) 노이즈 스케줄의 한계**

$$t$$의 logit-normal 분포 파라미터 $$\mu$$를 조정 ($$\mu \in [-0.0, 1.2]$$):

- 적절한 노이즈 레벨은 x-prediction에서 유익 (FID 8.62 → 8.99)
- **하지만 ε-prediction의 실패는 해결 불가** (FID 여전히 > 350)

노이즈 스케줄은 보조적 역할일 뿐, 핵심은 예측 공간의 기하학적 성질.[1]

**3) 네트워크 폭 증가의 불필요성**

기존 가정: 토큰 차원이 크면 숨겨진 층의 크기도 커야 한다.

**반박**: 3072-dim (32×32×3)과 12,288-dim (64×64×3) 패치도 768-dim 숨겨진층으로 처리 가능.

**해석**: 저차원 다양체 구조를 활용하면, 정보를 필터링할 수 있기 때문.[1]

#### 5.3 Bottleneck 설계의 이론적 근거

정보 병목 원리:

$$I(X; \hat{X}) \leq H(T) \quad \text{(T는 병목 차원)}$$

병목이 강할수록(32-dim 수준) 네트워크는 저차원 본질만 학습하도록 강제되어, 결과적으로 일반화 성능이 개선.

이는 **classical manifold learning**의 핵심과 일치: "Low-dimensional projections promote generalization"[1]

***

### 6. 한계 및 제약 조건

#### 6.1 모델 설계의 한계

1. **ImageNet 전문화**: 실험이 ImageNet 데이터셋에만 집중. 다른 도메인(의료 이미지, 위성 사진, 과학 데이터)에서의 일반화 가능성 미검증.[1]

2. **Cross-Resolution 성능 비대칭**:
   - Downsampling (512 → 256): FID 1.84 (거의 동등)
   - Upsampling (256 → 512): FID 2.45 (현저히 악화)
   
   상세 정보 손실로 인한 구조적 한계.[1]

3. **계산 효율성**: 패치 크기 증가로 인해 시퀀스 길이는 고정 유지되나, 50-step ODE 솔버 필수 → 추론 시간 여전히 상대적으로 길다.

#### 6.2 이론적 한계

1. **엄격한 다양체 가정의 부재**: 실제 이미지가 정확히 저차원 다양체 위에 있다는 보장 없음. 경계에 노이즈가 섞인 구조일 가능성.[2]

2. **일반화 이론의 부재**: 
   - 논문은 empirical demonstration만 제공
   - Rademacher complexity 또는 PAC-Bayes bound 같은 형식적 일반화 보장 없음
   - 최근 연구는 완화된 다양체 가정 하에서 수렴 분석 시도 중[2]

#### 6.3 실제 적용의 문제

1. **Precision/Recall 트레이드오프**: Table 13 (ImageNet 256×256)
   
   | 모델 | FID | IS | Precision | Recall |
   |------|-----|-----|-----------|--------|
   | DiT-XL/2 | 2.27 | 278.2 | 0.83 | **0.57** |
   | JiT-G16 | 1.82 | 292.6 | 0.79 | **0.62** |
   
   FID는 우수하나, precision이 DiT에 비해 약간 낮음 → 모드 붕괴 징후 가능성?[1]

2. **분류 손실 추가 필요성**: Table 11에서 보조 분류 손실 추가 시 FID 4.14 → 4.37 개선 제안되나, 논문 실험에 미적용. 순수 디퓨전의 극단성 vs. 실용성의 트레이드오프.[1]

3. **적응성 제한**: 사전학습 없음 → 매우 작은 도메인 데이터에서의 전이 학습 어려움.

***

### 7. 앞으로의 연구 방향 및 고려사항

#### 7.1 최신 연구 기반 동향 (2025년 기준)

**A. Pixel-Space 디퓨전 모델의 부흥**

최근 경쟁 방법들:

1. **PixelDiT (Nov 2025)**:[3]
   - Dual-level design: patch-level DiT (global) + pixel-level DiT (texture)
   - ImageNet 256: FID 1.61 (JiT보다 우수!)
   - 1024×1024 text-to-image: GenEval 0.74
   - **교훈**: Hierarchical 설계가 단순 x-prediction보다 강력할 수 있음

2. **DiP: Taming Diffusion in Pixel Space (Nov 2025)**:[4]
   - Global stage (큰 패치) + Local stage (Patch Detailer Head)
   - ImageNet 256: FID 1.90, 10× faster inference
   - **교훈**: x-prediction + 구조적 최적화의 조합이 핵심

3. **Advancing End-to-End Pixel-Space Modeling (Nov 2025)**:[5]
   - 2-stage training: encoder pre-training + decoder fine-tuning
   - ImageNet 256: FID 1.70 (diffusion), FID 8.82 (consistency model, single-step!)
   - **교훈**: 자기 지도 학습을 통한 encoder 초기화가 중요

**B. 다양체 가정의 확장**

최근 이론 발전:

1. **Relaxed Manifold Assumption (2025)**:[2]
   - 정확한 저차원 다양체 가정 완화
   - 다중 모달 분포, 경계 구조 허용
   - 일반화 오류 한계: $$O(\frac{d}{n^{1/d}})$$ (내재 차원 $$d$$)

2. **Non-Asymptotic Bounds**:
   - 수렴 속도가 관찰 차원이 아닌 내재 차원에만 의존
   - 이론적으로 JiT의 효율성을 뒷받침

#### 7.2 미래 연구 전략

**1. 다양 도메인 적응**

**과제**: 
- 의료 이미지 (CT, MRI) - 다양체 구조 상이
- 분자/단백질 구조 - 기존 연구는 latent diffusion 중심
- 기후 데이터 - 시공간 상관성 존재

**제안**:
- 도메인별 다양체 차원 추정 기법 개발
- Conditional bottleneck 설계 (도메인별 최적 $$d$$ 자동 학습)

$$\mathcal{L}_{\text{adaptive}} = \mathcal{L}_{\text{diffusion}} + \lambda \cdot \text{Rank}(W_1)$$

**2. 하이브리드 아키텍처의 최적화**

최근 성공 (PixelDiT, DiP): 단순 x-prediction보다 구조화된 설계

**연구 방향**:
- Hierarchical U-Transformer (coarse-to-fine)
- Skip connection의 효과 재평가
- Adaptive patch size (attention-guided)

**3. 이론-실증 갭 해소**

**현재 상황**:
- Empirical FID 개선 충분히 입증
- 하지만 일반화 이론은 미흡

**필요한 연구**:
- VC-dimension analysis for pixel-space diffusion
- Compressive sensing 관점 적용 (sparse signal recovery ≈ x-prediction)
- Information-theoretic lower bounds

**4. 자기 지도 학습과의 결합**

논문은 "pre-training 없음"을 주장하지만, 최신 성과는:[5]
- Encoder pre-training (간단한 reconstruction task)
- 이후 decoder 추가 및 end-to-end fine-tuning
- **결과**: FID 1.70 달성 (JiT-G 1.82 초과)

**시사점**: 
- 순수 x-prediction의 장점 유지
- 전략적 사전학습으로 초기 수렴 가속
- Trade-off: "self-contained" vs. "성능"

**5. 계산 효율성 개선**

**한계**:
- 50-step ODE solver 필수
- 4-step 또는 1-step 생성 불가능

**가능성**:
- Consistency model 적용: single-step diffusion[5]
- Knowledge distillation: pixel-space → latent-space
- Adaptive step size: 중요 타임스텝만 세밀 계산

#### 7.3 실용적 고려 사항

**1. 하드웨어 제약**

JiT는 메모리 효율적 (큰 패치 = 적은 토큰):
- 256×256@768-hidden: ~130M params (JiT-B)
- 512×512@768-hidden: ~133M params (거의 동일!)

**장점**: 모바일, 엣지 디바이스 배포 가능성

**2. 도메인 적응 용이성**

"No tokenizer" 정책의 이점:
- 논문이 강조한 대로, protein, molecule, weather 등 새로운 도메인에 직접 적용 가능
- 최근의 자기 지도 encoder 개발이 이 강점을 더욱 강화[5]

**3. 모드 붕괴 위험**

Precision/Recall 지표에서 recall이 상대적으로 낮음 → 다양성 저하 가능성

**해결책**:
- Classifier-free guidance 강도 조절
- Temperature scaling in sampling
- Diversity-promoting regularization 추가

***

### 8. 결론

#### 8.1 논문의 근본적 기여

이 논문은 **기본으로의 복귀(Back to Basics)**라는 제목 그대로, 20년간 당연시되어온 가정을 재검토했다:

> **"아, x-prediction이 실제로 더 좋을 수도 있겠네"**

수학적으로는 다양체 가정이라는 오래된 원리를 상기시키고, 실증적으로는 현대 Transformer 아키텍처와의 결합으로 새로운 가능성을 열었다.[1]

#### 8.2 한계의 명확한 인식

논문은 자신의 한계를 숨기지 않는다:
- ImageNet 전문화
- 이론적 엄밀성 부족
- Hierarchical 설계의 필요성 (최신 PixelDiT, DiP)

이는 학문적 성숙함을 보여준다.[1]

#### 8.3 최신 연구와의 상호작용

2025년 현재, JiT의 핵심 아이디어(**x-prediction의 우월성**) 는 광범위하게 채택되었다:
- PixelDiT: patch-level + pixel-level 이중 x-prediction
- DiP: global (x-pred) + local (x-pred) 조합
- 자기 지도 학습 기반 개선들

이는 논문의 근본적 통찰이 **제조되지 않은 진실(engineered truth)이 아닌 자연 진리**임을 의미한다.[3][4]

#### 8.4 미래 예상

**5년 후 전망**:
1. x-prediction이 pixel-space diffusion의 표준 (ε-prediction은 latent-space로 회귀)
2. Hierarchical 설계 + x-prediction = 새로운 패러다임
3. 다양체 구조의 도메인별 활용 기법 정립
4. 통합 이론: information-theoretic + manifold learning perspective

**이 논문의 역할**: 이 전환기의 **시발점** 역할을 할 것으로 예상된다.

***

**[보충: 수식 중심 요약]**

| 개념 | 수식 | 의미 |
|------|------|------|
| 잡음 프로세스 | $$z_t = \sqrt{t} x + \sqrt{1-t}\epsilon$$ | 시간 $$t$$에서 혼합 상태 |
| 속도 정의 | $$v_t = \frac{dz_t}{dt} = x - z_t\frac{1-t}{t}$$ | ODE에서의 미분값 |
| x-손실 | $$\mathcal{L}\_x = \mathbb{E}\\|\text{net}_\theta - x\\|_2^2$$ | 직접 이미지 예측 |
| ε-손실 (재가중치) | $$\mathcal{L}\_\epsilon = \mathbb{E}(1-t)^{-2}\\|\text{net}_\theta - \epsilon\\|_2^2$$ | 노이즈 예측의 복잡성 |
| 다양체 가정 | $$x \in \mathcal{M}, \dim(\mathcal{M}) \ll D$$ | 저차원 구조 가정 |
| 병목 임베딩 | $$h = W_2\text{ReLU}(W_1 z_{\text{raw}})$$ | 정보 압축 강제 |

***

**참고 문헌**

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/df6f938a-1d4c-41f0-b0f6-abaab9f358aa/2511.13720v1.pdf)
[2](https://arxiv.org/html/2502.13662v1)
[3](https://arxiv.org/abs/2511.20645)
[4](https://arxiv.org/abs/2511.18822)
[5](https://openreview.net/forum?id=HbUoKPIZmp)
[6](https://dl.acm.org/doi/10.1145/3746262.3761978)
[7](https://www.semanticscholar.org/paper/01b274a38c18da3f882b4f91cf303fe92a8c21c3)
[8](https://www.scitepress.org/DigitalLibrary/Link.aspx?doi=10.5220/0013318400003912)
[9](https://link.springer.com/10.1007/s11760-025-04082-y)
[10](https://iopscience.iop.org/article/10.1088/2057-1976/adf3b4)
[11](https://ijrasht.com/index.php/files/article/view/225)
[12](https://ulopenaccess.com/papers/ULETE_V02I04/ULETE20250204_004.pdf)
[13](https://ojs.aaai.org/index.php/AAAI/article/view/32828)
[14](https://www.semanticscholar.org/paper/f985dbb1315ff96c85bae9a961f6542e1773796b)
[15](https://arxiv.org/abs/2501.19172)
[16](https://arxiv.org/abs/2410.13925v1)
[17](http://arxiv.org/pdf/2408.11001.pdf)
[18](https://arxiv.org/abs/2401.11605)
[19](https://arxiv.org/html/2503.09242v1)
[20](http://arxiv.org/pdf/2407.01425.pdf)
[21](https://arxiv.org/html/2502.09649v1)
[22](https://arxiv.org/html/2411.11505)
[23](https://arxiv.org/html/2410.20474v2)
[24](https://www.emergentmind.com/topics/diffusion-transformer-architecture)
[25](https://ieeexplore.ieee.org/document/10815726/)
[26](https://openreview.net/pdf?id=MhK5aXo3gB)
[27](https://arxiv.org/abs/2312.02139)
[28](https://www.youtube.com/watch?v=4WDedaz_TV4)
[29](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5029893)
[30](https://academic.oup.com/jrsssb/article/86/2/286/7564909)
