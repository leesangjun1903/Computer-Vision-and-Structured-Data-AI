# RDDM : Residual Denoising Diffusion Models

**핵심 주장 및 기여**  
Residual Denoising Diffusion Models(RDDM)은 기존 단일 노이즈 기반 확산 프로세스를 “잔차(residual) 확산”과 “노이즈 확산”으로 분리한 **이중 확산 프레임워크**를 제안한다. 이를 통해 이미지 생성과 복원을 하나의 통일된 모델로 해석 가능하게 만들며, 배치 크기 1과 단순 L₁ 손실만으로도 SOTA 성능에 근접한 결과를 달성한다.[1]

***

## 1. 해결하고자 하는 문제  
기존 Denoising Diffusion Probabilistic Models(DDPM)과 DDIM은  
- 이미지 **복원** 시 불필요한 역방향 노이즈 초기화  
- 확산 과정이 **조건 이미지(degraded input)** 정보 포함 없이 비해석적(non-interpretable)  
라는 한계를 지닌다.  

RDDM은 이 문제를 “잔차”를 도입해 해결한다.

***

## 2. 제안하는 방법  
### 2.1 이중 확산(dual diffusion) 정의  
- **잔차 확산(residual diffusion)**: 목표 이미지 $$I_0$$에서 열화 이미지 $$I_{\text{in}}$$까지의 **방향성 확산**  
- **노이즈 확산(noise diffusion)**: 무작위 섭동(perturbation) 확산  

각 타임스텝 $$t$$에서의 전방 확산:  

$$
q(I_t\mid I_{t-1},I_{\text{res}})
= \mathcal{N}\bigl(I_t; I_{t-1} + \alpha_t I_{\text{res}},\,\beta_t^2\mathbf{I}\bigr)
$$  

여기서  

$$\displaystyle I_{\text{res}} = I_{\text{in}} - I_0,\;\alpha_t,\beta_t$$는 독립 스케줄.[1]

### 2.2 역방향 샘플링 및 학습목표  
역방향 과정에서 네트워크는  
- 잔차 $$I_{\text{res}}$$ 예측  
- 노이즈 $$\varepsilon$$ 예측  
두 가지 방식으로 학습 가능하며, 학습 중 자동 선택 알고리즘(AOSA)을 통해 SM-Res(잔차), SM-N(노이즈), SM-Res-N(잔차+노이즈) 방법 중 최적 방식을 결정한다.

간소화된 학습손실:  

$$
\mathcal{L}_{\text{res}}
= \mathbb{E}\bigl\|I_{\text{res}} - \widehat I_{\text{res}}\bigr\|_2^2,\quad
\mathcal{L}_{\varepsilon}
= \mathbb{E}\bigl\|\varepsilon - \widehat\varepsilon\bigr\|_2^2
$$

### 2.3 계수 변환(coefficient transformation)  
기존 DDPM/DDIM에서 학습된 모델도 RDDM으로 **무재학습 재활용** 가능하도록  

$$
\alpha_t = 1 - \bar\alpha_{t}^{\text{DDIM}},\quad
\beta_t = \sqrt{1 - \bar\alpha_{t}^{\text{DDIM}}}\quad
(\bar\alpha_{t}^{\text{DDIM}}: \text{누적 계수})
$$  

변환 공식을 제공하여 DDIM⇄RDDM 변환 지원.[1]

***

## 3. 모델 구조  
- **UNet 기반 네트워크**: 잔차 예측 네트워크 + 노이즈 예측 네트워크(또는 단일 네트워크 다중 출력)  
- 학습: 배치 크기 1, **단일 L₁ 손실**, 5단계 이하 샘플링으로도 수렴  

***

## 4. 성능 향상 및 한계  
### 4.1 이미지 생성  
- CelebA(256×256) 데이터셋에서 DDIM 대비 유사 FID/IS 유지  
- 학습 없이 DDIM 사전학습 모델 재사용 가능[1]

### 4.2 이미지 복원  
- **그림자 제거(ISTD), 저조도 향상(LOL), 강우 제거(RainDrop)** 등 4개 복원 작업에서  
  - 배치 크기 1, L₁만으로 SOTA 수준 근접 (예: ISTD PSNR 36.74dB, SSIM 0.979)[1]
- **5단계 이하** 샘플링으로 고성능 달성  

### 4.3 한계  
- 특정 작업 최적 SOTA 성능 위해선 더 큰 모델, 배치 크기 증가, 복합 손실 필요  
- 이미지 생성에서 최첨단 품질 확보 위해 계수 스케줄·네트워크 구조 추가 연구 필요  

***

## 5. 일반화 성능 향상 가능성  
- **AOSA**로 새로운 작업에 맞춰 자동으로 잔차/노이즈 예측 전략 선택  
- **계수 스케줄 독립 설계**로 다양한 확산 속도 조절 가능  
- UNet 구조 교체 없이 **다양한 영상 변환(Impaint, Translation)** 작업에 확장성 검증  

***

## 6. 향후 연구 영향 및 고려 사항  
- **다변량 확산(multidimensional diffusion)** 및 곡선 적분 관점 이론 심화  
- **적응적 계수 스케줄** 학습으로 샘플링 단계 획기적 감소  
- **다중 조건**(텍스트+이미지) 통합 생성 모델로 확장  
- **연산 효율성** 강화하여 자원 제약 환경에서의 확산 모델 적용  

RDDM은 잔차와 노이즈의 독립·병렬적 중요성을 강조하며, 이미지 생성과 복원을 아우르는 통합·해석 가능한 확산 프레임워크로서 향후 영상 처리·생성 연구에 새로운 패러다임을 제시한다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/fb2ba6ef-e958-4aff-a66c-491c4269a42e/2308.13712v3.pdf)
