# DiffIR: Efficient Diffusion Model for Image Restoration

**주요 시사점:** DiffIR은 고해상도 이미지 복원을 위해 기존 디퓨전 모델의 과도한 반복 및 대규모 연산을 해결하고, 압축된 IR prior 표현만을 추정하는 방식으로 최적화하여 복원 효율성과 안정성을 획기적으로 개선한다.[1]

## 1. 핵심 주장 및 기여
DiffIR은 이미지 복원(IR) 과제에 특화된 효율적 디퓨전 모델로,  
- **Compact IR Prior Extraction Network (CPEN)**로부터 압축된 prior 표현(IPR)을 추출하고,  
- **Dynamic IR Transformer (DIRformer)**로 이를 활용해 세밀한 복원 디테일을 추가하며,  
- **디퓨전 네트워크**를 통해 LQ 입력만으로 IPR을 정확히 예측함  
세 단계 구조를 두 단계 학습(staged training)으로 결합함으로써, 전통적 DM 대비 반복 횟수를 수십 배 절감하면서도 SOTA 성능을 달성한다.[1]

## 2. 해결 문제
- 전통적 디퓨전 모델은 IR에서 대부분 픽셀이 이미 주어졌음에도 이미지 합성을 위해 전체 이미지 또는 특징 맵을 반복적으로 추정하므로  
  - 과도한 계산 자원 소모  
  - LQ 입력과 일치하지 않는 불필요한 세부 생성  
문제를 유발한다.

## 3. 제안 방법
### 3.1. 두 단계 학습
1. **Pretraining**  
   - Ground-truth HQ 이미지를 CPEN에 입력해 IPR $$Z\in\mathbb{R}^{4\times C}$$을 생성  
   - DIRformer로 LQ→HQ 복원을 수행하며 손실 $$\mathcal{L}\_{\mathrm{rec}} = \|I_{GT}-I_{HQ}\|_1$$로 최적화  

$$
     Z = \mathrm{CPEN}_S^1\bigl(\mathrm{PixelUnshuffle}([I_{GT},I_{LQ}])\bigr)
   $$  

2. **Diffusion Training**  
   - 사전학습된 CPEN $$^1$$ 이 추출한 IPR을 디퓨전 과정으로 노이즈화(식 $$\mathrm{q}(Z_T|Z)$$)  
   - CPEN $$^2$$ 과 디퓨저가 LQ 이미지에서 직접 IPR을 역추정(식 $$\mathrm{p}(Z_{t-1}|Z_t,I_{LQ})$$)  
   - 추정된 IPR로 DIRformer를 joint optimization  

$$
     Z_{t-1} = \frac{1}{t}Z_t + \frac{t-1}{t}\bigl(Z_t - \epsilon_\theta(Z_t,t,D)\bigr)
   $$  
   
   최종 손실: $$\mathcal{L}\_{\mathrm{all}} = \mathcal{L}\_{\mathrm{rec}} + \mathcal{L}_{\mathrm{diff}}$$.[1]

### 3.2. 모델 구조
- **CPEN:** 잔차 블록·선형 레이어로 이루어진 압축형 prior 추출기  
- **DIRformer:** Unet 형태의 Dynamic Transformer 블록,  
  - Dynamic Gated FFN (DGFN)  
  - Dynamic Multi-Head Transposed Attention (DMTA)  
  prior를 모듈화 파라미터로 활용  
- **Denoising Network:** compact IPR에 특화된 경량 네트워크

## 4. 성능 향상 및 한계
- **연산 효율:** RePaint 대비 Mult-Adds 1,000배 절감, 유사 연산량 대비 FID·PSNR 성능 우위.[1]
- **SOTA 성능:** Inpainting, SR, Motion Deblurring 전 영역에서 최고 혹은 차상위 결과  
- **Joint Optimization 효과:** 전통 DM 최적화 대비 IPR 추정 오차에 강인하며 FID 0.4913 달성  
- **한계:**  
  - IPR 표현이 지나치게 단순화되면 복원 세부 표현 한계  
  - 복잡한 비정형 열화(예: 비균일 노이즈) 대응 미검증

## 5. 일반화 성능 향상 방안
- **다양한 열화 조건 학습:** 다양한 노이즈·블러 패턴을 포함한 IPR 사전학습으로 일반화 강화  
- **Prior 표현 확장:** 다차원·계층적 IPR로 복원 세부 다양성 확보  
- **도메인 적응:** fine-tuning 없이 신규 의료영상·위성영상 등으로 신속 전이 학습

## 6. 향후 연구 영향 및 고려 사항
DiffIR은 IR에 특화된 디퓨전 접근을 제시하며,  
- **효율적 generative prior 활용** 연구 촉진  
- **디퓨전-트랜스포머 결합** 새로운 모델 설계 방향 제시  
- **의료영상, 영상 복원 분야의 실시간 적용 가능성** 모색  

향후 연구 시, IPR 표현의 **표현력-효율성 균형**과 **비정형 열화 대응**을 고려하는 것이 중요하다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/97c1e582-25de-4ae1-9975-d1bfac666485/2303.09472v3.pdf)
