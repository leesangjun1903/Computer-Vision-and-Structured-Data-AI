# PSLD : Solving Linear Inverse Problems Provably via Posterior Sampling with Latent Diffusion Models

# 핵심 요약

**주장:** 사전학습된 잠재 확산 모델(Latent Diffusion Models, LDM)을 활용하여 선형 역문제(linear inverse problems)를 이론적으로 보증 가능한 방식으로 해결하는 최초의 프레임워크를 제시한다.[1]
**주요 기여:**  
1. 픽셀 공간이 아닌 잠재 공간에서 작동하는 확산 후방 샘플링 알고리즘인 PSLD(Posterior Sampling with Latent Diffusion)를 제안.  
2. 선형 모델 설정에서 샘플 복원을 보장하는 이론적 분석 수행.  
3. 랜덤·블록 인페인팅, 디노이징, 디블러링, 디스트라이핑, 슈퍼해상도 등 다양한 문제에서 최첨단 성능을 달성.[1]

***

## 1. 해결하고자 하는 문제

딥 생성 모델을 역문제(이미지 인페인팅·노이즈 제거·슈퍼해상도 등)에 활용하는 기존 방식은
- Supervised 학습으로 특정 문제에 맞춘 모델을 재훈련하거나  
- 픽셀 공간 확산 모델(pixel-space diffusion) 후방 샘플링(DPS, DDRM 등)을 사용  

두 가지로 나뉜다. 픽셀 공간 방식은 **고차원(ambient dimension)의 저주**에 직면하며, LDM과 같은 대형 생성 모델을 직접 활용하지 못한다.[1]

***

## 2. 제안 방법

### 2.1 모델 구조 및 알고리즘

1. **잠재 공간 확산 (LDM):**  
   - 인코더 $$E$$로 $$x_0$$를 잠재 $$z_0=E(x_0)$$로 맵핑  
   - 이 잠재에 Itô SDE 기반 확산 및 역확산을 적용  
   - 디코더 $$D$$로 이미지 복원  

2. **PSLD(Posterior Sampling with Latent Diffusion):**  
   확산 역과정에서 매 스텝마다 세 가지 업데이트를 수행한다:[1]
   1) **Denoising update:**  

$$z'_{i-1} = \sqrt{\alpha_i}z_i + \dots - \eta_i\nabla_{z_i}\|y - A\,D(\hat z_0)\|^2$$
   
   2) **Measurement consistency:**  

$$-\eta_i\nabla_{z_i}\|y - A\,D(\hat z_0)\|^2$$
   
   3) **Gluing update:**  

$$-\gamma_i\nabla_{z_i}\big\|\hat z_0 - E(A^T A x^*_0 + (I - A^T A)D(\hat z_0))\big\|^2$$  
   
   여기서 $$\hat z_0$$는 현재 스텝의 복원 잠재, $$y=A x_0$$는 관측치, $$x^*_0=D(\hat z_0)$$는 복원 이미지다.[1]

### 2.2 이론적 보증

- **정확한 샘플 복원:** 두 단계 확산(two-step diffusion)·선형 생성 모델 설정 하에서, 임의의 양의 스텝 크기 $$\eta$$로도 진정한 후방 분포 $$p(x_0\mid y)$$에서 샘플링됨을 보인다(정리 3.8).[1]
- **고차원 저주 회피:** 픽셀 공간이 아닌 잠재 차원에서 연산하므로, 계산 복잡도가 $$d$$가 아닌 $$k\ll d$$에 의존.[1]

***

## 3. 성능 향상 및 한계

### 3.1 실험 결과

- **FFHQ 256 검증 세트:**  
  - 랜덤 인페인팅에서 FID 21.34→33.48 (DPS)→PSLD, LPIPS 0.096→0.212 등 모든 지표에서 우수.[1]
  - 블록 인페인팅, 슈퍼해상도(4×), 가우시안 디블러 등 다양한 역문제에서 SOTA 성능 달성.[1]

- **일반화 성능:**  
  - OOD(이미지넷·웹) 샘플에서도 튜닝 없이 Stable Diffusion 기반 PSLD가 DPS 대비 우수.[1]
  - LDM-VQ-4 기반 PSLD조차 DPS 대비 동등 이상 성능, 강건한 일반화 입증.[1]

### 3.2 한계

- **데이터·모델 바이어스:** Stable Diffusion 학습 데이터(LAION)의 편향이 결과에 영향.  
- **비선형 역문제:** 본 연구는 선형 관측( $$y=Ax$$ )에 국한. 비선형 확장 필요.  

***

## 4. 일반화 성능 향상 가능성

- **대형 프리트레인 모델 활용:** PSLD는 모델 훈련 없이도 최신 대규모 LDM(예: Stable Diffusion) 사용 가능. 모델 성능 향상→역문제 성능 직결.  
- **단일 스텝 크기:** 임의의 $$\eta$$로 보장되므로, 하이퍼파라미터 튜닝 부담 경감.  
- **차원 축소:** 잠재 표현의 낮은 차원에서 역문제를 풀어, 과적합·노이즈 영향 완화.

***

## 5. 향후 연구 및 고려 사항

- **비선형 관측 확장:** DPS 기반 비선형 역문제 기법과 PSLD 통합 연구.  
- **바이어스 완화:** LAION 후속 데이터셋(DataComp 등)으로 기반 모델 재평가.  
- **다중 모달:** 비전 이외 오디오·의료 영상·시계열 데이터에 대한 잠재 확산 역문제 적용.  
- **효율화:** 샘플링 속도 개선(스텝 수·계산량 최적화) 및 실시간 복원 응용.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/53e91d77-3a7f-45a1-87f4-19f45ddbd332/2307.00619v1.pdf)
