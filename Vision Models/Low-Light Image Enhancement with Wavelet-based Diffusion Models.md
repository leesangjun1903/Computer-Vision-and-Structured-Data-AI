# Low-Light Image Enhancement with Wavelet-based Diffusion Models

**핵심 요약**  
Wavelet 기반 확산 모델(WCDM)을 도입하여 저조도 영상의 전역·국부 정보를 분리 처리함으로써 기존 확산 모델의 느린 연산 속도와 불안정한 복원 문제를 해결하고, 화질·효율성 모두에서 SOTA 성능을 달성한다.

***

## 1. 해결하려는 문제  
저조도 영상은 음영 구역의 디테일 소실, 과도·과소 노출, 색 왜곡 등으로 인해 시각 품질이 떨어지며, 객체 인식·분류·탐지 등 후속 컴퓨터 비전 과제의 성능까지 저하시킨다.  
기존 확산 모델은  
- 반복적 노이즈 제거 과정으로 매우 느리고  
- 무작위 초기 노이즈에서 출발해 복원 결과의 불안정성(콘텐츠 불일관성) 발생  
문제점을 갖는다.  

***

## 2. 제안 방법  
### 2.1 Wavelet 기반 도메인 분리  
입력 저조도 이미지 $$I_{low}$$에 2D DWT(Haar) 변환을 $$K$$단계 적용해  

$$
A_{low},\, V_{low},\, H_{low},\, D_{low}
$$  

평균 계수(전역 정보)와 세 방향 고주파 계수(국부 디테일)를 분리한다.  
공간 해상도는 $$4^K$$배 감소하지만 정보 손실 없이 연산량을 크게 줄일 수 있다.

### 2.2 Wavelet-based Conditional Diffusion Model (WCDM)  
- **확산 대상**: 평균 계수 $$A$$  
- **학습 전략**:  
  - 학습 시 순방향 확산(forward diffusion)과 역확산(denoising) 모두 수행하여, 추론 시 랜덤성으로 인한 콘텐츠 불안정성 제거  
  - 손실 함수:  

$$
      \mathcal{L}_{\text{diff}} = \| \epsilon - \epsilon_\theta(A_t,t)\|_2^2 
      + \| A_0 - \hat A_0\|_2^2
    $$  

- **추론**: 학습된 모델의 역확산만 수행  

### 2.3 High-Frequency Restoration Module (HFRM)  
고주파 계수 $$(V,H,D)$$를 재구성하기 위해  
1) 수평/수직 계수 간 상호 보완용 cross-attention  
2) progressive dilation ResBlock  
3) depth-wise separable convolution  
으로 디테일을 정교하게 복원  

***

## 3. 성능 및 한계  
### 3.1 화질 평가  
- 주요 데이터셋(LOLv1/LOLv2/LSRW)에서 PSNR, SSIM, LPIPS, FID 모두 SOTA 달성.[1]
- 초고해상도(UHD-LL)에서도 4K 복원 시 기존 모델보다 높은 PSNR·SSIM 확보.  
- 비참조 지표(NIQE/BRISQUE/PI)에서 평균 최저값 달성, 일반화 능력 입증.

### 3.2 효율성  
- 600×400 영상 복원에 0.157초·1.85 GB 사용, 확산 기반 모델 대비 70배 빠르고 메모리 3 GB 절감.  
- 4K 해상도 복원 시에도 주요 경쟁 모델은 OOM 발생, 본 기법만 안정적 처리 가능.

### 3.3 제약  
- 극저조도 환경(정보 손실 과도)에서는 복원 품질 감소.  
- 실시간 비디오 강화에는 아직 연산 효율 부족.  
- 다른 복원(task) 적용 시 사전 학습 필요성 존재.

***

## 4. 일반화 성능 향상 관점  
- Wavelet 변환으로 공간 정보 축소 시에도 전역 평균 계수에 확산 모델 학습 집중 → 과적합 방지  
- 학습 단계에서 역확산을 수행해 추론 시 콘텐츠 일관성 보장 → 랜덤성 억제  
- HFRM 교차 주의(attention) 설계로 다중 방향 디테일 상호 보완 → 다양한 장면에 강건  

이로써 학습 데이터와 다른 조건에서도 색·명암·디테일 복원 능력이 향상되며, 실제 야간 영상(얼굴 탐지) 전처리 성능 개선에도 기여함.

***

## 5. 향후 연구 과제  
- 극저조도(under-exposed) 및 과노출(over-exposed) 조건별 특화 모듈 개발  
- 실시간 비디오 처리용 경량화 및 연산 최적화  
- 본 프레임워크의 제거·복원·합성 등의 다른 영상 복원 과제(흐림, 우천, 안개)로 확장 검증  
- 다양한 wavelet 기저 및 adaptive scale 선택 기법 탐색  

***

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/dbf5f881-4707-4ddd-b4bb-41cc16a5fceb/2306.00306v3.pdf)
