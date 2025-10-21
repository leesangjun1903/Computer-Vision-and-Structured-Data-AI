# VmambaIR: Visual State Space Model for Image Restoration

# 핵심 요약

**VmambaIR**은 이미지 복원(Image Restoration) 분야에서 **선형 복잡도**를 가지는 **State Space Model(SSM)**을 도입하여, 기존 CNN, Transformer, Diffusion 모델의 한계를 극복한 **Visual State Space Model** 기반 네트워크이다.  
주요 기여는 다음과 같다:[1]
- Unet 구조에 **Omni Selective Scan(OSS) 블록** 삽입으로 6방향 정보 흐름 모델링  
- 효율적 피드포워드 네트워크(EFFN) 도입으로 계층 간 정보 조율  
- SOTA 성능 달성: 파라미터 및 연산량 대폭 절감하며 다양한 복원 과제에서 우수한 복원 품질  

# 문제 정의 및 제안 방법

## 해결하고자 하는 문제  
저해상도·노이즈·비선형 왜곡 등이 있는 입력 이미지에서 장거리 의존성(Long-range dependency)을 고비용 연산 없이 포착하고, 공간적 패턴을 균형 있게 복원하는 것.

## 모델 공식 및 구조  
1. **SSM 기초 방정식**  

$$
     \dot h(t) = Ah(t) + Bx(t), \quad y(t) = Ch(t) + Dx(t)
   $$  
   
연속 시스템을 이산화 시킨 후 ZOH(Zero-Order Hold)로 변환:[1]
   
$$
     \overline{A} = e^{A\Delta t}, \quad \overline{B} = A^{-1}(e^{A\Delta t}-I)B
   $$  

2. **네트워크 아키텍처**  
   - 입력 이미지 $$I_{in}\in\mathbb{R}^{H\times W\times3}$$을 shallow conv으로 임베딩  
   - **Unet** 기반 인코더–디코더: 각 레벨에 **OSS 블록** 배치, 멀티스케일 특징 추출  
   - 출력 후 tail 블록으로 해상도 복원 또는 레지듀얼 계산[1]

3. **OSS 블록 구성**  
   - **OSS 모듈**: 2개의 정보 흐름(Depth-wise conv + SiLU, SiLU) → **Omni Selective Scan** 적용  
   - **Efficient Feed-Forward Network(EFFN)**: LayerNorm → 1×1 Conv → Depth-wise Conv + Gated SiLU → 1×1 Conv[1]

4. **Omni Selective Scan**  
   - 평면(행·열) 및 채널 차원에서 **정·역방향 스캔** 수행  
   - 네 방향(행 전/후, 열 전/후) 후 Mamba 블록으로 모델링 → 합산 및 곱셈 → 채널 스캔 → 통합  
   - 선형 복잡도 유지하며 6방향 정보 흐름 완전 모델링[1]

# 성능 및 한계

## 성능 향상  
- **4× 단일 이미지 초해상화**: Urban100 PSNR +0.48dB, LPIPS 최저 기록[1]
- **실세계 초해상화**: 연산량 26% 수준, 파라미터 절반 이하, LPIPS↓, PSNR↑, SSIM↑[1]
- **이미지 제이닝**: Rain100H PSNR +0.1dB, SSIM↑, 파라미터·연산 절감[1]

## 모델 한계  
- 스캔 연산에서 reshape·permute 비용으로 순수 conv 대비 속도 저하  
- 복잡한 스캔 메커니즘으로 특정 하드웨어 최적화 필요  
- 거친 노이즈 분포나 비정형 왜곡에 대한 일반화 추가 검증 필요  

# 일반화 성능 향상 관점

- **선형 복잡도**: 대형 이미지 및 고해상도 처리 시 연산 효율성  
- **멀티스케일 Unet 구조**: 다양한 해상도 특징 조화  
- **OSS 블록 & EFFN**: 계층별 정보 융합으로 과적합 완화  
- **양방향·다차원 스캔**: 공간·채널 패턴을 균형 있게 학습, 다양한 복원 과제에 일관된 성능 발휘  

# 향후 연구 영향 및 고려 사항

VmambaIR은 SSM의 가능성을 입증함으로써, 다음 연구에 중요한 이정표가 될 것이다:
- **경량화 SSM 아키텍처**: 스캔 연산 최적화 및 하드웨어 특화 연산 개발  
- **비정형 잡음·왜곡 대응**: 자가교차 어텐션, 노이즈 모델 통합 연구  
- **다중 모달 복원**: 비전+언어, 의료 영상 등 멀티채널 데이터에 SSM 적용  
- **일반화 검증**: 실제 촬영 환경별 평가, 도메인 적응 기법 결합  

이와 같은 방향은 차세대 저수준 비전 과제에서 SSM 기반 네트워크의 **효율성과 확장성**을 더욱 강화할 것이다.  

***
 1[1]

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/49a9e8fa-0e51-4aa1-90e2-a2a188f2c206/2403.11423v1.pdf)
