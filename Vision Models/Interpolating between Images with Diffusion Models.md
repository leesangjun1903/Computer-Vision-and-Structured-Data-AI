
# Interpolating between Images with Diffusion Models

## 1. 핵심 주장 및 기여

본 논문은 MIT CSAIL의 Clinton J. Wang과 Polina Golland이 2023년 7월 발표한 연구로, **잠재 확산 모델(Latent Diffusion Models, LDM)을 활용하여 실제 이미지 간의 고품질 보간(interpolation)**을 최초로 수행하는 zero-shot 방법을 제안합니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/0f7124bd-05b5-4bec-ba82-78d2119a39df/2307.12560v1.pdf)

이 논문의 핵심 기여는 다음과 같습니다:

- **문제 해결**: 기존 이미지 보간 기술이 갖는 한계(GAN 기반 방법의 낮은 실제 이미지 재구성 능력, 비디오 보간의 스타일 차이 미처리)를 극복
- **Zero-shot 접근법**: 추가 학습 없이 사전학습된 Stable Diffusion 모델 활용으로 다양한 도메인의 이미지 쌍 보간 가능
- **다중 조건 제어**: 텍스트 프롬프트, 포즈 정보, CLIP 스코어링을 결합하여 의미론적으로 일관성 있는 보간 달성
- **새로운 평가 문제 제시**: FID(Fréchet Inception Distance)와 PPL(Perceptual Path Length) 같은 표준 메트릭이 보간 품질을 제대로 평가하지 못함을 밝힘

## 2. 해결하고자 하는 문제

### 2.1 기존 방법의 한계

논문은 세 가지 주요 문제를 식별합니다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/0f7124bd-05b5-4bec-ba82-78d2119a39df/2307.12560v1.pdf)

1. **GAN 기반 보간의 한계**: StyleGAN과 같은 모델은 학습된 이미지 다양체의 부분집합(예: 얼굴)만 효과적으로 표현하며, 임의의 실제 이미지 재구성에 실패
2. **비디오 보간 기술의 부족**: 연속 모션은 처리하지만 스타일 변화를 동반한 부드러운 전환 불가능
3. **스타일 전환 기술의 한계**: 점진적으로 스타일과 콘텐츠를 동시에 변환하지 못함

### 2.2 응용 분야

논문은 이미지 보간이 다음 분야에서 창의적 가치를 제공한다고 주장합니다:
- 예술 및 미디어 제작
- 디자인 프로세스
- 애니메이션 생성
- 스타일 간 부드러운 전환 시각화

## 3. 제안하는 방법 및 수식

### 3.1 잠재 확산 모델 기초

논문의 방법은 다음 수식으로 표현되는 LDM 구조에 기반합니다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/0f7124bd-05b5-4bec-ba82-78d2119a39df/2307.12560v1.pdf)

$$z_t = \alpha_t z_0 + \sigma_t \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

여기서:
- $z_0$: 원본 이미지의 잠재 벡터 (인코더 $E$에 의해 생성)
- $z_t$: 타임스텝 $t$에서의 노이즈 추가된 잠재 벡터
- $\alpha_t, \sigma_t$: 스케줄 파라미터
- $\epsilon$: 표준 정규분포 노이즈

확산 U-Net은 다음과 같이 정의됩니다:

$$\epsilon_\theta: (z_t; t, c_{\text{text}}, c_{\text{pose}}) \rightarrow \hat{\epsilon}$$

### 3.2 Add Noise-Interpolate-Denoise (ANID) 전략

논문의 핵심 기여는 다음 브랜칭 구조를 통한 계층적 보간입니다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/0f7124bd-05b5-4bec-ba82-78d2119a39df/2307.12560v1.pdf)

**알고리즘 개요:**
1. 입력 이미지 0과 N을 타임스텝 $t_K$로 확산
2. 공유 노이즈를 추가하여 보간: $z^i_{t_K} = \text{slerp}(z^0_{t_K}, z^N_{t_K}, i/N)$
3. 역확산으로 중간 이미지 생성
4. 각 부모 쌍에 대해 재귀적으로 반복

구체적으로, 노이즈 추가 단계는:

$$z^0_t = \alpha_t z^0_0 + \beta_t \epsilon_t$$
$$z^N_t = \alpha_t z^N_0 + \beta_t \epsilon_t$$

여기서 $\epsilon_t \sim \mathcal{N}(0, I)$는 **두 이미지 간 공유되는 노이즈**입니다. 이는 형제 이미지 분리를 유도하면서도 의미론적 연속성을 보장합니다.

### 3.3 텍스트 보간 (Textual Inversion)

두 이미지에 대한 텍스트 임베딩을 적응화하는 손실 함수: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/0f7124bd-05b5-4bec-ba82-78d2119a39df/2307.12560v1.pdf)

$$\mathcal{L}(c_{\text{text}}) = \|\hat{\epsilon}_\theta(\alpha_t z_0 + \sigma_t \epsilon; t, c_{\text{text}}) - \epsilon\|^2$$

최적화: 100-500 반복, 학습률 $10^{-4}$, 배치당 손실 계산

### 3.4 구면 선형 보간 (Slerp)

잠재 공간과 텍스트 임베딩 보간에 적용:

$$\text{slerp}(v_1, v_2, t) = \frac{\sin((1-t)\Omega)}{\sin(\Omega)} v_1 + \frac{\sin(t\Omega)}{\sin(\Omega)} v_2$$

여기서 $\Omega = \arccos(v_1 \cdot v_2 / (\|v_1\| \|v_2\|))$

### 3.5 노이즈 스케줄

DDIM 샘플링과 함께 사용되는 선형 노이즈 스케줄: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/0f7124bd-05b5-4bec-ba82-78d2119a39df/2307.12560v1.pdf)
- 최소 200개 타임스텝 권장 (그 이하는 품질 저하)
- 유효 범위: 25%~65%의 스케줄 (외부 범위는 알파 합성 또는 과도한 편차 초래)
- 부드러운 보간을 위해 $\text{frame schedule}(i)$는 경계에서 단조 감소

## 4. 모델 구조

### 4.1 전체 파이프라인

[그림 2 참조] 논문의 파이프라인은 4단계로 구성됩니다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/0f7124bd-05b5-4bec-ba82-78d2119a39df/2307.12560v1.pdf)

| 단계 | 설명 | 입력 | 출력 |
|------|------|------|------|
| **잠재 보간** | ANID 스킴으로 노이즈 추가 후 보간 | 원본 이미지 쌍 | 보간된 잠재 벡터 |
| **텍스트 조건화** | Textual Inversion으로 각 이미지별 임베딩 최적화 | 텍스트 프롬프트 | 적응화된 텍스트 임베딩 |
| **포즈 조건화** | OpenPose 검출 후 선형 보간, ControlNet으로 조건화 | 포즈 이미지 | 제어된 포즈 시퀀스 |
| **CLIP 선택** | 다중 후보 생성 및 CLIP 점수로 선별 (선택사항) | 노이즈 벡터 집합 | 최고 품질 이미지 |

### 4.2 핵심 컴포넌트

**인코더-디코더:**
- 인코더 $E: x \rightarrow z_0$ (VAE 기반)
- 디코더 $D: z_0 \rightarrow \hat{x}$

**U-Net 아키텍처:**
- 타임스텝 임베딩
- Cross-attention 레이어 (텍스트 조건)
- ControlNet 통합 (포즈 조건)

**ControlNet 모듈:**
- OpenPose로 추출한 키포인트를 제어 신호로 변환
- 연속적 포즈 변환으로 해부학적 오류(다중 팔, 얼굴) 방지

### 4.3 비교 대상 방법들

논문은 4가지 보간 스킴을 비교합니다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/0f7124bd-05b5-4bec-ba82-78d2119a39df/2307.12560v1.pdf)

| 방법 | 수식 | 특성 | FID | PPL |
|------|------|------|-----|-----|
| **Interpolate Only** | $z_i^0 = \text{slerp}(z^0_0, z^N_0, i/N)$ | 순수 잠재 보간 | 436 | 56±8 |
| **Interpolate-Denoise** | 공유 노이즈, 각 프레임별 타임스텝 | 부분 결합 | 179 | 172±32 |
| **Denoise-Interpolate-Denoise (DID)** | 계층적 역확산 후 보간 | 중간 결합 | 169 | 144±26 |
| **DID w/o Shared Noise** | 독립 노이즈 사용 | 형제 분리 | 199 | 133±22 |
| **ANID (제안)** | 공유 노이즈 + 계층적 구조 | **창의적 변환** | **214** | **193±27** |

## 5. 성능 향상 및 일반화

### 5.1 성능 특성

논문의 ANID 방법은 정량적 메트릭에서는 더 높은 값을 보이지만, **질적으로는 우월**합니다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/0f7124bd-05b5-4bec-ba82-78d2119a39df/2307.12560v1.pdf)

- **FID 216 vs DID 169**: ANID가 더 창의적인 이미지를 생성하여 분포 거리 증가
- **PPL 193±27 vs DID 144±26**: 더 큰 의미론적 변화로 인한 지각 거리 증가
- **논문의 주장**: 표준 메트릭이 "알파 블렌딩 같은" 단순 보간을 선호하며, 실제 품질 평가에 부적절

### 5.2 일반화 성능 향상 가능성

#### 5.2.1 Zero-shot 학습 능력

논문의 핵심 강점은 **추가 학습 없이 다양한 도메인에 적용** 가능합니다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/0f7124bd-05b5-4bec-ba82-78d2119a39df/2307.12560v1.pdf)
- 사진, 로고, UI, 미술작품, 광고, 만화, 게임 이미지 등 26개 다양한 이미지 쌍 실험
- 각 쌍별 하이퍼파라미터 조정 최소화
- 단일 모델로 광범위한 스타일과 콘텐츠 처리

#### 5.2.2 조건화 메커니즘의 확장성

포즈 조건화의 흥미로운 발견: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/0f7124bd-05b5-4bec-ba82-78d2119a39df/2307.12560v1.pdf)

> "포즈가 잘못 예측되더라도 포즈 조건화가 인접 프레임 간 갑작스러운 포즈 변화를 방지하여 더 나은 보간을 생성한다."

이는 **노이즈가 있는 조건 정보도 구조 유지에 도움**임을 시사하며, 일반화 강건성을 증가시킵니다.

#### 5.2.3 스타일-콘텐츠 분리

텍스트 임베딩 보간을 통해 스타일과 콘텐츠가 **점진적으로 변환**되도록 제어:
- 긍정/부정 프롬프트 모두 최적화
- 부정 프롬프트 공유로 일관성 유지
- 의미론적 정보 보존하면서 외형 변화

### 5.3 확장성 분석

#### 해상도 확장
논문은 고해상도 이미지에는 언급이 제한적이지만, LDM 기반 아키텍처의 스케일링 가능성은 확인:
- Stable Diffusion v2.1의 512×512 해상도 사용
- 더 높은 해상도는 계산 비용 증가

#### 도메인 전이
다양한 시각 도메인 간 전이 성공: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/0f7124bd-05b5-4bec-ba82-78d2119a39df/2307.12560v1.pdf)
- 실제 사진 ↔ 만화
- 로고 ↔ 자연 이미지
- 회화 스타일 ↔ 사진

## 6. 모델의 한계

### 6.1 명시적 한계

논문이 보고한 실패 사례: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/0f7124bd-05b5-4bec-ba82-78d2119a39df/2307.12560v1.pdf)

1. **큰 스타일 차이**: 매우 다른 스타일 간 보간 실패
2. **포즈 감지 실패**: 비정상적인 스타일(만화, 비인간)에서 OpenPose 신뢰도 저하
3. **의미론적 매핑 오류**: 객체 간 의미론적 대응을 이해하지 못함
4. **해부학적 오류**: 포즈 조건화에도 불구하고 추가 팔/다리 생성 가능
5. **텍스트 아티팩트**: 생성된 이미지에 부정확한 텍스트 삽입

### 6.2 내재적 한계

#### 평가 메트릭의 부재
- FID/PPL이 보간 품질을 제대로 평가하지 못함
- **"이는 향후 작업에서 해결해야 할 중요한 질문"** [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/0f7124bd-05b5-4bec-ba82-78d2119a39df/2307.12560v1.pdf)

#### 의미론적 일관성
- 두 이미지 간 객체 대응이 명확하지 않은 경우 실패
- 예: 완전히 다른 객체 간 보간

#### 계산 비용
- 다중 후보 생성과 CLIP 평가로 인한 추가 계산
- DDIM 200+ 타임스텝 필요로 비교적 느린 생성

## 7. 최신 관련 연구 비교 (2020년 이후)

### 7.1 직접 관련 논문들 (2023-2025)

| 논문 | 게시 | 핵심 기여 | 대비 본 논문 |
|------|------|---------|-----------|
| **NoiseDiffusion** (2024) | ICLR 2024 | 노이즈 유효성 검증 및 정정 메커니즘 | 본 논문보다 구체적인 노이즈 문제 진단 |
| **AID (Attention Interpolation)** | NeurIPS 2024 | 어텐션 레이어 보간으로 일관성/부드러움 향상 | 조건부 보간에 더 정교한 어텐션 제어 |
| **DreamMover** | ICCV 2024 | 대 모션 보간 시 의미론적 일관성 유지 | 큰 포즈 변화에 특화, 플로우 추정 통합 |
| **Motion-aware Latent Diffusion** | 2024 | 비디오 프레임 보간에 모션 정보 통합 | 비디오 시퀀스에 최적화 |
| **Adapting Image-to-Video Diffusion** | 2024 | 이미지-비디오 모델 재적응으로 대 모션 처리 | 대규모 모델 활용, 더 나은 FVD 성능 |

### 7.2 기초 기술 진전

**Textual Inversion 발전 (2023-2024):**
- 기본 TI에서 점진적 개선 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/0f7124bd-05b5-4bec-ba82-78d2119a39df/2307.12560v1.pdf)
- Compositional Inversion (2023): 합성 임베딩의 과적합 문제 해결
- Specialist Diffusion (2023): 극소수 샘플로 스타일 학습

**ControlNet 확장 (2023-2025):**
- 원본 ControlNet (2023): 기본 포즈 제어 [arxiv](https://arxiv.org/abs/2302.05543)
- HumanSD, Stable-Pose, Uni-ControlNet (2024-2025): 더 정교한 포즈 정렬
- Reinforcement Learning 기반 미세조정 (2025): 포즈 정확도 향상 [openaccess.thecvf](https://openaccess.thecvf.com/content/WACV2025W/ImageQuality/papers/Lee_Improving_Human_Pose-Conditioned_Generation_Fine-tuning_ControlNet_Models_with_Reinforcement_Learning_WACVW_2025_paper.pdf)

**노이즈 스케줄 최적화 (2024-2025):**
- Zero Terminal SNR (2024): 샘플링 안정성 개선 [openaccess.thecvf](https://openaccess.thecvf.com/content/WACV2024/papers/Lin_Common_Diffusion_Noise_Schedules_and_Sample_Steps_Are_Flawed_WACV_2024_paper.pdf)
- Improved Noise Schedule (2025): 학습 효율성 증진 [openaccess.thecvf](https://openaccess.thecvf.com/content/ICCV2025/papers/Hang_Improved_Noise_Schedule_for_Diffusion_Training_ICCV_2025_paper.pdf)
- Optimal Stepsize Distillation (2025): 동적 프로그래밍 기반 최적 스케줄 [arxiv](https://arxiv.org/html/2503.21774v1)

### 7.3 평가 메트릭 발전

본 논문이 지적한 FID/PPL의 한계는 커뮤니티에서도 인식: [arxiv](https://arxiv.org/abs/2406.09358)

- **Mode Interpolation 연구 (2024)**: 확산 모델의 "환각" 현상이 실제로는 **학습 데이터 모드 간 보간**임을 밝힘
- **Gini Coefficient 기반 평가**: 보간 부드러움을 정량화하는 새로운 메트릭 도입 (NeurIPS 2024 AID 논문)
- **개념적 일관성 메트릭**: 보간 시퀀스의 의미론적 흐름 평가

## 8. 향후 연구에 미치는 영향

### 8.1 직접적 영향

#### 1) 조건화 메커니즘 개선
본 논문의 다중 조건화(텍스트+포즈)가 후속 연구로 확장:
- AID (2024): 어텐션 기반 조건 보간
- DreamMover (2024): 플로우 기반 의미론적 대응
- Reinforcement Learning 기반 ControlNet 미세조정

#### 2) Zero-shot 일반화 탐구
- 본 논문이 **재학습 없이 다양한 도메인 처리 가능**함을 입증
- 이후 연구들이 zero-shot 구조 보존(CVPR 2024), zero-shot 비디오 강우 제거(2025) 등으로 확대

#### 3) 평가 메트릭 재검토
본 논문의 비판(FID/PPL 부족) → NeurIPS 2024의 새로운 평가 프레임워크 개발

### 8.2 근본적 이론적 기여

#### Score Smoothing과 생성 능력
논문의 발견(공유 노이즈 + 보간 = 의미론적 연속성)은 다음을 시사: [arxiv](https://arxiv.org/html/2502.19499v1)

$$\text{생성 능력} \propto \text{Score Function Smoothing}$$

즉, 확산 모델의 외삽(interpolation) 능력은:
- 학습된 점수 함수의 매끄러움에서 비롯
- 훈련과 추론 사이의 분포 불일치 극복 메커니즘

## 9. 향후 연구 시 고려할 점

### 9.1 기술적 고려사항

#### 1) 정량화 방법 개발
**문제**: FID (214) vs DID (169)인데, 질적으로는 ANID가 우수
**해결책**:
- User study 기반 평가 프레임워크 구축
- 의미론적 일관성 점수 (Gini coefficient, 지각 경로 길이 가중치 조정)
- 다중 참고자 기반 보간 품질 지표

#### 2) 스타일-콘텐츠 분리 고도화
**현재 한계**: 텍스트 임베딩 단순 보간
**개선 방향**:
- Disentangled 임베딩 학습 (스타일/콘텐츠 명시 분리)
- Compositional Inversion 활용으로 더 정밀한 제어

#### 3) 포즈 정확도 개선
**발견**: 부정확한 포즈도 일반적 구조 유지에 도움 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/0f7124bd-05b5-4bec-ba82-78d2119a39df/2307.12560v1.pdf)
**확장 연구**:
- 부정확한 조건의 견강성 정량화
- RL 기반 포즈 미세조정 (2025 연구 참조) [openaccess.thecvf](https://openaccess.thecvf.com/content/WACV2025W/ImageQuality/papers/Lee_Improving_Human_Pose-Conditioned_Generation_Fine-tuning_ControlNet_Models_with_Reinforcement_Learning_WACVW_2025_paper.pdf)

### 9.2 확장성 및 적용

#### 1) 고해상도 확장
- Latent Wavelet Diffusion (2025): 초고해상도 생성 [arxiv](https://arxiv.org/html/2506.00433v3)
- SVG-T2I (2024): VFM 특성 공간에서 고품질 합성 [arxiv](https://arxiv.org/html/2512.11749v1)

#### 2) 동적 확장
**영상 보간으로의 확장**:
- Motion-aware Latent Diffusion: 동작 정보 통합 [arxiv](https://arxiv.org/html/2404.13534v3)
- Adapting Image-to-Video: 대 모션 처리 [arxiv](https://arxiv.org/abs/2412.17042)
- 기술적 고려: 시간 일관성 유지, 플로우 기반 의미론적 대응

#### 3) 다중 모달 조건화
- 현재: 텍스트 + 포즈
- 향후: 세분화 맵, 바운딩 박스, 깊이 맵 등 다중 공간 조건

### 9.3 이론적 진전

#### 1) 일반화 메커니즘 이해
**핵심 질문**: 왜 공유 노이즈 + 보간이 작동하는가?
- 확산 프로세스의 보간 특성 수학적 모델링
- Score function의 연속성과 생성 능력의 관계 [arxiv](https://arxiv.org/html/2502.19499v1)

#### 2) 분포 외 일반화
**문제**: 학습 데이터에 없는 객체 쌍 보간
**접근**:
- 구조 보존 확산 모델 (CVPR 2024) - 구조 명시 보존
- 의미론적 대응 학습 (DreamMover - 플로우 기반)

#### 3) 계산 효율성
**최신 진전**:
- DDIM 스텝 최적화, [openaccess.thecvf](https://openaccess.thecvf.com/content/ICCV2025/papers/Hang_Improved_Noise_Schedule_for_Diffusion_Training_ICCV_2025_paper.pdf)
- 빠른 역변환 (Lightning-Fast Newton-Raphson, 2024) [arxiv](https://arxiv.org/pdf/2312.12540.pdf)
- 흐름 기반 모델과의 비교

## 10. 결론

### 주요 성과

Wang & Golland (2023)의 연구는 **확산 모델 기반 실제 이미지 보간의 첫 성공적 사례**로서:

1. **기술적**: Zero-shot 다중 조건화로 광범위한 도메인 적용 가능성 입증
2. **이론적**: 공유 노이즈 + 계층적 보간이 의미론적 연속성 생성 메커니즘 제시
3. **비판적**: 표준 평가 메트릭의 한계 지적으로 커뮤니티의 메트릭 개선 촉발

### 학문적 위상

이 논문은 다음의 기초를 마련했습니다:

| 분야 | 영향 |
|------|------|
| **조건화 기술** | 텍스트+포즈 결합 → 어텐션 보간(AID) → RL 기반 미세조정 |
| **보간 이론** | 의미론적 보간 → 플로우 기반 대응(DreamMover) → 동작 인식(Motion-aware) |
| **평가 방법** | FID 비판 → Gini 계수 + 지각 경로 길이 → 사용자 연구 기반 평가 |
| **Zero-shot 학습** | 보간 적용 → 구조 보존 → 비디오 강우 제거 등 다양한 작업 확대 |

### 미해결 문제와 향후 방향

1. **평가 메트릭**: 보간 품질을 포괄하는 정량화 방법 여전히 미흡
2. **의미론적 대응**: 완전히 다른 객체 간 자동 대응 불가능 → 플로우/의미론적 일관성 학습 필요
3. **계산 효율**: DDIM 200+ 스텝 필요 → 지식 증류, 흐름 기반 모델 탐색
4. **고해상도 확장**: 원본 512×512 제한 → 최신 고해상도 모델 통합 필요

***

## 참고문헌 및 인용

 Wang, C. J., & Golland, P. (2023). Interpolating between Images with Diffusion Models. arXiv:2307.12560 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/0f7124bd-05b5-4bec-ba82-78d2119a39df/2307.12560v1.pdf)
 Song, J., Meng, C., & Ermon, S. (2022). Denoising Diffusion Implicit Models. [arxiv](https://arxiv.org/abs/2409.09605)
 Rombach, R., et al. (2022). High-Resolution Image Synthesis with Latent Diffusion Models. CVPR 2022. [arxiv](https://arxiv.org/abs/2403.08840)
 Hang, T., et al. (2024). Adapting Image-to-Video Diffusion Models for Large-Motion Frame Interpolation. arXiv:2412.17042 [arxiv](https://arxiv.org/abs/2412.17042)
 Aithal, G., et al. (2024). Understanding Hallucinations in Diffusion Models through Mode Interpolation. NeurIPS 2024. [arxiv](https://arxiv.org/abs/2406.09358)
 Motion-aware Latent Diffusion Models (2024). arXiv:2404.13534 [arxiv](https://arxiv.org/html/2404.13534v3)
 On the Interpolation Effect of Score Smoothing (2025). arXiv:2502.19499 [arxiv](https://arxiv.org/html/2502.19499v1)
 Latent Wavelet Diffusion For Ultra-High-Resolution Image Synthesis (2025). [arxiv](https://arxiv.org/html/2506.00433v3)
[45-87] 최신 텍스추얼 인버전, ControlNet, DDIM, 포즈 조건화 관련 논문들
