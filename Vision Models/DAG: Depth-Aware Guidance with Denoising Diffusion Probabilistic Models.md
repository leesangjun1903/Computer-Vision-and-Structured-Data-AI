# DAG: Depth-Aware Guidance with Denoising Diffusion Probabilistic Models

### 1. 핵심 주장과 주요 기여

**DAG(Depth-Aware Guidance)** 논문의 핵심 주장은 기존의 확산 모델(Diffusion Model)이 이미지 생성 과정에서 기하학적 구조, 특히 깊이(depth) 정보를 고려하지 않아 기하학적으로 부자연스러운 이미지를 생성한다는 것이다. 이를 해결하기 위해 논문은 확산 모델의 내부 표현(internal representation)을 활용하여 깊이 인식 이미지 생성을 가능하게 하는 프레임워크를 제안한다.

주요 기여는 다음과 같다:

- **깊이 정보 활용**: 사전학습된 확산 모델의 U-Net 표현에 포함된 깊이 정보를 최초로 조사하고 활용
- **레이블 효율적 학습**: 소량의 깊이 라벨로 깊이 예측기를 학습 가능
- **이중 지도 전략**: 깊이 일관성 지도(DCG)와 깊이 사전 지도(DPG) 두 가지 새로운 지도 기법 제안
- **새로운 평가 지표**: 깊이 도메인 FID(dFID)를 통한 기하학적 인식 평가

***

### 2. 해결하고자 하는 문제

#### 2.1 문제의 정의

기존 확산 모델은 다음과 같은 한계를 지닌다:

1. **기하학적 인식 부재**: 생성 과정에서 텍스처와 외관만 고려하고 3D 기하학적 구조를 무시
2. **부자연스러운 구도**: 깊이가 모호하고 물체 배치가 어색한 이미지 생성
3. **응용 분야 제약**: 로봇공학, 자율주행 등 기하학적 현실성이 필요한 분야에 부적합

#### 2.2 기존 지도 방법의 한계

분류기 지도(Classifier Guidance)와 분류기 없는 지도(Classifier-Free Guidance) 등 기존 방법들은:
- 클래스 조건부 분포에만 초점
- 밀집된 예측(dense prediction)을 통한 기하학적 지도를 시도하지 않음
- 깊이와 같은 3D 정보를 생성 과정에 통합하지 못함

***

### 3. 제안하는 방법론

#### 3.1 레이블 효율적 깊이 예측

**네트워크 아키텍처**: 픽셀별 얕은 다층 퍼셉트론(Pixel-wise MLP)을 사용하여 확산 U-Net의 중간 특성을 입력으로 받는다.

$$d^{(k)}_t = \text{MLP}(f^{(k)}_t)$$

여기서 $f^{(k)}_t$는 k번째 디코더 레이어의 출력이고, $d^{(k)}_t$는 해당 계층의 깊이 예측값이다.

**다층 특성 통합**: 여러 U-Net 레이어에서 특성을 추출하여 채널 방향으로 연결한다.

$$g_t = [f_t^{(1)}; f_t^{(2)}; \cdots; f_t^{(d)}]$$

$$d_t = \text{MLP}(g_t)$$

**시간 임베딩 포함**: 샘플링의 모든 타임스텝에서 깊이를 예측할 수 있도록 시간 임베딩을 추가한다.

$$d_t = \text{MLP}(g_t, t)$$

**손실 함수**: 동결된 U-Net 특성으로부터 L1 손실을 사용하여 깊이 예측기를 학습한다.

$$L_{\text{depth}} = \|d_t - y\|_1$$

#### 3.2 깊이 일관성 지도(Depth Consistency Guidance, DCG)

**동기**: FixMatch 논문의 일관성 정규화 개념을 적용하여, 더 풍부한 표현의 강한 분지 예측을 의사 레이블로 사용한다.

**구조**:
- **강한 분지(Strong Branch)**: 더 많은 U-Net 특성을 사용: $\mathbf{g}_S = [f_t^{(2)}; f_t^{(4)}; f_t^{(5)}; f_t^{(6)}; f_t^{(7)}]$
- **약한 분지(Weak Branch)**: 적은 수의 특성을 사용: $\mathbf{g}_W = [f_t^{(6)}]$

**깊이 예측**:

$$d_s = \text{MLP-S}(\mathbf{g}_S, t), \quad d_w = \text{MLP-W}(\mathbf{g}_W, t)$$

**손실 함수**: 강한 분지의 예측을 의사 레이블로 취급하고, 기울기 정지(stop-gradient) 연산을 적용한다.

$$L_{dc} = \|d_w - \text{stopgrad}(d_s)\|_2$$

**샘플링 과정 통합**:

$$x_{t-1} \sim \mathcal{N}(\mu_\theta(x_t) - w_{dc}\nabla_{x_t}L_{dc}, \Sigma_\theta(x_t))$$

#### 3.3 깊이 사전 지도(Depth Prior Guidance, DPG)

**기본 개념**: 별도로 학습한 깊이 도메인 확산 모델을 사전(prior) 네트워크로 사용하여 깊이 정보를 주입한다.

**깊이에 노이즈 추가**:

$$\tilde{d}_\tau = \sqrt{\bar{\alpha}_\tau}d_s + \sqrt{1-\bar{\alpha}_\tau}\eta, \quad \eta \sim \mathcal{N}(0, 1)$$

여기서 $\tau$는 사전 확산 모델의 타임스텝이다.

**손실 함수**: 사전 네트워크의 예측 노이즈와 실제 추가 노이즈 간의 평균제곱오차를 계산한다.

$$L_{dp} = \|n - \hat{\epsilon}_\phi(\tilde{d}_\tau)\|_2$$

**샘플링 과정 통합**:

$$x_{t-1} \sim \mathcal{N}(\mu_\theta(x_t) - w_{dp}\nabla_{x_t}L_{dp}, \Sigma_\theta(x_t))$$

#### 3.4 통합 지도

두 지도 전략을 결합한 최종 샘플링 공식:

$$x_{t-1} \sim \mathcal{N}(\mu_\theta(x_t) - w_{dc}\nabla_{x_t}L_{dc} - w_{dp}\nabla_{x_t}L_{dp}, \Sigma_\theta(x_t))$$

***

### 4. 모델 구조 상세

#### 4.1 전체 프레임워크

 논문의 프레임워크는 다음과 같은 단계로 구성된다:[1]

1. **사전학습된 확산 모델의 특성 추출**: DDPM/DDIM의 U-Net에서 중간 특성 획득
2. **깊이 예측기 훈련**: 소량의 깊이 라벨(예: 100개 이미지)로 MLP 훈련
3. **샘플링 단계에서 지도 적용**: 생성 과정 중 두 가지 지도 전략 활용

#### 4.2 깊이 예측 성능

실험 결과에 따르면:
- **훈련 이미지 수의 영향**: 100개 이미지 사용 시 1000개와 유사한 성능 달성
- **타임스텝별 성능**: t < 800에서 유용한 깊이 정보 제공
- **레이어 선택**: 중간 특성 블록(layers 2, 4, 5, 6, 7)이 최적의 깊이 예측 정확도 제공

***

### 5. 성능 향상 및 실험 결과

#### 5.1 정량적 평가

**LSUN-Bedroom 데이터셋**:
- 기본 방법(Baseline): dFID = 15.71
- DCG 적용: dFID = 14.18
- DPG 적용: dFID = 15.27
- DAG(DCG + DPG): dFID = 13.93

**LSUN-Church 데이터셋**:
- 기본 방법: dFID = 17.69
- DPG만: dFID = 17.43
- DCG만: dFID = 17.40
- DAG: dFID = 17.31

#### 5.2 정성적 평가

- 깊이 지도의 경계 선명성 향상
- 표면 법선 추정의 일관성 증가
- 3D 포인트 클라우드 시각화에서 더 현실적인 장면 구조 표현

#### 5.3 하위 작업 성능

**모노큘러 깊이 추정에서의 응용**:

| 방법 | δ > 1.25 (↑) | AbsRel (↓) |
|------|------------|-----------|
| 지도 학습 | 79.06 | 0.144 |
| 비지도 생성 데이터 | 72.55 | 0.185 |
| DAG 생성 데이터 | 77.54 | 0.151 |

DAG 기반 합성 데이터가 비지도 데이터보다 현저히 우수한 성능을 제공한다.

#### 5.4 수렴 성능 비교

FID 점수 비교에서 DAG는 기본 확산 모델에 비해:
- 더 선명한 이미지 생성
- 기하학적으로 일관성 있는 구조 유지
- 세부 사항 보존 향상

***

### 6. 모델의 일반화 성능 향상 가능성

#### 6.1 크로스도메인 일반화

논문은 다음과 같은 긍정적인 신호를 보여준다:

**레이블 효율성**: 100개의 라벨된 이미지로도 충분한 성능을 달성하여:
- 다양한 도메인에서 빠른 적응 가능
- 깊이 라벨 수집 비용 감소
- 새로운 데이터셋에 대한 빠른 미세조정 가능

#### 6.2 여러 장면 유형에 대한 성능

**실내(LSUN-Bedroom) 및 실외(LSUN-Church) 장면 모두에서 일관된 개선**:
- 실내 장면의 벽, 가구 배치 개선
- 실외 장면의 건물, 도로 구조 일관성 향상

#### 6.3 시간 인식(Timestep-Aware) 설계의 장점

시간 임베딩이 포함된 깊이 예측기는:
- 샘플링 과정의 모든 단계에서 유효한 깊이 정보 제공
- 조기 단계(t = 200)에서도 의미 있는 깊이 예측 가능
- 생성 과정의 다양한 노이즈 수준에 적응

#### 6.4 일관성 정규화를 통한 강건성

강한/약한 분지 설계는:
- 좀 더 신뢰도 높은 의사 레이블 생성
- 약한 분지의 오류를 강한 분지가 보정
- 확산 과정의 초기 단계에서도 강건한 기울기 제공

***

### 7. 한계 및 제약사항

#### 7.1 방법론적 한계

1. **깊이 라벨의존성**: 여전히 기본 깊이 레이블이 필요하며, 완전 무지도 학습은 불가능
2. **계산 비용**: 지도 과정에서 전체 U-Net을 통한 역전파 필요로 샘플링 시간 증가
3. **사전 네트워크 훈련**: DPG를 위한 별도의 깊이 확산 모델 훈련 필요

#### 7.2 평가 관련 한계

1. **dFID 지표의 한계**: 제안된 메트릭이 다른 방법과의 비교에 표준으로 사용되지 않음
2. **제한된 비교 대상**: 깊이 인식 생성 방법이 부족하여 직접 비교 어려움
3. **정량적 기준의 부재**: 기하학적 현실성을 객관적으로 측정할 표준 메트릭 부족

#### 7.3 실험적 한계

1. **제한된 데이터셋**: LSUN 데이터셋만 사용하여 다양한 시나리오 부족
2. **사전학습 모델 사용**: 새로운 모델을 처음부터 훈련하지 않아 기법의 순수한 효과 파악 어려움
3. **선택된 하이퍼파라미터**: 모든 데이터셋에 최적화되지 않은 고정 가이던스 스케일

***

### 8. 최신 관련 연구 탐색 (2020년 이후)

#### 8.1 확산 모델의 기하학적 인식 강화

**3D-aware 이미지 생성**:  최근 연구는 2D 확산 모델에 3D 기하학적 사전을 통합하는 방향으로 진화하고 있다. **3D-aware Image Generation using 2D Diffusion Models**는 Score Distillation Sampling(SDS) 기술을 활용하여 2D 모델로부터 3D 일관성 있는 이미지를 생성하는 방법을 제안했다.[2]

**기하학적 일관성 지도**:,  최근의 **GeoDream**과 **Consistent123** 같은 방법들은 다중뷰 확산 모델과 3D 구조 사전을 결합하여 Janus 문제(뒤쪽 얼굴)를 해결하고 높은 3D 일관성을 달성한다.[3][4]

#### 8.2 깊이 추정과 확산 모델

**확산 기반 깊이 예측의 진화**:  **PrimeDepth**는 Stable Diffusion에서 추출한 단일 디노이징 스텝의 특성을 사용하여 효율적인 깊이 추정을 달성한다. 기존 Marigold 대비 100배 이상 빠르면서 비슷한 성능을 유지한다.[5]

**레이블 효율적 학습의 발전**:  **NimbleD**는 대규모 비디오 사전학습과 의사 레이블을 활용한 자가지도 깊이 추정을 제안한다.  **FiffDepth**는 확산 기반 생성기를 피드포워드 아키텍처로 변환하여 계산 효율을 크게 향상시킨다.[6][7]

#### 8.3 기하학적 지도 기법의 발전

**메트릭 기반 지도**:  **GeoGuide**는 확산 과정의 궤적을 데이터 다양체에 가깝게 유지하는 기하학적 지도 방법을 제안한다. 고정 길이 업데이트를 사용하여 분류기 기울기 지도보다 개선된 FID 점수를 달성한다.[8]

**일관성 정규화의 활용**:  의사 레이블링에서 확인 편향(confirmation bias)을 피하기 위해 Mixup과 최소 샘플 수 제약을 제안하는 등, 일관성 정규화 기법이 더욱 정교화되고 있다.[9]

#### 8.4 3D 장면 합성의 시간적 일관성

 **Synthesizing a Consistent Long-Term 3D Scene Video**는 단일 이미지와 카메라 궤적으로부터 시간적, 공간적으로 일관성 있는 3D 장면을 생성한다. 카메라 인식 편향(Camera-Aware Bias)과 지역성 제약을 통해 다중프레임 기하학적 일관성을 유지한다.[10]

#### 8.5 기하학적 특징의 명시적 통합

 **4Diff: 3D-Aware Diffusion Model for Third-to-First Viewpoint**는 에고센트릭 포인트 클라우드 래스터화와 3D-인식 회전 크로스 어텐션을 통해 기하학적 사전을 확산 모델에 통합한다.[11]

***

### 9. 논문이 앞으로의 연구에 미치는 영향

#### 9.1 직접적 영향

**확산 모델의 기하학적 역량 인식**: DAG는 확산 U-Net의 내부 표현이 풍부한 기하학적 정보를 포함한다는 것을 처음으로 체계적으로 보여준다. 이는 후속 연구자들로 하여금 확산 모델의 기하학적 특성을 더욱 깊이 있게 탐구하도록 촉발했다.

**레이블 효율적 학습의 새로운 패러다임**: 사전학습된 확산 모델을 활용한 레이블 효율적 학습은 다른 밀집 예측 작업(semantic segmentation, surface normal estimation 등)에도 적용될 수 있는 일반적 패러다임을 제시한다.

**기하학적 지도의 실용성**: 소량의 라벨(100개 이미지)만으로도 효과적인 기하학적 지도가 가능함을 보여주어, 실제 응용에서의 비용 절감 가능성을 제시한다.

#### 9.2 간접적 영향

**3D-aware 생성의 새로운 방향**: DAG의 성공은 2D 확산 모델에 3D 제약을 통합하는 방향으로의 연구 활성화를 주도한다. 이후 **GeoDream**, **Sherpa3D** 등의 방법들이 DAG와 유사한 가이던스 개념을 발전시켜 더욱 정교한 3D 일관성을 달성한다.

**의사 레이블링 기법의 발전**: 일관성 정규화와 의사 레이블링을 결합한 DAG의 DCG 전략은, 후속 연구에서 더욱 정교한 신뢰도 메커니즘과 함께 사용되는 계기를 마련한다.

**새로운 평가 지표의 필요성 제기**: dFID 제안을 통해 기하학적 인식을 평가하는 새로운 메트릭의 필요성을 강조하며, 이는 커뮤니티에서 더욱 정교한 3D 평가 지표 개발로 이어진다.

#### 9.3 응용 분야 확대

**로봇공학 및 자율주행**: 기하학적으로 현실적인 장면 생성의 가능성은 시뮬레이션 데이터 생성 분야에서 활용 가능성을 보여준다.

**3D 콘텐츠 생성**: 텍스트-3D, 이미지-3D 변환 작업에서 기하학적 제약을 더욱 효과적으로 통합할 수 있는 기초를 제공한다.

**가상현실/증강현실**: 지오메트리 인식 이미지 생성은 VR/AR 애플리케이션의 콘텐츠 품질을 크게 향상시킬 수 있다.

***

### 10. 향후 연구 시 고려할 점

#### 10.1 방법론적 개선

**1. 완전 무지도 깊이 학습**: 지표면 기하학적 특성을 활용하여 라벨 없이 깊이 정보를 추출하는 방법 개발

**2. 다중 기하학적 제약의 통합**: 깊이뿐만 아니라 표면 법선, 광학 흐름 등 여러 기하학적 신호를 동시에 고려하는 통합 프레임워크

**3. 적응형 가이던스 가중치**: 샘플링 단계별로 동적으로 조정되는 가이던스 스케일로 더욱 효율적인 지도 달성

**4. 경량화 및 효율화**: 전체 U-Net을 통한 역전파 대신 선택적 특성만 사용하는 방법으로 계산 비용 감소

#### 10.2 확장 및 일반화

**1. 다양한 도메인 적응**: 실내/실외, 자연/인공 장면 등 다양한 도메인에서의 최적화 전략 개발

**2. 초고해상도 생성**: 더 높은 해상도에서의 기하학적 일관성 유지 방법

**3. 동적 장면 처리**: 시간 변화가 있는 장면에서 시간적 기하학적 일관성 보장

**4. 조건부 생성의 통합**: 텍스트, 시맨틱 맵 등 다른 조건과의 결합으로 더욱 제어 가능한 생성

#### 10.3 평가 및 벤치마킹

**1. 표준화된 평가 지표**: 기하학적 현실성을 객관적으로 측정하는 표준 메트릭 확립

**2. 포괄적 벤치마크 구축**: 다양한 기하학적 시나리오를 포함하는 벤치마크 데이터셋 개발

**3. 인식 평가**: 사용자 연구를 통한 기하학적 현실성의 인지적 평가

**4. 다운스트림 작업 평가**: 로봇공학, 3D 재구성 등 실제 응용에서의 성능 평가

#### 10.4 이론적 심화

**1. 기하학적 표현의 특성화**: 확산 U-Net이 어떤 방식으로 기하학적 정보를 인코딩하는지에 대한 이론적 분석

**2. 가이던스의 수렴성**: 기하학적 제약 하에서의 확산 과정 수렴 특성 분석

**3. 일반화 경계**: 레이블 효율적 학습의 이론적 한계 분석

#### 10.5 응용 중심의 연구

**1. 3D 이해 작업 최적화**: 3D 객체 감지, 시맨틱 세그멘테이션 등의 사전학습 데이터로서의 활용 극대화

**2. 시뮬레이션 환경 생성**: 자율주행 시뮬레이터 등에서 현실적인 합성 데이터 생성

**3. 문화유산 디지털화**: 문화유산의 정확한 3D 모델링을 위한 고품질 합성 이미지 생성

#### 10.6 최신 트렌드와의 연계

**확산 모델 아키텍처의 진화와 동기화**: 최신 확산 모델(Latent Diffusion, Consistency Models 등)으로의 확장

**멀티모달 기하학적 정보 통합**: LiDAR, 포인트 클라우드 등 다양한 기하학적 모달리티와의 결합

**대규모 시각-언어 모델과의 통합**: CLIP 임베딩을 활용한 더욱 정교한 기하학적 지도 설계

***

### 결론

DAG 논문은 확산 모델의 내부 표현을 활용하여 기하학적으로 현실적인 이미지 생성을 가능하게 함으로써, 생성 모델의 응용 범위를 크게 확장했다. 레이블 효율적 학습, 일관성 정규화, 의사 레이블링 등의 기법은 이후 3D-aware 생성, 깊이 기반 지도 등의 후속 연구의 토대가 되었다. 

향후 연구는 **더욱 정교한 다중 기하학적 제약의 통합**, **계산 효율성의 향상**, **표준화된 평가 체계의 확립**, 그리고 **실제 응용 분야로의 적용 확대**에 초점을 맞춰야 할 것으로 예상된다. 특히 최신의 3D-aware 확산 모델 및 멀티모달 학습 패러다임과의 결합은 기하학적으로 일관성 있고 현실적인 콘텐츠 생성의 새로운 지평을 열 것이다.

[1](https://arxiv.org/abs/2306.03414)
[2](https://openaccess.thecvf.com/content/ICCV2023/papers/Xiang_3D-aware_Image_Generation_using_2D_Diffusion_Models_ICCV_2023_paper.pdf)
[3](https://arxiv.org/abs/2311.17971)
[4](https://dl.acm.org/doi/10.1145/3664647.3680994)
[5](https://arxiv.org/abs/2409.09144)
[6](https://arxiv.org/abs/2408.14177)
[7](https://arxiv.org/abs/2412.00671)
[8](https://openaccess.thecvf.com/content/WACV2025/papers/Poleski_GeoGuide_Geometric_Guidance_of_Diffusion_Models_WACV_2025_paper.pdf)
[9](https://arxiv.org/pdf/1908.02983.pdf)
[10](https://openaccess.thecvf.com/content/CVPR2022/papers/Ren_Look_Outside_the_Room_Synthesizing_a_Consistent_Long-Term_3D_Scene_CVPR_2022_paper.pdf)
[11](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/03536.pdf)
[12](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/294415b9-b72c-4643-bae2-4949cebc69ad/2212.08861v2.pdf)
[13](https://journal-laaroiba.com/ojs/index.php/manageria/article/view/5608)
[14](https://ieeexplore.ieee.org/document/10655542/)
[15](https://arxiv.org/abs/2306.12681)
[16](https://dergipark.org.tr/en/doi/10.30622/tarr.1526734)
[17](https://ieeexplore.ieee.org/document/10553327/)
[18](https://ieeexplore.ieee.org/document/10657231/)
[19](https://arxiv.org/abs/2312.06655)
[20](http://arxiv.org/pdf/2412.17162.pdf)
[21](https://arxiv.org/pdf/2302.04313v4.pdf)
[22](https://arxiv.org/abs/2305.19947)
[23](http://arxiv.org/pdf/2405.10858.pdf)
[24](http://arxiv.org/pdf/2410.24220.pdf)
[25](https://arxiv.org/abs/2405.03188)
[26](https://arxiv.org/html/2411.16076)
[27](http://arxiv.org/pdf/2310.05873v3.pdf)
[28](https://openaccess.thecvf.com/content/WACV2025/papers/Garcia_Fine-Tuning_Image-Conditional_Diffusion_Models_is_Easier_than_You_Think_WACV_2025_paper.pdf)
[29](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/03265.pdf)
[30](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/03381.pdf)
[31](https://www.emergentmind.com/topics/generative-3d-aware-diffusion-models)
[32](https://arxiv.org/html/2505.12486v1)
[33](https://arxiv.org/html/2410.11439v3)
[34](https://dl.acm.org/doi/10.1145/3664647.3681265)
[35](https://link.springer.com/10.1134/S1064562424602038)
[36](https://arxiv.org/abs/2406.12849)
[37](https://ieeexplore.ieee.org/document/10447100/)
[38](https://link.springer.com/10.1007/s11042-023-15757-4)
[39](https://mand-ycmm.org/index.php/eatij/article/view/730)
[40](https://ieeexplore.ieee.org/document/10900860/)
[41](https://arxiv.org/abs/2404.17335)
[42](https://arxiv.org/html/2409.19933)
[43](http://arxiv.org/pdf/2102.06685.pdf)
[44](http://arxiv.org/pdf/2403.12953.pdf)
[45](http://arxiv.org/pdf/2503.16709.pdf)
[46](http://arxiv.org/pdf/2407.18443.pdf)
[47](http://arxiv.org/pdf/2503.20211.pdf)
[48](http://arxiv.org/pdf/2403.08556.pdf)
[49](https://arxiv.org/html/2412.20390v1)
[50](https://proceedings.neurips.cc/paper_files/paper/2024/file/6b7e1e96243c9edc378f85e7d232e415-Paper-Conference.pdf)
[51](https://nkhan2.github.io/projects/geometry-guided-2025/)
[52](https://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_completion)
[53](http://www.kibme.org/resources/journal/20210216114019572.pdf)
[54](https://www.sciencedirect.com/science/article/abs/pii/S0957417425024005)
[55](https://openaccess.thecvf.com/content/CVPR2025/papers/Wang_TacoDepth_Towards_Efficient_Radar-Camera_Depth_Estimation_with_One-stage_Fusion_CVPR_2025_paper.pdf)
[56](https://www.sciencedirect.com/science/article/abs/pii/S0031320324009671)
[57](https://arxiv.org/html/2507.07374v1)
