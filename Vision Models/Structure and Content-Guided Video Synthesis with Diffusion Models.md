# Structure and Content-Guided Video Synthesis with Diffusion Models

### 1. 논문의 핵심 주장과 주요 기여

본 논문은 **구조(Structure)와 내용(Content)을 분리하여 비디오를 편집하는 새로운 접근**을 제시합니다. 주요 기여는 다음과 같습니다:[1]

- **Latent Diffusion Models의 비디오 확장**: 기존의 이미지 기반 사전학습 모델에 시간축 레이어를 도입하고, 이미지와 비디오 데이터를 결합하여 학습
- **구조-내용 분리 기반 편집**: 깊이 맵(Depth Map)을 통해 구조를 표현하고, CLIP 임베딩을 통해 내용을 표현
- **추론 시간 제어**: 별도의 비디오별 재학습 없이 단일 모델으로 다양한 비디오 편집 가능
- **시간 일관성 제어**: 이미지와 비디오 모델의 공동 학습으로부터 새로운 가이던스 메커니즘 도입

***

### 2. 해결하고자 하는 문제와 제안 방법

#### 2.1 핵심 문제

기존 비디오 편집 방법들의 한계:[1]
- **텍스트2비디오 편집**: 매 입력 비디오마다 재학습 필요 (예: Tune-a-Video)
- **이미지 전파 방식**: 프레임 간 오류 누적으로 시간 일관성 부족
- **균형 문제**: 공간적 디테일과 시간 일관성 사이의 트레이드오프

#### 2.2 제안 방법

##### 2.2.1 확산 모델 기초

논문에서 사용하는 확산 모델의 핵심 수식들:[1]

**순방향 프로세스 (Forward Process)**:
$$q(x_t|x_{t-1}) := \mathcal{N}(x_t, \sqrt{1-\beta_t}x_{t-1}, \beta_t I)$$

**역방향 프로세스 (Reverse Process)**:
$$p_\theta(x_{0:T}) = p(x_T)\prod_{t=1}^{T} p_\theta(x_{t-1}|x_t)$$

$$p_\theta(x_{t-1}|x_t) := \mathcal{N}(x_{t-1}, \mu_\theta(x_t, t), \Sigma_\theta(x_t, t))$$

**학습 목표 (Loss Function)**:

$$\mathcal{L} := \mathbb{E}_{t,q} \lambda_t \|\mu_t(x_t, x_0) - \mu_\theta(x_t, t)\|^2$$

여기서 $\mu_t(x_t, x_0)$는 순방향 프로세스의 후에 이용 가능한 닫힌 형태의 평균입니다.[1]

##### 2.2.2 Latent Diffusion Model

이미지 $x \in \mathbb{R}^{3 \times H \times W}$를 인코더 $E$로 잠재 공간으로 변환:[1]
$$z = E(x), \quad z \in \mathbb{R}^{4 \times H/8 \times W/8}$$

이를 통해 메모리 효율성을 높이고, 런타임을 개선합니다.

##### 2.2.3 시공간 확산 모델 (Spatio-Temporal Latent Diffusion)

**구조 표현 (Structure Representation)**:
- **모노큘러 깊이 추정**: MiDaS DPT-Large 모델 사용
- **정보 제거 프로세스**: 깊이 맵에 $t_s$번의 블러링 및 다운샘플링 적용
  - 훈련 시 $t_s \in [0, T_s]$ 범위에서 무작위 샘플링
  - 추론 시 사용자가 구조 보존 정도 제어 가능

**내용 표현 (Content Representation)**:
- CLIP 이미지 임베딩 $c$를 사용하여 내용 정보 인코딩
- 텍스트 입력의 경우 사전학습된 prior 모델로 텍스트 임베딩을 이미지 임베딩으로 변환

**조건화 메커니즘**:
- **구조**: Concatenation을 통해 깊이 표현 $s$를 입력 $z_t$에 연결
- **내용**: Cross-attention 블록을 통해 CLIP 임베딩 기반 정보 전달

##### 2.2.4 시간 일관성 제어 (Temporal Consistency Control)

분류자-무관 가이던스(Classifier-Free Guidance)를 확장:[1]

$$\tilde{\mu}_\theta(z_t, t, c, s) = \mu^\pi_\theta(z_t, t, \emptyset, s) + \omega_t(\mu_\theta(z_t, t, \emptyset, s) - \mu^\pi_\theta(z_t, t, \emptyset, s))$$
$$+ \omega(\mu_\theta(z_t, t, c, s) - \mu_\theta(z_t, t, \emptyset, s))$$

여기서:
- $\mu^\pi_\theta$: 이미지 모델(단일 프레임)의 예측
- $\mu_\theta$: 비디오 모델의 예측
- $\omega_t$: 시간 일관성 가이던스 스케일
- $\omega$: 내용 가이던스 스케일

이 수식의 직관은 비디오 모델과 이미지 모델 간의 차이를 외삽하여 시간 일관성을 제어합니다.[1]

***

### 3. 모델 구조

#### 3.1 아키텍처 설계

**U-Net 기반 시공간 확장**:[1]
- **공간층**: 2D 컨볼루션과 2D 트랜스포머 블록 (이미지-비디오 공유)
- **시간층**: 각 공간층 이후 1D 시간 컨볼루션 및 1D 시간 어텐션 추가
  - 시간 트랜스포머 블록은 학습 가능한 위치 인코딩 포함

**배치 처리**:
- 배치 크기 $b$, 프레임 수 $n$, 채널 $c$, 해상도 $w \times h$인 텐서 $b \times n \times c \times h \times w$를
  - 공간층: $(b \cdot n) \times c \times h \times w$로 재구성
  - 시간 컨볼루션: $(b \cdot h \cdot w) \times c \times n$으로 재구성
  - 시간 어텐션: $(b \cdot h \cdot w) \times n \times c$로 재구성

#### 3.2 학습 파이프라인

훈련 시 구조와 내용을 입력 비디오 자체에서 도출:[1]
$$s = s(x), \quad c = c(x)$$

손실 함수:

$$\lambda_t \|\mu_t(\mathcal{E}(x)_t, \mathcal{E}(x)_0) - \mu_\theta(\mathcal{E}(x)_t, t, s(x), c(x))\|^2$$

추론 시에는 입력 비디오 $y$에서 $s(y)$를 추출하고, 텍스트 또는 이미지 프롬프트 $t$에서 $c(t)$를 얻음:[1]
$$z \sim p_\theta(z|s(y), c(t)), \quad x = \mathcal{D}(z)$$

#### 3.3 학습 단계

다단계 학습 전략:[1]
1. **이미지 미세조정** (15k 스텝): CLIP 텍스트 임베딩에서 이미지 임베딩으로 조건화 변경
2. **시간 연결 추가 및 결합 학습** (75k 스텝): 시간층 도입, 이미지-비디오 결합 학습
3. **구조 조건화 추가** (25k 스텝): $t_s = 0$ (고정)으로 구조 조건화 학습
4. **가변적 구조 블러링** (10k 스텝): $t_s \in [0,7]$ 균일 샘플링[2]

***

### 4. 성능 평가 및 향상

#### 4.1 정량적 평가 메트릭

**프레임 일관성 (Frame Consistency)**:
- 연속 프레임의 CLIP 이미지 임베딩 간 코사인 유사도의 평균

**프롬프트 일관성 (Prompt Consistency)**:
- 모든 출력 프레임과 편집 프롬프트의 CLIP 임베딩 간 평균 코사인 유사도

#### 4.2 비교 결과

사용자 연구 (User Study) 기반 선호도 비교:[1]
- **제안 모델 vs SDEdit**: 약 3/4 비율로 제안 모델이 선호됨
- **제안 모델 vs Text2Live**: 88.24% 선호도
- **제안 모델 vs IVS (Propagation)**: 79.41-91.18% 선호도

정량적 평가에서 제안 모델은 프레임 일관성($0.9648 \pm 0.0031$)과 프롬프트 일관성($0.2805 \pm 0.0065$) 모두에서 우수한 성능을 보임.[1]

#### 4.3 시간 가이던스 스케일의 영향

$\omega_t$ 값에 따른 성능 변화:[1]
- $\omega_t = 0.50$: 손그려진 스타일, 낮은 프레임 일관성 (0.9238)
- $\omega_t = 1.00$: 균형잡힌 결과 (0.9648)
- $\omega_t = 1.50$: 부드러운 결과, 높은 프레임 일관성 (0.9722)

***

### 5. 일반화 성능 향상 가능성

#### 5.1 현재 논문의 일반화 메커니즘

**이미지-비디오 결합 학습의 이점**:[1]
- 대규모 이미지 데이터셋(240M 이미지)에서의 공간적 표현 학습
- 비디오 데이터셋(6.4M 클립)에서의 시간적 역학 학습
- 공유 가중치를 통해 일반화된 특성 추출

**구조 표현의 도메인 무관성**:[1]
- 깊이 맵 기반 구조는 특정 객체 클래스에 독립적
- 애니메이션, 실사, 풍경 등 다양한 콘텐츠에 적용 가능

**가변 깊이 블러링을 통한 유연성**:[1]
- 사용자가 추론 시 구조 보존 수준 조절 가능
- 다양한 편집 스타일 지원

#### 5.2 최신 연구에서의 일반화 발전

**깊이 기반 조건화의 확장**:[3]
- EVE (Efficient Zero-shot Video Editing)가 깊이 맵을 통한 구조 보존 강화
- 시간 일관성 제약을 명시적으로 모델링

**비디오-이미지 결합 학습 패러다임**:[4]
- BIVDiff는 이미지와 비디오 확산 모델을 브리징하여 강한 작업 일반화 달성
- 낮은 품질 비디오와 합성 고품질 이미지의 혼합 학습으로 도메인 외 일반화 개선

**3D 기하학 기반 가이던스의 등장**:[5]
- I2V3D는 3D 기하학적 조건화로 카메라 움직임, 객체 회전 등 정밀 제어
- 구조 표현의 다양화로 일반화 범위 확대

**시간 일관성 향상**:[6]
- Upscale-a-Video는 국소-전역 시간 전략으로 장시간 비디오의 일관성 향상
- 광학흐름 기반 재발생 잠재 전파로 수십 초 비디오의 안정성 확보

#### 5.3 일반화 성능 향상을 위한 미래 방향

**구조 표현의 다양화**:[1]
- 논문에서 언급: 인간 비디오 합성을 위해 포즈 추정이나 얼굴 랜드마크 활용 가능
- 최근 연구: SMPL 기반 신체 형태 가이드(Odo), DepthCrafter의 개선된 깊이 추정

**멀티모달 조건화 강화**:[7]
- Sora 이후 DiT(Diffusion Transformer) 기반 시공간 패치 처리로 확장성 증대
- 텍스트, 이미지, 오디오, 3D 조건의 통합 가이던스

**내용-구조 분리 개선**:[8][9]
- DiCoMoGAN: 구조화된 잠재 공간에서 텍스트 관련/무관 부분공간 명시적 분리
- Bitrate-Controlled Diffusion: 정보 이론 기반 동작과 내용의 완전 분리

**적응형 미세조정 전략**:[10]
- VideoCrafter2: 공간-시간 모듈 간 결합도 조절로 저품질 비디오 적응
- 맞춤형 적응(Customization)으로 특정 주제의 충실도 향상

***

### 6. 한계와 고려사항

#### 6.1 논문에서 명시된 한계

**구조-내용 완전 분리의 어려움**:[1]
- CLIP 임베딩과 깊이 맵 간의 기본적인 겹침 존재
- 깊이 맵이 객체 윤곽선을 포함하여 대규모 형태 변화를 제한

**미캡션 비디오 데이터에 대한 의존**:[1]
- 비디오-텍스트 쌍이 부족하여 구조, 내용을 입력 비디오 자체에서 도출
- 목표 출력에 대한 직접적인 감독 부재

**깊이 추정의 정확성**:[1]
- 단일 이미지 깊이 추정에 기반하여 복잡한 장면에서 오류 발생 가능

#### 6.2 최신 연구에서 지적된 추가 한계

**계산 효율성**:[11]
- 기존 확산 모델의 추론 속도 느림 (분 단위)
- 모바일 환경 배포의 어려움

**장시간 비디오 생성**:[12][7]
- 시간이 지남에 따라 아티팩트 축적
- 전역 시간 일관성 유지의 어려움

**일반화의 한계**:[13]
- 생성 성능과 다운스트림 작업 성능 간의 상관관계 부족
- 특정 도메인 편향 존재

***

### 7. 논문의 영향과 앞으로의 연구 고려사항

#### 7.1 학술적 영향

**확산 모델 기반 비디오 편집의 기초 확립**:[14]
- 708회 인용(2023년 ICCV 발표 이후) - 이 분야의 핵심 논문으로 자리매김
- 구조-내용 분리 개념의 표준화

**다양한 응용 분야로의 확산**:[1]
- 문자 기반 비디오 편집(Text2Live 대체)
- 이미지 기반 캐릭터 교체
- 마스킹을 통한 배경 편집

#### 7.2 후속 연구의 주요 방향

**1) 모듈 아키텍처 개선**
- DiT(Diffusion Transformer) 기반 구현으로 확장성 향상[7]
- 적응형 시공간 어텐션으로 효율성 증대

**2) 조건화 메커니즘 강화**
- 다중 기하학적 조건 통합(3D 기하학, 포즈, 광학흐름)[5][6]
- 언어 기반 세밀한 제어(LLM 가이더 통합)

**3) 내용-구조 분리 고도화**
- 정보 이론 기반 명시적 분리[9]
- 계층적 잠재 공간 설계로 더 정교한 제어

**4) 효율성 개선**
- 일관성 모델 기반 단일 스텝 생성[15]
- 프레임 삽입/보간을 통한 10배 가속[11]

**5) 평가 메트릭 정량화**
- 기존의 CLIP 기반 메트릭 한계 인식[13]
- FVD(Fréchet Video Distance), 시간 안정성 메트릭 활용

#### 7.3 실무 적용 시 고려사항

**데이터셋 구성**
- 다양한 도메인의 고품질 비디오-이미지 쌍 확보
- 특정 애플리케이션(예: 인간 합성)을 위한 도메인 특화 미세조정 데이터

**하이퍼파라미터 튜닝**
- 시간 가이던스 스케일 $\omega_t$의 세밀한 조정
- 구조 블러링 레벨 $t_s$의 작업별 최적화

**배포 환경 고려**
- 실시간 처리를 위한 경량화 전략
- 엣지 디바이스 최적화

**윤리 및 안전**
- 합성 비디오의 악용 방지 메커니즘
- 투명성 있는 생성 과정 공개

***

### 결론

"Structure and Content-Guided Video Synthesis with Diffusion Models" 논문은 **구조와 내용의 분리를 통해 제어 가능한 비디오 합성**이라는 우아한 솔루션을 제시했습니다. 깊이 맵 기반 구조 표현과 CLIP 임베딩 기반 내용 표현의 조합은 **시각적 조건화의 새로운 패러다임**을 열었으며, 이미지-비디오 결합 학습은 **일반화 성능의 향상**에 핵심 역할을 했습니다.

최신 연구 동향은 이 기초 위에서 **다양한 기하학적 조건의 통합**, **효율성 극대화**, **내용-구조 분리의 고도화** 방향으로 발전하고 있습니다. 앞으로의 연구는 계산 효율성, 장시간 비디오 안정성, 그리고 더욱 정교한 사용자 제어를 실현하는 방향으로 진행될 것으로 예상됩니다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/97e8f421-c5a8-4d28-bba1-27d8b2bf4798/2302.03011v1.pdf)
[2](http://arxiv.org/pdf/2407.08737.pdf)
[3](http://arxiv.org/pdf/2308.10648.pdf)
[4](http://arxiv.org/pdf/2312.02813.pdf)
[5](https://arxiv.org/html/2503.09733)
[6](https://openaccess.thecvf.com/content/CVPR2024/papers/Zhou_Upscale-A-Video_Temporal-Consistent_Diffusion_Model_for_Real-World_Video_Super-Resolution_CVPR_2024_paper.pdf)
[7](https://arxiv.org/html/2412.18688v2)
[8](https://bmvc2022.mpi-inf.mpg.de/0443.pdf)
[9](https://arxiv.org/html/2509.08376v1)
[10](https://openaccess.thecvf.com/content/CVPR2024/html/Chen_VideoCrafter2_Overcoming_Data_Limitations_for_High-Quality_Video_Diffusion_Models_CVPR_2024_paper.html)
[11](https://www.ijcai.org/proceedings/2024/1044.pdf)
[12](https://openreview.net/forum?id=2ODDBObKjH)
[13](https://openreview.net/forum?id=SIZhZrU41O)
[14](https://openaccess.thecvf.com/content/ICCV2023/papers/Esser_Structure_and_Content-Guided_Video_Synthesis_with_Diffusion_Models_ICCV_2023_paper.pdf)
[15](https://milvus.io/ai-quick-reference/what-are-the-latest-research-trends-in-diffusion-modeling)
[16](https://arxiv.org/pdf/2205.09853.pdf)
[17](https://arxiv.org/pdf/2409.11367.pdf)
[18](https://arxiv.org/html/2306.11173)
[19](http://arxiv.org/pdf/2410.05954.pdf)
[20](https://arxiv.org/html/2408.15241)
[21](http://arxiv.org/pdf/2212.05199.pdf)
[22](https://arxiv.org/abs/2302.00111)
[23](https://proceedings.neurips.cc/paper_files/paper/2023/file/1d5b9233ad716a43be5c0d3023cb82d0-Paper-Conference.pdf)
[24](https://lilianweng.github.io/posts/2024-04-12-diffusion-video/)
[25](https://arxiv.org/pdf/2408.16767v1.pdf)
[26](https://arxiv.org/pdf/1812.04605.pdf)
[27](https://arxiv.org/abs/2302.03011)
[28](https://arxiv.org/html/2409.02095)
[29](https://arxiv.org/pdf/1711.08682.pdf)
[30](https://arxiv.org/abs/2308.09091)
[31](https://arxiv.org/html/2508.13065v3)
[32](https://openaccess.thecvf.com/content/WACV2024/papers/Zhang_DR2_Disentangled_Recurrent_Representation_Learning_for_Data-Efficient_Speech_Video_Synthesis_WACV_2024_paper.pdf)
[33](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123660001.pdf)
[34](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/06048.pdf)
[35](https://velog.io/@jameskoo0503/Structure-and-Content-Guided-Video-Synthesis-with-Diffusion-Models-Paper-Review)
