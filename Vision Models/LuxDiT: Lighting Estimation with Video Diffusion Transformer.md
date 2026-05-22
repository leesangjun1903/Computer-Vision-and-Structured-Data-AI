
# LuxDiT: Lighting Estimation with Video Diffusion Transformer 

> **📌 논문 정보**
> - **제목**: LuxDiT: Lighting Estimation with Video Diffusion Transformer
> - **저자**: Ruofan Liang, Kai He, Zan Gojcic, Igor Gilitschenski, Sanja Fidler, Nandita Vijaykumar, Zian Wang (NVIDIA Toronto AI Lab 등)
> - **arXiv**: [2509.03680](https://arxiv.org/abs/2509.03680) (2025년 9월 3일 제출)
> - **학회**: NeurIPS 2025 (Accept)
> - **코드**: [github.com/nv-tlabs/LuxDiT](https://github.com/nv-tlabs/LuxDiT)
> - **프로젝트 페이지**: [research.nvidia.com/labs/toronto-ai/LuxDiT](https://research.nvidia.com/labs/toronto-ai/LuxDiT/)

---

## 1. 핵심 주장 및 주요 기여 요약

### 🔑 핵심 주장

본 논문은 조명 추정을 조건부 생성 태스크로 공식화하고, 합성 데이터로 학습된 뒤 실세계 장면에 적응된 신경망 조명 예측기 **LuxDiT**를 제안한다. 시각적 입력을 조건으로 하여, Diffusion Transformer(DiT)를 파인튜닝함으로써 노이즈로부터 HDR 파노라마를 합성한다.

조명 추정은 픽셀 정렬 태스크와 달리 장면 컨텍스트에 대한 **전역적(global) 추론**이 필요하다. DiT는 어텐션 기반 아키텍처 덕분에 전역 컨텍스트 집계에 적합하며, 생성적 사전 지식(generative priors)이 음영이나 반사 같은 간접적 단서로부터의 추론을 가능하게 한다.

### 📋 주요 기여 (3가지)

주요 기여는 다음 세 가지이다:
1. **DiT 기반 생성 아키텍처**: 시각적 입력으로부터 HDR 환경 맵을 합성하는 DiT 기반 생성 아키텍처
2. **LoRA 기반 파인튜닝 전략**: 입력 장면과 예측 조명 간 의미론적 정렬을 개선하기 위해 엄선된 HDR 파노라마를 이용한 LoRA 기반 파인튜닝 전략
3. **대규모 합성 데이터셋**: 무작위화된 기하학, 재질, 조명 조건으로 구성된 대규모 합성 데이터셋

---

## 2. 해결하고자 하는 문제, 제안 방법, 모델 구조, 성능 향상 및 한계

### 2-1. 해결하고자 하는 문제

단일 이미지 혹은 비디오로부터 장면 조명을 추정하는 것은 컴퓨터 비전과 그래픽스에서 오랜 난제이다. 학습 기반 접근법은 GT HDR 환경 맵의 희소성으로 제약받으며, 이는 수집 비용이 높고 다양성이 부족하다. 최근 생성 모델들도 조명 추정에는 어려움이 있는데, 간접적 시각 단서에 대한 의존, 전역(비국소) 컨텍스트 추론 필요성, HDR 출력 복원의 어려움 때문이다.

구체적으로 기존 방법들의 한계는 다음과 같다:

기존 잠재 확산 모델에 사용된 표준 VAE는 LDR 이미지로 학습되어 **HDR 콘텐츠를 충실히 인코딩하지 못**하며, 출력 파노라마가 입력과 공간적으로 정렬되지 않아 유연한 조건화 메커니즘이 필요하다.

DiffusionLight와 같은 선행 연구는 단일 추론으로 신뢰할 수 있는 조명 추정을 생성하지 못하고 HDR 출력을 직접 생성할 수 없어, 비용이 높은 테스트 타임 앙상블 전략에 의존해야 했다. 또한 별도의 추론 패스로 다중 노출을 샘플링하면 불일치가 생기고 복원된 조명의 다이나믹 레인지가 제한된다.

---

### 2-2. 제안하는 방법 (수식 포함)

#### (1) 문제 공식화: 조건부 디노이징 태스크

HDR 환경 맵 추정을 **조건부 디노이징 태스크**로 공식화한다. 입력 비디오 $\mathbf{I} \in \mathbb{R}^{L \times H \times W \times 3}$ (L 프레임)이 주어질 때, 대응하는 360° HDR 파노라마 시퀀스를 생성한다.

$$
\mathbf{E} = \mathcal{F}_\theta(\mathbf{I}), \quad \mathbf{E} \in \mathbb{R}^{L \times H_e \times W_e \times 3}
$$

여기서 $\mathcal{F}_\theta$는 DiT 기반 생성 모델이다.

#### (2) 이중 톤매핑 HDR 표현 (Dual Tonemapping)

출력은 두 개의 상호보완적 톤매핑 LDR 이미지로 표현된다: **Reinhard 톤매핑**($E_\text{ldr}$)과 **로그 강도 매핑**($E_\text{log}$).

이 두 가지 도전 과제를 해결하기 위해 이중 톤매핑 HDR 표현, 토큰 기반 조건화, 그리고 두 개의 조명 잠재 표현을 공동으로 디노이징하는 통합 트랜스포머 아키텍처를 활용한다.

구체적인 수식은 다음과 같다:

$$
E_\text{ldr} = \frac{E}{E + 1} \quad \text{(Reinhard tone mapping)}
$$

$$
E_\text{log} = \log(1 + E) \quad \text{(Log-intensity mapping)}
$$

최종 HDR 복원:

$$
E_\text{HDR} = \text{MLP}(E_\text{ldr}, E_\text{log})
$$

#### (3) VAE 인코딩 및 잠재 표현

톤매핑된 입력 $E_\text{ldr}$와 $E_\text{log}$는 사전학습된 VAE에 의해 잠재 텐서 $[z_\text{ldr}; z_\text{log}]$로 인코딩되며 형상은 $\mathbb{R}^{l \times h_e \times w_e \times C}$이다. 이들은 채널 차원을 따라 연결되어 확산 타겟 $z = [z_\text{ldr}; z_\text{log}] \in \mathbb{R}^{l \times h_e \times w_e \times 2C}$를 형성한다. 확산 네트워크의 입출력 프로젝션 레이어는 증가된 채널 차원을 수용하도록 확장된다.

$$
z = [z_\text{ldr}; z_\text{log}] \in \mathbb{R}^{l \times h_e \times w_e \times 2C}
$$

#### (4) 조건부 디노이징 목적 함수

잠재 확산 모델의 표준 디노이징 목적 함수는 다음과 같다:

$$
\mathcal{L}_\text{diff} = \mathbb{E}_{z_0, \epsilon, t}\left[\| \epsilon - \epsilon_\theta(z_t, t, \mathbf{c}) \|^2\right]
$$

여기서:
- $z_t$: 타임스텝 $t$에서 노이즈가 추가된 잠재 벡터
- $\epsilon \sim \mathcal{N}(0, I)$: 추가된 가우시안 노이즈
- $\mathbf{c}$: 시각적 입력(이미지/비디오) 조건
- $\epsilon_\theta$: DiT 기반 디노이저

#### (5) 방향성 맵(Directional Map) 가이던스

입력 이미지 또는 비디오 $I$가 주어지면, LuxDiT는 환경 맵 $E$를 두 개의 톤매핑 표현 $E_\text{ldr}$과 $E_\text{log}$로 예측하며, **방향성 맵(directional map) $E_\text{dir}$**의 가이던스를 받는다.

방향성 맵은 구형 좌표를 인코딩하여 공간적 조명 방향 정보를 DiT에 제공하는 역할을 한다.

---

### 2-3. 모델 구조

LuxDiT는 HDR 환경 맵 추정을 **조건부 디노이징 태스크**로 공식화한다. 모델은 계산 효율성을 위해 잠재 공간에서 동작하는 트랜스포머 기반 확산 백본(**CogVideoX** 아키텍처 기반)으로 구축된다. 입력은 단일 이미지 또는 비디오로, 사전학습된 VAE를 통해 인코딩된다.

환경 맵은 VAE로 인코딩되고, 결과 잠재 벡터들은 연결되어 시각적 입력과 함께 DiT에 의해 공동으로 처리된다. 출력 $E_\text{ldr}$과 $E_\text{log}$는 디코딩되어 경량 MLP에 의해 융합됨으로써 최종 HDR 파노라마를 복원한다.

모델 구조를 도식화하면 다음과 같다:

```
입력 I (이미지/비디오)
        ↓
   [VAE Encoder]                ← 시각적 입력 인코딩
        ↓
   [DiT (CogVideoX 기반)]       ← E_ldr + E_log + E_dir 공동 처리
        ↓
  [VAE Decoder × 2]             ← E_ldr, E_log 디코딩
        ↓
   [MLP Fusion]
        ↓
  최종 HDR 파노라마 E_HDR
```

**핵심 아키텍처 설계 이유:**

정확한 조명 추정은 모델이 그림자 방향, 표면 반사, 하이라이트 등 입력 이미지로부터 세밀한 음영 단서를 추출해야 하며, 픽셀 정렬 이미지-이미지 변환 태스크와 달리 경험적으로 관찰되는 특성들이 있다.

---

### 2-4. 학습 전략: 2단계 훈련

LuxDiT는 주로 다양한 조명 조건이 풍부한 합성 데이터셋으로 학습되고, 이후 실제 HDR 파노라마에 대한 파라미터 효율적인 LoRA 파인튜닝 단계를 거치는 **2단계 훈련 프로토콜**을 사용한다. 초기에는 Objaverse 등의 저장소를 활용하여 물리 기반 렌더링(PBR)으로 랜덤화된 3D 장면의 대규모 합성 데이터셋이 구성된다. 이 장면들은 샘플링된 HDR 환경 맵으로 조명되어, 간접 시각 효과(그림자, 반사, 상호 반사)와 기저 조명 사이의 복잡한 관계를 포착하는 입출력 쌍을 생성한다.

**Stage I: Synthetic Supervised Training**

$$
\mathcal{L}_\text{Stage I} = \mathcal{L}_\text{diff}(z_\text{HDR}, \mathbf{c}_\text{synthetic})
$$

**Stage II: LoRA-based Real-World Adaptation**

입력과 예측된 환경 맵 사이의 의미론적 정렬을 향상시키기 위해, 수집된 HDR 파노라마 데이터셋을 사용한 **저순위 적응(LoRA) 파인튜닝 전략**을 도입한다.

LoRA 파인튜닝은 선택된 트랜스포머 레이어에 훈련 가능한 저순위 행렬을 주입하며, 기본 가중치의 대부분은 고정되어, 치명적 망각(catastrophic forgetting) 없이 효율적인 적응이 가능하다. 파인튜닝은 실제 HDR 파노라마의 정선된 세트로 지도학습되어, 도시 또는 건축 장면에서 현실적인 조명이 필요한 경우의 불일치 방지 등 장면 콘텐츠와의 의미론적 매칭을 최적화한다.

LoRA 수식:

$$
W' = W_0 + \Delta W = W_0 + A \cdot B, \quad A \in \mathbb{R}^{d \times r},\ B \in \mathbb{R}^{r \times k},\ r \ll \min(d, k)
$$

---

### 2-5. 성능 향상

단일 이미지 또는 비디오가 주어지면 LuxDiT는 정확한 방향, 강도, 장면 일관성 있는 콘텐츠를 가진 HDR 환경 맵을 생성한다. **Laval Outdoor 벤치마크에서 태양광 방향 오차를 45% 감소**시키며, 비디오 입력에 대한 시간적 일관성을 개선하여, 가상 객체 삽입 등 다운스트림 애플리케이션에서 신뢰할 수 있는 사용을 가능하게 한다.

대규모 합성 사전학습과 LoRA 기반 파인튜닝을 활용하여 선행 방법 대비 **우수한 각도 정확도와 시간적 일관성**을 달성한다. 실험 결과는 감소된 피크 각도 오차와 향상된 가상 객체 삽입 품질을 보여주며, 방법의 최첨단 성능을 검증한다.

본 방법은 현실적인 각도 고주파수 세부 사항을 가진 정확한 조명 예측을 생성하며, **정량적 및 정성적 평가 모두에서 기존 최첨단 기술을 능가**한다.

---

### 2-6. 한계

LuxDiT는 고품질 조명 예측을 생성하지만, **확산 모델의 반복적 특성으로 인해 추론이 계산 집약적**이어서 실시간 애플리케이션 사용이 제한된다. 향후 연구에서는 모델 증류(distillation)나 더 효율적인 아키텍처를 탐색하여 추론을 가속화할 수 있다.

또한, 예측된 파노라마의 해상도는 데이터와 학습 규모에 의해 제한되며; **몰입형 애플리케이션을 위한 고해상도 출력 생성**을 위해서는 더 풍부하고 다양한 HDR 감독이 필요하다.

---

## 3. 모델의 일반화 성능 향상 가능성

### 3-1. 합성→실세계 일반화의 핵심 전략

대규모 합성 데이터셋에서 다양한 조명 조건으로 학습된 모델은 간접적 시각 단서로부터 조명을 추론하는 방법을 학습하고, **실세계 장면으로 효과적으로 일반화**한다.

실세계 HDR 조명 감독의 부재를 극복하기 위해 무작위화된 기하학, 재질, 조명 조건을 가진 대규모 합성 데이터셋을 구성한다. 이 데이터셋으로의 학습은 모델이 빛의 방향과 강도에 대한 **물리적으로 근거 있는(physically grounded) 단서**를 학습하게 한다.

### 3-2. LoRA를 활용한 도메인 갭 극복

LoRA 단계 동안, 실제 이미지 톤매핑에 직접 관여하는 피처만 목적 함수에 사용되며, 일반화와 커버리지는 고정된 기반 아키텍처를 통해 보존된다. 이 구성은 의미론적 정렬을 보장하고 희소한 HDR 감독에 대한 과적합을 방지한다.

### 3-3. 학습 데이터 다양성을 통한 일반화 강화

장면들은 샘플링된 HDR 환경 맵으로 조명되어, 간접 시각 효과(그림자, 반사율, 상호 반사)와 기저 조명 사이의 복잡한 관계를 포착하는 입출력 쌍을 생성한다. 또한, 실제 HDR 파노라마에서의 투시 크롭과 저다이나믹 레인지 파노라마 비디오 데이터셋(예: WEB360)을 학습에 포함함으로써, 모델이 합성 및 자연 도메인에 걸쳐 일반화 가능한 매핑을 학습한다.

### 3-4. 비디오 입력을 통한 시간적 일반화

LuxDiT는 일상적으로 촬영된 이미지와 비디오로부터 HDR 장면 조명을 추정하는 조건부 생성 모델로, 물리적으로 근거 있는 사전 지식 학습을 위한 **대규모 합성 데이터**와 의미론적 정렬을 개선하기 위한 **실제 HDR 파노라마 기반 LoRA 적응**을 결합한다.

---

## 4. 2020년 이후 관련 최신 연구 비교 분석

| 연구명 | 연도 | 발표처 | 핵심 방법 | LuxDiT 대비 특징 |
|:---|:---:|:---:|:---|:---|
| **DiffusionLight** | 2024 | CVPR | 가상 크롬볼 인페인팅 | 다단계 처리, 왜곡된 파노라마 |
| **EverLight** | 2023 | ICCV | 파라메트릭 추정 + GAN | 복잡/밝은 조명에 취약 |
| **IllumiDiff** | 2024 | TVCG | 단일 이미지 확산 모델 | 실내 한정 |
| **LightOctree** | 2024 | CVPR | 경량 3D 공간 일관성 | 정적 장면 한정 |
| **LuxDiT** | **2025** | **NeurIPS** | Video DiT + LoRA | **HDR 직접 생성, 비디오 지원** |

DiffusionLight는 사전학습된 텍스트-이미지 모델이 조명에 대한 암묵적 지식을 인코딩함을 보여주었으며, 가상 크롬볼 인페인팅으로 이를 추출한다. EverLight는 파라메트릭 조명 추정을 회귀하고 GAN으로 고주파 세부 사항을 추가하지만, 의사 레이블(pseudo-labeled) HDR 데이터에 의존하며 복잡하거나 밝은 조명에 어려움을 겪는다. DiffusionLight는 다중 노출에서 가상 크롬볼을 인페인팅하여 HDR 맵으로 병합하는 확산 모델을 사용하지만, 이 다단계 프로세스는 왜곡된 파노라마와 제한된 다이나믹 레인지를 생성한다.

기존 학습 기반 방법들은 GT HDR 맵이 있는 페어드 데이터셋의 희소성으로 제약받으며, GAN이나 확산 모델 같은 생성 모델들도 조명 단서의 비국소적·간접적 특성을 완전히 다루지 못했다.

**LuxDiT의 차별점 정리:**

LuxDiT는 트랜스포머 기반 확산 모델링, 대규모 합성 사전학습, LoRA 기반 의미론적 적응을 통합함으로써 HDR 조명 추정을 발전시킨다. 정량적·정성적 평가 모두에서 최첨단 성능을 달성하며, 이미지와 비디오 모두에 대해 견고하고 장면 일관적인 조명 예측을 지원한다.

---

## 5. 앞으로의 연구에 미치는 영향 및 고려할 점

### 5-1. 앞으로의 연구에 미치는 영향

이 접근법은 가상 객체 삽입, 리라이팅(relighting), AR 등 다운스트림 태스크로 확장 가능하며, 통합된 장면 재구성 및 외관 합성 연구를 위한 기반을 제공한다.

최근 공동 생성 모델링의 발전과 함께, LuxDiT는 통합된 역방향(inverse) 및 순방향(forward) 렌더링 프레임워크를 향한 한 걸음으로, 신경 순방향 렌더링 및 G-버퍼 추정 분야의 최근 발전을 보완한다.

**구체적 영향 영역:**

1. **AR/VR 가상 객체 삽입**: 정확한 조명을 보존하면서 장면 의미론을 유지함으로써, 다양한 조건에서 현실감 있는 가상 객체 삽입이 가능해진다.

2. **비디오 리라이팅 파이프라인**: 시간적으로 일관된 HDR 조명 추정을 통해 비디오 후처리에서의 응용이 가능해진다.

3. **역방향 렌더링(Inverse Rendering) 통합**: DiT 기반 생성 모델이 조명, 재질, 기하학을 동시에 추론하는 통합 프레임워크 구축 가능성을 열었다.

### 5-2. 향후 연구 시 고려할 점

**① 계산 효율성:**
확산 모델의 반복적 특성으로 인해 추론이 계산 집약적이어서 실시간 애플리케이션 사용이 제한된다. 모델 증류나 더 효율적인 아키텍처 탐색이 필요하다.

**② 고해상도 출력:**
예측된 파노라마의 해상도는 데이터와 학습 규모에 의해 제한되며, 몰입형 애플리케이션을 위한 고해상도 출력 생성을 위해서는 더 풍부하고 다양한 HDR 감독이 필요하다.

**③ 실세계 데이터 부족 문제:**
학습 기반 접근법은 GT HDR 환경 맵의 희소성으로 제약받으며, 이는 수집 비용이 높고 다양성이 부족하다. 더 효율적인 실세계 HDR 데이터 수집 방법론이나 자기지도 학습 방식 탐구가 필요하다.

**④ 공간적으로 변하는 조명(Spatially-Varying Lighting):**
현재 LuxDiT는 전역(global) 환경 맵을 예측하지만, 실내 장면에서의 공간적으로 변하는 조명 추정으로의 확장이 향후 과제이다.

**⑤ 합성-실세계 도메인 갭:**
파인튜닝이 장면 콘텐츠와의 의미론적 매칭을 최적화하여 도시/건축 장면에서의 불일치를 방지하지만, 더 다양한 실세계 도메인에 대한 일반화 검증이 지속적으로 필요하다.

---

## 📚 참고 자료 (출처)

| # | 출처 유형 | 링크 및 제목 |
|:---|:---|:---|
| 1 | **arXiv 논문 (주 참고)** | [arXiv:2509.03680 - LuxDiT: Lighting Estimation with Video Diffusion Transformer](https://arxiv.org/abs/2509.03680) |
| 2 | **NVIDIA 공식 프로젝트 페이지** | [research.nvidia.com/labs/toronto-ai/LuxDiT](https://research.nvidia.com/labs/toronto-ai/LuxDiT/) |
| 3 | **NVIDIA 공식 논문 PDF** | [research.nvidia.com/labs/toronto-ai/LuxDiT/assets/LuxDiT_paper.pdf](https://research.nvidia.com/labs/toronto-ai/LuxDiT/assets/LuxDiT_paper.pdf) |
| 4 | **OpenReview (NeurIPS 2025)** | [openreview.net/forum?id=nw6Kx91J48](https://openreview.net/forum?id=nw6Kx91J48) |
| 5 | **arXiv HTML 전문** | [arxiv.org/html/2509.03680v1](https://arxiv.org/html/2509.03680v1) |
| 6 | **GitHub 공식 코드** | [github.com/nv-tlabs/LuxDiT](https://github.com/nv-tlabs/LuxDiT) |
| 7 | **HuggingFace 논문 페이지** | [huggingface.co/papers/2509.03680](https://huggingface.co/papers/2509.03680) |
| 8 | **Emergent Mind 분석** | [emergentmind.com/papers/2509.03680](https://www.emergentmind.com/papers/2509.03680) |
| 9 | **Awesome-Illumination-Estimation (관련 연구 목록)** | [github.com/waldenlakes/Awesome-Illumination-Estimation](https://github.com/waldenlakes/Awesome-Illumination-Estimation) |

> ⚠️ **정확도 주의사항**: 손실 함수의 구체적인 수식 형태(예: HDR-specific loss term 세부 항목)와 일부 구현 세부 사항은 공개 PDF 전문에서 완전히 확인되지 않아, 일반적인 잠재 확산 모델 표준 수식으로 기술하였습니다. 정확한 수식 전체는 [공식 논문 PDF](https://research.nvidia.com/labs/toronto-ai/LuxDiT/assets/LuxDiT_paper.pdf)를 직접 확인하시기를 권장합니다.
