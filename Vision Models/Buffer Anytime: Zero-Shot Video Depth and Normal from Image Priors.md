
# Buffer Anytime: Zero-Shot Video Depth and Normal from Image Priors

> **논문 정보**
> - **저자**: Zhengfei Kuang, Tianyuan Zhang, Kai Zhang, Hao Tan, Sai Bi, Yiwei Hu, Zexiang Xu, Milos Hasan, Gordon Wetzstein, Fujun Luan
> - **소속**: Stanford University, MIT, Adobe Research
> - **게재**: CVPR 2025 (pp. 17660–17670)
> - **arXiv**: [2411.17249](https://arxiv.org/abs/2411.17249) (2024.11.26)

---

## 1. 핵심 주장 및 주요 기여 요약

Buffer Anytime은 비디오에서 깊이 맵(depth map)과 법선 맵(normal map)—논문에서 **geometric buffers**라 명명—을 추정하는 프레임워크로, **비디오-깊이 및 비디오-법선 쌍(paired) 학습 데이터가 전혀 필요 없다**는 점이 핵심입니다.

대규모 어노테이션된 비디오 데이터셋에 의존하는 대신, 단일 이미지 prior(image prior)와 시간적 일관성 제약(temporal consistency constraints)을 활용하며, **광학 흐름(optical flow) 기반의 스무스니스(smoothness)를 하이브리드 손실 함수로 결합한 제로샷(zero-shot) 학습 전략**을 제안합니다.

### 주요 기여 3가지

① **제로샷 파인튜닝 학습 체계**: 이미지 기하 버퍼 모델을 비디오 생성 모델로 전환하는 제로샷 훈련 방식 ② **하이브리드 학습 지도**: 이미지 모델의 정규화 손실(regularization loss)과 광학 흐름 기반의 스무스니스 손실(smoothness loss)로 구성된 하이브리드 훈련 지도 ③ **성능**: 이미지 기반 베이스라인 모델을 큰 폭으로 능가하며, 쌍 데이터로 훈련된 최신 비디오 모델과 비교 가능한 수준 달성

Depth Anything V2(깊이 추정)와 Marigold-E2E-FT(법선 추정)라는 두 최신 이미지 모델에 해당 훈련 체계를 적용하여 다양한 비디오 기하 추정 평가에서 유의미한 성능 향상을 시연합니다.

---

## 2. 상세 분석

### 2-1. 해결하고자 하는 문제

단일 이미지 깊이 추정기를 비디오의 각 프레임에 단순 적용하면 시간적 연속성이 무시되어 **플리커링(flickering)**이 발생하고, 카메라 움직임에 따른 깊이 범위 급변 문제가 생깁니다. 비디오 foundation model 위에 구축하는 방법도 있지만, 이는 고비용 훈련·추론, 불완전한 3D 일관성, 고정 길이 출력에 따른 스티칭 문제 등 고유한 한계를 가집니다.

즉, 본 논문은 다음 두 가지 핵심 난제를 동시에 해결하려 합니다:
- **데이터 병목(Data Bottleneck)**: 비디오-깊이/법선 쌍 데이터 획득의 어려움
- **시간 일관성(Temporal Consistency)**: 프레임 간 기하 예측의 일관성 유지

---

### 2-2. 제안 방법 및 수식

#### (A) 전체 학습 목표 — 하이브리드 손실 함수

하이브리드 손실은 사전 훈련된 이미지 모델에서 도출된 **정규화 손실(regularization loss)**과 광학 흐름 스무스니스 기반의 **안정화 손실(stabilization loss)**의 조합이며, 이를 통해 공간적 정확도와 시간적 일관성을 동시에 확보합니다.

전체 손실 함수는 다음과 같이 표현됩니다:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{stab}} + \omega_{\text{reg}} \cdot \mathcal{L}_{\text{reg}}$$

여기서:
- $\mathcal{L}_{\text{stab}}$: 광학 흐름 기반 시간적 안정화 손실 (Stabilization Loss)
- $\mathcal{L}_{\text{reg}}$: 이미지 prior 기반 정규화 손실 (Regularization Loss)
- $\omega_{\text{reg}}$: 두 손실 항목 간의 균형을 조정하는 하이퍼파라미터

#### (B) 안정화 손실 (Stabilization Loss)

광학 흐름 $\mathbf{F}_{t \to t'}$를 이용하여 인접 프레임 간 예측값의 일관성을 강제합니다:

$$\mathcal{L}_{\text{stab}} = \sum_{t} \sum_{\mathbf{p} \in \mathcal{V}_t} \left\| \hat{B}_t(\mathbf{p}) - \hat{B}_{t'}(\mathbf{p} + \mathbf{F}_{t \to t'}) \right\|_1$$

여기서:
- $\hat{B}_t(\mathbf{p})$: 시간 $t$에서 픽셀 $\mathbf{p}$의 예측 geometric buffer 값
- $\mathbf{F}_{t \to t'}$: 프레임 $t$에서 $t'$으로의 광학 흐름 벡터
- $\mathcal{V}_t$: 유효 픽셀 마스크 (occluded pixel 제외)

또한 광학 흐름의 부정확성으로 인해 깊이 프레임의 경계 영역에서 안정화 손실이 잘못 과대평가될 수 있어, **Canny 엣지 검출기**를 예측된 깊이 맵에 적용하고, 탐지된 엣지에 근접한 픽셀(맨해튼 거리 3픽셀 이내)의 손실을 필터링합니다.

#### (C) 정규화 손실 (Regularization Loss)

동결(frozen)된 이미지 prior 모델 $f_{\text{img}}$의 출력을 정규화 기준으로 사용:

$$\mathcal{L}_{\text{reg}} = \left\| \hat{B}_{t_r} - f_{\text{img}}(I_{t_r}) \right\|_1$$

전체 모델은 처음 세 가지 변형 모델보다 우수한 성능을 보이며, 모든 프레임에 정규화를 적용하는 방식(Ours all frames)은 표준 모델과 유사한 성능을 보여, **단일 무작위 프레임 정규화만으로도 이미지 prior와의 정렬을 충분히 유지**할 수 있음을 시사합니다.

#### (D) 광학 흐름 마스킹

임계값 하이퍼파라미터 $\tau_c$를 기반으로 신뢰할 수 없는 흐름 벡터를 필터링합니다. 구체적으로, 포워드-백워드 일관성 검사(forward-backward consistency check)를 통해 유효 마스크를 생성:

$$\mathcal{M}_t(\mathbf{p}) = \mathbb{1}\left[\left\| \mathbf{F}_{t \to t'}(\mathbf{p}) + \mathbf{F}_{t' \to t}(\mathbf{p} + \mathbf{F}_{t \to t'}) \right\|_2 < \tau_c \right]$$

---

### 2-3. 모델 구조

#### (A) Depth 추정 모델 (기반: Depth Anything V2)

깊이 추정 모델은 Depth Anything V2를 기반으로 하며, **ViT backbone을 동결(frozen)한 채로** fusion layer 사이에 temporal block을 삽입합니다.

#### (B) Normal 추정 모델 (기반: Marigold-E2E-FT)

공간 레이어(spatial layer) 사이에 temporal layer를 삽입하며, 원본 U-Net 레이어와 오토인코더는 훈련 중 고정됩니다. Temporal block은 **AnimateDiff의 블록 구조**와 유사하게, 여러 temporal attention block 뒤에 projection layer를 연결하며, 각 블록의 최종 projection layer는 **zero-initialization**하여 훈련 시작 시 이미지 모델과 동일하게 동작하도록 보장합니다.

```
[전체 구조 개요]

입력 비디오 프레임 {I_1, ..., I_T}
         │
         ▼
┌─────────────────────────────────┐
│ 공간 레이어 (Frozen)              │
│  (ViT Encoder or U-Net)         │
├─────────────────────────────────┤
│ Temporal Attention Block ← NEW  │
│  (only this part is trained)    │
├─────────────────────────────────┤
│ 공간 레이어 (Frozen) ... 반복    │
└─────────────────────────────────┘
         │
         ▼
출력: {B_1, ..., B_T}  (Depth or Normal maps)
```

핵심 이미지 모델은 동결(frozen) 상태를 유지하면서 temporal attention block만 통합하여 아키텍처를 적응시키며, 이를 통해 검증된 이미지 기반 기법의 이점을 유지하면서 시간적 추론 능력을 획득합니다.

---

### 2-4. 성능 향상 및 평가

ScanNet, KITTI, Bonn 등 다양한 데이터셋에 대한 실험 결과, 제안된 프레임워크는 경쟁력 있는 성능을 달성하면서 훈련 자원을 유의미하게 절감함을 입증합니다.

실험 결과, 이 방법은 이미지 기반 접근법들을 능가할 뿐만 아니라, **비디오-쌍 데이터를 전혀 사용하지 않았음에도** 대규모 쌍 비디오 데이터셋으로 훈련된 최신 비디오 모델과 비교 가능한 수준의 결과를 달성합니다.

| 비교 대상 | 학습 데이터 | 시간 일관성 | 정확도 |
|---|---|---|---|
| Depth Anything V2 (이미지) | 대규모 이미지 데이터 | ❌ 낮음 | ✅ 높음 |
| DepthCrafter (비디오) | 대규모 쌍 비디오 데이터 | ✅ 높음 | ✅ 높음 |
| **Buffer Anytime (제안)** | **쌍 비디오 데이터 없음** | **✅ 높음** | **✅ 유사** |

---

### 2-5. 한계

첫째, 백본 모델이 완전히 실패하는 극단적 케이스에서는 모델도 어려움을 겪을 수 있습니다. 둘째, 광학 흐름은 인접 프레임 간의 상관관계만 처리하므로, **장면에서 일시적으로 사라졌다가 다시 나타나는 객체**에 대한 일관된 깊이 정보 캡처가 실패할 수 있습니다.

---

## 3. 모델의 일반화 성능 향상 가능성

Buffer Anytime의 일반화 성능 향상 가능성은 여러 차원에서 분석할 수 있습니다.

### 3-1. 현재의 일반화 강점

Buffer Anytime은 "**zero-shot**"이라는 용어를 이 맥락에서 쌍 비디오-기하 실측 데이터(ground truth) 없이 학습하는 것으로 정의하며, 이미지 기하 모델의 지식과 기존 광학 흐름 방법을 결합하여 시간 일관성과 정확도 모두를 보장합니다.

이 접근법은 다음 측면에서 일반화 성능이 우수합니다:
- **도메인 불변성(Domain Invariance)**: 이미지 prior 모델(Depth Anything V2 등)이 이미 대규모 다양한 이미지 데이터로 훈련되어 있으므로, 이 지식이 비디오 도메인으로 전이됩니다.
- **플러그-앤-플레이(Plug-and-Play)**: 더 강력한 이미지 prior 모델이 등장할 때마다 자동으로 성능이 향상되는 구조입니다.

### 3-2. 일반화 향상을 위한 미래 방향

향후 유망한 방향으로는, ① **제한적인 비디오 감독을 포함한 하이브리드 훈련**으로 대형 이미지 모델과 통합하거나, ② **3D 공간에서 정의된 손실 함수** 등 더 정교한 프레임 간 일관성 가이던스를 개발하는 것이 제안됩니다. 이 프레임워크는 비디오 기반 기하 이해 작업에서 고비용 비디오 어노테이션에 대한 의존도를 줄이는 유망한 단계로 평가됩니다.

① 제한된 비디오 감독으로 이미지 모델을 향상시키면 특히 현재 방법으로는 적절히 처리되지 않는 어려운 시각 장면에서 정확도가 향상될 수 있으며, ② 현재의 광학 흐름 기술을 넘어서는 더 정교한 시간 일관성 방법을 연구하면 복잡한 비디오 시퀀스에서 간헐적으로 등장하는 물체에 대한 안정성도 개선될 수 있습니다.

---

## 4. 관련 최신 연구 비교 분석 (2020년 이후)

| 논문 | 연도 | 방법 | 학습 데이터 | 특징 |
|---|---|---|---|---|
| **Depth Anything V2** | 2024 | ViT 기반 대규모 학습 | 대규모 이미지 | 이미지 단위, 뛰어난 zero-shot 전이 |
| **Marigold-E2E-FT** | 2024 | Diffusion 기반 | 이미지 | 법선 추정, 단일 이미지 |
| **DepthCrafter** | 2024 | 비디오 diffusion | 쌍 비디오 데이터 | 시간 일관성 우수, 대규모 데이터 필요 |
| **RollingDepth** | 2024 | LDM + 최적화 등록 | 이미지 LDM | 짧은 비디오 스니펫 조합 방식 |
| **Buffer Anytime** | 2024 | Zero-shot, 하이브리드 손실 | **쌍 비디오 없음** | 시간 일관성 + 쌍 데이터 불필요 |

RollingDepth는 단일 이미지 LDM에서 파생된 **다중 프레임 깊이 추정기**와, 다양한 프레임 속도로 샘플링된 깊이 스니펫을 최적으로 조합하는 **강건한 최적화 기반 등록 알고리즘**이라는 두 가지 주요 구성 요소를 가지며, 수백 개의 프레임을 포함한 긴 비디오를 효율적으로 처리하면서 전용 비디오 깊이 추정기와 고성능 단일 프레임 모델 모두보다 정확한 깊이 비디오를 제공합니다.

---

## 5. 향후 연구에 미치는 영향과 연구 시 고려할 점

### 5-1. 연구에 미치는 영향

이 연구는 고급 이미지 모델과 비디오 작업 간의 시너지를 추가로 탐구하는 경로를 열어, 대규모 데이터 의존성을 줄이면서 높은 정확도와 시간적 일관성을 유지하는 방향으로 나아갑니다. 이러한 발전은 **구현된 AI(embodied AI), 자율 시스템, 새로운 3D/4D 재구성** 등 광범위한 응용 분야에서 큰 가능성을 열어줍니다.

구체적인 파급 효과:

1. **데이터 패러다임의 전환**: 쌍 데이터 없이도 비디오 기하 추정이 가능함을 증명함으로써, 데이터 획득의 병목이 높았던 의료 영상, 수중 탐사, 위성 영상 등에 적용 가능성 확대
2. **모듈형 AI 발전 지원**: 새로운 이미지 foundation model이 개발될 때마다 자동으로 성능이 향상되는 **플러그-앤-플레이 구조**로서, 이미지-비디오 전이 학습의 새로운 패러다임 제시
3. **비디오 이해 연구 가속화**: 비디오 광학 흐름, 장면 재구성, NeRF/3DGS 등과의 결합을 통한 4D 복원 연구에 직접 활용 가능

### 5-2. 미래 연구 시 고려할 점

① 현재 방법으로 적절히 처리되지 않는 **어려운 시각 장면**에서의 정확도 향상을 위해 제한된 비디오 감독을 활용한 이미지 모델 강화 연구가 필요하며, ② **복잡한 비디오 시퀀스에서 간헐적으로 등장하는 객체**에 대한 안정성 개선을 위해, 현재의 광학 흐름 기술을 넘어서는 더 정교한 시간 일관성 방법을 탐구해야 합니다.

추가 고려 사항:
- **스케일 모호성(Scale Ambiguity)**: Affine-invariant 깊이 예측의 절대 스케일 복원 문제
- **동적 객체 처리**: 빠른 움직임의 객체에서 광학 흐름 기반 제약이 실패할 수 있는 케이스
- **장기 시간 의존성**: 현재 temporal attention이 짧은 클립에 최적화되어 있어, 수백 프레임 이상의 장시간 비디오에서의 일관성 유지 방법 연구 필요
- **3D 공간 손실 함수**: 논문 자체에서 언급한 대로, 3D 공간에서 정의된 손실 함수 도입을 통한 기하학적 일관성 강화

---

## 📚 참고 자료 (출처)

1. **arXiv 논문 원본**: Kuang et al., "Buffer Anytime: Zero-Shot Video Depth and Normal from Image Priors," arXiv:2411.17249 (2024) — https://arxiv.org/abs/2411.17249
2. **CVPR 2025 공식 게재**: CVPR 2025 Open Access — https://openaccess.thecvf.com/content/CVPR2025/html/Kuang_Buffer_Anytime_Zero-Shot_Video_Depth_and_Normal_from_Image_Priors_CVPR_2025_paper.html
3. **IEEE Xplore 게재**: https://ieeexplore.ieee.org/document/11092371/
4. **프로젝트 공식 웹사이트**: https://bufferanytime.github.io/
5. **Emergent Mind 분석**: https://www.emergentmind.com/papers/2411.17249
6. **Hugging Face Papers**: https://huggingface.co/papers/2411.17249
7. **ResearchGate**: https://www.researchgate.net/publication/386143706
8. **관련 비교 논문 (RollingDepth / Video Depth without Video Models)**: arXiv:2411.19189 — https://huggingface.co/papers/2411.19189

> ⚠️ **정확도 주의**: 손실 함수의 구체적인 수식 표현 일부(특히 안정화 손실의 세부 형태)는 논문 HTML 본문에서 LaTeX 렌더링 없이 서술된 내용을 기반으로 재구성한 것으로, 정확한 수식은 [arXiv PDF 원문](https://arxiv.org/abs/2411.17249)을 직접 확인하시기를 강력히 권장합니다.
