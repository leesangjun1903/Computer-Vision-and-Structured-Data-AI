# Zero123++: a Single Image to Consistent Multi-view Diffusion Base Model

---

## 1. 핵심 주장과 주요 기여 요약

**Zero123++**는 단일 이미지로부터 **3D 일관성을 갖춘 멀티뷰 이미지**를 생성하는 이미지 조건부 확산 모델(image-conditioned diffusion model)이다. 기존 Zero-1-to-3 모델이 각 뷰를 **독립적으로** 생성하여 뷰 간 기하학적 불일치가 발생하는 문제를 해결하고, 사전 학습된 Stable Diffusion의 prior를 **최대한 보존**하면서 미세 조정(fine-tuning)하는 다양한 전략을 제안한다.

### 주요 기여:
1. **멀티뷰 타일링(Tiling) 전략**: 6개의 뷰를 $3 \times 2$ 레이아웃의 단일 이미지로 구성하여 **조인트 분포(joint distribution)**를 올바르게 모델링
2. **노이즈 스케줄 변경**: Scaled-linear에서 **linear 노이즈 스케줄**로 전환하여 글로벌 일관성 향상
3. **Scaled Reference Attention**: 로컬 이미지 조건부 입력을 위한 참조 어텐션 메커니즘 제안
4. **FlexDiffuse 기반 글로벌 조건부**: CLIP 이미지 임베딩을 활용한 글로벌 이미지 조건부 도입
5. **Depth ControlNet**: Zero123++ 위에 ControlNet을 학습하여 기하학적 제어 가능성 시연

---

## 2. 상세 설명

### 2.1 해결하고자 하는 문제

기존 **Zero-1-to-3**는 단일 이미지에서 새로운 뷰를 생성하지만, 다음과 같은 핵심 문제점이 있다:

| 문제 | 설명 |
|------|------|
| **뷰 간 불일치** | 각 뷰를 독립적으로 생성하여 기하학적·텍스처 불일치 발생 |
| **Stable Diffusion prior 활용 부족** | 로컬/글로벌 조건부 메커니즘을 효과적으로 활용하지 못함 |
| **해상도 제한** | 512 해상도에서 학습 불안정으로 256으로 축소, 품질 저하 |
| **Elevation 추정 필요** | 상대적 elevation 사용으로 추가 추정 모듈 필요 |

### 2.2 제안하는 방법

#### (A) 멀티뷰 조인트 생성

Zero-1-to-3는 각 뷰의 **조건부 주변 분포(conditional marginal distribution)**를 독립적으로 모델링하여 뷰 간 상관관계를 무시한다. Zero123++는 6개의 뷰를 $3 \times 2$ 타일로 배치하여 **조인트 분포**를 직접 모델링한다.

카메라 포즈는 **고정된 절대 elevation**과 **상대 azimuth**를 사용한다:
- Elevation: $30°$ (하향)과 $-20°$ (상향) 교대
- Azimuth: $30°$에서 시작하여 $60°$씩 증가 ($30°, 90°, 150°, 210°, 270°, 330°$)

이를 통해 추가적인 elevation 추정 모듈 없이 방향 모호성을 제거한다.

#### (B) 노이즈 스케줄: Linear Schedule로 전환

Stable Diffusion의 **scaled-linear 스케줄**은 낮은 SNR(Signal-to-Noise Ratio) 구간의 스텝이 매우 적어, 초기 디노이징 단계에서 글로벌 저주파 구조 결정이 어렵다. 확산 모델에서 노이즈 입력 $z_t$는 다음과 같이 정의된다:

$$z_t = \alpha_t \cdot x_0 + \sigma_t \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

여기서 **SNR**은 다음과 같다:

$$\text{SNR}(t) = \frac{\alpha_t^2}{\sigma_t^2}$$

Scaled-linear 스케줄은 고해상도에서 낮은 SNR 구간이 부족하여, 글로벌 구조(멀티뷰 일관성 등)를 학습하기 어렵다. Chen [2]가 지적했듯이, 고해상도 이미지는 동일한 절대 노이즈 수준에서 저해상도 이미지보다 **덜 노이즈**가 심한 것처럼 보인다. 이는 인접 픽셀의 중복성(redundancy) 때문이다.

**Linear 스케줄**로 전환하면 낮은 SNR 구간이 충분해져 글로벌 일관성 학습이 용이해진다. 이때 **v-prediction** 파라미터화를 채택한다:

$$v_t = \alpha_t \cdot \epsilon - \sigma_t \cdot x_0$$

v-prediction은 스케줄 변경에 대해 $\epsilon$-prediction이나 $x_0$-prediction보다 **본질적으로 더 안정적**이어서(Salimans & Ho, 2022), 스케줄 전환 시 사전학습 모델의 성능 저하를 최소화한다.

#### (C) Scaled Reference Attention (로컬 조건부)

Zero-1-to-3는 조건부 이미지를 noisy input과 **채널 차원으로 concatenate**하여, 입력과 타겟 이미지 간 **잘못된 픽셀 단위 공간 대응**을 강제한다.

Zero123++는 **Reference Attention**을 사용한다 (Figure 6 참조):
1. 디노이징 UNet을 참조 이미지에 대해 실행
2. 참조 이미지의 self-attention **Key(K)**와 **Value(V)** 행렬을 디노이징 입력의 해당 attention 레이어에 **append**
3. 참조 이미지에도 동일한 시간 스텝 $t$의 가우시안 노이즈를 추가

핵심적으로, 참조 이미지의 latent를 노이즈 추가 전에 **스케일링**한다. 실험 결과, **5배 스케일링**이 최적의 일관성을 보인다.

#### (D) FlexDiffuse 기반 글로벌 조건부

CLIP 이미지-텍스트 공간의 정렬을 활용하여 글로벌 이미지 조건부를 도입한다. 원래 프롬프트 임베딩 $T$에 CLIP 글로벌 이미지 임베딩 $I$를 가중합하여 수정된 임베딩을 얻는다:

$$T'_i = T_i + w_i \cdot I, \quad i = 1, 2, \ldots, L \tag{1}$$

여기서 $L$은 토큰 길이, $D$는 토큰 임베딩 차원, $I \in \mathbb{R}^D$는 CLIP 글로벌 이미지 임베딩이다. 학습 가능한 가중치 $\{w_i\}_{i=1,...,L}$는 FlexDiffuse의 선형 가이던스로 초기화된다:

$$w_i = \frac{i}{L} \tag{2}$$

글로벌 조건부 없이는 입력 이미지에서 보이지 않는 영역의 생성 품질이 현저히 저하된다 (Figure 8).

#### (E) 학습 전략

**Base Model**: Stable Diffusion 2 v-prediction 모델

**단계적 학습 스케줄** (Stable Diffusion Image Variations 모델에서 차용):
- **Phase 1**: Self-attention 레이어 + cross-attention의 KV 행렬만 학습
  - AdamW optimizer, cosine annealing, peak LR = $7 \times 10^{-5}$, 1000 warm-up steps
- **Phase 2**: Full UNet 학습
  - 보수적 상수 LR = $5 \times 10^{-6}$, 2000 warm-up steps

**Min-SNR 가중치 전략** (Hang et al., 2023)을 사용하여 학습 효율을 높인다.

**학습 데이터**: Objaverse 데이터셋 (~800k 객체), 랜덤 HDRI 환경 조명으로 렌더링

### 2.3 모델 구조

Zero123++의 전체 구조는 Stable Diffusion 2 v-prediction UNet을 기반으로 하며, 다음 요소들이 추가/수정된다:

```
┌─────────────────────────────────────────────────────┐
│  입력: 단일 이미지                                      │
│     ├── CLIP Image Encoder → 글로벌 임베딩 I           │
│     └── VAE Encoder → 참조 이미지 Latent (×5 스케일링)   │
│                                                       │
│  UNet (Stable Diffusion 2 v-prediction)               │
│     ├── Self-Attention: Reference Attention (K,V append)│
│     ├── Cross-Attention: 수정된 프롬프트 임베딩 T'        │
│     └── 노이즈 스케줄: Linear Schedule                   │
│                                                       │
│  출력: 3×2 타일 이미지 (6개 뷰)                         │
│     320×320 × 6 → 960×640 단일 프레임                   │
└─────────────────────────────────────────────────────┘
```

### 2.4 성능 비교

#### 정량적 결과 (LPIPS, 낮을수록 좋음):

| 모델 | LPIPS $\downarrow$ |
|------|------|
| Zero-1-to-3 | $0.210 \pm 0.059$ |
| Zero-1-to-3 XL | $0.188 \pm 0.053$ |
| **Zero123++ (Ours)** | $\mathbf{0.177 \pm 0.066}$ |
| Zero123++ + Depth ControlNet | $0.086$ |

주목할 점: Zero-1-to-3 모델들은 검증 세트를 학습 중 보았을 가능성이 있고, XL 변형은 Zero123++보다 훨씬 더 많은 데이터로 학습되었음에도 Zero123++가 최고 성능을 달성했다.

#### 정성적 비교:
- **vs. Zero-1-to-3 XL**: 멀티뷰 일관성 부족, 일부 뷰에서 아티팩트 발생
- **vs. SyncDreamer**: elevation 변경 미지원, 제한적 일반화
- **vs. MVDream** (텍스트→멀티뷰): Objaverse 편향으로 인한 만화풍 텍스처 스타일 변환 문제
- **Zero123++**: 실사 사진, AI 생성 이미지, 2D 일러스트레이션 등 다양한 입력에 대해 일관되고 고품질의 멀티뷰 생성

### 2.5 한계

1. **$\epsilon$-prediction의 로컬 디테일 우위**: v-prediction은 글로벌 일관성에 강하지만, 로컬 디테일에서는 $\epsilon$-prediction이 우수할 수 있음
2. **학습 데이터 규모**: Objaverse (~800k)로 학습되어, 더 큰 데이터셋(Objaverse-XL, 10M+)으로의 확장 필요
3. **멀티뷰 → 3D 메쉬 변환 갭**: 고품질 멀티뷰 이미지에서 고품질 3D 메쉬로의 변환에는 여전히 간극 존재
4. **고정된 뷰포인트**: 6개의 고정된 카메라 포즈만 지원하여 임의의 뷰 생성에는 제약

---

## 3. 모델의 일반화 성능 향상 가능성

Zero123++의 일반화 성능은 여러 설계 결정에 의해 뒷받침된다:

### 3.1 Stable Diffusion Prior 보존 전략

Zero123++의 핵심 설계 철학은 **사전학습된 2D 생성 prior를 최대한 보존**하는 것이다. 이를 위한 전략들:

| 전략 | 일반화에 미치는 영향 |
|------|------|
| **단계적 학습** | Phase 1에서 attention 레이어만 학습하여 SD prior 보존 |
| **Reference Attention** | 새로운 파라미터 추가 없이 SD의 기존 attention 메커니즘 재활용 |
| **FlexDiffuse 글로벌 조건부** | CLIP 이미지-텍스트 정렬을 활용, 최소한의 학습 가능 파라미터($\{w_i\}$)만 추가 |
| **v-prediction + Linear Schedule** | 스케줄 전환에도 사전학습 모델 성능 유지 |

### 3.2 도메인 외 일반화 능력

Figure 1과 Figure 10에서 보여주듯, Zero123++는 다음 입력에 대해 모두 일반화된다:
- **실사 사진** (소화기 등)
- **AI 생성 이미지** (SDXL로 생성한 "강아지가 로켓에 앉아 있는 이미지")
- **2D 애니메이션 일러스트레이션**

이는 Objaverse 데이터셋에서만 학습했음에도 불구하고, SD의 강력한 2D prior를 효과적으로 보존했기 때문이다.

### 3.3 일반화 성능 향상을 위한 잠재적 방향

1. **데이터 스케일링**: Objaverse-XL (10M+ 객체)로 학습 데이터 확장
2. **2단계 리파이너**: $\epsilon$-parametrized SDXL 모델을 리파이너로 사용하여 로컬 디테일 향상
3. **더 강력한 base model 활용**: SDXL 등 더 강한 prior를 가진 모델 기반 미세 조정
4. **ControlNet 확장**: Depth 외에도 normal map, edge 등 다양한 조건부를 통한 제어 가능성 확장

### 3.4 노이즈 스케줄의 일반화 관점에서의 역할

Linear 노이즈 스케줄은 낮은 SNR 구간을 충분히 확보하여, 모델이 **글로벌 구조를 먼저 결정**하고 **로컬 디테일을 나중에 생성**하는 계층적 생성 과정을 가능하게 한다. 이는 도메인 외 입력에 대해서도 안정적인 전체 구조 생성을 보장하여 일반화에 기여한다.

---

## 4. 향후 연구에 미치는 영향과 고려 사항

### 4.1 연구에 미치는 영향

#### (1) 멀티뷰 확산 모델 설계 패러다임 정립
Zero123++는 **독립 뷰 생성 → 조인트 분포 모델링**으로의 패러다임 전환을 촉진하였다. 이후 등장한 다수의 연구들(Instant3D, Wonder3D, SV3D 등)이 유사한 멀티뷰 동시 생성 전략을 채택하고 있다.

#### (2) 노이즈 스케줄의 중요성 재인식
노이즈 스케줄이 단순한 하이퍼파라미터가 아니라 **모델의 글로벌 구조 학습 능력**에 직결된다는 통찰은 이후 확산 모델 연구 전반에 영향을 미쳤다.

#### (3) Prior 보존 미세 조정 전략의 일반화
SD prior를 최대한 보존하며 미세 조정하는 전략(Reference Attention, FlexDiffuse, 단계적 학습)은 다른 도메인에서도 적용 가능한 일반적인 방법론이다.

#### (4) Feed-forward 3D 생성 파이프라인의 기반
멀티뷰 이미지를 먼저 생성하고, 이를 3D 재구성에 활용하는 **feed-forward 파이프라인**의 핵심 구성 요소로 자리매김하였다.

### 4.2 향후 연구 시 고려할 점

1. **뷰 수 및 해상도 확장**: 6개 뷰로는 복잡한 객체의 완전한 커버리지가 어려움. 더 많은 뷰 또는 적응적 뷰 선택 연구 필요
2. **3D 표현과의 직접 통합**: 멀티뷰 이미지 → 3D 메쉬 변환의 간극을 해소하기 위해, 3D 표현(NeRF, 3D Gaussian Splatting 등)을 확산 과정에 직접 통합하는 연구
3. **비정형 객체 및 장면 확장**: 현재 주로 단일 객체에 초점. 복잡한 장면, 비정형 구조에 대한 확장
4. **실시간 생성**: 추론 속도 최적화(distillation, consistency model 등)
5. **평가 메트릭 개선**: LPIPS만으로는 3D 일관성을 완전히 평가하기 어려움. 3D-aware 메트릭 개발 필요

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

| 모델/논문 | 연도 | 핵심 특징 | Zero123++와의 비교 |
|----------|------|----------|-----------------|
| **DreamFusion** (Poole et al.) | 2022 | SDS loss를 통한 텍스트→3D, per-shape 최적화 | 최적화 기반으로 느림; Zero123++는 feed-forward |
| **Zero-1-to-3** (Liu et al.) | 2023 | 단일 이미지→새로운 뷰, 독립 생성 | 뷰 간 불일치 문제; Zero123++의 직접적 개선 대상 |
| **SyncDreamer** (Liu et al.) | 2023 | Zero-1-to-3 위에 3D-aware 어텐션 추가 | 추가 레이어 필요; elevation 변경 미지원 |
| **One-2-3-45** (Liu et al.) | 2023 | Zero-1-to-3 + SparseNeuS로 45초 내 3D 생성 | elevation 추정 모듈 필요; Zero123++는 불필요 |
| **MVDream** (Shi et al.) | 2023 | 텍스트→멀티뷰, 3D self-attention | 만화풍 텍스처 편향; Zero123++는 사실적 텍스처 유지 |
| **ProlificDreamer** (Wang et al.) | 2023 | VSD(Variational Score Distillation) | 고품질이나 매우 느린 최적화; Zero123++와 상호보완 가능 |
| **DreamGaussian** (Tang et al.) | 2023 | 3D Gaussian Splatting + SDS | 빠른 생성이나 품질 한계; Zero123++를 base로 활용 가능 |
| **Consistent123** (Lin et al.) | 2023 | Zero-1-to-3 기반 case-aware diffusion prior | Zero-1-to-3 위의 추가 레이어; Zero123++는 근본적 재설계 |
| **Wonder3D** (Long et al.) | 2023 | 멀티뷰 RGB + Normal 동시 생성 | Normal map 동시 생성으로 3D 재구성 품질 향상; Zero123++와 상호보완 |
| **Instant3D** (Li et al.) | 2023 | Feed-forward 단일 이미지→3D | Zero123++와 유사한 멀티뷰 생성 후 재구성 파이프라인 |
| **SV3D** (Voleti et al., Stability AI) | 2024 | Video diffusion 기반 멀티뷰 생성 | Stable Video Diffusion 활용; 연속적 뷰 궤적 생성 가능 |
| **LGM** (Tang et al.) | 2024 | 멀티뷰 → 3D Gaussian, feed-forward | Zero123++ 등의 멀티뷰 모델을 입력으로 활용 |
| **InstantMesh** (Xu et al.) | 2024 | 멀티뷰 → 메쉬, LRM 기반 | Zero123++를 멀티뷰 생성기로 사용하는 파이프라인 |

### 핵심 트렌드:

1. **독립 뷰 생성 → 조인트 멀티뷰 생성**: Zero123++가 이 전환의 핵심 기여자
2. **최적화 기반 → Feed-forward 기반**: 생성 속도의 극적 향상
3. **2D prior 활용의 정교화**: SD prior를 효과적으로 재활용하는 전략의 발전
4. **멀티뷰 생성 + 3D 재구성의 분리**: 각 단계의 전문화로 전체 품질 향상

---

## 참고자료

1. Shi, R., Chen, H., Zhang, Z., Liu, M., Xu, C., Wei, X., Chen, L., Zeng, C., Su, H. (2023). "Zero123++: a Single Image to Consistent Multi-view Diffusion Base Model." *arXiv:2310.15110*.
2. Liu, R., Wu, R., Van Hoorick, B., Tokmakov, P., Zakharov, S., Vondrick, C. (2023). "Zero-1-to-3: Zero-shot one image to 3d object." *ICCV 2023*.
3. Liu, Y., Lin, C., Zeng, Z., Long, X., Liu, L., Komura, T., Wang, W. (2023). "SyncDreamer: Generating multiview-consistent images from a single-view image." *arXiv:2309.03453*.
4. Shi, Y., Wang, P., Ye, J., Long, M., Li, K., Yang, X. (2023). "MVDream: Multi-view diffusion for 3d generation." *arXiv:2308.16512*.
5. Chen, T. (2023). "On the importance of noise scheduling for diffusion models." *arXiv:2301.10972*.
6. Salimans, T. & Ho, J. (2022). "Progressive distillation for fast sampling of diffusion models." *arXiv:2202.00512*.
7. Rombach, R., Blattmann, A., Lorenz, D., Esser, P., Ommer, B. (2022). "High-resolution image synthesis with latent diffusion models." *CVPR 2022*.
8. Poole, B., Jain, A., Barron, J.T., Mildenhall, B. (2022). "DreamFusion: Text-to-3D using 2D diffusion." *arXiv:2209.14988*.
9. Wang, Z., Lu, C., Wang, Y., Bao, F., Li, C., Su, H., Zhu, J. (2023). "ProlificDreamer: High-fidelity and diverse text-to-3d generation with variational score distillation." *arXiv:2305.16213*.
10. Tang, J., Ren, J., Zhou, H., Liu, Z., Zeng, G. (2023). "DreamGaussian: Generative Gaussian Splatting for efficient 3D content creation." *arXiv:2309.16653*.
11. Liu, M., Xu, C., Jin, H., Chen, L., Xu, Z., Su, H. (2023). "One-2-3-45: Any single image to 3D mesh in 45 seconds without per-shape optimization." *arXiv:2306.16928*.
12. Hang, T., Gu, S., Li, C., Bao, J., Chen, D., Hu, H., Geng, X., Guo, B. (2023). "Efficient diffusion training via min-snr weighting strategy." *arXiv:2303.09556*.
13. Speed, T. (2022). "FlexDiffuse: An adaptation of stable diffusion with image guidance." GitHub.
14. Zhang, L. (2023). "Reference-only control." GitHub Discussion.
15. Zhang, L., Rao, A., Agrawala, M. (2023). "Adding conditional control to text-to-image diffusion models." *ICCV 2023*.
16. Deitke, M. et al. (2023). "Objaverse: A universe of annotated 3D objects." *CVPR 2023*.
17. Podell, D. et al. (2023). "SDXL: Improving latent diffusion models for high-resolution image synthesis." *arXiv:2307.01952*.
18. Lambda Labs. (2022). "Stable Diffusion Image Variations." HuggingFace.
