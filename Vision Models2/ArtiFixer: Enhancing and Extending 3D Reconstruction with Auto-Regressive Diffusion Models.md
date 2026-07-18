# ArtiFixer: Enhancing and Extending 3D Reconstruction with Auto-Regressive Diffusion Models

## 1. 핵심 주장 및 주요 기여 요약

ArtiFixer는 NVIDIA 연구진(Riccardo de Lutio 등 10인)이 제안한 프레임워크로, 3D 신경 재구성과 자동회귀(auto-regressive) 비디오 생성을 연결하여 희소(sparse) 3D Gaussian Splatting(3DGS)을 향상시키는 새로운 프레임워크입니다.

핵심 주장은 3D 비전 분야에서 오랫동안 존재해온 이분법, 즉 Neural Reconstruction(3DGS 등)은 높은 충실도를 제공하지만 관측되지 않은 영역에서 실패하는 반면, Video Generation(Sora, Wan 2.1 등)은 그럴듯한 콘텐츠를 만들지만 정밀한 기하학적 제어가 부족하다는 문제를 해결하는 것입니다. ArtiFixer는 이 두 세계를 병합합니다.

주요 기여는 다음 두 가지입니다:
1. 새로운 opacity mixing 전략을 사용하여 기존 관측과의 일관성을 유지하면서도 미관측 영역에서 새로운 콘텐츠를 추정(extrapolate)할 수 있는 강력한 양방향(bidirectional) 생성 모델을 학습
2. 이를 한 번의 패스로 수백 프레임을 생성하는 causal auto-regressive 모델로 증류(distill)하여, 직접 novel view를 생성하거나 기저 3D 표현을 개선하는 pseudo-supervision으로 활용

성능 면에서 일반적으로 벤치마크되는 데이터셋에서 기존의 모든 baseline을 상당한 차이로 능가하며, 기존 최신 기법 대비 1-3 dB PSNR을 향상시켰습니다.

## 2. 문제 정의, 방법론, 모델 구조, 성능, 한계

### 해결하고자 하는 문제

3D Gaussian Splatting과 같은 scene별 최적화 방법은 최신 novel view synthesis 품질을 제공하지만 관측이 부족한 영역에서 잘 추정하지 못합니다. 생성적 prior를 활용해 이러한 영역의 아티팩트를 보정하는 방법들이 유망하지만 현재 두 가지 단점을 가지고 있습니다.

구체적으로:
- **확장성(Scalability) 문제**: 기존 방법들은 이미지 확산 모델이나 양방향 비디오 모델을 사용하는데, 이는 한 번에 생성할 수 있는 뷰 수가 제한되어 일관성을 위해 비용이 많이 드는 반복적 증류 과정이 필요합니다.
- **품질 문제**: 기존 연구에서 사용된 생성기들은 기존 장면 내용과 일관되지 않은 출력을 만들고 완전히 미관측된 영역에서는 전적으로 실패하는 경향이 있습니다.

### 제안 방법 및 수식

논문은 이 두 가지 통찰을 활용하는 2단계 파이프라인을 제안합니다.

**Phase I: Opacity Mixing 전략**

ArtiFixer는 2단계 파이프라인으로, Phase I에서는 opacity mixing 전략을 사용하여 양방향 비디오 확산 모델을 미세조정합니다. 이 전략의 핵심 아이디어는 mode collapse 문제를 해결하기 위해, 단순히 손상된 이미지를 모델에 입력하는 대신 3DGS opacity map($O$)을 사용하는 것입니다. 수식으로 표현하면:

$$z_{mix} = O_z \cdot z_{deg} + (1 - O_z) \cdot \epsilon$$

여기서 $z_{deg}$는 손상된(degraded) 렌더링의 latent, $\epsilon$은 순수 가우시안 노이즈, $O_z$는 해당 위치의 opacity 값입니다. 고신뢰 영역(높은 opacity)에서는 모델이 재구성 결과에 충실하게 유지되고, "구멍"(낮은 opacity) 영역에서는 순수 가우시안 노이즈를 주입하여 확산 모델에 그럴듯한 바닥, 벽, 가구를 인페인팅할 수 있는 "창의적 자유"를 부여합니다.

**Phase II: Causal Auto-Regressive 증류**

양방향 모델(모든 프레임을 동시에 보는)은 대화형 사용에 너무 무겁기 때문에, ArtiFixer는 이를 Causal Auto-Regressive 모델로 증류합니다. block-causal mask와 rolling KV cache를 적용함으로써, 한 번의 패스로 수백 프레임을 생성할 수 있으며, 노이즈가 있는 3D 렌더링을 장기 생성에서 흔히 발생하는 drift를 방지하는 "guide rail"로 사용합니다.

학습 절차와 관련하여, causal 모델은 양방향 teacher 모델의 가중치로부터 초기화되며, ODE trajectory 데이터셋 생성이 필요한 기존 연구의 ODE 초기화 프로토콜보다 단순한 전략을 따릅니다. 대신 block-causal mask를 적용하고, Diffusion Forcing 방식처럼 각 입력 프레임에 서로 다른 노이즈 레벨을 가하고, 그 외에는 동일한 입력과 학습 프로토콜을 사용합니다. 이후 Self Forcing과 유사한 학습 전략을 채택하여 비디오 청크를 순차적으로 생성하고 KV caching을 통해 이전에 생성된 청크에 조건을 부여하되, 카메라 제어와 순수 노이즈로부터의 생성이 저하되는 것을 막기 위해 dropout을 계속 적용합니다.

### 모델 구조 (3가지 변형)

논문은 세 가지 변형을 평가합니다: 직접 auto-regressive 생성기로부터 novel view를 렌더링하는 ArtiFixer, 그 출력을 기저 3D 표현으로 다시 증류하는 ArtiFixer3D, 그리고 ArtiFixer3D 위에 auto-regressive 모델을 후처리로 재적용하는 ArtiFixer3D+(Difix3D+ 방식과 유사)입니다.

모든 변형이 유사한 렌더링을 만들지만, ArtiFixer의 결과가 약간 더 선명하고, ArtiFixer3D는 약간의 흐림을 대가로 소스 이미지와 더 일관되며, ArtiFixer3D+는 높은 일관성을 유지하면서 선명함을 회복합니다.

### 성능

- Nerfbusters와 DL3DV에서의 아티팩트 제거에서 모든 ArtiFixer 변형이 상당한 차이로 기존 방법들을 능가하며 PSNR을 2 dB 향상시켰습니다.
- 훈련 뷰가 관측하지 못한 넓은 영역을 만드는 프로토콜로 DL3DV 장면을 재구성했을 때, 다음으로 좋은 방법인 GenFusion보다 거의 3 dB PSNR을 능가했습니다.
- 효율성 측면에서 순수 노이즈가 아닌 렌더링에서 시작하기 때문에 대부분의 경우 4단계 미만의 denoising step으로도 그럴듯한 시각적 결과를 생성할 수 있습니다.

### 한계

논문에서 언급된 한계점들:
- 렌더링에서 시작하기 때문에 빠른 생성이 가능하지만, 빈 영역에서는 선명도와 시간적 일관성이 다소 저하됩니다.
- ArtiFixer3D는 소스 이미지와 더 일관되지만 명시적 3D 표현으로 인한 약간의 흐림을 대가로 하며, 이는 Table 1에서 PSNR과 SSIM의 소폭 증가 및 LPIPS와 FID의 작은 저하로 나타남이 확인됩니다.
- 효율성 관점에서 3D 증류는 여전히 때때로 바람직한데, 기존 방법들은 뷰 생성과 3D 재구성을 번갈아 수행하는 점진적 증류 과정을 요구하여 상당한 학습 시간 오버헤드를 초래합니다.

## 3. 모델의 일반화 성능 향상 가능성

ArtiFixer의 일반화 성능과 관련하여 특히 주목할 부분은 다음과 같습니다.

**1) 임의 길이 시퀀스로의 일반화**: 이 접근법은 단순하지만 (주어진 계산 예산에 대해 더 짧은 비디오의 더 다양한 세트로 학습하기 때문에) 학습 수렴을 가속화하고, 실험에서 보여지듯 임의 길이의 비디오로 일반화됩니다. 이는 고정된 프레임 수에 묶여 있던 기존 양방향 모델의 근본적 한계를 극복하는 지점입니다.

**2) Opacity 기반 조건화를 통한 미관측 영역 일반화**: Opacity mixing 전략은 모델이 관측된 영역에서는 사실성을 유지하고 미관측 영역에서는 창의적으로 콘텐츠를 생성하도록 학습시킵니다. 이는 다양한 정도의 재구성 손상(다양한 opacity 분포)에 대해 모델이 일관되게 대응할 수 있게 하는 핵심 메커니즘으로, 훈련 시 보지 못한 손상 패턴에도 유연하게 대처할 잠재력을 제공합니다.

**3) Auto-regressive 구조의 확장성**: 기존 연구는 확산 모델 출력을 3D 표현으로 증류하여 일관성을 확보하는데, 이는 시간적 불안정성을 보이거나 양방향 모델이 한 번에 생성할 수 있는 프레임 수에 제한을 받기 때문입니다. ArtiFixer의 auto-regressive 모델은 순차적으로 임의 길이 렌더링을 생성할 수 있어 이러한 제약에서 자유롭습니다. 이는 대규모 장면이나 긴 궤적으로 확장할 때 일반화 잠재력이 높음을 시사합니다.

**4) 다양한 손상 정도에 대한 강건성**: 초기 렌더링이 매우 손상된 경우에도 방법이 그럴듯하고 일관된 기하 구조를 재구성할 수 있다는 점은, 다양한 sparse-view 시나리오(소수 뷰, 넓은 미관측 영역 등)에 걸쳐 일반화될 가능성을 보여줍니다.

다만, 논문 자체에서 일반화 성능에 대한 명시적 분석(예: 도메인 전이, 다른 카메라 궤적, 실내/실외 장면 간 전이)을 깊이 다루었는지는 확인된 자료에서 명확히 드러나지 않으며, 이는 향후 연구에서 보완이 필요한 부분으로 보입니다.

## 4. 향후 연구에 미치는 영향 및 고려할 점

**영향**:
- ArtiFixer는 "reconstruction-then-generation" 파이프라인에서 반복적 증류 없이 단일 패스로 고품질 결과를 얻는 방향을 제시함으로써, 향후 3D 재구성과 생성 모델 결합 연구에서 auto-regressive 구조 채택을 가속화할 가능성이 있습니다.
- Opacity mixing과 같은 명시적 3D 신뢰도 신호를 diffusion 조건화에 활용하는 방식은, 다른 3D-aware 생성 작업(예: 4D 동적 장면 생성, 로보틱스 시뮬레이션)에도 확장 적용될 수 있습니다.
- 가상/증강현실 및 물리 AI를 위한 closed-loop simulation과 같이 photorealistic rendering이 필수적인 응용 분야에서 실용성이 높아 산업적 파급력도 클 것으로 예상됩니다.

**향후 연구 시 고려할 점**:
1. **효율성과 품질의 트레이드오프**: ArtiFixer, ArtiFixer3D, ArtiFixer3D+ 간의 선명도-일관성 트레이드오프가 확인된 만큼, 실제 응용에서 어떤 변형을 채택할지에 대한 명확한 가이드라인 마련이 필요합니다.
2. **미관측 영역의 완전한 "환각" 통제**: 완전히 관측되지 않은 큰 영역에 대한 콘텐츠 생성의 사실성(faithfulness) 검증 및 사용자 의도(텍스트 프롬프트 등)와의 정합성 향상이 중요한 과제입니다.
3. **다양한 3D 표현과의 통합**: 현재 3DGS 중심으로 설계된 프레임워크가 NeRF, mesh, 또는 다른 최신 표현(예: 3DGUT)과도 원활히 통합될 수 있는지에 대한 검증이 필요합니다.
4. **일반화 벤치마크 강화**: in-the-wild 캡처, 다양한 카메라 궤적, 실내외 혼합 시나리오 등에서의 체계적인 일반화 성능 평가가 후속 연구의 중요한 방향이 될 것입니다.

## 5. 2020년 이후 관련 최신 연구 비교 분석

| 연구 | 접근 방식 | 한계 |
|---|---|---|
| **ReconFusion** (Wu et al., 2024) | NeRF + diffusion prior | 이미지 기반 확산으로 다중 뷰 일관성 제한 |
| **Difix3D+** (Wu et al., 2025a) | 단일 스텝 이미지 확산 모델로 아티팩트 보정 | 관측 영역에서는 양호하지만 미관측 영역 처리 실패 |
| **GenFusion** (Wu et al., 2025b) | DiT 기반 비디오 확산 + 재구성/생성 closed-loop | 한 번에 16프레임만 생성하여 반복적 증류가 필요하며 특히 빈 영역에서 흐린 결과 초래 |
| **Gen3C** (Ren et al., 2025) | 카메라 제어 가능한 비디오 생성 | 렌더링은 더 선명하지만 종종 소스 콘텐츠를 반영하지 못함 |
| **3DGS-Enhancer** (Liu et al., 2024) | 2D diffusion prior로 view-consistent 향상 | 다중 뷰 일관성 확보에 한계 |
| **GSFixer** (2025) | Reference-guided 비디오 확산 prior | Difix3D+, GenFusion 대비 PSNR, SSIM, LPIPS 개선이나 여전히 최적화 기반 반복 필요 |
| **G4Splat** (2026) | Geometry-guided generative prior | GenFusion, See3D, GuidedVD은 미관측 영역을 환각할 수 있으나 완성 결과가 흐리고 floater로 손상 |
| **ArtiFixer** (2026, 본 논문) | Opacity mixing + causal auto-regressive 증류 | 단일 패스로 수백 프레임 생성, 반복 증류 불필요, 1-3dB PSNR 개선 |

이러한 비교에서 알 수 있듯이, ArtiFixer는 기존 연구들이 공통적으로 겪던 **(1) 제한된 프레임 수로 인한 반복적 증류 필요성**과 **(2) 미관측 영역에서의 일관성 부족**이라는 두 가지 문제를 opacity mixing과 causal auto-regressive 증류라는 조합으로 동시에 해결하려 한 점에서 차별성을 가집니다.

---

### 참고 문헌 및 출처
1. arXiv:2603.00492 - "ArtiFixer: Enhancing and Extending 3D Reconstruction with Auto-Regressive Diffusion Models" (de Lutio et al., 2026)
2. NVIDIA Research 프로젝트 페이지 - research.nvidia.com/labs/sil/projects/artifixer/
3. ArtiFixer 프로젝트 웹사이트 - artifixer2026.github.io
4. alphaXiv 개요 - alphaxiv.org/overview/2603.00492v1
5. wispaper.ai 블로그 요약
6. Cool Papers - papers.cool/arxiv/2603.00492
7. G4Splat: Geometry-Guided Gaussian Splatting with Generative Prior (arXiv:2510.12099)
8. GSFixer: Improving 3D Gaussian Splatting with Reference-Guided Video Diffusion Priors (arXiv:2508.09667)
9. FixingGS: Enhancing 3D Gaussian Splatting via Training-Free Score Distillation (arXiv:2509.18759)
10. GIFSplat: Generative Prior-Guided Iterative Feed-Forward 3D Gaussian Splatting (arXiv:2602.22571)
