# Prox·E: Fine-Grained 3D Shape Editing via Primitive-Based Abstractions

## 1. 핵심 주장과 주요 기여

**핵심 주장**: 기존 2D 이미지 편집 모델 기반 3D 편집 파이프라인은 외형(appearance) 수정에는 효과적이지만, "다리를 1.5배 넓히기"와 같이 정확한 메트릭 추론이 필요한 **미세한 구조적 편집(fine-grained structural editing)**에는 근본적으로 한계가 있다. 이는 픽셀 기반 확산 모델이 3D 공간에서의 메트릭 속성에 대한 명시적 이해가 부족하기 때문이다.

**주요 기여**:
1. 3D 형상을 슈퍼쿼드릭(superquadric) 기반 프록시로 추상화하고, VLM이 이 프록시를 직접 편집하도록 하는 **training-free 프레임워크** 제안
2. 원본 구조, 워핑된 구조, 편집된 프록시의 latent를 블렌딩하는 **proxy-induced denoising** 전략 개발
3. 구조 편집과 외형 편집을 분리하여, 구조는 프록시로 외형은 2D 이미지 편집기로 처리하는 하이브리드 접근

## 2. 문제, 방법론, 모델 구조, 성능, 한계

### 해결하고자 하는 문제
- 2D 확산 모델(Flux-Kontext, Nano-Banana)은 "의자 다리 줄이기", "좌석을 1.5배 넓히기" 같은 메트릭 기반 지시에서 실패 (Fig. 2)
- 단일/소수 뷰 편집이 전체 3D 형상에 일관되게 전파되기 어려움

### 제안 방법 (수식 포함)

**슈퍼쿼드릭 표현**:
$$f(x, y, z; \lambda) = \left( \left|\frac{x}{a_1}\right|^{\frac{2}{\epsilon_2}} + \left|\frac{y}{a_2}\right|^{\frac{2}{\epsilon_2}} \right)^{\frac{\epsilon_2}{\epsilon_1}} + \left|\frac{z}{a_3}\right|^{\frac{2}{\epsilon_1}} = 1$$

각 프리미티브는 11개 파라미터로 구성: 스케일 $\mathbf{a} \in \mathbb{R}^3_{>0}$, 형상 지수 $\epsilon \in \mathbb{R}^2_{>0}$, 이동 $\mathbf{t} \in \mathbb{R}^3$, 회전 $\mathbf{r} \in \mathbb{R}^3$

**변환 행렬**: 각 프리미티브의 local-to-world 변환은 $M = TRS$로 구성되며, 편집 전후 프리미티브 쌍 $(q_{orig}^{(i)}, q_{edit}^{(i)})$에 대해 상대 변환은:
$$M_{rel}^{(i)} = M_{edit}^{(i)} (M_{orig}^{(i)})^{-1}$$

**Proxy-Induced Denoising**: 세 가지 공간 마스크 $\mathcal{M}\_{uc}$(unchanged), $\mathcal{M}\_{ed}$(edited), $\mathcal{M}_{new}$(added/removed)를 정의하여 각 영역별로 다른 latent를 주입:

- **Unchanged 영역**: $z_t[v] \leftarrow z_t^{orig}[v]$ (원본 구조 완전 보존, $t_{uc}$까지 적용)
- **Edited 영역**: $z_t[v] \leftarrow z_t^{warp}[v]$ ( $t_{warp}$까지 워핑된 latent 주입)
- **New 영역**: 진화하는 latent $z_t$ 그대로 사용 (프록시로부터 생성)

**Appearance 단계**에서는 픽셀 위치 역변환을 통해 특징을 재배치:
$$v' = (M_{rel}^{(i)})^{-1} v$$

### 모델 구조 (Fig. 3 기준 3단계)
1. **Abstraction Editing**: SuperDec으로 포인트 클라우드→슈퍼쿼드릭 분해 → VLM이 색상 코드화된 JSON을 편집 (Chain-of-Thought + 시각적 검증 루프, 최대 3회 반복)
2. **Structure Generation**: TRELLIS의 구조 확산 모델 기반, DDIM inversion 후 3-way latent 블렌딩으로 구조 편집
3. **Appearance Refinement**: FLUX.1-Kontext로 단일 뷰 편집 → SLAT 공간에서 조건화된 확산으로 외형 적용

### 성능 향상
ShapeTalk 벤치마크(Table 1) 기준, ChangeIt3D, BlendedPC(학습 기반), Spice-E, EditP23, VoxHammer, TRELLIS 대비:
- **LPIPS 0.10, DINO-I 0.92**로 최고 정체성 보존
- **FID 32.60**으로 최고 품질
- **VQA 0.71**로 최고 편집 충실도
- 사용자 연구(44명)에서 모든 경쟁 모델 대비 최고 승률 (TRELLIS 대비 78.8% 편집 품질 승률)

Ablation(Table 2)에서 모든 latent 소스(원본/워핑/프록시)가 성능에 기여함을 확인; 프록시 없이는 identity는 좋지만 편집 충실도 저하.

### 한계
1. **분해 품질 의존성**: SuperDec이 서로 다른 부품을 하나의 프리미티브로 잘못 병합하면 세밀한 제어 불가능 (예: 의자 스핀들이 프레임에 흡수됨, Fig. 5)
2. **VLM 성능 의존성**: 공간 추론이 약한 VLM은 파이프라인을 안정적으로 지원하지 못함
3. **씬 편집 확장성**: TRELLIS의 복셀 해상도 제약으로 크거나 복잡한 씬에는 추가 파티셔닝 필요
4. **런타임**: 총 10분 28초로, VoxHammer(9분 7초) 다음으로 느림; SLAT inversion이 큰 비중 차지

## 3. 모델의 일반화 성능 향상 가능성

Prox·E의 일반화 가능성은 다음 세 가지 축에서 주목할 만하다:

**(1) 분해 백본에 대한 아키텍처 독립성 (Backbone-Agnostic Design)**
저자들은 명시적으로 "our framework is agnostic to the decomposition backbone. As more expressive 3D decomposition methods emerge, our pipeline will directly benefit from improved granularity and semantic disentanglement **without architectural modifications**"라고 언급한다. 이는 SuperDec을 더 발전된 primitive decomposition 방법(예: PrimitiveAnything, Neural Parts 등)으로 교체만 해도 성능이 자동으로 향상될 수 있음을 의미하며, 이는 모듈형 설계의 강력한 일반화 이점이다.

**(2) VLM 성능 향상에 따른 자연스러운 스케일링**
"we expect our framework to naturally benefit from continued improvements in reasoning and instruction-following abilities"라고 명시. Table 3에서 Qwen2.5-VL, SAIL-VL 등 여러 VLM으로 실험한 결과 VLM 역량에 따라 VQA 점수가 크게 달라짐을 보였는데(0.28→0.54까지), 이는 foundation model이 발전할수록 별도 재학습 없이 시스템 전체 성능이 향상되는 **training-free 프레임워크의 스케일링 이점**을 시사한다.

**(3) 범주 초월 일반화(Cross-category generalization)**
- ShapeTalk(의자/테이블/램프)로 정량 평가 후, Edit3D-Bench(100개의 고품질 다양한 객체)에서 정성적 일반화를 입증(Fig. 7, 8)
- 학습 기반 방법(ChangeIt3D, BlendedPC, Spice-E)은 카테고리별 체크포인트가 필요해 학습 분포를 벗어나면 일반화가 제한적인 반면, Prox·E는 "완전히 training-free이며 임의의 입력 형상에 직접 적용 가능"
- 씬 수준 편집(Fig. 13)으로의 확장 가능성도 시연되어, 객체 단위를 넘어선 일반화 잠재력을 보여줌

**일반화의 구조적 근거**: 이 프레임워크는 3D 형상별 특화 학습 없이 (1) 범용 3D decomposition, (2) 범용 VLM, (3) 범용 3D diffusion backbone(TRELLIS), (4) 범용 2D 이미지 편집기(FLUX-Kontext)라는 4개의 사전학습된 범용 컴포넌트를 조합함으로써, 각 컴포넌트의 개별 발전이 곧바로 전체 시스템의 일반화 성능 향상으로 이어지는 **compositional generalization** 구조를 갖는다.

## 4. 향후 연구에 미치는 영향과 고려사항

### 연구에 미치는 영향

1. **Neuro-symbolic 3D 편집 패러다임 제시**: 명시적 기하 프리미티브(symbolic)와 신경망 생성 모델(neural)을 결합하는 방식은, 순수 신경망 기반 접근(EditP23, NANO3D 등)의 한계(메트릭 추론 부족)를 극복하는 새로운 방향을 제시한다. 이는 향후 3D 편집 연구가 "얼마나 강력한 확산 모델인가"뿐 아니라 "얼마나 좋은 중간 표현(abstraction)을 사용하는가"에 집중하도록 유도할 수 있다.

2. **VLM을 공간 추론 에이전트로 활용하는 방법론**: 색상 코드화 + JSON 파라미터화 + 시각적 검증 루프(chain-of-thought + self-verification) 조합은, VLM이 3D 기하를 직접 조작하는 후속 연구(로보틱스의 3D 씬 이해, CAD 자동화 등)에 응용 가능한 템플릿을 제공한다.

3. **평가 방법론 기여**: CoT를 결합한 VQAScore(Fig. 10, 11)는 기존 블랙박스 방식보다 신뢰도 높은 평가를 가능케 하여, 향후 3D/이미지 편집 벤치마크의 표준 평가 프로토콜로 채택될 가능성이 있다.

### 향후 연구 시 고려할 점

1. **분해의 의미론적 정확성 문제**: 프리미티브 분해가 부정확하면 전체 파이프라인이 실패하므로, "의미 인식(semantic-aware)" decomposition 방법의 발전이 병행되어야 한다. 단순 기하학적 피팅을 넘어 기능적/의미적 단위로 분해하는 연구가 필요하다.

2. **연쇄적 오류(error propagation) 관리**: LLM 파싱→VLM 편집→구조 생성→외형 정제로 이어지는 다단계 파이프라인에서 초기 단계 오류(예: "sits closer to ground"를 "shorter legs"로 잘못 해석)가 후속 단계로 전파된다. 각 단계의 불확실성 정량화 및 견고성 확보가 중요한 연구 방향이다.

3. **동적/시간적 3D 편집으로의 확장**: 결론에서 "opens new opportunities for scalable and controllable generation in more complex and dynamic 3D settings"라고 언급했듯, 정적 형상 편집을 넘어 애니메이션, 동적 씬, 4D 편집으로의 확장이 자연스러운 다음 단계다.

4. **계산 효율성**: SLAT inversion이 4분 18초로 전체 런타임의 상당 부분을 차지하므로, 실시간/대화형 편집 워크플로우를 위해서는 inversion-free 또는 경량화된 대안이 필요하다.

5. **씬 단위 편집의 확장성**: 복셀 해상도 제약을 극복하기 위한 계층적(hierarchical) 또는 객체별 분할 처리 방식의 체계적 연구가 필요하다.

---

**참고 문헌 (출처)**:
- Sella, E., Phung, H., Amiel, N., Litany, O., Patashnik, O., & Averbuch-Elor, H. (2026). "Prox·E: Fine-Grained 3D Shape Editing via Primitive-Based Abstractions." SIGGRAPH Conference Papers '26. arXiv:2604.23774v2 [cs.GR]

논문 내 인용된 주요 비교 연구들 (본문에서 직접 언급됨):
- Xiang, J. et al. (2025). "Structured 3D latents for scalable and versatile 3D generation" (TRELLIS)
- Li, L. et al. (2025). "VoxHammer: Training-free precise and coherent 3d editing in native 3d space"
- Bar-On, R. et al. (2025). "EditP23: 3D Editing via Propagation of Image Prompts to Multi-View"
- Achlioptas, P. et al. (2022, 2023). "ChangeIt3D" / "ShapeTalk"
- Sella, E. et al. (2024, 2025). "Spice-E" / "Blended Point Cloud Diffusion"
- Fedele, E. et al. (2025). "Superdec: 3d scene decomposition with superquadric primitives"
- Labs, Black Forest et al. (2025). "FLUX.1 Kontext"
