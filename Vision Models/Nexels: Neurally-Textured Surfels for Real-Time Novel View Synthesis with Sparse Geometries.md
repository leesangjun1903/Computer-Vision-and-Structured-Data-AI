
# Nexels: Neurally-Textured Surfels for Real-Time Novel View Synthesis with Sparse Geometries

> **논문 정보**
> - **제목**: Nexels: Neurally-Textured Surfels for Real-Time Novel View Synthesis with Sparse Geometries
> - **저자**: Victor Rong, Jan Held, Victor Chu, Daniel Rebain, Marc Van Droogenbroeck, Kiriakos N. Kutulakos, Andrea Tagliasacchi, David B. Lindell (University of Toronto, Vector Institute, Simon Fraser University, University of Liège, University of British Columbia)
> - **arXiv**: [2512.13796](https://arxiv.org/abs/2512.13796) (December 2025)
> - **코드**: [github.com/victor-rong/nexels](https://github.com/victor-rong/nexels)

---

## 1. 핵심 주장 및 주요 기여 요약

Gaussian Splatting은 novel view synthesis에서 인상적인 결과를 달성했지만, 장면의 기하학적 구조가 단순함에도 불구하고 고도로 텍스처된 장면을 모델링하기 위해 수백만 개의 프리미티브가 필요하다.

이에 대한 핵심 주장은 다음과 같습니다:

**Nexels는 밀집 프리미티브에 대한 의존성을 제거하면서도 실시간 고품질 렌더링을 달성하는 novel view synthesis 최초의 표현 방식**이며, 장면 수준의 데이터셋을 모델링하는 데 최신 연구들이 제안하는 것보다 훨씬 적은 기하학적 요소가 필요함을 보여준다.

**주요 기여 5가지**:

1. 기하학 표현을 위해, 표면과 날카로운 경계를 더 잘 재구성할 수 있도록 **미분 가능한 quad indicator**를 도입하였다.
2. 외관 표현을 위해, 가장 관련성 높은 프리미티브에만 시점 의존적 텍스처를 제공하는 **월드 공간 neural field**를 학습함으로써 세밀한 디테일을 효율적으로 포착하면서 계산량을 낮게 유지한다.
3. 점 기반 표현이 고주파 텍스처를 가진 영역을 정확하게 재구성하는 데 있는 한계를 부각시키는 **새로운 데이터셋**을 도입하였다.
4. 실외 장면에서 3DGS 대비 **9.7배 적은 프리미티브**로 동등한 지각적 품질을 달성하고, 동시대 텍스처 방법들보다 **2배 이상 빠른 렌더링**과 더 나은 측광 품질을 달성하였다.

---

## 2. 해결하고자 하는 문제, 제안 방법, 모델 구조, 성능 및 한계

### 2-1. 해결하고자 하는 문제

이 논문은 3D Gaussian Splatting(3DGS)과 같은 점 기반 3D 장면 표현의 핵심적인 한계, 즉 기하학적 구조는 단순하지만 고도로 텍스처된 표면을 가진 장면을 표현할 때의 비효율성을 해결하고자 한다. 점 기반 렌더링은 기하학적 파라미터와 외관 파라미터를 강하게 결합시켜, 위상적으로 단순한 표면에서도 세밀한 외관 디테일을 위해 방대한 수의 프리미티브가 필요하다.

반면, 전통적인 메시 기반 접근법은 텍스처를 사용해 기하학과 외관을 분리하지만, 최적화 및 미분 가능성 문제로 인해 novel view synthesis에서 경쟁력 있는 성능을 달성하지 못했다.

### 2-2. 제안하는 방법

#### (a) 핵심 아이디어

Nexels는 neurally-textured surfel 프리미티브로, **하이브리드 explicit/implicit 표현**을 통해 기하학과 외관을 분리하여 렌더링 품질이나 속도를 희생하지 않고 메모리와 연산을 크게 줄인다.

#### (b) 기하학 표현: Surfel 파라미터화

각 서펠은 평균 위치 $\mu_i \in \mathbb{R}^3$, 회전 행렬 $R_i \in SO(3)$ (첫 두 축이 서펠 방향, 세 번째 축이 법선 역할을 함), 그리고 2D 스케일로 파라미터화된다.

기하학적 형태를 나타내기 위해 **일반화된 Gaussian 커널(generalized Gaussian kernel)** 기반의 미분 가능한 quad indicator를 사용합니다. 표준 3D Gaussian과 달리, 서펠은 flat한 2D 디스크 구조로 표면에 가깝게 정렬되어 표면 기하학을 희소하게 표현합니다. 픽셀 $\mathbf{p}$에 대한 서펠 $i$의 알파 기여는 다음과 같이 표현됩니다 (논문의 일반화된 커널 형태):

$$\alpha_i(\mathbf{p}) = o_i \cdot \exp\!\left(-\frac{1}{2} \mathbf{d}_i(\mathbf{p})^\top \Sigma_i^{-1} \mathbf{d}_i(\mathbf{p})\right)^{\gamma}$$

여기서 $o_i$는 불투명도, $\Sigma_i$는 2D 공분산, $\gamma$는 generalized Gaussian의 형태 파라미터입니다. $\gamma$가 커질수록 보다 평평한 디스크(quad-like) 형태에 가까워집니다.

#### (c) 외관 표현: Top-K 뉴럴 텍스처링

외관을 위해 **전역 neural field와 프리미티브별 색상의 조합**을 사용하며, neural field는 픽셀당 고정된 수의 프리미티브를 텍스처링함으로써 추가적인 계산량을 낮게 유지한다.

구체적으로, 렌더링 가중치가 가장 높은 $K$개의 프리미티브는 뉴럴 텍스처를 사용하고, 나머지 프리미티브는 일반적인 프리미티브별 색상으로 폴백(fallback)한다.

#### (d) 2패스 렌더링 파이프라인

Nexels의 렌더링은 두 패스로 구성된다. **Collection Pass**에서는 관련 서펠들의 알파 컴포지팅을 수행하고 블렌딩 가중치를 이용해 픽셀당 텍스처링을 위한 Top-K 프리미티브를 식별한다. **Texturing Pass**에서는 선택된 프리미티브들의 교차 위치를 계산하고 neural field를 쿼리하여 시점 의존적 텍스처를 얻는다. 최종 픽셀 색상은 비텍스처 기여와 뉴럴 텍스처 기여를 속도와 미분 가능성이 최적화된 방식으로 통합한다.

최종 픽셀 색상 $C(\mathbf{p})$는 다음과 같이 표현할 수 있습니다:

$$C(\mathbf{p}) = \sum_{i \in \mathcal{T}_K(\mathbf{p})} w_i \cdot f_{\text{neural}}(\mathbf{x}_i, \mathbf{d}) + \sum_{j \notin \mathcal{T}_K(\mathbf{p})} w_j \cdot \mathbf{c}_j$$

여기서:
- $\mathcal{T}_K(\mathbf{p})$: 픽셀 $\mathbf{p}$에서 블렌딩 가중치 상위 $K$개의 프리미티브 집합
- $w_i = \alpha_i \prod_{k < i}(1 - \alpha_k)$: alpha compositing 가중치
- $f_{\text{neural}}(\mathbf{x}_i, \mathbf{d})$: 교차점 $\mathbf{x}_i$와 시선 방향 $\mathbf{d}$로부터 neural field가 출력하는 뷰 의존적 색상
- $\mathbf{c}_j$: 프리미티브 $j$의 명시적 색상

뉴럴 텍스처를 위한 아키텍처로는 **Instant-NGP 아키텍처**를 사용한다.

#### (e) 적응적 밀도 제어

이 설계는 neural field 평가를 최소화하고 빠른 래스터화를 위해 명시적 기하학을 활용함으로써 실시간 성능을 유지한다. 적응적 밀도 제어 메커니즘은 효율적인 장면 표현을 지원한다.

---

### 2-3. 모델 구조 요약

```
입력: 다시점 이미지 + COLMAP 포인트 클라우드
       ↓
[기하학 분기]                    [외관 분기]
N개의 Surfel 집합               Instant-NGP 기반 전역 Neural Field
(μ_i, R_i, s_i, o_i)           (해시 그리드 + 소형 MLP)
       ↓                               ↓
  미분 가능한                    Top-K 프리미티브 텍스처링
  Quad Indicator                 (픽셀당 K=2 권장)
       ↓                               ↓
    Collection Pass ──────────────────→ Texturing Pass
       ↓                               ↓
              최종 픽셀 색상 C(p) 출력
```

---

### 2-4. 성능 향상

제안 방법은 실외 장면에서 3D Gaussian Splatting과 동등한 지각적 품질을 유지하면서 **9.7배 적은 프리미티브와 5.5배 적은 메모리**를 사용하며, 실내 장면에서는 **31배 적은 프리미티브와 3.7배 적은 메모리**를 사용한다.

또한 기존 텍스처 프리미티브 방법들보다 **2배 빠르게 렌더링**하면서 시각적 품질도 향상시킨다.

수치 비교 (Bicycle 장면, Mip-NeRF360):

| 방법 | LPIPS | 프리미티브 수 | FPS |
|------|-------|------------|-----|
| 3DGS | 0.216 | 4,400K | 93 |
| BBSplat | 0.302 | 400K | 13 |
| NeST-Splatting | 0.248 | 1,300K | 20 |
| **Nexels (ours)** | **0.216** | **400K** | **60** |

주요 구성 요소의 Ablation study에서 neural texture가 가장 중요한 구성 요소임이 확인되었다. Neural texture 없이는 LPIPS를 포함한 모든 시각적 지표가 악화되었으며, 질적으로도 텍스처 없는 렌더링은 배경 세부 정보를 잃는다.

$K$ 값의 최적화 실험에서, 어떤 텍스처라도 추가하면 지표가 급격히 향상되었으나(즉 $K=2$ 이상), 놀랍게도 $K$를 2 이상으로 늘리면 품질이 저하되었다. 이는 더 높은 $K$ 값에서 테스트 뷰 렌더링의 배경에 노이즈가 관찰되었으며, 훈련 뷰에서 텍스처가 적용된 적 없는 프리미티브가 테스트 뷰에서 텍스처링된다는 것을 시사한다.

---

### 2-5. 한계

기하학적 부정확성이 존재한다. 예를 들어, 40K의 낮은 프리미티브 수로 Bonsai 장면을 학습할 경우 얇은 구조물을 포착하지 못한다.

추가적으로 다음 한계를 확인할 수 있습니다:
- Nexels는 현재 GPU 텐서 코어에 의존적이다.
- BBSplat에 비해 훈련 시간이 더 길 수 있는데, 해시 그리드 룩업이 광선-프리미티브 교차마다 수행되어 광선당 수천 번의 랜덤 접근이 발생하기 때문이다.
- 각 장면에 대한 per-scene 최적화가 필요하여 제로샷 일반화에는 한계가 있다.

---

## 3. 모델의 일반화 성능 향상 가능성

### 3-1. 현재 일반화 관련 구조적 장점

Nexels는 희소한 서펠로 기하학을, 공유된 적응적 neural field로 외관을 결합하는 핵심 아이디어를 통해 **시각적 품질과 실시간 렌더링 성능을 유지하는 컴팩트한 표현**을 가능하게 한다.

이 **공유 전역 neural field** 구조는 일반화에 있어 중요한 함의를 가집니다:

1. **모듈화된 구조**: 시스템의 모듈성—명시적 기하학과 공유 뉴럴 텍스처—은 레이 트레이싱 및 레벨-오브-디테일(LoD) 방식과의 통합을 용이하게 한다.

2. **희소 기하학의 확장성**: 실험 결과는 프리미티브 수와 메모리를 대폭 줄이면서 강력한 성능을 보여주며, **대규모 및 도메인 적응 시나리오**로의 확장 가능성을 시사한다.

3. **도시 규모 장면으로의 확장**: 향후 연구에서 도시 규모 장면 모델링을 위한 **레벨-오브-디테일 구조 통합** 및 **레이 트레이싱 응용**이 유망하게 제안된다.

4. **Top-K 방식의 확장 가능성**: Top-K 방식은 neural field와 프리미티브별 피처를 결합하는 텍스처를 지원하도록 확장할 수 있다.

5. **AR/VR 및 로보틱스 응용**: 실용적으로, Nexels는 메모리와 렌더링 속도가 중요한 AR/VR, 텔레프레즌스, 로보틱스 등 대규모 환경에서의 novel view synthesis를 가능하게 한다.

### 3-2. 일반화 성능 향상을 위한 방향성

이론적 관점에서, Nexels는 하이브리드 explicit/implicit 필드 구성의 유용성을 입증하고, **희소 미분 가능 렌더링 시스템**에 대한 추가 연구의 템플릿 역할을 한다. 일반화된 Gaussian 커널은 서펠 및 프리미티브 모델링에 새로운 기회를 제시하며, **적응적 계층 구조**로의 확장 가능성을 열어준다.

특히 feed-forward 일반화(scene-agnostic)를 위해서는 다음을 고려할 수 있습니다:
- 대규모 3D 데이터셋(Objaverse, CO3D 등)으로 Nexels의 neural field 부분을 사전 학습
- Cross-scene shared neural field를 통한 텍스처 사전(texture prior) 학습
- 단안 또는 희소 입력 뷰(sparse view) 조건에서의 피드포워드 추론 파이프라인 개발

---

## 4. 최신 관련 연구 비교 분석 (2020년 이후)

| 방법 | 연도 | 기하학 | 외관 | 실시간? | 특징 |
|------|------|--------|------|---------|------|
| **NeRF** (Mildenhall et al.) | 2020 | 암시적 볼륨 | MLP | ❌ | 최초 신경 복사장 |
| **3DGS** (Kerbl et al.) | 2023 | 3D 가우시안 | SH | ✅ | 수백만 프리미티브 필요 |
| **2DGS** (Huang et al.) | 2024 | 2D 가우시안 디스크 | SH | ✅ | 더 나은 표면 재구성 |
| **BBSplat** (Svitov et al.) | 2024 | 텍스처드 평면 프리미티브 | RGB 텍스처 + 알파맵 | ⚠️ 13FPS | 학습 가능한 RGB 텍스처 및 알파맵을 가진 최적화 가능한 텍스처 평면 프리미티브 |
| **NeST-Splatting** | 2024 | 가우시안 기반 | 뉴럴 텍스처 | ⚠️ 20FPS | 1,300K 프리미티브 |
| **Nexels** (Rong et al.) | 2025 | Surfel (sparse) | Neural Field + per-prim | ✅ 60FPS | **9.7~31x 프리미티브 절감** |

**BBSplat과의 핵심 차이**: BBSplat은 더 빠른 훈련 시간을 가지지만, 알파 텍스처 사용으로 인해 높은 오버드로우가 발생하여 렌더링 속도가 20 FPS에 그친다. 반면 Nexels는 Top-K 방식으로 불필요한 neural field 쿼리를 줄여 60 FPS를 달성합니다.

결론적으로, Nexels는 고주파 외관을 낮은 메모리와 연산 비용으로 포착하며, **볼류메트릭 표현과 표면 표현 사이의 탐구에서 새로운 지평을 연다.**

---

## 5. 앞으로의 연구에 미치는 영향 및 고려할 점

### 5-1. 연구에 미치는 영향

1. **패러다임 전환의 촉매**: Nexels는 미분 가능한 렌더링 내에서 **희소하고 하이브리드한 표현**으로의 패러다임 전환을 시사하며, 대규모 및 도메인 적응 시나리오로의 확장에 유망한 방향을 제시한다.

2. **메모리 효율적 3D 표현의 기준 확립**: 수십만 개의 프리미티브로도 수백만 개와 동등한 품질이 가능함을 보임으로써, 이후 연구들은 기하학-외관 분리 설계를 더욱 적극적으로 채택할 것으로 예상됩니다.

3. **다운스트림 응용 확대**: AR/VR, 텔레프레즌스, 로보틱스 분야에서 메모리와 렌더링 속도가 제약되는 환경에서의 실용적 활용이 크게 확대될 수 있습니다.

4. **뉴럴 텍스처 설계의 새 기준**: Instant-NGP 기반 해시 그리드를 3D 표현의 외관 모듈로 사용하는 방식은 이후 다양한 sparse primitive 방법들에 표준 컴포넌트로 채택될 가능성이 높습니다.

---

### 5-2. 앞으로 연구 시 고려할 점

| 고려 사항 | 내용 |
|-----------|------|
| **Feed-forward 일반화** | 현재는 per-scene 최적화 방식이므로, 새로운 장면에 대한 빠른 적응(few-shot, generalizable NVS)을 위한 사전 학습 전략 필요 |
| **동적 장면 확장** | Nexels는 현재 정적 장면에 초점. 시간 축 서펠 파라미터화(4D surfel) 혹은 deformation field 결합 필요 |
| **계층적 LoD 구조** | 도시 규모 장면 모델링을 위한 레벨-오브-디테일 구조 통합이 향후 연구 과제로 남아 있음 |
| **K값의 테스트 시 일반화** | K를 2 이상으로 늘리면 품질이 저하되며, 훈련 뷰에서 텍스처된 적 없는 프리미티브가 테스트 뷰에서 텍스처링되는 문제를 해결할 정규화 또는 마스킹 전략 필요 |
| **얇은 구조물 처리** | 낮은 프리미티브 수에서 얇은 구조물을 놓치는 문제에 대한 적응적 프리미티브 배치 전략 필요 |
| **GPU 의존성 완화** | 현재 GPU 텐서 코어에 의존적이므로, 모바일 및 엣지 디바이스로의 이식성을 위한 경량화 연구 필요 |
| **레이 트레이싱 통합** | 희소 표면은 레이 트레이싱 응용에 적합하므로, 글로벌 조명 효과를 통합하는 물리 기반 렌더링으로의 확장 가능성 탐구 필요 |

---

## 참고 자료 및 출처

1. **arXiv 원문**: Rong et al., "Nexels: Neurally-Textured Surfels for Real-Time Novel View Synthesis with Sparse Geometries," arXiv:2512.13796, December 2025. https://arxiv.org/abs/2512.13796
2. **공식 프로젝트 페이지**: https://lessvrong.com/cs/nexels/
3. **공식 GitHub 코드**: https://github.com/victor-rong/nexels
4. **arXiv HTML 전문**: https://arxiv.org/html/2512.13796v1
5. **arXiv PDF**: https://arxiv.org/pdf/2512.13796
6. **EmergentMind 리뷰**: https://www.emergentmind.com/papers/2512.13796
7. **Moonlight 문헌 리뷰**: https://www.themoonlight.io/en/review/nexels-neurally-textured-surfels-for-real-time-novel-view-synthesis-with-sparse-geometries
8. **MrNeRF Twitter/X 요약**: https://x.com/janusch_patas/status/2001267099537379514
9. **비교 논문 - BBSplat**: Svitov et al., "BillBoard Splatting (BBSplat): Learnable Textured Primitives for Novel View Synthesis," arXiv:2411.08508, 2024. https://arxiv.org/abs/2411.08508
10. **비교 논문 - NeST-Splatting**: https://zhangxin-cg.github.io/nest-splatting/

> ⚠️ **정확도 주의사항**: 내부 수식(특히 alpha compositing 공식과 generalized Gaussian kernel의 구체적 지수 형태)은 공개된 HTML/arXiv 페이지에서 완전히 렌더링되지 않은 부분이 있어, 논문의 기술적 맥락과 관련 문헌을 바탕으로 표준적인 형태로 재구성하였습니다. 정확한 수식은 arXiv PDF 원문을 직접 확인하시기 바랍니다.
