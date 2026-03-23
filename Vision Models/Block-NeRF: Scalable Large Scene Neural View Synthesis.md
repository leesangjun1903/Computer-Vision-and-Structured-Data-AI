# Block-NeRF: Scalable Large Scene Neural View Synthesis

---

## 1. 핵심 주장 및 주요 기여 요약

Block-NeRF는 Neural Radiance Fields(NeRF)의 변형으로, 대규모 환경을 표현할 수 있도록 설계되었다. 핵심 주장은 도시 규모(city-scale)의 장면을 렌더링할 때, 장면을 개별적으로 학습된 여러 NeRF로 분해(decompose)하는 것이 필수적이며, 이 분해를 통해 렌더링 시간을 장면 크기에서 분리(decouple)하고, 임의로 큰 환경으로 확장 가능하며, 블록 단위의 환경 업데이트를 허용한다는 것이다.

### 주요 기여:
1. **장면 분해(Scene Decomposition)**: 대규모 표현을 독립적으로 최적화할 수 있는 여러 블록으로 분할한다.
2. **아키텍처 개선**: 각 개별 NeRF에 appearance embeddings, learned pose refinement, controllable exposure를 추가하고, 인접 Block-NeRF 간 외관 정렬(appearance alignment) 절차를 도입하여 매끄럽게 결합할 수 있도록 했다.
3. **최대 규모 신경 장면 표현**: 280만 장의 이미지로 Block-NeRF 그리드를 구축하여, 당시 최대 규모의 신경 장면 표현을 생성하고, 샌프란시스코의 전체 이웃(neighborhood)을 렌더링할 수 있음을 시연했다.

**저자 및 게재**: Matthew Tancik, Vincent Casser, Xinchen Yan, Sabeek Pradhan, Ben Mildenhall, Pratul P. Srinivasan, Jonathan T. Barron, Henrik Kretzschmar — CVPR 2022, pp. 8248-8258.

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

기존 NeRF 기반 연구는 소규모·객체 중심(object-centric) 재구성에 집중해왔으며, 도시 규모 환경으로 확장(scale-up)하면 모델 용량(capacity) 제한으로 인해 아티팩트(artifacts)와 낮은 시각적 충실도(visual fidelity)가 발생한다.

구체적으로 해결해야 할 세 가지 핵심 문제:

1. **확장성(Scalability)**: 단일 NeRF는 메모리와 모델 용량의 한계로 도시 전체를 표현 불가
2. **환경 변화에 대한 강건성(Robustness)**: 수개월에 걸쳐 다양한 환경 조건에서 수집된 데이터에 대해 NeRF를 강건하게 만들어야 한다.
3. **일시적 객체(Transient Objects)**: 제안 방법은 세그멘테이션 알고리즘을 사용한 마스킹으로 학습 중 일시적 객체를 필터링하여 처리한다.

### 2.2 제안 방법 (수식 포함)

#### (A) 기본 볼륨 렌더링 (Volume Rendering)

NeRF의 기본 볼륨 렌더링 수식은 다음과 같다. 카메라 원점 $\mathbf{o}$에서 방향 $\mathbf{d}$로 발사한 광선 $\mathbf{r}(t) = \mathbf{o} + t\mathbf{d}$에 대해:

$$\hat{C}(\mathbf{r}) = \int_{t_n}^{t_f} T(t) \, \sigma(\mathbf{r}(t)) \, \mathbf{c}(\mathbf{r}(t), \mathbf{d}) \, dt$$

여기서 누적 투과율(transmittance):

$$T(t) = \exp\left(-\int_{t_n}^{t} \sigma(\mathbf{r}(s)) \, ds\right)$$

이산화(discretized) 형태:

$$\hat{C}(\mathbf{r}) = \sum_{i=1}^{N} T_i \, \alpha_i \, \mathbf{c}_i, \quad \text{where} \quad T_i = \prod_{j=1}^{i-1}(1 - \alpha_j), \quad \alpha_i = 1 - \exp(-\sigma_i \delta_i)$$

#### (B) Block-NeRF 네트워크 구조

네트워크는 mip-NeRF 구조를 따른다. Block-NeRF는 세 가지 하위 네트워크로 구성된다:

1. **밀도 네트워크 $f_\sigma$**: 8개 레이어, 너비 512 (Mission Bay 실험) 또는 1024 (기타 실험)로 구성된다. 입력 좌표로부터 밀도 $\sigma$와 중간 특징 벡터를 출력한다.

$$(\sigma, \mathbf{h}) = f_\sigma(\gamma(\mathbf{x}))$$

여기서 $\gamma(\mathbf{x})$는 mip-NeRF의 **Integrated Positional Encoding (IPE)**이다. mip-NeRF는 NeRF의 ray tracing 대신 cone tracing을 도입하여, 원뿔 절두체(conical frustum)를 다변량 가우시안으로 근사하고 IPE 특징을 생성한다.

2. **색상 네트워크 $f_c$**: 3개 레이어, 너비 128로 구성된다. 중간 특징, 시선 방향, appearance embedding, exposure 입력을 받아 RGB 색상을 출력한다.

$$\mathbf{c} = f_c(\mathbf{h}, \mathbf{d}, \ell_a, \mathbf{e})$$

여기서:
- $\mathbf{d}$: 시선 방향 (viewing direction)
- $\ell_a$: appearance embedding 벡터 (32차원)
- $\mathbf{e}$: exposure 입력

3. **가시성 네트워크 $f_v$**: 4개 레이어, 너비 128로 구성된다. 특정 위치에서의 가시성(visibility)을 예측한다.

$$v = f_v(\mathbf{h}, \mathbf{d})$$

#### (C) Appearance Embedding

NeRF in the Wild에서 차용한 Appearance Embedding 기법으로, 네트워크에 보조 잠재 벡터(auxiliary latent vector)를 입력하여 외관 변화를 설명한다. 각 이미지에 잠재 벡터가 할당되어 해당 이미지에 고유한 외관 변화를 설명한다.

잠재 벡터는 Generative Latent Optimization 기법으로 네트워크와 함께 최적화되며, 각 잠재 벡터에 대해 연관된 이미지와의 복원 손실(reconstruction loss)을 최소화하는 것이 목표이다.

#### (D) Block 간 결합 (Interpolation)

추론 시, 카메라 위치에서 볼 수 있는 여러 Block-NeRF의 렌더링을 **역거리 가중치(Inverse Distance Weighting)**로 결합한다:

$$\hat{C}_{\text{final}} = \frac{\sum_{k} w_k \cdot \hat{C}_k}{\sum_{k} w_k}$$

여기서 $w_k$는 카메라와 $k$번째 블록 중심 간의 역거리에 기반한 가중치이다. Block-NeRF가 예측한 가시성(visibility)을 보간에 활용하는 실험도 수행했다. 이미지 전체의 평균 가시성을 사용하는 방법과 픽셀별 가시성을 직접 활용하는 방법을 고려했다.

#### (E) 학습 설정

각 Block-NeRF는 Adam 옵티마이저로 300K 반복 학습되며, 배치 크기는 16,384이다. 학습률은 $2 \times 10^{-3}$에서 $2 \times 10^{-5}$로 로그 스케일 어닐링되며, 처음 1,024 반복 동안 워밍업 단계가 있다.

손실 함수:

$$\mathcal{L} = \mathcal{L}_{\text{recon}} + \lambda_v \mathcal{L}_{\text{vis}}$$

여기서 $\mathcal{L}_{\text{recon}}$은 MSE 기반 복원 손실이고, 가시성은 MSE 손실로 감독되며 $\lambda_v = 10^{-6}$으로 스케일링된다.

### 2.3 모델 구조 다이어그램 (개념적)

```
입력 이미지 → 세그멘테이션 마스크 (일시적 객체 제거)
      ↓
   [Block 1 NeRF]  [Block 2 NeRF]  ...  [Block N NeRF]
      │                │                    │
      ├─ f_σ (밀도)     ├─ f_σ              ├─ f_σ
      ├─ f_c (색상)     ├─ f_c              ├─ f_c
      └─ f_v (가시성)   └─ f_v              └─ f_v
              │                │                    │
              └────────── 역거리 가중 보간 ──────────┘
                              ↓
                     최종 렌더링 이미지
```

### 2.4 성능 향상

장면을 여러 Block-NeRF로 분할하면, 전체 가중치 수를 동일하게 유지하더라도 재구성 정확도가 향상된다.

주요 ablation 결과:
- 포즈 최적화(pose optimization) 없이 학습하면 결과 장면이 흐릿해지고 포즈 불일치로 인한 중복 객체가 발생할 수 있다.
- 노출(exposure) 입력은 재구성을 약간 향상시키며, 더 중요하게는 추론 시 노출 변경 능력을 제공한다.
- 가시성 기반 보간(pixelwise/imagewise visibility)을 사용하면 더 선명한(sharper) 결과를 얻을 수 있다.

### 2.5 한계

1. **일시적 객체 처리의 불완전성**: 객체가 제대로 마스킹되지 않으면 렌더링에 아티팩트가 발생할 수 있다. 예를 들어, 자동차 자체는 올바르게 제거되더라도 차의 그림자가 남는 경우가 있다.
2. **식생(vegetation) 문제**: 계절에 따라 변하고 바람에 움직이는 식물은 흐릿한 표현을 초래한다.
3. **시간적 비일관성**: 건설 공사 등 학습 데이터의 시간적 비일관성은 자동으로 처리되지 않으며, 영향받은 블록의 수동 재학습이 필요하다.
4. **시간적 일관성 문제**: 가시성 기반 보간은 더 선명한 재구성을 생성하지만 시간적 비일관성이 발생할 수 있어, 정지 이미지 렌더링에만 적합하다.
5. **계산 비용**: 이 방법은 NeRF 모델의 높은 계산 비용을 그대로 상속하며, 전례 없는 규모로 적용한다.

---

## 3. 모델의 일반화 성능 향상 가능성

Block-NeRF가 일반화 성능을 향상시키는 핵심 메커니즘과 한계를 다음과 같이 분석한다:

### 3.1 일반화를 가능하게 하는 설계 요소

| 설계 요소 | 일반화 기여 |
|-----------|------------|
| **Appearance Embedding** | 조명, 날씨, 시간대 등 다양한 환경 조건에 대한 적응력 제공 |
| **Learned Pose Refinement** | 부정확한 카메라 포즈를 보정하여 다양한 데이터 소스 수용 |
| **Controllable Exposure** | 다양한 노출 조건의 이미지에 대한 강건성 확보 |
| **Transient Object Masking** | 동적 객체를 제거하여 정적 환경에 대한 일관된 표현 학습 |
| **블록 분해** | 각 블록이 국소 영역에 집중하여 지역적 특성에 대한 높은 적합도 달성 |

### 3.2 일반화 한계 및 향후 개선 방향

현재 Block-NeRF는 **장면별(per-scene) 최적화** 패러다임에 기반한다. 즉:

- 새로운 장면에 대해 처음부터 학습이 필요하다
- **Cross-scene 일반화**는 지원하지 않는다
- NeRF는 각 고유 장면에 대해 재학습이 필요하다.

일반화 향상을 위한 가능한 방향:

1. **사전학습된 특징(Pre-trained Features)**: Foundation model의 특징을 활용한 일반화 가능한 NeRF
2. **메타러닝(Meta-Learning)**: Few-shot 설정에서 빠른 적응을 위한 학습
3. **하이브리드 접근법**: 명시적(explicit) 표현과 암묵적(implicit) 표현의 결합
4. **시간적 모델링(Temporal Modeling)**: 동적 장면과 계절 변화를 명시적으로 모델링

### 3.3 Appearance Embedding의 일반화 수식

Appearance embedding 공간에서 새로운 환경 조건으로의 외삽(extrapolation)은 다음과 같이 표현할 수 있다:

$$\hat{C}_{\text{novel}} = f_c(\mathbf{h}, \mathbf{d}, \ell_a^{\text{interp}}, \mathbf{e}^{\text{target}})$$

여기서 $\ell_a^{\text{interp}}$는 학습된 appearance embedding 공간에서의 보간 벡터이다. 이를 통해 학습 시 관찰된 다양한 조건 사이에서 부드러운 전환이 가능하지만, 학습 분포 밖의 극단적 조건으로의 일반화는 여전히 도전적이다.

---

## 4. 향후 연구에 미치는 영향 및 고려 사항

### 4.1 연구 영향

1. **대규모 신경 렌더링의 패러다임 확립**: Block-NeRF는 "분할 후 정복(divide-and-conquer)" 전략이 대규모 장면 재구성의 표준 접근법이 될 수 있음을 입증했다.
2. **자율주행 시뮬레이션**: Block-NeRF는 NeRF를 확장하여 대규모의 사실적인 도시 지도를 재구성하며, 다양한 시나리오의 사실적인 주행 뷰를 시뮬레이션하여 고비용 데이터 수집을 절감할 수 있다.
3. **모듈식 업데이트**: 환경 변화 시 전체 모델이 아닌 영향받은 블록만 재학습하면 되므로, 실용적 배포에 유리하다.

### 4.2 향후 연구 시 고려할 점

1. **실시간 렌더링**: NeRF 기반 방법의 높은 계산 비용 문제 해결
2. **동적 장면 처리**: 시간에 따라 변화하는 객체의 명시적 모델링
3. **자동화된 블록 분할**: 최적의 블록 크기와 배치를 자동으로 결정하는 방법
4. **다중 모달 통합**: LiDAR, GPS 등 다양한 센서 데이터의 통합
5. **품질 메트릭**: 대규모 장면에 적합한 새로운 평가 지표 개발

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

| 방법 | 연도 | 핵심 특징 | Block-NeRF 대비 차이점 |
|------|------|-----------|----------------------|
| **NeRF** (Mildenhall et al.) | 2020 | 기본 볼륨 렌더링 + MLP | 소규모 장면만 가능 |
| **NeRF in the Wild** (Martin-Brualla et al.) | 2021 | Appearance/Transient embedding | Block-NeRF가 이를 채택하여 확장 |
| **mip-NeRF** (Barron et al.) | 2021 | 안티앨리어싱을 위한 cone tracing + IPE | Block-NeRF의 기반 아키텍처 |
| **Mega-NeRF** (Turki et al.) | 2022 | 드론 데이터 기반 대규모 장면 | 항공 뷰 중심, Block-NeRF는 지상 주행 뷰 |
| **mip-NeRF 360** (Barron et al.) | 2022 | 비바운드 장면을 위한 비선형 파라미터화 | 단일 장면 최적화, 도시 규모 미지원 |
| **Instant-NGP** (Müller et al., NVIDIA) | 2022 | 해시 기반 인코딩으로 실시간 학습 | 학습 속도 극대화, but 대규모 장면 미고려 |
| **3D Gaussian Splatting** (Kerbl et al.) | 2023 | 가우시안 프리미티브 기반 실시간 렌더링 | 근본적으로 다른 표현 방식 |
| **SCALAR-NeRF** | 2023+ | 확장 가능한 대규모 신경 장면 재구성 | Block-NeRF의 직접적 후속 연구 |
| **GF-NeRF** | 2023+ | 글로벌 가이드 초점 신경 복사장 | 대규모 장면의 고충실도 렌더링 |

### 3D Gaussian Splatting (3DGS) vs Block-NeRF

두 접근법 모두 고충실도 재구성을 생성하지만, 3DGS는 계산 효율성과 노이즈 감소에서 NeRF를 일관되게 능가한다. 3DGS는 실용적 응용에서 NeRF를 대체하고 있으며, 실시간 렌더링(100+ FPS vs 5 FPS), 분 단위 학습(시간 단위 대신), 동등하거나 더 나은 시각적 품질을 제공한다.

그러나 Block-NeRF의 핵심 기여인 **모듈식 분해 전략**과 **다중 시간/조건 데이터 통합 방법**은 3DGS 기반 대규모 시스템에도 적용 가능한 보편적 원리이다. 실제로 최근 연구에서는 5천만 개의 가우시안 타원체(Gaussian ellipsoids)로 대규모 도시 환경을 처리하는 가우시안 스플래팅 기반 SLAM 시스템이 개발되고 있으며, 전역 일관성을 보장하기 위한 Loop Closure 모듈이 설계되고 있다.

### 핵심 연구 동향 정리

```
[2020] NeRF → [2021] mip-NeRF / NeRF-W → [2022] Block-NeRF / Instant-NGP
                                                      ↓
                                          [2023] 3D Gaussian Splatting
                                                      ↓
                                    [2024-2025] 대규모 GS / 하이브리드 방법
```

---

## 참고자료

1. **Tancik, M. et al.** "Block-NeRF: Scalable Large Scene Neural View Synthesis." *CVPR 2022*, pp. 8248-8258. (arXiv: 2202.05263)
2. **Barron, J.T. et al.** "Mip-NeRF: A Multiscale Representation for Anti-Aliasing Neural Radiance Fields." *ICCV 2021*. (arXiv: 2103.13415)
3. **Mildenhall, B. et al.** "NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis." *ECCV 2020 / Communications of the ACM 2022*.
4. **Martin-Brualla, R. et al.** "NeRF in the Wild: Neural Radiance Fields for Unconstrained Photo Collections." *CVPR 2021*.
5. **Kerbl, B. et al.** "3D Gaussian Splatting for Real-Time Radiance Field Rendering." *ACM Trans. Graph. 2023*, 42(4).
6. **Waymo Research — Block-NeRF Project Page**: https://waymo.com/research/block-nerf/
7. **Block-NeRF Supplement** (CVPR 2022 Open Access): https://openaccess.thecvf.com/content/CVPR2022/supplemental/
8. **ar5iv HTML version**: https://ar5iv.labs.arxiv.org/html/2202.05263
9. **BeyondPixels: A Comprehensive Review of the Evolution of Neural Radiance Fields** (arXiv: 2306.03000v3)
10. **Neural Radiance Field — Wikipedia**: https://en.wikipedia.org/wiki/Neural_radiance_field
11. **PyImageSearch** — "3D Gaussian Splatting vs NeRF: The End Game of 3D Reconstruction?" (2024)
12. **Synced Review** — "UC Berkeley, Waymo & Google's Block-NeRF" (2022): https://medium.com/syncedreview/
13. **Paper Summary Blog by Alex Lau**: https://riven314.github.io/ (2022)
14. **GitHub — UnboundedNeRFPytorch**: https://github.com/sjtuytc/UnboundedNeRFPytorch

# Block-NeRF: Scalable Large Scene Neural View Synthesis

### 1. 핵심 주장 및 주요 기여

**Block-NeRF**는 Neural Radiance Fields(NeRF)를 **도시 규모의 대규모 환경**에 확장하기 위한 혁신적인 방법론입니다. 논문의 핵심 주장은 다음과 같습니다:[1]

1. **블록 분해 원칙**: 도시 규모 장면을 렌더링할 때 장면을 **개별적으로 학습된 여러 NeRF로 분해**하는 것이 필수적입니다. 이를 통해:[1]
   - 렌더링 시간을 장면 크기에서 분리
   - 임의로 큰 환경으로의 확장 가능성 제공
   - 환경의 블록 단위 업데이트 가능

2. **다중 환경 조건 대응**: 개발된 Block-NeRF는 **외관 임베딩(appearance embeddings), 학습된 포즈 개선(learned pose refinement), 제어 가능한 노출(controllable exposure)** 을 포함하여 여러 달에 걸쳐 수집된 다양한 환경 조건의 데이터를 처리합니다.[1]

3. **무결한 합성**: 인접한 NeRF 간의 **외관 정렬 절차**를 도입하여 여러 Block-NeRF의 렌더링을 원활하게 합성할 수 있습니다.[1]

**주요 기여**로는 샌프란시스코의 알라모 광장 지역을 **280만 개 이미지**로부터 35개의 Block-NeRF로 구성된 그리드로 재구성하여, 당시 가장 큰 신경 장면 표현을 달성한 것입니다.[1]

***

### 2. 해결하려는 문제와 제안하는 방법

#### 2.1 핵심 문제점

전통적인 NeRF를 단순히 확장하는 방식의 한계:

- **제한된 모델 용량**: 단일 MLP로는 도시 규모 환경의 복잡성을 표현할 수 없음
- **메모리 제약**: 거대한 네트워크가 단일 장치의 메모리에 맞지 않음
- **렌더링 시간**: 네트워크 크기에 따라 렌더링 시간 증가
- **업데이트 불가능성**: 환경 수정 시 전체 네트워크 재학습 필요
- **다양한 환경 조건**: 여러 달에 걸친 데이터 수집으로 인한 날씨, 조명, 기하학적 변화(건설, 주차차량)

#### 2.2 제안 방법

##### **기본 NeRF 렌더링 수식**

Block-NeRF는 mip-NeRF를 기반으로 구축되며, 기본 렌더링 방정식은 다음과 같습니다:[1]

$$ c_{out} = \sum_{i=1}^{N} w_i c_i, \quad \text{where} \quad w_i = T_i(1 - e^{-\Delta_i \sigma_i}) $$

```math
T_i = \exp\left(-\sum_{j < i} \Delta_j \sigma_j\right), \quad \Delta_i = t_i - t_{i-1}
```

여기서:
- $c_{out}$: 최종 출력 색상
- $w_i$: 가중치 (volume rendering weight)
- $T_i$: 투과율 (transmittance)
- $\sigma_i$: 부피 밀도 (volume density)
- $c_i$: 샘플 위치에서의 색상

##### **통합 위치 인코딩 (Integrated Positional Encoding)**

Mip-NeRF의 개선 사항인 통합 위치 인코딩:

$$ \gamma_{IPE}(\mu, \Sigma) = \mathbb{E}_{X \sim \mathcal{N}(\mu,\Sigma)}[\gamma_{PE}(X)] $$

여기서 $\gamma_{PE}(z) = [\sin(2^0 z), \cos(2^0 z), \ldots, \sin(2^{L-1} z), \cos(2^{L-1} z)]$ 는 정현파 위치 인코딩입니다.[1]

##### **주요 아키텍처 개선 사항**

**(1) 외관 임베딩 (Appearance Embeddings)**

NeRF-W의 Generative Latent Optimization을 도입하여 이미지별 외관 임베딩 벡터를 최적화합니다. 이는 다양한 조명 및 기상 조건을 처리합니다.[1]

**(2) 학습된 포즈 개선 (Learned Pose Refinement)**

카메라 포즈 오류를 보정하기 위해 정규화된 포즈 오프셋을 학습합니다. 각 주행 세그먼트별로 다음의 변환을 학습합니다:[1]

- **위치 오프셋**: 3D 위치 조정
- **회전 오프셋**: $3 \times 3$ 잔여 회전 행렬 (항등 행렬에 추가 및 정규화)

훈련 초기에 강하게 정규화하여 먼저 대략적인 구조를 학습한 후 포즈 오프셋을 수정하도록 합니다.[1]

**(3) 노출 입력 (Exposure Input)**

카메라 노출 정보를 모델에 입력하여 시각적 차이를 보정합니다:[1]

$$ \text{Exposure encoding} = \gamma_{PE}\left(\frac{\text{shutter speed} \times \text{analog gain}}{t}\right) $$

여기서 $t$는 스케일링 계수(논문에서는 1,000 사용)입니다.[1]

#### 2.3 모델 구조

Block-NeRF의 아키텍처(Figure 3):[1]

1. **밀도 네트워크 ($f_\sigma$)**: 
   - 입력: 3D 위치 $x$
   - 출력: 부피 밀도 $\sigma$ 및 특성 벡터

2. **색상 네트워크 ($f_c$)**:
   - 입력: $f_\sigma$의 특성 벡터 + 2D 시점 방향 $d$ + 노출 레벨 + 외관 임베딩
   - 출력: RGB 색상 $c$

3. **가시성 네트워크 ($f_v$)**:
   - 입력: 3D 위치 및 시점 방향
   - 출력: 투과율(transmittance) 예측 - 훈련 뷰에서의 가시성 근사
   - 용도: 추론 중 Block-NeRF 선택 및 필터링

**블록 배치 및 크기**:[1]

- 각 교차로에 하나의 Block-NeRF 배치
- 교차로 자신과 연결된 거리의 75%까지 커버
- 인접 블록 간 50% 겹침으로 외관 정렬 용이

#### 2.4 Block-NeRF 병합 절차

**(1) Block-NeRF 선택**

효율성을 위해 두 가지 필터링 메커니즘 사용:[1]
- 대상 시점으로부터 설정된 반경 내의 Block-NeRF만 고려
- 가시성 평균이 임계값 이하이면 해당 Block-NeRF 제외

**(2) 합성 (Compositing)**

역거리 가중치(Inverse Distance Weighting, IDW)를 사용한 2D 이미지 공간 보간:[1]

$$ w_i \propto \text{distance}(c, x_i)^{-p} $$

여기서:
- $c$: 카메라 원점
- $x_i$: Block-NeRF 중심
- $p$: 블렌딩 속도를 조절하는 파라미터 (알라모 광장: $p=4$, 미션 베이: $p=1$)

**(3) 외관 정렬 (Appearance Matching)**

인접한 Block-NeRF 쌍 사이의 외관 일관성을 보장하기 위해:[1]

1. 두 Block-NeRF 모두에서 높은 가시성을 가진 3D 매칭 위치 선택
2. 네트워크 가중치 고정 후 **대상 Block-NeRF의 외관 코드만 최적화**
3. 매칭 위치의 렌더 영역 간 $\ell^2$ 손실 최소화:

$$ \mathcal{L}_{matching} = \|R_1(\text{appearance}_1) - R_2(\text{appearance}_2)\|_2^2 $$

4. 한 근본 Block-NeRF로부터 시작하여 인접 블록들을 반복적으로 최적화[1]

***

### 3. 성능 향상 및 정량적 결과

#### 3.1 모델 구성 요소 비교 (Ablation Study)

**Table 1: Alamo Square 데이터셋의 단일 교차로에서의 성능 평가**[1]

| 모델 | PSNR↑ | SSIM↑ | LPIPS↓ |
|------|-------|-------|--------|
| **mip-NeRF** (기본선) | 17.86 | 0.563 | 0.509 |
| **-Appearance** | 20.13 | 0.611 | 0.458 |
| **-Exposure** | 23.55 | 0.649 | 0.418 |
| **-Pose Opt.** | 23.05 | 0.625 | 0.442 |
| **Full Block-NeRF** | **23.60** | **0.649** | **0.417** |

**주요 발견**:[1]
- 외관 임베딩 없이: 아티팩트 및 흐릿한 재구성 (PSNR 20.13)
- 포즈 최적화 없이: 블러 및 중복된 객체 (PSNR 23.05)
- 노출 입력: 경미한 개선이지만 추론 시 제어 가능성 제공

#### 3.2 블록 크기 및 배치 비교

**Table 2: Mission Bay 데이터셋에서의 블록 수에 따른 성능**[1]

| 블록 수 | 가중치 | 크기 | 연산 | PSNR↑ | SSIM↑ | LPIPS↓ |
|--------|-------|------|------|-------|-------|--------|
| 1 | 0.25M | 544 m | 1× | 23.83 | 0.825 | 0.381 |
| 4 | 0.25M | 271 m | 2× | 25.55 | 0.868 | 0.318 |
| 8 | 0.25M | 116 m | 2× | 26.59 | 0.890 | **0.278** |
| 16 | 0.25M | 54 m | 2× | **27.40** | **0.907** | 0.242 |

**고정 총 가중치 시나리오** (하단):
- 8개 블록 (0.13M/block): PSNR 25.92, SSIM 0.875
- 16개 블록 (0.07M/block): PSNR 25.98, SSIM 0.877

**결론**: 장면을 작은 블록으로 분할할수록 성능 향상. 총 가중치를 고정했을 때도 개선되어 **계산 효율성과 품질이 모두 개선**됨을 의미합니다.[1]

#### 3.3 합성 방법 비교

**Table 3: 다양한 보간 방법 비교**[1]

| 방법 | 시간 일관성 | PSNR↑ | SSIM↑ | LPIPS↓ |
|------|-----------|-------|-------|--------|
| **Nearest** | ✓ | 26.40 | 0.887 | 0.280 |
| **IDW 2D** | ✓ | 26.59 | 0.890 | 0.278 |
| **IDW 3D** | ✓ | 26.57 | 0.890 | 0.278 |
| **Pixelwise Visibility** | ✗ | 27.39 | 0.906 | **0.242** |
| **Imagewise Visibility** | ✗ | **27.41** | **0.907** | 0.242 |

**선택 기준**:[1]
- 플라이스루 비디오: **2D IDW** (시간 일관성)
- 정적 이미지: **이미지별 가시성** (최고 품질)

#### 3.4 대규모 재구성 성과

**Alamo Square 데이터셋**:[1]
- **지역 크기**: 약 960 m × 570 m
- **Block-NeRF 수**: 35개
- **데이터 규모**: 
  - 총 주행 시간: 13.4시간
  - 수집 횟수: 1,330회
  - **총 이미지: 2,818,745장**
- **각 블록당**: 64,575~108,216 이미지

**Mission Bay 데이터셋**:[1]
- 단일 캡처 (일관된 조건)
- 거리: 1.08 km
- 이미지: 12,000장 (12 카메라)

***

### 4. 모델의 일반화 성능 향상 가능성

#### 4.1 현재 구조의 일반화 능력

Block-NeRF의 일반화 성능을 향상시키는 메커니즘:

**1. 외관 임베딩의 역할**:[1]
- 이미지별 외관 코드 최적화를 통해 다양한 조명, 기상, 시간대 조건에 적응
- 32차원의 임베딩 벡터가 글로벌 및 저주파 속성(시간대, 색상 균형, 기상)을 캡처
- **제한 사항**: 픽셀 수준의 세부 일관성이 부족할 수 있음

**2. 포즈 개선의 일반화 효과**:[1]
- 주행 세그먼트별 학습된 포즈 오프셋이 카메라 캘리브레이션 오류 보정
- 초기 강 정규화 후 점진적 감소로 안정적 수렴
- 기하학적 일관성 개선

**3. 가시성 네트워크의 선택 메커니즘**:[1]
- 불필요한 Block-NeRF 제외로 인한 노이즈 감소
- 각 시점에 대해 관련 있는 블록만 렌더링하여 성능 극대화

#### 4.2 일반화 성능의 한계 및 개선 방향

##### **4.2.1 기존 한계**[1]

**(1) 동적 객체 처리**
- **현재**: 의미론적 분할을 이용한 마스킹으로 자동차와 보행자 제거
- **문제**: 
  - 자동차 그림자는 여전히 남음
  - 부정확한 마스킹 시 아티팩트 발생
  - 동적 객체 제어 불가능

**(2) 식생(Vegetation) 처리**
- **현재 문제**: 나무와 식물의 계절별 변화 및 바람에 의한 움직임
- **결과**: 흐릿한 나뭇잎 표현

**(3) 기하학적 변화**
- **현재 문제**: 건설 작업, 새 건물 등의 환경 변화 자동 처리 불가
- **필요 조치**: 영향받는 블록의 수동 재학습

**(4) 원거리 객체의 흐릿한 표현**[1]
- **원인**: 샘플링된 볼륨 표현의 한계
- **해결책**: NeRF++나 Mip-NeRF 360의 기법 적용 가능[2]

##### **4.2.2 최신 연구 기반 개선 방향 (2023-2025)**

**1. 동적 객체 통합**[3][2]
- D-NeRF 방식으로 시간을 추가 입력으로 추가
- 정적 환경 + 제어 가능한 객체 NeRF 조합
- 예: 환경의 Block-NeRF와 독립적인 차량/보행자 NeRF

**2. 의미론적 분할 기반 향상**[4]
- 최신 패노프틱 세그먼테이션 모델로 더 정확한 마스킹
- 동적 객체 학습 (NeRF-W 방식 확장)
- 그림자 분리 처리

**3. 다양한 환경 조건 적응**[5]
- 인프라 카메라(Thermal-NeRF) 등 다중 센서 입력 지원[6]
- 수중 또는 안개 환경 처리 (SeaThru-NeRF)[7]
- 도시 거리 장면 특화 (S-NeRF)[2]

**4. 계산 효율성 개선**[8]
- **NeuRas (2024)**: 신경 래스터화를 통한 실시간 렌더링
  - 대규모 장면에서 >100 FPS 달성
  - Block-NeRF의 느린 렌더링(픽셀당 5.9초)을 30배 이상 개선
- 캐싱 기법 (FastNeRF, PlenOctrees 등)
- 희소 복셀 그리드 표현

**5. 제한된 샘플에서의 일반화**[9]
- MutualNeRF (ICLR 2025): 정보 이론 기반 희소 뷰 합성
- 상호 정보량 최소화를 통한 최적 뷰포인트 선택
- 제한된 이미지로 더 나은 장면 재구성

**6. 대규모 장면 인식**[10][11]
- Global-guided Focal NeRF: 글로벌-로컬 하이브리드 표현
- NeRFusion: TSDF 융합 기반 효율적 대규모 재구성
- Grid-NeRF: 다중 해상도 특성 평면

#### 4.3 교차 블록 일반화

**외관 정렬의 한계**:[1]
- 현재: 100 반복 내 수렴으로 글로벌 저주파 속성만 맞춤
- 미래: 고급 색상 보정 알고리즘, 심화 학습 기반 정렬

**개선 제안**:
1. 블록 경계에서의 **완벽한 시각적 일관성** 달성
2. **가변 환경 조건 보간**: 시간대별 외관 자동 전환
3. **의미론적 일관성**: 객체 범주별 외관 일관성 유지

***

### 5. 한계 및 제약 사항

Block-NeRF의 주요 한계:[1]

| 한계 | 원인 | 영향 |
|------|------|------|
| **동적 객체 처리** | 마스킹 기반 필터링 | 제어 불가능, 그림자 남음 |
| **식생 표현** | 계절/바람에 의한 변화 | 나뭇잎 흐릿함 |
| **기하 변화** | 자동 처리 불가 | 건설 중인 지역 수동 재학습 필요 |
| **원거리 시각화** | 샘플링 밀도 부족 | 하늘/먼 건물 흐릿함 |
| **렌더링 속도** | MLP 기반 부피 렌더링 | 픽셀당 5.9초 (느림) |
| **임베딩 크기** | 32차원 고정 | 복잡한 조건 표현 제한 |

**근본적 과제**:
- NeRF vs. Gaussian Splatting: 2023년 이후 가우시안 스플래팅이 더 빠르고 고품질(2배~3배 빠른 학습, 수배 빠른 렌더링)[12]
- **Block-NeRF의 강점**: 암묵적 표현, 낮은 저장 요구량, 시간/의미 기반 조작 가능

***

### 6. 논문의 연구 영향 및 향후 고려 사항

#### 6.1 학술 영향 (Citation 분석)

**논문 인용 현황** (2024년 기준):[13]
- CVPR 2022 게재 (1,000+ 인용)
- 대규모 장면 NeRF 연구의 벤치마크

**파생 연구**:
1. **대규모 장면 분할 방식**
   - NeRFusion (2022): TSDF 융합으로 더 빠른 학습
   - Switch-NeRF: 동적 라우팅 게이팅
   - Grid-NeRF: 다중해상도 특성 평면

2. **도시 장면 특화**
   - S-NeRF: 거리 뷰 합성 (7~40% MSE 감소)[2]
   - CityNeRF: 위성 이미지 기반
   - Mega-NeRF: 드론 데이터 분할

3. **실시간 렌더링 진화**[8]
   - NeuRas (2024): 신경 래스터화로 >100 FPS 달성
   - 암묵적 NeRF 표현을 명시적 래스터화와 결합

#### 6.2 향후 연구 시 고려 사항

**1. 문제 정의 단계**
- [ ] **동적 장면의 필요성**: 제어 가능한 객체 필요 시 D-NeRF 계열 도입
- [ ] **실시간성 요구**: 대규모 장면이면서 실시간 필요 시 NeuRas 등 래스터화 방식 고려
- [ ] **환경 다양성**: 악천후/야간/실내라면 멀티모달 센서 (Thermal-NeRF 등) 검토

**2. 데이터 수집**
- [ ] **카메라 캘리브레이션**: 포즈 개선의 한계를 고려한 정확한 초기 포즈 제공
- [ ] **시간대 분포**: 외관 임베딩이 효과적이려면 시간 계절에 걸친 다양한 샘플 필요
- [ ] **동적 객체 마스킹**: 의미론적 분할의 정확도가 성능에 직결 (패노프틱 세그먼테이션 권장)

**3. 아키텍처 설계**
- [ ] **블록 크기 결정**: Table 2 분석 - 54~116m 범위에서 최적 성능
- [ ] **네트워크 용량**: 총 계산 예산을 고려한 블록당 가중치 결정
- [ ] **가시성 네트워크**: 선택 정확도 vs. 보수성 트레이드오프 설정

**4. 후처리 및 최적화**
- [ ] **외관 정렬 반복 수**: 100회 반복은 저주파 속성 정렬이나, 세부 일관성 검토
- [ ] **경계 처리**: IDW 가중치 파라미터 $p$ 튜닝 (알라모: 4, 미션베이: 1)
- [ ] **시간 일관성**: 동영상 생성 시 2D IDW 사용으로 시간적 깜빡임 방지

**5. 평가 전략**
- [ ] **정량 지표**: PSNR (→ LPIPS 권장), SSIM, LPIPS 모두 보고
- [ ] **정성 평가**: 블록 경계에서의 시각적 일관성 검사 (인간 평가자)
- [ ] **동적 평가**: 플라이스루 비디오의 시간적 일관성 평가

**6. 최신 대안 기술 검토 (2024-2025)**[12]
- **Gaussian Splatting**: 더 빠른 학습/렌더링 (Block-GS 등장)
- **하이브리드 표현**: 암묵적 NeRF + 명시적 메시 (NeuRas)
- **다중 해상도**: 글로벌 거친 표현 + 로컬 세밀 표현

***

### 결론

**Block-NeRF**는 신경 렌더링 분야에서 **대규모 환경 재구성**의 획기적 전환점을 제시한 논문입니다. 도시 규모 환경을 2.8백만 이미지로 재구성한 성과는 당시 기술의 한계를 명확히 보여주었고, 이후 대규모 장면 표현 연구의 기초가 되었습니다. 

**핵심 기여**인 블록 분해 방식, 외관 임베딩, 포즈 개선, 가시성 필터링은 현재도 많은 파생 연구에서 채택되고 있습니다. 그러나 2023년 Gaussian Splatting의 등장으로 인한 패러다임 전환과 실시간 성능의 중요성 증대로, 향후 연구는 **암묵적 표현의 해석 가능성을 유지하면서도 명시적 방식의 계산 효율을 결합하는 하이브리드 접근**으로 발전하고 있습니다.[8][12]

**실무 적용 시**: 동적 환경 요구 사항, 실시간성 필요성, 계산 자원 제약을 먼저 검토하여, Block-NeRF, Gaussian Splatting 기반 방식, 또는 NeuRas 같은 신경 래스터화 기법 중 최적의 선택을 이루어야 합니다.

***

**참고문헌**

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/37155355-1fbe-4830-9084-0d8093294f16/2202.05263v1.pdf)
[2](https://arxiv.org/abs/2303.00749)
[3](https://ieeexplore.ieee.org/document/9578753/)
[4](https://www.mdpi.com/2072-4292/16/2/301)
[5](https://arxiv.org/abs/2407.10267)
[6](https://ieeexplore.ieee.org/document/10802480/)
[7](https://ieeexplore.ieee.org/document/10203316/)
[8](https://waabi.ai/research/neuras)
[9](https://openreview.net/forum?id=5RPpwW82vs)
[10](https://arxiv.org/html/2403.12839v2)
[11](https://openaccess.thecvf.com/content/CVPR2022/html/Zhang_NeRFusion_Fusing_Radiance_Fields_for_Large-Scale_Scene_Reconstruction_CVPR_2022_paper.html)
[12](https://arxiv.org/html/2210.00379v6)
[13](https://openaccess.thecvf.com/content/CVPR2022/papers/Tancik_Block-NeRF_Scalable_Large_Scene_Neural_View_Synthesis_CVPR_2022_paper.pdf)
[14](https://www.semanticscholar.org/paper/6caf3307096a15832ace34a0d54cd28413503f8b)
[15](https://ieeexplore.ieee.org/document/9879876/)
[16](https://ieeexplore.ieee.org/document/10423594/)
[17](https://arxiv.org/abs/2409.12014)
[18](http://arxiv.org/pdf/2202.05263.pdf)
[19](http://arxiv.org/pdf/2404.06152.pdf)
[20](http://arxiv.org/pdf/2112.01523.pdf)
[21](http://arxiv.org/pdf/2404.00714.pdf)
[22](https://arxiv.org/pdf/2402.11141.pdf)
[23](https://isprs-archives.copernicus.org/articles/XLVIII-2-W3-2023/115/2023/isprs-archives-XLVIII-2-W3-2023-115-2023.pdf)
[24](https://isprs-archives.copernicus.org/articles/XLVIII-M-2-2023/1113/2023/isprs-archives-XLVIII-M-2-2023-1113-2023.pdf)
[25](https://waymo.com/research/block-nerf/)
[26](https://hsejun07.tistory.com/120)
[27](https://www.sciencedirect.com/science/article/pii/S2352340925002161)
[28](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/06388.pdf)
[29](https://wandb.ai/geekyrakshit/block-nerf/reports/Block-NeRF-Scalable-Large-Scene-Neural-View-Synthesis--VmlldzoxNjIyMzI4)
