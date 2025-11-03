# Attentive Group Equivariant Convolutional Networks

### 1. 핵심 주장과 주요 기여

이 논문은 **그룹 등변 신경망(Group Equivariant Convolutional Networks, G-CNN)**의 근본적인 한계를 해결하기 위해 주의 메커니즘(attention mechanism)을 통합하는 방법을 제시합니다.[1]

**핵심 문제**: 기존의 그룹 등변 합성곱 신경망은 대칭성 패턴(symmetry patterns)을 학습하는 데는 우수하지만, 이들 대칭성 간의 **의미 있는 관계**—예를 들어, 상대적 위치(relative positions), 방향(orientations), 스케일(scales)—을 학습할 명시적인 수단이 부족합니다.[1]

**주요 기여:**

1. **등변 시각 주의의 일반 이론 프레임워크**: 기존의 모든 시각 주의 방법들(SE-Net, CBAM 등)이 이 프레임워크의 특수한 경우임을 증명[1]

2. **등변 그룹 합성곱 네트워크(Attentive Group Convolutional Networks, α-networks)**: 제안된 이론적 프레임워크의 구체적인 구현

3. **일관된 성능 향상**: 벤치마크 데이터셋(rot-MNIST, CIFAR-10, PatchCamelyon)에서 기존 그룹 등변 네트워크를 능가함[1]

4. **해석가능성 강화**: 등변 주의 맵 시각화를 통해 학습된 개념의 투명성 제공[1]

---

### 2. 해결하는 문제와 제안 방법

#### 2.1 배경 문제

Figure 1의 예시처럼, 얼굴을 구성하는 요소들의 상대적 위치와 방향이 중요합니다. 같은 요소들이 의미 없는 배치로 조합되면 얼굴이 아닙니다. G-CNN은 각 대칭성을 개별적으로 학습하지만, 대칭성들 간의 조화(harmonic relationships)를 명시적으로 학습하지 못합니다.[1]

#### 2.2 수식 기반 방법론

**기본 그룹 합성곱(Group Convolution)**:[1]

```math
[f ⋆_G ψ](g) = \sum_{\tilde{c}=1}^{\tilde{N}_c} \int_G f_{\tilde{c}}(\tilde{g})L_g[ψ_{\tilde{c}}](\tilde{g}) d\tilde{g}
```

여기서:
- $$f, ψ: G → ℝ^{\tilde{N}_c}$$: 신호와 커널
- $$L_g$$: 그룹 원소 $$g$$에 의한 변환
- 이 연산은 그룹 등변성을 만족: $$L_{\bar{g}}[f ⋆_G ψ] = (L_{\bar{g}}[f] ⋆_G ψ)$$[1]

**제안된 등변 그룹 주의 합성곱**:[1]

```math
[f ⋆^α_G ψ](g) = \sum_{\tilde{c}=1}^{\tilde{N}_c} \int_G α_{\tilde{c}}(g, \tilde{g}) f_{\tilde{c}}(\tilde{g}) L_g[ψ_{\tilde{c}}](\tilde{g}) d\tilde{g}
```

여기서 $$α: G × G → ^{\tilde{N}_c}$$는 주의 맵입니다.[1]

**핵심 정리 (Theorem 1)**: 등변 그룹 합성곱이 등변 연산자가 되기 위한 필요충분조건:[1]

$$
∀g, \bar{g}, \tilde{g} ∈ G: \quad A[L_{\bar{g}}f](g, \tilde{g}) = A[f]({\bar{g}}^{-1}g, {\bar{g}}^{-1}\tilde{g}) 
$$

이 조건을 만족하면, 주의 연산자 자체가 그룹 등변이어야 합니다.

#### 2.3 효율적인 주의 맵 분해

계산 부담을 감소시키기 위해, 아핀 그룹 $$G = ℝ^d ⋊ H$$에서 주의를 공간 및 채널 성분으로 분해합니다:[1]

$$
α_{\tilde{c}}(g, \tilde{g}) := α_X((x, h), (\tilde{x}, \tilde{h})) · α^{\tilde{c}}_C(h, \tilde{h})
$$

**채널 주의(Channel Attention)**:[1]

$$
α^C(h, \tilde{h}) = σ \left( W_2(h^{-1}\tilde{h}) · [W_1(h^{-1}\tilde{h}) · s^{avg}_C(h, \tilde{h})]_+ + W_2(h^{-1}\tilde{h}) · [W_1(h^{-1}\tilde{h}) · s^{max}_C(h, \tilde{h})]_+ \right)
$$

여기서 $$W_i: H → ℝ^{N_{out} × N_{in}}$$는 그룹 정규 표현으로 변환되는 행렬값 커널입니다.

**공간 주의(Spatial Attention)**:[1]

$$
α_X(x, h, \tilde{h}) = σ \left( [s_X ⋆_{ℝ^d} L_h[ψ_X]](x, \tilde{h}) \right)
$$

**최종 등변 그룹 합성곱**:[1]

$$
[f ⋆^α_G ψ](x, h) = \sum_{\tilde{c}=1}^{\tilde{N}_c} \int_H α_X(x, h, \tilde{h}) · α^{\tilde{c}}_C(h, \tilde{h}) · \tilde{f}(x, h, \tilde{h}) d\tilde{h} 
$$

#### 2.4 잔차 주의 분기(Residual Attention Branch)

기존의 주의 방법은 $$α^+ = 1 + α$$로 계산하여, 주의 맵이 $$[1,2] $$ 범위로 제한되어 입력 억제 능력이 없습니다.[2][1]

이를 해결하기 위해 **잔차 주의 분기**를 제안:[1]

$$
α^- = (1 - α^+), \quad α^+ = 1 - α^-
$$

결과적으로 주의 맵이 $$[0,1] $$ 범위를 가지면서 동시에 잔차 연결의 이점을 유지합니다.[1]

---

### 3. 모델 구조와 네트워크 아키텍처

#### 3.1 주요 아키텍처 변형

논문은 다음의 그룹 등변 네트워크들의 등변 버전을 구성합니다:[1]

- **p4-CNN / p4m-CNN**: Cohen & Welling (2016)의 기본 그룹 등변 네트워크
  - p4: 이산 90° 회전에 대한 등변성 (SE(2)의 부분군)
  - p4m: 회전과 반사에 대한 등변성 (E(2))

- **DenseNet**: Veeling et al. (2018)의 그룹 등변 DenseNet

#### 3.2 주의 메커니즘 구성

Figure 6에 보인 대로, 순차적 채널-공간 주의가 적용됩니다:[1]

1. 중간 합성곱 결과 $$\tilde{f}(x, h, \tilde{h})$$ 계산
2. 채널 통계 $$s^{avg}_C, s^{max}_C$$ 계산
3. 채널 주의 $$α_C$$ 적용
4. 공간 통계 계산
5. 공간 주의 $$α_X$$ 적용
6. 최종 풀링 전에 주의 가중치 적용

***

### 4. 성능 향상 결과

#### 4.1 rot-MNIST 결과[1]

| 네트워크 | 테스트 오류율(%) | 파라미터 |
|---------|----------|---------|
| p4-CNN | 2.048 ± 0.045 | 24.61K |
| α-p4-CNN | **1.696 ± 0.021** | 73.13K |
| αF-p4-CNN | 1.795 ± 0.028 | 29.46K |

**분석**: 주의를 전체 그룹에 적용(α-p4-CNN)할 경우 0.35% 성능 개선. 입력에만 주의를 적용(αF-p4-CNN)한 경우에도 유의미한 개선.

#### 4.2 CIFAR-10 결과[1]

| 네트워크 | p4 기본 | αF-p4 | p4m 기본 | αF-p4m |
|---------|--------|-------|---------|---------|
| All-CNN | 9.32% | **8.8%** | 7.61% | **6.93%** |
| ResNet44 | 15.72% | **10.82%** | - | - |

**분석**: ResNet44에서 무려 4.9% 감소 (중대한 개선). 메모리 제약으로 완전한 주의(α-networks)는 실행 불가능하여 αF-networks만 사용됨.

#### 4.3 PatchCamelyon (의료 이미지)[1]

| 네트워크 | p4 기본 | αF-p4 | p4m 기본 | αF-p4m |
|---------|--------|-------|---------|---------|
| DenseNet | 12.45% | **11.34%** | 11.64% | **10.88%** |

**분석**: 흥미로운 발견으로, 네트워크가 세포 핵(nuclei)에 집중하고 배경을 제거하는 등변 주의 맵을 학습했습니다 (Figure 8).

#### 4.4 일반화 성능 분석

1. **데이터 효율성**: 주의 메커니즘이 대칭성 정보를 더 효율적으로 활용하여, 동일한 파라미터 수에서도 더 나은 성능을 달성

2. **오버핏팅 감소**: 의료 이미지 (PatchCamelyon)에서 특히 명확한 개선, 이는 제한된 학습 데이터 환경에서의 강화된 정규화 효과

3. **수렴 안정성**: 더 큰 네트워크(ResNet44)에서 특히 주목할 만한 개선으로, 복잡한 대칭 관계 학습의 안정화

***

### 5. 한계점

#### 5.1 계산 복잡도[1]

**주요 문제**: CIFAR-10에서 α-p4 All-CNN이 약 **72GB CUDA 메모리** 필요 (기본 p4-All-CNN: 5GB)

이유: 중간 합성곱 응답 $$\tilde{f}(x, h, \tilde{h})$$의 저장 (Equation 16 참조)

**해결책**: 입력 레벨 주의(αF-networks) 사용으로 메모리를 크게 감소시키되, 일부 성능 트레이드오프 발생

#### 5.2 근사 등변성[1]

CIFAR-10 실험에서 기존 아키텍처의 보폭(stride)과 입력 크기 조합으로 인해 **근사 등변성만 달성됨**.

이를 해결하기 위해 보폭 있는 합성곱을 보통 합성곱 + 최대 풀링으로 교체하여 정확한 등변성을 확보했습니다.

#### 5.3 적용 범위[1]

현재 구현은 2D 데이터와 SE(2), E(2) 그룹에 제한됨. 3D 확장(CT 스캔 등)은 미래 과제로 남음.

***

### 6. 특히 중요한 발견: 해석가능성

#### 6.1 등변 주의 맵 시각화[1]

Figure 7-8에서 보인 주요 특성:

1. **회전 등변성 보존**: 회전된 입력에 대해 동일한 객체에 대한 주의 맵이 일관되게 회전됨

2. **의료 이미지 적용**: PatchCamelyon에서 네트워크가 자동으로 세포 핵에 초점을 맞추고 배경을 억제

3. **신뢰성 보장**: 비등변 CNN의 주의와 달리, 등변 주의는 모든 회전에서 일관된 해석 제공 → **의료 이미지 진단에서 중요**

---

### 7. 일반화 성능과 관련 내용

#### 7.1 데이터 증강과의 상호작용[1]

흥미로운 관찰: "추가 대칭성을 포함하면 데이터 증강의 효과가 감소한다"

**해석**: 비등변 네트워크는 각 대칭 변형을 독립적으로 학습하지만, 등변 네트워크는 단일 개념으로 모두 학습하므로, 증강의 추가 이점이 감소함.

#### 7.2 로컬-글로벌 주의[1]

논문에서는 로컬 주의(특정 필터의 응답에만 초점)과 글로벌 주의(입력 전체)의 균형을 맞추며, 결과적으로 의미 있는 대칭 조합만 활성화됨.

---

### 8. 최신 연구 기반 영향과 앞으로의 연구 방향

#### 8.1 논문 이후의 연구 동향 (2020-2025)[3][4][2]

1. **의료 영상 응용 확대**:
   - 구형 CNN (Spherical CNN)을 사용한 뇌 이미지 등 회전대칭 도메인 적용[5]
   - 미분 프라이버시(Differential Privacy)와 등변 네트워크 결합[6]
   - 3D 의료 영상 (CT, MRI) 분석에서 SE(3) 등변성 추가[7][8]

2. **3D 확장**:
   - SE(3)-Transformers (2020): 3D 점 구름에서 회전 변환 등변 자기 주의[9]
   - E2PN: 효율적인 SE(3) 등변 점 네트워크[10]
   - Clebsch-Gordan Transformer (2025): O(N log N) 복잡도의 전역 주의로 해결[11]

3. **효율성 개선**:
   - 적응형 집계를 통한 메모리 감소 방법[12]
   - 가중치 공유 전략의 재검토[13]

#### 8.2 의료 영상 분야에서의 실제 영향

최근 연구(2023-2025)에서 주목할 만한 진전:[14][15][8][7]

- **혈관 방향 예측 (SIRE)**: 회전 등변 그래프 신경망으로 척도 불변성과 회전 등변성을 동시에 달성[7]
- **의료 이미지 정렬**: 등변 신경망을 글로벌 강직 운동(rigid motion) 보정과 국소 변형을 통합하는 프레임워크에 통합[14]
- **개인정보 보호 기반 분석**: 차등 프라이버시와 등변 네트워크 결합으로 민감한 의료 데이터 보호[6]

#### 8.3 앞으로 고려할 연구 방향

**논문 제시 방향**:[1]
1. **계산 효율성**: 완전 주의 네트워크의 메모리 요구사항 감소 방법
2. **3D 확장**: CT 스캔, 뇌 MRI 등 3D 의료 이미지로의 확대
3. **해석가능성 강화**: 의료 진단에서 주의 맵의 임상적 검증

**커뮤니티 확장 연구 (2020 이후)**:
1. **고차 주의 메커니즘**: Transformer 스타일의 다중 헤드 등변 주의[11]
2. **약한 감독 학습**: 부분 라벨링된 의료 데이터에서의 등변 네트워크 활용[16]
3. **다중 도메인 일반화**: 서로 다른 의료 센터의 이미지 간 강화된 일반화[16]

***

### 9. 결론 및 종합 분석

**학문적 기여:**

이 논문은 두 개의 중요한 독립적 영역—**그룹 등변성(group equivariance)**과 **주의 메커니즘(attention mechanisms)**—을 통합하는 **수학적으로 우아한 프레임워크**를 제시합니다. 특히 Theorem 1은 모든 기존 시각 주의 방법이 이 프레임워크의 특수한 경우임을 증명하여, 이전의 산발적인 주의 방법들에 대한 통일 이론을 제공합니다.[1]

**실용적 영향:**

- **의료 영상**: 일관된 해석가능성과 회전 불변 진단
- **데이터 효율성**: 제한된 훈련 데이터 환경에서의 강화된 성능
- **일반화 능력**: 다양한 방향과 스케일의 객체 인식

**한계와 미래:**

계산 복잡도가 주요 장애물이지만, 이후의 3D 확장, 고효율 주의, 그리고 Transformer 통합 연구들이 이를 극복해가고 있습니다. 특히 의료 영상 분석에서의 적용이 현재 활발히 진행 중이며, 진단 신뢰성과 개인정보 보호를 동시에 달성하는 방향으로 발전하고 있습니다.

***

## 참고 자료

 논문: "Attentive Group Equivariant Convolutional Networks", Romero et al., ICML 2020[1]
 "SE(3)-Transformers: 3D Roto-Translation Equivariant Attention Networks", Fuchs et al., NeurIPS 2020[2]
 "Efficient Equivariant Networks for 3D Point Clouds", Zhu et al., CVPR 2023[3]
 "Clebsch-Gordan Transformer: Fast and Global Equivariant Attention", Howell et al., 2025[4]
 "Adaptive aggregation of Monte Carlo augmented decomposed filters", 2024[13]
 "Adaptive aggregation..." related works[12]
 SE(3)-Transformers reference[9]
 UniMo Motion Correction Framework[14]
 SIRE: Rotation-Equivariant Vessel Orientation[7]
 Spherical CNN for Medical Imaging[15]
 Equivariant Spherical CNN[8]
 Differentially Private Equivariant Deep Learning[6]
 Generalizable and Explainable Deep Learning[16]
 Spherical CNNs for Medical Applications[5]
 Clebsch-Gordan Transformer[11]
 E2PN: Efficient SE(3)-Equivariant Point Network[10]

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/5f45b03e-dd02-46ff-955b-79216b866f9d/2002.03830v3.pdf)
[2](https://arxiv.org/pdf/1602.07576.pdf)
[3](http://arxiv.org/pdf/1911.07849.pdf)
[4](https://arxiv.org/pdf/2310.02970.pdf)
[5](https://pmc.ncbi.nlm.nih.gov/articles/PMC10350095/)
[6](https://arxiv.org/abs/2209.04338)
[7](https://arxiv.org/abs/2311.05400)
[8](https://arxiv.org/abs/2307.03298)
[9](https://papers.neurips.cc/paper/2020/file/15231a7ce4ba789d13b722cc5c955834-Paper.pdf)
[10](https://cvpr.thecvf.com/virtual/2023/poster/21183)
[11](https://www.arxiv.org/abs/2509.24093)
[12](http://arxiv.org/pdf/2305.10110.pdf)
[13](http://arxiv.org/pdf/2105.13926.pdf)
[14](https://www.semanticscholar.org/paper/59c6f62967c1972bd52ab1f4ca6727271901b1c1)
[15](https://www.semanticscholar.org/paper/9bf263e2fb9959582aaebeea12ebe8678bfa4b56)
[16](https://arxiv.org/html/2503.08420v1)
[17](https://arxiv.org/pdf/2002.03830.pdf)
[18](https://arxiv.org/pdf/2108.03348.pdf)
[19](http://arxiv.org/pdf/2104.04848.pdf)
[20](https://arxiv.org/abs/2002.03830)
[21](https://openaccess.thecvf.com/content/ICCV2021/papers/Chen_Equivariant_Imaging_Learning_Beyond_the_Range_Space_ICCV_2021_paper.pdf)
[22](https://openreview.net/pdf?id=niyvAOOnwPM)
[23](https://research.vu.nl/files/401942277/Romero20a_Attentive_group_equivariant_convolutional_networks.pdf)
[24](https://arxiv.org/pdf/2306.07783.pdf)
[25](https://arxiv.org/abs/1810.06889)
[26](http://proceedings.mlr.press/v119/romero20a/romero20a.pdf)
[27](https://dl.acm.org/doi/10.5555/3524938.3525696)
[28](https://ieeexplore.ieee.org/document/10635407/)
[29](https://www.ewadirect.com/proceedings/ace/article/view/13990)
[30](https://qims.amegroups.com/article/view/123439/html)
[31](https://arxiv.org/abs/2407.17219)
[32](https://ieeexplore.ieee.org/document/10695984/)
[33](https://www.cureus.com/articles/310907-leveraging-artificial-neural-networks-and-support-vector-machines-for-accurate-classification-of-breast-tumors-in-ultrasound-images)
[34](https://arxiv.org/pdf/2206.01136.pdf)
[35](https://arxiv.org/pdf/2206.15274.pdf)
[36](https://arxiv.org/html/2404.15786v1)
[37](https://arxiv.org/pdf/2209.01725.pdf)
[38](https://pmc.ncbi.nlm.nih.gov/articles/PMC12071792/)
[39](https://www.mdpi.com/2076-3417/13/18/10521/pdf?version=1695283635)
[40](https://arxiv.org/abs/2205.00630)
[41](https://uu.diva-portal.org/smash/get/diva2:1825662/FULLTEXT01.pdf)
[42](https://www.nature.com/articles/s41598-024-65597-x)
[43](https://arxiv.org/html/2402.16825)
[44](https://www.nature.com/articles/s41524-025-01535-3)
[45](https://academic.oup.com/bioinformatics/article/35/14/i530/5529148)
