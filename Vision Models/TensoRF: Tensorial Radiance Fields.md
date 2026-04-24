
# TensoRF: Tensorial Radiance Fields

> **논문 정보**
> - **제목**: TensoRF: Tensorial Radiance Fields
> - **저자**: Anpei Chen, Zexiang Xu, Andreas Geiger, Jingyi Yu, Hao Su
> - **발표**: ECCV 2022
> - **arXiv**: [2203.09517](https://arxiv.org/abs/2203.09517)
> - **공식 페이지**: [apchenstu.github.io/TensoRF](https://apchenstu.github.io/TensoRF/)
> - **GitHub**: [apchenstu/TensoRF](https://github.com/apchenstu/TensoRF)

---

## 1. 핵심 주장 및 주요 기여 요약

TensoRF는 radiance field를 모델링하고 재구성하는 새로운 접근 방식으로, NeRF와 달리 장면의 radiance field를 4D 텐서로 모델링한다. 이 텐서는 복셀(voxel)당 다채널 특징(per-voxel multi-channel features)을 갖는 3D 복셀 그리드를 표현한다.

핵심 아이디어는 4D 장면 텐서를 여러 개의 저랭크(low-rank) 컴팩트 텐서 성분으로 분해(factorize)하는 것이다. 전통적인 CP 분해(CANDECOMP/PARAFAC)는 텐서를 컴팩트 벡터를 가진 랭크-1 성분들의 합으로 분해하며, 이를 통해 NeRF 대비 성능 향상을 이끌어낸다. 성능을 더욱 향상시키기 위해 저자들은 새로운 VM(Vector-Matrix) 분해를 도입하였는데, 이는 텐서의 두 모드에 대한 저랭크 제약을 완화하고 텐서를 컴팩트 벡터 및 행렬 인수(factor)로 분해한다.

### 주요 기여 요약표

| 기여 항목 | 내용 |
|-----------|------|
| 표현 방식 | NeRF의 MLP 대신 4D 텐서 기반 표현 |
| 분해 방법 | CP Decomposition + 신규 VM Decomposition |
| 재구성 속도 | CP: < 30분 / VM: < 10분 |
| 모델 크기 | CP: < 4MB / VM: < 75MB |
| 렌더링 품질 | 기존 SOTA 초과 달성 (PSNR 최대 33.14) |

---

## 2. 해결하고자 하는 문제, 제안 방법, 모델 구조, 성능 및 한계

### 2.1 해결하고자 하는 문제

기존의 특징 그리드(feature grid)를 활용한 방법들은 렌더링 속도를 빠르게 만들 수 있었지만, 여전히 긴 재구성 시간과 높은 메모리 비용이라는 문제를 가지고 있어 NeRF의 컴팩트함을 희생해야 했다. TensoRF는 특징 그리드를 기반으로, 텐서 분해 기법을 활용하여 빠른 재구성과 컴팩트한 모델링을 동시에 달성하는 새로운 텐서 장면 표현을 제안한다.

NeRF와 같이 MLP를 활용하는 전통적인 방법들은 상당한 잠재력을 보여주지만, 훈련 시간과 메모리 비효율성 문제를 가진다. TensoRF 논문은 텐서 분해 기법을 활용하여 이러한 한계를 극복하는 혁신적인 접근 방식을 도입한다.

---

### 2.2 제안 방법 (수식 포함)

#### (a) 장면의 텐서 표현

Radiance field는 임의의 3D 위치 $\mathbf{x}$와 시점 방향 $\mathbf{d}$를 입력으로 받아 볼륨 밀도 $\sigma$와 시점 의존적 색상 $\mathbf{c}$를 출력하는 함수로, 미분 가능한 레이 마칭(ray marching)을 통한 볼륨 렌더링을 지원한다. TensoRF는 복셀당 다채널 특징을 갖는 정규 3D 그리드 $\mathcal{G}$를 사용하여 이 함수를 모델링하며, 기하 그리드 $\mathcal{G}_\sigma$와 외형 그리드 $\mathcal{G}_c$로 분리하여 볼륨 밀도 $\sigma$와 시점 의존적 색상 $\mathbf{c}$를 각각 모델링한다.

#### (b) CP 분해 (CANDECOMP/PARAFAC Decomposition)

CP 분해는 3D 텐서 $\mathcal{T} \in \mathbb{R}^{I \times J \times K}$를 랭크-1 성분들의 합으로 분해한다:

$$\mathcal{T} = \sum_{r=1}^{R} \mathbf{v}_r^X \otimes \mathbf{v}_r^Y \otimes \mathbf{v}_r^Z$$

여기서 $\mathbf{v}_r^X \in \mathbb{R}^I$, $\mathbf{v}_r^Y \in \mathbb{R}^J$, $\mathbf{v}_r^Z \in \mathbb{R}^K$는 각 축 방향의 벡터 인수이며, $\otimes$는 외적(outer product)이다.

CP 분해는 텐서를 컴팩트 벡터를 가진 랭크-1 성분들의 합으로 분해하며, 이를 통해 바닐라 NeRF 대비 개선을 이끌어낸다.

#### (c) VM 분해 (Vector-Matrix Decomposition) — 핵심 기여

VM 분해는 텐서의 두 모드에 대한 저랭크 제약을 완화하고, 텐서를 컴팩트한 벡터 및 행렬 인수로 분해하는 새로운 방식이다.

VM 분해는 3D 텐서를 다음과 같이 표현한다:

$$\mathcal{T} = \sum_{r=1}^{R} \mathbf{v}_r^X \otimes \mathbf{M}_r^{Y,Z} + \mathbf{v}_r^Y \otimes \mathbf{M}_r^{X,Z} + \mathbf{v}_r^Z \otimes \mathbf{M}_r^{X,Y}$$

여기서:
- $\mathbf{v}_r^X \in \mathbb{R}^{I}$: $X$ 축 방향의 벡터
- $\mathbf{M}_r^{Y,Z} \in \mathbb{R}^{J \times K}$: $Y$ - $Z$ 평면의 행렬
- $\otimes$: 벡터-행렬 외적(outer product)

기하 텐서 $\mathcal{G}_\sigma$는 다음과 같이 분해된다:

$$\mathcal{G}_\sigma = \sum_{r=1}^{R_\sigma} \mathbf{v}_{\sigma,r}^X \otimes \mathbf{M}_{\sigma,r}^{Y,Z} + \mathbf{v}_{\sigma,r}^Y \otimes \mathbf{M}_{\sigma,r}^{X,Z} + \mathbf{v}_{\sigma,r}^Z \otimes \mathbf{M}_{\sigma,r}^{X,Y}$$

외형(appearance) 텐서 $\mathcal{G}_c$도 동일하게 분해된다:

$$\mathcal{G}_c = \sum_{r=1}^{R_c} \mathbf{v}_{c,r}^X \otimes \mathbf{M}_{c,r}^{Y,Z} + \mathbf{v}_{c,r}^Y \otimes \mathbf{M}_{c,r}^{X,Z} + \mathbf{v}_{c,r}^Z \otimes \mathbf{M}_{c,r}^{X,Y}$$

#### (d) 특징 계산 및 렌더링

임의의 음영 위치 $\mathbf{x} = (x, y, z)$에서, 벡터($\mathbf{v}$)/행렬($\mathbf{M}$) 인수로부터 선형/쌍선형 샘플링된 값을 사용하여 해당 텐서 성분의 삼선형 보간값을 계산한다. 밀도 성분 값 $\mathcal{A}^\sigma(\mathbf{x})$는 합산되어 볼륨 밀도 $\sigma$를 직접 계산한다. 외형 값 $\mathcal{A}^c(\mathbf{x})$는 연결되어 외형 행렬 $\mathbf{B}$와 곱해진 후 디코딩 함수 $S$로 전달되어 RGB 색상 $\mathbf{c}$를 회귀한다. 디코딩 함수 $S$는 구면 조화 함수(SH) 또는 완전 연결 네트워크(FCN)일 수 있다.

수식으로 표현하면:
$$\sigma(\mathbf{x}) = \text{sum}\left(\mathcal{A}^\sigma(\mathbf{x})\right)$$

$$\mathbf{c}(\mathbf{x}, \mathbf{d}) = S\left(\mathbf{B} \cdot \bigoplus_m \mathcal{A}^c_m(\mathbf{x}),\ \mathbf{d}\right)$$

#### (e) 손실 함수 및 정규화

주어진 다중 시점 입력 이미지와 알려진 카메라 포즈를 바탕으로, 텐서 radiance field는 그라디언트 강하(gradient descent)를 통해 장면별로 최적화되며, 지면 진실(ground truth) 픽셀 색상만을 감독 신호로 사용하는 $L_2$ 렌더링 손실을 최소화한다. Radiance field는 텐서 분해로 설명되며, 최적화에서 전체 필드를 연관 짓고 정규화하는 전역 벡터와 행렬 세트로 모델링된다.

전체 손실 함수:

$$\mathcal{L} = \sum_{\mathbf{r} \in \mathcal{R}} \left\| \hat{\mathbf{C}}(\mathbf{r}) - \mathbf{C}(\mathbf{r}) \right\|_2^2 + \omega \cdot \mathcal{L}_\text{reg}$$

여기서 $\hat{\mathbf{C}}(\mathbf{r})$는 렌더링된 색상, $\mathbf{C}(\mathbf{r})$는 실제 색상, $\mathcal{L}_\text{reg}$는 정규화 항($L_1$ sparsity + Total Variation)이다.

이 방법은 $L_1$ sparsity 및 total variation 정규화를 통합하여 AR/VR 및 로봇 공학에서의 실용적인 응용을 위한 확장 가능한 최적화를 가능하게 한다.

---

### 2.3 모델 구조

장면은 해당하는 축을 따라 장면 외형과 기하를 설명하는 벡터 및 행렬의 집합을 사용하는 텐서 radiance field로 모델링된다. 이러한 벡터/행렬 인수는 벡터-행렬 외적을 통해 볼륨 밀도와 시점 의존적 RGB 색상을 계산하는 데 사용되며, 이는 효율적인 radiance field 재구성과 사실적인 렌더링으로 이어진다.

아키텍처 요약:

```
입력 이미지 (다중 시점) + 카메라 포즈
        ↓
[ 텐서 분해 기반 장면 표현 ]
  ├─ 기하 그리드 G_σ → VM 분해 → 볼륨 밀도 σ
  └─ 외형 그리드 G_c → VM 분해 → 외형 특징 벡터
        ↓
[ 디코딩 함수 S ]
  ├─ Spherical Harmonics (SH) : 빠른 렌더링
  └─ Fully-Connected Network (FCN) : 높은 품질
        ↓
[ 미분 가능 볼륨 렌더링 ]
        ↓
렌더링 이미지 (L2 Loss 최적화)
```

---

### 2.4 성능 향상

TensoRF with CP 분해는 NeRF 대비 빠른 재구성($< 30$분)과 더 나은 렌더링 품질, 더 작은 모델 크기($< 4$ MB)를 달성한다. 또한 TensoRF with VM 분해는 렌더링 품질을 더욱 향상시켜 이전 SOTA 방법들을 능가하면서 재구성 시간을 $< 10$분으로 줄이고 $< 75$ MB의 컴팩트한 모델 크기를 유지한다.

이 방법은 PSNR 값 최대 33.14를 달성하며, NeRF 대비 훈련 시간을 수 시간에서 수 분으로 단축한다.

| 방법 | 재구성 시간 | 모델 크기 | 렌더링 품질 |
|------|-----------|----------|------------|
| NeRF | ~수 시간 | ~5MB | 기준 |
| TensoRF (CP) | < 30분 | < 4MB | NeRF 초과 |
| TensoRF (VM) | < 10분 | < 75MB | SOTA 초과 |
| Instant-NGP | 수 분 | ~수십 MB | 높음 |
| Plenoxels | 수십 분 | 수백 MB | 높음 |

---

### 2.5 한계

Radiance field는 텐서 분해로 설명되며, 최적화에서 전체 필드를 연관 짓고 정규화하는 전역 벡터 및 행렬 세트로 모델링된다. 그러나 이것은 때때로 그라디언트 강하에서 과적합(overfitting) 및 지역 최솟값(local minima) 문제로 이어져 아웃라이어(outlier)를 초래할 수 있다.

다른 방법들은 다중 장면에 걸쳐 학습된 일반화 가능한 네트워크 모듈을 설계하여 이미지 의존적 radiance field 렌더링과 빠른 재구성을 달성한다. TensoRF의 접근 방식은 radiance field 표현에 초점을 맞추며 NeRF처럼 장면별 최적화(per-scene optimization)만을 고려한다. 이 표현만으로도 이미 어떤 cross-scene 일반화 없이 매우 효율적인 radiance field 재구성을 이끌어낼 수 있음을 보여준다. 일반화 가능한 설정으로의 확장은 향후 연구 과제로 남긴다.

---

## 3. 모델 일반화 성능 향상 가능성

TensoRF는 명시적으로 **장면별 최적화(per-scene optimization)** 방식을 취하며, 일반화 가능한 설정을 향후 과제로 남겼다. 그러나 이 한계를 극복할 수 있는 다양한 방향이 존재한다.

### 3.1 Cross-scene 일반화의 한계

일반화 가능한 네트워크 모듈을 다중 장면에 걸쳐 학습하여 이미지 의존적 렌더링과 빠른 재구성을 달성하는 다른 방법들과 달리, TensoRF는 장면별 최적화에만 집중하며 어떠한 cross-scene 일반화 없이도 매우 효율적인 재구성이 가능함을 보여준다. 일반화 가능한 설정으로의 확장은 향후 연구 과제로 남겨진다.

### 3.2 일반화 향상을 위한 잠재 방향

**① 텐서 분해와 메타러닝(Meta-learning)의 결합**

TensoRF의 컴팩트한 텐서 표현(벡터/행렬 인수)은 메타러닝 프레임워크(예: MAML)와 결합 시, 소수의 이미지로부터 텐서 인수를 빠르게 초기화하는 것이 이론적으로 가능하다.

**② 이미지 특징 기반 컨디셔닝**

이미지 특징 기반 방법들은 3D 포인트를 투영하여 사용 가능한 2D 이미지 특징에 NeRF를 컨디셔닝한다. TensoRF의 텐서 인수를 이미지 특징으로 컨디셔닝하면 새로운 장면에 대한 적응력을 향상시킬 수 있다.

**③ 텐서 인수 초기화 사전 학습**

좌표 네트워크와 텐서 radiance field를 결합한 연구에서는 좌표 네트워크가 물체 형상과 같은 전역 맥락을 포착하고, 다중 평면 인코딩이 세밀한 디테일 표현에 집중하게 하는 것이 가능함을 보인다.

**④ 동적 장면 확장**

TensoRF는 정적 장면의 빠른 고품질 재구성에서 잠재력을 보여준다. D-TensoRF는 이를 동적 장면으로 확장한 텐서 radiance field로, 특정 시간의 새로운 시점 합성이 가능하다. 이 연구에서는 동적 장면의 radiance field를 5D 텐서로 간주하며, 5D 텐서는 X, Y, Z, 시간 각각에 해당하는 축을 가진 4D 그리드를 나타낸다.

**⑤ Instant-NGP와의 결합 가능성**

해시 인코딩 기법(Instant-NGP)과 TensoRF의 인수 분해 기반 기법은 서로 직교적(orthogonal)이므로, 각 벡터/행렬 인수를 이 해시 기법으로 인코딩하는 조합을 향후 연구 방향으로 제시한다.

---

## 4. 앞으로의 연구에 미치는 영향 및 고려할 점

### 4.1 연구에 미치는 영향

TensoRF는 초고속 훈련 프로세스, 컴팩트한 메모리 사용량, SOTA 렌더링 품질을 동시에 달성하는 새로운 접근 방식을 제안한다.

**(1) 텐서 분해의 NeRF 적용 패러다임 정립**
TensoRF는 텐서 분해라는 고전적인 수학 도구를 NeRF 표현에 성공적으로 접목함으로써, 이후 연구들에 텐서 분해 기반 장면 표현의 가능성을 열었다. 이는 K-Planes, HexPlane, 4D Gaussian Splatting 등 다수의 후속 연구에 영향을 주었다.

**(2) 동적 장면으로의 확장 자극**
2023년의 NeRFPlayer 논문은 TensoRF와 유사한 접근법을 채택하여, 정적·변형·신규 등 세 가지 클래스로 장면을 분해한다.

**(3) 메모리 효율성 연구 방향 제시**
기존의 DVGO, Plenoxels 같은 방법들은 복셀별 특징을 직접 최적화하여 대용량 메모리가 필요하지만, TensoRF는 특징 그리드를 컴팩트한 성분으로 분해하여 현저히 높은 메모리 효율성을 달성한다.

**(4) 3D Gaussian Splatting과의 비교**
2022년 이후 3DGS 및 Diffusion/VFM 논문들의 기하급수적인 성장은 이러한 방법들의 획기적인 효율성과 누락된 정보를 합성하는 능력을 반영하며, 이는 초기 NeRF 변형들의 계산 비용 및 희소 입력에서의 과적합 한계를 직접적으로 해결한다.

---

### 4.2 2020년 이후 관련 최신 연구 비교 분석

| 연구 | 발표 | 핵심 방법 | TensoRF 대비 특이점 |
|------|------|----------|-------------------|
| **NeRF** (Mildenhall et al.) | ECCV 2020 | MLP 기반 | TensoRF의 기준점 |
| **Plenoxels** (Yu et al.) | CVPR 2022 | 희소 복셀 그리드 + SH | 직접 복셀 최적화, 고메모리 |
| **Instant-NGP** (Müller et al.) | SIGGRAPH 2022 | 다해상도 해시 인코딩 | 가장 빠른 훈련, 그러나 다른 방향성 |
| **TensoRF** (Chen et al.) | ECCV 2022 | CP/VM 텐서 분해 | 컴팩트함 + 빠른 속도 + 고품질 동시 달성 |
| **D-TensoRF** (Jang et al.) | arXiv 2022 | 5D 텐서로 동적 장면 확장 | TensoRF를 동적 장면에 적용 |
| **HexPlane/K-Planes** | 2023 | 다중 평면 특징 | TensoRF의 2D 평면 분해 확장 |
| **3D Gaussian Splatting** | SIGGRAPH 2023 | 가우시안 점군 | 실시간 렌더링 가능, 암시적 표현 탈피 |

최근 연구들은 TensoRF와 HexPlane의 다중 평면 표현을 따르면서 ReLU 기반 좌표 네트워크와 결합하는 방식으로 NeRF 성능을 향상시키고 있다.

---

### 4.3 앞으로 연구 시 고려할 점

**(1) 일반화 가능한 텐서 표현 개발**

NeRF 연구는 렌더링 효율성 향상, 적은 시점(few-view) 합성 최적화, 렌더링 품질 향상, 자기지도 학습(self-supervised learning) 개발 등 주요 분야에 집중하며, 실시간 렌더링 수요를 해결하고 3D 재구성의 정확도와 일반화 능력을 개선하며 대규모 어노테이션 데이터 의존성을 줄이는 것을 목표로 한다.

**(2) 희소 입력 조건에서의 과적합 방지**

텐서 특징과 좌표 네트워크를 결합한 최근 연구에서는 few-shot 조건에서 기준 모델보다 일관성 있게 뛰어난 성능을 보이며, 훈련 시점과 인접/비인접 시점 간 이미지 품질 차이가 줄어드는 강한 안정성을 보인다.

**(3) 해시 인코딩과 텐서 분해의 결합**

Instant-NGP의 다해상도 해시 기법과 TensoRF의 텐서 분해는 서로 직교적 기법으로, 이를 결합하면 메모리 효율과 훈련 속도를 동시에 향상시킬 수 있다.

**(4) 동적 장면 및 대규모 야외 장면으로의 확장**

희소 시점 3D 재구성은 로봇 공학, 증강/가상 현실(AR/VR), 자율 시스템과 같이 고밀도 이미지 획득이 비실용적인 응용 분야에서 필수적이다. 이러한 환경에서는 최소한의 이미지 겹침으로 인해 신뢰할 수 있는 대응점 매칭이 불가능하다.

**(5) 3D Gaussian Splatting과의 통합**

최근 3D 재구성의 최신 발전은 신경 암시적 모델(NeRF 및 그 정규화 버전), 명시적 포인트 클라우드 기반 접근법(3D Gaussian Splatting), 확산 모델 및 비전 기반 모델의 사전 지식을 활용하는 하이브리드 프레임워크를 포함한다. TensoRF의 텐서 표현을 3DGS의 렌더링 파이프라인과 통합하는 연구가 유망한 방향으로 떠오르고 있다.

---

## 📚 참고 자료 (출처 목록)

1. **논문 원문 (ECCV 2022)**: Chen, A., Xu, Z., Geiger, A., Yu, J., Su, H. — *TensoRF: Tensorial Radiance Fields*, ECCV 2022. [Springer Link](https://link.springer.com/chapter/10.1007/978-3-031-19824-3_20) / [ACM DL](https://dl.acm.org/doi/10.1007/978-3-031-19824-3_20)
2. **arXiv 프리프린트**: [arxiv.org/abs/2203.09517](https://arxiv.org/abs/2203.09517)
3. **공식 프로젝트 페이지**: [apchenstu.github.io/TensoRF](https://apchenstu.github.io/TensoRF/)
4. **GitHub 공개 코드**: [github.com/apchenstu/TensoRF](https://github.com/apchenstu/TensoRF)
5. **ECVA 공개 PDF**: [ecva.net/papers/eccv_2022/.../136920332.pdf](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136920332.pdf)
6. **ResearchGate 논문**: [researchgate.net TensoRF](https://www.researchgate.net/publication/359309205_TensoRF_Tensorial_Radiance_Fields)
7. **D-TensoRF (동적 장면 확장)**: Jang, H. et al., *D-TensoRF: Tensorial Radiance Fields for Dynamic Scenes*, arXiv 2022. [arxiv.org/abs/2212.02375](https://arxiv.org/abs/2212.02375)
8. **좌표 네트워크+텐서 결합 연구**: *Synergistic Integration of Coordinate Network and Tensorial Feature for Improving Neural Radiance Fields from Sparse Inputs*, arXiv 2024. [arxiv.org/html/2405.07857](https://arxiv.org/html/2405.07857)
9. **NeRF 서베이**: *Neural Radiance Fields for the Real World: A Survey*, arXiv 2025. [arxiv.org/html/2501.13104v1](https://arxiv.org/html/2501.13104v1)
10. **NeRF 시각 렌더링 리뷰**: *Neural Radiance Field-based Visual Rendering: A Comprehensive Review*, arXiv 2024. [arxiv.org/html/2404.00714v1](https://arxiv.org/html/2404.00714v1)
11. **Sparse-View 3D 재구성 서베이**: *Sparse-View 3D Reconstruction: Recent Advances and Open Challenges*, arXiv 2025. [arxiv.org/html/2507.16406v1](https://arxiv.org/html/2507.16406v1)
12. **Nerfstudio TensoRF 문서**: [docs.nerf.studio/nerfology/methods/tensorf.html](https://docs.nerf.studio/nerfology/methods/tensorf.html)
13. **Emergent Mind 논문 분석**: [emergentmind.com/papers/2203.09517](https://www.emergentmind.com/papers/2203.09517)
14. **Neural Fields 논문 데이터베이스**: [neuralfields.cs.brown.edu/paper_405.html](https://neuralfields.cs.brown.edu/paper_405.html)
15. **Awesome-NeRF GitHub 리스트**: [github.com/awesome-NeRF/awesome-NeRF](https://github.com/awesome-NeRF/awesome-NeRF)
16. **NeRF in 2023: Theory and Practice**: [it-jim.com/blog/nerf-in-2023-theory-and-practice](https://www.it-jim.com/blog/nerf-in-2023-theory-and-practice/)

# TensoRF: Tensorial Radiance Fields

### 1. 핵심 주장과 주요 기여

**TensoRF**의 중심 개념은 **4D 텐서로 모델링된 방사 필드(Radiance Field)를 저차(Low-rank) 텐서 성분으로 분해**하는 것입니다. NeRF의 순수 MLP 기반 접근과 달리, TensoRF는 3D 장면을 명시적 복셀 격자(Voxel Grid)로 표현합니다.[1]

**주요 기여:**

- **CP 분해(CANDECOMP/PARAFAC)의 적용**: 순수 벡터 외적을 통해 저차 성분으로 텐서를 분해하여 NeRF보다 빠른 학습과 더 작은 모델 크기(< 4MB)를 달성[1]

- **신규 VM 분해(Vector-Matrix Decomposition) 제안**: 두 모드의 제약을 완화하여 벡터와 행렬의 혼합으로 분해. 동일한 표현 용량으로 필요한 성분 수를 감소시켜 더 빠른 재구성(< 10분)과 우수한 렌더링 품질 달성[1]

- **메모리 효율성**: 공간 복잡도를 O(N³)에서 CP의 경우 O(N), VM의 경우 O(N²)로 감소[1]

***

### 2. 해결 문제 및 제안 방법

**문제점:**
NeRF는 MLP 기반이라 학습 시간이 과도하게 오래 걸리고(수십 시간~수일), 복셀 격자 기반 방법들은 메모리 사용량이 매우 많습니다.[1]

**제안 방법:**

#### **수학적 기반**

**CP 분해(3D 텐서 예시):**

$$T = \sum_{r=1}^{R} \mathbf{v}^1_r \circ \mathbf{v}^2_r \circ \mathbf{v}^3_r$$

여기서 $$\mathbf{v}^m_r$$는 m번째 모드의 r번째 벡터이고, $$\circ$$는 외적입니다.[1]

각 텐서 원소는:

$$T_{ijk} = \sum_{r=1}^{R} v^1_{r,i} \cdot v^2_{r,j} \cdot v^3_{r,k}$$

**VM 분해 (제안한 새로운 방법):**

$$\text{Tensor} = \sum_{r=1}^{R} \left( \mathbf{v}^X_r \circ \mathbf{M}^{YZ}_r + \mathbf{v}^Y_r \circ \mathbf{M}^{XZ}_r + \mathbf{v}^Z_r \circ \mathbf{M}^{XY}_r \right)$$

여기서 $$\mathbf{M}^{YZ}_r$$는 YZ 평면의 행렬(J×K)입니다.[1]

텐서 원소 계산:

$$T_{ijk} = \sum_{r=1}^{R} \sum_{m \in \{X,Y,Z\}} A^m_{r,ijk}$$

여기서 $$A^X_{r,ijk} = v^X_{r,i} \cdot M^{YZ}_{r,jk}$$ 등입니다.[1]

#### **모델 구조**

**밀도와 외관 그리드 분해:**

밀도 그리드($$\mathcal{G}_\sigma$$, 3D 텐서)의 VM 분해:

$$\text{Grid}_\sigma = \sum_{r=1}^{R} \left( \mathbf{v}^X_{\sigma,r} \circ \mathbf{M}^{YZ}_{\sigma,r} + \mathbf{v}^Y_{\sigma,r} \circ \mathbf{M}^{XZ}_{\sigma,r} + \mathbf{v}^Z_{\sigma,r} \circ \mathbf{M}^{XY}_{\sigma,r} \right)$$

외관 그리드($$\mathcal{G}_c$$, 4D 텐서, P개 채널)의 분해는 특성 차원에 벡터만 사용:

$$\text{Grid}_{rad} = \sum_{r=1}^{R_c} \left[ \mathbf{v}^X_{c,r} \circ \mathbf{M}^{YZ}_{c,r} \circ \mathbf{b}_r + \cdots \right]$$

행렬 $$\mathbf{B}$$는 모든 $$\mathbf{b}_r$$을 열로 스택하여 구성되는 글로벌 외관 사전입니다.[1]

**효율적 특성 계산:**

직접 계산:

$$\sigma_{ijk} = \sum_{r=1}^{R} \sum_{m \in \{X,Y,Z\}} A^m_{\sigma,r,ijk}$$

외관 특성(전체 P-채널 벡터):

$$\mathbf{d}_{c,ijk} = \mathbf{B} \cdot \mathbf{A}^c_{ijk}$$

여기서 $$\mathbf{A}^c_{ijk}$$는 모든 성분의 특성을 연결한 벡터입니다.[1]

**삼선형 보간:**

삼선형 보간의 핵심 장점:

$$A^X_r(x) = \mathbf{v}^X_r(x) \otimes \mathbf{M}^{YZ}_r(y,z)$$

여기서 $$\mathbf{v}^X_r(x)$$는 선형 보간, $$\mathbf{M}^{YZ}_r(y,z)$$는 이중선형 보간입니다. 이는 8개 텐서 값의 보간 대신 직접 보간된 값을 계산하므로 계산량이 8배 감소합니다.[1]

**볼륨 렌더링 및 재구성:**

연속 방사 필드:

$$C(\mathbf{r}) = \sum_{q=1}^{Q} T_q (1 - p_q) \left(1 - \exp\left(-\sigma(\mathbf{x}_q) \delta_q\right)\right)$$

렌더링 손실함수:

$$\mathcal{L}_{\text{total}} = \|\mathbf{C} - \tilde{\mathbf{C}}\|^2_2 + \lambda_1 \mathcal{L}_{L1} + \lambda_2 \mathcal{L}_{TV}$$

- $$\mathcal{L}_{L1}$$: L1 정규화(희소성 유도)
- $$\mathcal{L}_{TV}$$: 총변위 정규화(실제 데이터셋 사용 시)[1]

***

### 3. 성능 향상 및 한계

#### **성능 향상**

| 평가 항목 | NeRF | TensoRF-CP | TensoRF-VM(192) |
|---------|------|-----------|-----------------|
| **PSNR (Synthetic-NeRF)** | 31.01 | 31.56 | **33.14** |
| **SSIM** | 0.947 | 0.949 | **0.963** |
| **학습 시간** | 35시간 | 25.2분 | **8.1-17.4분** |
| **모델 크기** | 5.0MB | 3.9MB | 71.8MB |
| **속도 향상** | 1x | ~70x | ~100x |

Synthetic-NSVF 데이터셋:
- TensoRF-VM(192): PSNR **36.52** (DVGO 35.08 vs.)[1]

Tanks & Temples(실제 데이터):
- TensoRF-VM(192): PSNR **28.56** (NeRF 25.78 vs.)[1]

#### **주요 개선 사항**

1. **계산 효율성**: CP/VM 분해로 인한 저차 정규화가 자연스럽게 과적합을 방지[1]

2. **점진적 재구성**: 저해상도($$128^3$$)에서 시작해 선형 및 이중선형 업샘플링으로 단계적 해상도 증가 (2000, 3000, 4000, 5500, 7000 스텝에서)[1]

3. **이중 그리드 구조**: 밀도와 외관 그리드를 분리하여 각각 최적화[1]

#### **한계**

1. **제한된 장면 타입**: 단일 경계 박스를 가진 장면만 지원. **무한 장면(unbounded scenes)은 처리 불가**. 배경과 전경이 함께 있는 야외 장면 적용 불가[1]

2. **일반화 성능의 한계**: 논문은 **장면별 최적화(per-scene optimization)만 고려**하며, 일반화 가능한 모델로의 확장을 미래 연구로 제시[1]

3. **카메라 포즈 의존성**: 알려진 카메라 포즈를 요구하며, 자동 포즈 추정은 미지원[1]

***

### 4. 일반화 성능 향상 가능성[1]

#### **현재 제약:**

- TensoRF는 각 장면에 대해 별도로 최적화되어야 하므로 새로운 장면에 대해 재학습 필요
- 저차 텐서 분해의 정규화는 단일 장면 최적화에는 효과적이지만, 다중 장면 학습에는 제약 존재

#### **개선 방향:**

최신 연구(2023-2024)에서 제시된 해결책들:

1. **크로스 장면 일반화 방법들**: 여러 장면에서 학습하여 새로운 장면에 직접 적용 가능한 모델 개발 진행 중[2][3][4]

2. **저차 정규화의 활용**: 논문의 저차 텐서 분해 개념이 여러 후속 연구에 채택되어 모델 압축과 일반화 개선에 활용[5][6][7]

3. **동적 장면으로의 확장**: D-TensoRF는 시간 축을 추가한 5D 텐서로 동적 장면 처리 가능[8]

4. **역렌더링(Inverse Rendering) 확장**: TensoIR은 TensoRF를 기반으로 기하, 반사율, 조명 추정이 가능한 물리 기반 모델 개발[6]

***

### 5. 앞으로의 연구 시 고려할 점 (최신 연구 기반)[9][7][10][5][8][6]

#### **1단계: 기본 문제 해결**

- **무한 장면 처리**: Cubemap 기반 접근 또는 배경 표현 개선으로 야외 장면 지원[11]
- **일반화 성능**: 다중 장면 데이터에서의 저차 분해 효과 검증 필요. MRVM-NeRF 등 최신 방법에서 마스크 기반 모델링으로 일반화 개선 중[12]

#### **2단계: 응용 분야 확장**

- **의미론적 정보 통합**: NeRF-IS 같은 방법에서 TensoRF의 텐서 분해를 의미적 필드 모델링에 활용 가능[5]
- **얼굴 및 인간 재구성**: TIFace에서 TensoRF 기반 개선으로 정밀 재구성 달성[13]
- **멀티모달 응용**: 열화상, 적외선 등 다중 센서 데이터와의 융합[14]

#### **3단계: 성능 최적화**

- **하드웨어 가속**: Gen-NeRF 같은 알고리즘-하드웨어 공동설계로 실시간 일반화 NeRF 실현[2]
- **스파스 입력 강화**: 적은 입력 이미지 하에서의 강건성 개선. Simple-RF의 정규화 프레임워크 적용[9]

#### **4단계: 혁신적 확장**

- **생성 모델 통합**: 사전학습 확산 모델(Diffusion Model)을 활용한 더 나은 3D 표현 학습[15]
- **대규모 장면 모델링**: 계층적 로컬 필드 또는 멀티 스페이스 표현으로 대규모/복잡 장면 처리[16][17]
- **작업 특화 최적화**: 점유도 예측, 의미 분할 등 특정 비전 작업에 최적화된 텐서 분해 방법 개발[18]

#### **5단계: 근본적 이론 발전**

- **텐서 분해 이론 개선**: 블록 항 분해(BTD) 또는 새로운 분해 방식의 3D 비전 응용[19]
- **저차 정규화의 명시적 모델링**: 장면 복잡도에 따른 최적 텐서 순위 자동 결정[1]

#### **구체적 권장사항**

1. **멀티 스케일 텐서 분해** 도입: 거친 스케일에서 세밀한 스케일까지 계층적 표현[7]

2. **적응형 정규화**: 장면 특성에 따라 L1과 TV 손실의 가중치를 자동 조정[1]

3. **크로스 도메인 사전 학습**: 대규모 장면 데이터셋에서 사전학습된 기저 성분(basis components)을 활용한 전이 학습

4. **불확실성 정량화**: 테스트 시 모델 신뢰도를 추정하여 스파스 입력이나 out-of-distribution 장면에서의 견고성 향상

***

## 결론

TensoRF는 **텐서 분해를 3D 장면 재구성에 처음 적용**하여 NeRF 대비 100배 이상의 속도 향상과 우수한 화질을 동시에 달성했습니다. 특히 VM 분해의 도입으로 모델 표현력과 계산 효율의 균형을 효과적으로 해결했습니다. 앞으로는 **무한 장면 처리, 크로스 장면 일반화, 다양한 비전 작업으로의 확장**이 핵심 과제이며, 최신 연구에서 이들 방향의 개선이 활발히 진행 중입니다.[8][6][7][12][9][5][2][1]

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/243369e2-00e9-4057-a292-14e8c4249985/2203.09517v2.pdf)
[2](https://dl.acm.org/doi/10.1145/3579371.3589109)
[3](https://ieeexplore.ieee.org/document/10658469/)
[4](https://arxiv.org/abs/2312.09095)
[5](https://dl.acm.org/doi/10.1145/3595916.3626379)
[6](https://arxiv.org/html/2304.12461v2)
[7](https://arxiv.org/abs/2303.03808)
[8](https://arxiv.org/abs/2212.02375)
[9](https://arxiv.org/abs/2404.19015)
[10](https://kimjy99.github.io/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0/tensorf/)
[11](https://openaccess.thecvf.com/content/WACV2024/papers/Chang_Fast_Sun-Aligned_Outdoor_Scene_Relighting_Based_on_TensoRF_WACV_2024_paper.pdf)
[12](https://proceedings.iclr.cc/paper_files/paper/2024/file/8882d370cdafec9885b918a8cfac642e-Paper-Conference.pdf)
[13](https://arxiv.org/html/2312.09527v1)
[14](https://arxiv.org/abs/2403.11865)
[15](https://www.sciencedirect.com/science/article/abs/pii/S095741742402935X)
[16](https://arxiv.org/abs/2305.04268)
[17](https://arxiv.org/html/2403.12839v2)
[18](https://openaccess.thecvf.com/content/CVPR2024/papers/Zhao_LowRankOcc_Tensor_Decomposition_and_Low-Rank_Recovery_for_Vision-based_3D_Semantic_CVPR_2024_paper.pdf)
[19](https://dl.acm.org/doi/abs/10.1007/s10489-024-05476-0)
[20](http://www.dpi-journals.com/index.php/dtssehs/article/view/27372)
[21](https://www.semanticscholar.org/paper/d964a74722b87dbcc5e4ecd843e62be39b78eb85)
[22](https://www.semanticscholar.org/paper/eba6b0d9caef19aad9c476d114ad311c6ade9ca7)
[23](https://www.semanticscholar.org/paper/2c9da07d55729addab5c3682042de0e34bcea79a)
[24](https://ems.press/doi/10.4171/owr/2018/46)
[25](https://www.semanticscholar.org/paper/06325c8d7b81fc955ddb55a8e4fe90a79c1b8343)
[26](https://www.semanticscholar.org/paper/1364123ffc9d4421532ee19692ef6b9039530221)
[27](https://teseo.unitn.it/xy-rivista/article/view/2116)
[28](https://arxiv.org/abs/2203.09517)
[29](https://arxiv.org/pdf/2205.06407.pdf)
[30](http://arxiv.org/pdf/2402.16638.pdf)
[31](http://arxiv.org/pdf/1910.09499.pdf)
[32](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136920332.pdf)
[33](https://diglib.eg.org/bitstream/handle/10.1111/cgf15062/v43i2star_03_15062.pdf)
[34](https://openaccess.thecvf.com/content/CVPR2024/papers/Li_GP-NeRF_Generalized_Perception_NeRF_for_Context-Aware_3D_Scene_Understanding_CVPR_2024_paper.pdf)
[35](https://ar5iv.labs.arxiv.org/html/2203.09517)
[36](https://arxiv.org/abs/1807.10027)
[37](https://arxiv.org/abs/2304.11842)
[38](https://www.nature.com/articles/s40494-025-01695-x)
[39](https://arxiv.org/abs/2306.06359)
[40](https://ieeexplore.ieee.org/document/10205429/)
[41](https://link.springer.com/10.1007/s11390-024-4157-6)
[42](https://www.semanticscholar.org/paper/ebbdffb6dde100f70d8f5f3f67b8baffb467ea2b)
[43](https://ieeexplore.ieee.org/document/10204079/)
[44](https://arxiv.org/abs/2406.15707)
[45](http://arxiv.org/pdf/2304.11842.pdf)
[46](https://arxiv.org/html/2402.04632)
[47](https://arxiv.org/pdf/2208.04717.pdf)
[48](https://arxiv.org/html/2501.02807v1)
[49](https://arxiv.org/abs/2109.07448)
[50](http://arxiv.org/pdf/2309.05028.pdf)
[51](https://openaccess.thecvf.com/content/CVPR2024/papers/Min_Entangled_View-Epipolar_Information_Aggregation_for_Generalizable_Neural_Radiance_Fields_CVPR_2024_paper.pdf)
[52](https://vclab.kaist.ac.kr/cvpr2023p3/2831_progressively_optimized_local_-Camera-ready%20PDF.pdf)
[53](https://www.sciencedirect.com/science/article/abs/pii/S0926580524000785)
[54](https://dl.acm.org/doi/10.1145/3581783.3612246)
