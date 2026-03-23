# Mip-NeRF: A Multiscale Representation for Anti-Aliasing Neural Radiance Fields

---

## 1. 핵심 주장 및 주요 기여 요약

**핵심 주장:** NeRF의 렌더링 절차는 픽셀당 하나의 무한히 가는 광선(ray)을 샘플링하기 때문에, 학습·테스트 이미지가 서로 다른 해상도(스케일)에서 장면을 관측할 경우 심각한 **앨리어싱(aliasing)** 및 **과도한 블러(blur)** 아티팩트가 발생한다. Mip-NeRF는 이 문제를 **원뿔(cone) 트레이싱**과 **통합 위치 인코딩(Integrated Positional Encoding, IPE)**을 통해 해결하며, 연속적인 스케일 공간에서 장면을 표현하는 **멀티스케일 프리필터링(prefiltering)** 방법을 제안한다.

**주요 기여:**
1. **Cone Tracing:** 픽셀당 광선(ray) 대신 원뿔(cone)을 캐스팅하고, 원뿔 절두체(conical frustum)를 다변량 가우시안으로 근사하여 효율적으로 처리
2. **Integrated Positional Encoding (IPE):** 포인트가 아닌 볼륨(영역)에 대한 위치 인코딩의 기대값을 닫힌 형식(closed form)으로 계산하여 안티앨리어싱된 특징 표현 제공
3. **단일 멀티스케일 MLP:** NeRF의 별도 "coarse"/"fine" 두 MLP를 하나의 MLP로 통합 → 파라미터 50% 감소, 속도 7% 향상
4. **성능:** 단일 스케일 Blender 데이터셋에서 NeRF 대비 평균 오류 17% 감소, 제안한 멀티스케일 데이터셋에서 60% 감소, 브루트 포스 슈퍼샘플링 NeRF와 동등 정확도 달성하면서 22배 빠름

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

NeRF는 각 픽셀에 대해 **하나의 무한히 가는 광선**을 캐스팅하고, 광선 위의 **점(point)**을 샘플링하여 위치 인코딩(Positional Encoding, PE)을 수행한다. 이는 다음 문제를 야기한다:

- **스케일 불감증(scale ambiguity):** 서로 다른 거리에서 동일 위치를 관측하는 두 카메라가 동일한 포인트 샘플 PE를 생성할 수 있어, MLP가 멀티스케일 정보를 구별 불가
- **앨리어싱:** 저해상도(원거리) 렌더링 시 고주파 정보가 올바르게 필터링되지 않아 계단 현상("jaggies") 발생
- **과도한 블러:** 멀티스케일 이미지로 훈련 시 고해상도 이미지도 블러 처리됨
- **슈퍼샘플링의 비효율성:** 픽셀당 여러 광선을 캐스팅하는 슈퍼샘플링은 NeRF의 수백 번 MLP 쿼리 구조에서 비실용적

### 2.2 제안하는 방법 (수식 포함)

#### (A) NeRF의 기본 구조 (Preliminaries)

광선 정의:

$$\mathbf{r}(t) = \mathbf{o} + t\mathbf{d}$$

위치 인코딩(PE):

```math
\gamma(\mathbf{x}) = \left[\sin(\mathbf{x}), \cos(\mathbf{x}), \ldots, \sin(2^{L-1}\mathbf{x}), \cos(2^{L-1}\mathbf{x})\right]^{\mathrm{T}}
```

MLP 쿼리:

$$\forall t_k \in \mathbf{t}, \quad [\tau_k, \, \mathbf{c}_k] = \mathrm{MLP}(\gamma(\mathbf{r}(t_k)); \, \Theta)$$

볼륨 렌더링:

$$\mathbf{C}(\mathbf{r}; \Theta, \mathbf{t}) = \sum_k T_k \left(1 - \exp(-\tau_k(t_{k+1} - t_k))\right) \mathbf{c}_k$$

$$T_k = \exp\left(-\sum_{k' < k} \tau_{k'}(t_{k'+1} - t_{k'})\right)$$

#### (B) Cone Tracing과 원뿔 절두체의 가우시안 근사

Mip-NeRF는 광선 대신 **원뿔(cone)**을 캐스팅한다. 원뿔을 연속적인 **원뿔 절두체(conical frustum)**로 분할하고, 각 절두체를 **다변량 가우시안**으로 근사한다.

절두체 내 균일 분포의 평균과 분산 ($t_\mu = (t_0+t_1)/2$, $t_\delta = (t_1-t_0)/2$):

$$\mu_t = t_\mu + \frac{2t_\mu t_\delta^2}{3t_\mu^2 + t_\delta^2}$$

$$\sigma_t^2 = \frac{t_\delta^2}{3} - \frac{4t_\delta^4(12t_\mu^2 - t_\delta^2)}{15(3t_\mu^2 + t_\delta^2)^2}$$

$$\sigma_r^2 = \dot{r}^2 \left(\frac{t_\mu^2}{4} + \frac{5t_\delta^2}{12} - \frac{4t_\delta^4}{15(3t_\mu^2 + t_\delta^2)}\right)$$

월드 좌표로 변환:

$$\boldsymbol{\mu} = \mathbf{o} + \mu_t \mathbf{d}, \quad \boldsymbol{\Sigma} = \sigma_t^2(\mathbf{d}\mathbf{d}^{\mathrm{T}}) + \sigma_r^2\left(\mathbf{I} - \frac{\mathbf{d}\mathbf{d}^{\mathrm{T}}}{\|\mathbf{d}\|_2^2}\right)$$

#### (C) Integrated Positional Encoding (IPE)

PE를 푸리에 특징으로 재작성:

```math
\mathbf{P} = \begin{bmatrix} 1 & 0 & 0 & 2 & 0 & 0 & \cdots & 2^{L-1} & 0 & 0 \\ 0 & 1 & 0 & 0 & 2 & 0 & \cdots & 0 & 2^{L-1} & 0 \\ 0 & 0 & 1 & 0 & 0 & 2 & \cdots & 0 & 0 & 2^{L-1} \end{bmatrix}^{\mathrm{T}}
```

```math
\gamma(\mathbf{x}) = \begin{bmatrix} \sin(\mathbf{P}\mathbf{x}) \\ \cos(\mathbf{P}\mathbf{x}) \end{bmatrix}
```

리프트된 가우시안의 평균과 공분산:

$$\boldsymbol{\mu}_\gamma = \mathbf{P}\boldsymbol{\mu}, \quad \boldsymbol{\Sigma}_\gamma = \mathbf{P}\boldsymbol{\Sigma}\mathbf{P}^{\mathrm{T}}$$

가우시안 분포 하에서 사인/코사인의 기대값 (닫힌 형식):

$$\mathbb{E}_{x \sim \mathcal{N}(\mu, \sigma^2)}[\sin(x)] = \sin(\mu)\exp\left(-\tfrac{1}{2}\sigma^2\right)$$

$$\mathbb{E}_{x \sim \mathcal{N}(\mu, \sigma^2)}[\cos(x)] = \cos(\mu)\exp\left(-\tfrac{1}{2}\sigma^2\right)$$

**최종 IPE 특징:**

$$\gamma(\boldsymbol{\mu}, \boldsymbol{\Sigma}) = \begin{bmatrix} \sin(\boldsymbol{\mu}_\gamma) \circ \exp\left(-\tfrac{1}{2}\text{diag}(\boldsymbol{\Sigma}_\gamma)\right) \\ \cos(\boldsymbol{\mu}_\gamma) \circ \exp\left(-\tfrac{1}{2}\text{diag}(\boldsymbol{\Sigma}_\gamma)\right) \end{bmatrix}$$

여기서 $\circ$는 원소별 곱셈이다. 핵심 인사이트: **IPE는 절두체보다 주기가 큰 주파수는 보존하고, 주기가 작은 (빠르게 진동하는) 주파수는 0으로 감쇠시킨다** → 자동 안티앨리어싱.

효율적 대각 계산:

$$\text{diag}(\boldsymbol{\Sigma}_\gamma) = \left[\text{diag}(\boldsymbol{\Sigma}),\, 4\text{diag}(\boldsymbol{\Sigma}),\, \ldots,\, 4^{L-1}\text{diag}(\boldsymbol{\Sigma})\right]^{\mathrm{T}}$$

$$\text{diag}(\boldsymbol{\Sigma}) = \sigma_t^2(\mathbf{d} \circ \mathbf{d}) + \sigma_r^2\left(\mathbf{1} - \frac{\mathbf{d} \circ \mathbf{d}}{\|\mathbf{d}\|_2^2}\right)$$

#### (D) 단일 멀티스케일 MLP와 최적화

NeRF의 두 MLP를 하나로 통합한 최적화 문제:

```math
\min_{\Theta} \sum_{\mathbf{r} \in \mathcal{R}} \left(\lambda \|\mathbf{C}^*(\mathbf{r}) - \mathbf{C}(\mathbf{r}; \Theta, \mathbf{t}^c)\|_2^2 + \|\mathbf{C}^*(\mathbf{r}) - \mathbf{C}(\mathbf{r}; \Theta, \mathbf{t}^f)\|_2^2\right)
```
$\lambda = 0.1$로 설정하여 coarse/fine 손실을 균형화한다.

계층적 샘플링 시 가중치 스무딩:

```math
w'_k = \frac{1}{2}\left(\max(w_{k-1}, w_k) + \max(w_k, w_{k+1})\right) + \alpha
```

( $\alpha = 0.01$ ) — "blurpool" 필터링으로 넓고 부드러운 상한 봉투를 생성.

### 2.3 모델 구조

| 구성 요소 | NeRF | Mip-NeRF |
|---------|------|----------|
| 캐스팅 방법 | 광선(Ray) | 원뿔(Cone) |
| 인코딩 | PE (포인트) | IPE (볼륨/가우시안) |
| MLP 수 | 2개 (coarse + fine) | 1개 (멀티스케일) |
| 파라미터 수 | 1,191K | 612K |
| 스케일 모델링 | 불가 | 연속 스케일 |
| 입력 | $\gamma(\mathbf{x})$ (점 위치) | $\gamma(\boldsymbol{\mu}, \boldsymbol{\Sigma})$ (가우시안) |

MLP 구조 자체는 NeRF와 동일한 아키텍처(8-layer, 256-dim)를 사용하되, 입력 특징만 PE에서 IPE로 교체. 밀도 $\tau$에 shifted softplus ( $\log(1+\exp(x-1))$ ), 색상 $\mathbf{c}$에 widened sigmoid 사용.

### 2.4 성능 향상

**멀티스케일 Blender 데이터셋 (본 논문 제안):**

| 모델 | Avg. Error ↓ | 학습 시간 | 파라미터 |
|------|------------|--------|--------|
| NeRF | 0.0288 | 3.05h | 1,191K |
| Mip-NeRF | **0.0114** | 2.84h | 612K |

→ **평균 오류 60% 감소**, 7% 빠른 학습, 50% 적은 파라미터

**단일 스케일 Blender 데이터셋:**

| 모델 | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Avg. ↓ |
|------|--------|--------|---------|--------|
| NeRF (Jax) | 31.74 | 0.953 | 0.050 | 0.0194 |
| Mip-NeRF | **33.09** | **0.961** | **0.043** | **0.0161** |

→ **평균 오류 17% 감소**

**슈퍼샘플링 대비:** Mip-NeRF는 브루트 포스 슈퍼샘플링 NeRF와 동등 정확도를 달성하면서 **22배 빠름** (2.48 vs 55.52 sec/MP).

### 2.5 한계

1. **Forward-facing 장면 (NDC 공간):** NDC 좌표에서는 픽셀의 공간적 지지가 거리에 따라 증가하지 않으므로, Mip-NeRF의 이점이 미미 (PSNR 26.843 vs 26.838)
2. **가우시안 근사의 부정확성:** 절두체의 상·하단 반지름 차이가 클 때 (예: 넓은 FOV의 매크로 촬영) 다변량 가우시안 근사가 부정확
3. **실시간 렌더링 미지원:** 여전히 MLP 기반으로 실시간 렌더링에는 부적합
4. **언바운드 장면:** 본 논문은 바운드된 합성 장면에서만 평가; 실외 대규모 장면에 대한 확장은 다루지 않음
5. **동적 장면:** 정적 장면만 고려하며, 동적/변형 가능한 장면으로의 확장이 직접 논의되지 않음

---

## 3. 모델의 일반화 성능 향상 가능성

Mip-NeRF의 설계는 여러 측면에서 **일반화 성능 향상**에 직접적으로 기여한다:

### 3.1 스케일 일반화
- **핵심 메커니즘:** IPE는 카메라의 기하학적 구조(초점 거리, 거리)에 따라 자동으로 주파수를 조절한다. 큰 절두체(원거리/저해상도)에서는 고주파가 감쇠되고, 작은 절두체(근거리/고해상도)에서는 고주파가 보존된다.
- **실증 결과:** 멀티스케일 데이터셋에서 NeRF 대비 60% 오류 감소가 이를 입증. 특히 1/8 해상도에서 PSNR이 22.533(NeRF) → 35.602(Mip-NeRF)로 대폭 향상.

### 3.2 하이퍼파라미터 $L$의 제거
- NeRF에서 PE의 차수 $L$은 인터폴레이션 커널의 대역폭을 결정하며, 장면에 따라 수동 튜닝이 필요하다. 
- Mip-NeRF의 IPE에서는 $L$을 매우 큰 값으로 설정해도 성능이 저하되지 않는다 (Figure 7). 이는 IPE가 가우시안 크기에 따라 "자체적으로 주파수를 튜닝"하기 때문이다.
- 이로써 **새로운 장면이나 카메라 설정에 대한 일반화가 별도 튜닝 없이 가능**해진다.

### 3.3 단일 MLP의 멀티스케일 학습
- NeRF의 두 MLP(coarse/fine)는 각각 단일 스케일만 학습하므로 스케일 간 정보 공유가 불가능했다.
- Mip-NeRF의 단일 MLP는 모든 스케일에서의 장면 정보를 공유 학습하여, **정규화(regularization) 효과**를 가진다. 이는 미세 구조(세밀한 디테일)와 거시 구조(전체적 형태)를 동시에 학습하게 하여 일반화에 유리하다.

### 3.4 연속 스케일 표현
- 전통적 mipmap은 이산적 스케일만 지원하지만, Mip-NeRF는 **연속적인 스케일** 공간을 모델링한다. 이는 학습 시 보지 못한 임의의 스케일/해상도에서도 렌더링이 가능함을 의미한다.

### 3.5 일반화의 한계와 향후 개선 방향
- **장면 유형:** 현재는 바운드된 객체 중심 장면에서만 검증. 언바운드/실외 장면에서의 일반화는 좌표 공간의 재설계(예: Mip-NeRF 360의 contracted coordinates)가 필요
- **카메라 모델:** 핀홀 카메라를 가정; 왜곡이 심한 렌즈(어안 등)에서는 원뿔 근사가 부정확
- **희소 뷰:** 소수의 학습 이미지에서의 일반화는 별도로 다루지 않음

---

## 4. 향후 연구에 미치는 영향 및 고려할 점

### 4.1 연구에 미친 영향

Mip-NeRF는 NeRF 생태계에서 **안티앨리어싱과 멀티스케일 표현**이라는 근본적 문제를 해결하며, 이후 연구의 기초가 되었다:

1. **Mip-NeRF 360 (Barron et al., CVPR 2022):** Mip-NeRF를 360도 언바운드 장면으로 확장. 비선형 장면 수축(contraction), 디스토션 기반 정규화, 효율적 온라인 학습 제안.
2. **Zip-NeRF (Barron et al., ICCV 2023):** Mip-NeRF 360의 안티앨리어싱과 Instant NGP의 그리드 기반 가속을 결합. 해시 기반 특징 그리드에서의 멀티스케일 표현 문제를 해결.
3. **3D Gaussian Splatting (Kerbl et al., SIGGRAPH 2023):** 암시적 가우시안 표현을 명시적으로 사용하여 실시간 렌더링 달성. Mip-NeRF의 가우시안 기반 영역 표현 철학과 유사한 직관 공유.
4. **Tri-MipRF (Hu et al., CVPR 2023):** 3D 밉맵 특징을 도입하여 Mip-NeRF의 아이디어를 가속화된 NeRF 프레임워크에 적용.

### 4.2 향후 연구 시 고려할 점

1. **계산 효율성:** MLP 기반 쿼리는 여전히 병목. 해시 그리드, 텐서 분해 등 가속 구조와의 결합이 필수
2. **언바운드/대규모 장면:** Mip-NeRF의 원뿔 근사는 바운드된 장면에 최적화되어 있으므로, 공간 수축(spatial contraction)이나 적응적 좌표계 필요
3. **동적 장면 확장:** 시간 축을 포함한 4D 절두체로의 확장 가능성
4. **일반화 가능 NeRF (Generalizable NeRF):** Mip-NeRF는 장면별 최적화가 필요. 피드 포워드 방식의 일반화 가능 모델과 IPE를 결합하는 연구 방향
5. **희소 뷰 재구성:** 소수 입력 뷰에서 IPE의 정규화 효과를 활용한 연구
6. **렌즈 모델 확장:** 비핀홀 카메라(어안, 파노라마 등)에서의 원뿔/절두체 근사 개선
7. **신호 처리 관점의 좌표 공간 분석:** 논문에서도 NDC 공간과 안티앨리어싱의 상호작용에서 시사점을 언급하며, 좌표 공간 설계가 NeRF 성능에 미치는 영향에 대한 체계적 분석 권장

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

| 논문 | 연도 | 핵심 아이디어 | Mip-NeRF와의 관계 | 주요 차이점 |
|------|------|------------|----------------|----------|
| **NeRF** (Mildenhall et al.) | 2020 | 신경 복사 필드, PE, 볼륨 렌더링 | Mip-NeRF의 기반 | 포인트 샘플링, 스케일 인식 없음 |
| **Mip-NeRF 360** (Barron et al.) | 2022 | 비선형 장면 수축, 디스토션 정규화 | Mip-NeRF의 직접 확장 | 언바운드 360도 장면 지원, 프로포절 네트워크 도입 |
| **Instant NGP** (Müller et al.) | 2022 | 멀티해상도 해시 인코딩, CUDA 가속 | 다른 접근: 해시 기반 특징 vs IPE | 수초 내 학습 가능하나, 멀티스케일 안티앨리어싱 미흡 |
| **Zip-NeRF** (Barron et al.) | 2023 | Mip-NeRF 360 + Instant NGP 결합 | Mip-NeRF의 후속 진화 | 해시 그리드에서의 안티앨리어싱 해결, 속도와 품질 동시 달성 |
| **3D Gaussian Splatting** (Kerbl et al.) | 2023 | 명시적 3D 가우시안, 래스터화 | 다른 패러다임: 명시적 표현 | 실시간 렌더링, 멀티스케일 처리는 별도 연구 필요 |
| **Tri-MipRF** (Hu et al.) | 2023 | 삼면 밉맵 특징 + Instant NGP | Mip-NeRF의 밉맵 아이디어를 그리드에 적용 | 그리드 기반 밉맵으로 훈련·렌더링 가속 |
| **Mip-Splatting** (Yu et al.) | 2024 | 3DGS에 멀티스케일 필터링 적용 | Mip-NeRF의 안티앨리어싱 철학을 가우시안 스플래팅에 적용 | 3D 스무딩 + 2D 밉 필터로 스케일 일관성 달성 |
| **NeuMIP** (Kuznetsov et al.) | 2021 | 멀티해상도 신경 재질(material) | Mip-NeRF와 유사한 프리필터링 동기 | 재질(texture) 표현에 특화, 장면 재구성이 아닌 렌더링 파이프라인용 |

### 핵심 트렌드 분석

1. **안티앨리어싱의 보편화:** Mip-NeRF 이후 거의 모든 고품질 NeRF 후속 연구가 멀티스케일 처리를 고려. 이는 Mip-NeRF가 문제를 명확히 정의하고 해결 방향을 제시했기 때문.

2. **IPE → 그리드 기반 안티앨리어싱으로의 전환:** IPE는 MLP 기반에 최적화되어 있으나, Instant NGP 이후 해시 그리드 기반 표현이 주류가 되면서 Zip-NeRF, Tri-MipRF 등이 그리드에서의 안티앨리어싱 방법을 제안.

3. **명시적 표현에서의 멀티스케일:** 3D Gaussian Splatting에서도 Mip-Splatting이 등장하며, Mip-NeRF의 "스케일 인식" 원칙이 명시적 표현으로도 확산.

4. **속도-품질 트레이드오프의 진화:** Mip-NeRF(수 시간) → Zip-NeRF(수십 분) → 3DGS(수 분)로 학습 속도가 급격히 개선되면서, 안티앨리어싱과 속도의 동시 달성이 핵심 과제로 부상.

---

## 참고자료

1. Barron, J. T., Mildenhall, B., Tancik, M., Hedman, P., Martin-Brualla, R., & Srinivasan, P. P. (2021). *Mip-NeRF: A Multiscale Representation for Anti-Aliasing Neural Radiance Fields.* ICCV 2021. arXiv:2103.13415v3 (본 분석의 주 논문)
2. Mildenhall, B., Srinivasan, P. P., Tancik, M., Barron, J. T., Ramamoorthi, R., & Ng, R. (2020). *NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis.* ECCV 2020.
3. Barron, J. T., Mildenhall, B., Verbin, D., Srinivasan, P. P., & Hedman, P. (2022). *Mip-NeRF 360: Unbounded Anti-Aliased Neural Radiance Fields.* CVPR 2022.
4. Barron, J. T., Mildenhall, B., Verbin, D., Srinivasan, P. P., & Hedman, P. (2023). *Zip-NeRF: Anti-Aliased Grid-Based Neural Radiance Fields.* ICCV 2023.
5. Müller, T., Evans, A., Schied, C., & Keller, A. (2022). *Instant Neural Graphics Primitives with a Multiresolution Hash Encoding.* ACM ToG (SIGGRAPH) 2022.
6. Kerbl, B., Kopanas, G., Leimkühler, T., & Drettakis, G. (2023). *3D Gaussian Splatting for Real-Time Radiance Field Rendering.* ACM ToG (SIGGRAPH) 2023.
7. Hu, W., Wang, Y., Ma, L., Yang, B., Gao, L., Liu, X., & Ma, Y. (2023). *Tri-MipRF: Tri-Mip Representation for Efficient Anti-Aliasing Neural Radiance Fields.* ICCV 2023.
8. Yu, Z., Chen, A., Huang, B., Sattler, T., & Geiger, A. (2024). *Mip-Splatting: Alias-Free 3D Gaussian Splatting.* CVPR 2024.
9. Tancik, M., Srinivasan, P. P., Mildenhall, B., et al. (2020). *Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains.* NeurIPS 2020.
10. Kuznetsov, A., Mullia, K., Xu, Z., Hašan, M., & Ramamoorthi, R. (2021). *NeuMIP: Multi-Resolution Neural Materials.* ACM ToG 2021.
