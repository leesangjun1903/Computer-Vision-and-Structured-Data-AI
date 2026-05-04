# 3DGUT: Enabling Distorted Cameras and Secondary Rays in Gaussian Splatting

3DGUT은 기존 3D Gaussian Splatting(3DGS)의 핵심 병목인 “Jacobian 기반 EWA 투영”을 제거하고, Unscented Transform(UT)을 이용해 **어떤 비선형 카메라 모델과 롤링 셔터에도 자연스럽게 일반화되면서도, 래스터라이제이션의 실시간 속도와 2차 광선(반사·굴절)까지 모두 지원하는** 것을 핵심 주장으로 한다.[^1][^2]
이를 통해 3DGRT·EVER 같은 레이 트레이싱 기반 방법과 비슷한 표현력을 유지하면서도, 3DGS 수준에 가까운 FPS로 왜곡 카메라와 복잡한 센서 효과를 처리하는 것이 주요 기여다.[^3][^1]

***

## 핵심 논문 및 주요 참고 자료

- **3DGUT: Enabling Distorted Cameras and Secondary Rays in Gaussian Splatting**
Qi Wu et al., arXiv 2412.12507, CVPR 2025 (Oral).[^2][^1]
3DGS의 EWA 투영을 Unscented Transform 기반 입자 근사로 대체하여 왜곡 카메라, 롤링 셔터, 2차 광선을 동시에 지원하는 3DGUT를 제안.[^4][^1]
- **3D Gaussian Splatting for Real-Time Radiance Field Rendering**
Bernhard Kerbl et al., SIGGRAPH 2023.[^5][^1]
실시간 뷰 합성을 위한 3DGS 기본 프레임워크를 제안하며, 3DGUT는 이 구조를 그대로 계승하되 투영과 평가 부분만 치환.[^1]
- **3D Gaussian Ray Tracing (3DGRT)**
Nicolas Moenne-Loccoz et al., TOG 2024.[^6][^1]
3D 가우시안 입자를 직접 레이 트레이싱하여 왜곡 카메라와 2차 광선을 자연스럽게 처리하지만, 속도가 3DGS보다 3–4배 느린 한계를 가짐.[^6][^1]
- **EVER: Exact Volumetric Ellipsoid Rendering**
Alexander Mai et al., arXiv 2410.01804.[^7][^3]
가우시안 대신 타원체를 정확 적분하는 레이 트레이싱 기반 방법으로, 왜곡 카메라·심도 흐림·2차 효과를 고품질로 지원하나 레이 트레이싱 비용이 큼.[^3]
- **Fisheye-GS: Lightweight and Extensible Gaussian Splatting Module for Fisheye Cameras**
Zimu Liao et al., arXiv 2409.04751.[^8][^1]
특정 어안 카메라 모델에 대해 Jacobian을 유도하여 3DGS를 확장하지만, 카메라 모델별 수작업 도출이 필요하고 극단적 왜곡에서 성능 저하.[^9][^1]
- **On the Error Analysis of 3D Gaussian Splatting and an Optimal Projection Strategy**
Letian Huang et al., 2024.[^10][^1]
EWA 기반 투영의 오차 특성을 이론적으로 분석하고, 왜곡이 커질수록 Jacobian 선형화 오차가 급격히 커짐을 보이며 3DGUT 같은 대안의 필요성을 지지.[^11][^1]
- **Unified Sensor Simulation for Autonomous Driving (XSIM)**
arXiv 2602.05617, 2026.[^12]
3DGUT을 확장해 자율주행용 카메라·LiDAR 센서 시뮬레이션을 통합적으로 처리하며, 구면 카메라(360°/LiDAR)에서의 3DGUT 한계를 분석·보완.[^12]
- **SimULi: Real-Time LiDAR and Camera Simulation with Unscented Transforms**
arXiv 2510.12901, 2025.[^13][^14]
3DGUT을 기반으로 임의 카메라 모델과 LiDAR를 동시에 실시간 렌더링하는 프레임워크로, 일반화 성능 관점에서 3DGUT의 확장성을 잘 보여줌.[^13]
- **Evaluating Fisheye-Compatible 3D Gaussian Splatting Methods on Real 200° Fisheye Images**
arXiv 2508.06968, 2025.[^15][^16]
200° 어안 카메라에서 Fisheye-GS와 3DGUT를 비교·분석하며, FOV 200°–160°–120°에서 3DGUT가 왜곡 증가에도 안정적인 품질을 유지함을 보고.[^15]

***

## 1. 3DGUT의 핵심 주장과 주요 기여

1) **핵심 주장**[^2][^1]

- 3DGS의 EWA(Elliptical Weighted Average) 기반 투영은 **비선형 카메라 모델(어안, 높은 왜곡, 롤링 셔터 등)에 근본적으로 취약**하며, 각 카메라 모델마다 Jacobian을 직접 유도해야 하는 “프레임워크 레벨의 비일반성”이 존재한다.[^10][^1]
- 이를 해결하기 위해, **투영 함수를 선형화하는 대신 가우시안 입자 자체를 Unscented Transform으로 근사**하면, 어떤 비선형 투영 함수 $g(\cdot)$에도 Jacobian 없이 정확히 적용할 수 있고, 롤링 셔터처럼 시간 의존 extrinsic도 자연스럽게 모델링할 수 있다.[^1][^2]

2) **주요 기여**[^2][^1]

- **UT 기반 3D Gaussian Unscented Transform(3DGUT)**: 3D 가우시안을 $2N+1$개의 시그마 포인트로 근사한 뒤, 각 포인트를 임의의 카메라 투영 $g(\cdot)$에 통과시켜 2D conic(평균·공분산)을 재추정하는 새로운 래스터라이제이션 공식을 도입.[^1]
- **3D 평가 및 정렬을 통한 3DGRT와의 정합**: 입자 반응을 2D가 아니라 3D 상에서 최대 응답점에서 평가하고, per-ray 정렬(MLAB k-buffer)을 사용하여 3DGRT와 공통의 3D 표현을 공유, **하이브리드 렌더링(기본 광선은 splatting, 2차 광선은 tracing)**을 가능하게 함.[^6][^1]
- **성능·일반화**: MipNeRF360·Tanks\&Temples에서는 3DGS에 근접한 PSNR/SSIM과 FPS(예: MipNeRF360에서 약 265 FPS vs 3DGS 347 FPS, 3DGRT 52 FPS)를 달성하면서, 어안·롤링 셔터 데이터셋(ScanNet++, Waymo)에서는 Fisheye-GS·3DGRT 등을 능가하는 품질을 기록.[^8][^1]

***

## 2. 해결하고자 하는 문제

### 2.1 기존 3DGS/EWA 투영의 한계

3DGS는 장면을 평균 $\mu\in\mathbb{R}^3$와 공분산 $\Sigma\in\mathbb{R}^{3\times 3}$를 갖는 3D 가우시안 입자 집합으로 표현한다.[^1]
각 입자의 3D 영역에서의 밀도는 다음과 같이 정의된다.[^1]

$$
\rho(x) = \exp\left(-\frac{1}{2}(x-\mu)^{\top}\Sigma^{-1}(x-\mu)\right) \quad (1)
$$

공분산은 회전 $R\in SO(3)$와 스케일 $S\in\mathbb{R}^{3\times 3}$로 분해된다.[^1]

$$
\Sigma = R S S^{\top} R^{\top} \quad (2)
$$

EWA 스플래팅에서는, 카메라 좌표계로 변환한 뒤 투영 함수 $g$를 1차 테일러 전개로 선형화하여 Jacobian $J$를 사용해 2D 공분산을 계산한다.[^1]

$$
\Sigma' = J_{[:2,:3]} W \Sigma W^{\top} J_{[:2,:3]}^{\top} \quad (3)
$$

이 때 $W$는 월드→카메라 변환이다.[^1]
문제는 다음과 같다.[^10][^1]

- **선형화 오차**: 왜곡이 커질수록 $g$의 높은 차수 항이 무시되어, 투영된 가우시안이 실제 분포와 크게 어긋나고, 이는 “blur·늘어짐·popping” 같은 아티팩트로 이어진다.[^10][^1]
- **카메라 모델별 Jacobian 필요**: 어안, 복합(radial+tangential) 왜곡, 롤링 셔터 등마다 새로운 $J$를 수작업으로 도출해야 하며, 이는 구현·유지보수·오차 분석 측면에서 매우 비효율적이다.[^8][^1]
- **시간 의존성(롤링 셔터) 미지원**: EWA는 하나의 정적 투영 함수 $g$를 가정하므로, 라인별로 extrinsic이 달라지는 롤링 셔터를 모델링하기 어렵다.[^7][^1]

이로 인해, 3DGS는 사실상 “이상적인 pinhole 카메라”에 묶여 있고, 왜곡 카메라를 다루기 위해서는 레이 트레이싱(3DGRT, EVER)로 넘어가야 했으며, 이는 실시간성을 크게 떨어뜨렸다.[^3][^6][^1]

***

## 3. 제안 방법: Unscented Transform 기반 투영

### 3.1 시그마 포인트 설계

3DGUT은 “투영 함수를 선형화하지 말고, **3D 가우시안 분포 자체를 UT로 근사**하자”는 관점에서 출발한다.[^2][^1]
차원이 $N=3$인 가우시안 $(\mu,\Sigma)$에 대해, $2N+1=7$개의 시그마 포인트 $\{x_i\}_{i=0}^{6}$를 다음과 같이 정의한다.[^1]

$$
x_0 = \mu
$$

$$
x_i = \mu + \sqrt{3+\lambda}\,\Sigma^{1/2}_{[i]},\;\; i=1,2,3
$$

$$
x_{i} = \mu - \sqrt{3+\lambda}\,\Sigma^{1/2}_{[i-3]},\;\; i=4,5,6 \quad (6)
$$

여기서 $\Sigma^{1/2}$는 $\Sigma$의 matrix square-root로, 실제 구현에서는 $R,S$ 분해를 이용해 얻는다.[^1]
UT 가중치는 평균과 공분산에 대해 서로 다르게 정의된다.[^1]

$$
w_0^\mu = \frac{\lambda}{3+\lambda},\;\; w_i^\mu = \frac{1}{2(3+\lambda)}\;(i\ge1) \quad (7)
$$

$$
w_0^\Sigma = \frac{\lambda}{3+\lambda} + (1-\alpha^2+\beta),\;\;
w_i^\Sigma = \frac{1}{2(3+\lambda)}\;(i\ge1) \quad (8)
$$

$\lambda=\alpha^2(3+\kappa)-3$이며, 논문에서는 $\alpha=1.0,\beta=2.0,\kappa=0.0$을 사용한다.[^1]

### 3.2 임의 비선형 투영으로의 매핑

각 시그마 포인트를 카메라 투영 함수 $g:\mathbb{R}^3\to\mathbb{R}^2$에 통과시켜 2D 포인트 $v_{x_i}$를 얻는다.[^1]

$$
v_{x_i} = g(x_i)
$$

이제 2D conic(평균·공분산)을 샘플 평균과 공분산으로 근사한다.[^1]

$$
v_\mu = \sum_{i=0}^{6} w_i^\mu v_{x_i} \quad (9)
$$

$$
\Sigma' = \sum_{i=0}^{6} w_i^\Sigma (v_{x_i}-v_\mu)(v_{x_i}-v_\mu)^{\top} \quad (10)
$$

중요한 점은, **$g$가 어떤 형태이든(복합 왜곡, 어안, 롤링 셔터 포함) UT 공식에는 전혀 손을 대지 않아도 된다는 것**이다.[^2][^1]
즉, 카메라 모델별 Jacobian 유도 없이도, 투영된 가우시안의 1차·2차 통계량을 직접적으로 근사할 수 있다.[^10][^1]

이렇게 얻은 $(v_\mu,\Sigma')$는 더 이상 정확한 2D 가우시안이 아닐 수 있지만, **타일링·컬링을 위한 가속구조**로 사용하기에는 충분한 근사이며, Monte Carlo 레퍼런스와의 KL divergence 분석에서도 EWA보다 일관되게 낮은 오차를 보인다.[^10][^1]

***

## 4. 제안 방법: 3D 파티클 응답 평가와 정렬

### 4.1 3D 상의 최대 응답점 평가

EWA 기반 3DGS는 2D conic 상에서 입자 응답 $\rho_i$를 평가한다.[^1]
3DGUT/3DGRT는 반대로, **카메라 광선 $r(\tau)=o+\tau d$** 상에서 가우시안 응답이 최대가 되는 지점 $\tau_{\max}$를 찾고, 그 지점에서 $\rho$를 평가한다.[^6][^1]

$$
\tau_{\max} = \arg\max_{\tau} \rho(o+\tau d)
$$

정규분포의 특성상, 폐형식 해는 다음과 같이 얻을 수 있다.[^1]

$$
\tau_{\max} = 
\frac{(\mu - o)^{\top}\Sigma^{-1} d}{d^{\top}\Sigma^{-1} d} \quad (11)
$$

이 지점에서의 밀도 $\rho(o+\tau_{\max}d)$를 이용해 opacity $\alpha_i$를 정의한다.[^1]

$$
\alpha_i = \sigma_i \rho_i(o+\tau_{\max} d)
$$

이는 기존 3DGS가 필요로 하던 “투영 함수 $g$”에 대한 역전파를 완전히 제거하여, **수치적 안정성을 높이고, 카메라 모델에 독립적인 3D 평가**를 가능하게 한다.[^1]

### 4.2 레이 정렬(MLAB)과 3DGRT와의 정합

볼륨 렌더링 식은 3DGS와 동일한 front-to-back 알파 합성 형식을 유지한다.[^1]

$$
c(o,d) = \sum_{i=1}^{N} c_i(d)\,\alpha_i \prod_{j=1}^{i-1} (1-\alpha_j) \quad (5)
$$

차이는 **입자들이 해당 레이 상에서 어떤 순서로 합성되는지**에 있다.[^1]

- 3DGRT: 레이 트레이싱 구조(예: OptiX)로부터 정확한 $\tau_{\max}$ 순서를 얻는다.[^6][^1]
- 3DGS: 타일 단위로 깊이 정렬을 해, 레이별 순서와 정확히 일치하지 않아 popping이 발생할 수 있다.[^1]
- 3DGUT(Ours(sorted)): MLAB(Multi-Layer Alpha Blending, k-buffer)를 이용해 각 레이마다 k개의 가장 먼 hit를 저장하고, 나머지는 누적 블렌딩 함으로써, **레이별 깊이 순서에 최대한 근접하는 근사**를 제공한다.[^1]

이 설계 덕분에, **동일한 가우시안 표현을 가지고도 “splatting(기본 광선)”과 “tracing(2차 광선)”을 자유롭게 혼합하는 하이브리드 렌더링**이 가능해진다.[^6][^1]

***

## 5. 모델 구조와 학습 파이프라인

3DGUT의 표현 자체는 3DGS와 동일하며, **다음 파라미터들을 갖는 입자 집합**으로 장면을 표현한다.[^1]

- 위치: $\mu_i \in \mathbb{R}^3$
- 공분산 분해: $R_i$ (quaternion), $S_i$ (scale) → $\Sigma_i = R_i S_i S_i^{\top} R_i^{\top}$[^1]
- 불투명도: $\sigma_i$
- 뷰 의존 색: 구면 고조파(SH) 기반 $\phi_{\beta}(d)$[^1]

학습 파이프라인은 3DGS/3DGRT를 그대로 계승한다.[^1]

- **Loss**: L2 재구성 손실 + SSIM 기반 perceptual loss

$$
L = L_2 + 0.2 L_{\text{SSIM}}
$$

로 30k iteration 동안 최적화.[^1]
- **Gradient**: 2D screen-space gradient를 사용할 수 없기 때문에, 3DGRT와 같이 3D 위치 gradient를 카메라 거리로 정규화한 값을 사용.[^6][^1]
- **Densification/Pruning**: 300 iteration마다 한 번씩 점밀화 및 가지치기를 수행.[^1]
- **UT 하이퍼파라미터**: $\alpha=1.0,\beta=2.0,\kappa=0.0$으로 고정.[^1]

결과적으로, **기존 3DGS 코드베이스에서 “EWA 기반 2D 가우시안 추정 함수만 UT 버전으로 교체하면 되는 drop-in 모듈”**로 설계되어 있다.[^1]

***

## 6. 성능 향상: 품질·속도·카메라 일반화

### 6.1 표준 NVS 데이터셋 (Pinhole 카메라)

MipNeRF360, Tanks\&Temples 같이 **표준 pinhole 카메라 데이터셋**에서는 3DGUT가 3DGS·StopThePop·3DGRT·EVER와 유사한 PSNR/SSIM을 보인다.[^1]

- MipNeRF360에서 3DGUT는 3DGS, 3DGRT, EVER와 PSNR 27–28 dB 수준으로 비슷하고,[^1]
- FPS는 3DGS(약 347 FPS)보다는 다소 낮지만 265 FPS 수준으로, 3DGRT(약 52 FPS)나 EVER(약 36 FPS) 대비 크게 빠르다.[^3][^1]

즉, **pinhole 환경에서는 “품질은 유지, 카메라 일반성 보너스”를 얻는 형태**로 이해할 수 있다.[^1]

### 6.2 왜곡 카메라 및 롤링 셔터

보다 중요한 것은, **왜곡 및 시간 의존 효과가 있는 카메라에서의 일반화 성능**이다.[^1]

- **Scannet++ (어안 카메라, equidistant fisheye)**
    - 3DGS: undistortion 후 학습/렌더링 → 정보 손실 구역이 커지고 under-observed 영역 발생.[^1]
    - Fisheye-GS: equidistant fisheye 모델에 대한 Jacobian을 도출하여 EWA 투영을 확장.[^8][^1]
    - 3DGUT: Jacobian 없이 UT 기반 투영만으로 fisheye를 직접 지원.
        - PSNR에서 Fisheye-GS의 28.15 dB 대비 29.11 dB, SSIM도 0.901 → 0.910으로 향상되며,[^1]
사용 가우시안 개수는 1.07M→0.38M로 절반 이하로 감소.[^1]
- **Waymo (왜곡 + 롤링 셔터)**
    - 3DGS: rectified 이미지에서만 학습 가능, 실제 센서 모델을 직접 표현하지 못함.[^1]
    - 3DGRT/3DGUT: 완전한 카메라 모델과 롤링 셔터를 사용.
    - PSNR: 3DGS 29.83, 3DGRT 29.99, 3DGUT 30.16으로 3DGUT가 소폭 우위.[^1]

또한, Monte Carlo 기반 참조 투영과의 KL-divergence를 분석한 결과, **fisheye·radial distortion·롤링 셔터가 결합된 환경에서도 UT 기반 투영은 median KL이 거의 변하지 않지만, EWA는 왜곡이 커질수록 KL이 급격히 증가**함이 보고된다.[^10][^1]

### 6.3 2차 광선과 하이브리드 렌더링

3DGUT는 3DGRT와 동일한 3D 파티클 평가·정렬 공식을 공유하므로, **기본 광선은 3DGUT 래스터라이제이션으로, 2차 광선(반사·굴절·그림자)은 3DGRT로 트레이싱하는 하이브리드 렌더링**을 구현한다.[^6][^1]

- 예: 굴절 유리, 반사 거울 표면에서 기본 광선이 가우시안 표현과 교차하는 위치까지 splatting으로 렌더링한 후, 그 지점에서 추가 광선을 쏘아 3DGRT로 처리.[^1]
- 이는 레이 트레이싱만으로 장면 전체를 렌더링하는 EVER/3DGRT 대비 **2차 광선이 필요한 영역에만 tracing 비용을 지불**하게 해 준다.[^3][^6][^1]

***

## 7. 한계점

논문에서 직접 언급하는 한계는 다음과 같다.[^1]

- **EWA 대비 소폭 느린 속도**: UT 투영과 3D 평가·MLAB 정렬의 추가 비용 때문에, 순수 3DGS보다는 약간 느리다(예: MipNeRF360에서 2.88 ms → 3.77 ms).[^1]
- **심한 왜곡에서 2D 가우시안 근사 한계**: UT는 시그마 포인트를 정확히 투영하지만, 투영된 분포가 더 이상 “정확한 2D 가우시안”이 아니어서, $\Sigma'$로 근사할 때 오차가 생기며, 극단적인 왜곡에서는 어떤 파티클이 어떤 픽셀에 기여하는지 결정하는 단계의 정확도가 떨어질 수 있다.[^15][^1]
- **한 점 샘플 평가의 한계**: 여전히 각 입자를 레이 상의 단일 최대 응답점에서만 평가하므로, EVER처럼 엘립소이드를 정확 적분하는 방식에 비해 중첩 입자·고빈도 구조 표현에 제한이 있다.[^3][^1]

후속 작업으로 EVER·3DGEER 같은 “정확 적분 + 효율” 방법이 등장하고 있고, 3DGUT 저자들도 이러한 방향을 유망한 미래 연구로 언급한다.[^17][^3][^1]

***

## 8. 일반화 성능 관점에서의 의미

### 8.1 카메라/센서 모델에 대한 구조적 일반화

3DGUT의 가장 큰 이점은, **카메라 모델이 바뀌어도 코드 구조가 바뀌지 않는다는 점**이다.[^2][^1]

- UT 공식은 카메라 모델과 무관하며, 투영 함수 $g$만 바꾸면 된다.
- 롤링 셔터의 경우, 각 시그마 포인트에 대해 서로 다른 extrinsic을 적용하면 되므로, 시간 의존 투영도 “함수 평가만 늘어나는 형태”로 자연스럽게 포함된다.[^1]

이는 Fisheye-GS처럼 카메라 모델마다 Jacobian을 유도해야 하는 접근과 대비된다.[^8][^1]
이론적으로, **UT는 비선형 변환 후 분포의 1·2차 모멘트를 2차까지 정확히(또는 근사적으로) 추적하는 기법**이므로, 카메라 모델이 복잡해질수록 EWA보다 더 안정적이고 보수적인 일반화 성능을 제공한다.[^18][^1]

### 8.2 실험적 일반화: 강한 왜곡 및 고 FoV

2025년 “Evaluating Fisheye-Compatible 3D Gaussian Splatting Methods…”에서는, 실제 200° 어안 카메라 데이터에서 Fisheye-GS와 3DGUT를 비교한 결과를 보고한다.[^16][^15]

- FOV 120°–160°–200°에 걸쳐 실험한 결과, **3DGUT는 FoV가 200°로 커져도 품질 저하가 상대적으로 작고, Fisheye-GS는 160° 부근에서 최적, 200°에서는 왜곡 증가로 품질이 떨어진다**고 보고된다.[^15]
- 이는 “특정 카메라 모델에 최적화된 Jacobian 기반 방법보다 UT 기반 일반 공식을 사용하는 것이 극단적 왜곡 상황에서 더 robust할 수 있다”는 근거를 제공한다.[^15][^1]


### 8.3 멀티 센서·자율주행으로의 확장

후속 연구 XSIM·SimULi는 3DGUT을 자율주행 센서 시뮬레이션 프레임워크로 확장하며, **카메라·LiDAR·구면 카메라까지 단일 UT 기반 표현으로 통합**한다.[^14][^12][^13]

- XSIM: 구면 카메라(예: 360° 카메라, LiDAR)의 경우, azimuth 경계에서 UT 투영이 위상 불연속을 유발하는 문제를 발견하고, 이를 완화하는 phase modeling 메커니즘을 도입.[^12]
- SimULi: 3DGUT을 확장하여 arbitrary spinning LiDAR 모델과 다양한 카메라 모델을 지원하며, factorized 가우시안 표현과 ray-based culling으로 기존 방법 대비 10–20배 빠른 렌더링을 달성.[^13]

이들 작업은 **3DGUT의 카메라·센서 일반화 능력이 실제 자율주행 시스템에 적용 가능한 수준**임을 보여 주며, 3DGUT이 단순 NVS를 넘어 “물리 기반 시뮬레이터의 백엔드”로 확장될 잠재력을 입증한다.[^12][^13]

***

## 9. 2020년 이후 관련 연구 비교 분석

아래 표는 카메라·광선 처리 관점에서 3DGUT와 주요 관련 연구를 요약 비교한 것이다.[^8][^3][^6][^1]


| 방법 | 유형 | 카메라 모델 | 롤링 셔터 | 2차 광선 | 속도 (상대) | 특징 |
| :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| NeRF (Mildenhall 2020) | MLP 기반 볼륨 렌더링 | 일반 카메라, 이론상 어떤 모델도 가능 | 이론상 가능하지만 구현 복잡 | 레이 마칭으로 가능 | 느림 | 추론이 느려 실시간성 부족, 대신 표현력 높음.[^1] |
| 3DGS (Kerbl 2023) | 가우시안 + EWA splatting | 사실상 pinhole에 최적화 | 직접 지원 X | 직접 지원 X | 매우 빠름 | Jacobian 기반 EWA 투영, 왜곡 카메라에서 선형화 오차 큼.[^1][^5] |
| StopThePop (Radl 2024) | 3DGS + 정렬 | pinhole | X | X | 매우 빠름 | popping 감소용 정렬·k-buffer 사용, 카메라 모델은 3DGS와 동일.[^1] |
| Fisheye-GS (Liao 2024) | 3DGS 확장 | 특정 fisheye 모델 | 제한적 | X | 빠름 | 해당 fisheye 모델에 대해 Jacobian 유도, 극단적 왜곡·타 모델은 취약.[^1][^8][^9] |
| 3DGRT (Moenne-Loccoz 2024) | 가우시안 레이 트레이싱 | 임의 비선형 카메라 | O | O | 3DGS 대비 3–4배 느림 | 왜곡·2차 광선 지원, 레이 트레이싱 하드웨어 필요.[^1][^6] |
| EVER (Mai 2024) | 타원체 정확 적분 | 임의 비선형 카메라 | O | O | 중간 | 정확 볼륨 렌더링으로 품질 우수, 레이 트레이싱 기반.[^3][^7] |
| **3DGUT (Wu 2024)** | 3DGS + UT + 3D 평가 | 임의 비선형 카메라 | O | O (하이브리드) | 3DGS보다 약간 느림, 3DGRT보다 훨씬 빠름 | Jacobian 없이 UT로 투영, 3DGRT와 표현 일치, fisheye·롤링 셔터에서 SOTA.[^1][^2] |
| XSIM (2026) | 3DGUT 확장 | 자동차용 복합 카메라 | O | O | 실시간 | 롤링 셔터·왜곡·구면 카메라를 단일 프레임워크로 다루는 센서 시뮬레이터.[^12] |
| SimULi (2025) | 3DGUT 기반 멀티 센서 | 임의 카메라 + LiDAR | O | O | 실시간 | LiDAR + 카메라 통합 시뮬레이션, 3DGUT의 센서 일반화 사례.[^13][^14] |

3DGUT은 **“레이 트레이싱 수준의 일반성”과 “3DGS 수준의 속도”를 절충하는 지점**에 있으며, 이후 XSIM·SimULi·fisheye 평가 논문들이 이를 다양한 응용으로 확장하고 있다.[^12][^15][^1]

***

## 10. 앞으로의 연구에 미치는 영향과 고려할 점

### 10.1 연구 방향에 미치는 영향

1) **왜곡·롤링 셔터가 기본값이 되는 NVS/재구성 세팅**
3DGUT과 후속 연구들은 “pinhole만 고려한 NVS 벤치마크”에서 “현실적 센서 모델(어안, 광각, 롤링 셔터, 구면 카메라)”로 연구 초점을 옮기는 계기를 제공한다.[^19][^1]
특히 자율주행·로보틱스에서는 이런 카메라들이 사실상 표준이므로, 향후 NVS·SLAM·BEV perception 연구에서도 3DGUT류 표현이 기본 빌딩 블록으로 채택될 가능성이 크다.[^20][^12]
2) **멀티 센서 시뮬레이션과 데이터 생성**
XSIM·SimULi처럼, 3DGUT 기반 표현을 이용해 카메라·LiDAR·레이더 등 여러 센서를 동시에 시뮬레이션하는 연구가 활발해지고 있다.[^14][^13][^12]
이는 합성 데이터 생성, 안전 검증, 도메인 랜덤화 등에서 **일반화된 센서 모델이 필수임**을 보여 주며, 후속 연구가 센서 특성(노이즈, 응답 곡선, 동적 범위 등)까지 더 정교하게 모델링하도록 유도한다.[^12]
3) **Inverse Rendering·Relighting으로의 확장**
3DGUT은 3DGRT와 렌더링 공식을 공유하므로, 2차 광선까지 포함한 역문제(예: 재질 재구성, 조명 추정, inverse rendering)에 그대로 활용 가능하다.[^6][^1]
앞으로는 **UT 기반 래스터라이제이션 + 레이 트레이싱 결합 구조** 위에서 재질·조명·기하를 동시에 최적화하는 연구가 자연스럽게 등장할 것으로 보인다.[^17][^1]

### 10.2 향후 연구 시 고려할 점

연구자로서 3DGUT을 기반으로 새로운 작업을 설계할 때 고려할 사항은 다음과 같다.[^15][^10][^1]

- **정확도 vs 효율 트레이드오프**
    - UT 기반 투영과 단일 점 평가(최대 응답점)는 EWA·정확 적분 대비 “좋은 근사”지만, 극단적 왜곡·강한 중첩·복잡한 광학 효과에서는 여전히 한계가 있다.[^1]
    - EVER·3DGEER처럼 정밀한 적분을 도입하거나, UT의 시그마 포인트 수를 늘리는 방향은 정확도를 높이는 대신 효율을 떨어뜨리므로, 응용에 맞는 균형을 설계해야 한다.[^17][^3]
- **극단적 카메라 모델(200°+, 구면 카메라)에 대한 주의**
    - fisheye 200° 이상의 FoV나 구면 카메라에서는 UT 투영이 위상 불연속·경계 문제를 일으킬 수 있으며, XSIM에서처럼 phase modeling·토폴로지 인식이 필요하다.[^12][^15]
    - 이런 환경에서 새 방법을 제안할 때는, Monte Carlo 레퍼런스와의 KL 분석 등으로 **투영 근사 품질을 정량 평가**하는 것이 중요하다.[^10][^1]
- **학습 안정성과 최적화 전략**
    - 3DGS/3DGRT에서 쓰이는 densification·pruning·gradient 스케일링 전략은 UT 기반 투영과도 잘 맞지만, 카메라 모델에 따라 gradient 분포가 달라질 수 있다.[^1]
    - 롤링 셔터·동적 장면을 함께 다룰 경우, 시간 차원까지 포함한 입자 파라미터화(예: 시간별 opacity/위치)를 어떻게 설계할지, regularization을 어떻게 줄지에 대한 추가 연구가 필요하다.[^21][^12]
- **일반화 평가 프로토콜의 확장**
    - 3DGUT 후속 연구들은 대부분 fisheye·자율주행·구면 센서 등 “현실적인 난이도”를 가진 데이터셋 위에서 평가를 수행한다.[^15][^12][^1]
    - 새로운 방법을 제안할 때도, pinhole-only 벤치마크에 그치지 않고, **왜곡·롤링 셔터·멀티 센서·고 FoV 환경까지 포함한 평가 프로토콜**을 설계하는 것이 향후 일반화 성능 연구의 핵심이 될 것이다.[^22][^19]

요약하면, 3DGUT은 “투영 함수 대신 입자 분포를 근사한다”는 매우 단순한 아이디어로 카메라·센서 일반화 문제를 우아하게 해결했고, 이후 여러 후속 연구가 이를 다양한 센서·응용으로 확장하고 있다.[^4][^1]
향후 연구에서는 이 틀 안에서 **정확도·효율·표현력(정확 적분, 복잡 광학, 동적 장면)을 어떻게 균형 있게 확장할 것인지**가 핵심 연구 방향이 될 것이다.[^17][^1]
<span style="display:none">[^23][^24][^25][^26][^27][^28][^29][^30][^31][^32][^33][^34]</span>

<div align="center">⁂</div>

[^1]: 2412.12507v2.pdf

[^2]: https://arxiv.org/html/2412.12507v2

[^3]: https://arxiv.org/abs/2410.01804

[^4]: https://www.emergentmind.com/papers/2412.12507

[^5]: https://xoft.tistory.com/51

[^6]: https://github.com/nv-tlabs/3dgrut

[^7]: https://half-potato.gitlab.io/ever

[^8]: https://arxiv.org/abs/2409.04751

[^9]: https://github.com/zmliao/fisheye-gs

[^10]: https://www.semanticscholar.org/paper/On-the-Error-Analysis-of-3D-Gaussian-Splatting-and-Huang-Bai/ae002bbeeb8cdec0013bd4555b27542fc9ea5be2

[^11]: https://www.emergentmind.com/topics/fisheye-based-3d-gaussian-splatting

[^12]: https://arxiv.org/abs/2602.05617

[^13]: https://arxiv.org/abs/2510.12901

[^14]: https://arxiv.org/html/2510.12901v1

[^15]: https://arxiv.org/html/2508.06968v1

[^16]: https://arxiv.org/pdf/2508.06968.pdf

[^17]: https://arxiv.org/html/2505.24053v3

[^18]: https://arxiv.org/html/2604.00648v1

[^19]: https://arxiv.org/html/2508.06968v2

[^20]: https://arxiv.org/html/2511.17210v1

[^21]: https://linkinghub.elsevier.com/retrieve/pii/S1077314210001669

[^22]: https://arxiv.org/abs/2508.06968

[^23]: https://ieeexplore.ieee.org/document/11094280/

[^24]: https://www.mdpi.com/1424-8220/22/6/2388

[^25]: http://ieeexplore.ieee.org/document/6553816/

[^26]: https://www.semanticscholar.org/paper/5f9e6f2f84a6a9a64b1d5868e2782b4bae82b567

[^27]: http://ieeexplore.ieee.org/document/4027067/

[^28]: https://arxiv.org/abs/2412.12507

[^29]: https://arxiv.org/html/2412.12507v1

[^30]: https://arxiv.org/html/2409.04751v2

[^31]: https://arxiv.org/html/2411.15355v3

[^32]: https://research.nvidia.com/labs/toronto-ai/3DGUT/

[^33]: https://cvpr.thecvf.com/virtual/2025/poster/33729

[^34]: https://kimjy99.github.io/논문리뷰/3dgut/

