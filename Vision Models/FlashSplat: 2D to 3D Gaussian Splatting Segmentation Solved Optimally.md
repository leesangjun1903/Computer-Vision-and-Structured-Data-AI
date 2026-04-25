# FlashSplat: 2D to 3D Gaussian Splatting Segmentation Solved Optimally

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

FlashSplat은 3D Gaussian Splatting(3D-GS) 장면에서 2D 마스크를 3D 공간으로 "lifting"하는 세그멘테이션 문제를, **전역 최적(globally optimal) 선형 계획법(Linear Programming)**으로 단일 단계에 풀 수 있음을 보입니다. 핵심 통찰은 다음과 같습니다:

> "재구성된 3D-GS 장면에서, 2D 마스크의 렌더링은 각 Gaussian의 레이블에 대해 **선형 함수**이다."

이로 인해 반복적 경사 하강법 없이 closed-form으로 최적 레이블 할당이 가능합니다.

### 주요 기여 (5가지)

| 기여 | 내용 |
|------|------|
| ① 전역 최적 솔버 | 2D→3D 세그멘테이션을 ILP로 정식화, closed-form 해 도출 |
| ② 선형화 | 3D-GS 렌더링 과정을 선형화하여 이진/장면 세그멘테이션 모두 지원 |
| ③ Background Bias | 노이즈에 강인한 정규화된 최적 할당 도입 |
| ④ 속도 향상 | 30초 이내 완료, 기존 대비 약 50× 가속 |
| ⑤ 다운스트림 성능 | 객체 제거(Object Removal) 및 인페인팅(Inpainting) 우수 성능 |

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

**기존 방법의 문제점:**
- SAGA, Gaussian Grouping 등 기존 방법은 각 Gaussian에 레이블을 부여하기 위해 **30,000회 이상의 반복적 경사 하강법**을 사용
- 수렴 속도가 느리고, 지역 최적해(local optimum)에 빠질 위험
- 추가 학습 비용이 크고 실시간 응용에 부적합
- SAGS는 학습-프리(training-free)이지만, Gaussian 중심의 단순 투영에 기반하여 정확도가 낮음

**FlashSplat이 해결하려는 것:**
- 2D 마스크 집합 $\{M^v\}$로부터 각 3D Gaussian에 최적 레이블 $P_i$를 **단 한 번의 최적화**로 전역 최적으로 할당

---

### 2.2 제안 방법 및 수식

#### (1) 3DGS 렌더링의 알파 블렌딩

3D-GS의 렌더링은 다음 알파 컴포지션으로 정의됩니다:

$$X = \sum_{i \in \{G_i\}_B} x_i \alpha_i \prod_{j=1}^{i-1}(1 - \alpha_j) = \sum_{i \in \{G_i\}_B} x_i \alpha_i T_i $$

여기서:
- $\alpha_i$: 픽셀에서의 알파값 (불투명도 $o_i$와 2D Gaussian 확률의 곱)
- $T_i = \prod_{j=1}^{i-1}(1 - \alpha_j)$: 투과율 (앞선 $i-1$개 Gaussian을 통과한 비율)

**핵심 관찰**: 장면이 재구성되면 $\alpha_i$와 $T_i$는 **상수**가 됩니다. 따라서 레이블 $P_i$를 $x_i$로 대입하면 렌더링 함수는 $P_i$에 대해 **선형**이 됩니다.

#### (2) 이진 세그멘테이션: ILP 정식화

$$\min_{\{P_i\}} \quad \mathcal{F} = \sum_{v \in L} \sum_{M_{jk}^v \in M^v} \left| \sum_i P_i \alpha_i T_i - M_{jk}^v \right|$$

$$\text{subject to} \quad P_i \in \{0, 1\} $$

#### (3) Lemma 1 (알파 블렌딩의 경계)

$$0 \leq \sum P_i \alpha_i T_i \leq \sum \alpha_i T_i \leq 1 $$

이 성질을 이용해 목적함수를 전개하면:

$$\min \mathcal{F} = \sum_{v,i,j,k} P_i \alpha_i T_i \mathbb{I}(M_{jk}^v, 0) - \sum_{v,i,j,k} P_i \alpha_i T_i \mathbb{I}(M_{jk}^v, 1) + C $$

$$= C + \sum_i P_i (A_0^i - A_1^i) $$

여기서 $C = \sum_{v,j,k} M_{jk}^v$는 상수이며, $\mathbb{I}(\cdot, 0)$, $\mathbb{I}(\cdot, 1)$은 배경/전경 지시 함수입니다.

#### (4) 최적 할당: Majority Vote

$$P_i = \arg\max_n A_n, \quad n \in \{0, 1\}$$

$$\text{where} \quad A_n = \sum_{v,j,k} \alpha_i T_i \mathbb{I}(M_{jk}^v, n) $$

**직관**: 각 Gaussian이 전경/배경에 기여하는 가중합을 비교하여 다수결로 레이블 부여.

#### (5) 정규화된 ILP: Background Bias (노이즈 감소)

L1 정규화 후 배경 편향 $\gamma \in [-1, 1]$ 도입:

$$\bar{A}_e = \frac{A_e}{\sum_t A_t}, \quad \hat{A}_0 = \bar{A}_0 + \gamma$$

$$P_i = \arg\max_n \{\hat{A}_0, \bar{A}_1\}$$

- $\gamma > 0$: 전경 노이즈 감소 (배경 우선)
- $\gamma < 0$: 배경 노이즈 감소 (전경 확장)

#### (6) 씬 세그멘테이션 (다중 인스턴스 확장)

$$P_i = \arg\max_n A_n, \quad n \in \{0, t\}$$

$$A_t = \sum_{v,j,k} \alpha_i T_i \mathbb{I}(M_{jk}^v, t), \quad A_0 = A_{others} = \sum_{e \neq t} \sum_{v,j,k} \alpha_i T_i \mathbb{I}(M_{jk}^v, e) $$

기여 집합 $\{A_e\}$를 한 번만 누적하고 $\arg\max$로 각 객체의 Gaussian 부분집합 $\{G_i\}_e$를 결정합니다.

---

### 2.3 모델 구조

```
[입력] 재구성된 3D-GS 장면 {Gi} + 2D 마스크 집합 {M^v}
         ↓
[CUDA 커널] 타일 기반 래스터화로 αiTi 계산
         ↓
[기여 누적] Ae 행렬 (E × |{Gi}|) 구성
         ↓
[최적 할당] argmax + background bias γ 적용
         ↓
[출력] 각 Gaussian의 레이블 Pi / 씬 세그멘테이션 행렬 S
         ↓
[선택적] 깊이 유도 새로운 뷰 마스크 렌더링
```

**구현 세부:**
- CUDA 커널로 $A_e$ 계산: ~26초
- 이진 세그멘테이션 argmax: ~0.4ms
- $\gamma$ 조정은 $A_e$ 계산 후 1ms 이내 인터랙티브 조정 가능
- 피크 GPU 메모리: 8G (SAGA의 15G 대비 절반)

---

### 2.4 성능 향상

#### 정량적 결과 (NVOS 데이터셋)

| 방법 | mIOU (%) ↑ | mAcc (%) ↑ |
|------|-----------|-----------|
| NVOS | 39.4 | 73.6 |
| ISRF | 70.1 | 92.0 |
| SGISRF | 83.8 | 96.4 |
| SA3D | 90.3 | 98.2 |
| SAGA | 90.9 | 98.3 |
| **FlashSplat (ours)** | **91.8** | **98.6** |

#### 계산 비용 비교 (Figurines 씬, NVIDIA A6000)

| 방법 | 추가 학습 시간 | 최적화 스텝 | 세그멘테이션 시간 | 피크 메모리 |
|------|-------------|-----------|----------------|-----------|
| SAGA | 18분 | 30,000 | 0.5초 | 15G |
| Gaussian Grouping | 37분 | 30,000 | 0.3초 | 34G |
| **FlashSplat** | **26초** | **1** | **0.4ms** | **8G** |

#### Background Bias 효과 (Truck 씬)

| $\gamma$ | -0.8 | -0.4 | 0 | 0.4 | 0.8 |
|---------|------|------|---|-----|-----|
| mIoU | 82.4 | 89.6 | 92.3 | **94.2** | 93.8 |

---

### 2.5 한계점

1. **확장성 문제**: 대규모 장면에서 모든 마스크 픽셀을 순회해야 하므로 공간 해상도가 매우 클 경우 계산 부담 증가
2. **기하학적 모호성**: 깊이 유도 새로운 뷰 마스크 렌더링 시, 3D-GS 재구성에 기하학적 감독(supervision)이 없어 모호한 마스크 발생 가능
3. **Gaussian 공유 문제**: 단일 Gaussian이 여러 객체에 기여 가능 (Counter 씬에서 약 20%의 Gaussian이 2개 이상 객체에 공유됨)
4. **2D 마스크 품질 의존성**: 입력 2D 마스크(SAM 예측)의 품질에 따라 결과가 달라짐. $\gamma$로 완화 가능하나 완전 제거 불가
5. **폐색 처리 한계**: 전향(facing-forward) 뷰만 있는 씬(예: Horns)에서 큰 객체 제거 후 배경 아티팩트 발생

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 현재 일반화 강점

**① 적은 뷰로도 동작 (Data Efficiency)**

반복 최적화가 없으므로 전체 뷰의 약 10%(1/8 수준)로도 합리적인 세그멘테이션 결과 도출:

$$\text{필요 뷰 수}: \frac{1}{8} \text{ of total views} \rightarrow \text{decent segmentation}$$

이는 **희소 뷰(sparse view) 환경**에서의 일반화를 지원합니다.

**② 노이즈에 강인한 설계**

$\gamma$ 파라미터를 통한 background bias는 SAM이 생성한 노이즈가 있는 2D 마스크에 대해 강인성을 보입니다. 실험에서 3D 재구성이 2D 마스크의 깨진 영역(broken regions)을 자동 복원하는 효과가 확인되었습니다.

**③ 다양한 장면 유형 지원**

MIP-360 (360도 장면), T&T (대규모 야외), LLFF (전향 뷰), Instruct-NeRF2NeRF (편집 씬), LERF (언어 임베딩 씬) 등 다양한 데이터셋에서 검증.

**④ 학습-프리(Training-Free) 특성**

추가 학습 없이 재구성된 3D-GS만으로 작동하므로, **새로운 장면 유형에 바로 적용 가능**합니다. 도메인 특화 학습이 필요 없어 일반화 비용이 낮습니다.

**⑤ 이진 및 씬 세그멘테이션 통합**

동일한 수학적 프레임워크(ILP)로 이진 세그멘테이션과 다중 인스턴스 씬 세그멘테이션을 모두 처리합니다.

### 3.2 일반화 성능 향상을 위한 가능성

**① 기하학적 감독 통합**

현재 3D-GS 재구성은 뷰 공간 감독만 사용합니다. 깊이 맵, 법선 벡터 등 **명시적 기하학적 감독**을 추가하면 새로운 뷰에서의 마스크 렌더링 정확도가 향상될 것입니다.

$$\mathcal{L}_{total} = \mathcal{L}_{render} + \lambda \mathcal{L}_{geometry}$$

**② SAM2 또는 고품질 2D 세그멘테이션 모델과 결합**

2024년 발표된 SAM2(비디오 세그멘테이션 지원)를 활용하면 뷰 간 마스크 연관(association)의 품질이 향상되어, FlashSplat의 입력 품질이 개선됩니다.

**③ 동적 장면으로의 확장**

4D Gaussian Splatting (Dynamic 3DGS)에서도 시간 $t$에 따른 $\alpha_i(t)$, $T_i(t)$가 상수화 가능하다면 동일한 ILP 프레임워크 적용 가능:

$$A_n(t) = \sum_{v,j,k} \alpha_i(t) T_i(t) \mathbb{I}(M_{jk}^v, n)$$

**④ 적응적 $\gamma$ 설정**

현재 $\gamma$는 수동 설정입니다. 마스크 노이즈 수준을 자동으로 추정하여 $\gamma$를 적응적으로 결정하는 방법이 일반화 성능을 높일 수 있습니다.

**⑤ 시맨틱 언어 쿼리와의 결합**

LERF(Language Embedded Radiance Fields)처럼 언어 특징을 3D-GS에 임베딩한 경우, FlashSplat의 프레임워크에 언어 기반 마스크 생성을 연계하면 **개방형 어휘(open-vocabulary) 3D 세그멘테이션**으로 일반화 가능합니다.

---

## 4. 2020년 이후 관련 최신 연구 비교 분석

### 4.1 NeRF 기반 3D 세그멘테이션 연구

| 연구 | 연도 | 방법 | mIOU | 특징 |
|------|------|------|------|------|
| Semantic-NeRF | 2021 | NeRF + 시맨틱 레이블 학습 | - | 최초 NeRF 시맨틱 전파 |
| NVOS | 2022 | Neural Volumetric Object Selection | 39.4% | MLP 학습, 사용자 상호작용 |
| ISRF | 2022 | Interactive Segmentation of Radiance Fields | 70.1% | 2D 비전 모델 특징 증류 |
| SGISRF | 2023 | Scene-Generalizable Interactive Segmentation | 83.8% | 장면 일반화 강조 |
| SA3D | 2023 | Segment Anything in 3D with NeRFs | 90.3% | SAM + NeRF 결합 |

### 4.2 3D-GS 기반 세그멘테이션 연구

| 연구 | 연도 | 방법 | 최적화 시간 | 특징 |
|------|------|------|-----------|------|
| SAGA | 2023 | 2D 마스크 특징 증류 + 3D-GS | 18분 | 추가 특징 학습 필요 |
| Gaussian Grouping | 2023 | 비디오 트래커 + 식별 손실 | 37분 | 분류기 학습 필요 |
| SAGS | 2024 | Gaussian 중심 투영 (학습-프리) | ~수초 | 단순하나 정확도 낮음 |
| **FlashSplat** | **2024** | **ILP + closed-form** | **26초** | **전역 최적, 학습-프리** |

### 4.3 핵심 차별점 분석

```
특징 학습 기반 (SAGA, Gaussian Grouping):
  장점: 풍부한 특징 표현
  단점: 긴 학습 시간, 지역 최적 위험, 높은 메모리

투영 기반 (SAGS):
  장점: 학습-프리, 빠름
  단점: 3D Gaussian의 공간적 특성 무시, 정확도 낮음

FlashSplat (ILP 기반):
  장점: 전역 최적 보장, 학습-프리, 초고속 (26초)
  단점: 기하학적 모호성, 대규모 장면 확장성
```

---

## 5. 앞으로의 연구에 미치는 영향 및 고려사항

### 5.1 연구에 미치는 영향

**① 최적화 패러다임의 전환**

FlashSplat은 3D 장면 이해 분야에서 "반복 경사 하강법"에서 "닫힌형 선형 최적화"로의 패러다임 전환을 보여줍니다. 이는 3D-GS의 알파 블렌딩 구조가 갖는 선형성이라는 **수학적 성질을 명시적으로 활용**한 최초의 시도로, 유사한 접근법이 다른 3D 표현(예: 4D-GS, 동적 장면)에도 적용될 수 있음을 시사합니다.

**② 실시간 3D 장면 편집의 가능성**

30초 이내 세그멘테이션 완료 + 1ms argmax로, **인터랙티브 3D 장면 편집** (AR/VR, 로보틱스 조작)의 실용적 기반이 마련됩니다.

**③ 하위 작업(downstream task)에서의 파급 효과**

객체 제거, 인페인팅, 장면 편집 등 다양한 응용에서 세그멘테이션 품질이 직접 영향을 미치므로, FlashSplat의 접근법은 3D 콘텐츠 생성 파이프라인 전반에 영향을 줄 것입니다.

**④ 학습-프리 방법론의 재조명**

추가 학습 없이도 최고 수준의 성능을 달성함으로써, 3D 이해 연구에서 "모델 학습"의 필요성을 재검토하게 합니다.

### 5.2 앞으로 연구 시 고려할 점

**① 기하학적 감독 부재 문제 해결**

3D-GS 자체가 기하학적으로 정확하지 않으므로(뷰 공간 감독만 사용), 향후 연구에서는:
- 깊이 정규화(depth regularization)
- 법선 일관성(normal consistency) 손실
- MVS(Multi-View Stereo)와의 결합

등을 고려해야 합니다.

**② 동적/4D 장면으로의 확장**

$$A_n(t) = \sum_{v,j,k} \alpha_i(t) T_i(t) \mathbb{I}(M_{jk}^v, n)$$

시간 $t$에 따른 기여도 변화를 어떻게 처리할지, 그리고 변형 필드(deformation field)가 있는 경우 선형성이 유지되는지 검토 필요.

**③ 대규모 장면에서의 효율성**

현재 구현은 모든 마스크 픽셀을 순회하므로, 대규모 실외 장면(수백만 Gaussian)에서의 계산량이 문제가 될 수 있습니다. **계층적 접근법** 또는 **적응적 타일링**이 필요합니다.

**④ 2D 마스크 품질과의 연동**

FlashSplat의 성능은 SAM이 생성하는 2D 마스크 품질에 의존합니다. SAM2, Grounded SAM 등 더 강력한 2D 세그멘테이션 모델과의 통합, 그리고 마스크 노이즈 수준에 따른 적응적 $\gamma$ 선택이 중요한 연구 방향입니다.

**⑤ 언어 기반 세그멘테이션과의 통합**

LERF 등 언어 임베딩 방법과 결합하면:
$$P_i = \arg\max_n \{A_n^{visual} + \lambda A_n^{language}\}$$
형태의 멀티모달 최적 할당이 가능하여, 개방형 어휘 3D 세그멘테이션으로 일반화될 수 있습니다.

**⑥ 불확실성 정량화**

현재 방법은 결정론적 할당($\arg\max$)만 수행합니다. 향후 연구에서 각 Gaussian의 레이블 확신도(confidence)를 $A_n$의 분포로 정량화하면, 불확실한 경계 영역을 더 잘 처리할 수 있습니다:

$$\text{confidence}_i = \frac{|A_0^i - A_1^i|}{\sum_n A_n^i}$$

---

## 참고자료 (출처)

- **주요 논문**: Qiuhong Shen, Xingyi Yang, Xinchao Wang. "FlashSplat: 2D to 3D Gaussian Splatting Segmentation Solved Optimally." arXiv:2409.08270v1 [cs.CV], 12 Sep 2024. https://arxiv.org/abs/2409.08270
- **3D Gaussian Splatting**: Kerbl, B., et al. "3D Gaussian Splatting for Real-Time Radiance Field Rendering." ACM TOG 42(4), 2023.
- **SAGA**: Cen, J., et al. "Segment Any 3D Gaussians." arXiv:2312.00860, 2023.
- **Gaussian Grouping**: Ye, M., et al. "Gaussian Grouping: Segment and Edit Anything in 3D Scenes." arXiv:2312.00732, 2023.
- **SAGS**: Hu, X., et al. "Semantic Anything in 3D Gaussians." arXiv:2401.17857, 2024.
- **SAM**: Kirillov, A., et al. "Segment Anything." arXiv:2304.02643, 2023.
- **NVOS**: Ren, Z., et al. "Neural Volumetric Object Selection." CVPR 2022.
- **NeRF**: Mildenhall, B., et al. "NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis." ECCV 2020.
- **LERF**: Kerr, J., et al. "LERF: Language Embedded Radiance Fields." arXiv:2303.09553, 2023.
- **코드 저장소**: https://github.com/florinshen/FlashSplat
