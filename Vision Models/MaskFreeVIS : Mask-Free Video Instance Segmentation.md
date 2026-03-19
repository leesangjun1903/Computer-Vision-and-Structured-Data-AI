# Mask-Free Video Instance Segmentation

---

## 1. 핵심 주장과 주요 기여 요약

**MaskFreeVIS**는 비디오 인스턴스 분할(Video Instance Segmentation, VIS) 학습 시 **어떠한 마스크 어노테이션(비디오 마스크 또는 이미지 마스크)도 사용하지 않고**, 바운딩 박스 어노테이션만으로 경쟁력 있는 VIS 성능을 달성하는 최초의 방법론이다.

### 핵심 주장
- 비디오의 **시간적 마스크 일관성(Temporal Mask Consistency)** 제약을 활용하면, 고가의 마스크 어노테이션 없이도 고성능 VIS 모델을 학습할 수 있다.
- 기존 optical flow 기반의 일대일(one-to-one) 대응과 달리, **유연한 일대다(one-to-K) 대응**을 통해 폐색(occlusion), 균질 영역(homogeneous region) 등의 문제를 견고하게 처리할 수 있다.

### 주요 기여
1. **Temporal KNN-patch Loss (TK-Loss)**: 학습 가능한 파라미터 없이, 프레임 간 패치 매칭을 통해 시간적 마스크 일관성을 강제하는 비지도 목적 함수 제안
2. **MaskFreeVIS 프레임워크**: TK-Loss를 기존 SOTA VIS 방법(Mask2Former, SeqFormer 등)에 아키텍처 수정 없이 통합하여 마스크-프리 VIS 학습 가능
3. **최초의 마스크-프리 VIS**: YouTube-VIS 2019에서 ResNet-50 백본으로 **42.5% mask AP** 달성 (완전 지도 학습 대비 91.5% 성능), 4개 벤치마크에서 약지도-완전지도 간 성능 격차를 대폭 축소

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

VIS는 비디오의 모든 객체를 **탐지(detection), 추적(tracking), 분할(segmentation)**하는 과제이다. SOTA VIS 모델은 트랜스포머 기반으로 점점 더 많은 데이터를 요구하지만, **비디오 마스크 어노테이션은 바운딩 박스 대비 수 배 이상의 비용**이 소요된다. 이로 인해:

- 기존 VIS 데이터셋의 규모와 카테고리 다양성이 제한됨
- 기존 box-supervised 인스턴스 분할 방법(BoxInst, DiscoBox 등)은 **단일 이미지용**으로 설계되어 시간적 단서를 활용하지 못함
- Optical flow 기반 접근은 (i) 일대일 대응 가정의 한계, (ii) 대규모 네트워크로 인한 높은 계산 비용 문제가 존재

### 2.2 제안하는 방법 (수식 포함)

#### 2.2.1 Temporal KNN-patch Loss (TK-Loss)

TK-Loss는 4단계로 구성된다:

**1단계: 패치 후보 추출 (Patch Candidate Extraction)**

프레임 $t$에서 위치 $p = (x, y)$를 중심으로 한 $N \times N$ 타깃 이미지 패치 $X_p^t$에 대해, 프레임 $\hat{t}$에서 반경 $R$ 이내의 후보 위치 $\hat{p}$를 선택한다:

$$\|p - \hat{p}\| \leq R$$

**2단계: 시간적 KNN 매칭 (Temporal KNN-Matching)**

패치 간 거리를 다음과 같이 계산한다:

$$\mathbf{d}_{p \rightarrow \hat{p}}^{t \rightarrow \hat{t}} = \left\| X_p^t - X_{\hat{p}}^{\hat{t}} \right\|$$

(식 1)

$L_2$ 노름이 가장 효과적인 매칭 메트릭으로 확인되었다. 가장 작은 거리를 가진 상위 $K$개 매치를 선택하고, 최대 패치 거리 $D$를 초과하는 저신뢰 매치를 제거하여 대응 집합 $S_p^{t \rightarrow \hat{t}} = \{\hat{p}_i\}_i$를 구성한다.

**3단계: 일관성 손실 (Consistency Loss)**

$M_p^t \in [0, 1]$을 프레임 $t$의 위치 $p$에서의 예측 인스턴스 마스크라 할 때, 시간적 마스크 일관성 손실은:

$$\mathcal{L}_f^{t \rightarrow \hat{t}} = \frac{1}{HW} \sum_p \sum_{\hat{p}_i \in S_p^{t \rightarrow \hat{t}}} L_{\text{cons}}(M_p^t, M_{\hat{p}_i}^{\hat{t}})$$

(식 2)

여기서 마스크 일관성 측정 함수는:

$$L_{\text{cons}}(M_p^t, M_{\hat{p}}^{\hat{t}}) = -\log\left(M_p^t M_{\hat{p}}^{\hat{t}} + (1 - M_p^t)(1 - M_{\hat{p}}^{\hat{t}})\right)$$

(식 3)

이 함수는 두 예측이 정확히 배경($M_p^t = M_{\hat{p}}^{\hat{t}} = 0$) 또는 전경($M_p^t = M_{\hat{p}}^{\hat{t}} = 1$)일 때만 최솟값 0을 달성한다. 즉, 동일한 확률값 뿐 아니라 **확신 있는 전경/배경 예측**으로 수렴하도록 유도한다.

**4단계: 순환 튜브 연결 (Cyclic Tube Connection)**

$T$개 프레임으로 구성된 시간적 튜브에 대해 순환적으로 손실을 계산한다:

$$\mathcal{L}_{\text{temp}} = \sum_{t=1}^{T} \begin{cases} \mathcal{L}_f^{t \rightarrow (t+1)} & \text{if } t < T-1 \\ \mathcal{L}_f^{t \rightarrow 0} & \text{if } t = T-1 \end{cases}$$

(식 4)

순환 연결은 밀집 연결(dense connection) 대비 유사한 성능을 유지하면서 메모리 사용량을 대폭 절감한다.

#### 2.2.2 공간-시간 결합 정규화

**공간적 일관성 손실:**

Box Projection Loss:

$$\mathcal{L}_{\text{proj}} = \sum_{t=1}^{T} \sum_{d \in \{\vec{x}, \vec{y}\}} D\left(P'_d(M_p^t), P'_d(M_b^t)\right)$$

(식 5)

여기서 $D$는 Dice loss, $P'$는 $\vec{x}/\vec{y}$ 축 방향 투영 함수, $M_p^t$와 $M_b^t$는 각각 예측 마스크와 GT 박스 마스크이다.

Pairwise Loss:

$$\mathcal{L}_{\text{pair}} = \frac{1}{T} \sum_{t=1}^{T} \sum_{p'_i \in H \times W} L_{\text{cons}}(M_{p'_i}^t, M_{p'_j}^t)$$

(식 6)

색상 유사도가 $\sigma_{\text{pixel}}$ 이상인 공간적 이웃 픽셀 쌍에 대해 적용한다.

공간 손실 결합:

$$\mathcal{L}_{\text{spatial}} = \mathcal{L}_{\text{proj}} + \lambda_{\text{pair}} \mathcal{L}_{\text{pair}}$$

(식 7)

**전체 분할 목적 함수:**

$$\mathcal{L}_{\text{seg}} = \mathcal{L}_{\text{spatial}} + \lambda_{\text{temp}} \mathcal{L}_{\text{temp}}$$

(식 8)

### 2.3 모델 구조

MaskFreeVIS는 **별도의 아키텍처 수정을 요구하지 않는다.** 기존 SOTA VIS 모델(Mask2Former, SeqFormer, VITA, Unicorn)의 마스크 손실만 TK-Loss로 교체한다.

주요 구조적 특징:
- **파라미터-프리**: TK-Loss에 학습 가능한 파라미터가 없음
- **추론 비용 동일**: 추론 시 기존 방법과 동일한 파이프라인
- **Spatio-temporal Box Mask Matching**: 트랜스포머 기반 VIS의 set prediction에서 마스크 어노테이션 없이 인스턴스-시퀀스 매칭을 수행하기 위해, 예측/GT 박스 마스크 간 Dice IoU loss 기반 매칭 도입
- **CIE Lab 색공간** 변환 후 패치 어피니티 계산
- 하이퍼파라미터: 패치 크기 $N=3$, 검색 반경 $R=5$, $K=5$, 매칭 임계값 $D=0.05$, $\lambda_{\text{pair}}=1.0$, $\lambda_{\text{temp}}=0.1$

### 2.4 성능 향상

| 벤치마크 | 백본 | MaskFreeVIS AP | 완전 지도 AP | 비율 |
|---|---|---|---|---|
| YTVIS 2019 | R50 | 42.5 | 46.4 | 91.5% |
| YTVIS 2019 | R101 | 45.8 | 49.2 | 93.1% |
| YTVIS 2019 | Swin-L | 54.3 | 60.4 | 89.9% |
| YTVIS 2021 | R50 | 36.2 | 40.6 | 89.2% |
| OVIS | R50 | 15.7 | 19.6 | 80.1% |
| BDD100K MOTS | - | 23.8 mMOTSA | 29.6 | 80.4% |

핵심 성능 향상 요인:
- TK-Loss 단독으로 +5.0 AP (BoxProj 기준, Table 3)
- Spatial Pairwise Loss 대비 TK-Loss가 2.5배 이상의 개선폭 (+5.0 vs +2.0)
- Flow-based Matching 대비 +2.3 AP (파라미터 없이 달성)
- $K=1 \rightarrow K=5$로 증가 시 +1.7 AP (one-to-many 대응의 효과)

### 2.5 한계

1. **완전 지도 학습과의 성능 격차**: 여전히 약 4-10% AP의 차이 존재 (특히 OVIS, BDD100K 등 어려운 벤치마크에서 격차 확대)
2. **균질 색상 + 폐색 시나리오**: 유사한 색상의 인접 객체가 지속적으로 가까워지는 경우 패치 매칭이 실패 (Figure 10의 시계와 선반 사례)
3. **긴 시간 튜브의 한계**: 튜브 길이 5 이상에서 성능 포화, 시간적으로 먼 프레임 간 패치 대응이 약해짐
4. **원시 패치 기반 매칭의 본질적 한계**: 의미론적 이해 없이 저수준 색상/텍스처 유사도에만 의존
5. **COCO box pretraining 시 이미지 분할 성능 저하**: COCO에서 마스크 없이 학습 시 이미지 AP가 10+ 포인트 낮아, VIS의 초기 분할 품질에 영향

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 다양한 VIS 방법 및 백본에 대한 일반화

논문은 **4가지 베이스 VIS 방법**(Mask2Former, SeqFormer, VITA, Unicorn)과 **3가지 백본**(R50, R101, Swin-L)에 걸쳐 일관된 성능 향상을 보여준다. TK-Loss는 아키텍처 수정 없이 기존 마스크 손실만 교체하므로, 새로운 VIS 모델에도 즉시 적용 가능하다.

### 3.2 다양한 벤치마크에 대한 일반화

4개의 대규모 벤치마크(YTVIS 2019/2021, OVIS, BDD100K MOTS)에서 일관된 성능 향상을 확인:
- **OVIS** (폐색이 심한 벤치마크): +3.6 AP (VITA 기반)
- **BDD100K MOTS** (자율주행 벤치마크): +4.9 mMOTSA (Unicorn 기반)

이는 도메인과 난이도에 관계없이 시간적 마스크 일관성이 일반적으로 유효한 감독 신호임을 입증한다.

### 3.3 데이터 효율성 측면의 일반화

Figure 6에서 보듯, YTVIS 학습 데이터를 5%~100%까지 변경해도 TK-Loss는 **일관되게 3.0 AP 이상의 개선**을 제공한다. 특히 **50% 데이터의 MaskFreeVIS가 10% 데이터의 완전 지도 학습 모델을 상회**하는 결과는 데이터 효율성 측면에서의 강력한 일반화를 보여준다.

### 3.4 일반화 성능 향상의 핵심 요인

1. **파라미터-프리 설계**: 학습 가능한 파라미터가 없어 특정 데이터셋에 대한 과적합 위험이 없음
2. **One-to-K 대응의 유연성**: $K=0$ (폐색), $K=1$ (명확한 매칭), $K \geq 2$ (균질 영역)를 자동으로 처리
3. **높은 매칭 정확도**: YTVIS 2019에서 프레임 쌍당 평균 **95.7%**의 매칭 정확도
4. **도메인 독립적**: 원시 이미지 패치 기반으로 동작하므로, 사전 학습된 optical flow 모델의 도메인 갭 문제를 회피

### 3.5 일반화 향상을 위한 잠재적 방향

- **의미론적 패치 매칭**: 현재의 색상 기반 매칭에 학습된 특징(feature)을 보완적으로 활용
- **적응적 K 선택**: 영역 특성(균질성, 텍스처 복잡도)에 따라 $K$를 동적으로 조절
- **멀티스케일 패치 매칭**: 다양한 패치 크기를 동시에 활용하여 스케일 변화에 강건
- **대규모 비디오 데이터 활용**: 마스크 어노테이션 불필요하므로, 웹 비디오 등 대규모 미라벨 데이터로 확장 가능

---

## 4. 앞으로의 연구에 미치는 영향과 고려할 점

### 4.1 연구 영향

1. **어노테이션 패러다임의 전환**: 비디오 마스크 어노테이션이 VIS의 필수 요소가 아님을 실증적으로 증명하여, VIS 데이터셋의 **대규모 확장**과 **카테고리 다양화**의 가능성을 열었다.

2. **약지도 학습의 실용화**: 기존 약지도 VIS 방법(FlowIRN: 10.5 AP)과의 극적인 성능 차이(42.5 AP)를 통해, 약지도 학습 방법의 실제 배포 가능성을 크게 향상시켰다.

3. **시간적 일관성의 감독 신호로서의 가치 재확인**: 비디오의 시간적 일관성이 마스크 레이블을 효과적으로 대체할 수 있는 강력한 감독 신호임을 입증하여, 자기지도 학습(self-supervised learning) 및 비지도 학습 커뮤니티에 시사점을 제공한다.

4. **모듈형 설계 철학**: 손실 함수만 교체하는 방식으로, 향후 개발되는 새로운 VIS 아키텍처에도 즉시 적용 가능한 범용적 프레임워크를 제시하였다.

### 4.2 향후 연구 시 고려할 점

1. **마스크 품질 향상**: 현재 약 8-10%의 성능 격차를 줄이기 위해, 의미론적 특징과 저수준 패치 매칭의 결합이 필요
2. **어려운 시나리오 대응**: OVIS에서의 성능 격차(15.7 vs 19.6)가 보여주듯, 심한 폐색/변형 상황에서의 매칭 전략 개선이 필요
3. **장기 시간 모델링**: 튜브 길이 5에서 성능이 포화되는 한계를 극복하기 위한 계층적 시간 모델링
4. **Semi-supervised 확장**: 소량의 마스크 어노테이션과 TK-Loss의 결합을 통한 준지도 학습 탐구
5. **실시간 온라인 VIS**: 현재는 오프라인 설정에 집중하였으므로, 온라인 VIS(IDOL 등)에서의 적용 검증 필요
6. **Open-vocabulary VIS**: 마스크 프리 학습의 장점을 살려, 학습 시 보지 못한 카테고리로의 확장 연구

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

### 5.1 완전 지도 VIS 방법들

| 연구 | 연도 | 핵심 접근 | YTVIS19 AP (R50) | 마스크 필요 |
|---|---|---|---|---|
| VisTR (Wang et al.) | 2021 (CVPR) | Transformer 기반 end-to-end VIS | ~36.2 | ✓ |
| IFC (Hwang et al.) | 2021 (NeurIPS) | Inter-frame communication transformers | 42.8 | ✓ |
| SeqFormer (Wu et al.) | 2022 (ECCV) | Frame query decomposition + pseudo video | 47.4 | ✓ |
| Mask2Former-VIS (Cheng et al.) | 2022 | Masked attention + universal segmentation | 46.4~47.8 | ✓ |
| VMT (Ke et al.) | 2022 (ECCV) | Video Mask Transfiner for high-quality VIS | 47.9 | ✓ |
| VITA (Heo et al.) | 2022 (NeurIPS) | Object token association | 19.6 (OVIS) | ✓ |
| IDOL (Wu et al.) | 2022 (ECCV) | Online VIS with contrastive learning | - | ✓ |
| MinVIS (Huang et al.) | 2022 (NeurIPS) | VIS without video-based architecture | - | ✓ |
| **MaskFreeVIS (본 논문)** | **2023** | **TK-Loss로 마스크-프리 VIS** | **42.5** | **✗** |

### 5.2 약지도/Box-supervised 인스턴스 분할

| 연구 | 연도 | 도메인 | 핵심 접근 | 시간적 단서 |
|---|---|---|---|---|
| BoxInst (Tian et al.) | 2021 (CVPR) | 이미지 | Projection loss + Pairwise loss | ✗ |
| DiscoBox (Lan et al.) | 2021 (ICCV) | 이미지 | Teacher model guided pseudo mask | ✗ |
| Box-supervised w/ Level Set (Li et al.) | 2022 (ECCV) | 이미지 | Level set evolution | ✗ |
| FlowIRN (Liu et al.) | 2021 (CVPR) | 비디오 | Class-label only + optical flow | ✓ (flow) |
| SOLO-Track (Fu et al.) | 2021 (CVPR) | 비디오 | VIS without video annotations | 제한적 |
| **MaskFreeVIS** | **2023** | **비디오** | **TK-Loss (one-to-K patch matching)** | **✓ (patch)** |

### 5.3 핵심 비교 분석

**MaskFreeVIS vs. BoxInst (이미지 기반 약지도)**:
- BoxInst의 spatial loss만 사용 시 YTVIS 2019에서 38.6 AP
- TK-Loss 추가 시 42.5 AP → **+3.9 AP**, 이는 시간적 단서의 가치를 명확히 보여줌

**MaskFreeVIS vs. FlowIRN (비디오 기반 약지도)**:
- FlowIRN: 10.5 AP (분류 라벨만 사용)
- MaskFreeVIS: 42.5 AP → **4배 이상의 성능**, 바운딩 박스 수준의 약한 감독이면 충분함을 입증

**MaskFreeVIS vs. Flow-based Matching**:
- RAFT 기반 optical flow 매칭: 40.2 AP (추가 파라미터 필요)
- TK-Loss: 42.5 AP (파라미터-프리) → **+2.3 AP**, one-to-K의 우월성 확인

**MaskFreeVIS vs. 완전 지도 학습 (Mask2Former)**:
- 마스크-프리 설정에서 완전 지도 성능의 **91.5%** 달성
- COCO 이미지 마스크 + pseudo video 사용 시 **46.6 AP** (완전 지도 47.8 AP의 97.5%)

---

## 참고자료

1. **Ke, L., Danelljan, M., Ding, H., Tai, Y.-W., Tang, C.-K., & Yu, F.** (2023). "Mask-Free Video Instance Segmentation." *arXiv preprint arXiv:2303.15904v1*. [본 논문]
2. **Cheng, B., Choudhuri, A., Misra, I., Kirillov, A., Girdhar, R., & Schwing, A. G.** (2021). "Mask2Former for Video Instance Segmentation." *arXiv preprint arXiv:2112.10764*.
3. **Wu, J., Jiang, Y., Zhang, W., Bai, X., & Bai, S.** (2022). "SeqFormer: A Frustratingly Simple Model for Video Instance Segmentation." *ECCV 2022*.
4. **Tian, Z., Shen, C., Wang, X., & Chen, H.** (2021). "BoxInst: High-Performance Instance Segmentation with Box Annotations." *CVPR 2021*.
5. **Liu, Q., Ramanathan, V., Mahajan, D., Yuille, A., & Yang, Z.** (2021). "Weakly Supervised Instance Segmentation for Videos with Temporal Mask Consistency." *CVPR 2021*.
6. **Teed, Z. & Deng, J.** (2020). "RAFT: Recurrent All-Pairs Field Transforms for Optical Flow." *ECCV 2020*.
7. **Heo, M., Hwang, S., Oh, S. W., Lee, J.-Y., & Kim, S. J.** (2022). "VITA: Video Instance Segmentation via Object Token Association." *NeurIPS 2022*.
8. **Yan, B., Jiang, Y., Sun, P., Wang, D., Yuan, Z., Luo, P., & Lu, H.** (2022). "Towards Grand Unification of Object Tracking." *ECCV 2022*.
9. **Huang, D.-A., Yu, Z., & Anandkumar, A.** (2022). "MinVIS: A Minimal Video Instance Segmentation Framework without Video-Based Training." *NeurIPS 2022*.
10. **Yang, L., Fan, Y., & Xu, N.** (2019). "Video Instance Segmentation." *ICCV 2019*.
11. **Ke, L., Ding, H., Danelljan, M., Tai, Y.-W., Tang, C.-K., & Yu, F.** (2022). "Video Mask Transfiner for High-Quality Video Instance Segmentation." *ECCV 2022*.
12. **Lan, S., Yu, Z., Choy, C., et al.** (2021). "DiscoBox: Weakly Supervised Instance Segmentation and Semantic Correspondence from Box Supervision." *ICCV 2021*.
13. GitHub Repository: [https://github.com/SysCV/MaskFreeVis](https://github.com/SysCV/MaskFreeVis)
