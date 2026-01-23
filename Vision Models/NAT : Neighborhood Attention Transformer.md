# Neighborhood Attention Transformer

### 1. 핵심 주장 및 주요 기여 (간결한 요약)

**Neighborhood Attention Transformer (NAT)**는 컴퓨터 비전에서 자기주의(Self-Attention)의 이차 복잡도 문제를 해결하는 혁신적인 논문입니다. 세 가지 핵심 기여는: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e9eb3858-f8a8-4c62-b0f8-03abf8f6996a/2204.07143v5.pdf)

1. **Neighborhood Attention (NA)**: 각 픽셀을 가장 가까운 이웃 픽셀들로만 제한하는 픽셀 단위의 슬라이딩 윈도우 어텐션 메커니즘으로, 이차 복잡도를 선형 복잡도로 감소시킵니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e9eb3858-f8a8-4c62-b0f8-03abf8f6996a/2204.07143v5.pdf)

2. **NATTEN 패키지**: 효율적인 C++ 및 CUDA 커널을 포함하는 파이썬 패키지로, Swin Transformer의 윈도우 자기주의(WSA)보다 최대 40% 빠르고 25% 적은 메모리를 사용합니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e9eb3858-f8a8-4c62-b0f8-03abf8f6996a/2204.07143v5.pdf)

3. **NAT 아키텍처**: 계층적 계획(hierarchical design)을 활용하여 ImageNet-1K에서 83.2%(NAT-Tiny), MS-COCO에서 51.4% mAP, ADE20K에서 48.4% mIoU를 달성하며 동일 크기의 Swin Transformer를 능가합니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e9eb3858-f8a8-4c62-b0f8-03abf8f6996a/2204.07143v5.pdf)

***

### 2. 해결하고자 하는 문제, 제안 방법 및 모델 구조

#### 2.1 핵심 문제

Vision Transformer(ViT)는 전역 자기주의를 사용하여 우수한 성능을 보이지만, 다음의 문제가 존재합니다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e9eb3858-f8a8-4c62-b0f8-03abf8f6996a/2204.07143v5.pdf)

- **이차 복잡도**: 이미지 해상도에 대해 $O(n^2d)$ 시간 복잡도와 $O(n^2)$ 공간 복잡도 (여기서 $n$은 토큰 수, $d$는 임베딩 차원)
- **로컬 귀납적 편향 부족**: 합성곱과 달리 자기주의는 기본적으로 전역 1차원 연산
- **다운스트림 태스크 확장성**: 고해상도 이미지가 필요한 객체 탐지 및 의미 분할에 비실용적

#### 2.2 제안하는 방법 및 수식

**Neighborhood Attention (NA)**는 다음과 같이 정의됩니다. 입력 $X \in \mathbb{R}^{n \times d}$ (여기서 $n$은 토큰 수, $d$는 임베딩 차원)에 대해:

$i$번째 입력의 어텐션 가중치는 $k$개의 최근접 이웃을 사용하여 다음과 같이 정의됩니다:

$$A_i^k = \begin{bmatrix} Q_i K_{\rho_1(i)}^T + B(i, \rho_1(i)) \\ Q_i K_{\rho_2(i)}^T + B(i, \rho_2(i)) \\ \vdots \\ Q_i K_{\rho_k(i)}^T + B(i, \rho_k(i)) \end{bmatrix}$$

여기서 $\rho_j(i)$는 $i$의 $j$번째 최근접 이웃이고, $B(i, j)$는 상대적 위치 편향입니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e9eb3858-f8a8-4c62-b0f8-03abf8f6996a/2204.07143v5.pdf)

이웃 값들은:

$$V_i^k = \begin{bmatrix} V_{\rho_1(i)}^T \\ V_{\rho_2(i)}^T \\ \vdots \\ V_{\rho_k(i)}^T \end{bmatrix}^T$$

최종 Neighborhood Attention 출력:

$$NA^k(i) = \text{softmax}\left(\frac{A_i^k}{\sqrt{d}}\right)V_i^k$$

**핵심 특성**: $k$가 증가함에 따라 NA는 자기주의에 접근하며, 최대 이웃 크기에서 완전한 자기주의와 동일합니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e9eb3858-f8a8-4c62-b0f8-03abf8f6996a/2204.07143v5.pdf)

#### 2.3 복잡도 분석

|모듈|시간 복잡도 (FLOPs)|메모리|
|---|---|---|
|자기주의 (SA)|$3hwd^2 + 2h^2w^2d$|$3d^2 + h^2w^2$|
|윈도우 자기주의 (WSA)|$3hwd^2 + 2hwdk^2$|$3d^2 + hwk^2$|
|Neighborhood Attention (NA)|$3hwd^2 + 2hwdk^2$|$3d^2 + hwk^2$|
|합성곱|$hwd^2k^2$|$d^2k^2$|

NA와 WSA는 이론적으로 동일한 복잡도를 가지지만, NA는 구현 효율성과 병렬화 가능성에서 우수합니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e9eb3858-f8a8-4c62-b0f8-03abf8f6996a/2204.07143v5.pdf)

#### 2.4 모델 구조

**NAT (Neighborhood Attention Transformer)**는 계층적 설계를 사용합니다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e9eb3858-f8a8-4c62-b0f8-03abf8f6996a/2204.07143v5.pdf)

1. **토크나이저**: 2개의 3×3 합성곱(stride=2×2)으로 입력을 1/4 해상도로 축소

2. **4단계 구조**: 
   - 단계 1-4: 각각 H/4 × W/4, H/8 × W/8, H/16 × W/16, H/32 × W/32 해상도
   - 각 단계 사이 3×3 합성곱(stride=2×2)을 사용한 오버래핑 다운샘플러

3. **NAT 블록**: LayerNorm → NA → MLP → skip connection

| 변형 | 층 구성 | 임베딩 차원 | MLP 비율 | 파라미터 | FLOPs |
|------|--------|-----------|---------|--------|-------|
| NAT-Mini | (3,4,6,5) | 32×2 | 3 | 20M | 2.7G |
| NAT-Tiny | (3,4,18,5) | 32×2 | 3 | 28M | 4.3G |
| NAT-Small | (3,4,18,5) | 32×3 | 2 | 51M | 7.8G |
| NAT-Base | (3,4,18,5) | 32×4 | 2 | 90M | 13.7G |

***

### 3. 성능 향상 및 한계

#### 3.1 성능 향상

**ImageNet-1K 분류 (224×224):** [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e9eb3858-f8a8-4c62-b0f8-03abf8f6996a/2204.07143v5.pdf)
- NAT-Mini: 81.8% (Swin-Tiny 대비 +0.5%)
- NAT-Tiny: 83.2% (Swin-Tiny 대비 +1.9%, ConvNeXt-T 대비 +1.1%)
- NAT-Small: 83.7% (Swin-Small 대비 +0.7%)
- NAT-Base: 84.3% (Swin-Base 대비 +0.8%)

**MS-COCO 객체 탐지:** [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e9eb3858-f8a8-4c62-b0f8-03abf8f6996a/2204.07143v5.pdf)
- NAT-Tiny: 51.4% mAP (Swin-Tiny 대비 +1.0%)
- NAT-Small: 48.0% mAP (Cascade Mask R-CNN, Swin-S 대비 +0.1%)

**ADE20K 의미 분할:** [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e9eb3858-f8a8-4c62-b0f8-03abf8f6996a/2204.07143v5.pdf)
- NAT-Tiny: 48.4% mIoU (Swin-Tiny 대비 +2.6%)
- NAT-Small: 49.5% (단일 스케일) vs Swin-S 47.6%

#### 3.2 효율성 개선

- **메모리 사용**: NATTEN 구현으로 PyTorch 순수 구현 대비 약 9배 감소
- **처리량**: 최대 40% 빠른 추론 속도 (A100 GPU에서 측정)
- **메모리 효율성**: 25% 적은 메모리 사용량

#### 3.3 한계

**이론적 한계:** [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e9eb3858-f8a8-4c62-b0f8-03abf8f6996a/2204.07143v5.pdf)
1. **로컬 어텐션의 제약**: 전역 문맥 모델링 부족 → 이를 해결하기 위해 후속 연구인 **Dilated Neighborhood Attention (DiNA)**가 제안됨

2. **번역 등가성의 완전성**: 코너 케이스에서 어텐션 스팬이 반복되어 완벽한 번역 등가성을 깨뜨림 (Zero-padding 사용 문제 해결을 위한 설계)

3. **스케일링 한계**: 매우 큰 창 크기에서 성능 개선이 포화되는 경향

**실무적 한계:**
- NATTEN의 초기 단계: 일부 GPU 아키텍처(Hopper 등)에서 최적화 부족
- 다양한 하드웨어에 대한 지원 확장 필요

***

### 4. 모델의 일반화 성능 향상 가능성

#### 4.1 현재 일반화 성능

**다중 다운스트림 태스크에서의 성능:** [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e9eb3858-f8a8-4c62-b0f8-03abf8f6996a/2204.07143v5.pdf)

NA는 고정된 어텐션 스팬을 유지하면서도 재용성 있는 설계를 제공:
- 커널 크기 3×3부터 9×9까지 광범위한 옵션 제공
- 고정 이웃 정의로 인해 다양한 입력 해상도에 대응 가능

#### 4.2 성능 향상 전략

**1) 전역-로컬 하이브리드 어텐션:** [arxiv](https://arxiv.org/pdf/2209.15001.pdf)
DiNAT (Dilated Neighborhood Attention Transformer)에서:
$$\text{DiNA}(\text{dilation } \delta) = \text{Attention to k-dilated neighbors}$$

Dilation 값의 범위: $\delta \in [1, \lfloor \frac{n}{k} \rfloor]$

다층 구조에서:
- NA (dilation=1): 로컬 문맥
- DiNA (dilation>1): 희소 전역 어텐션

이를 통해 **기하급수적 수용장(receptive field) 확장** 가능:
$$RF_{\text{max}} = k^{\ell} \text{ (적절한 dilation 구성 시)}$$

**2) 상대적 위치 편향의 역할:**
상대적 위치 편향 $B(i, j)$를 통해:
- 로컬 귀납적 편향 도입
- 공간적 구조 정보 인코딩
- 일반화 성능 향상

**3) 오버래핑 합성곱 다운샘플러:** [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e9eb3858-f8a8-4c62-b0f8-03abf8f6996a/2204.07143v5.pdf)
패치 합병 대신 오버래핑 합성곱 사용으로:
- 유용한 귀납적 편향 추가
- 다운스트림 태스크 전이 학습 능력 향상
- ConvNeXt 아이디어 차용

#### 4.3 일반화 성능의 정량적 증거

**Ablation Study (ImageNet-1K):** [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e9eb3858-f8a8-4c62-b0f8-03abf8f6996a/2204.07143v5.pdf)

| 설정 | 정확도 | 개선도 |
|------|------|-------|
| Swin-T (기준) | 81.29% | - |
| + 오버래핑 다운샘플러 | 81.78% | +0.49% |
| + NAT 구성 (깊이/너비 조정) | 82.72% | +0.94% |
| + SASA 적용 | 82.54% | +0.82% |
| + NA 적용 (최종) | 83.20% | **+1.91%** |

**커널 크기 영향 (NAT-Tiny):** [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e9eb3858-f8a8-4c62-b0f8-03abf8f6996a/2204.07143v5.pdf)

| 커널 크기 | ImageNet | COCO mAP | ADE20K |
|---------|---------|---------|--------|
| 3×3 | 81.4% | 46.1 | 46.0% |
| 5×5 | 81.6% | 46.8 | 46.3% |
| 7×7 | 83.2% | 47.7 | 48.4% |
| 9×9 | 83.1% | 48.5 | 48.1% |

**관찰**: 7×7 커널이 최적값을 보이며, 더 큰 커널은 처리량 감소 대비 성능 개선이 미미함.

***

### 5. 2020년 이후 관련 최신 연구 비교 분석

#### 5.1 Vision Transformer 계열의 진화 경로

| 연도 | 논문 | 핵심 기여 | 복잡도 | ImageNet 정확도 |
|------|------|---------|--------|---------------|
| 2020 | Vision Transformer (ViT) | 패치 기반 전역 자기주의 | $O(n^2d)$ | 77.9% (ImageNet-21k 사전학습) |
| 2021 | Swin Transformer | 시프트된 윈도우 어텐션 | $O(n)$ | 87.3% |
| 2021 | DeiT | 데이터 효율적 ViT | $O(n^2d)$ | 81.8% (중간 스케일) |
| 2022 | NAT | 이웃 슬라이딩 윈도우 어텐션 | $O(n)$ | **83.2%** (NAT-T) |
| 2022 | ConvNeXt | Transformer 영감 CNN | $O(n)$ | 82.1% (ConvNeXt-T) |
| 2022 | DiNAT | Dilated 이웃 어텐션 | $O(n)$ | 84.5%+ |
| 2023 | GC ViT | 전역 문맥 ViT | $O(n)$ | 84.3% (51M) |

#### 5.2 로컬 어텐션 메커니즘 비교

**Stand-Alone Self-Attention (SASA) :** [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e9eb3858-f8a8-4c62-b0f8-03abf8f6996a/2204.07143v5.pdf)
- 초기 슬라이딩 윈도우 어텐션 구현
- 문제점: 코너 케이스에서 padding으로 인한 어텐션 스팬 감소
- 구현 비효율성으로 인해 실용적 채택 부족

**Swin Transformer :** [openaccess.thecvf](https://openaccess.thecvf.com/content/ICCV2021/papers/Liu_Swin_Transformer_Hierarchical_Vision_Transformer_Using_Shifted_Windows_ICCV_2021_paper.pdf)
- 비중첩 윈도우 파티션 → 시프트된 윈도우로 확장
- 장점: 효율적 구현, 계층적 구조
- 단점: 번역 등가성 위반, 복잡한 마스킹 연산, 고정 수용장

**Neighborhood Attention :** [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e9eb3858-f8a8-4c62-b0f8-03abf8f6996a/2204.07143v5.pdf)
- 픽셀 단위 슬라이딩 윈도우 (이웃 정의 사용)
- 개선점: 
  - 모든 픽셀이 동일 어텐션 스팬 유지
  - $k \to \infty$일 때 SA에 수렴
  - 번역 등가성 대부분 보존

**Dilated Neighborhood Attention :** [deepai](https://deepai.org/publication/dilated-neighborhood-attention-transformer)
- NA의 확장으로 희소 전역 어텐션 추가
- Dilation 패턴으로 기하급수적 수용장 확장
- DiNAT-Large: Swin-B 대비 +1.5% COCO 검출, +1.3% ADE20K 분할

#### 5.3 CNN vs Transformer 아키텍처 수렴

**ConvNeXt :** [emergentmind](https://www.emergentmind.com/topics/convnext-backbone)
Transformer 아이디어를 CNN에 통합:
- Patchify Stem: 4×4 비중첩 합성곱 (ViT 패치 모방)
- 깊이 분배: (3,3,9,3) 블록 구성 (Swin 영감)
- 깊이 separable 합성곱: 효율성 향상
- 역 병목 블록: Transformer MLP 구조 모방
- LayerNorm + GELU: Transformer 관례 채용

**성능 비교 (ImageNet-1K, 224×224):**

| 모델 | 파라미터 | FLOPs | 정확도 | 처리량 |
|------|--------|-------|-------|-------|
| ConvNeXt-T | 28M | 4.5G | 82.1% | 2491 imgs/s |
| Swin-T | 28M | 4.5G | 81.3% | 1730 imgs/s |
| NAT-T | 28M | 4.3G | **83.2%** | 1541 imgs/s |
| ConvNeXt-S | 50M | 8.7G | 83.1% | 1549 imgs/s |
| NAT-S | 51M | 7.8G | **83.7%** | 1051 imgs/s |

**분석**: 
- NAT는 유사 파라미터 대비 정확도 우위 (T: +1.1%, S: +0.6%)
- ConvNeXt는 처리량 우위 (하드웨어 최적화)
- NAT는 FLOPs 효율성 우위

#### 5.4 다운스트림 태스크 성능

**객체 탐지 (MS-COCO, Mask R-CNN 3×학습률 스케줄):** [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e9eb3858-f8a8-4c62-b0f8-03abf8f6996a/2204.07143v5.pdf)

| 백본 | 파라미터 | FLOPs | AP_box | AP_mask |
|------|--------|-------|--------|---------|
| Swin-T | 48M | 267G | 46.0 | 41.6 |
| ConvNeXt-T | 48M | 262G | 46.2 | 41.7 |
| NAT-T | 48M | 258G | **47.7** | **42.6** |
| Swin-S | 69M | 359G | 48.5 | 43.3 |
| NAT-S | 70M | 330G | 48.4 | 43.2 |

**의미 분할 (ADE20K, 512×512 입력):** [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e9eb3858-f8a8-4c62-b0f8-03abf8f6996a/2204.07143v5.pdf)

| 백본 | mIoU (단일) | mIoU (멀티) |
|------|-----------|-----------|
| Swin-T | 45.8% | 47.6% |
| ConvNeXt-T | 46.0% | 46.7% |
| NAT-T | **47.1%** | **48.4%** |
| Swin-S | 47.6% | 49.5% |
| NAT-S | 48.0% | 49.5% |

**주요 관찰**:
- NAT는 분할 태스크에서 특히 강점 (로컬 어텐션 + 오버래핑 다운샘플러)
- Swin과 ConvNeXt는 상보적 강점 존재

#### 5.5 최신 효율성 개선 방향

**1) Linear Attention 계열:** [openreview](https://openreview.net/forum?id=41Pdz4r5aB)
Performer, BigBird 영감 → 선형 복잡도 달성
- 문제점: 어텐션 스파이크니스 손실
- 개선: Mamba 구조와 결합

**2) 하이브리드 로컬-전역 어텐션:** [mdpi](https://www.mdpi.com/1424-8220/23/7/3447)
- PLG-ViT: 병렬 로컬-전역 자기주의 (Shifted Window 없음)
- GC ViT: 전역 문맥 모듈 + 로컬 어텐션
- RegionViT: 지역-로컬 어텐션

**3) 동적 윈도우 크기:** [ecva](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136850460.pdf)
VSA (Varied-Size Window Attention):
- 학습 가능한 가변 크기 윈도우
- 객체 크기 다양성에 대응

#### 5.6 번역 등가성 분석

Appendix C 분석: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e9eb3858-f8a8-4c62-b0f8-03abf8f6996a/2204.07143v5.pdf)

| 메커니즘 | 번역 등가성 | 비고 |
|---------|-----------|-----|
| 자기주의 | ✓ 완전 | 위치 불변 어텐션 |
| SASA | ✓ 완전 | Raster scan, zero-padding |
| WSA/SWSA | ✗ 위반 | 윈도우 파티션 고정 |
| NA | ✓ 대부분 | 코너에서 부분 이완 (이웃 반복) |

**NA의 설계 선택**: 코너 케이스에서 번역 등가성을 부분적으로 이완하되, 감소된 어텐션 스팬 문제를 해결하여 전체 성능 개선.

***

### 6. 논문이 앞으로의 연구에 미치는 영향 및 고려사항

#### 6.1 학술적 영향

**1) 슬라이딩 윈도우 어텐션의 재발견:**
논문 출판 당시, Swin Transformer의 성공으로 슬라이딩 윈도우 방식이 "비효율적"이라는 편견이 만연했습니다. NAT의 NATTEN 패키지 개발은 다음을 입증했습니다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e9eb3858-f8a8-4c62-b0f8-03abf8f6996a/2204.07143v5.pdf)
- 효율적인 저수준 구현으로 슬라이딩 윈도우가 윈도우 파티션보다 빠를 수 있음
- 병렬화 가능한 CUDA 커널 설계의 중요성

**2) 후속 연구 활성화:**
- **DiNAT (2022)**: NA의 Dilated 버전으로 전역 문맥 추가
- **DiNAT-IR (2025)**: 이미지 복원 태스크로 확장
- Hybrid 어텐션 연구 활성화 (GC ViT, PLG-ViT, RegionViT)

**3) 이론적 기여:**
- NA $\to$ SA의 연속적 수렴성 증명
- 번역 등가성에 대한 정밀한 분석
- 수용장(receptive field) 성장률 연구

#### 6.2 실무적 기여

**1) NATTEN 생태계:**
- PyTorch 통합 모듈 제공
- CPU/GPU 지원, 혼합 정밀도 지원
- 자동 미분 호환성

**2) 다운스트림 태스크 적용성:**
- 고해상도 입력 지원 (패딩 불필요)
- 임의 크기 피쳐맵 처리 가능
- Swin 기반 프레임워크와 직접 호환

**3) 하드웨어 친화성:**
메모리 계층 구조 활용:
- Tiled NA 알고리즘으로 공유 메모리 활용
- 뱅크 충돌 최소화
- 대역폭 향상

#### 6.3 앞으로의 연구 방향 및 고려사항

**1) 아키텍처 최적화:**

| 방향 | 내용 | 예상 효과 |
|------|------|---------|
| **Global-Local 하이브리드** | DiNAT처럼 희소 전역 어텐션 추가 | 수용장 확장, 전역 문맥 개선 |
| **동적 커널 크기** | 입력 복잡도에 따른 적응형 윈도우 | 객체 크기 다양성 대응 |
| **계층별 Dilation** | 깊이에 따른 다양한 dilation 값 | 효율적 계층적 정보 융합 |

**2) 구현 최적화:**
- Hopper 아키텍처 CUDA 지원 (논문 미래 계획)
- Implicit GEMM 기반 구현 (CUTLASS 활용)
- Multi-GPU 분산 학습 최적화

**3) 이론적 탐구:**
- Attention Sparsity와 일반화 경계 분석
- 다양한 dilation 패턴의 최적성 연구
- 위치 편향 설계의 이론적 근거

**4) 도메인별 확장:**

| 도메인 | 응용 | 고려사항 |
|--------|------|---------|
| **의료 영상** | CT/MRI 분석 | 고해상도 입력, 3D 어텐션 확장 |
| **자율주행** | 실시간 고해상도 처리 | 지연 시간 최소화, 엣지 배포 |
| **영상 생성** | 확산 모델 | 초고해상도 생성 (4K 이상) |
| **비디오** | 시간-공간 어텐션 | 시간축 이웃 정의 |

**5) 하이브리드 접근:**

$$\text{Hybrid Attention} = \alpha \cdot \text{NA}(\text{local}) + (1-\alpha) \cdot \text{DiNA}(\text{global})$$

ConvNeXt와의 결합:
- 로컬: NA (슬라이딩 윈도우)
- 전역: 대규모 합성곱 또는 희소 어텐션
- 각 레이어 또는 헤드별 선택

#### 6.4 제약사항 및 한계 인식

**1) 로컬 어텐션의 근본적 한계:**
- 매우 먼 거리 의존성 모델링 어려움
- 전역 패턴 인식 제약

**해결책**: DiNAT의 dilation 패턴 또는 다단계 피라미드 구조

**2) 커널 크기 선택의 민감성:**
- 7×7이 최적이지만 태스크/해상도별 조정 필요
- 과도한 dilation은 성능 포화

**해결책**: 자동 커널 크기 선택 학습 또는 멀티스케일 접근

**3) 메모리 여전히 선형이지만 상수 인자 존재:**
- 다중 헤드 어텐션: O( $hwdk^2 \cdot \text{num heads}$ )
- 배치 크기와의 상호작용

**해결책**: 그래디언트 체크포인팅, 혼합 정밀도 학습

#### 6.5 우선순위 연구 주제

**단기 (1-2년):**
1. DiNAT의 다양한 비전 태스크 적용
2. NATTEN의 추가 하드웨어 지원
3. 최적 커널 크기의 자동 결정

**중기 (2-4년):**
1. 3D 비전 (의료, 자율주행)
2. 멀티모달 모델에서의 NA 효율성
3. 차별화된 Transformer 이론 (일반화 경계)

**장기 (4년 이상):**
1. Quantum 또는 Neuromorphic 하드웨어에서의 효율적 구현
2. 뇌 영감 어텐션 메커니즘
3. Unified Architecture (CNN + Transformer + RNN 통합)

***

## 결론

**Neighborhood Attention Transformer**는 비전 트랜스포머 효율화의 중요한 이정표입니다. SASA의 초기 아이디어를 다시 살펴보면서도, Swin Transformer의 성공과 NA의 차별화된 설계(최근접 이웃 정의, 번역 등가성)를 통해 슬라이딩 윈도우 어텐션의 실무적 가치를 입증했습니다. NATTEN의 효율적 구현은 이론과 실무의 간격을 줄였으며, DiNAT로의 확장은 로컬 어텐션의 근본적 한계를 해결하는 방향을 제시했습니다.

향후 연구는 **로컬-전역 어텐션의 최적 결합**, **자동화된 아키텍처 설계**, 그리고 **다양한 도메인(의료, 자율주행, 비디오)으로의 확장**에 집중할 것으로 예상됩니다. ConvNeXt와의 성능 경쟁은 CNN과 Transformer의 수렴을 보여주며, 차별화의 핵심은 **하드웨어 친화성**, **구현 효율성**, 그리고 **귀납적 편향의 적절한 균형**에 있음을 시사합니다.

***

## 참고문헌
<span style="display:none">[^1_12][^1_13][^1_14][^1_15][^1_16][^1_17][^1_18][^1_19][^1_20][^1_21][^1_22][^1_23][^1_24][^1_25][^1_26][^1_27][^1_28][^1_29][^1_30][^1_31][^1_32][^1_33][^1_34][^1_35][^1_36][^1_37][^1_38][^1_39][^1_40][^1_41][^1_42][^1_43][^1_44][^1_45][^1_46][^1_47][^1_48][^1_49][^1_50][^1_51][^1_52][^1_53][^1_54][^1_55][^1_56][^1_57][^1_58][^1_59][^1_60][^1_61][^1_62][^1_63][^1_64][^1_65][^1_66][^1_67][^1_68][^1_69][^1_70][^1_71][^1_72][^1_73][^1_74][^1_75][^1_76][^1_77][^1_78][^1_79][^1_80][^1_81][^1_82][^1_83][^1_84][^1_85][^1_86][^1_87][^1_88][^1_89][^1_90][^1_91][^1_92][^1_93][^1_94][^1_95][^1_96]</span>

<div align="center">⁂</div>

[^1_1]: 2204.07143v5.pdf

[^1_2]: https://arxiv.org/pdf/2209.15001.pdf

[^1_3]: https://openaccess.thecvf.com/content/ICCV2021/papers/Liu_Swin_Transformer_Hierarchical_Vision_Transformer_Using_Shifted_Windows_ICCV_2021_paper.pdf

[^1_4]: https://deepai.org/publication/dilated-neighborhood-attention-transformer

[^1_5]: https://www.emergentmind.com/topics/convnext-backbone

[^1_6]: https://openaccess.thecvf.com/content/CVPR2022/papers/Liu_A_ConvNet_for_the_2020s_CVPR_2022_paper.pdf

[^1_7]: https://openreview.net/forum?id=41Pdz4r5aB

[^1_8]: https://www.mdpi.com/1424-8220/23/7/3447

[^1_9]: https://arxiv.org/pdf/2206.09959.pdf

[^1_10]: https://rpand002.github.io/data/ICLR_2022_regionvit.pdf

[^1_11]: https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136850460.pdf

[^1_12]: https://ieeexplore.ieee.org/document/9552005/

[^1_13]: https://link.springer.com/10.1007/978-3-030-58520-4

[^1_14]: https://e-journal.hamzanwadi.ac.id/index.php/infotek/article/view/31108

[^1_15]: https://ieeexplore.ieee.org/document/11043624/

[^1_16]: https://link.springer.com/10.1007/s42452-024-06048-0

[^1_17]: https://dl.acm.org/doi/10.1145/3723358

[^1_18]: https://ojs.istp-press.com/jait/article/view/645

[^1_19]: https://ieeexplore.ieee.org/document/10860237/

[^1_20]: https://www.mdpi.com/2076-3417/14/15/6471

[^1_21]: https://arxiv.org/pdf/2203.02358.pdf

[^1_22]: https://arxiv.org/pdf/2311.05988.pdf

[^1_23]: https://arxiv.org/pdf/2112.11435.pdf

[^1_24]: http://arxiv.org/pdf/2403.19882.pdf

[^1_25]: https://arxiv.org/pdf/2104.05707.pdf

[^1_26]: https://arxiv.org/pdf/2106.03714.pdf

[^1_27]: http://arxiv.org/pdf/2211.10526.pdf

[^1_28]: https://openreview.net/forum?id=tv0YEuJewa

[^1_29]: https://openaccess.thecvf.com/content/CVPR2023/papers/Pan_Slide-Transformer_Hierarchical_Vision_Transformer_With_Local_Self-Attention_CVPR_2023_paper.pdf

[^1_30]: https://arxiv.org/pdf/2103.14030.pdf

[^1_31]: https://openaccess.thecvf.com/content/CVPR2023/papers/Hassani_Neighborhood_Attention_Transformer_CVPR_2023_paper.pdf

[^1_32]: https://www.nature.com/articles/s41598-025-24844-5

[^1_33]: https://arxiv.org/abs/2103.14030

[^1_34]: https://github.com/SHI-Labs/Neighborhood-Attention-Transformer/blob/main/NAT.md

[^1_35]: https://kmhana.tistory.com/27

[^1_36]: https://www.youtube.com/watch?v=SndHALawoag

[^1_37]: https://www.emergentmind.com/papers/2204.07143

[^1_38]: https://viso.ai/deep-learning/vision-transformer-vit/

[^1_39]: https://openaccess.thecvf.com/content/ICCV2021/html/Liu_Swin_Transformer_Hierarchical_Vision_Transformer_Using_Shifted_Windows_ICCV_2021_paper.html

[^1_40]: https://github.com/vikhyat/e_natten

[^1_41]: https://ar5iv.labs.arxiv.org/html/2104.05707

[^1_42]: https://arxiv.org/html/2104.05707v2

[^1_43]: https://arxiv.org/pdf/2501.06480.pdf

[^1_44]: https://arxiv.org/html/2412.18778v1

[^1_45]: https://arxiv.org/html/2507.18405v2

[^1_46]: https://arxiv.org/abs/2204.07143

[^1_47]: https://arxiv.org/pdf/2203.01536.pdf

[^1_48]: https://arxiv.org/html/2511.14712v1

[^1_49]: https://arxiv.org/html/2508.17081v1

[^1_50]: https://arxiv.org/html/2501.06480v2

[^1_51]: https://arxiv.org/html/2504.16922v1

[^1_52]: https://www.cs.uoregon.edu/Reports/AREA-202307-Hassani.pdf

[^1_53]: https://www.semanticscholar.org/paper/Neighborhood-Attention-Transformer-Hassani-Walton/ad7bcec33f5206d4f28687a6a5a950de67010651

[^1_54]: https://www.emergentmind.com/topics/sliding-window-self-attention

[^1_55]: https://ieeexplore.ieee.org/document/9949350/

[^1_56]: https://ieeexplore.ieee.org/document/9954791/

[^1_57]: https://arxiv.org/abs/2206.09959

[^1_58]: https://ebooks.iospress.nl/doi/10.3233/FAIA220260

[^1_59]: https://arxiv.org/abs/2207.11347

[^1_60]: https://ieeexplore.ieee.org/document/9857212/

[^1_61]: https://ieeexplore.ieee.org/document/10074688/

[^1_62]: https://ieeexplore.ieee.org/document/9872771/

[^1_63]: https://kijoms.uokerbala.edu.iq/home/vol8/iss1/3

[^1_64]: https://www.techscience.com/cmc/v70n3/44938

[^1_65]: https://www.mdpi.com/1424-8220/23/23/9575/pdf?version=1701510364

[^1_66]: https://arxiv.org/pdf/2410.08049.pdf

[^1_67]: http://arxiv.org/pdf/2301.00808.pdf

[^1_68]: http://arxiv.org/pdf/2303.09975.pdf

[^1_69]: https://downloads.hindawi.com/journals/mpe/2022/7313612.pdf

[^1_70]: https://arxiv.org/abs/2202.13560

[^1_71]: https://www.mdpi.com/1424-8220/25/1/261

[^1_72]: https://arxiv.org/pdf/2306.00830.pdf

[^1_73]: https://www.kungfu.ai/blog-post/convnext-a-transformer-inspired-cnn-architecture

[^1_74]: https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/dinat.md

[^1_75]: https://www.deeplearning.ai/the-batch/convnext-v2-the-new-model-family-that-boosts-convnet-performance/

[^1_76]: https://kimjy99.github.io/논문리뷰/dinat/

[^1_77]: https://openaccess.thecvf.com/content/CVPR2025/papers/Gu_ACL_Activating_Capability_of_Linear_Attention_for_Image_Restoration_CVPR_2025_paper.pdf

[^1_78]: https://arxiv.org/abs/2209.15001

[^1_79]: https://proceedings.iclr.cc/paper_files/paper/2024/file/0a4c7cdfc0a4eb1b13bb84a9b6220c37-Paper-Conference.pdf

[^1_80]: https://www.geeksforgeeks.org/computer-vision/convnext/

[^1_81]: https://wikidocs.net/236104

[^1_82]: https://arxiv.org/abs/2505.16157

[^1_83]: https://arxiv.org/pdf/2201.03545.pdf

[^1_84]: https://arxiv.org/html/2510.16325v1

[^1_85]: https://arxiv.org/html/2508.20955v1

[^1_86]: https://arxiv.org/abs/2507.17892

[^1_87]: https://arxiv.org/html/2412.03814v1

[^1_88]: https://arxiv.org/pdf/2508.20955.pdf

[^1_89]: https://www.semanticscholar.org/paper/a883336e5c2e9f46f5012343227a6be4671c9ca0

[^1_90]: https://arxiv.org/html/2510.09107v2

[^1_91]: https://arxiv.org/html/2507.17892v1

[^1_92]: https://arxiv.org/abs/2501.16182

[^1_93]: https://www.emergentmind.com/topics/convnext-tiny-architecture

[^1_94]: https://huggingface.co/docs/transformers/v4.32.0/en/model_doc/dinat

[^1_95]: https://towardsdatascience.com/the-cnn-that-challenges-vit/

[^1_96]: https://huggingface.co/learn/computer-vision-course/unit2/cnns/convnext
