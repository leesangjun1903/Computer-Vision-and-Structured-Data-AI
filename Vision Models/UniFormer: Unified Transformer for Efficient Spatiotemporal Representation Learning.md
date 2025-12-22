# UniFormer: Unified Transformer for Efficient Spatiotemporal Representation Learning

### 1. 핵심 주장과 주요 기여 요약

**UniFormer**는 ICLR 2022에 발표된 혁신적인 논문으로, 3D 합성곱과 시공간 자가주의(spatiotemporal self-attention)의 장점을 단일 Transformer 포맷으로 통합하는 아키텍처를 제안합니다. 이 논문의 핵심 주장은 다음과 같습니다:

**핵심 문제**: 비디오 이해에는 두 가지 상충하는 과제가 있습니다:
- **로컬 중복성 감소**: 인접 프레임 간의 미묘한 움직임
- **글로벌 의존성 포착**: 장거리 프레임 간의 복잡한 관계

**주요 기여**:

| 기여도 | 내용 |
|--------|------|
| **다중 헤드 관계 집계기 (MHRA)** | 얕은 층에서는 로컬 관계, 깊은 층에서는 글로벌 관계 학습 |
| **동적 위치 임베딩 (DPE)** | 3D depthwise convolution으로 위치 정보 인코딩 |
| **결합 시공간 학습** | 분리된 공간-시간 주의 대신 결합 학습으로 성능 향상 |
| **계층적 구조** | 단계별 로컬-글로벌 전환으로 효율성과 정확도 균형 |

***

### 2. 문제 정의와 제안 방법 (수식 포함)

#### 2.1 핵심 문제 분석

논문에서는 기존 방법들의 한계를 다음과 같이 분석합니다:

**3D CNN의 한계**: 로컬 중복성 감소는 효과적이지만, 제한된 수용장(receptive field)으로 인해 장거리 의존성 포착 불가

**Vision Transformer의 한계**: TimeSformer를 예로 들면, 모든 층에서 전역 토큰-토큰 비교를 수행하므로 로컬 패턴을 배우는 데 과도한 계산 비용 발생

#### 2.2 제안된 방법: UniFormer 블록

**UniFormer 블록의 수식 표현**:

$$X = \text{DPE}(X_{in}) + X_{in} \quad \text{(1)}$$

$$Y = \text{MHRA}(\text{Norm}(X)) + X \quad \text{(2)}$$

$$Z = \text{FFN}(\text{Norm}(Y)) + Y \quad \text{(3)}$$

여기서:
- $X_{in} \in \mathbb{R}^{C \times T \times H \times W}$: 입력 토큰 텐서 (프레임 볼륨)
- DPE: 동적 위치 임베딩
- MHRA: 다중 헤드 관계 집계기
- FFN: 피드포워드 네트워크

#### 2.3 다중 헤드 관계 집계기 (MHRA)

**기본 구조**:

$$R_n(X) = A_n V_n(X) \quad \text{(4)}$$

$$\text{MHRA}(X) = \text{Concat}(R_1(X); R_2(X); \cdots; R_N(X))U \quad \text{(5)}$$

여기서:
- $X \in \mathbb{R}^{L \times C}$ (L = T×H×W): 토큰 수열
- $R_n(\cdot)$: n번째 헤드의 관계 집계기
- $U \in \mathbb{R}^{C \times C}$: 헤드를 통합하는 학습 가능한 매개변수 행렬
- $V_n(X) \in \mathbb{R}^{L \times C/N}$: 선형 변환을 통한 문맥 인코딩
- $A_n \in \mathbb{R}^{L \times L}$: 토큰 친화도 행렬

#### 2.4 로컬 MHRA (얕은 층)

로컬 MHRA는 작은 3D 이웃 영역에서 학습 가능한 매개변수 행렬을 사용합니다:

$$A^{\text{local}}_n(X_i, X_j) = a_{i-j}^n, \quad j \in \Omega^{t \times h \times w}_i \quad \text{(6)}$$

여기서:
- $a_n \in \mathbb{R}^{t \times h \times w}$: 튜브 크기의 관계 매개변수
- $\Omega^{t \times h \times w}_i$: 앵커 토큰 $X_i$ 주변의 작은 3D 이웃 영역

이는 다음과 같이 표현됩니다:

$$A^{\text{local}}_n(X_i, X_j) = a_{[\Delta t, \Delta h, \Delta w]} \quad \text{(6')}$$

여기서 $\Delta t = t_i - t_j, \Delta h = h_i - h_j, \Delta w = w_i - w_j$

**3D Convolution과의 연결**: 로컬 MHRA는 PWConv-DWConv-PWConv 형태로 재해석될 수 있습니다:
- $V_n(\cdot)$: Pointwise Convolution (선형 변환)
- $A^{\text{local}}_n$: Depthwise Convolution (채널별 작동)
- 멀티헤드 통합 $U$: Pointwise Convolution

#### 2.5 글로벌 MHRA (깊은 층)

깊은 층에서는 글로벌 시공간 뷰에서 콘텐츠 유사성을 비교합니다:

$$A^{\text{global}}_n(X_i, X_j) = \frac{e^{Q_n(X_i)^T K_n(X_j)}}{\sum_{j' \in \Omega^{T \times H \times W}} e^{Q_n(X_i)^T K_n(X_j')}} \quad \text{(7)}$$

여기서:
- $Q_n(\cdot), K_n(\cdot)$: 서로 다른 두 선형 변환
- $\Omega^{T \times H \times W}$: 전체 3D 튜브 (모든 토큰)
- 분모: 정규화를 위한 모든 토큰의 합

**핵심 특성**: Transformer의 표준 자가주의와 동일하지만, 공간과 시간을 **분리하지 않음** (TimeSformer와 다른 점)

#### 2.6 동적 위치 임베딩 (DPE)

$$\text{DPE}(X_{in}) = \text{DWConv}(X_{in}) \quad \text{(8)}$$

여기서:
- DWConv: 제로 패딩을 포함한 3D depthwise convolution
- 커널 크기: 3×3×3 (T×H×W)

**이점**:
- 공유 매개변수로 인한 효율성
- 경계의 절대 위치 인식 (zero padding의 효과)
- 임의의 입력 길이에 대한 친화적 구조

***

### 3. 모델 구조와 구체적 설계

#### 3.1 전체 아키텍처

| 단계 | 블록 수 | 채널 수 | MHRA 타입 | 튜브 크기 |
|------|--------|--------|----------|----------|
| **Stage 1** | 3 (S) / 5 (B) | 64 | 로컬 (Local) | 5×5×5 |
| **Stage 2** | 4 (S) / 8 (B) | 128 | 로컬 (Local) | 5×5×5 |
| **Stage 3** | 8 (S) / 20 (B) | 320 | 글로벌 (Global) | - |
| **Stage 4** | 3 (S) / 7 (B) | 512 | 글로벌 (Global) | - |

**헤더 설정**:
- 로컬 MHRA: 헤더 수 = 채널 수 (채널별 집계)
- 글로벌 MHRA: 헤더 크기 = 64 (표준 transformer)

**정규화 선택**:
- 로컬 MHRA: BatchNorm (CNN 유사성)
- 글로벌 MHRA: LayerNorm (Transformer 표준)

#### 3.2 피드포워드 네트워크 (FFN)

$$\text{FFN}(Z) = \text{Linear}_2(\text{GELU}(\text{Linear}_1(Z))) \quad \text{(9)}$$

- 확장 비율: 4 (모든 층에서)
- 비선형 활성화: GELU

***

### 4. 성능 향상 및 실험 결과

#### 4.1 주요 성능 지표

| 데이터셋 | 평가지표 | UniFormer-B | 이전 SOTA | 개선도 |
|---------|---------|------------|---------|--------|
| **Kinetics-400** | Top-1 정확도 | 82.9% | 80.7% (TimeSformer) | +2.2% |
| **Kinetics-600** | Top-1 정확도 | 84.8% | 82.2% (TimeSformer) | +2.6% |
| **Something-Something V1** | Top-1 정확도 | 60.9% | 56.8% (TDNEN) | +4.1% |
| **Something-Something V2** | Top-1 정확도 | 71.2% | 69.6% (Swin-B) | +1.6% |
| **GFLOPs/영상** | 효율성 | 168 (K400 기준) | 7140 (TimeSformer) | **10× 감소** |

#### 4.2 계산 효율성 분석

**ImageNet-1K 사전학습만으로 달성**:

```
UniFormer-B32f (Kinetics-400):
- Top-1 정확도: 82.9%
- GFLOPs: 1036
- 파라미터: 49.8M

vs. ViViT-L (JFT-300M 사전학습):
- Top-1 정확도: 82.8%
- GFLOPs: 17,352 (16.7배 높음)
- 사전학습 데이터셋: 300M 이미지
```

#### 4.3 구조 설계 연구 (Ablation Studies)

**Table 4(a): 로컬/글로벌 단계 구성**

| 구조 | K400 Top-1 | GFLOPs | 분석 |
|------|-----------|--------|------|
| LLLL (모두 로컬) | 81.9% | 31.6 | 글로벌 의존성 부족 |
| LLGG (제안) | **82.9%** | 41.8 | 최적 균형 |
| GGGG (모두 글로벌) | 82.1% | 72.0 | 로컬 특징 부족 |

**결론**: 처음 2단계는 로컬, 마지막 2단계는 글로벌이 최적

#### 4.4 이전 Transformer 대비 개선점

**TimeSformer와의 비교**:

```python
# TimeSformer (기존)
spatial_attn = all_tokens_vs_all_tokens  # 전체 토큰 비교
temporal_attn = all_tokens_vs_all_tokens # 전체 토큰 비교

# UniFormer (제안)
shallow_layers: local_affinity  # 5×5×5 이웃만
deep_layers: global_affinity    # 모든 토큰 (하지만 더 적은 수)
```

**핵심 이점**:
1. 얕은 층에서 로컬 중복성 제거 → 계산량 대폭 감소
2. 깊은 층에서 글로벌 관계 학습 가능
3. 결합 시공간 학습으로 더 판별성 있는 표현

***

### 5. 모델의 일반화 성능 향상 가능성 분석

#### 5.1 전이 학습 성능

**ImageNet → Kinetics-400 전이**:

| 모델 | ImageNet Top-1 | Kinetics-400 Top-1 | 전이 이득 |
|-----|---------------|--------------------|----------|
| UniFormer-S | 82.9% | 80.8% | -2.1% |
| UniFormer-B | 83.9% | 82.9% | -1.0% |
| ViT-Base | 79.9% | - | - |

**크로스-데이터셋 일반화**:

**Table 4(c): 사전학습 데이터셋별 성능**

```
Something-Something V1 (시간 관계 중심):

사전학습 | LLLL | LLGG | GGGG
--------|------|------|------
ImageNet | 49.2% | 52.0% | 비교 불가
Kinetics-400 | 49.2% | 53.8% | 비교 불가
(개선도) | +0.0% | +1.8% | -
```

**핵심 발견**: 
- 결합 시공간(Joint) 학습은 더 큰 데이터셋 사전학습에서 더 큰 이득
- 분리된 주의(Divided)보다 더 나은 전이 학습 성능

#### 5.2 도메인 외(Out-of-Distribution) 성능

**테스트 전략에 따른 성능**:

| 설정 | K400 | Sth-Sth V2 |
|------|------|-----------|
| 단일 클립 테스트 | 79.3% | 65.3% |
| 4클립 테스트 | 82.9% | 71.2% |

**다중 클립/크롭 테스트의 이점**:
- **Kinetics-400** (장면 중심): 멀티-클립 테스트 효과적 (더 많은 프레임 필요)
- **Something-Something** (시간 중심): 멀티-크롭 테스트 효과적 (공간 변동성 필요)

#### 5.3 일반화 능력의 메커니즘

**Grad-CAM 시각화 분석** (Figure 5):

```
LLLL (모두 로컬):
- 콘텐츠: 대체로 모호한 주의
- 한계: 글로벌 맥락 부재
- 특징: 스케이트보드와 풋볼 모두 놓침

GGGG (모두 글로벌):
- 콘텐츠: 비객체 영역에 과도한 주의
- 한계: 로컬 특징 부족
- 결과: 주요 객체 중복

LLGG (제안):
- 콘텐츠: 스케이트보드, 풋볼에 정확한 집중
- 강점: 로컬-글로벌 협력 효과
- 성능: 더 판별적 표현
```

***

### 6. 한계와 개선 가능성

#### 6.1 현재 한계

| 한계 | 설명 | 영향도 |
|------|------|--------|
| **하이브리드 최적화** | CNN-Transformer 혼합으로 인한 최적화 복잡성 | 높음 |
| **배치 정규화 선택** | 로컬(BN) vs 글로벌(LN)의 불일치 | 중간 |
| **튜브 크기 민감도** | 5×5×5 튜브 크기에 대한 강건성 검증 필요 | 낮음 |
| **데이터 의존성** | 더 큰 사전학습 데이터셋에서 더 나은 성능 | 중간 |

#### 6.2 개선 가능성

**1) 동적 로컬-글로벨 전환**
```python
# 현재: 고정된 단계별 전환
if layer <= 2:
    use_local_mhra()
else:
    use_global_mhra()

# 개선 제안: 데이터에 따른 동적 전환
attention_weight = learned_layer_routing()
output = attention_weight * local_mhra + (1-attention_weight) * global_mhra
```

**2) 위계적 토큰화**
```
현재:
- 선형 패칭: 모든 패치가 동일 중요도

개선:
- 계층적 패칭: 중요한 패치에 더 많은 계산
- 적응적 토큰 드롭: 중복된 영역에서 토큰 제거
```

***

### 7. 앞으로의 연구 영향 및 고려 사항

#### 7.1 논문이 미치는 영향

**학문적 영향**:

1. **Transformer 디자인 철학 변화**
   - 비디오: 단순 자가주의만으로는 불충분
   - 로컬 정보 처리의 중요성 재인식
   - 귀납적 편향(Inductive Bias) 재평가

2. **효율성-정확성 트레이드오프**
   - 10배 FLOPs 감소 달성 (168 vs 1680 GFLOPs)
   - 하이브리드 접근의 실용성 증명

3. **계층적 설계의 일반성**
   - 로컬-글로벌 패턴이 다양한 도메인에 적용 가능
   - 이후 MViT, VideoSwin 등의 영감

#### 7.2 이후 연구에 미친 직접적 영향

| 후속 연구 | 발표연도 | 주요 기여 | UniFormer와의 연관 |
|---------|--------|---------|-------------------|
| **VideoMAE** | 2022 | 마스크 오토인코딩 기반 사전학습 | 효율적 사전학습 방법 추가 |
| **MViTv2** | 2023 | 다중 스케일 계층적 주의 | 로컬-글로벌 개념 강화 |
| **InternVideo** | 2023 | 비전-언어 기초 모델 | 멀티모달 확장 |
| **EventSTU** | 2025 | 이벤트 기반 효율적 이해 | 계산 효율성 극대화 |

#### 7.3 향후 연구 시 고려할 점

**1) 도메인 특화 설계**
```
비디오 유형별 최적 구조:
- 액션 인식: 시간 의존성 높음 → 글로벌 MHRA 더 많이
- 객체 추적: 공간 관계 중요 → 로컬 MHRA 강화
- 장면 이해: 균형 필요 → LLGG 구조 유지
```

**2) 적응적 계산**
```python
# 제안: 토큰의 중요도에 따른 적응적 처리
class AdaptiveUniFormer:
    def forward(self, x):
        importance_scores = self.compute_importance(x)
        local_features = self.local_mhra(x)
        global_features = self.global_mhra(x)
        
        # 중요도에 따라 로컬/글로벌 혼합
        return self.adaptive_fusion(
            local_features, global_features, importance_scores
        )
```

**3) 멀티태스크 학습**
```
제안: 공동 사전학습
- 행동 인식 + 객체 검출 + 시맨틱 분할
- UniFormer 백본의 다양한 작업 전이 성능 향상
```

**4) 장시간 비디오 처리**
```
현재 한계: 수십 프레임 (≤1초)
개선 방향: 분단적 처리 → 계층적 연결
- 로컬 클립별 처리 (UniFormer)
- 클립 간 글로벌 관계 학습 (추가 모듈)
```

***

### 8. 2020년 이후 관련 최신 연구 비교 분석

#### 8.1 시공간 주의 메커니즘 비교

| 모델 | 발표 | 공간-시간 처리 | 효율성 | 성능 (K400) |
|------|------|---------------|--------|-----------|
| **TimeSformer** | 2021.02 | 분리된 주의 | 중간 | 80.7% |
| **ViViT** | 2021.03 | 인수분해 주의 | 낮음 | 80.6% |
| **UniFormer** | 2022.02 | 로컬-글로벌 | **높음** | **82.9%** |
| **VideoSwin** | 2021.06 | 윈도우 기반 | 높음 | 78-80% |
| **MViT** | 2021.04 | 계층적 풀링 | 높음 | 80.2% |

#### 8.2 효율성 비교 (GFLOPs/영상)

**Figure 2 재현**:

```
효율성-정확성 Pareto 최적선:

정확도 (%)
  83 |                            ◆ UniFormer-B
     |
  82 |            ◆ UniFormer-S        
     |       
  81 |    ◆ MViT        
     | ◆ Video Swin           
  80 |    ▲ TimeSformer (IN-21K pre)
     |
  79 |
     |______________|______________|__
        100         1000         10000
           GFLOPs/Video (로그 스케일)
```

**분석**:
- **로그 스케일에서 상단 좌측 우월**: UniFormer가 명확히 우수
- **ImageNet-1K만으로** IN-21K 사전학습 모델 능가
- 10배 FLOPs 감소로도 더 높은 정확도

#### 8.3 사전학습 전략 비교

**표: 사전학습 전략별 성능**

| 전략 | 모델 | 데이터 크기 | K400 Top-1 | 학습 비용 |
|------|------|-----------|-----------|----------|
| **감독학습** | ViViT-L | JFT-300M | 82.8% | **높음** |
| **감독학습** | Swin-B | IN-21K | 82.7% | 높음 |
| **자감독학습** | UniFormer-B | IN-1K only | **82.9%** | **낮음** |
| **자감독학습** | VATT | HowTo100M | 82.1% | 중간 |

**의미**: 자감독학습과 효율적 아키텍처로 대규모 감독학습 수준 달성

#### 8.4 도메인 일반화 성능

**최신 연구 (2023-2025)**:

1. **DA-ViT** (Cho et al., 2023)
   - 도메인 적응에 특화된 Vision Transformer
   - UniFormer와 결합 시 더 나은 도메인 이전 가능성

2. **DGMamba** (2024)
   - State Space Model로 도메인 일반화
   - O(N) 복잡도로 계산 효율성 향상

3. **EventSTU** (2025)
   - 이벤트 카메라 기반 효율적 비디오 이해
   - UniFormer 개념 활용하여 3배 FLOPs 감소

#### 8.5 멀티모달 확장

**VATT (Akbari et al., 2021)**:
```
비디오-오디오-텍스트 통합 학습
- DropToken: 50% 토큰 드롭 → 4배 FLOPs 감소
- 자감독학습으로 K400: 82.1% (당시 SOTA)
- MultiModal은 별도 논문이지만 효율성 기법 공유
```

***

### 9. 결론 및 향후 방향

#### 9.1 UniFormer의 위치

**2022년 시점**:
- **과거의 한계**: 3D CNN은 효율적이지만 제한적 수용장, 순수 Transformer는 비효율적
- **UniFormer의 해결책**: 로컬-글로벌 혼합으로 둘 다의 장점 활용
- **성과**: 10배 효율성 향상 + 새로운 SOTA 달성

**2024-2025 현재**:
- 기본 개념은 표준화됨 (MViT, Swin 등에서도 채택)
- 효율성 추구는 계속됨 (이벤트 기반 비전, Mamba 등)
- 멀티모달, 장시간 비디오로 확장 중

#### 9.2 추천 향후 연구 방향

**단기 (1-2년)**:
1. 적응적 로컬-글로벌 라우팅
2. 장시간 비디오 처리 (수분 길이)
3. 약한 감독학습 결합

**중기 (2-5년)**:
1. 비전-언어-오디오 멀티모달 통합
2. 도메인별 특화 모델 설계
3. 온디바이스 배포 최적화

**장기 (5년 이상)**:
1. 신경형 하드웨어 활용
2. 자동 아키텍처 설계 (AutoML)
3. 연속학습 (continual learning) 통합

***

## 참고문헌 (선택)

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/a1cc625f-afe3-4e21-9144-0971f1902c90/2201.04676v3.pdf)
[2](https://www.mdpi.com/2227-7102/14/8/826)
[3](https://www.researchprotocols.org/2024/1/e53191)
[4](https://www.semanticscholar.org/paper/87fcbe9ebb9730c59fd1d27a20fed2be139a75e0)
[5](https://www.mdpi.com/1424-8220/23/18/7938)
[6](https://ojs.bonviewpress.com/index.php/IJCE/article/view/2702)
[7](https://dom.lndb.lv/data/obj/1387745.html)
[8](https://www.researchprotocols.org/2025/1/e66975)
[9](https://www.psychiatrist.com/jcp/detecting-tardive-dyskinesia-using-video-based-artificial-intelligence/)
[10](https://net.ageditor.uy/index.php/net/article/view/110)
[11](https://www.mdpi.com/2072-4292/17/23/3805)
[12](https://arxiv.org/html/2411.00630)
[13](https://arxiv.org/pdf/2102.05095.pdf)
[14](http://arxiv.org/pdf/2101.08833.pdf)
[15](https://arxiv.org/html/2311.18825v2)
[16](https://arxiv.org/pdf/2201.04676.pdf)
[17](http://arxiv.org/pdf/2405.08204.pdf)
[18](https://arxiv.org/pdf/2108.09635.pdf)
[19](http://arxiv.org/pdf/2412.09828.pdf)
[20](https://openaccess.thecvf.com/content/WACV2023/papers/Ahn_STAR-Transformer_A_Spatio-Temporal_Cross_Attention_Transformer_for_Human_Action_Recognition_WACV_2023_paper.pdf)
[21](https://arxiv.org/pdf/2209.07474.pdf)
[22](https://www.cse.scu.edu/~yliu1/papers/A_Hybrid_Transformer-LSTM_Model_With_3D_Separable_Convolution_for_Video_Prediction.pdf)
[23](http://arxiv.org/pdf/2207.05526.pdf)
[24](https://openaccess.thecvf.com/content/ICCV2021/papers/Arnab_ViViT_A_Video_Vision_Transformer_ICCV_2021_paper.pdf)
[25](https://www.academia.edu/124930841/A_Hybrid_Transformer_LSTM_Model_With_3D_Separable_Convolution_for_Video_Prediction)
[26](https://www.sciencedirect.com/science/article/abs/pii/S0950705124014072)
[27](https://www.labellerr.com/blog/hands-on-with-vision-transformers-in-video-classification/)
[28](https://arxiv.org/pdf/2508.06528.pdf)
[29](http://www.arxiv.org/abs/2411.00630)
[30](https://openaccess.thecvf.com/content/CVPR2024/papers/Gritsenko_End-to-End_Spatio-Temporal_Action_Localisation_with_Video_Transformers_CVPR_2024_paper.pdf)
[31](https://openaccess.thecvf.com/content/CVPR2022/papers/Yang_Temporally_Efficient_Vision_Transformer_for_Video_Instance_Segmentation_CVPR_2022_paper.pdf)
[32](https://arxiv.org/html/2502.17863v1)
[33](https://openaccess.thecvf.com/content/ACCV2022/papers/Li_HaViT_Hybrid-attention_based_Vision_Transformer_for_Video_Classification_ACCV_2022_paper.pdf)
[34](https://arxiv.org/html/2402.13729v1)
[35](https://arxiv.org/html/2411.00630v1)
[36](https://arxiv.org/html/2404.06243v1)
[37](https://arxiv.org/html/2502.11168v1)
[38](https://www.emergentmind.com/topics/video-transformer-architecture)
[39](https://arxiv.org/abs/2410.23907)
[40](https://ieeexplore.ieee.org/document/10542106/)
[41](https://ieeexplore.ieee.org/document/10410871/)
[42](https://ieeexplore.ieee.org/document/10800516/)
[43](https://arxiv.org/abs/2412.09439)
[44](https://arxiv.org/abs/2407.12753)
[45](https://www.semanticscholar.org/paper/b021962b5ecd1fe2d94b5488ec0ed99004b8585a)
[46](https://www.semanticscholar.org/paper/86d5929b8ee9d8970e69df0719ea95b5961e5f93)
[47](https://ijarsct.co.in/Paper29737.pdf)
[48](https://www.semanticscholar.org/paper/c029a01aaeb84983f183114a2bf9d4fd10039f1c)
[49](https://arxiv.org/html/2403.09394v1)
[50](http://arxiv.org/pdf/2205.13535.pdf)
[51](https://arxiv.org/pdf/2404.04452.pdf)
[52](http://arxiv.org/pdf/2305.13311.pdf)
[53](https://www.mdpi.com/1424-8220/23/7/3447/pdf?version=1680001445)
[54](https://arxiv.org/abs/2104.11178)
[55](http://arxiv.org/pdf/2404.07794.pdf)
[56](http://arxiv.org/pdf/2305.17455.pdf)
[57](https://scholarworks.bwise.kr/cau/bitstream/2019.sw.cau/68695/1/Domain-Adaptive%20Vision%20Transformers%20for%20Generalizing%20Across%20Visual%20Domains.pdf)
[58](https://arxiv.org/abs/2504.12027)
[59](https://www.sciencedirect.com/science/article/abs/pii/S0957417425018615)
[60](https://eusipco2025.org/wp-content/uploads/pdfs/0000641.pdf)
[61](http://www.jdl.link/doc/2011/20221226_STAM%20A%20SpatioTemporal%20Attention%20based%20Memory.pdf)
[62](https://pmc.ncbi.nlm.nih.gov/articles/PMC12349062/)
[63](https://openaccess.thecvf.com/content/WACV2023/papers/Yang_TVT_Transferable_Vision_Transformer_for_Unsupervised_Domain_Adaptation_WACV_2023_paper.pdf)
[64](https://openaccess.thecvf.com/content/CVPR2024/papers/Son_CSTA_CNN-based_Spatiotemporal_Attention_for_Video_Summarization_CVPR_2024_paper.pdf)
[65](https://pmc.ncbi.nlm.nih.gov/articles/PMC11933460/)
[66](https://arxiv.org/html/2504.12027v2)
[67](https://arxiv.org/pdf/2308.03340.pdf)
[68](https://arxiv.org/html/2408.07675v2)
[69](https://arxiv.org/html/2510.26027v1)
[70](https://arxiv.org/html/2512.09579v1)
[71](https://arxiv.org/html/2505.24346v1)
[72](https://arxiv.org/html/2511.18920v1)
[73](https://arxiv.org/abs/2503.16546)
[74](https://arxiv.org/html/2404.04452v2)
[75](https://pmc.ncbi.nlm.nih.gov/articles/PMC12572492/)
[76](https://researchain.net/archives/pdf/Is-Space-Time-Attention-All-You-Need-For-Video-Understanding-1931067)
