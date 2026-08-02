# Self Forcing: Bridging the Train-Test Gap in Autoregressive Video Diffusion 

---

## 1. Executive Summary (10문장 이내)

Self Forcing은 자기회귀(AR) 비디오 확산 모델의 **노출 편향(Exposure Bias)** 문제를 해결하는 새로운 학습 패러다임이다. 기존 Teacher Forcing(TF)과 Diffusion Forcing(DF)은 훈련 시 정답(ground-truth) 컨텍스트를 사용하지만, 추론 시에는 모델 자신의 생성물을 컨텍스트로 사용하기 때문에 훈련-추론 간 분포 불일치가 발생한다. Self Forcing은 훈련 시에도 KV 캐싱을 활용한 자기회귀 롤아웃을 수행하여, 모델이 자신의 출력물을 컨텍스트로 사용하도록 강제함으로써 이 불일치를 근본적으로 해소한다. 홀리스틱(holistic) 비디오 레벨의 분포 매칭 손실(DMD, SiD, GAN)을 통해 생성된 전체 시퀀스의 품질을 직접 평가한다. 효율적인 구현을 위해 4단계 소수-스텝 확산 모델과 확률적 그래디언트 절단(stochastic gradient truncation) 전략을 채택하여 메모리 효율성을 확보하였다. 추가로, 롤링 KV 캐시(Rolling KV Cache) 메커니즘을 도입하여 $O(TL)$ 복잡도로 임의 길이의 비디오를 생성할 수 있게 하였다. 실험 결과, Self Forcing은 단일 H100 GPU에서 17 FPS의 실시간 비디오 생성(서브 세컨드 지연)을 달성하며, VBench 기준으로 기존 비인과적 확산 모델 대비 동등하거나 우월한 품질을 보인다. NeurIPS 2025에 채택된 이 연구는 게임, 라이브 스트리밍, 로봇공학 등 실시간 인터랙티브 콘텐츠 생성 분야에 직접적인 기여를 한다.

### 1-1. 연구의 목적과 필요성

**목적:**
자기회귀 비디오 확산 모델에서 발생하는 노출 편향 문제를 근본적으로 해결하고, 실시간 스트리밍 영상 생성이 가능한 새로운 훈련 패러다임을 제시한다.

**필요성:**

| 문제 | 설명 |
|------|------|
| 노출 편향 | TF/DF 모델은 훈련 시 정답 컨텍스트, 추론 시 자기 생성물을 사용 → 분포 불일치 |
| 오류 누적 | AR 생성에서 초기 오류가 이후 프레임으로 전파·증폭됨 (p.2) |
| 실시간 불가 | 양방향 어텐션 기반 비디오 확산 모델은 전체 영상을 동시에 생성해야 하므로 스트리밍 불가 (p.1) |
| VQ 품질 손실 | 기존 AR 모델은 lossy vector quantization에 의존하여 시각적 품질 저하 (p.1) |

---

## 2. 핵심 주장과 근거 표

| 주장 | 근거 | 위치 |
|------|------|------|
| Self Forcing이 노출 편향을 근본 해결 | 훈련 중 KV 캐싱 기반 자기회귀 롤아웃 수행으로 훈련-추론 분포 일치 | p.2, Figure 1(c) |
| 홀리스틱 손실이 프레임별 손실보다 우수 | 전체 비디오 시퀀스의 분포를 직접 매칭하여 error accumulation 방지 | p.6, Section 3.3 |
| 계산 효율성이 병렬 방법 대비 우수 | FlashAttention-3 활용 가능, 특수 마스킹 불필요 → 동일 훈련 시간 내 높은 품질 | p.9, Figure 6 |
| 롤링 KV 캐시가 $O(TL)$ 복잡도 달성 | 기존 슬라이딩 윈도우 대비 KV 재계산 불필요 | p.6, Figure 3 |
| Self Forcing이 VBench 최고 성능 달성 | 84.31(chunk-wise), 84.26(frame-wise) Total Score | p.8, Table 1 |
| 실시간 생성 가능 (17 FPS, <1s 지연) | 단일 H100 GPU에서 0.69s 지연 달성 | p.7-8, Table 1 |

---

## 2-1. 해결 문제, 제안 방법, 모델 구조, 성능 및 한계

### 해결하고자 하는 문제

**노출 편향(Exposure Bias):** 모델이 훈련 시에는 정답 컨텍스트 $x^{<i}$에 조건화되지만, 추론 시에는 자신이 생성한 불완전한 $\hat{x}^{<i}$에 조건화되어야 하는 분포 불일치 문제.

$$p_{\text{train}}(x^i | x^{<i}_{\text{GT}}) \neq p_{\text{infer}}(x^i | \hat{x}^{<i}_\theta) \quad \text{(p.2)}$$

### 제안하는 방법 (수식 포함)

**1. 자기회귀 비디오 확산 모델 기초 (p.3)**

$$p(x^{1:N}) = \prod_{i=1}^{N} p(x^i | x^{ < i})$$

각 프레임의 노이즈 추가 과정:

$$x^i_{t_i} = \Psi(x^i, \epsilon^i, t^i) = \alpha_{t_i} x^i + \sigma_{t_i} \epsilon^i, \quad \epsilon^i \sim \mathcal{N}(0, I)$$

기존 프레임별 손실:

$$\mathcal{L}^{\text{DM}}_\theta = \mathbb{E}_{x^i, t^i, \epsilon^i} \left[ w_{t^i} \| \hat{\epsilon}^i_\theta - \epsilon^i \|^2_2 \right]$$

**2. Self Forcing 학습 (p.4-5)**

모델 분포는 암묵적으로 정의됨:

$$p_\theta(x^i | x^{ < i}) := f_{\theta,t_1} \circ f_{\theta,t_2} \circ \cdots \circ f_{\theta,t_T}(x^i_{t_T})$$

여기서:

$$f_{\theta,t_j}(x^i_{t_j}) = \Psi(G_\theta(x^i_{t_j}, t_j, x^{ < i}), \epsilon_{t_{j-1}}, t_{j-1}), \quad x^i_{t_T} \sim \mathcal{N}(0, I)$$

**3. DMD 손실 (p.16)**

역 KL 발산의 그래디언트:

$$\nabla_\theta \mathbb{E}_t [D_{\text{KL}}(p_{\theta,t} \| p_{\text{data},t})] = -\mathbb{E}_{t, \hat{x}_t, \hat{x}} \left[ (s_{\text{real}}(\hat{x}_t, t) - s_{\text{fake}}(\hat{x}_t, t)) \frac{\partial \hat{x}}{\partial \theta} \right]$$

등가 손실 함수:

$$\mathcal{L}_{\text{DMD}}(\theta) = \mathbb{E}_{t,\hat{x}_t,\hat{x}} \left[ \frac{1}{2} \| \hat{x} - \text{sg}[\hat{x} - (f_\psi(\hat{x}_t, t) - f_\phi(\hat{x}_t, t))] \|^2 \right]$$

**4. SiD 손실 (p.17)**

$$\mathcal{L}_{\text{SiD}}(\theta) = \mathbb{E}_{t,\hat{x}_t,\hat{x}} \left[ (f_\phi(\hat{x}_t,t) - f_\psi(\hat{x}_t,t))^T (f_\psi(\hat{x}_t,t) - \hat{x}) + (1-\alpha)\|f_\phi(\hat{x}_t,t) - f_\psi(\hat{x}_t,t)\|^2 \right]$$

$\alpha = 1$일 때 Fisher 발산의 그래디언트와 동치.

**5. GAN 손실 (p.17)**

정규화 항:

$$\mathcal{L}_{\text{reg}} = \frac{1}{2}\mathbb{E}_{t,x_t,\hat{x}_t,\epsilon,\hat{\epsilon}} \left[ \|f_\psi(x_t) - f_\psi(x_t + \sigma\cdot\epsilon)\|^2_2 + \|f_\psi(\hat{x}_t) - f_\psi(\hat{x}_t + \sigma\cdot\hat{\epsilon})\|^2_2 \right]$$

판별자 손실:

$$\mathcal{L}_D(\psi) = -\mathbb{E}_{t,x_t,\hat{x}_t}[\log(\text{sigmoid}(f_\psi(x_t) - f_\psi(\hat{x}_t)))] + \lambda\mathcal{L}_{\text{reg}}$$

생성자 손실:

$$\mathcal{L}_G(\theta) = -\mathbb{E}_{t,x_t,\hat{x}_t}[\log(\text{sigmoid}(f_\psi(\hat{x}_t) - f_\psi(x_t)))]$$

**6. 데이터 예측 모델 (p.16, Appendix A)**

$$G_\theta(x, t, c) = c_{\text{skip}} \cdot \epsilon - c_{\text{out}} \cdot v_\theta(c_{\text{in}} \cdot x_t, c_{\text{noise}}(t'), c)$$

4-step 스케줄: $[t_4, t_3, t_2, t_1] = [1000, 750, 500, 250]$

### 모델 구조

| 구성 요소 | 내용 |
|-----------|------|
| 베이스 모델 | Wan2.1-T2V-1.3B (Flow Matching 기반) |
| 어텐션 | Causal DiT + KV 캐싱 (훈련 및 추론 모두 적용) |
| VAE | Causal 3D VAE (시간 압축) |
| 디노이징 스텝 | 4-step (few-step diffusion) |
| AR 단위 | 프레임별 또는 청크별 (3 latent frames/chunk) |
| 롤링 KV 캐시 | 고정 크기 $L$ 프레임의 최신 KV 임베딩 유지 |
| 손실 함수 | DMD / SiD / GAN (선택 가능) |

### 성능 향상

| 지표 | Self Forcing (chunk) | CausVid | Wan2.1 |
|------|---------------------|---------|--------|
| VBench Total | **84.31** | 81.20 | 84.26 |
| Throughput | **17.0 FPS** | 17.0 FPS | 0.78 FPS |
| Latency | 0.69s | 0.69s | 103s |

*(Table 1, p.8)*

### 한계

- 훈련 컨텍스트 길이를 초과하는 영상 생성 시 품질 저하 (p.10)
- 그래디언트 절단 전략으로 인해 장기 의존성 학습 제한 (p.10)
- 순차적 특성으로 인해 병렬 사전학습 단계에는 직접 적용 어려움

---

## 3. 각 주장의 페이지/Figure/Table 번호

| 주장 | 근거 위치 |
|------|-----------|
| 노출 편향 문제 정의 | p.2, Figure 1 |
| AR 분포 분해 공식 | p.3, Section 3.1 |
| Self Forcing 알고리즘 | p.5, Algorithm 1 |
| 롤링 KV 캐시 효율성 비교 | p.6-7, Figure 3 |
| VBench 정량 비교 | p.8, Table 1 |
| 사용자 선호도 연구 | p.7, Figure 4 |
| 어블레이션 연구 | p.9, Table 2 |
| 훈련 효율성 비교 | p.9-10, Figure 6 |
| 전체 VBench 16개 지표 | p.18, Figure 8 |
| 롤링 KV 캐시 학습 효과 | p.17-18, Figure 7 |
| DMD 손실 수식 | p.16, Eq.(2),(3), Appendix A |
| SiD 손실 수식 | p.17, Eq.(4), Appendix A |
| GAN 손실 수식 | p.17, Eq.(5),(6),(7), Appendix A |

---

## 4. 저자 직접 보고 결과 vs. 해석 분리

### 저자가 직접 보고한 결과

| 항목 | 저자 보고 내용 | 출처 |
|------|-------------|------|
| VBench Total Score | Self Forcing chunk-wise: 84.31, frame-wise: 84.26 | Table 1 |
| 처리량 | 17.0 FPS (chunk-wise), 8.9 FPS (frame-wise) | Table 1 |
| 지연시간 | chunk-wise: 0.69s, frame-wise: 0.45s | Table 1 |
| 사용자 선호도 | CausVid 대비 66.1%, Wan2.1 대비 62.7% 선호 | Figure 4 |
| 훈련 시간 | DMD 기준 64× H100에서 약 1.5시간 수렴 | p.9 |
| 롤링 KV 속도 | 로컬 어텐션 훈련 적용 시 16.1 FPS, 재계산 시 4.6 FPS | p.9 |
| 어블레이션 | SF(DMD) 84.31 > TF 83.58 > DF 82.95 (chunk-wise) | Table 2 |

### 필자의 해석

- **훈련 효율성의 역설:** Self Forcing이 순차적임에도 TF/DF보다 빠른 이유는 FlashAttention-3 사용 가능 여부의 차이가 핵심이며, 이는 특수 어텐션 마스크 구현의 오버헤드가 상당함을 시사한다.
- **CausVid 대비 개선의 본질:** 두 모델이 동일한 베이스 모델(Wan-1.3B)과 DMD를 사용하지만 VBench에서 3.11점 차이가 나는 것은, 분포 매칭의 대상이 되는 분포 자체의 정확성이 얼마나 중요한지를 명확히 보여준다.
- **VBench의 제한:** VBench는 개별 프레임 품질 및 의미 정렬 측면을 잘 측정하지만, 장기 시간적 일관성이나 사용자 인터랙티비티를 직접 측정하지는 않는다.

---

## 5. 통계적으로 취약한 부분과 비교 불가능한 수치

| 항목 | 문제점 |
|------|--------|
| ⚠️ 사용자 선호도 연구 | 각 프롬프트당 단 1명의 사용자만 평가(p.19, Appendix E). 평가자 간 신뢰도(inter-rater reliability) 미보고 |
| ⚠️ 모델 간 해상도 불일치 | Self Forcing(832×480) vs SkyReels-V2(960×540) vs NOVA(768×480) vs Pyramid Flow(640×384) — 해상도가 달라 직접적인 성능 비교에 한계 (Table 1) |
| ⚠️ 파라미터 수 불일치 | MAGI-1은 4.5B, Pyramid Flow는 2B로, 1.3B 모델들과 규모가 달라 공정한 비교 어려움 (Table 1) |
| ⚠️ VBench 평가 프롬프트 일관성 | 일부 베이스라인은 프롬프트 재작성(rewriting) 미지원—이 경우 결과 보고 방식 불명확 (p.16, Appendix A) |
| ⚠️ GAN 배치 크기 불일치 | GAN 훈련은 배치 768, DMD/SiD는 배치 64로 훈련 조건이 다름 (Table 3) |
| ⚠️ 장기 영상 품질 정량 평가 부재 | 롤링 KV 캐시를 활용한 장기 영상 생성의 품질은 정성적으로만 비교(Figure 7) |
| ⚠️ 훈련 데이터 규모 차이 | Self Forcing(DMD/SiD)은 데이터 불필요, GAN은 70k 비디오 사용 — 공정한 비교 조건 불일치 |

---

## 6. 문서가 답하지 않는 질문

| 미답 질문 | 설명 |
|-----------|------|
| 최적 청크 크기 | 3 latent frames/chunk 이외의 청크 크기 탐색 결과 미제공 |
| 스케일링 법칙 | 1.3B 이상의 더 큰 모델에서의 성능 변화 미탐구 |
| 도메인 일반화 | 특정 비디오 도메인(의료, 위성 등)에서의 일반화 능력 미검증 |
| 장기 시간 의존성 한계 정량화 | 훈련 컨텍스트 이상 길이에서 품질이 "얼마나" 저하되는지 정량 데이터 부재 |
| 다른 베이스 모델 적용 가능성 | Wan2.1 이외의 베이스 모델(CogVideoX, HunyuanVideo 등)에의 전이 성능 미검토 |
| 조건부 생성 (이미지→비디오) | 텍스트 조건 외 이미지 조건부 생성 실험 미포함 |
| 온라인 사용자 제어 반응성 | 실시간 인터랙션에서 사용자 입력 반영 지연 측정 부재 |
| 그래디언트 절단 길이의 민감도 | 절단 스텝 수 변화에 따른 성능 변화 분석 부재 |

---

## 7. 가장 중요한 그림 5개 해석

### Figure 1 (p.2) — 훈련 패러다임 비교
세 가지 학습 방식을 직관적으로 비교한다. (a) Teacher Forcing은 모든 컨텍스트가 정답 프레임 $x^1, x^2$이고, (b) Diffusion Forcing은 다양한 노이즈 레벨의 컨텍스트를 사용하지만, 두 방법 모두 생성된 출력이 추론 시 분포와 다르다. (c) Self Forcing은 이전에 자신이 생성한 프레임 $\hat{x}^1, \hat{x}^2$을 컨텍스트로 사용하며, 분포 매칭 손실을 전체 비디오에 적용한다. 이 그림은 연구의 핵심 직관을 가장 명확하게 표현한다.

### Figure 2 (p.4) — 어텐션 마스크 구성
(a) TF와 (b) DF는 특수한 블록-희소(block-sparse) 어텐션 마스크로 인과 의존성을 강제하고 전체 비디오를 병렬로 처리한다. 반면 (c) Self Forcing은 표준 전체 어텐션(full attention)을 사용하며, KV 캐싱을 통해 순차적으로 처리한다. 이것이 FlashAttention-3 활용을 가능하게 하여 훈련 효율성의 역설적 우수성을 설명하는 핵심 구조적 근거다.

### Figure 3 (p.7) — 비디오 외삽 효율성 비교
(a) 양방향 모델: $O(TL^2)$ — KV 캐시 불가, 완전 재계산 필요. (b) 기존 인과 모델: $O(L^2 + TL)$ — 슬라이딩 윈도우 이동 시 KV 재계산 필요. (c) Self Forcing 롤링 KV: $O(TL)$ — 재계산 없이 최신 $L$ 프레임만 유지하며 무한 길이 영상 생성 가능. 이 그림은 계산 복잡도의 질적 차이를 명확히 시각화한다.

### Figure 6 (p.10) — 훈련 효율성 비교
**(좌)** 반복당 훈련 시간: 4-step Self Forcing이 TF/DF 대비 유사하거나 낮은 시간 소요. **(우)** 동일한 벽시계 훈련 시간에서 VBench Total Score가 Self Forcing이 빠르게 상승하며 약 25분 이내에 DF(82.0)와 TF(82.5)를 초월한다. 이는 Self Forcing의 순차적 특성이 실제 학습 효율을 저해하지 않음을 실증적으로 보여준다.

### Figure 8 (Appendix C, p.18) — VBench 16개 지표 레이더 차트
Self Forcing(chunk-wise)이 특히 **semantic alignment** 관련 지표(scene, object class, multiple objects, human action)와 **미적 품질**(aesthetic quality, imaging quality)에서 우수하다. 단, frame-wise 변형은 dynamic degree가 높지만 background consistency, motion smoothness, temporal flickering 지표가 chunk-wise보다 열악함을 보여준다. 이는 AR 롤아웃 스텝이 길어질수록 시간적 일관성 유지가 어려워지는 고유한 트레이드오프를 보여준다.

---

## 8. 결론: 시사점, 후속 연구 계획 및 추가 방향

### 저자가 제시한 시사점

1. **병렬 사전훈련 + 순차 사후훈련 패러다임:** 병렬 훈련(효율)과 순차 추론(정확성) 간의 불일치를 해소하기 위해, 대규모 사전훈련 후 Self Forcing 기반 사후훈련(post-training)을 수행하는 새 패러다임을 제안한다. 이는 언어 모델에서의 강화학습(RLHF/GRPO)과 유사한 방향성이다 (p.10).

2. **AR-Diffusion-GAN의 통합:** 세 패러다임이 상호 배타적이 아니라 보완적이며, 중첩 방식으로 결합할 수 있음을 입증했다 (p.10).

3. **실시간 인터랙티브 응용의 실현:** 라이브 스트리밍, 게임 시뮬레이션, 로봇공학 등 지연에 민감한 응용 분야에서 실용적 비디오 생성이 가능해졌다 (p.2, p.10).

### 저자가 언급한 후속 연구 계획

- 훈련 컨텍스트 길이를 초과한 영상에 대한 외삽 기법 개선
- Mamba, SSM 등 순환 아키텍처와의 통합으로 메모리 효율과 장기 컨텍스트 모델링의 균형 확보 (p.10)

---

### 8-1. 모델의 일반화 성능 향상 가능성

현재 Self Forcing의 일반화 성능과 관련된 주요 제약 및 향상 가능성은 다음과 같다:

| 측면 | 현재 한계 | 향상 방향 |
|------|-----------|-----------|
| 훈련 길이 | 훈련 컨텍스트 이상 영상에서 품질 저하 | 계층적 롤아웃, SSM 기반 무한 컨텍스트 모델링 |
| 그래디언트 절단 | 마지막 디노이징 스텝만 역전파 → 장기 의존성 학습 제한 | 확률적 BPTT 길이 증가, gradient checkpointing |
| 도메인 특화 | Wan2.1 단일 모델에서만 검증 | 다양한 베이스 모델(CogVideoX, HunyuanVideo)로 일반화 실험 필요 |
| 조건부 제어 | 텍스트 조건만 검증 | 이미지 조건, 깊이, 포즈 등 다양한 조건부 신호에서의 일반화 |
| 데이터 다양성 | VidProM 기반 250k 프롬프트 | 저자원 언어, 특수 도메인, 멀티모달 입력에서의 일반화 |

**구체적 향상 전략:**

1. **커리큘럼 Self Forcing:** 초기에는 짧은 시퀀스로 훈련하고 점차 롤아웃 길이를 늘리는 커리큘럼 학습으로 장기 의존성 학습 유도
2. **적응형 KV 캐시 크기:** 콘텐츠의 복잡도에 따라 동적으로 캐시 크기를 조절하는 메커니즘
3. **도메인 어댑터(LoRA 기반):** Self Forcing으로 훈련된 베이스 모델 위에 경량 도메인 어댑터를 추가하여 특수 도메인 일반화
4. **멀티스케일 롤아웃:** 다양한 시간 해상도에서 Self Forcing을 적용하는 계층적 구조

---

### 8-2. 2020년 이후 관련 최신 연구 비교 분석

#### 주요 관련 연구 연표 및 비교

| 연구 | 연도 | 핵심 기여 | Self Forcing과의 관계 |
|------|------|-----------|----------------------|
| VideoGPT [94] | 2021 | VQ-VAE + Transformer AR 비디오 생성 | VQ 기반 품질 한계, Self Forcing으로 극복 |
| Video Diffusion Models [26] | 2022 | 확산 모델의 비디오 적용 | 양방향 어텐션의 원형, SF가 인과 구조로 발전 |
| Diffusion Forcing [8] | 2024 | 프레임별 독립적 노이즈 레벨 | SF의 직접 비교 대상, 노출 편향 미해결 |
| CausVid [100] | 2025 | DF + DMD 기반 AR 비디오 | SF의 가장 가까운 선행연구, 분포 불일치 문제 내포 |
| SkyReels-V2 [10] | 2025 | 양방향 DF 기반 무한 길이 영상 | KV 캐시 미지원으로 효율성 열위 |
| MAGI-1 [69] | 2025 | 대규모 AR 비디오 (4.5B) | 높은 지연시간, SF 대비 품질 열위 |
| NOVA [13] | 2025 | VQ 없는 AR 비디오 생성 | 소규모(0.6B), SF 대비 시맨틱 점수 경쟁력 |
| DeepSeek-R1 [21] | 2025 | LLM에서 RL 기반 사후훈련 | 언어 모델의 병렬 사전훈련+순차 사후훈련 패러다임과 유사 |

#### Self Forcing이 앞으로의 연구에 미치는 영향

1. **비디오 생성의 패러다임 전환 촉진:** "대규모 병렬 사전훈련 → Self Forcing 기반 순차 사후훈련"이라는 2단계 학습 파이프라인이 비디오 생성의 표준 레시피가 될 가능성이 높다.

2. **실시간 인터랙티브 AI 가속화:** 게임 엔진, 자율주행 시뮬레이터, VR/AR 콘텐츠 생성 등에서 실시간 동영상 생성이 실용화될 기반을 마련했다.

3. **연속 데이터 도메인으로의 확장:** 저자가 언급한 바와 같이(p.10), 오디오, 3D 모션, 로봇 궤적 등 순차적 연속 데이터를 다루는 모든 생성 모델에 Self Forcing 원리 적용 가능성이 있다.

4. **분포 매칭 손실 연구 촉진:** DMD, SiD, GAN 모두 Self Forcing 프레임워크 내에서 동등한 성능을 보임으로써, 이들 손실의 통합적 이해와 새로운 분포 매칭 방법 개발에 대한 관심을 높일 것이다.

#### 앞으로 연구 시 고려할 점

| 고려 사항 | 세부 내용 |
|-----------|-----------|
| **지연시간 vs. 품질 트레이드오프 정밀 측정** | 현재 VBench는 정적 프레임 품질 중심 — 시간적 일관성 및 인터랙티비티를 측정하는 새 벤치마크 개발 필요 |
| **사후훈련 데이터 의존성** | GAN 방식은 70k 비디오가 필요하나 DMD/SiD는 데이터 프리 — 데이터 효율적 사후훈련 방법론 심화 연구 필요 |
| **롤아웃 길이와 품질의 정량적 관계** | AR 롤아웃 단계 증가에 따른 성능 저하 곡선을 명확히 측정하고, 이를 이론적으로 분석할 필요 |
| **다중 GPU 스케일링** | 현재 64×H100 환경에서 검증됨 — 소규모 또는 엣지 환경에서의 적용 가능성 연구 필요 |
| **윤리적 안전장치** | 실시간 딥페이크 생성 위험이 증가함에 따라 워터마킹, 생성물 탐지기와의 공동 개발 필요 (p.18-19) |
| **SSM과의 통합** | Mamba [19] 등 상태공간 모델과 Self Forcing을 결합하여 $O(1)$ 상태 업데이트로 장기 의존성 문제 해결 시도 [63] |

---

## 참고 자료

본 분석은 다음 문서를 기반으로 작성되었습니다:

- **주 논문:** Xun Huang, Zhengqi Li, Guande He, Mingyuan Zhou, Eli Shechtman. "Self Forcing: Bridging the Train-Test Gap in Autoregressive Video Diffusion." *arXiv:2506.08009v2*, NeurIPS 2025.
- **프로젝트 페이지:** https://self-forcing.github.io/
- **직접 인용된 주요 참고문헌 (논문 내 번호 기준):**
  - [8] Chen et al., "Diffusion Forcing," NeurIPS 2024
  - [18] Goodfellow et al., "Generative Adversarial Nets," NeurIPS 2014
  - [31] Huang et al., "VBench," CVPR 2024
  - [83] Wang et al., "Wan: Open and Advanced Large-Scale Video Generative Models," arXiv 2025
  - [98,99] Yin et al., "Distribution Matching Distillation (DMD)," CVPR/NeurIPS 2024
  - [100] Yin et al., "CausVid," CVPR 2025
  - [112,113] Zhou et al., "Score Identity Distillation (SiD)," ICML/ICLR 2024/2025
  - [92] Xiao et al., "StreamingLLM (Rolling KV Cache)," ICLR 2024

> **주의:** 본 분석에서 정량적 수치, 수식, Figure/Table 번호는 모두 제공된 원문 PDF(arXiv:2506.08009v2)에 직접 근거하며, 불확실한 내용은 의도적으로 포함하지 않았습니다.
