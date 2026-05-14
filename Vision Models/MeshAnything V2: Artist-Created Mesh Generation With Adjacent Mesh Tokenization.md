
# MeshAnything V2: Artist-Created Mesh Generation With Adjacent Mesh Tokenization

> **논문 정보**
> - **제목**: MeshAnything V2: Artist-Created Mesh Generation With Adjacent Mesh Tokenization
> - **저자**: Yiwen Chen, Yikai Wang, Yihao Luo, Zhengyi Wang, Zilong Chen, Jun Zhu, Chi Zhang, Guosheng Lin
> - **소속**: Nanyang Technological University, Shengshu, Tsinghua University, Imperial College London, Westlake University
> - **arXiv**: [2408.02555](https://arxiv.org/abs/2408.02555) (v1: 2024.08.05, v3: 2024.12.01)
> - **학회**: ICCV 2025 채택
> - **프로젝트 페이지**: https://buaacyw.github.io/meshanything-v2/
> - **GitHub**: https://github.com/buaacyw/MeshAnythingV2

---

## 1. 핵심 주장 및 주요 기여 요약

### 1.1 배경과 핵심 주장

메시(mesh)는 게임, 영화, 가상현실 등 다양한 산업에서 지배적인 3D 표현 방식이지만, 수십 년간 인간 아티스트가 수작업으로 제작해 왔으며 이는 매우 시간 소모적이고 노동 집약적인 과정이다.

최근 메시를 오토리그레시브(Autoregressive) 방식으로 생성하는 연구들이 등장하였다. 이 접근법은 메시를 정점(vertex)들로 구성된 시퀀스로 처리하고, 언어 모델이 텍스트를 생성하듯 정점 단위로 생성한다. 이 방법들은 일정한 성과를 거두었지만 복잡한 메시 생성에는 여전히 어려움을 겪고 있으며, 그 주된 원인은 비효율적인 토크나이제이션(tokenization) 방법에 있다.

**핵심 주장**: 이 문제를 해결하기 위해 MeshAnything V2를 소개하며, 이 모델의 핵심 혁신은 Adjacent Mesh Tokenization(AMT)이다. 기존 방식이 각 면(face)을 세 개의 정점으로 표현하는 것과 달리, AMT는 가능한 경우 단 하나의 정점만을 사용함으로써 토큰 시퀀스 길이를 평균적으로 약 절반으로 줄인다. 이를 통해 토크나이제이션 과정이 간소화되고 더 압축적이고 잘 구조화된 시퀀스가 생성되며, 결과적으로 연산 비용을 높이지 않고도 이전 모델 대비 페이스(face) 한도를 두 배로 확장하는 데 성공했다.

### 1.2 주요 기여 목록

| 기여 항목 | 내용 |
|---|---|
| **AMT** | 새로운 메시 토크나이제이션 방법 |
| **Face Count Condition** | 면 개수 제어 기능 추가 |
| **Masking Invalid Predictions** | 추론 시 유효하지 않은 토큰 방지 |
| **Point Encoder 업데이트** | 복잡한 메시 처리를 위한 인코더 고도화 |
| **성능 향상** | CD, Edge CD, Normal Consistency 등 지표 개선 |

---

## 2. 상세 설명

### 2.1 해결하고자 하는 문제

메시 토크나이제이션이 오토리그레시브 메시 생성에 미치는 영향은 두 가지 측면으로 나눌 수 있다. 첫 번째는 **효율성**: 더 짧고 압축된 토큰 시퀀스로 메시를 표현하면 컨텍스트 길이가 줄어들어 메모리와 계산 복잡도가 감소한다. 두 번째는 **토큰 시퀀스의 규칙성**: 짧은 시퀀스가 항상 더 좋은 것은 아니며, 시퀀스의 규칙성과 패턴 일관성이 효과적인 시퀀스 학습에 핵심적이다. 결과적으로 효과적인 메시 토크나이제이션은 효율성과 규칙성 모두를 균형 있게 갖춰야 한다.

현재 메시 추출 방법들이 생성한 메시는 아티스트가 만든 메시(AM)에 비해 크게 열등하다. 구체적으로, 현재 방법들은 촘촘한 면에 의존하며 기하학적 특징을 무시하여 비효율성, 복잡한 후처리, 낮은 표현 품질을 초래한다.

---

### 2.2 제안하는 방법 (Adjacent Mesh Tokenization, AMT)

#### 2.2.1 핵심 알고리즘 직관

AMT는 Artist-Created Mesh(AM) 생성을 위한 새로운 토크나이제이션 방법이다. AMT는 각 면을 가능한 경우 단 하나의 정점으로 표현하여, 메시를 더 압축적이고 잘 구조화된 토큰 시퀀스로 처리한다. 현재 설명은 삼각형 메시를 기준으로 하지만, AMT는 가변 다각형 메시 생성으로도 쉽게 일반화될 수 있다.

**기존 방식 vs AMT 방식 비교**:

기존 모든 방법들은 메시를 면 시퀀스로 처리하고 하나의 면을 세 개의 정점으로 표현하여 매우 중복된 표현을 만든다. 이와 달리 AMT는 인접한 면을 공유 정점 하나만으로 표현하여 더 압축적이고 잘 구조화된 메시 표현을 제공함으로써 메시 생성의 효율성과 성능을 크게 향상시킨다.

#### 2.2.2 AMT 수식 표현

AMT의 핵심 아이디어를 수식으로 정리하면 다음과 같다.

**기존 표준 토크나이제이션:**

삼각형 면 $f_i$를 세 정점 $v_{i,1}, v_{i,2}, v_{i,3}$으로 표현:

$$\mathcal{T}_{\text{prev}} = [v_{1,1}, v_{1,2}, v_{1,3},\ v_{2,1}, v_{2,2}, v_{2,3},\ \ldots,\ v_{N,1}, v_{N,2}, v_{N,3}]$$

면 $N$개에 대해 총 토큰 길이 = $3N$ (각 정점당 3 토큰이면 $9N$)

**AMT:**

인접한 면 $f_i$와 $f_{i+1}$이 두 정점 $v_\text{shared}$를 공유할 경우, 새 정점 $v_\text{new}$ 하나만 추가:

$$\mathcal{T}_{\text{AMT}} = [v_{f_1,1}, v_{f_1,2}, v_{f_1,3},\ v_{f_2,\text{new}},\ v_{f_3,\text{new}},\ \ldots]$$

인접 면이 없을 경우 특수 토큰 `&`를 삽입:

```math
\mathcal{T}_{\text{AMT}} = [\ldots,\ \&,\ v_{f_k,1}, v_{f_k,2}, v_{f_k,3},\ \ldots]
```

인접 면을 찾을 수 없을 경우, AMT는 이 이벤트를 표시하기 위해 특수 토큰 `&`를 시퀀스에 추가하고, 아직 인코딩되지 않은 면에서 재시작한다. 이상적으로, AMT가 면당 하나의 정점만 사용하기 때문에 시퀀스 길이는 약 3분의 1로 줄어들 수 있다.

평균 토큰 감소율:

$$\text{Compression Ratio} \approx \frac{|\mathcal{T}_{\text{AMT}}|}{|\mathcal{T}_{\text{prev}}|} \approx 0.5$$

Objaverse에서의 실험 결과, AMT는 평균적으로 시퀀스 길이를 절반으로 줄이며, 어텐션 블록의 계산 부하와 메모리 사용량을 거의 4배 감소시킨다.

#### 2.2.3 포지셔널 임베딩 설계

Transformer의 AMT 시퀀스 패턴 학습을 용이하게 하기 위해, 절대 위치 인코딩 외에 추가적인 임베딩을 설계한다: 면을 세 정점으로 표현할 때는 세 새 정점에 특정 임베딩을 추가하고, 단일 정점으로 표현할 때는 그 단일 정점에 다른 임베딩을 추가한다. 그리고 `&` 토큰에도 별도의 임베딩을 제공한다.

#### 2.2.4 학습 목적 함수 (Objective)

MeshAnything V2는 주어진 형상에 정렬된 AM을 생성하는 것을 목표로 하며, 학습 목적 분포는 다음과 같다:

$$p(\mathcal{M} \mid \mathcal{S})$$

여기서 $\mathcal{M}$은 메시, $\mathcal{S}$는 형상 조건(포인트 클라우드)이다.

이 분포는 동일한 크기와 구조의 decoder-only transformer로 학습된다. $\mathcal{S}$를 transformer에 주입하기 위해 사전 학습된 포인트 클라우드 인코더로 고정 길이 토큰 시퀀스 $\mathcal{T}_S$로 인코딩하고, 이를 transformer 토큰 시퀀스의 prefix로 설정한다.

Cross-entropy 손실로 transformer를 학습한다:

$$\mathcal{L} = -\sum_{t=1}^{T} \log p\!\left(\mathcal{T}_{M}^{(t)} \mid \mathcal{T}_S,\ \mathcal{T}_{M}^{(1:t-1)}\right)$$

---

### 2.3 모델 구조

전체 파이프라인은 다음과 같이 구성된다:

```
입력 (3D Asset)
     ↓
포인트 클라우드 샘플링 (8,192 points)
     ↓
[Michelangelo Point Encoder]  ← 학습 중 업데이트
     ↓
T_S (고정 길이 포인트 클라우드 토큰 시퀀스)
     ↓
AMT로 변환된 메시 토큰 T_M과 concatenation
     ↓
[Decoder-only Transformer (GPT 계열)]
     ↓ (Autoregressive Generation)
AMT 토큰 시퀀스 생성
     ↓
DetokenizationAMT → 최종 Artist-Created Mesh
```

복잡한 메시를 수용하기 위해 포인트 클라우드당 8,192개의 포인트를 샘플링하며, 최대 1,600개 면까지의 복잡한 메시를 처리하기 위해 포인트 인코더를 학습 중에 업데이트한다. MeshAnything V2는 32개의 A800 GPU로 4일간 학습된다.

#### Face Count Condition

일부 응용 분야에서 면 개수에 대한 제어가 필요하다는 점을 고려하여, 메시 생성에 면 개수 조건을 추가하는 기능을 탐구했다. 허용되는 최대 면 개수와 같은 크기의 임베딩 북을 초기화하여 이를 구현했다.

#### Masking Invalid Predictions

MeshAnything V2의 추가적인 특징은 추론 시 유효하지 않은 토큰 시퀀스 생성을 방지하는 Masking Invalid Predictions을 통합하여 구조적으로 잘못된 메시 생성을 막는다.

---

### 2.4 성능 향상

MeshAnything V2는 지원하는 메시의 최대 면 수를 800개에서 1,600개로 두 배로 확장하였으며, 실험 결과 Chamfer Distance(CD), Edge Chamfer Distance(Edge CD), Normal Consistency 등 핵심 지표에서 유의미한 개선을 보였다.

Objaverse에서의 실험은 AMT가 평균적으로 시퀀스 길이를 절반으로 줄이고 어텐션 블록의 계산 부하와 메모리 사용을 거의 4배 감소시킴을 입증했다. 또한 AMT의 압축적이고 잘 구조화된 토큰 시퀀스 덕분에 모델 성능도 향상되었다. 더불어 AMT는 무조건적(unconditional) 또는 다른 조건부 메시 생성 설정에도 적용 가능하며, VQ-VAE 사용 여부에 관계없이 효과적이다.

---

### 2.5 한계점

1. **최대 면 수 제한**: MeshAnything은 1,600개 미만의 면을 가진 메시로 학습되어 1,600개 이상의 메시를 생성할 수 없다.

2. **입력 형상 품질 의존성**: 입력 메시의 형태는 충분히 선명해야 하며, 그렇지 않으면 1,600개 면만으로 표현하기 어렵다. 따라서 피드포워드 3D 생성 방법은 형상 품질 부족으로 인해 좋지 않은 결과를 낼 수 있으며, 3D 재구성, 스캐닝, SDS 기반 방법(DreamCraft3D 등) 또는 Rodin의 결과를 입력으로 사용하는 것을 권장한다.

3. **시퀀스 길이 한계**: MeshAnythingV2와 EdgeRunner는 기본 메시 시퀀스를 압축하는 향상된 토크나이제이션 방법을 제안하지만, 이 방법들도 상대적으로 긴 시퀀스로 변환되어 고폴리 메시에 대한 생성 모델의 학습 능력을 제한한다. 메시 생성 확장을 위해서는 더 압축적인 표현이 필요하다.

4. **계산 비용**: A6000 GPU에서 메시 하나를 생성하는 데 약 8GB 메모리와 45초가 소요된다(생성되는 메시의 면 수에 따라 다름).

---

## 3. 일반화 성능 향상 가능성

### 3.1 형상 조건 기반의 일반화

MeshGPT와 같이 직접 AM을 생성하는 방법에 비해, MeshAnything의 접근법은 복잡한 3D 형상 분포를 학습하는 것을 회피한다. 대신 최적화된 토폴로지를 통해 효율적으로 형상을 구성하는 데 집중하여 학습 부담을 크게 줄이고 확장성을 높인다. 이를 통해 다양한 3D 에셋 생성 방법과의 통합이 가능해진다.

기존 AM 생성 연구들은 소수의 카테고리에 국한된 반면, MeshAnything의 방법은 일반적인 형상에 대해 작동하는 것을 목표로 한다.

### 3.2 다양한 파이프라인과의 통합

MeshAnything V2는 주어진 형상에 정렬된 AM을 생성하는 오토리그레시브 트랜스포머이며, 다양한 3D 에셋 생산 파이프라인에 통합될 수 있어 고품질의 높은 제어 가능성을 가진 AM 생성을 달성할 수 있다.

### 3.3 AMT의 범용 적용 가능성

AMT는 무조건적 또는 다른 조건부 메시 생성 설정에도 적용 가능하며, VQ-VAE 사용 여부에 의해 그 효과가 영향을 받지 않는다.

AMT는 삼각형 메시를 기반으로 설명되지만, 가변 다각형의 메시 생성으로도 쉽게 일반화될 수 있다.

### 3.4 포인트 인코더 업데이트에 의한 일반화

이전 버전과 달리, 최대 1,600개 면을 가진 복잡한 메시를 처리하기 위한 정확도 부족 문제를 해결하기 위해 학습 중 포인트 인코더를 업데이트한다. 이는 단순히 복잡도를 높이는 것을 넘어 다양한 구조의 3D 형상에 대한 일반화 성능을 높이는 데 기여한다.

### 3.5 다양한 입력 조건 지원

더욱 정교한 신경망 구조와 다양한 데이터셋의 통합이 모델의 견고성과 성능을 더욱 향상시킬 수 있다.

또한, 포인트 클라우드를 조건 입력으로 활용함으로써 이미지, NeRF, SDF, Gaussian Splatting 등 다양한 3D 표현으로부터 파이프라인 연결이 가능하며, 이는 실질적인 일반화 범위를 확장한다.

---

## 4. 2020년 이후 관련 최신 연구 비교 분석

### 4.1 비교표

| 방법 | 연도 | 핵심 방식 | 최대 Face 수 | 조건 입력 | 범주 일반화 |
|---|---|---|---|---|---|
| **PolyGen** | 2020 | 두 개의 Transformer (vertex + face) | 소규모 | Class label | ❌ 제한적 |
| **MeshGPT** | 2023 | VQ-VAE + Autoregressive Transformer | ~800 | 없음 | ❌ ShapeNet 한정 |
| **MeshAnything V1** | 2024.06 | VQ-VAE + Point cloud conditioned Transformer | ~800 | Point cloud | ✅ Objaverse |
| **MeshAnything V2** | 2024.08 | AMT + Point cloud conditioned Transformer | **1,600** | Point cloud + face count | ✅ Objaverse |
| **EdgeRunner** | 2024.09 | Auto-regressive Auto-encoder | ~4,000 | 다양 | ✅ |
| **BPT (Blocked & Patchified Tokenization)** | 2024.11 | 블록 기반 압축 토크나이제이션 | 5,000+ | 다양 | ✅ |
| **Meshtron** | 2024.12 | Hourglass Transformer + truncation 학습 | ~16,000 | Point cloud | ✅ |

PolyGen은 정점과 면 생성을 위한 두 가지 오토리그레시브 모델을 사용했고, MeshGPT는 시퀀스를 단축하는 VQ-VAE를 결합했다. MeshAnything은 포인트 클라우드 인코더를 추가해 조건부 생성 기능을 MeshGPT에 도입했으며, MeshAnythingV2는 VQ-VAE를 더 효율적이고 손실 없는 메시 압축 알고리즘으로 대체하여 최대 1,600개 면까지 지원을 확장했다.

MeshAnything V2와 EdgeRunner는 각각 최대 면 수를 기존 800에서 1,600개와 4,000개로 늘렸으며, BPT와 TreeMeshGPT는 기하학 압축과 정점 압축을 최적화하여 5,000–8,000개 면을 지원하고, Meshtron은 hourglass transformer를 제안하여 최대 16,000개까지 확장했다.

그러나 MeshAnythingV2와 EdgeRunner를 포함한 이 방법들도 여전히 상대적으로 긴 시퀀스로 변환되어 고폴리 메시에서의 생성 모델 학습 능력을 제한하며, 메시 생성 확장을 위해서는 더욱 압축적인 표현이 요구된다.

Meshtron은 크게 확장된 컨텍스트 길이 덕분에 MeshAnythingV2보다 복잡한 형상을 훨씬 잘 처리한다.

---

## 5. 연구 영향 및 앞으로의 고려 사항

### 5.1 이 논문이 앞으로의 연구에 미치는 영향

#### 5.1.1 토크나이제이션 설계 패러다임의 전환

AMT는 메시 생성의 효율성과 성능을 크게 향상시키며, 메시 토크나이제이션 방법이 생성하는 시퀀스는 압축성과 규칙성 사이의 균형을 반드시 유지해야 한다는 것을 보여준다. 이 발견은 이후 BPT, TreeMeshGPT 등 후속 연구들의 설계 철학에 직접적인 영향을 미쳤다.

#### 5.1.2 3D 생성 파이프라인 통합 촉진

MeshAnything V2는 AI 기반 3D 콘텐츠 제작에서 중요한 발전을 나타내며, 아티스트가 만든 메시를 학습하고 모방하는 시스템의 능력은 게임, 시각 효과, 디자인 응용에서 자동화된 3D 모델링의 새로운 가능성을 열어준다.

#### 5.1.3 AMT의 범용성 입증

AMT는 무조건적 또는 다른 조건부 메시 생성 설정에도 적용 가능하며, VQ-VAE의 사용 여부와 무관하게 효과적임을 보여주어, 다양한 아키텍처에서 범용적으로 활용 가능한 기술임을 입증했다.

#### 5.1.4 오픈소스 기여

AMT로 뒷받침되는 MeshAnything V2 프레임워크는 아티스트 생성 메시 분야에서 중요한 진전을 나타내며, 토큰 시퀀스 길이를 극적으로 줄임으로써 이 분야를 오랫동안 괴롭혀 온 계산 비효율성을 해결하고, 더 복잡하고 까다로운 3D 에셋 생산 환경에 AI 기반 메시 생성 통합을 위한 새로운 길을 열었다.

---

### 5.2 앞으로의 연구 시 고려할 점

#### (1) 고폴리 메시로의 확장 필요

현재의 1,600면 상한은 실제 산업 수준의 복잡한 캐릭터나 환경 에셋에는 여전히 부족하다. Meshtron은 최대 64,000개의 삼각형 면을 달성하여 MeshAnythingV2(최대 1,600면)와 MeshGPT(최대 800면)를 양적 지표와 시각적 표면 세부 묘사 모두에서 능가한다. 따라서 더 효율적인 토크나이제이션 전략이나 계층적 트랜스포머 설계가 필요하다.

#### (2) 토크나이제이션의 압축성과 규칙성 균형 연구

토큰 시퀀스의 규칙성 측면을 더욱 발전시켜야 한다. 짧은 시퀀스가 항상 메시 생성에 더 좋은 것이 아니며, 규칙성과 패턴 일관성이 효과적인 시퀀스 학습에 매우 중요하다. 효과적인 메시 토크나이제이션은 효율성과 규칙성 모두를 균형 있게 갖춰야 한다.

#### (3) 다양한 입력 조건 확장

현재는 포인트 클라우드를 주 조건 입력으로 사용하지만, 텍스트 설명, 이미지, 스케치 등 다양한 모달리티에 대한 조건부 생성 기능 확장이 필요하다. 강력한 포인트 클라우드 조건화가 표면 정렬과 기하학적 충실도를 향상시키는 것으로 나타났다.

#### (4) 아웃오브디스트리뷰션(OOD) 일반화 강화

비아티스트 메시에서 샘플링된 분포 외(out-of-distribution) 포인트 클라우드로 조건화할 때 Meshtron은 입력 형상을 충실히 재현하는 반면 MeshAnythingV2는 어려움을 겪는다. 따라서 OOD 입력에 대한 강건성 향상이 중요한 연구 방향이다.

#### (5) 데이터셋 품질 및 다양성

모델 학습에서 가장 큰 도전 중 하나는 데이터셋 구축이다. 학습에 쌍을 이루는 형상 조건과 AM이 필요하며, 형상 조건은 추론 시 조건으로 사용될 수 있도록 최대한 다양한 3D 표현에서 효율적으로 도출되어야 하고, 3D 형상을 정확하게 표현할 충분한 정밀도도 갖춰야 한다.

#### (6) 강화학습 기반 메시 생성 품질 향상

최근 후속 연구인 **DeepMesh** (2025)에서는 강화학습(RL)을 활용한 오토리그레시브 아티스트 메시 생성이 제안되고 있으며, 보상 설계와 정책 최적화를 메시 품질 지표와 연결하는 방향이 주목받고 있다.

---

## 📚 참고 자료 / 출처

1. **MeshAnything V2 (arXiv)**: Chen, Y. et al. "MeshAnything V2: Artist-Created Mesh Generation With Adjacent Mesh Tokenization." arXiv:2408.02555 (2024). https://arxiv.org/abs/2408.02555
2. **MeshAnything V2 (ICCV 2025 paper)**: https://openaccess.thecvf.com/content/ICCV2025/papers/Chen_MeshAnything_V2_Artist-Created_Mesh_Generation_with_Adjacent_Mesh_Tokenization_ICCV_2025_paper.pdf
3. **MeshAnything V2 (Project Page)**: https://buaacyw.github.io/meshanything-v2/
4. **MeshAnything V2 (GitHub)**: https://github.com/buaacyw/MeshAnythingV2
5. **MeshAnything V2 (arXiv HTML v2)**: https://arxiv.org/html/2408.02555v2
6. **MeshAnything V2 (arXiv HTML v3)**: https://arxiv.org/html/2408.02555v3
7. **MeshAnything V1**: Chen, Y. et al. "MeshAnything: Artist-Created Mesh Generation with Autoregressive Transformers." arXiv:2406.10163 (2024). https://arxiv.org/abs/2406.10163
8. **MeshGPT**: Siddiqui, Y. et al. "MeshGPT: Generating Triangle Meshes with Decoder-Only Transformers." CVPR 2024. https://nihalsid.github.io/mesh-gpt/
9. **PolyGen**: Nash, C. et al. "PolyGen: An Autoregressive Generative Model of 3D Meshes." ICML 2020. http://proceedings.mlr.press/v119/nash20a/nash20a.pdf
10. **EdgeRunner**: Tang, J. et al. "EdgeRunner: Auto-regressive Auto-encoder for Artistic Mesh Generation." arXiv:2409.18114 (2024). https://arxiv.org/html/2409.18114v1
11. **Meshtron**: "Meshtron: High-Fidelity, Artist-Like 3D Mesh Generation at Scale." arXiv:2412.09548 (2024). https://arxiv.org/html/2412.09548v1
12. **BPT (Scaling Mesh Generation)**: "Scaling Mesh Generation via Compressive Tokenization." arXiv:2411.07025 (2024). https://arxiv.org/html/2411.07025v1
13. **HuggingFace Paper Page**: https://huggingface.co/papers/2408.02555
14. **ResearchGate**: https://www.researchgate.net/publication/382884617_MeshAnything_V2_Artist-Created_Mesh_Generation_With_Adjacent_Mesh_Tokenization
15. **ICCV 2025 Poster**: https://iccv.thecvf.com/virtual/2025/poster/1334
16. **Emergent Mind 요약**: https://www.emergentmind.com/papers/2408.02555
17. **Liner.com 리뷰**: https://liner.com/review/meshanything-v2-artistcreated-mesh-generation-with-adjacent-mesh-tokenization

> ⚠️ **주의**: 본 답변은 공개된 arXiv 논문, 프로젝트 페이지, GitHub, 후속 연구 인용 정보를 기반으로 작성되었습니다. 논문 내부의 세부 실험 수치(구체적인 ablation table 수치 등)는 원문 PDF를 직접 확인하시길 권장합니다.
