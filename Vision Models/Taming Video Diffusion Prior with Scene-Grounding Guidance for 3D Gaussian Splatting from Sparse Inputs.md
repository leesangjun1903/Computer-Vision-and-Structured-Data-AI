
# Taming Video Diffusion Prior with Scene-Grounding Guidance for 3D Gaussian Splatting from Sparse Inputs

## 1. 핵심 주장과 주요 기여 (요약)

이 논문은 희소 입력 환경에서 **3D Gaussian Splatting(3DGS)의 두 가지 중요한 문제**(외삽과 폐색)를 명시적으로 해결하는 첫 번째 연구입니다.

### 주요 기여:
1. **장면 기반 안내**: 렌더링된 3DGS 수열을 기반으로 비디오 확산 모델을 제어하는 훈련 무료 방법
2. **궤적 초기화 전략**: 3DGS 기반의 자동 카메라 궤적 선택
3. **최적화 방식 개선**: 지각 손실을 활용한 구멍 영역 채우기
4. **성능 향상**: Replica 3.5 dB, ScanNet 2.5 dB의 PSNR 개선

***

## 2. 핵심 수식 체계

### 2.1 조건부 스코어 함수 (베이즈 규칙)

$$\nabla_{\mathbf{x}_t}\log p(\mathbf{x}_t | Q) = \nabla_{\mathbf{x}_t}\log p(\mathbf{x}_t) + \nabla_{\mathbf{x}_t}\log p(Q|\mathbf{x}_t)$$

여기서 $Q$는 렌더링된 수열 기반 일관성 목표입니다.

### 2.2 장면 기반 손실 함수

$$L_S(M, S, X_t') = \|M \odot (S - X_t')\|_1 + \lambda_{\text{perc}} L_{\text{perc}}(M \odot S, M \odot X_t')$$

여기서:
- $M$: 폐색/시야 밖 영역 마스크
- $S$: 렌더링된 수열
- $X_t'$: 디노이징 예측 이미지
- $\odot$: 아다마르 곱 (요소별 곱셈)

### 2.3 최종 3DGS 최적화 손실

**입력 이미지:**
$$L_{\text{input}} = (1-\lambda_w)L_1(C_i, C_i^{\text{gt}}) + \lambda_w L_{\text{DSSIM}}(C_i, C_i^{\text{gt}})$$

**생성 이미지:**
$$L_{\text{gen}} = \lambda_{\text{gen1}} L_1(C_j, S_j) + \lambda_{\text{gen2}} L_{\text{perc}}(C_j, S_j)$$

***

## 3. 모델 구조의 특징

### 3.1 통합 파이프라인
```
희소 입력 → DUSt3R 초기화 → 기본 3DGS
    ↓
[궤적 초기화] ← [렌더링된 수열]
    ↓
장면 기반 안내 비디오 확산
    ↓
일관성 있는 생성 수열
    ↓
최종 3DGS 최적화
```

### 3.2 세 가지 핵심 기술

**1. 마스크 계산 (전송 지도)**

$$O(\mathbf{x}_p) = \prod_{i=1}^{K} (1 - \alpha_i), \quad M = O < \tau_{\text{mask}}$$

**2. 궤적 보간**
방위각: ±30°, ±15°, 0° / 반지름: 1배, 1/3배, 1/10배 깊이

**3. 지역적 샘플링 전략**
- 같은 수열에서: 70% (시각 품질)
- 다른 수열에서: 30% (망각 방지)

***

## 4. 성능 및 일반화 성능 향상

### 4.1 정량적 성과

| 데이터셋 | 기본선 | 제안 방법 | 향상도 |
|---------|--------|---------|--------|
| Replica | 22.80 dB | 26.35 dB | +3.5 dB |
| ScanNet | 21.41 dB | 23.89 dB | +2.5 dB |

### 4.2 관찰 불가능 영역의 극적 개선

| 영역 | 기본선 | 제안 방법 | 향상도 |
|------|--------|---------|--------|
| 관찰 가능 | 25.45 dB | 27.12 dB | +1.67 dB |
| 관찰 불가능 | 14.27 dB | 20.85 dB | **+6.58 dB** |

### 4.3 일반화 성능의 근본 원인

**1. 다중 뷰 일관성 학습**
- 비디오 확산 모델이 보유한 강력한 3D 구조 사전
- 시간적 일관성, 3D 기하학, 물리적 제약

**2. 장면 기반 안내의 이중 효과**
- **일관성 제약**: 인접 프레임의 높은 일관성
- **장면 기반 제약**: 환각 요소 제거

**3. 훈련 무료 방법의 장점**
- 도메인 시프트에 강건
- 새로운 장면에 즉시 적응
- 확산 모델의 일반화 능력 직접 활용

***

## 5. 한계 및 향후 과제

### 5.1 확인된 한계

**해상도 제약 (가장 심각)**
- Replica: 320×448 → 480×640 (업샘플링)
- ScanNet: 320×512 → 480×720 (업샘플링)
- **결과**: 과도한 평활화, 고주파 디테일 손실

**계산 비용**
- 생성 시간: 이미지당 수 분
- GPU 메모리: 32GB V100 필요
- 실시간 적용 불가능

**생성 품질 의존성**
- 초기 기본 3DGS 모델 품질에 의존
- 순환 의존성 잠재력

### 5.2 향후 해결 방향

**단기 (1-2년)**:
- 해상도 확대 (다단계 생성)
- 확산 모델 가속화
- 지역 세부 정제

**중기 (2-5년)**:
- 동적 장면 처리 (4D)
- 다양한 장면 타입 지원
- 다중 모달리티 통합

**장기 (5년 이상)**:
- 단일 이미지 한계 도전
- 인간 수준 재구성
- 차세대 표현 패러다임

***

## 6. 관련 최신 연구 동향 (2020 이후)

### 6.1 희소 입력 3DGS 발전 (2024-2025)

| 방법 | 핵심 기여 | 성능 향상 |
|------|---------|---------|
| **NexusGS** (2025) | 에피폴라 깊이 사전 | +0.5-1.0 dB |
| **HiSplat** (2025, ICLR) | 계층적 가우시안 | +0.82-3.19 dB |
| **GraphSplat** (2025) | 그래프 기반 특성 | 크로스 데이터셋 개선 |
| **MS-GS** (2025) | 다중 모양 + 의미론 | 광 조건 변화 처리 |

### 6.2 생성 모델 기반 3D 재구성 (2023-2025)

**확산 모델 활용**:
- **ReconFusion** (2024): SDS 기반 재구성
- **CAT3D** (2024): 다중 뷰 확산, 분 단위 생성
- **CAT4D** (2024): 4D 재구성
- **ViDAR** (2025): 비디오 확산 인식 4D

**훈련 무료 안내 (2024-2025)**:
- **TFG** (NeurIPS 2024): 통합 훈련 무료 안내
- **Dreamguider** (2024): 역전파 없는 안내
- **OC-Flow** (2025): 최적 제어 기반

### 6.3 폐색 처리 방법 (2023-2025)

- **OccluGaussian** (2025, ICCV): 폐색 인식 장면 분할
- **FSGS** (2024): 보이지 않는 뷰 정규화
- **GS-GS** (2025): 생성적 희소 뷰 가우시안

***

## 7. 향후 연구 시 고려사항

### 7.1 학술 연구자

**주요 고민**:
- 생성 사전의 최적 활용 방법
- 일관성과 다양성의 절충
- 도메인 특화 확산 모델 필요성

**연구 아이디어**:
- 다중 목표 최적화 프레임워크
- 불확실성 기반 가중 안내
- 하이브리드 명시/암시 표현

### 7.2 산업 응용 개발자

**실무 체크리스트**:
- GPU 메모리 (32GB 이상)
- 배치 처리 최적화
- 중간 결과 캐싱
- 모니터링 시스템

**적용 가능 분야**:
- ✓ 문화재 보존
- ✓ VR/XR 콘텐츠
- △ 건축 시각화
- ✗ 실시간 렌더링

### 7.3 시스템 엔지니어

**최적화 핵심**:
- 해상도-메모리 트레이드오프
- 생성 시간 최소화
- 캐시 효율성
- 병렬 처리

**하이퍼파라미터 권장값**:
- 입력 뷰: 6-9개
- 해상도: 480×640
- λ_perc: 10^-4
- ρ: 0.5 (지역/전역 샘플링 비율)

***

## 8. 최종 평가

| 평가 항목 | 점수 | 근거 |
|---------|------|------|
| **혁신도** | ★★★★★ | 훈련 무료 방식, 새로운 안내 기법 |
| **실용성** | ★★★★☆ | 산업 가능하나 계산 비용 있음 |
| **이론성** | ★★★★☆ | 베이즈 기초 확고, 최적성 증명 가능 |
| **영향력** | ★★★★★ | 3D 비전, 확산, 신경장 분야 광범위 |

***

## 참고: 제공된 상세 분석 문서

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/bcb15904-460c-43d5-a495-33758e2ae8db/2503.05082v1.pdf)
[2](https://ieeexplore.ieee.org/document/11092863/)
[3](https://ieeexplore.ieee.org/document/11125578/)
[4](https://ieeexplore.ieee.org/document/11247717/)
[5](https://ieeexplore.ieee.org/document/10887746/)
[6](https://arxiv.org/abs/2508.15457)
[7](https://arxiv.org/abs/2509.15548)
[8](https://link.springer.com/10.1007/s10489-025-06494-2)
[9](https://arxiv.org/abs/2506.10335)
[10](https://dl.acm.org/doi/10.1145/3746027.3755481)
[11](https://www.semanticscholar.org/paper/4d867cecda8646506bd14647af1014fe6557d8b5)
[12](https://arxiv.org/html/2412.10051v1)
[13](https://arxiv.org/html/2410.06245)
[14](https://arxiv.org/html/2502.02283v3)
[15](https://arxiv.org/pdf/2403.14627.pdf)
[16](https://arxiv.org/html/2401.02436v1)
[17](https://arxiv.org/html/2312.00206v2)
[18](https://arxiv.org/html/2503.04314v1)
[19](https://arxiv.org/html/2412.02245)
[20](https://proceedings.iclr.cc/paper_files/paper/2025/file/78da47a28386d3e2e5e156d8148cecdf-Paper-Conference.pdf)
[21](https://openreview.net/forum?id=mLVqiNH0aA)
[22](https://proceedings.neurips.cc/paper_files/paper/2024/file/cad4501fe7c1b53427b363daf1366b2f-Paper-Conference.pdf)
[23](https://arxiv.org/html/2312.00206v3)
[24](https://blog.outta.ai/289)
[25](https://arxiv.org/html/2508.15457)
[26](https://www.sciencedirect.com/science/article/abs/pii/S1568494625008415)
[27](https://kimjy99.github.io/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0/cat4d/)
[28](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/10096.pdf)
[29](https://openaccess.thecvf.com/content/CVPR2025/papers/Kong_Generative_Sparse-View_Gaussian_Splatting_CVPR_2025_paper.pdf)
[30](https://dl.acm.org/doi/10.1145/3746027.3761989)
[31](https://www.isca-archive.org/interspeech_2024/choi24c_interspeech.html)
[32](https://www.semanticscholar.org/paper/6c708659768e470f63d06f791ff8420e7ff0feac)
[33](https://arxiv.org/abs/2505.02527)
[34](https://journals.tsu.ru/philosophy/&journal_page=archive&id=2595&article_id=52952)
[35](https://www.semanticscholar.org/paper/150aec040041ad4ba473da610101820c767d63ff)
[36](https://aacrjournals.org/cebp/article/34/9_Supplement/B158/764755/Abstract-B158-Implementation-of-a-low-cost-tobacco)
[37](https://aacrjournals.org/clincancerres/article/31/13_Supplement/A035/763347/Abstract-A035-Denoising-Models-Enhance-Detection)
[38](https://www.po-rt.ru/articles/1943)
[39](https://ocs.editorial.upv.es/index.php/HEAD/HEAd25/paper/view/19996)
[40](https://arxiv.org/abs/2409.15761)
[41](https://arxiv.org/abs/2407.02687)
[42](https://arxiv.org/html/2406.02549)
[43](https://arxiv.org/pdf/2403.12404.pdf)
[44](https://arxiv.org/html/2406.07540)
[45](http://arxiv.org/pdf/2312.12487.pdf)
[46](https://arxiv.org/pdf/2210.09292.pdf)
[47](https://arxiv.org/html/2410.18070)
[48](https://theaisummer.com/classifier-free-guidance/)
[49](https://arxiv.org/html/2503.16177v2)
[50](https://isprs-annals.copernicus.org/articles/X-1-W1-2023/895/2023/isprs-annals-X-1-W1-2023-895-2023.pdf)
[51](https://papers.nips.cc/paper_files/paper/2024/file/2818054fc6de6dacdda0f142a3475933-Paper-Conference.pdf)
[52](https://openaccess.thecvf.com/content/ICCV2025/papers/Liu_OccluGaussian_Occlusion-Aware_Gaussian_Splatting_for_Large_Scene_Reconstruction_and_Rendering_ICCV_2025_paper.pdf)
[53](https://arxiv.org/html/2312.09095v2)
[54](https://dl.acm.org/doi/10.1145/3681758.3697997)
[55](https://dl.acm.org/doi/10.1145/3610548.3618188)
[56](https://openreview.net/forum?id=N8YbGX98vc)
[57](https://arxiv.org/abs/2505.19854)
[58](https://arxiv.org/abs/2505.20729)
[59](https://ieeexplore.ieee.org/document/11137711/)
[60](https://www.semanticscholar.org/paper/d98ac277478bbc568ede0c1f331d4e78ad745c7f)
[61](https://www.mdpi.com/2072-4292/15/12/3076)
[62](https://www.mdpi.com/2077-0472/14/3/391)
[63](https://ieeexplore.ieee.org/document/10205144/)
[64](http://biorxiv.org/lookup/doi/10.1101/2021.11.09.467984)
[65](https://linkinghub.elsevier.com/retrieve/pii/S0926580523002091)
[66](https://www.mdpi.com/2072-4292/15/15/3775)
[67](https://arxiv.org/html/2503.16318)
[68](https://arxiv.org/html/2409.08613v1)
[69](https://arxiv.org/html/2503.24391)
[70](http://arxiv.org/pdf/2312.14132v1.pdf)
[71](https://arxiv.org/html/2312.06706v1)
[72](https://arxiv.org/html/2410.23245)
[73](https://www.mdpi.com/1424-8220/25/5/1354)
[74](https://arxiv.org/pdf/1612.00603.pdf)
[75](https://learnopencv.com/dust3r-geometric-3d-vision/)
[76](https://pubmed.ncbi.nlm.nih.gov/39531569/)
[77](https://drexubery.github.io/ViewCrafter/)
[78](https://github.com/naver/dust3r)
[79](https://proceedings.neurips.cc/paper_files/paper/2023/file/b87738474533cab76c7bee4e08443aca-Paper-Conference.pdf)
[80](https://pubmed.ncbi.nlm.nih.gov/40986578/)
[81](https://kimjy99.github.io/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0/dust3r/)
[82](https://cwchenwang.github.io/outdoor-nerf-depth/data/paper.pdf)
[83](https://arxiv.org/html/2503.05638v1)
[84](https://ethanswinery.tistory.com/154)
