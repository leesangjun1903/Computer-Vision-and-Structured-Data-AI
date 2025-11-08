# (SCICO) Scientific Computational Imaging Code | Software
# 핵심 요약 및 기여

**핵심 주장**  
Scientific Computational Imaging Code (SCICO)는 과학적 영상 응용에서 발생하는 Inverse Problems 를 효과적으로 풀기 위해, 모듈화된 연산자(operator), 비용 함수(functional), 정규화 기법(regularizer), 최적화 알고리즘을 결합한 파이썬 기반 패키지를 제안한다. JAX를 기반으로 GPU/TPU 가속, JIT 컴파일, 자동 미분을 지원하여, 역연산자(adjoint) 구성과 대규모 반복 솔버 구현을 간소화한다.

**주요 기여**  
- 모듈형 빌딩 블록(Forward 모델, 데이터 충실도, 정규화, 최적화)을 제공하여 다양한 역문제에 재사용 가능  
- JAX 기반 설계로 자동미분을 통한 연산자 야코비안 및 어드조인트 계산 자동화  
- GPU/TPU 가속·JIT 컴파일로 대규모 영상 복원 문제의 성능 향상  
- DnCNN 기반 플러그앤플레이(prior) 프레임워크 구현으로 최신 딥러닝 기법 통합  
- 오픈소스·커뮤니티 기여 가능 구조로 확장성 및 재현성 확보  

# 문제 정의 및 제안 방법

## 해결하고자 하는 문제  
전통적 영상획득에서는 광학·하드웨어에 의존해 이미지를 취득하나, SCICO가 다루는 **과학적 계산 영상(computational imaging)** 은 측정 데이터 $$y$$ 로부터 물리 모델 $$A$$ 와 통계적 노이즈 $$\xi$$ 를 고려해 잠재 영상 $$x$$ 를 복원하는 아래의 **비유보 역문제(ill-posed inverse problem)** 이다.  
$$
y = A x + \xi
$$

## 제안 방법 (수식 포함)  
SCICO는 다음과 같은 **최적화 문제**를 일반 형태로 정의한다:  

$$
\hat{x} = \arg\min_{x} \; f(x) + \sum_{i=1}^N g_i\bigl(C_i x\bigr)
$$  

여기서  
- $$f(x) = \tfrac12\|A x - y\|_W^2$$ : 가중치 $$W$$ 를 포함한 데이터 충실도 항  
- $$g_i$$ : 비부정(Non-negative) 제약, Total Variation, $$\ell_{2,1}$$ 규제 등 정규화 함수  
- $$C_i$$ : Identity, finite difference, 기타 선형 연산자  

예를 들어, 다중 채널 토모그래피 재구성 문제는 다음과 같이 ADMM으로 풀 수 있다:  

$$
\hat{x} = \arg\min_{x \ge 0} \tfrac12\|A x - y\|\_W^2 + \alpha \|D x\|_{2,1}
$$  

$$
\text{ADMM 솔버: } 
\begin{cases}
\text{update }x \\
\text{update duals}
\end{cases}
\quad\text{(최대 반복 30회)}
$$

## 모델 구조  
1. **Forward 모델**($$A$$) : ASTRA 래퍼를 이용한 Radon 투영 연산자  
2. **데이터 충실도** : 가중 제곱 오차 $$\|A x - y\|_W^2$$  
3. **정규화**  
   - 비부정 제약 $$g_1(C_1 x)$$  
   - TV 유사 $$\ell_{2,1}$$ 규제 $$g_2(C_2 x)$$  
4. **최적화 알고리즘** : ADMM, 내부 선형 서브문제는 CG로 해결  

또한 SCICO는 DnCNN 기반 딥디노이저를 **플러그앤플레이(Plug-and-Play, PPP)** 기법으로 통합하여, 전통적 정규화 대신 학습 기반 사전(prior)을 적용 가능하도록 설계되었다.

## 성능 향상 및 한계  
- **성능**: JAX JIT 컴파일과 GPU 가속을 통해 반복 구조의 영상 복원 알고리즘에서 3–10배 이상의 속도 향상 확인  
- **정확도**: PPP-DnCNN 적용 시 PSNR이 14.3 dB → 20.7 dB로 개선[image:2]  
- **확장성**: 사용자 정의 연산자·정규화·솔버를 쉽게 추가 가능  
- **한계**:  
  - 딥러닝 기반 PPP는 학습 데이터 도메인에 따라 일반화 성능의 변동성  
  - 매우 고차원·대규모 데이터에 대한 메모리 요구량  
  - ADMM 및 CG 반복 횟수에 따른 계산 비용 부담  

# 모델 일반화 성능 향상 가능성  
- **자동미분 기반 어드조인트 계산**이 수기 파생 오류를 줄여, 다양한 선형 모델에도 일관적 적용  
- **플러그앤플레이 프레임워크**로 사전 학습된 다양한 딥 디노이저를 손쉽게 교체·비교 가능  
- JAX의 **호환성** 덕분에 새로운 학습 기반 정규화(예: score-based 모델, diffusion prior) 통합 용이  
- GPU/TPU 장점으로 대규모 실험 실행이 가능하여, 도메인별 일반화 효과를 체계적 검증  

# 향후 연구에 미치는 영향 및 고려 사항  
SCICO는 **컴퓨팅 자원**과 **모듈화 설계**를 결합하여, 향후 다음 영역에서 영향이 기대된다.  
- **차세대 영상 재구성**: learned regularizer, physics-informed neural networks 통합 연구 가속  
- **표준화된 벤치마크**: 과학적 영상 역문제 비교·재현성 확보  
- **자동화된 파이프라인**: JAX 생태계 활용한 end-to-end 학습·복원 파이프라인 구축  

*고려 사항*:  
- 딥러닝 사전의 **도메인 편향** 해소 및 **robustness** 검증  
- 대규모 3D/4D 영상 처리 시 **메모리 최적화** 전략  
- 동시성 실험을 위한 **분산 컴퓨팅** 및 **mixed-precision** 활용 전략  

---  

**결론**: SCICO는 과학적 영상 재구성 연구를 위한 유연하고 고성능의 오픈소스 프레임워크로, JAX 기반 설계를 통해 전통적 최적화와 최신 딥러닝 기법의 통합을 용이하게 하며, 향후 다양한 일반화 연구와 대규모 실험 인프라 구축에 중추적 역할을 할 것으로 기대된다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/eb367e5e-6836-426f-b03d-e33c49081634/1898364.pdf
