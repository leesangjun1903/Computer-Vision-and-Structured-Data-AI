# PROGPROMPT: Generating Situated Robot Task Plans using Large Language Models

# 1. 핵심 주장 및 주요 기여

**PROGPROMPT**는 로봇 작업 계획(task planning)에서 “상황에 맞춘(state-situated)” 계획 생성을 위해 자연어 대신 프로그래밍 언어 구조(파이썬 코드 형태)로 LLM을 프롬프트(prompt)하여, 실행 가능한(plan executability) 프로그램을 직접 생성한다.  
주요 기여:
- 로봇의 **가용 액션(primitive actions)** 과 **환경 객체(object list)** 를 import 구문과 리스트로 명시하여 LLM이 불가능한 행동을 생성하지 않도록 제약.
- **주석(comments)** 과 **assertion 기반 상태 피드백(Assertions & Recovery Actions)** 을 통해 계획 단계마다 전제조건을 확인하고 오류 복구(recovery)가 가능하도록 설계.
- 가상 시뮬레이터(VirtualHome)와 실제 로봇(FRANKA Panda) 모두에서 기존 자연어 기반 플래너 대비 성공율 및 실행 가능성(exec) 대폭 향상 입증.[1]

# 2. 문제 정의, 제안 방법, 모델 구조, 성능 및 한계

## 2.1 해결하고자 하는 문제  
고수준 자연어 지시문(예: “microwave salmon”)만으로 LLM이 생성한 자유형 텍스트(plan text)는  
1) 로봇의 실제 액션 API와 매핑(mapping) 불확실  
2) 현재 환경에 존재하지 않는 객체나 affordance를 언급  
3) 실행 중 오류 발생 시 복구 불가  
라는 한계를 갖는다.

## 2.2 제안 방법(PROGPROMPT)  
로봇 작업 계획을 파이썬 함수(program)로 모델링:
- 함수 이름 = 고수준 작업 지시(task name)
- 본문(body) = 액션 API 호출(action(obj1, obj2)), 자연어 주석, assertion 구문

프롬프트 구조:
```python
from actions import grab, putin, open, close, find, switchon, switchoff
objects = ['salmon','microwave',…]  # 환경 객체 리스트
# Example 1: throw_away_lime()
def throw_away_lime():
    # 1: find lime
    find('lime')
    assert('garbagecan' is 'opened') else: open('garbagecan')
    putin('lime','garbagecan')
    # Done
# … (고정 예제 3개)
# Target Task
def microwave_salmon():
    <LLM-generated code completion>
```

수식으로 정리하면, 작업 계획은 튜플 $$(O,P,A,T,I,G,t)$$ 중  
- $$O$$: 환경 객체 집합  
- $$A$$: 액션 API 집합  
- $$t$$: 자연어 지시문  
입력 시 프롬프트 함수 $$f_{\text{prompt}}(s,t,A,O)$$ → LLM → 실행 가능 프로그램 $$\pi$$ 생성.

## 2.3 모델 구조  
- LLM 백본: GPT-3 계열(GPT3, Codex, Davinci)  
- 프롬프트: 코드+주석+assertion 예제 3개 포함  
- 생성: 코드 완성(code completion) 방식  

## 2.4 성능 향상  
- 가상 환경(VirtualHome) 테스트:  
  평균 성공률(SR) 0.65 → 0.34 (기존) → 0.65로 약 두 배 개선[1]
  실행 가능성(Exec) 0.45 → 0.84, 목표 조건 재현율(GCR) 0.21 → 0.65 향상  
- 실제 로봇: distractor 포함 실험에서도 Plan SR 거의 100%, Exec=1.0 달성  

## 2.5 한계  
- 예제 코드에 없는 환경 특이성(affordance) 반영 어려움  
- API 캡(cap)으로 인한 코드 절단(cut-off) 발생  
- 실제 로봇에서 assertion 기반 폐루프(closed-loop) 미지원  
- 자동 평가 때 목표 상태 다수 가능성(ambiguity) 처리 미흡  

# 3. 모델의 일반화 성능 향상 가능성

PROGPROMPT는 다음 방면에서 **일반화(generalization)** 잠재력 보유:
1. **프롬프트 교체만으로 에이전트 전환**: import 문 목록 수정으로 다른 로봇 플랫폼에 즉시 적용.  
2. **환경 객체 리스트 동적 구성**: ViLD 같은 오픈 보카(object detection)로 신규 객체 포함 가능.  
3. **프로그래밍 언어 구조 확장**: 중첩된 제어 흐름(nested control flow), 실수 값 변수(real-valued state) 등 추가로 복잡한 작업 계획 표현력 강화.  
4. **추가 예제 도입**: few-shot 예제 개수·다양성 확대로 새로운 작업 유형에도 적응.

# 4. 향후 연구에 미치는 영향 및 고려 사항

**영향**:  
- 로봇 분야에서 LLM을 **실행 가능한 ‘계획 프로그램’** 생성기로 활용하는 새로운 패러다임 제시.  
- NLP-로보틱스 융합 연구에서 *프로그래밍 언어 프롬프트 디자인* 중요성 부각.

**고려 사항**:  
- 프롬프트 예제 다양성과 **환경 affordance 명시** 방안 연구  
- 실시간 state feedback 통합 및 **폐루프 복구 메커니즘** 강화  
- 다중 목표 상태 평가를 위한 **유연한 평가 지표** 마련  
- 코드 캡 문제 해결을 위한 **모델 입력 분할·후속 완성 전략** 검토

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/4bb0993a-abf7-41b4-8ec7-5e187235c341/2209.11302v1.pdf)
