# C++ Autograd Engine & Mini-GPT

이 프로젝트는 Andrej Karpathy의 [순수 파이썬 GPT 구현체(makemore)](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95)를 바탕으로, 자동 미분(Autograd)부터 어텐션(Attention) 엔진까지 외부 딥러닝 프레임워크(PyTorch, TensorFlow 등) 없이 **순수 C++로 밑바닥부터 포팅(Porting)**하며 학습한 결과물입니다.

## 📌 주요 기능 및 구현

- **Autograd Engine (`value.h`, `value.cpp`)**: 
  - 스칼라 값 기반의 연산 그래프(Computation Graph) 구축.
  - 덧셈, 곱셈, 나눗셈, 지수(`exp`), 로그(`log`), 멱승(`power`), `ReLU` 연산 객체화 및 오버로딩.
  - 위상 정렬(Topological Sort)을 이용한 연쇄 법칙(Chain Rule) 기반의 오차 역전파(`backward()`) 구현.
- **Tensor Operations (`tensor.h`, `tensor.cpp`)**:
  - `std::vector` 기반의 1차원, 2차원 매트릭스 자료구조.
  - `linear`, `softmax` (수치 안정성 최적화 적용), `rmsnorm` 수학 정규화 구현.
- **GPT Architecture (`model.h`, `model.cpp`)**:
  - `GPTLayer`: Multi-head Attention과 MLP 모듈로 구성된 단일 트랜스포머 레이어.
  - `GPT`: Token Embedding(`wte`), Positional Embedding(`wpe`), Language Model Head(`lm_head`) 초기화 및 Forward Pass 구현.
- **Optimizer & Training (`main.cpp`)**:
  - 파일 입출력을 통한 텍스트 데이터 로드 및 토크나이저(단어장) 구축.
  - `Adam Optimizer` 스텝 구현 (모멘텀 $m$, $v$ 계산 및 가중치 업데이트).
  - 메모리 제어를 포함한 훈련 순전파/역전파 루프.

## 🛠️ 컴파일 및 실행 방법

### 1. 훈련 데이터 준비
실행을 위해 동일한 디렉토리에 `input.txt` 파일이 필요합니다. 원래 실습에 사용된 Karpathy의 [names.txt](https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt)를 다운로드 받아 `input.txt`로 저장합니다.

```bash
# PowerShell 환경에서 다운로드 예시
Invoke-WebRequest -Uri "https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt" -OutFile "input.txt"
```

### 2. C++ 컴파일 (C++17 이상 권장)
프로젝트 내의 리소스들을 모두 엮어 컴파일합니다.

```bash
g++ -o gpt_train value.cpp tensor.cpp model.cpp main.cpp -std=c++17
```

### 3. 학습 실행
아래 명령어로 C++ 기반의 딥러닝 훈련 루프를 시작할 수 있습니다.
```bash
./gpt_train
```

## 📝 구조 및 하이퍼파라미터 (`config.h`)
- `N_LAYER`: 1
- `N_EMBD`: 16
- `BLOCK_SIZE`: 16
- `N_HEAD`: 4
- `LEARNING_RATE`: 0.01 (Adam)
- `NUM_STEPS`: 1000

## 📊 학습 결과 분석 (`output.txt`)

`./gpt_train > output.txt` 명령어로 1,000 step 학습을 수행하고, 그 출력 로그를 분석한 결과입니다.

### Loss 변화 추이

| 구간 (Step) | 평균 Loss |
|:-----------:|:---------:|
| 1 – 100 | 2.6135 |
| 101 – 200 | 2.3524 |
| 201 – 300 | 2.3112 |
| 301 – 400 | 2.3023 |
| 401 – 500 | 2.2753 |
| 501 – 600 | 2.2413 |
| 601 – 700 | 2.1636 |
| **701 – 800** | **2.1223** (최저 구간) |
| 801 – 900 | 2.1448 |
| 901 – 1000 | 2.2384 |

- **시작 Loss** (Step 1): **3.38** → **최종 Loss** (Step 1000): **1.45**
- 단일 스텝 최소 Loss: **1.26**
- 27개 문자에 대한 이론적 랜덤 확률 Loss = $-\ln(1/27) \approx 3.30$
- 학습 초반에 이 이론치를 빠르게 관통하였으므로 **학습이 정상적으로 작동**하고 있음을 확인.

### 예측 정확도

학습 중 매 스텝마다 각 문자에 대해 `Target(정답) vs Pred(예측)`을 기록하였으며, 총 **7,000개** 예측에 대한 통계는 아래와 같습니다.

| 지표 | 값 |
|------|-----|
| 전체 정확도 | **29.1%** (2,039 / 7,000) |
| 전반부 (Step 1~500) 정확도 | 26.7% |
| 후반부 (Step 501~1000) 정확도 | **31.6%** |
| 평균 Confidence | 31.62% |
| 랜덤 기준 기대 정확도 | ~3.7% (1/27) |

> 랜덤 추측 대비 약 **8배**의 정확도 향상이 관찰되었고, 전반부 → 후반부로 갈수록 정확도가 상승하여 모델이 점진적으로 패턴을 학습하고 있음을 보여줍니다.

### 글자별 정확도 분석 (주요 문자)

| 문자 | 출현 횟수 | 정확 예측 | 정확도 | 해석 |
|:----:|:---------:|:---------:|:------:|------|
| `#` (단어 끝) | 1,000 | 831 | **83.1%** | 단어 경계를 압도적으로 잘 학습 |
| `a` | 1,131 | 644 | **56.9%** | 가장 빈번한 글자, 높은 정확도 |
| `l` | 640 | 190 | **29.7%** | 중간 수준의 학습 |
| `i` | 564 | 115 | 20.4% | 보통 |
| `e` | 742 | 99 | 13.3% | 빈도 대비 낮은 정확도 |
| `n` | 539 | 59 | 10.9% | 위치 다양성이 높아 예측 어려움 |
| `y` | 305 | 1 | 0.3% | 거의 학습 실패 |
| `t`, `d`, `c` | 각 100~136 | 0 | **0.0%** | 학습되지 않음 |

### 인사이트: 이 모델이 배운 것 vs 못 배운 것

**✅ 잘 배운 것**
- **단어의 끝(`#`)**: 이름이 어디서 끝나야 하는지를 83.1%의 정확도로 파악. 이는 이름의 평균 길이와 글자 빈도 패턴에서 "경계"를 학습했다는 뜻.
- **가장 흔한 모음(`a`)**: 영어 이름에서 가장 빈번하게 등장하는 글자를 56.9%로 적극적으로 예측. 모델이 "확신이 없을 때 `a`를 찍는" 안전한 전략을 학습한 것으로 보임.

**❌ 못 배운 것**
- **드문 자음 (`t`, `d`, `c`, `y`)**: 출현 빈도가 줄어들수록 정확도가 급락. 4,192개의 초소형 파라미터로는 글자 간의 미세한 맥락적 차이를 표현할 용량이 부족.
- **복잡한 철자 흐름**: `charlotte` → `c-h-a-r-l-o-t-t-e` 같은 긴 의존성 패턴은 Embedding 16차원 / 1 Layer로는 한계.

### 한계 및 개선 방향

| 한계 | 원인 | 잠재적 해결 방향 |
|------|------|-----------------|
| Loss가 ~2.1 아래로 잘 내려가지 않음 | 모델 용량 부족 (4,192 params) | `N_EMBD`, `N_LAYER` 증가 |
| Step 700 이후 Loss 재상승 | 학습률이 아직 높거나, 오버피팅 징후 | Learning rate schedule 조절, Early stopping |
| `t`, `d`, `c` 등 정확도 0% | 해당 문자가 등장하는 맥락이 다양 | 더 넓은 `BLOCK_SIZE`, 더 많은 Head |
| `e`의 정확도가 빈도 대비 낮음 | 이름 내 위치 분포가 넓어 예측 어려움 | 데이터 증강 또는 모델 깊이 증가 |

> **결론**: 4,192개 파라미터만으로도 "이름의 끝 위치"와 "가장 흔한 글자"를 학습하는 데 성공하였으며, 이는 C++로 밑바닥부터 구현한 Autograd + Attention 엔진이 정상 동작함을 실증합니다.

---
**Reference / Origin**: 
This project is deeply inspired by and translated from Andrej Karpathy's clean Python GPT code.
