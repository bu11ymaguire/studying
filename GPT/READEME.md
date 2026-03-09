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

---
**Reference / Origin**: 
This project is deeply inspired by and translated from Andrej Karpathy's clean Python GPT code.
