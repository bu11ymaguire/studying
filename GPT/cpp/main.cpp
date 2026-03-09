#include<iostream>
#include<fstream>
#include<cmath>
#include<vector>
#include<string>
#include<algorithm>
#include "value.h"
#include "tensor.h"
#include "model.h"

int main(){
    std::ifstream file("input.txt");

    if(!file.is_open())
    {
        std::cerr << "input.txt 파일을 찾을 수 없습니다!" << std::endl;
        return 1;
    }

    std::vector<std::string> docs;
    std::string line;
    std::string all_chars = ""; 

    while(std::getline(file, line)) //고유 문자 추출을 위해 모든 글자를 이어붙인다.
    {
        if(!line.empty()){
            if(!line.empty()){
                docs.push_back(line);
                all_chars += line;
            }
        }
    }

    
    // 고유 문자 뽑아내기 및 정렬
    std::sort(all_chars.begin(), all_chars.end());
    auto last = std::unique(all_chars.begin(), all_chars.end());
    all_chars.erase(last, all_chars.end());


    int BOS = all_chars.size(); //BOS 특수 토큰 ID
    int VOCAB_SIZE = all_chars.size() + 1 ; 

    std::cout << "Num docs: " << docs.size() << std::endl;
    std::cout << "Vocab size: " << VOCAB_SIZE << std::endl;

    GPT model(VOCAB_SIZE);

    std::vector<Value*> params = model.get_all_params();
    std::cout << "Num params: " << params.size() << std::endl;

    // Adam 옵티마이저 버퍼 초기화
    std::vector<double> m(params.size(), 0.0);
    std::vector<double> v(params.size(), 0.0);

    for(int step = 0; step < NUM_STEPS; step++){
        // 문자열 데이터를 Tokenizer로 숫자 번역.
        std::string doc = docs[step % docs.size()];
        std::vector<int> tokens;

        tokens.push_back(BOS); // [BOS] 토큰 먼저 하나 넣기.

        for(char ch: doc){
            // all_chars 배열에서 ch 와 같은 글자의 위치 찾기.
            auto it = std::find(all_chars.begin(), all_chars.end(), ch);
            int idx = std::distance(all_chars.begin(), it);
            tokens.push_back(idx);
        }

        tokens.push_back(BOS);

        // 학습할 글자 길이(n) 정하기.
        int n = std::min(BLOCK_SIZE, static_cast<int>(tokens.size())-1);

        std::vector<std::vector<Vec>> keys(N_LAYER);
        std::vector<std::vector<Vec>> values(N_LAYER);

        std::vector<Value*> losses; 

        for(int pos_id = 0; pos_id < n; pos_id++){
            int token_id = tokens[pos_id];
            int target_id = tokens[pos_id + 1];

            Vec logits = model.forward(token_id, pos_id, keys, values);

            Vec probs = softmax(logits);

            Value* minus_one = new Value(-1.0);
            Value* prob_log = probs[target_id]->log();
            Value* loss_t = *prob_log * minus_one;

            losses.push_back(loss_t);
        }

        Value* loss = new Value(0.0);
        for(auto l : losses){
            loss = *loss + l;
        }

        Value* n_val = new Value(static_cast<double>(n));
        loss = *loss / n_val;

        loss->backward();

        double lr_t = LEARNING_RATE * (1.0 - (double)step/NUM_STEPS);

        for(size_t i = 0; i < params.size(); i++){
            Value* p = params[i];

            m[i] = BETA1 * m[i] + (1.0 - BETA1) * p->grad;

            v[i] = BETA2 * v[i] + (1.0 - BETA2) * (p->grad * p->grad);

            double m_hat = m[i] / (1.0 - std::pow(BETA1,step + 1));
            double v_hat = v[i] / (1.0 - std::pow(BETA2,step + 1));

            p->data -= lr_t * m_hat / (std::sqrt(v_hat) + EPS_ADAM);

            p->grad = 0.0;
        }

        std::cout << "step" << step + 1 << "/" << NUM_STEPS << " | loss " << loss->data << "\r" << std::flush;
    }

    return 0;
}