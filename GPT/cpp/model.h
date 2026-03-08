#pragma once
#include <vector>
#include "value.h"
#include "tensor.h"
#include "config.h"

struct GPTLayer{
    Mat attn_wq;
    Mat attn_wk;
    Mat attn_wv;
    Mat attn_wo; //Output 파라미터 행렬

    Mat mlp_fc1;
    Mat mlp_fc2;

    GPTLayer();
};

class GPT 
{
    public:

    Mat wte;
    Mat wpe;
    Mat lm_head;

    std::vector<GPTLayer> layers;

    GPT();

    std::vector<Value*> get_all_params();

    Vec forward(int token_id, int pos_id, std::vector<std::vector<Vec>>& keys,
    std::vector<std::vector<Vec>>& values);
};