#include "model.h"

GPTLayer::GPTLayer() // 개별 레이어 생성자.
{
    attn_wq = create_matrix(N_EMBD, N_EMBD);
    attn_wk = create_matrix(N_EMBD, N_EMBD);
    attn_wv = create_matrix(N_EMBD, N_EMBD);
    attn_wo = create_matrix(N_EMBD, N_EMBD);

    mlp_fc1 = create_matrix(4 * N_EMBD, N_EMBD);
    mlp_fc2 = create_matrix(N_EMBD, 4 * N_EMBD);
}

GPT::GPT() // 전체 모델 생성자
{

    int VOCAB_SIZE = 65;

    wte = create_matrix(VOCAB_SIZE, N_EMBD);
    wpe = create_matrix(BLOCK_SIZE, N_EMBD);
    lm_head = create_matrix(VOCAB_SIZE, N_EMBD);

    for(int i = 0; i < N_LAYER; i++)
    {
        layers.push_back(GPTLayer());
    }
}

std::vector<Value*> GPT::get_all_params(){ //모델 내 모든 가중치 1차원 리스트로 모아주기
    std::vector<Value*> p;

    auto add_mat = [&](const Mat& m) {
        for(const auto& row : m){
            for(auto val : row){
                p.push_back(val);
            }
        }
    };

    add_mat(wte);
    add_mat(wpe);
    add_mat(lm_head);

    for(const auto& element : layers)
    {
        add_mat(element.attn_wq);
        add_mat(element.attn_wk);
        add_mat(element.attn_wv);
        add_mat(element.attn_wo);

        add_mat(element.mlp_fc1);
        add_mat(element.mlp_fc2);
    }

    return p;
}

Vec GPT::forward(int token_id, int pos_id, std::vector<std::vector<Vec>>& keys,std::vector<std::vector<Vec>>& values)
{
    Vec tok_emb = this->wte[token_id];
    Vec pos_emb = this->wpe[pos_id];
    
    Vec x(N_EMBD);

    for(int i = 0; i < N_EMBD; i++){
        x[i] = *tok_emb[i] + pos_emb[i];
    }

    x = rmsnorm(x);

    for(int i = 0 ; i < N_LAYER ; i++)
    {
        Vec x_residual = x;
        x = rmsnorm(x);

        Vec q = linear(x,this->layers[i].attn_wq);
        Vec k = linear(x,this->layers[i].attn_wk);
        Vec v = linear(x,this->layers[i].attn_wv);
        
        keys[i].push_back(k);
        values[i].push_back(v);

        Vec x_attn;

        fot(int h = 0; h < N_HEAD; h++)
        {
            int hs = h * HEAD_DIM;

            Vec q_h;
            for(int j = hs; j < hs + HEAD_DIM; j++){
                q_h.push_back(q[j]);
            }
        }

        std::vector<Vec> k_h;
        for(const Vec& ki : keys[i])
        {
            
        }
    } 
}



/*
def gpt(token_id, pos_id, keys, values):
    tok_emb = state_dict['wte'][token_id] # token embedding
    pos_emb = state_dict['wpe'][pos_id] # position embedding
    x = [t + p for t, p in zip(tok_emb, pos_emb)] # joint token and position embedding
    x = rmsnorm(x) # note: not redundant due to backward pass via the residual connection

    for li in range(n_layer):
        # 1) Multi-head Attention block
        x_residual = x
        x = rmsnorm(x)
        q = linear(x, state_dict[f'layer{li}.attn_wq'])
        k = linear(x, state_dict[f'layer{li}.attn_wk'])
        v = linear(x, state_dict[f'layer{li}.attn_wv'])
        keys[li].append(k)
        values[li].append(v)
        x_attn = []
        for h in range(n_head):
            hs = h * head_dim
            q_h = q[hs:hs+head_dim]
            k_h = [ki[hs:hs+head_dim] for ki in keys[li]]
            v_h = [vi[hs:hs+head_dim] for vi in values[li]]
            attn_logits = [sum(q_h[j] * k_h[t][j] for j in range(head_dim)) / head_dim**0.5 for t in range(len(k_h))]
            attn_weights = softmax(attn_logits)
            head_out = [sum(attn_weights[t] * v_h[t][j] for t in range(len(v_h))) for j in range(head_dim)]
            x_attn.extend(head_out)
        x = linear(x_attn, state_dict[f'layer{li}.attn_wo'])
        x = [a + b for a, b in zip(x, x_residual)]
        # 2) MLP block
        x_residual = x
        x = rmsnorm(x)
        x = linear(x, state_dict[f'layer{li}.mlp_fc1'])
        x = [xi.relu() for xi in x]
        x = linear(x, state_dict[f'layer{li}.mlp_fc2'])
        x = [a + b for a, b in zip(x, x_residual)]

    logits = linear(x, state_dict['lm_head'])
    return logits
*/