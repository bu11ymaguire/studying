#include "tensor.h"
#include <random>
#include <cmath>

Mat create_matrix(int nout, int nin, double std)
{
    Mat result(nout, Vec(nin)); //nout * nin 크기의 행렬

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> dist(0.0, std);
    
    for(int r = 0; r < nout; r++)
    {
        for(int l = 0; l < nin; l++)
        {
            result[r][l] = new Value(dist(gen));
        }
    }

    return result;
}

Vec linear(const Vec& x, const Mat& w)
{
    Vec result;
    for(size_t r = 0; r < w.size(); ++r)
    {
        Value* row_sum = new Value(0.0);

        for(size_t c = 0; c <x.size(); ++c)
        {
            Value* prod = *w[r][c] * x[c];

            row_sum = *row_sum + prod;
        }

        result.push_back(row_sum);
    }
    return result;
}

Vec softmax(const Vec& logits)
{
    
    double max_val = logits[0]->data;
    for(size_t i = 1; i < logits.size(); ++i)
    {
        if (logits[i]->data > max_val){
            max_val = logits[i]->data;
        }
    }

    Value* v_max = new Value(max_val);

    Vec exps;

    for(size_t i = 0; i < logits.size(); ++i)
    {
        Value* diff =  *logits[i] - v_max;
        Value* e_val = diff->exp();
        exps.push_back(e_val);
    }

    Value* total = new Value(0.0);

    for(const auto& EXP: exps){
        total = *total + EXP;
    }

    Vec result;

    for(const auto& EXP: exps){
        result.push_back(*EXP/total);
    }

    return result;
}

Vec rmsnorm(const Vec& x)
{
    Value* ms = new Value(0.0);
    for(const auto& traversal: x)
    {
        ms = *ms +  (*traversal * traversal);
    }

    Value* n = new Value(static_cast<double>(x.size()));
    ms = *ms /n;

    Value* eps = new Value(1e-5);

    Value* scale = (*ms + eps)->power(-0.5);

    Vec result;
    for(const auto& traversal: x){
        result.push_back(*traversal * scale);
    }

    return result;
}