#include "tensor.h"
#include <random>

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