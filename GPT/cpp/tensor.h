#pragma once
#include <vector>
#include "value.h"

using Vec = std::vector<Value*>;

using Mat = std::vector<Vec>;

Mat create_matrix(int nout, int nin, double std_dev = 0.08);

Vec linear(const Vec& x, const Mat& w);

Vec softmax(const Vec& logits);

Vec rmsnorm(const Vec& x);