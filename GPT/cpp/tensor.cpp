#include "tensor.h"
#include <random>

Mat create_matrix(int nout, int nin, double std = 0.08){
    return vector<vector<Mat>>(nout,vector<Mat>(nin));
}