#pragma once

#include <vector>

class Value { //AutoGrad의 핵심부품.
public:
    double data;
    double grad;
    std::vector<Value*> children;
    std::vector<double> local_grads;

    Value(double data);
    Value(double data, std::vector<Value*> children, std::vector<double> local_grads); //AutoGrad를 위한 생성자 오버로딩.

    Value* operator+(Value* other);
    Value* operator*(Value* other);
    Value* operator/(Value* other);
    Value* operator-(Value* other);
    
    Value* power(double exponent);
    Value* log();
    Value* exp();
    Value* relu();

    void backward();
};