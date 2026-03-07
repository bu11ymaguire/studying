#include "value.h"
#include <cmath> //헤더 파일은 필요한 cpp에서만 include하는게 빌드에 더 효율적이다!

Value::Value(double data)
{ //최초의 Value 생성
    this->data = data;
    this->grad = 0.0;

    this->children = {};
    this->local_grads = {};
}

Value::Value(double data, std::vector<Value*>children, std::vector<double> local_grads)
{
    this->data = data;
    this->grad = 0.0;

    this->children = children;
    this->local_grads = local_grads;
}

Value* Value::operator+(Value* other)
{
   double result = this->data + other->data;
   std::vector<double> grads = {1.0, 1.0}; //덧셈의 국소점 미분은 1.0
   std::vector<Value*> parents = {this,other};

   return new Value(result, parents, grads);
}

Value* Value::operator*(Value* other)
{
    double result = this->data * other->data;
    std::vector<double> grads = {other->data, this->data}; //국소적 미분이니까 grad가 아니라 data를 가져오자!
    std::vector<Value*> parents = {this, other};

    return new Value(result, parents, grads);
}

Value* Value::relu()
{
    if(this->data > 0)
    {
        return new Value(this->data, std::vector<Value*>{this}, std::vector<double>{1.0});
        
    }else{
        return new Value(0.0, std::vector<Value*>{this}, std::vector<double>{0.0});
    }
}

Value* Value::power(double other)
{
    return new Value(std::pow(this->data,other),std::vector<Value*>{this}, std::vector<double>{other * std::pow(this->data,other - 1.0)});
}

Value* Value::log()
{
    return new Value(std::log(this->data),std::vector<Value*>{this}, std::vector<double>{1.0/this->data});
}

Value* Value::exp()
{
    return new Value(std::exp(this->data),std::vector<Value*>{this},std::vector<double>{std::exp(this->data)});
}

Value* Value::operator-(Value* other)
{
    /*
    double result = this->data - other->data;
    std::vector<double> grads = {(-1.0)*other->data, this->data};
    */
    Value* minus = new Value(-1.0);
    Value* for_plus = *other * minus;

    return *this + for_plus;
}

Value* Value::operator/(Value* other)
{
    Value* divide = other->power(-1.0);

    return *this * divide;
}