#include "value.h"

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

Value* Value::power()
{

}