#include "value.h"
#include <cmath>
#include <set> //헤더 파일은 필요한 cpp에서만 include하는게 빌드에 더 효율적이다!

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

void Value::backward(){
    std::vector<Value*> topo;
    std::set<Value*> visited;

    auto build_topo = [&](auto& self, Value* v) -> void {
        if (visited.find(v) == visited.end())
        {
            visited.insert(v);
        

        for(Value* child: v->children)
        {
            self(self,child);
        }

        topo.push_back(v);
    }};

    build_topo(build_topo,this);

    this->grad = 1.0; //현재 노드의 grad.
    
    for(auto it = topo.rbegin(); it != topo.rend(); ++it)
    {
        Value* v = *it; //위상정렬에서 역방향으로 갈때의 노드.

        for(size_t i = 0; i < v->children.size(); ++i) //해당 노드의 자식들을 순회.
        {
            v->children[i]->grad += v->local_grads[i] * v->grad;
        }
    }
}