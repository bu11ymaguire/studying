

class Value{
public:
    double data;
    double grad;

    Value(double data);

    Value* operator+(Value *other);
    Value* operator*(Value *other);
    Value* power(double exponent);
    Value* log();
    Value* exp();
    Value* relu();

    void backward();
};