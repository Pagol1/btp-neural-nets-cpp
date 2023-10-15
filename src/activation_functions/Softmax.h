#pragma once
#include "ActivationFunction.h"

class Softmax : public ActivationFunction {
public:
    Softmax();
    bool multiInput() override;
    bool evalSingle(TYPE &x, act_ret_single &ret) override;
    bool evalMulti(std::vector<TYPE> &x, act_ret_vector &ret) override;
};
