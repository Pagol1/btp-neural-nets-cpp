#pragma once
#include "ActivationFunction.h"

class ReLU : public ActivationFunction {
public:
    ReLU();
    bool multiInput() override;
    bool evalSingle(TYPE &x, act_ret_single &ret) override;
    bool evalMulti(std::vector<TYPE> &x, act_ret_vector &ret) override;
};
