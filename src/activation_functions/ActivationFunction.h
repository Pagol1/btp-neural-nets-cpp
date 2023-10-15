#pragma once
#include "my_defines.h"
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <cmath>

struct act_ret_single {
    TYPE &value;
    TYPE &grad;
};

struct act_ret_vector {
    std::vector<TYPE> &value;
    std::vector<std::vector<TYPE>> &grad;
};

class ActivationFunction {
public:
    virtual bool multiInput() = 0;
    virtual bool evalSingle(TYPE &x, act_ret_single &ret) = 0;
    virtual bool evalMulti(std::vector<TYPE> &x, act_ret_vector &ret) = 0;
};
