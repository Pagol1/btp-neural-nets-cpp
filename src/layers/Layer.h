// Layer.h
#pragma once
#include <vector>
#include <memory>
#include <random>
#include "my_defines.h"
#include "activation_functions/ActivationFunction.h"

struct layer_info {
    size_t &in_size;
    size_t &out_size;
    bool &has_w;
    TYPE &lr;
};

struct layer_data {
    std::vector<std::vector<TYPE>> &w;
    std::vector<TYPE> &b;
};

class Layer {
public:
    virtual bool hasWeights() = 0;
    virtual bool getInfo(layer_info &x) = 0;
    virtual bool getData(layer_data &x) = 0;
    virtual bool setData(layer_data &x) = 0;
    // Functional
    virtual bool forward(std::vector<TYPE> &in_x, act_ret_vector &ret) = 0;
    virtual bool backward(std::vector<TYPE> &grad_next, std::vector<TYPE> &x_cur, std::vector<std::vector<TYPE>> &cur_act_der, std::vector<TYPE> &grad_cur) = 0;
    virtual bool updateSGD() = 0;
};
