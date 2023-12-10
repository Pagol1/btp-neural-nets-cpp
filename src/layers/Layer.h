// Layer.h
#pragma once
#include <vector>
#include <memory>
#include <random>
#include <time.h>
#include "my_defines.h"
// #include "activation_functions/ActivationFunction.h"

struct layer_info {
    size_t &in_size;
    size_t &out_size;
    bool &has_w;
    bool &has_b;
    bool &has_cross;
    TYPE &lr;
};

struct layer_data {
    eigen_mat &w; //std::vector<std::vector<TYPE>> &w;
    bool &has_bias;
    eigen_vec &b; //std::vector<TYPE> &b;
};

struct ret_vector {
    eigen_vec &value;
    eigen_mat &grad;
};

class Layer {
public:
    virtual bool hasWeights() = 0;
    virtual bool hasBias() = 0;
    virtual bool hasCross() = 0;    // Has cross connections => Gradient will be a matrix not a vector
    virtual bool getInfo(layer_info &x) = 0;
    virtual bool getData(layer_data &x) = 0;
    virtual bool setData(layer_data &x) = 0;
    // Functional
    /* virtual bool forward(std::vector<TYPE> &in_x, act_ret_vector &ret) = 0; */
    virtual bool forward(eigen_vec &in_x, ret_vector &ret) = 0;
    /* virtual bool backward(std::vector<TYPE> &grad_next, std::vector<TYPE> &x_cur, std::vector<std::vector<TYPE>> &cur_act_der, std::vector<TYPE> &grad_cur) = 0; */
    virtual bool backward(eigen_vec &grad_next, eigen_vec &x_cur, eigen_mat &grad_der_mul, eigen_vec &grad_cur) = 0;
    virtual bool updateSGD(TYPE norm) = 0;
};
