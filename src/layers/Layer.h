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
#ifdef MIXED_PREC
    TYPE_N &lr;
#else   // MIXED_PREC
    TYPE &lr;
#endif  // MIXED_PREC
};

struct layer_data {
#ifdef MIXED_PREC
    eigen_mat_n &w;
    eigen_vec_n &b;
#else   // MIXED_PREC
    eigen_mat &w; //std::vector<std::vector<TYPE>> &w;
    eigen_vec &b; //std::vector<TYPE> &b;
#endif  // MIXED_PREC
    bool &has_bias;
};

struct ret_vector {
#ifdef MIXED_PREC
    eigen_vec_n &value;
    eigen_mat_n &grad;
#else   // MIXED_PREC
    eigen_vec &value;
    eigen_mat &grad;
#endif  // MIXED_PREC
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
#ifdef MIXED_PREC
    virtual bool forward(eigen_vec_n &in_x, ret_vector &ret) = 0;
    virtual bool backward(eigen_vec_n &grad_next, eigen_vec_n &x_cur, eigen_mat_n &grad_der_mul, eigen_vec_n &grad_cur) = 0;
    virtual bool updateSGD(TYPE_N norm) = 0;
#else   // MIXED_PREC
    virtual bool forward(eigen_vec &in_x, ret_vector &ret) = 0;
    virtual bool backward(eigen_vec &grad_next, eigen_vec &x_cur, eigen_mat &grad_der_mul, eigen_vec &grad_cur) = 0;
    virtual bool updateSGD(TYPE norm) = 0;
#endif  // MIXED_PREC
    virtual bool resetGrad() = 0;
};
