#pragma once
#include <cmath>
#include <Eigen/Core>
#include "my_defines.h"
#include "Layer.h"

class Softmax : public Layer {
public:
#ifdef MIXED_PREC
    Softmax(size_t input_size, size_t output_size, bool has_b, TYPE_N learn_r);
#else   // MIXED_PREC
    Softmax(size_t input_size, size_t output_size, bool has_b, TYPE learn_r);
#endif  // MIXED_REC

    // Override base class methods
    bool hasWeights() override;
    bool hasBias() override;
    bool hasCross() override;
    bool getInfo(layer_info &x) override;
    bool getData(layer_data &x) override;
    bool setData(layer_data &x) override;

#ifdef MIXED_PREC
    bool forward(eigen_vec_n &in_x, ret_vector &ret) override;
    bool backward(eigen_vec_n &grad_next, eigen_vec_n &x_cur, eigen_mat_n &grad_der_mul, eigen_vec_n &grad_cur) override;
    bool updateSGD(TYPE_N norm) override;
#else   // MIXED_PREC
    bool forward(eigen_vec &in_x, ret_vector &ret) override;
    bool backward(eigen_vec &grad_next, eigen_vec &x_cur, eigen_mat &grad_der_mul, eigen_vec &grad_cur) override;
    bool updateSGD(TYPE norm) override;
#endif  // MIXED_REC
    bool resetGrad() override;

private:
    bool has_bias;
    size_t in_size, out_size;
#ifdef MIXED_PREC
    TYPE_N lr;
#else   // MIXED_PREC
    TYPE lr;
#endif  // MIXED_REC
};
