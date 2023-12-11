#pragma once
#include <cmath>
#include <Eigen/Core>
#include "my_defines.h"
#include "Layer.h"

class Sigmoid : public Layer {
public:
    Sigmoid(size_t input_size, size_t output_size, bool has_b, TYPE learn_r);

    // Override base class methods
    bool hasWeights() override;
    bool hasBias() override;
    bool hasCross() override;
    bool getInfo(layer_info &x) override;
    bool getData(layer_data &x) override;
    bool setData(layer_data &x) override;

    bool forward(eigen_vec &in_x, ret_vector &ret) override;
    bool backward(eigen_vec &grad_next, eigen_vec &x_cur, eigen_mat &grad_der_mul, eigen_vec &grad_cur) override;
    bool updateSGD(TYPE norm) override;
    bool resetGrad() override;

private:
    bool has_bias;
    size_t in_size, out_size;
    TYPE lr;
};
