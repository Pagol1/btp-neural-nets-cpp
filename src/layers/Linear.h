#pragma once
#include <Eigen/Core>
#include "my_defines.h"
#include "Layer.h"

class Linear : public Layer {
public:
    Linear(size_t input_size, size_t output_size, bool has_b, TYPE learn_r);

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

private:
    bool has_bias;
    size_t in_size, out_size;
    TYPE lr;

    eigen_mat weights, w_diff;
    eigen_vec biases, b_diff;
#ifdef ADAM_P1
    eigen_mat m_w, v_w;
    eigen_vec m_b, v_b;
#endif
    /* Old
    std::vector<std::vector<TYPE>> weights;
    std::vector<std::vector<TYPE>> w_diff;
    std::vector<TYPE> biases;
    std::vector<TYPE> b_diff; */
};
