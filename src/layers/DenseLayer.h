#pragma once
#include "my_defines.h"
#include "Layer.h"

class DenseLayer : public Layer {
public:
    DenseLayer(size_t input_size, size_t output_size, bool has_w, TYPE learn_r, std::shared_ptr<ActivationFunction> activation);

    // Override base class methods
    bool hasWeights() override;
    bool getInfo(layer_info &x) override;
    bool getData(layer_data &x) override;
    bool setData(layer_data &x) override;

    bool forward(std::vector<TYPE> &in_x, act_ret_vector &ret) override;
    bool backward(std::vector<TYPE> &grad_next, std::vector<TYPE> &x_cur, std::vector<std::vector<TYPE>> &cur_act_der, std::vector<TYPE> &grad_cur) override;
    bool updateSGD() override;

private:
    bool has_weights;
    size_t in_size, out_size;
    TYPE lr;

    std::vector<std::vector<TYPE>> weights;
    std::vector<std::vector<TYPE>> w_diff;
    std::vector<TYPE> biases;
    std::vector<TYPE> b_diff;

    std::shared_ptr<ActivationFunction> activation_function;
};
