#include "DenseLayer.h"

/* TEST
#include <iostream>
#include "activation_functions/Sigmoid.h"
#include <prettyprint.h>
*/

DenseLayer::DenseLayer(size_t input_size, size_t output_size, bool has_w, TYPE learn_r, std::shared_ptr<ActivationFunction> activation) 
    : activation_function(activation), has_weights(has_w), in_size(input_size), out_size(output_size), lr(learn_r) {
    if (has_weights) {
        weights.resize(out_size);
        for (auto &r : weights) r.resize(in_size);
        biases.resize(out_size);
        w_diff.resize(out_size);
        for (auto &r : w_diff) r.resize(in_size, 0);
        b_diff.resize(out_size, 0);
        // Randomize
        std::random_device rd{};
        std::mt19937 gen{rd()};
        std::normal_distribution<float> dist{0, 0.5};
        auto random_TYPE = [&dist, &gen]{ return (TYPE) dist(gen); };
        for (auto &row : weights)
            for (auto &ele : row)
                ele = random_TYPE();
        for (auto &ele : biases)
            ele = random_TYPE();
    }
}

bool DenseLayer::hasWeights() {
    return has_weights;
}

bool DenseLayer::getInfo(layer_info &x) {
    x.in_size = in_size;
    x.out_size = out_size;
    x.has_w = has_weights;
    x.lr = lr;
    return true;
}

bool DenseLayer::getData(layer_data &x) {
    if (!has_weights) return false;
    x.w = weights;
    x.b = biases;
    return true;
}

bool DenseLayer::setData(layer_data &x) {
    if (!has_weights) return false;
    weights = x.w;
    biases = x.b;
    in_size = weights[0].size();
    out_size = weights.size();
    // Check sizes
    assertm(weights.size() == biases.size(), "loaded data size");
    for (auto &r : weights)
        assertm(r.size() == in_size, "loaded data size");
    // Reallocate
    w_diff.resize(out_size);
    for (auto &r : w_diff) r.resize(in_size, 0);
    b_diff.resize(out_size, 0);
    return true;
}

bool DenseLayer::forward(std::vector<TYPE> &in_x, act_ret_vector &ret) {
    if (!has_weights) {
        /* value = [out]
         * grad = [out, in]
         */
        return activation_function->evalMulti(in_x, ret);
    } else {
        /* value = [out]
         * grad = [1, out]
         */
        bool ret_stat = true;
        for (size_t j=0; j<out_size; ++j) {
            ret.value[j] = 0;
            for (size_t i=0; i<in_size; ++i) {
                ret.value[j] += weights[j][i] * in_x[i];
            }
            ret.value[j] += biases[j];
            act_ret_single temp = {ret.value[j], ret.grad[0][j]};
            ret_stat &= activation_function->evalSingle(ret.value[j], temp);
        }
        return ret_stat;
    }
}

/* next activation grad, previous activation, layer data, previous activation grad */
bool DenseLayer::backward(std::vector<TYPE> &grad_next, std::vector<TYPE> &x_cur, std::vector<std::vector<TYPE>> &cur_act_der, std::vector<TYPE> &grad_cur) {
    if (!has_weights) {
        for (size_t i = 0; i<in_size; ++i) {
            grad_cur[i] = 0;
            for (size_t j=0; j<out_size; ++j) {
                grad_cur[i] += cur_act_der[j][i] * grad_next[j];
            }
        }
        return true;
    } else {
        for (size_t i=0; i<in_size; ++i) grad_cur[i] = 0;

        for (size_t j=0; j<out_size; ++j) {
            b_diff[j] = cur_act_der[0][j] * grad_next[j];
            for (size_t i=0; i<in_size; ++i) {
                w_diff[j][i] = x_cur[i] * b_diff[j];
                grad_cur[i] += weights[j][i] * b_diff[j];
            }
        }
        return true;
    }
}

bool DenseLayer::updateSGD() {
    if (!has_weights) return false;
    for (size_t j=0; j<out_size; ++j) {
        for (size_t i=0; i<in_size; ++i) {
            weights[j][i] -= lr * w_diff[j][i];
            w_diff[j][i] = 0;
        }
        biases[j] -= lr * b_diff[j];
        b_diff[j] = 0;
    }
    return true;
}

/* TEST
int main()
{
    std::shared_ptr<Sigmoid> ptr_sig(new Sigmoid());
    DenseLayer layer(5, 2, true, 0.01, ptr_sig);
    
    std::vector<TYPE> x(5, 1);

    std::vector<TYPE> z_val(2, 1);
    std::vector<std::vector<TYPE>> z_grad(1);
    z_grad[0].resize(2, 0);

    act_ret_vector temp{z_val, z_grad};
    bool stat = layer.forward(x, temp);
    std::cout << stat << "\n";
    std::cout << x << "\n" << z_val << "\n" << z_grad << "\n";
    return 0;
}
*/
