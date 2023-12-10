#include "Linear.h"

#include <iostream>
/* TEST
#include <vector>
*/

Linear::Linear(size_t input_size, size_t output_size, bool has_b, 
        TYPE learn_r) :  has_bias(has_b), in_size(input_size), 
        out_size(output_size), lr(learn_r) {
    // Randomize - Kaiming He Initialization
    std::random_device rd{};
    std::mt19937 gen{rd()};
    std::normal_distribution<float> dist{0, 2/static_cast<float>(in_size)};
    auto random_TYPE = [&dist, &gen]{ return (TYPE) dist(gen); };
    // Define Weight Matrix
    /* weights.resize(out_size);
    for (auto &r : weights) r.resize(in_size); */
    weights = eigen_mat::NullaryExpr(out_size, in_size, random_TYPE);
    assertm(weights.cols() == in_size && weights.rows() == out_size, "weights init");
    ///////std::cout << "DBL: " << weights << "|" ;
    /* w_diff.resize(out_size);
    for (auto &r : w_diff) r.resize(in_size, 0);*/
    w_diff.setZero(out_size, in_size);
    assertm(w_diff.cols() == in_size && w_diff.rows() == out_size, "w_diff init");
    ///////std::cout << " " << w_diff << "|";
    /*
    for (auto &row : weights)
        for (auto &ele : row)
            ele = random_TYPE();
    */
    if (has_bias) {       
        biases = eigen_vec::NullaryExpr(out_size, random_TYPE); // biases.resize(out_size);
        assertm(biases.cols() == 1 && weights.rows() == out_size, "biases init");
        b_diff.setZero(out_size);   // b_diff.resize(out_size, 0);
        assertm(b_diff.cols() == 1 && b_diff.rows() == out_size, "b_diff init");
        /////////std::cout << " " << biases << "|" << b_diff << std::endl;
        /*for (auto &ele : biases)
            ele = random_TYPE();*/
    }
}

bool Linear::hasWeights() {
    return true;
}

bool Linear::hasBias() {
    return has_bias;
}

bool Linear::hasCross() {
    return true;
}

bool Linear::getInfo(layer_info &x) {
    x.in_size = in_size;
    x.out_size = out_size;
    x.has_w = true;
    x.has_b = has_bias;
    x.has_cross = true;
    x.lr = lr;
    return true;
}

bool Linear::getData(layer_data &x) {
    x.w = weights;
    x.has_bias = has_bias;
    x.b = biases;
    return true;
}

bool Linear::setData(layer_data &x) {
    weights = x.w;
    biases = x.b;
    in_size = weights.cols();
    out_size = weights.rows();
    // Check sizes
    assertm(weights.rows() == biases.cols(), "loaded data size");
    //Old: for (auto &r : weights)
    //    assertm(r.size() == in_size, "loaded data size");
    // Reallocate
    w_diff.resize(out_size, in_size);
    b_diff.resize(out_size, 1);
    /*w_diff.resize(out_size);
    for (auto &r : w_diff) r.resize(in_size, 0);
    b_diff.resize(out_size, 0);*/
    return true;
}

bool Linear::forward(eigen_vec &in_x, ret_vector &ret) {
    bool ret_stat = true;
    ret.value = weights * in_x;
    if (has_bias) ret.value += biases;
    ret.grad = weights.transpose();
    return ret_stat;
}
    /* Old Logic
    if (!has_weights) {
        return activation_function->evalMulti(in_x, ret);
    } else {
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
}*/

/* next activation grad, previous activation, layer data, previous activation grad */
bool Linear::backward(eigen_vec &grad_next, eigen_vec &x_cur, eigen_mat &grad_der_mul, eigen_vec &grad_cur) {
    grad_cur = grad_der_mul * grad_next;
    w_diff += grad_next * x_cur.transpose();
    if (has_bias) b_diff += grad_next;
    return true;
}
/*    if (!has_weights) {
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
}*/

bool Linear::updateSGD(TYPE norm) {
#ifdef L2_NORM
    weights = (1-2*lr)*weights.array();
#endif
    weights -= lr * w_diff / norm;
    w_diff.setZero(out_size, in_size);
    if (has_bias) {
#ifdef L2_NORM
        biases = (1-2*lr)*biases.array();
#endif
        biases -= lr * b_diff / norm;
        b_diff.setZero(out_size);
    }
    return true;
}
/*    if (!has_weights) return false;
    for (size_t j=0; j<out_size; ++j) {
        for (size_t i=0; i<in_size; ++i) {
            weights[j][i] -= lr * w_diff[j][i];
            w_diff[j][i] = 0;
        }
        biases[j] -= lr * b_diff[j];
        b_diff[j] = 0;
    }
    return true;
}*/

/* TEST
int main()
{
    Linear layer(5, 2, true, 0.01);
    
    eigen_vec x(5); 
    for (int i=0; i<5; ++i) x[i] = 1;
    
    eigen_vec z_val; z_val.resize(2);
    eigen_mat z_grad; z_grad.resize(2, 1);
    ret_vector temp{z_val, z_grad};
    bool stat = layer.forward(x, temp);
    std::cout << stat << "\n" << x << "\n\n" << z_val << "\n\n" << z_grad << "\n";
    return 0;
}*/
