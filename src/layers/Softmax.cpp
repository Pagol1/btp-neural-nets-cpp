#include "Softmax.h"

Softmax::Softmax(size_t input_size, size_t output_size, bool has_b, 
        TYPE learn_r) :  has_bias(has_b), in_size(input_size), 
        out_size(output_size), lr(learn_r) {
    assertm(in_size == out_size, "Softmax must have equal sizes!");
}

bool Softmax::hasWeights() {
    return false;
}

bool Softmax::hasBias() {
    return false;
}

bool Softmax::hasCross() {
    return false;
}

bool Softmax::getInfo(layer_info &x) {
    x.in_size = in_size;
    x.out_size = out_size;
    x.has_w = false;
    x.has_b = false;
    x.has_cross = true;
    x.lr = 0;
    return true;
}

bool Softmax::setData(layer_data &x) {
    return false;
}

bool Softmax::getData(layer_data &x) {
    return false;
}

bool Softmax::forward(eigen_vec &in_x, ret_vector &ret) {
    bool ret_stat = true;

    Eigen::Matrix<float, Eigen::Dynamic, 1> temp_vec(out_size);
    temp_vec = in_x.cast<float>();
    temp_vec = temp_vec.array() - temp_vec.maxCoeff();

    temp_vec = temp_vec.unaryExpr([](float val){ return std::exp(val); });
    ret.value = temp_vec.cast<TYPE>();
    ret.value = ret.value.array()/ret.value.sum();

    ret.grad = ret.value.array().matrix().asDiagonal();
    ret.grad -= ret.value * ret.value.transpose();
    return ret_stat;
}

bool Softmax::backward(eigen_vec &grad_next, eigen_vec &x_cur, eigen_mat &grad_der_mul, eigen_vec &grad_cur) {
    grad_cur = grad_der_mul * grad_next;
    return true;
}

bool Softmax::updateSGD(TYPE norm) {
    return true;
}

bool Softmax::resetGrad() {
    return true;
}
