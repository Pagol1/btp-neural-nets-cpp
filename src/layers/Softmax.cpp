#include "Softmax.h"

Softmax::Softmax(size_t input_size, size_t output_size, bool has_b, 
#ifdef MIXED_PREC
        TYPE_N learn_r) :  has_bias(has_b), in_size(input_size), 
#else   // MIXED_PREC
        TYPE learn_r) :  has_bias(has_b), in_size(input_size), 
#endif  // MIXED_REC
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

#ifdef MIXED_PREC
bool Softmax::forward(eigen_vec_n &in_x, ret_vector &ret) {
#else   // MIXED_PREC
bool Softmax::forward(eigen_vec &in_x, ret_vector &ret) {
#endif  // MIXED_REC
    bool ret_stat = true;

    Eigen::Matrix<float, Eigen::Dynamic, 1> temp_vec(out_size);
    temp_vec = in_x.cast<float>();
    temp_vec = temp_vec.array() - temp_vec.maxCoeff();

    temp_vec = temp_vec.unaryExpr([](float val){ return std::exp(val); });
#ifdef MIXED_PREC
    ret.value = temp_vec.cast<TYPE_N>();
    ret.value = (
            ret.value.cast<TYPE_W>().array()/ret.value.cast<TYPE_W>().sum()
        ).cast<TYPE_N>();
#else   // MIXED_PREC
    ret.value = temp_vec.cast<TYPE>();
    ret.value = ret.value.array()/ret.value.sum();
#endif  // MIXED_REC

#ifdef MIXED_PREC
    eigen_mat_w temp = ret.value.cast<TYPE_W>().array().matrix().asDiagonal();
    ret.grad = ( temp - (ret.value.cast<TYPE_W>() * 
                ret.value.cast<TYPE_W>().transpose()) ).cast<TYPE_N>();
#else   // MIXED_PREC
    ret.grad = ret.value.array().matrix().asDiagonal();
    ret.grad -= ret.value * ret.value.transpose();
#endif  // MIXED_REC
    return ret_stat;
}

#ifdef MIXED_PREC
bool Softmax::backward(eigen_vec_n &grad_next, eigen_vec_n &x_cur, eigen_mat_n &grad_der_mul, eigen_vec_n &grad_cur) {
#else   // MIXED_PREC
bool Softmax::backward(eigen_vec &grad_next, eigen_vec &x_cur, eigen_mat &grad_der_mul, eigen_vec &grad_cur) {
#endif  // MIXED_REC
    grad_cur = grad_der_mul * grad_next;
    return true;
}

#ifdef MIXED_PREC
bool Softmax::updateSGD(TYPE_N norm) {
#else   // MIXED_PREC
bool Softmax::updateSGD(TYPE norm) {
#endif  // MIXED_REC
    return true;
}

bool Softmax::resetGrad() {
    return true;
}
