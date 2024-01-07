#include "Sigmoid.h"
/*
#include <iostream>
#include <vector>
*/
Sigmoid::Sigmoid(size_t input_size, size_t output_size, bool has_b, 
#ifdef MIXED_PREC
        TYPE_N learn_r) :  has_bias(has_b), in_size(input_size), 
#else   // MIXED_PREC
        TYPE learn_r) :  has_bias(has_b), in_size(input_size), 
#endif  // MIXED_REC
        out_size(output_size), lr(learn_r) {
    assertm(in_size == out_size, "Sigmoid must have equal sizes!");
}

bool Sigmoid::hasWeights() {
    return false;
}

bool Sigmoid::hasBias() {
    return false;
}

bool Sigmoid::hasCross() {
    return false;
}

bool Sigmoid::getInfo(layer_info &x) {
    x.in_size = in_size;
    x.out_size = out_size;
    x.has_w = false;
    x.has_b = false;
    x.has_cross = false;
    x.lr = 0;
    return true;
}

bool Sigmoid::setData(layer_data &x) {
    return false;
}

bool Sigmoid::getData(layer_data &x) {
    return false;
}

#ifdef MIXED_PREC
bool Sigmoid::forward(eigen_vec_n &in_x, ret_vector &ret) {
#else   // MIXED_PREC
bool Sigmoid::forward(eigen_vec &in_x, ret_vector &ret) {
#endif  // MIXED_REC
    bool ret_stat = true;

    Eigen::Matrix<float, Eigen::Dynamic, 1> temp_vec(out_size);
    temp_vec = in_x.cast<float>();
    temp_vec = temp_vec.unaryExpr([](float val){ return 1/(1 + std::exp(-val)); });
#ifdef MIXED_PREC
    ret.value = temp_vec.cast<TYPE_N>();
#else   // MIXED_PREC
    ret.value = temp_vec.cast<TYPE>();
#endif  // MIXED_REC
 
    temp_vec = temp_vec.array() * (1.0 - temp_vec.array());
#ifdef MIXED_PREC
    ret.grad = temp_vec.cast<TYPE_N>();
#else   // MIXED_PREC
    ret.grad = temp_vec.cast<TYPE>();
#endif  // MIXED_REC
    return ret_stat;
}

#ifdef MIXED_PREC
bool Sigmoid::backward(eigen_vec_n &grad_next, eigen_vec_n &x_cur, eigen_mat_n &grad_der_mul, eigen_vec_n &grad_cur) {
#else   // MIXED_PREC
bool Sigmoid::backward(eigen_vec &grad_next, eigen_vec &x_cur, eigen_mat &grad_der_mul, eigen_vec &grad_cur) {
#endif  // MIXED_REC
    grad_cur = grad_der_mul.array().matrix().asDiagonal() * grad_next;
    return true;
}

#ifdef MIXED_PREC
bool Sigmoid::updateSGD(TYPE_N norm) {
#else   // MIXED_PREC
bool Sigmoid::updateSGD(TYPE norm) {
#endif  // MIXED_REC
    return true;
}

bool Sigmoid::resetGrad() {
    return true;
}
/* TEST
int main() {
    Sigmoid layer(5, 5, true, 0.01);
    
    eigen_vec x(5);
    std::vector<TYPE> vec{1, 0.5, -0.5, 0, -1};
    for (int i=0; i<vec.size(); ++i) x[i] = vec[i];
    eigen_vec z_val; z_val.resize(5);
    eigen_mat z_grad; z_grad.resize(5, 1);
    ret_vector temp{z_val, z_grad};
    bool stat = layer.forward(x, temp);
    std::cout << stat << "\n" << x << "\n\n" << z_val << "\n\n" << z_grad << "\n";
    return 0;
} */
