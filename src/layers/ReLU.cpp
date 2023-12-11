#include "ReLU.h"

ReLU::ReLU(size_t input_size, size_t output_size, bool has_b, 
        TYPE learn_r) :  has_bias(has_b), in_size(input_size), 
        out_size(output_size), lr(learn_r) {
    assertm(in_size == out_size, "ReLU must have equal sizes!");
}

bool ReLU::hasWeights() {
    return false;
}

bool ReLU::hasBias() {
    return false;
}

bool ReLU::hasCross() {
    return false;
}

bool ReLU::getInfo(layer_info &x) {
    x.in_size = in_size;
    x.out_size = out_size;
    x.has_w = false;
    x.has_b = false;
    x.has_cross = false;
    x.lr = 0;
    return true;
}

bool ReLU::setData(layer_data &x) {
    return false;
}

bool ReLU::getData(layer_data &x) {
    return false;
}

bool ReLU::forward(eigen_vec &in_x, ret_vector &ret) {
    bool ret_stat = true;
    
    Eigen::Matrix<float, Eigen::Dynamic, 1> temp_vec(out_size);
    temp_vec = in_x.cast<float>();
    temp_vec = temp_vec.unaryExpr([](float val){ return (float)(val >= 0); });

    eigen_vec mask(out_size);
    mask = temp_vec.cast<TYPE>();

    ret.value = in_x;
    ret.value = ret.value.array() * mask.array();
    ret.grad = mask;
    return ret_stat;
}

bool ReLU::backward(eigen_vec &grad_next, eigen_vec &x_cur, eigen_mat &grad_der_mul, eigen_vec &grad_cur) {
    grad_cur = grad_der_mul.array().matrix().asDiagonal() * grad_next;
    return true;
}

bool ReLU::updateSGD(TYPE norm) {
    return true;
}
/*
int main() {
    ReLU layer(5, 5, true, 0.01);
    
    eigen_vec x(5);
    std::vector<TYPE> vec{1, 0.5, -0.5, 0, -1};
    for (int i=0; i<vec.size(); ++i) x[i] = vec[i];
    eigen_vec z_val; z_val.resize(5);
    eigen_mat z_grad; z_grad.resize(5, 1);
    ret_vector temp{z_val, z_grad};
    bool stat = layer.forward(x, temp);
    std::cout << stat << "\n" << x << "\n\n" << z_val << "\n\n" << z_grad << "\n";
    return 0;
}*/
