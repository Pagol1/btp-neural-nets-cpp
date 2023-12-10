#include "ANN.h"

#include <iostream>
ANN::ANN(bool rec) : record(rec) {
    
}

bool ANN::addLayer(std::shared_ptr<Layer> new_layer) {
    size_t in_size, out_size;
    bool hw, hb, hc;
    TYPE lr;

    layer_info temp{in_size, out_size, hw, hb, hc, lr};
    layers.push_back(new_layer);
    bool stat = new_layer->getInfo(temp);

    z.push_back(eigen_vec(out_size));
    assertm(
            z[z.size()-1].cols() == 1 && 
            z[z.size()-1].rows() == out_size, 
            "z init");
    if (hc) grad_mul_list.push_back(eigen_mat(in_size, out_size));
    else grad_mul_list.push_back(eigen_mat(in_size, 1));
    assertm(
            grad_mul_list[grad_mul_list.size()-1].cols() == in_size && 
            grad_mul_list[grad_mul_list.size()-1].rows() == out_size, 
            "grad_mul_list init");
    grad_z.push_back(eigen_vec(in_size));
    assertm(
            grad_z[grad_z.size()-1].cols() == 1 && 
            grad_z[grad_z.size()-1].rows() == in_size, 
            "grad_z init");

    /*
    std::cout << "DBM: " << z[z.size()-1].rows() << " " 
        << grad_mul_list[grad_mul_list.size()-1].rows() << "|" 
        << grad_mul_list[grad_mul_list.size()-1].cols() << " "
        << grad_z[grad_z.size()-1].rows() << std::endl;
        */

    return stat;
}

bool ANN::saveData() {
    return false;
}

bool ANN::loadData() {
    return false;
}

bool ANN::getOutput(eigen_vec &out) {
    if (layers.size() == 0) return false;
    out = z[layers.size()-1];
    // std::cout << "DBM: " << z[layers.size()-2] << std::endl;
    return true;
}

bool ANN::forwardPass(eigen_vec &input) {
    x = input;
    if (layers.size() == 0) return false;
    ret_vector temp = {z[0], grad_mul_list[0]};
    bool stat = layers[0]->forward(input, temp);
    for (size_t i=1; i<layers.size(); ++i) {
        ret_vector temp = {z[i], grad_mul_list[i]};
        stat &= layers[i]->forward(z[i-1], temp);
    }
    return stat;
}

bool ANN::backwardPass(eigen_vec &grad_last, bool add_record) {
    if (layers.size() == 0) return false;
    else if (layers.size() > 1) {
        bool stat = layers[layers.size()-1]->backward(grad_last, z[layers.size()-2], grad_mul_list[layers.size()-1], grad_z[layers.size()-1]);
        for (size_t i=layers.size()-2; i>0; --i) {
            stat &= layers[i]->backward(grad_z[i+1], z[i-1], grad_mul_list[i], grad_z[i]);
        }
        stat &= layers[0]->backward(grad_z[1], x, grad_mul_list[0], grad_z[0]);
#if RECORD
/*            for (size_t i=0; i<layers.size()-2; ++i) {
                TYPE max_val_grad = *std::max_element(grad_z[i].begin(), grad_z[i].end());
                TYPE min_val_grad = *std::min_element(grad_z[i].begin(), grad_z[i].end());
                record_grad_min = (record_grad_min < min_val_grad) ? record_grad_min : min_val_grad;
                record_grad_max = (record_grad_max > max_val_grad) ? record_grad_max : max_val_grad;
            }*/
#endif
        return stat;
    }
    else {
        return layers[0]->backward(grad_last, x, grad_mul_list[0], grad_z[0]);
    }
}

bool ANN::getRecord(TYPE &min, TYPE &max) {
    if (!record) return false;
    min = record_grad_min; 
    max = record_grad_max;
    return record;
}
