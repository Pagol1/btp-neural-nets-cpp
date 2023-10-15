#include "ANN.h"

ANN::ANN(bool rec) : record(rec) {
    
}

bool ANN::addLayer(std::shared_ptr<Layer> new_layer) {
    size_t in_size, out_size;
    bool hw;
    TYPE lr;
    layer_info temp = {in_size, out_size, hw, lr};
    layers.push_back(new_layer);
    bool stat = new_layer->getInfo(temp);
    //
    z.push_back(std::vector<TYPE>(out_size, 0));
    if (hw) {
        act_der.push_back( std::vector<std::vector<TYPE>>(1, std::vector<TYPE>(out_size, 0)) );
    } else {
        act_der.push_back( std::vector<std::vector<TYPE>>(out_size, std::vector<TYPE>(in_size, 0)) );
    }
    grad_z.push_back(std::vector<TYPE>(in_size, 0));
    return stat;
}

bool ANN::saveData() {
    return false;
}

bool ANN::loadData() {
    return false;
}

bool ANN::getOutput(std::vector<TYPE> &out) {
    if (layers.size() == 0) return false;
    out = z[layers.size()-1];
    return true;
}

bool ANN::forwardPass(std::vector<TYPE> &input) {
    x = input;
    if (layers.size() == 0) return false;
    act_ret_vector temp = {z[0], act_der[0]};
    bool stat = layers[0]->forward(input, temp);
    for (size_t i=1; i<layers.size(); ++i) {
        act_ret_vector temp = {z[i], act_der[i]};
        stat &= layers[i]->forward(z[i-1], temp);
    }
    return stat;
}

bool ANN::backwardPass(std::vector<TYPE> &grad_last) {
    if (layers.size() == 0) return false;
    else if (layers.size() > 1) {
        bool stat = layers[layers.size()-1]->backward(grad_last, z[layers.size()-2], act_der[layers.size()-1], grad_z[layers.size()-1]);
        for (size_t i=layers.size()-2; i>0; --i) {
            stat &= layers[i]->backward(grad_z[i+1], z[i-1], act_der[i], grad_z[i]);
        }
        stat &= layers[0]->backward(grad_z[1], x, act_der[0], grad_z[0]);
        return stat;
    }
    else {
        return layers[0]->backward(grad_last, x, act_der[0], grad_z[0]);
    }
}
