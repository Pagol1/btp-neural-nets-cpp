#pragma once
#include "my_defines.h"
#include <vector>
#include <memory>
#include <iostream>
#include <algorithm>
#include <iterator>
#include <string>
#include "models/ANN.h"
#include "layers/DenseLayer.h"
#include "activation_functions/Sigmoid.h"
#include "activation_functions/Softmax.h"

class MNIST {
public:
    MNIST(std::vector<std::vector<uint8_t>> &tr_x, std::vector<uint8_t> &tr_y, std::vector<std::vector<uint8_t>> &te_x, std::vector<uint8_t> &te_y);
    bool getLossVector(int y, std::vector<TYPE> &loss);
    bool train();
    bool test();
private:
    ANN model;
    std::vector<std::shared_ptr<ActivationFunction>> act_list; 
    std::vector<std::shared_ptr<Layer>> layer_list;
    std::vector<std::vector<TYPE>> train_x, test_x;
    std::vector<int> train_y, test_y;
};
