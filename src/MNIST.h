#pragma once
#include "my_defines.h"
#include <vector>
#include <memory>
#include <iostream>
#include <algorithm>
#include <iterator>
#include <string>
#include <cmath>
#include <Eigen/Core>
#include "models/ANN.h"
#include "layers/Layer.h"
#include "layers/Linear.h"
#include "layers/Sigmoid.h"
#include "layers/Softmax.h"
#include "layers/ReLU.h"
//#include "layers/DenseLayer.h"
//#include "activation_functions/Sigmoid.h"
//#include "activation_functions/Softmax.h"
//#include "activation_functions/ReLU.h"

class MNIST {
public:
    MNIST(std::vector<std::vector<uint8_t>> &tr_x, std::vector<uint8_t> &tr_y, std::vector<std::vector<uint8_t>> &te_x, std::vector<uint8_t> &te_y);
#ifdef MIXED_PREC
    bool getLossVector(int y, eigen_vec_n &loss, TYPE_N &loss_val);
#else   // MIXED_PREC
    bool getLossVector(int y, eigen_vec &loss, TYPE &loss_val);
#endif  // MIXED_PREC
    bool train();
    bool test();
private:
    ANN model;
    //std::vector<std::shared_ptr<ActivationFunction>> act_list; 
    std::vector<std::shared_ptr<Layer>> layer_list;
#ifdef MIXED_PREC
    std::vector<eigen_vec_n> train_x, test_x;
#else   // MIXED_PREC
    std::vector<eigen_vec> train_x, test_x;
#endif  // MIXED_PREC
    std::vector<int> train_y, test_y;
};
