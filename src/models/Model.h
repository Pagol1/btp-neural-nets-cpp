#pragma once
#include "my_defines.h"
//#include "activation_functions/ActivationFunction.h"
#include "layers/Layer.h"
#include <algorithm>
#include <vector>
//#include <iostream>

class Model {
public:
    virtual bool addLayer(std::shared_ptr<Layer> new_layer) = 0;
    virtual bool saveData() = 0;
    virtual bool loadData() = 0;
    virtual bool getOutput(eigen_vec &out) = 0; //virtual bool getOutput(std::vector<TYPE> &out) = 0;
    virtual bool forwardPass(eigen_vec &input) = 0; //virtual bool forwardPass(std::vector<TYPE> &input) = 0;
    virtual bool backwardPass(eigen_vec &grad_last, bool add_record) = 0;   //virtual bool backwardPass(std::vector<TYPE> &grad_last, bool add_record) = 0;
};
