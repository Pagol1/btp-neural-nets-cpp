#pragma once
#include "Model.h"

class ANN : public Model {
public:
    ANN(bool rec);    
    bool addLayer(std::shared_ptr<Layer> new_layer) override;
    bool saveData() override;
    bool loadData() override;
    bool getOutput(eigen_vec &out) override;
    bool forwardPass(eigen_vec &input) override;
    bool backwardPass(eigen_vec &grad_last, bool add_record) override;
    bool getRecord(TYPE &min, TYPE &max);
private:
    std::vector<std::shared_ptr<Layer>> layers;
    bool record;
    std::vector<eigen_mat> record_grad; //std::vector<std::vector<TYPE>> record_grad;
    TYPE record_grad_min = 0, record_grad_max = 0;
    std::vector<eigen_vec> z;   //std::vector<std::vector<TYPE>> z;
    std::vector<eigen_mat> grad_mul_list;   //std::vector< std::vector<std::vector<TYPE>> > act_der;
    std::vector<eigen_vec> grad_z;  //std::vector<std::vector<TYPE>> grad_z;
    eigen_vec x;    //std::vector<TYPE> x;
};
