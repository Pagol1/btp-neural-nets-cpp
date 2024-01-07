#pragma once
#include "Model.h"

class ANN : public Model {
public:
    ANN(bool rec);    
    bool addLayer(std::shared_ptr<Layer> new_layer) override;
    bool saveData() override;
    bool loadData() override;
#ifdef MIXED_PREC
    bool getOutput(eigen_vec_n &out) override;
    bool forwardPass(eigen_vec_n &input) override;
    bool backwardPass(eigen_vec_n &grad_last, bool add_record) override;
    bool getRecord(TYPE_N &min, TYPE_N &max);
#else   // MIXED_PREC
    bool getOutput(eigen_vec &out) override;
    bool forwardPass(eigen_vec &input) override;
    bool backwardPass(eigen_vec &grad_last, bool add_record) override;
    bool getRecord(TYPE &min, TYPE &max);
#endif  // MIXED_REC
private:
    std::vector<std::shared_ptr<Layer>> layers;
    bool record;
#ifdef MIXED_PREC
    TYPE_N record_grad_min = 0, record_grad_max = 0;
#else   // MIXED_PREC
    TYPE record_grad_min = 0, record_grad_max = 0;
#endif  // MIXED_REC
#ifdef MIXED_PREC
    std::vector<eigen_mat_n> record_grad;
    std::vector<eigen_vec_n> z;
    std::vector<eigen_mat_n> grad_mul_list;
    std::vector<eigen_vec_n> grad_z;
    eigen_vec_n x;
#else   // MIXED_PREC
    std::vector<eigen_mat> record_grad; //std::vector<std::vector<TYPE>> record_grad;
    std::vector<eigen_vec> z;   //std::vector<std::vector<TYPE>> z;
    std::vector<eigen_mat> grad_mul_list;   //std::vector< std::vector<std::vector<TYPE>> > act_der;
    std::vector<eigen_vec> grad_z;  //std::vector<std::vector<TYPE>> grad_z;
    eigen_vec x;    //std::vector<TYPE> x;
#endif  // MIXED_REC
};
