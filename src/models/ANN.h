#pragma once
#include "Model.h"

class ANN : public Model {
public:
    ANN(bool rec);    
    bool addLayer(std::shared_ptr<Layer> new_layer) override;
    bool saveData() override;
    bool loadData() override;
    bool getOutput(std::vector<TYPE> &out) override;
    bool forwardPass(std::vector<TYPE> &input) override;
    bool backwardPass(std::vector<TYPE> &grad_last) override;
private:
    std::vector<std::shared_ptr<Layer>> layers;
    bool record;
    std::vector<std::vector<TYPE>> record_data;
    std::vector<std::vector<TYPE>> z;
    std::vector< std::vector<std::vector<TYPE>> > act_der;
    std::vector<std::vector<TYPE>> grad_z;
    std::vector<TYPE> x;
};
