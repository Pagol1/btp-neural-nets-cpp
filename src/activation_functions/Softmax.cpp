#include "Softmax.h"

Softmax::Softmax() {
    // Do Nothing
}

bool Softmax::multiInput() {
    return true;
}

bool Softmax::evalSingle(TYPE &x, act_ret_single &ret) {
    return false;
}

bool Softmax::evalMulti(std::vector<TYPE> &x, act_ret_vector &ret) {
    // Check sizes
    assertm(ret.value.size() == x.size(), "value size");
    assertm(ret.grad.size() == x.size(), "grad row size");
    for (auto &e : ret.grad) {
        assertm(e.size() == x.size(), "grad col size");
    }
    /* Compute */
#if (TYPE_FIXED) 
        float max_ele = (*std::max_element(x.begin(), x.end())).to_float();
#else 
        float max_ele = (*std::max_element(x.begin(), x.end()));
#endif
    for (size_t i=0; i<x.size(); ++i) {
#if (TYPE_FIXED)
        ret.value[i] = std::exp(x[i].to_float() - max_ele);
#else 
        ret.value[i] = std::exp(x[i] - max_ele);
#endif
    }
    TYPE sum = 0;
    for (auto &ele : ret.value) { 
        sum += ele;
    }
    for (auto &ele : ret.value) {
        ele /= sum;
    }
    for (int i=0; i<x.size(); ++i) {
        for (int j=0; j<x.size(); ++j) {
            ret.grad[i][j] = (i == j) - ret.value[i]*ret.value[j];
        }
    }
    return true;
}
