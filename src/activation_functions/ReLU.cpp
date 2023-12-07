#include "ReLU.h"
/* TEST ONLY
#include <iostream>
*/
ReLU::ReLU() {
    // Do nothing
}

bool ReLU::multiInput() {
    return false;
}

bool ReLU::evalSingle(TYPE &x, act_ret_single &ret) {
    ret.grad = (x >= 0);
    ret.value = x*ret.grad;
    return true;
}

bool ReLU::evalMulti(std::vector<TYPE> &x, act_ret_vector &ret) {
    return false;    
}

