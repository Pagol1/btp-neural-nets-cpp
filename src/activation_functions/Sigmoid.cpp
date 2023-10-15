#include "Sigmoid.h"
/* TEST ONLY
#include <iostream>
*/
Sigmoid::Sigmoid() {
    // Do nothing
}

bool Sigmoid::multiInput() {
    return false;
}

bool Sigmoid::evalSingle(TYPE &x, act_ret_single &ret) {
#if (TYPE_FIXED) 
    ret.value = 1/(1 + std::exp(-x.to_float()));
#else
    ret.value = 1/(1 + std::exp(-x));
#endif
    ret.grad = ret.value*(1-ret.value);
    return true;
}

bool Sigmoid::evalMulti(std::vector<TYPE> &x, act_ret_vector &ret) {
    return false;    
}

/* TEST ONLY
int main() {
    Sigmoid sig;
    TYPE x = 3.2567234782749374934681;
    act_ret_single ret = sig.evalSingle(x);
    std::cout << "x: " << x << "\n"
        << "sig_val: " << ret.value << "\n"
        << "sig_grad: " << ret.grad << "\n";
    return 0;
}*/
