#include "Linear.h"

#include <iostream>
/* TEST
#include <vector>
*/

Linear::Linear(size_t input_size, size_t output_size, bool has_b, 
#ifdef MIXED_PREC
        TYPE_N learn_r) :  has_bias(has_b), in_size(input_size), 
#else   // MIXED_PREC
        TYPE learn_r) :  has_bias(has_b), in_size(input_size), 
#endif  // MIXED_REC
        out_size(output_size), lr(learn_r) {
    // Randomize - Kaiming He Initialization
    std::random_device rd{};
    std::mt19937 gen{rd()};
    std::normal_distribution<float> dist{0, 2/static_cast<float>(in_size)};
#ifdef MIXED_PREC
    auto random_TYPE = [&dist, &gen]{ return (TYPE_N) dist(gen); };
#else   // MIXED_PREC
    auto random_TYPE = [&dist, &gen]{ return (TYPE) dist(gen); };
#endif  // MIXED_REC
    // Define Weight Matrix
    /* weights.resize(out_size);
    for (auto &r : weights) r.resize(in_size); */
#ifdef MIXED_PREC
    weights = eigen_mat_n::NullaryExpr(out_size, in_size, random_TYPE);
#else   // MIXED_PREC
    weights = eigen_mat::NullaryExpr(out_size, in_size, random_TYPE);
#endif  // MIXED_REC
    assertm(weights.cols() == in_size && weights.rows() == out_size, "weights init");
    ///////std::cout << "DBL: " << weights << "|" ;
    /* w_diff.resize(out_size);
    for (auto &r : w_diff) r.resize(in_size, 0);*/
    w_diff.setZero(out_size, in_size);
#ifdef ADAM_P1
    m_w.setZero(out_size, in_size);
    v_w.setZero(out_size, in_size);
#endif
    assertm(w_diff.cols() == in_size && w_diff.rows() == out_size, "w_diff init");
    ///////std::cout << " " << w_diff << "|";
    /*
    for (auto &row : weights)
        for (auto &ele : row)
            ele = random_TYPE();
    */
    if (has_bias) {       
#ifdef MIXED_PREC
        biases = eigen_vec_n::NullaryExpr(out_size, random_TYPE); // biases.resize(out_size);
#else   // MIXED_PREC
        biases = eigen_vec::NullaryExpr(out_size, random_TYPE); // biases.resize(out_size);
#endif  // MIXED_REC
        assertm(biases.cols() == 1 && weights.rows() == out_size, "biases init");
        b_diff.setZero(out_size);   // b_diff.resize(out_size, 0);
#ifdef ADAM_P1
        m_b.setZero(out_size);
        v_b.setZero(out_size);
#endif
        assertm(b_diff.cols() == 1 && b_diff.rows() == out_size, "b_diff init");
        /////////std::cout << " " << biases << "|" << b_diff << std::endl;
        /*for (auto &ele : biases)
            ele = random_TYPE();*/
    }
}

bool Linear::hasWeights() {
    return true;
}

bool Linear::hasBias() {
    return has_bias;
}

bool Linear::hasCross() {
    return true;
}

bool Linear::getInfo(layer_info &x) {
    x.in_size = in_size;
    x.out_size = out_size;
    x.has_w = true;
    x.has_b = has_bias;
    x.has_cross = true;
    x.lr = lr;
    return true;
}

bool Linear::getData(layer_data &x) {
    x.w = weights;
    x.has_bias = has_bias;
    x.b = biases;
    return true;
}

bool Linear::setData(layer_data &x) {
    weights = x.w;
    biases = x.b;
    in_size = weights.cols();
    out_size = weights.rows();
    // Check sizes
    assertm(weights.rows() == biases.cols(), "loaded data size");
    // Reallocate
    w_diff.resize(out_size, in_size);
    b_diff.resize(out_size, 1);
    return true;
}

#ifdef MIXED_PREC
bool Linear::forward(eigen_vec_n &in_x, ret_vector &ret) {
#else   // MIXED_PREC
bool Linear::forward(eigen_vec &in_x, ret_vector &ret) {
#endif  // MIXED_REC
    bool ret_stat = true;
#ifdef MIXED_PREC
    if (has_bias)
        ret.value = ( weights.cast<TYPE_W>() * in_x.cast<TYPE_W>() + 
                biases.cast<TYPE_W>() ).cast<TYPE_N>();
    else 
        ret.value = ( weights.cast<TYPE_W>() * in_x.cast<TYPE_W>()
                ).cast<TYPE_N>();
#else   // MIXED_PREC
    ret.value = weights * in_x;
    if (has_bias) ret.value += biases;
#endif  // MIXED_REC
    ret.grad = weights.transpose(); // Already TYPE_N
    return ret_stat;
}

/* next activation grad, previous activation, layer data, previous activation grad */
#ifdef MIXED_PREC
bool Linear::backward(eigen_vec_n &grad_next, eigen_vec_n &x_cur, eigen_mat_n &grad_der_mul, eigen_vec_n &grad_cur) {
#else   // MIXED_PREC
bool Linear::backward(eigen_vec &grad_next, eigen_vec &x_cur, eigen_mat &grad_der_mul, eigen_vec &grad_cur) {
#endif  // MIXED_PREC
#ifdef MIXED_PREC
    grad_cur = ( grad_der_mul.cast<TYPE_W>() * grad_next.cast<TYPE_W>()
            ).cast<TYPE_N>();
    w_diff += ( grad_next.cast<TYPE_W>() * x_cur.transpose().cast<TYPE_W>()
            ).cast<TYPE_N>();
#else   // MIXED_PREC
    grad_cur = grad_der_mul * grad_next;
    w_diff += grad_next * x_cur.transpose();
#endif  // MIXED_REC
    if (has_bias) b_diff += grad_next;
    return true;
}


#ifdef MIXED_PREC
bool Linear::updateSGD(TYPE_N norm) {
#else   // MIXED_PREC
bool Linear::updateSGD(TYPE norm) {
#endif  // MIXED_REC
    /* L2 Normalization [Decay] */
#ifdef L2_NORM
#ifdef MIXED_PREC
    TYPE_N t = (1-2*L2_NORM*lr);
    weights = (
            static_cast<TYPE_W>(t) * weights.cast<TYPE_W>().array()
        ).cast<TYPE_N>();
#else   // MIXED_PREC
    weights = (1-2*L2_NORM*lr)*weights.array();
#endif  // MIXED_REC
#endif

#ifdef ADAM_P1
    w_diff /= norm;
// TODO: Typecast ADAM_P1 and 1-ADAM_P1 from N to W explicitly
#ifdef MIXED_PREC
    t = ADAM_P1;
    m_w = (static_cast<TYPE_W>(t) * m_w.cast<TYPE_W>().array() + 
        static_cast<TYPE_W>(1-t) * w_diff.cast<TYPE_W>().array()
        ).cast<TYPE_N>();
    t = ADAM_P2;
    eigen_mat_n temp = (static_cast<TYPE_W>(1-t) * w_diff.cast<TYPE_W>().array() *
            w_diff.cast<TYPE_W>().array()).cast<TYPE_N>();
    v_w = ( static_cast<TYPE_W>(t) * v_w.cast<TYPE_W>().array()
            + temp.cast<TYPE_W>().array() ).cast<TYPE_N>();
    // Update
    t = 1-ADAM_P1;
    t =  lr/t;
    TYPE_N t_ = ADAM_EPS;
    t_.data_ |= 1;
    weights = (weights.cast<TYPE_W>().array() - ( static_cast<TYPE_W>(t) * 
                ( m_w.array()/(t_ + v_w.array().sqrt()) ).cast<TYPE_W>() )
        ).cast<TYPE_N>();
#else   // MIXED_PREC
    m_w = ADAM_P1*m_w.array() + (1-ADAM_P1)*w_diff.array();
    eigen_mat temp = (1-ADAM_P2)*w_diff.array()*w_diff.array();
    v_w =  ADAM_P2*v_w.array() + temp.array();
    // Update
    weights = weights.array() - (lr/(1-ADAM_P1))*m_w.array()/(ADAM_EPS + v_w.array().sqrt());
#endif  // MIXED_REC
#else   // ADAM_P1
    weights = (weights.cast<TYPE_W>().array() - 
            static_cast<TYPE_W>(lr) * (w_diff / norm).cast<TYPE_W>()
            ).cast<TYPE_N>();
#endif  // ADAM_P1
    w_diff.setZero(out_size, in_size);

    if (has_bias) {
#ifdef L2_NORM
#ifdef MIXED_PREC
        TYPE_N t = L2_NORM;
        biases = ( static_cast<TYPE_W>(1-2*t*lr) * 
                biases.cast<TYPE_W>().array() ).cast<TYPE_N>();
#else   // MIXED_PREC
        biases = (1-2*L2_NORM*lr)*biases.array();
#endif  // MIXED_REC
#endif
#ifdef ADAM_P1
#ifdef MIXED_PREC
        b_diff /= norm;
        t = ADAM_P1;
        m_b = ( static_cast<TYPE_W>(t)*m_b.cast<TYPE_W>().array() + 
            static_cast<TYPE_W>(1-t)*b_diff.cast<TYPE_W>().array()
            ).cast<TYPE_N>();
        t = ADAM_P2;
        eigen_mat_n temp = (static_cast<TYPE_W>(1-t) * 
                b_diff.cast<TYPE_W>().array() * b_diff.cast<TYPE_W>().array()
                ).cast<TYPE_N>();
        v_b = (static_cast<TYPE_W>(t)*v_b.cast<TYPE_W>().array() + 
                temp.cast<TYPE_W>().array()).cast<TYPE_N>();
        // Update    
        t = 1-ADAM_P1;
        t =  lr/t;
        TYPE_N t_ = ADAM_EPS;
        t_.data_ |= 1;
        biases = (biases.cast<TYPE_W>().array() - ( static_cast<TYPE_W>(t) * 
                (m_b.array()/(t_ + v_b.array().sqrt())).cast<TYPE_W>() )
                ).cast<TYPE_N>();
#else   // MIXED_PREC
        b_diff /= norm;
        m_b = ADAM_P1*m_b.array() + (1-ADAM_P1)*b_diff.array();
        eigen_mat temp = (1-ADAM_P2)*b_diff.array()*b_diff.array();
        v_b = ADAM_P2*v_b.array() + temp.array();
        // Update
        biases = biases.array() - (lr/(1-ADAM_P1))*m_b.array()/(ADAM_EPS + v_b.array().sqrt());
#endif  // MIXED_REC
#else   // ADAM_P1
#ifdef MIXED_PREC
        biases = (biases.cast<TYPE_W>() - static_cast<TYPE_W>(lr) * 
                (b_diff / norm).cast<TYPE_W>()).cast<TYPE_N>();
#else   // MIXED_PREC
        biases -= lr * b_diff / norm;
#endif  // MIXED_REC
#endif  // ADAM_P1
        b_diff.setZero(out_size);
    }
    return true;
}

bool Linear::resetGrad() {
    w_diff.setZero(out_size, in_size);
#ifdef ADAM_P1
    m_w.setZero(out_size, in_size);
    v_w.setZero(out_size, in_size);
#endif
    if (has_bias) {
        b_diff.setZero(out_size);
#ifdef ADAM_P1
        m_b.setZero(out_size);
        v_b.setZero(out_size);
#endif
    }
    return true;
}

/* TEST
int main()
{
    Linear layer(5, 2, true, 0.01);
    
    eigen_vec x(5); 
    for (int i=0; i<5; ++i) x[i] = 1;
    
    eigen_vec z_val; z_val.resize(2);
    eigen_mat z_grad; z_grad.resize(2, 1);
    ret_vector temp{z_val, z_grad};
    bool stat = layer.forward(x, temp);
    std::cout << stat << "\n" << x << "\n\n" << z_val << "\n\n" << z_grad << "\n";
    return 0;
}*/
