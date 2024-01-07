#pragma once
#include <fixed.h>
#include <cassert>
#include <Eigen/Dense>

#define MIXED_PREC

#ifdef MIXED_PREC
#define FXPW_I 24
#define FXPW_F 40
    typedef numeric::fixed<FXPW_I, FXPW_F> fixed_w;
#define FXPN_I 12
#define FXPN_F 20
    typedef numeric::fixed<FXPN_I, FXPN_F> fixed_n;
#else   // MIXED_PREC
#define FXP_I 16
#define FXP_F 48
    typedef numeric::fixed<FXP_I, FXP_F> fixed;
#endif   // MIXED_PREC

#define assertm(exp, msg) assert(((void)msg, exp))

#ifdef MIXED_PREC
#define TYPE_N fixed_n
#define TYPE_W fixed_w
#else   // MIXED_PREC
#define TYPE fixed
#endif   // MIXED_PREC
#define TYPE_FIXED 1
#define RECORD false

#define BATCH_SIZE 64
#define L2_NORM 1e-8

#ifdef MIXED_PREC
#define LOSS_ZERO ((1<<(FXPN_I+FXPN_F-2))-1)
#else   // MIXED_PREC
#define LOSS_ZERO ((1<<(FXP_I+FXP_F-2))-1)
#endif  // MIXED_PREC
/* Adam | Ref: https://www.ruder.io/optimizing-gradient-descent/ */
#define LR 0.0002
#define ADAM_P1 0.9
#define ADAM_P2 0.999
#define ADAM_EPS 1e-8

// Readings only
#define ACC_ONLY

#ifdef MIXED_PREC
    typedef Eigen::Matrix<TYPE_N, Eigen::Dynamic, Eigen::Dynamic> eigen_mat_n;
    typedef Eigen::Matrix<TYPE_N, Eigen::Dynamic, 1>              eigen_vec_n;
    typedef Eigen::Matrix<TYPE_W, Eigen::Dynamic, Eigen::Dynamic> eigen_mat_w;
    typedef Eigen::Matrix<TYPE_W, Eigen::Dynamic, 1>              eigen_vec_w;
#else   // MIXED_PREC
    typedef Eigen::Matrix<TYPE, Eigen::Dynamic, Eigen::Dynamic> eigen_mat;
    typedef Eigen::Matrix<TYPE, Eigen::Dynamic, 1>              eigen_vec;
#endif  // MIXED_PREC
