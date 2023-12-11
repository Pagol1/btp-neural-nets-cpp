#pragma once
#include <fixed.h>
#include <cassert>
#include <Eigen/Dense>

typedef numeric::fixed<24, 40> fixed;
#define assertm(exp, msg) assert(((void)msg, exp))

#define TYPE float
#define TYPE_FIXED 0
#define RECORD false

#define BATCH_SIZE 64
#define L2_NORM 1e-8
#define LOSS_ZERO 1000

/* Adam | Ref: https://www.ruder.io/optimizing-gradient-descent/ */
#define LR 0.0002
#define ADAM_P1 0.9
#define ADAM_P2 0.999
#define ADAM_EPS 1e-8

typedef Eigen::Matrix<TYPE, Eigen::Dynamic, Eigen::Dynamic> eigen_mat;
typedef Eigen::Matrix<TYPE, Eigen::Dynamic, 1>              eigen_vec;
