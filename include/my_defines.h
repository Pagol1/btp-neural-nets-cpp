#pragma once
#include <fixed.h>
#include <cassert>
#include <Eigen/Dense>

typedef numeric::fixed<24, 40> fixed;
#define assertm(exp, msg) assert(((void)msg, exp))

#define TYPE float
#define TYPE_FIXED 0
#define RECORD false

#define BATCH_SIZE 1000
#define L2_NORM
#define LR 1e-7
#define LOSS_ZERO 1000

typedef Eigen::Matrix<TYPE, Eigen::Dynamic, Eigen::Dynamic> eigen_mat;
typedef Eigen::Matrix<TYPE, Eigen::Dynamic, 1>              eigen_vec;
