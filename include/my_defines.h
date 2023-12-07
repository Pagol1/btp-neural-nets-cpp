#pragma once
#include <fixed.h>
#include <cassert>

typedef numeric::fixed<8, 24> fixed;
#define assertm(exp, msg) assert(((void)msg, exp))

#define TYPE fixed
#define TYPE_FIXED 1
#define RECORD false
