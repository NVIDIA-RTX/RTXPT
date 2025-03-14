/*
* Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#ifndef __MATH_CONSTANTS_HLSLI__ // using instead of "#pragma once" due to https://github.com/microsoft/DirectXShaderCompiler/issues/3943
#define __MATH_CONSTANTS_HLSLI__

#include "../../Config.h"    

/** This file contains useful numeric constants for use on the GPU.

    It should be included with #include rather than import, as imported files
    do not export macro definitions.

    Note the possible differences between declaring constants using static
    float vs macro definitions. The compiler may treat these differently with
    respect to what precision is used for compile-time constant propagation.
    Slang currently uses fp32 for constant propagation. We get higher
    precision using the pre-evaluated constants below. Ideally, all
    compile-time constants should be evaluated at fp64 or higher precision.
*/

#if !defined(__cplusplus)

// Constants from <math.h>
#define K_E                 2.71828182845904523536  // e
#define K_LOG2E             1.44269504088896340736  // log2(e)
#define K_LOG10E            0.434294481903251827651 // log10(e)
#define K_LN2               0.693147180559945309417 // ln(2)
#define K_LN10              2.30258509299404568402  // ln(10)
#define K_PI                3.14159265358979323846  // pi
#define K_PI_2              1.57079632679489661923  // pi/2
#define K_PI_4              0.785398163397448309616 // pi/4
#define K_1_PI              0.318309886183790671538 // 1/pi
#define K_2_PI              0.636619772367581343076 // 2/pi
#define K_2_SQRTPI          1.12837916709551257390  // 2/sqrt(pi)
#define K_SQRT2             1.41421356237309504880  // sqrt(2)
#define K_SQRT1_2           0.707106781186547524401 // 1/sqrt(2)

// Additional constants
#define K_2PI               6.28318530717958647693  // 2pi
#define K_4PI               12.5663706143591729539  // 4pi
#define K_4_PI              1.27323954473516268615  // 4/pi
#define K_1_2PI             0.159154943091895335769 // 1/2pi
#define K_1_4PI             0.079577471545947667884 // 1/4pi
#define K_SQRTPI            1.77245385090551602730  // sqrt(pi)
#define K_1_SQRT2           0.707106781186547524401 // 1/sqrt(2)

// Numeric limits from <stdint.h>
#define UINT32_MAX          4294967295
#define INT32_MIN           -2147483648
#define INT32_MAX           2147483647

// Numeric limits from <float.h>
#define DBL_DECIMAL_DIG     17                      // # of decimal digits of rounding precision
#define DBL_DIG             15                      // # of decimal digits of precision
#define DBL_EPSILON         2.2204460492503131e-016 // smallest such that 1.0+DBL_EPSILON != 1.0
#define DBL_HAS_SUBNORM     1                       // type does support subnormal numbers
#define DBL_MANT_DIG        53                      // # of bits in mantissa
#define DBL_MAX             1.7976931348623158e+308 // max value
#define DBL_MAX_10_EXP      308                     // max decimal exponent
#define DBL_MAX_EXP         1024                    // max binary exponent
#define DBL_MIN             2.2250738585072014e-308 // min positive value
#define DBL_MIN_10_EXP      (-307)                  // min decimal exponent
#define DBL_MIN_EXP         (-1021)                 // min binary exponent
#define DBL_RADIX           2                       // exponent radix
#define DBL_TRUE_MIN        4.9406564584124654e-324 // min positive value

#define FLT_DECIMAL_DIG     9                       // # of decimal digits of rounding precision
#define FLT_DIG             6                       // # of decimal digits of precision
#define FLT_EPSILON         1.192092896e-07F        // smallest such that 1.0+FLT_EPSILON != 1.0
#define FLT_HAS_SUBNORM     1                       // type does support subnormal numbers
#define FLT_GUARD           0
#define FLT_MANT_DIG        24                      // # of bits in mantissa
#define FLT_MAX             3.402823466e+38F        // max value
#define FLT_MAX_10_EXP      38                      // max decimal exponent
#define FLT_MAX_EXP         128                     // max binary exponent
#define FLT_MIN             1.175494351e-38F        // min normalized positive value
#define FLT_MIN_10_EXP      (-37)                   // min decimal exponent
#define FLT_MIN_EXP         (-125)                  // min binary exponent
#define FLT_NORMALIZE       0
#define FLT_RADIX           2                       // exponent radix
#define FLT_TRUE_MIN        1.401298464e-45F        // min positive value

#endif // #if !defined(__cplusplus)

// Numeric limits for half (IEEE754 binary16)
#define HLF_EPSILON         9.765625e-04F           // smallest such that 1.0+HLF_EPSILON != 1.0
#define HLF_HAS_SUBNORM     1                       // type does support subnormal numbers
#define HLF_MANT_DIG        11
#define HLF_MAX             6.5504e+4F              // max value
#define HLF_MAX_EXP         16                      // max binary exponent
#define HLF_MIN             6.097555160522461e-05F  // min normalized positive value
#define HLF_MIN_EXP         (-14)                   // min binary exponent
#define HLF_RADIX           2
#define HLF_TRUE_MIN        5.960464477539063e-08F  // min positive value

// Because sample values must be strictly less than 1, it�s useful to define a constant, OneMinusEpsilon, that represents the largest 
// representable floating-point constant that is less than 1. (https://www.pbr-book.org/3ed-2018/Sampling_and_Reconstruction/Sampling_Interface)
static const double         cDoubleOneMinusEpsilon = 0x1.fffffffffffffp-1;
static const float          cFloatOneMinusEpsilon = 0x1.fffffep-1;

#endif // __MATH_CONSTANTS_HLSLI__
