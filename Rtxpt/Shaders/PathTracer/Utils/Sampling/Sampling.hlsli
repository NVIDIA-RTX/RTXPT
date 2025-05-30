/*
* Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#ifndef __SAMPLING_HLSLI__ // using instead of "#pragma once" due to https://github.com/microsoft/DirectXShaderCompiler/issues/3943
#define __SAMPLING_HLSLI__

#include "../Math/MathConstants.hlsli"

// See https://www.pbr-book.org/4ed/Sampling_Algorithms/Sampling_1D_Functions#fragment-SamplingInlineFunctions-23

//Note: Lerp from https://www.pbr-book.org/4ed/Monte_Carlo_Integration/Sampling_Using_the_Inversion_Method#Lerp has different order of arguments than the 
//HLSL lerp https://learn.microsoft.com/en-us/windows/win32/direct3dhlsl/dx-graphics-hlsl-lerp ; Intentionally using PBR variant here for consistency, but renamed for clarity. 
float PBRLerp(float x, float a, float b) 
{
    return (1.0 - x) * a + x * b;
}

float Sqr(float x)
{
    return x*x;
}

float Logistic(float x, float s) 
{
    x = abs(x);
    return exp(-x / s) / (s * Sqr(1 + exp(-x / s)));
}

float SampleLogistic(float u, float s) 
{
    return -s * log(1.0 / u - 1.0);
}

float InvertLogisticSample(float x, float s) 
{
    return 1.0 / (1.0 + exp(-x / s));
}

float TrimmedLogisticPDF(float x, float s, float a, float b) 
{
    if (x < a || x > b) return 0;
    return Logistic(x, s) / (InvertLogisticSample(b, s) - InvertLogisticSample(a, s));
}

float SampleTrimmedLogistic(float u, float s, float a, float b) 
{
    u = PBRLerp(u, InvertLogisticSample(a, s), InvertLogisticSample(b, s));
    float x = SampleLogistic(u, s);
    return clamp(x, a, b);
}

float InvertTrimmedLogisticSample(float x, float s, float a, float b) 
{
    return (InvertLogisticSample(x, s) - InvertLogisticSample(a, s)) / (InvertLogisticSample(b, s) - InvertLogisticSample(a, s));
}


#endif // __SAMPLING_HLSLI__
