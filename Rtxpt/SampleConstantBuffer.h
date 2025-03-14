/*
* Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#ifndef __SAMPLE_CONSTANT_BUFFER_H__
#define __SAMPLE_CONSTANT_BUFFER_H__

#if !defined(__cplusplus) // not needed in the port so far
#pragma pack_matrix(row_major) // matrices below are expected in row_major
#else
using namespace donut::math;
#endif


#include <donut/shaders/view_cb.h>

#include "PathTracer/PathTracerShared.h"

#include "PathTracer/PathTracerDebug.hlsli"

#include "PathTracer/Lighting/LightingTypes.h"

struct SampleConstants
{    
    PlanarViewConstants view;
    PlanarViewConstants previousView;
    EnvMapSceneParams envMapSceneParams;
    EnvMapImportanceSamplingParams envMapImportanceSamplingParams;
    PathTracerConstants ptConsts;
    DebugConstants debug;
    float4 denoisingHitParamConsts;

    uint MaterialCount;
    uint _padding0;
    uint _padding1;
    uint _padding2;
};

// Used in a couple of places like multipass postprocess where you want to keep SampleConstants the same for all passes, but send just a few additional per-pass parameters 
// In path tracing used to pass subSampleIndex (when enabled).
// Set as 'push constants' (root constants)
struct SampleMiniConstants
{
    uint4 params;
};

#endif // __SAMPLE_CONSTANT_BUFFER_H__