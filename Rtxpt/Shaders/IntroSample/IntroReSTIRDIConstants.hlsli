/*
* Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#ifndef __INTRO_RESTIR_DI_CONSTANTS_HLSLI__
#define __INTRO_RESTIR_DI_CONSTANTS_HLSLI__

#define RTXDI_WITH_RESTIR_GI
#include <Rtxdi/RtxdiParameters.h>

struct IntroReSTIRDIConstants
{
    RTXDI_RuntimeParameters runtimeParams;
    RTXDI_Parameters restirDI;
    uint numAnalyticLocalLightSamples;  // ReSTIR DI Initial: analytic local-light RIS candidate count
    uint numTriangleLocalLightSamples;  // ReSTIR DI Initial: emissive-triangle RIS candidate count
    uint numInfiniteLightSamples;       // ReSTIR DI Initial: infinite (directional) light candidate count
    uint numEnvironmentSamples;         // ReSTIR DI Initial: environment-map RIS candidate count (0 disables the env stream)
    uint lightRegionCapacity;           // Number of slots in one light-buffer half
    uint lightCount;                    // Populated slots per half; layout must match before reuse
    uint temporalHistoryValid;          // Previous frame produced compatible reservoirs and G-buffer history
    uint padding;
};

#endif // __INTRO_RESTIR_DI_CONSTANTS_HLSLI__
