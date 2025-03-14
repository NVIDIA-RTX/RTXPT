/*
* Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#ifndef __LIGHTING_BINDINGS_HLSLI__    // using instead of "#pragma once" due to https://github.com/microsoft/DirectXShaderCompiler/issues/3943
#define __LIGHTING_BINDINGS_HLSLI__

#include <donut/shaders/bindless.h>
#include <donut/shaders/binding_helpers.hlsli>

#include "../PathTracer/Lighting/LightingTypes.h"

// Bindings 10-19 are scene lighting: environment map (distant lights) sampling, local lights sampling, etc.
TextureCube<float4> t_EnvironmentMap                                        : register(t10);
Texture2D<float>    t_EnvironmentMapImportanceMap                           : register(t11);

StructuredBuffer<LightingControlData>       t_LightsCB                      : register(t12);
StructuredBuffer<PolymorphicLightInfo>      t_Lights                        : register(t13);
StructuredBuffer<PolymorphicLightInfoEx>    t_LightsEx                      : register(t14);

Buffer<uint>                                t_LightProxyCounters            : register(t15);
Buffer<uint>                                t_LightProxyIndices             : register(t16);
Texture3D<uint>                             t_LightNarrowSamplingBuffer     : register(t17);
Texture2D<uint>                             t_EnvLookupMap                  : register(t18);

RWTexture2D<uint2>                          u_LightFeedbackBuffer           : register(u10);

#endif //__LIGHTING_BINDINGS_HLSLI__
