/*
* Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#ifndef __SAMPLER_BINDINGS_HLSLI__    // using instead of "#pragma once" due to https://github.com/microsoft/DirectXShaderCompiler/issues/3943
#define __SAMPLER_BINDINGS_HLSLI__

#include <donut/shaders/bindless.h>
#include <donut/shaders/binding_helpers.hlsli>

SamplerState s_MaterialSampler                              : register(s0);
SamplerState s_EnvironmentMapSampler                        : register(s1);
SamplerState s_EnvironmentMapImportanceSampler              : register(s2);

#endif //__SAMPLER_BINDINGS_HLSLI__
