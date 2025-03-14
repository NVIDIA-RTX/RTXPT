/*
* Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#ifndef __SHADER_RESOURCE_BINDINGS_HLSLI__ // using instead of "#pragma once" due to https://github.com/microsoft/DirectXShaderCompiler/issues/3943
#define __SHADER_RESOURCE_BINDINGS_HLSLI__

#include "../SampleConstantBuffer.h"
#include "../RTXDI/ShaderParameters.h"
#include <donut/shaders/binding_helpers.hlsli>

ConstantBuffer<SampleConstants>         g_Const                         : register(b0);
VK_PUSH_CONSTANT ConstantBuffer<SampleMiniConstants> g_MiniConst        : register(b1);

// All outputs are defined here
RWTexture2D<float4>                     u_Output                        : register(u0); // main HDR output

RWTexture2D<uint>                       u_Throughput                    : register(u4); // used by RTXDI, etc. Packed as R11G11B10_FLOAT
RWTexture2D<float4>                     u_MotionVectors                 : register(u5); // used by RTXDI, DLSS/TAA, etc.
RWTexture2D<float>                      u_Depth                         : register(u6); // used by RTXDI, DLSS/TAA, etc.

RWTexture2DArray<uint>                  u_StablePlanesHeader            : register(u40);
RWStructuredBuffer<StablePlane>         u_StablePlanesBuffer            : register(u42);
RWTexture2D<float4>                     u_StableRadiance                : register(u44);
RWStructuredBuffer<PackedPathTracerSurfaceData> u_SurfaceData           : register(u45);

// this is for debugging viz
RWTexture2D<float4>                     u_DebugVizOutput                : register(u50);
RWStructuredBuffer<DebugFeedbackStruct> u_FeedbackBuffer                : register(u51);
RWStructuredBuffer<DebugLineStruct>     u_DebugLinesBuffer              : register(u52);
RWStructuredBuffer<DeltaTreeVizPathVertex> u_DebugDeltaPathTree         : register(u53);
RWStructuredBuffer<PathPayload>         u_DeltaPathSearchStack          : register(u54);

// ReSTIR GI resources
RWTexture2D<float4>                     u_SecondarySurfacePositionNormal: register(u60);
RWTexture2D<float4>                     u_SecondarySurfaceRadiance      : register(u61);

// DLSS-RR inputs - leaving them globally accessible so we can move the writes where most optimal
RWTexture2D<float4>                     u_RRDiffuseAlbedo               : register(u70);
RWTexture2D<float4>                     u_RRSpecAlbedo                  : register(u71);
RWTexture2D<float4>                     u_RRNormalsAndRoughness         : register(u72);
RWTexture2D<float2>                     u_RRSpecMotionVectors           : register(u73);

#if RTXPT_STOCHASTIC_TEXTURE_FILTERING_ENABLE
// Stochastic Texture Filtering blue noise texture
Texture2D<float4>                       t_STBN2DTexture                 : register(t63);
#endif

#endif // #ifndef __SHADER_RESOURCE_BINDINGS_HLSLI__
