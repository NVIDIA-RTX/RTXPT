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
#include "BindingDataTypes.hlsli"
#include <donut/shaders/binding_helpers.hlsli>

ConstantBuffer<SampleConstants>         g_Const                         : register(b0);
VK_PUSH_CONSTANT ConstantBuffer<SampleMiniConstants> g_MiniConst        : register(b1);

Texture2D<float4>                       t_LdrColorScratch               : register(t6);

// All outputs are defined here
RWTexture2D<float4>                     u_OutputColor                   : register(u0); // main HDR output - RenderTargets::OutputColor
RWTexture2D<float4>                     u_ProcessedOutputColor          : register(u1); // tonemapping inputs - RenderTargets::ProcessedOutputColor
RWTexture2D<float4>                     u_PostTonemapOutputColor        : register(u2); // tonemapping outputs - RenderTargets::LdrColor

RWTexture2D<uint>                       u_Throughput                    : register(u4); // used by RTXDI, etc. Packed as R11G11B10_FLOAT
RWTexture2D<float4>                     u_MotionVectors                 : register(u5); // used by RTXDI, DLSS/TAA, etc.
RWTexture2D<float>                      u_Depth                         : register(u6); // used by RTXDI, DLSS/TAA, etc.
RWTexture2D<float>                      u_SpecularHitT                  : register(u7); // used by denoisers
RWTexture2D<float>                      u_ScratchFloat1                 : register(u8); // used by post-processing

RWTexture2DArray<uint>                  u_StablePlanesHeader            : register(u40);
RWStructuredBuffer<StablePlane>         u_StablePlanesBuffer            : register(u42);
RWTexture2D<float4>                     u_StableRadiance                : register(u44);
RWStructuredBuffer<PackedPathTracerSurfaceData> u_SurfaceData           : register(u45);

// this is for debugging viz
RWStructuredBuffer<DebugFeedbackStruct> u_FeedbackBuffer                : register(u51);
RWStructuredBuffer<DebugLineStruct>     u_DebugLinesBuffer              : register(u52);
RWStructuredBuffer<DeltaTreeVizPathVertex> u_DebugDeltaPathTree         : register(u53);
RWStructuredBuffer<PathPayload>         u_DeltaPathSearchStack          : register(u54);

// DLSS-RR inputs - leaving them globally accessible so we can move the writes where most optimal
RWTexture2D<float4>                     u_RRDiffuseAlbedo               : register(u70);
RWTexture2D<float4>                     u_RRSpecAlbedo                  : register(u71);
RWTexture2D<float4>                     u_RRNormalsAndRoughness         : register(u72);
RWTexture2D<float2>                     u_RRSpecMotionVectors           : register(u73);
RWTexture2D<float4>                     u_RRTransparencyLayer           : register(u74);
RWTexture2D<float4>                     u_DenoisingAvgLayerRadiance     : register(u75);

// Local cubemap for raster renderer
TextureCube<float4>                     t_LocalCubemapGGX               : register(t80);  // GGX-filtered local cubemap
TextureCube<float4>                     t_DiffuseIrradianceCube         : register(t81);  // SH-based diffuse irradiance
Texture2D<float4>                       t_SSRBlurChain                  : register(t82);  // SSR result with blur mips
Texture2D<float2>                       t_BRDFLUT                       : register(t83);  // Split-sum BRDF integration LUT
Texture2D<float>                        t_DepthHierarchy                : register(t84);  // Hi-Z depth pyramid for SSR

// SSR result UAV (depth hierarchy UAVs u80-84 are in a dedicated binding set in IntroSample)
RWTexture2D<float4>                     u_SSRResult                     : register(u85);  // SSR output (rgb=color, a=confidence)

// GTAO output (full-res, written by GTAORenderer, read by deferred lighting)
Texture2D<float>                        t_GTAOOutput                    : register(t86);

// Previous frame depth (full-res, for temporal reprojection / disocclusion detection)
Texture2D<float>                        t_PrevDepth                     : register(t87);

#endif // #ifndef __SHADER_RESOURCE_BINDINGS_HLSLI__
