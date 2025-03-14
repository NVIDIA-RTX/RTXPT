/*
* Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#ifndef __SCENE_BINDINGS_HLSLI__    // using instead of "#pragma once" due to https://github.com/microsoft/DirectXShaderCompiler/issues/3943
#define __SCENE_BINDINGS_HLSLI__

#include <donut/shaders/bindless.h>
#include <donut/shaders/binding_helpers.hlsli>

#include "../PathTracer/Materials/MaterialPT.h"
#include "../SubInstanceData.h"
#include "../OpacityMicroMap/OmmGeometryDebugData.hlsli"

// Bindings 0-9 are scene data: acceleration structures, geometry & instance data, materials, etc.
RaytracingAccelerationStructure SceneBVH                    : register(t0);
StructuredBuffer<SubInstanceData> t_SubInstanceData         : register(t1);
StructuredBuffer<InstanceData> t_InstanceData               : register(t2);
StructuredBuffer<GeometryData> t_GeometryData               : register(t3);
StructuredBuffer<GeometryDebugData> t_GeometryDebugData     : register(t4);
StructuredBuffer<MaterialPTData> t_MaterialPTData           : register(t5);

// Bindless
VK_BINDING(0, 1) ByteAddressBuffer t_BindlessBuffers[]      : register(t0, space1);
VK_BINDING(1, 1) Texture2D t_BindlessTextures[]             : register(t0, space2);

#endif //__SCENE_BINDINGS_HLSLI__
