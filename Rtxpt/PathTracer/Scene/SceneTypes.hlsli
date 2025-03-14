/*
* Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#ifndef __SCENE_TYPES_HLSLI__ // using instead of "#pragma once" due to https://github.com/microsoft/DirectXShaderCompiler/issues/3943
#define __SCENE_TYPES_HLSLI__

#include "../Config.h"    

#if defined(__cplusplus)
#error fix below first
// due to HLSL bug in one of the currently used compilers, it will still try to include these even though they're #ifdef-ed out
// #include "Utils/Math/PackedFormats.h"
#else
#include "../Utils/Math/PackedFormats.hlsli"
#endif

struct GeometryInstanceID
{
    uint data;
    static GeometryInstanceID   make( uint instanceIndex, uint geometryIndex )          { GeometryInstanceID ret; ret.data = (geometryIndex << 16) | instanceIndex; return ret; }
    uint                        getInstanceIndex( )                                     { return data & 0xFFFF; }
    uint                        getGeometryIndex( )                                     { return data >> 16; }
};

#endif // __SCENE_TYPES_HLSLI__