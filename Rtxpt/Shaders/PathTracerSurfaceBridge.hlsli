/*
* Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#ifndef __PATH_TRACER_SURFACE_BRIDGE_HLSLI__ // using instead of "#pragma once" due to https://github.com/microsoft/DirectXShaderCompiler/issues/3943
#define __PATH_TRACER_SURFACE_BRIDGE_HLSLI__

#include "PathTracerBridge.hlsli"

#include "PathTracer/Surface/Surface.hlsli"

namespace Bridge
{
    static PathTracer::Surface loadSurface( const uint instanceIndex, const uint geometryIndex, const uint triangleIndex, const float2 barycentrics,
        const float3 rayDir, const RayCone rayCone, const int pathVertexIndex, const uint2 pixelPos, DebugContext debug );
    
    static void ExportSurface(const PathState path, PathTracer::Surface surface, float sceneLength, float3 motionVectors );

}

#endif // __PATH_TRACER_SURFACE_BRIDGE_HLSLI__
