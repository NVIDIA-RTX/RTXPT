/*
* Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#ifndef __PATH_PAYLOAD_HLSLI__ // using instead of "#pragma once" due to https://github.com/microsoft/DirectXShaderCompiler/issues/3943
#define __PATH_PAYLOAD_HLSLI__

#include "Config.h"    


// Packed and aligned representation of PathState in a pre-raytrace state (no HitInfo, but path.origin and path.direction set)
// NOTE: we tried removing this and relying just on packed PathState, but we're blocked by various compiler bugs such as https://github.com/microsoft/DirectXShaderCompiler/issues/6464
struct PAYLOAD_QUALIFIER PathPayload
{
    uint4   packed[5] PAYLOAD_FIELD_RW_ALL; // normal reference codepath

#ifdef PATH_STATE_DEFINED
    static PathPayload pack(const PathState path);
    static PathState unpack(const PathPayload p);
#endif
};

#ifdef PATH_STATE_DEFINED

PathPayload PathPayload::pack(const PathState path)
{
    PathPayload p; // = {};

    // 0
    p.packed[0] = path.PackOriginId;

    // 1
    p.packed[1] = path.PackDirSceneLength;

    // 2
    p.packed[2].xy = path.pack23;
#if PATH_TRACER_MODE==PATH_TRACER_MODE_BUILD_STABLE_PLANES
    p.packed[2].zw = path.imageXformPacked;
#else
    p.packed[2].zw = path.pack45;
#endif

    // 3
#if RTXPT_NESTED_DIELECTRICS_QUALITY > 0
    p.packed[3].xy = uint2(path.interiorList.slots[0], path.interiorList.slots[1]); // all 32 bits necessary
#else
    p.packed[3].xy = 0;
#endif
    p.packed[3].z  = path.packedCounters;
    p.packed[3].w  = path.stableBranchID;

    // 4
    p.packed[4].x = path.rayCone.widthSpreadAngleFP16;
    p.packed[4].y = path.pack0;
    p.packed[4].z = path.pack1;
    p.packed[4].w = path.flagsAndVertexIndex;

    return p;
}

#if 0
bool isActive(PathPayload p)
{
    uint flagsAndVertexIndex = p.packed[1].w;
    const uint bit = ((uint)PathFlags::active) << kVertexIndexBitCount;
    return (flagsAndVertexIndex & bit) != 0;
}
#endif

PathState PathPayload::unpack(const PathPayload p)
{
    PathState path; // = {};


    // 0
    path.PackOriginId       = p.packed[0];

    // 1
    path.PackDirSceneLength = p.packed[1];

    // 2
    path.pack23             = p.packed[2].xy;
#if PATH_TRACER_MODE==PATH_TRACER_MODE_BUILD_STABLE_PLANES
    path.imageXformPacked   = p.packed[2].zw;
#else
    path.pack45             = p.packed[2].zw;
#endif

    // 3
#if RTXPT_NESTED_DIELECTRICS_QUALITY > 0
    path.interiorList.slots[0]  = p.packed[3].x;
    path.interiorList.slots[1]  = p.packed[3].y;
#endif
    path.packedCounters         = p.packed[3].z;
    path.stableBranchID         = p.packed[3].w;

    // 4
    path.rayCone.widthSpreadAngleFP16   = p.packed[4].x;
    path.pack0                          = p.packed[4].y;
    path.pack1                          = p.packed[4].z;
    path.flagsAndVertexIndex            = p.packed[4].w;   

    return path;
}

#endif

#endif // __PATH_PAYLOAD_HLSLI__
