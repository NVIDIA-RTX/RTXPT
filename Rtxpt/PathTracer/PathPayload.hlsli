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

// packed and aligned representation of PathState in a pre-raytrace state (no HitInfo, but path.origin and path.direction set)
struct PathPayload
{
#if PATH_TRACER_MODE==PATH_TRACER_MODE_REFERENCE      
    uint4   packed[5];                              // normal reference codepath
#else
    uint4   packed[6];                              // generate requires more for imageXForm or various additional radiances
#endif

#ifdef PATH_STATE_DEFINED
    static PathPayload pack(const PathState path);
    static PathState unpack(const PathPayload p, const PackedHitInfo packedHitInfo);
#endif
};

#ifdef PATH_STATE_DEFINED

PathPayload PathPayload::pack(const PathState path)
{
    PathPayload p; // = {};

    p.packed[0].xyz = asuint(path.origin);
    p.packed[0].w = path.id;

    p.packed[1].xyz = asuint(path.dir);
    p.packed[1].w = path.flagsAndVertexIndex;

    p.packed[2].xy = uint2(path.interiorList.slots[0], path.interiorList.slots[1]);
    p.packed[2].z  = path.rayCone.widthSpreadAngleFP16;
    p.packed[2].w  = path.packedCounters;

    p.packed[3].x = ((f32tof16(clamp(path.thp.x, 0, HLF_MAX))) << 16) | (f32tof16(clamp(path.thp.y, 0, HLF_MAX)));
    p.packed[3].y = ((f32tof16(clamp(path.thp.z, 0, HLF_MAX))) << 16) | (f32tof16(path.fireflyFilterK));
    p.packed[3].z = asuint(path.sceneLength);
    p.packed[3].w = path.stableBranchID;

#if PATH_TRACER_MODE!=PATH_TRACER_MODE_FILL_STABLE_PLANES
    float3 radianceVal = path.L;
#else
    float3 radianceVal = path.secondaryL;
#endif
    p.packed[4].x = ((f32tof16(clamp(radianceVal.x, 0, HLF_MAX))) << 16) | (f32tof16(clamp(radianceVal.y, 0, HLF_MAX)));
    p.packed[4].y = ((f32tof16(clamp(radianceVal.z, 0, HLF_MAX))) << 16); // empty space, was "| (f32tof16(clamp(path.pdf, 0, HLF_MAX)));"
    p.packed[4].z = 0; // warning, this is used in some scenarios below
    p.packed[4].w = (uint(path.packedMISInfo)<<16) | (f32tof16(clamp((float)path.bsdfScatterPdf, 0, HLF_MAX)));

#if PATH_TRACER_MODE==PATH_TRACER_MODE_BUILD_STABLE_PLANES
    uint p0 = ((f32tof16(path.imageXform[0].x)) << 16) | (f32tof16(path.imageXform[0].y));
    uint p1 = ((f32tof16(path.imageXform[0].z)) << 16) | (f32tof16(path.imageXform[1].x));
    uint p2 = ((f32tof16(path.imageXform[1].y)) << 16) | (f32tof16(path.imageXform[1].z));
    uint handedness = dot( cross(path.imageXform[0],path.imageXform[1]), path.imageXform[2] ) > 0;  // need to track handedness since mirror reflection flips it
    p.packed[5] = uint4(p0, p1, p2, handedness);    // 31 bits left unused in 'handedness' here :)
#elif PATH_TRACER_MODE==PATH_TRACER_MODE_FILL_STABLE_PLANES
    p.packed[4].z = asuint(path.denoiserSampleHitTFromPlane);
    p.packed[5] = ((f32tof16(clamp(path.denoiserDiffRadianceHitDist, 0, HLF_MAX))) << 16) | (f32tof16(clamp(path.denoiserSpecRadianceHitDist, 0, HLF_MAX)));
#endif

    return p;
}

PathState PathPayload::unpack(const PathPayload p, const PackedHitInfo packedHitInfo)
{
    PathState path; // = {};

    path.origin = asfloat(p.packed[0].xyz);
    path.id = p.packed[0].w;

    path.dir = asfloat(p.packed[1].xyz);
    path.flagsAndVertexIndex = p.packed[1].w;

    path.interiorList.slots = p.packed[2].xy;
    path.rayCone.widthSpreadAngleFP16 = p.packed[2].z;
    path.packedCounters = p.packed[2].w;

    path.thp.x = f16tof32(p.packed[3].x >> 16);
    path.thp.y = f16tof32(p.packed[3].x & 0xffff);
    path.thp.z = f16tof32(p.packed[3].y >> 16);
    path.fireflyFilterK = saturate((lpfloat)f16tof32(p.packed[3].y & 0xffff));
    path.sceneLength = asfloat(p.packed[3].z);
    path.stableBranchID = p.packed[3].w;

    float3 radianceVal = float3( f16tof32(p.packed[4].x >> 16), f16tof32(p.packed[4].x & 0xffff), f16tof32(p.packed[4].y >> 16) );
#if PATH_TRACER_MODE!=PATH_TRACER_MODE_FILL_STABLE_PLANES
    path.L = radianceVal;
#else
    path.secondaryL = radianceVal;
#endif
    // path.pdf = f16tof32(p.packed[4].y & 0xffff); <- removed from path state for now but might come back in

    path.hitPacked = packedHitInfo;

#if PATH_TRACER_MODE==PATH_TRACER_MODE_BUILD_STABLE_PLANES
    uint p0 = p.packed[5].x;
    uint p1 = p.packed[5].y;
    uint p2 = p.packed[5].z;
    uint handedness = p.packed[5].w;
    path.imageXform[0] = lpfloat3(f16tof32(p0 >> 16),     f16tof32(p0 & 0xffff),    f16tof32(p1 >> 16));
    path.imageXform[1] = lpfloat3(f16tof32(p1 & 0xffff),  f16tof32(p2 >> 16),       f16tof32(p2 & 0xffff));
    path.imageXform[0] = normalize(path.imageXform[0]); // not sure these are needed
    path.imageXform[1] = normalize(path.imageXform[1]); // not sure these are needed
    path.imageXform[2] = handedness?cross(path.imageXform[0],path.imageXform[1]):cross(path.imageXform[1],path.imageXform[0]);
#elif PATH_TRACER_MODE==PATH_TRACER_MODE_FILL_STABLE_PLANES
    path.denoiserSampleHitTFromPlane    = asfloat(p.packed[4].z);
    path.denoiserDiffRadianceHitDist    = lpfloat4(f16tof32(p.packed[5] >> 16));
    path.denoiserSpecRadianceHitDist    = lpfloat4(f16tof32(p.packed[5] & 0xffff));
#endif

    path.packedMISInfo  = (lpuint)(p.packed[4].w >> 16);
    path.bsdfScatterPdf = (lpfloat)f16tof32(p.packed[4].w & 0xffff);

    return path;
}

#endif

#endif // __PATH_PAYLOAD_HLSLI__
