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

#if NON_PATH_TRACING_PASS || defined(__cplusplus) || (__SHADER_TARGET_MAJOR < 6 || __SHADER_TARGET_MINOR < 8)
    #define RAYPAYLOAD_QUALIFIER
    #define RAYPAYLOAD_FIELD_QUALIFIER
#else
    #define RAYPAYLOAD_QUALIFIER        [raypayload] 
    #define RAYPAYLOAD_FIELD_QUALIFIER  : read(caller, closesthit, miss) : write(caller, closesthit, miss)
#endif


// packed and aligned representation of PathState in a pre-raytrace state (no HitInfo, but path.origin and path.direction set)
struct RAYPAYLOAD_QUALIFIER PathPayload
{
// #if PATH_TRACER_MODE==PATH_TRACER_MODE_REFERENCE      
    uint4   packed[5] RAYPAYLOAD_FIELD_QUALIFIER; // normal reference codepath

#ifdef PATH_STATE_DEFINED
    static PathPayload pack(const PathState path);
    static PathState unpack(const PathPayload p, const PackedHitInfo packedHitInfo);
#endif
};

#ifdef PATH_STATE_DEFINED

PathPayload PathPayload::pack(const PathState path)
{
    PathPayload p; // = {};

    // 0
    p.packed[0].xyz = asuint(path.origin);      // 3xfp32 absolutely necessary for precision
    p.packed[0].w   = path.id;                  // path.id packs at least pixel pos.xy, so 32bits necessary

    // 1
    //p.packed[1].xyz = asuint(path.dir);         // 3xfp32 absolutely necessary for precision
    p.packed[1].xy = asuint(Encode_Oct(path.dir));
#if PATH_TRACER_MODE==PATH_TRACER_MODE_FILL_STABLE_PLANES
    p.packed[1].z = ((f32tof16(clamp(path.specHitT, 0, HLF_MAX))) << 16) | (f32tof16(clamp(path.sceneLengthFromDenoisingLayer, 0, HLF_MAX)));
#else
    p.packed[1].z = 0;
#endif

    p.packed[1].w = path.flagsAndVertexIndex;   // all 32bits used

    // 2
    p.packed[2].xy = uint2(path.interiorList.slots[0], path.interiorList.slots[1]); // all 32 bits necessary
    p.packed[2].z  = path.rayCone.widthSpreadAngleFP16;                             // all 32 bits necessary
    p.packed[2].w  = path.packedCounters;                                           // all 32 bits necessary

    // 3
    p.packed[3].x = ((f32tof16(clamp(path.thp.x, 0, HLF_MAX))) << 16) | (f32tof16(clamp(path.thp.y, 0, HLF_MAX)));  // all 32 bits necessary
    p.packed[3].y = ((f32tof16(clamp(path.thp.z, 0, HLF_MAX))) << 16) | (f32tof16(path.fireflyFilterK));            // all 32 bits necessary
    p.packed[3].z = asuint(path.sceneLength);                                                                       // all 32 bits necessary (fp32 needed for precision)
    p.packed[3].w = path.stableBranchID;                                                                            // all 32 bits necessary

    // 4
#if PATH_TRACER_MODE==PATH_TRACER_MODE_BUILD_STABLE_PLANES
    p.packed[4].xy = PackOrthoMatrix(path.imageXform);
#else
    p.packed[4].x = ((f32tof16(clamp(path.L.x, 0, HLF_MAX))) << 16) | (f32tof16(clamp(path.L.y, 0, HLF_MAX)));
    p.packed[4].y = ((f32tof16(clamp(path.L.z, 0, HLF_MAX))) << 16) | (f32tof16(clamp(path.L.w, 0, HLF_MAX)));
#endif
    p.packed[4].z = ((f32tof16(clamp(0/*UNUSED_EMPTY*/, 0, HLF_MAX))) << 16) | (f32tof16(clamp(path.thpRuRuCorrection, 0, HLF_MAX))); 
    p.packed[4].w = (uint(path.packedMISInfo)<<16) | (f32tof16(clamp((float)path.bsdfScatterPdf, 0, HLF_MAX)));

    return p;
}

PathState PathPayload::unpack(const PathPayload p, const PackedHitInfo packedHitInfo)
{
    PathState path; // = {};

    // 0
    path.origin = asfloat(p.packed[0].xyz);
    path.id = p.packed[0].w;

    // 1
    //path.dir = asfloat(p.packed[1].xyz);
    path.dir = Decode_Oct(asfloat(p.packed[1].xy));
    path.flagsAndVertexIndex = p.packed[1].w;
#if PATH_TRACER_MODE==PATH_TRACER_MODE_FILL_STABLE_PLANES
    path.specHitT = f16tof32(p.packed[1].z >> 16);
    path.sceneLengthFromDenoisingLayer = f16tof32(p.packed[1].z & 0xffff);
#endif

    // 2
    path.interiorList.slots = p.packed[2].xy;
    path.rayCone.widthSpreadAngleFP16 = p.packed[2].z;
    path.packedCounters = p.packed[2].w;

    // 3
    path.thp.x = f16tof32(p.packed[3].x >> 16);
    path.thp.y = f16tof32(p.packed[3].x & 0xffff);
    path.thp.z = f16tof32(p.packed[3].y >> 16);
    path.fireflyFilterK = saturate((lpfloat)f16tof32(p.packed[3].y & 0xffff));
    path.sceneLength = asfloat(p.packed[3].z);
    path.stableBranchID = p.packed[3].w;

    // 4
#if PATH_TRACER_MODE==PATH_TRACER_MODE_BUILD_STABLE_PLANES
    path.imageXform = (lpfloat3x3)UnpackOrthoMatrix(p.packed[4].xy);
#else
    path.L.x    = f16tof32(p.packed[4].x >> 16);
    path.L.y    = f16tof32(p.packed[4].x & 0xffff);
    path.L.z    = f16tof32(p.packed[4].y >> 16);
    path.L.w    = f16tof32(p.packed[4].y & 0xffff);
#endif
    /*unused_value = f16tof32(p.packed[4].z >> 16);*/
    path.thpRuRuCorrection              = (lpfloat)f16tof32(p.packed[4].z & 0xffff);
    path.packedMISInfo                  = (lpuint)(p.packed[4].w >> 16);
    path.bsdfScatterPdf                 = (lpfloat)f16tof32(p.packed[4].w & 0xffff);

    //
    path.hitPacked = packedHitInfo;

    return path;
}

#endif

#endif // __PATH_PAYLOAD_HLSLI__
