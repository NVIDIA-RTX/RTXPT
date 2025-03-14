/*
* Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#ifndef __PATH_STATE_HLSLI__ // using instead of "#pragma once" due to https://github.com/microsoft/DirectXShaderCompiler/issues/3943
#define __PATH_STATE_HLSLI__

#define PATH_STATE_DEFINED

#include "Config.h"    
#include "SampleGenerators.hlsli"

#include "Scene/HitInfo.hlsli"
#include "Rendering/Materials/InteriorList.hlsli"

#include "Utils/Math/Ray.hlsli"

#include "Rendering/Materials/TexLODHelpers.hlsli"

#include "Lighting/LightingTypes.h"

// Be careful with changing these. PathFlags share 32-bit uint with vertexIndex. For now, we keep 10 bits for vertexIndex.
// PathFlags take higher bits, VertexIndex takes lower bits.
static const uint kVertexIndexBitCount = 10u;
static const uint kVertexIndexBitMask = (1u << kVertexIndexBitCount) - 1u;
static const uint kPathFlagsBitCount = 32u - kVertexIndexBitCount;
static const uint kPathFlagsBitMask = ((1u << kPathFlagsBitCount) - 1u) << kVertexIndexBitCount;
static const uint kStablePlaneIndexBitOffset    = 14+kVertexIndexBitCount; // if changing, must change PathFlags::stablePlaneIndexBit0
static const uint kStablePlaneIndexBitMask      = ((1u << 2)-1u) << kStablePlaneIndexBitOffset;

/** Path flags. The path flags are currently stored in kPathFlagsBitCount bits.
*/
enum class PathFlags
{
    active                          = (1<<0),   ///< Path is active/terminated.
    hit                             = (1<<1),   ///< Result of the scatter ray (0 = miss, 1 = hit).

    transmission                    = (1<<2),   ///< Scatter ray went through a transmission event.
    specular                        = (1<<3),   ///< Scatter ray went through a specular event.
    delta                           = (1<<4),   ///< Scatter ray went through a delta event.

    insideDielectricVolume          = (1<<5),   ///< Path vertex is inside a dielectric volume.
    //<removed, empty space>          = (1<<6),   ///<
    //<removed, empty space>          = (1<<7),   ///<

    diffusePrimaryHit               = (1<<8),   ///< Scatter ray went through a diffuse event on primary hit.
    specularPrimaryHit              = (1<<9),   ///< Scatter ray went through a specular event on primary hit.
    //<removed, empty space>          = (1<<10),  ///<
    deltaTransmissionPath           = (1<<11),  ///< Path started with and followed delta transmission events (whenever possible - TIR could be an exception) until it hit the first non-delta event.
    deltaOnlyPath                   = (1<<12),  ///< There was no non-delta events along the path so far.

    deltaTreeExplorer               = (1<<13),  ///< Debug exploreDeltaTree enabled and this path selected for debugging
    stablePlaneIndexBit0            = (1<<14),  ///< StablePlaneIndex, bit 0 -- just reserving space for kStablePlaneIndexBitOffset & kStablePlaneIndexBitMask which must be 14
    stablePlaneIndexBit1            = (1<<15),  ///< StablePlaneIndex, bit 1 -- just reserving space for kStablePlaneIndexBitOffset & kStablePlaneIndexBitMask which must be 14
    stablePlaneOnPlane              = (1<<16),  ///< Current vertex is on a stable plane; this is where we update stablePlaneBaseScatterDiff
    stablePlaneOnBranch             = (1<<17),  ///< Current vertex is on a stable plane or stable branch; all emission is stable and was already collected
    stablePlaneBaseScatterDiff      = (1<<18),  ///< When stepping off the last stable plane & branch, we had a diffuse scatter event
    stablePlaneOnDeltaBranch        = (1<<19),  ///< The first scatter from a stable plane was a delta event
    stablePlaneOnDominantBranch     = (1<<20),  ///< Are we on the dominant stable plane or one of its branches (landing on a new stable branch will re-set this flag accordingly)

    // Bits to kPathFlagsBitCount are still unused.
    // ^no more flag space! consider moving vertexIndex counter to PackedCounters
};

/** Bounce types. We keep separate counters for all of these.
*/
enum class PackedCounters // each packed to 8 bits, 4 max fits in 32bit uint
{
    DiffuseBounces              = 0,    ///< Diffuse reflection.
    RejectedHits                = 1,    ///< Number of false intersections rejected along the path. This is used as a safeguard to avoid deadlock in pathological cases.
    BouncesFromStablePlane      = 2,    ///< Number of bounces after the last stable plane the path was on (path.vertexIndex - currentStablePlaneVertexIndex)
    //SubSampleIndex              = 3     ///< Used when doing multiple (sub)samples per pixels: when the path gets terminated, this counter is incremented, and if still < 
};

// TODO: Compact encoding to reduce live registers, e.g. packed HitInfo, packed normals.
/** Live state for the path tracer.
*/
struct PathState
{
    uint        id;                     ///< Path ID encodes (pixel, sampleIdx) with 12 bits each for pixel x|y and 8 bits for sample index.
    uint        flagsAndVertexIndex;    ///< Higher kPathFlagsBitCount bits: Flags indicating the current status. This can be multiple PathFlags flags OR'ed together.
                                        ///< Lower kVertexIndexBitCount bits: Current vertex index (0 = camera, 1 = primary hit, 2 = secondary hit, etc.).

    float       sceneLength;            ///< [DO NOT COMPRESS TO 16bit float!] Path length in scene units (was 0.f at primary hit originally, in this implementation it includes camera to primary hit).

    lpfloat     fireflyFilterK;         ///< (0, 1] multiplier for the global firefly filter threshold if used; CAN be compressed to 16bit float!
    lpuint      packedMISInfo;          ///< See NEEBSDFMISInfo
    lpfloat     bsdfScatterPdf;         ///< 0 if delta lobe (zero roughness specular) bounce
    
    uint        packedCounters;         ///< Packed counters for different types of bounces and etc., see PackedCounters.

    uint        stableBranchID;         ///< Path 'stable delta tree' branch ID for finding matching StablePlane; Gets updated on scatter while path isDeltaOnlyPath;

    // Scatter ray
    float3      origin;                 ///< Origin of the scatter ray.
    float3      dir;                    ///< Scatter ray normalized direction.
    
    // removed until needed again for emissive triangle sampling MIS 
    // float3      normal;                 ///< Shading normal at the scatter ray origin.

    PackedHitInfo hitPacked;            ///< Hit information for the scatter ray. This is populated at committed triangle hits. 4 uints (16 bytes)

    float3      thp;                    ///< Path throughput.
#if PATH_TRACER_MODE!=PATH_TRACER_MODE_FILL_STABLE_PLANES // nothing to accumulate in this case - all goes to denoiser
    float3      L;                      ///< Accumulated path contribution.
#else
    lpfloat4    denoiserDiffRadianceHitDist;
    lpfloat4    denoiserSpecRadianceHitDist;
    float3      secondaryL;             ///< Radiance reflected and emitted by the secondary surface
    float       denoiserSampleHitTFromPlane;
#endif

    InteriorList interiorList;          ///< Interior list. Keeping track of a stack of materials with medium properties. Size depends on INTERIOR_LIST_SLOT_COUNT. 2 slots (8 bytes) by default.
    RayCone     rayCone;                ///< 4 or 8 bytes depending on USE_RAYCONES_WITH_FP16_IN_RAYPAYLOAD (on, so 4 bytes by default). 

#if PATH_TRACER_MODE==PATH_TRACER_MODE_BUILD_STABLE_PLANES
    lpfloat3x3  imageXform;             ///< Accumulated rotational image transform along the path. This can be float16_t.
#endif

    // Accessors

    bool isTerminated() { return !isActive(); }
    bool isActive() { return hasFlag(PathFlags::active); }
    bool isHit() { return hasFlag(PathFlags::hit); }
    bool wasScatterTransmission() { return hasFlag(PathFlags::transmission); }                      ///< Get flag indicating that last scatter ray went through a transmission event.
    bool wasScatterSpecular() { return hasFlag(PathFlags::specular); }                              ///< Get flag indicating that last scatter ray went through a specular event.
    bool wasScatterDelta() { return hasFlag(PathFlags::delta); }                                    ///< Get flag indicating that last scatter ray went through a delta event.
    bool isInsideDielectricVolume() { return hasFlag(PathFlags::insideDielectricVolume); }

    bool isDiffusePrimaryHit() { return hasFlag(PathFlags::diffusePrimaryHit); }
    bool isSpecularPrimaryHit() { return hasFlag(PathFlags::specularPrimaryHit); }
    bool isDeltaTransmissionPath() { return hasFlag(PathFlags::deltaTransmissionPath); }
    bool isDeltaOnlyPath() { return hasFlag(PathFlags::deltaOnlyPath); }

    void terminate() { setFlag(PathFlags::active, false); }
    void setActive() { setFlag(PathFlags::active); }
    //void setHit(HitInfo hitInfo) { hit = hitInfo; setFlag(PathFlags::hit); }
    void setHitPacked(PackedHitInfo hitInfoPacked) { hitPacked = hitInfoPacked; setFlag(PathFlags::hit); }
    void clearHit() { setFlag(PathFlags::hit, false); }

    void clearScatterEventFlags()
    {
        const uint bits = ( ((uint)PathFlags::transmission) | ((uint)PathFlags::specular) | ((uint)PathFlags::delta) ) << kVertexIndexBitCount;
        flagsAndVertexIndex &= ~bits;
    }

    void setScatterTransmission(bool value = true) { setFlag(PathFlags::transmission, value); }            ///< Set flag indicating that scatter ray went through a transmission event.
    void setScatterSpecular(bool value = true) { setFlag(PathFlags::specular, value); }                    ///< Set flag indicating that scatter ray went through a specular event.
    void setScatterDelta(bool value = true) { setFlag(PathFlags::delta, value); }                          ///< Set flag indicating that scatter ray went through a delta event.
    void setInsideDielectricVolume(bool value = true) { setFlag(PathFlags::insideDielectricVolume, value); }
    void setDiffusePrimaryHit(bool value = true) { setFlag(PathFlags::diffusePrimaryHit, value); }
    void setSpecularPrimaryHit(bool value = true) { setFlag(PathFlags::specularPrimaryHit, value); }
    void setDeltaTransmissionPath(bool value = true) { setFlag(PathFlags::deltaTransmissionPath, value); }
    void setDeltaOnlyPath(bool value = true) { setFlag(PathFlags::deltaOnlyPath, value); }

    bool hasFlag(PathFlags flag)
    {
        const uint bit = ((uint)flag) << kVertexIndexBitCount;
        return (flagsAndVertexIndex & bit) != 0;
    }

    void setFlag(PathFlags flag, bool value = true)
    {
        const uint bit = ((uint)flag) << kVertexIndexBitCount;
        if (value) flagsAndVertexIndex |= bit;
        else flagsAndVertexIndex &= ~bit;
    }

    uint getCounter(PackedCounters type)
    {
        const uint shift = ((uint)type) << 3;
        return (packedCounters >> shift) & 0xff;
    }

    void setCounter(PackedCounters type, uint bounces)
    {
        const uint shift = ((uint)type) << 3;
        packedCounters = (packedCounters & ~((uint)0xff << shift)) | ((bounces & 0xff) << shift);
    }

    void incrementCounter(PackedCounters type)
    {
        const uint shift = ((uint)type) << 3;
        // We assume that bounce counters cannot overflow.
        packedCounters += (1u << shift);
    }

    // fixed to 1 sample and moved to PathTracerTypes, PathIDToPixel - sorry - we might bring it back here
    // uint2 getPixel() { return uint2(id, id >> 12) & 0xfff; }
    // uint getSampleIdx() { return id >> 24; }

    // Unsafe - assumes that index is small enough.
    void setVertexIndex(uint index)
    {
        // Clear old vertex index.
        flagsAndVertexIndex &= kPathFlagsBitMask;
        // Set new vertex index (unsafe).
        flagsAndVertexIndex |= index;
    }

    uint getVertexIndex() { return flagsAndVertexIndex & kVertexIndexBitMask; }

    // Unsafe - assumes that vertex index never overflows.
    void incrementVertexIndex() { flagsAndVertexIndex += 1; }
    // Unsafe - assumes that vertex index will never be decremented below zero.
    void decrementVertexIndex() { flagsAndVertexIndex -= 1; }

    Ray getScatterRay()
    {
        return Ray::make(origin, dir, 0.f, kMaxRayTravel);
    }

    uint getStablePlaneIndex()                  { return (flagsAndVertexIndex & kStablePlaneIndexBitMask) >> kStablePlaneIndexBitOffset; }
    void setStablePlaneIndex(uint index)        { flagsAndVertexIndex &= ~kStablePlaneIndexBitMask; flagsAndVertexIndex |= index << kStablePlaneIndexBitOffset; }
};                                         

#endif // __PATH_STATE_HLSLI__