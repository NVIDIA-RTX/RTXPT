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
#include "Utils/Math/Ray.hlsli"
#include "Utils/SampleGenerators.hlsli"
#include "Scene/HitInfo.hlsli"
#include "Rendering/Materials/InteriorList.hlsli"
#include "Rendering/Materials/TexLODHelpers.hlsli"
#include "Lighting/LightingTypes.h"
#include "PathTracerHelpers.hlsli"

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
    terminateAtNextBounce           = (1<<6),   ///< This path is flagged for termination next bounce; in the next bounce it should not collect NEE - it's dead 
    
    // see https://github.com/NVIDIA-RTX/RTXDI/blob/main/Doc/RestirGI.md
    restirGIStarted                 = (1<<7),   ///< This path has started collecting data for ReSTIR GI - all radiance from now on is diverted into ReSTIR GI buffers
    restirGICollectSecondarySurface = (1<<8),   ///< This path has started collecting data for ReSTIR GI - next hit surface (or sky) info needs to be saved into ReSTIR GI buffers !!and the flag will be removed then!!

    //specularPrimaryHit              = (1<<9),   ///< Scatter ray went through a specular event on primary hit.
    //<removed, empty space>          = (1<<10),  ///<
    deltaTransmissionPath           = (1<<11),  ///< Path started with and followed delta transmission events (whenever possible - TIR could be an exception) until it hit the first non-delta event.
    deltaOnlyPath                   = (1<<12),  ///< There was no non-delta events along the path so far.

    deltaTreeExplorer               = (1<<13),  ///< Debug exploreDeltaTree enabled and this path selected for debugging
    stablePlaneIndexBit0            = (1<<14),  ///< StablePlaneIndex, bit 0 -- just reserving space for kStablePlaneIndexBitOffset & kStablePlaneIndexBitMask which must be 14
    stablePlaneIndexBit1            = (1<<15),  ///< StablePlaneIndex, bit 1 -- just reserving space for kStablePlaneIndexBitOffset & kStablePlaneIndexBitMask which must be 14
    stablePlaneOnPlane              = (1<<16),  ///< Current vertex is on a stable plane; this is where we update stablePlaneBaseScatterDiff
    stablePlaneOnBranch             = (1<<17),  ///< Current vertex is on a stable plane or stable branch; all emission is stable and was already collected
    stablePlaneBaseScatterDiff      = (1<<18),  ///< When stepping off the last stable plane & branch, we had a diffuse scatter event (this determines if the radiance is diffuse or specular for denoising purposes)
    //stablePlaneOnDeltaBranch        = (1<<19),  ///< The first scatter from a stable plane was a delta event
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
    uint        id;                     ///< See PathIDToPixel/PathIDFromPixel for encoding
    uint        flagsAndVertexIndex;    ///< Higher kPathFlagsBitCount bits: Flags indicating the current status. This can be multiple PathFlags flags OR'ed together.
                                        ///< Lower kVertexIndexBitCount bits: Current vertex index (0 = camera, 1 = primary hit, 2 = secondary hit, etc.).

    float       sceneLength;            ///< [DO NOT COMPRESS TO 16bit float!] Path length in scene units (was 0.f at primary hit originally, in this implementation it includes camera to primary hit).
#if PATH_TRACER_MODE==PATH_TRACER_MODE_FILL_STABLE_PLANES
    float       sceneLengthFromDenoisingLayer; ///< [can be compressed to 16bit in most cases] Path length in scene units between denoising layer vertex and the next vertex; useful for estimating denoising motion vectors
    float       specHitT;               ///< tracks rough hitT (weighted average of sceneLengthFromDenoisingLayer, with L.a used as a weight) 
#endif

    lpfloat     fireflyFilterK;         ///< (0, 1] multiplier for the global firefly filter threshold if used; CAN be compressed to 16bit float!
    lpuint      packedMISInfo;          ///< See NEEBSDFMISInfo
    lpfloat     bsdfScatterPdf;         ///< 0 if delta lobe (zero roughness specular) bounce
    
    uint        packedCounters;         ///< Packed counters for different types of bounces and etc., see PackedCounters.

    uint        stableBranchID;         ///< Path 'stable delta tree' branch ID for finding matching StablePlane; Gets updated on scatter while path isDeltaOnlyPath;

    // Scatter ray
    float3      origin;                 ///< Origin of the scatter ray.
    float3      dir;                    ///< Scatter ray normalized direction.
    
    PackedHitInfo hitPacked;            ///< Hit information for the scatter ray. This is populated at committed triangle hits. 4 uints (16 bytes)

    float3      thp;                    ///< Path throughput.
    lpfloat     thpRuRuCorrection;      ///< Since we use Russian Roulette to decide early termination for next frame, the correct place to apply RR thp boost that preserves unbiasedness is only AFTER emissive/sky is collected.

    InteriorList interiorList;          ///< Interior list. Keeping track of a stack of materials with medium properties. Size depends on INTERIOR_LIST_SLOT_COUNT. 2 slots (8 bytes) by default.
    RayCone     rayCone;                ///< 4 or 8 bytes depending on USE_RAYCONES_WITH_FP16_IN_RAYPAYLOAD (on, so 4 bytes by default). 

#if PATH_TRACER_MODE==PATH_TRACER_MODE_BUILD_STABLE_PLANES
    lpfloat3x3  imageXform;             ///< Accumulated rotational image transform along the path. This can be float16_t.
#else
    float4      L;                      ///< .rgb - accumulated path contribution; .a - specularness (weighted average)
#endif

    // Accessors

#if PATH_TRACER_MODE==PATH_TRACER_MODE_FILL_STABLE_PLANES
    float GetSceneLengthFromDenoisingLayer()    { return sceneLengthFromDenoisingLayer; }
    float GetSpecHitT()                         { return specHitT; }
#else
    float GetSceneLengthFromDenoisingLayer()    { return 0.0f; }
    float GetSpecHitT()                         { return 0.0f; }
#endif

    bool isTerminated() { return !isActive(); }
    bool isActive() { return hasFlag(PathFlags::active); }
    bool isHit() { return hasFlag(PathFlags::hit); }
    bool wasScatterTransmission() { return hasFlag(PathFlags::transmission); }                      ///< Get flag indicating that last scatter ray went through a transmission event.
    bool wasScatterSpecular() { return hasFlag(PathFlags::specular); }                              ///< Get flag indicating that last scatter ray went through a specular event.
    bool wasScatterDelta() { return hasFlag(PathFlags::delta); }                                    ///< Get flag indicating that last scatter ray went through a delta event.
    bool isInsideDielectricVolume() { return hasFlag(PathFlags::insideDielectricVolume); }

    // bool isDiffusePrimaryHit() { return hasFlag(PathFlags::diffusePrimaryHit); }
    // bool isSpecularPrimaryHit() { return hasFlag(PathFlags::specularPrimaryHit); }
    bool isDeltaTransmissionPath() { return hasFlag(PathFlags::deltaTransmissionPath); }
    bool isDeltaOnlyPath() { return hasFlag(PathFlags::deltaOnlyPath); }

    bool isTerminatingAtNextBounce() { return hasFlag(PathFlags::terminateAtNextBounce); }

    void terminate() { setFlag(PathFlags::active, false); }
    void setActive() { setFlag(PathFlags::active); }
    //void setHit(HitInfo hitInfo) { hit = hitInfo; setFlag(PathFlags::hit); }
    void setHitPacked(PackedHitInfo hitInfoPacked) { hitPacked = hitInfoPacked; setFlag(PathFlags::hit); }
    void clearHit() { setFlag(PathFlags::hit, false); }

    void setTerminateAtNextBounce()   { setFlag(PathFlags::terminateAtNextBounce); }

    void clearScatterEventFlags()
    {
        const uint bits = ( ((uint)PathFlags::transmission) | ((uint)PathFlags::specular) | ((uint)PathFlags::delta) ) << kVertexIndexBitCount;
        flagsAndVertexIndex &= ~bits;
    }

    void setScatterTransmission(bool value = true) { setFlag(PathFlags::transmission, value); }            ///< Set flag indicating that scatter ray went through a transmission event.
    void setScatterSpecular(bool value = true) { setFlag(PathFlags::specular, value); }                    ///< Set flag indicating that scatter ray went through a specular event.
    void setScatterDelta(bool value = true) { setFlag(PathFlags::delta, value); }                          ///< Set flag indicating that scatter ray went through a delta event.
    void setInsideDielectricVolume(bool value = true) { setFlag(PathFlags::insideDielectricVolume, value); }
    // void setDiffusePrimaryHit(bool value = true) { setFlag(PathFlags::diffusePrimaryHit, value); }
    // void setSpecularPrimaryHit(bool value = true) { setFlag(PathFlags::specularPrimaryHit, value); }
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

    uint2 GetPixelPos() { return PathIDToPixel(id); }

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