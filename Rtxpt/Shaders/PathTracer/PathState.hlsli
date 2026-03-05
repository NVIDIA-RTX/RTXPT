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
#include "Utils/Utils.hlsli"
#include "Utils/SampleGenerators.hlsli"
#include "Scene/HitInfo.hlsli"
#include "Rendering/Materials/InteriorList.hlsli"
#include "Rendering/Materials/TexLODHelpers.hlsli"
#include "Lighting/LightingTypes.hlsli"
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

    enableThreadReorder             = (1<<9),   ///< Path has become divergent, it's probably good to invoke Shader Execution Reordering for the next bounce
    // exportSpecHitTBlocked           = (1<<10),  ///< This is how PTMaterialFlags_PSDAutoBlockMVsAtSurface is implemented <- actually unnecessary, we can rely on GetMotionVectorSceneLength() != 0
    deltaTransmissionPath           = (1<<11),  ///< Path started with and followed delta transmission events (whenever possible - TIR could be an exception) until it hit the first non-delta event.
    deltaOnlyPath                   = (1<<12),  ///< There was no non-delta events along the path so far.

    deltaTreeExplorer               = (1<<13),  ///< Debug exploreDeltaTree enabled and this path selected for debugging
    stablePlaneIndexBit0            = (1<<14),  ///< StablePlaneIndex, bit 0 -- just reserving space for kStablePlaneIndexBitOffset & kStablePlaneIndexBitMask which must be 14
    stablePlaneIndexBit1            = (1<<15),  ///< StablePlaneIndex, bit 1 -- just reserving space for kStablePlaneIndexBitOffset & kStablePlaneIndexBitMask which must be 14
    stablePlaneOnPlane              = (1<<16),  ///< Current vertex is on a stable plane; this is where we update stablePlaneBaseScatterDiff
    stablePlaneOnBranch             = (1<<17),  ///< Current vertex is on a stable plane or stable branch; all emission is stable and was already collected
    stablePlaneBaseScatterDiff      = (1<<18),  ///< When stepping off the last stable plane & branch, we had a diffuse scatter event (this determines if the radiance is diffuse or specular for denoising purposes)
    exportSpecHitTQueued            = (1<<19),  ///< Export specular hitT distance on first next non-specular scatter (or sky) and clear the flag.
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

/** Live state for the path tracer.
*/
struct /*PAYLOAD_QUALIFIER*/ PathState
{
    // 1st 16 bytes
    // float3      origin  PAYLOAD_FIELD_RW_ALL;               ///< Origin of the scatter ray.
    // uint        id      PAYLOAD_FIELD_RW_ALL;               ///< See PathIDToPixel/PathIDFromPixel for encoding
    uint4       PackOriginId /*PAYLOAD_FIELD_RW_ALL*/;

    // 2nd 16 bytes
    // float3      dir     PAYLOAD_FIELD_RW_ALL;               ///< Scatter ray normalized direction.
    //float       sceneLength     PAYLOAD_FIELD_RW_ALL;       ///< [DO NOT COMPRESS TO 16bit float!] Path length in scene units (was 0.f at primary hit originally, in this implementation it includes camera to primary hit).
    uint4       PackDirSceneLength /*PAYLOAD_FIELD_RW_ALL*/;

    // 3rd 16 bytes
    //float3      thp;                                      ///< Path throughput.
    uint2       pack23  /*PAYLOAD_FIELD_RW_ALL*/;               ///< packed Thp
#if PATH_TRACER_MODE==PATH_TRACER_MODE_BUILD_STABLE_PLANES
    uint2       imageXformPacked PAYLOAD_FIELD_RW_ALL;      ///< Accumulated rotational image transform along the path. Use PackOrthoMatrix / UnpackOrthoMatrix to extract
#else
    //float4      L;                    ///< .rgb - accumulated path contribution; .a - specularness (weighted average)
    uint2       pack45  /*PAYLOAD_FIELD_RW_ALL*/;               ///< packed L
#endif

    // 4th 16 bytes
#if RTXPT_NESTED_DIELECTRICS_QUALITY > 0 || defined(__INTELLISENSE__)
    InteriorList interiorList   PAYLOAD_FIELD_RW_ALL;       ///< Interior list. Keeping track of a stack of materials with medium properties. Size depends on INTERIOR_LIST_SLOT_COUNT. 2 slots (8 bytes) by default.
#endif
    uint        packedCounters  /*PAYLOAD_FIELD_RW_ALL*/;       ///< Packed counters for different types of bounces and etc., see PackedCounters.
    uint        stableBranchID  /*PAYLOAD_FIELD_RW_ALL*/;       ///< Path 'stable delta tree' branch ID for finding matching StablePlane; Gets updated on scatter while path isDeltaOnlyPath;

    // 5th 16 bytes
    RayCone     rayCone         /*PAYLOAD_FIELD_RW_ALL*/;                ///< 4 or 8 bytes depending on USE_RAYCONES_WITH_FP16_IN_RAYPAYLOAD (on, so 4 bytes by default). 
    // lpfloat     fireflyFilterK;                          ///< (0, 1] multiplier for the global firefly filter threshold if used; CAN be compressed to 16bit float!
    // lpfloat     bsdfScatterPdf;                          ///< 0 if delta lobe (zero roughness specular) bounce
    uint        pack0           /*PAYLOAD_FIELD_RW_ALL*/;       ///< packed fireflyFilterK && bsdfScatterPdf
    // lpuint      packedMISInfo;                           ///< See NEEBSDFMISInfo
    // lpfloat     thpRuRuCorrection;                       ///< Since we use Russian Roulette to decide early termination for next frame, the correct place to apply RR thp boost that preserves unbiasedness is only AFTER emissive/sky is collected.
    uint        pack1           /*PAYLOAD_FIELD_RW_ALL*/;       ///< packed packedMISInfo and thpRuRuCorrection
    uint        flagsAndVertexIndex /*PAYLOAD_FIELD_RW_ALL*/;   ///< Higher kPathFlagsBitCount bits: Flags indicating the current status. This can be multiple PathFlags flags OR'ed together.
                                                            ///< Lower kVertexIndexBitCount bits: Current vertex index (0 = camera, 1 = primary hit, 2 = secondary hit, etc.).

    // Accessors
    #define PS_CONST

    float3      GetOrigin() PS_CONST            { return asfloat(PackOriginId.xyz); }
    void        SetOrigin(float3 origin)        { PackOriginId.xyz = asuint(origin); }

    uint        GetId() PS_CONST                   { return PackOriginId.w; }
    void        SetId(uint id)                  { PackOriginId.w = id; }

    float3      GetDir() PS_CONST               { return asfloat(PackDirSceneLength.xyz); }
    void        SetDir(float3 origin)           { PackDirSceneLength.xyz = asuint(origin); }

    float       GetSceneLength() PS_CONST       { return asfloat(PackDirSceneLength.w); }
    void        SetSceneLength(float sceneLength) { PackDirSceneLength.w = asuint(sceneLength); }

#if PATH_TRACER_MODE==PATH_TRACER_MODE_BUILD_STABLE_PLANES
    float3x3 GetImageXform() PS_CONST           { return UnpackOrthoMatrix(imageXformPacked); }
    void     SetImageXform(float3x3 p)          { imageXformPacked = PackOrthoMatrix(p); }
#endif

#if PATH_TRACER_MODE!=PATH_TRACER_MODE_BUILD_STABLE_PLANES
    void    SetFireflyFilterK_BsdfScatterPdf(float fireflyFilterK, float bsdfScatterPdf)            { pack0 = ((f32tof16(clamp(fireflyFilterK, 0, HLF_MAX))) << 16) | (f32tof16(clamp(bsdfScatterPdf, 0, HLF_MAX))); }
    lpfloat GetFireflyFilterK() PS_CONST        { return lpfloat(f16tof32(pack0 >> 16)); }
    lpfloat GetBsdfScatterPdf() PS_CONST        { return lpfloat(f16tof32(pack0 & 0xFFFF)); }

    void    SetPackedMISInfo_ThpRuRuCorrection( uint packedMISInfo, float thpRuRuCorrection )       { pack1 = (packedMISInfo<<16) | (f32tof16(clamp(thpRuRuCorrection, 0, HLF_MAX))); }
    lpuint  GetPackedMISInfo() PS_CONST         { return lpuint(pack1 >> 16); }
    lpfloat GetThpRuRuCorrection() PS_CONST     { return lpfloat(f16tof32(pack1 & 0xFFFF)); }
#else
    // see PTMaterialFlags_PSDAutoBlockMVsAtSurface - if enabled at surface, current SceneLength will be rememberded and used for all subsequent motion vector computation, as well as disable future Xform updates
    void    SetMotionVectorSceneLength(float l) { pack0 = asuint(l); }
    float   GetMotionVectorSceneLength()        { return asfloat(pack0); }
    //void    SetMotionVectorBounceNormal(float3 n) { pack1 = NDirToOctUnorm30(n); }        // too complex
    //float3  GetMotionVectorBounceNormal()         { return OctToNDirUnorm30(pack1); }     // too complex
    // there's no NEE while building stable planes as we're on delta branches only!
    lpuint  GetPackedMISInfo() PS_CONST         { return 0; }    // by convention, NEEBSDFMISInfo::empty().Pack16bit() is 0
    lpfloat GetBsdfScatterPdf() PS_CONST        { return 0.0; }
    // there's no RR while building stable planes!
    lpfloat GetThpRuRuCorrection() PS_CONST     { return 1.0f; }
#endif

    void    SetThp(float3 thp)                  { thp = clamp( thp, 0.xxx, HLF_MAX.xxx ); pack23 = Fp32ToFp16NoClamp(float4(thp,0)); }
    float3  GetThp() PS_CONST                   { return Fp16ToFp32( pack23 ).xyz; }

#if PATH_TRACER_MODE!=PATH_TRACER_MODE_BUILD_STABLE_PLANES
    void    SetL(float4 l)                      { l = clamp( l, 0.xxxx, HLF_MAX.xxxx ); pack45 = Fp32ToFp16NoClamp(l); }
    float4  GetL() PS_CONST                     { return Fp16ToFp32( pack45 ); }
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
    //void setHitPacked(PackedHitInfo hitInfoPacked) { hitPacked = hitInfoPacked; setFlag(PathFlags::hit); }
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

    uint2 GetPixelPos() { return PathIDToPixel( GetId() ); }

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
        return Ray::make(GetOrigin(), GetDir(), 0.f, kMaxRayTravel);
    }

    uint getStablePlaneIndex()                  { return (flagsAndVertexIndex & kStablePlaneIndexBitMask) >> kStablePlaneIndexBitOffset; }
    void setStablePlaneIndex(uint index)        { flagsAndVertexIndex &= ~kStablePlaneIndexBitMask; flagsAndVertexIndex |= index << kStablePlaneIndexBitOffset; }
};                                         

#endif // __PATH_STATE_HLSLI__