/*
* Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#ifndef __DENOISING_LAYERS_HLSLI__ // using instead of "#pragma once" due to https://github.com/microsoft/DirectXShaderCompiler/issues/3943
#define __DENOISING_LAYERS_HLSLI__

#include "Config.h"    

#include "PathTracerShared.h"
#include "BSDFs/IBSDF.hlsli"

#if !defined(__cplusplus) // shader only!
#include "Utils/ColorHelpers.hlsli"
#include "PathState.hlsli"
#endif
#include "Utils/Utils.hlsli"


// Largely based on ideas from https://www.mitsuba-renderer.org/~wenzel/papers/decomposition.pdf (Path-space Motion Estimation and Decomposition for Robust Animation Filtering),
// https://developer.nvidia.com/blog/rendering-perfect-reflections-and-refractions-in-path-traced-games/ (Rendering Perfect Reflections and Refractions in Path-Traced Games) and
// Falcor's NRD denoising (https://github.com/NVIDIAGameWorks/Falcor)

static const uint       cDenoisingLayerMaxVertexIndex      = 15;               // 15 is max, it's enough for denoising and will allow stableBranchID staying at 32bit-s
static const uint       cDenoisingLayerInvalidBranchID     = 0xFFFFFFFF;       // this means it's empty and any radiance there is invalid; within path tracer it also means one can start using it for the next exploration
static const uint       cDenoisingLayerEnqueuedBranchID    = 0xFFFFFFFF-1;     // this means it contains enqueued delta path exploration data; it should never be set to this value outside of path tracing passes (would indicate bug)
static const uint       cDenoisingLayerJustStartedID       = 0;                // this means the delta path is currently being explored; it should never be set to this value outside of path tracing passes (would indicate bug)

// Call after every scatter to update stable branch ID; deltaLobeID must be < 4, vertexIndex must be <= cDenoisingLayerMaxVertexIndex
uint DenoisingLayersAdvanceBranchID(const uint prevStableBranchID, const uint deltaLobeID);
uint DenoisingLayersGetParentLobeID(const uint stableBranchID);
uint DenoisingLayersVertexIndexFromBranchID(const uint stableBranchID);
bool DenoisingLayerIsOnPlane(const uint planeBranchID, const uint vertexBranchID);
bool DenoisingLayerIsOnStablePath(const uint planeBranchID, const uint planeVertexIndex, const uint vertexBranchID, const uint vertexIndex);
bool DenoisingLayerIsOnStablePath(const uint planeBranchID, const uint vertexBranchID);
float3 DenoisingLayerDebugVizColor(const uint planeIndex);

#if !defined(__cplusplus) // shader only!
static const float kSpecHeuristicBoost = 1.0f;
#endif

struct DenoisingLayer
{
    float3  RayOrigin;                      // Last surface hit position (minus offset to enable re-hitting again); if sky hit, then (+inf, +inf, +inf)
    float   LastRayTCurrent;                // space to pack something else
    float3  RayDir;                         // Last surface hit ray direction
    float   SceneLength;                    // total ray travel (PathState::sceneLength)
    uint3   PackedThpAndMVs;                // throughput and world space motion vector (ShadingData::toPrevPosW) packed in fp16; throughput might no longer be required since it's baked into bsdfestimate
    uint    VertexIndexAndRoughness;        // 16bits for vertex index (only 8 actually needed), 16bits for roughness
    uint3   DenoiserPackedBSDFEstimate;     // diff and spec bsdf estimates packed in fp16
    uint    PackedNormal;                   // normal packed in uint32 oct
    uint2   PackedNoisyRadianceAndSpecAvg;  // noisy radiance, and separate spec avg
    uint    FlagsAndVertexIndex;            // to be able to restart path correctly
    uint    PackedCounters;                 // to be able to restart path correctly

#if !defined(__cplusplus) // shader only!
    bool            IsEmpty()                   { return (VertexIndexAndRoughness >> 16) == 0; }

    float3          GetNormal()                 { return OctToNDirUnorm32(PackedNormal); }
    float           GetRoughness()              { return f16tof32(VertexIndexAndRoughness & 0xFFFF); }

    float3          GetNoisyRadiance()          { return Fp16ToFp32(PackedNoisyRadianceAndSpecAvg).xyz; }
    float4          GetNoisyRadianceAndSpecRA() { return Fp16ToFp32(PackedNoisyRadianceAndSpecAvg).xyzw; }

    float3          GetNoisyDiffRadiance()      { float4 l = Fp16ToFp32(PackedNoisyRadianceAndSpecAvg); float totalAvg = Average(l.rgb); return l.rgb * saturate(1.0 - (l.a*kSpecHeuristicBoost) / (totalAvg+1e-12)).xxx; }
    float3          GetNoisySpecRadiance()      { float4 l = Fp16ToFp32(PackedNoisyRadianceAndSpecAvg); float totalAvg = Average(l.rgb); return l.rgb * saturate((l.a*kSpecHeuristicBoost) / (totalAvg+1e-12)).xxx; }
    
    void            GetThpAndWorldMotionVector(out float3 throughput, out float3 toPrevPosW) { UnpackTwoFp32ToFp16(PackedThpAndMVs, throughput, toPrevPosW); }
    float3          GetThp()                    { float3 dummy, ret; UnpackTwoFp32ToFp16(PackedThpAndMVs, ret, dummy); return ret; }
    float3          GetWorldMotionVector()      { float3 dummy, ret; UnpackTwoFp32ToFp16(PackedThpAndMVs, dummy, ret); return ret; }
#endif

#if PATH_TRACER_MODE==PT_DENOISING_LAYERS_BUILD
    // This is used only during the stable path build to start new branches - just reusing the data; payload size mismatch will trigger compile error - easy to fix!
    void            PackCustomPayload( const uint4 packed[5] );
    void            UnpackCustomPayload( inout uint4 packed[5] );
#endif
};

struct DenoisingLayersContext
{
#if !defined(__cplusplus) // shader only!
    RWTexture2D<float4>                     StableRadianceUAV;
    RWTexture2DArray<uint>                  DenoisingLayersHeaderUAV;      // [0,1,2] are StableBranchIDs, [3] is asuint(FirstHitRayLength)
    RWStructuredBuffer<DenoisingLayer>         DenoisingLayersUAV;

    PathTracerConstants                     PTConstants;

    static DenoisingLayersContext make(RWTexture2DArray<uint> denoisingLayersHeaderUAV, RWStructuredBuffer<DenoisingLayer> denoisingLayersUAV, RWTexture2D<float4> stableRadianceUAV, PathTracerConstants ptConstants)
    {
        DenoisingLayersContext ret;
        ret.DenoisingLayersHeaderUAV       = denoisingLayersHeaderUAV;
        ret.DenoisingLayersUAV             = denoisingLayersUAV;
        ret.StableRadianceUAV           = stableRadianceUAV;
        ret.PTConstants                 = ptConstants;
        return ret;
    }

    static uint ComputeDominantAddress(uint2 pixelPos, RWTexture2DArray<uint> denoisingLayersHeaderUAV, RWStructuredBuffer<DenoisingLayer> denoisingLayersUAV, RWTexture2D<float4> stableRadianceUAV, PathTracerConstants ptConstants)
    {
        DenoisingLayersContext denoisingLayers = DenoisingLayersContext::make(denoisingLayersHeaderUAV, denoisingLayersUAV, stableRadianceUAV, ptConstants);

        uint dominantDenoisingLayerIndex = denoisingLayers.LoadDominantIndex(pixelPos);
        return denoisingLayers.PixelToAddress(pixelPos, dominantDenoisingLayerIndex);
    }

    // TODO: currently using scanline; update to more cache friendly addressing
    uint    PixelToAddress(uint2 pixelPos, uint planeIndex)
    {
        return GenericTSPixelToAddress(pixelPos, planeIndex, PTConstants.genericTSLineStride, PTConstants.genericTSPlaneStride);
    }
    
    uint    PixelToAddress(uint2 pixelPos)                                      { return PixelToAddress(pixelPos, 0); }
    
    DenoisingLayer LoadDenoisingLayer(const uint2 pixelPos, const uint planeIndex)
    {
        uint address = PixelToAddress( pixelPos, planeIndex );
        return DenoisingLayersUAV[address];
    }

    uint    GetBranchID(const uint2 pixelPos, const uint planeIndex)
    {
        return DenoisingLayersHeaderUAV[uint3(pixelPos,planeIndex)];
    }

    void    SetBranchID(const uint2 pixelPos, const uint planeIndex, uint stableBranchID)
    {
        DenoisingLayersHeaderUAV[uint3(pixelPos, planeIndex)] = stableBranchID;
    }

#if PATH_TRACER_MODE==PT_DENOISING_LAYERS_BUILD // should only be written in the first pass
    void                StoreStableRadiance(uint2 pixelPos, float3 radiance)            { StableRadianceUAV[pixelPos].xyzw = float4(clamp( radiance, 0, HLF_MAX ), 0); }
    void                AccumulateStableRadiance(uint2 pixelPos, float3 radiance)       { StableRadianceUAV[pixelPos].xyz += radiance; }
#endif
    float3              LoadStableRadiance(uint2 pixelPos)                              { return StableRadianceUAV[pixelPos].xyz; }

    // last 2 bits are for dominant SP index
    void                StoreFirstHitRayLengthAndClearDominantToZero(uint2 pixelPos, float length){ DenoisingLayersHeaderUAV[uint3(pixelPos, 3)] = asuint(min(kMaxRayTravel, length)) & 0xFFFFFFFC; }
    float               LoadFirstHitRayLength(uint2 pixelPos)                           { return asfloat(DenoisingLayersHeaderUAV[uint3(pixelPos, 3)] & 0xFFFFFFFC); }
    void                StoreDominantIndex(uint2 pixelPos, uint index)            { DenoisingLayersHeaderUAV[uint3(pixelPos, 3)] = (DenoisingLayersHeaderUAV[uint3(pixelPos, 3)] & 0xFFFFFFFC) | (0x3 & index); }
    uint                LoadDominantIndex(uint2 pixelPos)                         { return DenoisingLayersHeaderUAV[uint3(pixelPos, 3)] & 0x3; }


#if PATH_TRACER_MODE==PT_DENOISING_LAYERS_BUILD
    // this stores at surface hit, and decision taken to use vertex as denoising layer
    // NOTE 1: the PathState stored here is >before< any surface processing has happened (and actually requires short re-raytrace due to inability to store at exactly the surface)
    // NOTE 2: and the world location is "just before the hit"
    void                StoreDenoisingLayer(const uint2 pixelPos, const uint planeIndex, const uint vertexIndex, const float3 rayOrigin, const float3 rayDir, const uint stableBranchID, const float sceneLength, const float rayTCurrent,
                                            const float3 thp, const float3 toPrevPosW, const float roughness, const float3 worldNormal, const float3 diffBSDFEstimate, const float3 specBSDFEstimate, bool dominantSP,
                                            uint flagsAndVertexIndex, uint packedCounters)
    {
        uint address = PixelToAddress( pixelPos, planeIndex );
        DenoisingLayer sp;
        sp.RayOrigin                = rayOrigin;
        sp.RayDir                   = rayDir; //*clamp( sceneLength, 1e-7, kMaxRayTravel );
        sp.SceneLength              = sceneLength;
        sp.VertexIndexAndRoughness  = (vertexIndex << 16) | (f32tof16(roughness));
        sp.PackedThpAndMVs          = PackTwoFp32ToFp16(thp, toPrevPosW);

        // add throughput and clamp to maximum reasonable; note there is deliberately NO minimum reflectance floor here -
        // the estimates are stored as-is so each denoiser can pick its own floor at guide preparation time (see
        // PostProcess.hlsl: NRD needs a relatively high one to keep demodulation stable, DLSS-RR wants a much lower one)
        const float kMaxReflectance = 6.5504e+4F; // HLF_MAX
        const float3 fullDiffBSDFEstimate = clamp( diffBSDFEstimate /** thp*/, 0.0.xxx, kMaxReflectance.xxx );
        const float3 fullSpecBSDFEstimate = clamp( specBSDFEstimate /** thp*/, 0.0.xxx, kMaxReflectance.xxx );

        sp.DenoiserPackedBSDFEstimate   = PackTwoFp32ToFp16( fullDiffBSDFEstimate, fullSpecBSDFEstimate );
        sp.PackedNormal                 = NDirToOctUnorm32( worldNormal );
        sp.PackedNoisyRadianceAndSpecAvg = Fp32ToFp16(float4(0,0,0,0));
        sp.LastRayTCurrent              = rayTCurrent;
        sp.FlagsAndVertexIndex          = flagsAndVertexIndex;
        sp.PackedCounters               = packedCounters;
        DenoisingLayersUAV[address]     = sp;
        SetBranchID(pixelPos, planeIndex, stableBranchID);

        if (dominantSP && planeIndex != 0) // planeIndex 0 is dominant by default
            StoreDominantIndex(pixelPos, planeIndex); // we assume StoreFirstHitRayLengthAndClearDominantToZero was already called
    }

    // as we go on forking the delta paths, we need to store the payloads somewhere to be able to explore them later!
    void                StoreExplorationStart(uint2 pixelPos, uint planeIndex, const uint4 pathPayload[5])
    {
        uint address = PixelToAddress( pixelPos, planeIndex );
        DenoisingLayer sp;
        sp.PackCustomPayload(pathPayload);
        DenoisingLayersUAV[address] = sp;
        SetBranchID(pixelPos, planeIndex, cDenoisingLayerEnqueuedBranchID);
    }
    void                ExplorationStart(uint2 pixelPos, uint planeIndex, inout uint4 pathPayload[5])
    {
        uint address = PixelToAddress( pixelPos, planeIndex );
        DenoisingLayer sp = DenoisingLayersUAV[address];
        sp.UnpackCustomPayload(pathPayload);                        // transfer packed path data from sp to pathPayload
        SetBranchID(pixelPos, planeIndex, cDenoisingLayerJustStartedID);   // this plane is started now and consecutive calls to this function on this plane are incorrect
    }
    int                 FindNextToExplore(uint2 pixelPos, uint fromPlane)
    {
        for( int i = fromPlane; i < cDenoisingLayerCount; i++ )
            if( GetBranchID(pixelPos, i) == cDenoisingLayerEnqueuedBranchID )
                return i;
        return -1;
    }
    void                GetAvailableEmptyPlanes(const uint2 pixelPos, inout int availableCount, inout int availablePlanes[cDenoisingLayerCount])
    {
        // TODO optimize this; no need for this many reads, could just store availability
        availableCount = 0;
        for( int i = 1; i < min(PTConstants.GetActiveDenoisingLayerCount(), cDenoisingLayerCount); i++ )    // we know 1st isn't available so ignore it
            if( GetBranchID(pixelPos, i) == cDenoisingLayerInvalidBranchID )
                availablePlanes[availableCount++] = i;
    }
#endif
    // below is the stuff used during path tracing (build & fill), which is not required for denoising, RTXDI or any post-process
    void StartPixel(uint2 pixelPos)
    {
#if PATH_TRACER_MODE==PT_DENOISING_LAYERS_BUILD // if first pass, initialize data
        StoreStableRadiance(pixelPos, 0.xxx);         // reset to 0
        DenoisingLayersHeaderUAV[uint3(pixelPos, 0)] = cDenoisingLayerInvalidBranchID;
        DenoisingLayersHeaderUAV[uint3(pixelPos, 1)] = cDenoisingLayerInvalidBranchID;
        DenoisingLayersHeaderUAV[uint3(pixelPos, 2)] = cDenoisingLayerInvalidBranchID;
#endif // PATH_TRACER_MODE==PT_DENOISING_LAYERS_BUILD
    }

#if PATH_TRACER_MODE==PT_DENOISING_LAYERS_FILL
    void CommitDenoiserRadiance(inout PathState path)
    {
        const uint2 pixelPos                        = path.GetPixelPos();
        const uint planeIndex                       = path.getDenoisingLayerIndex();
        const bool baseScatterDiff                  = path.hasFlag(PathFlags::denoisingLayerBaseScatterDiff);
        const bool onDominantBranch                 = path.hasFlag(PathFlags::denoisingLayerOnDominantBranch);

        uint address = PixelToAddress( pixelPos, planeIndex ); 

        float4 accumRadiance        = path.GetL();
        const uint2 existingRadiancePacked = DenoisingLayersUAV[address].PackedNoisyRadianceAndSpecAvg;
        if ( existingRadiancePacked.x != 0 && existingRadiancePacked.y != 0 )
        {
            float4 existingRadiance = Fp16ToFp32(existingRadiancePacked);
            accumRadiance.rgba      += existingRadiance.rgba;
        }
        DenoisingLayersUAV[address].PackedNoisyRadianceAndSpecAvg = Fp32ToFp16(accumRadiance);
        
        path.SetL( float4(0,0,0,0) );
    }
#endif // #if PATH_TRACER_MODE==PT_DENOISING_LAYERS_FILL

    float3 GetAllRadiance(uint2 pixelPos, bool includeNoisy = true)
    {
        float3 pathL = LoadStableRadiance(pixelPos);
        if (includeNoisy)
        {
            for (int i = 0; i < cDenoisingLayerCount; i++)
            {
                if (GetBranchID(pixelPos, i) == cDenoisingLayerInvalidBranchID)
                    continue;
                pathL += DenoisingLayersUAV[PixelToAddress( pixelPos, i )].GetNoisyRadiance();
            }
        }
        return pathL;
    }

#endif // #if !defined(__cplusplus)
};

// Call after every scatter to update stable branch ID; deltaLobeID must be < 4, vertexIndex must be <= cDenoisingLayerMaxVertexIndex
inline uint DenoisingLayersAdvanceBranchID(const uint prevStableBranchID, const uint deltaLobeID)
{
    return (prevStableBranchID << 2) | deltaLobeID;
    // return prevStableBranchID | ( deltaLobeID << ( (vertexIndex-1)*2 ) );
}
inline uint DenoisingLayersGetParentLobeID(const uint stableBranchID)
{
    return stableBranchID & 0x3;
    // if( vertexIndex == 1 ) return 0; // parent is camera vertex, so just return 0
    // return (stableBranchID >> (vertexIndex-2)*2 ) & 0x3;
}
inline uint DenoisingLayersVertexIndexFromBranchID(const uint stableBranchID)
{
#if defined(__cplusplus)
    uint v = stableBranchID; unsigned r = 0; while (v >>= 1) r++; return r/2+1;
#else
    return firstbithigh(stableBranchID)/2+1;
#endif
}
inline bool DenoisingLayerIsOnPlane(const uint planeBranchID, const uint vertexBranchID) 
{ 
    return planeBranchID == vertexBranchID;
}
inline bool DenoisingLayerIsOnStablePath(const uint planeBranchID, const uint planeVertexIndex, const uint vertexBranchID, const uint vertexIndex) 
{
    if( vertexIndex > planeVertexIndex )
        return false;
    return (planeBranchID >> ((planeVertexIndex-vertexIndex)*2) ) == vertexBranchID;
}
inline bool DenoisingLayerIsOnStablePath(const uint planeBranchID, const uint vertexBranchID) 
{
    return DenoisingLayerIsOnStablePath(planeBranchID, DenoisingLayersVertexIndexFromBranchID(planeBranchID), vertexBranchID, DenoisingLayersVertexIndexFromBranchID(vertexBranchID));
}
inline float3 DenoisingLayerDebugVizColor(const uint planeIndex) 
{ 
    return float3( planeIndex==0 || planeIndex==3, planeIndex==1, planeIndex==2 || planeIndex==3 ); 
}

#if !defined(__cplusplus) // shader only!

inline uint3 DenoisingLayerDebugVizFourWaySplitCoord(const int dbgPlaneIndex, const uint2 pixelPos, const uint2 screenSize)
{
    if( dbgPlaneIndex >= 0 )
        return uint3( pixelPos.xy, dbgPlaneIndex );
    else
    {
        const uint2 halfSize = screenSize / 2;
        const uint2 quadrant = (pixelPos >= halfSize);
        uint3 ret;
        ret.xy = (pixelPos - quadrant * halfSize) * 2.0;
        ret.z = quadrant.x + quadrant.y * 2;
        return ret;
    }
}
#endif

#if PATH_TRACER_MODE==PT_DENOISING_LAYERS_BUILD
void DenoisingLayer::PackCustomPayload(const uint4 packed[5])
{
    // WARNING: take care when changing these - error could be subtle and very hard to track down later
    RayOrigin                       = asfloat(packed[0].xyz);
    LastRayTCurrent                 = asfloat(packed[0].w);
    RayDir                          = asfloat(packed[1].xyz);
    SceneLength                     = asfloat(packed[1].w);
    PackedThpAndMVs                 = packed[2].xyz;
    VertexIndexAndRoughness         = packed[2].w;
    DenoiserPackedBSDFEstimate      = packed[3].xyz;
    PackedNormal                    = packed[3].w;
    PackedNoisyRadianceAndSpecAvg   = packed[4].xy;
    FlagsAndVertexIndex             = packed[4].z;
    PackedCounters                  = packed[4].w;
}
void DenoisingLayer::UnpackCustomPayload(inout uint4 packed[5])
{
    // WARNING: take care when changing these - error could be subtle and very hard to track down later
    packed[0].xyz                   = asuint(RayOrigin);
    packed[0].w                     = asuint(LastRayTCurrent);
    packed[1].xyz                   = asuint(RayDir);
    packed[1].w                     = asuint(SceneLength);
    packed[2].xyz                   = PackedThpAndMVs;
    packed[2].w                     = VertexIndexAndRoughness;
    packed[3].xyz                   = DenoiserPackedBSDFEstimate;
    packed[3].w                     = PackedNormal;
    packed[4].xy                    = PackedNoisyRadianceAndSpecAvg;
    packed[4].z                     = FlagsAndVertexIndex;
    packed[4].w                     = PackedCounters;
}
#endif


#endif // __DENOISING_LAYERS_HLSLI__
