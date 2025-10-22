/*
* Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#ifndef __STABLE_PLANES_HLSLI__ // using instead of "#pragma once" due to https://github.com/microsoft/DirectXShaderCompiler/issues/3943
#define __STABLE_PLANES_HLSLI__

#include "Config.h"    

#include "PathTracerShared.h"
#include "Rendering/Materials/IBSDF.hlsli"

#if !defined(__cplusplus) // shader only!
#include "Utils/ColorHelpers.hlsli"
#include "PathState.hlsli"
#endif
#include "Utils/Utils.hlsli"


// Largely based on ideas from https://www.mitsuba-renderer.org/~wenzel/papers/decomposition.pdf (Path-space Motion Estimation and Decomposition for Robust Animation Filtering),
// https://developer.nvidia.com/blog/rendering-perfect-reflections-and-refractions-in-path-traced-games/ (Rendering Perfect Reflections and Refractions in Path-Traced Games) and
// Falcor's NRD denoising (https://github.com/NVIDIAGameWorks/Falcor)

static const uint       cStablePlaneMaxVertexIndex      = 15;               // 15 is max, it's enough for denoising and will allow stableBranchID staying at 32bit-s
static const uint       cStablePlaneInvalidBranchID     = 0xFFFFFFFF;       // this means it's empty and any radiance there is invalid; within path tracer it also means one can start using it for the next exploration
static const uint       cStablePlaneEnqueuedBranchID    = 0xFFFFFFFF-1;     // this means it contains enqueued delta path exploration data; it should never be set to this value outside of path tracing passes (would indicate bug)
static const uint       cStablePlaneJustStartedID       = 0;                // this means the delta path is currently being explored; it should never be set to this value outside of path tracing passes (would indicate bug)

// Call after every scatter to update stable branch ID; deltaLobeID must be < 4, vertexIndex must be <= cStablePlaneMaxVertexIndex
uint StablePlanesAdvanceBranchID(const uint prevStableBranchID, const uint deltaLobeID);
uint StablePlanesGetParentLobeID(const uint stableBranchID);
uint StablePlanesVertexIndexFromBranchID(const uint stableBranchID);
bool StablePlaneIsOnPlane(const uint planeBranchID, const uint vertexBranchID);
bool StablePlaneIsOnStablePath(const uint planeBranchID, const uint planeVertexIndex, const uint vertexBranchID, const uint vertexIndex);
bool StablePlaneIsOnStablePath(const uint planeBranchID, const uint vertexBranchID);
float StablePlaneAdvanceLayerSceneLength(float currentHitT, float currentSegmentT, uint bouncesFromPlane, bool pathIsDeltaOnlyPath);
//float4 StablePlaneCombineWithHitTCompensation(float4 currentRadianceHitDist, float3 newRadiance, float newHitT);
float3 StablePlaneDebugVizColor(const uint planeIndex);

#if !defined(__cplusplus) // shader only!
static const float kSpecHeuristicBoost = 1.0f;
#endif

struct StablePlane
{
    uint4   PackedHitInfo;                  // Hit surface info
    float3  RayDir;                         // Last surface hit direction
    float   SceneLength;                    // total ray travel (PathState::sceneLength)
    uint3   PackedThpAndMVs;                // throughput and motion vectors packed in fp16; throughput might no longer be required since it's baked into bsdfestimate
    uint    VertexIndexAndRoughness;        // 16bits for vertex index (only 8 actually needed), 16bits for roughness
    uint3   DenoiserPackedBSDFEstimate;     // diff and spec bsdf estimates packed in fp16
    uint    PackedNormal;                   // normal packed in uint32 oct
    uint2   PackedNoisyRadianceAndSpecAvg;  // noisy radiance, and separate spec avg
    float   NoisyRadianceSpecHitDist;       // spec hit distance - can be packed to fp16 
    uint    Padding;                        // space to pack something else (needs

#if !defined(__cplusplus) // shader only!
    bool            IsEmpty()                   { return (VertexIndexAndRoughness >> 16) == 0; }

    float3          GetNormal()                 { return OctToNDirUnorm32(PackedNormal); }
    float           GetRoughness()              { return f16tof32(VertexIndexAndRoughness & 0xFFFF); }

    float3          GetNoisyRadiance()          { return Fp16ToFp32(PackedNoisyRadianceAndSpecAvg).xyz; }
    float4          GetNoisyRadianceAndSpecRA() { return Fp16ToFp32(PackedNoisyRadianceAndSpecAvg).xyzw; }

    float3          GetNoisyDiffRadiance()      { float4 l = Fp16ToFp32(PackedNoisyRadianceAndSpecAvg); float totalAvg = Average(l.rgb); return l.rgb * saturate(1.0 - (l.a*kSpecHeuristicBoost) / (totalAvg+1e-12)).xxx; }
    float3          GetNoisySpecRadiance()      { float4 l = Fp16ToFp32(PackedNoisyRadianceAndSpecAvg); float totalAvg = Average(l.rgb); return l.rgb * saturate((l.a*kSpecHeuristicBoost) / (totalAvg+1e-12)).xxx; }
#endif

#if PATH_TRACER_MODE==PATH_TRACER_MODE_BUILD_STABLE_PLANES
    // This is used only during the stable path build to start new branches - just reusing the data; payload size mismatch will trigger compile error - easy to fix!
    void            PackCustomPayload( const uint4 packed[5] );
    void            UnpackCustomPayload( inout uint4 packed[5] );
#endif
};

struct StablePlanesContext
{
#if !defined(__cplusplus) // shader only!
    RWTexture2D<float4>                     StableRadianceUAV;
    RWTexture2DArray<uint>                  StablePlanesHeaderUAV;      // [0,1,2] are StableBranchIDs, [3] is asuint(FirstHitRayLength)
    RWStructuredBuffer<StablePlane>         StablePlanesUAV;

    PathTracerConstants                     PTConstants;

    static StablePlanesContext make(RWTexture2DArray<uint> stablePlanesHeaderUAV, RWStructuredBuffer<StablePlane> stablePlanesUAV, RWTexture2D<float4> stableRadianceUAV, PathTracerConstants ptConstants)
    {
        StablePlanesContext ret;
        ret.StablePlanesHeaderUAV       = stablePlanesHeaderUAV;
        ret.StablePlanesUAV             = stablePlanesUAV;
        ret.StableRadianceUAV           = stableRadianceUAV;
        ret.PTConstants                 = ptConstants;
        return ret;
    }

    static uint ComputeDominantAddress(uint2 pixelPos, RWTexture2DArray<uint> stablePlanesHeaderUAV, RWStructuredBuffer<StablePlane> stablePlanesUAV, RWTexture2D<float4> stableRadianceUAV, PathTracerConstants ptConstants)
    {
        StablePlanesContext stablePlanes = StablePlanesContext::make(stablePlanesHeaderUAV, stablePlanesUAV, stableRadianceUAV, ptConstants);

        uint dominantStablePlaneIndex = stablePlanes.LoadDominantIndex(pixelPos);
        return stablePlanes.PixelToAddress(pixelPos, dominantStablePlaneIndex);
    }

    // TODO: currently using scanline; update to more cache friendly addressing
    uint    PixelToAddress(uint2 pixelPos, uint planeIndex)
    {
        return GenericTSPixelToAddress(pixelPos, planeIndex, PTConstants.genericTSLineStride, PTConstants.genericTSPlaneStride);
    }
    
    uint    PixelToAddress(uint2 pixelPos)                                      { return PixelToAddress(pixelPos, 0); }
    
    StablePlane LoadStablePlane(const uint2 pixelPos, const uint planeIndex)
    {
        uint address = PixelToAddress( pixelPos, planeIndex );
        return StablePlanesUAV[address];
    }

    uint    GetBranchID(const uint2 pixelPos, const uint planeIndex)
    {
        return StablePlanesHeaderUAV[uint3(pixelPos,planeIndex)];
    }

    void    SetBranchID(const uint2 pixelPos, const uint planeIndex, uint stableBranchID)
    {
        StablePlanesHeaderUAV[uint3(pixelPos, planeIndex)] = stableBranchID;
    }

    static void UnpackStablePlane(const StablePlane sp, out uint vertexIndex, out uint4 packedHitInfo, out float3 rayDir, out float sceneLength, out float3 thp, out float3 motionVectors)
    {
        vertexIndex     = sp.VertexIndexAndRoughness>>16;
        packedHitInfo   = sp.PackedHitInfo;
        sceneLength     = sp.SceneLength;
        rayDir          = sp.RayDir;
        UnpackTwoFp32ToFp16(sp.PackedThpAndMVs, thp, motionVectors);
    }

    void    LoadStablePlane(const uint2 pixelPos, const uint planeIndex, out uint vertexIndex, out uint4 packedHitInfo, out uint stableBranchID, 
                            out float3 rayDir, out float sceneLength, out float3 thp, out float3 motionVectors )
    {
        stableBranchID = GetBranchID(pixelPos, planeIndex);
        UnpackStablePlane( LoadStablePlane(pixelPos, planeIndex), vertexIndex, packedHitInfo, rayDir, sceneLength, thp, motionVectors );
    }

#if PATH_TRACER_MODE==PATH_TRACER_MODE_BUILD_STABLE_PLANES // should only be written in the first pass
    void                StoreStableRadiance(uint2 pixelPos, float3 radiance)            { StableRadianceUAV[pixelPos].xyzw = float4(clamp( radiance, 0, HLF_MAX ), 0); }
    void                AccumulateStableRadiance(uint2 pixelPos, float3 radiance)       { StableRadianceUAV[pixelPos].xyz += radiance; }
#endif
    float3              LoadStableRadiance(uint2 pixelPos)                              { return StableRadianceUAV[pixelPos].xyz; }

    // last 2 bits are for dominant SP index
    void                StoreFirstHitRayLengthAndClearDominantToZero(uint2 pixelPos, float length){ StablePlanesHeaderUAV[uint3(pixelPos, 3)] = asuint(min(kMaxRayTravel, length)) & 0xFFFFFFFC; }
    float               LoadFirstHitRayLength(uint2 pixelPos)                           { return asfloat(StablePlanesHeaderUAV[uint3(pixelPos, 3)] & 0xFFFFFFFC); }
    void                StoreDominantIndex(uint2 pixelPos, uint index)            { StablePlanesHeaderUAV[uint3(pixelPos, 3)] = (StablePlanesHeaderUAV[uint3(pixelPos, 3)] & 0xFFFFFFFC) | (0x3 & index); }
    uint                LoadDominantIndex(uint2 pixelPos)                         { return StablePlanesHeaderUAV[uint3(pixelPos, 3)] & 0x3; }


#if PATH_TRACER_MODE==PATH_TRACER_MODE_BUILD_STABLE_PLANES
    // this stores at surface hit, with path processed in PathTracer::handleHit and decision taken to use vertex as stable plane
    void                StoreStablePlane(const uint2 pixelPos, const uint planeIndex, const uint vertexIndex, const uint4 packedHitInfo, const uint stableBranchID, const float3 rayDir, const float sceneLength,
                                            const float3 thp, const float3 motionVectors, const float roughness, const float3 worldNormal, const float3 diffBSDFEstimate, const float3 specBSDFEstimate, bool dominantSP )
    {
        uint address = PixelToAddress( pixelPos, planeIndex );
        StablePlane sp;
        sp.PackedHitInfo            = packedHitInfo;
        sp.RayDir                   = rayDir; //*clamp( sceneLength, 1e-7, kMaxRayTravel );
        sp.SceneLength              = sceneLength;
        sp.VertexIndexAndRoughness  = (vertexIndex << 16) | (f32tof16(roughness));
        sp.PackedThpAndMVs          = PackTwoFp32ToFp16(thp, motionVectors);

        // add throughput and clamp to minimum/maximum reasonable
        const float kNRDMinReflectance = 0.04f; const float kNRDMaxReflectance = 6.5504e+4F; // HLF_MAX
        float3 fullDiffBSDFEstimate = clamp( diffBSDFEstimate /** thp*/, kNRDMinReflectance.xxx, kNRDMaxReflectance.xxx );
        const float3 fullSpecBSDFEstimate = clamp( specBSDFEstimate /** thp*/, kNRDMinReflectance.xxx, kNRDMaxReflectance.xxx );

        sp.DenoiserPackedBSDFEstimate = PackTwoFp32ToFp16( fullDiffBSDFEstimate, fullSpecBSDFEstimate );
        sp.PackedNormal               = NDirToOctUnorm32( worldNormal );
        sp.PackedNoisyRadianceAndSpecAvg = Fp32ToFp16(float4(0,0,0,0));
        sp.NoisyRadianceSpecHitDist   = 0;
        sp.Padding                  = 0;
        StablePlanesUAV[address] = sp;
        SetBranchID(pixelPos, planeIndex, stableBranchID);

        if (dominantSP && planeIndex != 0) // planeIndex 0 is dominant by default
            StoreDominantIndex(pixelPos, planeIndex); // we assume StoreFirstHitRayLengthAndClearDominantToZero was already called
    }

    // as we go on forking the delta paths, we need to store the payloads somewhere to be able to explore them later!
    void                StoreExplorationStart(uint2 pixelPos, uint planeIndex, const uint4 pathPayload[5])
    {
        uint address = PixelToAddress( pixelPos, planeIndex );
        StablePlane sp;
        sp.PackCustomPayload(pathPayload);
        StablePlanesUAV[address] = sp;
        SetBranchID(pixelPos, planeIndex, cStablePlaneEnqueuedBranchID);
    }
    void                ExplorationStart(uint2 pixelPos, uint planeIndex, inout uint4 pathPayload[5])
    {
        uint address = PixelToAddress( pixelPos, planeIndex );
        StablePlane sp = StablePlanesUAV[address];
        sp.UnpackCustomPayload(pathPayload);                        // transfer packed path data from sp to pathPayload
        SetBranchID(pixelPos, planeIndex, cStablePlaneJustStartedID);   // this plane is started now and consecutive calls to this function on this plane are incorrect
    }
    int                 FindNextToExplore(uint2 pixelPos, uint fromPlane)
    {
        for( int i = fromPlane; i < cStablePlaneCount; i++ )
            if( GetBranchID(pixelPos, i) == cStablePlaneEnqueuedBranchID )
                return i;
        return -1;
    }
    void                GetAvailableEmptyPlanes(const uint2 pixelPos, inout int availableCount, inout int availablePlanes[cStablePlaneCount])
    {
        // TODO optimize this; no need for this many reads, could just store availability
        availableCount = 0;
        for( int i = 1; i < min(PTConstants.activeStablePlaneCount, cStablePlaneCount); i++ )    // we know 1st isn't available so ignore it
            if( GetBranchID(pixelPos, i) == cStablePlaneInvalidBranchID )
                availablePlanes[availableCount++] = i;
    }
#endif
    // below is the stuff used during path tracing (build & fill), which is not required for denoising, RTXDI or any post-process
    void StartPixel(uint2 pixelPos)
    {
#if PATH_TRACER_MODE==PATH_TRACER_MODE_BUILD_STABLE_PLANES // if first pass, initialize data
        StoreStableRadiance(pixelPos, 0.xxx);         // reset to 0
        StablePlanesHeaderUAV[uint3(pixelPos, 0)] = cStablePlaneInvalidBranchID;
        StablePlanesHeaderUAV[uint3(pixelPos, 1)] = cStablePlaneInvalidBranchID;
        StablePlanesHeaderUAV[uint3(pixelPos, 2)] = cStablePlaneInvalidBranchID;
#endif // PATH_TRACER_MODE==PATH_TRACER_MODE_BUILD_STABLE_PLANES
    }

#if PATH_TRACER_MODE==PATH_TRACER_MODE_FILL_STABLE_PLANES
    void CommitDenoiserRadiance(inout PathState path)
    {
        const uint2 pixelPos                        = path.GetPixelPos();
        const uint planeIndex                       = path.getStablePlaneIndex();
        const float sceneLengthFromDenoisingLayer   = path.sceneLengthFromDenoisingLayer;
        const bool baseScatterDiff                  = path.hasFlag(PathFlags::stablePlaneBaseScatterDiff);
        const bool onDominantBranch                 = path.hasFlag(PathFlags::stablePlaneOnDominantBranch);

        uint address = PixelToAddress( pixelPos, planeIndex ); 

        float4 accumRadiance        = path.L;
        float  accumSpecHitT        = path.specHitT;
        const uint2 existingRadiancePacked = StablePlanesUAV[address].PackedNoisyRadianceAndSpecAvg;
        if ( existingRadiancePacked.x != 0 && existingRadiancePacked.y != 0 )
        {
            float4 existingRadiance = Fp16ToFp32(existingRadiancePacked);
            float existingSpecHitT  = StablePlanesUAV[address].NoisyRadianceSpecHitDist;
            accumSpecHitT           = AccumulateHitT( existingRadiance.a, existingSpecHitT, accumRadiance.a, accumSpecHitT );
            accumRadiance.rgba      += existingRadiance.rgba;
        }
        StablePlanesUAV[address].PackedNoisyRadianceAndSpecAvg = Fp32ToFp16(accumRadiance);
        StablePlanesUAV[address].NoisyRadianceSpecHitDist      = accumSpecHitT;
        
        path.L = float4(0,0,0,0);
        path.sceneLengthFromDenoisingLayer  = 0.0;
        path.specHitT = 0.0;
    }
#endif // #if PATH_TRACER_MODE==PATH_TRACER_MODE_FILL_STABLE_PLANES

    float3 GetAllRadiance(uint2 pixelPos, bool includeNoisy = true)
    {
        float3 pathL = LoadStableRadiance(pixelPos);
        if (includeNoisy)
        {
            for (int i = 0; i < cStablePlaneCount; i++)
            {
                if (GetBranchID(pixelPos, i) == cStablePlaneInvalidBranchID)
                    continue;
                pathL += StablePlanesUAV[PixelToAddress( pixelPos, i )].GetNoisyRadiance();
            }
        }
        return pathL;
    }

#endif // #if !defined(__cplusplus)
};

// Call after every scatter to update stable branch ID; deltaLobeID must be < 4, vertexIndex must be <= cStablePlaneMaxVertexIndex
inline uint StablePlanesAdvanceBranchID(const uint prevStableBranchID, const uint deltaLobeID)
{
    return (prevStableBranchID << 2) | deltaLobeID;
    // return prevStableBranchID | ( deltaLobeID << ( (vertexIndex-1)*2 ) );
}
inline uint StablePlanesGetParentLobeID(const uint stableBranchID)
{
    return stableBranchID & 0x3;
    // if( vertexIndex == 1 ) return 0; // parent is camera vertex, so just return 0
    // return (stableBranchID >> (vertexIndex-2)*2 ) & 0x3;
}
inline uint StablePlanesVertexIndexFromBranchID(const uint stableBranchID)
{
#if defined(__cplusplus)
    uint v = stableBranchID; unsigned r = 0; while (v >>= 1) r++; return r/2+1;
#else
    return firstbithigh(stableBranchID)/2+1;
#endif
}
inline bool StablePlaneIsOnPlane(const uint planeBranchID, const uint vertexBranchID) 
{ 
    return planeBranchID == vertexBranchID;
}
inline bool StablePlaneIsOnStablePath(const uint planeBranchID, const uint planeVertexIndex, const uint vertexBranchID, const uint vertexIndex) 
{
    if( vertexIndex > planeVertexIndex )
        return false;
    return (planeBranchID >> ((planeVertexIndex-vertexIndex)*2) ) == vertexBranchID;
}
inline bool StablePlaneIsOnStablePath(const uint planeBranchID, const uint vertexBranchID) 
{
    return StablePlaneIsOnStablePath(planeBranchID, StablePlanesVertexIndexFromBranchID(planeBranchID), vertexBranchID, StablePlanesVertexIndexFromBranchID(vertexBranchID));
}
inline float3 StablePlaneDebugVizColor(const uint planeIndex) 
{ 
    return float3( planeIndex==0 || planeIndex==3, planeIndex==1, planeIndex==2 || planeIndex==3 ); 
}

#if !defined(__cplusplus) // shader only!
inline float StablePlaneAdvanceLayerSceneLength(float currentHitT, float currentSegmentT, uint bouncesFromPlane, bool pathIsDeltaOnlyPath)
{
    if (bouncesFromPlane==1)    // first hit from stable (denoising) plane always starts recording hitT
        return currentSegmentT;
#if 1 // allow one typical glass-like interface (2 bounces - one for glass entry one for glass exit bounce) to let the hitT computation pass through; this isn't perfect, it should only be done for transmission and mild refraction
    else if( pathIsDeltaOnlyPath && bouncesFromPlane > 1 && bouncesFromPlane <= 3 )
        return currentHitT+currentSegmentT;
#endif
    else
        return currentHitT;
}
//inline float4 StablePlaneCombineWithHitTCompensation(float4 currentRadianceHitDist, float3 newRadiance, float newHitT)
//{
//    float lNew = luminance(newRadiance);
//    if (lNew < 1e-5)
//        return currentRadianceHitDist;
//    float lOld = luminance(currentRadianceHitDist.rgb);
//    float weightNew = lNew / (lOld + lNew);
//    return float4( currentRadianceHitDist.rgb + newRadiance.rgb, abs(currentRadianceHitDist.w) * (1-weightNew) + newHitT * weightNew );
//}
inline uint3 StablePlaneDebugVizFourWaySplitCoord(const int dbgPlaneIndex, const uint2 pixelPos, const uint2 screenSize)
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

#if PATH_TRACER_MODE==PATH_TRACER_MODE_BUILD_STABLE_PLANES
void StablePlane::PackCustomPayload(const uint4 packed[5])
{
    // WARNING: take care when changing these - error could be subtle and very hard to track down later
    PackedHitInfo                   = packed[0];
    RayDir                          = asfloat(packed[1].xyz);
    SceneLength                     = asfloat(packed[1].w);
    PackedThpAndMVs                 = packed[2].xyz;
    VertexIndexAndRoughness         = packed[2].w;
    DenoiserPackedBSDFEstimate      = packed[3].xyz;
    PackedNormal                    = packed[3].w;
    PackedNoisyRadianceAndSpecAvg   = packed[4].xy;
    NoisyRadianceSpecHitDist        = asfloat(packed[4].z);
    Padding                         = packed[4].w;
}
void StablePlane::UnpackCustomPayload(inout uint4 packed[5])
{
    // WARNING: take care when changing these - error could be subtle and very hard to track down later
    packed[0]                       = PackedHitInfo;
    packed[1].xyz                   = asuint(RayDir);
    packed[1].w                     = asuint(SceneLength);
    packed[2].xyz                   = PackedThpAndMVs;
    packed[2].w                     = VertexIndexAndRoughness;
    packed[3].xyz                   = DenoiserPackedBSDFEstimate;
    packed[3].w                     = PackedNormal;
    packed[4].xy                    = PackedNoisyRadianceAndSpecAvg;
    packed[4].z                     = asuint(NoisyRadianceSpecHitDist);
    packed[4].w                     = Padding;
}
#endif


#endif // __STABLE_PLANES_HLSLI__
