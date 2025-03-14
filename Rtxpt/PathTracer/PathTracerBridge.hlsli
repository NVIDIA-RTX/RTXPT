/*
* Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#ifndef __PATH_TRACER_BRIDGE_HLSLI__ // using instead of "#pragma once" due to https://github.com/microsoft/DirectXShaderCompiler/issues/3943
#define __PATH_TRACER_BRIDGE_HLSLI__

#include "Config.h"
#include "../ShaderDebug.hlsli"
#include "PathTracerTypes.hlsli"
#include "Rendering/Volumes/HomogeneousVolumeSampler.hlsli"
#include "Lighting/EnvMap.hlsli"
#include "Lighting/LightSampler.hlsli"

namespace Bridge
{
    static uint getSampleIndex();
    
    // When using multiple samples within pixel in realtime mode (which share identical camera ray), only noisy part of radiance (i.e. not direct sky) needs to be attenuated!
    static float getNoisyRadianceAttenuation();

    static uint getMaxBounceLimit();
    
    static uint getMaxDiffuseBounceLimit();

    // Gets primary camera ray for given pixel position; Note: all realtime mode subSamples currently share same camera ray at subSampleIndex == 0 (otherwise denoising guidance buffers would be noisy)
    static Ray computeCameraRay(const uint2 pixelPos, const uint subSampleIndex);

    /** Helper to create a texture sampler instance.
    The method for computing texture level-of-detail depends on the configuration.
    \param[in] path Path state.
    \param[in] isPrimaryTriangleHit True if primary hit on a triangle.
    \return Texture sampler instance.
    */
    static ActiveTextureSampler createTextureSampler(
        const RayCone rayCone,
        const float3 rayDir,
        float coneTexLODValue,
        float3 normalW,
        bool isPrimaryHit,
        bool isTriangleHit,
        float texLODBias
#if RTXPT_STOCHASTIC_TEXTURE_FILTERING_ENABLE
        ,STF_SamplerState stfSamplerState
#endif
);

    static void loadSurfacePosNormOnly(out float3 posW, out float3 faceN, const TriangleHit triangleHit, DebugContext debug);

    static PathTracer::SurfaceData loadSurface(
        const uniform PathTracer::OptimizationHints optimizationHints, 
        const TriangleHit triangleHit, 
        const float3 rayDir, 
        const RayCone rayCone, 
        const int pathVertexIndex, 
        const uint2 pixelPosition,
        DebugContext debug);

    static void updateOutsideIoR(inout PathTracer::SurfaceData surfaceData, lpfloat outsideIoR);

    static lpfloat loadIoR(const uint materialID);

    static HomogeneousVolumeData loadHomogeneousVolumeData(const uint materialID);

    // 2.5D motion vectors
    static float3 computeMotionVector(float3 posW, float3 prevPosW);

    // 2.5D motion vectors
    static float3 computeSkyMotionVector(const uint2 pixelPos);

    // The normal AlphaTest
    static bool AlphaTest(
        uint instanceID, 
        uint instanceIndex, 
        uint geometryIndex, 
        uint triangleIndex, 
        float2 rayBarycentrics);

    // The alpha test function used for visibility rays.
    static bool AlphaTestVisibilityRay(
        uint instanceID,
        uint instanceIndex,
        uint geometryIndex,
        uint triangleIndex,
        float2 rayBarycentrics);

    // There's a relatively high cost to this when used in large shaders just due to register allocation required for alphaTest, even if all geometries are opaque.
    // Consider simplifying alpha testing - perhaps splitting it up from the main geometry path, load it with fewer indirections or something like that.
    static bool traceVisibilityRay(RayDesc ray, const RayCone rayCone, const int pathVertexIndex, DebugContext debug);

    static void traceScatterRay(const PathState path, inout RayDesc ray, inout RayQuery<RAY_FLAG_NONE> rayQuery, inout PackedHitInfo packedHitInfo, inout uint SERSortKey, DebugContext debug);

    static void StoreSecondarySurfacePositionAndNormal(uint2 pixelCoordinate, float3 worldPos, float3 normal);
    
    // If HasEnvMap returns false, Eval, EvalPdf and Sample will not be called.
    static bool HasEnvMap();
    
    // Used for evaluating environment map in given direction (but no importance sampling); available if HasEnvMap() returns true
    static EnvMap CreateEnvMap();

    // Used for environment map (distant lights) importance sampling; available if HasEnvMap() returns true
    static EnvMapSampler CreateEnvMapImportanceSampler();

    static LightSampler CreateLightSampler( const uint2 pixelPos, float rayConeWidthOverTotalPathTravel, bool isDebugPixel );
    static LightSampler CreateLightSampler( const uint2 pixelPos, bool isIndirect, bool isDebugPixel );

    static float DiffuseEnvironmentMapMIPOffset( );    ///< Use lower MIP level when sampling environment map. Only 0 produces unbiased results

};

#endif // __PATH_TRACER_BRIDGE_HLSLI__