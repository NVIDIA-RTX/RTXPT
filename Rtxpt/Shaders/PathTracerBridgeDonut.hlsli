/*
* Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#ifndef __PATH_TRACER_BRIDGE_DONUT_HLSLI__ // using instead of "#pragma once" due to https://github.com/microsoft/DirectXShaderCompiler/issues/3943
#define __PATH_TRACER_BRIDGE_DONUT_HLSLI__

// easier if we let Donut do this!
#define ENABLE_METAL_ROUGH_RECONSTRUCTION 1

#include "PathTracerBridge.hlsli"

#include "Bindings/SceneBindings.hlsli"
#include "Bindings/LightingBindings.hlsli"
#include "Bindings/SamplerBindings.hlsli"

// Donut-specific (native engine - we can include before PathTracer to avoid any collisions)
#include <donut/shaders/bindless.h>
#include <donut/shaders/utils.hlsli>

#include "DonutBridgeCommon.hlsli"


#if PATH_TRACER_MODE==PT_DENOISING_LAYERS_BUILD // build
#if PT_USE_RESTIR_GI || PT_USE_RESTIR_DI
#define EXPORT_GBUFFER 1
#endif
#endif


uint Bridge::getSampleIndex()
{
    return g_Const.ptConsts.sampleBaseIndex + g_MiniConst.params.x;
}

float Bridge::getNoisyRadianceAttenuation()
{
    // When using multiple samples within pixel in realtime mode (which share identical camera ray), only noisy part of radiance (i.e. not direct sky) needs to be attenuated!
#if PATH_TRACER_MODE != PT_DENOISING_LAYERS_BUILD
    return g_Const.ptConsts.invSubSampleCount;
#else
    return 1.0;
#endif
}

uint Bridge::getMaxBounceLimit()
{
    return g_Const.ptConsts.bounceCount;
}

uint Bridge::getMaxDiffuseBounceLimit()
{
    return g_Const.ptConsts.diffuseBounceCount;
}

/** Helper to create a texture sampler instance.
The method for computing texture level-of-detail depends on the configuration.
\param[in] path Path state.
\param[in] isPrimaryTriangleHit True if primary hit on a triangle.
\return Texture sampler instance.
*/
ActiveTextureSampler Bridge::createTextureSampler( const RayCone rayCone, const float3 rayDir, float coneTexLODValue, float3 normalW,bool isTriangleHit, float texLODBias
#if RTXPT_STOCHASTIC_TEXTURE_FILTERING_ENABLE
    ,STF_SamplerState stfSamplerState
#endif
)
{
#if ACTIVE_LOD_TEXTURE_SAMPLER == LOD_TEXTURE_SAMPLER_EXPLICIT
    return ExplicitLodTextureSampler::make(texLODBias
#if RTXPT_STOCHASTIC_TEXTURE_FILTERING_ENABLE
        ,stfSamplerState
#endif
    );
#elif ACTIVE_LOD_TEXTURE_SAMPLER == LOD_TEXTURE_SAMPLER_RAY_CONES
    float lambda = rayCone.computeLOD(coneTexLODValue, rayDir, normalW, true);
    lambda += texLODBias;
    return ExplicitRayConesLodTextureSampler::make(lambda
#if RTXPT_STOCHASTIC_TEXTURE_FILTERING_ENABLE
        ,stfSamplerState
#endif
    );
#endif  // ACTIVE_LOD_TEXTURE_SAMPLER
}

lpfloat Bridge::loadIoR(const uint materialDataIndex)
{
    if( materialDataIndex >= g_Const.MaterialCount )
        return 1.0;
    else
        return (lpfloat)t_PTMaterialData[materialDataIndex].IoR;
}

HomogeneousVolumeData Bridge::loadHomogeneousVolumeData(const uint materialDataIndex)
{
    HomogeneousVolumeData ptVolume;
    ptVolume.sigmaS = float3(0,0,0); 
    ptVolume.sigmaA = float3(0,0,0); 
    ptVolume.g = 0.0;

    if( materialDataIndex >= g_Const.MaterialCount )
        return ptVolume;

    VolumePTConstants volumeInfo = t_PTMaterialData[materialDataIndex].Volume;
        
    // these should be precomputed on the C++ side!!
    ptVolume.sigmaS = float3(0,0,0); // no scattering yet
    ptVolume.sigmaA = -log( clamp( volumeInfo.AttenuationColor, 1e-7, 1 ) ) / max( 1e-30, volumeInfo.AttenuationDistance.xxx );
    return ptVolume;
}

// There's a relatively high cost to this when used in large shaders just due to register allocation required for alphaTest, even if all geometries are opaque.
// Consider simplifying alpha testing - perhaps splitting it up from the main geometry path, load it with fewer indirections or something like that.
bool Bridge::traceVisibilityRay(RayDesc ray, const RayCone rayCone, const int pathVertexIndex, const uint2 pixelPos, DebugContext debug)
{
    // RAY_FLAG_FORCE_OMM_2_STATE?
    RayQuery<RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH, RAYQUERY_FLAG_ALLOW_OPACITY_MICROMAPS> rayQuery;
    rayQuery.TraceRayInline(SceneBVH, RAY_FLAG_NONE, 0xff, ray);

    while (rayQuery.Proceed())
    {
        if (rayQuery.CandidateType() == CANDIDATE_NON_OPAQUE_TRIANGLE)
        {
            [branch]if (Bridge::AlphaTestVisibilityRay(
                rayQuery.CandidateInstanceID(),
                rayQuery.CandidateInstanceIndex(),
                rayQuery.CandidateGeometryIndex(),
                rayQuery.CandidatePrimitiveIndex(),
                rayQuery.CandidateTriangleBarycentrics()
                //, debug
                ) )
            {
                rayQuery.CommitNonOpaqueTriangleHit();
                //break; // <- TODO: revisit - not needed when using RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH?
            }
        }
    }
        
#if ENABLE_DEBUG_VIZUALISATIONS && ENABLE_DEBUG_LINES_VIZ && !NON_PATH_TRACING_PASS && PATH_TRACER_MODE!=PT_DENOISING_LAYERS_BUILD
    float visible = rayQuery.CommittedStatus() == COMMITTED_TRIANGLE_HIT;
    if (rayQuery.CommittedStatus() == COMMITTED_TRIANGLE_HIT)
        ray.TMax = rayQuery.CommittedRayT();    // <- this gets passed via NvMakeHitWithRecordIndex/NvInvokeHitObject as RayTCurrent() or similar in ubershader path

    if( debug.IsDebugPixel(pixelPos) )
        debug.DrawLine(ray.Origin, ray.Origin+ray.Direction*ray.TMax, float4(visible.x, visible.x, 0.8, 0.2), float4(visible.x, visible.x, 0.8, 0.2));
#endif

    return !rayQuery.CommittedStatus() == COMMITTED_TRIANGLE_HIT;
}

void Bridge::traceScatterRay(const PathState path, inout RayQuery<RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES, RAYQUERY_FLAG_ALLOW_OPACITY_MICROMAPS> rayQuery, const float2 tMinMax, DebugContext debug)
{
    RayDesc ray = path.getScatterRay().toRayDesc();
    ray.TMin = tMinMax.x;
    ray.TMax = tMinMax.y;
    rayQuery.TraceRayInline(SceneBVH, RAY_FLAG_NONE, 0xff, ray);

#if 1
    while (rayQuery.Proceed())
    {
        if (rayQuery.CandidateType() == CANDIDATE_NON_OPAQUE_TRIANGLE)
        {
            // A.k.a. 'Anyhit' shader!
            [branch]if (Bridge::AlphaTest(
                rayQuery.CandidateInstanceID(),
                rayQuery.CandidateInstanceIndex(),
                rayQuery.CandidateGeometryIndex(),
                rayQuery.CandidatePrimitiveIndex(),
                rayQuery.CandidateTriangleBarycentrics()
                //, workingContext.Debug
                )
            )
            {
                rayQuery.CommitNonOpaqueTriangleHit();
            }
        }
    }
#else
    bool proceed = true;
	while (WaveActiveAnyTrue(proceed))
	{
		proceed = rayQuery.Proceed();
        if (proceed && rayQuery.CandidateType() == CANDIDATE_NON_OPAQUE_TRIANGLE)
        {
            // A.k.a. 'Anyhit' shader!
            if (Bridge::AlphaTest(
                rayQuery.CandidateInstanceID(),
                rayQuery.CandidateInstanceIndex(),
                rayQuery.CandidateGeometryIndex(),
                rayQuery.CandidatePrimitiveIndex(),
                rayQuery.CandidateTriangleBarycentrics()
                //, workingContext.Debug
                ) )
            {
                rayQuery.CommitNonOpaqueTriangleHit();
                // proceed = false;
            }
        }
    }

    // const uint ActivePathCount = WaveActiveCountBits(true);
    // const uint TotalLaneCount = WaveGetLaneCount();
    // const float ActivePathRatio =(float)ActivePathCount /(float)TotalLaneCount;
    // //if (path.getVertexIndex()==1)    
    // //    DebugPixel( path.GetPixelPos(), float4( saturate(GradientHeatMap(0.1)), 1 ) );
    // //else
    // if (path.getVertexIndex()==1)    
    //     DebugPixel( path.GetPixelPos(), float4( saturate(GradientHeatMap(1-sq(ActivePathRatio)+0.1)), 1 ) );


/*
    // Run SER if the number of active lanes that want to proceed drops below threashold (0.75 should be set from some cvar for further tuning)
	const uint threshold = (0.75 * WaveGetLaneCount());
	if (WaveActiveCountBits(proceed) < threshold)
	{
		uint serHint = (proceed ? 1 : 0);
		const uint serNumHintBits = 1;

		//NvHitObject hitObject = NvMakeNop();
		//NvReorderThread(hitObject, serHint, serNumHintBits);
#if SER_USE_SORTING
        SER_REORDER( serHint, serNumHintBits );
#endif
	}
*/

#endif

}

EnvMap Bridge::CreateEnvMap()
{
    return EnvMap::make( t_EnvironmentMap, s_EnvironmentMapSampler, g_Const.envMapSceneParams );
}

EnvMapSampler Bridge::CreateEnvMapImportanceSampler()
{
    return EnvMapSampler::make(
        s_EnvironmentMapImportanceSampler,
        t_EnvironmentMapImportanceMap,
        g_Const.envMapImportanceSamplingParams,
        t_EnvironmentMap,
        s_EnvironmentMapSampler,
        g_Const.envMapSceneParams/*,
        t_PresampledEnvMapBuffer*/
    );
}

LightSampler Bridge::CreateLightSampler( const uint2 pixelPos, float rayConeWidth, float totalPathLength )
{
    bool isScreenSpaceCoherent = LightSampler::IsScreenSpaceCoherentHeuristic( t_LightsCB, rayConeWidth, totalPathLength );
    return LightSampler::make( t_LightsCB, t_Lights, t_LightsEx, t_LightProxyCounters, t_LightProxyIndices, t_LightLocalSamplingBuffer, t_LightWeightsBoost, t_EnvLookupMap, u_LightFeedbackTotalWeight, u_LightFeedbackCandidates, pixelPos, isScreenSpaceCoherent );
}

LightSampler Bridge::CreateLightSampler( const uint2 pixelPos, bool isScreenSpaceCoherent )
{
    return LightSampler::make( t_LightsCB, t_Lights, t_LightsEx, t_LightProxyCounters, t_LightProxyIndices, t_LightLocalSamplingBuffer, t_LightWeightsBoost, t_EnvLookupMap, u_LightFeedbackTotalWeight, u_LightFeedbackCandidates, pixelPos, isScreenSpaceCoherent );
}

bool Bridge::HasEnvMap()
{
    return g_Const.envMapSceneParams.Enabled;
}

float Bridge::DiffuseEnvironmentMapMIPOffset( )
{
    return g_Const.ptConsts.EnvironmentMapDiffuseSampleMIPLevel;
}

void Bridge::ExportSurfaceInit(uint2 pixelPos)
{
#if !RTXPT_SKIP_STANDARD_OUTPUTS
    u_Depth[pixelPos] = 0;                  // this is a signal that data is invalid - there's (rare) cases where neither ExportSurface or ExportNonSurface get called
    u_SpecularHitT[pixelPos] = 0;           // it is common for this to be missing
#endif
    
    // u_MotionVectors[pixelPos] = float4( 0, 0, 0, 0 );   // this should not be strictly necessary as we already know from u_Depth[] that the signal is invalid
    // DebugPixel( pixelPos.xy, float4( 0.0.xxx, 1 ) ); 
}

void Bridge::ExportNonSurface(const PathState path, float3 virtualWorldPos, float3 motionVectors )
{
    uint2 pixelPos = path.GetPixelPos();

#if !RTXPT_SKIP_STANDARD_OUTPUTS
    u_MotionVectors[pixelPos]   = float4(motionVectors, 0);

    float4 clipPos = mul(float4( /*bridgedData.shadingData.posW*/virtualWorldPos, 1), g_Const.view.matWorldToClip);
    u_Depth[pixelPos] = clipPos.z / clipPos.w;
    u_Throughput[pixelPos] = 0;
#endif

    //DebugPixel( pixelPos.xy, float4( 0, 0, 0.2, 1 ) );

#if EXPORT_GBUFFER
    if (g_Const.ptConsts.useReSTIRDI || g_Const.ptConsts.useReSTIRGI)
    {
        // compute address for current output - it ping pongs based on frameIndex!
        const uint idxPingPong = (g_Const.ptConsts.frameIndex % 2) == (uint)0;
        const uint idx = GenericTSPixelToAddress(pixelPos, idxPingPong, g_Const.ptConsts.genericTSLineStride, g_Const.ptConsts.genericTSPlaneStride);

        PackedPathTracerSurfaceData emptyData = (PackedPathTracerSurfaceData)0;
        emptyData._viewDepth_planeHash_isEmpty_frontFacing = 0; // <- we actually only need to initialize this (because of isEmpty) but let's write everything to 0 for consistency
#if !RTXPT_SKIP_STANDARD_OUTPUTS
        u_SurfaceData[idx] = emptyData;
#endif
    }
#endif
}

void Bridge::ExportSpecHitTStart(const PathState path)
{
#if !RTXPT_SKIP_STANDARD_OUTPUTS
    u_SpecularHitT[path.GetPixelPos()] = -path.GetSceneLength();
#endif
}

void Bridge::ExportSpecHitTStop(const PathState path)
{
    uint2 pixelPos = path.GetPixelPos();

    float denoisingSceneLength = u_SpecularHitT[pixelPos];
    if (denoisingSceneLength < 0)  // 0 means not initialized - nothing to do here (shouldn't happen really!); >0 means we've already filled it up in previous pass when using multiple samples per pixel - again, nothing to do here
    {
        // if (denoisingSceneLength > 0)  
        //      DebugPixel( pixelPos.xy, float4( 1.0, 0, 0.0, 1 ) ); // bug!
    
        float specHitT = max( 0, path.GetSceneLength() + denoisingSceneLength );
#if !RTXPT_SKIP_STANDARD_OUTPUTS
        u_SpecularHitT[pixelPos] = specHitT;
#endif
    }
//     if (specHitT > 0.0)
//         //DebugPixel( pixelPos.xy, float4( 0.0, saturate(specHitT/10.0), 1.0, 1 ) );
//        DebugPixel( pixelPos.xy, float4( saturate(specHitT/10.0).xxx, 1 ) );
}

PathTracer::WorkingContext Bridge::GetWorkingContext()
{
    PathTracer::WorkingContext ret;
    ret.PtConsts = g_Const.ptConsts;
    ret.Debug.Init( g_Const.debug, u_FeedbackBuffer, u_DebugLinesBuffer, u_DebugDeltaPathTree, u_DeltaPathSearchStack );
    ret.DenoisingLayers = DenoisingLayersContext::make(u_DenoisingLayersHeader, u_DenoisingLayersBuffer, u_StableRadiance, g_Const.ptConsts);
    ret.OutputColor = u_OutputColor;
    return ret;
}


#endif // __PATH_TRACER_BRIDGE_DONUT_HLSLI__
