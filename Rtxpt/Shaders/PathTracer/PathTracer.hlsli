/*
* Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#ifndef __PATH_TRACER_HLSLI__ // using instead of "#pragma once" due to https://github.com/microsoft/DirectXShaderCompiler/issues/3943
#define __PATH_TRACER_HLSLI__

#include "PathTracerTypes.hlsli"

#include "Surface/ShadingData.hlsli"
#include "PathTracerSHaRC.hlsli"

// Global compile-time path tracer settings for debugging, performance or quality tweaks; could be in a separate file or Config.hlsli but it's convenient to have them in here where they're used.
namespace PathTracer
{
    static const bool           kUseBSDFSampling                = true;     // this setting will be ignored by ReSTIR-DI (RTXDI)

    static const float          kSpecularRoughnessThreshold     = 0.25f;
}

#include "PathTracerDenoisingLayers.hlsli"

namespace PathTracer
{
    /** Check if the path has finished all surface bounces and needs to be terminated.
        Note: This is expected to be called after GenerateScatterRay(), which increments the bounce counters.
        \param[in] path Path state.
        \return Returns true if path has processed all bounces.
    */
    inline bool HasFinishedSurfaceBounces(uint vertexIndex, uint diffuseBounces)
    {
        if (Bridge::getMaxBounceLimit()<vertexIndex)
            return true;
        return diffuseBounces > Bridge::getMaxDiffuseBounceLimit();
    }

    inline PathState EmptyPathInitialize(uint2 pixelPos, float pixelConeSpreadAngle)
    {
        PathState path;
        path.SetId                  ( PathIDFromPixel(pixelPos) );
        path.flagsAndVertexIndex    = 0;
        path.SetSceneLength         ( 0 );
        path.packedCounters         = 0;
        
        // path.setCounter(PackedCounters::SubSampleIndex, subSampleIndex);

#if RTXPT_NESTED_DIELECTRICS_QUALITY > 0 || defined(__INTELLISENSE__)
        for( uint i = 0; i < INTERIOR_LIST_SLOT_COUNT; i++ )
            path.interiorList.slots[i] = 0;
#endif

        path.SetOrigin               ( float3(0, 0, 0) );
        path.SetDir                  ( float3(0, 0, 0) );

        path.SetThp                 (float3(1, 1, 1));

        // path.setHitPacked( HitInfo::make().getData() );
        path.setActive();
        path.setDeltaOnlyPath(true);

        path.rayCone                = RayCone::make(0, pixelConeSpreadAngle);

        SharcPathInit();

#if PATH_TRACER_MODE==PT_DENOISING_LAYERS_BUILD
        path.SetImageXform( lpfloat3x3( 1.f, 0.f, 0.f,      0.f, 1.f, 0.f,     0.f, 0.f, 1.f) );
        path.setFlag(PathFlags::denoisingLayerOnDominantBranch, true); // denoising layer 0 starts being dominant but this can change; in the _FILL_PASS this is predetermined and can't change
        path.SetMotionVectorSceneLength(0);
        //path.SetMotionVectorBounceNormal(float3(0,0,1));
#else
        path.SetL( float4(0, 0, 0, 0) );
        path.SetFireflyFilterK_BsdfScatterPdf(1.0, 0.0);
        // these will be used for the first bounce
        path.SetPackedMISInfo_ThpRuRuCorrection(NEEBSDFMISInfo::empty().Pack16bit(), 1.0);
#endif
        path.setDenoisingLayerIndex(0);
        path.stableBranchID         = 1; // camera has 1; makes IDs unique
        
        if (HasFinishedSurfaceBounces(path.getVertexIndex()+1, path.getCounter(PackedCounters::DiffuseBounces)))
            path.setTerminateAtNextBounce();

        return path;
    }

    inline void StartPixel(const PathState path, const WorkingContext workingContext)
    {
        workingContext.DenoisingLayers.StartPixel(path.GetPixelPos());

#if PATH_TRACER_MODE==PT_REFERENCE
        // workingContext.OutputColor[path.GetPixelPos()] = float4( 0, 0, 0, 1 );
        Bridge::ExportSurfaceInit(path.GetPixelPos());
#endif

#if PATH_TRACER_MODE==PT_DENOISING_LAYERS_BUILD
#if PT_USE_RESTIR_GI
        // this is a fullscreen clear - there will be overwrite in the next pass (inefficiency)
        ReSTIRGI_Clear(path.GetPixelPos());
#endif
        Bridge::ExportSurfaceInit(path.GetPixelPos());
#endif // PATH_TRACER_MODE==PT_DENOISING_LAYERS_BUILD

#if PATH_TRACER_MODE!=PT_DENOISING_LAYERS_FILL    // Reset on denoising layers disabled (0) or denoising layers generate (1)
        workingContext.Debug.Reset(path.GetPixelPos(), 0);   // Setups per-pixel debugging - has to happen before any other debugging stuff in the frame
#endif
    }

    inline void SetupPathPrimaryRay(inout PathState path, const Ray ray)
    {
        path.SetOrigin(ray.origin);
        path.SetDir(ray.dir);
    }

    /** Update the path throughouput.
        \param[in,out] path Path state.
        \param[in] weight Vertex throughput.
    */
    inline void UpdatePathThroughput(inout PathState path, const float3 weight)
    {
        path.SetThp( path.GetThp() * weight );
    }

#if PATH_TRACER_MODE==PT_DENOISING_LAYERS_FILL && PT_USE_RESTIR_GI
    inline bool ShouldCollectGISecondaryRadiance( const PathState path )
    { 
        return path.hasFlag(PathFlags::restirGIStarted);
    }
#else
    inline bool ShouldCollectGISecondaryRadiance( const PathState path ) { return false; }
#endif

    inline void AccumulatePathRadiance(const WorkingContext workingContext, inout PathState path, float3 radiance, const float specularRadianceAvg, bool denoisingLayerOnBranch, bool collectGISecondaryRadiance )
    {
#if PATH_TRACER_MODE==PT_REFERENCE
        path.SetL( path.GetL() + float4( radiance, 0 ) );
#elif PATH_TRACER_MODE==PT_DENOISING_LAYERS_BUILD
        workingContext.DenoisingLayers.AccumulateStableRadiance(path.GetPixelPos(), radiance);
#elif PATH_TRACER_MODE==PT_DENOISING_LAYERS_FILL
        if ( !denoisingLayerOnBranch ) // stable radiance has already been captured above during _BUILD_ into separate stableRadiance buffer
        {
            #if PT_USE_RESTIR_GI
            if (collectGISecondaryRadiance) // divert to GI
                ReSTIRGI_AddSecondarySurfaceRadiance( path.GetPixelPos(), float3( radiance ) );
            else
            #endif
            {
                float4 newL = float4( radiance, specularRadianceAvg ) * Bridge::getNoisyRadianceAttenuation();
                float4 L = path.GetL( );
                path.SetL( L + newL );
            }
        }
#else
    #error mode unsupported
#endif
    }

    // store radiance and any other required data - we're not path tracing anymore
    inline void CommitPixel( const PathState path, const WorkingContext workingContext )
    {
#if PATH_TRACER_MODE==PT_REFERENCE
#if !RTXPT_SKIP_STANDARD_OUTPUTS
        workingContext.OutputColor[path.GetPixelPos()].rgba = float4( path.GetL().rgb, 1 );
#endif
#elif PATH_TRACER_MODE==PT_DENOISING_LAYERS_BUILD
#elif PATH_TRACER_MODE==PT_DENOISING_LAYERS_FILL
#if !RTXPT_SKIP_STANDARD_OUTPUTS
    workingContext.DenoisingLayers.CommitDenoiserRadiance(path);
#endif
#else
    #error mode unsupported
#endif
    }

    /** Apply russian roulette to terminate paths early.
        \param[in,out] path Path.
        \param[in] u Uniform random number in [0,1).
        \return Returns true if path needs to be terminated.
    */
    inline bool HandleRussianRoulette(inout PathState path, inout UniformSampleSequenceGenerator sampleGenerator, const WorkingContext workingContext)
    {
#if PT_ENABLE_RUSSIAN_ROULETTE==0 || PATH_TRACER_MODE==PT_DENOISING_LAYERS_BUILD // denoising layers must be stable, no RR!
        return false;
#else
        const float rrVal = sqrt(Luminance(path.GetThp())); // closer to perceptual
        
        // 'prob' is 'probability to terminate'
#if 0   // old "classic" one
        float prob = max(0.f, 1.f - rrVal);
#else   // a milder version of Falcor's Russian Roulette
        float prob = saturate( 0.85 - rrVal ); prob = prob*prob;
#endif

#if 1 // start stochastically terminating paths from 0.4 bounce limit, with increasing probability up to 0.6 (1-0.4)
        prob = saturate( prob + max( 0, ((float)path.getVertexIndex() / (float)Bridge::getMaxBounceLimit() - 0.4 ) ) );
#endif

        if (sampleNext1D(sampleGenerator) < prob)
            return true;
        
        float thpRuRuCorrection = lpfloat(1.0 / (1.0 - prob)); // note, thpRuRuCorrection is packed as fp16
        path.SetPackedMISInfo_ThpRuRuCorrection( path.GetPackedMISInfo(), thpRuRuCorrection );

        return false;
#endif // #if PT_ENABLE_RUSSIAN_ROULETTE==0 || PATH_TRACER_MODE==PT_DENOISING_LAYERS_BUILD
    }

    inline void UpdatePathTravelledLengthOnly(inout PathState path, const float rayTCurrent)
    {
        path.rayCone = path.rayCone.propagateDistance(rayTCurrent);             // Grow the cone footprint based on angle; angle itself can change on scatter
        path.SetSceneLength( min(path.GetSceneLength()+rayTCurrent, kMaxRayTravel ));    // Advance total travel length
    }

    // Called after ray tracing just before handleMiss or handleHit, to advance internal states related to travel
    inline void UpdatePathTravelled(inout PathState path, const float3 rayOrigin, const float3 rayDir, const float rayTCurrent, const WorkingContext workingContext)
    {
        #if 0 // these two do not need to be set here; they are only intended as "output" from ray tracer; but we need to be careful and not use as input in ClosestHit
        path.SetOrigin( rayOrigin );    
        path.SetDir( rayDir );
        #endif
        path.incrementVertexIndex();                                        // Advance to next path vertex (PathState::vertexIndex). (0 - camera, 1 - first bounce, ...)

        UpdatePathTravelledLengthOnly(path, rayTCurrent);

        // good place for debug viz
#if ENABLE_DEBUG_VIZUALISATIONS && ENABLE_DEBUG_LINES_VIZ && !NON_PATH_TRACING_PASS// && PATH_TRACER_MODE!=PT_DENOISING_LAYERS_BUILD <- let's actually show the build rays - maybe even add them some separate effect in the future
        if( workingContext.Debug.IsDebugPixel(path.GetPixelPos()) )
            workingContext.Debug.DrawLine(rayOrigin, rayOrigin+rayDir*rayTCurrent, float4(0.6.xxx, 0.5), float4(1.0.xxx, 1.0));
#endif
    }

    // Miss shader
    inline void HandleMiss(inout PathState path, const float3 rayOrigin, const float3 rayDir, const float rayTCurrent, const WorkingContext workingContext)
    {
        UpdatePathTravelled(path, rayOrigin, rayDir, rayTCurrent, workingContext);

#if PATH_TRACER_MODE==PT_DENOISING_LAYERS_FILL
        if (path.hasFlag(PathFlags::exportSpecHitTQueued)) // this is the end of the trip, so if exporting SpecHitT - do it now!
        {
            Bridge::ExportSpecHitTStop( path );
            path.setFlag( PathFlags::exportSpecHitTQueued, false );
        }
#endif

#if PATH_TRACER_MODE==PT_DENOISING_LAYERS_BUILD && ENABLE_DEBUG_DELTA_TREE_VIZUALISATION
        if (path.hasFlag(PathFlags::deltaTreeExplorer))
        {
            DeltaTreeVizHandleMiss(path, rayOrigin, rayDir, rayTCurrent, workingContext);
            return;
        }
#endif

#if PATH_TRACER_MODE==PT_DENOISING_LAYERS_FILL && PT_USE_RESTIR_GI
        if (path.hasFlag(PathFlags::restirGICollectSecondarySurface))
        {
            const float3 worldPos = rayOrigin + rayDir * kMaxSceneDistance;
            const float3 normal = -rayDir;

            ReSTIRGI_StoreSecondarySurfacePositionAndNormal(path.GetPixelPos(), worldPos, normal);
            // path.setFlag(PathFlags::restirGICollectSecondarySurface, false); // unnecessary, miss, path is ending
        }
#endif

        lpfloat3 environmentEmission = 0.f;

        NEEBSDFMISInfo misInfo = NEEBSDFMISInfo::Unpack16bit(path.GetPackedMISInfo());
        if ( !(misInfo.GetSkipEmissiveBRDF() && !path.wasScatterTransmission()) )
        {
            // raw source for our environment map
            EnvMap envMap = Bridge::CreateEnvMap(); 

            // sample lower MIP after second diffuse bounce; mip0 gets too costly for high res environment maps
            float mipLevel = (path.getCounter(PackedCounters::DiffuseBounces)>1)?(Bridge::DiffuseEnvironmentMapMIPOffset()):(0); 

            // convert to environment map's local dir (as it supports its own rotation matrix)
            float3 localDir = envMap.ToLocal(rayDir);     
            float3 Le = envMap.EvalLocal(localDir, mipLevel);

            // figure out MIS vs our lighting technique, if any
            float misWeight = 1.0f;
#if NEEAT_ENABLE_ENVMAP_PROCESSING && PATH_TRACER_MODE!=PT_DENOISING_LAYERS_BUILD   // NO MIS in PT_DENOISING_LAYERS_BUILD
            lpfloat bsdfScatterPdf = path.GetBsdfScatterPdf();
            if( misInfo.LightSamplingEnabled && bsdfScatterPdf != 0 )
            {
                // this is the NEE light sampler configured same as it was at previous vertex (previous vertex's "next event estimation" matches this light "event")
                LightSampler lightSampler = Bridge::CreateLightSampler( path.GetPixelPos(), misInfo.LightSamplingIsSSC );

#if RTXPT_USE_APPROXIMATE_MIS
                misWeight = lightSampler.ComputeLightVsBSDF_MIS_ForBSDF_Approx(bsdfScatterPdf, misInfo.CandidateSamples, misInfo.FullSamples);
#else
                uint environmentQuadLightIndex = lightSampler.LookupEnvLightByDirection( localDir ); //< figure out light index from the direction! it's guaranteed to be valid
                misWeight = lightSampler.ComputeBSDFMISForEnvironmentQuad(environmentQuadLightIndex, bsdfScatterPdf, misInfo.CandidateSamples, misInfo.FullSamples);
#endif

#if RTXPT_LIGHTING_ENABLE_BSDF_FEEDBACK
#if RTXPT_USE_APPROXIMATE_MIS
                uint environmentQuadLightIndex = lightSampler.LookupEnvLightByDirection( localDir );
#endif
                float simpleRandom = Hash32ToFloat( Hash32Combine( Hash32Combine(Hash32(path.getVertexIndex() + 0x0366FE2F), path.GetId()), Bridge::getSampleIndex() ) );   // note: using unique prime number salt for vertex index
                lightSampler.InsertFeedbackFromBSDF(environmentQuadLightIndex, Luminance(path.GetThp()*Le), misWeight, simpleRandom );
#endif
            }
#endif // #if NEEAT_ENABLE_ENVMAP_PROCESSING && PATH_TRACER_MODE!=PT_DENOISING_LAYERS_BUILD

            environmentEmission = ToLpRadiance(misWeight * Le);
        }

#if RTXPT_FIREFLY_FILTER && PATH_TRACER_MODE!=PT_DENOISING_LAYERS_BUILD
            float baseFFThreshold = workingContext.PtConsts.fireflyFilterThreshold;
            if( baseFFThreshold != 0 )
                environmentEmission = (lpfloat3)FireflyFilter( environmentEmission, baseFFThreshold, path.GetFireflyFilterK() );
#endif

#if PATH_TRACER_MODE!=PT_REFERENCE
        DenoisingLayersHandleMiss(path, environmentEmission, rayOrigin, rayDir, rayTCurrent, workingContext);
#else
        // in case of reference path tracer, dump guide buffers just in case ever needed; these are dumped every frame and are not accumulated so they'll jitter
        Bridge::ExportNonSurface(path, rayOrigin+rayDir*rayTCurrent, float3(0,0,0) );
#endif

        if (any(environmentEmission>0))
        {
            environmentEmission = clamp( environmentEmission, (lpfloat3)0, (lpfloat3)HLF_MAX );


            float3 radiance = path.GetThp() * environmentEmission;
#if RTXPT_DISCARD_NON_NEE_LIGHTING
            // Scales the accumulated radiance, NOT environmentEmission - SharcHandleMiss below needs the real value.
            radiance *= 0.0000000596;   // not 0, so the compiler doesn't compile everything out
#endif
            float specRadianceAvg = path.hasFlag(PathFlags::denoisingLayerBaseScatterDiff)?(0.0):(Average(radiance));
            AccumulatePathRadiance( workingContext, path, radiance, specRadianceAvg, path.hasFlag(PathFlags::denoisingLayerOnBranch), ShouldCollectGISecondaryRadiance(path) );
        }

        SharcHandleMiss(environmentEmission);

        path.clearHit();
        path.terminate();
    }
};

#endif // __PATH_TRACER_HLSLI__
