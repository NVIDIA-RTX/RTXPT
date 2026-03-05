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

#include "Scene/ShadingData.hlsli"

// Global compile-time path tracer settings for debugging, performance or quality tweaks; could be in a separate file or Config.hlsli but it's convenient to have them in here where they're used.
namespace PathTracer
{
    static const bool           kUseBSDFSampling                = true;     // this setting will be ignored by ReSTIR-DI (RTXDI)

    static const float          kSpecularRoughnessThreshold     = 0.25f;
}

#include "PathTracerNestedDielectrics.hlsli"
#include "PathTracerStablePlanes.hlsli"

#if !NON_PATH_TRACING_PASS
#include "PathTracerNEE.hlsli"
#endif

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

#if PATH_TRACER_MODE==PATH_TRACER_MODE_BUILD_STABLE_PLANES
        path.SetImageXform( lpfloat3x3( 1.f, 0.f, 0.f,      0.f, 1.f, 0.f,     0.f, 0.f, 1.f) );
        path.setFlag(PathFlags::stablePlaneOnDominantBranch, true); // stable plane 0 starts being dominant but this can change; in the _FILL_PASS this is predetermined and can't change
        path.SetMotionVectorSceneLength(0);
        //path.SetMotionVectorBounceNormal(float3(0,0,1));
#else
        path.SetL( float4(0, 0, 0, 0) );
        path.SetFireflyFilterK_BsdfScatterPdf(1.0, 0.0);
        // these will be used for the first bounce
        path.SetPackedMISInfo_ThpRuRuCorrection(NEEBSDFMISInfo::empty().Pack16bit(), 1.0);
#endif
        path.setStablePlaneIndex(0);
        path.stableBranchID         = 1; // camera has 1; makes IDs unique
        
        if (HasFinishedSurfaceBounces(path.getVertexIndex()+1, path.getCounter(PackedCounters::DiffuseBounces)))
            path.setTerminateAtNextBounce();

        return path;
    }

    inline void StartPixel(const PathState path, const WorkingContext workingContext)
    {
        workingContext.StablePlanes.StartPixel(path.GetPixelPos());

#if PATH_TRACER_MODE==PATH_TRACER_MODE_REFERENCE
        // workingContext.OutputColor[path.GetPixelPos()] = float4( 0, 0, 0, 1 );
        Bridge::ExportSurfaceInit(path.GetPixelPos());
#endif

#if PATH_TRACER_MODE==PATH_TRACER_MODE_BUILD_STABLE_PLANES
#if PT_USE_RESTIR_GI
        // this is a fullscreen clear - there will be overwrite in the next pass (inefficiency)
        ReSTIRGI_Clear(path.GetPixelPos());
#endif
        Bridge::ExportSurfaceInit(path.GetPixelPos());
#endif // PATH_TRACER_MODE==PATH_TRACER_MODE_BUILD_STABLE_PLANES

#if PATH_TRACER_MODE!=PATH_TRACER_MODE_FILL_STABLE_PLANES    // Reset on stable planes disabled (0) or stable planes generate (1)
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

#if PATH_TRACER_MODE==PATH_TRACER_MODE_FILL_STABLE_PLANES && PT_USE_RESTIR_GI
    inline bool ShouldCollectGISecondaryRadiance( const PathState path )
    { 
        return path.hasFlag(PathFlags::restirGIStarted);
    }
#else
    inline bool ShouldCollectGISecondaryRadiance( const PathState path ) { return false; }
#endif

    inline void AccumulatePathRadiance(const WorkingContext workingContext, inout PathState path, float3 radiance, const float specularRadianceAvg, bool stablePlaneOnBranch, bool collectGISecondaryRadiance )
    {
#if PATH_TRACER_MODE==PATH_TRACER_MODE_REFERENCE
        path.SetL( path.GetL() + float4( radiance, 0 ) );
#elif PATH_TRACER_MODE==PATH_TRACER_MODE_BUILD_STABLE_PLANES
        workingContext.StablePlanes.AccumulateStableRadiance(path.GetPixelPos(), radiance);
#elif PATH_TRACER_MODE==PATH_TRACER_MODE_FILL_STABLE_PLANES
        if ( !stablePlaneOnBranch ) // stable radiance has already been captured above during _BUILD_ into separate stableRadiance buffer
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
#if PATH_TRACER_MODE==PATH_TRACER_MODE_REFERENCE
        workingContext.OutputColor[path.GetPixelPos()].rgba = float4( path.GetL().rgb, 1 );
#elif PATH_TRACER_MODE==PATH_TRACER_MODE_BUILD_STABLE_PLANES
#elif PATH_TRACER_MODE==PATH_TRACER_MODE_FILL_STABLE_PLANES
    workingContext.StablePlanes.CommitDenoiserRadiance(path);
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
#if PT_ENABLE_RUSSIAN_ROULETTE==0 || PATH_TRACER_MODE==PATH_TRACER_MODE_BUILD_STABLE_PLANES // stable planes must be stable, no RR!
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
#endif // #if PT_ENABLE_RUSSIAN_ROULETTE==0 || PATH_TRACER_MODE==PATH_TRACER_MODE_BUILD_STABLE_PLANES
    }

    /** Generates a new scatter ray given a valid BSDF sample.
        \param[in] bs BSDF sample (assumed to be valid).
        \param[in] sd Shading data.
        \param[in] bsdf BSDF at the shading point.
        \param[in,out] path The path state.
        \return True if a ray was generated, false otherwise.
    */
    inline bool GenerateScatterRay(const BSDFSample bs, const ShadingData shadingData, const ActiveBSDF bsdf, inout PathState path, const WorkingContext workingContext)
    {
        path.SetDir(bs.wo);

        bool onDominantDenoisingLayer = path.hasFlag(PathFlags::stablePlaneOnPlane) && path.hasFlag(PathFlags::stablePlaneOnDominantBranch);

        #if PATH_TRACER_MODE==PATH_TRACER_MODE_FILL_STABLE_PLANES && PT_USE_RESTIR_GI
        if (onDominantDenoisingLayer && !path.hasFlag(PathFlags::restirGIStarted) && bs.pdf != 0 )
        {
            // if 
            //  a.) >previous< state was that we were on the stable plane and it's a dominant branch,
            //  b.) we haven't already started restirGI (should be unnecessary - there should be only 1 opportunity to start),
            //  c.) and the scatter event isn't a delta scatter
            // then, and only then we start collecting data for ReSTIR GI

            path.setFlag(PathFlags::restirGIStarted, true); // marks path to be permanently 
            path.setFlag(PathFlags::restirGICollectSecondarySurface, true);
            
            // ReSTIR GI decomposes the throughput of the primary scatter ray into the BRDF and PDF components.
            // The PDF component is applied here, and the BRDF component is applied in the ReSTIR GI final shading pass.
            #if 1
            ReSTIRGI_StorePrimarySurfaceScatterPdf(path.GetPixelPos(), bs.pdf);
            #else
            UpdatePathThroughput(path, 1.0 / bs.pdf);
            ReSTIRGI_StorePrimarySurfaceScatterPdf(path.GetPixelPos(), 1.0);
            #endif
        }
        else
        #endif
            UpdatePathThroughput(path, bs.weight);

        path.clearScatterEventFlags(); // removes PathFlags::transmission, PathFlags::specular, PathFlags::delta flags

        // Compute ray origin for next ray segment.
        path.SetOrigin(shadingData.computeNewRayOrigin(bs.isLobe(LobeType::Reflection)));
        
        const float roughness = bsdf.data.Roughness();

        // Handle reflection events.
        //if (bs.isLobe(LobeType::Reflection)) <- let's remove this and allow diffuse transmission to be classified as diffuse (makes sense, no?)
        {
            // We classify specular events as diffuse if the roughness is above some threshold.
            bool isDiffuse = bs.isLobe(LobeType::DiffuseReflection) || bs.isLobe(LobeType::DiffuseTransmission) || roughness > kSpecularRoughnessThreshold;

            if (isDiffuse)
            {
                // ...but in case of diffuse transmission, increase the counter only every second vertex to avoid overdarkening of diffuse volumes - this is a bit ad-hoc but works for now
                if ( ! (bs.isLobe(LobeType::DiffuseTransmission) && ((path.getVertexIndex() % 2)==1)) )
                    path.incrementCounter(PackedCounters::DiffuseBounces);
            }
            else
            {
                // path.incrementBounces(BounceType::Specular);
                path.setScatterSpecular();
            }
        }

        // Handle transmission events.
        if (bs.isLobe(LobeType::Transmission))
        {
            // path.incrementBounces(BounceType::Transmission);
            path.setScatterTransmission();

            // Update interior list and inside volume flag if needed.
            UpdateNestedDielectricsOnScatterTransmission(shadingData, path, workingContext);
        }

        float angleBefore = path.rayCone.getSpreadAngle();

        // Handle delta events.
        if (bs.isLobe(LobeType::Delta))
            path.setScatterDelta();
        else
        {
            path.setDeltaOnlyPath(false);
            path.rayCone = RayCone::make(path.rayCone.getWidth(), min( path.rayCone.getSpreadAngle() + ComputeRayConeSpreadAngleExpansionByScatterPDF( bs.pdf ), 2.0 * K_PI ) );
        }

#if PATH_TRACER_MODE==PATH_TRACER_MODE_FILL_STABLE_PLANES
        static const float          kSpecularRoughnessThresholdForHitT     = 0.35f;
        const bool isDiffuseForSpecHitT = bs.isLobe(LobeType::DiffuseReflection) || bs.isLobe(LobeType::DiffuseTransmission) || roughness > kSpecularRoughnessThresholdForHitT;
        if (onDominantDenoisingLayer && !isDiffuseForSpecHitT)
        {
            if (!shadingData.mtl.isPSDBlockMotionVectorsAtSurface())   // we've given up on this for now
            {
                path.setFlag(PathFlags::exportSpecHitTQueued, true);
                Bridge::ExportSpecHitTStart( path );
            }
        }
        else
        if (path.hasFlag(PathFlags::exportSpecHitTQueued))
        {
            const uint lobes = bsdf.getLobes(shadingData);
            const bool hasNonDeltaLobes = ((lobes & (uint) LobeType::NonDelta) != 0);

            // various heuristics to test & tune <shrug>
            const int maxHitTSpecBounces = 4;
            //if (isDiffuseForSpecHitT || path.getCounter(PackedCounters::BouncesFromStablePlane) > maxHitTSpecBounces) // if we've reached first diffuse surface, time to record specHitT
            //const bool isDiffuseForSpecHitT2 = bs.isLobe(LobeType::DiffuseReflection) || bs.isLobe(LobeType::DiffuseTransmission) || roughness > 0.08;
            //if (isDiffuseForSpecHitT2 || path.getCounter(PackedCounters::BouncesFromStablePlane) > maxHitTSpecBounces) // if we've reached first diffuse surface, time to record specHitT
            //if (!bs.isLobe(LobeType::Delta) || path.getCounter(PackedCounters::BouncesFromStablePlane) > maxHitTSpecBounces) // if we've reached first diffuse surface, time to record specHitT
            if (hasNonDeltaLobes || path.getCounter(PackedCounters::BouncesFromStablePlane) > maxHitTSpecBounces) // if we've reached first diffuse surface, time to record specHitT    
            {
                Bridge::ExportSpecHitTStop( path );
                path.setFlag( PathFlags::exportSpecHitTQueued, false );
            }
        }
#endif

#if PATH_TRACER_MODE!=PATH_TRACER_MODE_BUILD_STABLE_PLANES
#if RTXPT_FIREFLY_FILTER 
        // if bouncePDF then it's a delta event - expansion angle is 0
        float fireflyFilterK = ComputeNewScatterFireflyFilterK(path.GetFireflyFilterK(), bs.pdf, bs.lobeP);
#else
        float fireflyFilterK = 0;
#endif
        path.SetFireflyFilterK_BsdfScatterPdf( fireflyFilterK, bs.pdf );
#endif


#if PATH_TRACER_MODE==PATH_TRACER_MODE_FILL_STABLE_PLANES
        StablePlanesOnScatter(path, bs, workingContext);
#endif

        path.setFlag( PathFlags::enableThreadReorder, true );

        // Mark the path as valid only if it has a non-zero throughput. 
        return true; // any(path.thp > 0.f); <- This is an optimization. It's unnecessary for correctness.
    }

    /** Generates a new scatter ray using BSDF importance sampling.
        \param[in] sd Shading data.
        \param[in] bsdf BSDF at the shading point.
        \param[in,out] path The path state.
        \return True if a ray was generated, false otherwise.
    */
    inline bool GenerateScatterRay(const ShadingData shadingData, const ActiveBSDF bsdf, inout PathState path, const SampleGeneratorVertexBase sgBase, const WorkingContext workingContext)
    {
        // only ActiveBSDF::cRandomNumberCountForSampling are actually used; this is the best formula for compiler optimizations; sending SampleGenerator as an argument to bsdf.sample
        // keeps too much state alive and adding explicit two codepaths instead of using generic 
        float4 preGeneratedSamples; 
        {
#if 0 // slower, old path
            SampleGenerator sampleGenerator = SampleGenerator::make( sgBase, SampleGeneratorEffectSeed::ScatterBSDF, path.getCounter(PackedCounters::DiffuseBounces)<DisableLowDiscrepancySamplingAfterDiffuseBounceCount );
            [unroll] for( int i = 0; i < ActiveBSDF::cRandomNumberCountForSampling; i++ )
                preGeneratedSamples[i] = sampleNext1D(sampleGenerator);
#else
    #if RTXPT_ENABLE_LOW_DISCREPANCY_SAMPLER_FOR_BSDF
            [branch] if( path.getCounter(PackedCounters::DiffuseBounces)<DisableLowDiscrepancySamplingAfterDiffuseBounceCount )
                preGeneratedSamples = SampleGenerator::Generate( ActiveBSDF::cRandomNumberCountForSampling, sgBase, SampleGeneratorEffectSeed::ScatterBSDF );
            else
    #endif // RTXPT_ENABLE_LOW_DISCREPANCY_SAMPLER_FOR_BSDF
                preGeneratedSamples = UniformSampleSequenceGenerator::Generate( ActiveBSDF::cRandomNumberCountForSampling, sgBase, SampleGeneratorEffectSeed::ScatterBSDF );
#endif
        }

        BSDFSample result;
        bool valid = bsdf.sample(shadingData, preGeneratedSamples, result, kUseBSDFSampling);

        if (valid)
            return GenerateScatterRay(result, shadingData, bsdf, path, workingContext);
        else
            return false;
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
#if ENABLE_DEBUG_VIZUALISATIONS && ENABLE_DEBUG_LINES_VIZ && !NON_PATH_TRACING_PASS// && PATH_TRACER_MODE!=PATH_TRACER_MODE_BUILD_STABLE_PLANES <- let's actually show the build rays - maybe even add them some separate effect in the future
        if( workingContext.Debug.IsDebugPixel() )
            workingContext.Debug.DrawLine(rayOrigin, rayOrigin+rayDir*rayTCurrent, float4(0.6.xxx, 0.5), float4(1.0.xxx, 1.0));
#endif
    }

    // Miss shader
    inline void HandleMiss(inout PathState path, const float3 rayOrigin, const float3 rayDir, const float rayTCurrent, const WorkingContext workingContext)
    {
        UpdatePathTravelled(path, rayOrigin, rayDir, rayTCurrent, workingContext);

#if PATH_TRACER_MODE==PATH_TRACER_MODE_FILL_STABLE_PLANES
        if (path.hasFlag(PathFlags::exportSpecHitTQueued)) // this is the end of the trip, so if exporting SpecHitT - do it now!
        {
            Bridge::ExportSpecHitTStop( path );
            path.setFlag( PathFlags::exportSpecHitTQueued, false );
        }
#endif

#if PATH_TRACER_MODE==PATH_TRACER_MODE_BUILD_STABLE_PLANES && ENABLE_DEBUG_DELTA_TREE_VIZUALISATION
        if (path.hasFlag(PathFlags::deltaTreeExplorer))
        {
            DeltaTreeVizHandleMiss(path, rayOrigin, rayDir, rayTCurrent, workingContext);
            return;
        }
#endif

#if PATH_TRACER_MODE==PATH_TRACER_MODE_FILL_STABLE_PLANES && PT_USE_RESTIR_GI
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
                float simpleRandom = Hash32ToFloat( Hash32Combine( Hash32Combine(Hash32(path.getVertexIndex() + 0x0366FE2F), path.id), Bridge::getSampleIndex() ) );   // note: using unique prime number salt for vertex index
                lightSampler.InsertFeedbackFromBSDF(environmentQuadLightIndex, Average(path.GetThp()*Le), misWeight, simpleRandom );
#endif
            }

            environmentEmission = lpfloat3(misWeight * Le);
        }

#if RTXPT_FIREFLY_FILTER && PATH_TRACER_MODE!=PATH_TRACER_MODE_BUILD_STABLE_PLANES
            lpfloat baseFFThreshold = (lpfloat)workingContext.PtConsts.fireflyFilterThreshold;
            if( baseFFThreshold != 0 )
                environmentEmission = FireflyFilter( environmentEmission, baseFFThreshold, path.GetFireflyFilterK() );
#endif

#if PATH_TRACER_MODE!=PATH_TRACER_MODE_REFERENCE
        StablePlanesHandleMiss(path, environmentEmission, rayOrigin, rayDir, rayTCurrent, workingContext);
#else
        // in case of reference path tracer, dump guide buffers just in case ever needed; these are dumped every frame and are not accumulated so they'll jitter
        Bridge::ExportNonSurface(path, rayOrigin+rayDir*rayTCurrent, float3(0,0,0) );
#endif

        if (any(environmentEmission>0))
        {
#if RTXPT_DISCARD_NON_NEE_LIGHTING
            environmentEmission *= 1e-15f;
#endif

            float3 radiance = path.GetThp() * environmentEmission;
            float specRadianceAvg = path.hasFlag(PathFlags::stablePlaneBaseScatterDiff)?(0.0):(Average(radiance));
            AccumulatePathRadiance( workingContext, path, radiance, specRadianceAvg, path.hasFlag(PathFlags::stablePlaneOnBranch), ShouldCollectGISecondaryRadiance(path) );
        }

        path.clearHit();
        path.terminate();
    }

    inline void HandleHit(inout PathState path, const float3 rayOrigin, const float3 rayDir, const float rayTCurrent, float2 barycentrics, const WorkingContext workingContext)
    {
        UpdatePathTravelled(path, rayOrigin, rayDir, rayTCurrent, workingContext);
        
#if ENABLE_DEBUG_VIZUALISATIONS
        const bool debugPath = workingContext.Debug.IsDebugPixel(path.GetPixelPos());
#else
        const bool debugPath = false;
#endif

        SurfaceData surfaceData = Bridge::loadSurface( InstanceIndex(), GeometryIndex(), PrimitiveIndex(), barycentrics, rayDir, path.rayCone, path.getVertexIndex(), path.GetPixelPos(), workingContext.Debug);

        // if (surfaceData.shadingData.mtl.isPSDBlockMotionVectorsAtSurface())   // we've given up on this for now
        //     path.setFlag(PathFlags::exportSpecHitTBlocked, true);

#if PATH_TRACER_MODE==PATH_TRACER_MODE_FILL_STABLE_PLANES && PT_USE_RESTIR_GI
        if (path.hasFlag(PathFlags::restirGICollectSecondarySurface))
        {
            ReSTIRGI_StoreSecondarySurfacePositionAndNormal(path.GetPixelPos(), surfaceData.shadingData.posW, surfaceData.shadingData.N);
            path.setFlag(PathFlags::restirGICollectSecondarySurface, false);
        }
#endif

#if PATH_TRACER_MODE==PATH_TRACER_MODE_FILL_STABLE_PLANES
        // an example of debugging RayCone data for the specific pixel selected in the UI, at the first bounce (vertex index 1)
        // if( workingContext.Debug.IsDebugPixel() && path.getVertexIndex()==1 )
        //     workingContext.Debug.Print( 4, path.rayCone.getSpreadAngle(), path.rayCone.getWidth(), rayTCurrent, path.sceneLength);
#endif

        
        // Account for volume absorption. Shouldn't this be in the miss shader too? I don't know - I guess depends on whether we want to support open, infinite volumes?
        float volumeAbsorption = 0;   // used for stats

#if RTXPT_NESTED_DIELECTRICS_QUALITY > 0 || defined(__INTELLISENSE__)
        if (!path.interiorList.isEmpty())
        {
            const uint materialID = path.interiorList.getTopMaterialID();
            const HomogeneousVolumeData hvd = Bridge::loadHomogeneousVolumeData(materialID); // gScene.materials.getHomogeneousVolumeData(materialID);
            const float3 transmittance = HomogeneousVolumeSampler::evalTransmittance(hvd, rayTCurrent);
            volumeAbsorption = 1.0 - Luminance(transmittance);
            UpdatePathThroughput(path, transmittance);
        }
#endif

        // Reject false hits in nested dielectrics but also updates 'outside index of refraction' and dependent data
        bool rejectedFalseHit = !HandleNestedDielectrics(surfaceData, path, workingContext);

#if PATH_TRACER_MODE==PATH_TRACER_MODE_BUILD_STABLE_PLANES && ENABLE_DEBUG_DELTA_TREE_VIZUALISATION
        if (path.hasFlag(PathFlags::deltaTreeExplorer))
        {
            DeltaTreeVizHandleHit(path, rayOrigin, rayDir, rayTCurrent, surfaceData, rejectedFalseHit, HasFinishedSurfaceBounces(path.getVertexIndex(), path.getCounter(PackedCounters::DiffuseBounces)), volumeAbsorption, workingContext);
            return;
        }
#endif
        if (rejectedFalseHit)
        {
#if PATH_TRACER_MODE==PATH_TRACER_MODE_BUILD_STABLE_PLANES && !NESTED_DIELECTRICS_AVOID_TERMINATION
            if (path.isTerminated()) // HandleNestedDielectrics can terminate the ray if it ends up in a crazy loop - this needs proper handling to avoid holes in data
                StablePlanesHandleMiss(path, 0, rayOrigin, rayDir, rayTCurrent, workingContext);
#endif
            return;
        }

        // These will not change anymore, so make const shortcuts
        const ShadingData shadingData    = surfaceData.shadingData;
        const ActiveBSDF bsdf   = surfaceData.bsdf;

#if ENABLE_DEBUG_VIZUALISATIONS && ENABLE_DEBUG_LINES_VIZ && PATH_TRACER_MODE!=PATH_TRACER_MODE_BUILD_STABLE_PLANES
        if (debugPath)
        {
            // IoR debugging - .x - "outside", .y - "interior", .z - frontFacing, .w - "eta" (eta is isFrontFace?outsideIoR/insideIoR:insideIoR/outsideIoR)
            // workingContext.Debug.Print(path.getVertexIndex(), float4(shadingData.IoR, surfaceData.interiorIoR, shadingData.frontFacing, bsdf.data.eta) );

            // draw tangent space
            workingContext.Debug.DrawLine(shadingData.posW, shadingData.posW + shadingData.T * workingContext.Debug.LineScale(), float4(0.7, 0, 0, 0.4), float4(1.0, 0, 0, 0.4));
            workingContext.Debug.DrawLine(shadingData.posW, shadingData.posW + shadingData.B * workingContext.Debug.LineScale(), float4(0, 0.7, 0, 0.4), float4(0, 1.0, 0, 0.4));
            workingContext.Debug.DrawLine(shadingData.posW, shadingData.posW + shadingData.N * workingContext.Debug.LineScale(), float4(0, 0, 0.7, 0.4), float4(0, 0, 1.0, 0.4));

            // draw ray cone footprint
            float coneWidth = path.rayCone.getWidth();
            workingContext.Debug.DrawLine(shadingData.posW + (-shadingData.T+shadingData.B) * coneWidth, shadingData.posW + (+shadingData.T+shadingData.B) * coneWidth, float4(0.5, 0.0, 1.0, 0.3), float4(0.5, 1.0, 0.0, 0.3) );
            workingContext.Debug.DrawLine(shadingData.posW + (+shadingData.T+shadingData.B) * coneWidth, shadingData.posW + (+shadingData.T-shadingData.B) * coneWidth, float4(0.5, 0.0, 1.0, 0.3), float4(0.5, 1.0, 0.0, 0.3) );
            workingContext.Debug.DrawLine(shadingData.posW + (+shadingData.T-shadingData.B) * coneWidth, shadingData.posW + (-shadingData.T-shadingData.B) * coneWidth, float4(0.5, 0.0, 1.0, 0.3), float4(0.5, 1.0, 0.0, 0.3) );
            workingContext.Debug.DrawLine(shadingData.posW + (-shadingData.T-shadingData.B) * coneWidth, shadingData.posW + (-shadingData.T+shadingData.B) * coneWidth, float4(0.5, 0.0, 1.0, 0.3), float4(0.5, 1.0, 0.0, 0.3) );
        }
#endif

        // Collect emissive triangle radiance.
        lpfloat3 surfaceEmission = 0.0;
        NEEBSDFMISInfo misInfo = NEEBSDFMISInfo::Unpack16bit(path.GetPackedMISInfo());

#if !defined(RTXPT_MATERIAL_IS_EMISSIVE) || RTXPT_MATERIAL_IS_EMISSIVE

#if 0        
        if (workingContext.Debug.IsDebugPixel())
        {
            DebugPrint( "vi {0}, isEn {1}, isInd {2}, zeroB {3}, n {4}, t {5}, pdf {6}", vertexIndex, (uint)misInfo.LightSamplingEnabled, (uint)misInfo.LightSamplingIsSSC, (uint)misInfo.SkipEmissiveBSDF, misInfo.CandidateSamples, misInfo.FullSamples, path.bsdfScatterPdf );
        }
#endif

        // should we ignore this triangle's emission, and if not, is there any emission
        if ( !(misInfo.GetSkipEmissiveBRDF() && !path.wasScatterTransmission()) && (any(shadingData.emission>0)) )
        {
            // figure out MIS vs our lighting technique, if any
            float misWeight = 1.0f;
            lpfloat bsdfScatterPdf = path.GetBsdfScatterPdf();
            if( misInfo.LightSamplingEnabled && bsdfScatterPdf != 0 )
            {
                // if(surfaceData.neeTriangleLightIndex == 0xFFFFFFFF)                     // this is really bad if happens
                //     workingContext.Debug.DrawDebugViz( float4(0, 1, 0, 1) );

                // this is the NEE light sampler configured same as it was at previous vertex (previous vertex's "next event estimation" matches this light "event")
                LightSampler lightSampler = Bridge::CreateLightSampler( path.GetPixelPos(), misInfo.LightSamplingIsSSC );

#if RTXPT_USE_APPROXIMATE_MIS
                misWeight = lightSampler.ComputeLightVsBSDF_MIS_ForBSDF_Approx(bsdfScatterPdf, misInfo.CandidateSamples, misInfo.FullSamples);
#else
                misWeight = lightSampler.ComputeBSDFMISForEmissiveTriangle(surfaceData.neeTriangleLightIndex, bsdfScatterPdf, rayOrigin, shadingData.posW, misInfo.CandidateSamples, misInfo.FullSamples);
#endif

#if RTXPT_LIGHTING_ENABLE_BSDF_FEEDBACK
                float simpleRandom = Hash32ToFloat( Hash32Combine( Hash32Combine(Hash32(path.getVertexIndex() + 0x0367C2C7), path.id), Bridge::getSampleIndex() ) );   // note: using unique prime number salt for vertex index
                lightSampler.InsertFeedbackFromBSDF(surfaceData.neeTriangleLightIndex, Average(path.GetThp()*surfaceEmission), misWeight, simpleRandom );
#endif
            }

            surfaceEmission = lpfloat3(shadingData.emission * misWeight);
        }

#endif // !defined(RTXPT_MATERIAL_IS_EMISSIVE) || RTXPT_MATERIAL_IS_EMISSIVE

#if !defined(RTXPT_MATERIAL_IS_ANALYTIC_LIGHT_PROXY) || RTXPT_MATERIAL_IS_ANALYTIC_LIGHT_PROXY
        if (surfaceData.neeAnalyticLightIndex != 0xFFFFFFFF)
        {
            LightSampler lightSampler = Bridge::CreateLightSampler( path.GetPixelPos(), misInfo.LightSamplingIsSSC );

            bool applyMIS = misInfo.LightSamplingEnabled; // we don't want MIS if lighting was disabled - passing bsdfScatterPdf of 0 will disable it
#if RTXPT_USE_APPROXIMATE_MIS            
            bool approxMIS = true;
#else
            bool approxMIS = false;
#endif
            lightSampler.ComputeAnalyticLightProxyContributionWithMIS(surfaceEmission, surfaceData.neeAnalyticLightIndex, (applyMIS)?(path.GetBsdfScatterPdf()):(0.0), rayOrigin, rayDir, misInfo.CandidateSamples, misInfo.FullSamples, approxMIS);
        }
#endif

#if (!defined(RTXPT_MATERIAL_IS_ANALYTIC_LIGHT_PROXY) || RTXPT_MATERIAL_IS_ANALYTIC_LIGHT_PROXY) || (!defined(RTXPT_MATERIAL_IS_EMISSIVE) || RTXPT_MATERIAL_IS_EMISSIVE)
        if (any(surfaceEmission>0))
        {
#if RTXPT_FIREFLY_FILTER && PATH_TRACER_MODE!=PATH_TRACER_MODE_BUILD_STABLE_PLANES
            lpfloat baseFFThreshold = (lpfloat)workingContext.PtConsts.fireflyFilterThreshold;
            if( baseFFThreshold != 0 )
                surfaceEmission = FireflyFilter(surfaceEmission, baseFFThreshold, path.GetFireflyFilterK());
#endif

            if (any(surfaceEmission>0))
            {
#if RTXPT_DISCARD_NON_NEE_LIGHTING
                surfaceEmission *= 1e-15f;
#endif

                float3 radiance = path.GetThp() * surfaceEmission;
                float specRadianceAvg = path.hasFlag(PathFlags::stablePlaneBaseScatterDiff)?(0.0):(Average(radiance));
                AccumulatePathRadiance( workingContext, path, radiance, specRadianceAvg, path.hasFlag(PathFlags::stablePlaneOnBranch), ShouldCollectGISecondaryRadiance(path) );

                //if (path.getVertexIndex()==2)
                //workingContext.Debug.DrawDebugViz( path.GetPixelPos(), saturate(float4( 0, path.sceneLengthFromDenoisingLayer*1, 0, 1 )) );
            }
        }
#endif // #if (!defined(RTXPT_MATERIAL_IS_ANALYTIC_LIGHT_PROXY) || RTXPT_MATERIAL_IS_ANALYTIC_LIGHT_PROXY) && (!defined(RTXPT_MATERIAL_IS_EMISSIVE) || RTXPT_MATERIAL_IS_EMISSIVE)

        // Terminate after scatter ray on last vertex has been processed. Also terminates here if StablePlanesHandleHit terminated path. Also terminate based on "Russian roulette" if enabled.
        bool pathStopping = path.isTerminatingAtNextBounce();
        
#if PATH_TRACER_MODE!=PATH_TRACER_MODE_REFERENCE
        // this needs to happen before updating throughput
        StablePlanesHandleHit(path, rayOrigin, rayDir, rayTCurrent, workingContext, surfaceData, volumeAbsorption, surfaceEmission, pathStopping);
#else
        // in case of reference path tracer, dump guide buffers just in case ever needed; these are dumped every frame and are not accumulated so they'll jitter
        Bridge::ExportSurface(path, surfaceData, path.GetSceneLength(), float3(0,0,0) );
#endif

        if (pathStopping)
        {
            path.terminate();
            return;
        }

        // this is the correct place to apply Russian Roulette correction
        UpdatePathThroughput(path, path.GetThpRuRuCorrection().xxx);

#if PATH_TRACER_MODE==PATH_TRACER_MODE_BUILD_STABLE_PLANES
        // in build mode we've consumed emission and either updated or terminated path ourselves, so we must skip the rest of the function
        return;
#endif 
        
        const SampleGeneratorVertexBase sampleGeneratorVertexBase = SampleGeneratorVertexBase::make(path.GetId(), path.getVertexIndex(), Bridge::getSampleIndex() );

        UniformSampleSequenceGenerator uniformSG = UniformSampleSequenceGenerator::make( sampleGeneratorVertexBase, SampleGeneratorEffectSeed::Base );


        const PathState preScatterPath = path;

        // Generate the next path segment!
#if PATH_TRACER_MODE!=PATH_TRACER_MODE_BUILD_STABLE_PLANES
        bool scatterValid = GenerateScatterRay(shadingData, bsdf, path, sampleGeneratorVertexBase, workingContext);
#else
        bool scatterValid = false;
#endif
       
        // Compute NextEventEstimation a.k.a. direct light sampling!
#if NON_PATH_TRACING_PASS || PATH_TRACER_MODE==PATH_TRACER_MODE_BUILD_STABLE_PLANES || !PT_NEE_ENABLED
        NEEResult neeResult = NEEResult::empty();
#else
        NEEResult neeResult = HandleNEE(preScatterPath, shadingData, bsdf, uniformSG, workingContext); 
#endif

#if PATH_TRACER_MODE!=PATH_TRACER_MODE_BUILD_STABLE_PLANES        
        path.SetPackedMISInfo_ThpRuRuCorrection( neeResult.BSDFMISInfo.Pack16bit(), path.GetThpRuRuCorrection() );

        float4 neeRadianceAndSpecAvg = neeResult.GetRadianceAndSpecAvg();
        if ( any(neeRadianceAndSpecAvg>0) )
        {
#if RTXPT_DISCARD_NEE_LIGHTING
            neeRadianceAndSpecAvg *= 1e-15f;
#endif

            const int bouncesFromStablePlane = preScatterPath.getCounter(PackedCounters::BouncesFromStablePlane)+1;

            // NOTE: neeResult.RadianceAndSpecAvg has preScatterPath.thp baked in already
            float3 radiance = neeRadianceAndSpecAvg.rgb;
            float specRadianceAvg = 0;
            if (!preScatterPath.hasFlag(PathFlags::stablePlaneBaseScatterDiff))  // if stable plane scatter was not diffuse, it was specular - otherwise leave specRadianceAvg == 0
            {
                // if this is first bounce, some radiance could be specular, some not and that's fine; if this is more than 1 bounce after then it's as it was on the 1st bounce (so, all specular)
                bool pathIsDeltaOnlyPath = preScatterPath.isDeltaOnlyPath();
                bool specialCondition = (bouncesFromStablePlane==1) || (pathIsDeltaOnlyPath && bouncesFromStablePlane <= 3);
                specRadianceAvg = (specialCondition)?(neeRadianceAndSpecAvg.w):(Average(neeRadianceAndSpecAvg.rgb));
            }

            AccumulatePathRadiance( workingContext, path, radiance, specRadianceAvg, false, ShouldCollectGISecondaryRadiance(preScatterPath) );
        }
#endif // PATH_TRACER_MODE!=PATH_TRACER_MODE_BUILD_STABLE_PLANES
        
        if (!scatterValid)
        {   // this is very suboptimal from performance perspective, and should only happen very, very rarely (as in, almost never)
            path.terminate();
        }

        // adding +1 here because even though we've set up the path for the next bounce, we haven't actually raytraced it and done UpdatePathTravelled, which increments the vertex index
        bool shouldTerminate = HasFinishedSurfaceBounces(path.getVertexIndex()+1, path.getCounter(PackedCounters::DiffuseBounces));

#if PATH_TRACER_MODE!=PATH_TRACER_MODE_BUILD_STABLE_PLANES // in build mode we've consumed emission and either updated or terminated path ourselves
        shouldTerminate |= HandleRussianRoulette(path, uniformSG, workingContext);    // note: this will update path.thp!
#endif
        if (shouldTerminate)
            path.setTerminateAtNextBounce();
    }
};

#endif // __PATH_TRACER_HLSLI__
