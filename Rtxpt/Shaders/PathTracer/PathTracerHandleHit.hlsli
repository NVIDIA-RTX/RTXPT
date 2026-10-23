/*
* Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#ifndef __PATH_TRACER_HANDLE_HIT_HLSLI__ // using instead of "#pragma once" due to https://github.com/microsoft/DirectXShaderCompiler/issues/3943
#define __PATH_TRACER_HANDLE_HIT_HLSLI__

#include "../PathTracerSurfaceBridge.hlsli"

#include "PathTracerTypes.hlsli"

#include "Surface/Surface.hlsli"
#include "BSDFs/LobeType.hlsli"
#include "BSDFs/IBSDF.hlsli"

#include "Utils/HomogeneousVolumeSampler.hlsli"
#include "PathTracerNestedDielectrics.hlsli"

#include "PathTracerDenoisingLayersHandleHit.hlsli"

#if !NON_PATH_TRACING_PASS
#include "PathTracerNEE.hlsli"
#include "PathTracerSHaRC.hlsli"
#endif

namespace PathTracer
{
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

        bool onDominantDenoisingLayer = path.hasFlag(PathFlags::denoisingLayerOnPlane) && path.hasFlag(PathFlags::denoisingLayerOnDominantBranch);

        #if PATH_TRACER_MODE==PT_DENOISING_LAYERS_FILL && PT_USE_RESTIR_GI
        if (onDominantDenoisingLayer && !path.hasFlag(PathFlags::restirGIStarted) && bs.pdf != 0 )
        {
            // if 
            //  a.) >previous< state was that we were on the denoising layer and it's a dominant branch,
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
        path.SetOrigin(shadingData.ComputeNewRayOrigin(bs.isLobe(LobeType::Reflection)));
        
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

#if PATH_TRACER_MODE==PT_DENOISING_LAYERS_FILL
        static const float          kSpecularRoughnessThresholdForHitT     = 0.35f;
        const bool isDiffuseForSpecHitT = bs.isLobe(LobeType::DiffuseReflection) || bs.isLobe(LobeType::DiffuseTransmission) || roughness > kSpecularRoughnessThresholdForHitT;
        if (onDominantDenoisingLayer && !isDiffuseForSpecHitT)
        {
            if (!shadingData.shadingFlags.isPSDBlockMotionVectorsAtSurface())   // we've given up on this for now
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
            //if (isDiffuseForSpecHitT || path.getCounter(PackedCounters::BouncesFromDenoisingLayer) > maxHitTSpecBounces) // if we've reached first diffuse surface, time to record specHitT
            //const bool isDiffuseForSpecHitT2 = bs.isLobe(LobeType::DiffuseReflection) || bs.isLobe(LobeType::DiffuseTransmission) || roughness > 0.08;
            //if (isDiffuseForSpecHitT2 || path.getCounter(PackedCounters::BouncesFromDenoisingLayer) > maxHitTSpecBounces) // if we've reached first diffuse surface, time to record specHitT
            //if (!bs.isLobe(LobeType::Delta) || path.getCounter(PackedCounters::BouncesFromDenoisingLayer) > maxHitTSpecBounces) // if we've reached first diffuse surface, time to record specHitT
            if (hasNonDeltaLobes || path.getCounter(PackedCounters::BouncesFromDenoisingLayer) > maxHitTSpecBounces) // if we've reached first diffuse surface, time to record specHitT    
            {
                Bridge::ExportSpecHitTStop( path );
                path.setFlag( PathFlags::exportSpecHitTQueued, false );
            }
        }
#endif

#if PATH_TRACER_MODE!=PT_DENOISING_LAYERS_BUILD
#if RTXPT_FIREFLY_FILTER 
        // if bouncePDF then it's a delta event - expansion angle is 0
        float fireflyFilterK = ComputeNewScatterFireflyFilterK(path.GetFireflyFilterK(), bs.pdf, bs.lobeP);
#else
        float fireflyFilterK = 0;
#endif
        path.SetFireflyFilterK_BsdfScatterPdf( fireflyFilterK, bs.pdf );
#endif


#if PATH_TRACER_MODE==PT_DENOISING_LAYERS_FILL
        DenoisingLayersOnScatter(path, bs, workingContext);
#endif

        // if (path.rayCone.getWidth() / path.GetSceneLength() > 0.0005) <- logic was, this is a measure of coherence; however it doesn't help.
            path.setFlag( PathFlags::enableThreadReorder, true );

        // Mark the path as valid only if it has a non-zero throughput. 
        return true; // any(path.thp > 0.f); <- This is an optimization. It's unnecessary for correctness, and there's little benefit in practice as RR will kill the path.
    }

    /** Generates a new scatter ray using BSDF importance sampling.
        \param[in] sd Shading data.
        \param[in] bsdf BSDF at the shading point.
        \param[in,out] path The path state.
        \return True if a ray was generated, false otherwise.
    */
    inline bool GenerateScatterRay(const ShadingData shadingData, const ActiveBSDF bsdf, inout PathState path, const SampleGeneratorVertexBase sgBase, const WorkingContext workingContext)
    {
        // Note - one could adjust shading normal only for scatter - it kinda works but breaks symmetry between light sampling and BxDF
        // ShadingData shadingData = _shadingData;
        // shadingData.AdjustShadingNormal( float4(_shadingData.T, dot(_shadingData.T, _shadingData.B)>0?-1.0:1.0), true, false );

        // only ActiveBSDF::cRandomNumberCountForSampling are actually used; this is the best formula for compiler optimizations; sending SampleGenerator as an argument to bsdf.sample
        // keeps too much state alive and adding explicit two codepaths instead of using generic
        float4 preGeneratedSamples;
        {
    #if RTXPT_ENABLE_LOW_DISCREPANCY_SAMPLER_FOR_BSDF
            [branch] if( path.getCounter(PackedCounters::DiffuseBounces)<DisableLowDiscrepancySamplingAfterDiffuseBounceCount )
                preGeneratedSamples = SampleGenerator::Generate( ActiveBSDF::cRandomNumberCountForSampling, sgBase, SampleGeneratorEffectSeed::ScatterBSDF );
            else
    #endif // RTXPT_ENABLE_LOW_DISCREPANCY_SAMPLER_FOR_BSDF
                preGeneratedSamples = UniformSampleSequenceGenerator::Generate( ActiveBSDF::cRandomNumberCountForSampling, sgBase, SampleGeneratorEffectSeed::ScatterBSDF );
        }

        BSDFSample result;
        bool valid = bsdf.sample(shadingData, preGeneratedSamples, result, kUseBSDFSampling);

        if (valid)
            return GenerateScatterRay(result, shadingData, bsdf, path, workingContext);
        else
            return false;
    }

    inline void HandleHit(inout PathState path, const float3 rayOrigin, const float3 rayDir, const float rayTCurrent, float2 barycentrics, const WorkingContext workingContext)
    {
        UpdatePathTravelled(path, rayOrigin, rayDir, rayTCurrent, workingContext);
        
#if ENABLE_DEBUG_VIZUALISATIONS
        const bool debugPath = workingContext.Debug.IsDebugPixel(path.GetPixelPos());
#else
        const bool debugPath = false;
#endif

        Surface surface = Bridge::loadSurface( InstanceIndex(), GeometryIndex(), PrimitiveIndex(), barycentrics, rayDir, path.rayCone, path.getVertexIndex(), path.GetPixelPos(), workingContext.Debug);

        // if (surface.shadingData.shadingFlags.isPSDBlockMotionVectorsAtSurface())   // we've given up on this for now
        //     path.setFlag(PathFlags::exportSpecHitTBlocked, true);

#if PATH_TRACER_MODE==PT_DENOISING_LAYERS_FILL && PT_USE_RESTIR_GI
        if (path.hasFlag(PathFlags::restirGICollectSecondarySurface))
        {
            ReSTIRGI_StoreSecondarySurfacePositionAndNormal(path.GetPixelPos(), surface.shadingData.posW, surface.shadingData.N);
            path.setFlag(PathFlags::restirGICollectSecondarySurface, false);
        }
#endif

#if PATH_TRACER_MODE==PT_DENOISING_LAYERS_FILL
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

        // Covers the volume transmittance above; before the false-hit early-out below so it is never skipped.
        SharcThroughputCheckpoint(path); 

        // Reject false hits in nested dielectrics but also updates 'outside index of refraction' and dependent data
        bool rejectedFalseHit = !HandleNestedDielectrics(surface, path, workingContext);

#if PATH_TRACER_MODE==PT_DENOISING_LAYERS_BUILD && ENABLE_DEBUG_DELTA_TREE_VIZUALISATION
        if (path.hasFlag(PathFlags::deltaTreeExplorer))
        {
            DeltaTreeVizHandleHit(path, rayOrigin, rayDir, rayTCurrent, surface, rejectedFalseHit, HasFinishedSurfaceBounces(path.getVertexIndex(), path.getCounter(PackedCounters::DiffuseBounces)), volumeAbsorption, workingContext);
            return;
        }
#endif
        if (rejectedFalseHit)
        {
#if PATH_TRACER_MODE==PT_DENOISING_LAYERS_BUILD && !NESTED_DIELECTRICS_AVOID_TERMINATION
            if (path.isTerminated()) // HandleNestedDielectrics can terminate the ray if it ends up in a crazy loop - this needs proper handling to avoid holes in data
                DenoisingLayersHandleMiss(path, 0, rayOrigin, rayDir, rayTCurrent, workingContext);
#endif
            return;
        }

        // These will not change anymore, so make const shortcuts
        const ShadingData shadingData    = surface.shadingData;
        // bsdf is left non-const so SharcPrepareHit below can clamp its roughness on the SHaRC update pass
        ActiveBSDF bsdf   = surface.bsdf;

#if ENABLE_DEBUG_VIZUALISATIONS && ENABLE_DEBUG_LINES_VIZ && PATH_TRACER_MODE!=PT_DENOISING_LAYERS_BUILD
        if (debugPath)
        {
            // IoR debugging - .x - "outside", .y - "interior", .z - frontFacing, .w - "eta" (eta is isFrontFace?outsideIoR/insideIoR:insideIoR/outsideIoR)
            // workingContext.Debug.Print(path.getVertexIndex(), float4(shadingData.exteriorIoR, surface.shadingData.interiorIoR, shadingData.frontFacing, bsdf.data.eta) );

            // draw tangent space
            workingContext.Debug.DrawLine(shadingData.posW, shadingData.posW + shadingData.T() * workingContext.Debug.LineScale(), float4(0.7, 0, 0, 0.4), float4(1.0, 0, 0, 0.4));
            workingContext.Debug.DrawLine(shadingData.posW, shadingData.posW + shadingData.B() * workingContext.Debug.LineScale(), float4(0, 0.7, 0, 0.4), float4(0, 1.0, 0, 0.4));
            workingContext.Debug.DrawLine(shadingData.posW, shadingData.posW + shadingData.N * workingContext.Debug.LineScale(), float4(0, 0, 0.7, 0.4), float4(0, 0, 1.0, 0.4));

            // draw ray cone footprint
            float coneWidth = path.rayCone.getWidth();
            workingContext.Debug.DrawLine(shadingData.posW + (-shadingData.T()+shadingData.B()) * coneWidth, shadingData.posW + (+shadingData.T()+shadingData.B()) * coneWidth, float4(0.5, 0.0, 1.0, 0.3), float4(0.5, 1.0, 0.0, 0.3) );
            workingContext.Debug.DrawLine(shadingData.posW + (+shadingData.T()+shadingData.B()) * coneWidth, shadingData.posW + (+shadingData.T()-shadingData.B()) * coneWidth, float4(0.5, 0.0, 1.0, 0.3), float4(0.5, 1.0, 0.0, 0.3) );
            workingContext.Debug.DrawLine(shadingData.posW + (+shadingData.T()-shadingData.B()) * coneWidth, shadingData.posW + (-shadingData.T()-shadingData.B()) * coneWidth, float4(0.5, 0.0, 1.0, 0.3), float4(0.5, 1.0, 0.0, 0.3) );
            workingContext.Debug.DrawLine(shadingData.posW + (-shadingData.T()-shadingData.B()) * coneWidth, shadingData.posW + (-shadingData.T()+shadingData.B()) * coneWidth, float4(0.5, 0.0, 1.0, 0.3), float4(0.5, 1.0, 0.0, 0.3) );
        }
#endif

        // Collect emissive triangle radiance.
        lpfloat3 surfaceEmission = 0.0;
        NEEBSDFMISInfo misInfo = NEEBSDFMISInfo::Unpack16bit(path.GetPackedMISInfo());


        // should we ignore this triangle's emission, and if not, is there any emission
        if ( !(misInfo.GetSkipEmissiveBRDF() && !path.wasScatterTransmission()) )
        {
            float3 emission = shadingData.GetEmission();
            if( any(emission)>0 )
            {
                // figure out MIS vs our lighting technique, if any
                float misWeight = 1.0f;
#if PATH_TRACER_MODE!=PT_DENOISING_LAYERS_BUILD // NO MIS in PT_DENOISING_LAYERS_BUILD
                lpfloat bsdfScatterPdf = path.GetBsdfScatterPdf();
                if( misInfo.LightSamplingEnabled && bsdfScatterPdf != 0 )
                {
                    // if(surface.neeTriangleLightIndex == 0xFFFFFFFF)                     // this is really bad if happens
                    //     workingContext.Debug.DrawDebugViz( float4(0, 1, 0, 1) );

                    // this is the NEE light sampler configured same as it was at previous vertex (previous vertex's "next event estimation" matches this light "event")
                    LightSampler lightSampler = Bridge::CreateLightSampler( path.GetPixelPos(), misInfo.LightSamplingIsSSC );

#if RTXPT_USE_APPROXIMATE_MIS
                    misWeight = lightSampler.ComputeLightVsBSDF_MIS_ForBSDF_Approx(bsdfScatterPdf, misInfo.CandidateSamples, misInfo.FullSamples);
#else
                    misWeight = lightSampler.ComputeBSDFMISForEmissiveTriangle(surface.neeTriangleLightIndex, bsdfScatterPdf, rayOrigin, shadingData.posW, misInfo.CandidateSamples, misInfo.FullSamples);
#endif

#if RTXPT_LIGHTING_ENABLE_BSDF_FEEDBACK
                    float simpleRandom = Hash32ToFloat( Hash32Combine( Hash32Combine(Hash32(path.getVertexIndex() + 0x0367C2C7), path.GetId()), Bridge::getSampleIndex() ) );   // note: using unique prime number salt for vertex index
                    lightSampler.InsertFeedbackFromBSDF(surface.neeTriangleLightIndex, Luminance(path.GetThp()*emission), misWeight, simpleRandom );    // surfaceEmission isn't set yet
#endif
                }
#endif // #if PATH_TRACER_MODE!=PT_DENOISING_LAYERS_BUILD

                surfaceEmission = ToLpRadiance(emission * misWeight);
            }
        }

        if (surface.neeAnalyticLightIndex != 0xFFFFFFFF)
        {
            LightSampler lightSampler = Bridge::CreateLightSampler( path.GetPixelPos(), misInfo.LightSamplingIsSSC );

            bool applyMIS = misInfo.LightSamplingEnabled; // we don't want MIS if lighting was disabled - passing bsdfScatterPdf of 0 will disable it
            bool approxMIS = RTXPT_USE_APPROXIMATE_MIS?true:false;
            lightSampler.ComputeAnalyticLightProxyContributionWithMIS(surfaceEmission, surface.neeAnalyticLightIndex, (applyMIS)?(path.GetBsdfScatterPdf()):(0.0), rayOrigin, rayDir, misInfo.CandidateSamples, misInfo.FullSamples, approxMIS);
        }

        // Time to apply firefly filter!
        if (any(surfaceEmission>0))
        {
#if RTXPT_FIREFLY_FILTER && PATH_TRACER_MODE!=PT_DENOISING_LAYERS_BUILD
            float baseFFThreshold = workingContext.PtConsts.fireflyFilterThreshold;
            if( baseFFThreshold != 0 )
                surfaceEmission = (lpfloat3)FireflyFilter(surfaceEmission, baseFFThreshold, path.GetFireflyFilterK());
#endif
        }

        // SHaRC: build this vertex's cache key/payload (this also clamps bsdf roughness on the update pass), then
        // look the cache up.
        SharcPathVertex sharcVertex = SharcPrepareHit(shadingData, bsdf, surfaceEmission);

        // Set when SHaRC supplied radiance for this vertex. The cached value already includes this vertex's
        // emission, so the local emission accumulation below must be skipped or it would be double-counted.
        bool hasSharcCachedRadiance = false;
        {
            float3 sharcRadiance; bool sharcWroteDirectly;
            if (SharcQueryHit(sharcVertex, shadingData, bsdf, rayTCurrent, path, sharcRadiance, sharcWroteDirectly))
            {
                if (!sharcWroteDirectly)    // false for the debug visualizations, which write the pixel themselves
                {
                    // sharcRadiance is all radiance from this vertex onwards, so weight it by the path throughput.
                    float3 radiance = path.GetThp() * sharcRadiance;
                    float specRadianceAvg = path.hasFlag(PathFlags::denoisingLayerBaseScatterDiff) ? (0.0) : (Average(radiance));
                    AccumulatePathRadiance(workingContext, path, radiance, specRadianceAvg, path.hasFlag(PathFlags::denoisingLayerOnBranch), ShouldCollectGISecondaryRadiance(path));
                }
                // terminate at the *next* bounce, not immediately, so DenoisingLayersHandleHit still runs
                path.setTerminateAtNextBounce();
                hasSharcCachedRadiance = true;
            }
        }

        if (!hasSharcCachedRadiance && any(surfaceEmission>0))
        {
            surfaceEmission = clamp( surfaceEmission, (lpfloat3)0, (lpfloat3)HLF_MAX );

            float3 radiance = path.GetThp() * surfaceEmission;
#if RTXPT_DISCARD_NON_NEE_LIGHTING
            // Scales the accumulated radiance, NOT surfaceEmission - SharcDepositHit below still needs the real value.
            radiance *= 0.0000000596;   // not 0, so the compiler doesn't compile everything out
#endif
            float specRadianceAvg = path.hasFlag(PathFlags::denoisingLayerBaseScatterDiff)?(0.0):(Average(radiance));
            AccumulatePathRadiance( workingContext, path, radiance, specRadianceAvg, path.hasFlag(PathFlags::denoisingLayerOnBranch), ShouldCollectGISecondaryRadiance(path) );

            //if (path.getVertexIndex()==2)
            //workingContext.Debug.DrawDebugViz( path.GetPixelPos(), saturate(float4( 0, path.sceneLengthFromDenoisingLayer*1, 0, 1 )) );
        }

        // Terminate after scatter ray on last vertex has been processed. Also terminates here if DenoisingLayersHandleHit terminated path. Also terminate based on "Russian roulette" if enabled.
        bool pathStopping = path.isTerminatingAtNextBounce();
        
#if PATH_TRACER_MODE!=PT_REFERENCE
        // this needs to happen before updating throughput
        DenoisingLayersHandleHit(path, rayOrigin, rayDir, rayTCurrent, workingContext, surface, volumeAbsorption, surfaceEmission, pathStopping);
#else
        // in case of reference path tracer, dump guide buffers just in case ever needed; these are dumped every frame and are not accumulated so they'll jitter
        Bridge::ExportSurface(path, surface, path.GetSceneLength(), float3(0,0,0) );
#endif

        // Clean truncation: the emission collected above is the BSDF half of the MIS pair whose NEE half ran at the previous 
        // vertex, so every transport length we cover stays fully paired. If we ended after NEE, NEE would be one length further 
        // than requested by "bounce count" settings plus with no BSDF counterpart and MIS down-weight it'd be biased.
        if (pathStopping)
        {
            SharcHandlePathLimit(surfaceEmission);  // we return before SharcDepositHit, so push it back from here
            path.terminate();
            return;
        }

        // this is the correct place to apply Russian Roulette correction; no SHaRC counterpart needed - RR is
        // compiled out of the training pass, so the factor is always 1 there (see the #error in PathTracerSHaRC.hlsli)
        UpdatePathThroughput(path, path.GetThpRuRuCorrection().xxx);

#if PATH_TRACER_MODE==PT_DENOISING_LAYERS_BUILD
        // in build mode we've consumed emission and either updated or terminated path ourselves, so we must skip the rest of the function
        return;
#endif
        
        const SampleGeneratorVertexBase sampleGeneratorVertexBase = SampleGeneratorVertexBase::make(path.GetId(), path.getVertexIndex(), Bridge::getSampleIndex() );

        UniformSampleSequenceGenerator uniformSG = UniformSampleSequenceGenerator::make( sampleGeneratorVertexBase, SampleGeneratorEffectSeed::Base );


        const PathState preScatterPath = SharcAdjustPreScatterPath(path);   // identity unless this is the SHaRC update pass

        // Generate the next path segment!
#if PATH_TRACER_MODE!=PT_DENOISING_LAYERS_BUILD
        bool scatterValid = GenerateScatterRay(shadingData, bsdf, path, sampleGeneratorVertexBase, workingContext);
#else
        bool scatterValid = false;
#endif
       
        // Compute NextEventEstimation a.k.a. direct light sampling!
#if NON_PATH_TRACING_PASS || PATH_TRACER_MODE==PT_DENOISING_LAYERS_BUILD || !PT_NEE_ENABLED
        NEEResult neeResult = NEEResult::empty();
#else
        NEEResult neeResult = HandleNEE(preScatterPath, shadingData, bsdf, uniformSG, workingContext); 
#endif

#if PATH_TRACER_MODE!=PT_DENOISING_LAYERS_BUILD        
        path.SetPackedMISInfo_ThpRuRuCorrection( neeResult.BSDFMISInfo.Pack16bit(), path.GetThpRuRuCorrection() );

        float4 neeRadianceAndSpecAvg = neeResult.GetRadianceAndSpecAvg();
        if ( any(neeRadianceAndSpecAvg>0) )
        {
#if RTXPT_DISCARD_NEE_LIGHTING
            // Local copy only - SharcDepositHit below re-reads neeResult, so the cache keeps the real NEE radiance.
            neeRadianceAndSpecAvg *= 1e-15f;
#endif

            const int bouncesFromDenoisingLayer = preScatterPath.getCounter(PackedCounters::BouncesFromDenoisingLayer)+1;

            // NOTE: neeResult.RadianceAndSpecAvg has preScatterPath.thp baked in already
            float3 radiance = neeRadianceAndSpecAvg.rgb;
            float specRadianceAvg = 0;
            if (!preScatterPath.hasFlag(PathFlags::denoisingLayerBaseScatterDiff))  // if denoising layer scatter was not diffuse, it was specular - otherwise leave specRadianceAvg == 0
            {
                // if this is first bounce, some radiance could be specular, some not and that's fine; if this is more than 1 bounce after then it's as it was on the 1st bounce (so, all specular)
                bool pathIsDeltaOnlyPath = preScatterPath.isDeltaOnlyPath();
                bool specialCondition = (bouncesFromDenoisingLayer==1) || (pathIsDeltaOnlyPath && bouncesFromDenoisingLayer <= 3);
                specRadianceAvg = (specialCondition)?(neeRadianceAndSpecAvg.w):(Average(neeRadianceAndSpecAvg.rgb));
            }

            AccumulatePathRadiance( workingContext, path, radiance, specRadianceAvg, false, ShouldCollectGISecondaryRadiance(preScatterPath) );
        }
#endif // PATH_TRACER_MODE!=PT_DENOISING_LAYERS_BUILD

        // Deposit this vertex into the SHaRC cache. Placed AFTER HandleNEE so the NEE-sampled direct lighting is available.
        if (!SharcDepositHit(sharcVertex, neeResult.GetRadianceAndSpecAvg().rgb, surfaceEmission, path))
        {
            path.terminate();
            return;
        }

        if (!scatterValid)
        {   // this is very suboptimal from performance perspective, and should only happen very, very rarely (as in, almost never)
            path.terminate();
        }

        // adding +1 here because even though we've set up the path for the next bounce, we haven't actually raytraced it and done UpdatePathTravelled, which increments the vertex index
        bool shouldTerminate = HasFinishedSurfaceBounces(path.getVertexIndex()+1, path.getCounter(PackedCounters::DiffuseBounces));

#if PATH_TRACER_MODE!=PT_DENOISING_LAYERS_BUILD // in build mode we've consumed emission and either updated or terminated path ourselves
        shouldTerminate |= HandleRussianRoulette(path, uniformSG, workingContext);    // note: this will update path.thp!
#endif
        if (shouldTerminate)
            path.setTerminateAtNextBounce();
    }
}

#endif // __PATH_TRACER_HANDLE_HIT_HLSLI__
