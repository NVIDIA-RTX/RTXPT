/*
* Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#ifndef __PATH_TRACER_NEE_HLSLI__ // using instead of "#pragma once" due to https://github.com/microsoft/DirectXShaderCompiler/issues/3943
#define __PATH_TRACER_NEE_HLSLI__

#include "PathTracerTypes.hlsli"

// These are needed to link PolymorphicLight::EnvironmentQuadLight, which can do importance sampling for the direction, to actual environment map for sampling
float3 EnvironmentQuadLight::ToWorld(float3 localDir)  // Transform direction from local to world space.
{
    EnvMap envMap = Bridge::CreateEnvMap();
    return envMap.ToWorld(localDir);
}
//
float3 EnvironmentQuadLight::ToLocal(float3 worldDir)  // Transform direction from world to local space.
{
    EnvMap envMap = Bridge::CreateEnvMap();
    return envMap.ToLocal(worldDir);
}
//
float3 EnvironmentQuadLight::SampleLocalSpace(float3 localDir)
{
    EnvMap envMap = Bridge::CreateEnvMap();
    return envMap.EvalLocal(localDir, Bridge::DiffuseEnvironmentMapMIPOffset());
}
//

namespace PathTracer
{
    
#if 1   // switch this off to disable entire NEE codepath!

    inline float EvalSampleWeight( const PathLightSample lightSample, const ShadingData shadingData, const ActiveBSDF bsdf )
    {
    #if 0 // more costly version, does full BSDF - not really worth it unless special case colourful materials with colourful lights
        float3 bsdfThp = bsdf.eval(shadingData, lightSample.Direction, bsdfThpDiff, bsdfThpSpec);
        float weight = max3(bsdfThp*lightSample.Li); // used to be luminance
    #else // ignores colour but cheaper; allows us to use 8 instead of 6 candidate samples and still be a tiny bit faster
        float weight = max3(lightSample.Li) * bsdf.evalPdf(shadingData, lightSample.Direction, kUseBSDFSampling);
    #endif
        return weight;
    }
    
    // Weighted Reservoir Sampling, helper for picking 1 out of 'totalCandidates' samples. See https://agraphicsguynotes.com/posts/understanding_the_math_behind_restir_di/ for simple explanation.
    // Note: totalCandidates / RemainingCandidates logic is only there as a helper, as in our case we know the number of samples we'll draw but we'll still be streaming them
    struct WRSSingleSampleHelper
    {
        PathLightSample     PickedCandidate;        // a.k.a. reservoir; should be packable to uint4
        float               ReservoirTotalWeight;
        float               PickedCandidateWeight;

        static WRSSingleSampleHelper make()
        {
            WRSSingleSampleHelper ret;
            ret.PickedCandidate.Li      = float3(0,0,0); // this makes ret.PickedCandidate.Valid()==false
            ret.ReservoirTotalWeight    = 0;
            ret.PickedCandidateWeight   = 0;
            return ret;
        }

        void InsertCandidate( float randomValue, PathLightSample candidateSample, float candidateWeight )
        {
            // Perform Weighted Reservoir Sampling
            ReservoirTotalWeight += candidateWeight;
            float wrsThreshold = saturate(candidateWeight / ReservoirTotalWeight);
            if( randomValue < wrsThreshold )
            {
                PickedCandidate = candidateSample;  // TODO: pack here.
                PickedCandidateWeight = candidateWeight;
            }
        }

        PathLightSample GetResult(uint totalCandidates)
        {
            float pickedCandidatePdf = PickedCandidateWeight / ReservoirTotalWeight;   // adding candidate multiplier here to avoid multiplying each time
            pickedCandidatePdf *= (float)totalCandidates; // this is because we've skipped adding this term when drawing individual candidates
            PickedCandidate.Li  /= pickedCandidatePdf;
            // PickedCandidate.SelectionPdf *= pickedCandidatePdf; // correct pdf, even though already applied to .Li, is useful later
            return PickedCandidate;
        }
    };

    // Generates a light sample from all the available lights.
    inline PathLightSample GenerateLightSample(const WorkingContext workingContext, const ShadingData shadingData, const ActiveBSDF bsdf, const uint wrsCandidateSampleCount, inout UniformSampleSequenceGenerator sampleGeneratorLights, inout UniformSampleSequenceGenerator sampleGeneratorWRS, const LightSampler lightSampler, const bool sampleIsLocal)
    {
        // NvReorderThread(0, 32);
        WRSSingleSampleHelper wrs = WRSSingleSampleHelper::make();
        
        for (uint i = 0; i < wrsCandidateSampleCount; i++ )
        {
            uint lightIndex = 0; float selectionPdf = 0;

            float rnd = sampleNext1D(sampleGeneratorLights);
            if( sampleIsLocal )
                lightIndex = lightSampler.SampleLocal( rnd, selectionPdf );
            else
                lightIndex = lightSampler.SampleGlobal( rnd, selectionPdf );

            const PolymorphicLightInfoFull packedLightInfo = lightSampler.LoadLight(lightIndex);

            // TODO: for LD sampling try the "reuse/recycle" trick
            const float2 interiorSampleRnd = sampleNext2D(sampleGeneratorLights);

            // NvReorderThread(PolymorphicLight::DecodeType(packedLightInfo), 32);
            PolymorphicLightSample lightSample = PolymorphicLight::CalcSample( packedLightInfo, interiorSampleRnd, shadingData.posW );

			// an example on printf-debugging specific light type
            // if( PolymorphicLight::DecodeType(packedLightInfo) == PolymorphicLightType::kEnvironmentQuad )
            // {
            //     EnvironmentQuadLight envLight = EnvironmentQuadLight::Create(packedLightInfo);
            //     DebugPrint("", envLight.NodeDim, envLight.NodeX, envLight.NodeY );
            //     DebugPrint("", lightSample.Position, lightSample.Radiance, lightSample.SolidAnglePdf);
            // }
       
            // Setup generic path light sample.
            PathLightSample candidateSample;
            /*candidateSample.Pdf =*/ const float pdf = lightSample.SolidAnglePdf * selectionPdf;
            candidateSample.Li = pdf > 0.f ? (lightSample.Radiance / pdf) : float3(0,0,0);
            candidateSample.SolidAnglePdf = lightSample.SolidAnglePdf;
            float3 surfToLight = lightSample.Position-shadingData.posW;
            candidateSample.Distance = length(surfToLight);
            candidateSample.Direction = surfToLight / max( candidateSample.Distance, 1e-7 );
            candidateSample.LightIndex = lightIndex;
            candidateSample.SelectionPdf = selectionPdf;
            candidateSample.LightSampleableByBSDF = lightSample.LightSampleableByBSDF;
            // if( workingContext.Debug.IsDebugPixel() )
            //     workingContext.Debug.DrawLine(shadingData.posW, lightSample.Position, float3(1,0,0), float3(0,1,0) );

            // Perform Weighted Reservoir Sampling
            wrs.InsertCandidate( sampleNext1D(sampleGeneratorWRS), candidateSample, EvalSampleWeight( candidateSample, shadingData, bsdf ) );

            sampleGeneratorLights.AdvanceSampleIndex(); // only needed for LD sampling, resets the dimension; NO-OP for uniform random 
        }
        return wrs.GetResult(wrsCandidateSampleCount);
    }

    // This will ray cast and, if light visible, accumulate radiance properly, including doing weighted sum for 
    void ProcessLightSample(inout NEEResult accum, inout float specSum, PathLightSample lightSample, bool sampleIsLocal, uint localNEESamples, uint totalSamples,
                                const ShadingData shadingData, const ActiveBSDF bsdf, const PathState preScatterPath, LightSampler lightSampler,
                                inout UniformSampleSequenceGenerator sampleGeneratorFeedback, const WorkingContext workingContext)
    {
        RayDesc ray;
        bool visible = false;

        /*[branch]*/ if (lightSample.Valid())   // if sample's bad, skip; we tried casting the ray anyway but ignoring the results - didn't yield better perf
        {
            ray = lightSample.ComputeVisibilityRay(shadingData).toRayDesc();
            visible = Bridge::traceVisibilityRay(ray, preScatterPath.rayCone, preScatterPath.getVertexIndex(), workingContext.Debug);
        }

        // if( workingContext.Debug.IsDebugPixel() )
        //     DebugLine( shadingData.posW, shadingData.posW+lightSample.Direction*lightSample.Distance, float4(!visible,visible,0,1.0) );

#if 0 && SER_USE_SORTING
#if USE_NVAPI_REORDER_THREADS
        NvReorderThread(visible?(1):(0), 16);
#elif USE_DX_MAYBE_REORDER_THREADS
        dx::MaybeReorderThread(visible?(1):(0), 16);
#endif
#endif
        /*[branch]*/ if (visible)
        {
            // add compute grazing angle fadeout
            float fadeOut = (shadingData.shadowNoLFadeout>0)?(ComputeLowGrazingAngleFalloff( ray.Direction, shadingData.vertexN, shadingData.shadowNoLFadeout, 2.0 * shadingData.shadowNoLFadeout )):(1.0);

            // local (tile) vs global vs scatter (BSDF) multiple importance sampling
            float scatterPdfForDir = bsdf.evalPdf(shadingData, lightSample.Direction, kUseBSDFSampling);
            float misWeight = lightSampler.ComputeInternalMIS(shadingData.posW, lightSample, sampleIsLocal, localNEESamples, totalSamples, scatterPdfForDir);

            // apply MIS and other multipliers to light here - reduces register pressure and computation later
            lightSample.Li *= fadeOut * misWeight;

            // compute BSDF throughput!
            float4 bsdfThp = bsdf.eval(shadingData, lightSample.Direction);

            // compute radiance with MIS and other modifiers
            float3 radiance = bsdfThp.rgb * lightSample.Li;
            float radianceAvg = Average(radiance);
            float specAvg = bsdfThp.w * Average(lightSample.Li);

#if RTXPT_FIREFLY_FILTER   // firefly filter has cost - only enable if denoiser REALLY requires it
            if( workingContext.PtConsts.fireflyFilterThreshold != 0 )
            {
                const float pdf = lightSample.SelectionPdf * lightSample.SolidAnglePdf;
                float neeFireflyFilterK = ComputeNewScatterFireflyFilterK(preScatterPath.fireflyFilterK, pdf, 1.0);
                const float ffDampening = FireflyFilterShort(radianceAvg, workingContext.PtConsts.fireflyFilterThreshold, neeFireflyFilterK);
                radiance *= ffDampening;
                // radianceAvg *= ffDampening; // radianceAvg used for NEE weight later - testing suggests it's same or better if not firefly-filtered
            }
#endif

            // apply path throughput here
            float preScatterPathThpAvg = Average(preScatterPath.thp);
            radiance *= preScatterPath.thp;
            specAvg *= preScatterPathThpAvg;
            radianceAvg *= preScatterPathThpAvg;

            // accumulate radiances
            accum.AccumulateRadiance( radiance, specAvg );

            // for computing weighted average sample distance
            accum.SpecRadianceSourceDistance = accum.SpecRadianceSourceDistance + lightSample.Distance * specAvg;
            specSum += specAvg;
            
            // sample feedback for NEE-AT if needed!
            if( lightSample.LightIndex != RTXPT_INVALID_LIGHT_INDEX && lightSampler.IsTemporalFeedbackRequired() )
            {
                // compute light weigh as in "how much we want this light" - so include path throughput, BSDF and light; 
                float feedbackWeight = radianceAvg;

                float randomValues[RTXPT_LIGHTING_FEEDBACK_CANDIDATES_PER_PATH];
                for (int i = 0; i < RTXPT_LIGHTING_FEEDBACK_CANDIDATES_PER_PATH; i++)
                    randomValues[i] = sampleNext1D(sampleGeneratorFeedback);

                lightSampler.InsertFeedbackFromNEE(lightSample.LightIndex, feedbackWeight, randomValues);
            }
        }
    }
    
    void FinalizeLightSample( inout NEEResult accum, const float luminanceSum )
    {
        accum.SpecRadianceSourceDistance = accum.SpecRadianceSourceDistance / (luminanceSum + 1e-12);
    }
    
    // 'result' argument is expected to have been initialized to 'NEEResult::empty()'
    inline void HandleNEE_MultipleSamples(inout NEEResult inoutResult, const PathState preScatterPath, const ShadingData shadingData, const ActiveBSDF bsdf, 
                                            const SampleGeneratorVertexBase sgBase, const WorkingContext workingContext, const int sampleCountBoost)
    {
        // NvReorderThread(0, 32);

        LightSampler lightSampler = Bridge::CreateLightSampler( preScatterPath.GetPixelPos(), preScatterPath.rayCone.getWidth(), preScatterPath.sceneLength );

        // There's a cost to having these as a dynamic constant so an option for production code is to hard code
        const uint totalSamples = min(RTXPT_LIGHTING_MAX_TOTAL_SAMPLE_COUNT, (!lightSampler.IsEmpty()) ? (sampleCountBoost + workingContext.PtConsts.NEEFullSamples)     : (0));
        if (totalSamples == 0)
            return;

        UniformSampleSequenceGenerator sampleGenerator         = UniformSampleSequenceGenerator::make( sgBase, SampleGeneratorEffectSeed::NextEventEstimation );
        UniformSampleSequenceGenerator sampleGeneratorFeedback = UniformSampleSequenceGenerator::make( sgBase, SampleGeneratorEffectSeed::NextEventEstimationFeedback );
        UniformSampleSequenceGenerator sampleGeneratorLightSampler = UniformSampleSequenceGenerator::make( sgBase, SampleGeneratorEffectSeed::NextEventEstimationLightSampler );

        // in theory, using quasi-random sampling should help with picking light candidates; in practice it doesn't seem to help enough to justify the cost - even when we need to include picking sample on the light as well (see GenerateLightSample)
        // this code used to work for LD sampling in the past, leaving as a reference - you probably want to use the same stream for global and local samples this time, will make it easier
        // sampleGeneratorLightSampler = SampleGenerator::make( sgBase, SampleGeneratorEffectSeed::NextEventEstimationLightSamplerG, useLowDiscrepancyGen, globalNEESamples * workingContext.PtConsts.NEECandidateSamples );

        const uint localNEESamples  = lightSampler.ComputeLocalSampleCount(totalSamples);
        const uint globalNEESamples = totalSamples - localNEESamples;

        inoutResult.BSDFMISInfo.LightSamplingEnabled    = true;
        inoutResult.BSDFMISInfo.LightSamplingIsIndirect = lightSampler.IsIndirect;
        inoutResult.BSDFMISInfo.LocalNEESamples         = localNEESamples;
        inoutResult.BSDFMISInfo.TotalSamples            = totalSamples;

        // we must initialize to 0 since we're accumulating multiple samples
        inoutResult.SpecRadianceSourceDistance = 0;
        
        float specSum = 0.0; // for sample distance weighted average 
        
        int globalNEESamplesRemaining = globalNEESamples;
        bool sampleIsLocal = false;
        for (uint sampleIndex = 0; sampleIndex < totalSamples; sampleIndex++)
        {
            if (globalNEESamplesRemaining==0)
                sampleIsLocal = true;
            globalNEESamplesRemaining--;

#if PT_NEE_ANTI_LAG_PASS && PATH_TRACER_MODE==PATH_TRACER_MODE_BUILD_STABLE_PLANES // when anti-lag is enabled, in general do few more candidates as we know that we need to search for lights more
            const uint candidateSampleCount = PT_NEE_CANDIDATE_SAMPLES; //workingContext.PtConsts.NEECandidateSamples*2;
#else
            const uint candidateSampleCount = PT_NEE_CANDIDATE_SAMPLES; //workingContext.PtConsts.NEECandidateSamples;
#endif
            PathLightSample lightSample = GenerateLightSample(workingContext, shadingData, bsdf, candidateSampleCount, sampleGeneratorLightSampler, sampleGenerator, lightSampler, sampleIsLocal);

            // this computes the BSDF throughput and (if throughput>0) then casts shadow ray and handles radiance summing up & weighted averaging for 'sample distance' used by denoiser
            ProcessLightSample(inoutResult, specSum, lightSample, sampleIsLocal, localNEESamples, totalSamples, shadingData, bsdf, preScatterPath, lightSampler, sampleGeneratorFeedback, workingContext);
        }

        FinalizeLightSample(inoutResult, specSum);
    }
    
    inline NEEResult HandleNEE(const uniform OptimizationHints optimizationHints, const PathState preScatterPath,
                                    const ShadingData shadingData, const ActiveBSDF bsdf, const SampleGeneratorVertexBase sgBase, const WorkingContext workingContext)
    {
        // Determine if BSDF has non-delta lobes.
        const uint lobes = bsdf.getLobes(shadingData);
        const bool hasNonDeltaLobes = ((lobes & (uint) LobeType::NonDelta) != 0) && (!optimizationHints.OnlyDeltaLobes);

        const bool onDominantBranch = preScatterPath.hasFlag(PathFlags::stablePlaneOnDominantBranch);
        const bool onStablePlane = preScatterPath.hasFlag(PathFlags::stablePlaneOnPlane);

        // Check if we should apply NEE.
        const bool applyNEE = (PT_NEE_ENABLED && !optimizationHints.OnlyDeltaLobes) && hasNonDeltaLobes;

        NEEResult result = NEEResult::empty();
        
        if (!applyNEE)
            return result;
        
        // Check if sample from RTXDI should be applied instead of NEE.
#if PATH_TRACER_MODE==PATH_TRACER_MODE_FILL_STABLE_PLANES && PT_USE_RESTIR_DI
        // When ReSTIR DI is handling lighting, we skip NEE; at the moment RTXDI handles only reflection; in the case of first bounce transmission we still don't attemp to use
        // NEE due to complexity, and also the future where ReSTIR DI might handle transmission.
        if (hasNonDeltaLobes && onDominantBranch && onStablePlane)
        {
            result.BSDFMISInfo.SkipEmissiveBRDF = true;
            return result;
        }
#endif

        HandleNEE_MultipleSamples(result, preScatterPath, shadingData, bsdf, sgBase, workingContext, (onDominantBranch&&onStablePlane)?(PT_NEE_BOOST_SAMPLING_ON_DOMINANT_PLANE):(0));
       
        return result;
    }
    
#else // disabled NEE!

inline NEEResult HandleNEE(const uniform OptimizationHints optimizationHints, const PathState preScatterPath, 
                                const ShadingData shadingData, const ActiveBSDF bsdf, const SampleGeneratorVertexBase sgBase, const WorkingContext workingContext)
{
    return NEEResult::empty();
}
#endif
 
}


#endif // __PATH_TRACER_NEE_HLSLI__
