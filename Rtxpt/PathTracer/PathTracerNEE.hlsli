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
        float3 bsdfThpDiff, bsdfThpSpec;
        bsdf.eval(shadingData, lightSample.Direction, bsdfThpDiff, bsdfThpSpec);
        float3 bsdfThp = bsdfThpDiff + bsdfThpSpec;
        
        float weight = max3(bsdfThp*lightSample.Li); // used to be luminance
    #else // ignores colour but cheaper; allows us to use 8 instead of 6 candidate samples and still be a tiny bit faster
        float weight = max3(lightSample.Li) * bsdf.evalPdf(shadingData, lightSample.Direction, kUseBSDFSampling);
    #endif
        return weight;
    }
    
    // Weighted Reservoir Sampling, helper for picking 1 out of 'totalCandidates' samples. See https://agraphicsguynotes.com/posts/understanding_the_math_behind_restir_di/ for simple explanation.
    struct WRSSingleSampleHelper
    {
        PathLightSample     PickedCandidate;        // a.k.a. reservoir; should be packable to uint4
        float               ReservoirTotalWeight;
        float               PickedCandidateWeight;
        
        int                 RemainingCandidates;    // can be 16bit

        static WRSSingleSampleHelper make(int totalCandidates)
        {
            WRSSingleSampleHelper ret;
            ret.PickedCandidate.Li      = float3(0,0,0); // this makes ret.PickedCandidate.Valid()==false
            ret.ReservoirTotalWeight    = 0;
            ret.PickedCandidateWeight   = 0;
            ret.RemainingCandidates     = totalCandidates;
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
            RemainingCandidates--;
        }

        bool NotFinished()       
        { 
            return RemainingCandidates > 0; 
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
    inline PathLightSample GenerateLightSample(const WorkingContext workingContext, const ShadingData shadingData, const ActiveBSDF bsdf, const uint wrsCandidateSampleCount, inout SampleGenerator sampleGeneratorLowDiscrepancy, inout SampleGenerator sampleGeneratorRandom, const LightSampler lightSampler, const bool sampleIsNarrow)
    {
        WRSSingleSampleHelper wrs = WRSSingleSampleHelper::make(wrsCandidateSampleCount);
        
        do
        {
            uint lightIndex = 0; float selectionPdf = 0;

            float rnd = sampleNext1D(sampleGeneratorLowDiscrepancy);
            if( sampleIsNarrow )
                lightIndex = lightSampler.SampleNarrow( rnd, selectionPdf );
            else
                lightIndex = lightSampler.SampleGlobal( rnd, selectionPdf );

            const PolymorphicLightInfoFull packedLightInfo = lightSampler.LoadLight(lightIndex);

        #if 1
            const float2 interiorSampleRnd = sampleNext2D(sampleGeneratorRandom);
        #else
            const float2 interiorSampleRnd = sampleNext2D(sampleGeneratorLowDiscrepancy);
        #endif

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
            // if( workingContext.debug.IsDebugPixel() )
            //     workingContext.debug.DrawLine(shadingData.posW, lightSample.Position, float3(1,0,0), float3(0,1,0) );

            // Perform Weighted Reservoir Sampling
            wrs.InsertCandidate( sampleNext1D(sampleGeneratorRandom), candidateSample, EvalSampleWeight( candidateSample, shadingData, bsdf ) );

            sampleGeneratorLowDiscrepancy.AdvanceSampleIndex(); 

        } while( wrs.NotFinished() );
        return wrs.GetResult(wrsCandidateSampleCount);
    }

    // This will ray cast and, if light visible, accumulate radiance properly, including doing weighted sum for 
    void ProcessLightSample(inout NEEResult accum, inout float luminanceSum, PathLightSample lightSample, bool sampleIsNarrow, uint narrowNEESamples, uint totalSamples,
                                const ShadingData shadingData, const ActiveBSDF bsdf, const PathState preScatterPath, LightSampler lightSampler,
                                const bool useFeedback, inout LightFeedbackReservoir feedbackReservoir, inout SampleGenerator sampleGeneratorFeedback, const WorkingContext workingContext)
    {
        if (!lightSample.Valid())   // if sample's bad, skip; we tried casting the ray anyway but ignoring the results - didn't yield better perf
            return;

        const RayDesc ray = lightSample.ComputeVisibilityRay(shadingData).toRayDesc();
            
        bool visible = Bridge::traceVisibilityRay(ray, preScatterPath.rayCone, preScatterPath.getVertexIndex(), workingContext.debug);

        // if( workingContext.debug.IsDebugPixel() )
        //     DebugLine( shadingData.posW, shadingData.posW+lightSample.Direction*lightSample.Distance, float4(!visible,visible,0,1.0) );

        if (visible)
        {
            // add compute grazing angle fadeout
            float fadeOut = (shadingData.shadowNoLFadeout>0)?(ComputeLowGrazingAngleFalloff( ray.Direction, shadingData.vertexN, shadingData.shadowNoLFadeout, 2.0 * shadingData.shadowNoLFadeout )):(1.0);

            // narrow (tile) vs global vs scatter (BSDF) multiple importance sampling
            float scatterPdfForDir = bsdf.evalPdf(shadingData, lightSample.Direction, kUseBSDFSampling);
            float misWeight = lightSampler.ComputeInternalMIS(shadingData.posW, lightSample, sampleIsNarrow, narrowNEESamples, totalSamples, scatterPdfForDir);

            fadeOut *= misWeight;

            // compute BSDF throughput!                
            float3 bsdfThpDiff, bsdfThpSpec;
            bsdf.eval(shadingData, lightSample.Direction, bsdfThpDiff, bsdfThpSpec);

#if 1   // firefly filter has cost - only enable if denoiser REALLY requires it
            if( workingContext.ptConsts.fireflyFilterThreshold != 0 )
            {
                const float pdf = lightSample.SelectionPdf * lightSample.SolidAnglePdf;
                float neeFireflyFilterK = ComputeNewScatterFireflyFilterK(preScatterPath.fireflyFilterK, pdf, 1.0);
                fadeOut *= FireflyFilterShort(average(lightSample.Li*(bsdfThpDiff+bsdfThpSpec))*fadeOut, workingContext.ptConsts.fireflyFilterThreshold, neeFireflyFilterK);
            }
#endif
           
            // apply MIS and other modifiers
            lightSample.Li *= fadeOut;

            float3 diffRadiance = bsdfThpDiff * lightSample.Li;
            float3 specRadiance = bsdfThpSpec * lightSample.Li;

            // weighted sum for sample distance and sample feedback weight
            float3 combinedContribution = diffRadiance + specRadiance;
            float combinedContributionAvg = average(combinedContribution);

            // accumulate radiances
            lpfloat3 neeDiffuseRadiance, neeSpecularRadiance;
            accum.GetRadiances(neeDiffuseRadiance, neeSpecularRadiance);
            neeDiffuseRadiance = lpfloat3( min(neeDiffuseRadiance   + diffRadiance, HLF_MAX.xxx ) );
            neeSpecularRadiance = lpfloat3( min(neeSpecularRadiance + specRadiance, HLF_MAX.xxx ) );
            accum.SetRadiances(neeDiffuseRadiance, neeSpecularRadiance);

            // compute weighted sample distance
            accum.RadianceSourceDistance = lpfloat( min( accum.RadianceSourceDistance + lightSample.Distance * combinedContributionAvg, HLF_MAX ) );
            luminanceSum += combinedContributionAvg;
            
            // sample feedback for NEE-AT!
            if( useFeedback && lightSample.LightIndex != 0xFFFFFFFF ) // TODO: make these 0xFFFFFFFF named consts 
            {
                // compute light weigh as in "how much we want this light" - so include path throughput, BSDF and light; 
                float feedbackWeight = average(preScatterPath.thp) * combinedContributionAvg;

                // we can slightly boost samples that are underrepresented by the global sampler (and, likewise, dampen those that are easily sampled from global sampler)
                // benefit is non-conclusive, so disabled by default
                // feedbackWeight /= sqrt(lightSampler.SampleGlobalPDF(lightSample.LightIndex));

                lightSampler.InsertFeedbackFromNEE(feedbackReservoir, lightSample.LightIndex, feedbackWeight, sampleNext1D(sampleGeneratorFeedback) );
            }
        }
    }
    
    void FinalizeLightSample( inout NEEResult accum, const float luminanceSum )
    {
        accum.RadianceSourceDistance = lpfloat( min( accum.RadianceSourceDistance / (luminanceSum + 1e-30), HLF_MAX ) );
    }
    
    // 'result' argument is expected to have been initialized to 'NEEResult::empty()'
    inline void HandleNEE_MultipleSamples(inout NEEResult inoutResult, const PathState preScatterPath, const ShadingData shadingData, const ActiveBSDF bsdf, 
                                            const SampleGeneratorVertexBase sgBase, const WorkingContext workingContext, int sampleCountBoost)
    {
        LightSampler lightSampler = Bridge::CreateLightSampler( workingContext.pixelPos, preScatterPath.rayCone.getWidth() / preScatterPath.sceneLength, workingContext.debug.IsDebugPixel() );

        // more costly alternative
        // sampleCountBoost = lightSampler.IsIndirect?0:workingContext.ptConsts.NEEBoostSamplingOnDominantPlane;

        // There's a cost to having these as a dynamic constant so an option for production code is to hard code
        const uint totalSamples = min(RTXPT_LIGHTING_NEEAT_MAX_TOTAL_SAMPLE_COUNT, (!lightSampler.IsEmpty()) ? (sampleCountBoost + workingContext.ptConsts.NEEFullSamples)     : (0));
        if (totalSamples == 0)
            return;

        bool useLowDiscrepancyGen = true;

        SampleGenerator sampleGenerator         = SampleGenerator::make( sgBase, SampleGeneratorEffectSeed::NextEventEstimation );
        SampleGenerator sampleGeneratorLightSampler = SampleGenerator::make( sgBase, SampleGeneratorEffectSeed::NextEventEstimationLightSampler, useLowDiscrepancyGen, totalSamples * workingContext.ptConsts.NEECandidateSamples );
        SampleGenerator sampleGeneratorFeedback = SampleGenerator::make( sgBase, SampleGeneratorEffectSeed::NextEventEstimationFeedback );

        LightFeedbackReservoir feedbackReservoir;
        bool useFeedback = false;
#if PATH_TRACER_MODE!=PATH_TRACER_MODE_BUILD_STABLE_PLANES
        if( lightSampler.IsTemporalFeedbackRequired() )
        {
            feedbackReservoir = lightSampler.LoadFeedback();
            useFeedback = true;
        }
#endif
        const uint narrowNEESamples     = lightSampler.ComputeNarrowSampleCount(totalSamples);

        inoutResult.BSDFMISInfo.LightSamplingEnabled    = true;
        inoutResult.BSDFMISInfo.LightSamplingIsIndirect = lightSampler.IsIndirect;
        inoutResult.BSDFMISInfo.NarrowNEESamples        = narrowNEESamples;
        inoutResult.BSDFMISInfo.TotalSamples            = totalSamples;

        // we must initialize to 0 since we're accumulating multiple samples
        inoutResult.RadianceSourceDistance = 0;
        
        float luminanceSum = 0.0; // for sample distance weighted average 
        
        int narrowNEESamplesRemaining = narrowNEESamples;
        
        for (uint sampleIndex = 0; sampleIndex < totalSamples; sampleIndex++)
        {
            bool sampleIsNarrow = narrowNEESamplesRemaining>0;
            narrowNEESamplesRemaining--;

            PathLightSample lightSample = GenerateLightSample(workingContext, shadingData, bsdf, workingContext.ptConsts.NEECandidateSamples, sampleGeneratorLightSampler, sampleGenerator, lightSampler, sampleIsNarrow);

            // this computes the BSDF throughput and (if throughput>0) then casts shadow ray and handles radiance summing up & weighted averaging for 'sample distance' used by denoiser
            ProcessLightSample(inoutResult, luminanceSum, lightSample, sampleIsNarrow, narrowNEESamples, totalSamples, shadingData, bsdf, preScatterPath, lightSampler, useFeedback, feedbackReservoir, sampleGeneratorFeedback, workingContext);
        }

#if PATH_TRACER_MODE!=PATH_TRACER_MODE_BUILD_STABLE_PLANES
        if( useFeedback )
            lightSampler.StoreFeedback( feedbackReservoir, true );
#endif

        FinalizeLightSample(inoutResult, luminanceSum);
    }
    
    inline NEEResult HandleNEE(const uniform OptimizationHints optimizationHints, const PathState preScatterPath,
                                    const ShadingData shadingData, const ActiveBSDF bsdf, const SampleGeneratorVertexBase sgBase, const WorkingContext workingContext)
    {
        // Determine if BSDF has non-delta lobes.
        const uint lobes = bsdf.getLobes(shadingData);
        const bool hasNonDeltaLobes = ((lobes & (uint) LobeType::NonDelta) != 0) && (!optimizationHints.OnlyDeltaLobes);

        // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        // TODO: This is for performance reasons, to exclude non-visible samples. Check whether it's actually beneficial in practice (branchiness)
        // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        //const bool flagLightSampledUpper = (lobes & (uint) LobeType::NonDeltaReflection) != 0;
        const bool onDominantBranch = preScatterPath.hasFlag(PathFlags::stablePlaneOnDominantBranch);
        const bool onStablePlane = preScatterPath.hasFlag(PathFlags::stablePlaneOnPlane);

        // Check if we should apply NEE.
        const bool applyNEE = (workingContext.ptConsts.NEEEnabled && !optimizationHints.OnlyDeltaLobes) && hasNonDeltaLobes;

        NEEResult result = NEEResult::empty();
        
        if (!applyNEE)
            return result;
        
        // Check if sample from RTXDI should be applied instead of NEE.
#if PATH_TRACER_MODE==PATH_TRACER_MODE_FILL_STABLE_PLANES
        const bool applyReSTIRDI = workingContext.ptConsts.useReSTIRDI && hasNonDeltaLobes && onDominantBranch && onStablePlane;
#else
        const bool applyReSTIRDI = false;
#endif
        
        // When ReSTIR DI is handling lighting, we skip NEE; at the moment RTXDI handles only reflection; in the case of first bounce transmission we still don't attemp to use
        // NEE due to complexity, and also the future where ReSTIR DI might handle transmission.
        if (applyReSTIRDI)
        {
            result.BSDFMISInfo.SkipEmissiveBRDF = true;
            return result;
        }

        HandleNEE_MultipleSamples(result, preScatterPath, shadingData, bsdf, sgBase, workingContext, (onDominantBranch&&onStablePlane)?(workingContext.ptConsts.NEEBoostSamplingOnDominantPlane):(0));
       
        // Debugging tool to remove direct lighting from primary surfaces
        const bool suppressNEE = preScatterPath.hasFlag(PathFlags::stablePlaneOnDominantBranch) && preScatterPath.hasFlag(PathFlags::stablePlaneOnPlane) && workingContext.ptConsts.suppressPrimaryNEE;
        if (suppressNEE)    // keep it a valid sample so we don't add in normal path
            result.SetRadiances(0,0);
        
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
