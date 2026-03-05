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
    // switch this off to compile out entire NEE codepath!
#if (1 && PT_NEE_ENABLED) || defined(__INTELLISENSE__)

    inline float EvalSampleWeight( const LightSample lightSample, const ShadingData shadingData, const ActiveBSDF bsdf )
    {
    #if 0 // more costly version, does full BSDF - not really worth it unless special case colourful materials with colourful lights
        float3 bsdfThp = bsdf.eval(shadingData, lightSample.Direction, bsdfThpDiff, bsdfThpSpec);
        float weight = max3(bsdfThp*lightSample.Li); // used to be luminance
    #else // ignores colour but cheaper; allows us to use 8 instead of 6 candidate samples and still be a tiny bit faster
        float weight = max3(lightSample.Li) * bsdf.evalPdf(shadingData, lightSample.Direction, kUseBSDFSampling);
    #endif
        return weight;
    }
    
    // Weighted Reservoir Sampling
    // See https://pbr-book.org/4ed/Sampling_Algorithms/Reservoir_Sampling for code & basics and https://agraphicsguynotes.com/posts/understanding_the_math_behind_restir_di/ for a bit more.
    struct NEEWeightedReservoirSampler
    {
        LightSample     Candidate;        // a.k.a. 'reservoir'
        float           WeightSum;
        float           CandidateWeight;  // a.k.a. 'reservoirWeight'

        static NEEWeightedReservoirSampler make()
        {
            NEEWeightedReservoirSampler ret;
            ret.Candidate.Li    = float3(0,0,0); // this makes ret.Candidate.Valid()==false
            ret.WeightSum       = 0;
            ret.CandidateWeight = 0;
            return ret;
        }

        void Add( float randomValue, LightSample candidateSample, float candidateWeight )
        {
            // Perform Weighted Reservoir Sampling
            WeightSum += candidateWeight;
            float wrsThreshold = saturate(candidateWeight / WeightSum);
            if( randomValue < wrsThreshold )
            {
                Candidate = candidateSample;  // TODO: pack here.
                CandidateWeight = candidateWeight;
            }
        }

        float CandidateProbability()    // a.k.a. sampleProbability()
        {
            return CandidateWeight / WeightSum;
        }
    };

    // Generates a light sample from all the available lights.
    inline LightSample GenerateLightSample(const WorkingContext workingContext, const ShadingData shadingData, const ActiveBSDF bsdf, const uint candidateSampleCount, inout UniformSampleSequenceGenerator sampleGenerator, const LightSampler lightSampler)
    {
        // NvReorderThread(0, 32);
        NEEWeightedReservoirSampler wrs = NEEWeightedReservoirSampler::make();
        
        uint localCount, globalCount;
        lightSampler.GetCandidateSampleCounts(candidateSampleCount, localCount, globalCount);
        
        for (uint i = 0; i < candidateSampleCount; i++ )
        {
            const bool sampleIsLocal = i >= globalCount;
            uint lightIndex = 0; float selectionPdf = 0;

            float rnd = sampleNext1D(sampleGenerator);
            if( sampleIsLocal )
                lightIndex = lightSampler.SampleLocal( rnd, selectionPdf );
            else
                lightIndex = lightSampler.SampleGlobal( rnd, selectionPdf );

            const PolymorphicLightInfoFull packedLightInfo = lightSampler.LoadLight(lightIndex);

            // TODO: for LD sampling try the "reuse/recycle" trick
            const float2 interiorSampleRnd = sampleNext2D(sampleGenerator);

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
            LightSample candidateSample;
            /*candidateSample.Pdf =*/ const float pdf = lightSample.SolidAnglePdf * selectionPdf;
            candidateSample.Li = pdf > 0.f ? (lightSample.Radiance / pdf) : float3(0,0,0);
            candidateSample.SolidAnglePdf = lightSample.SolidAnglePdf;
            float3 surfToLight = lightSample.Position-shadingData.posW;
            candidateSample.Distance = length(surfToLight);
            candidateSample.Direction = surfToLight / max( candidateSample.Distance, 1e-7 );
            candidateSample.LightIndex = lightIndex;
            candidateSample.SelectionPdf = selectionPdf;
            candidateSample.LightSampleableByBSDF = lightSample.LightSampleableByBSDF;
            candidateSample.FromLocalDistribution = sampleIsLocal;
            // if( workingContext.Debug.IsDebugPixel() )
            //     workingContext.Debug.DrawLine(shadingData.posW, lightSample.Position, float3(1,0,0), float3(0,1,0) );

            // Perform Weighted Reservoir Sampling
            float wrsWeight = EvalSampleWeight( candidateSample, shadingData, bsdf );
            wrs.Add( sampleNext1D(sampleGenerator), candidateSample, wrsWeight );

            sampleGenerator.AdvanceSampleIndex(); // only needed for LD sampling, resets the dimension; NO-OP for uniform random 
        }

        LightSample pickedSample = wrs.Candidate;

#define LATE_WRS_MIS 1  // functionally identical, just faster if done after shadow test
#if !LATE_WRS_MIS
        float thisPdf, otherPdf;
        lightSampler.ComputeLightSelectionPdfs(pickedSample, localCount, globalCount, thisPdf, otherPdf);
        float wrsMIS = EvalMIS(RTXPT_NEE_MIS_HEURISTIC, 1, thisPdf, 1, otherPdf);   // these should use thisCount and otherCount which are localCount and globalCount in order matching wrs candidate source
        float thisCount = (float)(pickedSample.FromLocalDistribution)?(localCount):(globalCount);
        wrsMIS = wrsMIS / float(thisCount); // * float(candidateSampleCount); cancels with the wrs selection above
#else
        float wrsMIS = 1.0;
#endif

        pickedSample.Li *= wrsMIS / wrs.CandidateProbability(); // * candidateSampleCount; <- we'll include this in later if sample is visible

        return pickedSample;
    }

    // Computes shading surface visibility ray starting position with an offset to avoid self intersection at source, and a
    // shortening offset to avoid self-intersection at the light source end. 
    // Optimal selfIntersectionShorteningK default found empirically.
    RayDesc ComputeVisibilityRay(LightSample lightSample, const ShadingData shadingData, const float selfIntersectionShorteningK = 0.9985)
    {
        float3 surfaceShadingNormal = shadingData.N;

        // We must use **shading** normal to correctly figure out whether we're solving for BRDF or BTDF lobe (whether we want to cast the ray above or under the triangle).
        float faceSide = dot(surfaceShadingNormal, lightSample.Direction) >= 0 ? 1 : -1;

        float3 surfaceFaceNormal = shadingData.faceNCorrected * faceSide;
        float3 surfaceWorldPos = ComputeRayOrigin(shadingData.posW, surfaceFaceNormal);

        RayDesc ret; 
        ret.Origin = surfaceWorldPos; 
        ret.Direction = lightSample.Direction; 
        ret.TMin = 0.0; 
        ret.TMax = lightSample.Distance*selfIntersectionShorteningK;
        return ret;
    }

    // This will ray cast and, if light visible, accumulate radiance properly, including doing weighted sum for 
    void ProcessLightSample(inout NEEResult accum, LightSample lightSample, uint candidateSampleCount, uint fullSamples,
                                const ShadingData shadingData, const ActiveBSDF bsdf, const PathState preScatterPath, LightSampler lightSampler,
                                inout UniformSampleSequenceGenerator sampleGenerator, const WorkingContext workingContext)
    {
        bool visible = false;

        /*[branch]*/ if (lightSample.Valid())   // if sample's bad, skip; we tried casting the ray anyway but ignoring the results - didn't yield better perf
        {
            RayDesc ray = ComputeVisibilityRay(lightSample, shadingData);
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
            float fadeOut = (shadingData.shadowNoLFadeout>0)?(ComputeLowGrazingAngleFalloff( lightSample.Direction, shadingData.vertexN, shadingData.shadowNoLFadeout, 2.0 * shadingData.shadowNoLFadeout )):(1.0);

            float localCount, globalCount;
            lightSampler.GetCandidateSampleCounts(candidateSampleCount, localCount, globalCount);

            // Inner (WRS) MIS is essential for mixing samples from two pdf-s, one of which (Local) is mostly 0 except for the most important light sources contained by it
#if !LATE_WRS_MIS
            float wrsMIS = 1.0;
#else
            float thisPdf, otherPdf, thisCount, otherCount;
            lightSampler.ComputeLightSelectionPdfs(lightSample, localCount, globalCount, thisPdf, otherPdf, thisCount, otherCount);
            float wrsMIS = EvalMIS(RTXPT_NEE_MIS_HEURISTIC, 1, thisPdf, 1, otherPdf);   // these would normally use thisCount and otherCount which are localCount and globalCount in order matching wrs candidate source, but we've cancelled out few terms
            wrsMIS = wrsMIS / float(thisCount); // * float(candidateSampleCount); cancels with the wrs selection above
#endif

            // Outer, light sampling vs path scatter (BSDF) multiple importance sampling is also important but can be approximated which is an user setting
            float scatterPdfForDir = bsdf.evalPdf(shadingData, lightSample.Direction, kUseBSDFSampling);

#if RTXPT_USE_APPROXIMATE_MIS
            float pathMIS = lightSampler.ComputeLightVsBSDF_MIS_ForLight_Approx(lightSample, candidateSampleCount, fullSamples, scatterPdfForDir);
#else
            float pathMIS = lightSampler.ComputeLightVsBSDF_MIS_ForLight(shadingData.posW, lightSample, thisPdf, otherPdf, thisCount, otherCount, candidateSampleCount, fullSamples, scatterPdfForDir);
#endif

            // apply MIS and other multipliers to light here - reduces register pressure and computation later
            lightSample.Li *= fadeOut * wrsMIS * pathMIS / (float)fullSamples;

            // compute BSDF throughput!
            float4 bsdfThp = bsdf.eval(shadingData, lightSample.Direction);

            // compute radiance with MIS and other modifiers
            float3 radiance = bsdfThp.rgb * lightSample.Li;
            float radianceAvg = Average(radiance);
            float specAvg = bsdfThp.w * Average(lightSample.Li);

#if RTXPT_FIREFLY_FILTER && PATH_TRACER_MODE!=PATH_TRACER_MODE_BUILD_STABLE_PLANES  // firefly filter has cost - only enable if denoiser REALLY requires it
            if( workingContext.PtConsts.fireflyFilterThreshold != 0 )
            {
                const float pdf = lightSample.SelectionPdf * lightSample.SolidAnglePdf;
                float neeFireflyFilterK = ComputeNewScatterFireflyFilterK(preScatterPath.GetFireflyFilterK(), pdf, 1.0);
                const float ffDampening = FireflyFilterShort(radianceAvg, workingContext.PtConsts.fireflyFilterThreshold, neeFireflyFilterK);
                radiance *= ffDampening;
                // radianceAvg *= ffDampening; // radianceAvg used for NEE weight later - testing suggests it's same or better if not firefly-filtered
            }
#endif

            // apply path throughput here
            float3 preScatterThp = preScatterPath.GetThp();
            float preScatterPathThpAvg = Average(preScatterThp);
            radiance *= preScatterThp;
            specAvg *= preScatterPathThpAvg;
            radianceAvg *= preScatterPathThpAvg;

            // accumulate radiances
            accum.AccumulateRadiance( radiance, specAvg );
            
            // sample feedback for NEE-AT if needed!
            if( lightSample.LightIndex != RTXPT_INVALID_LIGHT_INDEX && lightSampler.IsTemporalFeedbackRequired() )
            {
                // compute light weigh as in "how much we want this light" - so include path throughput, BSDF and light; 
                float feedbackWeight = radianceAvg;

                lightSampler.InsertFeedbackFromNEE(lightSample.LightIndex, feedbackWeight, sampleNext1D(sampleGenerator));
            }
        }
    }
    
    inline NEEResult HandleNEE_MultipleSamples(const PathState preScatterPath, const ShadingData shadingData, const ActiveBSDF bsdf, 
                                            const LightSampler lightSampler, const uint fullSamples, inout UniformSampleSequenceGenerator sampleGenerator, const WorkingContext workingContext)
    {
        #ifdef RTXPT_NEE_TOTAL_CANDIDATE_SAMPLE_COUNT
        const uint candidateSampleCount = RTXPT_NEE_TOTAL_CANDIDATE_SAMPLE_COUNT;
        #else
        const uint candidateSampleCount = workingContext.PtConsts.NEECandidateSamples;
        #endif

        NEEResult result = NEEResult::empty();
        result.BSDFMISInfo.LightSamplingEnabled = true;
        result.BSDFMISInfo.LightSamplingIsSSC   = lightSampler.IsScreenSpaceCoherent;
        result.BSDFMISInfo.CandidateSamples     = candidateSampleCount;
        result.BSDFMISInfo.FullSamples          = fullSamples;
        
        for (uint sampleIndex = 0; sampleIndex < fullSamples; sampleIndex++)
        {
            LightSample lightSample = GenerateLightSample(workingContext, shadingData, bsdf, candidateSampleCount, sampleGenerator, lightSampler);

            // this computes the BSDF throughput and (if throughput>0) then casts shadow ray and handles radiance summing up & weighted averaging for 'sample distance' used by denoiser
            ProcessLightSample(result, lightSample, candidateSampleCount, fullSamples, shadingData, bsdf, preScatterPath, lightSampler, sampleGenerator, workingContext);
        }
        
        return result;
    }
    
    inline NEEResult HandleNEE(const PathState preScatterPath,
                                    const ShadingData shadingData, const ActiveBSDF bsdf, inout UniformSampleSequenceGenerator sampleGenerator, const WorkingContext workingContext)
    {
        LightSampler lightSampler = Bridge::CreateLightSampler( preScatterPath.GetPixelPos(), preScatterPath.rayCone.getWidth(), preScatterPath.GetSceneLength() );

        // There's a cost to having these as a dynamic constant so an option for production code is to hard code
        #ifdef RTXPT_NEE_FULL_SAMPLE_COUNT
        const uint fullSamples = RTXPT_NEE_FULL_SAMPLE_COUNT;
        #else
        const uint fullSamples = min(RTXPT_LIGHTING_MAX_SAMPLE_COUNT, workingContext.PtConsts.NEEFullSamples);
        #endif

        // Determine if BSDF has non-delta lobes.
        const uint lobes = bsdf.getLobes(shadingData);
        const bool hasNonDeltaLobes = ((lobes & (uint) LobeType::NonDelta) != 0);

        const bool onDominantBranch = preScatterPath.hasFlag(PathFlags::stablePlaneOnDominantBranch);
        const bool onStablePlane = preScatterPath.hasFlag(PathFlags::stablePlaneOnPlane);

        // Check if we should apply NEE.
        bool applyNEE = hasNonDeltaLobes;
        applyNEE &= !lightSampler.IsEmpty() && fullSamples > 0;

        if (!applyNEE)
            return NEEResult::empty();

        // Check if sample from RTXDI should be applied instead of NEE.
#if PATH_TRACER_MODE==PATH_TRACER_MODE_FILL_STABLE_PLANES && PT_USE_RESTIR_DI
        // When ReSTIR DI is handling lighting, we skip NEE; at the moment RTXDI handles only reflection; in the case of first bounce transmission we still don't attemp to use
        // NEE due to complexity, and also the future where ReSTIR DI might handle transmission.
        if (hasNonDeltaLobes && onDominantBranch && onStablePlane)
        {
            NEEResult result = NEEResult::empty();
            result.BSDFMISInfo.SkipEmissiveBRDF = true;
            return result;
        }
#endif

        // in theory, using quasi-random sampling should help with picking light candidates; in practice it doesn't seem to help enough to justify the cost - even when we need to include picking sample on the light as well (see GenerateLightSample)
        // this code used to work for LD sampling in the past, leaving as a reference - you probably want to use the same stream for global and local samples this time, will make it easier
        // sampleGeneratorLightSampler = SampleGenerator::make( sgBase, SampleGeneratorEffectSeed::NextEventEstimationLightSamplerG, useLowDiscrepancyGen, globalNEESamples * workingContext.PtConsts.NEECandidateSamples );

        return HandleNEE_MultipleSamples(preScatterPath, shadingData, bsdf, lightSampler, fullSamples, sampleGenerator, workingContext);
    }
    
#else // disabled NEE!

inline NEEResult HandleNEE(const PathState preScatterPath, 
                                const ShadingData shadingData, const ActiveBSDF bsdf, const SampleGeneratorVertexBase sgBase, const WorkingContext workingContext)
{
    return NEEResult::empty();
}
#endif
 
}


#endif // __PATH_TRACER_NEE_HLSLI__
