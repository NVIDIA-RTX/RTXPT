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
    
    static const uint           kMaxRejectedHits                = 16;       // Maximum number of rejected hits along a path (PackedCounters::RejectedHits counter, used by nested dielectrics). The path is terminated if the limit is reached to avoid getting stuck in pathological cases.
}

#include "PathTracerNestedDielectrics.hlsli"
#include "PathTracerStablePlanes.hlsli"

#if defined(RTXPT_COMPILE_WITH_NEE) && RTXPT_COMPILE_WITH_NEE!=0
#include "PathTracerNEE.hlsli"
#endif

namespace PathTracer
{
    inline PathState EmptyPathInitialize(uint2 pixelPos, float pixelConeSpreadAngle)
    {
        PathState path;
        path.id                     = PathIDFromPixel(pixelPos);
        path.flagsAndVertexIndex    = 0;
        path.sceneLength            = 0;
        path.fireflyFilterK         = 1.0;
        path.packedCounters         = 0;
        
        // path.setCounter(PackedCounters::SubSampleIndex, subSampleIndex);

        for( uint i = 0; i < INTERIOR_LIST_SLOT_COUNT; i++ )
            path.interiorList.slots[i] = 0;

        path.origin                 = float3(0, 0, 0);
        path.dir                    = float3(0, 0, 0);

        path.thp                    = float3(1, 1, 1);
#if PATH_TRACER_MODE!=PATH_TRACER_MODE_FILL_STABLE_PLANES
        path.L                      = float3(0, 0, 0);
#else
        path.denoiserSampleHitTFromPlane = 0.0;
        path.denoiserDiffRadianceHitDist = lpfloat4(0, 0, 0, 0);
        path.denoiserSpecRadianceHitDist = lpfloat4(0, 0, 0, 0);
        path.secondaryL             = 0.0;
#endif

        path.setHitPacked( HitInfo::make().getData() );
        path.setActive();
        path.setDeltaOnlyPath(true);

        path.rayCone                = RayCone::make(0, pixelConeSpreadAngle);

#if PATH_TRACER_MODE==PATH_TRACER_MODE_BUILD_STABLE_PLANES
        path.imageXform             = lpfloat3x3( 1.f, 0.f, 0.f,
                                                0.f, 1.f, 0.f,
                                                0.f, 0.f, 1.f);
        path.setFlag(PathFlags::stablePlaneOnDominantBranch, true); // stable plane 0 starts being dominant but this can change; in the _NOISY_PASS this is predetermined and can't change
#endif
        path.setStablePlaneIndex(0);
        path.stableBranchID         = 1; // camera has 1; makes IDs unique
        
        // these will be used for the first bounce
        path.packedMISInfo          = (lpuint)NEEBSDFMISInfo::empty().Pack16bit();
        path.bsdfScatterPdf         = 0;

        return path;
    }

    inline void SetupPathPrimaryRay(inout PathState path, const Ray ray)
    {
        path.origin = ray.origin;
        path.dir    = ray.dir;
    }

    /** Check if the path has finished all surface bounces and needs to be terminated.
        Note: This is expected to be called after GenerateScatterRay(), which increments the bounce counters.
        \param[in] path Path state.
        \return Returns true if path has processed all bounces.
    */
    inline bool HasFinishedSurfaceBounces(const PathState path)
    {
        if (Bridge::getMaxBounceLimit()<path.getVertexIndex())
            return true;
        const uint diffuseBounces = path.getCounter(PackedCounters::DiffuseBounces);
        return diffuseBounces > Bridge::getMaxDiffuseBounceLimit();
    }

    /** Update the path throughouput.
        \param[in,out] path Path state.
        \param[in] weight Vertex throughput.
    */
    inline void UpdatePathThroughput(inout PathState path, const float3 weight)
    {
        path.thp *= weight;
    }

    /** Apply russian roulette to terminate paths early.
        \param[in,out] path Path.
        \param[in] u Uniform random number in [0,1).
        \return Returns true if path needs to be terminated.
    */
    inline bool HandleRussianRoulette(inout PathState path, const SampleGeneratorVertexBase sgBase, const WorkingContext workingContext)
    {
#if PATH_TRACER_MODE==PATH_TRACER_MODE_BUILD_STABLE_PLANES  // stable planes must be stable, no RR!
        return false;
#else
        if( !workingContext.ptConsts.enableRussianRoulette )
            return false;

        SampleGenerator sampleGenerator = SampleGenerator::make( sgBase, SampleGeneratorEffectSeed::RussianRoulette, false ); // path.getCounter(PackedCounters::DiffuseBounces)<DisableLowDiscrepancySamplingAfterDiffuseBounceCount ); <- there is some benefit to using LD sampling here but quality gain does not clearly outweigh the cost

        const float rrVal = luminance(path.thp);
        
#if 0   // old "classic" one
        float prob = max(0.f, 1.f - rrVal);
#else   // a milder version of Falcor's Russian Roulette
        float prob = saturate( 0.8 - rrVal ); prob = prob*prob*prob*prob;
#endif

        if (sampleNext1D(sampleGenerator) < prob)
            return true;
        
        UpdatePathThroughput(path, 1.0 / (1.0 - prob)); // in theory we should also do 'path.fireflyFilterK *= (1.0 - prob);' here
        return false;
#endif
    }

    /** Generates a new scatter ray given a valid BSDF sample.
        \param[in] bs BSDF sample (assumed to be valid).
        \param[in] sd Shading data.
        \param[in] bsdf BSDF at the shading point.
        \param[in,out] path The path state.
        \return True if a ray was generated, false otherwise.
    */
    inline ScatterResult GenerateScatterRay(const BSDFSample bs, const ShadingData shadingData, const ActiveBSDF bsdf, inout PathState path, const WorkingContext workingContext)
    {
        ScatterResult result;
        
        if (path.hasFlag(PathFlags::stablePlaneOnPlane) && bs.pdf == 0)
        {
            // Set the flag to remember that this secondary path started with a delta branch,
            // so that its secondary radiance would not be directed into ReSTIR GI later.
            path.setFlag(PathFlags::stablePlaneOnDeltaBranch);
        }

        path.dir = bs.wo;
        if (workingContext.ptConsts.useReSTIRGI && path.hasFlag(PathFlags::stablePlaneOnPlane) && bs.pdf != 0 && path.hasFlag(PathFlags::stablePlaneOnDominantBranch))
        {
            // ReSTIR GI decomposes the throughput of the primary scatter ray into the BRDF and PDF components.
            // The PDF component is applied here, and the BRDF component is applied in the ReSTIR GI final shading pass.
            UpdatePathThroughput(path, 1.0 / bs.pdf);
        }
        else
        {
            // No ReSTIR GI, or not SP 0, or a secondary vertex, or a delta event - use full BRDF/PDF weight
            UpdatePathThroughput(path, bs.weight);
        }
        result.Pdf = bs.pdf;
        result.Dir = bs.wo;
        result.IsDelta = bs.isLobe(LobeType::Delta);
        result.IsTransmission = bs.isLobe(LobeType::Transmission);

        path.clearScatterEventFlags(); // removes PathFlags::transmission, PathFlags::specular, PathFlags::delta flags

        // Compute ray origin for next ray segment.
        path.origin = shadingData.computeNewRayOrigin(bs.isLobe(LobeType::Reflection));
        
        // Handle reflection events.
        if (bs.isLobe(LobeType::Reflection))
        {
            // We classify specular events as diffuse if the roughness is above some threshold.
            float roughness = bsdf.getProperties(shadingData).roughness;
            bool isDiffuse = bs.isLobe(LobeType::DiffuseReflection) || roughness > kSpecularRoughnessThreshold;

            if (isDiffuse)
            {
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

        // if bouncePDF then it's a delta event - expansion angle is 0
        path.fireflyFilterK = ComputeNewScatterFireflyFilterK(path.fireflyFilterK, bs.pdf, bs.lobeP);

        // Mark the path as valid only if it has a non-zero throughput.
        result.Valid = any(path.thp > 0.f);

#if PATH_TRACER_MODE==PATH_TRACER_MODE_FILL_STABLE_PLANES
        if (result.Valid)
            StablePlanesOnScatter(path, bs, workingContext);
#endif

        return result;
    }

    /** Generates a new scatter ray using BSDF importance sampling.
        \param[in] sd Shading data.
        \param[in] bsdf BSDF at the shading point.
        \param[in,out] path The path state.
        \return True if a ray was generated, false otherwise.
    */
    inline ScatterResult GenerateScatterRay(const ShadingData shadingData, const ActiveBSDF bsdf, inout PathState path, const SampleGeneratorVertexBase sgBase, const WorkingContext workingContext)
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
            [branch] if( path.getCounter(PackedCounters::DiffuseBounces)<DisableLowDiscrepancySamplingAfterDiffuseBounceCount )
                preGeneratedSamples = SampleGenerator::Generate( ActiveBSDF::cRandomNumberCountForSampling, sgBase, SampleGeneratorEffectSeed::ScatterBSDF );
            else
                preGeneratedSamples = UniformSampleSequenceGenerator::Generate( ActiveBSDF::cRandomNumberCountForSampling, sgBase, SampleGeneratorEffectSeed::ScatterBSDF );
#endif
        }

        BSDFSample result;
        bool valid = bsdf.sample(shadingData, preGeneratedSamples, result, kUseBSDFSampling);

        ScatterResult res;
        if (valid)
            res = GenerateScatterRay(result, shadingData, bsdf, path, workingContext);
        else
            res = ScatterResult::empty();

        return res;
    }

    // Called after ray tracing just before handleMiss or handleHit, to advance internal states related to travel
    inline void UpdatePathTravelled(inout PathState path, const float3 rayOrigin, const float3 rayDir, const float rayTCurrent, const WorkingContext workingContext, uniform bool incrementVertexIndex = true, uniform bool updateOriginDir = true)
    {
        if (updateOriginDir)    // make sure these two are up to date; they are only intended as "output" from ray tracer but could be used as input by subsystems
        {
            path.origin = rayOrigin;    
            path.dir = rayDir;
        }
        if (incrementVertexIndex)
            path.incrementVertexIndex();                                        // Advance to next path vertex (PathState::vertexIndex). (0 - camera, 1 - first bounce, ...)
        path.rayCone = path.rayCone.propagateDistance(rayTCurrent);             // Grow the cone footprint based on angle; angle itself can change on scatter
        path.sceneLength = min(path.sceneLength+rayTCurrent, kMaxRayTravel);    // Advance total travel length

        // good place for debug viz
#if ENABLE_DEBUG_VIZUALISATION && !NON_PATH_TRACING_PASS// && PATH_TRACER_MODE!=PATH_TRACER_MODE_BUILD_STABLE_PLANES <- let's actually show the build rays - maybe even add them some separate effect in the future
        if( workingContext.debug.IsDebugPixel() )
            workingContext.debug.DrawLine(rayOrigin, rayOrigin+rayDir*rayTCurrent, float4(0.6.xxx, 0.2), float4(1.0.xxx, 1.0));
#endif
    }

    // Miss shader
    inline void HandleMiss(inout PathState path, const float3 rayOrigin, const float3 rayDir, const float rayTCurrent, const WorkingContext workingContext)
    {
        UpdatePathTravelled(path, rayOrigin, rayDir, rayTCurrent, workingContext);

#if PATH_TRACER_MODE==PATH_TRACER_MODE_BUILD_STABLE_PLANES && ENABLE_DEBUG_DELTA_TREE_VIZUALISATION
        if (path.hasFlag(PathFlags::deltaTreeExplorer))
        {
            DeltaTreeVizHandleMiss(path, rayOrigin, rayDir, rayTCurrent, workingContext);
            return;
        }
#endif

        float3 environmentEmission = 0.f;

        NEEBSDFMISInfo misInfo = NEEBSDFMISInfo::Unpack16bit(path.packedMISInfo);
        if ( !(misInfo.SkipEmissiveBRDF && !path.wasScatterTransmission()) )
        {
            // raw source for our environment map
            EnvMap envMap = Bridge::CreateEnvMap(); 

            // sample lower MIP after second diffuse bounce; mip0 gets too costly for high res environment maps
            float mipLevel = (path.getCounter(PackedCounters::DiffuseBounces)>1)?(Bridge::DiffuseEnvironmentMapMIPOffset()):(0); 

            // convert to environment map's local dir (as it supports its own rotation matrix)
            float3 localDir = envMap.ToLocal(path.dir);     
            float3 Le = envMap.EvalLocal(localDir, mipLevel);

            // figure out MIS vs our lighting technique, if any
            float misWeight = 1.0f;
            if( misInfo.LightSamplingEnabled && path.bsdfScatterPdf != 0 )
            {
                // this is the NEE light sampler configured same as it was at previous vertex (previous vertex's "next event estimation" matches this light "event")
                LightSampler lightSampler = Bridge::CreateLightSampler( workingContext.pixelPos, misInfo.LightSamplingIsIndirect, workingContext.debug.IsDebugPixel() );
                uint environmentQuadLightIndex = lightSampler.LookupEnvLightByDirection( localDir ); //< figure out light index from the direction! it's guaranteed to be valid
                misWeight = lightSampler.ComputeBSDFMISForEnvironmentQuad(environmentQuadLightIndex, path.bsdfScatterPdf, misInfo.NarrowNEESamples, misInfo.TotalSamples);

                float simpleRandom = Hash32ToFloat( Hash32Combine( Hash32Combine(Hash32(path.getVertexIndex() + 0x0366FE2F), path.id), Bridge::getSampleIndex() ) );   // note: using unique prime number salt for vertex index
                lightSampler.InsertFeedbackFromBSDF(environmentQuadLightIndex, average(path.thp*Le), misWeight, simpleRandom );
            }

            environmentEmission = misWeight * Le;
        }

        if( workingContext.ptConsts.fireflyFilterThreshold != 0 )
            environmentEmission = FireflyFilter( environmentEmission, workingContext.ptConsts.fireflyFilterThreshold, path.fireflyFilterK );
        environmentEmission *= Bridge::getNoisyRadianceAttenuation();
        
        path.clearHit();
        path.terminate();

#if PATH_TRACER_MODE!=PATH_TRACER_MODE_REFERENCE
        if( !StablePlanesHandleMiss(path, environmentEmission, rayOrigin, rayDir, rayTCurrent, 0, workingContext) )
            return;
#endif

#if PATH_TRACER_MODE != PATH_TRACER_MODE_FILL_STABLE_PLANES // noisy mode should either output everything to denoising buffers, with stable stuff handled in MODE 1; there is no 'residual'
        if (any(environmentEmission>0))
            path.L += max( 0.xxx, path.thp*environmentEmission );   // add to path contribution!
#endif
    }

    // supports only TriangleHit for now; more to be added when needed
    inline void HandleHit(const uniform OptimizationHints optimizationHints, inout PathState path, const float3 rayOrigin, const float3 rayDir, const float rayTCurrent, const WorkingContext workingContext)
    {
        UpdatePathTravelled(path, rayOrigin, rayDir, rayTCurrent, workingContext);
        
        const uint2 pixelPos = PathIDToPixel(path.id);
        const SampleGeneratorVertexBase sampleGeneratorVertexBase = SampleGeneratorVertexBase::make(pixelPos, path.getVertexIndex(), Bridge::getSampleIndex() );
        
#if ENABLE_DEBUG_VIZUALISATION
        const bool debugPath = workingContext.debug.IsDebugPixel();
#else
        const bool debugPath = false;
#endif

        // Upon hit:
        // - Load vertex/material data
        // - Compute MIS weight if path.getVertexIndex() > 1 and emissive hit
        // - Add emitted radiance
        // - Sample light(s) using shadow rays
        // - Sample scatter ray or terminate

        // Few notes & code on figuring out which hit this is for purposes of injecting stuff "at primary surface hit" and etc.
        // Old code (no longer applicable):
        //      const bool isPrimaryHit     = path.getVertexIndex() == 1; <- this is not correct with Primary Surface Replacement enabled 
        // New suggested code (using `base hit` as a new description of primary hit that had PSR applied and is on stable plane zero):
        // #if PATH_TRACER_MODE==PATH_TRACER_MODE_REFERENCE
        //         const bool isBaseHit = path.getVertexIndex() == 1;
        // #elif PATH_TRACER_MODE==PATH_TRACER_MODE_FILL_STABLE_PLANES // build
        //         const bool isBaseHit = path.getCounter(PackedCounters::BouncesFromStablePlane) == 0 && path.getStablePlaneIndex() == 0;
        // #endif

        const TriangleHit triangleHit = TriangleHit::make(path.hitPacked);

        SurfaceData bridgedData = Bridge::loadSurface(optimizationHints, triangleHit, rayDir, path.rayCone, path.getVertexIndex(), pixelPos, workingContext.debug);

#if PATH_TRACER_MODE==PATH_TRACER_MODE_FILL_STABLE_PLANES
        // an example of debugging RayCone data for the specific pixel selected in the UI, at the first bounce (vertex index 1)
        // if( workingContext.debug.IsDebugPixel() && path.getVertexIndex()==1 )
        //     workingContext.debug.Print( 4, path.rayCone.getSpreadAngle(), path.rayCone.getWidth(), rayTCurrent, path.sceneLength);
#endif

        
        // Account for volume absorption.
        float volumeAbsorption = 0;   // used for stats
        if (!path.interiorList.isEmpty())
        {
            const uint materialID = path.interiorList.getTopMaterialID();
            const HomogeneousVolumeData hvd = Bridge::loadHomogeneousVolumeData(materialID); // gScene.materials.getHomogeneousVolumeData(materialID);
            const float3 transmittance = HomogeneousVolumeSampler::evalTransmittance(hvd, rayTCurrent);
            volumeAbsorption = 1 - luminance(transmittance);
            UpdatePathThroughput(path, transmittance);
        }

        // Reject false hits in nested dielectrics but also updates 'outside index of refraction' and dependent data
        bool rejectedFalseHit = !HandleNestedDielectrics(bridgedData, path, workingContext);

#if PATH_TRACER_MODE==PATH_TRACER_MODE_BUILD_STABLE_PLANES && ENABLE_DEBUG_DELTA_TREE_VIZUALISATION
        if (path.hasFlag(PathFlags::deltaTreeExplorer))
        {
            DeltaTreeVizHandleHit(path, rayOrigin, rayDir, rayTCurrent, bridgedData, rejectedFalseHit, HasFinishedSurfaceBounces(path), volumeAbsorption, workingContext);
            return;
        }
#endif
        if (rejectedFalseHit)
            return;

        // These will not change anymore, so make const shortcuts
        const ShadingData shadingData    = bridgedData.shadingData;
        const ActiveBSDF bsdf   = bridgedData.bsdf;

#if ENABLE_DEBUG_VIZUALISATION && PATH_TRACER_MODE!=PATH_TRACER_MODE_BUILD_STABLE_PLANES
        if (debugPath)
        {
            // IoR debugging - .x - "outside", .y - "interior", .z - frontFacing, .w - "eta" (eta is isFrontFace?outsideIoR/insideIoR:insideIoR/outsideIoR)
            // workingContext.debug.Print(path.getVertexIndex(), float4(shadingData.IoR, bridgedData.interiorIoR, shadingData.frontFacing, bsdf.data.eta) );

            // draw tangent space
            workingContext.debug.DrawLine(shadingData.posW, shadingData.posW + shadingData.T * workingContext.debug.LineScale(), float4(0.7, 0, 0, 0.5), float4(1.0, 0, 0, 0.5));
            workingContext.debug.DrawLine(shadingData.posW, shadingData.posW + shadingData.B * workingContext.debug.LineScale(), float4(0, 0.7, 0, 0.5), float4(0, 1.0, 0, 0.5));
            workingContext.debug.DrawLine(shadingData.posW, shadingData.posW + shadingData.N * workingContext.debug.LineScale(), float4(0, 0, 0.7, 0.5), float4(0, 0, 1.0, 0.5));

            // draw ray cone footprint
            float coneWidth = path.rayCone.getWidth();
            workingContext.debug.DrawLine(shadingData.posW + (-shadingData.T+shadingData.B) * coneWidth, shadingData.posW + (+shadingData.T+shadingData.B) * coneWidth, float4(0.5, 0.0, 1.0, 0.5), float4(0.5, 1.0, 0.0, 0.5) );
            workingContext.debug.DrawLine(shadingData.posW + (+shadingData.T+shadingData.B) * coneWidth, shadingData.posW + (+shadingData.T-shadingData.B) * coneWidth, float4(0.5, 0.0, 1.0, 0.5), float4(0.5, 1.0, 0.0, 0.5) );
            workingContext.debug.DrawLine(shadingData.posW + (+shadingData.T-shadingData.B) * coneWidth, shadingData.posW + (-shadingData.T-shadingData.B) * coneWidth, float4(0.5, 0.0, 1.0, 0.5), float4(0.5, 1.0, 0.0, 0.5) );
            workingContext.debug.DrawLine(shadingData.posW + (-shadingData.T-shadingData.B) * coneWidth, shadingData.posW + (-shadingData.T+shadingData.B) * coneWidth, float4(0.5, 0.0, 1.0, 0.5), float4(0.5, 1.0, 0.0, 0.5) );
        }
#endif

        BSDFProperties bsdfProperties = bsdf.getProperties(shadingData);

        // Collect emissive triangle radiance.
        float3 surfaceEmission = 0.0;
        NEEBSDFMISInfo misInfo = NEEBSDFMISInfo::Unpack16bit(path.packedMISInfo);

#if 0        
        if (workingContext.debug.IsDebugPixel())
        {
            DebugPrint( "vi {0}, isEn {1}, isInd {2}, zeroB {3}, n {4}, t {5}, pdf {6}", vertexIndex, (uint)misInfo.LightSamplingEnabled, (uint)misInfo.LightSamplingIsIndirect, (uint)misInfo.SkipEmissiveBSDF, misInfo.NarrowNEESamples, misInfo.TotalSamples, path.bsdfScatterPdf );
        }
#endif

        // should we ignore this triangle's emission, and if not, is there any emission
        if ( !(misInfo.SkipEmissiveBRDF && !path.wasScatterTransmission()) && (any(bsdfProperties.emission>0)) )
        {
            // figure out MIS vs our lighting technique, if any
            float misWeight = 1.0f;
            if( misInfo.LightSamplingEnabled && path.bsdfScatterPdf != 0 )
            {
                // if(bridgedData.neeLightIndex == 0xFFFFFFFF)                     // this is really bad if happens
                //     workingContext.debug.DrawDebugViz( float4(0, 1, 0, 1) );

                // this is the NEE light sampler configured same as it was at previous vertex (previous vertex's "next event estimation" matches this light "event")
                LightSampler lightSampler = Bridge::CreateLightSampler( workingContext.pixelPos, misInfo.LightSamplingIsIndirect, workingContext.debug.IsDebugPixel() );
                misWeight = lightSampler.ComputeBSDFMISForEmissiveTriangle(bridgedData.neeLightIndex, path.bsdfScatterPdf, path.origin, shadingData.posW, misInfo.NarrowNEESamples, misInfo.TotalSamples);

                float simpleRandom = Hash32ToFloat( Hash32Combine( Hash32Combine(Hash32(path.getVertexIndex() + 0x0367C2C7), path.id), Bridge::getSampleIndex() ) );   // note: using unique prime number salt for vertex index
                lightSampler.InsertFeedbackFromBSDF(bridgedData.neeLightIndex, average(path.thp*surfaceEmission), misWeight, simpleRandom );
            }

            surfaceEmission = bsdfProperties.emission * misWeight;
            if( workingContext.ptConsts.fireflyFilterThreshold != 0 )
                surfaceEmission = FireflyFilter(surfaceEmission, workingContext.ptConsts.fireflyFilterThreshold, path.fireflyFilterK);
            surfaceEmission *= Bridge::getNoisyRadianceAttenuation();

#if PATH_TRACER_MODE != PATH_TRACER_MODE_FILL_STABLE_PLANES // noisy mode should either output everything to denoising buffers, with stable stuff handled in MODE 1; there is no 'residual'
        if (any(surfaceEmission>0))
            path.L += max( 0.xxx, path.thp*surfaceEmission );   // add to path contribution!
#endif
        }
        
        // Terminate after scatter ray on last vertex has been processed. Also terminates here if StablePlanesHandleHit terminated path. Also terminate based on "Russian roulette" if enabled.
        bool pathStopping = HasFinishedSurfaceBounces(path);
        
#if PATH_TRACER_MODE!=PATH_TRACER_MODE_REFERENCE
        StablePlanesHandleHit(path, rayOrigin, rayDir, rayTCurrent, optimizationHints.SERSortKey, workingContext, bridgedData, volumeAbsorption, surfaceEmission, pathStopping);
#endif

#if PATH_TRACER_MODE!=PATH_TRACER_MODE_BUILD_STABLE_PLANES // in build mode we've consumed emission and either updated or terminated path ourselves
        pathStopping |= HandleRussianRoulette(path, sampleGeneratorVertexBase, workingContext);    // note: this will update path.thp!
#endif

        if (pathStopping)
        {
            path.terminate();
            return;
        }

#if PATH_TRACER_MODE==PATH_TRACER_MODE_BUILD_STABLE_PLANES 
        // in build mode we've consumed emission and either updated or terminated path ourselves, so we must skip the rest of the function
        return;
#endif 
        
        const PathState preScatterPath = path;

        // Generate the next path segment!
        ScatterResult scatterResult = GenerateScatterRay(shadingData, bsdf, path, sampleGeneratorVertexBase, workingContext);
        //        // debug-view invalid scatters
        //        if (!scatterResult.Valid && path.getVertexIndex() == 1)
        //            workingContext.debug.DrawDebugViz( float4( 1, 0, 0, 1 ) );
       
        // Compute NextEventEstimation a.k.a. direct light sampling!
#if defined(RTXPT_COMPILE_WITH_NEE) && RTXPT_COMPILE_WITH_NEE!=0
        NEEResult neeResult = HandleNEE(optimizationHints, preScatterPath, shadingData, bsdf, sampleGeneratorVertexBase, workingContext); 
#else
        NEEResult neeResult = NEEResult::empty();
#endif
        
        path.packedMISInfo  = (lpuint)neeResult.BSDFMISInfo.Pack16bit();
        path.bsdfScatterPdf = (lpfloat)scatterResult.Pdf;

        lpfloat3 neeDiffuseRadiance, neeSpecularRadiance;
        neeResult.GetRadiances(neeDiffuseRadiance, neeSpecularRadiance);
    
        if ( any( (neeDiffuseRadiance+neeSpecularRadiance) > 0 ) )
        {
            // add realtime multi-sample-per-pixel attenuation
            neeDiffuseRadiance *= (lpfloat3)Bridge::getNoisyRadianceAttenuation();
            neeSpecularRadiance *= (lpfloat3)Bridge::getNoisyRadianceAttenuation();

#if PATH_TRACER_MODE==PATH_TRACER_MODE_FILL_STABLE_PLANES // fill
            StablePlanesHandleNEE(preScatterPath, path, neeDiffuseRadiance, neeSpecularRadiance, neeResult.RadianceSourceDistance, workingContext);
#else
            float3 neeContribution = neeDiffuseRadiance + neeSpecularRadiance;
            path.L += max(0.xxx, preScatterPath.thp * neeContribution); // add to path contribution!
#endif
        }
        
        if (!scatterResult.Valid)
        {
            path.terminate();
        }
    }
};

#endif // __PATH_TRACER_HLSLI__
