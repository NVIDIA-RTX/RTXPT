/*
* Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#ifndef __PATH_TRACER_STABLE_PLANES_HLSLI__ // using instead of "#pragma once" due to https://github.com/microsoft/DirectXShaderCompiler/issues/3943
#define __PATH_TRACER_STABLE_PLANES_HLSLI__

#include "PathTracerTypes.hlsli"

#include "StablePlanes.hlsli"

namespace PathTracer
{
#if PATH_TRACER_MODE==PATH_TRACER_MODE_BUILD_STABLE_PLANES
    // splits out delta component, traces ray to next surface, saves the hit without any further processing
    // if verifyDominantFlag is true, will remove PathFlags::stablePlaneOnDominantBranch from result if not on dominant lobe (otherwise it stay as it was); if we're splitting more than 1 lobe then we can only follow one dominant so we must update - otherwise we can let the flag persist
    inline PathState SplitDeltaPath(const PathState oldPath, const float3 rayDir, const SurfaceData surfaceData, const DeltaLobe lobe, const uint deltaLobeIndex, const bool verifyDominantFlag, const WorkingContext workingContext)
    {
        const ShadingData shadingData = surfaceData.shadingData;

        // 1. first generate new virtual path - this is just for consistency with the rest of the code, most of it is unused and compiled out
        PathState newPath       = oldPath;
        // newPath.incrementVertexIndex(); <- not needed, happens in nextHit
        newPath.dir             = lobe.dir;
        newPath.thp             *= lobe.thp;
        //newPath.pdf             = 0;
        newPath.origin          = shadingData.computeNewRayOrigin(lobe.transmission==0);  // bool param is viewside
        newPath.stableBranchID  = StablePlanesAdvanceBranchID( oldPath.stableBranchID, deltaLobeIndex );

        newPath.setScatterDelta();

        // Handle reflection events.
        if (!lobe.transmission)
        {
            // newPath.incrementBounces(BounceType::Specular);
            newPath.setScatterSpecular();
        }
        else // transmission
        {
            // newPath.incrementBounces(BounceType::Transmission);
            newPath.setScatterTransmission();

            // Update interior list and inside volume flag.
            if (!shadingData.mtl.isThinSurface())
            {
                uint nestedPriority = shadingData.mtl.getNestedPriority();
                newPath.interiorList.handleIntersection(shadingData.materialID, nestedPriority, shadingData.frontFacing);
                newPath.setInsideDielectricVolume(!newPath.interiorList.isEmpty());
            }
        }

        // Compute local transform (rotation component only) and apply to path transform (path.imageXForm). This intentionally ignores curvature/skewing to avoid complexity and need for full 4x4 matrix.
        lpfloat3x3 localT;
        if (lobe.transmission)
        {
            localT = (lpfloat3x3)MatrixRotateFromTo(lobe.dir, rayDir);   // no need to refract again, we already have in and out vectors
        }
        else
        {
            const lpfloat3x3 toTangent  = lpfloat3x3(surfaceData.shadingData.T,surfaceData.shadingData.B,surfaceData.shadingData.N);
            const lpfloat3x3 mirror     = lpfloat3x3(1,0,0,0,1,0,0,0,-1); // reflect around z axis
            localT = mul(mirror,toTangent); 
            localT = mul(transpose(toTangent),localT);
        }
        newPath.imageXform = mul(newPath.imageXform, localT);

        // Testing the xforms - virt should always transform to rayDir here
        // float3 virt = mul( (float3x3)newPath.imageXform, lobe.Wo );
        // if (workingContext.debug.IsDebugPixel() && oldPath.getVertexIndex()==1)
        // {
        //     workingContext.debug.Print(oldPath.getVertexIndex()+0, rayDir );
        //     workingContext.debug.Print(oldPath.getVertexIndex()+1, lobe.Wo );
        //     workingContext.debug.Print(oldPath.getVertexIndex()+2, virt );
        // }

        // clear dominant flag if it this lobe isn't dominant but we were on a dominant path
        if (verifyDominantFlag && newPath.hasFlag(PathFlags::stablePlaneOnDominantBranch))
        {
            int psdDominantDeltaLobeIndex = int(shadingData.mtl.getPSDDominantDeltaLobeP1())-1;
            if ( deltaLobeIndex!=psdDominantDeltaLobeIndex )
                newPath.setFlag(PathFlags::stablePlaneOnDominantBranch, false);
        }

        return newPath;
    }
#endif // #if PATH_TRACER_MODE==PATH_TRACER_MODE_BUILD_STABLE_PLANES    
    
#if PATH_TRACER_MODE!=PATH_TRACER_MODE_REFERENCE
    inline void StablePlanesHandleHit(inout PathState path, const float3 rayOrigin, const float3 rayDir, const float rayTCurrent, uint SERSortKey, const WorkingContext workingContext, const SurfaceData bridgedData, float volumeAbsorption, float3 surfaceEmission, bool pathStopping)
    {
#if PATH_TRACER_MODE==PATH_TRACER_MODE_BUILD_STABLE_PLANES // build
        const uint vertexIndex = path.getVertexIndex();
        const uint currentSPIndex = path.getStablePlaneIndex();

        if (vertexIndex == 1)
            workingContext.stablePlanes.StoreFirstHitRayLengthAndClearDominantToZeroCenter(path.sceneLength);

        bool setAsBase = true;    // if path no longer stable, stop and set as a base
        float passthroughOverride = 0.0;
        if ( (vertexIndex < workingContext.ptConsts.maxStablePlaneVertexDepth) && !pathStopping) // Note: workingContext.ptConsts.maxStablePlaneVertexDepth already includes cStablePlaneMaxVertexIndex and MaxBounceCount
        {
            DeltaLobe deltaLobes[cMaxDeltaLobes]; uint deltaLobeCount; float nonDeltaPart;
            bridgedData.bsdf.evalDeltaLobes(bridgedData.shadingData, deltaLobes, deltaLobeCount, nonDeltaPart);
            bool potentiallyVolumeTransmission = false;

            float pathThp = average(path.thp);    // perhaps give it another 10% just to avoid blocking at the end of a dark volume since all other pixels will be dark

            const float nonDeltaIgnoreThreshold = (1e-5);
            const float deltaIgnoreThreshold    = (0.001f);   
            bool hasNonDeltaLobes = nonDeltaPart > nonDeltaIgnoreThreshold;
            passthroughOverride = saturate(1.0-nonDeltaPart*10.0); // if setting as base and no (or low) non-delta lobe, override denoising settings for better passthrough

#define USE_THP_PRIORITIZED_PRUNING 0

            // prune non-noticeable delta lobes
#if USE_THP_PRIORITIZED_PRUNING
            const float neverIgnoreThreshold = workingContext.ptConsts.stablePlanesSplitStopThreshold / float(vertexIndex); // TODO: add screen space dither to threshold?
            float nonZeroDeltaLobesThp[cMaxDeltaLobes];
#endif
            int nonZeroDeltaLobes[cMaxDeltaLobes];
            int nonZeroDeltaLobeCount = 0;
            for (int k = 0; k < deltaLobeCount; k++)
            {
                DeltaLobe lobe = deltaLobes[k];
                const float thp = average(lobe.thp);
                if (thp>deltaIgnoreThreshold)
                {
                    nonZeroDeltaLobes[nonZeroDeltaLobeCount] = k;	// TODO: there is a weirdness on AMD where this doesn't get filled 
#if USE_THP_PRIORITIZED_PRUNING
                    nonZeroDeltaLobesThp[nonZeroDeltaLobeCount] = thp*pathThp;
#endif
                    nonZeroDeltaLobeCount++;
                    potentiallyVolumeTransmission |= lobe.transmission; // we don't have a more clever way to do this at the moment
                }
            }

            #if 0 // a way of debugging the loop above - right click on the pixel to debug; info is in the Debugging panel in the UI
            if (workingContext.debug.IsDebugPixel() && vertexIndex == 1)
            {
                workingContext.debug.Print( 0, deltaLobeCount );
                workingContext.debug.Print( 1, float4( nonZeroDeltaLobeCount, nonZeroDeltaLobes[0], nonZeroDeltaLobes[1], nonZeroDeltaLobes[2] ) );
                //workingContext.debug.Print( 2, float4( nonZeroDeltaLobeCount, nonZeroDeltaLobesThp[0], nonZeroDeltaLobesThp[1], nonZeroDeltaLobesThp[2] ) );
            }
            #endif

            if( nonZeroDeltaLobeCount > 0)
            {
                // sorting is a bad idea because it causes edges where data goes to one or the other side which prevents denoiser from sharing data alongside the edge and shows up as a seam
                // // bubble-sort ascending (biggest thp at the end, so we can just pop from back when forking later)
                // for (int i = 0; i < nonZeroDeltaLobeCount; i++)
                //     for (int j = i+1; j < nonZeroDeltaLobeCount; j++)
                //         if (nonZeroDeltaLobesThp[i] > nonZeroDeltaLobesThp[j])
                //         {
                //             swap( nonZeroDeltaLobesThp[i], nonZeroDeltaLobesThp[j] );
                //             swap( nonZeroDeltaLobes[i], nonZeroDeltaLobes[j] );
                //         }

                // in case plane index 0, we must stop at first non-direct junction; we can only continue if there's only one delta lobe and no non-delta at all (this then becomes just primary surface replacement)
                bool allowPSR = workingContext.ptConsts.allowPrimarySurfaceReplacement && (nonZeroDeltaLobeCount == 1) && (currentSPIndex == 0) && !potentiallyVolumeTransmission;
                bool canReuseExisting = (currentSPIndex != 0) && (nonZeroDeltaLobeCount > 0);
                canReuseExisting |= allowPSR;
                canReuseExisting &= !hasNonDeltaLobes;       // stop on any diffuse lobe

                int availablePlaneCount = 0; int availablePlanes[cStablePlaneCount];
            
                workingContext.stablePlanes.GetAvailableEmptyPlanes(availablePlaneCount, availablePlanes);

                // when "Dominant bounce" material setting is set to "None (Surface)", and we're already on a split (currentSPIndex > 0), then make sure we can't reuse current path to go on, it has to stop and collect guide buffers
                canReuseExisting &= (currentSPIndex == 0) || (bridgedData.shadingData.mtl.getPSDDominantDeltaLobeP1()>0);
            
                // an example of debugging path decomposition logic for the specific pixel selected in the UI, at the second bounce (vertex index 2)
                // if (workingContext.debug.IsDebugPixel() && vertexIndex == 2)
                //     workingContext.debug.Print( 0, currentSPIndex, availablePlaneCount, canReuseExisting, nonZeroDeltaLobeCount );

#if USE_THP_PRIORITIZED_PRUNING
                float prunedThpTotal = 0.0;
                // prune lowest importance lobes that we can't handle
                while ((availablePlaneCount+canReuseExisting) < nonZeroDeltaLobeCount)
                {
                    int lowestThpIndex = 0;
                    float lowestThpValue = nonZeroDeltaLobesThp[0];
                    for (int i = 1; i < nonZeroDeltaLobeCount; i++)
                        if (nonZeroDeltaLobesThp[i] < lowestThpValue)
                        {
                            lowestThpIndex = i;
                            lowestThpValue = nonZeroDeltaLobesThp[i];
                        }
                    for (int j = lowestThpIndex; j < nonZeroDeltaLobeCount-1; j++)
                    {
                        nonZeroDeltaLobesThp[j] = nonZeroDeltaLobesThp[j+1];
                        nonZeroDeltaLobes[j]    = nonZeroDeltaLobes[j+1];
                    }
                    nonZeroDeltaLobeCount--;
                    prunedThpTotal += lowestThpValue;

                    // do not ignore the junction if we'll be completely missing a significant contribution
                    if (prunedThpTotal>neverIgnoreThreshold)
                        canReuseExisting = false;
                }
#else // #if USE_THP_PRIORITIZED_PRUNING
                nonZeroDeltaLobeCount = min(nonZeroDeltaLobeCount, availablePlaneCount+canReuseExisting);
#endif // #if USE_THP_PRIORITIZED_PRUNING

                // remove one lobe from the list for later reuse
                int lobeForReuse = -1;                                                  // could be one-liner with ?
                if (canReuseExisting)                                                       // could be one-liner with ?
                {                                                                           // could be one-liner with ?
                    lobeForReuse = nonZeroDeltaLobes[nonZeroDeltaLobeCount-1];          // could be one-liner with ?
                    nonZeroDeltaLobeCount--;                                                // could be one-liner with ?
                }                                                                           // could be one-liner with ?

                for( int i = 0; i < nonZeroDeltaLobeCount; i++ )
                {
                    const int lobeToExplore = nonZeroDeltaLobes[i];
                    // split and then trace ray & enqueue hit for further traversal after this path is completed
                    PathState splitPath = PathTracer::SplitDeltaPath(path, rayDir, bridgedData, deltaLobes[lobeToExplore], lobeToExplore, true, workingContext);
                    splitPath.setStablePlaneIndex(availablePlanes[i]);
                    workingContext.stablePlanes.StoreExplorationStart(availablePlanes[i], PathPayload::pack(splitPath).packed);

                    #if 0 // way of debugging stable plane index 1 / split path stuff
                    if (splitPath.getStablePlaneIndex() == 1)
                    {
                        // PathPayload pathPayload = PathPayload::pack(splitPath); 
                        // // const uint4 packed[6] = pathPayload.packed;
                        // 
                        // PathState testPath = PathPayload::unpack(pathPayload, PACKED_HIT_INFO_ZERO);

                        workingContext.debug.DrawDebugViz( float4(DbgShowNormalSRGB(deltaLobes[lobeToExplore].dir), 1) ); 

                        if (workingContext.debug.IsDebugPixel() /*&& vertexIndex == 2*/)
                            workingContext.debug.Print( 0, float4( deltaLobes[lobeToExplore].dir, deltaLobes[lobeToExplore].probability ) );

                    }
                    #endif

                }

                // and use current path to reuse lobe
                if ( lobeForReuse != -1 )
                {
                    setAsBase = false;
                    // split and use current path to continue on the best lobe
                    path = PathTracer::SplitDeltaPath(path, rayDir, bridgedData, deltaLobes[lobeForReuse], lobeForReuse, nonZeroDeltaLobeCount>0, workingContext);
                }
            }
        }

        // we've reached the end of stable delta path exploration on this plane; figure out surface properties including motion vectors and store
        if (setAsBase)
        {
            // move surface world pos to first transform starting point reference frame; we then rotate it with the rotation-only imageXform, and place it back into worldspace (we used to have the whole transform but this takes too much space in payload...)
            const Ray cameraRay = Bridge::computeCameraRay( workingContext.pixelPos, /*path.getCounter(PackedCounters::SubSampleIndex)*/ 0 );   // note: all realtime mode subSamples currently share same camera ray at subSampleIndex == 0 (otherwise denoising guidance buffers would be noisy)
            float3 worldXFormFrom   = cameraRay.origin + cameraRay.dir * workingContext.stablePlanes.LoadFirstHitRayLength(workingContext.pixelPos);
            float3 imagePosW        = mul((float3x3)path.imageXform, bridgedData.shadingData.posW-worldXFormFrom)+worldXFormFrom;
            float3 prevImagePosW    = mul((float3x3)path.imageXform, bridgedData.prevPosW-worldXFormFrom)+worldXFormFrom;

            // figure out motion vectors 
            float3 motionVectors = Bridge::computeMotionVector(imagePosW.xyz, prevImagePosW.xyz); // <- simplify/improve this (for debugging to ignore imageXform, just feed in 'bridgedData.shadingData.posW, bridgedData.prevPosW')
            
            // denoising guide helpers
            const float roughness   = saturate(bridgedData.bsdf.data.Roughness());
            float3 worldNormal  = bridgedData.shadingData.N;

            worldNormal = normalize(mul((float3x3)path.imageXform, worldNormal));

            float3 diffBSDFEstimate, specBSDFEstimate;
            BSDFProperties bsdfProperties = bridgedData.bsdf.getProperties(bridgedData.shadingData);
            bsdfProperties.estimateSpecDiffBSDF(diffBSDFEstimate, specBSDFEstimate, bridgedData.shadingData.N, bridgedData.shadingData.V);
            
            // diffBSDFEstimate = lerp( diffBSDFEstimate, 0.5.xxx, passthroughOverride * 0.5 );
            // specBSDFEstimate = lerp( specBSDFEstimate, 0.5.xxx, passthroughOverride * 0.5 );

            workingContext.stablePlanes.StoreStablePlaneCenter(currentSPIndex, vertexIndex, path.hitPacked, SERSortKey, path.stableBranchID, rayDir, path.sceneLength, path.thp, 
                motionVectors, roughness, worldNormal, diffBSDFEstimate, specBSDFEstimate, path.hasFlag(PathFlags::stablePlaneOnDominantBranch));
            
            // since we're building the planes and we've found a base plane, terminate here and the nextHit contains logic for continuing from other split paths if any (enqueued with .StoreExplorationStart)
            path.terminate();
        }
#elif PATH_TRACER_MODE==PATH_TRACER_MODE_FILL_STABLE_PLANES // fill
        
        const int bouncesFromStablePlane = path.getCounter(PackedCounters::BouncesFromStablePlane);
        if (workingContext.ptConsts.useReSTIRGI && bouncesFromStablePlane == 1 && path.hasFlag(PathFlags::stablePlaneOnDominantBranch))
        {
            // Store the position and orientation of the first secondary vertex for ReSTIR GI
            Bridge::StoreSecondarySurfacePositionAndNormal(PathIDToPixel(path.id), bridgedData.shadingData.posW, bridgedData.shadingData.N);

            // an example of debugging secondary surface pos & normal for the specific pixel selected in the UI, at all touched bounces
            // if (workingContext.debug.IsDebugPixel())
            //     workingContext.debug.Print(path.getVertexIndex(), bridgedData.shadingData.posW);
        }

        path.denoiserSampleHitTFromPlane = StablePlaneAccumulateSampleHitT( path.denoiserSampleHitTFromPlane, rayTCurrent, bouncesFromStablePlane, path.isDeltaOnlyPath() );

        if (!path.hasFlag(PathFlags::stablePlaneOnBranch)) // not on stable branch, we need to capture emission
        {
            path.secondaryL += surfaceEmission * path.thp;
        }
#endif
    }

#if PATH_TRACER_MODE==PATH_TRACER_MODE_FILL_STABLE_PLANES // fill only
    inline void StablePlanesOnScatter(inout PathState path, const BSDFSample bs, const WorkingContext workingContext)
    {
        const bool wasOnStablePlane = path.hasFlag(PathFlags::stablePlaneOnPlane);
        if( wasOnStablePlane ) // if we previously were on plane, remember the first scatter type
        {
            path.setFlag(PathFlags::stablePlaneBaseScatterDiff, (bs.lobe & (uint)LobeType::Diffuse)!=0);
            //path.setFlag(PathFlags::stablePlaneBaseTransmission, (bs.isLobe(LobeType::Transmission))!=0);
        }
        path.setFlag(PathFlags::stablePlaneOnPlane, false);     // assume we're no longer going to be on stable plane

        const uint nextVertexIndex = path.getVertexIndex()+1;   // since below we're updating states for the next surface hit, we're using the next one

        // update stableBranchID if we're still on delta paths, and make sure we're still on a path (this effectively predicts the future hit based on pre-generated branches)
        if (path.hasFlag(PathFlags::stablePlaneOnBranch) && nextVertexIndex <= cStablePlaneMaxVertexIndex)
        {
            path.stableBranchID = StablePlanesAdvanceBranchID( path.stableBranchID, bs.getDeltaLobeIndex() );
            bool onStablePath = false;
            for (uint spi = 0; spi < cStablePlaneCount; spi++)
            {
                const uint planeBranchID = workingContext.stablePlanes.GetBranchIDCenter(spi);
                if (planeBranchID == cStablePlaneInvalidBranchID)
                    continue;

                // changing the stable plane for the future
                if (StablePlaneIsOnPlane(planeBranchID, path.stableBranchID))
                {
                    workingContext.stablePlanes.CommitDenoiserRadiance(path.getStablePlaneIndex(), path.denoiserSampleHitTFromPlane,
                        path.denoiserDiffRadianceHitDist, path.denoiserSpecRadianceHitDist,
                        path.secondaryL, path.hasFlag(PathFlags::stablePlaneBaseScatterDiff),
                        path.hasFlag(PathFlags::stablePlaneOnDeltaBranch),
                        path.hasFlag(PathFlags::stablePlaneOnDominantBranch));
                    path.setFlag(PathFlags::stablePlaneOnPlane, true);
                    path.setFlag(PathFlags::stablePlaneOnDeltaBranch, false);
                    path.setStablePlaneIndex(spi);
                    path.setFlag(PathFlags::stablePlaneOnDominantBranch, spi == workingContext.stablePlanes.LoadDominantIndexCenter());
                    path.setCounter(PackedCounters::BouncesFromStablePlane, 0);
                    path.denoiserSampleHitTFromPlane    = 0.0;
                    path.denoiserDiffRadianceHitDist    = lpfloat4(0,0,0,0);
                    path.denoiserSpecRadianceHitDist    = lpfloat4(0,0,0,0);
                    path.secondaryL                     = 0;
                    onStablePath = true;
                    break;
                }

                const uint planeVertexIndex = StablePlanesVertexIndexFromBranchID(planeBranchID);

                onStablePath |= StablePlaneIsOnStablePath(planeBranchID, StablePlanesVertexIndexFromBranchID(planeBranchID), path.stableBranchID, nextVertexIndex);
            }
            path.setFlag(PathFlags::stablePlaneOnBranch, onStablePath);
        }
        else
        {
            // if we fell off the path, we stay on the last stable plane index and just keep depositing radiance there
            path.stableBranchID = cStablePlaneInvalidBranchID;
            path.setFlag(PathFlags::stablePlaneOnBranch, false);
            path.incrementCounter(PackedCounters::BouncesFromStablePlane); 
        }
        if (!path.hasFlag(PathFlags::stablePlaneOnPlane))
            path.incrementCounter(PackedCounters::BouncesFromStablePlane);
    }

    inline void StablePlanesHandleNEE(const PathState preScatterPath, inout PathState outPath, lpfloat3 diffuseRadiance, lpfloat3 specularRadiance, lpfloat sampleDistance, const WorkingContext workingContext)
    {
        uint stablePlaneIndex = preScatterPath.getStablePlaneIndex();

        const int bouncesFromStablePlane = 1+preScatterPath.getCounter(PackedCounters::BouncesFromStablePlane); // since it's NEE, it's +1!
        float accSampleDistance = StablePlaneAccumulateSampleHitT( preScatterPath.denoiserSampleHitTFromPlane, sampleDistance, bouncesFromStablePlane, false );
        accSampleDistance = lpfloat( min(accSampleDistance, HLF_MAX) );

        lpfloat3 preScatterThpLP = (lpfloat3)preScatterPath.thp;
        diffuseRadiance *= preScatterThpLP;     // account for the path throughput: have to do it here since we're adding samples at different path vertices
        specularRadiance *= preScatterThpLP;    // account for the path throughput: have to do it here since we're adding samples at different path vertices

        if (preScatterPath.hasFlag(PathFlags::stablePlaneOnPlane)) // we cound use bouncesFromStablePlane==0 for perf here, but needs testing
            //(path.getVertexIndex() == captureVertexIndex)
        {

            // this is the NEE at the denoising base surface - we've got separate inputs for diffuse and specular lobes, so capture them separately
            // there is no need to do StablePlaneCombineWithHitTCompensation - this always happens first
            outPath.denoiserDiffRadianceHitDist = lpfloat4( diffuseRadiance, accSampleDistance );
            outPath.denoiserSpecRadianceHitDist = lpfloat4( specularRadiance, accSampleDistance );
        }
        else //if( !(workingContext.ptConsts.stablePlanesSkipIndirectNoiseP0 && preScatterPath.getStablePlaneIndex() == 0 && (bouncesFromStablePlane)>2) )
        {
            // this is one of the subsequent bounces past the denoising surface - it got scattered through a diffuse or specular lobe so we use that to determine where to capture
            float3 neeSum = diffuseRadiance+specularRadiance;
            outPath.secondaryL += neeSum;	// <- there's a bug somewhere on this codepath unfortunately
        }
    }
#endif

    inline bool StablePlanesHandleMiss(inout PathState path, float3 emission, const float3 rayOrigin, const float3 rayDir, const float rayTCurrent, uint SERSortKey, const WorkingContext workingContext)
    {
#if PATH_TRACER_MODE==PATH_TRACER_MODE_BUILD_STABLE_PLANES // build
        const uint vertexIndex = path.getVertexIndex();
        if (vertexIndex == 1)
            workingContext.stablePlanes.StoreFirstHitRayLengthAndClearDominantToZeroCenter(kMaxRayTravel);
        float3 motionVectors = Bridge::computeSkyMotionVector(workingContext.pixelPos);
        workingContext.stablePlanes.StoreStablePlaneCenter(path.getStablePlaneIndex(), vertexIndex, path.hitPacked, SERSortKey, path.stableBranchID, rayDir, path.sceneLength, path.thp, 
            motionVectors, 1, -rayDir, 1.xxx, 1.xxx, path.hasFlag(PathFlags::stablePlaneOnDominantBranch));
        return true; // collect the emission!
#elif PATH_TRACER_MODE==PATH_TRACER_MODE_FILL_STABLE_PLANES // fill

        const int bouncesFromStablePlane = path.getCounter(PackedCounters::BouncesFromStablePlane);
        if (workingContext.ptConsts.useReSTIRGI && bouncesFromStablePlane == 1 && path.hasFlag(PathFlags::stablePlaneOnDominantBranch))
        {
            // Imagine a position and orientation for the environment and store it for ReSTIR GI
            const float3 worldPos = rayOrigin + rayDir * kMaxSceneDistance;
            const float3 normal = -rayDir;
            Bridge::StoreSecondarySurfacePositionAndNormal(PathIDToPixel(path.id), worldPos, normal);
        }

        path.denoiserSampleHitTFromPlane = StablePlaneAccumulateSampleHitT( path.denoiserSampleHitTFromPlane, kMaxSceneDistance, bouncesFromStablePlane, path.isDeltaOnlyPath() );

        // an example of debugging path decomposition info for the specific pixel selected in the UI, at all touched bounces
        //if (workingContext.debug.IsDebugPixel())
        //    workingContext.debug.Print(path.getVertexIndex(), path.getStablePlaneIndex(), bouncesFromStablePlane, path.denoiserSampleHitTFromPlane, 42);

        if (!path.hasFlag(PathFlags::stablePlaneOnBranch)) // not on stable branch, we need to capture emission
        {
            path.secondaryL += emission * path.thp;
        }
        return false; // tell path tracer that we're handling this radiance (or it was handled in MODE==1)
#endif
    }
#endif

// used only for debug visualization
#if PATH_TRACER_MODE==PATH_TRACER_MODE_BUILD_STABLE_PLANES && ENABLE_DEBUG_DELTA_TREE_VIZUALISATION
inline void DeltaTreeVizHandleMiss(inout PathState path, const float3 rayOrigin, const float3 rayDir, const float rayTCurrent, const WorkingContext workingContext)
{
    if (path.hasFlag(PathFlags::deltaTreeExplorer))
    {
        DeltaTreeVizPathVertex info = DeltaTreeVizPathVertex::make(path.getVertexIndex(), path.stableBranchID, 0xFFFFFFFF, path.thp, 0.0, rayOrigin + rayDir * rayTCurrent, path.hasFlag(PathFlags::stablePlaneOnDominantBranch)); // empty info for sky
        workingContext.debug.DeltaTreeVertexAdd( info );
        return;
    }
}
inline void DeltaTreeVizHandleHit(inout PathState path, const float3 rayOrigin, const float3 rayDir, const float rayTCurrent, const SurfaceData bridgedData, bool rejectedFalseHit, bool hasFinishedSurfaceBounces, float volumeAbsorption, const WorkingContext workingContext)
{
    uint vertexIndex = path.getVertexIndex();
    if (rejectedFalseHit)
    {
        // just continue - it has already been updated with an offset
        PathPayload packedPayload = PathPayload::pack(path);
        workingContext.debug.DeltaSearchStackPush(packedPayload);
    }
    else
    {
        DeltaTreeVizPathVertex info = DeltaTreeVizPathVertex::make(vertexIndex, path.stableBranchID, bridgedData.shadingData.materialID, path.thp, volumeAbsorption, bridgedData.shadingData.posW, path.hasFlag(PathFlags::stablePlaneOnDominantBranch));

        bridgedData.bsdf.evalDeltaLobes(bridgedData.shadingData, info.deltaLobes, info.deltaLobeCount, info.nonDeltaPart);

        // use deltaTreeContinueRecursion to give up on searching after buffers are filled in; can easily happen in complex meshes with clean reflection/transmission materials
        bool deltaTreeContinueRecursion = workingContext.debug.DeltaTreeVertexAdd( info );
        deltaTreeContinueRecursion &= vertexIndex <= cStablePlaneMaxVertexIndex;

        if (!hasFinishedSurfaceBounces)
        {
            for (int i = info.deltaLobeCount-1; (i >= 0) && deltaTreeContinueRecursion; i--) // reverse-iterate to un-reverse outputs
            {
                DeltaLobe lobe = info.deltaLobes[i];

                if (luminance(path.thp*lobe.thp)>cDeltaTreeVizThpIgnoreThreshold)
                {
                    PathState deltaPath = PathTracer::SplitDeltaPath(path, rayDir, bridgedData, lobe, i, false, workingContext);
                    deltaPath.incrementCounter(PackedCounters::BouncesFromStablePlane); 

                    // update stable plane index state
                    deltaPath.setFlag(PathFlags::stablePlaneOnPlane, false); // assume we're no longer on stable plane
                    if (deltaPath.getVertexIndex() <= cStablePlaneMaxVertexIndex)
                        for (uint spi = 0; spi < cStablePlaneCount; spi++)
                        {
                            const uint planeBranchID = workingContext.stablePlanes.GetBranchIDCenter(spi);
                            if (planeBranchID != cStablePlaneInvalidBranchID && StablePlaneIsOnPlane(planeBranchID, deltaPath.stableBranchID))
                            {
                                deltaPath.setFlag(PathFlags::stablePlaneOnPlane, true);
                                deltaPath.setStablePlaneIndex(spi);
                                deltaPath.setCounter(PackedCounters::BouncesFromStablePlane, 0);

                                // picking dominant flag from the actual build pass stable planes to be faithful debug for the StablePlanes system, which executed before this
                                const uint stablePlaneIndex = deltaPath.getStablePlaneIndex();
                                const uint dominantSPIndex = workingContext.stablePlanes.LoadDominantIndexCenter();
                                deltaPath.setFlag(PathFlags::stablePlaneOnDominantBranch, stablePlaneIndex == dominantSPIndex && deltaPath.hasFlag(PathFlags::stablePlaneOnPlane) );
                            }
                        }
                    
                    deltaTreeContinueRecursion &= workingContext.debug.DeltaSearchStackPush(PathPayload::pack(deltaPath));
                }
            }
        }
    }
}
#endif    
}

#endif // __PATH_TRACER_STABLE_PLANES_HLSLI__
