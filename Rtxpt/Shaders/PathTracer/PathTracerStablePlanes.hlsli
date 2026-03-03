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

    inline void UpdatePathTravelledLengthOnly(inout PathState path, const float rayTCurrent);

#if PATH_TRACER_MODE==PATH_TRACER_MODE_BUILD_STABLE_PLANES || defined(__INTELLISENSE__)
    // splits out delta component, traces ray to next surface, saves the hit without any further processing
    // if verifyDominantFlag is true, will remove PathFlags::stablePlaneOnDominantBranch from result if not on dominant lobe (otherwise it stay as it was); if we're splitting more than 1 lobe then we can only follow one dominant so we must update - otherwise we can let the flag persist
    inline PathState SplitDeltaPath(const PathState oldPath, const float3 rayDir, const SurfaceData surfaceData, const DeltaLobe lobe, const uint deltaLobeIndex, const bool verifyDominantFlag, const WorkingContext workingContext)
    {
        const ShadingData shadingData = surfaceData.shadingData;

        // 1. first generate new virtual path - this is just for consistency with the rest of the code, most of it is unused and compiled out
        PathState newPath       = oldPath;
        // newPath.incrementVertexIndex(); <- not needed, happens in nextHit
        newPath.SetDir( lobe.dir );
        newPath.SetThp( newPath.GetThp() * lobe.thp );
        //newPath.pdf             = 0;
        newPath.SetOrigin( shadingData.computeNewRayOrigin(lobe.transmission==0) );  // bool param is viewside
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

#if RTXPT_NESTED_DIELECTRICS_QUALITY > 0 || defined(__INTELLISENSE__)
            // Update interior list and inside volume flag.
            if (!shadingData.mtl.isThinSurface())
            {
                uint nestedPriority = shadingData.mtl.getNestedPriority();
                newPath.interiorList.handleIntersection(shadingData.materialID, nestedPriority, shadingData.frontFacing);
                newPath.setInsideDielectricVolume(!newPath.interiorList.isEmpty());
            }
#endif
        }

        // Compute local transform (rotation component only) and apply to path transform (path.imageXform). This intentionally ignores curvature/skewing to avoid complexity and need for full 4x4 matrix.
        if (!newPath.GetMotionVectorSceneLength()!=0) // <- check if we've stopped updating transform!
        {
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
            newPath.SetImageXform( mul(newPath.GetImageXform(), localT) );
        }

        // Testing the xforms - virt should always transform to rayDir here
        // float3 virt = mul( (float3x3)newPath.imageXform, lobe.Wo );
        // if (workingContext.Debug.IsDebugPixel() && oldPath.getVertexIndex()==1)
        // {
        //     workingContext.Debug.Print(oldPath.getVertexIndex()+0, rayDir );
        //     workingContext.Debug.Print(oldPath.getVertexIndex()+1, lobe.Wo );
        //     workingContext.Debug.Print(oldPath.getVertexIndex()+2, virt );
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
    inline void StablePlanesHandleHit(inout PathState path, const float3 rayOrigin, const float3 rayDir, const float rayTCurrent, const WorkingContext workingContext, const SurfaceData surfaceData, float volumeAbsorption, float3 surfaceEmission, bool pathStopping)
    {
#if PATH_TRACER_MODE==PATH_TRACER_MODE_BUILD_STABLE_PLANES || defined(__INTELLISENSE__)// build
        const uint vertexIndex = path.getVertexIndex();
        const uint currentSPIndex = path.getStablePlaneIndex();
        const uint2 pixelPos = path.GetPixelPos();

        if (surfaceData.shadingData.mtl.isPSDBlockMotionVectorsAtSurface() && path.GetMotionVectorSceneLength() == 0)
        {
            path.SetMotionVectorSceneLength(path.GetSceneLength());
            // path.SetMotionVectorBounceNormal(surfaceData.shadingData.N);
        }

        if (vertexIndex == 1)
            workingContext.StablePlanes.StoreFirstHitRayLengthAndClearDominantToZero(pixelPos, path.GetSceneLength());

        bool setAsBase = true;    // if path no longer stable, stop and set as a base
        float passthroughOverride = 0.0;
        if ( (vertexIndex < workingContext.PtConsts.maxStablePlaneVertexDepth) && !pathStopping) // Note: workingContext.PtConsts.maxStablePlaneVertexDepth already includes cStablePlaneMaxVertexIndex and MaxBounceCount
        {
            DeltaLobe deltaLobes[cMaxDeltaLobes]; uint deltaLobeCount; float nonDeltaPart;
            surfaceData.bsdf.evalDeltaLobes(surfaceData.shadingData, deltaLobes, deltaLobeCount, nonDeltaPart);
            deltaLobeCount = max( cMaxDeltaLobes-1, deltaLobeCount );

            bool potentiallyVolumeTransmission = false;

            float pathThp = Average(path.GetThp());    // perhaps give it another 10% just to avoid blocking at the end of a dark volume since all other pixels will be dark

            const float nonDeltaIgnoreThreshold = (1e-5);
            const float deltaIgnoreThreshold    = (0.001f);   
            bool hasNonDeltaLobes = nonDeltaPart > nonDeltaIgnoreThreshold;
            passthroughOverride = saturate(1.0-nonDeltaPart*10.0); // if setting as base and no (or low) non-delta lobe, override denoising settings for better passthrough

#define USE_THP_PRIORITIZED_PRUNING 0

            // prune non-noticeable delta lobes
#if USE_THP_PRIORITIZED_PRUNING
            const float neverIgnoreThreshold = workingContext.PtConsts.stablePlanesSplitStopThreshold / float(vertexIndex); // TODO: add screen space dither to threshold?
            float nonZeroDeltaLobesThp[cMaxDeltaLobes];
#endif
            int nonZeroDeltaLobes[cMaxDeltaLobes]; for (int i = 0; i < cMaxDeltaLobes; i++ ) nonZeroDeltaLobes[i] = 0;
            int nonZeroDeltaLobeCount = 0;
            for (int k = 0; k < deltaLobeCount; k++)
            {
                DeltaLobe lobe = deltaLobes[k];
                const float thp = Average(lobe.thp);
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
            if (workingContext.Debug.IsDebugPixel() && vertexIndex == 1)
            {
                workingContext.Debug.Print( 0, deltaLobeCount );
                workingContext.Debug.Print( 1, float4( nonZeroDeltaLobeCount, nonZeroDeltaLobes[0], nonZeroDeltaLobes[1], nonZeroDeltaLobes[2] ) );
                //workingContext.Debug.Print( 2, float4( nonZeroDeltaLobeCount, nonZeroDeltaLobesThp[0], nonZeroDeltaLobesThp[1], nonZeroDeltaLobesThp[2] ) );
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
                bool allowPSR = workingContext.PtConsts.allowPrimarySurfaceReplacement && (nonZeroDeltaLobeCount == 1) && (currentSPIndex == 0) && !potentiallyVolumeTransmission;
                allowPSR &= !surfaceData.shadingData.mtl.isPSDBlockMotionVectorsAtSurface();
                bool canReuseExisting = (currentSPIndex != 0) && (nonZeroDeltaLobeCount > 0);
                canReuseExisting |= allowPSR;
                canReuseExisting &= !hasNonDeltaLobes;       // stop on any diffuse lobe

                int availablePlaneCount = 0; int availablePlanes[cStablePlaneCount];
            
                workingContext.StablePlanes.GetAvailableEmptyPlanes(pixelPos, availablePlaneCount, availablePlanes);

                // when "Dominant bounce" material setting is set to "None (Surface)", and we're already on a split (currentSPIndex > 0), then make sure we can't reuse current path to go on, it has to stop and collect guide buffers
                canReuseExisting &= (currentSPIndex == 0) || (surfaceData.shadingData.mtl.getPSDDominantDeltaLobeP1()>0);
            
                // an example of debugging path decomposition logic for the specific pixel selected in the UI, at the second bounce (vertex index 2)
                // if (workingContext.Debug.IsDebugPixel() && vertexIndex == 2)
                //     workingContext.Debug.Print( 0, currentSPIndex, availablePlaneCount, canReuseExisting, nonZeroDeltaLobeCount );

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
                    PathState splitPath = PathTracer::SplitDeltaPath(path, rayDir, surfaceData, deltaLobes[lobeToExplore], lobeToExplore, true, workingContext);
                    splitPath.setStablePlaneIndex(availablePlanes[i]);
                    workingContext.StablePlanes.StoreExplorationStart(pixelPos, availablePlanes[i], PathPayload::pack(splitPath).packed);

                    #if 0 // way of debugging stable plane index 1 / split path stuff
                    if (splitPath.getStablePlaneIndex() == 1)
                    {
                        // PathPayload pathPayload = PathPayload::pack(splitPath); 
                        // // const uint4 packed[5] = pathPayload.packed;
                        // 
                        // PathState testPath = PathPayload::unpack(pathPayload, PACKED_HIT_INFO_ZERO);

                        workingContext.Debug.DrawDebugViz( float4(DbgShowNormalSRGB(deltaLobes[lobeToExplore].dir), 1) ); 

                        if (workingContext.Debug.IsDebugPixel() /*&& vertexIndex == 2*/)
                            workingContext.Debug.Print( 0, float4( deltaLobes[lobeToExplore].dir, deltaLobes[lobeToExplore].probability ) );

                    }
                    #endif

                }

                // and use current path to reuse lobe
                if ( lobeForReuse != -1 )
                {
                    setAsBase = false;
                    // split and use current path to continue on the best lobe
                    path = PathTracer::SplitDeltaPath(path, rayDir, surfaceData, deltaLobes[lobeForReuse], lobeForReuse, nonZeroDeltaLobeCount>0, workingContext);
                }
            }
        }

        // we've reached the end of stable delta path exploration on this plane; figure out surface properties including motion vectors and store
        if (setAsBase)
        {
            // move surface world pos to first transform starting point reference frame; we then rotate it with the rotation-only imageXform, and place it back into worldspace (we used to have the whole transform but this takes too much space in payload...)
            const Ray cameraRay = Bridge::computeCameraRay( pixelPos );   // note: all realtime mode subSamples currently share same camera ray at subSampleIndex == 0 (otherwise denoising guidance buffers would be noisy)
            
            const float3x3 imageXform = path.GetImageXform();
    
            const bool blockedAtSurface = path.GetMotionVectorSceneLength() != 0;

            // this is the world position either directly or as seen through all the reflections & refractions (e.g. mirrored few times); it's enough to compute camera motion vectors
            float sceneLengthForMVs = (blockedAtSurface)?(path.GetMotionVectorSceneLength()):(path.GetSceneLength());
            float3 virtualWorldPos = cameraRay.origin + cameraRay.dir * sceneLengthForMVs;
            // this is actual world space motion - it's only good to compute motion vectors if not reflected/refracted...
            float3 worldMotion = surfaceData.prevPosW - surfaceData.shadingData.posW;
            // ...but if we apply all stacked reflection/refraction transforms to match virtualWorldPos, we can then use it to compute screen space motion vectors 
            float3 virtualWorldMotion = mul(imageXform, worldMotion);
            //if (path.GetMotionVectorSceneLength() != 0) // we zero the motion or not - one is better sometimes, the other other times :|
            //    virtualWorldMotion = 0;
            float3 motionVectors = Bridge::computeMotionVector(virtualWorldPos, virtualWorldPos+virtualWorldMotion);
            
            // denoising guide helpers
            float roughness     = saturate(surfaceData.bsdf.data.Roughness());
            float3 worldNormal  = surfaceData.shadingData.N;
            worldNormal = normalize(mul((float3x3)imageXform, worldNormal));

            float3 diffBSDFEstimate, specBSDFEstimate;
            //BSDFProperties bsdfProperties = surfaceData.bsdf.getProperties(surfaceData.shadingData);
            surfaceData.bsdf.estimateSpecDiffBSDF(diffBSDFEstimate, specBSDFEstimate, surfaceData.shadingData.N, surfaceData.shadingData.V);

            // if blockedAtSurface is true, one of the previous vertices landed on high curvature surface which breaks our motion vector calculations; 
            // so we stop calculating MVs at that surface and consider everything else as zero roughness specular for denoising purposes (as DLSS-RR can infer specular MVs)
            if (blockedAtSurface)
            {
                roughness *= kSpecularRoughnessThreshold * 0.95;         // these ensure spec MVs are used
                //specBSDFEstimate = saturate(specBSDFEstimate+0.2.xxx);  // these ensure spec MVs are used
            }
            
            bool isDominant = path.hasFlag(PathFlags::stablePlaneOnDominantBranch);
            workingContext.StablePlanes.StoreStablePlane(pixelPos, currentSPIndex, vertexIndex, rayOrigin, rayDir, path.stableBranchID, path.GetSceneLength(), rayTCurrent, path.GetThp(), 
                motionVectors, roughness, worldNormal, diffBSDFEstimate, specBSDFEstimate, isDominant, 0, 0); //path.flagsAndVertexIndex, path.packedCounters);

            if (isDominant)
            {
                Bridge::ExportSurface(path, surfaceData, sceneLengthForMVs, motionVectors );
                // path.setFlag( PathFlags::surfaceExported, true );
            }
            
            // since we're building the planes and we've found a base plane, terminate here and the nextHit contains logic for continuing from other split paths if any (enqueued with .StoreExplorationStart)
            path.terminate();
        }
#endif
    }

#if PATH_TRACER_MODE==PATH_TRACER_MODE_FILL_STABLE_PLANES || defined(__INTELLISENSE__) // fill only
    inline void StablePlanesOnScatter(inout PathState path, const BSDFSample bs, const WorkingContext workingContext)
    {
        const uint2 pixelPos = path.GetPixelPos();
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
                const uint planeBranchID = workingContext.StablePlanes.GetBranchID(pixelPos, spi);
                if (planeBranchID == cStablePlaneInvalidBranchID)
                    continue;

                // changing the stable plane for the future
                if (StablePlaneIsOnPlane(planeBranchID, path.stableBranchID))
                {
                    workingContext.StablePlanes.CommitDenoiserRadiance(path);
                    path.setStablePlaneIndex(spi);
                    path.setFlag(PathFlags::stablePlaneOnDominantBranch, spi == workingContext.StablePlanes.LoadDominantIndex(pixelPos));
                    path.setFlag(PathFlags::stablePlaneOnPlane, true);
                    path.setCounter(PackedCounters::BouncesFromStablePlane, 0);
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
#endif

    inline void StablePlanesHandleMiss(inout PathState path, float3 emission, const float3 rayOrigin, const float3 rayDir, const float rayTCurrent, const WorkingContext workingContext)
    {
        const uint2 pixelPos = path.GetPixelPos();
#if PATH_TRACER_MODE==PATH_TRACER_MODE_BUILD_STABLE_PLANES || defined(__INTELLISENSE__) // build; during build, every miss is a denoising layer base by traversal design, so store the properties
        const uint vertexIndex = path.getVertexIndex();
        if (vertexIndex == 1)
            workingContext.StablePlanes.StoreFirstHitRayLengthAndClearDominantToZero(pixelPos, kMaxRayTravel);
        
        const Ray cameraRay = Bridge::computeCameraRay( pixelPos );
        const bool blockedAtSurface = path.GetMotionVectorSceneLength() != 0;
        const float sceneLengthForMVs = (blockedAtSurface) ? (path.GetMotionVectorSceneLength()) : (kEnvironmentMapSceneDistance);
        const float3 virtualWorldPos = cameraRay.origin + cameraRay.dir * sceneLengthForMVs;
        float3 motionVectors = Bridge::computeMotionVector(virtualWorldPos, virtualWorldPos);

        bool isDominant = path.hasFlag(PathFlags::stablePlaneOnDominantBranch);
        
        // trick for RR - sky denoising needs some guide albedo
        float3 skyAlbedo = sqrt(ReinhardMax(emission));

        // if blockedAtSurface is true, one of the previous vertices landed on high curvature surface which breaks our motion vector calculations; 
        // so we stop calculating MVs at that surface and consider everything else as zero roughness specular for denoising purposes (as DLSS-RR can infer specular MVs)
        workingContext.StablePlanes.StoreStablePlane(pixelPos, path.getStablePlaneIndex(), vertexIndex, rayOrigin, rayDir, path.stableBranchID, (blockedAtSurface)?(sceneLengthForMVs):(1.#INF), 0, // inf indicates MISS
            path.GetThp(), motionVectors, (blockedAtSurface)?0.1:1, -rayDir, skyAlbedo, (blockedAtSurface)?0.5.xxx:0.xxx, path.hasFlag(PathFlags::stablePlaneOnDominantBranch), 0, 0);

        if (isDominant)
        {
            Bridge::ExportNonSurface(path, virtualWorldPos, motionVectors );
            // path.setFlag( PathFlags::surfaceExported, true );
        }

#endif
    }
#endif

// used only for debug visualization
#if PATH_TRACER_MODE==PATH_TRACER_MODE_BUILD_STABLE_PLANES && ENABLE_DEBUG_DELTA_TREE_VIZUALISATION 
inline void DeltaTreeVizHandleMiss(inout PathState path, const float3 rayOrigin, const float3 rayDir, const float rayTCurrent, const WorkingContext workingContext)
{
    if (path.hasFlag(PathFlags::deltaTreeExplorer))
    {
        DeltaTreeVizPathVertex info = DeltaTreeVizPathVertex::make(path.getVertexIndex(), path.stableBranchID, 0xFFFFFFFF, path.GetThp(), 0.0, rayOrigin + rayDir * rayTCurrent, path.hasFlag(PathFlags::stablePlaneOnDominantBranch)); // empty info for sky
        workingContext.Debug.DeltaTreeVertexAdd( info );
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
        workingContext.Debug.DeltaSearchStackPush(packedPayload);
    }
    else
    {
        DeltaTreeVizPathVertex info = DeltaTreeVizPathVertex::make(vertexIndex, path.stableBranchID, bridgedData.shadingData.materialID, path.GetThp(), volumeAbsorption, bridgedData.shadingData.posW, path.hasFlag(PathFlags::stablePlaneOnDominantBranch));

        bridgedData.bsdf.evalDeltaLobes(bridgedData.shadingData, info.deltaLobes, info.deltaLobeCount, info.nonDeltaPart);

        // use deltaTreeContinueRecursion to give up on searching after buffers are filled in; can easily happen in complex meshes with clean reflection/transmission materials
        bool deltaTreeContinueRecursion = workingContext.Debug.DeltaTreeVertexAdd( info );
        deltaTreeContinueRecursion &= vertexIndex <= cStablePlaneMaxVertexIndex;

        if (!hasFinishedSurfaceBounces)
        {
            for (int i = info.deltaLobeCount-1; (i >= 0) && deltaTreeContinueRecursion; i--) // reverse-iterate to un-reverse outputs
            {
                DeltaLobe lobe = info.deltaLobes[i];

                if (luminance(path.GetThp()*lobe.thp)>cDeltaTreeVizThpIgnoreThreshold)
                {
                    PathState deltaPath = PathTracer::SplitDeltaPath(path, rayDir, bridgedData, lobe, i, false, workingContext);
                    deltaPath.incrementCounter(PackedCounters::BouncesFromStablePlane); 

                    // update stable plane index state
                    deltaPath.setFlag(PathFlags::stablePlaneOnPlane, false); // assume we're no longer on stable plane
                    if (deltaPath.getVertexIndex() <= cStablePlaneMaxVertexIndex)
                        for (uint spi = 0; spi < cStablePlaneCount; spi++)
                        {
                            const uint planeBranchID = workingContext.StablePlanes.GetBranchID(pixelPos, spi);
                            if (planeBranchID != cStablePlaneInvalidBranchID && StablePlaneIsOnPlane(planeBranchID, deltaPath.stableBranchID))
                            {
                                deltaPath.setFlag(PathFlags::stablePlaneOnPlane, true);
                                deltaPath.setStablePlaneIndex(spi);
                                deltaPath.setCounter(PackedCounters::BouncesFromStablePlane, 0);

                                // picking dominant flag from the actual build pass stable planes to be faithful debug for the StablePlanes system, which executed before this
                                const uint stablePlaneIndex = deltaPath.getStablePlaneIndex();
                                const uint dominantSPIndex = workingContext.StablePlanes.LoadDominantIndex(pixelPos);
                                deltaPath.setFlag(PathFlags::stablePlaneOnDominantBranch, stablePlaneIndex == dominantSPIndex && deltaPath.hasFlag(PathFlags::stablePlaneOnPlane) );
                            }
                        }
                    
                    deltaTreeContinueRecursion &= workingContext.Debug.DeltaSearchStackPush(PathPayload::pack(deltaPath));
                }
            }
        }
    }
}
#endif    
}

#endif // __PATH_TRACER_STABLE_PLANES_HLSLI__
