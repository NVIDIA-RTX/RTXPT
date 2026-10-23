/*
* Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#ifndef __PATH_TRACER_DENOISING_LAYERS_HLSLI__ // using instead of "#pragma once" due to https://github.com/microsoft/DirectXShaderCompiler/issues/3943
#define __PATH_TRACER_DENOISING_LAYERS_HLSLI__

#include "PathTracerTypes.hlsli"

#include "DenoisingLayers.hlsli"

namespace PathTracer
{

    inline void DenoisingLayersHandleMiss(inout PathState path, float3 emission, const float3 rayOrigin, const float3 rayDir, const float rayTCurrent, const WorkingContext workingContext)
    {
        const uint2 pixelPos = path.GetPixelPos();
#if PATH_TRACER_MODE==PT_DENOISING_LAYERS_BUILD || defined(__INTELLISENSE__) // build; during build, every miss is a denoising layer base by traversal design, so store the properties
        const uint vertexIndex = path.getVertexIndex();
        if (vertexIndex == 1)
            workingContext.DenoisingLayers.StoreFirstHitRayLengthAndClearDominantToZero(pixelPos, kMaxRayTravel);
        
        const Ray cameraRay = Bridge::computeCameraRay( pixelPos );
        const bool blockedAtSurface = path.GetMotionVectorSceneLength() != 0;
        const float sceneLengthForMVs = (blockedAtSurface) ? (path.GetMotionVectorSceneLength()) : (kEnvironmentMapSceneDistance);
        const float3 virtualWorldPos = cameraRay.origin + cameraRay.dir * sceneLengthForMVs;
        float3 motionVectors = Bridge::computeMotionVector(virtualWorldPos, virtualWorldPos);

        bool isDominant = path.hasFlag(PathFlags::denoisingLayerOnDominantBranch);
        
        // trick for RR - sky denoising needs some guide albedo
        float3 skyAlbedo = sqrt(ReinhardMax(emission));

        // if blockedAtSurface is true, one of the previous vertices landed on high curvature surface which breaks our motion vector calculations; 
        // so we stop calculating MVs at that surface and consider everything else as zero roughness specular for denoising purposes (as DLSS-RR can infer specular MVs)
        workingContext.DenoisingLayers.StoreDenoisingLayer(pixelPos, path.getDenoisingLayerIndex(), vertexIndex, rayOrigin, rayDir, path.stableBranchID, (blockedAtSurface)?(sceneLengthForMVs):(1.#INF), 0, // inf indicates MISS
            path.GetThp(), float3(0,0,0), (blockedAtSurface)?0.1:1, -rayDir, skyAlbedo, (blockedAtSurface)?0.5.xxx:0.xxx, path.hasFlag(PathFlags::denoisingLayerOnDominantBranch), 0, 0);

        if (isDominant)
        {
            Bridge::ExportNonSurface(path, virtualWorldPos, motionVectors );
            // path.setFlag( PathFlags::surfaceExported, true );
        }

#endif
    }

}

#endif // __PATH_TRACER_DENOISING_LAYERS_HLSLI__
