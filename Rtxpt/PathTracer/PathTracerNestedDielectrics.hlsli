/*
* Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#ifndef __PATH_TRACER_NESTED_DIELECTRICS_HLSLI__ // using instead of "#pragma once" due to https://github.com/microsoft/DirectXShaderCompiler/issues/3943
#define __PATH_TRACER_NESTED_DIELECTRICS_HLSLI__

#include "PathTracerTypes.hlsli"

namespace PathTracer
{
    /** Compute index of refraction for medium on the outside of the current dielectric volume.
        \param[in] interiorList Interior list.
        \param[in] materialID Material ID of intersected material.
        \param[in] entering True if material is entered, false if material is left.
        \return Index of refraction.
    */
    inline lpfloat ComputeOutsideIoR(const InteriorList interiorList, const uint materialID, const bool entering)
    {
        // The top element holds the material ID of currently highest priority material.
        // This is the material on the outside when entering a new medium.
        uint outsideMaterialID = interiorList.getTopMaterialID();

        if (!entering)
        {
            // If exiting the currently highest priority material, look at the next element
            // on the stack to find out what is on the outside.
            if (outsideMaterialID == materialID) outsideMaterialID = interiorList.getNextMaterialID();
        }

        // If no material, assume the default IoR for vacuum.
        if (outsideMaterialID == InteriorList::kNoMaterial) return 1.f;

        // this is implemented in \Falcor\Scene\Material\MaterialSystem.hlsli 
        // and probably need to get ported to Bridge::XXX - yet to decide
        return Bridge::loadIoR(outsideMaterialID);
    }
    
    /** Handle hits on dielectrics.
    \return True if this is an valid intersection, false if it is rejected.
    */
    inline bool HandleNestedDielectrics(inout SurfaceData surfaceData, inout PathState path, const WorkingContext workingContext)
    {
        // Check for false intersections.
        uint nestedPriority = surfaceData.shadingData.mtl.getNestedPriority();
        if (!path.interiorList.isTrueIntersection(nestedPriority))
        {
            // If it is a false intersection, we reject the hit and continue the path
            // on the other side of the interface.
            // If the offset position is not quite large enough, due to self-intersections
            // it is possible we repeatedly hit the same surface and try to reject it.
            // This has happened in a few occasions with large transmissive triangles.
            // As a workaround, count number of rejected hits and terminate the path if too many.
            if (path.getCounter(PackedCounters::RejectedHits) < kMaxRejectedHits)
            {
#if 0 && ENABLE_DEBUG_VIZUALISATION // do debugging for rejected pixels too!
                if (workingContext.debug.IsDebugPixel())
                {
                    // IoR debugging - .x - "outside", .y - "interior", .z - frontFacing, .w - "eta" (eta is isFrontFace?outsideIoR/insideIoR:insideIoR/outsideIoR)
                    // workingContext.debug.Print(path.getVertexIndex()-1, float4(-42,-42,-42,-42) ); //float4(surfaceData.shadingData.IoR, surfaceData.interiorIoR, surfaceData.shadingData.frontFacing, surfaceData.bsdf.data.eta) );
                    // path segment
                    workingContext.debug.DrawLine(path.origin, surfaceData.shadingData.posW, 0.4.xxx, 0.1.xxx);
                    workingContext.debug.DrawLine(surfaceData.shadingData.posW, surfaceData.shadingData.posW + surfaceData.shadingData.T * workingContext.debug.LineScale()*0.2, float3(0.1, 0, 0), float3(0.5, 0, 0));
                    workingContext.debug.DrawLine(surfaceData.shadingData.posW, surfaceData.shadingData.posW + surfaceData.shadingData.B * workingContext.debug.LineScale()*0.2, float3(0, 0.1, 0), float3(0, 0.5, 0));
                    workingContext.debug.DrawLine(surfaceData.shadingData.posW, surfaceData.shadingData.posW + surfaceData.shadingData.N * workingContext.debug.LineScale()*0.2, float3(0, 0, 0.1), float3(0, 0, 0.5));
                }
#endif

                path.incrementCounter(PackedCounters::RejectedHits);
                path.interiorList.handleIntersection(surfaceData.shadingData.materialID, nestedPriority, surfaceData.shadingData.frontFacing);
                path.origin = surfaceData.shadingData.computeNewRayOrigin(false);
                path.decrementVertexIndex();
            }
            else
            {
                path.terminate();
            }
            return false;
        }

        // Compute index of refraction for medium on the outside.
        Bridge::updateOutsideIoR( surfaceData, ComputeOutsideIoR(path.interiorList, surfaceData.shadingData.materialID, surfaceData.shadingData.frontFacing) );

        return true;
    }
    
    /** Update dielectric stack after valid scatter.
    */
    inline void UpdateNestedDielectricsOnScatterTransmission(const ShadingData shadingData, inout PathState path, const WorkingContext workingContext)
    {
        if (!shadingData.mtl.isThinSurface())
        {
            uint nestedPriority = shadingData.mtl.getNestedPriority();
            path.interiorList.handleIntersection(shadingData.materialID, nestedPriority, shadingData.frontFacing);
            path.setInsideDielectricVolume(!path.interiorList.isEmpty());
        }
    }
}

#endif // __PATH_TRACER_NESTED_DIELECTRICS_HLSLI__
