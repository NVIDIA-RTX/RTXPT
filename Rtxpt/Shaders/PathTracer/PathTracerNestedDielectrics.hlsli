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
#include "Surface/Surface.hlsli"

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
    inline bool HandleNestedDielectrics(inout Surface surface, inout PathState path, const WorkingContext workingContext)
    {
#if RTXPT_NESTED_DIELECTRICS_QUALITY == 1
    static const uint           kMaxRejectedDielectricHits      = 4;    // Maximum number of rejected hits along a path (PackedCounters::RejectedHits counter, used by nested dielectrics). The path is terminated if the limit is reached to avoid getting stuck in pathological cases.
    #define NESTED_DIELECTRICS_AVOID_TERMINATION 1                      // simply revert back to non-nested dielectrics when we hit the rejected hit limit - we get approx result and there's less code
#elif RTXPT_NESTED_DIELECTRICS_QUALITY == 2
    static const uint           kMaxRejectedDielectricHits      = 16;   // Maximum number of rejected hits along a path (PackedCounters::RejectedHits counter, used by nested dielectrics). The path is terminated if the limit is reached to avoid getting stuck in pathological cases.
    #define NESTED_DIELECTRICS_AVOID_TERMINATION 0
#endif

#if RTXPT_NESTED_DIELECTRICS_QUALITY > 0 || defined(__INTELLISENSE__)

        if (surface.shadingData.shadingFlags.isThinSurface())
            return true; 

        // Check for false intersections.
        uint nestedPriority = surface.shadingData.shadingFlags.getNestedPriority();
        if (
#if NESTED_DIELECTRICS_AVOID_TERMINATION
            path.getCounter(PackedCounters::RejectedHits) < kMaxRejectedDielectricHits && 
#endif
            !path.interiorList.isTrueIntersection(nestedPriority))
        {
            // If it is a false intersection, we reject the hit and continue the path
            // on the other side of the interface.
            // If the offset position is not quite large enough, due to self-intersections
            // it is possible we repeatedly hit the same surface and try to reject it.
            // This has happened in a few occasions with large transmissive triangles.
            // As a workaround, count number of rejected hits and terminate the path if too many.
#if !NESTED_DIELECTRICS_AVOID_TERMINATION
            if (path.getCounter(PackedCounters::RejectedHits) < kMaxRejectedDielectricHits)
#endif
            {
#if 0 && ENABLE_DEBUG_VIZUALISATIONS && ENABLE_DEBUG_LINES_VIZ // do debugging for rejected pixels too!
                if (workingContext.Debug.IsDebugPixel(path.GetPixelPos()))
                {
                    // IoR debugging - .x - "outside", .y - "interior", .z - frontFacing, .w - "eta" (eta is isFrontFace?outsideIoR/insideIoR:insideIoR/outsideIoR)
                    // workingContext.Debug.Print(path.getVertexIndex()-1, float4(-42,-42,-42,-42) ); //float4(surface.shadingData.exteriorIoR, surface.shadingData.interiorIoR, surface.shadingData.frontFacing, surface.bsdf.data.eta) );
                    // path segment
                    workingContext.Debug.DrawLine(path.origin-, surface.shadingData.posW, 0.4.xxx, 0.1.xxx);
                    workingContext.Debug.DrawLine(surface.shadingData.posW, surface.shadingData.posW + surface.shadingData.T() * workingContext.Debug.LineScale()*0.2, float3(0.1, 0, 0), float3(0.5, 0, 0));
                    workingContext.Debug.DrawLine(surface.shadingData.posW, surface.shadingData.posW + surface.shadingData.B() * workingContext.Debug.LineScale()*0.2, float3(0, 0.1, 0), float3(0, 0.5, 0));
                    workingContext.Debug.DrawLine(surface.shadingData.posW, surface.shadingData.posW + surface.shadingData.N * workingContext.Debug.LineScale()*0.2, float3(0, 0, 0.1), float3(0, 0, 0.5));
                }
#endif

                path.incrementCounter(PackedCounters::RejectedHits);
                path.interiorList.handleIntersection(surface.shadingData.MaterialID(), nestedPriority, surface.shadingData.frontFacing);
                path.SetOrigin( surface.shadingData.ComputeNewRayOrigin(false) );
                path.decrementVertexIndex();
            }
#if !NESTED_DIELECTRICS_AVOID_TERMINATION
            else
            {
                path.terminate();
            }
#endif
            return false;
        }

        // Compute index of refraction for medium on the outside.
        surface.shadingData.SetExteriorIoR(ComputeOutsideIoR(path.interiorList, surface.shadingData.MaterialID(), surface.shadingData.frontFacing));

#endif // #if RTXPT_NESTED_DIELECTRICS_QUALITY > 0 || defined(__INTELLISENSE__)

        return true;
    }
    
    /** Update dielectric stack after valid scatter.
    */
    inline void UpdateNestedDielectricsOnScatterTransmission(const ShadingData shadingData, inout PathState path, const WorkingContext workingContext)
    {
#if RTXPT_NESTED_DIELECTRICS_QUALITY > 0 || defined(__INTELLISENSE__)
        if (!shadingData.shadingFlags.isThinSurface())
        {
            uint nestedPriority = shadingData.shadingFlags.getNestedPriority();
            path.interiorList.handleIntersection(shadingData.MaterialID(), nestedPriority, shadingData.frontFacing);
            path.setInsideDielectricVolume(!path.interiorList.isEmpty());
        }
#endif // #if RTXPT_NESTED_DIELECTRICS_QUALITY > 0 || defined(__INTELLISENSE__)
    }

}

#endif // __PATH_TRACER_NESTED_DIELECTRICS_HLSLI__
