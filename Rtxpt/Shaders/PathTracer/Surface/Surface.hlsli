/*
* Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#ifndef __SURFACE_HLSLI__ // using instead of "#pragma once" due to https://github.com/microsoft/DirectXShaderCompiler/issues/3943
#define __SURFACE_HLSLI__

#include "../Config.h"    

#include "ShadingData.hlsli"

#include "../BSDFs/StandardBSDF.hlsli"

#ifndef ActiveBSDF
#error This file requires BSDF specialization config - for example, "#define ActiveBSDF StandardBSDF"
#endif

namespace PathTracer
{
    /** All surface data returned by the Bridge::loadSurface
    */
    struct Surface
    {
        ShadingData     shadingData;
        ActiveBSDF      bsdf;
//#if !RTXPT_USE_APPROXIMATE_MIS
        uint            neeTriangleLightIndex;  // 0xFFFFFFFF if none
//#endif
        uint            neeAnalyticLightIndex;  // 0xFFFFFFFF if none
        static Surface make( ShadingData shadingData, ActiveBSDF bsdf,
                                uint neeTriangleLightIndex, uint neeAnalyticLightIndex )
        {
            Surface ret;
            ret.shadingData             = shadingData;
            ret.bsdf                    = bsdf;
//#if !RTXPT_USE_APPROXIMATE_MIS
            ret.neeTriangleLightIndex   = neeTriangleLightIndex;
//#endif
            ret.neeAnalyticLightIndex   = neeAnalyticLightIndex;
            return ret; 
        }
    };
}

#endif // __SURFACE_HLSLI__