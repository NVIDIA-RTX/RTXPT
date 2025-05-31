/*
* Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#ifndef __RAY_HLSLI__ // using instead of "#pragma once" due to https://github.com/microsoft/DirectXShaderCompiler/issues/3943
#define __RAY_HLSLI__

#include "MathConstants.hlsli"

/** Ray type.
    This has equal layout to DXR RayDesc but adds additional functionality.
*/
struct Ray
{
    float3 origin;
    float tMin;
    float3 dir;
    float tMax;

    /** Initializes a ray.
    */
    void __init(float3 origin, float3 dir, float tMin, float tMax)
    {
        this.origin = origin;
        this.dir = dir;
        this.tMin = tMin;
        this.tMax = tMax;
    }
    static Ray make(float3 origin, float3 dir, float tMin = 0.f, float tMax = FLT_MAX)
    {
        Ray ret; 
        ret.__init(origin, dir, tMin, tMax);
        return ret;
    }

    /** Convert to DXR RayDesc.
    */
    RayDesc toRayDesc()
    {
        // return { origin, tMin, dir, tMax }; // error G3B4BA6C5: generalized initializer lists are incompatible with HLSL
        RayDesc ret; ret.Origin = origin; ret.TMin = tMin; ret.Direction = dir; ret.TMax = tMax;
        return ret; 
    }

    /** Evaluate position on the ray.
        \param[in] t Ray parameter.
        \return Returns evaluated position.
    */
    float3 eval(float t)
    {
        return origin + t * dir;
    }
};

#endif // __RAY_HLSLI__