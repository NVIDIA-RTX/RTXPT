/*
* Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#ifndef __MATERIAL_TYPES_HLSLI__ // using instead of "#pragma once" due to https://github.com/microsoft/DirectXShaderCompiler/issues/3943
#define __MATERIAL_TYPES_HLSLI__

#include "../Config.h"

// rename to MaterialPropertiesAtSurface?
struct MaterialProperties
{
    float3 shadingNormal;
    float3 geometryNormal;
    lpfloat3 diffuseAlbedo; // BRDF input Cdiff
    lpfloat3 specularF0; // BRDF input F0
    lpfloat3 emissiveColor;
    lpfloat opacity;
    //lpfloat occlusion;
    lpfloat roughness;
    lpfloat3 baseColor; // native in metal-rough, derived in spec-gloss
    lpfloat metalness; // native in metal-rough, derived in spec-gloss
    lpfloat transmission;
    lpfloat diffuseTransmission;
    lpfloat ior;
    lpfloat shadowNoLFadeout;
    uint flags;

    static MaterialProperties make()
    {
        MaterialProperties result;
        result.shadingNormal = 0;
        result.geometryNormal = 0;
        result.diffuseAlbedo = 0;
        result.specularF0 = 0;
        result.emissiveColor = 0;
        result.opacity = 1;
        //result.occlusion = 1;
        result.roughness = 0;
        result.baseColor = 0;
        result.metalness = 0;
        result.transmission = 0;
        result.diffuseTransmission = 0;
        result.ior = 1.5;
        result.flags = 0;
        return result;
    }
};

#endif // __MATERIAL_TYPES_HLSLI__