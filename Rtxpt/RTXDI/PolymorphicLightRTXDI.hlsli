/*
* Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#ifndef POLYMORPHIC_LIGHT_RTXDI_HLSLI
#define POLYMORPHIC_LIGHT_RTXDI_HLSLI

#include "HelperFunctions.hlsli"
#include "../PathTracer/Utils/Utils.hlsli"
#include "../PathTracer/Lighting/LightShaping.hlsli"
#include "../PathTracer/Utils/ColorHelpers.hlsli"
#include "../PathTracer/Lighting/EnvMap.hlsli"
#include "../PathTracer/PathTracerHelpers.hlsli"
#include <Rtxdi/Utils/Checkerboard.hlsli>

// This is a adapter for PolymorphicLight, enabling features as needed by RTXDI

// Polymorphic light config - RTXDI will also need ENV 
#define POLYLIGHT_SPHERE_ENABLE         1
#define POLYLIGHT_POINT_ENABLE          1
#define POLYLIGHT_TRIANGLE_ENABLE       1
#define POLYLIGHT_DIRECTIONAL_ENABLE    1   // probably not needed as baked in envmap but this need testing
#define POLYLIGHT_ENV_ENABLE            1
#define POLYLIGHT_QT_ENV_ENABLE         0   // environment map quad tree in equal area octahedral mapping
#define POLYLIGHT_CONFIGURED

#include "../PathTracer/Lighting/PolymorphicLight.hlsli"


PolymorphicLightSample EnvironmentLight::CalcSample(in const float2 random, in const float3 viewerPosition)
{
    PolymorphicLightSample pls;
       
    EnvMapSampler envMapSampler = EnvMapSampler::make( s_EnvironmentMapImportanceSampler, t_EnvironmentMapImportanceMap, g_Const.envMapImportanceSamplingParams,
                                                        t_EnvironmentMap, s_EnvironmentMapSampler, g_Const.envMapSceneParams );

    float3 worldDir = Decode_Oct( random );
    pls.Position = viewerPosition + worldDir * DISTANT_LIGHT_DISTANCE;
    pls.Normal = -worldDir;
    pls.Radiance = envMapSampler.Eval(worldDir);
    pls.SolidAnglePdf = envMapSampler.MIPDescentEvalPdf(worldDir);
        
    return pls;
}    

inline EnvironmentLight EnvironmentLight::Create(in const PolymorphicLightInfoFull lightInfo)
{
    EnvironmentLight envLight;

    return envLight;
}

#endif // POLYMORPHIC_LIGHT_RTXDI_HLSLI