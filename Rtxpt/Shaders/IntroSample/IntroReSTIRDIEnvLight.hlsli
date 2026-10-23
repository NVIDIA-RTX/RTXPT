/*
* Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#ifndef __INTRO_RESTIR_DI_ENV_LIGHT_HLSLI__
#define __INTRO_RESTIR_DI_ENV_LIGHT_HLSLI__

// Environment-map support for ReSTIR DI (opt-in via INTRO_RTXDI_ENABLE_ENV_LIGHT, which also turns on
// POLYLIGHT_ENV_ENABLE in IntroRTXDIBridge.hlsli). The environment is treated as one synthetic
// kEnvironment light appended to the light buffer by PrepareLights; this header supplies the polymorphic
// EnvironmentLight::CalcSample implementation (declared but intentionally left undefined in
// PolymorphicLight.hlsli, since the sampler isn't packed in the light record) plus the importance-map
// resource the intro samples with. Include AFTER PolymorphicLight.hlsli.

#if INTRO_RTXDI_ENABLE_ENV_LIGHT

#include "../Bindings/SamplerBindings.hlsli"          // s_EnvironmentMapSampler (s1), s_EnvironmentMapImportanceSampler (s2)
#include "../PathTracer/Utils/Utils.hlsli"            // Encode_Oct / Decode_Oct
#include "../PathTracer/Lighting/EnvMap.hlsli"        // EnvMapSampler

// Hierarchical importance map (whole mip chain). Already bound by the shared binding set (Sample.cpp) to
// EnvMapBaker's importance map at t11; the intro just declares it here so its ReSTIR passes can sample it.
Texture2D<float> t_EnvironmentMapImportanceMap : register(t11);

// Build the MIP-descent env sampler from the already-bound importance map / env cube and the
// constant-buffer params. t_EnvironmentMap (t10) comes from IntroLightBindings.hlsli.
EnvMapSampler IntroMakeEnvMapSampler()
{
    return EnvMapSampler::make(
        s_EnvironmentMapImportanceSampler, t_EnvironmentMapImportanceMap, g_Const.envMapImportanceSamplingParams,
        t_EnvironmentMap, s_EnvironmentMapSampler, g_Const.envMapSceneParams);
}

// PolymorphicLight interface implementation. Environment sampling data is not packed in the light record,
// so CalcSample reconstructs the direction from the reservoir UV (octahedral) and evaluates the globally
// bound importance map / env cube. Mirrors the advanced sample's PolymorphicLightRTXDI.hlsli.
PolymorphicLightSample EnvironmentLight::CalcSample(in const float2 random, in const float3 viewerPosition)
{
    EnvMapSampler envMapSampler = IntroMakeEnvMapSampler();
    float3 worldDir = Decode_Oct(random);

    PolymorphicLightSample pls;
    pls.Position = viewerPosition + worldDir * DISTANT_LIGHT_DISTANCE;
    pls.Normal = -worldDir;
    pls.Radiance = envMapSampler.Eval(worldDir);
    pls.SolidAnglePdf = envMapSampler.MIPDescentEvalPdf(worldDir);
    return pls;
}

EnvironmentLight EnvironmentLight::Create(in const PolymorphicLightInfoFull lightInfo)
{
    EnvironmentLight envLight;
    return envLight;
}

#endif // INTRO_RTXDI_ENABLE_ENV_LIGHT
#endif // __INTRO_RESTIR_DI_ENV_LIGHT_HLSLI__
