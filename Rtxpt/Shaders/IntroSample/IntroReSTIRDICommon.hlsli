/*
* Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#ifndef __INTRO_RESTIR_DI_COMMON_HLSLI__
#define __INTRO_RESTIR_DI_COMMON_HLSLI__

#include <Rtxdi/DI/Reservoir.hlsli>
#include <Rtxdi/DI/ReservoirStorage.hlsli>
#include <Rtxdi/DI/InitialSampling.hlsli>

// Local utils

RAB_LightInfo RAB_LoadTriangleLightInfo(uint index, bool previousFrame)
{
    return PolymorphicLightInfoFull::make(t_IntroRTXDILightsBase[index]);
}

void RTXDI_RandomlySelectAreaLightUniformly(
    float rnd,
    RTXDI_LightBufferRegion region,
    out RAB_LightInfo lightInfo,
    out uint lightIndex,
    out float invSourcePdf)
{
    invSourcePdf = float(region.numLights);
    lightIndex = region.firstLightIndex + min(uint(floor(rnd * region.numLights)), region.numLights - 1);
    lightInfo = RAB_LoadTriangleLightInfo(lightIndex, false);
}

void RTXDI_SelectNextLocalAreaLight(
    RTXDI_LocalLightSelectionContext ctx,
    float rnd,
    out RAB_LightInfo lightInfo,
    out uint lightIndex,
    out float invSourcePdf)
{
    switch (ctx.mode)
    {
#if RTXDI_ENABLE_PRESAMPLING
    case RTXDI_LocalLightContextSamplingMode_RIS:
        RTXDI_RandomlySelectLocalLightFromRISTile(rnd, ctx.risTileInfo, lightInfo, lightIndex, invSourcePdf);
        break;
#endif // RTXDI_ENABLE_PRESAMPLING
    default:
    case RTXDI_LocalLightContextSamplingMode_UNIFORM:
        RTXDI_RandomlySelectAreaLightUniformly(rnd, ctx.lightBufferRegion, lightInfo, lightIndex, invSourcePdf);
        break;
    }
}

RAB_LightSample RAB_SampleAreaLight(RAB_LightInfo lightInfo, RAB_Surface surface, float2 uv)
{
    PolymorphicLightSample pls = TriangleLight::Create(lightInfo).CalcSample(uv, surface.worldPos); // TriangleLight::CalcSample(lightInfo, uv, surface.worldPos);

    RAB_LightSample lightSample;
    lightSample.position = pls.Position;
    lightSample.normal = pls.Normal;
    lightSample.radiance = pls.Radiance;
    lightSample.solidAnglePdf = pls.SolidAnglePdf;
    lightSample.lightType = PolymorphicLightType::kTriangle;// PolymorphicLight::DecodeType(lightInfo);
    return lightSample;
}

bool RTXDI_StreamLocalAreaLightAtUVIntoReservoir(
    inout RTXDI_RandomSamplerState rng,
    RTXDI_InitialSamplingMisData misData,
    RAB_Surface surface,
    float brdfCutoff,
    float localLightMisWeight,
    uint lightIndex,
    float2 uv,
    float invSourcePdf,
    RAB_LightInfo lightInfo,
    inout RTXDI_DIReservoir state,
    inout RAB_LightSample o_selectedSample)
{
    RAB_LightSample candidateSample = RAB_SampleAreaLight(lightInfo, surface, uv);
    float blendedSourcePdf = RTXDI_LightBrdfMisWeight(surface, candidateSample, 1.0 / invSourcePdf,
        misData.localLightMisWeight, false, misData.brdfMisWeight, brdfCutoff);
    float targetPdf = RAB_GetLightSampleTargetPdfForSurface(candidateSample, surface);
    float risRnd = RTXDI_GetNextRandom(rng);

    if (blendedSourcePdf == 0)
    {
        return false;
    }
    bool selected = RTXDI_StreamSample(state, lightIndex, uv, risRnd, targetPdf, 1.0 / blendedSourcePdf);

    if (selected) {
        o_selectedSample = candidateSample;
    }
    return true;
}

RTXDI_DIReservoir RTXDI_SampleLocalAreaLightsInternal(
    inout RTXDI_RandomSamplerState rng,
    inout RTXDI_RandomSamplerState coherentRng,
    RAB_Surface surface,
    RTXDI_DIInitialSamplingParameters sampleParams,
	RTXDI_InitialSamplingMisData misData,
    ReSTIRDI_LocalLightSamplingMode localLightSamplingMode,
    RTXDI_LightBufferRegion localLightBufferRegion,
#if RTXDI_ENABLE_PRESAMPLING
    RTXDI_RISBufferSegmentParameters localLightRISBufferSegmentParams,
#if RTXDI_REGIR_MODE != RTXDI_REGIR_DISABLED
    ReGIR_Parameters regirParams,
#endif
#endif
    out RAB_LightSample o_selectedSample)
{
    RTXDI_DIReservoir state = RTXDI_EmptyDIReservoir();

    RTXDI_LocalLightSelectionContext lightSelectionContext = RTXDI_InitializeLocalLightSelectionContext(coherentRng, localLightSamplingMode, localLightBufferRegion
#if RTXDI_ENABLE_PRESAMPLING
    ,localLightRISBufferSegmentParams
#if RTXDI_REGIR_MODE != RTXDI_REGIR_DISABLED
    ,regirParams
    ,surface
#endif
#endif
    );

    float rnd = RTXDI_GetNextRandom(rng) / sampleParams.numLocalLightSamples;

    for (uint i = 0; i < sampleParams.numLocalLightSamples; i++)
    {
        uint lightIndex;
        RAB_LightInfo lightInfo;
        float invSourcePdf;

        RTXDI_SelectNextLocalAreaLight(lightSelectionContext, rnd, lightInfo, lightIndex, invSourcePdf);
        float2 uv = RTXDI_RandomlySelectLocalLightUV(rng);
        RTXDI_StreamLocalAreaLightAtUVIntoReservoir(rng, misData, surface, sampleParams.brdfCutoff, misData.localLightMisWeight, lightIndex, uv, invSourcePdf, lightInfo, state, o_selectedSample);

        rnd += 1.0f / sampleParams.numLocalLightSamples;
    }

    RTXDI_FinalizeResampling(state, 1.0, misData.numMisSamples);
    state.M = 1;

    return state;
}

RTXDI_DIReservoir RTXDI_SampleLocalAreaLights(
    inout RTXDI_RandomSamplerState rng,
    inout RTXDI_RandomSamplerState coherentRng,
    RAB_Surface surface,
    RTXDI_DIInitialSamplingParameters sampleParams,
	RTXDI_InitialSamplingMisData misData,
    ReSTIRDI_LocalLightSamplingMode localLightSamplingMode,
    RTXDI_LightBufferRegion localLightBufferRegion,
#if RTXDI_ENABLE_PRESAMPLING
    RTXDI_RISBufferSegmentParameters localLightRISBufferSegmentParams,
#if RTXDI_REGIR_MODE != RTXDI_REGIR_DISABLED
    ReGIR_Parameters regirParams,
#endif
#endif
    out RAB_LightSample o_selectedSample)
{
    o_selectedSample = RAB_EmptyLightSample();

    if (localLightBufferRegion.numLights == 0)
        return RTXDI_EmptyDIReservoir();

    if (sampleParams.numLocalLightSamples == 0)
        return RTXDI_EmptyDIReservoir();

    return RTXDI_SampleLocalAreaLightsInternal(rng, coherentRng, surface, sampleParams, misData, localLightSamplingMode, localLightBufferRegion,
#if RTXDI_ENABLE_PRESAMPLING
    localLightRISBufferSegmentParams,
#if RTXDI_REGIR_MODE != RTXDI_REGIR_DISABLED
    regirParams,
#endif
#endif
    o_selectedSample);
}

#if INTRO_RTXDI_ENABLE_ENV_LIGHT
// Custom environment-map RIS stream. Mirrors RTXDI_SampleLocalAreaLights, but instead of selecting from a
// light-buffer region it importance-samples the environment map directly (MIP descent - no presampled RIS
// tiles) and streams each candidate under the synthetic env light's real index, storing the sampled
// octahedral direction as the reservoir UV. The winning sample resolves through the normal polymorphic
// path (RAB_LoadLightInfo -> EnvironmentLight::CalcSample) in every downstream pass.
RTXDI_DIReservoir RTXDI_SampleEnvironmentMapCustom(
    inout RTXDI_RandomSamplerState rng,
    RAB_Surface surface,
    RTXDI_DIInitialSamplingParameters sampleParams,
    RTXDI_InitialSamplingMisData misData,
    out RAB_LightSample o_selectedSample)
{
    o_selectedSample = RAB_EmptyLightSample();
    RTXDI_DIReservoir state = RTXDI_EmptyDIReservoir();

    if (sampleParams.numEnvironmentSamples == 0 || g_Const.lightControl.EnvironmentLightPresent == 0)
        return state;

    uint envLightIndex = g_Const.lightControl.EnvironmentLightIndex;
    RAB_LightInfo envLightInfo = RAB_LoadLightInfo(envLightIndex, false);
    EnvMapSampler envSampler = IntroMakeEnvMapSampler();

    for (uint i = 0; i < sampleParams.numEnvironmentSamples; i++)
    {
        float2 rnd = float2(RTXDI_GetNextRandom(rng), RTXDI_GetNextRandom(rng));
        DistantLightSample ds = envSampler.MIPDescentSample(rnd);
        if (ds.Pdf <= 0)
            continue;

        float2 uv = Encode_Oct(ds.Dir);
        // RIS source pdf is 1: there is no discrete light selection for the single environment light, and its
        // solid-angle importance-sampling density is already carried by candidateSample.solidAnglePdf
        // (EnvironmentLight::CalcSample -> MIPDescentEvalPdf), which targetPdf and the shading both divide by.
        // Feeding 1/ds.Pdf here would double-count that density and blow the estimate up by orders of magnitude.
        float invSourcePdf = 1.0;

        RAB_LightSample candidateSample = RAB_SamplePolymorphicLight(envLightInfo, surface, uv);
        float blendedSourcePdf = RTXDI_LightBrdfMisWeight(surface, candidateSample, 1.0 / invSourcePdf,
            misData.environmentMapMisWeight, true, misData.brdfMisWeight, sampleParams.brdfCutoff);
        if (blendedSourcePdf == 0)
            continue;

        float targetPdf = RAB_GetLightSampleTargetPdfForSurface(candidateSample, surface);
        float risRnd = RTXDI_GetNextRandom(rng);
        bool selected = RTXDI_StreamSample(state, envLightIndex, uv, risRnd, targetPdf, 1.0 / blendedSourcePdf);
        if (selected)
            o_selectedSample = candidateSample;
    }

    RTXDI_FinalizeResampling(state, 1.0, misData.numMisSamples);
    state.M = 1;
    return state;
}
#endif // INTRO_RTXDI_ENABLE_ENV_LIGHT

#endif // __INTRO_RESTIR_DI_COMMON_HLSLI__
