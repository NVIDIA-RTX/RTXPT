/*
* Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#pragma once

#include <donut/core/math/math.h>

#include <cmath>
#include <cstdint>

struct PolymorphicLightInfoFull;
struct PolymorphicLightInfo;

namespace donut::engine
{
    class Light;
}

namespace PackingUtils
{
    uint32_t floatToUInt(float _V, float _Scale);

    uint32_t FLOAT3_to_R8G8B8_UNORM(float unpackedInputX, float unpackedInputY, float unpackedInputZ);

    donut::math::float2 OctWrap(donut::math::float2 v);

    donut::math::float2 Encode_Oct(donut::math::float3 n3);

    uint32_t NDirToOctUnorm32(donut::math::float3 n);

    // Modified from the DX fallback layer sample.
    uint16_t fp32ToFp16(float v);

    void packLightColor(const donut::math::float3& color, PolymorphicLightInfoFull& lightInfo);
    void packLightColor(const donut::math::float3& color, PolymorphicLightInfo& lightInfo);

    // TODO: This should probably renamed as IsPointOrSpotLight, and we should have a separate IsAnalyticLight
    // that returns true for all analytical lights, including distant disk lights (e.g. sun) and any analytical area lights
    // supported in the future, like rectangular barndoor lights or linear segment lights.
    // For now, leaving as is to reduce refactoring.
    bool IsAnalyticLight(const donut::engine::Light& light);
    // True for analytic lights that have no position (currently directional only). These
    // are sampled separately by RTXDI as "infinite" lights in the intro sample's RTXDI
    // path. The full sample bakes them into the environment cubemap instead, so the
    // existing IsAnalyticLight predicate intentionally still returns false for them.
    bool IsInfiniteAnalyticLight(const donut::engine::Light& light);
    PolymorphicLightInfoFull ConvertLight(const donut::engine::Light& light);

    // Pack a directional ("infinite") light directly from raw parameters, matching the
    // LightType_Directional layout produced by ConvertLight. Used for lights that have no
    // scene-graph Light object (e.g. the procedural sky sun).
    //   travelDirection   - world-space direction the light travels (i.e. -toLight); normalized internally
    //   irradiance        - per-channel illuminance on a surface perpendicular to the light
    //   angularSizeRadians - full angular diameter of the light disk, in radians
    PolymorphicLightInfoFull MakeDirectionalLight(const donut::math::float3& travelDirection, const donut::math::float3& irradiance, float angularSizeRadians);

    // Pack a synthetic kEnvironment light. The environment sampler is not stored in the light record
    // (EnvironmentLight::CalcSample evaluates the globally-bound importance map / env cube), so only the
    // polymorphic type needs to be set - the record is a type marker resolved through the normal path.
    PolymorphicLightInfoFull MakeEnvironmentLight();
}
