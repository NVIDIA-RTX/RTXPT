/*
* Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "PackingUtils.h"

#include <donut/engine/Scene.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>

using namespace std;
using namespace donut::math;
using namespace donut::engine;

#include "../Shaders/PathTracer/Lighting/PolymorphicLight.h"

namespace PackingUtils
{
    uint32_t floatToUInt(float _V, float _Scale)
    {
        return uint32_t(floor(_V * _Scale + 0.5f));
    }

    uint32_t FLOAT3_to_R8G8B8_UNORM(float unpackedInputX, float unpackedInputY, float unpackedInputZ)
    {
        return (floatToUInt(saturate(unpackedInputX), 0xFF) & 0xFF) |
            ((floatToUInt(saturate(unpackedInputY), 0xFF) & 0xFF) << 8) |
            ((floatToUInt(saturate(unpackedInputZ), 0xFF) & 0xFF) << 16);
    }

    float2 OctWrap(float2 v)
    {
        return float2(
            (1.0f - abs(v.y)) * ((v.x >= 0.0f) ? 1.0f : -1.0f),
            (1.0f - abs(v.x)) * ((v.y >= 0.0f) ? 1.0f : -1.0f));
    }

    float2 Encode_Oct(float3 n3)
    {
        n3 /= (abs(n3.x) + abs(n3.y) + abs(n3.z));
        donut::math::float2 n = n3.xy();
        n = n3.z >= 0.0f ? n : OctWrap(n);
        n = n * 0.5f + 0.5f;
        return n;
    }

    uint32_t NDirToOctUnorm32(float3 n)
    {
        float2 p = Encode_Oct(n);
        p = saturate(p * 0.5f + 0.5f);
        return uint32_t(p.x * 0xfffe) | (uint32_t(p.y * 0xfffe) << 16);
    }

    // Modified from the DX fallback layer sample.
    inline uint16_t fp32ToFp16(float v)
    {
        static const union FU {
            uint32_t ui;
            float f;
        } multiple = { 0x07800000 }; // 2**-112

        FU biasedFloat;
        biasedFloat.f = v * multiple.f;
        const uint32_t bits = biasedFloat.ui;

        const uint32_t sign = bits & 0x80000000;
        uint32_t body = bits & 0x0fffffff;

        return (uint16_t)(sign >> 16 | body >> 13) & 0xffff;
    }

    void packLightColor(const float3& color, PolymorphicLightInfoFull& lightInfo)
    {
        packLightColor(color, lightInfo.Base);
    }

    void packLightColor(const float3& color, PolymorphicLightInfo& lightInfo)
    {
        float maxRadiance = std::max(color.x, std::max(color.y, color.z));
        if (maxRadiance <= 0.f)
            return;

        float logRadiance = (::log2f(maxRadiance) - kPolymorphicLightMinLog2Radiance) / (kPolymorphicLightMaxLog2Radiance - kPolymorphicLightMinLog2Radiance);
        logRadiance = saturate(logRadiance);
        uint32_t packedRadiance = std::min(uint32_t(ceilf(logRadiance * 65534.f)) + 1, 0xffffu);
        float unpackedRadiance = ::exp2f((float(packedRadiance - 1) / 65534.f) * (kPolymorphicLightMaxLog2Radiance - kPolymorphicLightMinLog2Radiance) + kPolymorphicLightMinLog2Radiance);

        lightInfo.ColorTypeAndFlags |= FLOAT3_to_R8G8B8_UNORM(color.x / unpackedRadiance, color.y / unpackedRadiance, color.z / unpackedRadiance);
        lightInfo.LogRadiance |= packedRadiance;
        assert((lightInfo.LogRadiance & 0xFFFF0000) == 0);
    }

    bool IsAnalyticLight(const Light& light)
    {
        switch (light.GetLightType())
        {
        case LightType_Spot:
        case LightType_Point:
            return true;
        default:
            return false;
        }
    }

    bool IsInfiniteAnalyticLight(const Light& light)
    {
        switch (light.GetLightType())
        {
        case LightType_Directional:
            return true;
        default:
            return false;
        }
    }

    PolymorphicLightInfoFull ConvertLight(const Light& light)
    {
        PolymorphicLightInfo polymorphic; memset(&polymorphic, 0, sizeof(polymorphic));
        PolymorphicLightInfoEx polymorphicEx = PolymorphicLightInfoEx::empty();

        switch (light.GetLightType())
        {
        case LightType_Spot:
        {
            auto& spot = dynamic_cast<const SpotLight&>(light);
            if (spot.radius == 0.f)
            {
                float3 flux = spot.color * spot.intensity;

                polymorphic.ColorTypeAndFlags = (uint32_t)PolymorphicLightType::kPoint << kPolymorphicLightTypeShift | ((spot.outerAngle < 0) ? (kPolymorphicLightShapingUseMinFalloff) : (0));
                packLightColor(flux, polymorphic);
                polymorphic.Center = float3(spot.GetPosition());
                polymorphic.Direction1 = NDirToOctUnorm32(float3(normalize(spot.GetDirection())));
                polymorphic.Direction2 = fp32ToFp16(dm::radians(abs(spot.outerAngle)));
                polymorphic.Direction2 |= fp32ToFp16(dm::radians(spot.innerAngle)) << 16;
            }
            else
            {
                float projectedArea = dm::PI_f * (spot.radius * spot.radius);
                float3 radiance = spot.color * spot.intensity / projectedArea;
                float softness = saturate(1.f - spot.innerAngle / abs(spot.outerAngle));

                polymorphic.ColorTypeAndFlags = (uint32_t)PolymorphicLightType::kSphere << kPolymorphicLightTypeShift | ((spot.outerAngle < 0) ? (kPolymorphicLightShapingUseMinFalloff) : (0));
                polymorphic.ColorTypeAndFlags |= kPolymorphicLightShapingEnableBit;
                packLightColor(radiance, polymorphic);
                polymorphic.Center = float3(spot.GetPosition());
                polymorphic.Scalars = fp32ToFp16(spot.radius);
                if (abs(spot.outerAngle) > 0)
                {
                    polymorphic.ColorTypeAndFlags |= kPolymorphicLightShapingEnableBit;
                    polymorphicEx.PrimaryAxis = NDirToOctUnorm32(float3(normalize(spot.GetDirection())));
                    polymorphicEx.CosConeAngleAndSoftness = fp32ToFp16(cosf(dm::radians(abs(spot.outerAngle))));
                    polymorphicEx.CosConeAngleAndSoftness |= fp32ToFp16(softness) << 16;
                }
            }

            // example for the IES profile - few things need connecting
            /* case LightType_Spot: {
            *    // Spot Light with ies profile
                 auto& spot = static_cast<const SpotLightWithProfile&>(light);
                 float projectedArea = dm::PI_f * square(spot.radius);
                 float3 radiance = spot.color * spot.intensity / projectedArea;
                 float softness = saturate(1.f - spot.innerAngle / abs(spot.outerAngle));

                 polymorphic.colorTypeAndFlags = (uint32_t)PolymorphicLightType::kSphere << kPolymorphicLightTypeShift;
                 polymorphic.colorTypeAndFlags |= kPolymorphicLightShapingEnableBit;
                 packLightColor(radiance, polymorphic);
                 polymorphic.center = float3(spot.GetPosition());
                 polymorphic.scalars = fp32ToFp16(spot.radius);
                 polymorphic.primaryAxis = packNormalizedVector(float3(normalize(spot.GetDirection())));
                 polymorphic.cosConeAngleAndSoftness = fp32ToFp16(cosf(dm::radians(abs(spot.outerAngle))));
                 polymorphic.cosConeAngleAndSoftness |= fp32ToFp16(softness) << 16;

                 if (spot.profileTextureIndex >= 0)
                 {
                     polymorphic.iesProfileIndex = spot.profileTextureIndex; <- note, shader side needs fixing too
                     polymorphic.colorTypeAndFlags |= kPolymorphicLightIesProfileEnableBit;
                 }

                 return true;
             }*/

        } break;
        case LightType_Point:
        {
            auto& point = dynamic_cast<const PointLight&>(light);
            if (point.radius == 0.f)
            {
                float3 flux = point.color * point.intensity;

                polymorphic.ColorTypeAndFlags = (uint32_t)PolymorphicLightType::kPoint << kPolymorphicLightTypeShift;
                packLightColor(flux, polymorphic);
                polymorphic.Center = float3(point.GetPosition());
                // Set the default values so we can use the same path for spot lights 
                polymorphic.Direction2 = fp32ToFp16(dm::PI_f) | fp32ToFp16(0.0f) << 16;
            }
            else
            {
                float projectedArea = dm::PI_f * (point.radius * point.radius);
                float3 radiance = point.color * point.intensity / projectedArea;

                polymorphic.ColorTypeAndFlags = (uint32_t)PolymorphicLightType::kSphere << kPolymorphicLightTypeShift;
                packLightColor(radiance, polymorphic);
                polymorphic.Center = float3(point.GetPosition());
                polymorphic.Scalars = fp32ToFp16(point.radius);
            }
        }
        break;
        case LightType_Directional:
        {
            auto& dir = dynamic_cast<const DirectionalLight&>(light);
            // Donut treats the DirectionalLight's "intensity" as irradiance (illuminance on a
            // perpendicular surface). Delegate the packing to MakeDirectionalLight so scene-graph
            // directionals and the procedural-sky sun use identical layout/units.
            float clampedAngularSizeDeg = std::clamp(dir.angularSize, 0.f, 90.f);
            return MakeDirectionalLight(float3(normalize(dir.GetDirection())), float3(dir.color) * dir.irradiance, dm::radians(clampedAngularSizeDeg));
        }
        default:
            break;
        }

        return PolymorphicLightInfoFull::make(polymorphic, polymorphicEx);
    }

    PolymorphicLightInfoFull MakeDirectionalLight(const float3& travelDirection, const float3& irradiance, float angularSizeRadians)
    {
        PolymorphicLightInfo polymorphic; memset(&polymorphic, 0, sizeof(polymorphic));
        PolymorphicLightInfoEx polymorphicEx = PolymorphicLightInfoEx::empty();

        // Match the shader-side DirectionalLight::Create layout in PolymorphicLight.hlsli:
        //   Direction1 = octahedral-encoded travel direction
        //   Scalars    = fp16(halfAngle) | fp16(solidAngle) << 16
        //   Radiance   = packLightColor(...)
        // The shader uses SolidAnglePdf = 1/solidAngle and lightSample.Radiance = radiance, so the
        // integrated contribution equals irradiance (color folded into radiance below).
        constexpr float kMinHalfAngleRadians = 1e-3f; // avoid the delta-light special case (shader assumes cosHalfAngle != 1)
        float halfAngle = std::max(0.5f * angularSizeRadians, kMinHalfAngleRadians);
        float solidAngle = 2.f * dm::PI_f * (1.f - std::cos(halfAngle));
        float3 radiance = irradiance / std::max(solidAngle, 1e-12f);

        polymorphic.ColorTypeAndFlags = (uint32_t)PolymorphicLightType::kDirectional << kPolymorphicLightTypeShift;
        packLightColor(radiance, polymorphic);
        polymorphic.Direction1 = NDirToOctUnorm32(float3(normalize(travelDirection)));
        polymorphic.Scalars = uint32_t(fp32ToFp16(halfAngle)) | (uint32_t(fp32ToFp16(solidAngle)) << 16);

        return PolymorphicLightInfoFull::make(polymorphic, polymorphicEx);
    }

    PolymorphicLightInfoFull MakeEnvironmentLight()
    {
        PolymorphicLightInfo polymorphic; memset(&polymorphic, 0, sizeof(polymorphic));
        PolymorphicLightInfoEx polymorphicEx = PolymorphicLightInfoEx::empty();

        polymorphic.ColorTypeAndFlags = (uint32_t)PolymorphicLightType::kEnvironment << kPolymorphicLightTypeShift;

        return PolymorphicLightInfoFull::make(polymorphic, polymorphicEx);
    }
} // namespace PackingUtils
