/*
* Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

// Note:
// This is a simple example of a procedural sky, used to stress test the path tracer dynamic environment map capability.
// The shaders were borrowed from https://www.shadertoy.com/view/tdSXzD by user 'stilltravelling' - much appreciated.
// 
// There's a much better version on https://www.shadertoy.com/view/cdlyWr that includes moon as well as well as moving stars.
// It would be nice to upgrade to it at some point^ with proper parametrization for earth rotation (stars moving), sun 
// position, moon position, etc.


#pragma once

#include <donut/engine/BindingCache.h>
#include <nvrhi/nvrhi.h>
#include <donut/core/math/math.h>
#include <memory>

#include <donut/core/math/math.h>

using namespace donut::math;

#include "SampleProceduralSky.hlsli"

namespace donut::engine
{
    class FramebufferFactory;
    class TextureCache;
    class TextureHandle;
    class ShaderFactory;
    class CommonRenderPasses;
    struct TextureData;
    struct LoadedTexture;
}

class SampleProceduralSky 
{
public:
    SampleProceduralSky( nvrhi::IDevice* device, std::shared_ptr<donut::engine::TextureCache> textureCache, std::shared_ptr<donut::engine::CommonRenderPasses> commonPasses, nvrhi::ICommandList* commandList );
    ~SampleProceduralSky();

    nvrhi::TextureHandle            GetTransmittanceTexture() const;
    nvrhi::TextureHandle            GetScatterringTexture() const;
    nvrhi::TextureHandle            GetIrradianceTexture() const;
    nvrhi::TextureHandle            GetCloudsTexture() const;
    nvrhi::TextureHandle            GetNoiseTexture() const;

    bool                            Update( double sceneTime, ProceduralSkyConstants & outConstants );

    void                            DebugGUI(float indent);

private:
    nvrhi::DeviceHandle                             m_device;
    std::shared_ptr<donut::engine::TextureCache>    m_textureCache;

    std::shared_ptr<donut::engine::LoadedTexture>   m_transmittanceTexture;
    std::shared_ptr<donut::engine::LoadedTexture>   m_scatterringTexture;
    std::shared_ptr<donut::engine::LoadedTexture>   m_irradianceTexture;
    std::shared_ptr<donut::engine::LoadedTexture>   m_cloudsTexture;
    std::shared_ptr<donut::engine::LoadedTexture>   m_noiseTexture;

    float3                          m_colorTint                 = float3(1.45f, 1.29f, 1.27f);
    float                           m_brightness                = 0.7f;
    float                           m_sunBrightness             = 5.0f;
    float                           m_cloudsMovementSpeed       = 0.8f;
    float                           m_timeOfDayMovementSpeed    = 300.0f;
    float                           m_sunTimeOfDayOffset        = -0.4f;
    float                           m_sunEastWestRotation       = 0.0f;

    float                           m_cloudDensityOffset        = 0.75f;
    float                           m_cloudTransmittance        = 2.5f;
    float                           m_cloudScattering           = 2.0f;

    ProceduralSkyConstants          m_lastConstants;
};
