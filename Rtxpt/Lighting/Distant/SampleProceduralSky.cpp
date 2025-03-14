/*
* Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "SampleProceduralSky.h"
#include <donut/engine/ShaderFactory.h>
#include <donut/engine/FramebufferFactory.h>
#include <donut/engine/CommonRenderPasses.h>
#include <donut/engine/TextureCache.h>

#include <nvrhi/utils.h>

#include <donut/app/imgui_renderer.h>

#include "../../SampleCommon.h"

using namespace donut;
using namespace donut::math;
using namespace donut::engine;

std::filesystem::path GetLocalPath(std::string subfolder);  // defined in Sample.cpp

extern const char* g_assetsFolder;

SampleProceduralSky::SampleProceduralSky( nvrhi::IDevice* device, std::shared_ptr<donut::engine::TextureCache> textureCache, std::shared_ptr<donut::engine::CommonRenderPasses> commonPasses, nvrhi::ICommandList* commandList )
    : m_device(device)
    , m_textureCache(textureCache)
{
    auto path = GetLocalPath(g_assetsFolder);

    m_transmittanceTexture  = textureCache->LoadTextureFromFile(path.string() + "/StandaloneTextures/q2rtx_env/transmittance_earth.dds", false, commonPasses.get(), commandList);
    m_scatterringTexture    = textureCache->LoadTextureFromFile(path.string() + "/StandaloneTextures/q2rtx_env/inscatter_earth.dds", false, commonPasses.get(), commandList);
    m_irradianceTexture     = textureCache->LoadTextureFromFile(path.string() + "/StandaloneTextures/q2rtx_env/irradiance_earth.dds", false, commonPasses.get(), commandList);
    m_cloudsTexture         = textureCache->LoadTextureFromFile(path.string() + "/StandaloneTextures/q2rtx_env/clouds.dds", false, commonPasses.get(), commandList);
    m_noiseTexture          = textureCache->LoadTextureFromFile(path.string() + "/StandaloneTextures/RGBANoiseMedium.png", false, commonPasses.get(), commandList);

    // Make sure the texture is loaded
     commandList->close();
    m_device->executeCommandList(commandList);
    m_device->waitForIdle();
    commandList->open();

    memset( &m_lastConstants, 0, sizeof(m_lastConstants) );
}

SampleProceduralSky::~SampleProceduralSky()
{
    if (m_noiseTexture != nullptr)
        m_textureCache->UnloadTexture(m_noiseTexture);
}

nvrhi::TextureHandle SampleProceduralSky::GetTransmittanceTexture() const { return m_transmittanceTexture->texture; }
nvrhi::TextureHandle SampleProceduralSky::GetScatterringTexture() const { return m_scatterringTexture->texture; }
nvrhi::TextureHandle SampleProceduralSky::GetIrradianceTexture() const { return m_irradianceTexture->texture; }
nvrhi::TextureHandle SampleProceduralSky::GetCloudsTexture() const { return m_cloudsTexture->texture; }
nvrhi::TextureHandle SampleProceduralSky::GetNoiseTexture() const { return m_noiseTexture->texture; }

bool SampleProceduralSky::Update( double sceneTime, ProceduralSkyConstants & outConstants )
{
    memset(&outConstants, 0, sizeof(outConstants));

    outConstants.FinalRadianceMultiplier = float3(m_brightness, m_brightness, m_brightness) * m_colorTint * m_sunBrightness;

    const float cloudsLoopLength = 60 * 60 * 24;
    float cloudsTime = (float)fmod(sceneTime*m_cloudsMovementSpeed, cloudsLoopLength);

    outConstants.CloudsTime = cloudsTime;

    outConstants.GroundAlbedo       = 0.3f, 0.15f, 0.14f;
    outConstants.SunAngularRadius   = 0.00930609557f;    // avg angular radius of 0.5332 degrees

    outConstants.SkyParams.StarIrradiance                   = float3(1.47399998f, 1.85039997f, 1.91198003f) * m_sunBrightness;
    outConstants.SkyParams.StarAngularRadius                = outConstants.SunAngularRadius;
    outConstants.SkyParams.RayleightScatteringRGB           = float3(0.00580233941f, 0.0135577619f, 0.0331000052f);
    outConstants.SkyParams.PlanetSurfaceRadius              = 6360.00000f;
    outConstants.SkyParams.MieScatteringRGB                 = float3(0.00149850000f, 0.00149850000f, 0.00149850000f);
    outConstants.SkyParams.PlanetAtmosphereRadius           = 6420.00000f;
    outConstants.SkyParams.MieHenyeyGreensteinG             = 0.8f;
    outConstants.SkyParams.SqDistanceToHorizontalBoundary   = 766800.000f;
    outConstants.SkyParams.AtmosphereHeight                 = 60.0f;
    outConstants.SkyParams.reserved                         = 0.0f;

    outConstants.sun_tan_half_angle = tan(outConstants.SunAngularRadius * 0.5f);
    outConstants.sun_cos_half_angle = cos(outConstants.SunAngularRadius * 0.5f);
    outConstants.sun_solid_angle    = 2 * PI_f * (float)(1.0 - cos(0.5 * outConstants.SunAngularRadius)); // use double for precision

    outConstants.cloud_density_offset = m_cloudDensityOffset;
    outConstants.sky_transmittance  = m_cloudTransmittance;
    outConstants.sky_phase_g        = 0.9f;
    outConstants.sky_amb_phase_g    = 0.3f;
    outConstants.sky_scattering     = m_cloudScattering;
    outConstants.physical_sky_ground_radiance = (0.177055925f, 0.0584776886f, 0.00655480893f);

    // All this needs rework - stars need to rotate too :)

    float timeOfTheDay = (float)fmod((sceneTime*m_timeOfDayMovementSpeed) / float(60 * 60 * 24) + m_sunTimeOfDayOffset + 1.0f, 2.0f) - 1.0f;
    
    float3 sunDir = normalize( float3( std::cos(timeOfTheDay * PI_f), 0, std::sin(timeOfTheDay * PI_f) ) );

    const float rotateX = -0.8f;    // in radians
    const float rotateY = -1.1f;    // in radians
    const float rotateZ = dm::radians(m_sunEastWestRotation);
    affine3 earthRot = dm::rotation(float3(rotateX,0,0)) * dm::rotation(float3(0,rotateY,0)) * dm::rotation(float3(0,0,rotateZ)) ;
    sunDir = earthRot.transformVector(sunDir);
    outConstants.SunDir = sunDir;

    bool changes = memcmp(&outConstants, &m_lastConstants, sizeof(outConstants)) != 0;
    m_lastConstants = outConstants;

    return changes;
}

void SampleProceduralSky::DebugGUI(float indent)
{
    if (ImGui::CollapsingHeader("Procedural Sky", ImGuiTreeNodeFlags_DefaultOpen))
    {
        ImGui::TextWrapped("This is a simple procedural sky used to stress test environment map sampling; See https://www.shadertoy.com/view/tdSXzD for the original effect.");
        RAII_SCOPE( ImGui::Indent( indent );, ImGui::Unindent( indent ); );
        ImGui::InputFloat("Brightness", &m_brightness); m_brightness = dm::clamp(m_brightness, 0.0f, 32768.0f);
        ImGui::InputFloat("Sun Brightness", &m_sunBrightness); m_sunBrightness = dm::clamp(m_sunBrightness, 0.0f, 32768.0f);
        ImGui::InputFloat("Cloud movement speed", &m_cloudsMovementSpeed); m_cloudsMovementSpeed = dm::clamp(m_cloudsMovementSpeed, 0.0f, 10000.0f);
        ImGui::InputFloat("Sun movement speed", &m_timeOfDayMovementSpeed); m_timeOfDayMovementSpeed = dm::clamp(m_timeOfDayMovementSpeed, 0.0f, 10000.0f);
        ImGui::SliderFloat("Sun time of day offset", &m_sunTimeOfDayOffset, -1.0f, 1.0f, "%.3f"); m_sunTimeOfDayOffset = dm::clamp(m_sunTimeOfDayOffset, -1.0f, 1.0f);
        ImGui::SliderFloat("Sun east west rotation", &m_sunEastWestRotation, -180.0f, 180.0f, "%.3f"); m_sunEastWestRotation = dm::clamp(m_sunEastWestRotation, -180.0f, 180.0f);
        
         ImGui::SliderFloat("Cloud density offset", &m_cloudDensityOffset, 0.0f, 1.0f);
        // ImGui::SliderFloat("Cloud transmittance", &m_cloudTransmittance, 0.0f, 10.0f);
        // ImGui::SliderFloat("Cloud scattering", &m_cloudScattering, 0.0f, 10.0f );
    }
}
