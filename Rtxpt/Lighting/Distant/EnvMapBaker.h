/*
* Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#pragma once

#include <donut/engine/BindingCache.h>
#include <nvrhi/nvrhi.h>
#include <donut/core/math/math.h>
#include <memory>

#include <donut/core/math/math.h>

#include <filesystem>


using namespace donut::math;

#include "EnvMapBaker.hlsl"

#include "SampleProceduralSky.h"

namespace donut::engine
{
    class FramebufferFactory;
    class TextureCache;
    class TextureHandle;
    class ShaderFactory;
    class CommonRenderPasses;
    struct TextureData;
}

class EnvMapImportanceSamplingBaker;

// This is used to bake the cubemap with the given inputs. Inputs can be equirectangular envmap image or procedural sky, and directional lights.
// There's a low resolution (half by half) pass which can be used to speed up baking - currently only used for procedural sky.
class EnvMapBaker 
{
public:
    struct BakeSettings
    {
        // Use this if input envmap is FP32 and outside of max FP16 (65504.0, which is the max we support for perf reasons) - you can premultiply with say 1/16 to avoid clamping 
        // and later use envmap sampling Intensity setting of 16 to offset! This will also help with baking in small sharp bright directional lights. Values lower than (roughly) 1/1024 can result in image quality loss.
        float   EnvMapRadianceScale = 1.0;  

        //BakeSettings() {}
        //BakeSettings(const float envMapRadianceScale) : EnvMapRadianceScale(envMapRadianceScale) {}
    };

    constexpr static uint           c_MaxDirLights  = EMB_MAXDIRLIGHTS;    // Can't have any more than this number of directional lights baked into cubemap, sorry.
    constexpr static const char *   c_ProcSkyName   = "==PROCEDURAL_SKY==";
    constexpr static const char *   c_SceneDefault  = "==SCENE_DEFAULT==";

public:
    EnvMapBaker( nvrhi::IDevice* device, std::shared_ptr<donut::engine::TextureCache> textureCache, std::shared_ptr<donut::engine::ShaderFactory> shaderFactory, std::shared_ptr<donut::engine::CommonRenderPasses> commonPasses );
    ~EnvMapBaker();

    void                            SceneReloaded()                 { m_targetResolution = 0; } // change default target resolution on each scene load

    void                            CreateRenderPasses();

    void                            PreUpdate( nvrhi::ICommandList* commandList, std::string envMapBackgroundPath );    // use to update to figure out GetTargetCubeResolution() default cubemap resolution and needed before Update; Ignore return if not needed.
    // Returns 'true' if contents changed; note: directionalLights must be transformed to Environment map local space. 
    bool                            Update( nvrhi::ICommandList * commandList, const BakeSettings & settings, double sceneTime, EMB_DirectionalLight const * directionalLights, uint directionaLightCount );

    nvrhi::TextureHandle            GetEnvMapCube() const           { return (m_outputIsCompressed)?(m_cubemapBC6H):(m_cubemap); }
    nvrhi::SamplerHandle            GetEnvMapCubeSampler() const    { return m_linearSampler; }
    uint                            GetEnvMapCubeDim() const        { return m_cubeDim; }
    uint64_t                        GetEnvMapVersion() const        { return m_versionID; }

    bool                            DebugGUI(float indent);

    bool                            IsProcedural() const            { return m_dbgOverrideSource == std::string(c_ProcSkyName) || m_loadedSourceBackgroundPath == std::string(c_ProcSkyName); }

    void                            SetTargetCubeResolution(uint res)   { m_targetResolution = res; }
    int                             GetTargetCubeResolution() const;

    const std::shared_ptr<EnvMapImportanceSamplingBaker>
                                    GetImportanceSampling() const   { return m_importanceSamplingBaker; }

    
private:
    void                            InitBuffers(uint cubeDim);
    void                            UnloadSourceBackgrounds();

private:
    nvrhi::DeviceHandle             m_device;
    std::shared_ptr<donut::engine::TextureCache> m_textureCache;
    std::shared_ptr<donut::engine::CommonRenderPasses> m_commonPasses;
    std::shared_ptr<donut::engine::FramebufferFactory> m_framebufferFactory;
    std::shared_ptr<donut::engine::ShaderFactory> m_shaderFactory;

    nvrhi::ShaderHandle             m_lowResPrePassLayerCS;
    nvrhi::ComputePipelineHandle    m_lowResPrePassLayerPSO;

    nvrhi::ShaderHandle             m_baseLayerCS;
    nvrhi::ComputePipelineHandle    m_baseLayerPSO;

    nvrhi::ShaderHandle             m_MIPReduceCS;
    nvrhi::ComputePipelineHandle    m_MIPReducePSO;

    nvrhi::BindingLayoutHandle      m_commonBindingLayout;
    nvrhi::BindingLayoutHandle      m_reduceBindingLayout;

    bool                            m_BC6UCompressionEnabled = true;
    nvrhi::ShaderHandle             m_BC6UCompressLowCS;
    nvrhi::ComputePipelineHandle    m_BC6UCompressLowPSO;
    nvrhi::ShaderHandle             m_BC6UCompressHighCS;
    nvrhi::ComputePipelineHandle    m_BC6UCompressHighPSO;
    nvrhi::BindingLayoutHandle      m_BC6UCompressBindingLayout;


    donut::engine::BindingCache     m_bindingCache;

    nvrhi::BufferHandle             m_constantBuffer;

    nvrhi::SamplerHandle            m_pointSampler;
    nvrhi::SamplerHandle            m_linearSampler;
    nvrhi::SamplerHandle            m_equiRectSampler;

    std::string                     m_sourceBackgroundPath;
    std::string                     m_loadedSourceBackgroundPath;
    std::shared_ptr<donut::engine::TextureData>    m_loadedSourceBackgroundTextureEquirect;
    std::shared_ptr<donut::engine::TextureData>    m_loadedSourceBackgroundTextureCubemap;

    nvrhi::TextureHandle            m_cubemap;
    nvrhi::TextureDesc              m_cubemapDesc;
    nvrhi::TextureHandle            m_cubemapBC6H;
    nvrhi::TextureHandle            m_cubemapBC6HScratch;
    uint                            m_cubeDim = 0;

    uint                            m_targetResolution = 0;

    // optional low res pre-pass goes into this cubemap and is (additively) sampled in the full res pass
    nvrhi::TextureHandle            m_cubemapLowRes;
    uint                            m_cubeDimLowResDim = 0;

    uint64_t                        m_versionID = -1;
    bool                            m_renderPassesDirty = true;

    EMB_DirectionalLight            m_bakedLights[c_MaxDirLights];
    uint                            m_bakedLightCount = 0;

    int                             m_compressionQuality = 1;       // 0 - disabled; 1 - low quality; 2 - high quality
    bool                            m_outputIsCompressed = false;   // updated in Update() - it reflects current state of textures while 'm_compressionQuality' reflects required (future) state

    std::shared_ptr<SampleProceduralSky>
                                    m_proceduralSky;
    bool                            m_dbgForceDynamic = false;
    std::filesystem::path           m_dbgLocalMediaFolder;
    std::vector<std::filesystem::path> m_dbgLocalMediaEnvironmentMaps;
    std::string                     m_dbgOverrideSource = c_SceneDefault;

    std::string                     m_dbgSaveBaked = "";

    std::shared_ptr<EnvMapImportanceSamplingBaker>
                                    m_importanceSamplingBaker;
};
