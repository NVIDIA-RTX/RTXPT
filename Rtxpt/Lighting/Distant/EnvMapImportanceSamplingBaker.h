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

#include "../../PathTracer/Lighting/LightingTypes.h"

#include "EnvMapImportanceSamplingBaker.hlsl"

namespace donut::engine
{
    class FramebufferFactory;
    class TextureCache;
    class TextureHandle;
    class ShaderFactory;
    class CommonRenderPasses;
    struct TextureData;
}
namespace donut::render
{
    class MipMapGenPass;
}

// This is used to 
//  * pre-process importance sampling for a given cubemap source (baked by EnvMapBaker).
//  * provide all buffers and constants required for importance sampling the environment map.
// It supports 3+ approaches:
//  - uniform reference
//  - classic MIP descent (implementation originates in https://github.com/NVIDIAGameWorks/Falcor)
//  - presampled lights (use MIP descent to pre-generate a bunch of lights each frame)
class EnvMapImportanceSamplingBaker
{
public:

public:
    EnvMapImportanceSamplingBaker( nvrhi::IDevice* device, std::shared_ptr<donut::engine::TextureCache> textureCache, std::shared_ptr<donut::engine::ShaderFactory> shaderFactory, std::shared_ptr<donut::engine::CommonRenderPasses> commonPasses );
    ~EnvMapImportanceSamplingBaker();

    void                            CreateRenderPasses();

    void                            PreUpdate(nvrhi::TextureHandle sourceCubemap);
    void                            Update(nvrhi::CommandListHandle commandList, nvrhi::TextureHandle sourceCubemap);

    int                             GetImportanceMapResolution();
    int                             GetImportanceMapMIPLevels();
    nvrhi::TextureHandle            GetImportanceMap() const { return m_importanceMapTexture; }
    
    // TODO: this will be obsolete and will be removed
    void                            ExecutePresampling(nvrhi::CommandListHandle commandList, nvrhi::TextureHandle sourceCubemap, int sampleIndex);
    //nvrhi::BufferHandle             GetPresampledBuffer() const { return m_presampledBuffer; }

    nvrhi::SamplerHandle            GetImportanceMapSampler() const { return m_pointClampSampler; }

    // nvrhi::TextureHandle            GetEnvMapCube() const           { return m_cubemap; }

    bool                            DebugGUI(float indent);

    EnvMapImportanceSamplingParams  GetShaderParams()                   { return m_envMapImportanceSamplingParams; }

private:
    void                            CreateImportanceMap();
    void                            GenerateImportanceMap(nvrhi::CommandListHandle commandList, nvrhi::TextureHandle sourceCubemap);
    void                            FillBakerConsts(EnvMapImportanceSamplingBakerConstants & constants, nvrhi::TextureHandle sourceCubemap, int sampleIndex);

private:
    nvrhi::DeviceHandle             m_device;
    std::shared_ptr<donut::engine::TextureCache> m_textureCache;
    std::shared_ptr<donut::engine::CommonRenderPasses> m_commonPasses;
    std::shared_ptr<donut::engine::FramebufferFactory> m_framebufferFactory;
    std::shared_ptr<donut::engine::ShaderFactory> m_shaderFactory;
    donut::engine::BindingCache     m_bindingCache;

    nvrhi::SamplerHandle            m_pointClampSampler;
    nvrhi::SamplerHandle            m_linearWrapSampler;

    nvrhi::BufferHandle             m_builderConstants;  // EnvMapImportanceSamplingBakerConstants

    // MIP hierarchy needed for MIP descent importance sampling approach (always needed)
    nvrhi::TextureHandle            m_importanceMapTexture;
    nvrhi::ShaderHandle             m_importanceMapComputeShader;
    nvrhi::BindingLayoutHandle      m_importanceMapBindingLayout;
    nvrhi::ComputePipelineHandle    m_importanceMapPipeline;
    nvrhi::BindingSetHandle         m_importanceMapBindingSet;
    std::shared_ptr<donut::render::MipMapGenPass> m_MIPMapPass;

#if 0
    // Pre-sampling approach (faster for path tracing, but limited)
    nvrhi::BufferHandle             m_presampledBuffer;
    nvrhi::ShaderHandle             m_presamplingCS;
    nvrhi::BindingLayoutHandle      m_presamplingBindingLayout;
    nvrhi::ComputePipelineHandle    m_presamplingPipeline;
    nvrhi::BindingSetHandle         m_presamplingBindingSet;
#endif

    EnvMapImportanceSamplingParams  m_envMapImportanceSamplingParams;
};
