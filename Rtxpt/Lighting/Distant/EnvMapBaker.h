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

#include <nvrhi/nvrhi.h>
#include <donut/core/math/math.h>
#include <memory>

#include <donut/core/math/math.h>

#include <filesystem>

#include "../../SampleCommon/SampleCommon.h"

using namespace donut::math;

#include "EnvMapBaker.hlsl"
#include "BRDFLUTGenerator.hlsl"

#include "SampleProceduralSky.h"

namespace donut::engine
{
    class BindingCache;
    class FramebufferFactory;
    class TextureCache;
    class TextureHandle;
    class ShaderFactory;
    class CommonRenderPasses;
    struct TextureData;
}

class ShaderDebug;
class EnvMapImportanceSamplingBaker;
class ComputePipelineBaker;
class ComputeShaderVariant;

// This is used to bake the cubemap with the given inputs. Inputs can be equirectangular envmap image or procedural sky, and directional lights.
// There's a low resolution (half by half) pass which can be used to speed up baking - currently only used for procedural sky.
// Also provides shared cubemap processing functionality (mips, GGX filtering, SH projection) via ProcessCubemap().
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

    // Options for ProcessCubemap() - shared cubemap processing
    struct CubemapProcessingOptions
    {
        bool generateMips = true;           // Solid-angle weighted mip chain
        bool ggxPrefilter = false;          // GGX specular convolution (for IBL specular)
        bool projectToSH = false;           // Project to SH and reproject to low-res diffuse cube
        bool bc6hCompress = false;          // BC6H compression
        bool generateImportanceMap = false; // For path tracer importance sampling
    };

    // Results from ProcessCubemap() - caller provides the destination textures
    struct CubemapProcessingResults
    {
        nvrhi::TextureHandle filteredCubemap;       // GGX-filtered mip chain (for specular)
        nvrhi::TextureHandle diffuseIrradianceCube; // Low-res SH-reprojected cubemap (for diffuse)
    };

    constexpr static uint           c_MaxDirLights  = EMB_MAXDIRLIGHTS;    // Can't have any more than this number of directional lights baked into cubemap, sorry.
    constexpr static uint           c_BRDFLUTSize   = BRDF_LUT_SIZE;       // Size of the BRDF integration LUT
    constexpr static uint           c_IrradianceCubeSize = 32;             // Size of diffuse irradiance cubemap
    

public:
    EnvMapBaker( nvrhi::IDevice* device, std::shared_ptr<donut::engine::TextureCache> textureCache, bool enableRasterPrecompute );
    ~EnvMapBaker();

    void                            SceneReloaded()                 { m_targetResolution = 0; } // change default target resolution on each scene load

    void                            CreateRenderPasses(std::shared_ptr<ShaderDebug> shaderDebug, std::shared_ptr<donut::engine::ShaderFactory> shaderFactory, std::shared_ptr<ComputePipelineBaker> computePipelineBaker);

    void                            PreUpdate( nvrhi::ICommandList* commandList, std::shared_ptr<donut::engine::CommonRenderPasses> commonPasses, std::string envMapBackgroundPath );    // use to update to figure out GetTargetCubeResolution() default cubemap resolution and needed before Update; Ignore return if not needed.
    // Returns 'true' if contents changed; note: directionalLights must be transformed to Environment map local space. 
    bool                            Update( nvrhi::ICommandList * commandList, donut::engine::BindingCache & bindingCache, std::shared_ptr<donut::engine::CommonRenderPasses> commonPasses, const BakeSettings & settings, double sceneTime, EMB_DirectionalLight const * directionalLights, uint directionaLightCount, bool forceInstantUpdate );

    nvrhi::TextureHandle            GetEnvMapCube() const           { return (m_outputIsCompressed)?(m_cubemapBC6H):(m_cubemap); }
    nvrhi::SamplerHandle            GetEnvMapCubeSampler() const    { return m_linearSampler; }
    uint                            GetEnvMapCubeDim() const        { return m_cubeDim; }
    uint64_t                        GetEnvMapVersion() const        { return m_versionID; }

    bool                            DebugGUI(float indent);

    bool                            IsProcedural() const            { return IsProceduralSky( m_loadedSourceBackgroundPath.c_str() ); }
    const std::shared_ptr<SampleProceduralSky> &
                                    GetProceduralSky() const        { return m_proceduralSky; }

    void                            SetTargetCubeResolution(uint res)   { m_targetResolution = res; }
    int                             GetTargetCubeResolution() const;

    const std::shared_ptr<EnvMapImportanceSamplingBaker>
                                    GetImportanceSampling() const   { return m_importanceSamplingBaker; }

    // BRDF LUT for split-sum IBL approximation (generated once at startup)
    nvrhi::TextureHandle            GetBRDFLUT() const              { return m_brdfLUT; }
    bool                            IsBRDFLUTReady() const          { return m_brdfLUT != nullptr && m_brdfLUTGenerated; }

    // Process an external cubemap with the specified options
    // This allows reusing the mip generation, GGX filtering, and SH projection for external cubemaps (e.g., local RT cubemap)
    void                            ProcessCubemap(
                                        nvrhi::ICommandList* commandList,
                                        donut::engine::BindingCache& bindingCache,
                                        nvrhi::TextureHandle sourceCubemap,
                                        const CubemapProcessingOptions& options,
                                        const CubemapProcessingResults& results);
    
    // Generate the BRDF integration LUT (should be called once during initialization)
    bool                            GenerateBRDFLUT(nvrhi::ICommandList* commandList, donut::engine::BindingCache& bindingCache);
    
private:
    void                            GGXPrefilterCubemap(nvrhi::ICommandList* commandList, donut::engine::BindingCache& bindingCache, 
                                        nvrhi::TextureHandle srcCubemap, nvrhi::TextureHandle dstCubemap);
    void                            ConvolveDiffuseIrradiance(nvrhi::ICommandList* commandList, donut::engine::BindingCache& bindingCache,
                                        nvrhi::TextureHandle srcCubemap, nvrhi::TextureHandle dstCubemap);
    void                            GenerateCubemapMips(nvrhi::ICommandList* commandList, donut::engine::BindingCache& bindingCache, 
                                        nvrhi::TextureHandle cubemap);

    void                            InitBuffers(uint cubeDim);
    void                            UnloadSourceBackgrounds();

private:
    nvrhi::DeviceHandle             m_device;
    std::shared_ptr<donut::engine::TextureCache> m_textureCache;
    std::shared_ptr<ShaderDebug>    m_shaderDebug;

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

    std::string                     m_dbgSaveBaked = "";

    std::shared_ptr<EnvMapImportanceSamplingBaker>
                                    m_importanceSamplingBaker;

    bool                            m_enableRasterPrecompute = false;

    // BRDF LUT for split-sum IBL
    nvrhi::TextureHandle            m_brdfLUT;
    nvrhi::ShaderHandle             m_brdfLUTCS;
    nvrhi::ComputePipelineHandle    m_brdfLUTPSO;
    nvrhi::BindingLayoutHandle      m_brdfLUTBindingLayout;
    nvrhi::BufferHandle             m_brdfLUTConstantBuffer;
    bool                            m_brdfLUTGenerated = false;

    // GGX pre-filtering for specular IBL - managed by ComputePipelineBaker for hot reload
    std::shared_ptr<ComputeShaderVariant> m_ggxPrefilterVariant;
    nvrhi::BindingLayoutHandle      m_ggxPrefilterBindingLayout;

    // Diffuse irradiance convolution (SH-based) - managed by ComputePipelineBaker for hot reload
    std::shared_ptr<ComputeShaderVariant> m_irradianceConvolveVariant;
    nvrhi::BindingLayoutHandle      m_irradianceConvolveBindingLayout;

    // Reference to compute pipeline baker for hot reload support
    std::shared_ptr<ComputePipelineBaker> m_computePipelineBaker;
};
