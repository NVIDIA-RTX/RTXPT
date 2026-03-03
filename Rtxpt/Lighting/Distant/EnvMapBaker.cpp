/*
* Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "EnvMapBaker.h"
#include "CubemapProcessing.hlsl"

#include "EnvMapImportanceSamplingBaker.h"
#include "../../SampleCommon/ComputePipelineBaker.h"

#include <donut/engine/BindingCache.h>
#include <donut/engine/ShaderFactory.h>
#include <donut/engine/FramebufferFactory.h>
#include <donut/engine/CommonRenderPasses.h>
#include <donut/engine/TextureCache.h>

#include <donut/app/UserInterfaceUtils.h>

#include <nvrhi/utils.h>

#include <donut/app/imgui_renderer.h>
#include <donut/core/vfs/VFS.h>
#include <donut/engine/DDSFile.h>

#include <fstream>

#include "../../Misc/ShaderDebug.h"


using namespace donut;
using namespace donut::math;
using namespace donut::engine;

std::filesystem::path GetLocalPath(std::string subfolder);

static const int    c_BlockCompressionBlockSize = 4; 

EnvMapBaker::EnvMapBaker( nvrhi::IDevice* device, std::shared_ptr<donut::engine::TextureCache> textureCache, bool enableRasterPrecompute )
    : m_device(device)
    , m_textureCache(textureCache)
{
}

EnvMapBaker::~EnvMapBaker()
{
    UnloadSourceBackgrounds();
}

void EnvMapBaker::CreateRenderPasses(std::shared_ptr<ShaderDebug> shaderDebug, std::shared_ptr<donut::engine::ShaderFactory> shaderFactory, std::shared_ptr<ComputePipelineBaker> computePipelineBaker)
{
    m_importanceSamplingBaker = nullptr;
    m_shaderDebug = shaderDebug;
    m_computePipelineBaker = computePipelineBaker;

    // Release existing compute shader variants from the baker
    if (m_computePipelineBaker && m_enableRasterPrecompute)
    {
        if (m_ggxPrefilterVariant)
            m_computePipelineBaker->ReleaseVariant(m_ggxPrefilterVariant);
        if (m_irradianceConvolveVariant)
            m_computePipelineBaker->ReleaseVariant(m_irradianceConvolveVariant);
    }
    m_ggxPrefilterVariant = nullptr;
    m_irradianceConvolveVariant = nullptr;

    // early de-alloc when re-creating passes
    m_lowResPrePassLayerCS = m_baseLayerCS = m_MIPReduceCS = nullptr;
    m_commonBindingLayout = m_reduceBindingLayout = nullptr;
    m_lowResPrePassLayerPSO = m_baseLayerPSO = m_MIPReducePSO = m_BC6UCompressLowPSO = m_BC6UCompressHighPSO = nullptr;
    m_pointSampler = m_linearSampler = m_equiRectSampler = nullptr;
    
    // Reset new processing shaders
    m_brdfLUTCS = nullptr;
    m_brdfLUTPSO = nullptr;
    m_brdfLUTBindingLayout = m_ggxPrefilterBindingLayout = m_irradianceConvolveBindingLayout = nullptr;

    std::vector<donut::engine::ShaderMacro> shaderMacros;
    //shaderMacros.push_back(donut::engine::ShaderMacro({              "BLEND_DEBUG_BUFFER", "1" }));

    m_lowResPrePassLayerCS = shaderFactory->CreateShader("app/Lighting/Distant/EnvMapBaker.hlsl", "LowResPrePassLayerCS", &shaderMacros, nvrhi::ShaderType::Compute);
    m_baseLayerCS = shaderFactory->CreateShader("app/Lighting/Distant/EnvMapBaker.hlsl", "BaseLayerCS", &shaderMacros, nvrhi::ShaderType::Compute);
    m_MIPReduceCS = shaderFactory->CreateShader("app/Lighting/Distant/EnvMapBaker.hlsl", "MIPReduceCS", &shaderMacros, nvrhi::ShaderType::Compute);

    {
        nvrhi::BindingLayoutDesc layoutDesc;
        layoutDesc.visibility = nvrhi::ShaderType::Compute;
        layoutDesc.bindings = {
            nvrhi::BindingLayoutItem::VolatileConstantBuffer(0),
            //nvrhi::BindingLayoutItem::PushConstants(1, sizeof(SampleMiniConstants)),
            nvrhi::BindingLayoutItem::Texture_UAV(0),
            nvrhi::BindingLayoutItem::Texture_UAV(1),
            nvrhi::BindingLayoutItem::Texture_SRV(0),
            nvrhi::BindingLayoutItem::Texture_SRV(1),
            nvrhi::BindingLayoutItem::Texture_SRV(2),
            nvrhi::BindingLayoutItem::Texture_SRV(10),
            nvrhi::BindingLayoutItem::Texture_SRV(11),
            nvrhi::BindingLayoutItem::Texture_SRV(12),
            nvrhi::BindingLayoutItem::Texture_SRV(13),
            nvrhi::BindingLayoutItem::Texture_SRV(14),
            nvrhi::BindingLayoutItem::Sampler(0),
            nvrhi::BindingLayoutItem::Sampler(1),
            nvrhi::BindingLayoutItem::Sampler(2),
            nvrhi::BindingLayoutItem::RawBuffer_UAV(SHADER_DEBUG_BUFFER_UAV_INDEX),
            nvrhi::BindingLayoutItem::Texture_UAV(SHADER_DEBUG_VIZ_TEXTURE_UAV_INDEX)
        };
        m_commonBindingLayout = m_device->createBindingLayout(layoutDesc);
    }

    {
        nvrhi::BindingLayoutDesc layoutDesc;
        layoutDesc.visibility = nvrhi::ShaderType::Compute;
        layoutDesc.bindings = {
            nvrhi::BindingLayoutItem::VolatileConstantBuffer(0),
            //nvrhi::BindingLayoutItem::PushConstants(1, sizeof(SampleMiniConstants)),
            nvrhi::BindingLayoutItem::Texture_UAV(0),
            nvrhi::BindingLayoutItem::Texture_UAV(1),
            nvrhi::BindingLayoutItem::Sampler(0),
            nvrhi::BindingLayoutItem::Sampler(1),
            nvrhi::BindingLayoutItem::Sampler(2),
            nvrhi::BindingLayoutItem::RawBuffer_UAV(SHADER_DEBUG_BUFFER_UAV_INDEX),
            nvrhi::BindingLayoutItem::Texture_UAV(SHADER_DEBUG_VIZ_TEXTURE_UAV_INDEX)
        };
        m_reduceBindingLayout = m_device->createBindingLayout(layoutDesc);
    }

    nvrhi::ComputePipelineDesc pipelineDesc;

    pipelineDesc.bindingLayouts = { m_commonBindingLayout };
    pipelineDesc.CS = m_lowResPrePassLayerCS;
    m_lowResPrePassLayerPSO = m_device->createComputePipeline(pipelineDesc);

    pipelineDesc.bindingLayouts = { m_commonBindingLayout };
    pipelineDesc.CS = m_baseLayerCS;
    m_baseLayerPSO = m_device->createComputePipeline(pipelineDesc);

    pipelineDesc.bindingLayouts = { m_reduceBindingLayout };
    pipelineDesc.CS = m_MIPReduceCS;
    m_MIPReducePSO = m_device->createComputePipeline(pipelineDesc);

    nvrhi::SamplerDesc samplerDesc;
    samplerDesc.setBorderColor(nvrhi::Color(0.f));
    samplerDesc.setAllFilters(true);
    samplerDesc.setMipFilter(true);
    samplerDesc.setAllAddressModes(nvrhi::SamplerAddressMode::Wrap);
    m_linearSampler = m_device->createSampler(samplerDesc);

    samplerDesc.setAllFilters(false);
    m_pointSampler = m_device->createSampler(samplerDesc);

    samplerDesc = nvrhi::SamplerDesc();
    samplerDesc.setAddressU(nvrhi::SamplerAddressMode::Wrap);
    samplerDesc.setAllFilters(true);
    m_equiRectSampler = m_device->createSampler(samplerDesc);

    m_importanceSamplingBaker = std::make_shared<EnvMapImportanceSamplingBaker>(m_device, shaderFactory);
    m_importanceSamplingBaker->CreateRenderPasses();

    if (m_BC6UCompressionEnabled)
    {
        std::vector<donut::engine::ShaderMacro> smQ0 = { donut::engine::ShaderMacro({ "QUALITY", "0" }) };
        std::vector<donut::engine::ShaderMacro> smQ1 = { donut::engine::ShaderMacro({ "QUALITY", "1" }) };
        m_BC6UCompressLowCS = shaderFactory->CreateShader("app/Lighting/Distant/BC6UCompress.hlsl", "CSMain", &smQ0, nvrhi::ShaderType::Compute);
        m_BC6UCompressHighCS = shaderFactory->CreateShader("app/Lighting/Distant/BC6UCompress.hlsl", "CSMain", &smQ1, nvrhi::ShaderType::Compute);

        nvrhi::BindingLayoutDesc layoutDesc;
        layoutDesc.visibility = nvrhi::ShaderType::Compute;
        layoutDesc.bindings = {
            nvrhi::BindingLayoutItem::Texture_UAV(0),
            nvrhi::BindingLayoutItem::Texture_SRV(0),
            nvrhi::BindingLayoutItem::Sampler(0)
        };
        m_BC6UCompressBindingLayout = m_device->createBindingLayout(layoutDesc);

        nvrhi::ComputePipelineDesc pipelineDesc;
        pipelineDesc.bindingLayouts = { m_BC6UCompressBindingLayout };
        pipelineDesc.CS = m_BC6UCompressLowCS;
        m_BC6UCompressLowPSO = m_device->createComputePipeline(pipelineDesc);
        pipelineDesc.CS = m_BC6UCompressHighCS;
        m_BC6UCompressHighPSO = m_device->createComputePipeline(pipelineDesc);
    }

    // === BRDF LUT Generation ===
    if (m_enableRasterPrecompute)
    {
        m_brdfLUTCS = shaderFactory->CreateShader("app/Lighting/Distant/BRDFLUTGenerator.hlsl", "main", nullptr, nvrhi::ShaderType::Compute);
        
        nvrhi::BindingLayoutDesc layoutDesc;
        layoutDesc.visibility = nvrhi::ShaderType::Compute;
        layoutDesc.bindings = {
            nvrhi::BindingLayoutItem::VolatileConstantBuffer(0),
            nvrhi::BindingLayoutItem::Texture_UAV(0)
        };
        m_brdfLUTBindingLayout = m_device->createBindingLayout(layoutDesc);
        
        nvrhi::ComputePipelineDesc pipelineDesc;
        pipelineDesc.bindingLayouts = { m_brdfLUTBindingLayout };
        pipelineDesc.CS = m_brdfLUTCS;
        m_brdfLUTPSO = m_device->createComputePipeline(pipelineDesc);
        
        m_brdfLUTConstantBuffer = m_device->createBuffer(nvrhi::utils::CreateVolatileConstantBufferDesc(
            sizeof(BRDFLUTConstants), "BRDFLUTConstants", engine::c_MaxRenderPassConstantBufferVersions));
        
        // Create BRDF LUT texture
        nvrhi::TextureDesc lutDesc;
        lutDesc.width = c_BRDFLUTSize;
        lutDesc.height = c_BRDFLUTSize;
        lutDesc.format = nvrhi::Format::RG16_FLOAT;
        lutDesc.dimension = nvrhi::TextureDimension::Texture2D;
        lutDesc.debugName = "BRDF_LUT";
        lutDesc.isUAV = true;
        lutDesc.initialState = nvrhi::ResourceStates::UnorderedAccess;
        lutDesc.keepInitialState = true;
        m_brdfLUT = m_device->createTexture(lutDesc);
    }
    
    // === GGX Pre-filtering ===
    if (m_enableRasterPrecompute)
    {
        nvrhi::BindingLayoutDesc layoutDesc;
        layoutDesc.visibility = nvrhi::ShaderType::Compute;
        layoutDesc.bindings = {
            nvrhi::BindingLayoutItem::PushConstants(0, sizeof(GGXPrefilterConstants)),
            nvrhi::BindingLayoutItem::Texture_SRV(0),
            nvrhi::BindingLayoutItem::Texture_UAV(0),
            nvrhi::BindingLayoutItem::Sampler(0)
        };
        m_ggxPrefilterBindingLayout = m_device->createBindingLayout(layoutDesc);
                
        if (m_computePipelineBaker)
        {
            m_ggxPrefilterVariant = m_computePipelineBaker->CreateVariant(
                "../Lighting/Distant/CubemapProcessing.hlsl",  // source path relative to ShadersPath (Rtxpt/Shaders)
                "GGXPrefilterCS",                               // entry point
                {},                                             // macros (empty)
                { m_ggxPrefilterBindingLayout },                // binding layouts
                "GGXPrefilter");                                // debug name
        }
    }
    
    // === Diffuse Irradiance Convolution ===
    if (m_enableRasterPrecompute)
    {
        nvrhi::BindingLayoutDesc layoutDesc;
        layoutDesc.visibility = nvrhi::ShaderType::Compute;
        layoutDesc.bindings = {
            nvrhi::BindingLayoutItem::PushConstants(0, sizeof(IrradianceConvolveConstants)),
            nvrhi::BindingLayoutItem::Texture_SRV(0),
            nvrhi::BindingLayoutItem::Texture_UAV(0),
            nvrhi::BindingLayoutItem::Sampler(0)
        };
        m_irradianceConvolveBindingLayout = m_device->createBindingLayout(layoutDesc);
                
        if (m_computePipelineBaker)
        {
            m_irradianceConvolveVariant = m_computePipelineBaker->CreateVariant(
                "../Lighting/Distant/CubemapProcessing.hlsl",  // source path relative to ShadersPath (Rtxpt/Shaders)
                "ConvolveIrradianceCS",                         // entry point
                {},                                             // macros (empty)
                { m_irradianceConvolveBindingLayout },          // binding layouts
                "IrradianceConvolve");                          // debug name
        }
    }

    // if we've recompiled shaders, force re-bake to avoid having stale data!
    m_renderPassesDirty = true;
    m_brdfLUTGenerated = false; // Force BRDF LUT regeneration
}

void EnvMapBaker::UnloadSourceBackgrounds()
{
    if (m_loadedSourceBackgroundTextureEquirect != nullptr)
        m_textureCache->UnloadTexture(m_loadedSourceBackgroundTextureEquirect);
    m_loadedSourceBackgroundTextureEquirect = nullptr;
    if (m_loadedSourceBackgroundTextureCubemap != nullptr)
        m_textureCache->UnloadTexture(m_loadedSourceBackgroundTextureCubemap);
    m_loadedSourceBackgroundTextureCubemap = nullptr;
    m_loadedSourceBackgroundPath = "";
}

void EnvMapBaker::InitBuffers(uint cubeDim)
{
    m_cubeDim = cubeDim;

    // Main constant buffer
    m_constantBuffer = m_device->createBuffer(nvrhi::utils::CreateVolatileConstantBufferDesc(
        sizeof(EnvMapBakerConstants), "EnvMapBakerConstants", engine::c_MaxRenderPassConstantBufferVersions * 5));	// *5 we could be updating few times per frame

    // Main cubemap texture
    {
        nvrhi::TextureDesc desc;

        uint mipLevels = uint(std::log2( (float)m_cubeDim / c_BlockCompressionBlockSize ) + 0.5f );  // stop at the BC block min size (4x4)

        desc.width  = m_cubeDim;
        desc.height = m_cubeDim;
        desc.depth  = 1;
        desc.arraySize = 6;
        desc.mipLevels = mipLevels;
        desc.format = nvrhi::Format::RGBA16_FLOAT; //(m_device->getGraphicsAPI() == nvrhi::GraphicsAPI::VULKAN)?nvrhi::Format::RGBA32_FLOAT:nvrhi::Format::RGBA16_FLOAT;
        desc.dimension = nvrhi::TextureDimension::TextureCube;
        desc.debugName = "EnvMapBakerMainCube";
        desc.isUAV = true;
        desc.sharedResourceFlags = nvrhi::SharedResourceFlags::None;
        desc.initialState = nvrhi::ResourceStates::UnorderedAccess;
        desc.keepInitialState = true;

        m_cubemap = m_device->createTexture(desc);
        
        m_cubemapDesc = desc; // save original cubemap settings

        // low res cubemap used for fast procedural generation or etc.
        m_cubeDimLowResDim = m_cubeDim / 2; assert( m_cubeDimLowResDim > 0 );

        desc.width = m_cubeDimLowResDim;
        desc.height = m_cubeDimLowResDim;
        desc.debugName = "EnvMapBakerMainCubeLowRes";
        desc.mipLevels = 1;
        m_cubemapLowRes = m_device->createTexture(desc);

        if (m_BC6UCompressionEnabled)
        {
            // BC6H compression resources: final compressed
            desc = m_cubemapDesc; // restore original cubemap settings
            desc.format = nvrhi::Format::BC6H_UFLOAT;
            desc.initialState = nvrhi::ResourceStates::CopyDest;
            desc.debugName = "EnvMapBakerMainCubeBC6H";
            desc.isUAV = false;
            m_cubemapBC6H = m_device->createTexture(desc);
            // BC6H compression resources: compression scratch (UAV target)
            desc.format = nvrhi::Format::RGBA32_UINT;
            desc.initialState = nvrhi::ResourceStates::UnorderedAccess;
            desc.isUAV = true;
            desc.width = m_cubeDim / c_BlockCompressionBlockSize;
            desc.height = m_cubeDim / c_BlockCompressionBlockSize;
            // desc.mipLevels is ensured to be based on "width / c_BlockCompressionBlockSize" - see above 'mipLevels'
            desc.debugName = "EnvMapBakerMainCubeBC6HScratch";
            m_cubemapBC6HScratch = m_device->createTexture(desc);
        }
    }

    m_bakedLightCount = 0;
}

bool isnear(EMB_DirectionalLight const & a, EMB_DirectionalLight const & b)
{
    return 
        dm::isnear( a.AngularSize, b.AngularSize ) &&
        dm::all(dm::isnear( a.ColorIntensity, b.ColorIntensity )) &&
        dm::all(dm::isnear( a.Direction, b.Direction ));
}

int EnvMapBaker::GetTargetCubeResolution() const     
{ 
    assert( m_targetResolution != 0 ); // PreUpdate() needs to be called to establish this value early
    return m_targetResolution; 
}

void EnvMapBaker::PreUpdate(nvrhi::ICommandList* commandList, std::shared_ptr<donut::engine::CommonRenderPasses> commonPasses, std::string envMapBackgroundPath)
{
    if( m_device->getGraphicsAPI() == nvrhi::GraphicsAPI::VULKAN && m_BC6UCompressionEnabled )
    {
        log::warning("There is an unresolved bug in BC6U cubemap compression on Vulkan. Disabling compression until it is fixed.");
        m_BC6UCompressionEnabled = false;
    }

    bool proceduralSkyEnabled = IsProceduralSky( envMapBackgroundPath.c_str() );

    if (m_targetResolution == 0)
        m_targetResolution = (proceduralSkyEnabled) ? (1024) : (2048);
    
    m_sourceBackgroundPath = envMapBackgroundPath;

    if (!m_BC6UCompressionEnabled)
        m_compressionQuality = 0;

    bool newBuffers = false;
    if (m_targetResolution != m_cubeDim)
    {
        newBuffers = true;
        m_renderPassesDirty = true;
        InitBuffers(m_targetResolution);
    }
    m_importanceSamplingBaker->PreUpdate(m_cubemap, newBuffers);

    // Load static (background) environment map or procedural sky if enabled
    if (m_sourceBackgroundPath != m_loadedSourceBackgroundPath)
    {
        UnloadSourceBackgrounds();
        m_loadedSourceBackgroundPath = m_sourceBackgroundPath;

        if (!proceduralSkyEnabled)
        {
            std::filesystem::path fullPath = GetLocalPath(c_AssetsFolder) / m_loadedSourceBackgroundPath;
            m_textureCache->LoadTextureFromFile(fullPath, false, commonPasses.get(), commandList);
            commandList->close();
            m_device->executeCommandList(commandList);
            m_device->waitForIdle();
            commandList->open();

            std::shared_ptr<TextureData> loadedTexture = m_textureCache->GetLoadedTexture(fullPath);
            if (loadedTexture != nullptr && loadedTexture->format != nvrhi::Format::UNKNOWN)
            {
                if (loadedTexture->arraySize == 6)
                    m_loadedSourceBackgroundTextureCubemap = loadedTexture;
                else
                    m_loadedSourceBackgroundTextureEquirect = loadedTexture;
            }
            else
                m_loadedSourceBackgroundPath = "";
        }

        m_renderPassesDirty = true;
    }

    if (proceduralSkyEnabled && m_proceduralSky == nullptr)
        m_proceduralSky = std::make_shared<SampleProceduralSky>(m_device, m_textureCache, commonPasses, commandList);
}

bool EnvMapBaker::Update(nvrhi::ICommandList* commandList, donut::engine::BindingCache & bindingCache, std::shared_ptr<donut::engine::CommonRenderPasses> commonPasses, const BakeSettings & _settings, double sceneTime, EMB_DirectionalLight const * directionalLights, uint directionaLightCount, bool forceInstantUpdate)
{
    BakeSettings settings = _settings;

    bool contentsChanged = m_dbgForceDynamic || m_renderPassesDirty;
    m_renderPassesDirty = false;

    bool proceduralSkyEnabled = IsProceduralSky( m_sourceBackgroundPath.c_str() );

    if (m_dbgSaveBaked != "") // re-bake if saving
    {
        contentsChanged = true;
        if (m_dbgSaveBaked == "<<REFRESH>>")
            m_dbgSaveBaked = "";                    // second pass, need to refresh
        else
            settings.EnvMapRadianceScale = 1.0f;    // need to remove scale for saving screenshot
    }

    assert( directionaLightCount <= c_MaxDirLights ); 
    directionaLightCount = std::min( directionaLightCount, c_MaxDirLights );
    if (directionaLightCount != m_bakedLightCount)
        contentsChanged |= true;
    else if (directionaLightCount > 0)
    {
        for (uint i = 0; i < directionaLightCount; i++)
            if( !isnear(directionalLights[i], m_bakedLights[i]) )
                contentsChanged |= true;
    }

    ProceduralSkyConstants procSkyConsts; memset(&procSkyConsts, 0, sizeof(procSkyConsts));
    if (m_proceduralSky != nullptr && proceduralSkyEnabled)
        contentsChanged |= m_proceduralSky->Update(sceneTime, procSkyConsts, m_sourceBackgroundPath, forceInstantUpdate);

    if (!contentsChanged)
        return contentsChanged;

    RAII_SCOPE(commandList->beginMarker("EnvMapBaker"); , commandList->endMarker(); );

    // Constants
    {
        EnvMapBakerConstants consts; memset(&consts, 0, sizeof(consts));

        if (m_proceduralSky != nullptr && proceduralSkyEnabled)
        {
            consts.ProcSkyEnabled   = 1;
            consts.ProcSkyConsts    = procSkyConsts;
        }

        // Copy over directional lights
        consts.DirectionalLightCount = m_bakedLightCount = directionaLightCount;
        for (uint i = 0; i < m_bakedLightCount; i++)
            consts.DirectionalLights[i] = m_bakedLights[i] = directionalLights[i];

        consts.CubeDim = m_cubeDim;
        consts.CubeDimLowRes = m_cubeDimLowResDim;
        consts.ScaleColor = float3(settings.EnvMapRadianceScale,settings.EnvMapRadianceScale,settings.EnvMapRadianceScale);
        consts.BackgroundSourceType = 0;
        if (m_loadedSourceBackgroundTextureEquirect != nullptr)
            consts.BackgroundSourceType = 1;
        else if (m_loadedSourceBackgroundTextureCubemap != nullptr)
            consts.BackgroundSourceType = 2;
     
        commandList->writeBuffer(m_constantBuffer, &consts, sizeof(consts));
    }

    // Bindings
    nvrhi::BindingSetDesc bindingSetDesc;
    bindingSetDesc.bindings = {
            nvrhi::BindingSetItem::ConstantBuffer(0, m_constantBuffer),
            //nvrhi::BindingSetItem::PushConstants(1, sizeof(SampleMiniConstants)),
            nvrhi::BindingSetItem::Texture_UAV(0, m_cubemapLowRes, nvrhi::Format::UNKNOWN, nvrhi::TextureSubresourceSet(0, 1, 0, 6)).setDimension(nvrhi::TextureDimension::Texture2DArray),
            nvrhi::BindingSetItem::Texture_UAV(1, m_cubemap, nvrhi::Format::UNKNOWN, nvrhi::TextureSubresourceSet(1, 1, 0, 6)).setDimension(nvrhi::TextureDimension::Texture2DArray),
            nvrhi::BindingSetItem::Texture_SRV(0, (m_loadedSourceBackgroundTextureEquirect != nullptr) ? (m_loadedSourceBackgroundTextureEquirect->texture) : ((nvrhi::TextureHandle)commonPasses->m_BlackTexture.Get())),
            nvrhi::BindingSetItem::Texture_SRV(1, (m_loadedSourceBackgroundTextureCubemap != nullptr) ? (m_loadedSourceBackgroundTextureCubemap->texture) : ((nvrhi::TextureHandle)commonPasses->m_BlackCubeMapArray.Get())),
            nvrhi::BindingSetItem::Texture_SRV(2, (nvrhi::TextureHandle)commonPasses->m_BlackCubeMapArray.Get()),
            nvrhi::BindingSetItem::Texture_SRV(10, (m_proceduralSky != nullptr && proceduralSkyEnabled) ? (m_proceduralSky->GetTransmittanceTexture()) : ((nvrhi::TextureHandle)commonPasses->m_BlackTexture.Get())),
            nvrhi::BindingSetItem::Texture_SRV(11, (m_proceduralSky != nullptr && proceduralSkyEnabled) ? (m_proceduralSky->GetScatterringTexture()) : ((nvrhi::TextureHandle)commonPasses->m_BlackTexture3D.Get())),
            nvrhi::BindingSetItem::Texture_SRV(12, (m_proceduralSky != nullptr && proceduralSkyEnabled) ? (m_proceduralSky->GetIrradianceTexture()) : ((nvrhi::TextureHandle)commonPasses->m_BlackTexture.Get())),
            nvrhi::BindingSetItem::Texture_SRV(13, (m_proceduralSky != nullptr && proceduralSkyEnabled) ? (m_proceduralSky->GetCloudsTexture()) : ((nvrhi::TextureHandle)commonPasses->m_BlackTexture.Get())),
            nvrhi::BindingSetItem::Texture_SRV(14, (m_proceduralSky != nullptr && proceduralSkyEnabled) ? (m_proceduralSky->GetNoiseTexture()) : ((nvrhi::TextureHandle)commonPasses->m_BlackTexture.Get())),
            //nvrhi::BindingSetItem::Texture_UAV(0, m_Cubemap),
            nvrhi::BindingSetItem::Sampler(0, m_pointSampler),
            nvrhi::BindingSetItem::Sampler(1, m_linearSampler),
            nvrhi::BindingSetItem::Sampler(2, m_equiRectSampler),
            nvrhi::BindingSetItem::RawBuffer_UAV(SHADER_DEBUG_BUFFER_UAV_INDEX, m_shaderDebug->GetGPUWriteBuffer()),
            nvrhi::BindingSetItem::Texture_UAV(SHADER_DEBUG_VIZ_TEXTURE_UAV_INDEX, m_shaderDebug->GetDebugVizTexture()),
    };
    nvrhi::BindingSetHandle bindingSetLowResPrePass = bindingCache.GetOrCreateBindingSet(bindingSetDesc, m_commonBindingLayout);
    bindingSetDesc.bindings[5] = nvrhi::BindingSetItem::Texture_SRV(2, m_cubemapLowRes );
    bindingSetDesc.bindings[1] = nvrhi::BindingSetItem::Texture_UAV(0, m_cubemap, nvrhi::Format::UNKNOWN, nvrhi::TextureSubresourceSet(0, 1, 0, 6)).setDimension(nvrhi::TextureDimension::Texture2DArray);
    nvrhi::BindingSetHandle bindingSetBake = bindingCache.GetOrCreateBindingSet(bindingSetDesc, m_commonBindingLayout);

    {
        // Low res pre-pass (only needed for proc sky)
        if (proceduralSkyEnabled)
        {
            RAII_SCOPE(commandList->beginMarker("ProcSkyLowResPrePass"); , commandList->endMarker(); );

            nvrhi::ComputeState state;
            state.bindings = { bindingSetLowResPrePass };
            state.pipeline = m_lowResPrePassLayerPSO;

            commandList->setComputeState(state);

            const dm::uint  threads = EMB_NUM_COMPUTE_THREADS_PER_DIM;
            const dm::uint2 dispatchSize = dm::uint2((m_cubeDimLowResDim + threads - 1) / threads, (m_cubeDimLowResDim + threads - 1) / threads);
            assert(m_cubeDim % EMB_NUM_COMPUTE_THREADS_PER_DIM == 0); // if not, shaders need fixing!
            //commandList->setPushConstants(&miniConsts, sizeof(miniConsts));
            commandList->dispatch(dispatchSize.x, dispatchSize.y, 6); // <- 6 cubemap faces! :)
        }

        // Base bake
        {
            RAII_SCOPE(commandList->beginMarker("ProcSkyBaseBake");, commandList->endMarker(); );
            nvrhi::ComputeState state;
            state.bindings = { bindingSetBake };
            state.pipeline = m_baseLayerPSO;

            commandList->setComputeState(state);

            const dm::uint  threads = EMB_NUM_COMPUTE_THREADS_PER_DIM;
            const dm::uint2 dispatchSize = dm::uint2((m_cubeDim/2 + threads - 1) / threads, (m_cubeDim/2 + threads - 1) / threads);
            assert( m_cubeDim % EMB_NUM_COMPUTE_THREADS_PER_DIM == 0 ); // if not, shaders need fixing!
            //commandList->setPushConstants(&miniConsts, sizeof(miniConsts));
            commandList->dispatch(dispatchSize.x, dispatchSize.y, 6); // <- 6 cubemap faces! :)
        }

        commandList->setTextureState(m_cubemap, nvrhi::AllSubresources, nvrhi::ResourceStates::UnorderedAccess);
    }

    {
        RAII_SCOPE( commandList->beginMarker("EnvMapBakerMIPs");, commandList->endMarker(); );

        // Downsample MIPs - TODO: do it as a 2 or 4 layers at a time for better perf 
        uint mipLevels = m_cubemap->getDesc().mipLevels;
        for (uint i = 2; i < mipLevels; i++)
        {
            nvrhi::BindingSetDesc localBindingSetDesc;
            localBindingSetDesc.bindings = {
                    nvrhi::BindingSetItem::ConstantBuffer(0, m_constantBuffer),
                    //nvrhi::BindingSetItem::PushConstants(1, sizeof(SampleMiniConstants)),
                    nvrhi::BindingSetItem::Texture_UAV(0, m_cubemap, nvrhi::Format::UNKNOWN, nvrhi::TextureSubresourceSet(i, 1, 0, 6)).setDimension(nvrhi::TextureDimension::Texture2DArray),
                    nvrhi::BindingSetItem::Texture_UAV(1, m_cubemap, nvrhi::Format::UNKNOWN, nvrhi::TextureSubresourceSet(i - 1, 1, 0, 6)).setDimension(nvrhi::TextureDimension::Texture2DArray),
                    nvrhi::BindingSetItem::Sampler(0, m_pointSampler),
                    nvrhi::BindingSetItem::Sampler(1, m_linearSampler),
                    nvrhi::BindingSetItem::Sampler(2, m_equiRectSampler),
                    nvrhi::BindingSetItem::RawBuffer_UAV(SHADER_DEBUG_BUFFER_UAV_INDEX, m_shaderDebug->GetGPUWriteBuffer()),
                    nvrhi::BindingSetItem::Texture_UAV(SHADER_DEBUG_VIZ_TEXTURE_UAV_INDEX, m_shaderDebug->GetDebugVizTexture()),
            };
            nvrhi::BindingSetHandle localBindingSet = bindingCache.GetOrCreateBindingSet(localBindingSetDesc, m_reduceBindingLayout);
        
            nvrhi::ComputeState state;
            state.bindings = { localBindingSet };
            state.pipeline = m_MIPReducePSO;

            commandList->setComputeState(state);

            uint destinationRes = m_cubemap->getDesc().width >> i;

            const dm::uint  threads = EMB_NUM_COMPUTE_THREADS_PER_DIM;
            const dm::uint2 dispatchSize = dm::uint2((destinationRes + threads - 1) / threads, (destinationRes + threads - 1) / threads);
            //commandList->setPushConstants(&miniConsts, sizeof(miniConsts));
            commandList->dispatch(dispatchSize.x, dispatchSize.y, 6); // <- 6 cubemap faces! :)

            commandList->setTextureState(m_cubemap, nvrhi::AllSubresources, nvrhi::ResourceStates::UnorderedAccess);
        }
    }

    if (m_compressionQuality>0 && m_BC6UCompressionEnabled)
    {
        RAII_SCOPE(commandList->beginMarker("BC6UCompression"); , commandList->endMarker(); );

        uint mipLevels = m_cubemap->getDesc().mipLevels; assert( mipLevels == m_cubemapBC6HScratch->getDesc().mipLevels );
        for (uint i = 0; i < mipLevels; i++)
        {
            nvrhi::BindingSetDesc localBindingSetDesc;
            localBindingSetDesc.bindings = {
                    nvrhi::BindingSetItem::Texture_UAV(0, m_cubemapBC6HScratch, nvrhi::Format::UNKNOWN, nvrhi::TextureSubresourceSet(i, 1, 0, 6)).setDimension(nvrhi::TextureDimension::Texture2DArray),
                    nvrhi::BindingSetItem::Texture_SRV(0, m_cubemap, nvrhi::Format::UNKNOWN, nvrhi::TextureSubresourceSet(i, 1, 0, 6)).setDimension(nvrhi::TextureDimension::Texture2DArray),
                    nvrhi::BindingSetItem::Sampler(0, m_pointSampler),
            };
            nvrhi::BindingSetHandle localBindingSet = bindingCache.GetOrCreateBindingSet(localBindingSetDesc, m_BC6UCompressBindingLayout);

            nvrhi::ComputeState state;
            state.bindings = { localBindingSet };
            state.pipeline = m_compressionQuality==1?m_BC6UCompressLowPSO:m_BC6UCompressHighPSO;

            commandList->setComputeState(state);

            uint destinationRes = m_cubemapBC6HScratch->getDesc().width;

            const dm::uint  threads = 8;
            const dm::uint2 dispatchSize = dm::uint2((destinationRes + threads - 1) / threads, (destinationRes + threads - 1) / threads);
            //commandList->setPushConstants(&miniConsts, sizeof(miniConsts));
            commandList->dispatch(dispatchSize.x, dispatchSize.y, 6); // <- 6 cubemap faces! :)
        }

        // TODO: upgrade to CopyResource
        for (uint im = 0; im < mipLevels; im++)
            for( uint ia = 0; ia < 6; ia++)
            {
                nvrhi::TextureSlice slice; slice.setArraySlice(ia); slice.setMipLevel(im);
                commandList->copyTexture(m_cubemapBC6H, slice, m_cubemapBC6HScratch, slice);
            }

        m_outputIsCompressed = true;
    }
    else
        m_outputIsCompressed = false;

    m_importanceSamplingBaker->Update(commandList, m_cubemap);

    m_versionID++; 

    if (m_dbgSaveBaked != "")
    {
        nvrhi::TextureDesc outCubemapDesc = m_cubemapDesc;
        outCubemapDesc.mipLevels = 1; // remove this if you want to export all MIPs; not needed here because we regenerate them anyway on load
        nvrhi::StagingTextureHandle cubemapStaging = m_device->createStagingTexture(outCubemapDesc, nvrhi::CpuAccessMode::Read);

        for (int m = 0; m < (int)outCubemapDesc.mipLevels; m++)
            for( int i = 0; i < 6; i++ )
            {
                nvrhi::TextureSlice slice; 
                slice.arraySlice = i;
                slice.mipLevel = m;
                commandList->copyTexture(cubemapStaging, slice, m_cubemap, slice);
            }

        commandList->close();
        m_device->executeCommandList(commandList);
        m_device->waitForIdle();

        auto blob = SaveStagingTextureAsDDS(m_device, cubemapStaging);

        if (blob != nullptr)
        {
            std::fstream myfile;
            myfile.open(m_dbgSaveBaked.c_str(), std::ios::out | std::ios::trunc | std::ios::binary);
            if (myfile.is_open())
            {
                myfile.write((char*)blob->data(), blob->size());
                myfile.close();
                donut::log::info("Image saved successfully %s.", m_dbgSaveBaked.c_str());
            }
            else
                donut::log::fatal("Unable to write into file %s. ", m_dbgSaveBaked.c_str());
        }
        else
            donut::log::fatal("Unable to bake cubemap for image %s. ", m_dbgSaveBaked.c_str());

        m_dbgSaveBaked = "<<REFRESH>>"; // need to re-bake one more time with normal settings

        commandList->open();
    }

    return contentsChanged;
}

std::string CubeResToString(uint res)
{
    char resRet[1024]; 
    snprintf(resRet, sizeof(resRet), "%d x %d x 6", res, res );
    return resRet;
}

bool EnvMapBaker::DebugGUI(float indent)
{
    bool resetAccumulation = false;
    #define IMAGE_QUALITY_OPTION(code) do{if (code) resetAccumulation = true;} while(false)

    std::string currentRes = CubeResToString(m_targetResolution);
    if (ImGui::BeginCombo("Target cube res", currentRes.c_str()))
    {
        uint resolutions[] = {512, 1024, 2048, 4096};
        for (int i = 0; i < (int)std::size(resolutions); i++)   // note, std::size is equivalent of _countof :)
        {
            std::string itemName = CubeResToString(resolutions[i]);
            bool is_selected = itemName == currentRes;
            if (ImGui::Selectable(itemName.c_str(), is_selected))
                m_targetResolution = resolutions[i];
            if (is_selected)
                ImGui::SetItemDefaultFocus();
        }
        ImGui::EndCombo();
        resetAccumulation = true;
    }
    IMAGE_QUALITY_OPTION(ImGui::Checkbox("Force dynamic", &m_dbgForceDynamic));
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Force re-generate every frame even if static (for performance testing only)");

    if (m_BC6UCompressionEnabled)
    {
        if (ImGui::Combo("BC6U compression", &m_compressionQuality, "Off\0Fast\0Quality\0\0"))
        {
            m_renderPassesDirty = true;
            resetAccumulation = true;
        }
    }
    else
    {
        ImGui::Text("BC6U compression not currently supported in Vulkan");
    }

    if (ImGui::Button("Save baked cubemap"))
    {
        std::string fileName;
        if (donut::app::FileDialog(false, "DDS files\0*.dds\0\0", fileName))
            m_dbgSaveBaked = fileName;
    }
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Save baked cubemap. It will be rebaked with EnvMapRadianceScale set to 1.0 before saving.");

    return resetAccumulation;
}

bool EnvMapBaker::GenerateBRDFLUT(nvrhi::ICommandList* commandList, donut::engine::BindingCache& bindingCache)
{
    if (!m_enableRasterPrecompute || m_brdfLUTGenerated)
        return false;
    
    RAII_SCOPE(commandList->beginMarker("GenerateBRDFLUT");, commandList->endMarker(););
    
    BRDFLUTConstants consts;
    consts.LUTSize = c_BRDFLUTSize;
    consts.SampleCount = BRDF_LUT_SAMPLE_COUNT;
    consts.Padding0 = 0;
    consts.Padding1 = 0;
    commandList->writeBuffer(m_brdfLUTConstantBuffer, &consts, sizeof(consts));
    
    nvrhi::BindingSetDesc bindingSetDesc;
    bindingSetDesc.bindings = {
        nvrhi::BindingSetItem::ConstantBuffer(0, m_brdfLUTConstantBuffer),
        nvrhi::BindingSetItem::Texture_UAV(0, m_brdfLUT)
    };
    nvrhi::BindingSetHandle bindingSet = bindingCache.GetOrCreateBindingSet(bindingSetDesc, m_brdfLUTBindingLayout);
    
    nvrhi::ComputeState state;
    state.bindings = { bindingSet };
    state.pipeline = m_brdfLUTPSO;
    commandList->setComputeState(state);
    
    const uint threads = 8;
    const dm::uint2 dispatchSize = dm::uint2(div_ceil(c_BRDFLUTSize, threads), div_ceil(c_BRDFLUTSize, threads));
    commandList->dispatch(dispatchSize.x, dispatchSize.y, 1);
        
    m_brdfLUTGenerated = true;
    return true;
}

void EnvMapBaker::GGXPrefilterCubemap(nvrhi::ICommandList* commandList, donut::engine::BindingCache& bindingCache, 
    nvrhi::TextureHandle srcCubemap, nvrhi::TextureHandle dstCubemap)
{
    if (!m_enableRasterPrecompute)
        return;

    RAII_SCOPE(commandList->beginMarker("GGXPrefilter");, commandList->endMarker(););
    
    // Get pipeline from hot-reloadable variant
    auto pipeline = m_ggxPrefilterVariant ? m_ggxPrefilterVariant->GetPipeline() : nullptr;
    if (!pipeline)
        return; // Pipeline not yet compiled or compilation failed
    
    uint srcSize = srcCubemap->getDesc().width;
    uint mipLevels = dstCubemap->getDesc().mipLevels;
    
    GGXPrefilterConstants consts;
    consts.SrcCubemapSize = srcSize;
    consts.MaxMipLevels = mipLevels;
    consts.SampleCount = 64;

    for (uint mip = 0; mip < mipLevels; mip++)
    {
        uint dstMipSize = srcSize >> mip;
        float roughness = static_cast<float>(mip) / static_cast<float>(mipLevels - 1);
        roughness = max(0.001f, roughness);
        
        consts.DstMipSize = dstMipSize;
        consts.MipLevel = mip;
        consts.Roughness = roughness;
        consts.Padding1 = 0;
        consts.Padding2 = 0;
        
        nvrhi::BindingSetDesc bindingSetDesc;
        bindingSetDesc.bindings = {
            nvrhi::BindingSetItem::PushConstants(0, sizeof(GGXPrefilterConstants)),
            nvrhi::BindingSetItem::Texture_SRV(0, srcCubemap),
            nvrhi::BindingSetItem::Texture_UAV(0, dstCubemap, nvrhi::Format::UNKNOWN, nvrhi::TextureSubresourceSet(mip, 1, 0, 6)).setDimension(nvrhi::TextureDimension::Texture2DArray),
            nvrhi::BindingSetItem::Sampler(0, m_linearSampler)
        };
        nvrhi::BindingSetHandle bindingSet = bindingCache.GetOrCreateBindingSet(bindingSetDesc, m_ggxPrefilterBindingLayout);
        
        nvrhi::ComputeState state;
        state.bindings = { bindingSet };
        state.pipeline = pipeline;
        commandList->setComputeState(state);
        commandList->setPushConstants(&consts, sizeof(consts));
        
        const uint threads = CUBEMAP_PROCESS_THREADS;
        const dm::uint2 dispatchSize = dm::uint2(div_ceil(dstMipSize,threads),div_ceil(dstMipSize, threads));
        commandList->dispatch(dispatchSize.x, dispatchSize.y, 6);
        
        // UAV barrier between mips
        commandList->setTextureState(dstCubemap, nvrhi::AllSubresources, nvrhi::ResourceStates::UnorderedAccess);
    }
}

void EnvMapBaker::ConvolveDiffuseIrradiance(nvrhi::ICommandList* commandList, donut::engine::BindingCache& bindingCache,
    nvrhi::TextureHandle srcCubemap, nvrhi::TextureHandle dstCubemap)
{
    if (!m_enableRasterPrecompute)
        return;

    RAII_SCOPE(commandList->beginMarker("ConvolveIrradiance");, commandList->endMarker(););
    
    // Get pipeline from hot-reloadable variant
    auto pipeline = m_irradianceConvolveVariant ? m_irradianceConvolveVariant->GetPipeline() : nullptr;
    if (!pipeline)
        return; // Pipeline not yet compiled or compilation failed
    
    IrradianceConvolveConstants consts;
    consts.SrcCubemapSize = srcCubemap->getDesc().width;
    consts.DstCubemapSize = c_IrradianceCubeSize;
    consts.SampleCount = 1024;
    consts.Padding = 0;
    
    nvrhi::BindingSetDesc bindingSetDesc;
    bindingSetDesc.bindings = {
        nvrhi::BindingSetItem::PushConstants(0, sizeof(IrradianceConvolveConstants)),
        nvrhi::BindingSetItem::Texture_SRV(0, srcCubemap),
        nvrhi::BindingSetItem::Texture_UAV(0, dstCubemap, nvrhi::Format::UNKNOWN, nvrhi::TextureSubresourceSet(0, 1, 0, 6)).setDimension(nvrhi::TextureDimension::Texture2DArray),
        nvrhi::BindingSetItem::Sampler(0, m_linearSampler)
    };
    nvrhi::BindingSetHandle bindingSet = bindingCache.GetOrCreateBindingSet(bindingSetDesc, m_irradianceConvolveBindingLayout);
    
    nvrhi::ComputeState state;
    state.bindings = { bindingSet };
    state.pipeline = pipeline;
    commandList->setComputeState(state);
    commandList->setPushConstants(&consts, sizeof(IrradianceConvolveConstants));
    
    // Dispatch once per face - 1024 threads per group, one group per face (32x32 resolution)
    commandList->dispatch(1, 1, 6);
}

void EnvMapBaker::GenerateCubemapMips(nvrhi::ICommandList* commandList, donut::engine::BindingCache& bindingCache, 
    nvrhi::TextureHandle cubemap)
{
    // Reuse the existing MIPReduceCS for solid-angle weighted mip generation
    RAII_SCOPE(commandList->beginMarker("GenerateCubemapMips");, commandList->endMarker(););
    
    // Write dummy constants (MIPReduceCS doesn't actually read them, but binding layout requires it)
    EnvMapBakerConstants dummyConsts = {};
    commandList->writeBuffer(m_constantBuffer, &dummyConsts, sizeof(dummyConsts));
    
    uint mipLevels = cubemap->getDesc().mipLevels;
    uint baseSize = cubemap->getDesc().width;
    
    for (uint i = 1; i < mipLevels; i++)
    {
        nvrhi::BindingSetDesc localBindingSetDesc;
        localBindingSetDesc.bindings = {
            nvrhi::BindingSetItem::ConstantBuffer(0, m_constantBuffer),
            nvrhi::BindingSetItem::Texture_UAV(0, cubemap, nvrhi::Format::UNKNOWN, nvrhi::TextureSubresourceSet(i, 1, 0, 6)).setDimension(nvrhi::TextureDimension::Texture2DArray),
            nvrhi::BindingSetItem::Texture_UAV(1, cubemap, nvrhi::Format::UNKNOWN, nvrhi::TextureSubresourceSet(i - 1, 1, 0, 6)).setDimension(nvrhi::TextureDimension::Texture2DArray),
            nvrhi::BindingSetItem::Sampler(0, m_pointSampler),
            nvrhi::BindingSetItem::Sampler(1, m_linearSampler),
            nvrhi::BindingSetItem::Sampler(2, m_equiRectSampler),
            nvrhi::BindingSetItem::RawBuffer_UAV(SHADER_DEBUG_BUFFER_UAV_INDEX, m_shaderDebug->GetGPUWriteBuffer()),
            nvrhi::BindingSetItem::Texture_UAV(SHADER_DEBUG_VIZ_TEXTURE_UAV_INDEX, m_shaderDebug->GetDebugVizTexture()),
        };
        nvrhi::BindingSetHandle localBindingSet = bindingCache.GetOrCreateBindingSet(localBindingSetDesc, m_reduceBindingLayout);
        
        nvrhi::ComputeState state;
        state.bindings = { localBindingSet };
        state.pipeline = m_MIPReducePSO;
        commandList->setComputeState(state);
        
        uint destinationRes = baseSize >> i;
        
        const dm::uint threads = EMB_NUM_COMPUTE_THREADS_PER_DIM;
        const dm::uint2 dispatchSize = dm::uint2((destinationRes + threads - 1) / threads, (destinationRes + threads - 1) / threads);
        commandList->dispatch(dispatchSize.x, dispatchSize.y, 6);
    }
}

void EnvMapBaker::ProcessCubemap(
    nvrhi::ICommandList* commandList,
    donut::engine::BindingCache& bindingCache,
    nvrhi::TextureHandle sourceCubemap,
    const CubemapProcessingOptions& options,
    const CubemapProcessingResults& results)
{
    RAII_SCOPE(commandList->beginMarker("ProcessCubemap");, commandList->endMarker(););
    
    // Generate mips for source if needed
    if (options.generateMips)
    {
        GenerateCubemapMips(commandList, bindingCache, sourceCubemap);
    }
    
    // GGX pre-filtering for specular
    if (options.ggxPrefilter)
    {
        assert(results.filteredCubemap);
        GGXPrefilterCubemap(commandList, bindingCache, sourceCubemap, results.filteredCubemap);
    }
    
    // SH-based diffuse irradiance
    if (options.projectToSH)
    {
        assert(results.diffuseIrradianceCube);
        ConvolveDiffuseIrradiance(commandList, bindingCache, sourceCubemap, results.diffuseIrradianceCube);
    }
}
