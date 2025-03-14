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

#include "EnvMapImportanceSamplingBaker.h"

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

#include "../../SampleCommon.h"

using namespace donut;
using namespace donut::math;
using namespace donut::engine;

std::filesystem::path GetLocalPath(std::string subfolder);

extern const char* g_assetsFolder;

static const int    c_BlockCompressionBlockSize = 4; 

EnvMapBaker::EnvMapBaker( nvrhi::IDevice* device, std::shared_ptr<donut::engine::TextureCache> textureCache, std::shared_ptr<donut::engine::ShaderFactory> shaderFactory, std::shared_ptr<engine::CommonRenderPasses> commonPasses )
    : m_device(device)
    , m_textureCache(textureCache)
    , m_commonPasses(commonPasses)
    , m_bindingCache(device)
    , m_shaderFactory(shaderFactory)
{
    m_importanceSamplingBaker = std::make_shared<EnvMapImportanceSamplingBaker>(device, textureCache, shaderFactory, commonPasses);
}

EnvMapBaker::~EnvMapBaker()
{
    UnloadSourceBackgrounds();
}

void EnvMapBaker::CreateRenderPasses()
{
    std::vector<donut::engine::ShaderMacro> shaderMacros;
    //shaderMacros.push_back(donut::engine::ShaderMacro({              "BLEND_DEBUG_BUFFER", "1" }));

    m_lowResPrePassLayerCS = m_shaderFactory->CreateShader("app/Lighting/Distant/EnvMapBaker.hlsl", "LowResPrePassLayerCS", &shaderMacros, nvrhi::ShaderType::Compute);
    m_baseLayerCS = m_shaderFactory->CreateShader("app/Lighting/Distant/EnvMapBaker.hlsl", "BaseLayerCS", &shaderMacros, nvrhi::ShaderType::Compute);
    m_MIPReduceCS = m_shaderFactory->CreateShader("app/Lighting/Distant/EnvMapBaker.hlsl", "MIPReduceCS", &shaderMacros, nvrhi::ShaderType::Compute);

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
            nvrhi::BindingLayoutItem::Sampler(2)
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
            nvrhi::BindingLayoutItem::Sampler(2)
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

    // Get all scenes in "Assets" folder
    m_dbgLocalMediaEnvironmentMaps.clear();
    m_dbgLocalMediaFolder = GetLocalPath(g_assetsFolder) / "EnvironmentMaps";
    for (const auto& file : std::filesystem::directory_iterator(m_dbgLocalMediaFolder))
    {
        if (!file.is_regular_file()) continue;
        if (file.path().extension() == ".exr" || file.path().extension() == ".hdr" || file.path().extension() == ".dds")
            m_dbgLocalMediaEnvironmentMaps.push_back(file.path());
    }

    m_importanceSamplingBaker->CreateRenderPasses();

    if (m_BC6UCompressionEnabled)
    {
        std::vector<donut::engine::ShaderMacro> smQ0 = { donut::engine::ShaderMacro({ "QUALITY", "0" }) };
        std::vector<donut::engine::ShaderMacro> smQ1 = { donut::engine::ShaderMacro({ "QUALITY", "1" }) };
        m_BC6UCompressLowCS = m_shaderFactory->CreateShader("app/Lighting/Distant/BC6UCompress.hlsl", "CSMain", &smQ0, nvrhi::ShaderType::Compute);
        m_BC6UCompressHighCS = m_shaderFactory->CreateShader("app/Lighting/Distant/BC6UCompress.hlsl", "CSMain", &smQ1, nvrhi::ShaderType::Compute);

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

    // if we've recompiled shaders, force re-bake to avoid having stale data!
    m_renderPassesDirty = true;
}

void EnvMapBaker::UnloadSourceBackgrounds()
{
    if (m_loadedSourceBackgroundTextureEquirect != nullptr)
        m_textureCache->UnloadTexture(m_loadedSourceBackgroundTextureEquirect);
    m_loadedSourceBackgroundTextureEquirect = nullptr;
    if (m_loadedSourceBackgroundTextureCubemap != nullptr)
        m_textureCache->UnloadTexture(m_loadedSourceBackgroundTextureCubemap);
    m_loadedSourceBackgroundTextureCubemap = nullptr;
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
        desc.format = nvrhi::Format::RGBA16_FLOAT;
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

void EnvMapBaker::PreUpdate(nvrhi::ICommandList* commandList, std::string envMapBackgroundPath)
{
    if( m_device->getGraphicsAPI() == nvrhi::GraphicsAPI::VULKAN && m_BC6UCompressionEnabled )
    {
        log::warning("There is an unresolved bug in BC6U cubemap compression on Vulkan. Disabling compression until it is fixed.");
        m_BC6UCompressionEnabled = false;
    }

    if (m_dbgOverrideSource != c_SceneDefault)
    {
        if (m_dbgOverrideSource != c_ProcSkyName)
            envMapBackgroundPath = m_dbgLocalMediaFolder.string() + "/" + m_dbgOverrideSource;
        else
            envMapBackgroundPath = c_ProcSkyName;
    }

    bool proceduralSkyEnabled = envMapBackgroundPath == std::string(c_ProcSkyName);

    if (m_targetResolution == 0)
        m_targetResolution = (proceduralSkyEnabled) ? (1024) : (2048);
    
    m_sourceBackgroundPath = envMapBackgroundPath;

    if (!m_BC6UCompressionEnabled)
        m_compressionQuality = 0;

    if (m_targetResolution != m_cubeDim)
    {
        m_renderPassesDirty = true;
        InitBuffers(m_targetResolution);
    }
    m_importanceSamplingBaker->PreUpdate(m_cubemap);

    // Load static (background) environment map or procedural sky if enabled
    if (m_sourceBackgroundPath != m_loadedSourceBackgroundPath)
    {
        m_loadedSourceBackgroundPath = m_sourceBackgroundPath;
        UnloadSourceBackgrounds();

        if (!proceduralSkyEnabled)
        {
            std::filesystem::path fullPath = GetLocalPath(g_assetsFolder) / m_loadedSourceBackgroundPath;
            m_textureCache->LoadTextureFromFile(fullPath, false, m_commonPasses.get(), commandList);
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
        else
        {
            if (m_proceduralSky == nullptr)
                m_proceduralSky = std::make_shared<SampleProceduralSky>(m_device, m_textureCache, m_commonPasses, commandList);
        }

        m_renderPassesDirty = true;
    }
}

bool EnvMapBaker::Update(nvrhi::ICommandList* commandList, const BakeSettings & _settings, double sceneTime, EMB_DirectionalLight const * directionalLights, uint directionaLightCount)
{
    BakeSettings settings = _settings;

    bool contentsChanged = m_dbgForceDynamic || m_renderPassesDirty;
    m_renderPassesDirty = false;

    bool proceduralSkyEnabled = m_sourceBackgroundPath == std::string(c_ProcSkyName);

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
        contentsChanged |= m_proceduralSky->Update(sceneTime, procSkyConsts);

    if (!contentsChanged)
        return contentsChanged;

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
            nvrhi::BindingSetItem::Texture_SRV(0, (m_loadedSourceBackgroundTextureEquirect != nullptr) ? (m_loadedSourceBackgroundTextureEquirect->texture) : ((nvrhi::TextureHandle)m_commonPasses->m_BlackTexture.Get())),
            nvrhi::BindingSetItem::Texture_SRV(1, (m_loadedSourceBackgroundTextureCubemap != nullptr) ? (m_loadedSourceBackgroundTextureCubemap->texture) : ((nvrhi::TextureHandle)m_commonPasses->m_BlackCubeMapArray.Get())),
            nvrhi::BindingSetItem::Texture_SRV(2, (nvrhi::TextureHandle)m_commonPasses->m_BlackCubeMapArray.Get()),
            nvrhi::BindingSetItem::Texture_SRV(10, (m_proceduralSky != nullptr && proceduralSkyEnabled) ? (m_proceduralSky->GetTransmittanceTexture()) : ((nvrhi::TextureHandle)m_commonPasses->m_BlackTexture.Get())),
            nvrhi::BindingSetItem::Texture_SRV(11, (m_proceduralSky != nullptr && proceduralSkyEnabled) ? (m_proceduralSky->GetScatterringTexture()) : ((nvrhi::TextureHandle)m_commonPasses->m_BlackTexture3D.Get())),
            nvrhi::BindingSetItem::Texture_SRV(12, (m_proceduralSky != nullptr && proceduralSkyEnabled) ? (m_proceduralSky->GetIrradianceTexture()) : ((nvrhi::TextureHandle)m_commonPasses->m_BlackTexture.Get())),
            nvrhi::BindingSetItem::Texture_SRV(13, (m_proceduralSky != nullptr && proceduralSkyEnabled) ? (m_proceduralSky->GetCloudsTexture()) : ((nvrhi::TextureHandle)m_commonPasses->m_BlackTexture.Get())),
            nvrhi::BindingSetItem::Texture_SRV(14, (m_proceduralSky != nullptr && proceduralSkyEnabled) ? (m_proceduralSky->GetNoiseTexture()) : ((nvrhi::TextureHandle)m_commonPasses->m_BlackTexture.Get())),
            //nvrhi::BindingSetItem::Texture_UAV(0, m_Cubemap),
            nvrhi::BindingSetItem::Sampler(0, m_pointSampler),
            nvrhi::BindingSetItem::Sampler(1, m_linearSampler),
            nvrhi::BindingSetItem::Sampler(2, m_equiRectSampler)
    };
    nvrhi::BindingSetHandle bindingSetLowResPrePass = m_bindingCache.GetOrCreateBindingSet(bindingSetDesc, m_commonBindingLayout);
    bindingSetDesc.bindings[5] = nvrhi::BindingSetItem::Texture_SRV(2, m_cubemapLowRes );
    bindingSetDesc.bindings[1] = nvrhi::BindingSetItem::Texture_UAV(0, m_cubemap, nvrhi::Format::UNKNOWN, nvrhi::TextureSubresourceSet(0, 1, 0, 6)).setDimension(nvrhi::TextureDimension::Texture2DArray);
    nvrhi::BindingSetHandle bindingSetBake = m_bindingCache.GetOrCreateBindingSet(bindingSetDesc, m_commonBindingLayout);

    {
        RAII_SCOPE( commandList->beginMarker("EnvMapBaker");, commandList->endMarker(); );

        // Low res pre-pass (only needed for proc sky)
        if (proceduralSkyEnabled)
        {
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
                    nvrhi::BindingSetItem::Sampler(2, m_equiRectSampler)
            };
            nvrhi::BindingSetHandle localBindingSet = m_bindingCache.GetOrCreateBindingSet(localBindingSetDesc, m_reduceBindingLayout);
        
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
            nvrhi::BindingSetHandle localBindingSet = m_bindingCache.GetOrCreateBindingSet(localBindingSetDesc, m_BC6UCompressBindingLayout);

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
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Force re-generate every frame even if static");


    //ImGui::PushItemWidth(-60.0f);
    if (ImGui::BeginCombo("Override source", m_dbgOverrideSource.c_str()))
    {
        for (int i = -2; i < (int)m_dbgLocalMediaEnvironmentMaps.size(); i++)
        {
            std::string itemName;
            if (i == -2)
                itemName = c_SceneDefault;
            else if (i == -1)
                itemName = c_ProcSkyName;
            else 
                itemName = m_dbgLocalMediaEnvironmentMaps[i].filename().string();
                
            bool is_selected = itemName == m_dbgOverrideSource;
            if (ImGui::Selectable(itemName.c_str(), is_selected))
                m_dbgOverrideSource = itemName;
            if (is_selected)
                ImGui::SetItemDefaultFocus();
        }
        ImGui::EndCombo();
        resetAccumulation = true;
    }
    //ImGui::PopItemWidth();

    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Overrides scene's default environment map");

    if (m_proceduralSky != nullptr && m_loadedSourceBackgroundPath == std::string(c_ProcSkyName))
        m_proceduralSky->DebugGUI(indent);

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
