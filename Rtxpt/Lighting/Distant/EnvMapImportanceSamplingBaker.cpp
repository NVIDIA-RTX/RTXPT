/*
* Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "EnvMapImportanceSamplingBaker.h"

#include <donut/engine/ShaderFactory.h>
#include <donut/engine/FramebufferFactory.h>
#include <donut/engine/CommonRenderPasses.h>
#include <donut/engine/TextureCache.h>
#include <donut/render/MipMapGenPass.h>

#include <nvrhi/utils.h>

#include <donut/app/imgui_renderer.h>

#include "../../SampleCommon.h"

using namespace donut;
using namespace donut::math;
using namespace donut::engine;

EnvMapImportanceSamplingBaker::EnvMapImportanceSamplingBaker( nvrhi::IDevice* device, std::shared_ptr<donut::engine::TextureCache> textureCache, std::shared_ptr<donut::engine::ShaderFactory> shaderFactory, std::shared_ptr<engine::CommonRenderPasses> commonPasses )
    : m_device(device)
    , m_textureCache(textureCache)
    , m_commonPasses(commonPasses)
    , m_bindingCache(device)
    , m_shaderFactory(shaderFactory)
{
}

EnvMapImportanceSamplingBaker::~EnvMapImportanceSamplingBaker()
{
}

void EnvMapImportanceSamplingBaker::CreateRenderPasses()
{
    // Samplers
    {
        nvrhi::SamplerDesc samplerDesc;
        samplerDesc.setBorderColor(nvrhi::Color(0.f));
        samplerDesc.setAllFilters(true);
        samplerDesc.setMipFilter(true);
        samplerDesc.setAllAddressModes(nvrhi::SamplerAddressMode::Wrap);
        m_linearWrapSampler = m_device->createSampler(samplerDesc);

        samplerDesc.setAllFilters(false);
        samplerDesc.setAllAddressModes(nvrhi::SamplerAddressMode::Clamp);
        m_pointClampSampler = m_device->createSampler(samplerDesc);
    }

    {
        nvrhi::BufferDesc constBufferDesc;
        constBufferDesc.byteSize = sizeof(EnvMapImportanceSamplingBakerConstants);
        constBufferDesc.debugName = "EnvMapImportanceSamplingBakerConstants";
        constBufferDesc.isConstantBuffer = true;
        constBufferDesc.isVolatile = true;
        constBufferDesc.maxVersions = 16;
        m_builderConstants = m_device->createBuffer(constBufferDesc);
    }   

    //Create importance map (for MIP descent) builder shader and resources
    {
        m_importanceMapComputeShader = m_shaderFactory->CreateShader("app/Lighting/Distant/EnvMapImportanceSamplingBaker.hlsl", "BuildMIPDescentImportanceMapCS", nullptr, nvrhi::ShaderType::Compute);
        assert(m_importanceMapComputeShader);

        nvrhi::BindingLayoutDesc layoutDesc;
        layoutDesc.visibility = nvrhi::ShaderType::Compute;
        layoutDesc.bindings = {
            nvrhi::BindingLayoutItem::VolatileConstantBuffer(0),
            nvrhi::BindingLayoutItem::Texture_SRV(0),
            nvrhi::BindingLayoutItem::Texture_UAV(0),
            nvrhi::BindingLayoutItem::Sampler(0),
            nvrhi::BindingLayoutItem::Sampler(1)
        };
        m_importanceMapBindingLayout = m_device->createBindingLayout(layoutDesc);

        nvrhi::ComputePipelineDesc pipelineDesc;
        pipelineDesc.setComputeShader(m_importanceMapComputeShader);
        pipelineDesc.addBindingLayout(m_importanceMapBindingLayout);
        m_importanceMapPipeline = m_device->createComputePipeline(pipelineDesc);
        
        m_importanceMapBindingSet = nullptr;
    }

    // Pre-sampling builder shader and resources
#if 0
    {
        // Stuff for presampling goes below
        m_presamplingCS = m_shaderFactory->CreateShader("app/Lighting/Distant/EnvMapImportanceSamplingBaker.hlsl", "PreSampleCS", nullptr, nvrhi::ShaderType::Compute);
        assert(m_presamplingCS);

        nvrhi::BindingLayoutDesc layoutDesc;
        layoutDesc.visibility = nvrhi::ShaderType::Compute;
        layoutDesc.bindings = {
            nvrhi::BindingLayoutItem::VolatileConstantBuffer(0),
            nvrhi::BindingLayoutItem::Texture_SRV(0),
            nvrhi::BindingLayoutItem::Texture_SRV(1),
            nvrhi::BindingLayoutItem::TypedBuffer_UAV(0),
            nvrhi::BindingLayoutItem::Sampler(0),
            nvrhi::BindingLayoutItem::Sampler(1)
        };
        m_presamplingBindingLayout = m_device->createBindingLayout(layoutDesc);

        nvrhi::ComputePipelineDesc pipelineDesc;
        pipelineDesc.setComputeShader(m_presamplingCS);
        pipelineDesc.addBindingLayout(m_presamplingBindingLayout);
        m_presamplingPipeline = m_device->createComputePipeline(pipelineDesc);

        // buffer that stores pre-generated samples which get updated once per frame
        nvrhi::BufferDesc buffDesc;
        buffDesc.byteSize = sizeof(uint32_t) * 2 * std::max(ENVMAP_PRESAMPLED_COUNT, 1u); // RG32_UINT (2 UINTs) per element
        buffDesc.format = nvrhi::Format::RG32_UINT;
        buffDesc.canHaveTypedViews = true;
        buffDesc.initialState = nvrhi::ResourceStates::ShaderResource;
        buffDesc.keepInitialState = true;
        buffDesc.debugName = "PresampledEnvironmentSamples";
        buffDesc.canHaveUAVs = true;
        m_presampledBuffer = m_device->createBuffer(buffDesc);
        assert(m_presampledBuffer);

        m_presamplingBindingSet = nullptr;
    }
#endif

    memset( &m_envMapImportanceSamplingParams, 0, sizeof(m_envMapImportanceSamplingParams) );

    m_importanceMapTexture = nullptr;
    m_MIPMapPass = nullptr;
}

int EnvMapImportanceSamplingBaker::GetImportanceMapResolution()
{
    return EMISB_IMPORTANCE_MAP_DIM;
}

int EnvMapImportanceSamplingBaker::GetImportanceMapMIPLevels()
{
    const uint32_t dimensions = GetImportanceMapResolution();
    uint32_t mips = (uint32_t)log2(dimensions) + 1;
    assert((1u << (mips - 1)) == dimensions);
    assert(mips > 1 && mips <= 12);
    return mips;
}

void EnvMapImportanceSamplingBaker::CreateImportanceMap()
{
    const uint32_t dimensions = GetImportanceMapResolution();
    const uint32_t samples = EMISB_IMPORTANCE_SAMPLES_PER_PIXEL;

    assert(ispow2(dimensions) && ispow2(samples));

    uint32_t mips = GetImportanceMapMIPLevels();

    nvrhi::TextureDesc texDesc;
    texDesc.format = nvrhi::Format::R32_FLOAT;
    texDesc.width = dimensions;
    texDesc.height = dimensions;
    texDesc.mipLevels = mips;
    texDesc.isRenderTarget = true;
    texDesc.isUAV = true;
    texDesc.debugName = "ImportanceMap";
    texDesc.setInitialState(nvrhi::ResourceStates::UnorderedAccess);
    texDesc.keepInitialState = true;
    m_importanceMapTexture = m_device->createTexture(texDesc);

    m_MIPMapPass = std::make_unique<donut::render::MipMapGenPass>(m_device, m_shaderFactory, m_importanceMapTexture, donut::render::MipMapGenPass::MODE_COLOR);
}

void EnvMapImportanceSamplingBaker::FillBakerConsts(EnvMapImportanceSamplingBakerConstants & constants, nvrhi::TextureHandle sourceCubemap, int sampleIndex)
{
    const uint32_t dimensions = EMISB_IMPORTANCE_MAP_DIM;
    const uint32_t samples = EMISB_IMPORTANCE_SAMPLES_PER_PIXEL;
    uint32_t samplesX = std::max(1u, (uint32_t)std::sqrt(samples));
    uint32_t samplesY = samples / samplesX;

    constants.SourceCubeDim = sourceCubemap->getDesc().width;
    constants.SourceCubeMIPCount = sourceCubemap->getDesc().mipLevels;
    constants.ImportanceMapDim = dimensions;
    constants.ImportanceMapDimInSamples = uint2(dimensions * samplesX, dimensions * samplesY);
    constants.ImportanceMapNumSamples = uint2(samplesX, samplesY);
    constants.ImportanceMapInvSamples = 1.f / (samplesX * samplesY);
    constants.ImportanceMapBaseMip = m_importanceMapTexture->getDesc().mipLevels - 1;
    constants.SampleIndex = sampleIndex;
}

void EnvMapImportanceSamplingBaker::PreUpdate(nvrhi::TextureHandle sourceCubemap)
{
    assert(sourceCubemap);

    if (m_importanceMapTexture == nullptr)
        CreateImportanceMap();

    if (m_importanceMapBindingSet == nullptr)
    {
        nvrhi::BindingSetDesc bindingSetDesc;
        bindingSetDesc.bindings = {
            nvrhi::BindingSetItem::ConstantBuffer(0, m_builderConstants),
            nvrhi::BindingSetItem::Texture_SRV(0, sourceCubemap),
            nvrhi::BindingSetItem::Texture_UAV(0, m_importanceMapTexture),
            nvrhi::BindingSetItem::Sampler(0, m_pointClampSampler),
            nvrhi::BindingSetItem::Sampler(1, m_linearWrapSampler),
        };
        m_importanceMapBindingSet = m_device->createBindingSet(bindingSetDesc, m_importanceMapBindingLayout);
    }
}

void EnvMapImportanceSamplingBaker::GenerateImportanceMap(nvrhi::CommandListHandle commandList, nvrhi::TextureHandle sourceCubemap)
{
    nvrhi::ComputeState state;
    state.pipeline = m_importanceMapPipeline;
    state.bindings = { m_importanceMapBindingSet };

    uint32_t groupCount = (EMISB_IMPORTANCE_MAP_DIM+EMISB_NUM_COMPUTE_THREADS_PER_DIM-1) / EMISB_NUM_COMPUTE_THREADS_PER_DIM;

    EnvMapImportanceSamplingBakerConstants constants;
    FillBakerConsts( constants, sourceCubemap, -1 );    // sampleIndex not relevant during importance map generation

    {
        RAII_SCOPE(commandList->beginMarker("GenIM"); , commandList->endMarker(); );
        commandList->writeBuffer(m_builderConstants, &constants, sizeof(EnvMapImportanceSamplingBakerConstants));
        commandList->setComputeState(state);
        commandList->dispatch(groupCount, groupCount);
    }

    {
        //RAII_SCOPE(commandList->beginMarker("GenMIPs");, commandList->endMarker(); );
        m_MIPMapPass->Dispatch(commandList);
    }

    commandList->setTextureState(m_importanceMapTexture, nvrhi::AllSubresources, nvrhi::ResourceStates::UnorderedAccess);
    commandList->commitBarriers();

    m_envMapImportanceSamplingParams.ImportanceBaseMip = constants.ImportanceMapBaseMip;
    m_envMapImportanceSamplingParams.ImportanceInvDim = 1.f / float2((float)m_importanceMapTexture->getDesc().width, (float)m_importanceMapTexture->getDesc().height);
    m_envMapImportanceSamplingParams.padding0 = 0;
}

void EnvMapImportanceSamplingBaker::Update(nvrhi::CommandListHandle commandList, nvrhi::TextureHandle sourceCubemap)
{
    RAII_SCOPE( commandList->beginMarker("ISBake");, commandList->endMarker(); );

    GenerateImportanceMap(commandList, sourceCubemap);
}

void EnvMapImportanceSamplingBaker::ExecutePresampling(nvrhi::CommandListHandle commandList, nvrhi::TextureHandle sourceCubemap, int sampleIndex)
{
    assert( false );
#if 0
    assert(m_presampledBuffer);

    if (!m_presamplingBindingSet)
    {
        nvrhi::BindingSetDesc bindingSetDesc;
        bindingSetDesc.bindings = {
            nvrhi::BindingSetItem::ConstantBuffer(0, m_builderConstants),
            nvrhi::BindingSetItem::Texture_SRV(0, sourceCubemap),
            nvrhi::BindingSetItem::Texture_SRV(1, m_importanceMapTexture),
            nvrhi::BindingSetItem::TypedBuffer_UAV(0, m_presampledBuffer),
            nvrhi::BindingSetItem::Sampler(0, m_pointClampSampler),
            nvrhi::BindingSetItem::Sampler(1, m_linearWrapSampler),
        };
        m_presamplingBindingSet = m_device->createBindingSet(bindingSetDesc, m_presamplingBindingLayout);
    }

    EnvMapImportanceSamplingBakerConstants constants;
    FillBakerConsts(constants, sourceCubemap, sampleIndex);    // sampleIndex not relevant during importance map generation

    {
        RAII_SCOPE(commandList->beginMarker("Pre-sampling");, commandList->endMarker(); );
        commandList->writeBuffer(m_builderConstants, &constants, sizeof(EnvMapImportanceSamplingBakerConstants));

        nvrhi::ComputeState state;
        state.pipeline = m_presamplingPipeline;
        state.bindings = { m_presamplingBindingSet };
        commandList->setComputeState(state);
        static_assert((ENVMAP_PRESAMPLED_COUNT % 256) == 0);
        uint32_t groupCount = ENVMAP_PRESAMPLED_COUNT / 256;
        commandList->dispatch(groupCount, groupCount);
    }
#endif
}


bool EnvMapImportanceSamplingBaker::DebugGUI(float indent)
{
    return false;
}
