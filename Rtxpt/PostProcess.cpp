/*
* Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "PostProcess.h"

#include <donut/engine/FramebufferFactory.h>
#include <donut/engine/CommonRenderPasses.h>
#include "ShaderDebug.h"

using namespace donut;
using namespace donut::math;
using namespace donut::engine;

PostProcess::PostProcess( nvrhi::IDevice* device, std::shared_ptr<donut::engine::ShaderFactory> shaderFactory, 
    std::shared_ptr<engine::CommonRenderPasses> commonPasses, std::shared_ptr<ShaderDebug> shaderDebug
    )
    : m_device(device)
    , m_commonPasses(commonPasses)
    , m_bindingCache(device)
    , m_shaderDebug(shaderDebug)
{

    for (uint32_t i = 0; i < (uint32_t)ComputePassType::MaxCount; i++)
    {
        std::vector<donut::engine::ShaderMacro> shaderMacros;
        switch ((ComputePassType)i)
        {
        case(ComputePassType::StablePlanesDebugViz):
            shaderMacros.push_back(donut::engine::ShaderMacro({ "STABLE_PLANES_DEBUG_VIZ", "1" }));
            break;
        case(ComputePassType::RELAXDenoiserPrepareInputs):
            shaderMacros.push_back(donut::engine::ShaderMacro({ "DENOISER_PREPARE_INPUTS", "1" }));
            shaderMacros.push_back(donut::engine::ShaderMacro({ "USE_RELAX", "1" }));
            break;
        case(ComputePassType::REBLURDenoiserPrepareInputs):
            shaderMacros.push_back(donut::engine::ShaderMacro({ "DENOISER_PREPARE_INPUTS", "1" }));
            shaderMacros.push_back(donut::engine::ShaderMacro({ "USE_RELAX", "0" }));
            break;
        case(ComputePassType::RELAXDenoiserFinalMerge): 
            shaderMacros.push_back(donut::engine::ShaderMacro({ "DENOISER_FINAL_MERGE", "1" })); 
            shaderMacros.push_back(donut::engine::ShaderMacro({ "USE_RELAX", "1" }));
            break;
        case(ComputePassType::REBLURDenoiserFinalMerge): 
            shaderMacros.push_back(donut::engine::ShaderMacro({ "DENOISER_FINAL_MERGE", "1" }));
            shaderMacros.push_back(donut::engine::ShaderMacro({ "USE_RELAX", "0" }));
            break;
        case(ComputePassType::DLSSRRDenoiserPrepareInputs):
            shaderMacros.push_back(donut::engine::ShaderMacro({ "DENOISER_PREPARE_INPUTS", "1" }));
            shaderMacros.push_back(donut::engine::ShaderMacro({ "DENOISER_DLSS_RR", "1" }));
            break;
        case(ComputePassType::DummyPlaceholder): shaderMacros.push_back(donut::engine::ShaderMacro({ "DUMMY_PLACEHOLDER_EFFECT", "1" })); break;
        };
        m_computeShaders[i] = shaderFactory->CreateShader("app/PostProcess.hlsl", "main", &shaderMacros, nvrhi::ShaderType::Compute);
    }
    //m_MainCS = shaderFactory->CreateShader("app/PostProcess.hlsl", "main", &std::vector<donut::engine::ShaderMacro>(1, donut::engine::ShaderMacro("USE_CS", "1")), nvrhi::ShaderType::Compute);

    nvrhi::BindingLayoutDesc layoutDesc;

    layoutDesc.visibility = nvrhi::ShaderType::Compute | nvrhi::ShaderType::Pixel;
    layoutDesc.bindings = {
        nvrhi::BindingLayoutItem::VolatileConstantBuffer(0),
        nvrhi::BindingLayoutItem::PushConstants(1, sizeof(SampleMiniConstants)),
        nvrhi::BindingLayoutItem::Texture_SRV(0),
        nvrhi::BindingLayoutItem::Texture_UAV(0),
        nvrhi::BindingLayoutItem::Texture_UAV(1),
        nvrhi::BindingLayoutItem::Texture_SRV(2),
        nvrhi::BindingLayoutItem::Texture_SRV(3),
        nvrhi::BindingLayoutItem::Texture_SRV(4),
        nvrhi::BindingLayoutItem::Texture_SRV(5),
        nvrhi::BindingLayoutItem::Texture_SRV(6),
        nvrhi::BindingLayoutItem::Texture_SRV(7),
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(10),
        nvrhi::BindingLayoutItem::Sampler(0)
    };
    m_bindingLayoutCS = m_device->createBindingLayout(layoutDesc);

    nvrhi::SamplerDesc samplerDesc;
    samplerDesc.setBorderColor(nvrhi::Color(0.f));
    samplerDesc.setAllFilters(true);
    samplerDesc.setMipFilter(false);
    samplerDesc.setAllAddressModes(nvrhi::SamplerAddressMode::Wrap);
    m_linearSampler = m_device->createSampler(samplerDesc);

    samplerDesc.setAllFilters(false);
    m_pointSampler = m_device->createSampler(samplerDesc);
}

void PostProcess::Apply(nvrhi::ICommandList* commandList, ComputePassType passType, nvrhi::BufferHandle consts, SampleMiniConstants & miniConsts, nvrhi::BindingSetHandle bindingSet, nvrhi::BindingLayoutHandle bindingLayout, uint32_t width, uint32_t height)
{
    uint passIndex = (uint32_t)passType;

    if (m_computePSOs[passIndex] == nullptr)
    {
        nvrhi::ComputePipelineDesc pipelineDesc;
        pipelineDesc.bindingLayouts = { bindingLayout };
        pipelineDesc.CS = m_computeShaders[passIndex];
        m_computePSOs[passIndex] = m_device->createComputePipeline(pipelineDesc);
    }

    nvrhi::ComputeState state;
    state.bindings = { bindingSet };
    state.pipeline = m_computePSOs[passIndex];

    commandList->setComputeState(state);

    const dm::uint  threads = NUM_COMPUTE_THREADS_PER_DIM;
    const dm::uint2 dispatchSize = dm::uint2((width + threads - 1) / threads, (height + threads - 1) / threads);
    commandList->setPushConstants(&miniConsts, sizeof(miniConsts));
    commandList->dispatch(dispatchSize.x, dispatchSize.y);
}

void PostProcess::Apply( nvrhi::ICommandList* commandList, ComputePassType passType, int pass, nvrhi::BufferHandle consts, SampleMiniConstants & miniConsts, nvrhi::ITexture* workTexture, RenderTargets & renderTargets, nvrhi::ITexture* sourceTexture)
{
    // commandList->beginMarker("PostProcessCS");

    assert((uint32_t)passType >= 0 && passType < ComputePassType::MaxCount);

    nvrhi::BindingSetDesc bindingSetDesc;
    bindingSetDesc.bindings = {
    		nvrhi::BindingSetItem::ConstantBuffer(0, consts),
            nvrhi::BindingSetItem::PushConstants(1, sizeof(SampleMiniConstants)), 
            nvrhi::BindingSetItem::Texture_SRV(0, (sourceTexture!=nullptr)?(sourceTexture):(m_commonPasses->m_WhiteTexture.Get())),
            nvrhi::BindingSetItem::Texture_UAV(0, workTexture),
            nvrhi::BindingSetItem::Texture_UAV(1, m_shaderDebug->GetDebugVizTexture()),
    		//nvrhi::BindingSetItem::StructuredBuffer_SRV(1, renderTargets.DenoiserPixelDataBuffer),
            nvrhi::BindingSetItem::Texture_SRV(2, renderTargets.DenoiserOutDiffRadianceHitDist[pass]),
            nvrhi::BindingSetItem::Texture_SRV(3, renderTargets.DenoiserOutSpecRadianceHitDist[pass]),
            nvrhi::BindingSetItem::Texture_SRV(4, m_commonPasses->m_WhiteTexture.Get()),
            nvrhi::BindingSetItem::Texture_SRV(5, (renderTargets.DenoiserOutValidation!=nullptr)?(renderTargets.DenoiserOutValidation):((nvrhi::TextureHandle)m_commonPasses->m_WhiteTexture.Get())),
            nvrhi::BindingSetItem::Texture_SRV(6, renderTargets.DenoiserViewspaceZ),
            nvrhi::BindingSetItem::Texture_SRV(7, renderTargets.DenoiserDisocclusionThresholdMix),
            nvrhi::BindingSetItem::StructuredBuffer_SRV(10, renderTargets.StablePlanesBuffer),
            nvrhi::BindingSetItem::Sampler(0, (true) ? m_linearSampler : m_pointSampler)
    	};

    nvrhi::BindingSetHandle bindingSet = m_bindingCache.GetOrCreateBindingSet(bindingSetDesc, m_bindingLayoutCS);

    Apply(commandList, passType, consts, miniConsts, bindingSet, m_bindingLayoutCS, workTexture->getDesc().width, workTexture->getDesc().height);

    // commandList->endMarker();
}

