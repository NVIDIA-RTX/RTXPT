/*
* Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "AccumulationPass.h"

#include <donut/engine/ShaderFactory.h>
#include <donut/engine/View.h>
#include <donut/core/log.h>

using namespace donut::math;
#include "RTXDI/ShaderParameters.h"

using namespace donut::engine;


AccumulationPass::AccumulationPass(nvrhi::IDevice* device, std::shared_ptr<ShaderFactory> shaderFactory)
    : m_device(device)
    , m_shaderFactory(shaderFactory)
{
    nvrhi::BindingLayoutDesc bindingLayoutDesc;
    bindingLayoutDesc.visibility = nvrhi::ShaderType::Compute;
    bindingLayoutDesc.bindings = {
        nvrhi::BindingLayoutItem::Texture_SRV(0),
        nvrhi::BindingLayoutItem::Texture_UAV(0),
        nvrhi::BindingLayoutItem::Sampler(0),
        nvrhi::BindingLayoutItem::PushConstants(0, sizeof(AccumulationConstants))
    };

    m_bindingLayout = m_device->createBindingLayout(bindingLayoutDesc);

    auto samplerDesc = nvrhi::SamplerDesc().setAllFilters(true);

    m_sampler = m_device->createSampler(samplerDesc);
}

void AccumulationPass::CreatePipeline()
{
    m_computeShader = m_shaderFactory->CreateShader("app/AccumulationPass.hlsl", "main", nullptr, nvrhi::ShaderType::Compute);

    nvrhi::ComputePipelineDesc pipelineDesc;
    pipelineDesc.bindingLayouts = { m_bindingLayout };
    pipelineDesc.CS = m_computeShader;
    m_computePipeline = m_device->createComputePipeline(pipelineDesc);
}

void AccumulationPass::CreateBindingSet(nvrhi::ITexture* inputTexture, nvrhi::ITexture* outputTexture)
{
    nvrhi::BindingSetDesc bindingSetDesc;

    bindingSetDesc.bindings = {
        nvrhi::BindingSetItem::Texture_SRV(0, inputTexture),
        nvrhi::BindingSetItem::Texture_UAV(0, outputTexture),
        nvrhi::BindingSetItem::Sampler(0, m_sampler),
        nvrhi::BindingSetItem::PushConstants(0, sizeof(AccumulationConstants))
    };

    m_bindingSet = m_device->createBindingSet(bindingSetDesc, m_bindingLayout);

    m_compositedColor = outputTexture;
}

void AccumulationPass::Render(
    nvrhi::ICommandList* commandList,
    const donut::engine::IView& sourceView,
    const donut::engine::IView& upscaledView,
    float accumulationWeight)
{
    commandList->beginMarker("Accumulation");

    const auto sourceViewport = sourceView.GetViewportState().viewports[0];
    const auto upscaledViewport = upscaledView.GetViewportState().viewports[0];

    const auto& inputDesc = m_compositedColor->getDesc();

    AccumulationConstants constants = {};
    constants.inputSize = float2(sourceViewport.width(), sourceViewport.height());
    constants.inputTextureSizeInv = float2(1.f / float(inputDesc.width), 1.f / float(inputDesc.height));
    constants.outputSize = float2(upscaledViewport.width(), upscaledViewport.height());
    constants.pixelOffset = sourceView.GetPixelOffset();
    constants.blendFactor = accumulationWeight;

    nvrhi::ComputeState state;
    state.bindings = { m_bindingSet };
    state.pipeline = m_computePipeline;
    commandList->setComputeState(state);

    commandList->setPushConstants(&constants, sizeof(constants));
    
    commandList->dispatch(
        dm::div_ceil(upscaledView.GetViewExtent().width(), 8), 
        dm::div_ceil(upscaledView.GetViewExtent().height(), 8), 
        1);

    commandList->endMarker();
}
