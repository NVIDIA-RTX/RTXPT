/*
* Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "GeneratePdfMipsPass.h"
#include <donut/engine/ShaderFactory.h>
#include <nvrhi/utils.h>

#include <donut/core/math/math.h>
#include <donut/core/log.h>

using namespace donut::math;

#include "ShaderParameters.h"

GenerateMipsPass::GenerateMipsPass(
    nvrhi::IDevice* device, 
    std::shared_ptr<donut::engine::ShaderFactory> shaderFactory,
    nvrhi::ITexture* sourceEnvironmentMap,
    nvrhi::ITexture* destinationTexture)
    : m_sourceTexture(sourceEnvironmentMap)
    , m_destinationTexture(destinationTexture)
{
    donut::log::debug("Initializing GenerateMipsPass...");

    const auto& destinationDesc = m_destinationTexture->getDesc();

    nvrhi::SamplerDesc samplerDesc;
    samplerDesc.setBorderColor(nvrhi::Color(0.f));
    samplerDesc.setAllFilters(true);
    samplerDesc.setMipFilter(true);
    samplerDesc.setAllAddressModes(nvrhi::SamplerAddressMode::Wrap);
    m_linearSampler = device->createSampler(samplerDesc);

    nvrhi::BindingSetDesc bindingSetDesc;
    bindingSetDesc.bindings = {
        nvrhi::BindingSetItem::PushConstants(0, sizeof(PreprocessEnvironmentMapConstants)),
        nvrhi::BindingSetItem::Sampler(0, m_linearSampler)
    };

    if (sourceEnvironmentMap) 
    {
        bindingSetDesc.bindings.push_back(nvrhi::BindingSetItem::Texture_SRV(0, sourceEnvironmentMap));
    };

    for (uint32_t mipLevel = 0; mipLevel < destinationDesc.mipLevels; mipLevel++)
    {
        bindingSetDesc.bindings.push_back(nvrhi::BindingSetItem::Texture_UAV(
            mipLevel, 
            m_destinationTexture,
            nvrhi::Format::UNKNOWN, 
            nvrhi::TextureSubresourceSet(mipLevel, 1, 0, 1)));
    }

    nvrhi::BindingLayoutHandle bindingLayout;
    nvrhi::utils::CreateBindingSetAndLayout(device, nvrhi::ShaderType::Compute, 0,
        bindingSetDesc, bindingLayout, m_bindingSet);

    std::vector<donut::engine::ShaderMacro> macros = { { "INPUT_ENVIRONMENT_MAP", sourceEnvironmentMap ? "1" : "0" } };

    nvrhi::ShaderHandle shader = shaderFactory->CreateShader("app/RTXDI/PreprocessEnvironmentMap.hlsl", "main", &macros, nvrhi::ShaderType::Compute);

    nvrhi::ComputePipelineDesc pipelineDesc;
    pipelineDesc.bindingLayouts = { bindingLayout };
    pipelineDesc.CS = shader;
    n_pipeline = device->createComputePipeline(pipelineDesc);
}

GenerateMipsPass::~GenerateMipsPass(){}

void GenerateMipsPass::Process(nvrhi::ICommandList* commandList)
{
    commandList->beginMarker("GenerateMips");
    
    const auto& destDesc = m_destinationTexture->getDesc();

    constexpr uint32_t mipLevelsPerPass = 5;
    uint32_t width = destDesc.width;
    uint32_t height = destDesc.height;

    for (uint32_t sourceMipLevel = 0; sourceMipLevel < destDesc.mipLevels; sourceMipLevel += mipLevelsPerPass)
    {
        nvrhi::ComputeState state;
        state.pipeline = n_pipeline;
        state.bindings = { m_bindingSet };
        commandList->setComputeState(state);

        PreprocessEnvironmentMapConstants constants{};
        constants.sourceSize = { destDesc.width, destDesc.height };
        constants.numDestMipLevels = destDesc.mipLevels;
        constants.sourceMipLevel = sourceMipLevel;
        commandList->setPushConstants(&constants, sizeof(constants));

        commandList->dispatch(div_ceil(width, 32), div_ceil(height, 32), 1);

        width = std::max(1u, width >> mipLevelsPerPass);
        height = std::max(1u, height >> mipLevelsPerPass);
        
        commandList->clearState(); // make sure nvrhi inserts a barrier
    }

    commandList->endMarker();
}
