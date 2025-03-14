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
#include <memory>

namespace donut::engine
{
    class ShaderFactory;
}

class GenerateMipsPass
{
public:
    GenerateMipsPass(
        nvrhi::IDevice* device,
        std::shared_ptr<donut::engine::ShaderFactory> shaderFactory,
        nvrhi::ITexture* sourceEnvironmentMap,
        nvrhi::ITexture* destinationTexture);
    ~GenerateMipsPass();
    void Process(nvrhi::ICommandList* commandList);

private:
    nvrhi::ComputePipelineHandle n_pipeline;
    nvrhi::BindingSetHandle m_bindingSet;
    nvrhi::TextureHandle m_sourceTexture;
    nvrhi::TextureHandle m_destinationTexture;
    nvrhi::SamplerHandle m_linearSampler;
};
