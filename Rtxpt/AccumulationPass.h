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
    class IView;
}

class AccumulationPass
{
public:
    AccumulationPass(nvrhi::IDevice* device, std::shared_ptr<donut::engine::ShaderFactory> shaderFactory);

    void CreatePipeline();
    void CreateBindingSet(nvrhi::ITexture* inputTexture, nvrhi::ITexture* outputTexture);
    void Render(nvrhi::ICommandList* commandList, const donut::engine::IView& sourceView, const donut::engine::IView& upscaledView, float accumulationWeight);

private:
    nvrhi::DeviceHandle m_device;
    nvrhi::ShaderHandle m_computeShader;
    nvrhi::ComputePipelineHandle m_computePipeline;
    nvrhi::BindingLayoutHandle m_bindingLayout;
    nvrhi::BindingSetHandle m_bindingSet;
    nvrhi::SamplerHandle m_sampler;
    nvrhi::TextureHandle m_compositedColor;

    std::shared_ptr<donut::engine::ShaderFactory> m_shaderFactory;
};
