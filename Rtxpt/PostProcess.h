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
#include <donut/engine/ShaderFactory.h>
#include <donut/engine/CommonRenderPasses.h>
#include <donut/core/math/math.h>
#include <memory>

using namespace donut::math;

#include "PathTracer/PathTracerShared.h"
#include "RenderTargets.h"
#include "PostProcess.hlsl"
#include "SampleConstantBuffer.h"

namespace donut::engine
{
    class FramebufferFactory;
}

class ShaderDebug;

// TODO: either replace this whole approach with instances of ComputePass OR replace internal m_computeShaders/m_computePSOs with ComputePass
class PostProcess 
{
public:
    enum class ComputePassType
    {
        StablePlanesDebugViz,
        RELAXDenoiserPrepareInputs,
        REBLURDenoiserPrepareInputs,
        RELAXDenoiserFinalMerge,
        REBLURDenoiserFinalMerge,
        DLSSRRDenoiserPrepareInputs,
        DummyPlaceholder,

        MaxCount
    };

private:

    nvrhi::DeviceHandle             m_device;
    std::shared_ptr<donut::engine::CommonRenderPasses> m_commonPasses;

    nvrhi::ShaderHandle             m_computeShaders[(uint32_t)ComputePassType::MaxCount];
    nvrhi::ComputePipelineHandle    m_computePSOs[(uint32_t)ComputePassType::MaxCount];

    nvrhi::SamplerHandle            m_pointSampler;
    nvrhi::SamplerHandle            m_linearSampler;

    nvrhi::BindingLayoutHandle      m_bindingLayoutCS;
    nvrhi::BindingSetHandle         m_bindingSetCS;

    donut::engine::BindingCache     m_bindingCache;

    std::shared_ptr<ShaderDebug>    m_shaderDebug;

public:

    PostProcess(
        nvrhi::IDevice* device,
        std::shared_ptr<donut::engine::ShaderFactory> shaderFactory,
        std::shared_ptr<donut::engine::CommonRenderPasses> commonPasses,
        std::shared_ptr<ShaderDebug> shaderDebug
        //, std::shared_ptr<engine::FramebufferFactory> colorFramebufferFactory
    );

    void Apply(nvrhi::ICommandList* commandList, ComputePassType passType, nvrhi::BufferHandle consts, SampleMiniConstants & miniConsts, nvrhi::BindingSetHandle bindingSet, nvrhi::BindingLayoutHandle bindingLayout, uint32_t width, uint32_t height);
    void Apply(nvrhi::ICommandList* commandList, ComputePassType passType, int pass, nvrhi::BufferHandle consts, SampleMiniConstants & miniConsts, nvrhi::ITexture* workTexture, RenderTargets & renderTargets, nvrhi::ITexture* sourceTexture);
};
