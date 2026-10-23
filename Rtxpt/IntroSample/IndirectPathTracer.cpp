/*
* Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "IndirectPathTracer.h"
#include "SampleCommon/PTPipelineBaker.h"
#include "SampleCommon/SampleCommon.h"
#include "Shaders/SampleConstantBuffer.h"

IndirectPathTracer::IndirectPathTracer(nvrhi::IDevice* device)
    : m_device(device)
{
}

void IndirectPathTracer::CreateRTPipelines(PTPipelineBaker& pipelineBaker)
{
    m_pipeline = pipelineBaker.CreateVariant("IntroSample/IndirectPathTracer.hlsl", {}, "IndPT");
    m_sharcPipeline = pipelineBaker.CreateVariant("IntroSample/IndirectPathTracer.hlsl", {{"IPT_SHARC_ENABLED","1"}}, "IPTSh");
}

void IndirectPathTracer::DestroyRTPipelines(PTPipelineBaker& pipelineBaker)
{
    pipelineBaker.ReleaseVariant(m_pipeline);
    pipelineBaker.ReleaseVariant(m_sharcPipeline);
}

bool IndirectPathTracer::IsReady() const
{
    return m_pipeline && m_pipeline->GetShaderTable();
}

bool IndirectPathTracer::IsSharcReady() const
{
    return m_sharcPipeline && m_sharcPipeline->GetShaderTable();
}

void IndirectPathTracer::OnRenderTargetsRecreated(uint32_t renderWidth, uint32_t renderHeight)
{
    nvrhi::TextureDesc desc;
    desc.width = renderWidth;
    desc.height = renderHeight;
    desc.format = nvrhi::Format::RGBA16_FLOAT;
    desc.dimension = nvrhi::TextureDimension::Texture2D;
    desc.debugName = "IndirectPathTracerOutput";
    desc.isUAV = true;
    desc.initialState = nvrhi::ResourceStates::UnorderedAccess;
    desc.keepInitialState = true;
    m_output = m_device->createTexture(desc);
}

void IndirectPathTracer::Render(nvrhi::CommandListHandle commandList,
                                uint32_t renderWidth, uint32_t renderHeight,
                                nvrhi::BindingSetHandle globalBindingSet,
                                nvrhi::BindingSetHandle descriptorTable,
                                bool useSharcVariant,
                                uint32_t sharcCapacity,
                                bool primaryEmissiveNEE,
                                bool primaryEnvironmentNEE)
{
    ScopedPerfMarker perfMarker("IndirectPathTracer", commandList);

    bool useSharc = useSharcVariant && IsSharcReady();
    auto shaderTable = useSharc ? m_sharcPipeline->GetShaderTable() : m_pipeline->GetShaderTable();
    if (!shaderTable || !m_output)
        return;

    nvrhi::rt::State state;
    state.shaderTable = shaderTable;
    state.bindings = { globalBindingSet, descriptorTable };
    commandList->setRayTracingState(state);

    SampleMiniConstants miniConstants = {};
    miniConstants.params.x = useSharc ? sharcCapacity : 0;
    miniConstants.params.y = (primaryEmissiveNEE ? 1u : 0u) | (primaryEnvironmentNEE ? 2u : 0u);

    commandList->setPushConstants(&miniConstants, sizeof(miniConstants));

    nvrhi::rt::DispatchRaysArguments args;
    args.width = renderWidth;
    args.height = renderHeight;
    args.depth = 1;
    commandList->dispatchRays(args);
}
