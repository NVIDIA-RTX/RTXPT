/*
* Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "RTGIRenderer.h"
#include "SampleCommon/PTPipelineBaker.h"
#include "SampleCommon/SampleCommon.h"
#include "Shaders/SampleConstantBuffer.h"

RTGIRenderer::RTGIRenderer(nvrhi::IDevice* device)
    : m_device(device)
{
}

void RTGIRenderer::CreateRTPipelines(PTPipelineBaker& pipelineBaker)
{
    m_rtgiPipeline = pipelineBaker.CreateVariant("IntroSample/RTGI.hlsl", {}, "RTGI");
    m_rtgiSharcPipeline = pipelineBaker.CreateVariant("IntroSample/RTGI.hlsl", {{"RTGI_SHARC_ENABLED","1"}}, "GISHa");
}

void RTGIRenderer::DestroyRTPipelines(PTPipelineBaker& pipelineBaker)
{
    pipelineBaker.ReleaseVariant(m_rtgiPipeline);
    pipelineBaker.ReleaseVariant(m_rtgiSharcPipeline);
}

bool RTGIRenderer::IsReady() const
{
    return m_rtgiPipeline && m_rtgiPipeline->GetShaderTable();
}

bool RTGIRenderer::IsSharcReady() const
{
    return m_rtgiSharcPipeline && m_rtgiSharcPipeline->GetShaderTable();
}

void RTGIRenderer::OnRenderTargetsRecreated(uint32_t renderWidth, uint32_t renderHeight)
{
    nvrhi::TextureDesc desc;
    desc.width = renderWidth;
    desc.height = renderHeight;
    desc.format = nvrhi::Format::RGBA16_FLOAT;
    desc.dimension = nvrhi::TextureDimension::Texture2D;
    desc.debugName = "RTGIOutput";
    desc.isUAV = true;
    desc.initialState = nvrhi::ResourceStates::UnorderedAccess;
    desc.keepInitialState = true;
    m_output = m_device->createTexture(desc);
}

void RTGIRenderer::Render(nvrhi::CommandListHandle commandList,
                          uint32_t renderWidth, uint32_t renderHeight,
                          nvrhi::BindingSetHandle globalBindingSet,
                          nvrhi::BindingSetHandle descriptorTable,
                          bool useSharcVariant,
                          uint32_t sharcCapacity,
                          uint32_t envMip,
                          bool includeHitEmissive)
{
    ScopedPerfMarker perfMarker("RTGI", commandList);

    bool useSharc = useSharcVariant && IsSharcReady();
    auto shaderTable = useSharc ? m_rtgiSharcPipeline->GetShaderTable()
                                : m_rtgiPipeline->GetShaderTable();
    if (!shaderTable || !m_output)
        return;

    nvrhi::rt::State state;
    state.shaderTable = shaderTable;
    state.bindings = { globalBindingSet, descriptorTable };
    commandList->setRayTracingState(state);

    SampleMiniConstants miniConstants = {};
    if (useSharc)
        miniConstants.params.x = sharcCapacity;
    miniConstants.params.y = includeHitEmissive ? 1u : 0u;
    miniConstants.params.z = envMip;
    commandList->setPushConstants(&miniConstants, sizeof(miniConstants));

    nvrhi::rt::DispatchRaysArguments args;
    args.width = renderWidth;
    args.height = renderHeight;
    args.depth = 1;
    commandList->dispatchRays(args);
}
