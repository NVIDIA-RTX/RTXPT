/*
* Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "SHaRCRenderer.h"
#include "SampleCommon/ComputePipelineBaker.h"
#include "SampleCommon/PTPipelineBaker.h"
#include "SampleCommon/SampleCommon.h"
#include "Shaders/SampleConstantBuffer.h"

static uint32_t DivCeil(uint32_t x, uint32_t y) { return (x + y - 1) / y; }

SHaRCRenderer::SHaRCRenderer(nvrhi::IDevice* device)
    : m_device(device)
{
}

SHaRCRenderer::~SHaRCRenderer()
{
}

void SHaRCRenderer::CreatePipelines(nvrhi::BindingLayoutHandle globalBindingLayout,
                                     nvrhi::BindingLayoutHandle globalBindlessLayout,
                                     ComputePipelineBaker& computeBaker)
{
    m_resolveVariant = computeBaker.CreateVariant(
        "IntroSample/SHaRCResolve.hlsl",
        "main",
        {},
        { globalBindingLayout, globalBindlessLayout },
        "SHaRC");
}

void SHaRCRenderer::DestroyPipelines(ComputePipelineBaker& computeBaker)
{
    computeBaker.ReleaseVariant(m_resolveVariant);
}

void SHaRCRenderer::CreateRTPipeline(PTPipelineBaker& pipelineBaker)
{
    m_updatePipeline = pipelineBaker.CreateVariant("IntroSample/SHaRCUpdate.hlsl", {}, "SHaUpd");
}

void SHaRCRenderer::DestroyRTPipeline(PTPipelineBaker& pipelineBaker)
{
    pipelineBaker.ReleaseVariant(m_updatePipeline);
}

bool SHaRCRenderer::IsUpdateReady() const
{
    return m_updatePipeline && m_updatePipeline->GetShaderTable() && m_hashEntries;
}

void SHaRCRenderer::DispatchUpdate(nvrhi::CommandListHandle commandList,
                                    uint32_t renderWidth, uint32_t renderHeight,
                                    nvrhi::BindingSetHandle globalBindingSet,
                                    nvrhi::BindingSetHandle descriptorTable)
{
    ScopedPerfMarker perfMarker("SHaRC Update", commandList);

    if (!IsUpdateReady())
        return;

    if (m_needsClear)
    {
        commandList->clearBufferUInt(m_hashEntries, 0);
        commandList->clearBufferUInt(m_accumulation, 0);
        commandList->clearBufferUInt(m_resolved, 0);
        m_needsClear = false;

        commandList->setBufferState(m_hashEntries, nvrhi::ResourceStates::UnorderedAccess);
        commandList->setBufferState(m_accumulation, nvrhi::ResourceStates::UnorderedAccess);
        commandList->setBufferState(m_resolved, nvrhi::ResourceStates::UnorderedAccess);
        commandList->commitBarriers();
    }

    nvrhi::rt::State state;
    state.shaderTable = m_updatePipeline->GetShaderTable();
    state.bindings = { globalBindingSet, descriptorTable };
    commandList->setRayTracingState(state);

    SampleMiniConstants miniConstants = {};
    miniConstants.params.x = m_capacity;
    commandList->setPushConstants(&miniConstants, sizeof(miniConstants));

    nvrhi::rt::DispatchRaysArguments args;
    args.width = (renderWidth + 3) / 4;
    args.height = (renderHeight + 3) / 4;
    args.depth = 1;
    commandList->dispatchRays(args);

    commandList->setBufferState(m_hashEntries, nvrhi::ResourceStates::UnorderedAccess);
    commandList->setBufferState(m_accumulation, nvrhi::ResourceStates::UnorderedAccess);
    commandList->setBufferState(m_resolved, nvrhi::ResourceStates::UnorderedAccess);
    commandList->commitBarriers();
}

void SHaRCRenderer::CreateBuffers()
{
    m_needsClear = true;

    // Hash entries: uint64_t per element (8 bytes)
    {
        nvrhi::BufferDesc desc;
        desc.byteSize = m_capacity * sizeof(uint64_t);
        desc.structStride = sizeof(uint64_t);
        desc.debugName = "SHaRC_HashEntries";
        desc.canHaveUAVs = true;
        desc.initialState = nvrhi::ResourceStates::UnorderedAccess;
        desc.keepInitialState = true;
        m_hashEntries = m_device->createBuffer(desc);
    }

    // Accumulation: uint4 per element (16 bytes)
    {
        nvrhi::BufferDesc desc;
        desc.byteSize = m_capacity * 16;
        desc.structStride = 16;
        desc.debugName = "SHaRC_Accumulation";
        desc.canHaveUAVs = true;
        desc.initialState = nvrhi::ResourceStates::UnorderedAccess;
        desc.keepInitialState = true;
        m_accumulation = m_device->createBuffer(desc);
    }

    // Resolved: SharcPackedData per element (float16_t4 + uint + uint = 16 bytes)
    {
        nvrhi::BufferDesc desc;
        desc.byteSize = m_capacity * 16;
        desc.structStride = 16;
        desc.debugName = "SHaRC_Resolved";
        desc.canHaveUAVs = true;
        desc.initialState = nvrhi::ResourceStates::UnorderedAccess;
        desc.keepInitialState = true;
        m_resolved = m_device->createBuffer(desc);
    }
}

void SHaRCRenderer::Resolve(nvrhi::CommandListHandle commandList,
                             nvrhi::BindingSetHandle globalBindingSet,
                             nvrhi::BindingSetHandle descriptorTable)
{
    ScopedPerfMarker perfMarker("SHaRC Resolve", commandList);

    auto pipeline = m_resolveVariant ? m_resolveVariant->GetPipeline() : nullptr;
    if (!pipeline || !m_hashEntries)
        return;

    nvrhi::ComputeState state;
    state.bindings = { globalBindingSet, descriptorTable };
    state.pipeline = pipeline;
    commandList->setComputeState(state);

    SampleMiniConstants miniConstants = {};
    miniConstants.params.x = m_capacity;
    commandList->setPushConstants(&miniConstants, sizeof(miniConstants));

    commandList->dispatch(DivCeil(m_capacity, 256), 1, 1);

    commandList->setBufferState(m_hashEntries, nvrhi::ResourceStates::UnorderedAccess);
    commandList->setBufferState(m_accumulation, nvrhi::ResourceStates::UnorderedAccess);
    commandList->setBufferState(m_resolved, nvrhi::ResourceStates::UnorderedAccess);
    commandList->commitBarriers();
}
