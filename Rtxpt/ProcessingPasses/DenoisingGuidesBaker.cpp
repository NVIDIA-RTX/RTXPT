///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Copyright (C) 2016-2021, Intel Corporation 
// 
// SPDX-License-Identifier: MIT
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
// Author(s):  Filip Strugar (filip.strugar@intel.com)
//
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <nvrhi/utils.h>
#include <donut/engine/ShaderFactory.h>
#include <donut/engine/CommonRenderPasses.h>
#include "DenoisingGuidesBaker.h"
#include "../SampleCommon/SampleCommon.h"
#include <donut/app/imgui_renderer.h>

using namespace donut::math;

#include "DenoisingGuidesBaker.hlsl"

#include "../Shaders/PathTracer/PathTracerDebug.hlsli"

#include "../SampleCommon/RenderTargets.h"

#include "../Shaders/SampleConstantBuffer.h"

static_assert( sizeof(DenoisingGuidesBakerConstants) == sizeof(SampleMiniConstants) );

DenoisingGuidesBaker::DenoisingGuidesBaker( nvrhi::IDevice* device, std::shared_ptr<donut::engine::ShaderFactory> shaderFactory, const std::unique_ptr<RenderTargets> & renderTargets, const std::shared_ptr<ShaderDebug> & shaderDebug, nvrhi::BindingLayoutHandle bindingLayout )
    : m_device(device)
    , m_renderTargets(renderTargets)
    , m_shaderDebug(shaderDebug)
    , m_bindingLayout(bindingLayout)
{ 

    // nvrhi::BindingLayoutDesc layoutDesc;
    // layoutDesc.visibility = nvrhi::ShaderType::Compute;
    // layoutDesc.bindings = { 
    //     nvrhi::BindingLayoutItem::VolatileConstantBuffer(0),
    //     nvrhi::BindingLayoutItem::Texture_UAV(0) };
    // m_bindingLayout = m_device->createBindingLayout(layoutDesc);

    nvrhi::ComputePipelineDesc pipelineDesc;

    // These need to know about the scene
    pipelineDesc.bindingLayouts = { m_bindingLayout };
    m_csDenoiseSpecHitT.Init(m_device, *shaderFactory, "app/ProcessingPasses/DenoisingGuidesBaker.hlsl", "DenoiseSpecHitT", std::vector<donut::engine::ShaderMacro>(), pipelineDesc.bindingLayouts);
    m_csComputeAvgLayerRadiance.Init(m_device, *shaderFactory, "app/ProcessingPasses/DenoisingGuidesBaker.hlsl", "ComputeAvgLayerRadiance", std::vector<donut::engine::ShaderMacro>(), pipelineDesc.bindingLayouts);
    m_csPrepareDLSSRRGuides.Init(m_device, *shaderFactory, "app/ProcessingPasses/DenoisingGuidesBaker.hlsl", "PrepareDLSSRRGuides", std::vector<donut::engine::ShaderMacro>(), pipelineDesc.bindingLayouts);
    m_csDebugViz.Init(m_device, *shaderFactory, "app/ProcessingPasses/DenoisingGuidesBaker.hlsl", "DebugViz", std::vector<donut::engine::ShaderMacro>(), pipelineDesc.bindingLayouts);

    //m_constantBuffer = m_device->createBuffer(nvrhi::utils::CreateVolatileConstantBufferDesc(sizeof(DenoisingGuidesBakerConstants), "DenoisingGuidesBakerConstants", donut::engine::c_MaxRenderPassConstantBufferVersions));
}

DenoisingGuidesBaker::~DenoisingGuidesBaker( )
{
}

#pragma optimize("", off)

void DenoisingGuidesBaker::DenoiseSpecHitT(nvrhi::ICommandList* commandList, nvrhi::BindingSetHandle bindingSet)
{
    RTXPT_GPU_SCOPE(commandList, "DenoiseSpecHitT");

    int threadGroupCountX = div_ceil(m_renderTargets->RenderSize.x, DGB_2D_THREADGROUP_SIZE);
    int threadGroupCountY = div_ceil(m_renderTargets->RenderSize.y, DGB_2D_THREADGROUP_SIZE);

    static int passCount = 1;
    DenoisingGuidesBakerConstants consts;
    for( int pass = 0; pass < passCount; pass++ )
    {
        // ping
        commandList->setTextureState(m_renderTargets->SpecularHitT, nvrhi::AllSubresources, nvrhi::ResourceStates::UnorderedAccess);
        consts = DenoisingGuidesBakerConstants { .RenderResolution = m_renderTargets->RenderSize, .DisplayResolution = m_renderTargets->DisplaySize, .DebugView = (int)0, .Ping = 1 };
        m_csDenoiseSpecHitT.Execute(commandList, threadGroupCountX, threadGroupCountY, 1, bindingSet, nullptr, nullptr, &consts, sizeof(consts));
        // pong
        commandList->setTextureState(m_renderTargets->SpecularHitT, nvrhi::AllSubresources, nvrhi::ResourceStates::UnorderedAccess);
        commandList->setTextureState(m_renderTargets->ScratchFloat1, nvrhi::AllSubresources, nvrhi::ResourceStates::UnorderedAccess);
        consts = DenoisingGuidesBakerConstants { .RenderResolution = m_renderTargets->RenderSize, .DisplayResolution = m_renderTargets->DisplaySize, .DebugView = (int)0, .Ping = 0 };
        m_csDenoiseSpecHitT.Execute(commandList, threadGroupCountX, threadGroupCountY, 1, bindingSet, nullptr, nullptr, &consts, sizeof(consts));
    }
    commandList->setTextureState(m_renderTargets->SpecularHitT, nvrhi::AllSubresources, nvrhi::ResourceStates::UnorderedAccess);
}

void DenoisingGuidesBaker::ComputeAvgLayerRadiance(nvrhi::ICommandList* commandList, nvrhi::BindingSetHandle bindingSet)
{
    RTXPT_GPU_SCOPE(commandList, "ComputeAvgLayerRadiance");

    const auto& texDesc = m_renderTargets->DenoiserAvgLayerRadianceHalfRes->getDesc();
    int halfWidth = (int)texDesc.width;
    int halfHeight = (int)texDesc.height;

    int threadGroupCountX = div_ceil(halfWidth, DGB_2D_THREADGROUP_SIZE);
    int threadGroupCountY = div_ceil(halfHeight, DGB_2D_THREADGROUP_SIZE);

    DenoisingGuidesBakerConstants consts { .RenderResolution = m_renderTargets->RenderSize, .DisplayResolution = m_renderTargets->DisplaySize, .DebugView = 0, .Ping = 0 };
    m_csComputeAvgLayerRadiance.Execute(commandList, threadGroupCountX, threadGroupCountY, 1, bindingSet, nullptr, nullptr, &consts, sizeof(consts));

    commandList->setTextureState(m_renderTargets->DenoiserAvgLayerRadianceHalfRes, nvrhi::AllSubresources, nvrhi::ResourceStates::UnorderedAccess);
}

void DenoisingGuidesBaker::PrepareDLSSRRGuides(nvrhi::ICommandList* commandList, nvrhi::BindingSetHandle bindingSet)
{
    RTXPT_GPU_SCOPE(commandList, "DLSSRR_PrepareInputs");

    // Dispatch size is taken from OutputColor rather than RenderSize to stay identical to the PostProcess pass this
    // replaced. The shader bounds-checks against g_Const.ptConsts.imageWidth/Height and does not read the push constants
    // at all (the old call site passed all zeros); they are filled here only for consistency with the passes above.
    const nvrhi::TextureDesc & tdesc = m_renderTargets->OutputColor->getDesc();

    int threadGroupCountX = div_ceil((int)tdesc.width, DGB_2D_THREADGROUP_SIZE);
    int threadGroupCountY = div_ceil((int)tdesc.height, DGB_2D_THREADGROUP_SIZE);

    DenoisingGuidesBakerConstants consts { .RenderResolution = m_renderTargets->RenderSize, .DisplayResolution = m_renderTargets->DisplaySize, .DebugView = 0, .Ping = 0 };
    m_csPrepareDLSSRRGuides.Execute(commandList, threadGroupCountX, threadGroupCountY, 1, bindingSet, nullptr, nullptr, &consts, sizeof(consts));
}

void DenoisingGuidesBaker::RenderDebugViz( nvrhi::ICommandList * commandList, DebugViewType debugView, nvrhi::BindingSetHandle bindingSet )
{
    //if( !m_settings.Enabled )
    //    return;

    
    DenoisingGuidesBakerConstants consts { .RenderResolution = m_renderTargets->RenderSize, .DisplayResolution = m_renderTargets->DisplaySize, .DebugView = (int)debugView } ;

    RTXPT_GPU_SCOPE(commandList, "DebugViz");

    int threadGroupCountX = div_ceil(consts.RenderResolution.x, DGB_2D_THREADGROUP_SIZE);
    int threadGroupCountY = div_ceil(consts.RenderResolution.y, DGB_2D_THREADGROUP_SIZE);

    m_csDebugViz.Execute(commandList, threadGroupCountX, threadGroupCountY, 1, bindingSet, nullptr, nullptr, &consts, sizeof(consts) );
}

bool DenoisingGuidesBaker::DebugGUI(float indent)
{
    //ImGui::PushItemWidth(120.0f);

    // ImGui::InputInt2("BoxPos", &m_settings.BoxPos.x);
    // ImGui::InputInt2("BoxSize", &m_settings.BoxSize.x);

    //ImGui::PopItemWidth();
    return false;
}

