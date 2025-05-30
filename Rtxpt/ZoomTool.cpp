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
#include "ZoomTool.h"
#include "SampleCommon.h"
#include <donut/app/imgui_renderer.h>

#define GLFW_INCLUDE_NONE // Do not include any OpenGL headers
#include <GLFW/glfw3.h>

using namespace donut::math;

#include "ZoomTool.hlsl"


ZoomTool::ZoomTool( nvrhi::IDevice* device, std::shared_ptr<donut::engine::ShaderFactory> shaderFactory )
    : m_device(device)
    , m_bindingCache(device)
{ 

    nvrhi::BindingLayoutDesc layoutDesc;
    layoutDesc.visibility = nvrhi::ShaderType::Compute;
    layoutDesc.bindings = { 
        nvrhi::BindingLayoutItem::VolatileConstantBuffer(0),
        nvrhi::BindingLayoutItem::Texture_UAV(0) };
    m_bindingLayout = m_device->createBindingLayout(layoutDesc);

    nvrhi::ComputePipelineDesc pipelineDesc;

    // These need to know about the scene
    pipelineDesc.bindingLayouts = { m_bindingLayout };
    m_CSZoomTool.Init(m_device, *shaderFactory, "app/ZoomTool.hlsl", "main", std::vector<donut::engine::ShaderMacro>(), pipelineDesc.bindingLayouts);

    m_constantBuffer = m_device->createBuffer(nvrhi::utils::CreateVolatileConstantBufferDesc(sizeof(ZoomToolShaderConstants), "ZoomToolShaderConstants", donut::engine::c_MaxRenderPassConstantBufferVersions));
}

ZoomTool::~ZoomTool( )
{
}

bool ZoomTool::KeyboardUpdate(int key, int scancode, int action, int mods)
{
    if (key == GLFW_KEY_Z && action == GLFW_PRESS && mods == GLFW_MOD_CONTROL)
        m_settings.Enabled = !m_settings.Enabled;

    if (key == GLFW_KEY_Z && mods == GLFW_MOD_CONTROL)
        return true;

    return false;
}

void ZoomTool::MousePosUpdate(double xpos, double ypos)
{
    m_lastMousePos = float2(xpos, ypos);
}

bool ZoomTool::MouseButtonUpdate(int button, int action, int mods)
{
    if (m_settings.Enabled)
    {
        if (button == GLFW_MOUSE_BUTTON_1)
        {
            if (action == GLFW_PRESS)
                m_settings.BoxPos = int2(m_lastMousePos - float2(m_settings.BoxSize.x / 2, m_settings.BoxSize.y / 2));
            return true;
        }
    }
    return false;
}

void ZoomTool::Render( nvrhi::ICommandList * commandList, nvrhi::TextureHandle colorInOut )
{
    if( !m_settings.Enabled )
        return;

    ZoomToolShaderConstants consts{};

    consts.SourceRectangle = float4((float)m_settings.BoxPos.x, (float)m_settings.BoxPos.y, (float)m_settings.BoxPos.x + m_settings.BoxSize.x, (float)m_settings.BoxPos.y + m_settings.BoxSize.y);
    consts.ZoomFactor = m_settings.ZoomFactor;

    {
        RAII_SCOPE(commandList->beginMarker("ZoomTool");, commandList->endMarker(); );

        commandList->writeBuffer(m_constantBuffer, &consts, sizeof(consts));
        nvrhi::BindingSetDesc bindingSetDesc; bindingSetDesc.bindings = {
            nvrhi::BindingSetItem::ConstantBuffer(0, m_constantBuffer),
            nvrhi::BindingSetItem::Texture_UAV(0, colorInOut, nvrhi::Format::RGBA8_UNORM) };

        nvrhi::BindingSetHandle bindingSet = m_bindingCache.GetOrCreateBindingSet(bindingSetDesc, m_bindingLayout);

        int threadGroupCountX = (colorInOut->getDesc().width + 16 - 1) / 16;
        int threadGroupCountY = (colorInOut->getDesc().height + 16 - 1) / 16;

        m_CSZoomTool.Execute(commandList, threadGroupCountX, threadGroupCountY, 1, bindingSet);
    }
}

bool ZoomTool::DebugGUI(float indent)
{
    //ImGui::PushItemWidth(120.0f);

    ImGui::Checkbox("Enabled", &m_settings.Enabled);
    ImGui::InputInt("ZoomFactor", &m_settings.ZoomFactor, 1);
    m_settings.ZoomFactor = donut::math::clamp(m_settings.ZoomFactor, 2, 32);

    ImGui::InputInt2("BoxPos", &m_settings.BoxPos.x);
    ImGui::InputInt2("BoxSize", &m_settings.BoxSize.x);

    //ImGui::PopItemWidth();
    return false;
}

/*
void vaZoomTool::UIPanelTick( vaApplicationBase & application )
{
#ifdef VA_IMGUI_INTEGRATION_ENABLED
#endif

    // Some misc input handling
    {
        if( application.HasFocus( ) && !vaInputMouseBase::GetCurrent( )->IsCaptured( ) 
#ifdef VA_IMGUI_INTEGRATION_ENABLED
        && !ImGui::GetIO().WantTextInput && !ImGui::GetIO().WantCaptureMouse
#endif
        )
        {
            HandleMouseInputs( *vaInputMouseBase::GetCurrent() );
        }
    }
}

*/