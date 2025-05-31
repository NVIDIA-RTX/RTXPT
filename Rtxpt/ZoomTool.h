/*
* Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// See https://github.com/fstrugar/XeGTAO/blob/master/Source/Rendering/Misc/vaZoomTool.cpp
// Copyright (C) 2016-2021, Intel Corporation, 
// SPDX-License-Identifier: MIT, Author(s):  Filip Strugar (filip.strugar@intel.com)
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


#pragma once

#include <memory>
#include "ComputePass.h"
#include <donut/core/math/math.h>
#include <donut/engine/BindingCache.h>

namespace donut::engine
{
    class ShaderFactory;
}

class ZoomTool
{
public:
    struct ZoomSettings
    {
        bool                Enabled     = false;
        int                 ZoomFactor  = 4;
        donut::math::int2   BoxPos      = { 400, 300 };
        donut::math::int2   BoxSize     = { 128, 96 };
    };

    ZoomTool( nvrhi::IDevice* device, std::shared_ptr<donut::engine::ShaderFactory> shaderFactory );
    ~ZoomTool( );

public:
    ZoomSettings &                  Settings( )         { return m_settings; }

     void                           Render( nvrhi::ICommandList * commandList, nvrhi::TextureHandle colorInOut );

     bool                           KeyboardUpdate(int key, int scancode, int action, int mods);
     void                           MousePosUpdate(double xpos, double ypos);
     bool                           MouseButtonUpdate(int button, int action, int mods);

     bool                           Enabled( ) const    { return m_settings.Enabled; }

     bool                           DebugGUI(float indent);

// private:
//     virtual void                    UIPanelTickAlways( vaApplicationBase & application ) override;
//     virtual void                    UIPanelTick( vaApplicationBase & application ) override;

private:
    const nvrhi::DeviceHandle       m_device;
    
    ZoomSettings                    m_settings;
    nvrhi::BufferHandle             m_constantBuffer;
    ComputePass                     m_CSZoomTool;

    nvrhi::BindingLayoutHandle      m_bindingLayout;
    donut::engine::BindingCache     m_bindingCache;

    donut::math::float2             m_lastMousePos;
};
    
