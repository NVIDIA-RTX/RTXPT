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

#include <memory>
#include "../SampleCommon/ComputePass.h"
#include <donut/core/math/math.h>

enum class DebugViewType;

namespace donut::engine
{
    class ShaderFactory;
}

class RenderTargets;
class ShaderDebug;

class DenoisingGuidesBaker
{
public:

    DenoisingGuidesBaker( nvrhi::IDevice* device, std::shared_ptr<donut::engine::ShaderFactory> shaderFactory, const std::unique_ptr<RenderTargets> & renderTargets, const std::shared_ptr<ShaderDebug> & shaderDebug, nvrhi::BindingLayoutHandle globalBindingLayout );
    ~DenoisingGuidesBaker( );

public:
    void                            DenoiseSpecHitT( nvrhi::ICommandList * commandList, nvrhi::BindingSetHandle bindingSet );
    void                            ComputeAvgLayerRadiance( nvrhi::ICommandList * commandList, nvrhi::BindingSetHandle bindingSet );
    // Writes the DLSS-RR guide buffers (albedo/spec albedo/normals+roughness/spec motion vectors) and the denoiser input
    // colour. Must be dispatched at the same point in the frame as the old PostProcess DLSSRRDenoiserPrepareInputs pass -
    // it consumes the denoising layers and writes u_OutputColor, so it is not interchangeable with the passes above.
    void                            PrepareDLSSRRGuides( nvrhi::ICommandList * commandList, nvrhi::BindingSetHandle bindingSet );
    void                            RenderDebugViz( nvrhi::ICommandList * commandList, DebugViewType debugView, nvrhi::BindingSetHandle bindingSet );

    bool                            DebugGUI(float indent);


private:
    const nvrhi::DeviceHandle       m_device;
    
    ComputePass                     m_csDenoiseSpecHitT;
    ComputePass                     m_csComputeAvgLayerRadiance;
    ComputePass                     m_csPrepareDLSSRRGuides;
    ComputePass                     m_csDebugViz;

    nvrhi::BindingLayoutHandle      m_bindingLayout;

    const std::unique_ptr<RenderTargets> & m_renderTargets;
    const std::shared_ptr<ShaderDebug> & m_shaderDebug;
};
    
