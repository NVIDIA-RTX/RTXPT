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
#include <donut/core/math/math.h>

#include "ShaderDebug.hlsli"

namespace donut::engine
{
    class ShaderFactory;
    class CommonRenderPasses;
}

// HOW TO USE
//
// 1.) In your render pass C++ side, #include "../../ShaderDebug.h"
// 2.) You also need access to the global shared_ptr<ShaderDebug> 
// 3.) Add " nvrhi::BindingLayoutItem::RawBuffer_UAV(SHADER_DEBUG_BUFFER_UAV_INDEX), " to your binding layout(s) and
// 4.) Add " nvrhi::BindingSetItem::RawBuffer_UAV(SHADER_DEBUG_BUFFER_UAV_INDEX, m_shaderPrintf->GetGPUWriteBuffer()), " to your binding set(s)
// 5.) 

class ShaderDebug
{
private:
    nvrhi::DeviceHandle             m_device;

    static constexpr int            c_swapchainCount = 3;
    nvrhi::BufferHandle             m_bufferGPU;                    // write
    nvrhi::BufferHandle             m_indirectDrawBufferGPU;
    nvrhi::BufferHandle             m_bufferCPU[c_swapchainCount];  // readback
    unsigned int                    m_currentBufferIndex = 0;

    nvrhi::ShaderHandle             m_trianglesVertexShader;
    nvrhi::ShaderHandle             m_trianglesPixelShader;
    nvrhi::GraphicsPipelineHandle   m_trianglesPipeline;
    nvrhi::ShaderHandle             m_linesVertexShader;
    nvrhi::ShaderHandle             m_linesPixelShader;
    nvrhi::GraphicsPipelineHandle   m_linesPipeline;
    nvrhi::BindingLayoutHandle      m_geometryBindingLayout;
    nvrhi::BindingSetHandle         m_geometryBindingSet;

    nvrhi::TextureHandle            m_debugVizOutput;

    nvrhi::ShaderHandle             m_blendDebugVizPS;
    nvrhi::GraphicsPipelineHandle   m_blendDebugVizPSO;


    std::shared_ptr<donut::engine::ShaderFactory> m_shaderFactory;
    std::shared_ptr<donut::engine::CommonRenderPasses> m_commonPasses;

    std::array<char, SHADER_DEBUG_HEADER_IN_BYTES>              m_initHeader;
    std::array<char, SHADER_DEBUG_BUFFER_IN_BYTES_NO_TRIANGLES> m_lastBuffer;

public:
    ShaderDebug( nvrhi::IDevice* device, nvrhi::ICommandList* commandList, std::shared_ptr<donut::engine::ShaderFactory> shaderFactory, std::shared_ptr<donut::engine::CommonRenderPasses> commonPasses );

    void                    CreateRenderPasses( nvrhi::IFramebuffer * frameBuffer, nvrhi::TextureHandle depthBuffer );

    void                    BeginFrame( nvrhi::ICommandList* commandList, const donut::math::float4x4 & matWorldToClip );
    void                    ClearDebugVizTexture(nvrhi::CommandListHandle commandList); // not merged with BeginFrame since sometimes you want it to persist between frames

    void                    EndFrameAndOutput( nvrhi::ICommandList* commandList, nvrhi::IFramebuffer * frameBuffer, nvrhi::TextureHandle depthBuffer, const nvrhi::Viewport & viewport );

    nvrhi::BufferHandle     GetGPUWriteBuffer()         { return m_bufferGPU; };
    nvrhi::TextureHandle    GetDebugVizTexture()        { return m_debugVizOutput; };

    //void                    DrawLastBuffer( view? Frame buffer? Depth buffer? See line drawing! )

private:
    void                    OutputLastBufferPrints();
    void                    DrawCurrentBufferGeometry(nvrhi::ICommandList* commandList, nvrhi::IFramebuffer * frameBuffer, nvrhi::TextureHandle depthBuffer, const nvrhi::Viewport & viewport);
};
