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

#include "../Shaders/Libraries/ShaderDebug/ShaderDebug.hlsl"
#include "SampleCommon/SampleCommon.h"

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

    nvrhi::BufferHandle             m_bufferGPU;                    // write
    nvrhi::BufferHandle             m_indirectDrawBufferGPU;
    nvrhi::BufferHandle             m_copyDispatchArgs;             // { header's CopyGroupCount, 1, 1 }, see CopyForOutput
    nvrhi::BufferHandle             m_bufferCPU[c_SwapchainCount+1];  // readback
    unsigned int                    m_currentBufferIndex = 0;

    nvrhi::ShaderHandle             m_trianglesVertexShader;
    nvrhi::ShaderHandle             m_trianglesPixelShader;
    nvrhi::GraphicsPipelineHandle   m_trianglesPipeline;
    nvrhi::ShaderHandle             m_linesVertexShader;
    nvrhi::ShaderHandle             m_linesPixelShader;
    nvrhi::GraphicsPipelineHandle   m_linesPipeline;
    nvrhi::BindingLayoutHandle      m_geometryBindingLayout;

    nvrhi::TextureHandle            m_debugVizOutput;
    // The overlay draws from these copies (see CopyForOutput), as the next (async) prefix may rewrite the originals meanwhile
    nvrhi::TextureHandle            m_debugVizOutputCopy;
    nvrhi::BufferHandle             m_bufferGPUCopy;
    nvrhi::BindingSetHandle         m_geometryCopyBindingSet;
    nvrhi::TextureHandle            m_geometryCopyDepth;
    nvrhi::ShaderHandle             m_copyGeometryCS;
    nvrhi::ComputePipelineHandle    m_copyGeometryPSO;
    nvrhi::BindingLayoutHandle      m_copyGeometryBindingLayout;
    nvrhi::BindingSetHandle         m_copyGeometryBindingSet;

    nvrhi::ShaderHandle             m_blendDebugVizPS;
    nvrhi::GraphicsPipelineHandle   m_blendDebugVizPSO;
    nvrhi::BindingLayoutHandle      m_blendBindingLayout;
    nvrhi::BindingSetHandle         m_blendCopyBindingSet;

    bool                            m_closed = false;


    std::shared_ptr<donut::engine::ShaderFactory> m_shaderFactory;
    std::shared_ptr<donut::engine::CommonRenderPasses> m_commonPasses;

    std::array<char, SHADER_DEBUG_HEADER_IN_BYTES>              m_initHeader;
    std::array<char, SHADER_DEBUG_BUFFER_IN_BYTES_NO_TRIANGLES> m_lastBuffer = {};

public:
    ShaderDebug( nvrhi::IDevice* device, nvrhi::ICommandList* commandList, std::shared_ptr<donut::engine::ShaderFactory> shaderFactory, std::shared_ptr<donut::engine::CommonRenderPasses> commonPasses );

    void                    CreateRenderPasses( nvrhi::IFramebuffer * frameBuffer, nvrhi::TextureHandle depthBuffer );

    void                    BeginFrame( nvrhi::ICommandList* commandList, const donut::math::float4x4 & matWorldToClip );
    void                    ClearDebugVizTexture(nvrhi::CommandListHandle commandList); // not merged with BeginFrame since sometimes you want it to persist between frames

    // End of the frame's ShaderDebug use (end of the prefix): read back (any queue), then copy what the overlay reads (any
    // queue); OutputFromCopies draws the overlay later in the frame (graphics queue), depthCopy holding the same frame's depth
    void                    EndFrame( nvrhi::ICommandList* commandList );
    void                    CopyForOutput( nvrhi::ICommandList* commandList );
    void                    OutputFromCopies( nvrhi::ICommandList* commandList, nvrhi::IFramebuffer * frameBuffer, nvrhi::TextureHandle depthCopy, const nvrhi::Viewport & viewport );

    // From EndFrame to the next frame the buffer and viz texture belong to the next prefix, which may already run async
    void                    SetClosed( bool closed )    { m_closed = closed; }
    nvrhi::BufferHandle     GetGPUWriteBuffer()         { assert(!m_closed && "ShaderDebug used after EndFrame"); return m_bufferGPU; };
    nvrhi::TextureHandle    GetDebugVizTexture()        { assert(!m_closed && "ShaderDebug used after EndFrame"); return m_debugVizOutput; };

    //void                    DrawLastBuffer( view? Frame buffer? Depth buffer? See line drawing! )

private:
    void                    OutputLastBufferPrints();
    void                    DrawCurrentBufferGeometry(nvrhi::ICommandList* commandList, nvrhi::IFramebuffer * frameBuffer, const nvrhi::Viewport & viewport, nvrhi::IBindingSet* bindingSet, nvrhi::IBuffer* buffer);
    void                    BlendDebugViz(nvrhi::ICommandList* commandList, nvrhi::IFramebuffer * frameBuffer, const nvrhi::Viewport & viewport, nvrhi::IBindingSet* bindingSet);
};
