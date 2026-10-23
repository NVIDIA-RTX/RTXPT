/*
* Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "ShaderDebug.h"

#include <donut/engine/ShaderFactory.h>
#include <donut/engine/CommonRenderPasses.h>
#include <donut/engine/View.h>
#include <donut/core/log.h>
#include "../SampleCommon/SampleCommon.h"

#include <algorithm>
#include <chrono>
//#include <iostream>
#include <thread>

using namespace donut;
using namespace donut::math;
using namespace donut::engine;

ShaderDebug::ShaderDebug( nvrhi::IDevice* device, nvrhi::ICommandList* commandList, std::shared_ptr<donut::engine::ShaderFactory> shaderFactory, std::shared_ptr<donut::engine::CommonRenderPasses> commonPasses )
    : m_device(device)
    , m_shaderFactory(shaderFactory)
    , m_commonPasses(commonPasses)
{
    nvrhi::BufferDesc bufferDesc;

    bufferDesc.byteSize = 4*4;
    bufferDesc.isConstantBuffer = false;
    bufferDesc.isVolatile = false;
    bufferDesc.canHaveUAVs = false;
    bufferDesc.canHaveRawViews = false;
    bufferDesc.cpuAccess = nvrhi::CpuAccessMode::None;
    bufferDesc.maxVersions = 0;
    bufferDesc.structStride = 0;
    bufferDesc.keepInitialState = true;
    bufferDesc.initialState = nvrhi::ResourceStates::Common;
    bufferDesc.isDrawIndirectArgs = true;
    bufferDesc.debugName = "ShaderDebugIndirectDrawBufferGPU";
    m_indirectDrawBufferGPU = m_device->createBuffer(bufferDesc);
    bufferDesc.byteSize = 3*4;
    bufferDesc.initialState = nvrhi::ResourceStates::CopyDest;  // rests where CopyForOutput copies into it, so that copy needs no barrier
    bufferDesc.debugName = "ShaderDebugCopyDispatchArgs";
    m_copyDispatchArgs = m_device->createBuffer(bufferDesc);

    bufferDesc.byteSize = SHADER_DEBUG_BUFFER_IN_BYTES;
    bufferDesc.isConstantBuffer = false;
    bufferDesc.isVolatile = false;
    bufferDesc.canHaveUAVs = true;
    bufferDesc.canHaveRawViews = true;
    bufferDesc.cpuAccess = nvrhi::CpuAccessMode::None;
    bufferDesc.maxVersions = 0;
    bufferDesc.structStride = 0;
    bufferDesc.keepInitialState = true;
    bufferDesc.initialState = nvrhi::ResourceStates::Common;
    bufferDesc.isDrawIndirectArgs = false;
    bufferDesc.debugName = "ShaderDebugBufferGPU";
    m_bufferGPU = m_device->createBuffer(bufferDesc);


    bufferDesc.canHaveUAVs = false;
    bufferDesc.canHaveRawViews = false;
    bufferDesc.cpuAccess = nvrhi::CpuAccessMode::Read;
    bufferDesc.structStride = 0;
    bufferDesc.keepInitialState = false;
    bufferDesc.initialState = nvrhi::ResourceStates::Unknown;
    bufferDesc.isDrawIndirectArgs = false;
    bufferDesc.byteSize = SHADER_DEBUG_BUFFER_IN_BYTES_NO_TRIANGLES;
    for (int i = 0; i < (int)std::size(m_bufferCPU); i++)
    {
        bufferDesc.debugName = "ShaderDebugBufferCPU_" + std::to_string(i);
        m_bufferCPU[i] = m_device->createBuffer(bufferDesc);
    }

    memset(m_initHeader.data(), 0, m_initHeader.size());
    reinterpret_cast<ShaderDebugHeader*>(m_initHeader.data())->VertexCountPerInstance = 3;  // needed for indirect draw
    commandList->writeBuffer(m_bufferGPU, m_initHeader.data(), m_initHeader.size() );
    const uint32_t copyDispatchArgs[3] = { 0, 1, 1 };   // CopyForOutput only ever rewrites the group count
    commandList->writeBuffer(m_copyDispatchArgs, copyDispatchArgs, sizeof(copyDispatchArgs));

    for (int i = 0; i < (int)std::size(m_bufferCPU); i++)
        commandList->copyBuffer(m_bufferCPU[i], 0, m_bufferGPU, 0, SHADER_DEBUG_BUFFER_IN_BYTES_NO_TRIANGLES );
}

void ShaderDebug::CreateRenderPasses( nvrhi::IFramebuffer * frameBuffer, nvrhi::TextureHandle depthBuffer )
{
    nvrhi::TextureDesc desc;
    //desc.width = frameBuffer->getDesc().colorAttachments[0].texture->getDesc().width;
    //desc.height = frameBuffer->getDesc().colorAttachments[0].texture->getDesc().height;
    desc.width  = depthBuffer->getDesc().width;
    desc.height = depthBuffer->getDesc().height;
    desc.debugName = "DebugVizOutput";
    desc.format = nvrhi::Format::RGBA16_FLOAT;
    desc.clearValue = nvrhi::Color(0.0f, 0.0f, 0.0f, 0.0f);   // avoid the debug layer warnings... not actually cleared except for debug purposes
    desc.isUAV = true;
    desc.keepInitialState = true;
    desc.initialState = nvrhi::ResourceStates::UnorderedAccess;
    // Only recreate on a size change: this also runs on AS-only rebuilds, where binding caches that bound the texture
    // are not cleared and would keep a replaced one alive.
    if (m_debugVizOutput == nullptr || m_debugVizOutput->getDesc().width != desc.width || m_debugVizOutput->getDesc().height != desc.height)
        m_debugVizOutput = m_device->createTexture(desc);


    {
        nvrhi::BindingLayoutDesc bindingLayoutDesc;
        bindingLayoutDesc.visibility = nvrhi::ShaderType::All;
        bindingLayoutDesc.bindings = {
            nvrhi::BindingLayoutItem::Texture_SRV(0),
            nvrhi::BindingLayoutItem::Texture_SRV(1),
            nvrhi::BindingLayoutItem::RawBuffer_UAV(SHADER_DEBUG_BUFFER_UAV_INDEX)
        };

        m_geometryBindingLayout = m_device->createBindingLayout(bindingLayoutDesc);
    }

    {   // The viz blend only reads the viz texture (its copy, see CopyForOutput)
        nvrhi::BindingLayoutDesc bindingLayoutDesc;
        bindingLayoutDesc.visibility = nvrhi::ShaderType::Pixel;
        bindingLayoutDesc.bindings = { nvrhi::BindingLayoutItem::Texture_SRV(1) };
        m_blendBindingLayout = m_device->createBindingLayout(bindingLayoutDesc);

        m_blendCopyBindingSet = nullptr;    // the copy sets are recreated against the new layouts by CopyForOutput/OutputFromCopies
        m_geometryCopyBindingSet = nullptr;
    }

    {
        std::vector<donut::engine::ShaderMacro> shaderMacros;
        shaderMacros.push_back(donut::engine::ShaderMacro({              "BLEND_DEBUG_BUFFER", "1" })); 

        m_blendDebugVizPS = m_shaderFactory->CreateShader("app/Shaders/Libraries/ShaderDebug/ShaderDebug.hlsl", "main", &shaderMacros, nvrhi::ShaderType::Pixel);

        nvrhi::GraphicsPipelineDesc pipelineDesc;
        pipelineDesc.bindingLayouts = { m_blendBindingLayout };
        pipelineDesc.primType = nvrhi::PrimitiveType::TriangleStrip;
        pipelineDesc.VS = m_commonPasses->m_FullscreenVS;
        pipelineDesc.PS = m_blendDebugVizPS;
        pipelineDesc.renderState.rasterState.setCullNone();
        pipelineDesc.renderState.depthStencilState.depthTestEnable = false;
        pipelineDesc.renderState.depthStencilState.stencilEnable = false;
        pipelineDesc.renderState.blendState.targets[0].enableBlend()
            .setSrcBlend(nvrhi::BlendFactor::SrcAlpha)
            .setDestBlend(nvrhi::BlendFactor::InvSrcAlpha)
            .setSrcBlendAlpha(nvrhi::BlendFactor::Zero)
            .setDestBlendAlpha(nvrhi::BlendFactor::One);
        m_blendDebugVizPSO = m_device->createGraphicsPipeline(pipelineDesc, frameBuffer);
    }

    // debug triangles and lines...
    {
        std::vector<ShaderMacro> drawTrianglesMacro = { ShaderMacro("DRAW_TRIANGLES_SHADERS", "1") };
        m_trianglesVertexShader = m_shaderFactory->CreateShader("app/Shaders/Libraries/ShaderDebug/ShaderDebug.hlsl", "main_vs", &drawTrianglesMacro, nvrhi::ShaderType::Vertex);
        m_trianglesPixelShader = m_shaderFactory->CreateShader("app/Shaders/Libraries/ShaderDebug/ShaderDebug.hlsl", "main_ps", &drawTrianglesMacro, nvrhi::ShaderType::Pixel);
    }

    {
        std::vector<ShaderMacro> drawLinesMacro = { ShaderMacro("DRAW_LINES_SHADERS", "1") };
        m_linesVertexShader = m_shaderFactory->CreateShader("app/Shaders/Libraries/ShaderDebug/ShaderDebug.hlsl", "main_vs", &drawLinesMacro, nvrhi::ShaderType::Vertex);
        m_linesPixelShader = m_shaderFactory->CreateShader("app/Shaders/Libraries/ShaderDebug/ShaderDebug.hlsl", "main_ps", &drawLinesMacro, nvrhi::ShaderType::Pixel);
    }

    {   // Copies the geometry the overlay draws (see CopyForOutput)
        nvrhi::BindingLayoutDesc bindingLayoutDesc;
        bindingLayoutDesc.visibility = nvrhi::ShaderType::Compute;
        bindingLayoutDesc.bindings = {
            nvrhi::BindingLayoutItem::RawBuffer_UAV(SHADER_DEBUG_BUFFER_UAV_INDEX),
            nvrhi::BindingLayoutItem::RawBuffer_UAV(0)
        };
        m_copyGeometryBindingLayout = m_device->createBindingLayout(bindingLayoutDesc);
        m_copyGeometryBindingSet = nullptr;

        std::vector<ShaderMacro> copyGeometryMacro = { ShaderMacro("COPY_GEOMETRY_FOR_OUTPUT", "1") };
        m_copyGeometryCS = m_shaderFactory->CreateShader("app/Shaders/Libraries/ShaderDebug/ShaderDebug.hlsl", "main_cs", &copyGeometryMacro, nvrhi::ShaderType::Compute);
        m_copyGeometryPSO = m_device->createComputePipeline(nvrhi::ComputePipelineDesc().setComputeShader(m_copyGeometryCS).addBindingLayout(m_copyGeometryBindingLayout));
    }

    nvrhi::GraphicsPipelineDesc psoDesc;
    psoDesc.VS = m_trianglesVertexShader;
    psoDesc.PS = m_trianglesPixelShader;
    psoDesc.bindingLayouts = { m_geometryBindingLayout };
    psoDesc.primType = nvrhi::PrimitiveType::TriangleList;
    psoDesc.renderState.depthStencilState.depthTestEnable = false;
    psoDesc.renderState.rasterState.cullMode = nvrhi::RasterCullMode::None;
    psoDesc.renderState.blendState.targets[0].enableBlend().setSrcBlend(nvrhi::BlendFactor::SrcAlpha)
        .setDestBlend(nvrhi::BlendFactor::InvSrcAlpha).setSrcBlendAlpha(nvrhi::BlendFactor::Zero).setDestBlendAlpha(nvrhi::BlendFactor::One);
    m_trianglesPipeline = m_device->createGraphicsPipeline(psoDesc, frameBuffer);

    psoDesc.VS = m_linesVertexShader;
    psoDesc.PS = m_linesPixelShader;
    psoDesc.bindingLayouts = { m_geometryBindingLayout };
    psoDesc.primType = nvrhi::PrimitiveType::LineList;
    m_linesPipeline = m_device->createGraphicsPipeline(psoDesc, frameBuffer);
}

void ShaderDebug::BeginFrame( nvrhi::ICommandList* commandList, const float4x4& matWorldToClip )
{
    ZoneScoped;

    ShaderDebugHeader* header = reinterpret_cast<ShaderDebugHeader*>(m_initHeader.data());
    memcpy( header->WorldViewProjectionMatrix, matWorldToClip.m_data, sizeof(float)*16 );

    commandList->writeBuffer(m_bufferGPU, m_initHeader.data(), m_initHeader.size());   // only need to clear the counters
}
    
void ShaderDebug::ClearDebugVizTexture(nvrhi::CommandListHandle commandList)
{
    commandList->clearTextureFloat(m_debugVizOutput, nvrhi::AllSubresources, nvrhi::Color(0, 0, 0, 0));
}

void ShaderDebug::EndFrame( nvrhi::ICommandList* commandList )
{
    // map and copy CPU side buffer so we can process it (later)
    void* pData;
    {
        // Own zone because nvrhi's Device::mapBuffer starts with 'if (b->lastUseFence) WaitForFence(...)' - it is a GPU
        // sync point on the main thread, not just a Map. The readback buffers rotate over c_SwapchainCount+1 entries so
        // the fence should be long signalled, but a sampling profiler cannot tell a wait from work, so measure it.
        ZoneScopedN("ShaderDebugMapReadback");
        pData = m_device->mapBuffer(m_bufferCPU[m_currentBufferIndex], nvrhi::CpuAccessMode::Read);
    }

    static uint32_t s_breakpointReportCount = 0;
    bool logBreakpoint = false;
    {
        ZoneScopedN("ShaderDebugCopyReadback");
        // Copy only the header plus however many print bytes the shaders actually wrote, rather than the whole
        // SHADER_DEBUG_BUFFER_IN_BYTES_NO_TRIANGLES (~256KB) every frame. Nothing reads past that point: the only
        // consumers are OutputLastBufferPrints, which bails immediately when this same counter is 0 (the common case,
        // so the copy is normally 96 bytes), and DrawCurrentBufferGeometry, which needs the header alone - its triangle
        // data is GPU-resident and drawn indirectly.
        //
        // The clamp must match OutputLastBufferPrints' exactly: the counter is a SIGNED int that the shader side
        // increments without bound, so it can arrive negative or past the end of the print region, and either would
        // read outside the mapped range here.
        static_assert( SHADER_DEBUG_HEADER_IN_BYTES + SHADER_DEBUG_PRINT_BUFFER_IN_BYTES == SHADER_DEBUG_BUFFER_IN_BYTES_NO_TRIANGLES );
        const ShaderDebugHeader * header = reinterpret_cast<const ShaderDebugHeader*>(pData);
        const size_t printBytes = (size_t)std::clamp( header->PrintItemsCounterInBytes, 0, (int)SHADER_DEBUG_PRINT_BUFFER_IN_BYTES );

        // A shader hit BREAKPOINT() (Shaders/Libraries/ShaderDebug/ShaderDebug.hlsl)
        // in Shaders/Bindless.hlsli when a descriptor index does not refer to the resource type the shader is about
        // to interpret it as. Without this the trap fires and reports nothing: BREAKPOINT defaults to a plain store
        // rather than a hang, so the failure is otherwise completely silent.
        //
        // BeginFrame rewrites the header every frame, so this is per-frame rather than latched. Rate-limited
        // because a bad index usually fires for a large fraction of the threads, every frame.
        if ( (uint32_t)header->BreakpointSentinel == SHADER_DEBUG_BREAKPOINT_SENTINEL )
        {
            if ( s_breakpointReportCount < 8 )
            {
                s_breakpointReportCount++;
                logBreakpoint = true;
            }
        }

        // NOTE: m_lastBuffer is persistent, so everything past this point still holds the PREVIOUS frame's bytes - that
        // is fine because nothing reads there, but do not be fooled by it when inspecting m_lastBuffer in a debugger.
        assert( m_lastBuffer.size() == SHADER_DEBUG_BUFFER_IN_BYTES_NO_TRIANGLES );
        memcpy(m_lastBuffer.data(), pData, SHADER_DEBUG_HEADER_IN_BYTES + printBytes);
    }

    m_device->unmapBuffer(m_bufferCPU[m_currentBufferIndex]);

    // copy latest GPU side buffer 
    commandList->copyBuffer(m_bufferCPU[m_currentBufferIndex], 0, m_bufferGPU, 0, SHADER_DEBUG_BUFFER_IN_BYTES_NO_TRIANGLES );
    m_currentBufferIndex = (m_currentBufferIndex+1)%(int)std::size(m_bufferCPU);

    OutputLastBufferPrints();

    if (logBreakpoint)  // do it here so any DebugPrint-s from before are shown in the Output before the breakpoint popup appears
    {
        // The sentinel carries no source location, so this cannot say WHICH BREAKPOINT() fired - grep for
        // the macro to see the candidates. The bindless validation is the most common one by far.
        donut::log::error("Shader BREAKPOINT() was hit%s. Set RTXPT_BREAKPOINT_HANG to 1 in "
            "Shaders/Libraries/ShaderDebug/ShaderDebug.hlsl to trap it on the GPU instead.",
            (s_breakpointReportCount == 8) ? " (further reports suppressed)" : "");
    }
}

void ShaderDebug::CopyForOutput( nvrhi::ICommandList* commandList )
{
    // The copies rest in CopyDest, so copying into them needs no barrier on the compute queue
    const nvrhi::TextureDesc& vizDesc = m_debugVizOutput->getDesc();
    if (m_debugVizOutputCopy == nullptr || m_debugVizOutputCopy->getDesc().width != vizDesc.width || m_debugVizOutputCopy->getDesc().height != vizDesc.height)
    {
        nvrhi::TextureDesc desc = vizDesc;
        desc.debugName = "DebugVizOutputCopy";
        desc.isUAV = false;
        desc.initialState = nvrhi::ResourceStates::CopyDest;
        m_debugVizOutputCopy = m_device->createTexture(desc);
        m_blendCopyBindingSet = nullptr;
        m_geometryCopyBindingSet = nullptr;
    }
    if (m_blendCopyBindingSet == nullptr)
    {
        nvrhi::BindingSetDesc bindingSetDesc;
        bindingSetDesc.bindings = { nvrhi::BindingSetItem::Texture_SRV(1, m_debugVizOutputCopy) };
        m_blendCopyBindingSet = m_device->createBindingSet(bindingSetDesc, m_blendBindingLayout);
    }
    commandList->copyTexture(m_debugVizOutputCopy, nvrhi::TextureSlice(), m_debugVizOutput, nvrhi::TextureSlice());

    if (m_bufferGPUCopy == nullptr)
    {
        nvrhi::BufferDesc desc = m_bufferGPU->getDesc();
        desc.debugName = "ShaderDebugBufferGPUCopy";
        desc.initialState = nvrhi::ResourceStates::CopyDest;
        m_bufferGPUCopy = m_device->createBuffer(desc);
        m_geometryCopyBindingSet = nullptr;
    }
    if (m_copyGeometryBindingSet == nullptr)
    {
        nvrhi::BindingSetDesc bindingSetDesc;
        bindingSetDesc.bindings = {
            nvrhi::BindingSetItem::RawBuffer_UAV(SHADER_DEBUG_BUFFER_UAV_INDEX, m_bufferGPU),
            nvrhi::BindingSetItem::RawBuffer_UAV(0, m_bufferGPUCopy),
        };
        m_copyGeometryBindingSet = m_device->createBindingSet(bindingSetDesc, m_copyGeometryBindingLayout);
    }

    // Every frame, without the CPU knowing whether there is any geometry: the header (draw arguments, matrix), then only
    // the triangles and lines this frame added, as counted by the shaders that added them (no groups when there are none)
    commandList->copyBuffer(m_bufferGPUCopy, 0, m_bufferGPU, 0, SHADER_DEBUG_HEADER_IN_BYTES);
    commandList->copyBuffer(m_copyDispatchArgs, 0, m_bufferGPU, SHADER_DEBUG_COPY_GROUP_COUNT_OFFSET, sizeof(uint32_t));
    nvrhi::ComputeState state;
    state.pipeline = m_copyGeometryPSO;
    state.bindings = { m_copyGeometryBindingSet };
    state.indirectParams = m_copyDispatchArgs;
    commandList->setComputeState(state);
    commandList->dispatchIndirect(0);
}

void ShaderDebug::OutputFromCopies( nvrhi::ICommandList* commandList, nvrhi::IFramebuffer * frameBuffer, nvrhi::TextureHandle depthCopy, const nvrhi::Viewport & viewport )
{
    RTXPT_GPU_SCOPE(commandList, "ShaderDebug");

    if (m_bufferGPUCopy != nullptr)
    {
        if (m_geometryCopyBindingSet == nullptr || m_geometryCopyDepth != depthCopy)
        {
            nvrhi::BindingSetDesc bindingSetDesc;
            bindingSetDesc.bindings = {
                nvrhi::BindingSetItem::Texture_SRV(0, depthCopy),
                nvrhi::BindingSetItem::Texture_SRV(1, m_debugVizOutputCopy),
                nvrhi::BindingSetItem::RawBuffer_UAV(SHADER_DEBUG_BUFFER_UAV_INDEX, m_bufferGPUCopy),
            };
            m_geometryCopyBindingSet = m_device->createBindingSet(bindingSetDesc, m_geometryBindingLayout);
            m_geometryCopyDepth = depthCopy;
        }
        DrawCurrentBufferGeometry(commandList, frameBuffer, viewport, m_geometryCopyBindingSet, m_bufferGPUCopy);
    }

    if (m_blendCopyBindingSet != nullptr)
        BlendDebugViz(commandList, frameBuffer, viewport, m_blendCopyBindingSet);
}

void ShaderDebug::BlendDebugViz( nvrhi::ICommandList* commandList, nvrhi::IFramebuffer * frameBuffer, const nvrhi::Viewport & viewport, nvrhi::IBindingSet* bindingSet )
{
    // Draw debug viz overlay with blending
    nvrhi::GraphicsState state;
    state.pipeline = m_blendDebugVizPSO;
    state.framebuffer = frameBuffer;
    state.bindings = { bindingSet };
    state.viewport.addViewportAndScissorRect(viewport);
    commandList->setGraphicsState(state);

    nvrhi::DrawArguments args;
    args.instanceCount = 1;
    args.vertexCount = 4;
    commandList->draw(args);
}

static std::string ProcessSingleValue(const char*& currentData, const char* endData, ShaderDebugArgCode valueType )
{
    size_t neededSize = 1 * sizeof(uint);
    if (currentData + neededSize > endData)
    {
        assert(false && "Buffer overflow - there's a bug somewhere");
        return "";
    }
    std::string ret;
    switch (valueType)
    {
    case(ShaderDebugArgCode::DebugPrint_Uint):    ret = std::to_string( *reinterpret_cast<const unsigned int *>(currentData) ); break;
    case(ShaderDebugArgCode::DebugPrint_Int):     ret = std::to_string( *reinterpret_cast<const int *>(currentData) ); break;
    case(ShaderDebugArgCode::DebugPrint_Float):   ret = std::to_string( *reinterpret_cast<const float *>(currentData) ); break;
    default: assert(false && "Wrong value type - there's a bug somewhere"); return "";
    }

    currentData += sizeof(float);
    return ret;
}

static std::string ProcessArg( const char * & currentData, const char * endData )
{
    char firstByteCode = *reinterpret_cast<const char*>(currentData);
    currentData += sizeof(char);

    ShaderDebugArgCode argCode = static_cast<ShaderDebugArgCode>(firstByteCode);
    if( argCode == ShaderDebugArgCode::DebugPrint_ErrorType ) { assert(false); return ""; }

    uint elementCount = 0;
    ShaderDebugArgCode typeCode = ShaderDebugArgCode::DebugPrint_ErrorType;
    switch(argCode)
    {
        case(ShaderDebugArgCode::DebugPrint_Uint):    elementCount = 1; typeCode = ShaderDebugArgCode::DebugPrint_Uint; break;
        case(ShaderDebugArgCode::DebugPrint_Uint2):   elementCount = 2; typeCode = ShaderDebugArgCode::DebugPrint_Uint; break;
        case(ShaderDebugArgCode::DebugPrint_Uint3):   elementCount = 3; typeCode = ShaderDebugArgCode::DebugPrint_Uint; break;
        case(ShaderDebugArgCode::DebugPrint_Uint4):   elementCount = 4; typeCode = ShaderDebugArgCode::DebugPrint_Uint; break;
        case(ShaderDebugArgCode::DebugPrint_Int):     elementCount = 1; typeCode = ShaderDebugArgCode::DebugPrint_Int; break;
        case(ShaderDebugArgCode::DebugPrint_Int2):    elementCount = 2; typeCode = ShaderDebugArgCode::DebugPrint_Int; break;
        case(ShaderDebugArgCode::DebugPrint_Int3):    elementCount = 3; typeCode = ShaderDebugArgCode::DebugPrint_Int; break;
        case(ShaderDebugArgCode::DebugPrint_Int4):    elementCount = 4; typeCode = ShaderDebugArgCode::DebugPrint_Int; break;
        case(ShaderDebugArgCode::DebugPrint_Float):   elementCount = 1; typeCode = ShaderDebugArgCode::DebugPrint_Float; break;
        case(ShaderDebugArgCode::DebugPrint_Float2):  elementCount = 2; typeCode = ShaderDebugArgCode::DebugPrint_Float; break;
        case(ShaderDebugArgCode::DebugPrint_Float3):  elementCount = 3; typeCode = ShaderDebugArgCode::DebugPrint_Float; break;
        case(ShaderDebugArgCode::DebugPrint_Float4):  elementCount = 4; typeCode = ShaderDebugArgCode::DebugPrint_Float; break;
        default: assert(false && "Wrong data type - there's a bug somewhere"); return "";
    }
    size_t neededSize = elementCount * sizeof(uint);
    if( currentData + neededSize > endData )
    {
        assert(false && "Buffer overflow - there's a bug somewhere");
        return "";
    }

    if( elementCount == 1 ) 
        return ProcessSingleValue( currentData, endData, typeCode );
    std::string ret = "(";
    for( uint i = 0; i < elementCount; i++ )
    {
        if( i != 0 ) ret += ", ";
        ret += ProcessSingleValue( currentData, endData, typeCode );
    }
    return ret + ")";
}

void ShaderDebug::OutputLastBufferPrints()
{
    const char * rawData = m_lastBuffer.data();
    uint shaderPrintByteCount = *reinterpret_cast<const uint*>(rawData);
    uint shaderTriangleByteCount = *reinterpret_cast<const uint*>(rawData+4);

    if( shaderPrintByteCount == 0 ) 
        return;
    bool hadOverflow = false;
    if( shaderPrintByteCount > SHADER_DEBUG_PRINT_BUFFER_IN_BYTES )
    {
        hadOverflow = true;
        shaderPrintByteCount = SHADER_DEBUG_PRINT_BUFFER_IN_BYTES;
    }

    const char * currentData  = rawData + SHADER_DEBUG_HEADER_IN_BYTES;
    const char * endData      = currentData + shaderPrintByteCount;

    bool hadText = false;

    std::vector<std::pair<int, std::string>> unformattedArgs;

    while( currentData+sizeof(ShaderDebugPrintItem) < endData )
    {
        const ShaderDebugPrintItem & header = *reinterpret_cast<const ShaderDebugPrintItem *>(currentData);
        currentData += sizeof(ShaderDebugPrintItem);
        if( (header.NumArgs > SHADER_PRINTF_MAX_DEBUG_PRINT_ARGS) || (header.StringSize > (endData - currentData)) )
        {
            assert( false && "Buffer overflow or out of bounds - there's a bug somewhere" ); break; // out of bounds?
        }

        const char * itemStartPtr = currentData;

        std::string text = (header.StringSize==0)?(""):(std::string(currentData, header.StringSize));
        currentData += header.StringSize; // +1 is for the null which is added for safety

        unformattedArgs.clear();
        for( int i = 0; i < header.NumArgs; i++ )
        {
            std::string arg = ProcessArg(currentData, endData);
            if( arg == "" )
            {
                assert( false && "Buffer overflow or out of bounds - there's a bug somewhere" ); break; // out of bounds?
            }
            std::string placeholder = "{"+std::to_string(i)+"}";    // all this string processing is quite awful performance-wise but hey whatchyagonnado
            size_t index = text.find(placeholder, 0);
            if (index == std::string::npos) 
            {
                unformattedArgs.push_back(make_pair(i, arg));
                continue;
            }
            text.replace(index, placeholder.size(), arg);
        }

        if( unformattedArgs.size() > 0 )
        {
            text += " [unformatted args] ";
            for( auto it : unformattedArgs )
                text += std::to_string(it.first) + ": " + it.second + " ";
        }

        log::message(log::Severity::Info, "Shader: %s", text.c_str());

        assert(currentData <= itemStartPtr + header.NumBytes);
        currentData = itemStartPtr + header.NumBytes;

        hadText = true;
    }

    if( hadText )
    {
        // if no sleep, VS debug output freezes out
        using namespace std::chrono_literals;
        std::this_thread::sleep_for(20ms);
    }

    if( hadOverflow )
    {
        log::message(log::Severity::Info, "ShaderDebug: ============================================================================================");
        log::message(log::Severity::Info, "ShaderDebug: ==== INSUFFICIENT SPACE IN SHADER_DEBUG_PRINT_BUFFER_IN_BYTES TO STORE ALL DebugPrint-s ====");
        log::message(log::Severity::Info, "ShaderDebug: ============================================================================================");
    }
}

void ShaderDebug::DrawCurrentBufferGeometry(nvrhi::ICommandList* commandList, nvrhi::IFramebuffer * frameBuffer, const nvrhi::Viewport & viewport, nvrhi::IBindingSet* bindingSet, nvrhi::IBuffer* buffer)
{
    RTXPT_GPU_SCOPE(commandList, "Tris");

    ShaderDebugHeader & header = *reinterpret_cast<ShaderDebugHeader*>(m_lastBuffer.data());

    // copy args from master buffer to indirect
    commandList->copyBuffer(m_indirectDrawBufferGPU, 0, buffer, 4*4, 4*4 );

    nvrhi::GraphicsState state;
    state.bindings = { bindingSet };
    //state.vertexBuffers = { {0, 0, 0} };
    state.pipeline = m_trianglesPipeline;
    state.framebuffer = frameBuffer;
    state.viewport.addViewportAndScissorRect(viewport);
    state.indirectParams = m_indirectDrawBufferGPU;
    commandList->setGraphicsState(state);
    commandList->drawIndirect( 0 );

    state.pipeline = m_linesPipeline;
    commandList->setGraphicsState(state);
    commandList->drawIndirect(0);
}
