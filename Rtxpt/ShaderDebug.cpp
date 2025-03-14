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
#include "SampleCommon.h"

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
    for (int i = 0; i < c_swapchainCount; i++)
    {
        bufferDesc.debugName = "ShaderDebugBufferCPU_" + std::to_string(i);
        m_bufferCPU[i] = m_device->createBuffer(bufferDesc);
    }

    memset(m_initHeader.data(), 0, m_initHeader.size());
    reinterpret_cast<ShaderDebugHeader*>(m_initHeader.data())->VertexCountPerInstance = 3;  // needed for indirect draw
    commandList->writeBuffer(m_bufferGPU, m_initHeader.data(), m_initHeader.size() );

    for (int i = 0; i < c_swapchainCount; i++)
        commandList->copyBuffer(m_bufferCPU[i], 0, m_bufferGPU, 0, SHADER_DEBUG_BUFFER_IN_BYTES_NO_TRIANGLES );
}

void ShaderDebug::CreateRenderPasses( nvrhi::IFramebuffer * frameBuffer, nvrhi::TextureHandle depthBuffer )
{
    nvrhi::TextureDesc desc;
    desc.width = frameBuffer->getDesc().colorAttachments[0].texture->getDesc().width;
    desc.height = frameBuffer->getDesc().colorAttachments[0].texture->getDesc().height;
    desc.debugName = "DebugVizOutput";
    desc.format = nvrhi::Format::RGBA16_FLOAT;
    desc.clearValue = nvrhi::Color(0.0f, 0.0f, 0.0f, 0.0f);   // avoid the debug layer warnings... not actually cleared except for debug purposes
    desc.isUAV = true;
    desc.keepInitialState = true;
    desc.initialState = nvrhi::ResourceStates::UnorderedAccess;
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

    {
        std::vector<donut::engine::ShaderMacro> shaderMacros;
        shaderMacros.push_back(donut::engine::ShaderMacro({              "BLEND_DEBUG_BUFFER", "1" })); 

        m_blendDebugVizPS = m_shaderFactory->CreateShader("app/ShaderDebug.hlsl", "main", &shaderMacros, nvrhi::ShaderType::Pixel);

        nvrhi::GraphicsPipelineDesc pipelineDesc;
        pipelineDesc.bindingLayouts = { m_geometryBindingLayout };
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
        m_trianglesVertexShader = m_shaderFactory->CreateShader("app/ShaderDebug.hlsl", "main_vs", &drawTrianglesMacro, nvrhi::ShaderType::Vertex);
        m_trianglesPixelShader = m_shaderFactory->CreateShader("app/ShaderDebug.hlsl", "main_ps", &drawTrianglesMacro, nvrhi::ShaderType::Pixel);
    }

    {
        std::vector<ShaderMacro> drawLinesMacro = { ShaderMacro("DRAW_LINES_SHADERS", "1") };
        m_linesVertexShader = m_shaderFactory->CreateShader("app/ShaderDebug.hlsl", "main_vs", &drawLinesMacro, nvrhi::ShaderType::Vertex);
        m_linesPixelShader = m_shaderFactory->CreateShader("app/ShaderDebug.hlsl", "main_ps", &drawLinesMacro, nvrhi::ShaderType::Pixel);
    }

    nvrhi::BindingSetDesc bindingSetDesc;
    bindingSetDesc.bindings = {
        nvrhi::BindingSetItem::Texture_SRV(0, depthBuffer),
        nvrhi::BindingSetItem::Texture_SRV(1, m_debugVizOutput),
        nvrhi::BindingSetItem::RawBuffer_UAV(SHADER_DEBUG_BUFFER_UAV_INDEX, GetGPUWriteBuffer()),
    };
    m_geometryBindingSet = m_device->createBindingSet(bindingSetDesc, m_geometryBindingLayout);

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
    ShaderDebugHeader* header = reinterpret_cast<ShaderDebugHeader*>(m_initHeader.data());
    memcpy( header->WorldViewProjectionMatrix, matWorldToClip.m_data, sizeof(float)*16 );

    commandList->writeBuffer(m_bufferGPU, m_initHeader.data(), m_initHeader.size());   // only need to clear the counters
}
    
void ShaderDebug::ClearDebugVizTexture(nvrhi::CommandListHandle commandList)
{
    commandList->clearTextureFloat(m_debugVizOutput, nvrhi::AllSubresources, nvrhi::Color(0, 0, 0, 0));
}

void ShaderDebug::EndFrameAndOutput( nvrhi::ICommandList* commandList, nvrhi::IFramebuffer * frameBuffer, nvrhi::TextureHandle depthBuffer, const nvrhi::Viewport & viewport )
{
    RAII_SCOPE( commandList->beginMarker("ShaderDebug");, commandList->endMarker(); );

    // map and copy CPU side buffer so we can process it (later)
    void* pData = m_device->mapBuffer(m_bufferCPU[m_currentBufferIndex], nvrhi::CpuAccessMode::Read);
    memcpy(&m_lastBuffer, pData, SHADER_DEBUG_BUFFER_IN_BYTES_NO_TRIANGLES); assert( m_lastBuffer.size() == SHADER_DEBUG_BUFFER_IN_BYTES_NO_TRIANGLES );
    m_device->unmapBuffer(m_bufferCPU[m_currentBufferIndex]);

    // copy latest GPU side buffer 
    commandList->copyBuffer(m_bufferCPU[m_currentBufferIndex], 0, m_bufferGPU, 0, SHADER_DEBUG_BUFFER_IN_BYTES_NO_TRIANGLES );
    m_currentBufferIndex = (m_currentBufferIndex+1)%c_swapchainCount;

    OutputLastBufferPrints();
    
    DrawCurrentBufferGeometry(commandList, frameBuffer, depthBuffer, viewport );

    // Draw debug viz overlay with blending
    {
        nvrhi::BindingSetHandle bindingSet = m_geometryBindingSet; //m_bindingCache.GetOrCreateBindingSet(bindingSetDesc, m_BindingLayoutPS);

        nvrhi::GraphicsState state;
        state.pipeline = m_blendDebugVizPSO;
        state.framebuffer = frameBuffer;
        state.bindings = { bindingSet };
        nvrhi::ViewportState viewportState;
        auto desc = frameBuffer->getDesc().colorAttachments[0].texture->getDesc();
        viewportState.addViewport(nvrhi::Viewport((float)desc.width, (float)desc.height));
        viewportState.addScissorRect(nvrhi::Rect(desc.width, desc.height));
        state.viewport = viewportState;
        commandList->setGraphicsState(state);

        nvrhi::DrawArguments args;
        args.instanceCount = 1;
        args.vertexCount = 4;
        commandList->draw(args);
    }
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

void ShaderDebug::DrawCurrentBufferGeometry(nvrhi::ICommandList* commandList, nvrhi::IFramebuffer * frameBuffer, nvrhi::TextureHandle depthBuffer, const nvrhi::Viewport & viewport)
{
    RAII_SCOPE(commandList->beginMarker("Tris"); , commandList->endMarker(); );

    ShaderDebugHeader & header = *reinterpret_cast<ShaderDebugHeader*>(m_lastBuffer.data());

    // copy args from master buffer to indirect
    commandList->copyBuffer(m_indirectDrawBufferGPU, 0, m_bufferGPU, 4*4, 4*4 );    
    
    nvrhi::GraphicsState state;
    state.bindings = { m_geometryBindingSet };
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
