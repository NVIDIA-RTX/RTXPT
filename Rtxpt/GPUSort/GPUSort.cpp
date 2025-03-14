/*
* Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "GPUSort.h"

#include <donut/engine/ShaderFactory.h>
#include <donut/engine/FramebufferFactory.h>
#include <donut/engine/CommonRenderPasses.h>
#include <donut/engine/TextureCache.h>

//#include <donut/app/UserInterfaceUtils.h>

#include <nvrhi/utils.h>

#include "../SampleCommon.h"
#include "../ShaderDebug.h"

#define FFX_CPP
#include "FFX_ParallelSort.h"

using namespace donut;
using namespace donut::math;
using namespace donut::engine;


GPUSort::GPUSort(nvrhi::IDevice* device, std::shared_ptr<donut::engine::ShaderFactory> shaderFactory)
    : m_device(device)
    , m_bindingCache(device)
    , m_shaderFactory(shaderFactory)
{
}


GPUSort::~GPUSort()
{
}

void GPUSort::CreateRenderPasses(std::shared_ptr<engine::CommonRenderPasses> commonPasses, std::shared_ptr<ShaderDebug> shaderDebug)
{
    m_commonPasses = commonPasses;
    m_shaderDebug = shaderDebug;

    {
        nvrhi::BindingLayoutDesc layoutDesc;
        layoutDesc.visibility = nvrhi::ShaderType::Compute;
        layoutDesc.bindings = {
            //nvrhi::BindingLayoutItem::VolatileConstantBuffer(0),
            nvrhi::BindingLayoutItem::PushConstants(1, 4*sizeof(uint32_t)),
            nvrhi::BindingLayoutItem::TypedBuffer_SRV(0),       // SrcKeys
            nvrhi::BindingLayoutItem::TypedBuffer_SRV(1),       // SrcIndices
            nvrhi::BindingLayoutItem::TypedBuffer_UAV(0),       // ScratchBuffer
            nvrhi::BindingLayoutItem::TypedBuffer_UAV(1),       // ReducedScratchBuffer
            nvrhi::BindingLayoutItem::TypedBuffer_UAV(2),       // DstIndices
            nvrhi::BindingLayoutItem::StructuredBuffer_UAV(3),  // ConstsUAV
            //nvrhi::BindingLayoutItem::StructuredBuffer_UAV(4),  // DispIndUAV
            nvrhi::BindingLayoutItem::RawBuffer_UAV(SHADER_DEBUG_BUFFER_UAV_INDEX),
        };
        m_commonBindingLayout = m_device->createBindingLayout(layoutDesc);


        layoutDesc.bindings = {
            nvrhi::BindingLayoutItem::StructuredBuffer_UAV(3),  // ConstsUAV
            nvrhi::BindingLayoutItem::StructuredBuffer_UAV(4),  // DispIndUAV
            //nvrhi::BindingLayoutItem::RawBuffer_UAV(SHADER_DEBUG_BUFFER_UAV_INDEX),
        };
        m_initBindingLayout = m_device->createBindingLayout(layoutDesc);
    }

    auto CreateCSPSOPair = [&]( nvrhi::ShaderHandle & shaderHandle, nvrhi::ComputePipelineHandle & psoHandle, const std::string & name, bool initOnly, bool specialInitIndicesFirstPass )
    {
        std::vector<donut::engine::ShaderMacro> shaderMacros;

        if( specialInitIndicesFirstPass )
            shaderMacros.push_back(donut::engine::ShaderMacro({ "RTXPT_GPUSORT_FIRST_PASS_INIT_INDICES", "1" }));

        shaderHandle = m_shaderFactory->CreateShader("app/GPUSort/GPUSort.hlsl", name.c_str(), &shaderMacros, nvrhi::ShaderType::Compute);
        nvrhi::ComputePipelineDesc pipelineDesc;
        pipelineDesc.bindingLayouts = { (initOnly)?(m_initBindingLayout):(m_commonBindingLayout) };
        pipelineDesc.CS = shaderHandle;
        psoHandle = m_device->createComputePipeline(pipelineDesc);
    };

#define CREATE_CS_PSO_PAIR( X, y, z ) CreateCSPSOPair( m_CS##X, m_PSO##X, #X, y, z );

    CREATE_CS_PSO_PAIR(SetupIndirect, true, false);
    CREATE_CS_PSO_PAIR(Count, false, false);
    CREATE_CS_PSO_PAIR(CountIIFP, false, true);
    CREATE_CS_PSO_PAIR(CountReduce, false, false);
    CREATE_CS_PSO_PAIR(ScanPrefix, false, false);
    CREATE_CS_PSO_PAIR(ScanAdd, false, false);
    CREATE_CS_PSO_PAIR(Scatter, false, false);
    CREATE_CS_PSO_PAIR(ScatterIIFP, false, true);
    CREATE_CS_PSO_PAIR(Validate, false, false);

    {
        nvrhi::BufferDesc bufferDesc;
        bufferDesc.initialState = nvrhi::ResourceStates::UnorderedAccess;
        bufferDesc.keepInitialState = true;
        bufferDesc.canHaveUAVs = true;
        
        // TODO: why not unify these two?
        bufferDesc.byteSize = sizeof(FFX_ParallelSortCB) * 1;
        bufferDesc.structStride = sizeof(FFX_ParallelSortCB);
        bufferDesc.debugName = "FFX_ParallelSortCB";
        m_controlBuffer = m_device->createBuffer(bufferDesc);

        bufferDesc.isDrawIndirectArgs = true;
        bufferDesc.byteSize = sizeof(FFX_DispatchIndirectBuffer) * 1;
        bufferDesc.structStride = sizeof(FFX_DispatchIndirectBuffer);
        bufferDesc.debugName = "FFX_DispatchIndirectBuffer";
        m_dispatchIndirectBuffer = m_device->createBuffer(bufferDesc);
    }

}

void GPUSort::ReCreateWorkingBuffers(uint32_t maxItemCount)
{
// TODO: test; this should cause crashes.
//    static uint debugTest = 0; debugTest++; 
//    if (debugTest % 1234 == 0) 
//        m_scratchMaxItemCountSize = 0;

    if( maxItemCount <= m_scratchMaxItemCountSize )
        return;
    
    m_scratchMaxItemCountSize = maxItemCount;

    // Allocate the scratch buffers needed for radix sort
    uint32_t scratchBufferSize;
    uint32_t reducedScratchBufferSize;
    uint32_t scratchIndicesBufferSize = maxItemCount;
    FFX_ParallelSort_CalculateScratchResourceSize(maxItemCount, scratchBufferSize, reducedScratchBufferSize);

    if( m_scratchBuffer != nullptr )
    {
        m_device->waitForIdle();    // make sure buffers are no longer used by the GPU
        m_scratchBuffer = nullptr;
        m_reducedScratchBuffer = nullptr;
        m_reducedScratchBuffer = nullptr;
    }

    nvrhi::BufferDesc bufferDesc;
    bufferDesc.initialState = nvrhi::ResourceStates::UnorderedAccess;
    bufferDesc.keepInitialState = true;
    bufferDesc.canHaveUAVs = true;
    bufferDesc.format = nvrhi::Format::R32_UINT;
    bufferDesc.structStride = 0;
    bufferDesc.canHaveTypedViews = true;

    bufferDesc.byteSize = scratchBufferSize;
    bufferDesc.debugName = "SortScratchBuffer";
    m_scratchBuffer = m_device->createBuffer(bufferDesc);

    bufferDesc.byteSize = reducedScratchBufferSize;
    bufferDesc.debugName = "SortReducedScratchBuffer";
    m_reducedScratchBuffer = m_device->createBuffer(bufferDesc);

    bufferDesc.byteSize = sizeof(uint32_t) * maxItemCount;
    bufferDesc.debugName = "SortScratchIndicesBuffer";
    m_scratchIndicesBuffer = m_device->createBuffer(bufferDesc);
}

uint32_t FloorLog2(uint32_t n)  { assert(n > 0); return n == 1 ? 0 : 1 + FloorLog2(n >> 1); }
uint32_t CeilLog2(uint32_t n)   { assert(n > 0); return n == 1 ? 0 : FloorLog2(n - 1) + 1; };

void GPUSort::Sort(nvrhi::ICommandList * commandList, nvrhi::BufferHandle controlBuffer, uint32_t itemCountByteOffset, nvrhi::BufferHandle bufferKeys, nvrhi::BufferHandle bufferIndices, uint32_t maxItemCount, bool resetIndices)
{
    RAII_SCOPE( commandList->beginMarker("GPUSort");, commandList->endMarker(); );

    ReCreateWorkingBuffers(maxItemCount);

    commandList->copyBuffer( m_controlBuffer, 0, controlBuffer, itemCountByteOffset, sizeof(uint32_t) );

    // // implicit copy barrier should deal with this automatically so not needed
    // commandList->setBufferState(m_controlBuffer, nvrhi::ResourceStates::UnorderedAccess);
    // commandList->commitBarriers();

    // Bindings
    nvrhi::BindingSetDesc bindingSetDescInit, bindingSetDescPing, bindingSetDescPong;
    bindingSetDescInit.bindings = { 
            nvrhi::BindingSetItem::StructuredBuffer_UAV(3, m_controlBuffer),            // ConstsUAV
            nvrhi::BindingSetItem::StructuredBuffer_UAV(4, m_dispatchIndirectBuffer),   // DispIndUAV
            //nvrhi::BindingSetItem::RawBuffer_UAV(SHADER_DEBUG_BUFFER_UAV_INDEX),
    };
    bindingSetDescPing.bindings = {
            nvrhi::BindingSetItem::PushConstants(1, 4*sizeof(uint32_t)), 
            nvrhi::BindingSetItem::TypedBuffer_SRV(0, bufferKeys),                      // SrcKeys
            nvrhi::BindingSetItem::TypedBuffer_SRV(1, bufferIndices),                   // SrcIndices
            nvrhi::BindingSetItem::TypedBuffer_UAV(0, m_scratchBuffer),                 // ScratchBuffer
            nvrhi::BindingSetItem::TypedBuffer_UAV(1, m_reducedScratchBuffer),          // ReducedScratchBuffer
            nvrhi::BindingSetItem::TypedBuffer_UAV(2, m_scratchIndicesBuffer),          // DstIndices
            nvrhi::BindingSetItem::StructuredBuffer_UAV(3, m_controlBuffer),            // ConstsUAV
            //nvrhi::BindingSetItem::StructuredBuffer_UAV(4, m_dispatchIndirectBuffer),   // DispIndUAV
            nvrhi::BindingSetItem::RawBuffer_UAV(SHADER_DEBUG_BUFFER_UAV_INDEX, m_shaderDebug->GetGPUWriteBuffer()),
        };
    bindingSetDescPong.bindings = {
            nvrhi::BindingSetItem::PushConstants(1, 4 * sizeof(uint32_t)),
            nvrhi::BindingSetItem::TypedBuffer_SRV(0, bufferKeys),                      // SrcKeys
            nvrhi::BindingSetItem::TypedBuffer_SRV(1, m_scratchIndicesBuffer),          // SrcIndices
            nvrhi::BindingSetItem::TypedBuffer_UAV(0, m_scratchBuffer),                 // ScratchBuffer
            nvrhi::BindingSetItem::TypedBuffer_UAV(1, m_reducedScratchBuffer),          // ReducedScratchBuffer
            nvrhi::BindingSetItem::TypedBuffer_UAV(2, bufferIndices),                   // DstIndices
            nvrhi::BindingSetItem::StructuredBuffer_UAV(3, m_controlBuffer),            // ConstsUAV
            //nvrhi::BindingSetItem::StructuredBuffer_UAV(4, m_dispatchIndirectBuffer),   // DispIndUAV
            nvrhi::BindingSetItem::RawBuffer_UAV(SHADER_DEBUG_BUFFER_UAV_INDEX, m_shaderDebug->GetGPUWriteBuffer()),
    };

    nvrhi::BindingSetHandle bindingSetInit = m_bindingCache.GetOrCreateBindingSet(bindingSetDescInit, m_initBindingLayout);
    nvrhi::BindingSetHandle bindingSetPing = m_bindingCache.GetOrCreateBindingSet(bindingSetDescPing, m_commonBindingLayout);
    nvrhi::BindingSetHandle bindingSetPong = m_bindingCache.GetOrCreateBindingSet(bindingSetDescPong, m_commonBindingLayout);

    auto RunCSPass = [ & ]( const char * markerName, const nvrhi::ComputePipelineHandle & pso, uint dispatchSizeX, uint indirectDispatchArgOffset, bool pong, uint rootConst, bool initOnly = false )
    {
#if 0 // enable for markers for each pass!
        RAII_SCOPE( commandList->beginMarker(markerName);, commandList->endMarker(); );
#endif

        nvrhi::ComputeState state;
        state.bindings = { (!pong)?(bindingSetPing):(bindingSetPong) };
        if( initOnly )
            state.bindings = {bindingSetInit};
        state.pipeline = pso;

        if (dispatchSizeX == 0)
            state.indirectParams = m_dispatchIndirectBuffer;

        commandList->setComputeState(state);

        uint4 tinyRootConsts = { rootConst, 0, 0, 0 };
        if( !initOnly )
            commandList->setPushConstants(&tinyRootConsts, sizeof(tinyRootConsts));
        if (dispatchSizeX > 0)
            commandList->dispatch(dispatchSizeX, 1, 1);
        else
            commandList->dispatchIndirect(indirectDispatchArgOffset);
    };

    auto ScratchBarriers = [ & ]()
    {
        commandList->setBufferState(m_scratchBuffer, nvrhi::ResourceStates::UnorderedAccess);
        commandList->setBufferState(m_reducedScratchBuffer, nvrhi::ResourceStates::UnorderedAccess);
        commandList->commitBarriers();
    };

    RunCSPass( "SetupIndirect", m_PSOSetupIndirect, 1, 0, false, 0, true );

    // add barrier to make sure no race condition with subsequent passes 
    commandList->setBufferState(m_controlBuffer, nvrhi::ResourceStates::UnorderedAccess);
    commandList->commitBarriers();

    uint32_t maxKeyValue = 0xFFFFFFFF;  // optimization - sorting smaller keys can be faster! not exposed now, but should work just fine!
    uint32_t bitsNeeded = CeilLog2(maxKeyValue);
    bitsNeeded = std::max(1U, bitsNeeded);
    uint32_t bitsRounding = FFX_PARALLELSORT_SORT_BITS_PER_PASS * 2;
    bitsNeeded = ((bitsNeeded + bitsRounding - 1) / bitsRounding) * bitsRounding;

    bool pong = false;
    for (uint32_t Shift = 0; Shift < bitsNeeded; Shift += FFX_PARALLELSORT_SORT_BITS_PER_PASS)
    {
        RunCSPass("Count", (Shift==0 && resetIndices)?m_PSOCountIIFP:m_PSOCount, 0, 0 * 4, pong, Shift);        // SEE RTXPT_GPUSORT_FIRST_PASS_INIT_INDICES
        ScratchBarriers();
        RunCSPass("CountReduce", m_PSOCountReduce, 0, 4 * 4, pong, Shift);
        ScratchBarriers();
        RunCSPass("ScanPrefix", m_PSOScanPrefix, 1, 0, pong, Shift);
        ScratchBarriers();
        RunCSPass("ScanAdd", m_PSOScanAdd, 0, 4 * 4, pong, Shift);
        ScratchBarriers();
        RunCSPass("Scatter", (Shift==0 && resetIndices)?m_PSOScatterIIFP:m_PSOScatter, 0, 0 * 4, pong, Shift);  // SEE RTXPT_GPUSORT_FIRST_PASS_INIT_INDICES
        // ScratchBarriers(); not needed because ping pong forces UAV<->SRV transition which adds a barrier

        pong = !pong;
    }

#if 0
    // Will show up in Debug Output
    RunCSPass("Validate", m_PSOValidate, 1, 0 * 0, false, 0);
#endif
}
