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

#include <donut/engine/BindingCache.h>
#include <nvrhi/nvrhi.h>
#include <donut/core/math/math.h>
#include <memory>

#include <donut/core/math/math.h>

#include <filesystem>

using namespace donut::math;

namespace donut::engine
{
    class ExtendedScene;
    class FramebufferFactory;
    class TextureCache;
    class TextureHandle;
    class ShaderFactory;
    class CommonRenderPasses;
    struct TextureData;
}

class ShaderDebug;

// This uses older version of AMD's https://github.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK/blob/main/docs/techniques/parallel-sort.md
// that was ported from https://github.com/GameTechDev/XeGTAO/blob/master/Source/Rendering/Shaders/FFX_ParallelSort.h 
// See Sort() below for HowTo's.

class GPUSort
{
public:
    GPUSort(nvrhi::IDevice* device, std::shared_ptr<donut::engine::ShaderFactory> shaderFactory);
    ~GPUSort();

    void                            CreateRenderPasses(std::shared_ptr<donut::engine::CommonRenderPasses> commonPasses, std::shared_ptr<ShaderDebug> shaderDebug);

    // Provide number of items to be sorted as a uint32 variable within the controlBuffer at the itemCountByteOffset. Must be less than or equal to maxItemCount. If bigger, behaviour is undefined.
    // Provide sort keys in the keys buffer (uint32). These will NOT be touched - only indices are sorted.
    // Provide indices in the indices buffer (uint32). These have to be already initialized to [1, 2, 3, 4, 5, ..., itemCount-1] and will be sorted in-place. Also known as the "payload". If you do not want to pre-initialize them, use `resetIndices`
    // Provide maxItemCount as the maximum possible itemCount - this will be used to create needed scratch buffers at runtime.
    // Do all necessary resource barriers before and after the call to Sort.
    void                            Sort(nvrhi::ICommandList * commandList, nvrhi::BufferHandle controlBuffer, uint itemCountByteOffset, nvrhi::BufferHandle bufferKeys, nvrhi::BufferHandle bufferIndices, uint maxItemCount, bool resetIndices);

private:
    void                            ReCreateWorkingBuffers(uint32_t maxItemCount);

private:
    nvrhi::DeviceHandle             m_device;
    std::shared_ptr<donut::engine::CommonRenderPasses> m_commonPasses;
    std::shared_ptr<donut::engine::FramebufferFactory> m_framebufferFactory;
    std::shared_ptr<donut::engine::ShaderFactory> m_shaderFactory;
    std::shared_ptr<ShaderDebug>   m_shaderDebug;

    nvrhi::ShaderHandle             m_CSSetupIndirect;
    nvrhi::ComputePipelineHandle    m_PSOSetupIndirect;
    nvrhi::ShaderHandle             m_CSCount;
    nvrhi::ComputePipelineHandle    m_PSOCount;
    nvrhi::ShaderHandle             m_CSCountIIFP;      // init indices first pass
    nvrhi::ComputePipelineHandle    m_PSOCountIIFP;     // init indices first pass
    nvrhi::ShaderHandle             m_CSCountReduce;
    nvrhi::ComputePipelineHandle    m_PSOCountReduce;
    nvrhi::ShaderHandle             m_CSScanPrefix;
    nvrhi::ComputePipelineHandle    m_PSOScanPrefix;
    nvrhi::ShaderHandle             m_CSScanAdd;
    nvrhi::ComputePipelineHandle    m_PSOScanAdd;
    nvrhi::ShaderHandle             m_CSScatter;
    nvrhi::ComputePipelineHandle    m_PSOScatter;
    nvrhi::ShaderHandle             m_CSScatterIIFP;    // init indices first pass
    nvrhi::ComputePipelineHandle    m_PSOScatterIIFP;   // init indices first pass
    nvrhi::ShaderHandle             m_CSValidate;
    nvrhi::ComputePipelineHandle    m_PSOValidate;

    nvrhi::BindingLayoutHandle      m_initBindingLayout;
    nvrhi::BindingLayoutHandle      m_commonBindingLayout;
    donut::engine::BindingCache     m_bindingCache;

    nvrhi::BufferHandle             m_controlBuffer;
    nvrhi::BufferHandle             m_dispatchIndirectBuffer;
    nvrhi::BufferHandle             m_scratchBuffer;
    nvrhi::BufferHandle             m_reducedScratchBuffer;
    nvrhi::BufferHandle             m_scratchIndicesBuffer;

    uint32_t                        m_scratchMaxItemCountSize = 0;
};
