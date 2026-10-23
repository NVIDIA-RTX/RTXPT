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
#include <cstdint>

class ComputePipelineBaker;
class ComputeShaderVariant;
class PTPipelineBaker;
class PTPipelineVariant;

// SHaRC (Spatial Hash Radiance Cache) renderer.
// Owns the three GPU buffers (hash entries, accumulation, resolved),
// the resolve compute pass, and the sparse update RT pipeline.
class SHaRCRenderer
{
public:
    static constexpr uint32_t kDefaultCapacity = 1u << 22; // 2^22 = ~4M elements

    SHaRCRenderer(nvrhi::IDevice* device);
    ~SHaRCRenderer();

    void CreatePipelines(nvrhi::BindingLayoutHandle globalBindingLayout,
                         nvrhi::BindingLayoutHandle globalBindlessLayout,
                         ComputePipelineBaker& computeBaker);
    void DestroyPipelines(ComputePipelineBaker& computeBaker);

    void CreateRTPipeline(PTPipelineBaker& pipelineBaker);
    void DestroyRTPipeline(PTPipelineBaker& pipelineBaker);

    void CreateBuffers();

    // Sparse multi-bounce path tracing at 1/4 x 1/4 resolution to populate the cache.
    void DispatchUpdate(nvrhi::CommandListHandle commandList,
                        uint32_t renderWidth, uint32_t renderHeight,
                        nvrhi::BindingSetHandle globalBindingSet,
                        nvrhi::BindingSetHandle descriptorTable);

    // Combines new cache data with previous frame data.
    void Resolve(nvrhi::CommandListHandle commandList,
                 nvrhi::BindingSetHandle globalBindingSet,
                 nvrhi::BindingSetHandle descriptorTable);

    bool IsUpdateReady() const;

    nvrhi::BufferHandle GetHashEntries()   const { return m_hashEntries; }
    nvrhi::BufferHandle GetAccumulation()  const { return m_accumulation; }
    nvrhi::BufferHandle GetResolved()      const { return m_resolved; }

    uint32_t GetCapacity() const { return m_capacity; }

private:
    nvrhi::DeviceHandle m_device;
    uint32_t            m_capacity = kDefaultCapacity;
    bool                m_needsClear = true;

    nvrhi::BufferHandle m_hashEntries;   // RWStructuredBuffer<uint64_t> -- 8 B/elem
    nvrhi::BufferHandle m_accumulation;  // RWStructuredBuffer<uint4>    -- 16 B/elem
    nvrhi::BufferHandle m_resolved;      // RWStructuredBuffer<SharcPackedData> -- 16 B/elem

    std::shared_ptr<ComputeShaderVariant> m_resolveVariant;
    std::shared_ptr<PTPipelineVariant>    m_updatePipeline;
};
