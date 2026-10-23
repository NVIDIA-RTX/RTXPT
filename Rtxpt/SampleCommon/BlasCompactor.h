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

#include <cstdint>
#include <vector>

namespace bvh
{
    class BlasPool;

    // Builds bottom-level acceleration structures into a fixed staging heap and moves the compacted
    // results into a pool. This results in better packed memory for BLASs.
    //
    // Work is done in batches bounded by the staging heap, each batch costing two GPU round trips: one
    // to read back compacted sizes, one to let the compacting copies finish before their sources are
    // released. Those round trips are also what lets NVRHI recycle build scratch, which otherwise grows
    // for the whole load.
    // Future work: Spread this cost over many frames and allow streaming BLASs in and out instead of front
    // loading all builds.
    class BlasCompactor
    {
    public:
        BlasCompactor(nvrhi::IDevice* device, BlasPool& pool);
        ~BlasCompactor();

        BlasCompactor(const BlasCompactor&) = delete;
        BlasCompactor& operator=(const BlasCompactor&) = delete;

        // False when this backend or device cannot support the pooled path; the caller must then build
        // acceleration structures the ordinary way.
        bool IsAvailable();

        // Builds every descriptor and returns the pooled results in the same order, null for any that
        // failed. Closes, submits and reopens the command list once per batch, so the caller must own it
        // and must not rely on recording state across the call.
        std::vector<nvrhi::rt::AccelStructHandle> BuildAll(nvrhi::ICommandList* commandList,
                                                           const std::vector<nvrhi::rt::AccelStructDesc>& descs);

        uint32_t GetLastBatchCount() const { return m_lastBatchCount; }

    private:
        struct Entry
        {
            size_t                       Index = 0;      // position in the caller's array
            const nvrhi::rt::AccelStructDesc* Desc = nullptr;
            uint64_t                     SizeBytes = 0;  // staging footprint, heap-alignment rounded
            nvrhi::rt::AccelStructHandle Staging;
        };

        bool BuildBatch(nvrhi::ICommandList* commandList, std::vector<Entry>& batch,
                        std::vector<nvrhi::rt::AccelStructHandle>& results);
        bool EnsureSizeBuffers(size_t entryCount);

        nvrhi::IDevice*     m_device = nullptr;
        BlasPool&           m_pool;
        nvrhi::BufferHandle m_sizeBuffer;
        nvrhi::BufferHandle m_sizeReadback;
        bool                m_available = false;
        bool                m_checked = false;
        uint32_t            m_lastBatchCount = 0;
    };
}
