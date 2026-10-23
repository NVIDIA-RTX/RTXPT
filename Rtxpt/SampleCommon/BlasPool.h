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

    // A BLAS living at an offset inside a pool block rather than in an allocation of its own. It answers
    // only for its device address, because that is all the rest of the renderer asks of a BLAS: both the
    // address table in Sample::UploadTLASInstanceData and the InstanceDesc the CPU TLAS path fills take
    // an address, never the object.
    //
    // It must never reach an NVRHI entry point. NVRHI checked_casts IAccelStruct* to its own concrete
    // type, and checked_cast is a plain static_cast outside Debug, so the cast would quietly succeed and
    // the callee would read whatever happens to sit where its members would be.
    //
    // What keeps that from happening is structural, not defensive: the only NVRHI entry points left that
    // take a BLAS are the skinned refit in Sample::UpdateSkinnedBLASs, and a skinned mesh is never given
    // a pooled handle - Sample::CreateBlases forks on bvh::IsSkinnedMesh before assigning one. No assert
    // can enforce that in Release; preserving the fork is what enforces it.
    class PooledAccelStruct final : public nvrhi::RefCounter<nvrhi::rt::IAccelStruct>
    {
    public:
        const nvrhi::rt::AccelStructDesc& getDesc() const override  { return m_desc; }
        bool                              isCompacted() const override { return true; }
        uint64_t                          getDeviceAddress() const override { return m_deviceAddress; }

        // Null rather than the block's ID3D12Resource: a caller asking a BLAS for its native resource
        // wants that BLAS's size, and the block's size is not it.
        nvrhi::Object getNativeObject(nvrhi::ObjectType objectType) override { (void)objectType; return nullptr; }

        // Distinguishes a pooled BLAS from one NVRHI allocated. Used to assert the pooled/skinned
        // invariant where a handle is assigned, which is the one place the two can drift apart.
        static bool IsPooled(const nvrhi::rt::IAccelStruct* as);

    private:
        friend class BlasPool;
        PooledAccelStruct() = default;
        ~PooledAccelStruct() override;

        BlasPool* m_pool         = nullptr;
        uint32_t  m_blockIndex   = 0;
        uint64_t  m_blockOffset  = 0;
        uint64_t  m_sizeBytes    = 0;
        uint64_t  m_deviceAddress = 0;
        nvrhi::rt::AccelStructDesc m_desc;
    };

    // Suballocates compacted BLASes out of a small number of large buffers, so that a scene's BLASes cost
    // what they measure rather than a 64KB page each.
    //
    // Not thread safe: allocation happens only from the serial BLAS build path. Reading a device address
    // off an already-allocated PooledAccelStruct is const and is safe from the TLAS worker threads.
    class BlasPool
    {
    public:
        struct Stats
        {
            uint64_t ReservedBytes = 0;  // committed across all blocks
            uint64_t UsedBytes     = 0;  // handed out to live allocations
            uint32_t BlockCount    = 0;
            uint32_t LiveCount     = 0;
        };

        explicit BlasPool(nvrhi::IDevice* device);
        ~BlasPool();

        BlasPool(const BlasPool&) = delete;
        BlasPool& operator=(const BlasPool&) = delete;

        // sizeBytes is the compacted size from postbuild info; rounded up to the acceleration structure
        // alignment internally. Returns null if the pool cannot serve the request, in which case the
        // caller must fall back to an NVRHI-allocated acceleration structure.
        nvrhi::rt::AccelStructHandle Allocate(uint64_t sizeBytes, const nvrhi::rt::AccelStructDesc& sourceDesc);

        // Bumped by every allocation and every free. Sample::UpdateTLAS re-bakes the BLAS address table
        // when this moves; bumping it inside the allocator rather than at call sites is what stops a
        // future caller from forgetting and leaving the table pointing at freed memory.
        uint64_t GetGeneration() const { return m_generation; }

        Stats GetStats() const;

        // Drops every block. Asserts nothing is still allocated - that assert is the tripwire for a
        // pooled handle outliving the pool.
        void ReleaseAll();


    private:
        friend class PooledAccelStruct;

        struct Range
        {
            uint64_t Offset = 0;
            uint64_t Size   = 0;
        };

        struct Block
        {
            nvrhi::BufferHandle Buffer;
            uint64_t            BaseAddress = 0;
            uint64_t            Size        = 0;
            uint64_t            UsedBytes   = 0;
            std::vector<Range>  Free;          // sorted by Offset, non-overlapping, coalesced
        };

        void     Free(const PooledAccelStruct& as);
        bool     CreateBlock(uint64_t minimumSize, uint32_t& outBlockIndex);
        bool     AllocateFromBlock(Block& block, uint64_t size, uint64_t& outOffset);

        nvrhi::IDevice*    m_device = nullptr;
        std::vector<Block> m_blocks;
        uint64_t           m_generation   = 0;
        uint64_t           m_totalUsed    = 0;
        uint32_t           m_liveCount    = 0;
    };
}
