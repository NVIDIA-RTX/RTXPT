/*
* Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "BlasPool.h"

#include <donut/core/log.h>

#include <algorithm>
#include <cassert>
#include <string>

#if DONUT_WITH_DX12
#include <nvrhi/d3d12.h>
#endif

namespace bvh
{
namespace
{
    // DXR requires every acceleration structure to start on a 256-byte boundary.
    constexpr uint64_t c_accelStructAlignment = 256;
#if DONUT_WITH_DX12
    static_assert(c_accelStructAlignment == D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BYTE_ALIGNMENT);
#endif

    // First block is generous enough that small scenes need exactly one; later blocks track what the
    // scene has already proven it needs, capped so a single block never dominates the budget. A flat
    // large size would commit a whole block for a scene whose entire BLAS set is smaller than one.
    constexpr uint64_t c_firstBlockBytes = 32ull << 20;
    constexpr uint64_t c_maxBlockBytes   = 64ull << 20;

    uint64_t AlignUp(uint64_t value, uint64_t alignment)
    {
        return (value + alignment - 1) & ~(alignment - 1);
    }
}

PooledAccelStruct::~PooledAccelStruct()
{
    if (m_pool != nullptr)
        m_pool->Free(*this);
}

bool PooledAccelStruct::IsPooled(const nvrhi::rt::IAccelStruct* as)
{
    // dynamic_cast rather than a tag member: the whole point is to be called on pointers that may not be
    // this type, and reading a member off one of those would be exactly the bug being guarded against.
    return dynamic_cast<const PooledAccelStruct*>(as) != nullptr;
}

BlasPool::BlasPool(nvrhi::IDevice* device)
    : m_device(device)
{
}

BlasPool::~BlasPool()
{
    assert(m_liveCount == 0); // a pooled handle has outlived the pool that backs it
}

bool BlasPool::CreateBlock(uint64_t minimumSize, uint32_t& outBlockIndex)
{
    if (m_device == nullptr)
        return false;

    // A single allocation larger than the cap simply gets a block sized to hold it - still an ordinary
    // block, so it can host later allocations and is reclaimed like any other.
    uint64_t blockSize = m_blocks.empty() ? c_firstBlockBytes
                                         : std::clamp(m_totalUsed, c_firstBlockBytes, c_maxBlockBytes);
    blockSize = std::max(blockSize, minimumSize);

    Block block;
    block.Size = blockSize;

    nvrhi::BufferDesc desc;
    desc.byteSize = blockSize;
    desc.canHaveUAVs = true;              // acceleration structure storage requires ALLOW_UNORDERED_ACCESS
    desc.isAccelStructStorage = true;
    desc.initialState = nvrhi::ResourceStates::AccelStructBuildBlas;
    desc.keepInitialState = true;
    desc.debugName = "BlasPoolBlock" + std::to_string(m_blocks.size());
    block.Buffer = m_device->createBuffer(desc);
    if (block.Buffer == nullptr)
    {
        donut::log::warning("[BlasPool] failed to create a %llu MB block", blockSize >> 20);
        return false;
    }

    block.BaseAddress = block.Buffer->getGpuVirtualAddress();

    // Committed buffers start on a 64KB boundary, so every suballocated offset inherits the alignment
    // an acceleration structure needs.
    assert((block.BaseAddress % c_accelStructAlignment) == 0);

    block.Free.push_back({ 0, blockSize });
    outBlockIndex = (uint32_t)m_blocks.size();
    m_blocks.push_back(std::move(block));
    return true;
}

bool BlasPool::AllocateFromBlock(Block& block, uint64_t size, uint64_t& outOffset)
{
    for (size_t i = 0; i < block.Free.size(); i++)
    {
        Range& range = block.Free[i];
        if (range.Size < size)
            continue;

        outOffset = range.Offset;
        if (range.Size == size)
            block.Free.erase(block.Free.begin() + i);
        else
        {
            range.Offset += size;
            range.Size   -= size;
        }
        block.UsedBytes += size;
        return true;
    }
    return false;
}

nvrhi::rt::AccelStructHandle BlasPool::Allocate(uint64_t sizeBytes, const nvrhi::rt::AccelStructDesc& sourceDesc)
{
    if (sizeBytes == 0)
        return nullptr;

    const uint64_t size = AlignUp(sizeBytes, c_accelStructAlignment);

    uint32_t blockIndex = ~0u;
    uint64_t offset = 0;
    for (uint32_t i = 0; i < (uint32_t)m_blocks.size(); i++)
    {
        if (AllocateFromBlock(m_blocks[i], size, offset))
        {
            blockIndex = i;
            break;
        }
    }

    if (blockIndex == ~0u)
    {
        if (!CreateBlock(size, blockIndex))
            return nullptr;
        if (!AllocateFromBlock(m_blocks[blockIndex], size, offset))
        {
            assert(false); // a fresh block must be able to hold what it was sized for
            return nullptr;
        }
    }

    PooledAccelStruct* as = new PooledAccelStruct();
    as->m_pool          = this;
    as->m_blockIndex    = blockIndex;
    as->m_blockOffset   = offset;
    as->m_sizeBytes     = size;
    as->m_deviceAddress = m_blocks[blockIndex].BaseAddress + offset;
    as->m_desc          = sourceDesc;
    // The geometry descriptors hold raw IBuffer pointers that outlive nothing in particular; NVRHI
    // clears them on its own acceleration structures for the same reason. Flags and name are what
    // callers actually read back.
    as->m_desc.bottomLevelGeometries.clear();

    m_totalUsed += size;
    m_liveCount++;
    m_generation++;

    return nvrhi::rt::AccelStructHandle::Create(as);
}

void BlasPool::Free(const PooledAccelStruct& as)
{
    assert(as.m_blockIndex < m_blocks.size());
    Block& block = m_blocks[as.m_blockIndex];

    const Range freed { as.m_blockOffset, as.m_sizeBytes };

    // Insert in offset order, then coalesce with either neighbour it now touches.
    size_t insertAt = 0;
    while (insertAt < block.Free.size() && block.Free[insertAt].Offset < freed.Offset)
        insertAt++;
    block.Free.insert(block.Free.begin() + insertAt, freed);

    if (insertAt + 1 < block.Free.size() &&
        block.Free[insertAt].Offset + block.Free[insertAt].Size == block.Free[insertAt + 1].Offset)
    {
        block.Free[insertAt].Size += block.Free[insertAt + 1].Size;
        block.Free.erase(block.Free.begin() + insertAt + 1);
    }
    if (insertAt > 0 &&
        block.Free[insertAt - 1].Offset + block.Free[insertAt - 1].Size == block.Free[insertAt].Offset)
    {
        block.Free[insertAt - 1].Size += block.Free[insertAt].Size;
        block.Free.erase(block.Free.begin() + insertAt);
    }

    assert(block.UsedBytes >= as.m_sizeBytes);
    block.UsedBytes -= as.m_sizeBytes;
    m_totalUsed -= as.m_sizeBytes;
    assert(m_liveCount > 0);
    m_liveCount--;
    m_generation++;
}

BlasPool::Stats BlasPool::GetStats() const
{
    Stats stats;
    for (const Block& block : m_blocks)
    {
        stats.ReservedBytes += block.Size;
        stats.UsedBytes     += block.UsedBytes;
    }
    stats.BlockCount = (uint32_t)m_blocks.size();
    stats.LiveCount  = m_liveCount;
    return stats;
}

void BlasPool::ReleaseAll()
{
    assert(m_liveCount == 0); // something still holds a pooled handle; releasing now would dangle it
    m_blocks.clear();
    m_totalUsed = 0;
    m_generation++;
}

}
