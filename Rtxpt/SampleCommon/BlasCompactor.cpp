/*
* Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "BlasCompactor.h"
#include "BlasPool.h"

#include <donut/core/log.h>

#include <nvrhi/utils.h>

#include <algorithm>
#include <cassert>

#if DONUT_WITH_DX12
#include <nvrhi/d3d12.h>
#include <wrl/client.h>
#endif

namespace bvh
{
namespace
{
    // Bounds how much uncompacted build output is resident at once. Measured on Caldera: 43 batches,
    // peak 32MB of staging plus ~25MB of build scratch. Larger trades load-time memory for fewer round
    // trips; it is not what sets the peak on a scene with one enormous mesh.
    constexpr uint64_t c_stagingBytes = 32ull << 20;

    uint64_t AlignUp(uint64_t value, uint64_t alignment)
    {
        return (alignment == 0) ? value : ((value + alignment - 1) & ~(alignment - 1));
    }

#if DONUT_WITH_DX12
    ID3D12GraphicsCommandList4* GetNativeCommandList(nvrhi::ICommandList* commandList)
    {
        ID3D12GraphicsCommandList* base = commandList->getNativeObject(nvrhi::ObjectTypes::D3D12_GraphicsCommandList);
        if (base == nullptr)
            return nullptr;
        // QueryInterface would addref; the command list outlives the call and NVRHI owns it.
        Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList4> list4;
        if (FAILED(base->QueryInterface(IID_PPV_ARGS(&list4))))
            return nullptr;
        return list4.Get();
    }

    void GlobalUavBarrier(ID3D12GraphicsCommandList4* commandList)
    {
        D3D12_RESOURCE_BARRIER barrier = {};
        barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
        barrier.UAV.pResource = nullptr;
        commandList->ResourceBarrier(1, &barrier);
    }

#endif
}

BlasCompactor::BlasCompactor(nvrhi::IDevice* device, BlasPool& pool)
    : m_device(device)
    , m_pool(pool)
{
}

BlasCompactor::~BlasCompactor() = default;

bool BlasCompactor::IsAvailable()
{
    if (m_checked)
        return m_available;
    m_checked = true;

    // Vulkan has equivalents for both - vkCmdWriteAccelerationStructuresPropertiesKHR and a compacting
    // vkCmdCopyAccelerationStructureKHR - but NVRHI exposes neither, and its copyRaytracingAccelerationStructure
    // is clone-only and takes an IAccelStruct rather than a destination address. So the gate is on this
    // implementation, not on the capability; a Vulkan path would need the same two calls written natively.
    m_available = m_device->getGraphicsAPI() == nvrhi::GraphicsAPI::D3D12;
    if (!m_available)
        donut::log::info("[BlasCompactor] BLAS pooling needs the D3D12 backend; building acceleration structures individually");
    return m_available;
}

bool BlasCompactor::EnsureSizeBuffers(size_t entryCount)
{
    const uint64_t required = sizeof(uint64_t) * std::max<size_t>(1, entryCount);
    if (m_sizeBuffer != nullptr && m_sizeBuffer->getDesc().byteSize >= required)
        return true;

    nvrhi::BufferDesc desc;
    desc.byteSize = required;
    desc.canHaveUAVs = true;
    desc.debugName = "BlasCompactedSizes";
    desc.initialState = nvrhi::ResourceStates::UnorderedAccess;
    desc.keepInitialState = true;
    m_sizeBuffer = m_device->createBuffer(desc);

    nvrhi::BufferDesc readbackDesc;
    readbackDesc.byteSize = required;
    readbackDesc.cpuAccess = nvrhi::CpuAccessMode::Read;
    readbackDesc.debugName = "BlasCompactedSizesReadback";
    m_sizeReadback = m_device->createBuffer(readbackDesc);

    return m_sizeBuffer != nullptr && m_sizeReadback != nullptr;
}

std::vector<nvrhi::rt::AccelStructHandle> BlasCompactor::BuildAll(nvrhi::ICommandList* commandList,
                                                                 const std::vector<nvrhi::rt::AccelStructDesc>& descs)
{
    std::vector<nvrhi::rt::AccelStructHandle> results(descs.size());
    if (descs.empty() || !IsAvailable())
        return results;

    // Sizing pass. A virtual acceleration structure runs the prebuild query without committing any
    // memory, which is what makes it affordable to measure every mesh before deciding batch membership.
    std::vector<Entry> pending;
    pending.reserve(descs.size());
    for (size_t i = 0; i < descs.size(); i++)
    {
        // Measured with a virtual structure so nothing is committed: the prebuild query has run, but no
        // memory is bound. The handle is dropped straight away - it is only ever asked for its size.
        nvrhi::rt::AccelStructDesc virtualDesc = descs[i];
        virtualDesc.isVirtual = true;
        nvrhi::rt::AccelStructHandle measured = m_device->createAccelStruct(virtualDesc);
        if (measured == nullptr)
            continue;

        const nvrhi::MemoryRequirements memory = m_device->getAccelStructMemoryRequirements(measured);
        Entry entry;
        entry.Index = i;
        entry.Desc = &descs[i];
        entry.SizeBytes = AlignUp(memory.size, std::max<uint64_t>(memory.alignment, 1));
        pending.push_back(std::move(entry));
    }

    // Largest first: it packs the staging heap better, and it front-loads the batch that overshoots.
    std::sort(pending.begin(), pending.end(),
              [](const Entry& a, const Entry& b) { return a.SizeBytes > b.SizeBytes; });

    m_lastBatchCount = 0;
    std::vector<Entry> batch;
    uint64_t batchBytes = 0;
    for (Entry& entry : pending)
    {
        if (!batch.empty() && batchBytes + entry.SizeBytes > c_stagingBytes)
        {
            BuildBatch(commandList, batch, results);
            batch.clear();
            batchBytes = 0;
        }

        batchBytes += entry.SizeBytes;
        batch.push_back(std::move(entry));

        // Too large for the heap even on its own. It becomes a batch of one, built into an allocation of
        // its own rather than the heap - rare enough to be worth a note, not an error.
        if (batch.size() == 1 && batchBytes > c_stagingBytes)
        {
            donut::log::info("[BlasCompactor] '%s' needs %llu MB of build space, more than the %llu MB staging heap; building it on its own",
                             batch[0].Desc->debugName.c_str(), batchBytes >> 20, c_stagingBytes >> 20);
            BuildBatch(commandList, batch, results);
            batch.clear();
            batchBytes = 0;
        }
    }
    if (!batch.empty())
        BuildBatch(commandList, batch, results);

    return results;
}

bool BlasCompactor::BuildBatch(nvrhi::ICommandList* commandList, std::vector<Entry>& batch,
                               std::vector<nvrhi::rt::AccelStructHandle>& results)
{
#if DONUT_WITH_DX12
    if (batch.empty() || !EnsureSizeBuffers(batch.size()))
        return false;

    m_lastBatchCount++;

    ID3D12GraphicsCommandList4* nativeList = GetNativeCommandList(commandList);
    if (nativeList == nullptr)
        return false;

    // Allocate this batch's staging structures. NVRHI cannot place an acceleration structure into a heap
    // we own - createBuffer returns before applying isAccelStructStorage when isVirtual is set, so the
    // placed resource would lack D3D12_RESOURCE_FLAG_RAYTRACING_ACCELERATION_STRUCTURE and the build is
    // invalid - so the batch budget bounds ordinary allocations instead. Same peak, and they are released
    // together at the end of the batch.
    for (Entry& entry : batch)
    {
        nvrhi::rt::AccelStructDesc stagingDesc = *entry.Desc;
        stagingDesc.isVirtual = false;
        entry.Staging = m_device->createAccelStruct(stagingDesc);
        if (entry.Staging == nullptr)
        {
            donut::log::error("[BlasCompactor] could not allocate staging for '%s'", entry.Desc->debugName.c_str());
            continue;
        }
    }

    std::vector<D3D12_GPU_VIRTUAL_ADDRESS> sources;
    sources.reserve(batch.size());
    for (Entry& entry : batch)
    {
        if (entry.Staging == nullptr)
            continue;
        nvrhi::utils::BuildBottomLevelAccelStruct(commandList, entry.Staging, *entry.Desc);
        sources.push_back(entry.Staging->getDeviceAddress());
    }
    if (sources.empty())
        return false;

    // The postbuild query must see finished builds; a size read from an unfinished one would
    // under-allocate its pool range and let the compacting copy run into its neighbour.
    commandList->commitBarriers();
    GlobalUavBarrier(nativeList);

    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_DESC postbuildDesc = {};
    postbuildDesc.InfoType = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_COMPACTED_SIZE;
    postbuildDesc.DestBuffer = m_sizeBuffer->getGpuVirtualAddress();
    nativeList->EmitRaytracingAccelerationStructurePostbuildInfo(&postbuildDesc, (UINT)sources.size(), sources.data());

    commandList->setBufferState(m_sizeBuffer, nvrhi::ResourceStates::CopySource);
    commandList->commitBarriers();
    commandList->copyBuffer(m_sizeReadback, 0, m_sizeBuffer, 0, sizeof(uint64_t) * sources.size());

    // Round trip: the compacted sizes have to reach the CPU before pool ranges can be assigned.
    commandList->close();
    m_device->executeCommandList(commandList);
    m_device->waitForIdle();
    commandList->open();

    std::vector<uint64_t> compactedSizes(sources.size(), 0);
    if (void* mapped = m_device->mapBuffer(m_sizeReadback, nvrhi::CpuAccessMode::Read))
    {
        memcpy(compactedSizes.data(), mapped, sizeof(uint64_t) * compactedSizes.size());
        m_device->unmapBuffer(m_sizeReadback);
    }

    nativeList = GetNativeCommandList(commandList);
    if (nativeList == nullptr)
        return false;

    size_t sourceIndex = 0;
    for (Entry& entry : batch)
    {
        if (entry.Staging == nullptr)
            continue;
        const uint64_t compacted = compactedSizes[sourceIndex++];

        // A compacted structure is never larger than the build it came from; anything else means the
        // size was read before the build finished.
        assert(compacted > 0 && compacted <= entry.SizeBytes);
        if (compacted == 0 || compacted > entry.SizeBytes)
        {
            donut::log::error("[BlasCompactor] implausible compacted size %llu for '%s' (build reserved %llu)",
                              compacted, entry.Desc->debugName.c_str(), entry.SizeBytes);
            continue;
        }

        nvrhi::rt::AccelStructHandle pooled = m_pool.Allocate(compacted, *entry.Desc);
        if (pooled == nullptr)
        {
            donut::log::error("[BlasCompactor] pool allocation failed for '%s'", entry.Desc->debugName.c_str());
            continue;
        }

        nativeList->CopyRaytracingAccelerationStructure(
            pooled->getDeviceAddress(), entry.Staging->getDeviceAddress(),
            D3D12_RAYTRACING_ACCELERATION_STRUCTURE_COPY_MODE_COMPACT);

        results[entry.Index] = pooled;
    }

    // Order the copies against whatever reads the pool next, and against the next batch reusing the heap.
    commandList->commitBarriers();
    GlobalUavBarrier(nativeList);

    // The copies must retire before their sources are released. NVRHI does not know a native copy read
    // them, so nothing else keeps the staging structures alive - dropping them early frees memory the
    // GPU is still reading, which faults nowhere and quietly corrupts the results.
    commandList->close();
    m_device->executeCommandList(commandList);
    m_device->waitForIdle();
    commandList->open();

    batch.clear();
    m_device->runGarbageCollection();
    return true;
#else
    (void)commandList; (void)batch; (void)results;
    return false;
#endif
}
}
