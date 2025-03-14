/*
* Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "OmmBuildQueue.h"

#include <donut/core/math/math.h>
#include <donut/core/vfs/VFS.h>
#include <donut/engine/ShaderFactory.h>
#include <nvrhi/common/misc.h>
#include <nvrhi/utils.h>
#include "../Materials/MaterialsBaker.h"
#include "../OpacityMicroMap/OmmBaker.h"

namespace
{
    static omm::GpuBakeNvrhi::Input GetBakeInput(omm::GpuBakeNvrhi::Operation op, const donut::engine::MeshInfo& mesh, const OmmBuildQueue::BuildInput::Geometry& geometry)
    {
        using namespace donut;
        using namespace donut::math;

        const donut::engine::MeshGeometry* meshGeometry = mesh.geometries[geometry.geometryIndexInMesh].get();

        const uint32_t indexOffset = mesh.indexOffset + meshGeometry->indexOffsetInMesh;
        const uint32_t vertexOffset = (mesh.vertexOffset + meshGeometry->vertexOffsetInMesh);

        MaterialPT & materialPT = *MaterialPT::FromDonut(meshGeometry->material);

        omm::GpuBakeNvrhi::Input params;
        params.operation = op;
        params.alphaTexture = materialPT.BaseTexture.Loaded->texture;
        params.alphaCutoff = materialPT.AlphaCutoff;
        params.alphaCutoffGreater = geometry.alphaCutoffGT;
        params.alphaCutoffLessEqual = geometry.alphaCutoffLE;
        params.bilinearFilter = true;
        params.enableLevelLineIntersection = geometry.enableLevelLineIntersection;
        params.sampleMode = nvrhi::SamplerAddressMode::Wrap;
        params.indexBuffer = mesh.buffers->indexBuffer;
        params.texCoordBuffer = mesh.buffers->vertexBuffer;
        params.texCoordBufferOffsetInBytes = (uint)(vertexOffset * sizeof(float2) + mesh.buffers->getVertexBufferRange(engine::VertexAttribute::TexCoord1).byteOffset);
        params.texCoordStrideInBytes = sizeof(float2);
        params.indexBufferOffsetInBytes = size_t(indexOffset) * sizeof(uint32_t);
        params.numIndices = meshGeometry->numIndices;
        params.maxSubdivisionLevel = geometry.maxSubdivisionLevel;
        params.dynamicSubdivisionScale = geometry.dynamicSubdivisionScale;
        params.format = geometry.format;
        params.minimalMemoryMode = false;
        params.enableStats = true;
        params.force32BitIndices = geometry.force32BitIndices;
        params.enableSpecialIndices = geometry.enableSpecialIndices;
        params.computeOnly = geometry.computeOnly;
        params.enableNsightDebugMode = geometry.enableNsightDebugMode;
        params.enableTexCoordDeduplication = geometry.enableTexCoordDeduplication;
        params.maxOutOmmArraySize = geometry.maxOmmArrayDataSizeInMB << 20;
        
        return params;
    }

    enum class BufferConfig
    {
        RawUAV,
        RawUAVAndASBuildInput,
        Readback
    };

    static nvrhi::BufferHandle AllocateBuffer(nvrhi::DeviceHandle device, const char* name, size_t byteSize, BufferConfig cfg)
    {
        nvrhi::BufferDesc desc;
        desc.byteSize = byteSize;
        desc.debugName = name;
        desc.format = nvrhi::Format::R32_UINT;
        if (cfg == BufferConfig::RawUAV)
        {
            desc.canHaveUAVs = true;
            desc.canHaveRawViews = true;
        }
        else if (cfg == BufferConfig::RawUAVAndASBuildInput)
        {
            desc.canHaveUAVs = true;
            desc.canHaveRawViews = true;
            desc.isAccelStructBuildInput = true;
        }
        else
        {
            desc.canHaveUAVs = false;
            desc.cpuAccess = nvrhi::CpuAccessMode::Read;
        }
        return device->createBuffer(desc);
    }

    static nvrhi::rt::OpacityMicromapDesc GetOpacityMicromapDesc(
        nvrhi::BufferHandle ommArrayBuffer,
        size_t ommArrayBufferOffset,
        nvrhi::BufferHandle ommDescBuffer,
        size_t ommDescBufferOffset,
        const std::vector<nvrhi::rt::OpacityMicromapUsageCount>& usageDescs,
        nvrhi::rt::OpacityMicromapBuildFlags flags)
    {
        nvrhi::rt::OpacityMicromapDesc desc;
        desc.debugName = "OmmArray";
        desc.flags = flags;
        desc.counts = usageDescs;
        desc.inputBuffer = ommArrayBuffer;
        desc.inputBufferOffset = ommArrayBufferOffset;
        desc.perOmmDescs = ommDescBuffer;
        desc.perOmmDescsOffset = ommDescBufferOffset;
        return desc;
    }

    struct LinearBufferAllocator
    {
        LinearBufferAllocator()
        {
        }

        size_t Allocate(size_t sizeInBytes, size_t alignment)
        {
            m_offset = nvrhi::align(m_offset, alignment);
            size_t offset = m_offset;
            m_offset += sizeInBytes;
            return offset;
        }

        nvrhi::BufferHandle CreateBuffer(nvrhi::DeviceHandle device, const char* name, BufferConfig config)
        {
            if (m_offset == 0)
                return nullptr;

            nvrhi::BufferHandle handle = AllocateBuffer(device, name, m_offset, config);
            m_offset = 0;
            return handle;
        }

    private:
        size_t m_offset = 0;
    };
}

OmmBuildQueue::OmmBuildQueue(
    nvrhi::DeviceHandle device, 
    std::shared_ptr<donut::engine::DescriptorTableManager> descriptorTable, 
    std::shared_ptr<donut::engine::ShaderFactory> shaderFactory)
    : m_device(device)
    , m_descriptorTable(descriptorTable)
    , m_shaderFactory(shaderFactory)
{
}

void OmmBuildQueue::Initialize(nvrhi::CommandListHandle commandList)
{
    omm::GpuBakeNvrhi::ShaderProvider provider;

    provider.bindingOffsets = nvrhi::VulkanBindingOffsets { .shaderResource = 200, .sampler = 100, .constantBuffer = 300, .unorderedAccess = 400 };
    //provider.bindingOffsets = nvrhi::VulkanBindingOffsets { .shaderResource = UINT_MAX, .sampler = UINT_MAX, .constantBuffer = UINT_MAX, .unorderedAccess = UINT_MAX }; <- if MR gets accepted

    provider.shaders = [this](nvrhi::ShaderType type, const char* shaderName, const char* shaderEntryName)->nvrhi::ShaderHandle
    {
        std::vector<donut::engine::ShaderMacro> defines = { donut::engine::ShaderMacro("COMPILER_DXC", "1") };
        std::string shaderNameStr = std::string("omm/Omm/libraries/omm-lib/shaders/") + shaderName;
        return m_shaderFactory->CreateShader(shaderNameStr.c_str(), shaderEntryName, &defines, type);
    };

    omm::GpuBakeNvrhi::MessageCallback messageCb = [](omm::MessageSeverity severity, const char* message) {
        donut::log::info("[OMM SDK]: %d %s", severity, message);
    };

    m_baker = std::make_unique<omm::GpuBakeNvrhi>(m_device, commandList, false /*debug*/, &provider, messageCb);
}

OmmBuildQueue::~OmmBuildQueue()
{
}

void OmmBuildQueue::RunSetup(nvrhi::CommandListHandle commandList, BuildTask& task)
{
    assert(task.state == BuildState::None);

    std::shared_ptr < donut::engine::MeshInfo > mesh = task.input.mesh;

    LinearBufferAllocator ommIndexBufferAllocator;
    LinearBufferAllocator ommDescBufferAllocator;
    LinearBufferAllocator ommDescArrayHistogramBufferAllocator;
    LinearBufferAllocator ommIndexArrayHistogramBufferAllocator;
    LinearBufferAllocator ommPostBuildInfoBufferAllocator;
    LinearBufferAllocator ommReadbackBufferAllocator;

    std::vector< BufferInfo > bufferInfos;

    for (const OmmBuildQueue::BuildInput::Geometry& geom : task.input.geometries)
    {
        omm::GpuBakeNvrhi::Input input = GetBakeInput(omm::GpuBakeNvrhi::Operation::Setup, *mesh.get(), geom);

        omm::GpuBakeNvrhi::PreDispatchInfo setupInfo;
        m_baker->GetPreDispatchInfo(input, setupInfo);

        BufferInfo info;
        info.ommIndexFormat                         = setupInfo.ommIndexFormat;
        info.ommIndexCount                          = setupInfo.ommIndexCount;
        info.ommIndexOffset                         = ommIndexBufferAllocator.Allocate(setupInfo.ommIndexBufferSize, 256);
        info.ommDescArrayOffset                     = ommDescBufferAllocator.Allocate(setupInfo.ommDescBufferSize, 256);
        info.ommDescArrayHistogramOffset            = ommDescArrayHistogramBufferAllocator.Allocate(setupInfo.ommDescArrayHistogramSize, 256);
        info.ommDescArrayHistogramSize              = setupInfo.ommDescArrayHistogramSize;
        info.ommDescArrayHistogramReadbackOffset    = ommReadbackBufferAllocator.Allocate(setupInfo.ommDescArrayHistogramSize, 256);
        info.ommIndexHistogramOffset                = ommIndexArrayHistogramBufferAllocator.Allocate(setupInfo.ommIndexHistogramSize, 256);
        info.ommIndexHistogramSize                  = setupInfo.ommIndexHistogramSize;
        info.ommIndexHistogramReadbackOffset        = ommReadbackBufferAllocator.Allocate(setupInfo.ommIndexHistogramSize, 256);
        info.ommPostDispatchInfoOffset              = ommPostBuildInfoBufferAllocator.Allocate(setupInfo.ommPostDispatchInfoBufferSize, 256);
        info.ommPostDispatchInfoReadbackOffset      = ommReadbackBufferAllocator.Allocate(setupInfo.ommPostDispatchInfoBufferSize, 256);
        info.ommArrayDataOffset                     = 0xFFFFFFFF; // Not yet known

        bufferInfos.push_back(info);
    }

    Buffers buffers;
    buffers.ommIndexBuffer                  = ommIndexBufferAllocator.CreateBuffer(m_device, "OmmIndexBuffer", BufferConfig::RawUAVAndASBuildInput);
    buffers.ommDescBuffer                   = ommDescBufferAllocator.CreateBuffer(m_device, "OmmDescBuffer", BufferConfig::RawUAVAndASBuildInput);
    buffers.ommDescArrayHistogramBuffer     = ommDescArrayHistogramBufferAllocator.CreateBuffer(m_device, "OmmDescArrayHistogramBuffer", BufferConfig::RawUAV);
    buffers.ommIndexArrayHistogramBuffer    = ommIndexArrayHistogramBufferAllocator.CreateBuffer(m_device, "OmmIndexArrayHistogramBuffer", BufferConfig::RawUAV);
    buffers.ommPostDispatchInfoBuffer       = ommPostBuildInfoBufferAllocator.CreateBuffer(m_device, "OmmPostBuildInfoBuffer", BufferConfig::RawUAV);
    buffers.ommReadbackBuffer               = ommReadbackBufferAllocator.CreateBuffer(m_device, "OmmGenericReadbackBuffer", BufferConfig::Readback);

    commandList->beginTrackingBufferState(buffers.ommIndexBuffer, nvrhi::ResourceStates::Common);
    commandList->beginTrackingBufferState(buffers.ommDescBuffer, nvrhi::ResourceStates::Common);
    commandList->beginTrackingBufferState(buffers.ommDescArrayHistogramBuffer, nvrhi::ResourceStates::Common);
    commandList->beginTrackingBufferState(buffers.ommIndexArrayHistogramBuffer, nvrhi::ResourceStates::Common);
    commandList->beginTrackingBufferState(buffers.ommPostDispatchInfoBuffer, nvrhi::ResourceStates::Common);
    commandList->beginTrackingBufferState(buffers.ommReadbackBuffer, nvrhi::ResourceStates::Common);

    // Dispatch all setup tasks.
    for (uint32_t i = 0; i < task.input.geometries.size(); ++i)
    {
        const BufferInfo& bufferInfo = bufferInfos[i];
        const OmmBuildQueue::BuildInput::Geometry& geom = task.input.geometries[i];

        omm::GpuBakeNvrhi::Input input              = GetBakeInput(omm::GpuBakeNvrhi::Operation::Setup, *mesh.get(), geom);

        omm::GpuBakeNvrhi::Buffers output;
        output.ommDescBuffer                        = buffers.ommDescBuffer;
        output.ommDescBufferOffset                  = (uint32_t)bufferInfo.ommDescArrayOffset;
        output.ommIndexBuffer                       = buffers.ommIndexBuffer;
        output.ommIndexBufferOffset                 = (uint32_t)bufferInfo.ommIndexOffset;
        output.ommDescArrayHistogramBuffer          = buffers.ommDescArrayHistogramBuffer;
        output.ommDescArrayHistogramBufferOffset    = (uint32_t)bufferInfo.ommDescArrayHistogramOffset;
        output.ommIndexHistogramBuffer              = buffers.ommIndexArrayHistogramBuffer;
        output.ommIndexHistogramBufferOffset        = (uint32_t)bufferInfo.ommIndexHistogramOffset;
        output.ommPostDispatchInfoBuffer            = buffers.ommPostDispatchInfoBuffer;
        output.ommPostDispatchInfoBufferOffset      = (uint32_t)bufferInfo.ommPostDispatchInfoOffset;

        m_baker->Dispatch(commandList, input, output);

        omm::GpuBakeNvrhi::PreDispatchInfo setupInfo;
        m_baker->GetPreDispatchInfo(input, setupInfo);

        commandList->copyBuffer(buffers.ommReadbackBuffer, bufferInfo.ommDescArrayHistogramReadbackOffset,  buffers.ommDescArrayHistogramBuffer,  bufferInfo.ommDescArrayHistogramOffset,      setupInfo.ommDescArrayHistogramSize);
        commandList->copyBuffer(buffers.ommReadbackBuffer, bufferInfo.ommIndexHistogramReadbackOffset,  buffers.ommIndexArrayHistogramBuffer, bufferInfo.ommIndexHistogramOffset, setupInfo.ommIndexHistogramSize);
        commandList->copyBuffer(buffers.ommReadbackBuffer, bufferInfo.ommPostDispatchInfoReadbackOffset,   buffers.ommPostDispatchInfoBuffer,       bufferInfo.ommPostDispatchInfoOffset,  setupInfo.ommPostDispatchInfoBufferSize);
    }

    nvrhi::EventQueryHandle query = m_device->createEventQuery();
    m_device->setEventQuery(query, nvrhi::CommandQueue::Graphics);

    task.query = query;
    task.state = BuildState::Setup;
    task.buffers = std::move(buffers);
    task.bufferInfos = std::move(bufferInfos);
}

void OmmBuildQueue::RunBakeAndBuild(nvrhi::CommandListHandle commandList, BuildTask& task)
{
    assert(task.state == BuildState::None || task.state == BuildState::Setup);

    LinearBufferAllocator ommArrayDataBufferAllocator;
    {
        void* pReadbackData = m_device->mapBuffer(task.buffers.ommReadbackBuffer, nvrhi::CpuAccessMode::Read);
        for (uint32_t i = 0; i < task.input.geometries.size(); ++i)
        {
            BufferInfo& bufferInfo = task.bufferInfos[i];
            
            m_baker->ReadUsageDescBuffer((uint8_t*)pReadbackData + bufferInfo.ommDescArrayHistogramReadbackOffset, bufferInfo.ommDescArrayHistogramSize, bufferInfo.ommArrayHistogram);
            
            m_baker->ReadUsageDescBuffer((uint8_t*)pReadbackData + bufferInfo.ommIndexHistogramReadbackOffset, bufferInfo.ommIndexHistogramSize, bufferInfo.ommIndexHistogram);

            omm::GpuBakeNvrhi::PostDispatchInfo postDispatchInfo;
            m_baker->ReadPostDispatchInfo((uint8_t*)pReadbackData + bufferInfo.ommPostDispatchInfoReadbackOffset, sizeof(postDispatchInfo), postDispatchInfo);
            size_t ommArrayDataOffset = ommArrayDataBufferAllocator.Allocate(postDispatchInfo.ommArrayBufferSize, 256);

            if ((ommArrayDataOffset) > std::numeric_limits<uint32_t>::max())
            {
                assert(false && "OH OH. Exceeding 4gb of ommArrayData");
            }

            bufferInfo.ommArrayDataOffset = (uint32_t)ommArrayDataOffset;
        }
        m_device->unmapBuffer(task.buffers.ommReadbackBuffer);

        task.buffers.ommArrayDataBuffer = ommArrayDataBufferAllocator.CreateBuffer(m_device, "OmmArrayBuffer", BufferConfig::RawUAVAndASBuildInput);
    }

    // Dispatch the bake which will fill the ommArrayData
    {
        commandList->beginTrackingBufferState(task.buffers.ommIndexBuffer, nvrhi::ResourceStates::Common);
        commandList->beginTrackingBufferState(task.buffers.ommDescBuffer, nvrhi::ResourceStates::Common);
        commandList->beginTrackingBufferState(task.buffers.ommDescArrayHistogramBuffer, nvrhi::ResourceStates::Common);
        commandList->beginTrackingBufferState(task.buffers.ommIndexArrayHistogramBuffer, nvrhi::ResourceStates::Common);
        commandList->beginTrackingBufferState(task.buffers.ommPostDispatchInfoBuffer, nvrhi::ResourceStates::Common);
        commandList->beginTrackingBufferState(task.buffers.ommArrayDataBuffer, nvrhi::ResourceStates::Common);

        commandList->clearBufferUInt(task.buffers.ommArrayDataBuffer, 0);

        std::shared_ptr < donut::engine::MeshInfo > mesh = task.input.mesh;

        for (uint32_t i = 0; i < task.input.geometries.size(); ++i)
        {
            const BufferInfo& bufferInfo = task.bufferInfos[i];
            const OmmBuildQueue::BuildInput::Geometry& geom = task.input.geometries[i];

            omm::GpuBakeNvrhi::Input input              = GetBakeInput(omm::GpuBakeNvrhi::Operation::Bake, *mesh.get(), geom);

            omm::GpuBakeNvrhi::Buffers output;
            output.ommArrayBuffer                       = task.buffers.ommArrayDataBuffer;
            output.ommArrayBufferOffset                 = (uint32_t)bufferInfo.ommArrayDataOffset;
            output.ommDescBuffer                        = task.buffers.ommDescBuffer;
            output.ommDescBufferOffset                  = (uint32_t)bufferInfo.ommDescArrayOffset;
            output.ommIndexBuffer                       = task.buffers.ommIndexBuffer;
            output.ommIndexBufferOffset                 = (uint32_t)bufferInfo.ommIndexOffset;
            output.ommDescArrayHistogramBuffer          = task.buffers.ommDescArrayHistogramBuffer;
            output.ommDescArrayHistogramBufferOffset    = (uint32_t)bufferInfo.ommDescArrayHistogramOffset;
            output.ommIndexHistogramBuffer              = task.buffers.ommIndexArrayHistogramBuffer;
            output.ommIndexHistogramBufferOffset        = (uint32_t)bufferInfo.ommIndexHistogramOffset;
            output.ommPostDispatchInfoBuffer            = task.buffers.ommPostDispatchInfoBuffer;
            output.ommPostDispatchInfoBufferOffset      = (uint32_t)bufferInfo.ommPostDispatchInfoOffset;

            m_baker->Dispatch(commandList, input, output);

            omm::GpuBakeNvrhi::PreDispatchInfo setupInfo;
            m_baker->GetPreDispatchInfo(input, setupInfo);

            commandList->copyBuffer(task.buffers.ommReadbackBuffer, bufferInfo.ommPostDispatchInfoReadbackOffset, task.buffers.ommPostDispatchInfoBuffer, bufferInfo.ommPostDispatchInfoOffset, setupInfo.ommPostDispatchInfoBufferSize);
        }
    }

    std::vector<bvh::OmmAttachment> ommAttachment;
    ommAttachment.resize(task.input.mesh->geometries.size());

    // Build the OMM Arrays
    {
        for (uint32_t i = 0; i < task.input.geometries.size(); ++i)
        {
            const OmmBuildQueue::BuildInput::Geometry& geom = task.input.geometries[i];

            const BufferInfo& bufferInfo = task.bufferInfos[i];

            nvrhi::rt::OpacityMicromapDesc desc = GetOpacityMicromapDesc(
                task.buffers.ommArrayDataBuffer,    bufferInfo.ommArrayDataOffset, 
                task.buffers.ommDescBuffer,         bufferInfo.ommDescArrayOffset,
                bufferInfo.ommArrayHistogram,       geom.flags);

            nvrhi::rt::OpacityMicromapHandle ommBuffer = m_device->createOpacityMicromap(desc);

            std::static_pointer_cast<MeshInfoEx>(task.input.mesh)->OpacityMicroMaps.push_back(ommBuffer);

            commandList->buildOpacityMicromap(ommBuffer, desc);

            ommAttachment[geom.geometryIndexInMesh] =
            {
                .ommBuffer                    = ommBuffer,
                .ommIndexFormat               = bufferInfo.ommIndexFormat,
                .ommIndexHistogram            = bufferInfo.ommIndexHistogram,
                .ommIndexBuffer               = task.buffers.ommIndexBuffer,
                .ommIndexBufferOffset         = (uint32_t)bufferInfo.ommIndexOffset,
                .ommArrayDataBuffer           = task.buffers.ommArrayDataBuffer,
                .ommArrayDataBufferOffset     = (uint32_t)bufferInfo.ommArrayDataOffset
            };
        }
    }

    // Build a BLAS with OMM Attached.
    {
        nvrhi::rt::AccelStructDesc blasDesc = bvh::GetMeshBlasDesc(task.input.bvhCfg , *task.input.mesh, &ommAttachment);
        assert((int)blasDesc.bottomLevelGeometries.size() < (1 << 12)); // we can only hold 13 bits for the geometry index in the HitInfo - see GeometryInstanceID in SceneTypes.hlsli
        nvrhi::rt::AccelStructHandle as = m_device->createAccelStruct(blasDesc);
        nvrhi::utils::BuildBottomLevelAccelStruct(commandList, as, blasDesc);
        
        // Store results
        std::static_pointer_cast<MeshInfoEx>(task.input.mesh)->AccelStructOMM = as;
    }

    nvrhi::EventQueryHandle query = m_device->createEventQuery();
    m_device->setEventQuery(query, nvrhi::CommandQueue::Graphics);

    task.query = query;
    task.state = BuildState::BakeAndBuild;
}

void OmmBuildQueue::Finalize(nvrhi::CommandListHandle commandList, BuildTask& task)
{
    std::unique_ptr<MeshDebugData> debugData = std::make_unique<MeshDebugData>();
    debugData->ommArrayDataBuffer = task.buffers.ommArrayDataBuffer;
    debugData->ommDescBuffer = task.buffers.ommDescBuffer;
    debugData->ommIndexBuffer = task.buffers.ommIndexBuffer;

    debugData->ommArrayDataBufferDescriptor =
        std::make_shared<donut::engine::DescriptorHandle>(m_descriptorTable->CreateDescriptorHandle(nvrhi::BindingSetItem::RawBuffer_SRV(0, task.buffers.ommArrayDataBuffer)));
    debugData->ommDescBufferDescriptor =
        std::make_shared<donut::engine::DescriptorHandle>(m_descriptorTable->CreateDescriptorHandle(nvrhi::BindingSetItem::RawBuffer_SRV(0, task.buffers.ommDescBuffer)));
    debugData->ommIndexBufferDescriptor =
        std::make_shared<donut::engine::DescriptorHandle>(m_descriptorTable->CreateDescriptorHandle(nvrhi::BindingSetItem::RawBuffer_SRV(0, task.buffers.ommIndexBuffer)));

    auto mesh = std::static_pointer_cast<MeshInfoEx>(task.input.mesh);
    assert(!mesh->DebugData);
    mesh->DebugData = std::move(debugData);
    mesh->DebugDataDirty = true;

    void* pReadbackData = m_device->mapBuffer(task.buffers.ommReadbackBuffer, nvrhi::CpuAccessMode::Read);
    for (uint32_t i = 0; i < task.input.geometries.size(); ++i)
    {
        // Get debug info
        BufferInfo& bufferInfo = task.bufferInfos[i];
        omm::GpuBakeNvrhi::PostDispatchInfo postDispatchInfo;
        m_baker->ReadPostDispatchInfo((uint8_t*)pReadbackData + bufferInfo.ommPostDispatchInfoReadbackOffset, sizeof(postDispatchInfo), postDispatchInfo);

        const OmmBuildQueue::BuildInput::Geometry& geom = task.input.geometries[i];
        MeshGeometryEx* meshGeometry = static_cast<MeshGeometryEx*>(task.input.mesh->geometries[geom.geometryIndexInMesh].get());

        meshGeometry->DebugData.ommArrayDataOffset = (uint32_t)bufferInfo.ommArrayDataOffset;
        meshGeometry->DebugData.ommDescBufferOffset = (uint32_t)bufferInfo.ommDescArrayOffset;
        meshGeometry->DebugData.ommIndexBufferOffset = (uint32_t)bufferInfo.ommIndexOffset;
        meshGeometry->DebugData.ommIndexBufferFormat = bufferInfo.ommIndexFormat;
        meshGeometry->DebugData.ommStatsTotalKnown = postDispatchInfo.ommTotalOpaqueCount + postDispatchInfo.ommTotalTransparentCount;
        meshGeometry->DebugData.ommStatsTotalUnknown = postDispatchInfo.ommTotalUnknownCount;
    }
    m_device->unmapBuffer(task.buffers.ommReadbackBuffer);

    task.buffers.ommReadbackBuffer = nullptr;
    m_baker->Clear();
}

void OmmBuildQueue::Update(nvrhi::CommandListHandle commandList)
{
    for (auto it = m_pending.begin(); it != m_pending.end();)
    {
        BuildTask& task = *it;

        bool isPending = true;
        switch (task.state)
        {
        case BuildState::None:
        {
            RunSetup(commandList, task);
            break;
        }
        case BuildState::Setup:
        {
            if (m_device->pollEventQuery(task.query))
            {
                RunBakeAndBuild(commandList, task);
            }
            break;
        }
        case BuildState::BakeAndBuild:
        {
            if (m_device->pollEventQuery(task.query))
            {
                Finalize(commandList, task);
                isPending = false;
            }
            break;
        }
        default:
        {
            assert(false);
            break;
        }
        }

        if (!isPending)
            it = m_pending.erase(it);
        else
            ++it;

        break; // we do one operation per frame for now.
    }
}

void OmmBuildQueue::QueueBuild(const BuildInput& input)
{
    m_pending.push_back(input);
}

uint32_t OmmBuildQueue::NumPendingBuilds() const
{
    return (uint32_t)m_pending.size();
}

void OmmBuildQueue::CancelPendingBuilds()
{
    m_pending.clear();
}