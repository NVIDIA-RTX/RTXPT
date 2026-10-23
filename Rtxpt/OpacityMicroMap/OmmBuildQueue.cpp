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
#include "../SampleCommon/BlasCompactor.h"

#include <donut/core/math/math.h>
#include <donut/core/vfs/VFS.h>
#include <donut/engine/ShaderFactory.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <limits>
#include <sstream>
#include <nvrhi/common/misc.h>
#include <nvrhi/utils.h>
#include "../Materials/MaterialsBaker.h"
#include "../OpacityMicroMap/OmmBaker.h"
#include "OmmCpuInputIdentity.h"

#if DONUT_WITH_VULKAN
#ifndef VULKAN_HPP_DISPATCH_LOADER_DYNAMIC
#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1
#endif
#include <vulkan/vulkan.hpp>
#endif

namespace
{
    void CommitOmmWrites(nvrhi::IDevice* device, nvrhi::ICommandList& commandList)
    {
#if DONUT_WITH_VULKAN
        if (device->getGraphicsAPI() == nvrhi::GraphicsAPI::VULKAN)
        {
            // The pinned SDK resets outputs and private scratch buffers to Common.
            // Vulkan maps Common to TOP_OF_PIPE/no access, so a later NVRHI state
            // transition cannot recover the shader writes. Cover subsequent copies,
            // scratch reuse, micromap builds and BLAS reads with a real dependency.
            // This also covers hardware micromap writes before a consumer BLAS.
            commandList.clearState(); // Native barriers must be outside a render pass.
            commandList.commitBarriers();
            const vk::MemoryBarrier2 barrier = vk::MemoryBarrier2()
                .setSrcStageMask(vk::PipelineStageFlagBits2::eAllCommands)
                .setSrcAccessMask(vk::AccessFlagBits2::eMemoryWrite)
                .setDstStageMask(vk::PipelineStageFlagBits2::eAllCommands)
                .setDstAccessMask(vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite);
            const vk::DependencyInfo dependency = vk::DependencyInfo().setMemoryBarriers(barrier);
            const VkCommandBuffer nativeCommandBuffer = commandList.getNativeObject(nvrhi::ObjectTypes::VK_CommandBuffer);
            vk::CommandBuffer(nativeCommandBuffer).pipelineBarrier2(dependency);
        }
#else
        (void)device;
        (void)commandList;
#endif
    }

    static omm::GpuBakeNvrhi::Input GetBakeInput(omm::GpuBakeNvrhi::Operation op, const donut::engine::MeshInfo& mesh, const OmmBuildQueue::BuildInput::Geometry& geometry)
    {
        using namespace donut;
        using namespace donut::math;

        const donut::engine::MeshGeometry* meshGeometry = mesh.geometries[geometry.geometryIndexInMesh].get();

        const uint32_t indexOffset = mesh.indexOffset + meshGeometry->indexOffsetInMesh;
        const uint32_t vertexOffset = (mesh.vertexOffset + meshGeometry->vertexOffsetInMesh);

        PTMaterial & materialPT = *PTMaterial::SafeCast(meshGeometry->material);

        omm::GpuBakeNvrhi::Input params;
        params.operation = op;
        params.alphaTexture = materialPT.BaseTexture.Loaded->texture;
        const float runtimeAlphaCutoff = SubInstanceData::DecodeAlphaCutoff(SubInstanceData::QuantizeAlphaCutoff(materialPT.AlphaCutoff));
        // The SDK uses > while traversal uses >=. The predecessor preserves equality;
        // zero uses a normal negative value so GPU denormal flushing cannot erase it.
        params.alphaCutoff = runtimeAlphaCutoff == 0.0f ? -std::numeric_limits<float>::min()
            : std::nextafter(runtimeAlphaCutoff, -std::numeric_limits<float>::infinity());
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

    bool OmmDiagnosticsEnabled()
    {
        static const bool enabled = [] {
            const char* value = std::getenv("RTXPT_OMM_DIAGNOSTICS");
            return value != nullptr && value[0] == '1' && value[1] == '\0';
        }();
        return enabled;
    }

    void LogOmmDiagnostic(const std::string& message)
    {
        donut::log::info("[OMM diagnostics] %s", message.c_str());
        // Mirror debugger output so redirected noninteractive runs can collect the diagnostics.
        std::fprintf(stdout, "[OMM diagnostics] %s\n", message.c_str());
        std::fflush(stdout);
    }

    void AppendOmmDiagnosticBuffer(std::ostream& message, nvrhi::IDevice* device,
        const char* name, nvrhi::IBuffer* buffer)
    {
        uint64_t allocatedBytes = 0;
        bool known = buffer == nullptr;
        if (buffer != nullptr && device->getGraphicsAPI() != nvrhi::GraphicsAPI::D3D11)
        {
            allocatedBytes = device->getBufferMemoryRequirements(buffer).size;
            known = allocatedBytes != 0;
        }
        message << ' ' << name << "_capacity_bytes=" << (buffer != nullptr ? buffer->getDesc().byteSize : 0)
            << ' ' << name << "_known_allocated_bytes=" << allocatedBytes
            << ' ' << name << "_unknown_allocations=" << (known ? 0 : 1);
    }

    void LogOmmBuildDiagnostic(nvrhi::IDevice* device, const OmmBuildQueue::BuildInput& input, const std::vector<omm::GpuBakeNvrhi::Input>& bakeInputs, const MeshInfoEx& mesh,
        nvrhi::IBuffer* bakeArray, nvrhi::IBuffer* bakeDesc, nvrhi::IBuffer* bakeIndex, bool sharedBake = false)
    {
        if (!OmmDiagnosticsEnabled())
            return;

        // The public NVRHI API exposes buffer requirements, but no OMM allocation size.
        const uint64_t hardwareAllocatedBytes = 0;
        const size_t hardwareUnknownAllocations = mesh.OpacityMicroMaps.size();

        uint64_t indexCount = 0;
        for (const auto& geometry : input.geometries)
            indexCount += mesh.geometries[geometry.geometryIndexInMesh]->numIndices;

        std::ostringstream message;
        message << "finalized mesh=" << std::quoted(mesh.name)
            << " mesh_geometries=" << mesh.geometries.size()
            << " baked_geometries=" << input.geometries.size()
            << " baked_indices=" << indexCount
            << " shared_bake=" << sharedBake
            << " hardware_count=" << mesh.OpacityMicroMaps.size()
            << " hardware_known_allocated_bytes=" << hardwareAllocatedBytes
            << " hardware_unknown_allocations=" << hardwareUnknownAllocations;
        message << " debug_data_retained=" << (mesh.DebugData != nullptr);
        AppendOmmDiagnosticBuffer(message, device, "raw_array", mesh.DebugData ? mesh.DebugData->ommArrayDataBuffer.Get() : nullptr);
        AppendOmmDiagnosticBuffer(message, device, "raw_desc", mesh.DebugData ? mesh.DebugData->ommDescBuffer.Get() : nullptr);
        AppendOmmDiagnosticBuffer(message, device, "raw_index", mesh.DebugData ? mesh.DebugData->ommIndexBuffer.Get() : nullptr);
        AppendOmmDiagnosticBuffer(message, device, "bake_array", bakeArray);
        AppendOmmDiagnosticBuffer(message, device, "bake_desc", bakeDesc);
        AppendOmmDiagnosticBuffer(message, device, "bake_index", bakeIndex);
        LogOmmDiagnostic(message.str());

        for (size_t i = 0; i < input.geometries.size(); ++i)
        {
            const auto& geometry = input.geometries[i];
            const auto& bake = bakeInputs[i];
            std::ostringstream geometryMessage;
            geometryMessage << std::setprecision(std::numeric_limits<float>::max_digits10);
            geometryMessage << "geometry mesh=" << std::quoted(mesh.name)
                << " geometry_index=" << geometry.geometryIndexInMesh
                << " indices=" << bake.numIndices
                << " index_offset_bytes=" << bake.indexBufferOffsetInBytes
                << " uv_offset_bytes=" << bake.texCoordBufferOffsetInBytes
                << " alpha_texture=" << std::quoted(bake.alphaTexture->getDesc().debugName)
                << " alpha_cutoff=" << bake.alphaCutoff
                << " alpha_gt=" << int(bake.alphaCutoffGreater)
                << " alpha_le=" << int(bake.alphaCutoffLessEqual)
                << " max_subdivision=" << bake.maxSubdivisionLevel
                << " dynamic_scale=" << bake.dynamicSubdivisionScale
                << " format=" << int(bake.format)
                << " max_array_bytes=" << bake.maxOutOmmArraySize;
            LogOmmDiagnostic(geometryMessage.str());
        }
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
            desc.initialState = nvrhi::ResourceStates::Common;
            desc.keepInitialState = true;
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
    nvrhi::DeviceHandle& device, 
    std::shared_ptr<donut::engine::DescriptorTableManager> descriptorTable, 
    std::shared_ptr<donut::engine::ShaderFactory> shaderFactory)
    : m_device(device)
    , m_descriptorTable(std::move(descriptorTable))
    , m_shaderFactory(std::move(shaderFactory))
{
}

void OmmBuildQueue::CreateBaker(nvrhi::ICommandList& commandList)
{
    omm::GpuBakeNvrhi::ShaderProvider provider;

    // These shaders are compiled by donut_compile_shaders with NVRHI's default
    // register shifts, rather than the SDK's built-in shader register shifts.
    provider.bindingOffsets = nvrhi::VulkanBindingOffsets {};

    provider.shaders = [this](nvrhi::ShaderType type, const char* shaderName, const char* shaderEntryName)->nvrhi::ShaderHandle
    {
        std::vector<donut::engine::ShaderMacro> defines = { donut::engine::ShaderMacro("COMPILER_DXC", "1") };
        std::string shaderNameStr = std::string("omm/Omm/libraries/omm-lib/shaders/") + shaderName;
        return m_shaderFactory->CreateShader(shaderNameStr.c_str(), shaderEntryName, &defines, type);
    };

    omm::GpuBakeNvrhi::MessageCallback messageCb = [](omm::MessageSeverity severity, const char* message) {
        donut::log::info("[OMM SDK]: %d %s", severity, message);
    };

    // Record SDK uploads before bake dispatches on the caller's command list.
    m_baker = std::make_unique<omm::GpuBakeNvrhi>(m_device, &commandList, false /*debug*/, &provider, messageCb);
}

OmmBuildQueue::~OmmBuildQueue()
{
}

bool OmmBuildQueue::MakeInputKey(const BuildTask& task, omm::BakeInputKey& key) const
{
    const auto& mesh = static_cast<const MeshInfoEx&>(*task.input.mesh);
    const auto& identity = mesh.OmmCpuInputIdentity;
    if (!identity || !identity->IsValidFor(mesh))
        return false;
    key.reserve(2 + task.bakeInputs.size() * 26);
    key = { uint32_t(mesh.geometries.size()), uint32_t(task.input.geometries.size()) };
    for (size_t i = 0; i < task.bakeInputs.size(); ++i)
    {
        const auto slot = task.input.geometries[i].geometryIndexInMesh;
        const uint64_t id = identity->GetGeometryId(slot);
        if (!id) return false;
        const auto& geometry = identity->geometries[slot];
        const auto& input = task.bakeInputs[i];
        const uint64_t indexOffset = (uint64_t(identity->indexOffset) + geometry.indexOffset) * sizeof(uint32_t);
        const uint64_t uvOffset = identity->texCoordRange.byteOffset
            + (uint64_t(identity->vertexOffset) + geometry.vertexOffset) * sizeof(donut::math::float2);
        // Match the captured SDK stream too, including offsets that its 32-bit fields
        // cannot represent. Unsupported or replaced inputs simply bake independently.
        if (input.indexBuffer != identity->indexBuffer || input.texCoordBuffer != identity->vertexBuffer
            || input.indexBufferOffsetInBytes != indexOffset || input.texCoordBufferOffsetInBytes != uvOffset
            || input.numIndices != geometry.numIndices || input.texCoordStrideInBytes != sizeof(donut::math::float2)
            || input.texCoordFormat != nvrhi::Format::R32_FLOAT)
            return false;
        key.push_back(slot);
        omm::AppendBakeInputKey(key, input, task.input.geometries[i].flags, id);
    }
    return true;
}

void OmmBuildQueue::FindSharedBake(BuildTask& task)
{
    omm::BakeInputKey key;
    const bool valid = MakeInputKey(task, key);
    if (valid)
    {
        auto [entry, inserted] = m_sharedBakes.try_emplace(std::move(key));
        if (inserted)
        {
            entry->second = std::make_shared<SharedBake>();
            for (const auto& input : task.bakeInputs)
                entry->second->alphaTextures.push_back(input.alphaTexture);
        }
        task.sharedBake = entry->second;
        if (!inserted)
        {
            task.state = BuildState::WaitingForSharedBake;
            return;
        }
    }
}

void OmmBuildQueue::RunSharedBuild(nvrhi::ICommandList& commandList, BuildTask& task)
{
    auto& mesh = static_cast<MeshInfoEx&>(*task.input.mesh);
    for (const auto& attachment : task.sharedBake->attachments)
        if (attachment.ommBuffer)
            mesh.OpacityMicroMaps.push_back(attachment.ommBuffer);
    BuildBLASWithOMM(commandList, task, task.sharedBake->attachments);
    task.state = BuildState::SharedBuild;
}

void OmmBuildQueue::FinalizeSharedBuild(BuildTask& task)
{
    auto& mesh = static_cast<MeshInfoEx&>(*task.input.mesh);
    if (m_retainDebugData)
        mesh.DebugData = task.sharedBake->debugData;
    for (size_t i = 0; i < mesh.geometries.size(); ++i)
        static_cast<MeshGeometryEx&>(*mesh.geometries[i]).DebugData = task.sharedBake->geometryDebugData[i];
    mesh.MarkDebugDataDirty();
    LogOmmBuildDiagnostic(m_device, task.input, task.bakeInputs, mesh, nullptr, nullptr, nullptr, true);
}

void OmmBuildQueue::SetRetainDebugData(bool retain)
{
    m_retainDebugData = retain;
    if (!retain)
        for (auto& entry : m_sharedBakes)
            entry.second->debugData.reset();
}

void OmmBuildQueue::RunSetup(nvrhi::ICommandList& commandList, BuildTask& task)
{
    assert(task.state == BuildState::None);

    LinearBufferAllocator ommIndexBufferAllocator;
    LinearBufferAllocator ommDescBufferAllocator;
    LinearBufferAllocator ommDescArrayHistogramBufferAllocator;
    LinearBufferAllocator ommIndexArrayHistogramBufferAllocator;
    LinearBufferAllocator ommPostBuildInfoBufferAllocator;
    LinearBufferAllocator ommReadbackBufferAllocator;

    std::vector<BufferInfo>& bufferInfos = task.bufferInfos;

    for (const auto& input : task.bakeInputs)
    {
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

    Buffers& buffers = task.buffers;
    buffers.ommIndexBuffer                  = ommIndexBufferAllocator.CreateBuffer(m_device, "OmmIndexBuffer", BufferConfig::RawUAVAndASBuildInput);
    buffers.ommDescBuffer                   = ommDescBufferAllocator.CreateBuffer(m_device, "OmmDescBuffer", BufferConfig::RawUAVAndASBuildInput);
    buffers.ommDescArrayHistogramBuffer     = ommDescArrayHistogramBufferAllocator.CreateBuffer(m_device, "OmmDescArrayHistogramBuffer", BufferConfig::RawUAV);
    buffers.ommIndexArrayHistogramBuffer    = ommIndexArrayHistogramBufferAllocator.CreateBuffer(m_device, "OmmIndexArrayHistogramBuffer", BufferConfig::RawUAV);
    buffers.ommPostDispatchInfoBuffer       = ommPostBuildInfoBufferAllocator.CreateBuffer(m_device, "OmmPostBuildInfoBuffer", BufferConfig::RawUAV);
    buffers.ommReadbackBuffer               = ommReadbackBufferAllocator.CreateBuffer(m_device, "OmmGenericReadbackBuffer", BufferConfig::Readback);

    commandList.beginTrackingBufferState(buffers.ommIndexBuffer, nvrhi::ResourceStates::Common);
    commandList.beginTrackingBufferState(buffers.ommDescBuffer, nvrhi::ResourceStates::Common);
    commandList.beginTrackingBufferState(buffers.ommDescArrayHistogramBuffer, nvrhi::ResourceStates::Common);
    commandList.beginTrackingBufferState(buffers.ommIndexArrayHistogramBuffer, nvrhi::ResourceStates::Common);
    commandList.beginTrackingBufferState(buffers.ommPostDispatchInfoBuffer, nvrhi::ResourceStates::Common);
    commandList.beginTrackingBufferState(buffers.ommReadbackBuffer, nvrhi::ResourceStates::Common);

    // Dispatch all setup tasks.
    for (uint32_t i = 0; i < task.input.geometries.size(); ++i)
    {
        const BufferInfo& bufferInfo = bufferInfos[i];
        const auto& input = task.bakeInputs[i];

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

        m_baker->Dispatch(&commandList, input, output);
        CommitOmmWrites(m_device, commandList);

        omm::GpuBakeNvrhi::PreDispatchInfo setupInfo;
        m_baker->GetPreDispatchInfo(input, setupInfo);

        commandList.copyBuffer(buffers.ommReadbackBuffer, bufferInfo.ommDescArrayHistogramReadbackOffset,  buffers.ommDescArrayHistogramBuffer,  bufferInfo.ommDescArrayHistogramOffset,      setupInfo.ommDescArrayHistogramSize);
        commandList.copyBuffer(buffers.ommReadbackBuffer, bufferInfo.ommIndexHistogramReadbackOffset,  buffers.ommIndexArrayHistogramBuffer, bufferInfo.ommIndexHistogramOffset, setupInfo.ommIndexHistogramSize);
        commandList.copyBuffer(buffers.ommReadbackBuffer, bufferInfo.ommPostDispatchInfoReadbackOffset,   buffers.ommPostDispatchInfoBuffer,       bufferInfo.ommPostDispatchInfoOffset,  setupInfo.ommPostDispatchInfoBufferSize);
    }

    task.state = BuildState::Setup;
}

void OmmBuildQueue::AllocateOMMArrayDataBuffer(BuildTask& task)
{
    LinearBufferAllocator ommArrayDataBufferAllocator;
    
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

void OmmBuildQueue::BakeOmmArrayData(nvrhi::ICommandList& commandList, BuildTask& task)
{
    commandList.beginTrackingBufferState(task.buffers.ommIndexBuffer, nvrhi::ResourceStates::Common);
    commandList.beginTrackingBufferState(task.buffers.ommDescBuffer, nvrhi::ResourceStates::Common);
    commandList.beginTrackingBufferState(task.buffers.ommDescArrayHistogramBuffer, nvrhi::ResourceStates::Common);
    commandList.beginTrackingBufferState(task.buffers.ommIndexArrayHistogramBuffer, nvrhi::ResourceStates::Common);
    commandList.beginTrackingBufferState(task.buffers.ommPostDispatchInfoBuffer, nvrhi::ResourceStates::Common);
    commandList.beginTrackingBufferState(task.buffers.ommArrayDataBuffer, nvrhi::ResourceStates::Common);

    commandList.clearBufferUInt(task.buffers.ommArrayDataBuffer, 0);

    for (uint32_t i = 0; i < task.input.geometries.size(); ++i)
    {
        const BufferInfo& bufferInfo = task.bufferInfos[i];
        auto input = task.bakeInputs[i];
        input.operation = omm::GpuBakeNvrhi::Operation::Bake;

        omm::GpuBakeNvrhi::Buffers output;
        output.ommArrayBuffer = task.buffers.ommArrayDataBuffer;
        output.ommArrayBufferOffset = (uint32_t)bufferInfo.ommArrayDataOffset;
        output.ommDescBuffer = task.buffers.ommDescBuffer;
        output.ommDescBufferOffset = (uint32_t)bufferInfo.ommDescArrayOffset;
        output.ommIndexBuffer = task.buffers.ommIndexBuffer;
        output.ommIndexBufferOffset = (uint32_t)bufferInfo.ommIndexOffset;
        output.ommDescArrayHistogramBuffer = task.buffers.ommDescArrayHistogramBuffer;
        output.ommDescArrayHistogramBufferOffset = (uint32_t)bufferInfo.ommDescArrayHistogramOffset;
        output.ommIndexHistogramBuffer = task.buffers.ommIndexArrayHistogramBuffer;
        output.ommIndexHistogramBufferOffset = (uint32_t)bufferInfo.ommIndexHistogramOffset;
        output.ommPostDispatchInfoBuffer = task.buffers.ommPostDispatchInfoBuffer;
        output.ommPostDispatchInfoBufferOffset = (uint32_t)bufferInfo.ommPostDispatchInfoOffset;

        m_baker->Dispatch(&commandList, input, output);
        CommitOmmWrites(m_device, commandList);

        omm::GpuBakeNvrhi::PreDispatchInfo setupInfo;
        m_baker->GetPreDispatchInfo(input, setupInfo);

        commandList.copyBuffer(task.buffers.ommReadbackBuffer, bufferInfo.ommPostDispatchInfoReadbackOffset, task.buffers.ommPostDispatchInfoBuffer, bufferInfo.ommPostDispatchInfoOffset, setupInfo.ommPostDispatchInfoBufferSize);
    }
}

std::vector<bvh::OmmAttachment> OmmBuildQueue::BuildOMMAttachments(nvrhi::ICommandList& commandList, BuildTask& task)
{
    std::vector<bvh::OmmAttachment> ommAttachment;
    ommAttachment.resize(task.input.mesh->geometries.size());

    // Build the OMM Arrays
    for (uint32_t i = 0; i < task.input.geometries.size(); ++i)
    {
        const OmmBuildQueue::BuildInput::Geometry& geom = task.input.geometries[i];

        const BufferInfo& bufferInfo = task.bufferInfos[i];

        nvrhi::rt::OpacityMicromapDesc desc = GetOpacityMicromapDesc(
            task.buffers.ommArrayDataBuffer, bufferInfo.ommArrayDataOffset,
            task.buffers.ommDescBuffer, bufferInfo.ommDescArrayOffset,
            bufferInfo.ommArrayHistogram, geom.flags);

        nvrhi::rt::OpacityMicromapHandle ommBuffer = m_device->createOpacityMicromap(desc);

        std::static_pointer_cast<MeshInfoEx>(task.input.mesh)->OpacityMicroMaps.push_back(ommBuffer);

        commandList.buildOpacityMicromap(ommBuffer, desc);

        ommAttachment[geom.geometryIndexInMesh] =
        {
            .ommBuffer = ommBuffer,
            .ommIndexFormat = bufferInfo.ommIndexFormat,
            .ommIndexHistogram = bufferInfo.ommIndexHistogram,
            .ommIndexBuffer = task.buffers.ommIndexBuffer,
            .ommIndexBufferOffset = (uint32_t)bufferInfo.ommIndexOffset,
            .ommArrayDataBuffer = task.buffers.ommArrayDataBuffer,
            .ommArrayDataBufferOffset = (uint32_t)bufferInfo.ommArrayDataOffset
        };
    }

    return ommAttachment;
}

void OmmBuildQueue::BuildBLASWithOMM(nvrhi::ICommandList& commandList, BuildTask& task, const std::vector<bvh::OmmAttachment>& ommAttachment)
{
    // Explicitly declare the handoff for fresh and cached triangle-to-micromap
    // indices: the pinned Vulkan backend does not track this BLAS input itself.
    for (const auto& attachment : ommAttachment)
        if (attachment.ommIndexBuffer)
            commandList.setBufferState(attachment.ommIndexBuffer, nvrhi::ResourceStates::AccelStructBuildInput);
    commandList.commitBarriers();
    CommitOmmWrites(m_device, commandList);

    nvrhi::rt::AccelStructDesc blasDesc = bvh::GetMeshBlasDesc(task.input.bvhCfg , *task.input.mesh, ommAttachment.data(), false);
    // The old 8-bit limit here assumed instanceID was packed as (instanceIndex << 8) | geometryIndex. That scheme is gone:
    // BuildTLAS now sets instanceID to the flat GetGeometryInstanceIndex(), and the shaders index with 'instanceID +
    // geometryIndex' (DonutBridgeCommon.hlsli AlphaTest) using DXR 1.1's GeometryIndex() - exactly what the old comment
    // suggested doing. The real per-BLAS geometry limit is RTXDI's 12-bit field, same as the non-OMM path in Sample.cpp.
    assert((int)blasDesc.bottomLevelGeometries.size() < (1 << 12)); // RTXDI instanceAndGeometryIndex: low 12 bits are geometryIndex
    nvrhi::rt::AccelStructHandle as;
    if (m_blasCompactor != nullptr && m_blasCompactor->IsAvailable())
    {
        // One mesh per frame arrives here, so this is a batch of one. BuildAll submits and waits twice
        // internally, which is the same close/execute/reopen SubmitAndSubscribeQuery already does around
        // this work - the queue's own event-query gate is unaffected.
        const std::vector<nvrhi::rt::AccelStructDesc> descs = { blasDesc };
        as = m_blasCompactor->BuildAll(&commandList, descs)[0];
    }
    if (as == nullptr)
    {
        // No pool, or pooling failed for this mesh; an ordinary acceleration structure still renders.
        as = m_device->createAccelStruct(blasDesc);
        nvrhi::utils::BuildBottomLevelAccelStruct(&commandList, as, blasDesc);
    }

    // Store results
    std::static_pointer_cast<MeshInfoEx>(task.input.mesh)->AccelStructOMM = as;
}

void OmmBuildQueue::RunBakeAndBuild(nvrhi::ICommandList& commandList, BuildTask& task)
{
    assert(task.state == BuildState::None || task.state == BuildState::Setup);

    AllocateOMMArrayDataBuffer(task);

    BakeOmmArrayData(commandList, task); // Dispatch the bake which will fill the ommArrayData

    std::vector<bvh::OmmAttachment> ommAttachment = BuildOMMAttachments(commandList, task);

    BuildBLASWithOMM(commandList, task, ommAttachment); // Build a BLAS with OMM Attached.

    if (task.sharedBake)
    {
        task.sharedBake->attachments = std::move(ommAttachment);
        for (auto& attachment : task.sharedBake->attachments)
            attachment.ommArrayDataBuffer = nullptr;
    }

    task.state = BuildState::BakeAndBuild;
}

void OmmBuildQueue::SubmitAndSubscribeQuery(nvrhi::ICommandList& commandList)
{
    // Need to submit the command list for the event query to be relevant. Ideally, we would have the query point to the next fence value instead of the last submitted one, or we would defer query set up to happen on command list submission.
    commandList.close();
    m_device->executeCommandList(&commandList, nvrhi::CommandQueue::Graphics);
    commandList.open();

    // Subscribe to this submission
    m_device->setEventQuery(m_InFlightQuery, nvrhi::CommandQueue::Graphics);
}

void OmmBuildQueue::Finalize(nvrhi::ICommandList& commandList, BuildTask& task)
{
    auto mesh = std::static_pointer_cast<MeshInfoEx>(task.input.mesh);
    assert(!mesh->DebugData);
    // The completion query has retired the builds. Raw array/descriptor buffers are
    // debug-only; the batch cache keeps triangle indices for later consumer BLASes.
    if (m_retainDebugData)
    {
        auto debugData = std::make_shared<MeshDebugData>();
        debugData->ommArrayDataBuffer = task.buffers.ommArrayDataBuffer;
        debugData->ommDescBuffer = task.buffers.ommDescBuffer;
        debugData->ommIndexBuffer = task.buffers.ommIndexBuffer;

        debugData->ommArrayDataBufferDescriptor =
            std::make_shared<donut::engine::DescriptorHandle>(m_descriptorTable->CreateDescriptorHandle(nvrhi::BindingSetItem::RawBuffer_SRV(0, task.buffers.ommArrayDataBuffer)));
        debugData->ommDescBufferDescriptor =
            std::make_shared<donut::engine::DescriptorHandle>(m_descriptorTable->CreateDescriptorHandle(nvrhi::BindingSetItem::RawBuffer_SRV(0, task.buffers.ommDescBuffer)));
        debugData->ommIndexBufferDescriptor =
            std::make_shared<donut::engine::DescriptorHandle>(m_descriptorTable->CreateDescriptorHandle(nvrhi::BindingSetItem::RawBuffer_SRV(0, task.buffers.ommIndexBuffer)));

        mesh->DebugData = std::move(debugData);
    }
    mesh->MarkDebugDataDirty();

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

    if (task.sharedBake)
    {
        task.sharedBake->debugData = mesh->DebugData;
        for (const auto& geometry : mesh->geometries)
            task.sharedBake->geometryDebugData.push_back(static_cast<const MeshGeometryEx&>(*geometry).DebugData);
        task.sharedBake->ready = true;
    }

    LogOmmBuildDiagnostic(m_device, task.input, task.bakeInputs, *mesh,
        task.buffers.ommArrayDataBuffer, task.buffers.ommDescBuffer, task.buffers.ommIndexBuffer);

    task.buffers.ommReadbackBuffer = nullptr;
    m_baker->Clear();
}

void OmmBuildQueue::BuildTask::Reset()
{
    bufferInfos.clear();
}

void OmmBuildQueue::ConsumeOneTask(nvrhi::ICommandList& commandList, BuildState taskState)
{
    for (auto& task : m_pending)
    {
        if (task.state == taskState && (taskState != BuildState::WaitingForSharedBake || task.sharedBake->ready))
        {
            bool finished = ExecuteTask(commandList, task);
            if (finished)
            {
                // Remove the task from the queue
                task.Reset(); // No need to swap the whole BufferInfo vector just to pop it
                std::swap(m_pending.back(), task);
                m_pending.pop_back();
            }

            return;
        }
    }
}

bool OmmBuildQueue::ExecuteTask(nvrhi::ICommandList& commandList, BuildTask& task)
{
    switch (task.state)
    {
    case BuildState::None:
    {
        RunSetup(commandList, task);
        break;
    }
    case BuildState::WaitingForSharedBake:
    {
        RunSharedBuild(commandList, task);
        break;
    }
    case BuildState::SharedBuild:
    {
        FinalizeSharedBuild(task);
        return true;
    }
    case BuildState::Setup:
    {
        RunBakeAndBuild(commandList, task);
        break;
    }
    case BuildState::BakeAndBuild:
    {
        Finalize(commandList, task);
        return true;
    }
    default:
    {
        assert(false);
        break;
    }
    }

    return false;
}

bool OmmBuildQueue::ReadyToRecordWork()
{
    if (!m_InFlightQuery)
    {
        // Create a query to check for command list completion
        m_InFlightQuery = m_device->createEventQuery();
        return true;
    }

    if (m_device->pollEventQuery(m_InFlightQuery))
    {
        return true;
    }

    return false;
}

void OmmBuildQueue::Update(nvrhi::ICommandList& commandList)
{
    if (ReadyToRecordWork() && !m_pending.empty())
    {
        if (m_baker == nullptr)
            CreateBaker(commandList);

        // Work backwards from the final state to avoid consuming consecutive states of the same task in a single frame.
        ConsumeOneTask(commandList, BuildState::SharedBuild);
        ConsumeOneTask(commandList, BuildState::BakeAndBuild);
        ConsumeOneTask(commandList, BuildState::WaitingForSharedBake);
        ConsumeOneTask(commandList, BuildState::Setup);
        ConsumeOneTask(commandList, BuildState::None);

        SubmitAndSubscribeQuery(commandList);

        // Last task finalized. Anything already submitted keeps the baker's resources alive until the GPU retires it.
        if (m_pending.empty())
        {
            m_baker = nullptr;
            m_sharedBakes.clear();
            if (OmmDiagnosticsEnabled())
                LogOmmDiagnostic("queue_drained baker_released=1");
        }
    }
}

void OmmBuildQueue::QueueBuild(const BuildInput& input)
{
    BuildTask task(input);
    for (const auto& geometry : input.geometries)
        task.bakeInputs.push_back(GetBakeInput(omm::GpuBakeNvrhi::Operation::Setup, *input.mesh, geometry));
    FindSharedBake(task);
    m_pending.push_back(std::move(task));
}

uint32_t OmmBuildQueue::NumPendingBuilds() const
{
    return (uint32_t)m_pending.size();
}

void OmmBuildQueue::CancelPendingBuilds()
{
    m_pending.clear();
    m_sharedBakes.clear();
    // Nothing left to bake, so release the baker: that drops its scratch pool and its binding cache, whose entries would
    // otherwise keep the dropped tasks' buffers and the mesh IB/VB alive. Update recreates it when work is queued.
    m_baker = nullptr;
}
