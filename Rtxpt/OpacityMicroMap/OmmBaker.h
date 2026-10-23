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

#include <filesystem>
#include <memory>

#include <nvrhi/nvrhi.h>
#include <donut/core/math/math.h>
#include <donut\engine\SceneTypes.h>

#include "../SampleCommon/ComputePass.h"

#include "../Shaders/Misc/OmmGeometryDebugData.hlsli"

namespace bvh { class BlasCompactor; }
namespace omm { struct CpuMeshInputIdentity; }

namespace donut::engine
{
    class Scene;
    class FramebufferFactory;
    class TextureCache;
    class TextureHandle;
    class ShaderFactory;
    class CommonRenderPasses;
    struct TextureData;
    struct LoadedTexture;
}

struct MeshGeometryDebugData
{
    uint32_t ommArrayDataOffset = 0xFFFFFFFF; // for use by applications
    uint32_t ommDescBufferOffset = 0xFFFFFFFF; // for use by applications
    uint32_t ommIndexBufferOffset = 0xFFFFFFFF; // for use by applications
    nvrhi::Format ommIndexBufferFormat = nvrhi::Format::R32_UINT; // for use by applications
    uint64_t ommStatsTotalKnown = 0;
    uint64_t ommStatsTotalUnknown = 0;
};

struct MeshDebugData
{
    std::shared_ptr<donut::engine::DescriptorHandle> ommArrayDataBufferDescriptor;
    std::shared_ptr<donut::engine::DescriptorHandle> ommDescBufferDescriptor;
    std::shared_ptr<donut::engine::DescriptorHandle> ommIndexBufferDescriptor;
    nvrhi::BufferHandle ommArrayDataBuffer; // for use by applications
    nvrhi::BufferHandle ommDescBuffer; // for use by applications
    nvrhi::BufferHandle ommIndexBuffer; // for use by applications
};

struct MeshGeometryEx : public donut::engine::MeshGeometry
{
    // (Debug) OMM buffers.
    MeshGeometryDebugData DebugData;

    virtual ~MeshGeometryEx() = default;
};

struct MeshInfoEx : public donut::engine::MeshInfo
{
    std::shared_ptr<omm::CpuMeshInputIdentity> OmmCpuInputIdentity;
    // Static UV/index buffers are immutable after upload. Before editing a shared
    // buffer, cancel pending OMM builds, invalidate every affected mesh, then rebuild.
    void InvalidateOmmCpuInputIdentity() { OmmCpuInputIdentity.reset(); }

    nvrhi::rt::AccelStructHandle AccelStructOMM; // for use by application
    std::vector<nvrhi::rt::OpacityMicromapHandle> OpacityMicroMaps; // for use by application

    std::shared_ptr<MeshDebugData> DebugData;

    // Marking debug data dirty MUST go through MarkDebugDataDirty(): it raises both the per-mesh flag and the shared
    // 'anything dirty' signal that lets OmmBaker::Update() skip scanning every mesh in the scene. The per-mesh flag is
    // private precisely so the two cannot drift - setting it directly would silently reintroduce the full-scene poll,
    // or worse, leave a mesh dirty that Update() never revisits.
    void MarkDebugDataDirty()                   { m_debugDataDirty = true; s_anyDebugDataDirty = true; }
    bool IsDebugDataDirty() const               { return m_debugDataDirty; }
    void ClearDebugDataDirty()                  { m_debugDataDirty = false; }

    // Shared across all meshes; there is a single OmmBaker instance, so a static is equivalent to a member here without
    // needing a back-pointer from OmmBuildQueue (which has no OmmBaker reference).
    static bool AnyDebugDataDirty()             { return s_anyDebugDataDirty; }
    static void ClearAnyDebugDataDirty()        { s_anyDebugDataDirty = false; }

    // A newly constructed mesh starts dirty, so it must also raise the shared flag - otherwise a mesh created after a
    // scan would never be picked up. A user-provided ctor also skips MeshInfo's zero-init, and GltfImporter only '|='s
    // into objectSpaceBounds, so it must start empty here or the bounds absorb heap garbage.
    MeshInfoEx()                                { objectSpaceBounds = dm::box3::empty(); s_anyDebugDataDirty = true; }
    virtual ~MeshInfoEx() = default;

private:
    bool m_debugDataDirty = true;   // set via MarkDebugDataDirty() only
    static bool s_anyDebugDataDirty;
};

enum class OpacityMicroMapDebugView : uint32_t
{
    Disabled,
    InWorld,
    Overlay,
};

struct OpacityMicroMapUIData
{
    struct BuildState
    {
        // ~~ Application is expected to tweak these settings ~~ 
        int MaxSubdivisionLevel = 6;
        bool EnableDynamicSubdivision = true;
        float DynamicSubdivisionScale = 2.f;
        nvrhi::rt::OpacityMicromapBuildFlags Flag = nvrhi::rt::OpacityMicromapBuildFlags::FastTrace;
        nvrhi::rt::OpacityMicromapFormat Format = nvrhi::rt::OpacityMicromapFormat::OC1_4_State;
        int AlphaCutoffGT = 1 /*Opaque*/;
        int AlphaCutoffLE = 0 /*Transparent*/;

        // ~~ Debug settings, application is expected to leave to default ~~ 
        bool ComputeOnly = true;
        bool LevelLineIntersection = true;
        bool EnableTexCoordDeduplication = true;
        bool Force32BitIndices = false;
        bool EnableNsightDebugMode = false;
        bool EnableSpecialIndices = true;
        int MaxOmmArrayDataSizeInMB = 100;

        bool operator == (const BuildState& other) const = default;
    };

    bool                                Enable = true;
    bool                                Force2State = false;
    bool                                OnlyOMMs = false;

    // Amortize the builds over multiple frames
    std::optional<BuildState>           ActiveState;
    BuildState                          DesiredState;
    bool                                TriggerRebuild = true;

    // --- Stats --- 
    // Build progress counts meshes, matching OmmBuildQueue tasks.
    uint32_t                            BuildsLeftInQueue = 0;
    uint32_t                            BuildsQueued = 0;

    OpacityMicroMapDebugView            DebugView = OpacityMicroMapDebugView::Disabled;
};

class OmmBaker
{
public:
    OmmBaker(nvrhi::DeviceHandle device,
        std::shared_ptr<donut::engine::DescriptorTableManager> descriptorTableManager,
        std::shared_ptr<donut::engine::TextureCache> textureCache,
        std::shared_ptr<donut::engine::ShaderFactory> shaderFactory);
    ~OmmBaker();

    void                            CreateRenderPasses(nvrhi::BindingLayoutHandle bindlessLayout, std::shared_ptr<donut::engine::CommonRenderPasses> commonPasses);

    bool                            Update(nvrhi::ICommandList& commandList, const donut::engine::Scene& scene);

    OpacityMicroMapUIData &         UIData()    { return m_uiData; }
    // BuildOpacityMicromaps would flush the command list to the graphics queue
    bool                            NeedsGraphicsQueue() const;
    bool                            DebugGUI(float indent, const donut::engine::Scene& scene);

    void                            SceneLoaded(const donut::engine::Scene& scene);
    void                            SceneUnloading();

    void                            CreateOpacityMicromaps(const donut::engine::Scene& scene);
    void                            DestroyOpacityMicromaps(nvrhi::ICommandList& commandList, const donut::engine::Scene& scene);
    void                            BuildOpacityMicromaps(nvrhi::ICommandList& commandList, const donut::engine::Scene& scene);
    void                            WriteGeometryDebugBuffer(nvrhi::ICommandList& commandList);
    void                            UpdateDebugGeometry(const donut::engine::MeshInfo& mesh);

    [[nodiscard]] nvrhi::IBuffer*   GetGeometryDebugBuffer() const { return m_geometryDebugBuffer; }

    void                            SetGlobalShaderMacros(std::vector<donut::engine::ShaderMacro> & macros);
    uint64_t                        GetMemoryGeneration() const { return m_memoryGeneration; }

    // Routes the OMM BLAS builds through the shared pool; see OmmBuildQueue::SetBlasCompactor.
    void                            SetBlasCompactor(bvh::BlasCompactor* compactor);


private:
    void                            ReleaseDebugData(nvrhi::ICommandList& commandList, const donut::engine::Scene& scene);

    nvrhi::DeviceHandle             m_device;
    std::shared_ptr<donut::engine::TextureCache> m_textureCache;
    std::shared_ptr<donut::engine::CommonRenderPasses> m_commonPasses;
    std::shared_ptr<donut::engine::FramebufferFactory> m_framebufferFactory;
    std::shared_ptr<donut::engine::DescriptorTableManager> m_descriptorTableManager;
    std::shared_ptr<donut::engine::ShaderFactory> m_shaderFactory;

    nvrhi::BindingLayoutHandle      m_commonBindingLayout;
    nvrhi::BindingLayoutHandle      m_bindlessLayout;

    ComputePass                     m_examplePass;

    std::unique_ptr<class OmmBuildQueue>  m_ommBuildQueue;

    OpacityMicroMapUIData           m_uiData;
    bool                            m_retainDebugData = false;
    uint64_t                        m_memoryGeneration = 0;

    std::vector<class GeometryDebugData> m_geometryDebugDataPtr;
    nvrhi::BufferHandle             m_geometryDebugBuffer;
};
