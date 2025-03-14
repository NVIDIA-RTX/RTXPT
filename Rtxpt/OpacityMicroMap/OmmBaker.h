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

#include <donut/engine/BindingCache.h>
#include <nvrhi/nvrhi.h>
#include <donut/core/math/math.h>
#include <donut\engine\SceneTypes.h>

#include "../ComputePass.h"

using namespace donut::math;

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
    nvrhi::rt::AccelStructHandle AccelStructOMM; // for use by application
    std::vector<nvrhi::rt::OpacityMicromapHandle> OpacityMicroMaps; // for use by application

    std::unique_ptr<MeshDebugData> DebugData;
    bool DebugDataDirty = true; // set this to true to make Scene update the debug data

    virtual ~MeshInfoEx() = default;
};

struct OpacityMicroMapUIData
{
    struct BuildState
    {
        // ~~ Application is expected to tweak these settings ~~ 
        int MaxSubdivisionLevel = 12;
        bool EnableDynamicSubdivision = true;
        float DynamicSubdivisionScale = 1.f;
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

        bool operator == (const BuildState& o) const {
            return
                MaxSubdivisionLevel == o.MaxSubdivisionLevel &&
                EnableDynamicSubdivision == o.EnableDynamicSubdivision &&
                DynamicSubdivisionScale == o.DynamicSubdivisionScale &&
                Flag == o.Flag &&
                Format == o.Format &&
                AlphaCutoffGT == o.AlphaCutoffGT &&
                AlphaCutoffLE == o.AlphaCutoffLE &&
                ComputeOnly == o.ComputeOnly &&
                LevelLineIntersection == o.LevelLineIntersection &&
                EnableTexCoordDeduplication == o.EnableTexCoordDeduplication &&
                Force32BitIndices == o.Force32BitIndices &&
                EnableNsightDebugMode == o.EnableNsightDebugMode &&
                EnableSpecialIndices == o.EnableSpecialIndices &&
                MaxOmmArrayDataSizeInMB == o.MaxOmmArrayDataSizeInMB
                ;
        }
    };

    bool                                Enable = true;
    bool                                Force2State = false;
    bool                                OnlyOMMs = false;

    // Amortize the builds over multiple frames
    std::optional<BuildState>           ActiveState;
    BuildState                          DesiredState;
    bool                                TriggerRebuild = true;

    // --- Stats --- 
    // build progress of active tasks
    uint32_t                            BuildsLeftInQueue = 0;
    uint32_t                            BuildsQueued = 0;
};

class OmmBaker
{
public:
    OmmBaker(nvrhi::IDevice* device, std::shared_ptr<donut::engine::DescriptorTableManager> descriptorTableManager, std::shared_ptr<donut::engine::TextureCache> textureCache, std::shared_ptr<donut::engine::ShaderFactory> shaderFactory, nvrhi::CommandListHandle mainCommandList, bool enabled);
    ~OmmBaker();

    bool                            IsEnabled() const { return m_enabled; }

    void                            CreateRenderPasses(nvrhi::IBindingLayout* bindlessLayout, std::shared_ptr<donut::engine::CommonRenderPasses> commonPasses, const std::shared_ptr<donut::engine::Scene>& scene);

    void                            Update(nvrhi::ICommandList * commandList, const std::shared_ptr<donut::engine::Scene> & scene);

    OpacityMicroMapUIData &         UIData()    { return m_uiData; }
    bool                            DebugGUI(float indent, const std::shared_ptr<donut::engine::Scene>& scene);

    void                            SceneLoaded(const std::shared_ptr<donut::engine::Scene>& scene);
    void                            SceneUnloading();

    void                            CreateOpacityMicromaps(const std::shared_ptr<donut::engine::Scene>& scene);
    void                            DestroyOpacityMicromaps(nvrhi::ICommandList* commandList, const std::shared_ptr<donut::engine::Scene>& scene);
    void                            BuildOpacityMicromaps(nvrhi::ICommandList* commandList, const std::shared_ptr<donut::engine::Scene>& scene);
    void                            WriteGeometryDebugBuffer(nvrhi::ICommandList* commandList);
    void                            UpdateDebugGeometry(const std::shared_ptr<struct donut::engine::MeshInfo>& mesh);

    [[nodiscard]] nvrhi::IBuffer*   GetGeometryDebugBuffer() const { return m_geometryDebugBuffer; }


private:
    nvrhi::DeviceHandle             m_device;
    std::shared_ptr<donut::engine::TextureCache> m_textureCache;
    std::shared_ptr<donut::engine::CommonRenderPasses> m_commonPasses;
    std::shared_ptr<donut::engine::FramebufferFactory> m_framebufferFactory;
    std::shared_ptr<donut::engine::DescriptorTableManager> m_descriptorTableManager;
    std::shared_ptr<donut::engine::ShaderFactory> m_shaderFactory;

    nvrhi::BindingLayoutHandle      m_commonBindingLayout;
    nvrhi::BindingLayoutHandle      m_bindlessLayout;
    donut::engine::BindingCache     m_bindingCache;

    ComputePass                     m_examplePass;

    std::unique_ptr<class OmmBuildQueue>  m_ommBuildQueue;

    OpacityMicroMapUIData           m_uiData;

    std::unique_ptr<std::vector<class GeometryDebugData>> m_geometryDebugDataPtr;
    nvrhi::BufferHandle             m_geometryDebugBuffer;

    bool                            m_enabled;
};
