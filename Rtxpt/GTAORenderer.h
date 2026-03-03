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

class ComputePipelineBaker;
class ComputeShaderVariant;
struct SampleConstants;

// Ground Truth Ambient Occlusion (GTAO) renderer.
// Implements horizon-based ambient occlusion with radiometrically-correct
// integration (Jimenez et al., "Practical Real-Time Strategies for Accurate
// Indirect Occlusion"), including spatio-temporal filtering and bilateral upscale.
class GTAORenderer
{
public:
    GTAORenderer(nvrhi::IDevice* device);
    ~GTAORenderer();

    void CreatePipelines(nvrhi::BindingLayoutHandle globalBindingLayout,
                         nvrhi::BindingLayoutHandle globalBindlessLayout,
                        ComputePipelineBaker& computeBaker);
    void DestroyPipelines(ComputePipelineBaker& computeBaker);

    void OnRenderTargetsRecreated(uint32_t renderWidth, uint32_t renderHeight);

    void Render(nvrhi::CommandListHandle commandList,
                const SampleConstants& constants,
                uint32_t frameIndex,
                nvrhi::TextureHandle depthHierarchy,
                nvrhi::TextureHandle fullResNormals,
                nvrhi::TextureHandle fullResMotionVectors,
                nvrhi::TextureHandle prevDepth,
                nvrhi::BindingSetHandle globalBindingSet);

    nvrhi::TextureHandle GetOutputTexture() const { return m_GTAOOutput; }

private:
    void ComputeAO(nvrhi::CommandListHandle commandList);
    void SpatialFilter(nvrhi::CommandListHandle commandList);
    void TemporalAccumulate(nvrhi::CommandListHandle commandList);

    nvrhi::DeviceHandle                     m_device;

    // Render targets
    nvrhi::TextureHandle    m_GTAORaw;        // Half-res R8_UNORM: raw horizon-based AO
    nvrhi::TextureHandle    m_GTAOFiltered;   // Half-res R8_UNORM: spatially filtered AO
    nvrhi::TextureHandle    m_GTAOHistory[2]; // Half-res R8_UNORM: ping-pong history buffers
    uint32_t                m_historyReadIdx = 0;
    nvrhi::TextureHandle    m_GTAOOutput;     // Full-res R8_UNORM: final upscaled AO
    uint32_t                m_halfWidth = 0;
    uint32_t                m_halfHeight = 0;
    uint32_t                m_fullWidth = 0;
    uint32_t                m_fullHeight = 0;

    // Compute pipelines (ComputePipelineBaker for hot reload)
    std::shared_ptr<ComputeShaderVariant>   m_computeAOVariant;
    std::shared_ptr<ComputeShaderVariant>   m_spatialFilterVariant;
    std::shared_ptr<ComputeShaderVariant>   m_temporalAccumVariant;

    // Dedicated binding layouts and sets per subpass
    nvrhi::BindingLayoutHandle  m_computeAOBindingLayout;
    nvrhi::BindingSetHandle     m_computeAOBindingSet;
    nvrhi::BindingLayoutHandle  m_spatialFilterBindingLayout;
    nvrhi::BindingSetHandle     m_spatialFilterBindingSet;
    nvrhi::BindingLayoutHandle  m_temporalAccumBindingLayout;
    nvrhi::BindingSetHandle     m_temporalAccumBindingSet;

    nvrhi::BufferHandle         m_constantBuffer;
    nvrhi::SamplerHandle        m_pointClampSampler;

    // Cached references to textures and global binding set for binding set creation
    nvrhi::TextureHandle        m_cachedDepthHierarchy;
    nvrhi::TextureHandle        m_cachedNormals;
    nvrhi::TextureHandle        m_cachedMotionVectors;
    nvrhi::TextureHandle        m_cachedPrevDepth;
    nvrhi::BindingSetHandle     m_cachedGlobalBindingSet;
    uint32_t                    m_cachedFrameIndex = 0;
};
