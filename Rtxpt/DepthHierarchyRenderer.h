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

// Generates a Hi-Z depth mip pyramid from a full-resolution depth buffer
// using a single-pass LDS reduction shader. The resulting depth hierarchy
// is used for screen-space effects such as SSR and GTAO.
class DepthHierarchyRenderer
{
public:
    DepthHierarchyRenderer(nvrhi::IDevice* device);
    ~DepthHierarchyRenderer();

    void CreatePipelines(ComputePipelineBaker& computeBaker);
    void DestroyPipelines(ComputePipelineBaker& computeBaker);

    // Call when render targets are (re)created. Takes the DepthHierarchy texture owned by RenderTargets.
    void OnRenderTargetsRecreated(uint32_t width, uint32_t height);

    // Copies depth mip 0 and dispatches the reduction shader to fill remaining mips.
    void Generate(nvrhi::CommandListHandle commandList,
                  nvrhi::TextureHandle depthTexture,
                  uint32_t width, uint32_t height);

    uint32_t GetNumMips() const { return m_numMips; }

    nvrhi::TextureHandle m_depthHierarchy;  // Depth min/max mip pyramid for Hi-Z SSR

private:
    nvrhi::DeviceHandle                     m_device;

    std::shared_ptr<ComputeShaderVariant>   m_variant;
    nvrhi::BindingLayoutHandle              m_bindingLayout;
    nvrhi::BindingSetHandle                 m_bindingSet;
    nvrhi::SamplerHandle                    m_maxSampler;
    uint32_t                                m_numMips = 0;
};
