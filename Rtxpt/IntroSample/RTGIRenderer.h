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

class PTPipelineBaker;
class PTPipelineVariant;

// Ray-traced global illumination renderer.
// Traces cosine-weighted hemisphere rays from GBuffer surfaces, shading hits
// with simple IBL. The output (rgb=irradiance) is consumed by
// the DeferredLightingCombine pass to provide one-bounce diffuse GI. No blur is applied;
// denoising is left to DLSS-RR.
class RTGIRenderer
{
public:
    RTGIRenderer(nvrhi::IDevice* device);
    ~RTGIRenderer() = default;

    void CreateRTPipelines(PTPipelineBaker& pipelineBaker);
    void DestroyRTPipelines(PTPipelineBaker& pipelineBaker);

    void OnRenderTargetsRecreated(uint32_t renderWidth, uint32_t renderHeight);

    // Dispatches the RTGI ray generation pass.
    // When useSharcVariant is true, uses the RTGI+SHaRC pipeline and passes
    // sharcCapacity via push constants.
    // envMip is the fixed mip level RTGI samples the prefiltered environment cube at, both for a GI
    // ray's miss and (via ComputeHitLighting) the one-sample environment term at its hit; forwarded
    // to the shader through SampleMiniConstants::params.z. Trades environment/lighting detail against
    // firefly noise - see "Misses and fireflies" in the Step 2 docs. The caller is responsible for
    // clamping it to the prefiltered cube's actual max mip.
    // includeHitEmissive controls whether emission at the GI hit is counted (params.y). Set false
    // when direct lighting samples emissive triangles explicitly (e.g. ReSTIR DI) to avoid double
    // counting.
    void Render(nvrhi::CommandListHandle commandList,
                uint32_t renderWidth, uint32_t renderHeight,
                nvrhi::BindingSetHandle globalBindingSet,
                nvrhi::BindingSetHandle descriptorTable,
                bool useSharcVariant = false,
                uint32_t sharcCapacity = 0,
                uint32_t envMip = 5,
                bool includeHitEmissive = true);

    bool IsReady() const;
    bool IsSharcReady() const;

    nvrhi::TextureHandle GetOutput() const { return m_output; }

private:
    nvrhi::DeviceHandle  m_device;
    nvrhi::TextureHandle m_output;  // Screen-sized RGBA16F (rgb=irradiance; alpha unused)

    std::shared_ptr<PTPipelineVariant> m_rtgiPipeline;
    std::shared_ptr<PTPipelineVariant> m_rtgiSharcPipeline;
};
