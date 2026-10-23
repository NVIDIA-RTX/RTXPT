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

// Stochastic indirect lighting via path tracing.
// Traces rays from GBuffer surfaces to gather indirect illumination.
// Writes diffuse/transmission and specular contributions for the deferred lighting combine pass.
// Denoising is left to DLSS-RR.
class IndirectPathTracer
{
public:
    IndirectPathTracer(nvrhi::IDevice* device);
    ~IndirectPathTracer() = default;

    void CreateRTPipelines(PTPipelineBaker& baker);
    void DestroyRTPipelines(PTPipelineBaker& pipelineBaker);

    void OnRenderTargetsRecreated(uint32_t renderWidth, uint32_t renderHeight);

    // Dispatches the indirect path tracing ray generation pass.
    // When useSharcVariant is true, uses the SHaRC-enabled pipeline and passes
    // sharcCapacity via push constants.
    // Primary NEE coverage prevents counting emitter/environment paths twice in ReSTIR DI mode.
    void Render(nvrhi::CommandListHandle commandList,
                uint32_t renderWidth, uint32_t renderHeight,
                nvrhi::BindingSetHandle globalBindingSet,
                nvrhi::BindingSetHandle descriptorTable,
                bool useSharcVariant = false,
                uint32_t sharcCapacity = 0,
                bool primaryEmissiveNEE = false,
                bool primaryEnvironmentNEE = false);

    bool IsReady() const;
    bool IsSharcReady() const;

    nvrhi::TextureHandle GetOutput() const { return m_output; }

private:
    nvrhi::DeviceHandle  m_device;
    nvrhi::TextureHandle m_output;  // Screen-sized RGBA16F (rgb=irradiance; alpha unused)

    std::shared_ptr<PTPipelineVariant> m_pipeline;
    std::shared_ptr<PTPipelineVariant> m_sharcPipeline;
};
