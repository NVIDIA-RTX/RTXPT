/*
* Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
#include <vector>

namespace donut::engine
{
    class ShaderFactory;
    struct ShaderMacro;
}

class RenderTargets;
class PathTracePerfLog;
struct SampleUIData;
struct SampleMiniConstants;

// Realtime sub-sampling (sub-1spp): owns the compaction pass of the list based pattern and sets up the fill pass ray
// launch for the selected rate and pattern.
class AdaptiveSampling
{
public:
    AdaptiveSampling(nvrhi::IDevice* device, std::shared_ptr<donut::engine::ShaderFactory> shaderFactory, const std::unique_ptr<RenderTargets>& renderTargets, nvrhi::BindingLayoutHandle bindingLayout, nvrhi::BindingLayoutHandle bindlessLayout);

    // Macros for the fill pass raygen. Static: the RT pipelines (and their macro callbacks) are created before this object.
    static void GetFillPassMacros(const SampleUIData& ui, std::vector<donut::engine::ShaderMacro>& macros);

    // Sets up the fill pass for sub-sampling: fills the push constants, runs the compaction pass when the pattern
    // needs it, and returns the ray launch size for the pattern (see SubSampling_ThreadToPixel in Shaders/Misc/SubSampling.h).
    nvrhi::rt::DispatchRaysArguments PrepareSubSampling(nvrhi::ICommandList* commandList, const SampleUIData& ui, nvrhi::IBindingSet* bindingSet, nvrhi::IDescriptorTable* descriptorTable,
                                                        PathTracePerfLog& perfLog, uint32_t width, uint32_t height, SampleMiniConstants& miniConstants);

private:
    const std::unique_ptr<RenderTargets>&   m_renderTargets;

    nvrhi::ShaderHandle                     m_subSamplingCompactionCS;
    nvrhi::ComputePipelineHandle            m_subSamplingCompactionPSO;
};
