/*
* Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "AdaptiveSampling.h"

#include "SampleCommon.h"
#include "PathTracePerfLog.h"
#include "RenderTargets.h"
#include "../SampleUI.h"
#include "../Shaders/SampleConstantBuffer.h"
#include "../Shaders/Misc/SubSampling.h"

#include <donut/engine/ShaderFactory.h>

#include <algorithm>
#include <cmath>

static_assert((int)SubSamplingPattern::BlockJitter          == RTXPT_SUB_SAMPLING_PATTERN_BLOCK_JITTER);
static_assert((int)SubSamplingPattern::StochasticCompacted  == RTXPT_SUB_SAMPLING_PATTERN_STOCHASTIC_COMPACTED);
static_assert((int)SubSamplingPattern::BlockStratified      == RTXPT_SUB_SAMPLING_PATTERN_BLOCK_STRATIFIED);
static_assert((int)SubSamplingPattern::SobolBlock           == RTXPT_SUB_SAMPLING_PATTERN_SOBOL_BLOCK);

AdaptiveSampling::AdaptiveSampling(nvrhi::IDevice* device, std::shared_ptr<donut::engine::ShaderFactory> shaderFactory, const std::unique_ptr<RenderTargets>& renderTargets, nvrhi::BindingLayoutHandle bindingLayout, nvrhi::BindingLayoutHandle bindlessLayout)
    : m_renderTargets(renderTargets)
{
    m_subSamplingCompactionCS = shaderFactory->CreateShader("app/ProcessingPasses/SubSamplingCompaction.hlsl", "main", nullptr, nvrhi::ShaderType::Compute);
    nvrhi::ComputePipelineDesc pipelineDesc;
    pipelineDesc.bindingLayouts = { bindingLayout, bindlessLayout };
    pipelineDesc.CS = m_subSamplingCompactionCS;
    m_subSamplingCompactionPSO = device->createComputePipeline(pipelineDesc);
}

void AdaptiveSampling::GetFillPassMacros(const SampleUIData& ui, std::vector<donut::engine::ShaderMacro>& macros)
{
    macros.push_back({ "RTXPT_SUB_SAMPLING_SCALE_X", std::to_string(ui.SubSamplingScaleX()) });
    macros.push_back({ "RTXPT_SUB_SAMPLING_SCALE_Y", std::to_string(ui.SubSamplingScaleY()) });
    macros.push_back({ "RTXPT_SUB_SAMPLING_RATE_NUM", std::to_string(ui.SubSamplingRateNum()) });
    // baked rather than push constants so the unused patterns compile out; switching them recompiles the fill pass
    macros.push_back({ "RTXPT_SUB_SAMPLING_PATTERN", std::to_string((int)ui.ActualSubSamplingPattern()) });
    macros.push_back({ "RTXPT_SUB_SAMPLING_BLOCK_SIZE", std::to_string(ui.SubSamplingStratifiedBlockSize()) });
}

nvrhi::rt::DispatchRaysArguments AdaptiveSampling::PrepareSubSampling(nvrhi::ICommandList* commandList, const SampleUIData& ui, nvrhi::IBindingSet* bindingSet, nvrhi::IDescriptorTable* descriptorTable,
                                                                      PathTracePerfLog& perfLog, uint32_t width, uint32_t height, SampleMiniConstants& miniConstants)
{
    const uint32_t scaleX = ui.SubSamplingScaleX();
    const uint32_t scaleY = ui.SubSamplingScaleY();
    const uint32_t rateNum = ui.SubSamplingRateNum();
    const uint32_t rateDen = scaleX * scaleY;
    const SubSamplingPattern pattern = ui.ActualSubSamplingPattern();
    const uint32_t blockSize = ui.SubSamplingStratifiedBlockSize();

    miniConstants.params = uint4(0, 0, 0, 0);
    miniConstants.params1 = uint4(0, 0, rateDen, rateNum);     // the rate, for the compaction pass

    nvrhi::rt::DispatchRaysArguments args;
    switch (pattern)
    {
    case SubSamplingPattern::StochasticCompacted:
    {
        RAII_SCOPE(GpuProfiler::Begin(commandList, "SubSamplingCompaction"); perfLog.Begin(commandList, PathTracePerfLog::Compaction);, perfLog.End(commandList, PathTracePerfLog::Compaction); GpuProfiler::End(commandList); );

        const uint32_t zero = 0;
        commandList->writeBuffer(m_renderTargets->SubSamplingPixelList, &zero, sizeof(zero), 0);
        // Explicit, because nvrhi skips a binding set's state transitions when the compute bindings did not change
        // since the last setComputeState (e.g. VBufferExport uses the same ones), which would leave the reset copy
        // unordered with the compaction's atomics. setComputeState commits it.
        commandList->setBufferState(m_renderTargets->SubSamplingPixelList, nvrhi::ResourceStates::UnorderedAccess);

        nvrhi::ComputeState computeState;
        computeState.bindings = { bindingSet, descriptorTable };
        computeState.pipeline = m_subSamplingCompactionPSO;
        commandList->setComputeState(computeState);
        commandList->setPushConstants(&miniConstants, sizeof(miniConstants));
        commandList->dispatch(div_ceil(width, RTXPT_SUB_SAMPLING_COMPACTION_TILE), div_ceil(height, RTXPT_SUB_SAMPLING_COMPACTION_TILE));

        // the raygen reads what the compaction wrote
        commandList->setBufferState(m_renderTargets->SubSamplingPixelList, nvrhi::ResourceStates::UnorderedAccess);

        // The CPU doesn't know how many pixels got selected, and there is no indirect dispatchRays, so launch
        // conservatively; threads past the GPU-side count exit straight away.
        // The count is Binomial(n, rate): mean + 8 sigma is exceeded with probability ~1e-15 per frame, and
        // even then it only leaves a handful of pixels untraced for that one frame.
        const double n = double(width) * double(height);
        const double p = double(rateNum) / double(rateDen);
        const double bound = std::ceil(n * p + 8.0 * std::sqrt(n * p * (1.0 - p))) + 256.0;
        args.width = (uint32_t)std::min(n, bound);
        args.height = 1;
        break;
    }
    case SubSamplingPattern::BlockStratified:
    case SubSamplingPattern::SobolBlock:
        args.width = div_ceil(width, blockSize) * (blockSize * blockSize * rateNum / rateDen);
        args.height = div_ceil(height, blockSize);
        break;
    case SubSamplingPattern::BlockJitter:
    default:
        args.width = div_ceil(width, scaleX) * rateNum;
        args.height = div_ceil(height, scaleY);
        break;
    }
    return args;
}
