/*
* Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "RayTracingPass.h"

#include <donut/engine/ShaderFactory.h>
#include <donut/core/math/math.h>
#include <donut/core/log.h>
#include <nvrhi/utils.h>

using namespace donut::engine;


bool RayTracingPass::Init(
    nvrhi::IDevice* device,
    donut::engine::ShaderFactory& shaderFactory,
    const char* shaderName,
    const std::vector<donut::engine::ShaderMacro>& extraMacros,
    bool useRayQuery,
    uint32_t computeGroupSize,
    nvrhi::IBindingLayout* bindingLayout,
    nvrhi::IBindingLayout* extraBindingLayout,
    nvrhi::IBindingLayout* bindlessLayout)
{
    donut::log::debug("Initializing RayTracingPass %s...", shaderName);

    ComputeGroupSize = computeGroupSize;

    std::vector<donut::engine::ShaderMacro> macros = { { "USE_RAY_QUERY", "1" } };

    macros.insert(macros.end(), extraMacros.begin(), extraMacros.end());

    if (useRayQuery)
    {
        ComputeShader = shaderFactory.CreateShader(shaderName, "main", &macros, nvrhi::ShaderType::Compute);
        if (!ComputeShader)
            return false;

        nvrhi::ComputePipelineDesc pipelineDesc;
        if(extraBindingLayout)
            pipelineDesc.bindingLayouts.push_back(extraBindingLayout);
        if (bindlessLayout)
            pipelineDesc.bindingLayouts.push_back(bindlessLayout);
        if (bindingLayout)
            pipelineDesc.bindingLayouts.push_back(bindingLayout);
        pipelineDesc.CS = ComputeShader;
        ComputePipeline = device->createComputePipeline(pipelineDesc);

        if (!ComputePipeline)
            return false;

        return true;
    }

    macros[0].definition = "0"; // USE_RAY_QUERY
    ShaderLibrary = shaderFactory.CreateShaderLibrary(shaderName, &macros);
    if (!ShaderLibrary)
        return false;

    nvrhi::rt::PipelineDesc rtPipelineDesc;
    rtPipelineDesc.globalBindingLayouts = { extraBindingLayout, bindlessLayout };
    if (bindingLayout)
        rtPipelineDesc.globalBindingLayouts.push_back(bindingLayout);
    rtPipelineDesc.shaders = {
        { "", ShaderLibrary->getShader("RayGen", nvrhi::ShaderType::RayGeneration), nullptr },
        { "", ShaderLibrary->getShader("Miss", nvrhi::ShaderType::Miss), nullptr }
    };

    rtPipelineDesc.hitGroups = {
        {
            "HitGroup",
            ShaderLibrary->getShader("ClosestHit", nvrhi::ShaderType::ClosestHit),
            ShaderLibrary->getShader("AnyHit", nvrhi::ShaderType::AnyHit),
            nullptr, // intersectionShader
            nullptr, // localBindingLayout
            false // isProceduralPrimitive
        },
    };

    rtPipelineDesc.maxAttributeSize = 8;
    rtPipelineDesc.maxPayloadSize = 40;
    rtPipelineDesc.maxRecursionDepth = 1;

    RayTracingPipeline = device->createRayTracingPipeline(rtPipelineDesc);
    if (!RayTracingPipeline)
        return false;

    ShaderTable = RayTracingPipeline->createShaderTable();
    if (!ShaderTable)
        return false;

    ShaderTable->setRayGenerationShader("RayGen");
    ShaderTable->addMissShader("Miss");
    ShaderTable->addHitGroup("HitGroup");

    return true;
}

void RayTracingPass::Execute(
    nvrhi::ICommandList* commandList,
    int width,
    int height,
    nvrhi::IBindingSet* bindingSet,
    nvrhi::IBindingSet* extraBindingSet,
    nvrhi::IDescriptorTable* descriptorTable,
    const void* pushConstants,
    const size_t pushConstantSize)
{
    if (ComputePipeline)
    {
        nvrhi::ComputeState state;
        state.bindings = { extraBindingSet };
        if (descriptorTable)
            state.bindings.push_back(descriptorTable);
        if (bindingSet)
            state.bindings.push_back(bindingSet);
        state.pipeline = ComputePipeline;
        commandList->setComputeState(state);

        if (pushConstants)
            commandList->setPushConstants(pushConstants, pushConstantSize);

        commandList->dispatch(dm::div_ceil(width, ComputeGroupSize), dm::div_ceil(height, ComputeGroupSize), 1);
    }
    else
    {
        nvrhi::rt::State state;
        state.bindings = { extraBindingSet };
        if (descriptorTable)
            state.bindings.push_back(descriptorTable);
        if (bindingSet)
            state.bindings.push_back(bindingSet);
        state.shaderTable = ShaderTable;
        commandList->setRayTracingState(state);

        if (pushConstants)
            commandList->setPushConstants(pushConstants, pushConstantSize);

        nvrhi::rt::DispatchRaysArguments args;
        args.width = width;
        args.height = height;
        args.depth = 1;
        commandList->dispatchRays(args);
    }
}
