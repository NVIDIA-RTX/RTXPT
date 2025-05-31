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

namespace donut::engine
{
    class ShaderFactory;
    struct ShaderMacro;
}


struct RayTracingPass
{
    nvrhi::ShaderHandle ComputeShader;
    nvrhi::ComputePipelineHandle ComputePipeline;

    nvrhi::ShaderLibraryHandle ShaderLibrary;
    nvrhi::rt::PipelineHandle RayTracingPipeline;
    nvrhi::rt::ShaderTableHandle ShaderTable;

    uint32_t ComputeGroupSize = 0;

    bool Init(
        nvrhi::IDevice* device,
        donut::engine::ShaderFactory& shaderFactory,
        const char* shaderName,
        const std::vector<donut::engine::ShaderMacro>& extraMacros,
        bool useRayQuery,
        uint32_t computeGroupSize,
        nvrhi::IBindingLayout* bindingLayout,
        nvrhi::IBindingLayout* extraBindingLayout = nullptr,
        nvrhi::IBindingLayout* bindlessLayout = nullptr);

    void Execute(
        nvrhi::ICommandList* commandList,
        int width,
        int height,
        nvrhi::IBindingSet* bindingSet,
        nvrhi::IBindingSet* extraBindingSet,
        nvrhi::IDescriptorTable* descriptorTable,
        const void* pushConstants = nullptr,
        size_t pushConstantSize = 0);
};