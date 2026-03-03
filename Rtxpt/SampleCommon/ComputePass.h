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


class ComputePass
{
public:
    bool Init(
        nvrhi::IDevice* device,
        donut::engine::ShaderFactory& shaderFactory,
        const char* fileName,
        const char* entry, 
        const std::vector<donut::engine::ShaderMacro>& macros,
        nvrhi::IBindingLayout* bindingLayout,
        nvrhi::IBindingLayout* extraBindingLayout = nullptr,
        nvrhi::IBindingLayout* bindlessLayout = nullptr);

    bool Init(
        nvrhi::IDevice* device,
        donut::engine::ShaderFactory& shaderFactory,
        const char* fileName,
        const char* entry,
        const std::vector<donut::engine::ShaderMacro>& macros,
        nvrhi::BindingLayoutVector & bindingLayouts );

    void Execute(
        nvrhi::ICommandList* commandList,
        int width,
        int height,
        int depth,
        nvrhi::IBindingSet* bindingSet,
        nvrhi::IBindingSet* extraBindingSet = nullptr,
        nvrhi::IDescriptorTable* descriptorTable = nullptr,
        const void* pushConstants = nullptr,
        size_t pushConstantSize = 0);

    void Execute(
        nvrhi::ICommandList* commandList,
        int width,
        int height,
        int depth,
        const nvrhi::BindingSetVector & bindings);

private:
    nvrhi::ShaderHandle             m_computeShader;
    nvrhi::ComputePipelineHandle    m_computePipeline;
    nvrhi::ShaderLibraryHandle      m_shaderLibrary;
};