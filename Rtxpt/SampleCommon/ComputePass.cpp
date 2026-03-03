/*
* Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "ComputePass.h"

#include <donut/engine/ShaderFactory.h>
#include <donut/core/math/math.h>
#include <donut/core/log.h>
#include <nvrhi/utils.h>

using namespace donut::engine;


bool ComputePass::Init(
	nvrhi::IDevice* device, 
	donut::engine::ShaderFactory& shaderFactory, 
	const char* fileName, 
    const char* entry, 
	const std::vector<donut::engine::ShaderMacro>& macros, 
	nvrhi::IBindingLayout* bindingLayout,
	nvrhi::IBindingLayout* extraBindingLayout /*= nullptr*/, 
	nvrhi::IBindingLayout* bindlessLayout /*= nullptr*/)
{
	m_computeShader = shaderFactory.CreateShader(fileName, entry, &macros, nvrhi::ShaderType::Compute);
	if (!m_computeShader)
		return false;

	nvrhi::ComputePipelineDesc pipelineDesc;
	if(extraBindingLayout)
		pipelineDesc.bindingLayouts.push_back(extraBindingLayout);
	if (bindlessLayout)
		pipelineDesc.bindingLayouts.push_back(bindlessLayout);
	if (bindingLayout)
		pipelineDesc.bindingLayouts.push_back(bindingLayout);
	pipelineDesc.CS = m_computeShader;
	m_computePipeline = device->createComputePipeline(pipelineDesc);

	if (!m_computePipeline)
		return false;

	return true;
}

bool ComputePass::Init(
    nvrhi::IDevice* device,
    donut::engine::ShaderFactory& shaderFactory,
    const char* fileName,
    const char* entry,
    const std::vector<donut::engine::ShaderMacro>& macros,
    nvrhi::BindingLayoutVector & bindingLayouts )
{
    m_computeShader = shaderFactory.CreateShader(fileName, entry, &macros, nvrhi::ShaderType::Compute);
    if (!m_computeShader)
        return false;

    nvrhi::ComputePipelineDesc pipelineDesc;
    pipelineDesc.bindingLayouts = bindingLayouts;
    pipelineDesc.CS = m_computeShader;
    m_computePipeline = device->createComputePipeline(pipelineDesc);

    if (!m_computePipeline)
        return false;

    return true;
}


void ComputePass::Execute(
	nvrhi::ICommandList* commandList, 
	int width, 
	int height, 
	int depth, 
	nvrhi::IBindingSet* bindingSet, 
	nvrhi::IBindingSet* extraBindingSet /*= nullptr*/, 
	nvrhi::IDescriptorTable* descriptorTable /*= nullptr*/, 
	const void* pushConstants /*= nullptr*/, 
	size_t pushConstantSize /*= 0*/)
{
	nvrhi::ComputeState state;
	state.bindings;
    if (extraBindingSet)
        state.bindings.push_back(extraBindingSet);
	if (descriptorTable)
		state.bindings.push_back(descriptorTable);
	if (bindingSet)
		state.bindings.push_back(bindingSet);
	state.pipeline = m_computePipeline;
	commandList->setComputeState(state);

	if (pushConstants)
		commandList->setPushConstants(pushConstants, pushConstantSize);

	commandList->dispatch(width, height, depth);
}

void ComputePass::Execute(
    nvrhi::ICommandList* commandList,
    int width,
    int height,
    int depth,
    const nvrhi::BindingSetVector & bindings)
{
    nvrhi::ComputeState state;
    state.bindings = bindings;
    state.pipeline = m_computePipeline;
    commandList->setComputeState(state);

    commandList->dispatch(width, height, depth);
}
