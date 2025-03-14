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
#include <donut/core/math/math.h>
#include <memory>
#include <unordered_map>
#include <rtxdi/ImportanceSamplingContext.h>
#include "../ComputePass.h"
#include "../RayTracingPass.h"
#include "RtxdiResources.h"
#include "RtxdiApplicationSettings.h"

#include "../PathTracer/Lighting/LightingTypes.h"

class RenderTargets;
class PrepareLightsPass;
class GenerateMipsPass;
class EnvMapBaker;
struct ReGirIndirectConstants;
class ShaderDebug;

namespace donut::engine
{
	class ExtendedScene;
	class ShaderFactory;
	class CommonRenderPasses;
	struct ShaderMacro;
	class PlanarView;
}

struct RtxdiUserSettings
{
	struct
	{
		rtxdi::ReSTIRDI_ResamplingMode resamplingMode = GetReSTIRDI_ResamplingMode();
		ReSTIRDI_InitialSamplingParameters initialSamplingParams = getReSTIRDIInitialSamplingParams();
		ReSTIRDI_TemporalResamplingParameters temporalResamplingParams = getReSTIRDITemporalResamplingParams();
		ReSTIRDI_SpatialResamplingParameters spatialResamplingParams = getReSTIRDISpatialResamplingParams();
		ReSTIRDI_ShadingParameters shadingParams = getReSTIRDIShadingParams();
	} restirDI;

	struct  
	{
		rtxdi::ReSTIRGI_ResamplingMode resamplingMode = GetReSTIRGI_ResamplingMode();
		ReSTIRGI_TemporalResamplingParameters temporalResamplingParams = getReSTIRGITemporalResamplingParams();
		ReSTIRGI_SpatialResamplingParameters spatialResamplingParams = getReSTIRGISpatialResamplingParams();
		ReSTIRGI_FinalShadingParameters finalShadingParams = getReSTIRGIFinalShadingParams();
	} restirGI;

	struct 
	{
		rtxdi::ReGIRStaticParameters regirStaticParams = {};
		rtxdi::ReGIRDynamicParameters regirDynamicParameters = getReGIRDynamicParams();
	} regir;

	struct
	{
		int numIndirectSamples = 6;
	} regirIndirect;

	float rayEpsilon = 1.0e-4f;
	bool reStirGIEnableTemporalResampling = true;
	bool reStirGIVaryAgeThreshold = true;
};

struct RtxdiBridgeParameters
{
	uint32_t frameIndex;
	donut::math::uint2 frameDims;
	donut::math::float3 cameraPosition;

	RtxdiUserSettings userSettings;

    bool usingLightSampling;
    bool usingReGIR;
};

class RtxdiPass
{
public:
	RtxdiPass(
		nvrhi::IDevice* device,
		std::shared_ptr<donut::engine::ShaderFactory> shaderFactory,
		std::shared_ptr<donut::engine::CommonRenderPasses> commonRenderPasses,
		nvrhi::BindingLayoutHandle bindlessLayout);
	~RtxdiPass();

	void Reset();
    void PrepareResources(
        nvrhi::CommandListHandle commandList,
        const RenderTargets& renderTargets,
        std::shared_ptr<EnvMapBaker> envMap,
        EnvMapSceneParams envMapSceneParams,
        const std::shared_ptr<donut::engine::ExtendedScene> scene,
        std::shared_ptr<class MaterialsBaker> materialsBaker,
        std::shared_ptr<class OmmBaker> ommBaker,
        nvrhi::BufferHandle subInstanceDataBuffer,
        const RtxdiBridgeParameters& bridgeParams,
        const nvrhi::BindingLayoutHandle extraBindingLayout,
        std::shared_ptr<ShaderDebug> shaderDebug);
	void BeginFrame(
		nvrhi::CommandListHandle commandList,
		const RenderTargets& renderTargets,
		const nvrhi::BindingLayoutHandle extraBindingLayout,
		nvrhi::BindingSetHandle extraBindingSet);
	void Execute(
		nvrhi::CommandListHandle commandList,
		nvrhi::BindingSetHandle extraBindingSet, bool skipFinal);
	void ExecuteGI(nvrhi::CommandListHandle commandList,
		nvrhi::BindingSetHandle extraBindingSet, bool skipFinal);
    void ExecuteFusedDIGIFinal(nvrhi::CommandListHandle commandList,
        nvrhi::BindingSetHandle extraBindingSet);
	void EndFrame();
	
	std::shared_ptr<RtxdiResources> GetRTXDIResources() { return m_rtxdiResources; }
	nvrhi::BufferHandle GetRTXDIConstants() { return m_rtxdiConstantBuffer; }

private:
	void CheckContextStaticParameters();
	void UpdateContextDynamicParameters();
	void CreatePipelines(nvrhi::BindingLayoutHandle extraBindingLayout = nullptr, bool useRayQuery = true);
	void CreateBindingSet(const RenderTargets& renderTargets);

	std::unique_ptr<rtxdi::ImportanceSamplingContext> m_ImportanceSamplingContext;
	std::shared_ptr<RtxdiResources> m_rtxdiResources;
	std::unique_ptr<PrepareLightsPass> m_PrepareLightsPass;
	std::unique_ptr<GenerateMipsPass> m_LocalLightPdfMipmapPass;

	nvrhi::DeviceHandle m_device; 
	std::shared_ptr<donut::engine::ShaderFactory> m_shaderFactory;
	std::shared_ptr<donut::engine::CommonRenderPasses> m_CommonRenderPasses;
	std::shared_ptr<donut::engine::ExtendedScene> m_Scene;
	nvrhi::BindingLayoutHandle m_bindingLayout;
	nvrhi::BindingLayoutHandle m_bindlessLayout;
	nvrhi::BindingSetHandle m_bindingSet;
	nvrhi::BindingSetHandle m_PrevBindingSet;
	nvrhi::BufferHandle m_rtxdiConstantBuffer;

	ComputePass m_PresampleLightsPass;
	ComputePass m_PresampleEnvMapPass;
	ComputePass m_PresampleReGIRPass;
	ComputePass m_FinalSamplingPass;
	RayTracingPass m_GenerateInitialSamplesPass;
	RayTracingPass m_SpatialResamplingPass;
	RayTracingPass m_TemporalResamplingPass;

	RayTracingPass m_GITemporalResamplingPass;
	RayTracingPass m_GISpatialResamplingPass;
	RayTracingPass m_GIFinalShadingPass;

    RayTracingPass m_FusedDIGIFinalShadingPass;

	void ExecuteComputePass(
		nvrhi::CommandListHandle& commandList, 
		ComputePass& pass, 
		const char* passName, 
		dm::int3 dispatchSize, 
		nvrhi::BindingSetHandle extraBindingSet = nullptr);

	void ExecuteRayTracingPass(
		nvrhi::CommandListHandle& commandList,
		RayTracingPass& pass,
		const char* passName,
		dm::int2 dispatchSize, 
		nvrhi::IBindingSet* extraBindingSet = nullptr
	);

	donut::engine::ShaderMacro GetReGirMacro(const rtxdi::ReGIRStaticParameters& regirParameters);

	void FillConstants(nvrhi::CommandListHandle commandList);
	void FillSharedConstants(struct RtxdiBridgeConstants& bridgeConstants) const;
	void FillDIConstants(ReSTIRDI_Parameters& diParams);
	void FillGIConstants(ReSTIRGI_Parameters& giParams);
	void FillReGIRConstant(ReGIR_Parameters& regirParams);
	void FillReGirIndirectConstants(ReGirIndirectConstants& regirIndirectConstants);

	RtxdiBridgeParameters m_BridgeParameters;
	uint32_t m_CurrentReservoirIndex;
	uint32_t m_PreviousReservoirIndex;
};

