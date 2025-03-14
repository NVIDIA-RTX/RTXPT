/*
* Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "RtxdiPass.h"
#include <rtxdi/ImportanceSamplingContext.h>

#include "RtxdiResources.h"
#include "PrepareLightsPass.h"
#include "donut/engine/ShaderFactory.h"
#include "donut/engine/CommonRenderPasses.h"
#include "donut/engine/View.h"
#include "GeneratePdfMipsPass.h"
#include "../Lighting/Distant/EnvMapBaker.h"
#include "../Lighting/Distant/EnvMapImportanceSamplingBaker.h"
#include "../ExtendedScene.h"
#include "ShaderParameters.h"

#include "../RenderTargets.h"

using namespace donut::math;
using namespace donut::engine;
using namespace donut::render;

RtxdiPass::RtxdiPass(
	nvrhi::IDevice* device,
	std::shared_ptr<donut::engine::ShaderFactory> shaderFactory,
	std::shared_ptr<donut::engine::CommonRenderPasses> commonRenderPasses,
	nvrhi::BindingLayoutHandle bindlessLayout) :
		m_device(device),
		m_shaderFactory(shaderFactory),
		m_CommonRenderPasses(commonRenderPasses),
		m_bindlessLayout(bindlessLayout),
		m_PreviousReservoirIndex(0)
{
	//Create binding layouts
	nvrhi::BindingLayoutDesc layoutDesc;
	layoutDesc.visibility = nvrhi::ShaderType::All;
	layoutDesc.bindings = {
		nvrhi::BindingLayoutItem::StructuredBuffer_SRV(21),		//t_LightDataBuffer
		nvrhi::BindingLayoutItem::TypedBuffer_SRV(22),			//t_NeighborOffsets
		nvrhi::BindingLayoutItem::TypedBuffer_SRV(23),			//t_LightIndexMappingBuffer
		nvrhi::BindingLayoutItem::Texture_SRV(25),				//t_LocalLightPdfTexture
		nvrhi::BindingLayoutItem::StructuredBuffer_SRV(26),		//t_GeometryInstanceToLight
		
		nvrhi::BindingLayoutItem::StructuredBuffer_UAV(13),		//u_LightReservoirs
		nvrhi::BindingLayoutItem::StructuredBuffer_UAV(14),		//u_GIReservoirs
		nvrhi::BindingLayoutItem::TypedBuffer_UAV(15),			//u_RisBuffer
		nvrhi::BindingLayoutItem::TypedBuffer_UAV(16),			//u_RisLightDataBuffer

		nvrhi::BindingLayoutItem::VolatileConstantBuffer(5),	//g_RtxdiBridgeConst

		nvrhi::BindingLayoutItem::Sampler(4)
	};
	m_bindingLayout = m_device->createBindingLayout(layoutDesc);

	m_rtxdiConstantBuffer = m_device->createBuffer(nvrhi::utils::CreateVolatileConstantBufferDesc(sizeof(RtxdiBridgeConstants), "RtxdiBridgeConstants", 16));
}

RtxdiPass::~RtxdiPass(){}

// Check for changes in the static parameters, these will require the importance sampling context to be recreated
void RtxdiPass::CheckContextStaticParameters()
{
	if (m_ImportanceSamplingContext != nullptr)
	{
		auto& reGIRContext = m_ImportanceSamplingContext->GetReGIRContext();

		bool needsReset = false;
		if (reGIRContext.GetReGIRStaticParameters().Mode != m_BridgeParameters.userSettings.regir.regirStaticParams.Mode)
			needsReset = true;
		if (reGIRContext.GetReGIRStaticParameters().LightsPerCell != m_BridgeParameters.userSettings.regir.regirStaticParams.LightsPerCell)
			needsReset = true;

		if (needsReset)
			Reset();
	}
}

void RtxdiPass::UpdateContextDynamicParameters()
{
	// ReSTIR DI
	m_ImportanceSamplingContext->GetReSTIRDIContext().SetFrameIndex(m_BridgeParameters.frameIndex);
	m_ImportanceSamplingContext->GetReSTIRDIContext().SetInitialSamplingParameters(m_BridgeParameters.userSettings.restirDI.initialSamplingParams);
	m_ImportanceSamplingContext->GetReSTIRDIContext().SetResamplingMode(m_BridgeParameters.userSettings.restirDI.resamplingMode);
	m_ImportanceSamplingContext->GetReSTIRDIContext().SetTemporalResamplingParameters(m_BridgeParameters.userSettings.restirDI.temporalResamplingParams);
	m_ImportanceSamplingContext->GetReSTIRDIContext().SetSpatialResamplingParameters(m_BridgeParameters.userSettings.restirDI.spatialResamplingParams);
	m_ImportanceSamplingContext->GetReSTIRDIContext().SetShadingParameters(m_BridgeParameters.userSettings.restirDI.shadingParams);

	// ReSTIR GI
	m_ImportanceSamplingContext->GetReSTIRGIContext().SetFrameIndex(m_BridgeParameters.frameIndex);
	m_ImportanceSamplingContext->GetReSTIRGIContext().SetResamplingMode(m_BridgeParameters.userSettings.restirGI.resamplingMode);
	m_ImportanceSamplingContext->GetReSTIRGIContext().SetTemporalResamplingParameters(m_BridgeParameters.userSettings.restirGI.temporalResamplingParams);
	m_ImportanceSamplingContext->GetReSTIRGIContext().SetSpatialResamplingParameters(m_BridgeParameters.userSettings.restirGI.spatialResamplingParams);
	m_ImportanceSamplingContext->GetReSTIRGIContext().SetFinalShadingParameters(m_BridgeParameters.userSettings.restirGI.finalShadingParams);

	// ReGIR
	auto regirParams = m_BridgeParameters.userSettings.regir.regirDynamicParameters;
	regirParams.center = { m_BridgeParameters.cameraPosition.x, m_BridgeParameters.cameraPosition.y,m_BridgeParameters.cameraPosition.z };
	m_ImportanceSamplingContext->GetReGIRContext().SetDynamicParameters(regirParams);
}

void RtxdiPass::CreatePipelines(nvrhi::BindingLayoutHandle extraBindingLayout /*= nullptr*/, bool useRayQuery /*= true*/)
{
	const auto& reGIRParams = m_ImportanceSamplingContext->GetReGIRContext().GetReGIRStaticParameters();
	
	std::vector<donut::engine::ShaderMacro> regirMacros = { GetReGirMacro(reGIRParams) };

	m_PresampleLightsPass.Init(m_device, *m_shaderFactory, "app/RTXDI/PresampleLights.hlsl", "main", {}, m_bindingLayout, extraBindingLayout, m_bindlessLayout);
	m_PresampleEnvMapPass.Init(m_device, *m_shaderFactory, "app/RTXDI/PresampleEnvironmentMap.hlsl", "main", {}, m_bindingLayout, extraBindingLayout, m_bindlessLayout);
	if (reGIRParams.Mode != rtxdi::ReGIRMode::Disabled)
	{
		m_PresampleReGIRPass.Init(m_device, *m_shaderFactory, "app/RTXDI/PresampleReGIR.hlsl", "main", regirMacros, m_bindingLayout, extraBindingLayout, m_bindlessLayout);
	}
	
	m_GenerateInitialSamplesPass.Init(m_device, *m_shaderFactory, "app/RTXDI/GenerateInitialSamples.hlsl", 
		regirMacros, useRayQuery, RTXDI_SCREEN_SPACE_GROUP_SIZE, m_bindingLayout, extraBindingLayout, m_bindlessLayout);
	m_SpatialResamplingPass.Init(m_device, *m_shaderFactory, "app/RTXDI/SpatialResampling.hlsl",
		{}, useRayQuery, RTXDI_SCREEN_SPACE_GROUP_SIZE, m_bindingLayout, extraBindingLayout, m_bindlessLayout);
	m_TemporalResamplingPass.Init(m_device, *m_shaderFactory, "app/RTXDI/TemporalResampling.hlsl",
		{}, useRayQuery, RTXDI_SCREEN_SPACE_GROUP_SIZE, m_bindingLayout, extraBindingLayout, m_bindlessLayout);
	
	m_FinalSamplingPass.Init(m_device, *m_shaderFactory, "app/RTXDI/DIFinalShading.hlsl", "main", { { "USE_RAY_QUERY", "1" } }, m_bindingLayout, extraBindingLayout, m_bindlessLayout);
	
	m_GISpatialResamplingPass.Init(m_device, *m_shaderFactory, "app/RTXDI/GISpatialResampling.hlsl",
		{}, useRayQuery, RTXDI_SCREEN_SPACE_GROUP_SIZE, m_bindingLayout, extraBindingLayout, m_bindlessLayout);
	m_GITemporalResamplingPass.Init(m_device, *m_shaderFactory, "app/RTXDI/GITemporalResampling.hlsl",
		{}, useRayQuery, RTXDI_SCREEN_SPACE_GROUP_SIZE, m_bindingLayout, extraBindingLayout, m_bindlessLayout);
	m_GIFinalShadingPass.Init(m_device, *m_shaderFactory, "app/RTXDI/GIFinalShading.hlsl",
		{}, useRayQuery, RTXDI_SCREEN_SPACE_GROUP_SIZE, m_bindingLayout, extraBindingLayout, m_bindlessLayout);
    m_FusedDIGIFinalShadingPass.Init(m_device, *m_shaderFactory, "app/RTXDI/FusedDIGIFinalShading.hlsl",
        {}, useRayQuery, RTXDI_SCREEN_SPACE_GROUP_SIZE, m_bindingLayout, extraBindingLayout, m_bindlessLayout);
}

void RtxdiPass::CreateBindingSet(const RenderTargets& renderTargets)
{
	for (int currentFrame = 0; currentFrame <= 1; currentFrame++)
	{
		nvrhi::BindingSetDesc bindingSetDesc;
		bindingSetDesc.bindings = {
			// RTXDI resources
			nvrhi::BindingSetItem::StructuredBuffer_SRV(21, m_rtxdiResources->LightDataBuffer),
			nvrhi::BindingSetItem::TypedBuffer_SRV(22, m_rtxdiResources->NeighborOffsetsBuffer),
			nvrhi::BindingSetItem::TypedBuffer_SRV(23, m_rtxdiResources->LightIndexMappingBuffer),
			nvrhi::BindingSetItem::Texture_SRV(25, m_rtxdiResources->LocalLightPdfTexture),
			nvrhi::BindingSetItem::StructuredBuffer_SRV(26, m_rtxdiResources->GeometryInstanceToLightBuffer),
			
			// Render targets
			nvrhi::BindingSetItem::StructuredBuffer_UAV(13, m_rtxdiResources->LightReservoirBuffer),
			nvrhi::BindingSetItem::StructuredBuffer_UAV(14, m_rtxdiResources->GIReservoirBuffer),
			nvrhi::BindingSetItem::TypedBuffer_UAV(15, m_rtxdiResources->RisBuffer),
			nvrhi::BindingSetItem::TypedBuffer_UAV(16, m_rtxdiResources->RisLightDataBuffer),
			
			nvrhi::BindingSetItem::ConstantBuffer(5, m_rtxdiConstantBuffer),

			nvrhi::BindingSetItem::Sampler(4, m_CommonRenderPasses->m_LinearWrapSampler)
		};

		const nvrhi::BindingSetHandle bindingSet = m_device->createBindingSet(bindingSetDesc, m_bindingLayout);
		if (currentFrame)
			m_bindingSet = bindingSet;
		else
			m_PrevBindingSet = bindingSet;
	}
}

void RtxdiPass::Reset()
{
	m_ImportanceSamplingContext = nullptr;
	m_rtxdiResources = nullptr;
	m_LocalLightPdfMipmapPass = nullptr;
	m_bindingSet = nullptr;
}

void RtxdiPass::PrepareResources(
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
    std::shared_ptr<ShaderDebug> shaderDebug)
{
    m_Scene = scene;
    m_BridgeParameters = bridgeParams;

    CheckContextStaticParameters();

    if (!m_ImportanceSamplingContext)
    {
        // Set static parameters for ReSTIR DI, ReSTIR GI and ReGIR
        rtxdi::ImportanceSamplingContext_StaticParameters staticParameters = {};
        staticParameters.renderWidth = m_BridgeParameters.frameDims.x;
        staticParameters.renderHeight = m_BridgeParameters.frameDims.y;
        staticParameters.regirStaticParams = m_BridgeParameters.userSettings.regir.regirStaticParams;

        m_ImportanceSamplingContext = std::make_unique<rtxdi::ImportanceSamplingContext>(staticParameters);

        // RTXDI context settings affect the shader permutations
        CreatePipelines(extraBindingLayout, true);
    }

    UpdateContextDynamicParameters();

    if (!m_PrepareLightsPass)
    {
        m_PrepareLightsPass = std::make_unique<PrepareLightsPass>(m_device, m_shaderFactory, m_CommonRenderPasses, nullptr, materialsBaker, ommBaker, subInstanceDataBuffer, m_bindlessLayout, shaderDebug);
        m_PrepareLightsPass->CreatePipeline();
    }

    m_PrepareLightsPass->SetScene(m_Scene, envMap, envMapSceneParams);

    //Check if resources have changed
    bool envMapPresent = envMap != nullptr;
    uint32_t numEmissiveMeshes, numEmissiveTriangles = 0;
    m_PrepareLightsPass->CountLightsInScene(numEmissiveMeshes, numEmissiveTriangles);
    uint32_t numPrimitiveLights = uint32_t(m_Scene->GetSceneGraph()->GetLights().size());
    uint32_t numGeometryInstances = uint32_t(m_Scene->GetSceneGraph()->GetGeometryInstancesCount());

    if (m_rtxdiResources && (
        numEmissiveMeshes > m_rtxdiResources->GetMaxEmissiveMeshes() ||
        numEmissiveTriangles > m_rtxdiResources->GetMaxEmissiveTriangles() ||
        numPrimitiveLights > m_rtxdiResources->GetMaxPrimitiveLights() ||
        numGeometryInstances > m_rtxdiResources->GetMaxGeometryInstances()))
    {
        m_rtxdiResources = nullptr;
    }

    bool rtxdiResourceCreated = false;

    if (!m_rtxdiResources)
    {
        uint32_t meshAllocationQuantum = 128;
        uint32_t triangleAllocationQuantum = 1024;
        uint32_t primitiveAllocationQuantum = 128;

        m_rtxdiResources = std::make_shared<RtxdiResources>(
            m_device,
            m_ImportanceSamplingContext->GetReSTIRDIContext(),
            m_ImportanceSamplingContext->GetRISBufferSegmentAllocator(),
            (numEmissiveMeshes + meshAllocationQuantum - 1) & ~(meshAllocationQuantum - 1),
            (numEmissiveTriangles + triangleAllocationQuantum - 1) & ~(triangleAllocationQuantum - 1),
            (numPrimitiveLights + primitiveAllocationQuantum - 1) & ~(primitiveAllocationQuantum - 1),
            numGeometryInstances);

        rtxdiResourceCreated = true;
    }

    if (rtxdiResourceCreated)
    {
        m_PrepareLightsPass->CreateBindingSet(*m_rtxdiResources, renderTargets);
        m_rtxdiResources->InitializeNeighborOffsets(commandList, m_ImportanceSamplingContext->GetNeighborOffsetCount());
        m_LocalLightPdfMipmapPass = nullptr;
    }

    if (rtxdiResourceCreated || m_bindingSet == nullptr)
    {
        CreateBindingSet(renderTargets);
    }
}

void RtxdiPass::BeginFrame(
    nvrhi::CommandListHandle commandList,
    const RenderTargets & renderTargets,
    const nvrhi::BindingLayoutHandle extraBindingLayout,
    nvrhi::BindingSetHandle extraBindingSet )
{
	// Light preparation is only needed for ReStirDI and ReGIR
	if (m_BridgeParameters.usingLightSampling)
	{
		//This pass needs to happen before we fill the constant buffers 
		commandList->beginMarker("Prepare Light");
		RTXDI_LightBufferParameters lightBufferParams = m_PrepareLightsPass->Process(commandList);
		commandList->endMarker();

		m_ImportanceSamplingContext->SetLightBufferParams(lightBufferParams);
	}

	FillConstants(commandList);

	// In cases where the RTXDI context is only needed for ReSTIR GI we can skip pdf, presampling and ReGir passes
	if (!m_BridgeParameters.usingLightSampling)
		return;

	if (!m_LocalLightPdfMipmapPass)
	{
		m_LocalLightPdfMipmapPass = std::make_unique<GenerateMipsPass>(
			m_device,
			m_shaderFactory,
			nullptr,
			m_rtxdiResources->LocalLightPdfTexture);
	}

	commandList->beginMarker("GeneratePDFTextures");

	m_LocalLightPdfMipmapPass->Process(commandList);

	commandList->endMarker();

	// Pre-sample lights
	const auto lightBufferParams = m_ImportanceSamplingContext->GetLightBufferParameters();
	if (m_ImportanceSamplingContext->IsLocalLightPowerRISEnabled() && lightBufferParams.localLightBufferRegion.numLights > 0)
	{
		dm::int3 presampleDispatchSize = {
			dm::div_ceil(m_ImportanceSamplingContext->GetLocalLightRISBufferSegmentParams().tileSize, RTXDI_PRESAMPLING_GROUP_SIZE),
			int(m_ImportanceSamplingContext->GetLocalLightRISBufferSegmentParams().tileCount),
			1
		};

		nvrhi::utils::BufferUavBarrier(commandList, m_rtxdiResources->RisBuffer);

		ExecuteComputePass(commandList, m_PresampleLightsPass, "Pre-sample Lights", presampleDispatchSize, extraBindingSet);
	}

	if (lightBufferParams.environmentLightParams.lightPresent)
	{
		dm::int3 presampleDispatchSize = {
			dm::div_ceil(m_ImportanceSamplingContext->GetEnvironmentLightRISBufferSegmentParams().tileSize, RTXDI_PRESAMPLING_GROUP_SIZE),
			int(m_ImportanceSamplingContext->GetEnvironmentLightRISBufferSegmentParams().tileCount),
			1
		};

		nvrhi::utils::BufferUavBarrier(commandList, m_rtxdiResources->RisBuffer);

		ExecuteComputePass(commandList, m_PresampleEnvMapPass, "Pre-sample Envmap", presampleDispatchSize, extraBindingSet);
	}

	//Build ReGIR structure 
	if (m_ImportanceSamplingContext->IsReGIREnabled() && m_BridgeParameters.usingReGIR)
	{
		dm::int3 worldGridDispatchSize = {
			dm::div_ceil(m_ImportanceSamplingContext->GetReGIRContext().GetReGIRLightSlotCount(), RTXDI_GRID_BUILD_GROUP_SIZE), 1, 1 };
		ExecuteComputePass(commandList, m_PresampleReGIRPass, "Pre-sample ReGir", worldGridDispatchSize, extraBindingSet);
	}
}

void RtxdiPass::Execute(
	nvrhi::CommandListHandle commandList,
	nvrhi::BindingSetHandle extraBindingSet,
    bool skipFinal
)
{
	commandList->beginMarker("ReSTIR DI");

	auto& reSTIRDI = m_ImportanceSamplingContext->GetReSTIRDIContext();
	dm::int2 dispatchSize = { (int) reSTIRDI.GetStaticParameters().RenderWidth, (int)reSTIRDI.GetStaticParameters().RenderHeight };
	
	// Not implemented
	//if (reSTIRDI.GetResamplingMode() == rtxdi::ReSTIRDI_ResamplingMode::FusedSpatiotemporal)
	//{
	//	// TODO: combine initial, temporal, spatial and final sampling in one pass
	//       // In case this is implemented, probably no point in doing fused ReSTIR-DI and ReSTIR-GI sampling, so 
	//       // then remove skipFinal and related logic.
	//}
	//else
	{
		//Generate sample, pick re-sampling method, final sampling
		ExecuteRayTracingPass(commandList, m_GenerateInitialSamplesPass, "Generate Initial Samples", dispatchSize, extraBindingSet);

		if (reSTIRDI.GetResamplingMode() == rtxdi::ReSTIRDI_ResamplingMode::Temporal ||
			reSTIRDI.GetResamplingMode() == rtxdi::ReSTIRDI_ResamplingMode::TemporalAndSpatial)
		{
			nvrhi::utils::BufferUavBarrier(commandList, m_rtxdiResources->LightReservoirBuffer);
			ExecuteRayTracingPass(commandList, m_TemporalResamplingPass, "Temporal Re-sampling", dispatchSize, extraBindingSet);
		}
		
		if (reSTIRDI.GetResamplingMode() == rtxdi::ReSTIRDI_ResamplingMode::Spatial ||
			reSTIRDI.GetResamplingMode() == rtxdi::ReSTIRDI_ResamplingMode::TemporalAndSpatial)
		{
			nvrhi::utils::BufferUavBarrier(commandList, m_rtxdiResources->LightReservoirBuffer);
			ExecuteRayTracingPass(commandList, m_SpatialResamplingPass, "Spatial Re-sampling", dispatchSize, extraBindingSet);

		}

        //Full screen light sampling pass
        nvrhi::utils::BufferUavBarrier(commandList, m_rtxdiResources->LightReservoirBuffer);

        if (!skipFinal)
        {
            dm::int3 screenSpaceDispatchSize = {
                ((int)reSTIRDI.GetStaticParameters().RenderWidth + RTXDI_SCREEN_SPACE_GROUP_SIZE - 1) / RTXDI_SCREEN_SPACE_GROUP_SIZE,
                ((int)reSTIRDI.GetStaticParameters().RenderHeight + RTXDI_SCREEN_SPACE_GROUP_SIZE - 1) / RTXDI_SCREEN_SPACE_GROUP_SIZE,
                1 };

            ExecuteComputePass(commandList, m_FinalSamplingPass, "Final Sampling", screenSpaceDispatchSize, extraBindingSet);
        }
    }
	commandList->endMarker();
}

void RtxdiPass::FillConstants(nvrhi::CommandListHandle commandList)
{
	// Set the ReGir center and the camera position 
	RtxdiBridgeConstants bridgeConstants{};
	bridgeConstants.lightBufferParams = m_ImportanceSamplingContext->GetLightBufferParameters();
	bridgeConstants.localLightsRISBufferSegmentParams = m_ImportanceSamplingContext->GetLocalLightRISBufferSegmentParams();
	bridgeConstants.environmentLightRISBufferSegmentParams = m_ImportanceSamplingContext->GetEnvironmentLightRISBufferSegmentParams();
	bridgeConstants.runtimeParams = m_ImportanceSamplingContext->GetReSTIRDIContext().GetRuntimeParams();

	FillSharedConstants(bridgeConstants);
	FillDIConstants(bridgeConstants.restirDI);
	FillGIConstants(bridgeConstants.restirGI);
	FillReGIRConstant(bridgeConstants.regir);
	FillReGirIndirectConstants(bridgeConstants.regirIndirect);

	commandList->writeBuffer(m_rtxdiConstantBuffer, &bridgeConstants, sizeof(RtxdiBridgeConstants));
}

void RtxdiPass::FillSharedConstants(struct RtxdiBridgeConstants& bridgeConstants) const
{
	bridgeConstants.frameIndex = m_BridgeParameters.frameIndex;
	bridgeConstants.frameDim = m_BridgeParameters.frameDims;
	bridgeConstants.rayEpsilon = m_BridgeParameters.userSettings.rayEpsilon;
	bridgeConstants.localLightPdfTextureSize = uint2(m_rtxdiResources->LocalLightPdfTexture->getDesc().width, m_rtxdiResources->LocalLightPdfTexture->getDesc().height);
	bridgeConstants.localLightPdfLastMipLevel = m_rtxdiResources->LocalLightPdfTexture->getDesc().mipLevels - 1 ;
	bridgeConstants.maxLights = uint32_t(m_rtxdiResources->LightDataBuffer->getDesc().byteSize / (sizeof(PolymorphicLightInfo) * 2));
	bridgeConstants.reStirGIVaryAgeThreshold = m_BridgeParameters.userSettings.reStirGIVaryAgeThreshold;

	const auto& giSampleMode = m_ImportanceSamplingContext->GetReSTIRGIContext().GetResamplingMode();
	bridgeConstants.reStirGIEnableTemporalResampling = (giSampleMode == rtxdi::ReSTIRGI_ResamplingMode::Temporal) || (giSampleMode == rtxdi::ReSTIRGI_ResamplingMode::TemporalAndSpatial) ? 1 : 0;
}

void RtxdiPass::FillDIConstants(ReSTIRDI_Parameters& diParams)
{
	const auto& reSTIRDI = m_ImportanceSamplingContext->GetReSTIRDIContext();
	const auto& lightBufferParams = m_ImportanceSamplingContext->GetLightBufferParameters();

	diParams.reservoirBufferParams = reSTIRDI.GetReservoirBufferParameters();
	diParams.bufferIndices = reSTIRDI.GetBufferIndices();
	diParams.initialSamplingParams = reSTIRDI.GetInitialSamplingParameters();
	diParams.initialSamplingParams.environmentMapImportanceSampling = lightBufferParams.environmentLightParams.lightPresent;
	if (!diParams.initialSamplingParams.environmentMapImportanceSampling)
		diParams.initialSamplingParams.numPrimaryEnvironmentSamples = 0;
	diParams.temporalResamplingParams = reSTIRDI.GetTemporalResamplingParameters();
	diParams.spatialResamplingParams = reSTIRDI.GetSpatialResamplingParameters();
	diParams.shadingParams = reSTIRDI.GetShadingParameters();
}

void RtxdiPass::FillGIConstants(ReSTIRGI_Parameters& giParams)
{
	const auto& reSTIRGI = m_ImportanceSamplingContext->GetReSTIRGIContext();

	giParams.reservoirBufferParams = reSTIRGI.GetReservoirBufferParameters();
	giParams.bufferIndices = reSTIRGI.GetBufferIndices();
	giParams.temporalResamplingParams = reSTIRGI.GetTemporalResamplingParameters();
	giParams.spatialResamplingParams = reSTIRGI.GetSpatialResamplingParameters();
	giParams.finalShadingParams = reSTIRGI.GetFinalShadingParameters();
}


void RtxdiPass::FillReGIRConstant(ReGIR_Parameters& regirParams)
{
	const auto& regir = m_ImportanceSamplingContext->GetReGIRContext();
	auto staticParams = regir.GetReGIRStaticParameters();
	auto dynamicParams = regir.GetReGIRDynamicParameters();
	auto gridParams = regir.GetReGIRGridCalculatedParameters();
	auto onionParams = regir.GetReGIROnionCalculatedParameters();

	regirParams.gridParams.cellsX = staticParams.gridParameters.GridSize.x;
	regirParams.gridParams.cellsY = staticParams.gridParameters.GridSize.y;
	regirParams.gridParams.cellsZ = staticParams.gridParameters.GridSize.z;

	regirParams.commonParams.numRegirBuildSamples = dynamicParams.regirNumBuildSamples;
	regirParams.commonParams.risBufferOffset = regir.GetReGIRCellOffset();
	regirParams.commonParams.lightsPerCell = staticParams.LightsPerCell;
	regirParams.commonParams.centerX = dynamicParams.center.x;
	regirParams.commonParams.centerY = dynamicParams.center.y;
	regirParams.commonParams.centerZ = dynamicParams.center.z;
	regirParams.commonParams.cellSize = (staticParams.Mode == rtxdi::ReGIRMode::Onion)
		? dynamicParams.regirCellSize * 0.5f // Onion operates with radii, while "size" feels more like diameter
		: dynamicParams.regirCellSize;
	regirParams.commonParams.localLightSamplingFallbackMode = static_cast<uint32_t>(dynamicParams.fallbackSamplingMode);
	regirParams.commonParams.localLightPresamplingMode = static_cast<uint32_t>(dynamicParams.presamplingMode);
	regirParams.commonParams.samplingJitter = std::max(0.f, dynamicParams.regirSamplingJitter * 2.f);
	regirParams.onionParams.cubicRootFactor = onionParams.regirOnionCubicRootFactor;
	regirParams.onionParams.linearFactor = onionParams.regirOnionLinearFactor;
	regirParams.onionParams.numLayerGroups = uint32_t(onionParams.regirOnionLayers.size());

	assert(onionParams.regirOnionLayers.size() <= RTXDI_ONION_MAX_LAYER_GROUPS);
	for (int group = 0; group < int(onionParams.regirOnionLayers.size()); group++)
	{
		regirParams.onionParams.layers[group] = onionParams.regirOnionLayers[group];
		regirParams.onionParams.layers[group].innerRadius *= regirParams.commonParams.cellSize;
		regirParams.onionParams.layers[group].outerRadius *= regirParams.commonParams.cellSize;
	}

	assert(onionParams.regirOnionRings.size() <= RTXDI_ONION_MAX_RINGS);
	for (int n = 0; n < int(onionParams.regirOnionRings.size()); n++)
	{
		regirParams.onionParams.rings[n] = onionParams.regirOnionRings[n];
	}

	regirParams.onionParams.cubicRootFactor = regir.GetReGIROnionCalculatedParameters().regirOnionCubicRootFactor;
}


void RtxdiPass::FillReGirIndirectConstants(ReGirIndirectConstants& regirIndirectConstants)
{
	regirIndirectConstants.numIndirectSamples = m_BridgeParameters.userSettings.regirIndirect.numIndirectSamples;
}

void RtxdiPass::ExecuteGI(nvrhi::CommandListHandle commandList, nvrhi::BindingSetHandle extraBindingSet, bool skipFinal)
{
	commandList->beginMarker("ReSTIR GI");

	auto& reSTIRGI = m_ImportanceSamplingContext->GetReSTIRGIContext();

	dm::int2 dispatchSize = { (int)reSTIRGI.GetStaticParams().RenderWidth, (int)reSTIRGI.GetStaticParams().RenderHeight };

	ExecuteRayTracingPass(commandList, m_GITemporalResamplingPass, "Temporal Resampling", dispatchSize, extraBindingSet);

	if (reSTIRGI.GetResamplingMode() == rtxdi::ReSTIRGI_ResamplingMode::Spatial || reSTIRGI.GetResamplingMode() == rtxdi::ReSTIRGI_ResamplingMode::TemporalAndSpatial)
	{
		nvrhi::utils::BufferUavBarrier(commandList, m_rtxdiResources->GIReservoirBuffer);

		ExecuteRayTracingPass(commandList, m_GISpatialResamplingPass, "Spatial Resampling", dispatchSize, extraBindingSet);
	}

	nvrhi::utils::BufferUavBarrier(commandList, m_rtxdiResources->GIReservoirBuffer);

    if (!skipFinal)
	    ExecuteRayTracingPass(commandList, m_GIFinalShadingPass, "Final Shading", dispatchSize, extraBindingSet);

	commandList->endMarker(); // ReSTIR GI
}

void RtxdiPass::ExecuteFusedDIGIFinal(nvrhi::CommandListHandle commandList, nvrhi::BindingSetHandle extraBindingSet)
{
	auto& reSTIRDI = m_ImportanceSamplingContext->GetReSTIRDIContext();
	dm::int2 dispatchSize = { (int)reSTIRDI.GetStaticParameters().RenderWidth, (int)reSTIRDI.GetStaticParameters().RenderHeight };

    ExecuteRayTracingPass(commandList, m_FusedDIGIFinalShadingPass, "Fused DI GI Final Shading", dispatchSize, extraBindingSet);
}

void RtxdiPass::EndFrame(){}

void RtxdiPass::ExecuteComputePass(
	nvrhi::CommandListHandle& commandList,
	ComputePass& pass,
	const char* passName,
	dm::int3 dispatchSize,
	nvrhi::BindingSetHandle extraBindingSet /*= nullptr*/)
{
	commandList->beginMarker(passName);
    
    uint4 unusedPushConstants = {0,0,0,0};  // shared bindings require them
	pass.Execute(commandList, dispatchSize.x, dispatchSize.y, dispatchSize.z, m_bindingSet,
		extraBindingSet, m_Scene->GetDescriptorTable(), &unusedPushConstants, sizeof(unusedPushConstants));

	commandList->endMarker();
}

void RtxdiPass::ExecuteRayTracingPass(
	nvrhi::CommandListHandle& commandList, 
	RayTracingPass& pass, 
	const char* passName, 
	dm::int2 dispatchSize, 
	nvrhi::IBindingSet* extraBindingSet /* = nullptr */
)
{
	commandList->beginMarker(passName);
	
    uint4 unusedPushConstants = { 0,0,0,0 };  // shared bindings require them
	pass.Execute(commandList, dispatchSize.x, dispatchSize.y, m_bindingSet, 
		extraBindingSet, m_Scene->GetDescriptorTable(), &unusedPushConstants, sizeof(unusedPushConstants));

	commandList->endMarker();
}

donut::engine::ShaderMacro RtxdiPass::GetReGirMacro(const rtxdi::ReGIRStaticParameters& regirParameters)
{
	std::string regirMode;

	switch (regirParameters.Mode)
	{
	case rtxdi::ReGIRMode::Disabled :
		regirMode = "RTXDI_REGIR_DISABLED";
		break;
	case rtxdi::ReGIRMode::Grid :
		regirMode = "RTXDI_REGIR_GRID";
		break;
	case rtxdi::ReGIRMode::Onion :
		regirMode = "RTXDI_REGIR_ONION";
		break;
	}

	return { "RTXDI_REGIR_MODE", regirMode };
}
