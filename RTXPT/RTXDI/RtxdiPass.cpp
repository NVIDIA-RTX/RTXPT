/*
* Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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
		m_Device(device),
		m_ShaderFactory(shaderFactory),
		m_CommonRenderPasses(commonRenderPasses),
		m_BindlessLayout(bindlessLayout),
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
	m_BindingLayout = m_Device->createBindingLayout(layoutDesc);

	m_RtxdiConstantBuffer = m_Device->createBuffer(nvrhi::utils::CreateVolatileConstantBufferDesc(sizeof(RtxdiBridgeConstants), "RtxdiBridgeConstants", 16));
}

RtxdiPass::~RtxdiPass(){}

// Check for changes in the static parameters, these will require the importance sampling context to be recreated
void RtxdiPass::CheckContextStaticParameters()
{
	if (m_ImportanceSamplingContext != nullptr)
	{
		auto& reGIRContext = m_ImportanceSamplingContext->getReGIRContext();

		bool needsReset = false;
		if (reGIRContext.getReGIRStaticParameters().Mode != m_BridgeParameters.userSettings.regir.regirStaticParams.Mode)
			needsReset = true;
		if (reGIRContext.getReGIRStaticParameters().LightsPerCell != m_BridgeParameters.userSettings.regir.regirStaticParams.LightsPerCell)
			needsReset = true;

		if (needsReset)
			Reset();
	}
}

void RtxdiPass::UpdateContextDynamicParameters()
{
	// ReSTIR DI
	m_ImportanceSamplingContext->getReSTIRDIContext().setFrameIndex(m_BridgeParameters.frameIndex);
	m_ImportanceSamplingContext->getReSTIRDIContext().setInitialSamplingParameters(m_BridgeParameters.userSettings.restirDI.initialSamplingParams);
	m_ImportanceSamplingContext->getReSTIRDIContext().setResamplingMode(m_BridgeParameters.userSettings.restirDI.resamplingMode);
	m_ImportanceSamplingContext->getReSTIRDIContext().setTemporalResamplingParameters(m_BridgeParameters.userSettings.restirDI.temporalResamplingParams);
	m_ImportanceSamplingContext->getReSTIRDIContext().setSpatialResamplingParameters(m_BridgeParameters.userSettings.restirDI.spatialResamplingParams);
	m_ImportanceSamplingContext->getReSTIRDIContext().setShadingParameters(m_BridgeParameters.userSettings.restirDI.shadingParams);

	// ReSTIR GI
	m_ImportanceSamplingContext->getReSTIRGIContext().setFrameIndex(m_BridgeParameters.frameIndex);
	m_ImportanceSamplingContext->getReSTIRGIContext().setResamplingMode(m_BridgeParameters.userSettings.restirGI.resamplingMode);
	m_ImportanceSamplingContext->getReSTIRGIContext().setTemporalResamplingParameters(m_BridgeParameters.userSettings.restirGI.temporalResamplingParams);
	m_ImportanceSamplingContext->getReSTIRGIContext().setSpatialResamplingParameters(m_BridgeParameters.userSettings.restirGI.spatialResamplingParams);
	m_ImportanceSamplingContext->getReSTIRGIContext().setFinalShadingParameters(m_BridgeParameters.userSettings.restirGI.finalShadingParams);

	// ReGIR
	auto regirParams = m_BridgeParameters.userSettings.regir.regirDynamicParameters;
	regirParams.center = { m_BridgeParameters.cameraPosition.x, m_BridgeParameters.cameraPosition.y,m_BridgeParameters.cameraPosition.z };
	m_ImportanceSamplingContext->getReGIRContext().setDynamicParameters(regirParams);
}

void RtxdiPass::CreatePipelines(nvrhi::BindingLayoutHandle extraBindingLayout /*= nullptr*/, bool useRayQuery /*= true*/)
{
	const auto& reGIRParams = m_ImportanceSamplingContext->getReGIRContext().getReGIRStaticParameters();
	
	std::vector<donut::engine::ShaderMacro> regirMacros = { GetReGirMacro(reGIRParams) };

	m_PresampleLightsPass.Init(m_Device, *m_ShaderFactory, "app/RTXDI/PresampleLights.hlsl", {}, m_BindingLayout, extraBindingLayout, m_BindlessLayout);
	m_PresampleEnvMapPass.Init(m_Device, *m_ShaderFactory, "app/RTXDI/PresampleEnvironmentMap.hlsl", {}, m_BindingLayout, extraBindingLayout, m_BindlessLayout);
	if (reGIRParams.Mode != rtxdi::ReGIRMode::Disabled)
	{
		m_PresampleReGIRPass.Init(m_Device, *m_ShaderFactory, "app/RTXDI/PresampleReGIR.hlsl", regirMacros, m_BindingLayout, extraBindingLayout, m_BindlessLayout);
	}
	
	m_GenerateInitialSamplesPass.Init(m_Device, *m_ShaderFactory, "app/RTXDI/GenerateInitialSamples.hlsl", 
		regirMacros, useRayQuery, RTXDI_SCREEN_SPACE_GROUP_SIZE, m_BindingLayout, extraBindingLayout, m_BindlessLayout);
	m_SpatialResamplingPass.Init(m_Device, *m_ShaderFactory, "app/RTXDI/SpatialResampling.hlsl",
		{}, useRayQuery, RTXDI_SCREEN_SPACE_GROUP_SIZE, m_BindingLayout, extraBindingLayout, m_BindlessLayout);
	m_TemporalResamplingPass.Init(m_Device, *m_ShaderFactory, "app/RTXDI/TemporalResampling.hlsl",
		{}, useRayQuery, RTXDI_SCREEN_SPACE_GROUP_SIZE, m_BindingLayout, extraBindingLayout, m_BindlessLayout);
	
	m_FinalSamplingPass.Init(m_Device, *m_ShaderFactory, "app/RTXDI/DIFinalShading.hlsl", { { "USE_RAY_QUERY", "1" } }, m_BindingLayout, extraBindingLayout, m_BindlessLayout);
	
	m_GISpatialResamplingPass.Init(m_Device, *m_ShaderFactory, "app/RTXDI/GISpatialResampling.hlsl",
		{}, useRayQuery, RTXDI_SCREEN_SPACE_GROUP_SIZE, m_BindingLayout, extraBindingLayout, m_BindlessLayout);
	m_GITemporalResamplingPass.Init(m_Device, *m_ShaderFactory, "app/RTXDI/GITemporalResampling.hlsl",
		{}, useRayQuery, RTXDI_SCREEN_SPACE_GROUP_SIZE, m_BindingLayout, extraBindingLayout, m_BindlessLayout);
	m_GIFinalShadingPass.Init(m_Device, *m_ShaderFactory, "app/RTXDI/GIFinalShading.hlsl",
		{}, useRayQuery, RTXDI_SCREEN_SPACE_GROUP_SIZE, m_BindingLayout, extraBindingLayout, m_BindlessLayout);
    m_FusedDIGIFinalShadingPass.Init(m_Device, *m_ShaderFactory, "app/RTXDI/FusedDIGIFinalShading.hlsl",
        {}, useRayQuery, RTXDI_SCREEN_SPACE_GROUP_SIZE, m_BindingLayout, extraBindingLayout, m_BindlessLayout);
}

void RtxdiPass::CreateBindingSet(const RenderTargets& renderTargets)
{
	for (int currentFrame = 0; currentFrame <= 1; currentFrame++)
	{
		nvrhi::BindingSetDesc bindingSetDesc;
		bindingSetDesc.bindings = {
			// RTXDI resources
			nvrhi::BindingSetItem::StructuredBuffer_SRV(21, m_RtxdiResources->LightDataBuffer),
			nvrhi::BindingSetItem::TypedBuffer_SRV(22, m_RtxdiResources->NeighborOffsetsBuffer),
			nvrhi::BindingSetItem::TypedBuffer_SRV(23, m_RtxdiResources->LightIndexMappingBuffer),
			nvrhi::BindingSetItem::Texture_SRV(25, m_RtxdiResources->LocalLightPdfTexture),
			nvrhi::BindingSetItem::StructuredBuffer_SRV(26, m_RtxdiResources->GeometryInstanceToLightBuffer),
			
			// Render targets
			nvrhi::BindingSetItem::StructuredBuffer_UAV(13, m_RtxdiResources->LightReservoirBuffer),
			nvrhi::BindingSetItem::StructuredBuffer_UAV(14, m_RtxdiResources->GIReservoirBuffer),
			nvrhi::BindingSetItem::TypedBuffer_UAV(15, m_RtxdiResources->RisBuffer),
			nvrhi::BindingSetItem::TypedBuffer_UAV(16, m_RtxdiResources->RisLightDataBuffer),
			
			nvrhi::BindingSetItem::ConstantBuffer(5, m_RtxdiConstantBuffer),

			nvrhi::BindingSetItem::Sampler(4, m_CommonRenderPasses->m_LinearWrapSampler)
		};

		const nvrhi::BindingSetHandle bindingSet = m_Device->createBindingSet(bindingSetDesc, m_BindingLayout);
		if (currentFrame)
			m_BindingSet = bindingSet;
		else
			m_PrevBindingSet = bindingSet;
	}
}

void RtxdiPass::Reset()
{
	m_ImportanceSamplingContext = nullptr;
	m_RtxdiResources = nullptr;
	m_LocalLightPdfMipmapPass = nullptr;
	m_BindingSet = nullptr;
}

void RtxdiPass::PrepareResources(
    nvrhi::CommandListHandle commandList,
    const RenderTargets& renderTargets,
    std::shared_ptr<EnvMapBaker> envMap,
    EnvMapSceneParams envMapSceneParams,
    const std::shared_ptr<donut::engine::ExtendedScene> scene,
    const RtxdiBridgeParameters& bridgeParams,
    const nvrhi::BindingLayoutHandle extraBindingLayout)
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
        m_PrepareLightsPass = std::make_unique<PrepareLightsPass>(m_Device, m_ShaderFactory, m_CommonRenderPasses, nullptr, m_BindlessLayout);
        m_PrepareLightsPass->CreatePipeline();
    }

    m_PrepareLightsPass->SetScene(m_Scene, envMap, envMapSceneParams);

    //Check if resources have changed
    bool envMapPresent = envMap != nullptr;
    uint32_t numEmissiveMeshes, numEmissiveTriangles = 0;
    m_PrepareLightsPass->CountLightsInScene(numEmissiveMeshes, numEmissiveTriangles);
    uint32_t numPrimitiveLights = uint32_t(m_Scene->GetSceneGraph()->GetLights().size());
    uint32_t numGeometryInstances = uint32_t(m_Scene->GetSceneGraph()->GetGeometryInstancesCount());

    if (m_RtxdiResources && (
        numEmissiveMeshes > m_RtxdiResources->GetMaxEmissiveMeshes() ||
        numEmissiveTriangles > m_RtxdiResources->GetMaxEmissiveTriangles() ||
        numPrimitiveLights > m_RtxdiResources->GetMaxPrimitiveLights() ||
        numGeometryInstances > m_RtxdiResources->GetMaxGeometryInstances()))
    {
        m_RtxdiResources = nullptr;
    }

    bool rtxdiResourceCreated = false;

    if (!m_RtxdiResources)
    {
        uint32_t meshAllocationQuantum = 128;
        uint32_t triangleAllocationQuantum = 1024;
        uint32_t primitiveAllocationQuantum = 128;

        m_RtxdiResources = std::make_shared<RtxdiResources>(
            m_Device,
            m_ImportanceSamplingContext->getReSTIRDIContext(),
            m_ImportanceSamplingContext->getRISBufferSegmentAllocator(),
            (numEmissiveMeshes + meshAllocationQuantum - 1) & ~(meshAllocationQuantum - 1),
            (numEmissiveTriangles + triangleAllocationQuantum - 1) & ~(triangleAllocationQuantum - 1),
            (numPrimitiveLights + primitiveAllocationQuantum - 1) & ~(primitiveAllocationQuantum - 1),
            numGeometryInstances);

        rtxdiResourceCreated = true;
    }

    if (rtxdiResourceCreated)
    {
        m_PrepareLightsPass->CreateBindingSet(*m_RtxdiResources, renderTargets);
        m_RtxdiResources->InitializeNeighborOffsets(commandList, m_ImportanceSamplingContext->getNeighborOffsetCount());
        m_LocalLightPdfMipmapPass = nullptr;
    }

    if (rtxdiResourceCreated || m_BindingSet == nullptr)
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

		m_ImportanceSamplingContext->setLightBufferParams(lightBufferParams);
	}

	FillConstants(commandList);

	// In cases where the RTXDI context is only needed for ReSTIR GI we can skip pdf, presampling and ReGir passes
	if (!m_BridgeParameters.usingLightSampling)
		return;

	if (!m_LocalLightPdfMipmapPass)
	{
		m_LocalLightPdfMipmapPass = std::make_unique<GenerateMipsPass>(
			m_Device,
			m_ShaderFactory,
			nullptr,
			m_RtxdiResources->LocalLightPdfTexture);
	}

	commandList->beginMarker("GeneratePDFTextures");

	m_LocalLightPdfMipmapPass->Process(commandList);

	commandList->endMarker();

	// Pre-sample lights
	const auto lightBufferParams = m_ImportanceSamplingContext->getLightBufferParameters();
	if (m_ImportanceSamplingContext->isLocalLightPowerRISEnabled() && lightBufferParams.localLightBufferRegion.numLights > 0)
	{
		dm::int3 presampleDispatchSize = {
			dm::div_ceil(m_ImportanceSamplingContext->getLocalLightRISBufferSegmentParams().tileSize, RTXDI_PRESAMPLING_GROUP_SIZE),
			int(m_ImportanceSamplingContext->getLocalLightRISBufferSegmentParams().tileCount),
			1
		};

		nvrhi::utils::BufferUavBarrier(commandList, m_RtxdiResources->RisBuffer);

		ExecuteComputePass(commandList, m_PresampleLightsPass, "Pre-sample Lights", presampleDispatchSize, extraBindingSet);
	}

	if (lightBufferParams.environmentLightParams.lightPresent)
	{
		dm::int3 presampleDispatchSize = {
			dm::div_ceil(m_ImportanceSamplingContext->getEnvironmentLightRISBufferSegmentParams().tileSize, RTXDI_PRESAMPLING_GROUP_SIZE),
			int(m_ImportanceSamplingContext->getEnvironmentLightRISBufferSegmentParams().tileCount),
			1
		};

		nvrhi::utils::BufferUavBarrier(commandList, m_RtxdiResources->RisBuffer);

		ExecuteComputePass(commandList, m_PresampleEnvMapPass, "Pre-sample Envmap", presampleDispatchSize, extraBindingSet);
	}

	//Build ReGIR structure 
	if (m_ImportanceSamplingContext->isReGIREnabled() && m_BridgeParameters.usingReGIR)
	{
		dm::int3 worldGridDispatchSize = {
			dm::div_ceil(m_ImportanceSamplingContext->getReGIRContext().getReGIRLightSlotCount(), RTXDI_GRID_BUILD_GROUP_SIZE), 1, 1 };
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

	auto& reSTIRDI = m_ImportanceSamplingContext->getReSTIRDIContext();
	dm::int2 dispatchSize = { (int) reSTIRDI.getStaticParameters().RenderWidth, (int)reSTIRDI.getStaticParameters().RenderHeight };
	
	// Not implemented
	//if (reSTIRDI.getResamplingMode() == rtxdi::ReSTIRDI_ResamplingMode::FusedSpatiotemporal)
	//{
	//	// TODO: combine initial, temporal, spatial and final sampling in one pass
	//       // In case this is implemented, probably no point in doing fused ReSTIR-DI and ReSTIR-GI sampling, so 
	//       // then remove skipFinal and related logic.
	//}
	//else
	{
		//Generate sample, pick re-sampling method, final sampling
		ExecuteRayTracingPass(commandList, m_GenerateInitialSamplesPass, "Generate Initial Samples", dispatchSize, extraBindingSet);

		if (reSTIRDI.getResamplingMode() == rtxdi::ReSTIRDI_ResamplingMode::Temporal ||
			reSTIRDI.getResamplingMode() == rtxdi::ReSTIRDI_ResamplingMode::TemporalAndSpatial)
		{
			nvrhi::utils::BufferUavBarrier(commandList, m_RtxdiResources->LightReservoirBuffer);
			ExecuteRayTracingPass(commandList, m_TemporalResamplingPass, "Temporal Re-sampling", dispatchSize, extraBindingSet);
		}
		
		if (reSTIRDI.getResamplingMode() == rtxdi::ReSTIRDI_ResamplingMode::Spatial ||
			reSTIRDI.getResamplingMode() == rtxdi::ReSTIRDI_ResamplingMode::TemporalAndSpatial)
		{
			nvrhi::utils::BufferUavBarrier(commandList, m_RtxdiResources->LightReservoirBuffer);
			ExecuteRayTracingPass(commandList, m_SpatialResamplingPass, "Spatial Re-sampling", dispatchSize, extraBindingSet);

		}

        //Full screen light sampling pass
        nvrhi::utils::BufferUavBarrier(commandList, m_RtxdiResources->LightReservoirBuffer);

        if (!skipFinal)
        {
            dm::int3 screenSpaceDispatchSize = {
                ((int)reSTIRDI.getStaticParameters().RenderWidth + RTXDI_SCREEN_SPACE_GROUP_SIZE - 1) / RTXDI_SCREEN_SPACE_GROUP_SIZE,
                ((int)reSTIRDI.getStaticParameters().RenderHeight + RTXDI_SCREEN_SPACE_GROUP_SIZE - 1) / RTXDI_SCREEN_SPACE_GROUP_SIZE,
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
	bridgeConstants.lightBufferParams = m_ImportanceSamplingContext->getLightBufferParameters();
	bridgeConstants.localLightsRISBufferSegmentParams = m_ImportanceSamplingContext->getLocalLightRISBufferSegmentParams();
	bridgeConstants.environmentLightRISBufferSegmentParams = m_ImportanceSamplingContext->getEnvironmentLightRISBufferSegmentParams();
	bridgeConstants.runtimeParams = m_ImportanceSamplingContext->getReSTIRDIContext().getRuntimeParams();

	FillSharedConstants(bridgeConstants);
	FillDIConstants(bridgeConstants.restirDI);
	FillGIConstants(bridgeConstants.restirGI);
	FillReGIRConstant(bridgeConstants.regir);
	FillReGirIndirectConstants(bridgeConstants.regirIndirect);

	commandList->writeBuffer(m_RtxdiConstantBuffer, &bridgeConstants, sizeof(RtxdiBridgeConstants));
}

void RtxdiPass::FillSharedConstants(struct RtxdiBridgeConstants& bridgeConstants) const
{
	bridgeConstants.frameIndex = m_BridgeParameters.frameIndex;
	bridgeConstants.frameDim = m_BridgeParameters.frameDims;
	bridgeConstants.rayEpsilon = m_BridgeParameters.userSettings.rayEpsilon;
	bridgeConstants.localLightPdfTextureSize = uint2(m_RtxdiResources->LocalLightPdfTexture->getDesc().width, m_RtxdiResources->LocalLightPdfTexture->getDesc().height);
	bridgeConstants.localLightPdfLastMipLevel = m_RtxdiResources->LocalLightPdfTexture->getDesc().mipLevels - 1 ;
	bridgeConstants.maxLights = uint32_t(m_RtxdiResources->LightDataBuffer->getDesc().byteSize / (sizeof(PolymorphicLightInfo) * 2));
	bridgeConstants.reStirGIVaryAgeThreshold = m_BridgeParameters.userSettings.reStirGIVaryAgeThreshold;

	const auto& giSampleMode = m_ImportanceSamplingContext->getReSTIRGIContext().getResamplingMode();
	bridgeConstants.reStirGIEnableTemporalResampling = (giSampleMode == rtxdi::ReSTIRGI_ResamplingMode::Temporal) || (giSampleMode == rtxdi::ReSTIRGI_ResamplingMode::TemporalAndSpatial) ? 1 : 0;
}

void RtxdiPass::FillDIConstants(ReSTIRDI_Parameters& diParams)
{
	const auto& reSTIRDI = m_ImportanceSamplingContext->getReSTIRDIContext();
	const auto& lightBufferParams = m_ImportanceSamplingContext->getLightBufferParameters();

	diParams.reservoirBufferParams = reSTIRDI.getReservoirBufferParameters();
	diParams.bufferIndices = reSTIRDI.getBufferIndices();
	diParams.initialSamplingParams = reSTIRDI.getInitialSamplingParameters();
	diParams.initialSamplingParams.environmentMapImportanceSampling = lightBufferParams.environmentLightParams.lightPresent;
	if (!diParams.initialSamplingParams.environmentMapImportanceSampling)
		diParams.initialSamplingParams.numPrimaryEnvironmentSamples = 0;
	diParams.temporalResamplingParams = reSTIRDI.getTemporalResamplingParameters();
	diParams.spatialResamplingParams = reSTIRDI.getSpatialResamplingParameters();
	diParams.shadingParams = reSTIRDI.getShadingParameters();
}

void RtxdiPass::FillGIConstants(ReSTIRGI_Parameters& giParams)
{
	const auto& reSTIRGI = m_ImportanceSamplingContext->getReSTIRGIContext();

	giParams.reservoirBufferParams = reSTIRGI.getReservoirBufferParameters();
	giParams.bufferIndices = reSTIRGI.getBufferIndices();
	giParams.temporalResamplingParams = reSTIRGI.getTemporalResamplingParameters();
	giParams.spatialResamplingParams = reSTIRGI.getSpatialResamplingParameters();
	giParams.finalShadingParams = reSTIRGI.getFinalShadingParameters();
}


void RtxdiPass::FillReGIRConstant(ReGIR_Parameters& regirParams)
{
	const auto& regir = m_ImportanceSamplingContext->getReGIRContext();
	auto staticParams = regir.getReGIRStaticParameters();
	auto dynamicParams = regir.getReGIRDynamicParameters();
	auto gridParams = regir.getReGIRGridCalculatedParameters();
	auto onionParams = regir.getReGIROnionCalculatedParameters();

	regirParams.gridParams.cellsX = staticParams.gridParameters.GridSize.x;
	regirParams.gridParams.cellsY = staticParams.gridParameters.GridSize.y;
	regirParams.gridParams.cellsZ = staticParams.gridParameters.GridSize.z;

	regirParams.commonParams.numRegirBuildSamples = dynamicParams.regirNumBuildSamples;
	regirParams.commonParams.risBufferOffset = regir.getReGIRCellOffset();
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

	regirParams.onionParams.cubicRootFactor = regir.getReGIROnionCalculatedParameters().regirOnionCubicRootFactor;
}


void RtxdiPass::FillReGirIndirectConstants(ReGirIndirectConstants& regirIndirectConstants)
{
	regirIndirectConstants.numIndirectSamples = m_BridgeParameters.userSettings.regirIndirect.numIndirectSamples;
}

void RtxdiPass::ExecuteGI(nvrhi::CommandListHandle commandList, nvrhi::BindingSetHandle extraBindingSet, bool skipFinal)
{
	commandList->beginMarker("ReSTIR GI");

	auto& reSTIRGI = m_ImportanceSamplingContext->getReSTIRGIContext();

	dm::int2 dispatchSize = { (int)reSTIRGI.getStaticParams().RenderWidth, (int)reSTIRGI.getStaticParams().RenderHeight };

	ExecuteRayTracingPass(commandList, m_GITemporalResamplingPass, "Temporal Resampling", dispatchSize, extraBindingSet);

	if (reSTIRGI.getResamplingMode() == rtxdi::ReSTIRGI_ResamplingMode::Spatial || reSTIRGI.getResamplingMode() == rtxdi::ReSTIRGI_ResamplingMode::TemporalAndSpatial)
	{
		nvrhi::utils::BufferUavBarrier(commandList, m_RtxdiResources->GIReservoirBuffer);

		ExecuteRayTracingPass(commandList, m_GISpatialResamplingPass, "Spatial Resampling", dispatchSize, extraBindingSet);
	}

	nvrhi::utils::BufferUavBarrier(commandList, m_RtxdiResources->GIReservoirBuffer);

    if (!skipFinal)
	    ExecuteRayTracingPass(commandList, m_GIFinalShadingPass, "Final Shading", dispatchSize, extraBindingSet);

	commandList->endMarker(); // ReSTIR GI
}

void RtxdiPass::ExecuteFusedDIGIFinal(nvrhi::CommandListHandle commandList, nvrhi::BindingSetHandle extraBindingSet)
{
	auto& reSTIRDI = m_ImportanceSamplingContext->getReSTIRDIContext();
	dm::int2 dispatchSize = { (int)reSTIRDI.getStaticParameters().RenderWidth, (int)reSTIRDI.getStaticParameters().RenderHeight };

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
	pass.Execute(commandList, dispatchSize.x, dispatchSize.y, dispatchSize.z, m_BindingSet,
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
	pass.Execute(commandList, dispatchSize.x, dispatchSize.y, m_BindingSet, 
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
