/*
* Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "RtxdiApplicationSettings.h"

const rtxdi::ReSTIRDI_ResamplingMode GetReSTIRDI_ResamplingMode() { return rtxdi::ReSTIRDI_ResamplingMode::TemporalAndSpatial; }

const RTXDI_DIInitialSamplingParameters getReSTIRDIInitialSamplingParams()
{
	RTXDI_DIInitialSamplingParameters params = {};
	params.brdfCutoff = 0.0001f;
	params.enableInitialVisibility = true;
	params.environmentMapImportanceSampling = 1;
	params.localLightSamplingMode = ReSTIRDI_LocalLightSamplingMode::ReGIR_RIS;
	params.numBrdfSamples = 1;
	params.numEnvironmentSamples = 1;
	params.numInfiniteLightSamples = 1;
	params.numLocalLightSamples = 8;

	return params;
}

const RTXDI_DITemporalResamplingParameters getReSTIRDITemporalResamplingParams()
{
	RTXDI_DITemporalResamplingParameters params = {};
	params.enablePermutationSampling = true;
	params.maxHistoryLength = 20;
	params.permutationSamplingThreshold = 0.9f;
	params.biasCorrectionMode = ReSTIRDI_TemporalBiasCorrectionMode::Raytraced;
	params.depthThreshold = 0.1f;
	params.normalThreshold = 0.5f;

	return params;
}

const RTXDI_BoilingFilterParameters getReSTIRDIBoilingFilterParams()
{
	RTXDI_BoilingFilterParameters params = {};
	params.enableBoilingFilter = true;
	params.boilingFilterStrength = 0.2f;

	return params;
}

const RTXDI_DISpatialResamplingParameters getReSTIRDISpatialResamplingParams()
{
	RTXDI_DISpatialResamplingParameters params = {};
	params.numDisocclusionBoostSamples = 8;
	params.numSamples = 1;
	params.biasCorrectionMode = ReSTIRDI_SpatialBiasCorrectionMode::Raytraced;
	params.depthThreshold = 0.1f;
	params.normalThreshold = 0.5f;
	params.samplingRadius = 32.0f;
	params.discountNaiveSamples = true;

	return params;
}

const RTXDI_ShadingParameters getReSTIRDIShadingParams()
{
	RTXDI_ShadingParameters params = {};
	params.enableDenoiserInputPacking = false;
	params.enableFinalVisibility = true;
	params.finalVisibilityMaxAge = 4;
	params.finalVisibilityMaxDistance = 16.f;
	params.reuseFinalVisibility = false;

	return params;
}

const rtxdi::ReSTIRGI_ResamplingMode GetReSTIRGI_ResamplingMode() { return rtxdi::ReSTIRGI_ResamplingMode::TemporalAndSpatial; }

const RTXDI_GITemporalResamplingParameters getReSTIRGITemporalResamplingParams()
{
	RTXDI_GITemporalResamplingParameters params = {};
	params.depthThreshold = 0.1f;
	params.enableFallbackSampling = true;
	params.enablePermutationSampling = true;
	params.maxHistoryLength = 10;
	params.maxReservoirAge = 50;
	params.normalThreshold = 0.6f;
	params.biasCorrectionMode = RTXDI_GIBiasCorrectionMode::Basic;

	return params;
}

const RTXDI_BoilingFilterParameters getReSTIRGIBoilingFilterParams()
{
	RTXDI_BoilingFilterParameters params = {};
	params.enableBoilingFilter = true;
	params.boilingFilterStrength = 0.35f;

	return params;
}

const RTXDI_GISpatialResamplingParameters getReSTIRGISpatialResamplingParams()
{
	RTXDI_GISpatialResamplingParameters params = {};
	params.numSamples = 2;
	params.biasCorrectionMode = RTXDI_GIBiasCorrectionMode::Basic;
	params.depthThreshold = 0.1f;
	params.normalThreshold = 0.5f;
	params.samplingRadius = 32.0f;

	return params;
}

const RTXDI_GIFinalShadingParameters getReSTIRGIFinalShadingParams()
{
	RTXDI_GIFinalShadingParameters params = {};
	params.enableFinalMIS = true;
	params.enableFinalVisibility = true;
	return params;
}

const rtxdi::ReGIRDynamicParameters getReGIRDynamicParams()
{
	rtxdi::ReGIRDynamicParameters params = {};
	return params;
}