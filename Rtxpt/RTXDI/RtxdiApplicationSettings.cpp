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

const ReSTIRDI_InitialSamplingParameters getReSTIRDIInitialSamplingParams()
{
	ReSTIRDI_InitialSamplingParameters params = {};
	params.brdfCutoff = 0.0001f;
	params.enableInitialVisibility = true;
	params.environmentMapImportanceSampling = 1;
	params.localLightSamplingMode = ReSTIRDI_LocalLightSamplingMode::ReGIR_RIS;
	params.numPrimaryBrdfSamples = 1;
	params.numPrimaryEnvironmentSamples = 1;
	params.numPrimaryInfiniteLightSamples = 1;
	params.numPrimaryLocalLightSamples = 8;

	return params;
}

const ReSTIRDI_TemporalResamplingParameters getReSTIRDITemporalResamplingParams()
{
	ReSTIRDI_TemporalResamplingParameters params = {};
	params.boilingFilterStrength = 0.2f;
	params.discardInvisibleSamples = false;
	params.enableBoilingFilter = true;
	params.enablePermutationSampling = false;
	params.maxHistoryLength = 20;
	params.permutationSamplingThreshold = 0.9f;
	params.temporalBiasCorrection = ReSTIRDI_TemporalBiasCorrectionMode::Raytraced;
	params.temporalDepthThreshold = 0.1f;
	params.temporalNormalThreshold = 0.5f;

	return params;
}

const ReSTIRDI_SpatialResamplingParameters getReSTIRDISpatialResamplingParams()
{
	ReSTIRDI_SpatialResamplingParameters params = {};
	params.numDisocclusionBoostSamples = 8;
	params.numSpatialSamples = 1;
	params.spatialBiasCorrection = ReSTIRDI_SpatialBiasCorrectionMode::Raytraced;
	params.spatialDepthThreshold = 0.1f;
	params.spatialNormalThreshold = 0.5f;
	params.spatialSamplingRadius = 32.0f;
	params.discountNaiveSamples = true;

	return params;
}

const ReSTIRDI_ShadingParameters getReSTIRDIShadingParams()
{
	ReSTIRDI_ShadingParameters params = {};
	params.enableDenoiserInputPacking = false;
	params.enableFinalVisibility = true;
	params.finalVisibilityMaxAge = 4;
	params.finalVisibilityMaxDistance = 16.f;
	params.reuseFinalVisibility = false;

	return params;
}

const rtxdi::ReSTIRGI_ResamplingMode GetReSTIRGI_ResamplingMode() { return rtxdi::ReSTIRGI_ResamplingMode::TemporalAndSpatial; }

const ReSTIRGI_TemporalResamplingParameters getReSTIRGITemporalResamplingParams()
{
	ReSTIRGI_TemporalResamplingParameters params = {};
	params.boilingFilterStrength = 0.6f;
	params.depthThreshold = 0.1f;
	params.enableBoilingFilter = true;
	params.enableFallbackSampling = true;
	params.enablePermutationSampling = false;
	params.maxHistoryLength = 10;
	params.maxReservoirAge = 30;
	params.normalThreshold = 0.6f;
    params.enablePermutationSampling = true;
	params.temporalBiasCorrectionMode = ResTIRGI_TemporalBiasCorrectionMode::Basic;

	return params;
}

const ReSTIRGI_SpatialResamplingParameters getReSTIRGISpatialResamplingParams()
{
	ReSTIRGI_SpatialResamplingParameters params = {};
	params.numSpatialSamples = 2;
	params.spatialBiasCorrectionMode = ResTIRGI_SpatialBiasCorrectionMode::Basic;
	params.spatialDepthThreshold = 0.1f;
	params.spatialNormalThreshold = 0.5f;
	params.spatialSamplingRadius = 32.0f;

	return params;
}

const ReSTIRGI_FinalShadingParameters getReSTIRGIFinalShadingParams()
{
	ReSTIRGI_FinalShadingParameters params = {};
	params.enableFinalMIS = true;
	params.enableFinalVisibility = true;
	return params;
}

const rtxdi::ReGIRDynamicParameters getReGIRDynamicParams()
{
	rtxdi::ReGIRDynamicParameters params = {};
	return params;
}