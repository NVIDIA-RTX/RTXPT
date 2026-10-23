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

#include <rtxdi/ImportanceSamplingContext.h>

// Set default values RTXDI settings
const rtxdi::ReSTIRDI_ResamplingMode GetReSTIRDI_ResamplingMode();

const RTXDI_DIInitialSamplingParameters getReSTIRDIInitialSamplingParams();

const RTXDI_DITemporalResamplingParameters getReSTIRDITemporalResamplingParams();

const RTXDI_BoilingFilterParameters getReSTIRDIBoilingFilterParams();

const RTXDI_DISpatialResamplingParameters getReSTIRDISpatialResamplingParams();

const RTXDI_ShadingParameters getReSTIRDIShadingParams();

const rtxdi::ReSTIRGI_ResamplingMode GetReSTIRGI_ResamplingMode();

const RTXDI_GITemporalResamplingParameters getReSTIRGITemporalResamplingParams();

const RTXDI_BoilingFilterParameters getReSTIRGIBoilingFilterParams();

const RTXDI_GISpatialResamplingParameters getReSTIRGISpatialResamplingParams();

const RTXDI_GIFinalShadingParameters getReSTIRGIFinalShadingParams();

const rtxdi::ReGIRDynamicParameters getReGIRDynamicParams();
