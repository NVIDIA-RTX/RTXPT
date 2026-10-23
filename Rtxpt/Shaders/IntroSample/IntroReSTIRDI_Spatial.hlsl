/*
* Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

// ReSTIR DI spatial resampling: for each pixel, combine the current reservoir with a
// handful of neighbor reservoirs whose surfaces are similar enough, using the SDK's
// RTXDI_DISpatialResampling. Source buffer is either Temporal output (when temporal
// reuse is enabled) or Initial output (spatial-only mode); destination is the spatial
// output buffer that Final shades.

#define IGNORE_RAY_PAYLOAD 1

// Resolve environment reservoirs (from the Initial pass) through the shared polymorphic path during reuse.
#define INTRO_RTXDI_ENABLE_ENV_LIGHT 1

#include "../Bindings/ShaderResourceBindings.hlsli"
#include "../Bindings/GBufferBindings.hlsli"
#include "IntroBridge.hlsli"
#include "IntroSampleCommon.hlsli"
#include "IntroRTXDIBridge.hlsli"
#include "IntroReSTIRDICommon.hlsli"
#include <Rtxdi/Utils/Math.hlsli>               // RTXDI_IsValidNeighbor
#include <Rtxdi/Utils/ReservoirAddressing.hlsli>// RTXDI_PixelPosToReservoirPos
#include <Rtxdi/DI/SpatialResampling.hlsli>

[numthreads(NUM_COMPUTE_THREADS_PER_DIM, NUM_COMPUTE_THREADS_PER_DIM, 1)]
void main(uint2 dispatchThreadID : SV_DispatchThreadID)
{
    const uint2 pixelPos = dispatchThreadID.xy;
    if (any(pixelPos >= uint2(g_Const.ptConsts.imageWidth, g_Const.ptConsts.imageHeight)))
        return;

    // Distinct RNG stream from Initial / Temporal so spatial neighbor selection isn't
    // correlated with the RIS noise or the temporal jitter.
    RNGState rng;
    rng.initRngSeed(pixelPos, g_Const.ptConsts.frameIndex * 97 + 13);

    RAB_Surface surface = RAB_GetGBufferSurface(int2(pixelPos), false);

    RTXDI_DIReservoir resampledReservoir = RTXDI_EmptyDIReservoir();

    if (RAB_IsSurfaceValid(surface))
    {
        // The SDK has already set spatialResamplingInputBufferIndex to either the
        // initial output (Spatial-only mode) or the temporal output (TemporalAndSpatial
        // mode), so we just trust the constants and don't branch on it here.
        RTXDI_DIReservoir centerReservoir = RTXDI_LoadDIReservoir(
            g_IntroReSTIRConst.restirDI.reservoirBufferParams,
            pixelPos,
            g_IntroReSTIRConst.restirDI.bufferIndices.spatialResamplingInputBufferIndex);

        RAB_LightSample selectedSample = RAB_EmptyLightSample();

        resampledReservoir = RTXDI_DISpatialResampling(
            pixelPos,
            surface,
            centerReservoir,
            rng,
            g_IntroReSTIRConst.runtimeParams,
            g_IntroReSTIRConst.restirDI.reservoirBufferParams,
            g_IntroReSTIRConst.restirDI.bufferIndices.spatialResamplingInputBufferIndex,
            g_IntroReSTIRConst.restirDI.spatialResamplingParams,
            selectedSample);
    }

    RTXDI_StoreDIReservoir(
        resampledReservoir,
        g_IntroReSTIRConst.restirDI.reservoirBufferParams,
        pixelPos,
        g_IntroReSTIRConst.restirDI.bufferIndices.spatialResamplingOutputBufferIndex);
}
