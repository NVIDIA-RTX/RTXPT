/*
* Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

// ReSTIR DI temporal resampling: combine the per-pixel reservoir from the Initial pass
// with the temporally-reprojected reservoir from the previous frame.

#define IGNORE_RAY_PAYLOAD 1

// Resolve environment reservoirs (from the Initial pass) through the shared polymorphic path during reuse.
#define INTRO_RTXDI_ENABLE_ENV_LIGHT 1

// Boiling filter needs group-shared memory, so it is only available in this compute pass.
// The group tile is NUM_COMPUTE_THREADS_PER_DIM x NUM_COMPUTE_THREADS_PER_DIM (see numthreads).
#define RTXDI_ENABLE_BOILING_FILTER
#define RTXDI_BOILING_FILTER_GROUP_SIZE NUM_COMPUTE_THREADS_PER_DIM

#include "../Bindings/ShaderResourceBindings.hlsli"
#include "../Bindings/GBufferBindings.hlsli"
#include "IntroBridge.hlsli"
#include "IntroSampleCommon.hlsli"
#include "IntroRTXDIBridge.hlsli"
#include "IntroReSTIRDICommon.hlsli"
#include <Rtxdi/Utils/Math.hlsli>             // RTXDI_IsValidNeighbor (surface similarity test)
#include <Rtxdi/Utils/ReservoirAddressing.hlsli>  // RTXDI_ApplyPermutationSampling, RTXDI_PixelPosToReservoirPos
#include <Rtxdi/DI/TemporalResampling.hlsli>
#include <Rtxdi/DI/BoilingFilter.hlsli>       // RTXDI_BoilingFilter (group-local firefly cull)

[numthreads(NUM_COMPUTE_THREADS_PER_DIM, NUM_COMPUTE_THREADS_PER_DIM, 1)]
void main(uint2 dispatchThreadID : SV_DispatchThreadID, uint2 localIndex : SV_GroupThreadID)
{
    const uint2 pixelPos = dispatchThreadID.xy;

    // NOTE: do NOT early-return for out-of-bounds pixels here. The boiling filter below issues a
    // group-wide barrier, so every thread in the tile must reach it. Out-of-bounds / invalid-surface
    // threads simply carry an empty reservoir (weight 0, ignored by the filter) and skip the store.
    const bool inBounds = all(pixelPos < uint2(g_Const.ptConsts.imageWidth, g_Const.ptConsts.imageHeight));

    // Different stream than the Initial pass (offset frameIndex multiplier) so temporal RNG
    // doesn't correlate with the initial-sampling RIS noise.
    RNGState rng;
    rng.initRngSeed(pixelPos, g_Const.ptConsts.frameIndex * 53 + 7);

    RAB_Surface surface = RAB_EmptySurface();
    if (inBounds)
        surface = RAB_GetGBufferSurface(int2(pixelPos), false);

    RTXDI_DIReservoir resampledReservoir = RTXDI_EmptyDIReservoir();

    if (RAB_IsSurfaceValid(surface))
    {
        // Current-frame initial reservoir for this pixel.
        RTXDI_DIReservoir currentReservoir = RTXDI_LoadDIReservoir(
            g_IntroReSTIRConst.restirDI.reservoirBufferParams,
            pixelPos,
            g_IntroReSTIRConst.restirDI.bufferIndices.initialSamplingOutputBufferIndex);

        // Startup, resource/layout changes and gaps in ReSTIR rendering have no compatible history.
        // Keep this frame's initial sample without reading previous textures or reservoirs.
        resampledReservoir = currentReservoir;
        if (g_IntroReSTIRConst.temporalHistoryValid != 0)
        {
            // Screen-space motion vector (xy = pixel delta, z = view-depth delta).
            float3 motion = u_MotionVectors[pixelPos].xyz;

            int2 temporalSamplePixelPos = -1;
            RAB_LightSample selectedSample = RAB_EmptyLightSample();

            resampledReservoir = RTXDI_DITemporalResampling(
                pixelPos,
                surface,
                currentReservoir,
                rng,
                g_IntroReSTIRConst.runtimeParams,
                g_IntroReSTIRConst.restirDI.reservoirBufferParams,
                motion,
                g_IntroReSTIRConst.restirDI.bufferIndices.temporalResamplingInputBufferIndex,
                g_IntroReSTIRConst.restirDI.temporalResamplingParams,
                temporalSamplePixelPos,
                selectedSample);
        }
    }

    // Group-local firefly cull: discards reservoirs whose weight is far above the tile average.
    // Uniform branch (constant across the group) so all threads either run the barrier or none do.
    if (g_IntroReSTIRConst.restirDI.boilingFilterParams.enableBoilingFilter)
    {
        RTXDI_BoilingFilter(localIndex, g_IntroReSTIRConst.restirDI.boilingFilterParams.boilingFilterStrength, resampledReservoir);
    }

    if (!inBounds)
        return;

    RTXDI_StoreDIReservoir(
        resampledReservoir,
        g_IntroReSTIRConst.restirDI.reservoirBufferParams,
        pixelPos,
        g_IntroReSTIRConst.restirDI.bufferIndices.temporalResamplingOutputBufferIndex);
}
