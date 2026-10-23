/*
* Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

// ReSTIR DI initial sampling: build per-pixel reservoirs using RTXDI RIS.

#define IGNORE_RAY_PAYLOAD 1
//#define OVERRIDE_POLYLIGHT_TRIANGLE_ENABLE

// Compile in the environment-map RIS stream + the polymorphic kEnvironment light so this pass can sample
// the sky/HDRI and (via the shared resolution path) shade the winning env reservoir.
#define INTRO_RTXDI_ENABLE_ENV_LIGHT 1

#include "../Bindings/ShaderResourceBindings.hlsli"
#include "../Bindings/GBufferBindings.hlsli"
#include "IntroBridge.hlsli"
#include "IntroSampleCommon.hlsli"
#include "IntroRTXDIBridge.hlsli"
#include "IntroReSTIRDICommon.hlsli"
#include "IntroReSTIRDI_Shading.hlsli"

// Combine independent RIS streams for analytic locals, emissive triangles, infinite lights and
// the optional environment. Each stream has its own candidate budget and normalized reservoir.
// Carry the winning sample's direction/distance into the visibility test to avoid reconstructing
// the polymorphic light after selection. RAB_GetLightDirDistance also handles directional lights,
// whose 'position' stores a unit direction.
void IntroReSTIRDI_BuildCombinedReservoir(
    IntroRTXDISurface surface,
    inout RNGState rng,
    out RTXDI_DIReservoir reservoir,
    out float3 o_selectedLightDir,
    out float o_selectedLightDistance)
{
    reservoir = RTXDI_EmptyDIReservoir();
    o_selectedLightDir = float3(0, 0, 0);
    o_selectedLightDistance = 0;

    RTXDI_LightBufferRegion analyticRegion = MakeIntroRTXDILocalLightRegionAnalyticOnly();
    RTXDI_LightBufferRegion triangleRegion = MakeIntroRTXDILocalLightRegionTriangleOnly();
    RTXDI_LightBufferRegion infiniteRegion = MakeIntroRTXDIInfiniteLightRegion();
    if (!surface.valid)
        return;

    RTXDI_DIInitialSamplingParameters baseParams = MakeIntroRTXDIInitialSamplingParameters();

    // The analytic-local and emissive-triangle candidate counts are controlled independently from
    // the UI (g_IntroReSTIRConst). Each stream needs its own MIS data because numMisSamples /
    // localLightMisWeight derive from that stream's sample count, and RTXDI_SampleLocalLights
    // normalizes its reservoir by numMisSamples.
    RTXDI_DIInitialSamplingParameters analyticParams = baseParams;
    RTXDI_DIInitialSamplingParameters triangleParams = baseParams;
    analyticParams.numLocalLightSamples = g_IntroReSTIRConst.numAnalyticLocalLightSamples;
    triangleParams.numLocalLightSamples = g_IntroReSTIRConst.numTriangleLocalLightSamples;

    RTXDI_InitialSamplingMisData analyticMis = RTXDI_ComputeInitialSamplingMisData(analyticParams);
    RTXDI_InitialSamplingMisData triangleMis = RTXDI_ComputeInitialSamplingMisData(triangleParams);

    // Coherent RNG is only consumed by presampled / ReGIR selection modes (disabled here); in uniform
    // mode it is unused, so a single snapshot shared by both local streams is fine.
    RNGState coherentRng = rng;

    // Reused per stream: streams run serially, so a single capture slot is enough - each stream
    // overwrites it, and the survivor (o_selectedLightDir/Distance) is resolved at combine time.
    RAB_LightSample streamSample = RAB_EmptyLightSample();

    // Analytic locals are sampled first, ahead of triangles/infinite. This stream has the most
    // expensive candidate loop, so running it before any cross-stream "survivor" state exists keeps
    // its loop's live set minimal (the selected-sample carry below is otherwise live across it). The
    // first positive-weight combine into an empty reservoir always wins, so its random argument is 0.5.
    RTXDI_DIReservoir analyticReservoir = RTXDI_SampleLocalLights(
        rng,
        coherentRng,
        surface,
        analyticParams,
        analyticMis,
        analyticParams.localLightSamplingMode,
        analyticRegion,
        streamSample);
    if (RTXDI_CombineDIReservoirs(reservoir, analyticReservoir, 0.5, analyticReservoir.targetPdf))
        RAB_GetLightDirDistance(surface, streamSample, o_selectedLightDir, o_selectedLightDistance);

    RTXDI_DIReservoir triangleReservoir = RTXDI_SampleLocalAreaLights(
        rng,
        coherentRng,
        surface,
        triangleParams,
        triangleMis,
        triangleParams.localLightSamplingMode,
        triangleRegion,
        streamSample);
    if (RTXDI_CombineDIReservoirs(reservoir, triangleReservoir, RTXDI_GetNextRandom(rng), triangleReservoir.targetPdf))
        RAB_GetLightDirDistance(surface, streamSample, o_selectedLightDir, o_selectedLightDistance);

    RTXDI_DIReservoir infiniteReservoir = RTXDI_SampleInfiniteLights(
        rng,
        surface,
        g_IntroReSTIRConst.numInfiniteLightSamples,
        infiniteRegion,
        streamSample);
    if (RTXDI_CombineDIReservoirs(reservoir, infiniteReservoir, RTXDI_GetNextRandom(rng), infiniteReservoir.targetPdf))
        RAB_GetLightDirDistance(surface, streamSample, o_selectedLightDir, o_selectedLightDistance);

#if INTRO_RTXDI_ENABLE_ENV_LIGHT
    // Environment map: importance-sampled as its own stream (numEnvironmentSamples from the UI). Its MIS
    // data derives from the env sample count exactly like the local streams above. RAB_GetLightDirDistance
    // resolves the winning env sample's direction (its CalcSample places the light along the sampled
    // direction at DISTANT_LIGHT_DISTANCE, so the generic distant handling applies).
    RTXDI_DIInitialSamplingParameters envParams = baseParams;
    RTXDI_InitialSamplingMisData envMis = RTXDI_ComputeInitialSamplingMisData(envParams);
    RTXDI_DIReservoir envReservoir = RTXDI_SampleEnvironmentMapCustom(rng, surface, envParams, envMis, streamSample);
    if (RTXDI_CombineDIReservoirs(reservoir, envReservoir, RTXDI_GetNextRandom(rng), envReservoir.targetPdf))
        RAB_GetLightDirDistance(surface, streamSample, o_selectedLightDir, o_selectedLightDistance);
#endif

    // Finalize and reset M to 1, exactly as RTXDI_SampleLightsForSurface does. The first combine
    // uses a fixed 0.5 because the target reservoir is empty (selection is deterministic there).
    RTXDI_FinalizeResampling(reservoir, 1.0, 1.0);
    reservoir.M = 1;
}

[numthreads(NUM_COMPUTE_THREADS_PER_DIM, NUM_COMPUTE_THREADS_PER_DIM, 1)]
void main(uint2 dispatchThreadID : SV_DispatchThreadID)
{
    uint2 pixelPos = dispatchThreadID.xy;
    
    if (any(pixelPos >= uint2(g_Const.ptConsts.imageWidth, g_Const.ptConsts.imageHeight)))
        return;

    PackedGBufferSurface packedSurface;
    packedSurface.LoadFromRenderTargets(pixelPos);
    float depth = u_Depth[pixelPos];

    RNGState rng;
    rng.initRngSeed(pixelPos, g_Const.ptConsts.frameIndex * 37);

    if (depth <= 0)
        return;

    IntroRTXDISurface surface = BuildIntroRTXDISurface(pixelPos, depth, packedSurface);

    // Resolve the reservoir store address up front, before the candidate loop. Carrying the
    // single 32-bit pointer across the loop is cheaper than carrying pixelPos.x/y: the pixel
    // coordinate comes from a swizzled SV_DispatchThreadID that the compiler spills to shared
    // rather than rematerialize, and it is otherwise unused inside the loop. Folding it to the
    // pointer here removes two loop-crossing values (and thus two shared-memory spill slots),
    // which on this kernel is shared-pipe bandwidth that was contending with the TTU.
    const uint outputIndex = g_IntroReSTIRConst.restirDI.bufferIndices.initialSamplingOutputBufferIndex;
    const uint reservoirPointer = RTXDI_ReservoirPositionToPointer(
        g_IntroReSTIRConst.restirDI.reservoirBufferParams, pixelPos, outputIndex);

    RTXDI_DIReservoir reservoir;
    float3 carriedLightDir;
    float carriedLightDistance;
    IntroReSTIRDI_BuildCombinedReservoir(surface, rng, reservoir, carriedLightDir, carriedLightDistance);
    
    if (RTXDI_IsValidDIReservoir(reservoir))
    {
        // Use the light dir/distance carried out of the builder (the winning stream's selected
        // sample, resolved at selection time) instead of reconstructing the sample here. This keeps
        // the heavy polymorphic reconstruct off the path to the shadow ray, freeing the TTU.
        float3 lightDir = carriedLightDir;
        float lightDistance = carriedLightDistance;

        float rayOffset = max(kIntroRTXDIRayEpsilon, kIntroRTXDIRayEpsilon * (0.001 / max(depth, 1e-6)));
        bool visible = TraceVisibilityRay(surface.worldPos + surface.normal * rayOffset,
                                        lightDir,
                                        max(rayOffset, lightDistance - rayOffset));
        if (!visible)
            RTXDI_StoreVisibilityInDIReservoir(reservoir, float3(0, 0, 0), true);
    }

    // Store via the pre-resolved pointer (see above) so pixelPos does not need to survive the loop.
    RTXDI_LIGHT_RESERVOIR_BUFFER[reservoirPointer] = RTXDI_PackDIReservoir(reservoir);
}
