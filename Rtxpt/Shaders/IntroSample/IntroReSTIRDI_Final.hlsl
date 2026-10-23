/*
* Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

// ReSTIR DI final shading: load the reservoir from the shading-input buffer
// (initialSamplingOutputBufferIndex in None mode, temporalResamplingOutputBufferIndex
// once temporal resampling runs, etc.), reconstruct the chosen light sample, trace a
// visibility ray, and emit shaded direct lighting.

#define IGNORE_RAY_PAYLOAD 1

// Resolve the winning environment reservoir through the shared polymorphic path when shading.
#define INTRO_RTXDI_ENABLE_ENV_LIGHT 1

#include "../Bindings/ShaderResourceBindings.hlsli"
#include "../Bindings/GBufferBindings.hlsli"
#include "IntroBridge.hlsli"
#include "IntroSampleCommon.hlsli"
#include "IntroRTXDIBridge.hlsli"
#include "IntroReSTIRDICommon.hlsli"
#include "IntroReSTIRDI_Shading.hlsli"

[numthreads(NUM_COMPUTE_THREADS_PER_DIM, NUM_COMPUTE_THREADS_PER_DIM, 1)]
void main(uint2 dispatchThreadID : SV_DispatchThreadID)
{
    const uint2 pixelPos = dispatchThreadID.xy;
    if (any(pixelPos >= uint2(g_Const.ptConsts.imageWidth, g_Const.ptConsts.imageHeight)))
        return;

    PackedGBufferSurface packedSurface;
    packedSurface.LoadFromRenderTargets(pixelPos);
    float depth = u_Depth[pixelPos];

    float3 directLight = 0;
    if (depth > 0)
    {
        IntroRTXDISurface surface = BuildIntroRTXDISurface(pixelPos, depth, packedSurface);

        const uint reservoirIndex = g_IntroReSTIRConst.restirDI.bufferIndices.shadingInputBufferIndex;

        RTXDI_DIReservoir reservoir = RTXDI_LoadDIReservoir(
            g_IntroReSTIRConst.restirDI.reservoirBufferParams,
            pixelPos,
            reservoirIndex);

        if (RTXDI_IsValidDIReservoir(reservoir))
        {
            // Reconstruct the RIS-selected light sample from the reservoir-stored light
            // index and sample UV. Matches the standard RTXDI Final shading pattern (see
            // External/Rtxdi/Include/Rtxdi/DI/DIFinalShading.hlsli) and works for all
            // PolymorphicLight types - including directional, whose CalcSample returns
            // a non-zero solidAnglePdf of 1 / solidAngle.
            const uint lightIdx = RTXDI_GetDIReservoirLightIndex(reservoir);
            const float2 lightUV = RTXDI_GetDIReservoirSampleUV(reservoir);
            RAB_LightInfo lightInfo = RAB_LoadLightInfo(lightIdx, false);
            RAB_LightSample lightSample = RAB_SamplePolymorphicLight(lightInfo, surface, lightUV);

            if (lightSample.solidAnglePdf > 0)
            {
                float3 lightDir;
                float lightDistance;
                RAB_GetLightDirDistance(surface, lightSample, lightDir, lightDistance);

                const float NdotL = saturate(dot(surface.normal, lightDir));
                const float rayOffset = max(kIntroRTXDIRayEpsilon,
                    kIntroRTXDIRayEpsilon * (0.001 / max(depth, 1e-6)));

                bool visible = false;
                if (NdotL > 0)
                {
                    visible = TraceVisibilityRay(surface.worldPos + surface.normal * rayOffset,
                                             lightDir,
                                             max(rayOffset, lightDistance - rayOffset));
                }

                if (visible)
                {
                    const float invPdf = RTXDI_GetDIReservoirInvPdf(reservoir);
                    const float3 incidentRadiance = (lightSample.radiance / lightSample.solidAnglePdf) * invPdf;
                    directLight = IntroRTXDIEvaluateBRDF(surface, lightDir, NdotL) * incidentRadiance * NdotL;
                }
                else if (NdotL > 0)
                {
                    // Shadow ray reported the sample is blocked. Mark zero visibility in
                    // the reservoir so next frame's Temporal/Spatial passes can prune it
                    // without re-tracing - this is the amortization trick from RTXDI's
                    // enableFinalVisibility. NdotL == 0 is left alone because a small
                    // normal jitter next frame could legitimately bring the light back
                    // above the horizon.
                    RTXDI_StoreVisibilityInDIReservoir(reservoir, float3(0, 0, 0), true);
                    RTXDI_StoreDIReservoir(reservoir, g_IntroReSTIRConst.restirDI.reservoirBufferParams,
                        pixelPos, reservoirIndex);
                }
            }
        }
    }

    u_DirectLighting[pixelPos] = float4(directLight, 0);
}
