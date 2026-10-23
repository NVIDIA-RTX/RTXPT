/*
* Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#ifndef __PATH_TRACER_SHARC_HLSLI__ // using instead of "#pragma once" due to https://github.com/microsoft/DirectXShaderCompiler/issues/3943
#define __PATH_TRACER_SHARC_HLSLI__

#if RTXPT_SHARC_TRAIN && RTXPT_SHARC_CONSUME
#error SHaRC: both train and consume can't be on at the same time!
#endif
#if RTXPT_SHARC_TRAIN && PATH_TRACER_MODE != PT_REFERENCE
#error SHaRC: training can only happen in reference mode
#endif
#if RTXPT_SHARC_CONSUME && PATH_TRACER_MODE != PT_DENOISING_LAYERS_FILL
#error SHaRC: consuming can only happen in real-time "fill" pass
#endif
#if RTXPT_SHARC_TRAIN && PT_ENABLE_RUSSIAN_ROULETTE
#error SHaRC: russian roulette not beneficial for SHaRC as the importance of radiance might not be high to path currently depositing, but can be for different ones later reading from the voxel
#endif


// All SHaRC (spatial hash radiance cache) glue for the path tracer lives here, so PathTracer.hlsli and
// PathTracerHandleHit.hlsli stay free of RTXPT_SHARC_CONSUME / RTXPT_SHARC_TRAIN blocks. Every function below compiles away
// when both macros are 0, so call sites need no #if of their own.
//
// This header deliberately does NOT call back into PathTracer.hlsli (AccumulatePathRadiance and friends): keeping
// the dependency one-way lets it be included early, before those exist. The caller owns radiance accumulation.

#include "PathTracerTypes.hlsli"

#include "DenoisingLayers.hlsli"

namespace PathTracer
{
    // ---------------------------------------------------------------------------------------------------------
    // Per-path state.
    //
    // On the training pass the SHaRC propagation state (SharcState - a sliding window of the last few path vertices
    // plus their propagation weights) is produced in the closest hit shader at bounce N and consumed in the closest
    // hit shader at bounce N+1, or in raygen/miss if the path escapes, so it has to survive the TraceRay boundary.
    // It used to do that by riding in the ray payload, which cost 5 extra uint4 (80 bytes) on EVERY PathPayload and
    // kept them live for the whole of the hit shader, shadow rays included. It now lives in a per-thread slot of a
    // scratch buffer instead, touched only where it is actually used.
    //
    // The slot is keyed on DispatchRaysIndex(), which is valid in raygen, closesthit, anyhit and miss and follows the
    // work through SER reordering - so nothing about this has to travel in the payload. It also means these helpers
    // are ray tracing only; nothing compiles RTXPT_SHARC_TRAIN into a compute shader.
    //
    // Layout, stride RTXPT_SHARC_PATH_STATE_STRIDE == 16 * (1 + SHARC_PROPAGATION_DEPTH) bytes:
    //      +0              uint4 { pathLength, -, -, - }
    //      +16 + 16*i      uint4 { cacheIndices[i], asuint(sampleWeights[i].xyz) }
    // All SHARC_PROPAGATION_DEPTH entries are persisted: SharcUpdateHit clamps pathLength to SHARC_PROPAGATION_DEPTH,
    // and every read loop in the SDK is bounded by pathLength.
    //
    // Consume mode does not use any of this: with SHARC_UPDATE off the SDK compiles SharcState down to a single
    // unused placeholder, so there is no per-path state to carry.
    // ---------------------------------------------------------------------------------------------------------

#if RTXPT_SHARC_TRAIN || defined(__INTELLISENSE__)

    // Byte offset of this thread's slot in the path state buffer.
    inline uint SharcPathStateOffset()
    {
        const uint2 rayIndex    = DispatchRaysIndex().xy;
        const uint  slotIndex   = rayIndex.y * DispatchRaysDimensions().x + rayIndex.x;
#if RTXPT_SHADER_DEBUG
        // The buffer is sized from the training dispatch size, so this can only fire if the two have drifted apart
        // on the C++ side. Worth trapping: out of range raw buffer writes are silently dropped and reads return 0,
        // which just pins pathLength at 0 and quietly stops the cache filling, with no error anywhere.
        if (slotIndex >= g_Const.ptConsts.sharc.PathStateCapacity)
            BREAKPOINT();
#endif
        return slotIndex * RTXPT_SHARC_PATH_STATE_STRIDE;
    }

    // Just the header. Both callers below use it to skip the rest of the load when the path has stored nothing yet,
    // which is the case on every path's first hit.
    inline uint SharcLoadPathLength(const uint baseOffset)
    {
        return GetSharcPathStateBuffer(g_Const.ptConsts.sharc).Load(baseOffset);
    }

    inline SharcState SharcLoadState(const uint baseOffset)
    {
        RWByteAddressBuffer pathStateBuffer = GetSharcPathStateBuffer(g_Const.ptConsts.sharc);

        SharcState sharcState;
        sharcState.pathLength = pathStateBuffer.Load(baseOffset);

        [unroll] for (uint i = 0; i < SHARC_PROPAGATION_DEPTH; i++)
        {
            const uint4 entry = pathStateBuffer.Load4(baseOffset + 16 + i * 16);
            sharcState.cacheIndices[i]  = entry.x;
            // Per component, not asfloat(entry.yzw): DXC 1.9.2607+ turns this uint3->float3 into an i96 at SM 6.9 and fails validation (DXC #9040 / PR #9041)
            sharcState.sampleWeights[i] = (SharcSampleWeight)float3(asfloat(entry.y), asfloat(entry.z), asfloat(entry.w));
        }

        return sharcState;
    }

    inline void SharcStoreState(const uint baseOffset, const SharcState sharcState)
    {
        RWByteAddressBuffer pathStateBuffer = GetSharcPathStateBuffer(g_Const.ptConsts.sharc);

        pathStateBuffer.Store(baseOffset, sharcState.pathLength);

        [unroll] for (uint i = 0; i < SHARC_PROPAGATION_DEPTH; i++)
            pathStateBuffer.Store4(baseOffset + 16 + i * 16, uint4(sharcState.cacheIndices[i], asuint((float3)sharcState.sampleWeights[i])));
    }

#endif // RTXPT_SHARC_TRAIN

    // Reset the per-path propagation state. Call once per path, from EmptyPathInitialize.
    inline void SharcPathInit()
    {
#if RTXPT_SHARC_TRAIN
        // SharcInit() only zeroes pathLength, and every read loop in the SDK is bounded by it, so the entries above
        // it are never read and do not need clearing - one 4 byte store per path is the whole of it.
        GetSharcPathStateBuffer(g_Const.ptConsts.sharc).Store(SharcPathStateOffset(), 0);
#endif
    }

    // Fold the throughput accumulated since the last checkpoint into the stored vertices' propagation weights, then
    // reset it. SharcSetThroughput MULTIPLIES, so the reset is what keeps each push per-segment instead of
    // compounding - this is Integration.md's "path throughput should be reset to 1.0 on each bounce".
    inline void SharcThroughputCheckpoint(inout PathState path)
    {
#if RTXPT_SHARC_TRAIN
        float3 thp = path.GetThp();
        if (all(thp == 1.0f))
            return;
        path.SetThp(float3(1,1,1));

        const uint baseOffset = SharcPathStateOffset();
        // Safe to restore as a perf win - the loop below is bounded by pathLength, so it is already a no-op at the
        // first vertex - but only BELOW the reset above. Above it, the camera->first-vertex transmittance survives
        // into the next segment and contaminates every weight from there on.
        // if (SharcLoadPathLength(baseOffset) == 0)
        //     return; 

        SharcState sharcState = SharcLoadState(baseOffset);
        SharcSetThroughput(sharcState, thp);
        SharcStoreState(baseOffset, sharcState);
#endif
    }

    // Snapshot of the path handed to HandleNEE. The cache wants the LOCAL (un-throughput-weighted) direct lighting
    // at this vertex, and HandleNEE's only use of the snapshot's throughput is the final scaling of its own result -
    // so neutralising it here makes neeResult carry exactly what SharcDepositHit needs, with no separate channel.
    // Safe because the training pass writes none of the standard outputs (RTXPT_SKIP_STANDARD_OUTPUTS), and this
    // touches the copy only: the real path keeps its throughput for the scatter ray and the propagation weights.
    // Note this also drops the throughput factor from the NEE-AT feedback weight in the update pass.
    inline PathState SharcAdjustPreScatterPath(const PathState path)
    {
#if RTXPT_SHARC_TRAIN
        PathState ret = path;
        ret.SetThp(float3(1.0f, 1.0f, 1.0f));
        return ret;
#else
        return path;
#endif
    }

    // Back-propagate a final radiance contribution into every vertex this path has stored. sampleData is 0, so this
    // adds radiance without allocating a cache entry or counting a sample anywhere.
    inline void SharcBackPropagate(const float3 radiance)
    {
#if RTXPT_SHARC_TRAIN
        const uint baseOffset = SharcPathStateOffset();
        if (SharcLoadPathLength(baseOffset) == 0)
            return;                     // nothing stored - SharcUpdateMiss would do nothing, so skip the full load

        SharcParameters sharcParameters = GetSharcParameters(g_Const.ptConsts.sharc);
        SharcUpdateMiss(sharcParameters, SharcLoadState(baseOffset), radiance);
#endif
    }

    // Path escaped the scene.
    inline void SharcHandleMiss(const float3 environmentEmission)   { SharcBackPropagate(environmentEmission); }

    // Path ran out of bounces at a surface instead of escaping. NOT SharcDepositHit: that would also store this
    // vertex and count a sample for it, and that sample would be emission-only, biasing the voxel dark.
    inline void SharcHandlePathLimit(const float3 surfaceEmission)  { SharcBackPropagate(surfaceEmission); }
}

// -------------------------------------------------------------------------------------------------------------
// Per-hit helpers. These name ShadingData and ActiveBSDF, so they can only be declared where a BSDF specialization
// has actually been pulled in. Surface.hlsli is the header that does that (it includes StandardBSDF.hlsli and
// #errors without ActiveBSDF), so its include guard is the exact condition - checking defined(ActiveBSDF) alone
// would also fire for a translation unit that has the macro but not the types. PathTracerSample.hlsl includes
// neither, and never includes PathTracerHandleHit.hlsli, so it simply skips this section.
// -------------------------------------------------------------------------------------------------------------
#if defined(__SURFACE_HLSLI__) || defined(__INTELLISENSE__)

#include "BSDFs/LobeType.hlsli"

namespace PathTracer
{
    // Opaque per-vertex SHaRC payload, carried by HandleHit from the cache lookup through to the deposit.
    struct SharcPathVertex
    {
#if RTXPT_SHARC_CONSUME || RTXPT_SHARC_TRAIN
        SharcHitData HitData;
#else
        uint Unused;    // a struct must not be empty
#endif
    };

    // Build the cache key/payload for this hit. On the update pass this also biases the BSDF roughness upward
    // (hence the non-const bsdf) so glossy lobes spread more samples into the cache.
    inline SharcPathVertex SharcPrepareHit(const ShadingData shadingData, inout ActiveBSDF bsdf, const float3 surfaceEmission)
    {
        SharcPathVertex sharcVertex = (SharcPathVertex) 0;

#if (RTXPT_SHARC_CONSUME || RTXPT_SHARC_TRAIN) || defined(__INTELLISENSE__)
        sharcVertex.HitData.positionWorld = shadingData.posW;
        sharcVertex.HitData.normalWorld = shadingData.faceNCorrected; // geometry (face) normal, view-corrected; piecewise-constant per triangle -> stable hash key

#if SHARC_MATERIAL_DEMODULATION || defined(__INTELLISENSE__)
        // Preserve material detail by demodulating diffuse/specular response from the cached radiance.
        sharcVertex.HitData.materialDemodulation = float3(1.0f, 1.0f, 1.0f);
        if (g_Const.ptConsts.sharc.MaterialDemodulation)
        {
            const float3 diffuseAlbedo = (float3) bsdf.data.Diffuse();
            const float3 specularF0 = (float3) bsdf.data.Specular();
            const float3 specularFAvg = specularF0 + (1.0f - specularF0) * 0.047619f; // 1/21
            sharcVertex.HitData.materialDemodulation = max(diffuseAlbedo, 0.05f) + max(specularF0, 0.02f) * Luminance(specularFAvg);
        }
#endif // SHARC_MATERIAL_DEMODULATION

#if SHARC_SEPARATE_EMISSIVE || defined(__INTELLISENSE__)
        // Stored here so SharcGetCachedRadiance re-adds it at query time, and excluded from the directLighting
        // argument to SharcUpdateHit in SharcDepositHit below.
        sharcVertex.HitData.emissive = g_Const.ptConsts.sharc.SeparateEmissive ? surfaceEmission : float3(0.0f, 0.0f, 0.0f);
#endif // SHARC_SEPARATE_EMISSIVE

#if RTXPT_SHARC_TRAIN || defined(__INTELLISENSE__)
        // Bias glossy lobes wider - a razor-sharp lobe stores nothing a spatially averaged voxel can reuse. Skipped
        // for pure delta surfaces (smooth glass, mirrors): there it would swap delta transmission for a rough one and
        // bend training paths away from the real transport, and the query pass ignores those surfaces anyway.
        if ((bsdf.getLobes(shadingData) & (uint) LobeType::NonDelta) != 0)
            bsdf.data.SetRoughness((lpfloat) max(g_Const.ptConsts.sharc.MinRoughnessOverride, (float) bsdf.data.Roughness()));
#endif // RTXPT_SHARC_TRAIN
#endif // RTXPT_SHARC_CONSUME || RTXPT_SHARC_TRAIN

        return sharcVertex;
    }

    // Cache lookup for this vertex, plus the debug visualizations.
    // Returns true when SHaRC produced radiance for this vertex - the caller must then skip its own emission
    // accumulation (the cached value already contains this vertex's emission) and stop the path at the next bounce.
    //   outRadiance      - radiance from this vertex onwards, NOT multiplied by path throughput; caller accumulates
    //   outWroteDirectly - a debug visualization already wrote the pixel; caller must not accumulate outRadiance
    inline bool SharcQueryHit(const SharcPathVertex sharcVertex, const ShadingData shadingData, const ActiveBSDF bsdf,
                              const float rayTCurrent, inout PathState path, out float3 outRadiance, out bool outWroteDirectly)
    {
        outRadiance = float3(0.0f, 0.0f, 0.0f);
        outWroteDirectly = false;

#if RTXPT_SHARC_CONSUME || defined(__INTELLISENSE__)
        SharcParameters sharcParameters = GetSharcParameters(g_Const.ptConsts.sharc);

        bool hasRadiance = false;
        {
            uint gridLevel = HashGridGetLevel(shadingData.posW, sharcParameters.hashGridParameters);
            float voxelSize = HashGridGetVoxelSize(gridLevel, sharcParameters.hashGridParameters);
            bool isValidHit = rayTCurrent > voxelSize * sqrt(3.0f);

            // Use RTXPT path's ray cone directly instead of an RTXGI roughness-based footprint.
            isValidHit &= path.rayCone.getWidth() > voxelSize * g_Const.ptConsts.sharc.TerminateThreshold;

// TODO: disabled pending investigation
#if 0
            // The cone is seeded with PixelConeSpreadAngle and, absent any widening, 
            // getWidth() == PixelConeSpreadAngle * GetSceneLength(), so this strict > is false.
            // isValidHit &= (path.rayCone.getWidth() / path.GetSceneLength()) > g_Const.ptConsts.camera.PixelConeSpreadAngle;

            // The FILL pass owns only the noisy part of the path. While this vertex is still on the stable
            // branch (primary hit and any delta chain the BUILD pass already captured), AccumulatePathRadiance
            // drops its radiance - consuming here would discard the cached value and then end the path.
            // isValidHit &= !path.hasFlag(PathFlags::denoisingLayerOnBranch);
#endif

            // Skip the query when the surface has no non-delta lobes (pure specular/transmission).
            // Affects lookups only, not what is stored, so no cache clear is needed.
            const bool hasNonDeltaLobes = ((bsdf.getLobes(shadingData) & (uint) LobeType::NonDelta) != 0);
            isValidHit &= hasNonDeltaLobes;

            float3 sharcRadiance;
            if (isValidHit && SharcGetCachedRadiance(sharcParameters, sharcVertex.HitData, sharcRadiance, false))
            {
                outRadiance = sharcRadiance;
                hasRadiance = true;
            }
        }

#if (RTXPT_SHARC_ENABLE_DEBUG && PATH_TRACER_MODE != PT_DENOISING_LAYERS_BUILD) || defined(__INTELLISENSE__)
        // Debug visualizations, selected via the "Debug Mode" combo (SharcIntegration::DebugGUI ->
        // g_Const.ptConsts.sharc.SharcDebug). Runs after the lookup above and deliberately overrides it, so the
        // selected mode is what reaches the screen whether or not this vertex had a cache hit.
        if (g_Const.ptConsts.sharc.SharcDebug > 0)
        {
            float3 debugColor = float3(0.f, 0.f, 0.f);
            if (g_Const.ptConsts.sharc.SharcDebug == 1)
            {
                // cached radiance; black on a miss, where SharcGetCachedRadiance leaves its output unwritten
                float3 cachedRadiance;
                if (SharcGetCachedRadiance(sharcParameters, sharcVertex.HitData, cachedRadiance, false))
                    debugColor = cachedRadiance;
            }
            else if (g_Const.ptConsts.sharc.SharcDebug == 2)
            {
                debugColor = HashGridDebugColoredHash(sharcVertex.HitData.positionWorld, sharcVertex.HitData.normalWorld, sharcParameters.hashGridParameters);
            }
            else if (g_Const.ptConsts.sharc.SharcDebug == 3)
            {
                debugColor = HashGridDebugOccupancy(path.GetPixelPos(), g_Const.ptConsts.camera.ViewportSize, sharcParameters.hashGridData);
            }
            else if (g_Const.ptConsts.sharc.SharcDebug == 4)
            {
                debugColor = HashGridDebugHashCollisions(sharcVertex.HitData.positionWorld, sharcVertex.HitData.normalWorld, sharcParameters.hashGridParameters, sharcParameters.hashGridData);
            }
            else if (g_Const.ptConsts.sharc.SharcDebug == 5)
            {
                uint dbgGridLevel = HashGridGetLevel(sharcVertex.HitData.positionWorld, sharcParameters.hashGridParameters);
                float dbgVoxelSize = HashGridGetVoxelSize(dbgGridLevel, sharcParameters.hashGridParameters);
                debugColor = float3(dbgVoxelSize, dbgVoxelSize, dbgVoxelSize);
            }
            else if (g_Const.ptConsts.sharc.SharcDebug == 6)
            {
                uint dbgGridLevel = HashGridGetLevel(sharcVertex.HitData.positionWorld, sharcParameters.hashGridParameters);
                debugColor = float3(dbgGridLevel, dbgGridLevel, dbgGridLevel);
            }

            // also possible, direct debug display that avoids tonemapping and everything:
            // workingContext.Debug.DrawDebugViz(path.GetPixelPos(), float4(debugColor, 1));

            // written here rather than returned: PathState::SetL does not exist in PT_DENOISING_LAYERS_BUILD, and
            // this whole block is compiled out there (RTXPT_SHARC_CONSUME is 0 for that mode).
            path.SetL(float4(debugColor, 1.0f));
            outRadiance = float3(0.0f, 0.0f, 0.0f);
            outWroteDirectly = true;
            hasRadiance = true;
        }
#endif // RTXPT_SHARC_ENABLE_DEBUG

        return hasRadiance;
#else
        return false;
#endif // RTXPT_SHARC_CONSUME
    }

    // Deposit this vertex into the cache, then advance the propagation weights by the per-segment scatter
    // throughput. Must run after HandleNEE so neeRadiance carries the NEE-sampled direct lighting.
    // Returns false when the path should stop: cache resampling decided the cached estimate is good enough, or the
    // hash grid had no free slot for this vertex.
    // neeRadiance must be the LOCAL direct lighting, i.e. not multiplied by path throughput - see
    // SharcAdjustPreScatterPath, which is what makes neeResult's own radiance already local on the update pass.
    inline bool SharcDepositHit(const SharcPathVertex sharcVertex, const float3 neeRadiance, const float3 surfaceEmission, inout PathState path)
    {
#if RTXPT_SHARC_TRAIN || defined(__INTELLISENSE__)
        SharcParameters sharcParameters = GetSharcParameters(g_Const.ptConsts.sharc);

        float3 directLighting = neeRadiance;
#if SHARC_SEPARATE_EMISSIVE
        // emission is propagated via HitData.emissive inside SharcUpdateHit; add it to directLighting only when
        // separate-emissive is disabled.
        if (!g_Const.ptConsts.sharc.SeparateEmissive)
            directLighting += surfaceEmission;
#else
        directLighting += surfaceEmission;
#endif // SHARC_SEPARATE_EMISSIVE

        const uint baseOffset = SharcPathStateOffset();
        SharcState sharcState = SharcLoadState(baseOffset);

        float random = Hash32ToFloat(Hash32Combine(Hash32Combine(Hash32(path.getVertexIndex() + 0x0366FE2F), path.GetId()), Bridge::getSampleIndex())); // unique prime number salt for vertex index
        if (!SharcUpdateHit(sharcParameters, sharcState, sharcVertex.HitData, directLighting, random))
            return false;   // cache resampling early-out (the cached estimate is good enough) or no free hash grid
                            // slot for this vertex: stop tracing either way.
                            // Deliberately no store: the caller terminates the path on false, so the state
                            // SharcUpdateHit just mutated is dead - nothing reads this slot again before the next
                            // path resets it in SharcPathInit.

        // Throughput accumulated since the checkpoint at the top of HandleHit, i.e. just this vertex's scatter
        // weight. Reset afterwards for the same reason as there.
        SharcSetThroughput(sharcState, path.GetThp());
        SharcStoreState(baseOffset, sharcState);
        path.SetThp(float3(1,1,1));
#endif // RTXPT_SHARC_TRAIN
        return true;
    }
}

#endif // defined(__SURFACE_HLSLI__)

#endif // __PATH_TRACER_SHARC_HLSLI__
