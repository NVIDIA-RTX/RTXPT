/*
* Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#ifndef __SHARC_BINDINGS_HLSLI__    // using instead of "#pragma once" due to https://github.com/microsoft/DirectXShaderCompiler/issues/3943
#define __SHARC_BINDINGS_HLSLI__

#include <donut/shaders/bindless.h>
#include <donut/shaders/binding_helpers.hlsli>

// ALL OF THESE MUST BE REMOVED FROM HERE IN THE FUTURE - make them dynamic macro or const buffer settings
// Must be defined before SharcCommon.h so the SDK declares the matching SharcHitData fields
// (materialDemodulation, emissive). Each feature is enabled at runtime via SharcConstants
// so the UI can toggle without recompiling shaders.
#define SHARC_MATERIAL_DEMODULATION 1
#define SHARC_SEPARATE_EMISSIVE     1
#define SHARC_RESAMPLING_DEPTH_MIN  2
// 3 live vertices - what SHaRC 1.6.3's default of 4 actually kept (it only used DEPTH-1); 1.7.2 fixed that and lowered
// the default to 2. TODO: check if longer paths are beneficial at the cost of increased register pressure
#define SHARC_PROPAGATION_DEPTH     3
#define HASH_GRID_USE_NORMALS       1

// SHaRC code requires these macros to be set depending on which stage we are in
#if RTXPT_SHARC_TRAIN
#define SHARC_UPDATE 1
#endif

#include "../../../External/Sharc/include/SharcCommon.h"

#if RTXPT_SHARC_TRAIN
#undef SHARC_UPDATE
#endif

// HashGridCommon.h defaults HASH_GRID_ENABLE_64_BIT_ATOMICS to 1 (64-bit atomics need shader model 6.6+).
// The 0 path makes the SDK fall back to a spin lock over a separate lock buffer. This integration does not
// allocate one - fail explicitly here rather than silently spinning on an unbound resource.
#if !HASH_GRID_ENABLE_64_BIT_ATOMICS
#error "SHaRC requires 64-bit atomics (shader model 6.6+); the lock buffer fallback path is not wired up."
#endif

#include "../Libraries/SHaRC/SharcConstants.h"

#include "../Bindless.hlsli"

// Custom types require custom handling - otherwise, all generic types should be in Bindless.hlsli
// Both are 16 bytes (SharcAccumulationData is a uint4; SharcPackedData is float16_t4 + 2 uints - see
// External/SHARC/include/SharcTypes.h) and SharcIntegration creates both buffers with that structStride, so the
// STRUCTURED variant is what matches. Registering them with the plain RTXPT_BINDLESS_REGISTER_TYPE implies an
// expected stride of 0, which fails the metadata compare on every single fetch and fires BREAKPOINT() constantly -
// exhausting ShaderDebug's 8-report budget so that real traps never get reported.
RTXPT_BINDLESS_REGISTER_STRUCTURED_TYPE( RWStructuredBuffer<SharcAccumulationData>, true, 16 )
RTXPT_BINDLESS_REGISTER_STRUCTURED_TYPE( RWStructuredBuffer<SharcPackedData>,       true, 16 )

// The stride is spelled out in SharcConstants.h because the C++ side sizes the path state buffer with it and cannot
// see SHARC_PROPAGATION_DEPTH (an SDK macro). One uint4 header plus one uint4 per propagation entry: [0] is the
// header, [1+i] is entry i - see SharcLoadState/SharcStoreState in PathTracer/PathTracerSHaRC.hlsli.
#if RTXPT_SHARC_PATH_STATE_STRIDE != (16 * (1 + SHARC_PROPAGATION_DEPTH))
#error "RTXPT_SHARC_PATH_STATE_STRIDE (SharcConstants.h) is out of step with SHARC_PROPAGATION_DEPTH - update both."
#endif


// The cache buffers have no binding slots of their own - they are fetched from the descriptor heap using
// indices carried in the constants (SharcIntegration refreshes them every frame). u90-u92 still exist in the
// shared global binding layout, but they belong to IntroSample's separate SHaRC implementation now.

// Per-thread scratch holding the SHaRC propagation state across the TraceRay boundary during the training pass -
// this is what used to ride in the ray payload (PathPayload slots 5..9). Untyped, so it needs no bindless type
// registration of its own and the packing can change without touching Bindless.hlsli.
RWByteAddressBuffer GetSharcPathStateBuffer(SharcConstants sharcConst)
{
    return GetBindlessUniform< RWByteAddressBuffer >( sharcConst.PathStateBindlessIndex );
}

SharcParameters GetSharcParameters(SharcConstants sharcConst)
{
    // DELIBERATELY NOT ZERO-INITIALISED. Writing `= (SharcParameters)0` here makes DXC route the bindless handles
    // through dx.op.createHandleForLib(dx.op.createHandleFromHeap(...)) instead of straight to annotateHandle, and
    // the NVIDIA driver access-violates on that DXIL shape while creating the DXR state object - a hard crash at
    // pipeline creation, with no D3D12 error first. Plain ResourceDescriptorHeap use in a ray tracing library is
    // fine; it is specifically zero-initialising a struct that holds resource handles that triggers it.
    // Every field below is assigned unconditionally, so nothing is left undefined:
    //   hashGridParameters (4 fields), hashGridData.capacity + .hashEntriesBuffer, radianceScale,
    //   accumulationBuffer, resolvedBuffer.
    //   (hashGridData.lockBuffer is compiled out - see the 64-bit atomics #error above.)
    SharcParameters sharcParameters;

    // Fetched into locals, not straight into the struct: DXC's SPIR-V backend (1.9.2602 to 1.9.2609 at least) fails a
    // heap fetch assigned to a struct member with "cannot handle associated counter variable assignment". It handles
    // the hidden counter before evaluating the heap access, which is what creates the counter. DXIL is unaffected.
    RWStructuredBuffer<uint64_t>                hashEntries  = GetBindlessUniform< RWStructuredBuffer<uint64_t> >( sharcConst.HashEntriesBindlessIndex );
    RWStructuredBuffer<SharcAccumulationData>   accumulation = GetBindlessUniform< RWStructuredBuffer<SharcAccumulationData> >( sharcConst.AccumulationBindlessIndex );
    RWStructuredBuffer<SharcPackedData>         resolved     = GetBindlessUniform< RWStructuredBuffer<SharcPackedData> >( sharcConst.ResolvedBindlessIndex );

    sharcParameters.hashGridParameters.cameraPosition = sharcConst.CameraPosition.xyz;
    sharcParameters.hashGridParameters.sceneScale = sharcConst.SceneScale;
    sharcParameters.hashGridParameters.logarithmBase = SHARC_GRID_LOGARITHM_BASE;
    sharcParameters.hashGridParameters.levelBias = SHARC_GRID_LEVEL_BIAS;

    sharcParameters.hashGridData.capacity = sharcConst.EntriesNum;
    sharcParameters.hashGridData.hashEntriesBuffer = hashEntries;

    sharcParameters.radianceScale = 1.0e3f; // 1.0e3f; // Replace SHARC_RADIANCE_SCALE with SharcParameters::radianceScale.

    sharcParameters.accumulationBuffer = accumulation;
    sharcParameters.resolvedBuffer     = resolved;

    return sharcParameters;
}

#endif //__SHARC_BINDINGS_HLSLI__
