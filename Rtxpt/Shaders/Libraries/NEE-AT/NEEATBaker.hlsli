/*
* Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#ifndef __NEEAT_BAKER_HLSL__
#define __NEEAT_BAKER_HLSL__

// this will enable various valdiation passes which will debug print or otherwise indicate issues with the algorithm or input data / buffers setup
#define LLB_ENABLE_VALIDATION           0

#define LLB_NUM_COMPUTE_THREADS         128
#define LLB_NUM_COMPUTE_THREADS_2D      8
#define LLB_LOCAL_BLOCK_SIZE            32

#define LLB_PREPROCESS_BLOCK_SIZE_OUTER 16
#define LLB_PREPROCESS_BLOCK_SIZE_INNER (LLB_PREPROCESS_BLOCK_SIZE_OUTER-2)

#define RTXPT_LIGHTING_CPJ_BLOCKSIZE    1024

#define LLB_MAX_TRIANGLES_PER_TASK      32
#define LLB_MAX_PROC_TASKS              (RTXPT_LIGHTING_MAX_LIGHTS / LLB_MAX_TRIANGLES_PER_TASK * 2)
struct EmissiveTrianglesProcTask
{
    uint InstanceIndex;
    uint GeometryIndex;
    uint TriangleIndexFrom;
    uint TriangleIndexTo;
    uint DestinationBufferOffset;
    uint HistoricBufferOffset;
    uint EmissiveLightMappingOffset;
    uint Padding0;
};

#define LLB_MAX_PROXIES_PER_TASK        32
#define LLB_MAX_PROXY_PROC_TASKS        (RTXPT_LIGHTING_MAX_LIGHTS+(RTXPT_LIGHTING_MAX_SAMPLING_PROXIES+LLB_MAX_PROXIES_PER_TASK-1) / LLB_MAX_PROXIES_PER_TASK)
struct SamplingProxyBuildProcTask
{
    uint LightIndex; // <- index into u_lightsBuffer
    uint ProxyIndexBase; // useful for figuring out sampling proxy index within its own proxies
    uint FillProxyIndexFrom; // this task needs to fill from this index                                  
    uint FillProxyIndexTo; // this task needs to fill to this index                                    
};

#define LLB_SCRATCH_BUFFER_SIZE         (32*1024*1024) // this is in bytes

// count in float-s; buffer size in bytes should be 2 * sizeof(float) * RTXPT_LIGHTING_WEIGHTS_COUNT_HALF
#define RTXPT_LIGHTING_WEIGHTS_COUNT_HALF     (RTXPT_LIGHTING_MAX_LIGHTS+1)   // +1 is purely because perLightProxyCounters needs one more to store invalid feedback; "half" is because we need 2x so we can ping-pong with historic

#if defined(__cplusplus)
static_assert( sizeof(EmissiveTrianglesProcTask) * LLB_MAX_PROC_TASKS <= LLB_SCRATCH_BUFFER_SIZE ); // does it fit
static_assert( sizeof(SamplingProxyBuildProcTask) * LLB_MAX_PROXY_PROC_TASKS <= LLB_SCRATCH_BUFFER_SIZE ); // does it fit
static_assert( (RTXPT_LIGHTING_MAX_LIGHTS / LLB_MAX_TRIANGLES_PER_TASK * 2) <= LLB_MAX_PROC_TASKS );
#endif


#if !defined(__cplusplus)
#pragma pack_matrix(row_major)

#include "../MicroRng.hlsli"

#endif


#endif // __NEEAT_BAKER_HLSL__