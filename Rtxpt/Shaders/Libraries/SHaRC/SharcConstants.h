/*
* Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#ifndef __SHARC_CONSTANTS_H__
#define __SHARC_CONSTANTS_H__

#if defined(__cplusplus)
using namespace donut::math;
#endif

// Size, in bytes, of one thread's slot in the SHaRC path state buffer (see PathTracerSHaRC.hlsli for the layout).
// Shared with the C++ side, which sizes the buffer with it; SharcBindings.hlsli #errors if it ever drifts out of
// step with SHARC_PROPAGATION_DEPTH, which is an SDK macro the C++ side cannot see.
#define RTXPT_SHARC_PATH_STATE_STRIDE   64

struct SharcConstants
{
    int AccumulationFrameNum;
    int StaleFrameNum;
    int SharcDebug;
    int padding0;

    int EntriesNum;
    int DownscaleFactor;
    float SceneScale;
    float MinRoughnessOverride;

    uint MaterialDemodulation;
    uint SeparateEmissive;
    uint PathStateCapacity;             // number of thread slots in the path state buffer; only used by the RTXPT_SHADER_DEBUG bounds check
    float TerminateThreshold;

    // Descriptor heap indices for the cache buffers, fetched via GetBindlessUniform (see
    // Rtxpt/Shaders/Bindless.hlsli). Refreshed every frame from DescriptorHandle::GetIndexInHeap() - they are NOT
    // stable across scene loads, because growing the descriptor table moves its base.
    uint HashEntriesBindlessIndex;
    uint AccumulationBindlessIndex;
    uint ResolvedBindlessIndex;
    uint PathStateBindlessIndex;        // per-thread SHaRC propagation state; see GetSharcPathStateBuffer

    float4 CameraPosition;
    float4 CameraPositionPrev;
};

#endif // __SHARC_CONSTANTS_H__
