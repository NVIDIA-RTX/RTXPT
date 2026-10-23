/*
* Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

// Stream compaction for the list based sub-sampling pattern (STOCHASTIC_COMPACTED: white noise coin flip per pixel):
// the selected pixels are appended to u_SubSamplingPixelList, which the fill pass raygen then walks with a 1D launch.
// Compaction is two level - within the group (CTA) using wave intrinsics plus one groupshared atomic per wave,
// then globally with a single atomic per group - so each group writes one contiguous, tile-ordered range.
// Expects the list count (u_SubSamplingPixelList[0]) to be reset to 0 before the dispatch; push constants as for
// the fill pass (see SubSampling.h).

#define NON_PATH_TRACING_PASS 1

#include "../Shaders/Bindings/ShaderResourceBindings.hlsli"
#include "../Shaders/Misc/SubSampling.h"

groupshared uint gs_groupCount;     // selected pixels in this group; each wave reserves its range from it
groupshared uint gs_groupOffset;    // this group's first slot in the global list

[numthreads(RTXPT_SUB_SAMPLING_COMPACTION_TILE, RTXPT_SUB_SAMPLING_COMPACTION_TILE, 1)]
void main( uint2 dispatchThreadID : SV_DispatchThreadID, uint groupIndex : SV_GroupIndex )
{
    // No early out: every lane has to take part in the wave intrinsics and group barriers below.
    const uint2 pixelPos = dispatchThreadID;
    const uint frameSeed = SubSampling_FrameSeed();
    const bool selected = all(pixelPos < g_Const.ptConsts.camera.ViewportSize) && SubSampling_IsPixelSelected(pixelPos, frameSeed, g_MiniConst.params1.w, g_MiniConst.params1.z);

    if (groupIndex == 0)
        gs_groupCount = 0;
    GroupMemoryBarrierWithGroupSync();

    // 1.) CTA level: each wave reserves its range with one groupshared atomic, broadcast to its lanes - no assumption
    // about how lanes map to waves. (the wave total must be computed with all lanes active)
    const uint laneOffset = WavePrefixCountBits(selected);
    const uint waveTotal  = WaveActiveCountBits(selected);
    uint waveOffset = 0;
    if (WaveIsFirstLane())
        InterlockedAdd(gs_groupCount, waveTotal, waveOffset);
    waveOffset = WaveReadLaneFirst(waveOffset);
    GroupMemoryBarrierWithGroupSync();

    // 2.) Global: reserve the group's range with one atomic.
    if (groupIndex == 0)
    {
        uint groupOffset = 0;
        if (gs_groupCount > 0)
            u_SubSamplingPixelList.InterlockedAdd(0, gs_groupCount, groupOffset);
        gs_groupOffset = groupOffset;
    }
    GroupMemoryBarrierWithGroupSync();

    if (selected)
    {
        const uint index = gs_groupOffset + waveOffset + laneOffset;
        u_SubSamplingPixelList.Store(RTXPT_SUB_SAMPLING_LIST_HEADER_BYTES + index * 4, pixelPos.x | (pixelPos.y << 16));
    }
}
