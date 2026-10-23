/*
* Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#ifndef __INTRO_SAMPLE_SHADOW_RAY_HLSLI__
#define __INTRO_SAMPLE_SHADOW_RAY_HLSLI__

// Trace a ray using inline RayQuery. Returns true if the path is unoccluded.
// Requires SceneBVH (from SceneBindings.hlsli) to be declared before including this header.
bool TraceVisibilityRay(float3 origin, float3 direction, float distance)
{
    RayDesc ray;
    ray.Origin = origin;
    ray.Direction = direction;
    ray.TMin = 0;
    ray.TMax = distance;
    
#if (__SHADER_TARGET_MAJOR > 6) || (__SHADER_TARGET_MAJOR == 6 && __SHADER_TARGET_MINOR >= 9)
    RayQuery<RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH, RAYQUERY_FLAG_ALLOW_OPACITY_MICROMAPS> query;
#else
    RayQuery<RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> query;
#endif

#if 1
    query.TraceRayInline(SceneBVH, RAY_FLAG_NONE, 0xff, ray);

    while (query.Proceed())
    {
        if (query.CandidateType() == CANDIDATE_NON_OPAQUE_TRIANGLE)
        {
            uint subIdx = query.CandidateInstanceID() + query.CandidateGeometryIndex();
            bool excludeFromNEE = (t_SubInstanceData[subIdx].GlobalGeometryIndex_FlagsAndAlphaCutoff & SubInstanceData::Flags_ExcludeFromNEE) != 0;
            if (!excludeFromNEE)
            {
                query.CommitNonOpaqueTriangleHit();
            }
        }
    }
#else
    query.TraceRayInline(SceneBVH, RAY_FLAG_FORCE_OPAQUE, 0xff, ray);
    query.Proceed();
#endif
    return query.CommittedStatus() == COMMITTED_NOTHING;
}

#endif // __INTRO_SAMPLE_SHADOW_RAY_HLSLI__
