/*
* Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "PathTracer/Config.h" // must always be included first

#include "SERUtils.hlsli"

#if PATH_TRACER_MODE==PATH_TRACER_MODE_BUILD_STABLE_PLANES
#define SER_USE_SORTING 0
#else
#define SER_USE_SORTING 1
#endif

#include "PathTracer/PathTracerTypes.hlsli"

#include "Bindings/ShaderResourceBindings.hlsli"
#if PT_USE_RESTIR_GI
#include "Bindings/ReSTIRBindings.hlsli"
#endif

#include "PathTracerBridgeDonut.hlsli"
#include "PathTracer/PathTracer.hlsli"

// TODO: move this to PathTracer once SER is unified
// NOTE: this only works correctly (reconstructs the path fully) with primary hit (plus PSR so all mirrors are fine)
#if PATH_TRACER_MODE==PATH_TRACER_MODE_FILL_STABLE_PLANES || defined(__INTELLISENSE__)
float2 FirstHitFromVBuffer(inout PathState path, const uint basePlaneIndex, const PathTracer::WorkingContext workingContext)
{
    const uint2 pixelPos = path.GetPixelPos();

    float2 tMinMax = float2(0, kMaxRayTravel);

    StablePlane sp = workingContext.StablePlanes.LoadStablePlane(pixelPos, basePlaneIndex);
    uint stableBranchID = workingContext.StablePlanes.GetBranchID(pixelPos, basePlaneIndex); 
    float sceneLength = sp.SceneLength;
    float lastRayTCurrent = sp.LastRayTCurrent;
    uint vertexIndex = sp.VertexIndexAndRoughness>>16; 
    float3 thp, dummy;
    UnpackTwoFp32ToFp16(sp.PackedThpAndMVs, thp, dummy);

    bool isMiss = false;
    if (!isfinite(sceneLength))
    {
        sceneLength = kMaxRayTravel;
        isMiss = true;
    }
    else
    {
        // we know where/what we want to hit, so reduce the ray travel for performance reasons
        tMinMax.x = lastRayTCurrent * 0.99;
        tMinMax.y = lastRayTCurrent * 1.01;
        sceneLength -= lastRayTCurrent; // we'll move the path by the unaccounted travel so far, minus the ray we're about to cast
    }
    // these are currently unnecessary; also check the side where they're set, these are disabled there as well to avoid unnecessary work
    // path.flagsAndVertexIndex = sp.FlagsAndVertexIndex;      // vertexIndex will be overwritten below
    // path.packedCounters = sp.PackedCounters;                // only "rejected hits" can be non-zero

    // this only works for primary surface replacement cases - in this case sceneLength and rayT become kind of the same
    path.setVertexIndex(vertexIndex-1); // decrement counter by 1 since we'll be processing hit (and calling PathTracer::updatePathTravelled) inside hit/miss shader

    // we're starting from the plane 0 (that's our vbuffer)
    path.SetDir( sp.RayDir );
    path.SetOrigin( sp.RayOrigin );
    path.setFlag(PathFlags::stablePlaneOnPlane , true);
    path.setFlag(PathFlags::stablePlaneOnBranch, true);
    path.setStablePlaneIndex(basePlaneIndex);
    path.stableBranchID = stableBranchID;
    path.SetThp( thp );
    path.SetL( float4(0,0,0,0) );
    const uint dominantSPIndex = workingContext.StablePlanes.LoadDominantIndex(pixelPos);
    path.setFlag(PathFlags::stablePlaneOnDominantBranch, dominantSPIndex == basePlaneIndex ); // dominant plane has been determined in _BUILD_PASS; see if it's basePlaneIndex and set flag
    path.setCounter(PackedCounters::BouncesFromStablePlane, 0);
    if (PathTracer::HasFinishedSurfaceBounces(path.getVertexIndex()+1, path.getCounter(PackedCounters::DiffuseBounces)))
        path.setTerminateAtNextBounce();

    // update path state as it starts from camera at sceneLength == 0, and rayCones need expanding
    PathTracer::UpdatePathTravelledLengthOnly(path, sceneLength); // move path internal state by the unaccounted travel, but don't increment vertex index or update origin/rayDir

    if (isMiss) // this means sky/miss
    {
        // inline miss shader! it marks the ray as terminated / !isActive and avoids additional raycast
        PathTracer::HandleMiss(path, path.GetOrigin(), path.GetDir(), sceneLength, workingContext);
    }
    
    return tMinMax;
}
#endif // #if PATH_TRACER_MODE==PATH_TRACER_MODE_FILL_STABLE_PLANES

void postProcessHit(inout PathState path, const PathTracer::WorkingContext workingContext)
{
#if PATH_TRACER_MODE==PATH_TRACER_MODE_BUILD_STABLE_PLANES   // explore enqueued stable planes, if any
    int nextPlaneToExplore;
    const uint2 pixelPos = path.GetPixelPos();
    if (!path.isActive() && (nextPlaneToExplore=workingContext.StablePlanes.FindNextToExplore(pixelPos, path.getStablePlaneIndex()+1))!=-1 )
    {
        PathPayload payload;
        workingContext.StablePlanes.ExplorationStart(pixelPos, nextPlaneToExplore, payload.packed);
        path = PathPayload::unpack(payload);

		#if 0 // way of debugging contents of stable plane index 1
        if (path.getStablePlaneIndex()==1)
            workingContext.Debug.DrawDebugViz( float4( DbgShowNormalSRGB(path.dir), 1 ) );
        #endif
    }
#endif
}

void nextHit(inout PathState path, inout float2 tMinMax, const PathTracer::WorkingContext workingContext)
{
#if defined(SER_HIT_OBJECT) || defined(__INTELLISENSE__)
    RayQuery<RAY_FLAG_NONE, RTXPT_FLAG_ALLOW_OPACITY_MICROMAPS> rayQuery;
    Bridge::traceScatterRay(path, rayQuery, tMinMax, workingContext.Debug);   // this outputs ray and rayQuery; if there was a hit, ray.TMax is rayQuery.ComittedRayT

    SER_HIT_OBJECT hit;
    if (rayQuery.CommittedStatus() != COMMITTED_TRIANGLE_HIT)
    {
        #if 1 // inline miss shader!
            PathTracer::HandleMiss(path, rayQuery.WorldRayOrigin(), rayQuery.WorldRayDirection(), kMaxRayTravel, workingContext);
            return;
        #else
            SER_HIT_OBJECT_INIT_FROM_MISS( hit, rayQuery );
        #endif
    }
    else
    {
        SER_HIT_OBJECT_INIT_FROM_RAYQ( hit, rayQuery );
    }

#if SER_SORT_ENABLED && SER_USE_SORTING || defined(__INTELLISENSE__)
    if (path.hasFlag( PathFlags::enableThreadReorder ))
    {
        uint terminateAtNextBounceBit = path.isTerminatingAtNextBounce();
        //uint inDielectricBounceBit = (path.getCounter(PackedCounters::RejectedHits)>0);

    #if RTXPT_DISABLE_SER_TERMINATION_HINT
        SER_REORDER_HIT(hit, 0, 0);
    #else
        SER_REORDER_HIT(hit, terminateAtNextBounceBit, 1);
    #endif
    }
#endif

    PathPayload payload = PathPayload::pack(path);
    SER_INVOKE_HIT(hit, payload);
    path = PathPayload::unpack(payload);
#else
    // refactor...
    RayDesc ray = path.getScatterRay().toRayDesc();
    ray.TMin = tMinMax.x;
    ray.TMax = tMinMax.y;
    PathPayload payload = PathPayload::pack(path);
    TraceRay( SceneBVH, RAY_FLAG_NONE, 0xff, 0, 1, 0, ray, payload );
    path = PathPayload::unpack(payload);
#endif
    tMinMax = float2(0, kMaxRayTravel); // reset - it's only designed to be used the first time after reading the ray from FirstHitFromVBuffer
}

#if ENABLE_DEBUG_DELTA_TREE_VIZUALISATION
// figure out where to move this so it's not in th emain path tracer code
void DeltaTreeVizExplorePixel(PathTracer::WorkingContext workingContext);
#endif

void ValidateNaNs(inout PathState path, PathTracer::WorkingContext workingContext)
{
#if 1   // sanitize NaNs/infinities
    bool somethingWrong = false;
#if PATH_TRACER_MODE!=PATH_TRACER_MODE_BUILD_STABLE_PLANES
    float4 L = path.GetL();
    somethingWrong |= any(isnan(L)) || !all(isfinite(L));
#endif
    somethingWrong |= any(isnan(path.GetThp())) || !all(isfinite(path.GetThp()));
    [branch] if (somethingWrong)
    {
#if ENABLE_DEBUG_VIZUALISATIONS
        uint2 pixelPos = path.GetPixelPos();
        workingContext.Debug.DrawDebugViz( pixelPos, float4(0, 0, 0, 1 ) );
        for( int k = 1; k < 6; k++ )
        {
            workingContext.Debug.DrawDebugViz( pixelPos+uint2(+k,+0), float4(1-(k/2)%2, (k/2)%2, k%5, 1 ) );
            workingContext.Debug.DrawDebugViz( pixelPos+uint2(-k,+0), float4(1-(k/2)%2, (k/2)%2, k%5, 1 ) );
            workingContext.Debug.DrawDebugViz( pixelPos+uint2(+0,+k), float4(1-(k/2)%2, (k/2)%2, k%5, 1 ) );
            workingContext.Debug.DrawDebugViz( pixelPos+uint2(+0,-k), float4(1-(k/2)%2, (k/2)%2, k%5, 1 ) );
        }
#endif
 #if PATH_TRACER_MODE!=PATH_TRACER_MODE_BUILD_STABLE_PLANES
        path.SetL(0.xxxx);
 #endif
        path.SetThp(0.xxx);
    }
#endif    
}

[shader("raygeneration")]
void RAYGEN_ENTRY()
{
    uint2 pixelPos = DispatchRaysIndex().xy;

    //float3 lastOrigin = float3(1e30f,1e30f,1e30f);
    PathState path;

    path = PathTracer::EmptyPathInitialize(pixelPos, g_Const.ptConsts.camera.PixelConeSpreadAngle);
    PathTracer::SetupPathPrimaryRay(path, Bridge::computeCameraRay(pixelPos));  // note: all realtime mode subSamples currently share same camera ray at subSampleIndex == 0 (otherwise denoising guidance buffers would be noisy)

    PathTracer::WorkingContext workingContext = GetWorkingContext();
    // clear, initialize any global backing memory, etc
    PathTracer::StartPixel(path, workingContext);

    // TODO: move this to PathTracer once DX SER API comes out of Preview and works on Vulkan
#if PATH_TRACER_MODE==PATH_TRACER_MODE_FILL_STABLE_PLANES    // we're continuing from base stable plane (index 0) here to avoid unnecessary path tracing
    float2 tMinMax = FirstHitFromVBuffer(path, 0, workingContext);
#else
    float2 tMinMax = float2(0, kMaxRayTravel);
#endif

    // Main path tracing loop
    while (path.isActive()) // NOTE: might be not active in the first instance
    {
        nextHit(path, tMinMax, workingContext);
        postProcessHit(path, workingContext);
    }
    
#if 0
    ValidateNaNs(path, workingContext);
#endif
       
    // store radiance and any other required data - we're not path tracing anymore
    PathTracer::CommitPixel( path, workingContext );
        
#if PATH_TRACER_MODE==PATH_TRACER_MODE_BUILD_STABLE_PLANES && ENABLE_DEBUG_DELTA_TREE_VIZUALISATION
    DeltaTreeVizExplorePixel(workingContext, 0);
    return;
#endif

    //if ( length(float2(pixelPos.xy) - float2(800, 500)) < 100 ) // draw a circle for testing
    //    u_OutputColor[pixelPos].z += 10;

    // if( workingContext.Debug.IsDebugPixel() )
    //     workingContext.Debug.Print( 0, Bridge::getSampleIndex(), Hash32(Bridge::getSampleIndex()) );

    //    if (all(pixelPos > uint2(400, 400)) && all(pixelPos < uint2(600, 600)))
    //        u_OutputColor[pixelPos] = float4( g_Const.ptConsts.preExposedGrayLuminance.xxx, 1 ); 

}

#if ENABLE_DEBUG_DELTA_TREE_VIZUALISATION
void DeltaTreeVizExplorePixel(PathTracer::WorkingContext workingContext)
{
    if (workingContext.Debug.constants.exploreDeltaTree && workingContext.Debug.IsDebugPixel())
    {
        // setup path normally
        PathState path = PathTracer::EmptyPathInitialize( workingContext.Debug.pixelPos, g_Const.ptConsts.camera.pixelConeSpreadAngle );
        PathTracer::SetupPathPrimaryRay( path, Bridge::computeCameraRay( workingContext.Debug.pixelPos ) );
        // but then make delta lobes split into their own subpaths that get saved into debug stack with workingContext.Debug.DeltaSearchStackPush()
        path.setFlag(PathFlags::deltaTreeExplorer);
        // start with just primary ray
        nextHit(path, workingContext, true); <- no longer valid

        PathPayload statePacked; int loop = 0;
        while ( workingContext.Debug.DeltaSearchStackPop(statePacked) )
        {
            loop++; 
            PathState deltaPathState = PathPayload::unpack( statePacked );
            nextHit(deltaPathState, workingContext, true); <- no longer valid
        }
        for (int i = 0; i < cStablePlaneCount; i++)
            workingContext.Debug.DeltaTreeStoreStablePlaneID( i, workingContext.StablePlanes.GetBranchIDCenter(i) );
        workingContext.Debug.DeltaTreeStoreDominantStablePlaneIndex( workingContext.StablePlanes.LoadDominantIndexCenter() );
    }
}
#endif
// Miss only required for the full TraceRay support - should be compiled out normally
[shader("miss")]
void MISS_ENTRY(inout PathPayload payload : SV_RayPayload)
{
//#if USE_NVAPI_HIT_OBJECT_EXTENSION || USE_DX_HIT_OBJECT_EXTENSION
//    // we inline misses in rgs, so this is a no-op.
//#else
    PathState path = PathPayload::unpack(payload);
    PathTracer::HandleMiss(path, WorldRayOrigin(), WorldRayDirection(), RayTCurrent(), GetWorkingContext());
    payload = PathPayload::pack(path);
//#endif
}
