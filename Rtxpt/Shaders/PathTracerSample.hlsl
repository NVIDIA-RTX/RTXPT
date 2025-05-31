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

#define RTXPT_COMPILE_WITH_NEE 1

// this sets up various approaches - different combinations are possible
// this will get significantly simplified once SER API is part of DirectX - we'll then be able to default to "TraceRayInline" version in all variants
#if USE_NVAPI_HIT_OBJECT_EXTENSION
#define NV_HITOBJECT_USE_MACRO_API 1
#include <NVAPI/nvHLSLExtns.h>
#endif

#if PATH_TRACER_MODE==PATH_TRACER_MODE_BUILD_STABLE_PLANES
#define SER_USE_SORTING 0
#else
#define SER_USE_SORTING 1
#endif

#include "PathTracer/PathTracerTypes.hlsli"

#include "Bindings/ShaderResourceBindings.hlsli"

#include "PathTracerBridgeDonut.hlsli"
#include "PathTracer/PathTracer.hlsli"

PathTracer::WorkingContext getWorkingContext(uint2 pixelPos)
{
    PathTracer::WorkingContext ret;
    ret.ptConsts = g_Const.ptConsts;
    ret.pixelPos = pixelPos;
    //ret.pixelStorageIndex = PixelCoordToIndex(pixelPos, g_Const.ptConsts.imageWidth);
    ret.debug.Init( pixelPos, g_Const.debug, u_FeedbackBuffer, u_DebugLinesBuffer, u_DebugDeltaPathTree, u_DeltaPathSearchStack, u_DebugVizOutput );
    ret.stablePlanes = StablePlanesContext::make(pixelPos, u_StablePlanesHeader, u_StablePlanesBuffer, u_StableRadiance, u_SecondarySurfaceRadiance, g_Const.ptConsts);

    return ret;
}

PathTracer::WorkingContext getWorkingContext(const PathState path)
{
    return getWorkingContext(PathIDToPixel(path.id));
}

#if PATH_TRACER_MODE!=PATH_TRACER_MODE_REFERENCE
void firstHitFromBasePlane(inout PathState path, const uint basePlaneIndex, const PathTracer::WorkingContext workingContext)
{
    PackedHitInfo packedHitInfo; float3 rayDir; uint vertexIndex; uint stableBranchID; float sceneLength; float3 thp; float3 motionVectors;
    workingContext.stablePlanes.LoadStablePlane(workingContext.pixelPos, basePlaneIndex, vertexIndex, packedHitInfo, stableBranchID, rayDir, sceneLength, thp, motionVectors);

    // reconstruct ray; this is the ray we used to get to this hit, and Direction and rayTCurrent will not be identical due to compression
    RayDesc ray;
    ray.Direction   = rayDir;
    ray.Origin      = path.origin;  // initialized by 'PathTracer::pathSetupPrimaryRay' - WARNING, THIS WILL NOT BE CORRECT FOR NON-PRIMARY BOUNCES
    ray.TMin        = 0;
    ray.TMax        = sceneLength;  // total ray travel so far - used to correctly update rayCone and similar

    // this only works for primary surface replacement cases - in this case sceneLength and rayT become kind of the same
    path.setVertexIndex(vertexIndex-1); // decrement counter by 1 since we'll be processing hit (and calling PathTracer::updatePathTravelled) inside hit/miss shader

    // we're starting from the plane 0 (that's our vbuffer)
    path.setFlag(PathFlags::stablePlaneOnPlane , true);
    path.setFlag(PathFlags::stablePlaneOnBranch, true);
    path.setStablePlaneIndex(basePlaneIndex);
    path.stableBranchID = stableBranchID;
    path.thp = thp;
#if PATH_TRACER_MODE==PATH_TRACER_MODE_FILL_STABLE_PLANES
    const uint dominantSPIndex = workingContext.stablePlanes.LoadDominantIndexCenter();
    path.setFlag(PathFlags::stablePlaneOnDominantBranch, dominantSPIndex == basePlaneIndex ); // dominant plane has been determined in _BUILD_PASS; see if it's basePlaneIndex and set flag
    path.setCounter(PackedCounters::BouncesFromStablePlane, 0);
    path.denoiserSampleHitTFromPlane = 0.0;
    path.denoiserDiffRadianceHitDist = lpfloat4(0,0,0,0);
    path.denoiserSpecRadianceHitDist = lpfloat4(0,0,0,0);
#endif
    if (PathTracer::HasFinishedSurfaceBounces(path.getVertexIndex()+1, path.getCounter(PackedCounters::DiffuseBounces)))
        path.setTerminateAtNextBounce();

    if (!IsValid(packedHitInfo))
    {
        // inline miss shader!
        PathTracer::HandleMiss(path, ray.Origin, ray.Direction, ray.TMax, workingContext);
    }
    else
    {
#if USE_NVAPI_HIT_OBJECT_EXTENSION
        NvHitObject hit;
        if (IsValid(packedHitInfo))
        {
            const TriangleHit triangleHit = TriangleHit::make(packedHitInfo); // if valid, we know it's a triangle hit (no support for curved surfaces yet)
            const uint instanceIndex    = triangleHit.instanceID.getInstanceIndex();
            const uint geometryIndex    = triangleHit.instanceID.getGeometryIndex();
            const uint primitiveIndex   = triangleHit.primitiveIndex;
    
            BuiltInTriangleIntersectionAttributes attrib;
            attrib.barycentrics = triangleHit.barycentrics;
            NvMakeHit( SceneBVH, instanceIndex, geometryIndex, primitiveIndex, 0, 0, 1, ray, attrib, hit );

            PathPayload payload = PathPayload::pack(path);
            
            // NOTE: Shader Execution Reordering here in the first bounce is almost always detrimental to performance because rays are still mostly coherent
            NvInvokeHitObject(SceneBVH, hit, payload);
            path = PathPayload::unpack(payload, PACKED_HIT_INFO_ZERO);  // init dummy hitinfo - it's not included in the payload to minimize register pressure
        }
#else
        // All this below is one long hack to re-cast a tiny ray with a tiny step so correct shaders get called by TraceRay; using computeRayOrigin to offset back ensuring 
        // we'll hit the same triangle.
        // None of this is needed in SER pass, and will be removed once SER API becomes more widely available.
    
        float3 surfaceHitPosW; float3 surfaceHitFaceNormW; 
        Bridge::loadSurfacePosNormOnly(surfaceHitPosW, surfaceHitFaceNormW, TriangleHit::make(packedHitInfo), workingContext.debug);   // recover surface triangle position
        bool frontFacing = dot( -ray.Direction, surfaceHitFaceNormW ) >= 0.0;

        // ensure we'll hit the same triangle again (additional offset found empirially - it's still imperfect for glancing rays)
        const float offsetEpsilon = 8e-5;
        float3 newOrigin = ComputeRayOrigin(surfaceHitPosW, (frontFacing)?(surfaceHitFaceNormW):(-surfaceHitFaceNormW)) - ray.Direction * offsetEpsilon;

        // update path state as we'll skip everything up to the surface, thus we must account for the skip which is 'length(newOrigin-ray.Origin)'
        PathTracer::UpdatePathTravelled(path, ray.Origin, ray.Direction, length(newOrigin-ray.Origin), workingContext, false, false); // move path internal state by the unaccounted travel, but don't increment vertex index or update origin/rayDir

        ray.Origin = newOrigin; // move to a new starting point; leave ray.TMax as is as we can't reliably compute min travel to ensure hit but we know it's less than current TMax (sceneLength)
    
        PathPayload payload = PathPayload::pack(path);
        TraceRay( SceneBVH, RAY_FLAG_NONE, 0xff, 0, 1, 0, ray, payload );
        path = PathPayload::unpack(payload, PACKED_HIT_INFO_ZERO);
#endif // USE_NVAPI_HIT_OBJECT_EXTENSION
    }
}
#endif

void nextHit(inout PathState path, const PathTracer::WorkingContext workingContext, uniform bool skipStablePlaneExploration)
{
#if USE_DX_HIT_OBJECT_EXTENSION
    RayDesc ray; RayQuery<RAY_FLAG_NONE> rayQuery;
    PackedHitInfo packedHitInfo;
    Bridge::traceScatterRay(path, ray, rayQuery, packedHitInfo, workingContext.debug);   // this outputs ray and rayQuery; if there was a hit, ray.TMax is rayQuery.ComittedRayT

    dx::HitObject hit; // default-initialized to HitObject::MakeNop
    if (rayQuery.CommittedStatus() != COMMITTED_TRIANGLE_HIT)
    {
        // inline miss shader!
        PathTracer::HandleMiss(path, ray.Origin, ray.Direction, ray.TMax, workingContext );
    }
    else
    {
        BuiltInTriangleIntersectionAttributes attrib;
        attrib.barycentrics = rayQuery.CommittedTriangleBarycentrics();
        hit = dx::HitObject::FromRayQuery/*WithAttrs*/(rayQuery);
        hit.SetShaderTableIndex(rayQuery.CommittedInstanceContributionToHitGroupIndex()+rayQuery.CommittedGeometryIndex()); // set the record index

        //NvMakeHitWithRecordIndex( rayQuery.CommittedInstanceContributionToHitGroupIndex()+rayQuery.CommittedGeometryIndex(), SceneBVH, rayQuery.CommittedInstanceIndex(), rayQuery.CommittedGeometryIndex(), rayQuery.CommittedPrimitiveIndex(), 0, ray, attrib, hit );
        uint vertexIndex = path.getVertexIndex();
        bool terminateAtNextBounce = path.isTerminatingAtNextBounce();
        PathPayload payload = PathPayload::pack(path);

        // NOTE: only doing sorting when (vertexIndex > 0) seems to help perf but was not tested enough
#if USE_DX_MAYBE_REORDER_THREADS && SER_USE_SORTING
        //dx::MaybeReorderThread(hit, 0, 0);
        dx::MaybeReorderThread(hit, (terminateAtNextBounce)?(1):(0), 1);
#endif

        dx::HitObject::Invoke( hit, payload );
        path = PathPayload::unpack(payload, PACKED_HIT_INFO_ZERO);  // init dummy hitinfo - it's not included in the payload to minimize register pressure
    }
#elif USE_NVAPI_HIT_OBJECT_EXTENSION
    RayDesc ray; RayQuery<RAY_FLAG_NONE> rayQuery;
    PackedHitInfo packedHitInfo;
    Bridge::traceScatterRay(path, ray, rayQuery, packedHitInfo, workingContext.debug);   // this outputs ray and rayQuery; if there was a hit, ray.TMax is rayQuery.ComittedRayT

    NvHitObject hit;
    if (rayQuery.CommittedStatus() != COMMITTED_TRIANGLE_HIT)
    {
        //NvReorderThread(0, 1);
        // inline miss shader!
        PathTracer::HandleMiss(path, ray.Origin, ray.Direction, ray.TMax, workingContext );
    }
    else
    {
        BuiltInTriangleIntersectionAttributes attrib;
        attrib.barycentrics = rayQuery.CommittedTriangleBarycentrics();
        NvMakeHitWithRecordIndex( rayQuery.CommittedInstanceContributionToHitGroupIndex()+rayQuery.CommittedGeometryIndex(), SceneBVH, rayQuery.CommittedInstanceIndex(), rayQuery.CommittedGeometryIndex(), rayQuery.CommittedPrimitiveIndex(), 0, ray, attrib, hit );
        uint vertexIndex = path.getVertexIndex();
        bool terminateAtNextBounce = path.isTerminatingAtNextBounce();
        PathPayload payload = PathPayload::pack(path);

        // NOTE: only doing sorting when (vertexIndex > 0) can help perf in some cases
#if USE_NVAPI_REORDER_THREADS && SER_USE_SORTING
        //NvReorderThread(hit, 0, 0);
        NvReorderThread(hit, (terminateAtNextBounce)?(1):(0), 1);         // for testing it is interesting to compare to NvReorderThread(SERSortKey, 16);  - manual sort key isn't nearly as efficient
#endif

        NvInvokeHitObject(SceneBVH, hit, payload);
        path = PathPayload::unpack(payload, PACKED_HIT_INFO_ZERO);  // init dummy hitinfo - it's not included in the payload to minimize register pressure
    }
#else
    // refactor...
    RayDesc ray = path.getScatterRay().toRayDesc();
    PathPayload payload = PathPayload::pack(path);
    TraceRay( SceneBVH, RAY_FLAG_NONE, 0xff, 0, 1, 0, ray, payload );
    path = PathPayload::unpack(payload, PACKED_HIT_INFO_ZERO);  // init dummy hitinfo - it's not included in the payload to minimize register pressure
#endif

#if PATH_TRACER_MODE==PATH_TRACER_MODE_BUILD_STABLE_PLANES   // explore enqueued stable planes, if any
    int nextPlaneToExplore;
    if (!path.isActive() && (nextPlaneToExplore=workingContext.stablePlanes.FindNextToExplore(path.getStablePlaneIndex()+1))!=-1 )
    {
        float3 prevL = path.L;  // save non-noisy radiance captured so far
        PathPayload payload;
        workingContext.stablePlanes.ExplorationStart(nextPlaneToExplore, payload.packed);
        //PathState newPayload = PathPayload::unpack(payload, PACKED_HIT_INFO_ZERO);
        path = PathPayload::unpack(payload, PACKED_HIT_INFO_ZERO);
        path.L = prevL;         // keep accumulating non-noisy radiance captured so far

		#if 0 // way of debugging contents of stable plane index 1
        if (path.getStablePlaneIndex()==1)
            workingContext.debug.DrawDebugViz( float4( DbgShowNormalSRGB(path.dir), 1 ) );
        #endif
    }
#endif
}

#if ENABLE_DEBUG_DELTA_TREE_VIZUALISATION
// figure out where to move this so it's not in th emain path tracer code
void DeltaTreeVizExplorePixel(PathTracer::WorkingContext workingContext, uint subSampleIndex);
#endif

void SanitizeNaNs(inout PathState path, PathTracer::WorkingContext workingContext)
{
#if 1   // sanitize NaNs/infinities
    bool somethingWrong = false;
#if PATH_TRACER_MODE!=PATH_TRACER_MODE_FILL_STABLE_PLANES
    somethingWrong |= any(isnan(path.L)) || !all(isfinite(path.L));
#endif
    somethingWrong |= any(isnan(path.thp)) || !all(isfinite(path.thp));
    [branch] if (somethingWrong)
    {
#if ENABLE_DEBUG_VIZUALISATIONS
        workingContext.debug.DrawDebugViz( workingContext.pixelPos, float4(0, 0, 0, 1 ) );
        for( int k = 1; k < 6; k++ )
        {
            workingContext.debug.DrawDebugViz( workingContext.pixelPos+uint2(+k,+0), float4(1-(k/2)%2, (k/2)%2, k%5, 1 ) );
            workingContext.debug.DrawDebugViz( workingContext.pixelPos+uint2(-k,+0), float4(1-(k/2)%2, (k/2)%2, k%5, 1 ) );
            workingContext.debug.DrawDebugViz( workingContext.pixelPos+uint2(+0,+k), float4(1-(k/2)%2, (k/2)%2, k%5, 1 ) );
            workingContext.debug.DrawDebugViz( workingContext.pixelPos+uint2(+0,-k), float4(1-(k/2)%2, (k/2)%2, k%5, 1 ) );
        }
#endif
#if PATH_TRACER_MODE!=PATH_TRACER_MODE_FILL_STABLE_PLANES
        path.L = 0;
#endif
        path.thp = 0;
    }
#endif    
}

[shader("raygeneration")]
void RayGen()
{
    uint2 pixelPos = DispatchRaysIndex().xy;

    PathTracer::WorkingContext workingContext = getWorkingContext( pixelPos );

#if PATH_TRACER_MODE!=PATH_TRACER_MODE_REFERENCE
    workingContext.stablePlanes.StartPathTracingPass();
#endif

#if PATH_TRACER_MODE!=PATH_TRACER_MODE_FILL_STABLE_PLANES    // Reset on stable planes disabled (0) or stable planes generate (1)
    workingContext.debug.Reset(0);   // Setups per-pixel debugging - has to happen before any other debugging stuff in the frame
#endif

    uint subSampleIndex = 0;

    float3 lastOrigin = float3(1e30f,1e30f,1e30f);
    PathState path;

    path = PathTracer::EmptyPathInitialize(pixelPos, g_Const.ptConsts.camera.PixelConeSpreadAngle);
    PathTracer::SetupPathPrimaryRay(path, Bridge::computeCameraRay(pixelPos, /*subSampleIndex*/0));  // note: all realtime mode subSamples currently share same camera ray at subSampleIndex == 0 (otherwise denoising guidance buffers would be noisy)

#if PATH_TRACER_MODE==PATH_TRACER_MODE_FILL_STABLE_PLANES    // we're continuing from base stable plane (index 0) here to avoid unnecessary path tracing
    firstHitFromBasePlane(path, 0, workingContext);
    lastOrigin = path.origin;
#endif

#if PATH_TRACER_MODE==PATH_TRACER_MODE_BUILD_STABLE_PLANES     // BUILD
    u_SecondarySurfacePositionNormal[pixelPos] = 0;
#endif

    // Main path tracing loop
    while (path.isActive())
        nextHit(path, workingContext, false);
    
    SanitizeNaNs(path, workingContext);
       
#if PATH_TRACER_MODE==PATH_TRACER_MODE_FILL_STABLE_PLANES
    workingContext.stablePlanes.CommitDenoiserRadiance(path.getStablePlaneIndex(), path.denoiserSampleHitTFromPlane,
    path.denoiserDiffRadianceHitDist, path.denoiserSpecRadianceHitDist,
    path.secondaryL, path.hasFlag(PathFlags::stablePlaneBaseScatterDiff),
    path.hasFlag(PathFlags::stablePlaneOnDeltaBranch),
    path.hasFlag(PathFlags::stablePlaneOnDominantBranch));
#endif    

#if PATH_TRACER_MODE==PATH_TRACER_MODE_REFERENCE
    float3 pathRadiance = path.L;
#elif PATH_TRACER_MODE==PATH_TRACER_MODE_BUILD_STABLE_PLANES
    float3 pathRadiance = workingContext.stablePlanes.CompletePathTracingBuild(path.L);
#elif PATH_TRACER_MODE==PATH_TRACER_MODE_FILL_STABLE_PLANES
    float3 pathRadiance = workingContext.stablePlanes.CompletePathTracingFill(g_Const.ptConsts.denoisingEnabled);
#endif
        
#if PATH_TRACER_MODE==PATH_TRACER_MODE_BUILD_STABLE_PLANES && ENABLE_DEBUG_DELTA_TREE_VIZUALISATION
    DeltaTreeVizExplorePixel(workingContext, 0);
    return;
#endif

    //if ( length(float2(pixelPos.xy) - float2(800, 500)) < 100 ) // draw a circle for testing
    //    pathRadiance.z += 10;

#if PATH_TRACER_MODE!=PATH_TRACER_MODE_BUILD_STABLE_PLANES
    u_OutputColor[pixelPos] = float4( pathRadiance, 1 );   // <- alpha 1 is important for screenshots
#endif
    
    // if( workingContext.debug.IsDebugPixel() )
    //     workingContext.debug.Print( 0, Bridge::getSampleIndex(), Hash32(Bridge::getSampleIndex()) );

//  debugging examples:
//    if( workingContext.debug.IsDebugPixel() )
//    {
//        workingContext.debug.Print( 0, pathRadiance);
//        workingContext.debug.Print( 1, path.thp);
//        workingContext.debug.Print( 2, path.getCounter(PackedCounters::DiffuseBounces));
//        workingContext.debug.Print( 3, path.getCounter(PackedCounters::RejectedHits));
//    }
//    if (all(pixelPos > uint2(400, 400)) && all(pixelPos < uint2(600, 600)))
//        u_OutputColor[pixelPos] = float4( g_Const.ptConsts.preExposedGrayLuminance.xxx, 1 ); 

}

#if ENABLE_DEBUG_DELTA_TREE_VIZUALISATION
void DeltaTreeVizExplorePixel(PathTracer::WorkingContext workingContext, uint subSampleIndex)
{
    if (workingContext.debug.constants.exploreDeltaTree && workingContext.debug.IsDebugPixel())
    {
        // setup path normally
        PathState path = PathTracer::EmptyPathInitialize( workingContext.debug.pixelPos, g_Const.ptConsts.camera.pixelConeSpreadAngle, subSampleIndex );
        PathTracer::SetupPathPrimaryRay( path, Bridge::computeCameraRay( workingContext.debug.pixelPos, /*subSampleIndex*/0 ) );  // note: all realtime mode subSamples currently share same camera ray at subSampleIndex == 0 (otherwise denoising guidance buffers would be noisy)
        // but then make delta lobes split into their own subpaths that get saved into debug stack with workingContext.debug.DeltaSearchStackPush()
        path.setFlag(PathFlags::deltaTreeExplorer);
        // start with just primary ray
        nextHit(path, workingContext, true);

        PathPayload statePacked; int loop = 0;
        while ( workingContext.debug.DeltaSearchStackPop(statePacked) )
        {
            loop++; 
            PathState deltaPathState = PathPayload::unpack( statePacked, PACKED_HIT_INFO_ZERO );
            nextHit(deltaPathState, workingContext, true);
        }
        for (int i = 0; i < cStablePlaneCount; i++)
            workingContext.debug.DeltaTreeStoreStablePlaneID( i, workingContext.stablePlanes.GetBranchIDCenter(i) );
        workingContext.debug.DeltaTreeStoreDominantStablePlaneIndex( workingContext.stablePlanes.LoadDominantIndexCenter() );
    }
}
#endif

void HandleHitUnpacked(const uniform PathTracer::OptimizationHints optimizationHints, const PackedHitInfo packedHitInfo, inout PathState path, float3 worldRayOrigin, float3 worldRayDirection, float rayT, const PathTracer::WorkingContext workingContext)
{
    // reconstruct previous origin & dir (avoids actually unpacking .origin from PathPayload); TODO: refactor this so the scatter ray (next ray) is in a separate payload
    path.origin = worldRayOrigin;
    path.dir = worldRayDirection;
    path.setHitPacked( packedHitInfo );
    PathTracer::HandleHit(optimizationHints, path, worldRayOrigin, worldRayDirection, rayT, workingContext);
}

void HandleHit(const uniform PathTracer::OptimizationHints optimizationHints, const PackedHitInfo packedHitInfo, inout PathPayload payload)
{
    PathState path = PathPayload::unpack(payload, packedHitInfo);
    PathTracer::WorkingContext workingContext = getWorkingContext(PathIDToPixel(path.id));
    HandleHitUnpacked(optimizationHints, packedHitInfo, path, WorldRayOrigin(), WorldRayDirection(), RayTCurrent(), workingContext);
    payload = PathPayload::pack( path );
}

#define CLOSEST_HIT_VARIANT( name, NoTextures, NoTransmission, OnlyDeltaLobes )     \
[shader("closesthit")] void ClosestHit##name(inout PathPayload payload : SV_RayPayload, in BuiltInTriangleIntersectionAttributes attrib) \
{ \
    HandleHit( PathTracer::OptimizationHints::make( NoTextures, NoTransmission, OnlyDeltaLobes ), TriangleHit::make( InstanceIndex(), GeometryIndex(), PrimitiveIndex(), attrib.barycentrics ).pack(), payload); \
}

//hints: NoTextures, NoTransmission, OnlyDeltaLobes
#if 1 // 3bit 8-variant version
CLOSEST_HIT_VARIANT( 000, false, false, false );
CLOSEST_HIT_VARIANT( 001, false, false, true  );
CLOSEST_HIT_VARIANT( 010, false, true,  false );
CLOSEST_HIT_VARIANT( 011, false, true,  true  );
CLOSEST_HIT_VARIANT( 100, true,  false, false );
CLOSEST_HIT_VARIANT( 101, true,  false, true  );
CLOSEST_HIT_VARIANT( 110, true,  true,  false );
CLOSEST_HIT_VARIANT( 111, true,  true,  true  );
#endif

// These two are required for the full TraceRay support
[shader("miss")]
void Miss(inout PathPayload payload : SV_RayPayload)
{
#if USE_NVAPI_HIT_OBJECT_EXTENSION || USE_DX_HIT_OBJECT_EXTENSION
    // we inline misses in rgs, so this is a no-op.
#else
    PathState path = PathPayload::unpack(payload, PACKED_HIT_INFO_ZERO);
    PathTracer::HandleMiss(path, WorldRayOrigin(), WorldRayDirection(), RayTCurrent(), getWorkingContext(PathIDToPixel(path.id)));
    payload = PathPayload::pack(path);
#endif
}

[shader("anyhit")]
void AnyHit(inout PathPayload payload, in BuiltInTriangleIntersectionAttributes attrib/* : SV_IntersectionAttributes*/)
{
    if (!Bridge::AlphaTest(InstanceID(), InstanceIndex(), GeometryIndex(), PrimitiveIndex(), attrib.barycentrics/*, getWorkingContext( PathIDToPixel(path.id) ).debug*/ ))
        IgnoreHit();
}
