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
#if PT_USE_RESTIR_GI
#include "Bindings/ReSTIRBindings.hlsli"
#endif

#include "PathTracerBridgeDonut.hlsli"
#include "PathTracer/PathTracer.hlsli"

PathTracer::WorkingContext GetWorkingContext(const PathState path)
{
    PathTracer::WorkingContext ret;
    ret.PtConsts = g_Const.ptConsts;
    ret.PixelID = path.id;
    ret.Debug.Init( g_Const.debug, u_FeedbackBuffer, u_DebugLinesBuffer, u_DebugDeltaPathTree, u_DeltaPathSearchStack, u_DebugVizOutput );
    ret.StablePlanes = StablePlanesContext::make(u_StablePlanesHeader, u_StablePlanesBuffer, u_StableRadiance, g_Const.ptConsts);
    ret.OutputColor = u_OutputColor;
    return ret;
}

// TODO: move this to PathTracer once SER is unified
#if PATH_TRACER_MODE!=PATH_TRACER_MODE_REFERENCE
void firstHitFromBasePlane(inout PathState path, const uint basePlaneIndex, const PathTracer::WorkingContext workingContext)
{
    const uint2 pixelPos = path.GetPixelPos();
    PackedHitInfo packedHitInfo; float3 rayDir; uint vertexIndex; uint stableBranchID; float sceneLength; float3 thp; float3 motionVectors;
    workingContext.StablePlanes.LoadStablePlane(pixelPos, basePlaneIndex, vertexIndex, packedHitInfo, stableBranchID, rayDir, sceneLength, thp, motionVectors);

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
#if PATH_TRACER_MODE!=PATH_TRACER_MODE_BUILD_STABLE_PLANES
    path.L = float4(0,0,0,0);
#endif
#if PATH_TRACER_MODE==PATH_TRACER_MODE_FILL_STABLE_PLANES
    path.sceneLengthFromDenoisingLayer = 0.0;
    path.specHitT = 0;
    const uint dominantSPIndex = workingContext.StablePlanes.LoadDominantIndex(pixelPos);
    path.setFlag(PathFlags::stablePlaneOnDominantBranch, dominantSPIndex == basePlaneIndex ); // dominant plane has been determined in _BUILD_PASS; see if it's basePlaneIndex and set flag
    path.setCounter(PackedCounters::BouncesFromStablePlane, 0);
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
        Bridge::loadSurfacePosNormOnly(surfaceHitPosW, surfaceHitFaceNormW, TriangleHit::make(packedHitInfo), workingContext.Debug);   // recover surface triangle position
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
    RayDesc ray; RayQuery<RAY_FLAG_NONE, RTXPT_FLAG_ALLOW_OPACITY_MICROMAPS> rayQuery;
    PackedHitInfo packedHitInfo;
    Bridge::traceScatterRay(path, ray, rayQuery, packedHitInfo, workingContext.Debug);   // this outputs ray and rayQuery; if there was a hit, ray.TMax is rayQuery.ComittedRayT

    dx::HitObject hit; // default-initialized to HitObject::MakeNop
    if (rayQuery.CommittedStatus() != COMMITTED_TRIANGLE_HIT)
    {
        // dx::MaybeReorderThread(0, 32);
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
    RayDesc ray; RayQuery<RAY_FLAG_NONE, RTXPT_FLAG_ALLOW_OPACITY_MICROMAPS> rayQuery;
    PackedHitInfo packedHitInfo;
    Bridge::traceScatterRay(path, ray, rayQuery, packedHitInfo, workingContext.Debug);   // this outputs ray and rayQuery; if there was a hit, ray.TMax is rayQuery.ComittedRayT

    NvHitObject hit;
    if (rayQuery.CommittedStatus() != COMMITTED_TRIANGLE_HIT)
    {
        // NvReorderThread(0, 32);
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
        NvReorderThread(hit, (terminateAtNextBounce)?(1):(0), 1);
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
    const uint2 pixelPos = path.GetPixelPos();
    if (!path.isActive() && (nextPlaneToExplore=workingContext.StablePlanes.FindNextToExplore(pixelPos, path.getStablePlaneIndex()+1))!=-1 )
    {
        PathPayload payload;
        workingContext.StablePlanes.ExplorationStart(pixelPos, nextPlaneToExplore, payload.packed);
        path = PathPayload::unpack(payload, PACKED_HIT_INFO_ZERO);

		#if 0 // way of debugging contents of stable plane index 1
        if (path.getStablePlaneIndex()==1)
            workingContext.Debug.DrawDebugViz( float4( DbgShowNormalSRGB(path.dir), 1 ) );
        #endif
    }
#endif
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
    somethingWrong |= any(isnan(path.L)) || !all(isfinite(path.L));
#endif
    somethingWrong |= any(isnan(path.thp)) || !all(isfinite(path.thp));
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

    //float3 lastOrigin = float3(1e30f,1e30f,1e30f);
    PathState path;

    path = PathTracer::EmptyPathInitialize(pixelPos, g_Const.ptConsts.camera.PixelConeSpreadAngle);
    PathTracer::SetupPathPrimaryRay(path, Bridge::computeCameraRay(pixelPos));  // note: all realtime mode subSamples currently share same camera ray at subSampleIndex == 0 (otherwise denoising guidance buffers would be noisy)

    PathTracer::WorkingContext workingContext = GetWorkingContext(path);
    // clear, initialize any global backing memory, etc
    PathTracer::StartPixel(path, workingContext);

    // TODO: move this to PathTracer once DX SER API comes out of Preview and works on Vulkan
#if PATH_TRACER_MODE==PATH_TRACER_MODE_FILL_STABLE_PLANES    // we're continuing from base stable plane (index 0) here to avoid unnecessary path tracing
    firstHitFromBasePlane(path, 0, workingContext);
    //lastOrigin = path.origin;
#endif

    // Main path tracing loop
    while (path.isActive())
        nextHit(path, workingContext, false);
    
#if 0
    ValidateNaNs(path, workingContext);
#endif
       
    // store radiance and any other required data
    PathTracer::CommitPixel( path, workingContext );
        
#if PATH_TRACER_MODE==PATH_TRACER_MODE_BUILD_STABLE_PLANES && ENABLE_DEBUG_DELTA_TREE_VIZUALISATION
    DeltaTreeVizExplorePixel(workingContext, 0);
    return;
#endif

    //if (g_MiniConst.params.x == 0)
    //{
    //        uint address = StablePlanesContext::ComputeDominantAddress(pixelPos, u_StablePlanesHeader, u_StablePlanesBuffer, u_StableRadiance, g_Const.ptConsts);
    //
    //        float4 radiance     = Fp16ToFp32(u_StablePlanesBuffer[address].PackedNoisyRadianceAndSpecAvg);
    //        float specHitDist   = u_StablePlanesBuffer[address].NoisyRadianceSpecHitDist;
    //        u_StablePlanesBuffer[address].PackedNoisyRadianceAndSpecAvg = Fp32ToFp16(radiance);
    //        u_StablePlanesBuffer[address].NoisyRadianceSpecHitDist = specHitDist;
    //}


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
        nextHit(path, workingContext, true);

        PathPayload statePacked; int loop = 0;
        while ( workingContext.Debug.DeltaSearchStackPop(statePacked) )
        {
            loop++; 
            PathState deltaPathState = PathPayload::unpack( statePacked, PACKED_HIT_INFO_ZERO );
            nextHit(deltaPathState, workingContext, true);
        }
        for (int i = 0; i < cStablePlaneCount; i++)
            workingContext.Debug.DeltaTreeStoreStablePlaneID( i, workingContext.StablePlanes.GetBranchIDCenter(i) );
        workingContext.Debug.DeltaTreeStoreDominantStablePlaneIndex( workingContext.StablePlanes.LoadDominantIndexCenter() );
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
    PathTracer::WorkingContext workingContext = GetWorkingContext(path);
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
    PathTracer::HandleMiss(path, WorldRayOrigin(), WorldRayDirection(), RayTCurrent(), GetWorkingContext(path));
    payload = PathPayload::pack(path);
#endif
}

[shader("anyhit")]
void AnyHit(inout PathPayload payload, in BuiltInTriangleIntersectionAttributes attrib/* : SV_IntersectionAttributes*/)
{
    if (!Bridge::AlphaTest(InstanceID(), InstanceIndex(), GeometryIndex(), PrimitiveIndex(), attrib.barycentrics/*, GetWorkingContext( path ).debug*/ ))
        IgnoreHit();
}
