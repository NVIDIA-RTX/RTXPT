/*
* Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#define NON_PATH_TRACING_PASS 1

#include "Shaders/Bindings/ShaderResourceBindings.hlsli"

#include "Shaders/PathTracerBridgeDonut.hlsli"
#include "Shaders/PathTracer/PathTracer.hlsli"
#include "RTXDI/SurfaceData.hlsli"

[numthreads(NUM_COMPUTE_THREADS_PER_DIM, NUM_COMPUTE_THREADS_PER_DIM, 1)]
void main( uint2 dispatchThreadID : SV_DispatchThreadID )
{
    const uint2 pixelPos = dispatchThreadID.xy;
    if( any(pixelPos >= uint2(g_Const.ptConsts.imageWidth, g_Const.ptConsts.imageHeight) ) )
        return;

    // Load the primary hit from the V-buffer (stable planes are now what used to be v-buffer).
    StablePlanesContext stablePlanes = StablePlanesContext::make(u_StablePlanesHeader, u_StablePlanesBuffer, u_StableRadiance, g_Const.ptConsts);

    uint dominantStablePlaneIndex = stablePlanes.LoadDominantIndex(pixelPos);
    uint stableBranchID = stablePlanes.GetBranchID(pixelPos, dominantStablePlaneIndex);
	StablePlane sp = stablePlanes.LoadStablePlane(pixelPos, dominantStablePlaneIndex);
    PackedHitInfo packedHitInfo; float3 rayDir; uint vertexIndex; float sceneLength; float3 pathThp; float3 motionVectors;
    StablePlanesContext::UnpackStablePlane(sp, vertexIndex, packedHitInfo, rayDir, sceneLength, pathThp, motionVectors);

    u_MotionVectors[pixelPos]   = float4(motionVectors, 0);

    // Useful for debugging
    // DebugContext debug; debug.Init( pixelPos, 0, g_Const.debug, u_FeedbackBuffer, u_DebugLinesBuffer, u_DebugDeltaPathTree, u_DeltaPathSearchStack, u_DebugVizOutput );

    const Ray cameraRay = Bridge::computeCameraRay( pixelPos );
    const HitInfo hit = HitInfo(packedHitInfo);
    bool hitSurface = hit.isValid();// && hit.getType() == HitType::Triangle;

#if 0  // for testing correctness: compute first hit surface motion vector
    {
        float3 virtualWorldPos = cameraRay.origin + cameraRay.dir * stablePlanes.LoadFirstHitRayLength(pixelPos);
        u_MotionVectors[pixelPos].xyz = Bridge::computeMotionVector(virtualWorldPos, virtualWorldPos);
    }
#endif

    // Prepare per-pixel surface data for denoising, RTXDI and etc.
    if (hitSurface)
    {
        float3 virtualWorldPos = cameraRay.origin + cameraRay.dir * sceneLength;
        float4 clipPos = mul(float4(/*bridgedData.shadingData.posW*/virtualWorldPos, 1), g_Const.view.matWorldToClip);
        u_Depth[pixelPos] = clipPos.z / clipPos.w;
        u_Throughput[pixelPos] = Pack_R11G11B10_FLOAT(saturate(pathThp));
    }
    else
    {
        u_Depth[pixelPos] = 0;
        u_Throughput[pixelPos] = 0;
    }

    if (g_Const.ptConsts.useReSTIRDI || g_Const.ptConsts.useReSTIRGI)
    {
        // compute address for current output - it ping pongs based on frameIndex!
        const uint idxPingPong = (g_Const.ptConsts.frameIndex % 2) == (uint)0;
        const uint idx = GenericTSPixelToAddress(pixelPos, idxPingPong, g_Const.ptConsts.genericTSLineStride, g_Const.ptConsts.genericTSPlaneStride);

        // TODO: make a variant that already takes StablePlanes inputs so they're not re-loaded again
        u_SurfaceData[idx] = ExtractPackedGbufferSurfaceData(pixelPos, sp, dominantStablePlaneIndex, stableBranchID);
    }

    if (g_Const.debug.debugViewType==(int)DebugViewType::VBufferMotionVectors)
        u_DebugVizOutput[pixelPos] = float4( 0.5.xxx + u_MotionVectors[pixelPos].xyz * float3(0.2, 0.2, 10), 1);
    else if (g_Const.debug.debugViewType==(int)DebugViewType::VBufferDepth)
        u_DebugVizOutput[pixelPos] = float4( u_Depth[pixelPos].xxx * 100.0, 1 );

}