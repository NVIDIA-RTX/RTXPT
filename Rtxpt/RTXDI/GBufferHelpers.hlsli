/*
* Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

RayDesc setupPrimaryRay(uint2 pixelPosition, PlanarViewConstants view)
{
    float2 uv = (float2(pixelPosition)+0.5) * view.viewportSizeInv;
    float4 clipPos = float4(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, (1.0 / 256.0), 1);
    float4 worldPos = mul(clipPos, view.matClipToWorld);
    worldPos.xyz /= worldPos.w;

    RayDesc ray;
    ray.Origin = view.cameraDirectionOrPosition.xyz;
    ray.Direction = normalize(worldPos.xyz - ray.Origin);
    ray.TMin = 0;
    ray.TMax = 1000;
    return ray;
}

float3 getMotionVector(
    PlanarViewConstants view,
    PlanarViewConstants viewPrev,
    InstanceData instance,
    float3 objectSpacePosition,
    float3 prevObjectSpacePosition,
    out float o_clipDepth,
    out float o_viewDepth)
{
    float3 worldSpacePosition = mul(instance.transform, float4(objectSpacePosition, 1.0)).xyz;
    float3 prevWorldSpacePosition = mul(instance.prevTransform, float4(prevObjectSpacePosition, 1.0)).xyz;

    float4 clipPos = mul(float4(worldSpacePosition, 1.0), view.matWorldToClip);
    clipPos.xyz /= clipPos.w;
    float4 prevClipPos = mul(float4(prevWorldSpacePosition, 1.0), viewPrev.matWorldToClip);
    prevClipPos.xyz /= prevClipPos.w;

    o_clipDepth = clipPos.z;
    o_viewDepth = clipPos.w;

    if (clipPos.w <= 0 || prevClipPos.w <= 0)
        return 0;

    float3 motion;
    motion.xy = (prevClipPos.xy - clipPos.xy) * view.clipToWindowScale;
    motion.xy += (view.pixelOffset - viewPrev.pixelOffset);
    motion.z = prevClipPos.w - clipPos.w;
    return motion;
}

float2 getEnvironmentMotionVector(
    PlanarViewConstants view,
    PlanarViewConstants viewPrev,
    float2 windowPos)
{
    float4 clipPos;
    clipPos.xy = windowPos * view.windowToClipScale + view.windowToClipBias;
    clipPos.z = 0;
    clipPos.w = 1;

    float4 worldPos = mul(clipPos, view.matClipToWorld);
    float4 prevClipPos = mul(worldPos, viewPrev.matWorldToClip);

    prevClipPos.xyz /= prevClipPos.w;

    float2 motion = (prevClipPos.xy - clipPos.xy) * view.clipToWindowScale;
    motion += view.pixelOffset - viewPrev.pixelOffset;

    return motion;
}

float3 getPreviousWorldPos(
    PlanarViewConstants viewPrev,
    int2 windowPos,
    float viewDepth,
    float3 motionVector)
{
    float2 prevWindowPos = float2(windowPos.xy) + motionVector.xy + 0.5;
    float prevViewDepth = viewDepth + motionVector.z;


    float2 uv = prevWindowPos * viewPrev.viewportSizeInv;
    float4 clipPos = float4(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, 0.5, 1);
    float4 viewPos = mul(clipPos, viewPrev.matClipToView);
    viewPos.xy /= viewPos.z;
    viewPos.zw = 1.0;
    viewPos.xyz *= prevViewDepth;
    return mul(viewPos, viewPrev.matViewToWorld).xyz;
}

float3 viewDepthToWorldPos(
    PlanarViewConstants view,
    int2 pixelPosition,
    float viewDepth)
{
    float2 uv = (float2(pixelPosition)+0.5) * view.viewportSizeInv;
    float4 clipPos = float4(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, 0.5, 1);
    float4 viewPos = mul(clipPos, view.matClipToView);
    viewPos.xy /= viewPos.z;
    viewPos.zw = 1.0;
    viewPos.xyz *= viewDepth;
    return mul(viewPos, view.matViewToWorld).xyz;
}

/*
// The motion vectors rendered by the G-buffer pass match what is expected by NRD and DLSS.
// In case of dynamic resolution, there is a difference that needs to be corrected...
//
// The rendered motion vectors are computed as:
//     (previousUV - currentUV) * currentViewportSize
//
// The motion vectors necessary for pixel reprojection are:
//     (previousUV * previousViewportSize - currentUV * currentViewportSize)
//
float3 convertMotionVectorToPixelSpace(
    PlanarViewConstants view,
    PlanarViewConstants viewPrev,
    int2 pixelPosition,
    float3 motionVector)
{
    float2 curerntPixelCenter = float2(pixelPosition.xy) + 0.5;
    float2 previousPosition = curerntPixelCenter + motionVector.xy;
    previousPosition *= viewPrev.viewportSize * view.viewportSizeInv;
    motionVector.xy = previousPosition - curerntPixelCenter;
    return motionVector;
}
*/