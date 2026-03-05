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

#include "PathTracer/PathTracerTypes.hlsli"

#include "Bindings/ShaderResourceBindings.hlsli"
#include "Bindings/GBufferBindings.hlsli"

#include "PathTracerBridgeDonut.hlsli"
#include "PathTracer/PathTracer.hlsli"
#include "Bindings/GBufferBindings.hlsli"
#include "IntroCommon.hlsli"

[shader("raygeneration")]
void RAYGEN_ENTRY()
{
    uint2 pixelPos = DispatchRaysIndex().xy;

    // Trace primary visibility rays, starting at the camera.
    Ray cameraRay = Bridge::computeCameraRay(pixelPos);
    RayDesc rayDesc;
    rayDesc.Origin = cameraRay.origin;
    rayDesc.Direction = cameraRay.dir;
    rayDesc.TMin = 0;
    rayDesc.TMax = 10000;
    
    PayloadLite payload; // Intentionally uninitialized to reduce state across the tracing boundary.
    TraceRay(SceneBVH, RAY_FLAG_NONE, 0xff, 0, 1, 0, rayDesc, payload);
    
    // Process Bounce
    PackedGBufferSurface gBuffer; // Intentionally uninitialized to save registers
    gBuffer.SetBaseColor(0);
    
    // Dummy. Need to read this so we can share hit shaders with introPathTracer without payload qualifiers complaining.
    // This is suboptimal, but this populateGBuuffer pass is just a stand in for your regular G-Buffer raster.
    gBuffer.SetBaseColor(payload.worldPos * payload.ambientOcclusion);
    
    if (payload.hitDistance >= 0)
    {
        // Read material info
        gBuffer.SetBaseColor(payload.baseColor);
        gBuffer.SetSpecNormal(payload.normal);
        gBuffer.SetRoughnessMetal(payload.roughness, payload.metal);
        gBuffer.packedMaterialInfo = payload.shaderId;
    }
    else
    {
        // Dummy data fill.
        // Maybe we can even skip this section and write to the UAVs directly inside the if.
        gBuffer.SetBaseColor(0);
        gBuffer.SetSpecNormal(float3(0, 0, 1));
        gBuffer.SetRoughnessMetal(0, 0);
        gBuffer.packedMaterialInfo = ShaderIdInvalid;
    }

    gBuffer.SetMaterialInfo(payload.shaderId, payload.ambientOcclusion);
    gBuffer.StoreIntoRenderTargets(pixelPos);
    
    // Convert world position to NDC depth for SSR and other passes
    if (payload.hitDistance >= 0)
    {
        float4 clipPos = mul(float4(payload.worldPos, 1), g_Const.view.matWorldToClip);
        u_Depth[pixelPos] = clipPos.z / clipPos.w;
    }
    else
    {
        u_Depth[pixelPos] = 0;
    }
    
    u_MotionVectors[pixelPos] = float4(payload.motionVector, 0);
}

[shader("miss")]
void MISS_ENTRY(inout PayloadLite payload : SV_RayPayload)
{
    payload.hitDistance = -1;
}
