/*
* Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

// Local Cubemap Ray Tracing Pass
// Traces rays from camera position to generate a local environment cubemap
// Uses the same hit shaders as the GBuffer pass, then performs simple IBL lighting

#include "PathTracer/Config.h" // must always be included first

#include "PathTracer/PathTracerTypes.hlsli"

#include "Bindings/ShaderResourceBindings.hlsli"
#include "Bindings/GBufferBindings.hlsli"
#include "Bindings/LightingBindings.hlsli"
#include "Bindings/SamplerBindings.hlsli"

#include "PathTracerBridgeDonut.hlsli"
#include "PathTracer/PathTracer.hlsli"
#include "IntroCommon.hlsli"

// Local cubemap output - bound via the main binding set
RWTexture2DArray<float4> u_LocalCubemap : register(u10);

// Convert cubemap face pixel to world direction
float3 CubemapPixelToDirection(uint2 pixel, uint face, uint cubeSize)
{
    float2 uv = (float2(pixel) + 0.5) / float(cubeSize);
    float2 ndc = uv * 2.0 - 1.0;
    
    float3 dir;
    switch (face)
    {
        case 0: dir = float3(1.0, -ndc.y, -ndc.x); break;  // +X
        case 1: dir = float3(-1.0, -ndc.y, ndc.x); break;  // -X
        case 2: dir = float3(ndc.x, 1.0, ndc.y); break;    // +Y
        case 3: dir = float3(ndc.x, -1.0, -ndc.y); break;  // -Y
        case 4: dir = float3(ndc.x, -ndc.y, 1.0); break;   // +Z
        case 5: dir = float3(-ndc.x, -ndc.y, -1.0); break; // -Z
        default: dir = float3(0, 0, 1); break;
    }
    
    return normalize(dir);
}

// Simple IBL lighting using the global environment map
// Takes extracted values (not PayloadLite) so DXC can see the payload reads in the caller
float3 ComputeSimpleLighting(float3 baseColor, float3 normal, float roughness, float metal, float ao, float3 viewDir)
{
    float3 R = reflect(-viewDir, normal);
    
    // Get environment map mip count (assume 8-10 mips for a typical 256-4096 cubemap)
    float maxMipLevel = 8.0;
    
    // Diffuse IBL - sample at high mip for blurred/irradiance-like result
    float diffuseMip = maxMipLevel * 0.8;
    //float3 diffuseIBL = t_EnvironmentMap.SampleLevel(s_EnvironmentMapSampler, normal, diffuseMip).rgb;
    float3 diffuseIBL = t_DiffuseIrradianceCube.SampleLevel(s_EnvironmentMapSampler, normal, 0).rgb;
    
    // Specular IBL - sample at roughness-based mip
    roughness = max(0.2, roughness); // Mitigate fireflies with high contrast cubemaps.
    float specularMip = roughness * maxMipLevel;
    float3 specularIBL = t_EnvironmentMap.SampleLevel(s_EnvironmentMapSampler, R, specularMip).rgb;
    
    // Preintegrated brdf (Split sum approximation)
    float NdotV = saturate(dot(normal, viewDir));
    float2 brdfLUT = t_BRDFLUT.SampleLevel(s_EnvironmentMapSampler, float2(NdotV, roughness), 0).rg;
    float3 F0 = lerp(float3(0.04, 0.04, 0.04), baseColor, metal);
    float3 specular = specularIBL * (F0 * brdfLUT.x + brdfLUT.y);
    
    // Combine diffuse and specular
    // For metals, diffuse is essentially zero (energy goes to specular)
    float3 diffuse = baseColor * (1.0 - metal) * diffuseIBL;
    
    // Combine with ambient occlusion
    float3 radiance = (diffuse + specular) * ao;
    
    return radiance;
}

[shader("raygeneration")]
void RAYGEN_ENTRY()
{
    // DispatchRaysIndex: x = pixel.x, y = pixel.y, z encodes face info
    // We dispatch with dimensions (CubemapSize, CubemapSize, 2) for 2 faces per frame
    uint2 pixel = DispatchRaysIndex().xy;
    uint faceIndex = DispatchRaysIndex().z + g_MiniConst.params.x;  // 0 or 1
    
    // Get cubemap size from dispatch dimensions
    uint cubeSize = DispatchRaysDimensions().x;
        
    // Skip if face index is invalid (>= 6)
    if (faceIndex >= 6)
        return;
    
    // Compute ray direction for this cubemap texel
    float3 rayDir = CubemapPixelToDirection(pixel, faceIndex, cubeSize);
        
    // Set up ray
    RayDesc rayDesc;
    rayDesc.Origin = g_Const.ptConsts.camera.PosW.xyz;;
    rayDesc.Direction = rayDir;
    rayDesc.TMin = 0.00;
    rayDesc.TMax = 10000;
    
    // Trace ray using the same payload as GBuffer pass
    PayloadLite payload;
    TraceRay(SceneBVH, RAY_FLAG_NONE, 0xff, 0, 1, 0, rayDesc, payload);
    
    float3 radiance;
    
    if (payload.hitDistance >= 0)
    {
        // Hit geometry - extract payload fields explicitly so DXC sees the reads
        // (DXC's payload qualifier analysis doesn't see through function calls)
        float3 baseColor = payload.baseColor;
        float3 normal = payload.normal;
        float roughness = payload.roughness;
        float metal = payload.metal;
        float ao = payload.ambientOcclusion;
        
        // Dummy reads to satisfy payload qualifier requirements (these fields are written by hit shaders)
        float3 dummy = payload.worldPos + payload.motionVector;
        uint dummyId = payload.shaderId;
        (void)dummy; (void)dummyId;
        
        // Compute simple IBL lighting
        radiance = ComputeSimpleLighting(baseColor, normal, roughness, metal, ao, -rayDir);
    }
    else
    {
        // Miss - sample global environment map directly
        radiance = t_EnvironmentMap.SampleLevel(s_EnvironmentMapSampler, rayDir, 0).rgb;
    }
    
    // Clamp to avoid fireflies (FP16 max)
    radiance = min(radiance, float3(65504, 65504, 65504));
    
    // Write to cubemap
    u_LocalCubemap[uint3(pixel, faceIndex)] = float4(radiance, 1.0);
}

[shader("miss")]
void MISS_ENTRY(inout PayloadLite payload : SV_RayPayload)
{
    payload.hitDistance = -1;
}
