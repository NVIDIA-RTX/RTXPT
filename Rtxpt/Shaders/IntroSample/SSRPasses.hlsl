/*
* Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

// Screen-Space Reflections Passes
// - DepthHierarchyCS: Generate Hi-Z depth pyramid
// - SSRCS: Hierarchical ray marching SSR
// - SSRBlurCS: Roughness-aware blur mip chain

#ifndef __SSR_PASSES_HLSL__
#define __SSR_PASSES_HLSL__

#include <donut/shaders/utils.hlsli>

#define SSR_THREADS 8

// SSR Push Constants (96 bytes) - self-contained, no constant buffer needed
// Contains packed 3x3 view matrix, projection params, screen size, and SSR params
struct SSRPushConstants
{
    // View matrix 3x3 (row major) packed with projection diagonal
    float4 viewRow0_P00;      // viewRow0.xyz, proj[0][0]
    float4 viewRow1_P11;      // viewRow1.xyz, proj[1][1]
    float4 viewRow2_P22;      // viewRow2.xyz, proj[2][2]
    
    // Projection and reconstruction params
    float4 projParams;        // rcpP00, rcpP11, P32, nearZ
    
    // Screen size and additional params
    float4 screenAndParams;   // screenWidth, screenHeight, farZ, maxRayDistance
    
    // SSR-specific params (maxSteps and maxMipLevel stored as float bits)
    float4 ssrParams;         // maxSteps (asuint), thickness, jitter, maxMipLevel (asuint)
};

struct SSRBlurConstants
{
    uint2   SrcSize;
    uint2   DstSize;
    uint    SrcMipLevel;
    uint    DstMipLevel;
    uint    Padding0;
    uint    Padding1;
};

// Mark as non-path-tracing pass to avoid ray payload qualifier issues in compute shaders
#define NON_PATH_TRACING_PASS 1

// Use global bindings for shared resources
#include "Bindings/ShaderResourceBindings.hlsli"
#include "Bindings/GBufferBindings.hlsli"
#include "Bindings/SamplerBindings.hlsli"

//==================================================================================================
// Single-Pass Depth Hierarchy Generation (using LDS reduction)
// Uses dedicated binding set to avoid duplicate barriers with global SRV
// SrcSize is passed via push constants (g_MiniConst.x, g_MiniConst.y)
//==================================================================================================

// Local bindings for depth hierarchy pass (dedicated binding set)
// Note: g_MiniConst comes from global bindings include above
SamplerState            s_SSRDH_Max         : register(s0);  // For sampling level 0
Texture2D<float>        t_SrcDepth          : register(t0);  // Source depth
RWTexture2D<float>      u_HierMip1          : register(u0);  // Output mip 1
RWTexture2D<float>      u_HierMip2          : register(u1);  // Output mip 2
RWTexture2D<float>      u_HierMip3          : register(u2);  // Output mip 3
RWTexture2D<float>      u_HierMip4          : register(u3);  // Output mip 4
RWTexture2D<float>      u_HierMip5          : register(u4);  // Output mip 5
RWTexture2D<float>      u_HierMip6          : register(u5);  // Output mip 6
RWTexture2D<float>      u_HierMip7          : register(u6);  // Output mip 7
RWTexture2D<float>      u_HierMip8          : register(u7);  // Output mip 8
RWTexture2D<float>      u_HierMip9          : register(u8);  // Output mip 9
RWTexture2D<float>      u_HierMip10         : register(u9);  // Output mip 10
RWTexture2D<float>      u_HierMip11         : register(u10);  // Output mip 11
RWTexture2D<float>      u_HierMip12         : register(u11);  // Output mip 12

// Groupshared memory for the tile reduction
// We use 16x16 because that's the result after the first 2x2 reduction
groupshared float gs_DepthTile[64];

// Helper to reduce 4 depth values (max for reverse-Z Hi-Z)
float ReduceDepth4(float d00, float d10, float d01, float d11)
{
    return max(max(d00, d10), max(d01, d11));
}

// Decode 8-bit Morton code to 4-bit x,y (maps thread 0-255 to 16x16 tile)
uint2 MortonDecode8(uint index)
{
    // Extract even bits for x
    uint x = index & 0x55;      // 0b01010101 - bits 0,2,4,6
    x = (x | (x >> 1)) & 0x33;  // 0b00110011
    x = (x | (x >> 2)) & 0x0F;  // 0b00001111

    // Extract odd bits for y (shift first, then same pattern)
    uint y = (index >> 1) & 0x55;
    y = (y | (y >> 1)) & 0x33;
    y = (y | (y >> 2)) & 0x0F;

    return uint2(x, y);
}

[numthreads(256, 1, 1)]
void DepthHierarchyCS(
    uint3 groupID : SV_GroupID,
    uint groupIndex : SV_GroupIndex)
{
    // ====== First reduction: Max Sampler 64x64 -> 32x32 ======
    uint2 srcPos = MortonDecode8(groupIndex) * 4; // Each thread first works on its local 4x4 tile

    uint2 srcSize = uint2(g_MiniConst.params.x, g_MiniConst.params.y);

    // Calculate base position for this 64x64 tile in source image
    uint2 tileBase = groupID.xy * 64;
    float4 d4x4;

    uint2 mip1Size = srcSize / 2;
    // The first reduction happens implicitly in the sampler
    uint2 quadCenter = tileBase + srcPos + uint2(1, 1);
    uint2 dstPos = quadCenter / 2;
    float2 uv = float2(quadCenter) / srcSize;
    d4x4.x = t_SrcDepth.SampleLevel(s_SSRDH_Max, uv, 0);
    if (dstPos.x < mip1Size.x && dstPos.y < mip1Size.y)
        u_HierMip1[dstPos] = d4x4.x;

    quadCenter = tileBase + srcPos + uint2(3, 1);
    dstPos = quadCenter / 2;
    uv = float2(quadCenter) / srcSize;
    d4x4.y = t_SrcDepth.SampleLevel(s_SSRDH_Max, uv, 0);
    if (dstPos.x < mip1Size.x && dstPos.y < mip1Size.y)
        u_HierMip1[dstPos] = d4x4.y;

    quadCenter = tileBase + srcPos + uint2(1, 3);
    dstPos = quadCenter / 2;
    uv = float2(quadCenter) / srcSize;
    d4x4.z = t_SrcDepth.SampleLevel(s_SSRDH_Max, uv, 0);
    if (dstPos.x < mip1Size.x && dstPos.y < mip1Size.y)
        u_HierMip1[dstPos] = d4x4.z;

    quadCenter = tileBase + srcPos + uint2(3, 3);
    dstPos = quadCenter / 2;
    uv = float2(quadCenter) / srcSize;
    d4x4.w = t_SrcDepth.SampleLevel(s_SSRDH_Max, uv, 0);
    if (dstPos.x < mip1Size.x && dstPos.y < mip1Size.y)
        u_HierMip1[dstPos] = d4x4.w;

    // ====== Second reduction: Thread ALU 32x32 -> 16x16 ======
    float reduced = ReduceDepth4(d4x4.x, d4x4.y, d4x4.z, d4x4.w);
    dstPos = (tileBase + srcPos) / 4;
    uint2 mip2Size = mip1Size / 2;
    if(all(dstPos < mip2Size))
        u_HierMip2[dstPos] = reduced;

    // ====== Third reduction: Quad Shuffle 16x16 -> 8x8 ======
    reduced = max(QuadReadAcrossX(reduced), reduced);
    reduced = max(QuadReadAcrossY(reduced), reduced); // Already includes the x reduction
    dstPos = (tileBase + srcPos) / 8;
    uint2 mip3Size = mip2Size / 2;
    if (all(dstPos < mip3Size) && (groupIndex % 4 == 0)) // Only 1 in 4 threads writes
        u_HierMip3[dstPos] = reduced;

    // ====== 4th reduction: LDS within the wave 8x8 -> 4x4 ======
    if (groupIndex % 4 == 0)
    {
        gs_DepthTile[groupIndex / 4] = reduced;
    }

    // No need for a sync group because we're reading and writing within the wave
    uint2 mip4Size = mip3Size / 2;
    if (groupIndex % 16 == 0)
    {
        uint baseRead = groupIndex / 4;
        d4x4.x = gs_DepthTile[baseRead + 0];
        d4x4.y = gs_DepthTile[baseRead + 1];
        d4x4.z = gs_DepthTile[baseRead + 2];
        d4x4.w = gs_DepthTile[baseRead + 3];

        reduced = ReduceDepth4(d4x4.x, d4x4.y, d4x4.z, d4x4.w);
        dstPos = (tileBase + srcPos) / 16;

        if (all(dstPos < mip4Size))
        {
            u_HierMip4[dstPos] = reduced;
        }
    }

    // ====== 5th reduction: Cross-wave LDS 4x4 -> 2x2 ======
    GroupMemoryBarrierWithGroupSync(); // Barrier before to ensure we don't overwrite anyone's LDS before everyone has read it.

    if (any(dstPos >= mip4Size))
        reduced = 0;

    if((groupIndex % 16 == 0))
    {
        reduced = WaveActiveMax(reduced);
        if(WaveIsFirstLane())
            gs_DepthTile[groupIndex / 32] = reduced; // This reduces 4x4 to 2x4
    }

    GroupMemoryBarrierWithGroupSync(); // Barrier after write

    // From now on, we only need 1 thread per group
    if (groupIndex == 0)
    {
        // Reading 2 shared values per thread reduces 2x4 to 2x2
        d4x4.x = max(gs_DepthTile[0], gs_DepthTile[1]);
        d4x4.y = max(gs_DepthTile[2], gs_DepthTile[3]);
        d4x4.z = max(gs_DepthTile[4], gs_DepthTile[5]);
        d4x4.w = max(gs_DepthTile[6], gs_DepthTile[7]);

        uint2 mip5Size = mip4Size / 2;
        dstPos = groupID.xy * 2;
        if (dstPos.x + 0 < mip5Size.x && dstPos.y + 0 < mip5Size.y)
            u_HierMip5[dstPos + uint2(0, 0)] = d4x4.x;
        if (dstPos.x + 1 < mip5Size.x && dstPos.y + 0 < mip5Size.y)
            u_HierMip5[dstPos + uint2(1, 0)] = d4x4.y;
        if (dstPos.x + 0 < mip5Size.x && dstPos.y + 1 < mip5Size.y)
            u_HierMip5[dstPos + uint2(0, 1)] = d4x4.z;
        if (dstPos.x + 1 < mip5Size.x && dstPos.y + 1 < mip5Size.y)
            u_HierMip5[dstPos + uint2(1, 1)] = d4x4.w;

        // ====== Sixth reduction: ALU within the thread 2x2 -> 1x1 ======
        reduced = ReduceDepth4(d4x4.x, d4x4.y, d4x4.z, d4x4.w);
        uint2 mip6Size = mip5Size / 2;
        dstPos = groupID.xy;
        if (dstPos.x < mip6Size.x && dstPos.y < mip6Size.y)
        {
            u_HierMip6[dstPos] = reduced;
        }
    }

    //
    // ====== Full sync barrier across the whole dispatch ======
    // ====== 7th reduction: Hardware sampling 64x64 -> 32x32 ======
    AllMemoryBarrierWithGroupSync();

    // Only one group needs to keep working
    if (groupID.x > 0 || groupID.y > 0)
        return;

    uint2 mip6Size = mip4Size / 4;
    uint2 mip7Size = mip6Size / 2;
    
    quadCenter = srcPos + uint2(1, 1);
    uv = float2(quadCenter) / (srcSize/64);
    d4x4.x = (quadCenter.x < mip6Size.x && quadCenter.y < mip6Size.y) ? t_SrcDepth.SampleLevel(s_SSRDH_Max, uv, 6) : 0;
    dstPos = quadCenter / 2;
    if (dstPos.x < mip7Size.x && dstPos.y < mip7Size.y)
        u_HierMip7[dstPos] = d4x4.x;

    quadCenter = srcPos + uint2(3, 1);
    uv = float2(quadCenter) / (srcSize / 64);
    d4x4.y = (quadCenter.x < mip6Size.x && quadCenter.y < mip6Size.y) ? t_SrcDepth.SampleLevel(s_SSRDH_Max, uv, 6) : 0;
    dstPos = quadCenter / 2;
    if (dstPos.x < mip7Size.x && dstPos.y < mip7Size.y)
        u_HierMip7[dstPos] = d4x4.y;

    quadCenter = srcPos + uint2(1, 3);
    uv = float2(quadCenter) / (srcSize / 64);
    d4x4.z = (quadCenter.x < mip6Size.x && quadCenter.y < mip6Size.y) ? t_SrcDepth.SampleLevel(s_SSRDH_Max, uv, 6) : 0;
    dstPos = quadCenter / 2;
    if (dstPos.x < mip7Size.x && dstPos.y < mip7Size.y)
        u_HierMip7[dstPos] = d4x4.z;

    quadCenter = srcPos + uint2(3, 3);
    uv = float2(quadCenter) / (srcSize / 64);
    d4x4.w = (quadCenter.x < mip6Size.x && quadCenter.y < mip6Size.y) ? t_SrcDepth.SampleLevel(s_SSRDH_Max, uv, 6) : 0;
    dstPos = quadCenter / 2;
    if (dstPos.x < mip7Size.x && dstPos.y < mip7Size.y)
        u_HierMip7[dstPos] = d4x4.w;

    // ====== 8th reduction: ALU within the thread 32x32 -> 16x16 ======
    reduced = ReduceDepth4(d4x4.x, d4x4.y, d4x4.z, d4x4.w);
    dstPos = srcPos / 4;
    uint2 mip8Size = mip7Size / 2;
    if (dstPos.x < mip8Size.x && dstPos.y < mip8Size.y)
        u_HierMip8[dstPos] = reduced;

    // ====== 9th reduction: Quad Shuffle 16x16 -> 8x8 ======
    reduced = max(QuadReadAcrossX(reduced), reduced);
    reduced = max(QuadReadAcrossY(reduced), reduced); // Already includes the x reduction
    dstPos = srcPos / 8;
    uint2 mip9Size = mip8Size / 2;
    if (dstPos.x < mip9Size.x && dstPos.y < mip9Size.y)
        u_HierMip9[dstPos] = reduced;

    // ====== 10th reduction: LDS within the wave 8x8 -> 4x4 ======
    if (groupIndex % 4 == 0)
    {
        gs_DepthTile[groupIndex / 4] = reduced;
    }

    // No need for a sync group because we're reading and writing within the wave
    uint2 mip10Size = mip9Size / 2;
    if (groupIndex % 16 == 0)
    {
        uint baseRead = groupIndex / 4;
        d4x4.x = gs_DepthTile[baseRead + 0];
        d4x4.y = gs_DepthTile[baseRead + 1];
        d4x4.z = gs_DepthTile[baseRead + 2];
        d4x4.w = gs_DepthTile[baseRead + 3];

        reduced = ReduceDepth4(d4x4.x, d4x4.y, d4x4.z, d4x4.w);
        dstPos = (tileBase + srcPos) / 16;

        if (all(dstPos < mip10Size))
        {
            u_HierMip10[dstPos] = reduced;
        }
    }

    // ====== 11th reduction: Cross-wave LDS 4x4 -> 2x2 ======
    GroupMemoryBarrierWithGroupSync(); // Barrier before to ensure we don't overwrite anyone's LDS before everyone has read it.

    if (any(dstPos >= mip10Size))
        reduced = 0;

    if ((groupIndex % 16 == 0))
    {
        reduced = WaveActiveMax(reduced);
        if (WaveIsFirstLane())
            gs_DepthTile[groupIndex / 32] = reduced; // This reduces 4x4 to 2x4
    }

    GroupMemoryBarrierWithGroupSync();

    // From now on, we only need 1 thread per group
    if (groupIndex == 0)
    {
        // Reading 2 shared values per thread reduces 2x4 to 2x2
        d4x4.x = max(gs_DepthTile[0], gs_DepthTile[1]);
        d4x4.y = max(gs_DepthTile[2], gs_DepthTile[3]);
        d4x4.z = max(gs_DepthTile[4], gs_DepthTile[5]);
        d4x4.w = max(gs_DepthTile[6], gs_DepthTile[7]);

        uint2 mip11Size = mip10Size / 2;
        dstPos = groupID.xy * 2;
        if (dstPos.x + 0 < mip11Size.x && dstPos.y + 0 < mip11Size.y)
            u_HierMip11[dstPos + uint2(0, 0)] = d4x4.x;
        if (dstPos.x + 1 < mip11Size.x && dstPos.y + 0 < mip11Size.y)
            u_HierMip11[dstPos + uint2(1, 0)] = d4x4.y;
        if (dstPos.x + 0 < mip11Size.x && dstPos.y + 1 < mip11Size.y)
            u_HierMip11[dstPos + uint2(0, 1)] = d4x4.z;
        if (dstPos.x + 1 < mip11Size.x && dstPos.y + 1 < mip11Size.y)
            u_HierMip11[dstPos + uint2(1, 1)] = d4x4.w;

        // ====== 12th reduction: ALU within the thread 2x2 -> 1x1 ======
        // reduced = ReduceDepth4(d4x4.x, d4x4.y, d4x4.z, d4x4.w);
        // uint2 mip6Size = mip5Size / 2;
        // dstPos = groupID.xy;
        // if (dstPos.x < mip6Size.x && dstPos.y < mip6Size.y)
        // {
        //     u_HierMip6[dstPos] = reduced;
        // }
    }
}

//==================================================================================================
// Hierarchical Screen-Space Reflections
// Uses dedicated binding layout with custom push constants - no constant buffers
//==================================================================================================

// SSR dedicated bindings (self-contained, no global dependencies)
// Push constants are configured via binding layout on CPU side
ConstantBuffer<SSRPushConstants> g_SSR : register(b0);

Texture2D<float>    t_SSR_DepthHierarchy    : register(t0);  // Hi-Z mip chain
Texture2D<float>    t_SSR_Depth             : register(t1);  // Full-res depth
Texture2D<uint>     t_SSR_Normal            : register(t2);  // Packed normal (R32_UINT)
Texture2D<float2>   t_SSR_RoughnessMetal    : register(t3);  // Roughness/metallic
Texture2D<float4>   t_SSR_BaseColor        : register(t4);  // Scene color for sampling
TextureCube<float4> t_SSR_CubemapGGX        : register(t5);  // GGX-filtered local cubemap
TextureCube<float4> t_SSR_IrradianceCube    : register(t6);  // Diffuse irradiance cubemap
Texture2D<float2>   t_SSR_BRDFLUT           : register(t7);  // Split-sum BRDF LUT
RWTexture2D<float4> u_SSR_Result            : register(u0);  // Output
SamplerState        s_SSR_LinearClamp       : register(s0);  // For depth hierarchy sampling
SamplerState        s_SSR_NearestClamp      : register(s1);  // For depth hierarchy sampling

// Helper functions to extract push constant parameters
float3x3 SSR_GetViewMatrix3x3()
{
    return float3x3(
        g_SSR.viewRow0_P00.xyz,
        g_SSR.viewRow1_P11.xyz,
        g_SSR.viewRow2_P22.xyz
    );
}

float SSR_GetP00() { return g_SSR.viewRow0_P00.w; }
float SSR_GetP11() { return g_SSR.viewRow1_P11.w; }
float SSR_GetP22() { return g_SSR.viewRow2_P22.w; }
float SSR_GetP32() { return g_SSR.projParams.z; }

float SSR_GetRcpP00() { return g_SSR.projParams.x; }
float SSR_GetRcpP11() { return g_SSR.projParams.y; }
float SSR_GetNearZ()  { return g_SSR.projParams.z; }
float SSR_GetFarZ()   { return g_SSR.screenAndParams.z; }

uint2 SSR_GetScreenSize() { return uint2(g_SSR.screenAndParams.xy); }

uint  SSR_GetMaxSteps()       { return asuint(g_SSR.ssrParams.x); }
float SSR_GetThickness()      { return g_SSR.ssrParams.y; }
float SSR_GetJitter()         { return g_SSR.ssrParams.z; }
uint  SSR_GetMaxMipLevel()    { return asuint(g_SSR.ssrParams.w); }

// Analytic view position reconstruction using projection parameters
float3 SSR_GetViewPosition(uint2 pixel, float depth)
{
    float2 screenSize = float2(SSR_GetScreenSize());
    float2 uv = (float2(pixel) + 0.5) / screenSize;
    float2 ndc = uv * 2.0 - 1.0;
    ndc.y = -ndc.y;
    
    // Linearize depth (infinite reverse-Z: depth = nearZ / viewZ)
    float nearZ = SSR_GetNearZ();
    float viewZ = nearZ / depth;
    
    // Reconstruct view position using inverse projection parameters
    return float3(ndc.x * SSR_GetRcpP00() * viewZ, ndc.y * SSR_GetRcpP11() * viewZ, viewZ);
}

// Screen projection using minimal projection params
float2 SSR_ProjectToScreen(float3 viewPos)
{
    // For left-handed perspective (clip.w = viewZ): ndc.xy = (viewPos.xy * Pii) / viewPos.z
    float rcpZ = 1.0 / viewPos.z;
    float2 ndc = float2(viewPos.x * SSR_GetP00() * rcpZ, viewPos.y * SSR_GetP11() * rcpZ);
    ndc.y = -ndc.y;
    return ndc * 0.5 + 0.5;
}

// Get projected depth for ray intersection testing
float SSR_GetProjectedDepth(float3 viewPos)
{
    // For infinite reverse-Z: clip.z = P32 (constant), clip.w = viewPos.z
    // depth = clip.z / clip.w = P32 / viewPos.z
    return SSR_GetP32() / viewPos.z;
}

float stepToEdge(float2 startUV, float2 dir, float curT, uint2 curMipResolution)
{
    float2 curUV = startUV + dir * curT;
    uint2 curPixel = uint2(curUV * curMipResolution);

    uint2 topLeftToEdge = select(dir > 0, 1, 0);
    uint2 nextEdge = topLeftToEdge + curPixel;
    float2 edgeUV = float2(nextEdge) / curMipResolution;
    float2 curToEdge = edgeUV - curUV;

    // Note: Divisions by zero are ok in this context. Div by zero will result in +inf, which will be discarded by the min()
    return curT + min(curToEdge.x / dir.x, curToEdge.y / dir.y); // No need for abs. curToEdge and dir already have the same signs.
}

// Fresnel-Schlick with roughness for IBL (same as in RasterDeferredLighting.hlsl)
float3 SSR_FresnelSchlickRoughness(float cosTheta, float3 F0, float roughness)
{
    return F0 + (max(float3(1.0 - roughness, 1.0 - roughness, 1.0 - roughness), F0) - F0) * pow(saturate(1.0 - cosTheta), 5.0);
}

[numthreads(SSR_THREADS, SSR_THREADS, 1)]
void SSRCS(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint2 pixel = dispatchThreadID.xy;
    uint2 screenSize = SSR_GetScreenSize();
    
    if (pixel.x >= screenSize.x || pixel.y >= screenSize.y)
        return;
        
    // Load depth and check for sky
    float depth = t_SSR_Depth[pixel];
    if (depth <= 0)
    {
        u_SSR_Result[pixel] = 0;
        return;
    }
        
    // Get view-space position
    float3 viewPos = SSR_GetViewPosition(pixel, depth);
    
    // Load and transform normal to view space
    uint packedNormal = t_SSR_Normal[pixel];
    float3 worldNormal = octToNdirUnorm32(packedNormal);
    float3 viewNormal = mul(worldNormal, SSR_GetViewMatrix3x3());
    
    // Calculate reflection direction in view space
    float3 viewDir = normalize(viewPos);
    float3 reflectDir = reflect(viewDir, viewNormal);
    
    // Don't trace rays going towards camera
    // Should probably just blend these out
    if (reflectDir.z < 0)
    {
        u_SSR_Result[pixel] = 0;
        return;
    }

    // Get SSR parameters from push constants
    float thickness = SSR_GetThickness();
    float jitter = SSR_GetJitter();
    int maxMipLevel = min(11, (int)SSR_GetMaxMipLevel());
    
    // Ray march through screen space
    float3 rayStart = viewPos;
    float3 rayEnd = viewPos + reflectDir * 100;
    
    float2 startUV = (pixel + 0.5) / screenSize.xy;
    float2 endUV = SSR_ProjectToScreen(rayEnd);
    float endDepthNDC = SSR_GetProjectedDepth(rayEnd);
    
    float2 rayDir2D = endUV - startUV;
    float rayLength = length(rayDir2D);
    
    if (rayLength < 0.001)
    {
        u_SSR_Result[pixel] = 0;
        return;
    }
    
    rayDir2D /= rayLength;
    
    float t = stepToEdge(startUV, rayDir2D, 0, screenSize) + (0.5 / screenSize.x); // Start tracing at the next pixel
    int mipLevel = 0;

    uint stepCount = 0;
    uint maxSteps = SSR_GetMaxSteps();
    float rayOverScene;
    while (stepCount < maxSteps && mipLevel >= 0)
    {
        uint2 curMipResolution = uint2(screenSize) / (1u << uint(mipLevel));

        // Sample the cell t is currently in (not the next cell)
        float2 curUV = startUV + rayDir2D * t;
        uint2 curCell = uint2(curUV * float2(curMipResolution));
        float2 sampleUV = (float2(curCell) + 0.5) / float2(curMipResolution);
        
        // Check bounds
        if (any(curUV < 0) || any(curUV >= 1))
            break;
        
        // Sample depth hierarchy at current mip
        float sceneDepth = t_SSR_DepthHierarchy.SampleLevel(s_SSR_NearestClamp, sampleUV, mipLevel);
        
        // Evaluate ray depth at the far edge of this cell
        float tEdge = stepToEdge(startUV, rayDir2D, t, curMipResolution) + 0.0001;
        float rayT = tEdge / rayLength;
        float rayDepth = lerp(depth, endDepthNDC, rayT);
        
        // Check for intersection (reverse-Z: ray on top of surface when rayDepth > sceneDepth)
        rayOverScene = rayDepth - sceneDepth;
        
        if (rayOverScene >= 0) // Ray clears this entire cell. Advance past it.
        {
            t = tEdge;
            mipLevel++;
            if (mipLevel > maxMipLevel)
            {
                break;
            }
        }
        else // Ray goes behind something in this cell. Refine.
        {
            rayOverScene /= rayDepth; // For edge normalization
            mipLevel--;
        }
        
        stepCount++;
    }

    float confidence = 0;
    if (mipLevel == -1) // Hit on mip 0
    {
        confidence = saturate(1 + rayOverScene * 50);
        confidence *= saturate(reflectDir.z * 10);
    }
    
    // Sample scene color at hit point (t is inside the hit cell when mipLevel == -1)
    float3 reflectionColor = float3(0, 0, 0);
    if (confidence > 0)
    {
        float2 hitUV = startUV + rayDir2D * t;
        if (all(hitUV >= 0) && all(hitUV < 1))
        {
            uint2 hitPixel = uint2(hitUV * float2(screenSize));

            packedNormal = t_SSR_Normal[hitPixel];
            float3 hitNormal = octToNdirUnorm32(packedNormal);
            float3 viewNormal = mul(hitNormal, SSR_GetViewMatrix3x3());

            // IBL lighting at the hit point using local cubemaps
            float3 hitBaseColor = t_SSR_BaseColor[hitPixel].xyz;
            float2 hitRM = t_SSR_RoughnessMetal[hitPixel];
            float hitRoughness = hitRM.x;
            float hitMetallic = hitRM.y;

            // Convert view-space reflection direction to world space for cubemap sampling
            // View matrix is orthonormal, so transpose = inverse
            float3x3 viewToWorld = transpose(SSR_GetViewMatrix3x3());
            float3 worldViewDir = mul(reflectDir, viewToWorld);  // view-space ray dir -> world space
            float3 hitViewDir = -worldViewDir;                   // direction from hit surface toward camera
            float3 hitReflDir = reflect(worldViewDir, hitNormal); // reflect ray off hit surface

            float hitNdotV = saturate(dot(hitNormal, hitViewDir));

            // F0 for metals/dielectrics
            float3 F0 = lerp(float3(0.04, 0.04, 0.04), hitBaseColor, hitMetallic);

            // Diffuse IBL
            float3 irradiance = t_SSR_IrradianceCube.SampleLevel(s_SSR_LinearClamp, hitNormal, 0).rgb;
            float3 kD = (1.0 - F0) * (1.0 - hitMetallic);
            float3 diffuse = kD * hitBaseColor * irradiance;

            // Specular IBL (split-sum approximation)
            float ggxMip = hitRoughness * 7.0;
            float3 prefiltered = t_SSR_CubemapGGX.SampleLevel(s_SSR_LinearClamp, hitReflDir, ggxMip).rgb;
            float2 brdf = t_SSR_BRDFLUT.SampleLevel(s_SSR_LinearClamp, float2(hitNdotV, hitRoughness), 0).rg;
            float3 specular = prefiltered * (F0 * brdf.x + brdf.y);

            reflectionColor = diffuse + specular;
        }
        else
        {
            confidence = 0;
        }
    }
        
    u_SSR_Result[pixel] = float4(reflectionColor, confidence);
}

//==================================================================================================
// SSR Blur Mip Chain Generation
// Note: This pass still needs a dedicated binding set for per-mip UAV bindings
//==================================================================================================

ConstantBuffer<SSRBlurConstants>    g_SSRBlurConst      : register(b0);
Texture2D<float4>                   t_SSRSrc            : register(t0);
RWTexture2D<float4>                 u_SSRBlurDst        : register(u0);

[numthreads(SSR_THREADS, SSR_THREADS, 1)]
void SSRBlurCS(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint2 dstPixel = dispatchThreadID.xy;
    
    if (dstPixel.x >= g_SSRBlurConst.DstSize.x || dstPixel.y >= g_SSRBlurConst.DstSize.y)
        return;
    
    float2 uv = (float2(dstPixel) + 0.5) / float2(g_SSRBlurConst.DstSize);
    float2 texelSize = 1.0 / float2(g_SSRBlurConst.SrcSize);
    
    // Gaussian blur kernel (simplified 5x5)
    float4 result = float4(0, 0, 0, 0);
    float totalWeight = 0;
    
    // 3x3 blur with weights
    float weights[9] = { 1, 2, 1, 2, 4, 2, 1, 2, 1 };
    int2 offsets[9] = { 
        int2(-1,-1), int2(0,-1), int2(1,-1),
        int2(-1, 0), int2(0, 0), int2(1, 0),
        int2(-1, 1), int2(0, 1), int2(1, 1)
    };
    
    for (int i = 0; i < 9; i++)
    {
        float2 sampleUV = uv + float2(offsets[i]) * texelSize;
        float4 sample = t_SSRSrc.SampleLevel(s_MaterialSampler, sampleUV, g_SSRBlurConst.SrcMipLevel);
        sample.xyz *= sample.w;
        result += sample * weights[i];
        totalWeight += weights[i];
    }
    
    result /= totalWeight;
    if (result.w > 0)
        result.xyz /= result.w;
    
    u_SSRBlurDst[dstPixel] = result;
}

#endif // __SSR_PASSES_HLSL__
