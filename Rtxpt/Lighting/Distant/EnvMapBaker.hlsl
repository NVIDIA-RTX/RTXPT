/*
* Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#ifndef __ENVMAP_BAKER_HLSL__
#define __ENVMAP_BAKER_HLSL__

#include "SampleProceduralSky.hlsli"

#define EMB_NUM_COMPUTE_THREADS_PER_DIM     8
#define EMB_MAXDIRLIGHTS                    16

struct EMB_DirectionalLight
{
    float4      ColorIntensity;     // Intensity unit should be 'W/sr'
    float3      Direction;          // Light incoming direction (so, i.e.,  .Direction = -sunDirection)
    float       AngularSize;
};

struct EnvMapBakerConstants
{
    EMB_DirectionalLight    DirectionalLights[EMB_MAXDIRLIGHTS];

    ProceduralSkyConstants  ProcSkyConsts;

	float3                  ScaleColor;
    uint                    DirectionalLightCount;

	uint                    CubeDim;
    uint                    CubeDimLowRes;
    uint                    ProcSkyEnabled;
    uint                    BackgroundSourceType;                   // 0 - disabled; 1 - t_SrcEquirectangularEnvMap; 2 - t_SrcCubemapEnvMap
};

#if !defined(__cplusplus)

#include "../../PathTracer/Utils/Math/MathHelpers.hlsli"

ConstantBuffer<EnvMapBakerConstants>    g_Const                     : register(b0);

RWTexture2DArray<float4>                u_EnvMapCubeFacesDst0       : register(u0);
RWTexture2DArray<float4>                u_EnvMapCubeFacesDst1       : register(u1);

RWTexture2DArray<float4>                u_EnvMapCubeFacesDst        : register(u0);
RWTexture2DArray<float4>                u_EnvMapCubeFacesSrc        : register(u1);

Texture2D<float4>                       t_SrcEquirectangularEnvMap  : register(t0);
TextureCube<float4>                     t_SrcCubemapEnvMap          : register(t1);

Texture2DArray<float4>                  t_EnvMapCubeFaces           : register(t0);

TextureCube<float4>                     t_LowResPrePassCube         : register(t2);

Texture2D                               t_ProcSkyTransmittance      : register(t10);
Texture3D                               t_ProcSkyScatter            : register(t11);
Texture2D                               t_ProcSkyIrradiance         : register(t12);
Texture3D                               t_ProcSkyClouds             : register(t13);
Texture2D                               t_ProcSkyNoise              : register(t14);

SamplerState                            s_Point                     : register(s0);
SamplerState                            s_Linear                    : register(s1);
SamplerState                            s_EquiRectSampler           : register(s2);

// Originally from https://github.com/GameTechDev/XeGTAO/blob/0b276c0ce820475c2adf6e2f3b696b696c172f43/Source/Rendering/Shaders/vaShared.hlsl#L4
float3 CubemapGetDirectionFor(uint face, float2 uv)
{
    // map [0, dim] to [-1,1] with (-1,-1) at bottom left
    float cx = (uv.x * 2.0) - 1;
    float cy = 1 - (uv.y * 2.0);    // <- not entirely sure about this bit

    float3 dir;
    const float l = sqrt(cx * cx + cy * cy + 1);
    switch (face) 
    {
        case 0:  dir = float3(   1, cy, -cx ); break;  // PX
        case 1:  dir = float3(  -1, cy,  cx ); break;  // NX
        case 2:  dir = float3(  cx,  1, -cy ); break;  // PY
        case 3:  dir = float3(  cx, -1,  cy ); break;  // NY
        case 4:  dir = float3(  cx, cy,   1 ); break;  // PZ
        case 5:  dir = float3( -cx, cy,  -1 ); break;  // NZ
        default: dir = 0.0.xxx; break;
    }
    return dir * (1 / l);
}
float3 CubemapGetDirectionFor( uint cubeDim, uint face, uint ux, uint uy )
{
    return CubemapGetDirectionFor( face, float2(ux+0.5,uy+0.5) / cubeDim.xx );
}

float3 SampleSource( uint2 pixel, uint face, float2 offset )
{
    float3 direction = CubemapGetDirectionFor( face, (float2(pixel) + offset + 0.5.xx ) / float(g_Const.CubeDim).xx );
    if (g_Const.BackgroundSourceType == 1)
    {
        float2 uv = world_to_latlong_map(direction);
        return t_SrcEquirectangularEnvMap.SampleLevel( s_EquiRectSampler, uv, 0 ).rgb;
    }
    else if (g_Const.BackgroundSourceType == 2)
        return t_SrcCubemapEnvMap.SampleLevel( s_Linear, direction, 0 ).rgb;
    else
        return float3(0,0,0);
}

// from "filament\libs\ibl\src\CubemapUtils.cpp"
// Area of a cube face's quadrant projected onto a sphere
static inline float SphereQuadrantArea(float x, float y) 
{
    return atan2( x*y, sqrt( x*x + y*y + 1 ) );
}
// the sum of all segments should be 4*pi*r^2 or ~12.56637 :)
float CubemapTexelSolidAngle(float cubeDim, uint2 coord) 
{
    const float iDim = 1.0f / cubeDim;
    float s = ((coord.x + 0.5f) * 2 * iDim) - 1;
    float t = ((coord.y + 0.5f) * 2 * iDim) - 1;
    const float x0 = s - iDim;
    const float y0 = t - iDim;
    const float x1 = s + iDim;
    const float y1 = t + iDim;
    float solidAngle =  SphereQuadrantArea( x0, y0 ) -
                        SphereQuadrantArea( x0, y1 ) -
                        SphereQuadrantArea( x1, y0 ) +
                        SphereQuadrantArea( x1, y1 );
    return max( 1e-6, abs(solidAngle) );
}

// Note: a lot of this can be optimized if need be; currently the cost is tiny, but could grow in case of many directional lights
// It's also not completely physically correct especially for very small angular sizes due to the way coverage is computed.
float3 ComputeLightContribution( uint2 pixel, uint face, const EMB_DirectionalLight light )
{
#if 0 // this provides a binary "either in or out of cone" coverage that is a.) incorrect and b.) aliased
    float3 direction = CubemapGetDirectionFor( face, (float2(pixel) + 0.5.xx ) / float(g_Const.CubeDim).xx );
    float angle = acos( clamp( dot(-light.Direction, direction), -1.0, 1.0 ) );
    float pixelCoverage = angle < (light.AngularSize*0.5);
#else // this provides a coverage approximation that is still not physically correct but removes aliasing
    const float fadeRangeInTexels = 1.1;
    float3 direction0 = CubemapGetDirectionFor( face, (float2(pixel) + 0.5.xx + 0.5 * float2(-fadeRangeInTexels, -fadeRangeInTexels)) / float(g_Const.CubeDim).xx );
    float3 direction1 = CubemapGetDirectionFor( face, (float2(pixel) + 0.5.xx + 0.5 * float2(+fadeRangeInTexels, -fadeRangeInTexels)) / float(g_Const.CubeDim).xx );
    float3 direction2 = CubemapGetDirectionFor( face, (float2(pixel) + 0.5.xx + 0.5 * float2(-fadeRangeInTexels, +fadeRangeInTexels)) / float(g_Const.CubeDim).xx );
    float3 direction3 = CubemapGetDirectionFor( face, (float2(pixel) + 0.5.xx + 0.5 * float2(+fadeRangeInTexels, +fadeRangeInTexels)) / float(g_Const.CubeDim).xx );
    float dotMin = min( min( dot(-light.Direction, direction0), dot(-light.Direction, direction1) ), min( dot(-light.Direction, direction2), dot(-light.Direction, direction3) ) );
    float dotMax = max( max( dot(-light.Direction, direction0), dot(-light.Direction, direction1) ), max( dot(-light.Direction, direction2), dot(-light.Direction, direction3) ) );

    float angleMin = acos( clamp( dotMax, -1.0, 1.0 ) );
    float angleMax = acos( clamp( dotMin, -1.0, 1.0 ) );

    float pixelCoverage = saturate( ((light.AngularSize*0.5)-angleMin) / (angleMax-angleMin+1e-24) );
    pixelCoverage = pow(pixelCoverage, 4); // just roughly account for tonemapping and gamma curve to get nicer AA
#endif

    float lightSolidAngle = 2 * K_PI * ( 1 - cos(light.AngularSize*0.5) );

    return pixelCoverage * light.ColorIntensity.rgb * (light.ColorIntensity.a / lightSolidAngle);
}

ProceduralSkyWorkingContext GetProcSkyContext()
{
    ProceduralSkyWorkingContext pswc;
    pswc.Consts                 = g_Const.ProcSkyConsts;
    pswc.SamplerLinearWrap      = s_Linear;
    pswc.TransmittanceTexture   = t_ProcSkyTransmittance;
    pswc.ScatterTexture         = t_ProcSkyScatter;
    pswc.IrradianceTexture      = t_ProcSkyIrradiance;
    pswc.CloudsTexture          = t_ProcSkyClouds;
    pswc.NoiseTexture           = t_ProcSkyNoise;
    return pswc;
}

float4 GenerateTexel( const uint2 cubePixelPos, const uint cubeFace, const uint cubeRes )
{
    float3 envCol = 0;

#if 1
    envCol += SampleSource( cubePixelPos, cubeFace, float2(0,0) );
#else   // 2x2 filter
    float blurOffset = 0.1f;
    envCol += SampleSource( cubePixelPos, cubeFace, float2(-blurOffset,-blurOffset) );
    envCol += SampleSource( cubePixelPos, cubeFace, float2(+blurOffset,-blurOffset) );
    envCol += SampleSource( cubePixelPos, cubeFace, float2(-blurOffset,+blurOffset) );
    envCol += SampleSource( cubePixelPos, cubeFace, float2(+blurOffset,+blurOffset) );
    envCol /= 4.0;
#endif

    for (uint i = 0; i < g_Const.DirectionalLightCount; i++ )
        envCol += ComputeLightContribution( cubePixelPos, cubeFace, g_Const.DirectionalLights[i] );

    float3 cubeDir          = CubemapGetDirectionFor( cubeFace, (float2(cubePixelPos) + 0.5.xx ) / float(g_Const.CubeDim).xx );
    float3 cubeDirRight     = CubemapGetDirectionFor( cubeFace, (float2(cubePixelPos+float2(1,0)) + 0.5.xx ) / float(g_Const.CubeDim).xx ) - cubeDir;
    float3 cubeDirBottom    = CubemapGetDirectionFor( cubeFace, (float2(cubePixelPos+float2(0,1)) + 0.5.xx ) / float(g_Const.CubeDim).xx ) - cubeDir;

    if (g_Const.ProcSkyEnabled)
    {
        float3x3 toLocal = {    1, 0, 0, 
                                0, 0, 1, 
                                0, 1, 0 };
        float3 localDirection = mul( cubeDir, toLocal );

        envCol += ProceduralSky( g_Const.CubeDim, uint3(cubePixelPos, cubeFace), localDirection, GetProcSkyContext(), t_LowResPrePassCube, cubeDir, cubeDirRight, cubeDirBottom );
    }
    
    envCol *= g_Const.ScaleColor;
   
    // clamp within meaningful range (negative - meaningless, and fp16 max val)
    const float fp16Max = HLF_MAX; //65504.0;
    envCol = clamp( envCol, 0.0.xxx, fp16Max.xxx );    
   
    return float4( envCol, 1 );
}

// (Low res layer): process top (MIP 0) environment map cubemap (all 6 faces!)
[numthreads(EMB_NUM_COMPUTE_THREADS_PER_DIM, EMB_NUM_COMPUTE_THREADS_PER_DIM, 1)]
void LowResPrePassLayerCS( uint3 dispatchThreadID : SV_DispatchThreadID )
{
    const uint cubeFace = dispatchThreadID.z;
    uint2 cubePixelPos = dispatchThreadID.xy;
    const uint cubeRes = g_Const.CubeDimLowRes;

    if (g_Const.ProcSkyEnabled)
    {
        float3 direction = CubemapGetDirectionFor( cubeFace, (float2(cubePixelPos) + 0.5.xx ) / float(cubeRes).xx );
        float3x3 toLocal = {    1, 0, 0, 
                                0, 0, 1, 
                                0, 1, 0 };
        float3 localDirection = mul( direction, toLocal );

        u_EnvMapCubeFacesDst0[dispatchThreadID.xyz] = ProceduralSkyLowRes( cubeRes, uint3(cubePixelPos, cubeFace), localDirection, GetProcSkyContext() );
    }
}

// Process top (MIP 0) of our dynamic (or static) environment map cubemap (all 6 faces!)
[numthreads(EMB_NUM_COMPUTE_THREADS_PER_DIM, EMB_NUM_COMPUTE_THREADS_PER_DIM, 1)]
void BaseLayerCS( uint3 dispatchThreadID : SV_DispatchThreadID )
{
    const uint cubeFace = dispatchThreadID.z;
    uint2 cubePixelPos = dispatchThreadID.xy;

    const uint cubeRes = g_Const.CubeDim;
   
#if 0 // simple 1 per pixel

    u_EnvMapCubeFacesDst[ dispatchThreadID.xyz ] = GenerateTexel(cubePixelPos, cubeFace, cubeRes);

#else // 4 per pixel + first MIP downsample

    // note: in this case, cubePixelPos is the 1st mip pixel pos!
    uint3 src00 = uint3(cubePixelPos.x*2 + 0, cubePixelPos.y*2 + 0, cubeFace);
    uint3 src01 = uint3(cubePixelPos.x*2 + 0, cubePixelPos.y*2 + 1, cubeFace);
    uint3 src10 = uint3(cubePixelPos.x*2 + 1, cubePixelPos.y*2 + 0, cubeFace);
    uint3 src11 = uint3(cubePixelPos.x*2 + 1, cubePixelPos.y*2 + 1, cubeFace);
    
    float w00 = CubemapTexelSolidAngle(cubeRes, src00.xy);
    float w01 = CubemapTexelSolidAngle(cubeRes, src01.xy);
    float w10 = CubemapTexelSolidAngle(cubeRes, src10.xy);
    float w11 = CubemapTexelSolidAngle(cubeRes, src11.xy);
    
    float wsum = w00+w01+w10+w11;
    
    float4 env00 = GenerateTexel(src00.xy, cubeFace, cubeRes);
    float4 env01 = GenerateTexel(src01.xy, cubeFace, cubeRes);
    float4 env10 = GenerateTexel(src10.xy, cubeFace, cubeRes);
    float4 env11 = GenerateTexel(src11.xy, cubeFace, cubeRes);

    u_EnvMapCubeFacesDst0[src00] = env00;
    u_EnvMapCubeFacesDst0[src01] = env01;
    u_EnvMapCubeFacesDst0[src10] = env10;
    u_EnvMapCubeFacesDst0[src11] = env11;
    
    float4 c00 = env00 * w00;
    float4 c01 = env01 * w01;
    float4 c10 = env10 * w10;
    float4 c11 = env11 * w11;
    
    float4 wavg = (c00+c01+c10+c11) / wsum;
    
    u_EnvMapCubeFacesDst1[uint3(cubePixelPos, cubeFace)] = wavg;
#endif
}

// Downsample a single cubemap MIP layer using physically correct (solid angle weighted) approach. Values are expected to be in range supported by output texture.
// Brute force, not tuned for speed.
[numthreads(EMB_NUM_COMPUTE_THREADS_PER_DIM, EMB_NUM_COMPUTE_THREADS_PER_DIM, 1)]
void MIPReduceCS( uint3 dispatchThreadID : SV_DispatchThreadID )
{
    const uint cubeFace = dispatchThreadID.z;

    uint2 cubePixelPos = dispatchThreadID.xy;
    
    uint width, height, elements;
    u_EnvMapCubeFacesDst.GetDimensions(width, height, elements);
    const uint destinationRes = width;
    
    // skip out of range texels
    if (any(cubePixelPos>=uint2(destinationRes.xx)))
        return;
    
    uint3 src00 = uint3(cubePixelPos.x*2 + 0, cubePixelPos.y*2 + 0, cubeFace);
    uint3 src01 = uint3(cubePixelPos.x*2 + 0, cubePixelPos.y*2 + 1, cubeFace);
    uint3 src10 = uint3(cubePixelPos.x*2 + 1, cubePixelPos.y*2 + 0, cubeFace);
    uint3 src11 = uint3(cubePixelPos.x*2 + 1, cubePixelPos.y*2 + 1, cubeFace);
    
    float w00 = CubemapTexelSolidAngle(destinationRes*2, src00.xy);
    float w01 = CubemapTexelSolidAngle(destinationRes*2, src01.xy);
    float w10 = CubemapTexelSolidAngle(destinationRes*2, src10.xy);
    float w11 = CubemapTexelSolidAngle(destinationRes*2, src11.xy);
    
    float wsum = w00+w01+w10+w11;
    
    float4 c00 = u_EnvMapCubeFacesSrc[src00] * w00;
    float4 c01 = u_EnvMapCubeFacesSrc[src01] * w01;
    float4 c10 = u_EnvMapCubeFacesSrc[src10] * w10;
    float4 c11 = u_EnvMapCubeFacesSrc[src11] * w11;
    
    float4 wavg = (c00+c01+c10+c11) / wsum;
    u_EnvMapCubeFacesDst[ dispatchThreadID.xyz ] = float4(wavg);
}

#endif // #if !defined(__cplusplus)

#endif // __ENVMAP_BAKER_HLSL__