/*
* Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

// Shared, binding-free cubemap helpers (face direction math, texel solid angle)

#ifndef __CUBEMAP_UTILS_HLSLI__
#define __CUBEMAP_UTILS_HLSLI__

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

// results are in 00 01 10 11 order
float4 CubemapTexelSolidAngle4(float cubeDim, uint2 coordTL)
{
    const float iDim = 1.0f / cubeDim;
    float s = ((coordTL.x + 0.5f) * 2 * iDim) - 1;
    float t = ((coordTL.y + 0.5f) * 2 * iDim) - 1;
    const float x0 = s - iDim;
    const float y0 = t - iDim;
    const float x1 = s + iDim;
    const float y1 = t + iDim;
    const float x2 = s + iDim*3;
    const float y2 = t + iDim*3;

    float sqa00 = SphereQuadrantArea( x0, y0 );
    float sqa01 = SphereQuadrantArea( x0, y1 );
    float sqa10 = SphereQuadrantArea( x1, y0 );
    float sqa11 = SphereQuadrantArea( x1, y1 );
    float sqa20 = SphereQuadrantArea( x2, y0 );
    float sqa21 = SphereQuadrantArea( x2, y1 );
    float sqa02 = SphereQuadrantArea( x0, y2 );
    float sqa12 = SphereQuadrantArea( x1, y2 );
    float sqa22 = SphereQuadrantArea( x2, y2 );

    float solidAngle00 =  max( 1e-6, abs( sqa00 - sqa01 - sqa10 + sqa11 ) );
    float solidAngle10 =  max( 1e-6, abs( sqa10 - sqa11 - sqa20 + sqa21 ) );
    float solidAngle01 =  max( 1e-6, abs( sqa01 - sqa02 - sqa11 + sqa12 ) );
    float solidAngle11 =  max( 1e-6, abs( sqa11 - sqa12 - sqa21 + sqa22 ) );
    return float4( solidAngle00, solidAngle01, solidAngle10, solidAngle11 );
}

#endif // __CUBEMAP_UTILS_HLSLI__
