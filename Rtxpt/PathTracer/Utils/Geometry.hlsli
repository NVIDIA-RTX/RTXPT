/*
* Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#ifndef __GEOMETRY_HLSLI__
#define __GEOMETRY_HLSLI__

#include "Utils.hlsli"

// Constructs an orthonormal basis based on the provided normal. See https://graphics.pixar.com/library/OrthonormalB/paper.pdf
void BranchlessONB(in const float3 normal, out float3 tangent, out float3 bitangent)
{
    float sign = (normal.z >= 0) ? 1 : -1;
    float a = -1.0 / (sign + normal.z);
    float b = normal.x * normal.y * a;
    tangent = float3(1.0f + sign * normal.x * normal.x * a, sign * b, -sign * normal.x);
    bitangent = float3(b, sign + normal.y * normal.y * a, -normal.y);
}

float2 SampleDiskUniform(float2 rand)
{
    float angle = 2 * K_PI * rand.x;
    return float2(cos(angle), sin(angle)) * sqrt(rand.y);
}

// Returns uniformly sampled barycentric coordinates inside a triangle
float3 SampleTriangleUniform(float2 rndSample)
{
    float sqrtx = sqrt(rndSample.x);

    return float3(
        1 - sqrtx,
        sqrtx * (1 - rndSample.y),
        sqrtx * rndSample.y);
}

// http://www.pbr-book.org/3ed-2018/Monte_Carlo_Integration/2D_Sampling_with_Multidimensional_Transformations.html
float3 SampleSphereUniform( const float2 u )
{
    float z = 1.0 - 2.0 * u[0];
    float r = sqrt( max( 0.0, 1.0 - z * z) );
    float phi = 2 * K_PI * u[1];
    return float3(r * cos(phi), r * sin(phi), z);
}
float SampleSphereUniformPDF( )
{
    return 1.0 / (4.0 * K_PI);
}

// http://www.rorydriscoll.com/2009/01/07/better-sampling/
float3 SampleHemisphereCosineWeighted( const float2 u )
{
    const float r = sqrt(u.x);
    const float theta = 2 * K_PI * u.y;

    const float x = r * cos(theta);
    const float y = r * sin(theta);

    return float3(x, y, sqrt(max(0.0, 1.0 - u.x)));
}
float SampleHemisphereCosineWeightedPDF( float cosTheta )
{ 
    return max( 0, cosTheta ) / K_PI;
}
float3 SampleHemisphereCosineWeighted( const float2 u, out float pdf )
{
    float3 ret = SampleHemisphereCosineWeighted( u );
    pdf = SampleHemisphereCosineWeightedPDF( ret.z );
    return ret;
}

// For converting an area measure pdf to solid angle measure pdf
float pdfAtoW(float pdfA, float distance_, float cosTheta)
{
    return pdfA * sq(distance_) / max(cosTheta, 2e-9f); //< TODO: take this epsilon out as a global const
}

#endif // __GEOMETRY_HLSLI__
