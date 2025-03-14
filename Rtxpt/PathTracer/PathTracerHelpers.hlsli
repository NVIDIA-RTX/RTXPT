/*
* Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#ifndef __PATH_TRACER_HELPERS_HLSLI__ // using instead of "#pragma once" due to https://github.com/microsoft/DirectXShaderCompiler/issues/3943
#define __PATH_TRACER_HELPERS_HLSLI__

#include "Config.h"    
#include "PathTracerShared.h"
#include "Utils/Math/MathConstants.hlsli"
#include "Utils/Math/Quaternion.hlsli"
#include "Utils/Math/Ray.hlsli"
#include "Utils/ColorHelpers.hlsli"

// Computes new ray origin based on hit position to avoid self-intersections. 
// The function assumes that the hit position has been computed by barycentric interpolation, and not from the ray t which is less accurate.
// Described in Ray Tracing Gems, Chapter 6, "A Fast and Robust Method for Avoiding Self-Intersection" by Carsten Wächter and Nikolaus Binder.
float3 ComputeRayOrigin(float3 worldPosition, float3 faceNormal)  // expects triangle faceNormal pointing towards the intended ray direction
{
    const float origin = 1.f / 16.f;
    const float fScale = 3.f / 65536.f;
    const float iScale = 3 * 256.f;

    // Per-component integer offset to bit representation of fp32 position.
    int3 iOff = int3(faceNormal * iScale);
    float3 iPos = asfloat(asint(worldPosition) + select(worldPosition < 0.f, -iOff, iOff));

    // Select per-component between small fixed offset or above variable offset depending on distance to origin.
    float3 fOff = faceNormal * fScale;
    return select(abs(worldPosition) < origin, worldPosition + fOff, iPos);
}

// When applying direct lighting to opaque low poly geometry, there will be areas where triangle face is facing away from the light
// source, but interpolated geometry normal (and normalmapped normal...) is facing towards thus still receiving light, resulting in 
// sharp triangle-shaped shadows when using shadow rays.
// Easiest (and fairly robust) solution to this is to apply NdotL-based light falloff at low grazing angles.
float ComputeLowGrazingAngleFalloff( float3 lightDirection, float3 interpolatedGeometryNormal, float falloffFrom, float falloffRange )
{
    // This could be optimized to dot()*a+b but then it's less readable
    return saturate( ( dot( lightDirection, interpolatedGeometryNormal) - falloffFrom ) / falloffRange ); // one could use, like in NRD Sample, saturate( ( smoothstep( 0.03, 0.1, dot( ray.Direction, shadingData.vertexN ) ) ) ); // also sqrt gives a visually smoother transition
}


float BalanceHeuristic(float nf, float fPdf, float ng, float gPdf) 
{
    float f = nf * fPdf;
    float g = ng * gPdf;
    return (f) / (f + g);
}
float PowerHeuristic(float nf, float fPdf, float ng, float gPdf) 
{
    float f = nf * fPdf;
    float g = ng * gPdf;
    return (f * f) / (f * f + g * g);
}

// !! Ported from ..\Falcor\Source\Falcor\Scene\Camera\Camera.hlsli !!
/** Computes the primary ray's direction, non-normalized assuming pinhole camera model.
    The camera jitter is taken into account to compute the sample position on the image plane.
    \param[in] data original Falcor CameraData equivalent.
    \param[in] pixel Pixel coordinates with origin in top-left.
    \param[in] jitter Per-pixel jitter.
    \return Returns the non-normalized ray direction
*/
float3 ComputeNonNormalizedRayDirPinhole( PathTracerCameraData data, uint2 pixel, float2 jitter )
{
    // Compute sample position in screen space in [0,1] with origin at the top-left corner.
    // The camera jitter offsets the sample by +-0.5 pixels from the pixel center.
    float2 p = (pixel + float2(0.5f, 0.5f) + jitter) / (float2)data.ViewportSize;
    // if (applyJitter) p += float2(-data.jitterX, data.jitterY);
    float2 ndc = float2(2, -2) * p + float2(-1, 1);

    // Compute the non-normalized ray direction assuming a pinhole camera.
    return ndc.x * data.CameraU + ndc.y * data.CameraV + data.CameraW;
}

// !! Ported from ..\Falcor\Source\Falcor\Scene\Camera\Camera.hlsli !!
/** Computes a camera ray for a given pixel assuming a pinhole camera model.
    The camera jitter is taken into account to compute the sample position on the image plane.
    \param[in] data original Falcor CameraData equivalent.
    \param[in] pixel Pixel coordinates with origin in top-left.
    \param[in] frameDim Image plane dimensions in pixels.
    \param[in] jitter Per-pixel jitter.
    \return Returns the camera ray.
*/
Ray ComputeRayPinhole( PathTracerCameraData data, uint2 pixel, float2 jitter )
{
    Ray ray;

    // Compute the normalized ray direction assuming a pinhole camera.
    ray.origin      = data.PosW;
    ray.dir         = normalize( ComputeNonNormalizedRayDirPinhole( data, pixel, jitter ) );

    float invCos    = 1.f / dot(normalize(data.CameraW), ray.dir);
    ray.tMin        = data.NearZ * invCos;
    ray.tMax        = data.FarZ * invCos;

#if 1    // plays nicer with debugging code
    ray.origin += ray.dir * ray.tMin;
    ray.tMin = 0;
    ray.tMax -= ray.tMin;
#endif
    return ray;
}

// !! Ported from ..\Falcor\Source\Falcor\Scene\Camera\Camera.hlsli !!
/** Computes a camera ray for a given pixel assuming a thin-lens camera model.
    The camera jitter is taken into account to compute the sample position on the image plane.
    \param[in] data original Falcor CameraData equivalent.
    \param[in] pixel Pixel coordinates with origin in top-left.
    \param[in] jitter Per-pixel jitter.
    \param[in] sample2D Uniform 2D sample.
    \return Returns the camera ray.
*/
Ray ComputeRayThinlens( PathTracerCameraData data, uint2 pixel, float2 jitter, float2 sample2D )
{
    Ray ray;

    // Sample position in screen space in [0,1] with origin at the top-left corner.
    // The camera jitter offsets the sample by +-0.5 pixels from the pixel center.
    float2 p = (pixel + float2(0.5f, 0.5f) + float2(-jitter.x, jitter.y)) / (float2)data.ViewportSize;
    float2 ndc = float2(2, -2) * p + float2(-1, 1);

    // Compute the normalized ray direction assuming a thin-lens camera.
    ray.origin  = data.PosW;
    ray.dir     = ndc.x * data.CameraU + ndc.y * data.CameraV + data.CameraW;
    float2 apertureSample = sample_disk(sample2D); // Sample lies in the unit disk [-1,1]^2
    float3 rayTarget = ray.origin + ray.dir;
    ray.origin  += data.ApertureRadius * (apertureSample.x * normalize(data.CameraU) + apertureSample.y * normalize(data.CameraV));
    ray.dir     = normalize(rayTarget - ray.origin);

    float invCos = 1.f / dot(normalize(data.CameraW), ray.dir);
    ray.tMin = data.NearZ * invCos;
    ray.tMax = data.FarZ * invCos;

#if 1    // plays nicer with debugging code
    ray.origin += ray.dir * ray.tMin;
    ray.tMin = 0;
    ray.tMax -= ray.tMin;
#endif
    return ray;
}

// "Improved Shader and Texture Level of Detail Using Ray Cones", Chapter 4 "Integrating BRDF Roughness" / Appendix A, "BRDF Roughness Derivations"; Journal of Computer Graphics Techniques, Vol. 10, No. 1, 2021
// https://www.jcgt.org/published/0010/01/01/paper.pdf
float RoughnessToVariance(float roughness)
{
    float ggxAlpha = roughness * roughness;
    float s = ggxAlpha * ggxAlpha;
    s = min(s, 0.99); // Prevents division by zero.
    return (s / (1.0f - s)) * 0.5;
}
float GetAngleFromGGXRoughness(float roughness)
{
    float sigma2 = RoughnessToVariance(roughness);
    return sqrt(sigma2);
}

// https://www.jcgt.org/published/0010/01/01/paper.pdf "Improved Shader and Texture Level of Detail Using Ray Cones", Chapter 4 "Integrating BRDF Roughness" 
// covers case of Lambertian (diffuse) materials.
// 'diffuseToAngleFactor' is based on "Improved Shader and Texture Level of Detail Using Ray Cones", Chapter 3. Curvature Approximations:
// "...On the other hand, when ray cones are used inside a Monte Carlo path tracer, one would prefer slightly underestimating the spread angle,
// since antialiasing will be handled by stochastic supersampling anyway, and the main objective would be to avoid introducing overblur in the results."
float ComputeRayConeSpreadAngleExpansionByRoughness(float roughness, float diffuseToAngleFactor = 0.6)
{
    return diffuseToAngleFactor * GetAngleFromGGXRoughness(sqrt(roughness));
}

// Experimental ray cone spread heuristic: assume pdf comes from an uniform sphere cap lobe. Then we can compute cone spread
// angle alpha (a plane angle) from the uniform sphere cap solid angle (omega), which can be derived from pdf 
// (omega = 1 / uniform_sphere_cap_pdf). 
// The formula is alpha = 2 * acos( 1 - omega / 2*PI ) - see https://rechneronline.de/winkel/solid-angle.php
// (This heuristic starts to break down for BSDFs with overlapping lobes but seems good enough in most cases - perhaps BSDF should be responsible providing the scatter angle).
//
// growthFactor 0.3 is very conservative underestimation, see https://www.jcgt.org/published/0010/01/01/paper.pdf, "Improved Shader and Texture Level of Detail Using Ray Cones", 
// Chapter 3. Curvature Approximations            "...On the other hand, when ray cones are used inside a Monte Carlo path tracer, one would prefer slightly underestimating the 
// spread angle, since antialiasing will be handled by stochastic supersampling anyway, and the main objective would be to avoid introducing overblur in the results."
float ComputeRayConeSpreadAngleExpansionByScatterPDF(float bsdfScatterPdf, const float growthFactor = 0.3)
{
    return growthFactor * 2.0 * FastACos(max( -1.0, 1.0 - (1.0 / bsdfScatterPdf) / (2.0 * K_PI) ) );    // fast acos would work just fine
}

// Ad-hoc heuristic: reduce firefly threshold the more we bounce, but make it dependent on pdf and lobe probability
lpfloat ComputeNewScatterFireflyFilterK(const lpfloat currentK, float bouncePDF, float lobeP)
{
    const float minK = 0.00001;
    float angle = (bouncePDF==0)?(0):(ComputeRayConeSpreadAngleExpansionByScatterPDF(bouncePDF, 1.0));
    const float k = 32;                 // found empirically
    float p = k / (k+angle*angle);      // found empirically
    p *= FastSqrt(lobeP);               // square root behaves better empirically
    return (lpfloat)max( minK, currentK * p );
}

// Experimental: Biased cap to maximum radiance based on current vs starting ray cone spread angle, used as a rough estimate of probability of the path.
float3 FireflyFilter(float3 signalIn, const float threshold, const float fireflyFilterK)
{
    float fireflyFilterThreshold = threshold * fireflyFilterK;
    float maxR = average( signalIn );   // we used to use 'luminance', but that will actually result in hue shift towards blue!
    if( maxR > fireflyFilterThreshold )
        signalIn = signalIn / maxR * fireflyFilterThreshold;
    return signalIn;
}
float FireflyFilterShort(float signalAverage, const float threshold, const float fireflyFilterK)
{
    float fireflyFilterThreshold = threshold * fireflyFilterK;
    float maxR = signalAverage;
    return ( maxR > fireflyFilterThreshold )?( 1.0 / maxR * fireflyFilterThreshold ):(1.0);
}

void swap(inout float a, inout float b) { float t = a; a = b; b = t; }
void swap(inout int a, inout int b)     { int t = a; a = b; b = t; }

// "Efficiently building a matrix to rotate one vector to another"
// http://cs.brown.edu/research/pubs/pdfs/1999/Moller-1999-EBA.pdf / https://dl.acm.org/doi/10.1080/10867651.1999.10487509
// (using https://github.com/assimp/assimp/blob/master/include/assimp/matrix3x3.inl#L275 as a code reference as it seems to be best, see https://github.com/assimp/assimp/blob/master/LICENSE)
float3x3 MatrixRotateFromTo( const float3 from, const float3 to, uniform bool columnMajor = true )
{
    const float e = dot(from, to);
    const float f = abs(e); //(e < 0)? -e:e;

    if( f > float( 1.0f - 1e-10f ) )
        return float3x3(1,0,0,0,1,0,0,0,1);

    const float3 v      = cross( from, to );
    /* ... use this hand optimized version (9 mults less) */
    const float h       = (1.0f)/(1.0f + e);      /* optimization by Gottfried Chen */
    const float hvx     = h * v.x;
    const float hvz     = h * v.z;
    const float hvxy    = hvx * v.y;
    const float hvxz    = hvx * v.z;
    const float hvyz    = hvz * v.y;

    float3x3 mtx;
    if (columnMajor)
    {
        mtx[0][0] = e + hvx * v.x;
        mtx[0][1] = hvxy - v.z;
        mtx[0][2] = hvxz + v.y;
        mtx[1][0] = hvxy + v.z;
        mtx[1][1] = e + h * v.y * v.y;
        mtx[1][2] = hvyz - v.x;
        mtx[2][0] = hvxz - v.y;
        mtx[2][1] = hvyz + v.x;
        mtx[2][2] = e + hvz * v.z;
    }
    else
    {
        mtx[0][0] = e + hvx * v.x;
        mtx[1][0] = hvxy - v.z;
        mtx[2][0] = hvxz + v.y;
        mtx[0][1] = hvxy + v.z;
        mtx[1][1] = e + h * v.y * v.y;
        mtx[2][1] = hvyz - v.x;
        mtx[0][2] = hvxz - v.y;
        mtx[1][2] = hvyz + v.x;
        mtx[2][2] = e + hvz * v.z;
    }
    return mtx;
}

// column-major
float4 QuaternionFromRotationMatrix( float3x3 mat )
{
    mat = transpose(mat);
    float4 ret;
    // from http://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/, converted to row-major
    float trace = mat[0][0] + mat[1][1] + mat[2][2]; // I removed + 1.0f; see discussion with Ethan
    if( trace > 0 ) // I changed M_EPSILON to 0
    {
        float s = 0.5f / sqrt(trace+ 1.0f);
        ret.w = 0.25f / s;
        ret.x = ( mat[1][2] - mat[2][1] ) * s;
        ret.y = ( mat[2][0] - mat[0][2] ) * s;
        ret.z = ( mat[0][1] - mat[1][0] ) * s;
    } 
    else 
    {
        if ( mat[0][0] > mat[1][1] && mat[0][0] > mat[2][2] ) 
        {
            float s = 2.0f * sqrt( 1.0f + mat[0][0] - mat[1][1] - mat[2][2]);
            ret.w = (mat[1][2] - mat[2][1] ) / s;
            ret.x = 0.25f * s;
            ret.y = (mat[1][0] + mat[0][1] ) / s;
            ret.z = (mat[2][0] + mat[0][2] ) / s;
        } 
        else if (mat[1][1] > mat[2][2]) 
        {
            float s = 2.0f * sqrt( 1.0f + mat[1][1] - mat[0][0] - mat[2][2]);
            ret.w = (mat[2][0] - mat[0][2] ) / s;
            ret.x = (mat[1][0] + mat[0][1] ) / s;
            ret.y = 0.25f * s;
            ret.z = (mat[2][1] + mat[1][2] ) / s;
        } 
        else 
        {
            float s = 2.0f * sqrt( 1.0f + mat[2][2] - mat[0][0] - mat[1][1] );
            ret.w = (mat[0][1] - mat[1][0] ) / s;
            ret.x = (mat[2][0] + mat[0][2] ) / s;
            ret.y = (mat[2][1] + mat[1][2] ) / s;
            ret.z = 0.25f * s;
        }
    }
    return ret;
}

#endif // __PATH_TRACER_HELPERS_HLSLI__