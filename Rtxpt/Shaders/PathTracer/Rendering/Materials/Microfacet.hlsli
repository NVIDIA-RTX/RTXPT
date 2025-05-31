/*
* Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#ifndef __MICROFACET_HLSLI__ // using instead of "#pragma once" due to https://github.com/microsoft/DirectXShaderCompiler/issues/3943
#define __MICROFACET_HLSLI__

#if !defined(__cplusplus)

#include "../../Utils/Math/MathConstants.hlsli"
#include "BxDFConfig.hlsli"

/** Evaluates the GGX (Trowbridge-Reitz) normal distribution function (D).

    Introduced by Trowbridge and Reitz, "Average irregularity representation of a rough surface for ray reflection", Journal of the Optical Society of America, vol. 65(5), 1975.
    See the correct normalization factor in Walter et al. https://dl.acm.org/citation.cfm?id=2383874
    We use the simpler, but equivalent expression in Eqn 19 from http://blog.selfshadow.com/publications/s2012-shading-course/hoffman/s2012_pbs_physics_math_notes.pdf

    For microfacet models, D is evaluated for the direction h to find the density of potentially active microfacets (those for which microfacet normal m = h).
    The 'alpha' parameter is the standard GGX width, e.g., it is the square of the linear roughness parameter in Disney's BRDF.
    Note there is a singularity (0/0 = NaN) at NdotH = 1 and alpha = 0, so alpha should be clamped to some epsilon.

    \param[in] alpha GGX width parameter (should be clamped to small epsilon beforehand).
    \param[in] cosTheta Dot product between shading normal and half vector, in positive hemisphere.
    \return D(h)
*/
float evalNdfGGX(float alpha, float cosTheta)
{
    float a2 = alpha * alpha;
    float d = ((cosTheta * a2 - cosTheta) * cosTheta + 1);
    return a2 / (d * d * K_PI);
}

/** Evaluates the PDF for sampling the GGX normal distribution function using Walter et al. 2007's method.
    See https://www.cs.cornell.edu/~srm/publications/EGSR07-btdf.pdf

    \param[in] alpha GGX width parameter (should be clamped to small epsilon beforehand).
    \param[in] cosTheta Dot product between shading normal and half vector, in positive hemisphere.
    \return D(h) * cosTheta
*/
float evalPdfGGX_NDF(float alpha, float3 wi, float3 h)
{
    float cosTheta = h.z;
    return evalNdfGGX(alpha, cosTheta) * cosTheta / (max(0.f, dot(wi, h)) * 4.0f);  // "1.0 / max(0.f, dot(wi, h)) * 4.0f" term used to be applied externally
}

/** Samples the GGX (Trowbridge-Reitz) normal distribution function (D) using Walter et al. 2007's method.
    Note that the sampled half vector may lie in the negative hemisphere. Such samples should be discarded.
    See Eqn 35 & 36 in https://www.cs.cornell.edu/~srm/publications/EGSR07-btdf.pdf
    See Listing A.1 in https://seblagarde.files.wordpress.com/2015/07/course_notes_moving_frostbite_to_pbr_v32.pdf

    \param[in] alpha GGX width parameter (should be clamped to small epsilon beforehand).
    \param[in] u Uniform random number (2D).
    \param[out] pdf Sampling probability.
    \return Sampled half vector in local space.
*/
float3 sampleGGX_NDF(float alpha, float2 u)
{
    float alphaSqr = alpha * alpha;
    float phi = u.y * (2 * K_PI);
    float tanThetaSqr = alphaSqr * u.x / (1 - u.x);
    float cosTheta = 1 / sqrt(1 + tanThetaSqr);
    float r = sqrt(max(1 - cosTheta * cosTheta, 0));

    return float3(cos(phi) * r, sin(phi) * r, cosTheta);
}

float evalG1GGX(float alphaSqr, float cosTheta);

/** Evaluates the PDF for sampling the GGX distribution of visible normals (VNDF).
    See http://jcgt.org/published/0007/04/01/paper.pdf

    \param[in] alpha GGX width parameter (should be clamped to small epsilon beforehand).
    \param[in] wi Incident direction in local space, in the positive hemisphere.
    \param[in] h Half vector in local space, in the positive hemisphere.
    \return D_V(h) = G1(wi) * D(h) * max(0,dot(wi,h)) / wi.z
*/
float evalPdfGGX_VNDF(float alpha, float3 wi, float3 h)
{
    float G1 = evalG1GGX(alpha * alpha, wi.z);
    float D = evalNdfGGX(alpha, h.z);
    
#if 0   // old code; "1.0 / max(0.f, dot(wi, h)) * 4.0f" term used to be applied externally
    return G1 * D * max(0.f, dot(wi, h)) / wi.z;
#else
    return G1 * D * max(0.f, dot(wi, h)) / (wi.z * max(0.f, dot(wi, h)) * 4.0f);   // <- corrected?
#endif
}

/** Evaluates the PDF for sampling the GGX distribution of >bounded< visible normals (BVNDF).
    See https://gpuopen.com/download/publications/Bounded_VNDF_Sampling_for_Smith-GGX_Reflections.pdf, 
    Adapted from listing 2.

    \param[in] alpha GGX width parameter (should be clamped to small epsilon beforehand).
    \param[in] wi Incident direction in local space, in the positive hemisphere.
    \param[in] h Half vector in local space, in the positive hemisphere.
    \return pdf
*/
float evalPdfGGX_BVNDF( float _alpha, float3 i, float3 m ) 
{
    float2 alpha = _alpha.xx;                               // TODO: add support for anisotropic roughness
    //float3 m = normalize( i + o );
    float ndf = evalNdfGGX(_alpha, m.z); //D(m , alpha);    // TODO: add support for anisotropic roughness
    float2 ai = alpha * i.xy ;
    float len2 = dot(ai, ai );
    float t = sqrt ( len2 + i.z * i.z );
#if 0   // our i.z is always in positive hemisphere
    if ( i.z >= 0.0f )
#endif
    {
        float a = saturate(min(alpha.x, alpha.y)); // Eq. 6
        float s = 1.0f + length(float2(i.x, i.y)); // Omit sgn for a <=1
        float a2 = a * a;
        float s2 = s * s;
        float k = (1.0f - a2) * s2 / (s2 + a2 * i.z * i.z); // Eq. 5
        return ndf / (2.0f * (k * i.z + t)); // Eq. 8 * || dm/do ||
    }
#if 0   // our i.z is always in positive hemisphere
    // Numerically stable form of the previous PDF for i.z < 0
    return ndf * ( t - i.z ) / (2.0f * len2 ) ; // = Eq. 7 * || dm/do ||
#endif
}


/** Samples the GGX (Trowbridge-Reitz) using the distribution of visible normals (VNDF).
    The GGX VDNF yields significant variance reduction compared to sampling of the GGX NDF.
    See http://jcgt.org/published/0007/04/01/paper.pdf

    \param[in] alpha Isotropic GGX width parameter (should be clamped to small epsilon beforehand).
    \param[in] wi Incident direction in local space, in the positive hemisphere.
    \param[in] u Uniform random number (2D).
    // \param[out] pdf Sampling probability. - removed for simplicity / removing code duplication; use 'evalPdfGGX_VNDF', compiler is smart enough to optimize things out
    \return Sampled half vector in local space, in the positive hemisphere.
*/
float3 sampleGGX_VNDF(float alpha, float3 wi, float2 u)
{
    float alpha_x = alpha, alpha_y = alpha;

    // Transform the view vector to the hemisphere configuration.
    float3 Vh = normalize(float3(alpha_x * wi.x, alpha_y * wi.y, wi.z));

    // Construct orthonormal basis (Vh,T1,T2).
#if 0
    float3 T1 = (Vh.z < 0.9999f) ? normalize(cross(float3(0, 0, 1), Vh)) : float3(1, 0, 0); // TODO: fp32 precision
#else   
    // from latest http://jcgt.org/published/0007/04/01/paper.pdf - fewer instructions than above; 0.0002 threshold found empirically and matches above variant
    float lensq = Vh.x * Vh.x + Vh.y * Vh.y;
    float3 T1 = lensq > 0.0002f ? float3(-Vh.y, Vh.x, 0) * rsqrt(lensq) : float3(1,0,0);
#endif
    float3 T2 = cross(Vh, T1);

    // Parameterization of the projected area of the hemisphere.
    float r = sqrt(u.x);
    float phi = (2.f * K_PI) * u.y;
    float t1 = r * cos(phi);
    float t2 = r * sin(phi);
    float s = 0.5f * (1.f + Vh.z);
    t2 = (1.f - s) * sqrt(1.f - t1 * t1) + s * t2;

    // Reproject onto hemisphere.
    float3 Nh = t1 * T1 + t2 * T2 + sqrt(max(0.f, 1.f - t1 * t1 - t2 * t2)) * Vh;

    // Transform the normal back to the ellipsoid configuration. This is our half vector.
    float3 h = normalize(float3(alpha_x * Nh.x, alpha_y * Nh.y, max(0.f, Nh.z)));

    // pdf = evalPdfGGX_VNDF(alpha, wi, h);
    return h;
}

/** Samples the GGX using the >bounded< distribution of visible normals (VNDF).
    See https://gpuopen.com/download/publications/Bounded_VNDF_Sampling_for_Smith-GGX_Reflections.pdf,
    Adapted from listing 1.

    \param[in] alpha Isotropic GGX width parameter (should be clamped to small epsilon beforehand).
    \param[in] wi Incident direction in local space, in the positive hemisphere.
    \param[in] u Uniform random number (2D).
    \return Sampled half vector in local space, in the positive hemisphere.
*/
float3 sampleGGX_BVNDF(float _alpha, float3 i, float2 rand)
{
    float2 alpha = _alpha.xx;                               // TODO: add support for anisotropic roughness
    
    float3 i_std = normalize ( float3 ( i.xy * alpha, i.z ) ) ;
    // Sample a spherical cap
    float phi = 2.0f * K_PI * rand.x ;
    float a = saturate( min( alpha.x, alpha.y ) ); // Eq. 6
    float s = 1.0f + length( float2( i.x, i.y ) ); // Omit sgn for a <=1
    float a2 = a * a; float s2 = s * s;
    float k = (1.0f - a2) * s2 / (s2 + a2 * i.z * i.z); // Eq. 5
    float b = i.z > 0 ? k * i_std.z : i_std.z;
    float z = mad (1.0f - rand.y , 1.0f + b, -b );
    float sinTheta = sqrt( saturate( 1.0f - z * z ) );
    float3 o_std = float3( sinTheta * cos( phi ), sinTheta * sin( phi ), z );
    // Compute the microfacet normal m
    float3 m_std = i_std + o_std ;
    
    float3 m = normalize( float3( m_std.xy * alpha , m_std.z ) );
    
    // Transform the normal back to the ellipsoid configuration. This is our half vector. From this we can compute reflection vector with reflect(-ViewVector, h);
    return normalize( float3( m_std.xy * alpha , m_std.z ) );
}

/** Evaluates the Smith masking function (G1) for the GGX normal distribution.
    See Eq 34 in https://www.cs.cornell.edu/~srm/publications/EGSR07-btdf.pdf

    The evaluated direction is assumed to be in the positive hemisphere relative the half vector.
    This is the case when both incident and outgoing direction are in the same hemisphere, but care should be taken with transmission.

    \param[in] alphaSqr Squared GGX width parameter.
    \param[in] cosTheta Dot product between shading normal and evaluated direction, in the positive hemisphere.
*/
float evalG1GGX(float alphaSqr, float cosTheta)
{
    if (cosTheta <= 0) return 0;
    float cosThetaSqr = cosTheta * cosTheta;
    float tanThetaSqr = max(1 - cosThetaSqr, 0) / cosThetaSqr;
    return 2 / (1 + sqrt(1 + alphaSqr * tanThetaSqr));
}

/** Evaluates the Smith lambda function for the GGX normal distribution.
    See Eq 72 in http://jcgt.org/published/0003/02/03/paper.pdf

    \param[in] alphaSqr Squared GGX width parameter.
    \param[in] cosTheta Dot product between shading normal and the evaluated direction, in the positive hemisphere.
*/
float evalLambdaGGX(float alphaSqr, float cosTheta)
{
    if (cosTheta <= 0) return 0;
    float cosThetaSqr = cosTheta * cosTheta;
    float tanThetaSqr = max(1 - cosThetaSqr, 0) / cosThetaSqr;
    return 0.5 * (-1 + sqrt(1 + alphaSqr * tanThetaSqr));
}

/** Evaluates the separable form of the masking-shadowing function for the GGX normal distribution, using Smith's approximation.
    See Eq 98 in http://jcgt.org/published/0003/02/03/paper.pdf

    \param[in] alpha GGX width parameter (should be clamped to small epsilon beforehand).
    \param[in] cosThetaI Dot product between shading normal and incident direction, in positive hemisphere.
    \param[in] cosThetaO Dot product between shading normal and outgoing direction, in positive hemisphere.
    \return G(cosThetaI, cosThetaO)
*/
float evalMaskingSmithGGXSeparable(float alpha, float cosThetaI, float cosThetaO)
{
    float alphaSqr = alpha * alpha;
    float lambdaI = evalLambdaGGX(alphaSqr, cosThetaI);
    float lambdaO = evalLambdaGGX(alphaSqr, cosThetaO);
    return 1 / ((1 + lambdaI) * (1 + lambdaO));
}

/** Evaluates the height-correlated form of the masking-shadowing function for the GGX normal distribution, using Smith's approximation.
    See Eq 99 in http://jcgt.org/published/0003/02/03/paper.pdf

    Eric Heitz recommends using it in favor of the separable form as it is more accurate and of similar complexity.
    The function is only valid for cosThetaI > 0 and cosThetaO > 0  and should be clamped to 0 otherwise.

    \param[in] alpha GGX width parameter (should be clamped to small epsilon beforehand).
    \param[in] cosThetaI Dot product between shading normal and incident direction, in positive hemisphere.
    \param[in] cosThetaO Dot product between shading normal and outgoing direction, in positive hemisphere.
    \return G(cosThetaI, cosThetaO)
*/
float evalMaskingSmithGGXCorrelated(float alpha, float cosThetaI, float cosThetaO)
{
    float alphaSqr = alpha * alpha;
    float lambdaI = evalLambdaGGX(alphaSqr, cosThetaI);
    float lambdaO = evalLambdaGGX(alphaSqr, cosThetaO);
    return 1 / (1 + lambdaI + lambdaO);
}

/** Approximate pre-integrated specular BRDF. The approximation assumes GGX VNDF and Schlick's approximation.
    See Eq 4 in [Ray Tracing Gems, Chapter 32]

    \param[in] specularReflectance Reflectance from a direction parallel to the normal.
    \param[in] alpha GGX width parameter (should be clamped to small epsilon beforehand).
    \param[in] cosTheta Dot product between shading normal and evaluated direction, in the positive hemisphere.
*/
float3 approxSpecularIntegralGGX(float3 specularReflectance, float alpha, float cosTheta)
{
    cosTheta = abs(cosTheta);

    float4 X;
    X.x = 1.f;
    X.y = cosTheta;
    X.z = cosTheta * cosTheta;
    X.w = cosTheta * X.z;

    float4 Y;
    Y.x = 1.f;
    Y.y = alpha;
    Y.z = alpha * alpha;
    Y.w = alpha * Y.z;

    // Select coefficients based on BRDF version being in use (either separable or correlated G term)
#if SpecularMaskingFunction == SpecularMaskingFunctionSmithGGXSeparable
    float2x2 M1 = float2x2(
        0.99044f, -1.28514f,
        1.29678f, -0.755907f
    );

    float3x3 M2 = float3x3(
        1.0f, 2.92338f, 59.4188f,
        20.3225f, -27.0302f, 222.592f,
        121.563f, 626.13f, 316.627f
    );

    float2x2 M3 = float2x2(
        0.0365463f, 3.32707f,
        9.0632f, -9.04756f
    );

    float3x3 M4 = float3x3(
        1.0f, 3.59685f, -1.36772f,
        9.04401f, -16.3174f, 9.22949f,
        5.56589f, 19.7886f, -20.2123f
    );
#elif SpecularMaskingFunction == SpecularMaskingFunctionSmithGGXCorrelated
    float2x2 M1 = float2x2(
        0.995367f, -1.38839f,
        -0.24751f, 1.97442f
    );

    float3x3 M2 = float3x3(
        1.0f, 2.68132f, 52.366f,
        16.0932f, -3.98452f, 59.3013f,
        -5.18731f, 255.259f, 2544.07f
    );

    float2x2 M3 = float2x2(
        -0.0564526f, 3.82901f,
        16.91f, -11.0303f
    );

    float3x3 M4 = float3x3(
        1.0f, 4.11118f, -1.37886f,
        19.3254f, -28.9947f, 16.9514f,
        0.545386f, 96.0994f, -79.4492f
    );
#endif

    float bias = dot(mul(M1, X.xy), Y.xy) * rcp(dot(mul(M2, X.xyw), Y.xyw));
    float scale = dot(mul(M3, X.xy), Y.xy) * rcp(dot(mul(M4, X.xzw), Y.xyw));

    // This is a hack for specular reflectance of 0
    float specularReflectanceLuma = dot(specularReflectance, float3( (1.f / 3.f).xxx ));
    bias *= saturate(specularReflectanceLuma * 50.0f);

    return mad(specularReflectance, max(0.0, scale), max(0.0, bias));
}

#endif // #if !defined(__cplusplus)

#endif // #ifdef __MICROFACET_HLSLI__
