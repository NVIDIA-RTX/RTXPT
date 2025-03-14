/*
* Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#ifndef __MATH_HELPERS_HLSLI__ // using instead of "#pragma once" due to https://github.com/microsoft/DirectXShaderCompiler/issues/3943
#define __MATH_HELPERS_HLSLI__

#include "../../Config.h"    

/** This file contains various math utility helper functions.

    Included functionality (in order):

    - Sherical coordinates mapping functions
    - Octahedral mapping functions
    - Sampling functions (disk, sphere, triangle etc.)
    - Misc stuff (matrix inversion, bounding cones etc.)

*/

// Include math constants (K_PI etc.)
#include "MathConstants.hlsli"

/******************************************************************************

    Spherical coordinates

    Functions for converting Cartesian coordinates to spherical coordinates
    using standard mathematical notations.

    The latitude-longitude map uses (phi,theta) as positions in two dimensions.
    Its using using other conventions to orient and wrap the map the same way
    as in common 3D software (e.g. Autodesk Maya).

******************************************************************************/

/** Converts Cartesian coordinates to spherical coordinates (unsigned normalized).
    'theta' is the polar angle (inclination) between the +z axis and the vector from origin to p, normalized to [0,1].
    'phi' is the azimuthal angle from the +x axis in the xy-plane, normalized to [0,1].
    \param[in] p Cartesian coordinates (x,y,z).
    \return Spherical coordinates (theta,phi).
*/
float2 cartesian_to_spherical_unorm(float3 p)
{
    p = normalize(p);
    float2 sph;
    sph.x = acos(p.z) * K_1_PI;
    sph.y = atan2(-p.y, -p.x) * K_1_2PI + 0.5f;
    return sph;
}

/** Converts Cartesian coordinates to spherical coordinates (radians).
    'theta' is the polar angle (inclination) between the +z axis and the vector from origin to p, in the range [0,pi].
    'phi' is the azimuthal angle from the +x axis in the xy-plane, in the range [0,2pi].
    \param[in] p Cartesian coordinates (x,y,z).
    \return Spherical coordinates (theta,phi).
*/
float2 cartesian_to_spherical_rad(float3 p)
{
    p = normalize(p);
    float2 sph;
    sph.x = acos(p.z);
    sph.y = atan2(-p.y, -p.x) + K_PI;
    return sph;
}

/** Converts spherical coordinates (radians) to Cartesian coordinates.
    Inverse of cartesian_to_spherical_rad.
    \param[in] sph Spherical coordinates (theta,phi).
    \return Cartesian coordinates (x,y,z).
*/
float3 spherical_to_cartesian_rad(float2 sph)
{
    float3 p;
    p.x = -cos(sph.y - K_PI) * sin(sph.x);
    p.y = -sin(sph.y - K_PI) * sin(sph.x);
    p.z = cos(sph.x);
    return p;
}

/** Convert world space direction to (u,v) coord in latitude-longitude map (unsigned normalized).
    The map is centered around the -z axis and wrapping around in clockwise order (left to right).
    \param[in] dir World space direction (unnormalized).
    \return Position in latitude-longitude map in [0,1] for each component.
*/
float2 world_to_latlong_map(float3 dir)
{
    float3 p = normalize(dir);
    float2 uv;
    uv.x = atan2(p.x, -p.z) * K_1_2PI + 0.5f;
    uv.y = acos(p.y) * K_1_PI;
    return uv;
}

/** Convert a coordinate in latitude-longitude map (unsigned normalized) to a world space direction.
    The map is centered around the -z axis and wrapping around in clockwise order (left to right).
    \param[in] latlong Position in latitude-longitude map in [0,1] for each component.
    \return Normalized direction in world space.
*/
float3 latlong_map_to_world(float2 latlong)
{
    float phi = K_PI * (2.f * saturate(latlong.x) - 1.f);
    float theta = K_PI * saturate(latlong.y);
    float sinTheta = sin(theta);
    float cosTheta = cos(theta);
    float sinPhi = sin(phi);
    float cosPhi = cos(phi);
    return float3(sinTheta * sinPhi, cosTheta, -sinTheta * cosPhi);
}

/******************************************************************************

    Octahedral mapping

    The center of the map represents the +z axis and its corners -z.
    The rotated inner square is the xy-plane projected onto the upper hemi-
    sphere, the outer four triangles folds down over the lower hemisphere.
    There are versions for equal-area and non-equal area (slightly faster).

    For details refer to:
    - Clarberg 2008, "Fast Equal-Area Mapping of the (Hemi)Sphere using SIMD".
    - Cigolle et al. 2014, "Survey of Efficient Representations for Independent Unit Vectors".

******************************************************************************/

/** Helper function to reflect the folds of the lower hemisphere
    over the diagonals in the octahedral map.
*/
float2 oct_wrap(float2 v)
{
    return (1.f - abs(v.yx)) * select(v.xy >= 0.f, 1.f, -1.f);
}

/** Converts normalized direction to the octahedral map (non-equal area, signed normalized).
    \param[in] n Normalized direction.
    \return Position in octahedral map in [-1,1] for each component.
*/
float2 ndir_to_oct_snorm(float3 n)
{
    // Project the sphere onto the octahedron (|x|+|y|+|z| = 1) and then onto the xy-plane.
    float2 p = n.xy * (1.f / (abs(n.x) + abs(n.y) + abs(n.z)));
    p = (n.z < 0.f) ? oct_wrap(p) : p;
    return p;
}

/** Converts normalized direction to the octahedral map (non-equal area, unsigned normalized).
    \param[in] n Normalized direction.
    \return Position in octahedral map in [0,1] for each component.
*/
float2 ndir_to_oct_unorm(float3 n)
{
    return ndir_to_oct_snorm(n) * 0.5f + 0.5f;
}

/** Converts point in the octahedral map to normalized direction (non-equal area, signed normalized).
    \param[in] p Position in octahedral map in [-1,1] for each component.
    \return Normalized direction.
*/
float3 oct_to_ndir_snorm(float2 p)
{
    float3 n = float3(p.xy, 1.0 - abs(p.x) - abs(p.y));
    n.xy = (n.z < 0.0) ? oct_wrap(n.xy) : n.xy;
    return normalize(n);
}

/** Converts point in the octahedral map to normalized direction (non-equal area, unsigned normalized).
    \param[in] p Position in octahedral map in [0,1] for each component.
    \return Normalized direction.
*/
float3 oct_to_ndir_unorm(float2 p)
{
    return oct_to_ndir_snorm(p * 2.f - 1.f);
}

/** Converts normalized direction to the octahedral map (equal-area, unsigned normalized).
    \param[in] n Normalized direction.
    \return Position in octahedral map in [0,1] for each component.
*/
float2 ndir_to_oct_equal_area_unorm(float3 n)
{
    // Use atan2 to avoid explicit div-by-zero check in atan(y/x).
    float r = sqrt(1.f - abs(n.z));
    float phi = atan2(abs(n.y), abs(n.x));

    // Compute p = (u,v) in the first quadrant.
    float2 p;
    p.y = r * phi * K_2_PI;
    p.x = r - p.y;

    // Reflect p over the diagonals, and move to the correct quadrant.
    if (n.z < 0.f) p = 1.f - p.yx;
    p *= sign(n.xy);

    return saturate(p * 0.5f + 0.5f);
}

/** Converts point in the octahedral map to normalized direction (equal area, unsigned normalized).
    \param[in] p Position in octahedral map in [0,1] for each component.
    \return Normalized direction.
*/
float3 oct_to_ndir_equal_area_unorm(float2 p)
{
    p = p * 2.f - 1.f;

    // Compute radius r without branching. The radius r=0 at +z (center) and at -z (corners).
    float d = 1.f - (abs(p.x) + abs(p.y));
    float r = 1.f - abs(d);

    // Compute phi in [0,pi/2] (first quadrant) and sin/cos without branching.
    // TODO: Analyze fp32 precision, do we need a small epsilon instead of 0.0 here?
    float phi = (r > 0.f) ? ((abs(p.y) - abs(p.x)) / r + 1.f) * K_PI_4 : 0.f;

    // Convert to Cartesian coordinates. Note that sign(x)=0 for x=0, but that's fine here.
    float f = r * sqrt(2.f - r*r);
    float x = f * sign(p.x) * cos(phi);
    float y = f * sign(p.y) * sin(phi);
    float z = sign(d) * (1.f - r*r);

    return float3(x, y, z);
}

/******************************************************************************

    Sampling functions

******************************************************************************/

/** Uniform sampling of the unit disk using polar coordinates.
    \param[in] u Uniform random number in [0,1)^2.
    \return Sampled point on the unit disk.
*/
float2 sample_disk(float2 u)
{
    float2 p;
    float r = sqrt(u.x);
    float phi = K_2PI * u.y;
    p.x = r * cos(phi);
    p.y = r * sin(phi);
    return p;
}

/** Uniform sampling of direction within a cone
    \param[in] u Uniform random number in [0,1)^2.
    \param[in] cosTheta Cosine of the cone half-angle
    \return Sampled direction within the cone with (0,0,1) axis
*/
float3 sample_cone(float2 u, float cosTheta)
{
    float z = u.x * (1.f - cosTheta) + cosTheta;
    float r = sqrt(1.f - z*z);
    float phi = K_2PI * u.y;
    return float3(r * cos(phi), r * sin(phi), z);
}

/** Uniform sampling of the unit sphere using spherical coordinates.
    \param[in] u Uniform random numbers in [0,1)^2.
    \return Sampled point on the unit sphere.
*/
float3 sample_sphere(float2 u)
{
    float phi = K_2PI * u.y;
    float cosTheta = 1.0f - 2.0f * u.x;
    float sinTheta = sqrt(max(0.0f, 1.0f - cosTheta * cosTheta));
    return float3(sinTheta * cos(phi), sinTheta * sin(phi), cosTheta);
}

/** Uniform sampling of the unit hemisphere using sphere sampling.
    \param[in] u Uniform random numbers in [0,1)^2.
    \return Sampled point on the unit hemisphere.
*/
float3 sample_hemisphere(float2 u)
{
    float3 w = sample_sphere(u);
    w.z = abs(w.z);
    return w;
}

/** Uniform sampling of the unit disk using Shirley's concentric mapping.
    \param[in] u Uniform random numbers in [0,1)^2.
    \return Sampled point on the unit disk.
*/
float2 sample_disk_concentric(float2 u)
{
    u = 2.f * u - 1.f;
    if (u.x == 0.f && u.y == 0.f) return u;
    float phi, r;
    if (abs(u.x) > abs(u.y))
    {
        r = u.x;
        phi = (u.y / u.x) * K_PI_4;
    }
    else
    {
        r = u.y;
        phi = K_PI_2 - (u.x / u.y) * K_PI_4;
    }
    return r * float2(cos(phi), sin(phi));
}

/** Cosine-weighted sampling of the hemisphere using Shirley's concentric mapping.
    \param[in] u Uniform random numbers in [0,1)^2.
    \param[out] pdf Probability density of the sampled direction (= cos(theta)/pi).
    \return Sampled direction in the local frame (+z axis up).
*/
float3 sample_cosine_hemisphere_concentric(float2 u, out float pdf)
{
    float2 d = sample_disk_concentric(u);
    float z = sqrt(max(0.f, 1.f - dot(d, d)));
    pdf = z * K_1_PI;
    return float3(d, z);
}

/** Cosine-weighted sampling of the hemisphere using a polar coordinates.
    \param[in] u Uniform random numbers in [0,1)^2.
    \param[out] pdf Probability density of the sampled direction (= cos(theta)/pi).
    \return Sampled direction in the local frame (+z axis up).
*/
float3 sample_cosine_hemisphere_polar(float2 u, out float pdf)
{
    float3 p;
    float r = sqrt(u.x);
    float phi = K_2PI * u.y;
    p.x = r * cos(phi);
    p.y = r * sin(phi);
    p.z = sqrt(1.f - u.x);
    pdf = p.z * K_1_PI;
    return p;
}

/** Cosine-weighted sampling of the hemisphere using a polar coordinates.
    This overload does not compute the pdf for the generated sample.
    \param[in] u Uniform random numbers in [0,1)^2.
    \return Sampled direction in the local frame (+z axis up).
*/
float3 sample_cosine_hemisphere_polar(float2 u)
{
    float pdf;
    return sample_cosine_hemisphere_polar(u, pdf);
}

/** Uniform sampling of a triangle.
    \param[in] u Uniform random numbers in [0,1)^2.
    \return Barycentric coordinates (1-u-v,u,v) of the sampled point.
*/
float3 sample_triangle(float2 u)
{
    float su = sqrt(u.x);
    float2 b = float2(1.f - su, u.y * su);
    return float3(1.f - b.x - b.y, b.x, b.y);
}

/******************************************************************************

    Motion vectors

******************************************************************************/

/** Calculate screen-space motion vector.
    \param[in] pixelCrd Sample in current frame expressed in pixel coordinates with origin in the top-left corner.
    \param[in] prevPosH Sample in previous frame expressed in homogeneous clip space coordinates. Note that the definition differs between D3D12 and Vulkan.
    \param[in] renderTargetDim Render target dimension in pixels.
    \return Motion vector pointing from current to previous position expressed in sceen space [0,1] with origin in the top-left corner.
*/
float2 calcMotionVector(float2 pixelCrd, float4 prevPosH, float2 renderTargetDim)
{
    float2 prevCrd = prevPosH.xy / prevPosH.w;
#ifdef FALCOR_FLIP_Y
    prevCrd *= float2(0.5, 0.5);
#else
    prevCrd *= float2(0.5, -0.5);
#endif
    prevCrd += 0.5f;
    float2 normalizedCrd = pixelCrd / renderTargetDim;
    return prevCrd - normalizedCrd;
}

/******************************************************************************

    Miscellaneous functions

******************************************************************************/

/** Inverts a 2x2 matrix.
*/
float2x2 inverse(float2x2 M)
{
    float2x2 inv;
    float invdet = 1.0f / determinant(M);
    inv[0][0] = M[1][1] * invdet;
    inv[1][1] = M[0][0] * invdet;
    inv[0][1] = -M[0][1] * invdet;
    inv[1][0] = -M[1][0] * invdet;
    return inv;
}

/** Inverts a 2x3 matrix.
*/
float2x3 inverse(float2x3 M)
{
    float2x2 N = float2x2(M._m00, M._m01, M._m10, M._m11);
    float2x2 Ni = inverse(N);
    float2 t = -mul(Ni, float2(M._m02, M._m12));
    float2x3 Mi = float2x3(Ni._m00, Ni._m01, t.x, Ni._m10, Ni._m11, t.y);
    return Mi;
}

/** Inverts a 3x3 matrix.
*/
float3x3 inverse(float3x3 M)
{
    float3x3 inv;
    float invdet = 1.0f / determinant(M);
    inv[0][0] = (M[1][1] * M[2][2] - M[2][1] * M[1][2]) * invdet;
    inv[0][1] = (M[0][2] * M[2][1] - M[0][1] * M[2][2]) * invdet;
    inv[0][2] = (M[0][1] * M[1][2] - M[0][2] * M[1][1]) * invdet;
    inv[1][0] = (M[1][2] * M[2][0] - M[1][0] * M[2][2]) * invdet;
    inv[1][1] = (M[0][0] * M[2][2] - M[0][2] * M[2][0]) * invdet;
    inv[1][2] = (M[1][0] * M[0][2] - M[0][0] * M[1][2]) * invdet;
    inv[2][0] = (M[1][0] * M[2][1] - M[2][0] * M[1][1]) * invdet;
    inv[2][1] = (M[2][0] * M[0][1] - M[0][0] * M[2][1]) * invdet;
    inv[2][2] = (M[0][0] * M[1][1] - M[1][0] * M[0][1]) * invdet;
    return inv;
}

/** Generate a vector that is orthogonal to the input vector.
    This can be used to invent a tangent frame for meshes that don't have real tangents/bitangents.
    \param[in] u Unit vector.
    \return v Unit vector that is orthogonal to u.
*/
float3 perp_stark(float3 u)
{
    // TODO: Validate this and look at numerical precision etc. Are there better ways to do it?
    float3 a = abs(u);
    uint uyx = (a.x - a.y) < 0 ? 1 : 0;
    uint uzx = (a.x - a.z) < 0 ? 1 : 0;
    uint uzy = (a.y - a.z) < 0 ? 1 : 0;
    uint xm = uyx & uzx;
    uint ym = (1 ^ xm) & uzy;
    uint zm = 1 ^ (xm | ym);  // 1 ^ (xm & ym)
    float3 v = normalize(cross(u, float3(xm, ym, zm)));
    return v;
}
// fp16 variant
half3 perp_stark(half3 u)
{
    // TODO: Validate this and look at numerical precision etc. Are there better ways to do it?
    half3 a = abs(u);
    uint uyx = (a.x - a.y) < 0 ? 1 : 0;
    uint uzx = (a.x - a.z) < 0 ? 1 : 0;
    uint uzy = (a.y - a.z) < 0 ? 1 : 0;
    uint xm = uyx & uzx;
    uint ym = (1 ^ xm) & uzy;
    uint zm = 1 ^ (xm | ym);  // 1 ^ (xm & ym)
    half3 v = normalize(cross(u, half3(xm, ym, zm)));
    return v;
}

float  sqr(float  v) { return v*v; }

float2 sqr(float2 v) { return v*v; }

float3 sqr(float3 v) { return v*v; }

float4 sqr(float4 v) { return v*v; }

/** Error function.
*/
float erf(float x)
{
    // From "Numerical Recipes in C, The Art of Scientific Computing"
    // (Second Edition) by Press et al. 1992. Page 221.
    // Maxiumum error: 1.2 x 10^-7.
    float t = 1.0f / (1.0f + 0.5f * abs(x));
    float p =  0.17087277f;
    p = -0.82215223f + p * t;
    p =  1.48851587f + p * t;
    p = -1.13520398f + p * t;
    p =  0.27886807f + p * t;
    p = -0.18628806f + p * t;
    p =  0.09678418f + p * t;
    p =  0.37409196f + p * t;
    p =  1.00002368f + p * t;
    p = -1.26551223f + p * t;
    float tau = t * exp(-x * x + p);
    return x >= 0.0f ? 1.0f - tau : tau - 1.0f;
}

/** Inverse error function.
*/
float erfinv(float x)
{
    // From "Approximating the erfinv function" by Mike Giles 2012.
    // Maximum error: 7 x 10^-7.
    if (x <= -1.0f)
    {
        return -1.0f / 0.0f;
    }
    else if (x >= 1.0f)
    {
        return 1.0f / 0.0f;
    }

    float w = - log((1.0f - x) * (1.0f + x));
    float p;
    if (w < 5.0f)
    {
        w = w - 2.5f;
        p =  2.81022636e-08f;
        p =  3.43273939e-07f + p * w;
        p = -3.5233877e-06f  + p * w;
        p = -4.39150654e-06f + p * w;
        p =  0.00021858087f  + p * w;
        p = -0.00125372503f  + p * w;
        p = -0.00417768164f  + p * w;
        p =  0.246640727f    + p * w;
        p =  1.50140941f     + p * w;
    }
    else
    {
        w = sqrt(w) - 3.0f;
        p = -0.000200214257f;
        p =  0.000100950558f + p * w;
        p =  0.00134934322f  + p * w;
        p = -0.00367342844f  + p * w;
        p =  0.00573950773f  + p * w;
        p = -0.0076224613f   + p * w;
        p =  0.00943887047f  + p * w;
        p =  1.00167406f     + p * w;
        p =  2.83297682f     + p * w;
    }
    return p * x;
}

/** Logarithm of the Gamma function.
*/
float lgamma(float x_)
{
    // Lanczos approximation from
    // "Numerical Recipes in C, The Art of Scientific Computing"
    // (Second Edition) by Press et al. 1992. Page 214.
    // Maxiumum error: 2 x 10^-10.
    const float g = 5.0;
    const float coeffs[7] = {
          1.000000000190015,
         76.18009172947146,
        -86.50532032941677,
         24.01409824083091,
         -1.231739572450155,
          0.1208650973866179e-2,
         -0.5395239384953e-5
    };

    bool reflect = x_ < 0.5f;
    float x = reflect ? -x_ : x_ - 1.0f;
    float base = x + g + 0.5f;

    float s = 0.0f;
    for (int i = 6; i >= 1; --i)
    {
        s += coeffs[i] / (x + (float) i);
    }
    s += coeffs[0];

    float res = ((log(sqrt(2*K_PI)) + log(s)) - base) + log(base) * (x + 0.5f);
    if (reflect)
    {
        // Apply the Gamma reflection formula (in log-space).
        res = log(abs(K_PI / sin(K_PI * x_))) - res;
    }
    return res;
}

/** Gamma function.
*/
float gamma(float x)
{
    return exp(lgamma(x));
}

/** Beta function.
*/
float beta(float x, float y)
{
    return gamma(x) * gamma(y) / gamma(x + y);
}

/** Given two vectors, will orthonormalized them, preserving v1 and v2.
    Vector v1 must be already normalized, v2 will be renormalized as part of the adjustment.
    Vector v2 is assumed to be non-zero and already somewhat orthogonal to v1, as this code does not handle v2=0 or v2=v1

    \param[in] v1 A normalized vector to orthogonalize v2 to.
    \param[in,out] v2 Vector to be adjusted to be orthonormal to v2.
*/
void orthogonalizeVectors(const float3 v1, inout float3 v2)
{
    v2 = normalize(v2 - dot(v1, v2) * v1);
}

/** Rotate a 3D vector counterclockwise around the X-axis.
    The rotation angle is positive for a rotation that is counterclockwise when looking along the x-axis towards the origin.
    \param[in] v Original vector.
    \param[in] angle Angle in radians.
    \return Transformed vector.
*/
float3 rotate_x(const float3 v, const float angle)
{
    float c, s;
    sincos(angle, s, c);
    return float3(v.x, v.y * c - v.z * s, v.y * s + v.z * c);
}

/** Rotate a 3D vector counterclockwise around the Y-axis.
    The rotation angle is positive for a rotation that is counterclockwise when looking along the y-axis towards the origin.
    \param[in] v Original vector.
    \param[in] angle Angle in radians.
    \return Transformed vector.
*/
float3 rotate_y(const float3 v, const float angle)
{
    float c, s;
    sincos(angle, s, c);
    return float3(v.x * c + v.z * s, v.y, -v.x * s + v.z * c);
}

/** Rotate a 3D vector counterclockwise around the Z-axis.
    The rotation angle is positive for a rotation that is counterclockwise when looking along the z-axis towards the origin.
    \param[in] v Original vector.
    \param[in] angle Angle in radians.
    \return Transformed vector.
*/
float3 rotate_z(const float3 v, const float angle)
{
    float c, s;
    sincos(angle, s, c);
    return float3(v.x * c - v.y * s, v.x * s + v.y * c, v.z);
}
#endif // __MATH_HELPERS_HLSLI__