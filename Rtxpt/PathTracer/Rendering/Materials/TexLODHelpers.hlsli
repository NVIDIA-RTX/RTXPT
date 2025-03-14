/*
* Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#ifndef __TEX_LOD_HELPERS_HLSLI__ // using instead of "#pragma once" due to https://github.com/microsoft/DirectXShaderCompiler/issues/3943
#define __TEX_LOD_HELPERS_HLSLI__

/** Helper functions for the texture level-of-detail (LOD) system.

    Supports texture LOD both for ray differentials (Igehy, SIGGRAPH 1999) and a method based on ray cones,
    described in
    * "Strategies for Texture Level-of-Detail for Real-Time Ray Tracing," by Tomas Akenine-Moller et al., Ray Tracing Gems, 2019,
    * "Improved Shader and Texture Level-of-Detail using Ray Cones" by Akenine-Moller et al., Journal of Graphics Tools, 2021,
    * "Refraction Ray Cones for Texture Level of Detail" by Boksansky et al., to appear in Ray Tracing Gems II, 2021.

    Note that the actual texture lookups are baked into the TextureSampler interfaces.

    See WhittedRayTracer.* for an example using these functions.
*/

#include "../../Config.h"
#include "../../Utils/Math/MathConstants.hlsli"
#include "../../Utils/Utils.hlsli"
//import Scene.SceneTypes; // Needed for ray bounce helpers.

// Modes for calculating spread angle from curvature
#define TEXLOD_SPREADANGLE_RTG1                     0     // 0: Original approach derived from RTG.
#define TEXLOD_SPREADANGLE_ARC_LENGTH_UNOPTIMIZED   1     // 1: New arc-length integration approach, unoptimized.
#define TEXLOD_SPREADANGLE_ARC_LENGTH_OPTIMIZED     2     // 2: New arc-length integration approach, optimized.

// Chose one of modes above, default to optimized arc length approach (2)
#define TEXLOD_SPREADANGLE_FROM_CURVATURE_MODE TEXLOD_SPREADANGLE_ARC_LENGTH_OPTIMIZED

// Uncomment to use FP16 for ray cone payload
#define USE_RAYCONES_WITH_FP16_IN_RAYPAYLOAD

// log2 function clamped to a valid domain. The range of lod values that can be returned is [-126, 127].
float SafeLog2(float x) { return log2(clamp(x, FLT_MIN, FLT_MAX)); }

// ----------------------------------------------------------------------------
// Ray cone helpers
// ----------------------------------------------------------------------------

/** Describes a ray cone for texture level-of-detail.

    Representing a ray cone based on width and spread angle. Has both FP32 and FP16 support.
    Use #define USE_RAYCONES_WITH_FP16_IN_RAYPAYLOAD to use FP16

    Note: spread angle is the whole (not half) cone angle! See https://research.nvidia.com/publication/2021-08_refraction-ray-cones-texture-level-detail
*/
struct RayCone
{
#ifndef USE_RAYCONES_WITH_FP16_IN_RAYPAYLOAD
    float width;
    float spreadAngle;
    float getWidth()            { return width; }
    float getSpreadAngle()      { return spreadAngle; }
#else
    uint widthSpreadAngleFP16;
    float getWidth()            { return f16tof32(widthSpreadAngleFP16 >> 16); }
    float getSpreadAngle()      { return f16tof32(widthSpreadAngleFP16); }
#endif

    /** Initializes a ray cone struct.
        \param[in] width The width of the ray cone.
        \param[in] angle The angle of the ray cone.
    */
    void __init(float width, float angle)
    {
#ifndef USE_RAYCONES_WITH_FP16_IN_RAYPAYLOAD
        this.width = width;
        this.spreadAngle = angle;
#else
        this.widthSpreadAngleFP16 = (f32tof16(width) << 16) | f32tof16(angle);
#endif
    }
    static RayCone make(float width, float angle) { RayCone ret; ret.__init(width, angle); return ret; }

    /** Propagate the raycone to the next hit point (hitT distance away).
        \param[in] hitT Distance to the hit point.
        \return The propagated ray cone.
    */
    RayCone propagateDistance(float hitT)
    {
        float angle = getSpreadAngle();
        float width = getWidth();
        return RayCone::make(angle * hitT + width, angle);
    }

    /** Add surface spread angle to the current RayCone and returns the updated RayCone.
        \param[in] surfaceSpreadAngle Angle to be added.
        \return The updated ray cone.
    */
    RayCone addToSpreadAngle(float surfaceSpreadAngle)
    {
        float angle = getSpreadAngle();
        return RayCone::make(getWidth(), angle + surfaceSpreadAngle);
    }

    /** Compute texture level of details based on ray cone. Commented out, since we handle texture resolution as part of the texture lookup in Falcor.
        Keeping this here for now, since other may find it easier to understand.
        Note: call propagateDistance() before computeLOD()
    */
    float computeLOD(float triLODConstant, float3 rayDir, float3 normal, float textureWidth, float textureHeight, uniform bool moreDetailOnSlopes = false)
    {
        float lambda = triLODConstant;                                  // Constant per triangle.
        float filterWidth = getWidth();
        float distTerm = abs(filterWidth);
        float normalTerm = abs(dot(rayDir, normal));
        if( moreDetailOnSlopes ) normalTerm = sqrt( normalTerm );
        lambda += 0.5f * SafeLog2(textureWidth * textureHeight * distTerm / normalTerm);
        return lambda;
    }

    /** Compute texture level of details based on ray cone.
        Note that this versions excludes texture dimension dependency, which is instead added back in
        using the ExplicitRayConesLodTextureSampler:ITextureSampler in order to support baseColor, specular, etc per surfaces.
        \param[in] triLODConstant Value computed by computeRayConeTriangleLODValue().
        \param[in] rayDir Ray direction.
        \param[in] normal Normal at the hit point.
        \return The level of detail, lambda.
    */
    float computeLOD(float triLODConstant, float3 rayDir, float3 normal, uniform bool moreDetailOnSlopes = false)     // Note: call propagateDistance() before computeLOD()
    {
        float lambda = triLODConstant; // constant per triangle
        float filterWidth = getWidth();
        float distTerm = abs(filterWidth);
        float normalTerm = abs(dot(rayDir, normal));
        if( moreDetailOnSlopes ) normalTerm = sqrt( normalTerm );
        lambda += SafeLog2(distTerm / normalTerm);
        return lambda;
    }
};

/** Compute the triangle LOD value based on triangle vertices and texture coordinates, used by ray cones.
    \param[in] vertices Triangle vertices.
    \param[in] txcoords Texture coordinates at triangle vertices.
    \param[in] worldMat 3x3 world matrix.
    \return Triangle LOD value.
*/
float computeRayConeTriangleLODValue(float3 vertices[3], float2 txcoords[3], float3x3 worldMat)
{
    float2 tx10 = txcoords[1] - txcoords[0];
    float2 tx20 = txcoords[2] - txcoords[0];
    float Ta = abs(tx10.x * tx20.y - tx20.x * tx10.y);

    // We need the area of the triangle, which is length(triangleNormal) in worldspace, and
    // could not figure out a way with fewer than two 3x3 mtx multiplies for ray cones.
    float3 edge01 = mul(vertices[1] - vertices[0], worldMat);
    float3 edge02 = mul(vertices[2] - vertices[0], worldMat);

    float3 triangleNormal = cross(edge01, edge02);              // In world space, by design.
    float Pa = length(triangleNormal);                          // Twice the area of the triangle.
    return 0.5f * SafeLog2(Ta / Pa);                            // Value used by texture LOD cones model. 
}

/** Compute screen space spread angle at the first hit point based on ddx and ddy of normal and position.
    \param[in] positionW Position of the hit point in world space.
    \param[in] normalW Normal of the hit point in world space.
    \return Spread angle at hit point.
*/
float computeScreenSpaceSurfaceSpreadAngle(float3 positionW, float3 normalW)
{
    float3 dNdx = ddx(normalW);
    float3 dNdy = ddy(normalW);
    float3 dPdx = ddx(positionW);
    float3 dPdy = ddy(positionW);

    float beta = sqrt(dot(dNdx, dNdx) + dot(dNdy, dNdy)) * sign(dot(dNdx, dPdx) + dot(dNdy, dPdy));
    return beta;
}

/** Compute screen space spread angle at the first hit point based on ddx and ddy of normal and position.
    \param[in] rightVector The difference vector between normalized eye ray direction at (x + 1, y) and (x, y).
    \param[in] cameraUpVector The difference vector between normalized eye ray direction at (x, y + 1) and (x, y).
    \param[in] dNdx Differential normal in the x-direction.
    \param[in] dNdy Differential normal in the y-direction.
    \return Spread angle at hit point.
*/
float computeScreenSpaceSurfaceSpreadAngle(float3 rightVector, float3 upVector, float3 dNdx, float3 dNdy)
{
    float betaX = atan(length(dNdx));
    float betaY = atan(length(dNdy));
    float betaCurvature = sqrt(betaX * betaX + betaY * betaY) * (betaX >= betaY ? sign(dot(rightVector, dNdx)) : sign(dot(upVector, dNdy)));
    return betaCurvature;
}

/** Compute spread from estimated curvature from a triangle for ray cones.
    \param[in] curvature Curvature value.
    \param[in] rayConeWidth The width of the ray cone.
    \param[in] rayDir The ray direction.
    \param[in] normal The normal.
    \return Spread angle.
*/
float computeSpreadAngleFromCurvatureIso(float curvature, float rayConeWidth, float3 rayDir, float3 normal)
{
    float dn = -dot(rayDir, normal);
    dn = abs(dn) < 1.0e-5 ? sign(dn) * 1.0e-5 : dn;

#if TEXLOD_SPREADANGLE_FROM_CURVATURE_MODE == TEXLOD_SPREADANGLE_RTG1
    // Original approach.
    float s = sign(curvature);
    float curvatureScaled = curvature * rayConeWidth * 0.5 / dn;
    float surfaceSpreadAngle = 2.0 * atan(abs(curvatureScaled) / sqrt(2.0)) * s;
#elif TEXLOD_SPREADANGLE_FROM_CURVATURE_MODE == TEXLOD_SPREADANGLE_ARC_LENGTH_UNOPTIMIZED
    // New approach, unoptimized: https://www.math24.net/curvature-plane-curves/

    float r = 1.0 / (curvature);
    float chord = (rayConeWidth) / (dn);
    float arcLength = asin(chord / (2.0 * r)) * (2.0 * r);
    float deltaPhi = (curvature) * (arcLength);

    float surfaceSpreadAngle = deltaPhi;
#else // TEXLOD_SPREADANGLE_FROM_CURVATURE_MODE == TEXLOD_SPREADANGLE_ARC_LENGTH_OPTIMIZED
    // New approach : Fast Approximation.
    float deltaPhi = (curvature * rayConeWidth / dn);
    float surfaceSpreadAngle = deltaPhi;
#endif

    return surfaceSpreadAngle;
}

/** Exploit ray cone to compute an approximate anisotropic filter. The idea is to find the width (2*radius) of the ray cone at
    the intersection point, and approximate the ray cone as a cylinder at that point with that radius. Then intersect the
    cylinder with the triangle plane to find the ellipse of anisotropy. Finally, convert to gradients in texture coordinates.
    \param[in] intersectionPoint The intersection point.
    \param[in] faceNormal The normal of the triangle.
    \param[in] rayConeDir Direction of the ray cone.
    \param[in] rayConeWidthAtIntersection Width of the cone at the intersection point (use: raycone.getWidth()).
    \param[in] positions Positions of the triangle.
    \param[in] txcoords Texture coordinates of the vertices of the triangle.
    \param[in] interpolatedTexCoordsAtIntersection Interpolated texture coordinates at the intersection point.
    \param[in] texGradientX First gradient of texture coordinates, which can be fed into SampleGrad().
    \param[in] texGradientY Second gradient of texture coordinates, which can be fed into SampleGrad().
*/
void computeAnisotropicEllipseAxes(float3 intersectionPoint, float3 faceNormal, float3 rayConeDir,
    float rayConeRadiusAtIntersection, float3 positions[3], float2 txcoords[3], float2 interpolatedTexCoordsAtIntersection,
    out float2 texGradientX, out float2 texGradientY)
{
    // Compute ellipse axes.
    float3 ellipseAxis0 = rayConeDir - dot(faceNormal, rayConeDir) * faceNormal;                // Project rayConeDir onto the plane.
    float3 rayDirPlaneProjection0 = ellipseAxis0 - dot(rayConeDir, ellipseAxis0) * rayConeDir;  // Project axis onto the plane defined by the ray cone dir.
    ellipseAxis0 *= rayConeRadiusAtIntersection / max(0.0001f, length(rayDirPlaneProjection0)); // Using uniform triangles to find the scale.

    float3 ellipseAxis1 = cross(faceNormal, ellipseAxis0);
    float3 rayDirPlaneProjection1 = ellipseAxis1 - dot(rayConeDir, ellipseAxis1) * rayConeDir;
    ellipseAxis1 *= rayConeRadiusAtIntersection / max(0.0001f, length(rayDirPlaneProjection1));

    // Compute texture coordinate gradients.
    float3 edgeP;
    float u, v, Atriangle, Au, Av;
    float3 d = intersectionPoint - positions[0];
    float3 edge01 = positions[1] - positions[0];
    float3 edge02 = positions[2] - positions[0];
    float oneOverAreaTriangle = 1.0f / dot(faceNormal, cross(edge01, edge02));

    // Compute barycentrics.
    edgeP = d + ellipseAxis0;
    u = dot(faceNormal, cross(edgeP, edge02)) * oneOverAreaTriangle;
    v = dot(faceNormal, cross(edge01, edgeP)) * oneOverAreaTriangle;
    texGradientX = (1.0f - u - v) * txcoords[0] + u * txcoords[1] + v * txcoords[2] - interpolatedTexCoordsAtIntersection;

    edgeP = d + ellipseAxis1;
    u = dot(faceNormal, cross(edgeP, edge02)) * oneOverAreaTriangle;
    v = dot(faceNormal, cross(edge01, edgeP)) * oneOverAreaTriangle;
    texGradientY = (1.0f - u - v) * txcoords[0] + u * txcoords[1] + v * txcoords[2] - interpolatedTexCoordsAtIntersection;
}

/** Refracts a ray and handles total internal reflection (TIR) in 3D.
    \param[in] rayDir The ray direction to be refracted.
    \param[in] normal The normal at the hit point.
    \param[in] eta The raio of indices of refraction (entering / exiting).
    \param[out] refractedRayDir The refracted vector.
    \return Returns false if total internal reflection occured, otherwise true.
*/
bool refractWithTIR(float3 rayDir, float3 normal, float eta, out float3 refractedRayDir)
{
    float NdotD = dot(normal, rayDir);
    float k = 1.0f - eta * eta * (1.0f - NdotD * NdotD);
    if (k < 0.0f)
    {
        refractedRayDir = float3(0.0, 0.0, 0.0);
        return false;
    }
    else
    {
        refractedRayDir = rayDir * eta - normal * (eta * NdotD + sqrt(k));
        return true;
    }
}

/** Refracts a ray and handles total internal reflection (TIR) in 2D.
    \param[in] rayDir The ray direction to be refracted.
    \param[in] normal The normal at the hit point.
    \param[in] eta The raio of indices of refraction (entering / exiting).
    \param[out] refractedRayDir The refracted vector.
    \return Returns false if total internal reflection occured, otherwise true.
*/
bool refractWithTIR(float2 rayDir, float2 normal, float eta, out float2 refractedRayDir)
{
    float NdotD = dot(normal, rayDir);
    float k = 1.0f - eta * eta * (1.0f - NdotD * NdotD);
    if (k < 0.0f)
    {
        refractedRayDir = float2(0.0,0.0);
        return false;
    }
    else
    {
        refractedRayDir = rayDir * eta - normal * (eta * NdotD + sqrt(k));
        return true;
    }
}

/** Helper function rotate a vector by both +angle and -angle.
    \param[in] vec A vector to be rotated.
    \param[in] angle The angle used for rotation.
    \param[out] rotatedVecPlus The in vector rotated by +angle.
    \param[out] rotatedVecMinus The in vector rotated by -angle.
*/
void rotate2DPlusMinus(float2 vec, float angle, out float2 rotatedVecPlus, out float2 rotatedVecMinus)
{
    float c = cos(angle);
    float s = sin(angle);
    float cx = c * vec.x;
    float sy = s * vec.y;
    float sx = s * vec.x;
    float cy = c * vec.y;
    rotatedVecPlus =  float2(cx - sy, +sx + cy);    // Rotate +angle,
    rotatedVecMinus = float2(cx + sy, -sx + cy);    // Rotate -angle.
}

/** Helper function that returns an orthogonal vector to the in vector: 90 degrees counter-clockwise rotation.
    \param[in] vec A vector to be rotate 90 degrees counter-clockwise.
    \return The in vector rotated 90 degrees counter-clockwise.
*/
float2 orthogonal(float2 vec)
{
    return float2(-vec.y, vec.x);
}

/** Computes RayCone for a given refracted ray direction. Note that the incident ray cone should be called with propagateDistance(hitT); before computeRayConeForRefraction() is called.
    \param[in,out] rayCone A ray cone to be refracted, result is returned here as well.
    \param[in] rayOrg Ray origin.
    \param[in] rayDir Ray direction.
    \param[in] hitPoint The hit point.
    \param[in] normal The normal at the hit point.
    \param[in] normalSpreadAngle The spread angle at the normal at the hit point.
    \param[in] eta Ratio of indices of refraction (enteringIndexOfRefraction / exitingIndexOfRefraction).
    \param[in] refractedRayDir The refracted ray direction.
*/
void computeRayConeForRefraction(inout RayCone rayCone, float3 rayOrg, float3 rayDir, float3 hitPoint, float3 normal, float normalSpreadAngle,
    float eta, float3 refractedRayDir)
{
    // We have refractedRayDir, which is the direction of the refracted ray cone,
    // but we also need the rayCone.width and the rayCone.spreadAngle. These are computed in 2D,
    // with xAxis and yAxis as the 3D axes. hitPoint is the origin of this 2D coordinate system.
    float3 xAxis = normalize(rayDir - normal * dot(normal, rayDir));
    float3 yAxis = normal;

    float2 refractedDir2D = float2(dot(refractedRayDir, xAxis), dot(refractedRayDir, yAxis));           // Project to 2D.
    float2 incidentDir2D = float2(dot(rayDir, xAxis), dot(rayDir, yAxis));                              // Project to 2D.
    float2 incidentDir2D_u, incidentDir2D_l;                                                            // Upper (_u) and lower (_l) line of ray cone in 2D.
    float2 incidentDirOrtho2D = orthogonal(incidentDir2D);

    float widthSign = rayCone.getWidth() > 0.0f ? 1.0f : -1.0f;

    rotate2DPlusMinus(incidentDir2D, rayCone.getSpreadAngle() * widthSign * 0.5f, incidentDir2D_u, incidentDir2D_l);

    // Note: since we assume that the incident ray cone has been propagated to the hitpoint, we start the width-vector
    // from the origin (0,0), and so, we do not need to add rayOrigin2D to tu and tl.
    float2 tu = +incidentDirOrtho2D * rayCone.getWidth() * 0.5f;                               // Top, upper point on the incoming ray cone (in 2D).
    float2 tl = -tu;                                                                           // Top, lower point on the incoming ray cone (in 2D).
    // Intersect 2D rays (tu + t * incidentDir2D_u, and similar for _l) with y = 0.
    // Optimized becuase y will always be 0.0f, so only need to compute x.
    float hitPoint_u_x = tu.x + incidentDir2D_u.x * (-tu.y / incidentDir2D_u.y);
    float hitPoint_l_x = tl.x + incidentDir2D_l.x * (-tl.y / incidentDir2D_l.y);

    float normalSign = hitPoint_u_x > hitPoint_l_x ? +1.0f : -1.0f;

    float2 normal2D = float2(0.0f, 1.0f);
    float2 normal2D_u, normal2D_l;

    rotate2DPlusMinus(normal2D, -normalSpreadAngle * normalSign * 0.5f, normal2D_u, normal2D_l);

    // Refract in 2D.
    float2 refractedDir2D_u, refractedDir2D_l;
    if (!refractWithTIR(incidentDir2D_u, normal2D_u, eta, refractedDir2D_u))
    {
        refractedDir2D_u = incidentDir2D_u - normal2D_u * dot(normal2D_u, incidentDir2D_u);
        refractedDir2D_u = normalize(refractedDir2D_u);
    }
    if (!refractWithTIR(incidentDir2D_l, normal2D_l, eta, refractedDir2D_l))
    {
        refractedDir2D_l = incidentDir2D_l - normal2D_l * dot(normal2D_l, incidentDir2D_l);
        refractedDir2D_l = normalize(refractedDir2D_l);
    }

    float signA = (refractedDir2D_u.x * refractedDir2D_l.y - refractedDir2D_u.y * refractedDir2D_l.x) * normalSign < 0.0f ? +1.0f : -1.0f;
    float spreadAngle = acos(dot(refractedDir2D_u, refractedDir2D_l)) * signA;

    // Now compute the width of the refracted cone.
    float2 refractDirOrtho2D = orthogonal(refractedDir2D);

    // Intersect line (0,0) + t * refractDirOrtho2D with the line: hitPoint_u + s * refractedDir2D_u, but optimized since hitPoint_ul.y=0.
    float width = (-hitPoint_u_x * refractedDir2D_u.y) / dot(refractDirOrtho2D, orthogonal(refractedDir2D_u));
    // Intersect line (0,0) + t * refractDirOrtho2D with the line: hitPoint_l + s * refractedDir2D_l.
    width += (hitPoint_l_x * refractedDir2D_l.y) / dot(refractDirOrtho2D, orthogonal(refractedDir2D_l));

    rayCone = RayCone::make(width, spreadAngle);
}

/** Refracts a ray cone. Note that teh incident ray cone should be called with propagate(0.0f, hitT); before refractRayCone() is called.
    \param[in,out] rayCone A ray cone to be refracted, result is returned here as well.
    \param[in] rayOrg Ray origin.
    \param[in] rayDir Ray direction.
    \param[in] hitPoint The hit point.
    \param[in] normal The normal at the hit point.
    \param[in] normalSpreadAngle The spread angle at the normal at the hit point.
    \param[in] eta Ratio of indices of refraction (enteringIndexOfRefraction / exitingIndexOfRefraction).
    \param[out] refractedRayDir The refracted ray direction (unless the ray was totally internally reflcted (TIR:ed).
    \return Whether the ray was not totally internally reflected, i.e., returns true without TIR, and false in cases of TIR
*/
bool refractRayCone(inout RayCone rayCone, float3 rayOrg, float3 rayDir, float3 hitPoint, float3 normal, float normalSpreadAngle,
    float eta, out float3 refractedRayDir)
{
    if (!refractWithTIR(rayDir, normal, eta, refractedRayDir))
    {
        return false;               // total internal reflection
    }

    computeRayConeForRefraction(rayCone, rayOrg, rayDir, hitPoint, normal, normalSpreadAngle, eta, refractedRayDir);

    return true;
}

// ----------------------------------------------------------------------------
// Environment map sampling helpers
// ----------------------------------------------------------------------------

/** Compute the LOD used when performing a lookup in an environment map when using ray cones
    and under the assumption of using a longitude-latitude environment map. See Chapter 21 in Ray Tracing Gems 1.
    \param[in] spreadAngle The spread angle of the ray cone.
    \param[in] environmentMap The environment map.
    \return The level of detail, lambda.
*/
float computeEnvironmentMapLOD(float spreadAngle, Texture2D environmentMap)
{
    uint txw, txh;
    environmentMap.GetDimensions(txw, txh);
    return SafeLog2(abs(spreadAngle) * txh * K_1_PI);                           // From chapter 21 in Ray Tracing Gems.
}

/** Compute the LOD used when performing a lookup in an environment map when using ray differentials
    and under the assumption of using a longitude-latitude environment map. See Chapter 21 in Ray Tracing Gems 1.
    \param[in] spreadAngle The spread angle of the ray cone.
    \param[in] environmentMap The environment map.
    \return The level of detail, lambda.
*/
float computeEnvironmentMapLOD(float3 dDdx, float3 dDdy, Texture2D environmentMap)
{
    uint txw, txh;
    environmentMap.GetDimensions(txw, txh);
    return SafeLog2(length(dDdx + dDdy) * txh * K_1_PI);                        // From chapter 21 in Ray Tracing Gems.
}

#endif // __TEX_LOD_HELPERS_HLSLI__
