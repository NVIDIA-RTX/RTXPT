/*
* Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "../Utils/Utils.hlsli"

#ifndef LIGHT_SHAPING_HLSLI
#define LIGHT_SHAPING_HLSLI

struct LightShaping
{
    float cosConeAngle;
    float3 primaryAxis;
    float cosConeSoftness;
    uint isSpot;
    int iesProfileIndex;
};

LightShaping unpackLightShaping(const PolymorphicLightInfoFull lightInfoFull)
{
    LightShaping shaping = (LightShaping)0;
    shaping.isSpot = false;

    if( lightInfoFull.Base.HasLightShaping() )
    {
        shaping.isSpot = true; //(lightInfoFull.Base.ColorTypeAndFlags & kPolymorphicLightShapingEnableBit) != 0;
        shaping.primaryAxis = OctToNDirUnorm32(lightInfoFull.Extended.PrimaryAxis);
        shaping.cosConeAngle = f16tof32(lightInfoFull.Extended.CosConeAngleAndSoftness);
        shaping.cosConeSoftness = f16tof32(lightInfoFull.Extended.CosConeAngleAndSoftness >> 16);
        shaping.iesProfileIndex = 0; // shaping.iesProfileIndex = (lightInfoFull.Base.ColorTypeAndFlags & kPolymorphicLightIesProfileEnableBit) ? lightInfoFull.Extended.IesProfileIndex : -1;
    }
    
    return shaping;
}

float evaluateIesProfile(int profileIndex, float3 emissionDirection_, float3 lightPrimaryAxis)
{
#if 0   // currently disabled until we implement scene side code and find a scene with appropriate test cases 
    if (profileIndex < 0)
        return 1.0;

    float3 xAxis;
    float3 yAxis;
    BranchlessONB(lightPrimaryAxis, xAxis, yAxis);

    float3 emissionDirection;
    emissionDirection.x = dot(emissionDirection_, xAxis);
    emissionDirection.y = dot(emissionDirection_, yAxis);
    emissionDirection.z = dot(emissionDirection_, lightPrimaryAxis);
    emissionDirection = normalize(emissionDirection);

    const float angle = acos(emissionDirection.z);
    const float normAngle = angle / c_pi;

    const float tangentAngle = atan2(emissionDirection.y, emissionDirection.x);
    const float normTangentAngle = tangentAngle * .5f / c_pi + .5f;

    Texture2D<float4> iesProfileTexture = t_BindlessTextures[NonUniformResourceIndex(profileIndex)];

    float iesMultiplier = iesProfileTexture.SampleLevel(IES_SAMPLER, float2(normAngle, normTangentAngle), 0).x;

    return iesMultiplier;
#else
    return 1.0;
#endif
}

float3 evaluateLightShaping(LightShaping shaping, float3 surfacePosition, float3 lightSamplePosition)
{
    if (!shaping.isSpot)
        return 1.0;

    const float3 lightToSurface = normalize(surfacePosition - lightSamplePosition);

    const float cosTheta = dot(shaping.primaryAxis, lightToSurface);

    const float softSpotlight = smoothstep(shaping.cosConeAngle, 
        shaping.cosConeAngle + shaping.cosConeSoftness, cosTheta);

    if (softSpotlight <= 0)
        return 0.0;

    const float iesMultiplier = evaluateIesProfile(shaping.iesProfileIndex,
        lightToSurface, shaping.primaryAxis);

    return softSpotlight * iesMultiplier;
}

// Computes the conservative cone of influence for a spherical light source that has a shaping angle.
// The cone angle and axis are the same as the shaping angle and axis, and the cone vertex is the
// light center offset against the axis by the distance that is necessary to inscribe the sphere into the cone.
// Assumes nonzero cone angle.
float3 getConeVertexForSphericalSource(float3 sphereCenter, float sphereRadius, float3 coneAxis, float coneHalfAngle)
{
    // Compute the sine of the clamped half angle. When the angle is more than 90 degrees (half a pi),
    // the offset should be exactly one sphere radius.
    float sinHalfAngle = sin(min(coneHalfAngle, K_PI * 0.5));
    
    // Offset is the hypotenuse of a right triangle whose vertices are: the light center; the cone vertex; 
    // and any point on the circle where the cone touches the sphere.
    float offset = sphereRadius / sinHalfAngle;

    // Compute the cone vertex assuming that the aforementioned hypotenuse is collinear with the cone axis.
    float3 coneVertex = sphereCenter - coneAxis * offset;

    return coneVertex;
}

// Tests whether a sphere intersects with a cone. Returns true if they do intersect.
bool testSphereConeIntersection(float3 coneVertex, float3 coneAxis, float coneHalfAngle, float3 sphereCenter, float sphereRadius)
{
    // The intersection is determined by comparing three angles in the plane that goes through the cone axis
    // and the sphere center. The geometry and solution should be clear from the variable names.

    float3 coneVertexToSphereCenter = sphereCenter - coneVertex;
    float distanceFromConeVertexToSphereCenter = length(coneVertexToSphereCenter);

    if (distanceFromConeVertexToSphereCenter <= sphereRadius)
        return true;

    float invDistanceFromConeVertexToSphereCenter = rcp(distanceFromConeVertexToSphereCenter);

    float angleBetweenConeAxisAndVertexToSphere = acos(dot(coneVertexToSphereCenter, coneAxis) * invDistanceFromConeVertexToSphereCenter);

    float halfAngleSubtendedBySphereAtVertex = asin(sphereRadius * invDistanceFromConeVertexToSphereCenter);

    bool intersection = angleBetweenConeAxisAndVertexToSphere <= halfAngleSubtendedBySphereAtVertex + coneHalfAngle;

    return intersection;
}

// Tests whether a sphere intersects with the cone of influence of a shaped spherical light.
// Returns true if they do intersect. Also returns true if the light is not shaped.
// The test is a bit conservative, i.e. some volume behind the light will be considered intersecting.
// That volume is coming from the conservative cone of influence, and it gets larger for wide lights
// with a small shaping angle because the cone goes further back to include the light sphere.
bool testSphereIntersectionForShapedLight(float3 lightCenter, float lightRadius, LightShaping shaping, float3 sphereCenter, float sphereRadius)
{
    if (!shaping.isSpot)
        return true;

    // Recover the angle from the cosine because the further algorithms operate with raw angles or sines.
    float coneHalfAngle = acos(shaping.cosConeAngle);

    // Compute the conservative cone of influence.
    float3 coneVertex = getConeVertexForSphericalSource(lightCenter, lightRadius, shaping.primaryAxis, coneHalfAngle);

    // Test the intersection of the given sphere with the cone of influence.
    return testSphereConeIntersection(coneVertex, shaping.primaryAxis, coneHalfAngle, sphereCenter, sphereRadius);
}

// Returns the approximate ratio of the flux of a shaped sphere light and onmidirectional sphere light.
float getShapingFluxFactor(LightShaping shaping)
{
    if (!shaping.isSpot)
        return 1.0;

    float solidAngleOverTwoPi = (1.0 - shaping.cosConeAngle);

    // TODO: this is pulled out of thin air, need a more grounded expression
    solidAngleOverTwoPi *= lerp(1.0, 0.5, shaping.cosConeSoftness);

    // TODO: account for IES profiles

    return solidAngleOverTwoPi * 0.5; // (solidAngle / (4 * c_pi))
}

#endif // LIGHT_SHAPING_HLSLI