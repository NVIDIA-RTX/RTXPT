/*
* Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#ifndef __POLYMORPHIC_LIGHT_HLSLI__
#define __POLYMORPHIC_LIGHT_HLSLI__

#include "LightShaping.hlsli"
#include "../Utils/Geometry.hlsli"
#include "../PathTracerHelpers.hlsli"

#include "PolymorphicLight.h"

#ifndef POLYLIGHT_CONFIGURED
#error Lighting system requires compile time configuration - see LightingConfig.h (used for all core path tracing) for details; do not enable lights if unused to reduce cost of switch statements and register pressure and similar.
#endif

#define FLT_EPSILON_MINI                (2e-9f)
#define MAX_SOLID_ANGLE_PDF             (1e10f)

#if (!defined(POLYLIGHT_SPHERE_ENABLE) || !defined(POLYLIGHT_POINT_ENABLE) || !defined(POLYLIGHT_TRIANGLE_ENABLE) || !defined(POLYLIGHT_DIRECTIONAL_ENABLE) || !defined(POLYLIGHT_ENV_ENABLE) || !defined(POLYLIGHT_LIGHT_CLUSTER))
#error Each light type must be explicitly enabled or disabled
#endif

struct PolymorphicLightSample
{
    float3    Position;
    float3    Normal;
    float3    Radiance;
    float     SolidAnglePdf;
    bool      LightSampleableByBSDF;    // 'true' if there's a geometric counterpart that the path can sample; 'false' if not "visible" in the scene except as indirect diffuse (not physically correct)
};

struct PolymorphicLight
{
    static PolymorphicLightSample   CalcSample(in const PolymorphicLightInfoFull lightInfo, in const float2 random, in const float3 viewerPosition);

    // This will calculate solid angle for MIS between BSDF and light sampling. Since BSDF cannot sample a non-physical light, this will return 0 for everything except emissive triangles and environment lights.
    // In case of emissive triangles and environment lighting, the returned value is the same as returned by CalcSample in PolymorphicLightSample::SolidAnglePdf; otherwise it is different (zero).
    static float                    CalcSolidAnglePdfForMIS(in const float3 viewerPosition, in const float3 lightSamplePosition); // Note: not implemented as the caller should know the light type and avoid the switch

    // TODO: rename GetPower to getWeight so it's also applicable to infinitely distant light sources, with a scale factor?
    static float GetPower( in const PolymorphicLightInfoFull lightInfo );
    static float GetWeightForVolume( in const PolymorphicLightInfoFull lightInfo, in const float3 volumeCenter, in const float volumeRadius );

    static PolymorphicLightType     DecodeType(PolymorphicLightInfoFull lightInfo);
    static PolymorphicLightType     DecodeType(PolymorphicLightInfo lightInfo);
    static float                    UnpackRadiance(uint logRadiance);
    static float3                   UnpackColor(PolymorphicLightInfo lightInfo);
    static void                     PackColor(float3 radiance, inout PolymorphicLightInfo lightInfo);
    static bool                     PackCompactInfo(PolymorphicLightInfoFull lightInfo, out uint4 res1, out uint4 res2);
    static PolymorphicLightInfoFull UnpackCompactInfo(const uint4 data1, const uint4 data2);

};

#define VOLUME_SAMPLE_MODE_AVERAGE2 0
#define VOLUME_SAMPLE_MODE_CLOSEST2 1

#define VOLUME_SAMPLE_MODE2 VOLUME_SAMPLE_MODE_AVERAGE2

// Computes estimated distance between a given point in space and a random point inside
// a spherical volume. Since the geometry of this solution is spherically symmetric,
// only the distance from the volume center to the point and the volume radius matter here.
float getAverageDistanceToVolume(float distanceToCenter, float volumeRadius)
{
    // The expression and factor are fitted to a Monte Carlo estimated curve.
    // At distanceToCenter == 0, this function returns (0.75 * volumeRadius) which is analytically accurate.
    // At infinity, the result asymptotically approaches distanceToCenter.
    const float nonlinearFactor = 1.1547;

    float distance = distanceToCenter + volumeRadius * sq(volumeRadius)
        / sq(distanceToCenter + volumeRadius * nonlinearFactor);

#if VOLUME_SAMPLE_MODE2 == VOLUME_SAMPLE_MODE_CLOSEST2
    // if we're outside the volume, find the closest point
    if (distanceToCenter > volumeRadius)
    {
        distance = distanceToCenter - volumeRadius;
    }
#endif
    return distance;
}

#if POLYLIGHT_SPHERE_ENABLE || defined(__INTELLISENSE__)
// Note: Sphere lights always assume an interaction point is not going to be inside of the sphere, so special logic handling this case
// can be avoided in sampling logic (for PDF/radiance calculation), as well as individual PDF calculation and radiance evaluation.
struct SphereLight
{
    float3 position;
    float radius; // Note: Assumed to always be >0 to avoid point light special cases
    float3 radiance;
    LightShaping shaping;

    // Interface methods

    static float3 sphericalDirection(float sinTheta, float cosTheta, float sinPhi, float cosPhi, float3 x, float3 y, float3 z)
    {
        return sinTheta * cosPhi * x + sinTheta * sinPhi * y + cosTheta * z;
    }

    PolymorphicLightSample CalcSample(in const float2 random, in const float3 viewerPosition)
    {
        const float3 lightVector = position - viewerPosition;
        const float lightDistance2 = dot(lightVector, lightVector);
        const float radius2 = sq(radius);

        // Note: Sampling based on PBRT's solid angle sphere sampling, resulting in fewer rays occluded by the light itself,
        // ignoring special case for when viewing inside the light (which should just use normal spherical area sampling)
        // for performance. Similarly condition ignored in PDF calculation as well.

        // Note 2: In RTXPT we only emit light from the surface of the sphere - this makes it consistent the way emissive 
        // triangles work (they are single sided), and when using analytic light proxies
#if 1
        if (lightDistance2 < radius2)
        {
            PolymorphicLightSample lightSample;
            lightSample.Position = position;
            lightSample.Normal   = float3(0.0, 0.0, 0.0); // -lightVector/lightDistance;
            lightSample.Radiance = float3(0.0, 0.0, 0.0);
            lightSample.SolidAnglePdf = 1.0;
            lightSample.LightSampleableByBSDF = false;
            return lightSample;
        }
#endif

        const float lightDistance = sqrt(lightDistance2);

        // Compute theta and phi for cone sampling

        const float2 u = random;
        const float sinThetaMax2 = radius2 / lightDistance2;
        const float cosThetaMax = sqrt(max(0.0f, 1.0f - sinThetaMax2));
        const float phi = 2.0f * K_PI * u.x;
        const float cosTheta = lerp(cosThetaMax, 1.0f, u.y);
        const float sinTheta = sqrt(max(0.0f, 1.0f - sq(cosTheta)));
        const float sinTheta2 = sinTheta * sinTheta;

        // Calculate the alpha value representing the spherical coordinates of the sample point

        const float cLIGHT_SAMPING_EPSILON = 1e-10;
        const float dc = lightDistance;
        const float dc2 = lightDistance2;
        const float ds = dc * cosTheta - sqrt(max(cLIGHT_SAMPING_EPSILON, radius2 - dc2 * sinTheta2));
        const float cosAlpha = (dc2 + radius2 - sq(ds)) / (2.0f * dc * radius);
        const float sinAlpha = sqrt(max(0.0f, 1.0f - sq(cosAlpha)));

        // Construct a coordinate frame to sample in around the direction of the light vector

        const float3 sampleSpaceNormal = normalize(lightVector);
        float3 sampleSpaceTangent;
        float3 sampleSpaceBitangent;
        BranchlessONB(sampleSpaceNormal, sampleSpaceTangent, sampleSpaceBitangent);

        // Calculate sample position and normal on the sphere

        float sinPhi;
        float cosPhi;
        sincos(phi, sinPhi, cosPhi);

        const float3 radiusVector = sphericalDirection(
            sinAlpha, cosAlpha, sinPhi, cosPhi, -sampleSpaceTangent, -sampleSpaceBitangent, -sampleSpaceNormal);
        const float3 spherePositionSample = position + radius * radiusVector;
        const float3 sphereNormalSample = normalize(radiusVector);
        // Note: Reprojection for position to minimize error here skipped for performance

        // Calculate the pdf

        // Note: The cone already represents a solid angle effectively so its pdf is already a solid angle pdf
        const float solidAnglePdf = 1.0f / (2.0f * K_PI * (1.0f - cosThetaMax));

        // Create the light sample

        PolymorphicLightSample lightSample;

        lightSample.Position        = spherePositionSample;
        lightSample.Normal          = sphereNormalSample;
        lightSample.Radiance        = radiance;
        lightSample.SolidAnglePdf   = solidAnglePdf;
        lightSample.LightSampleableByBSDF = false;

        return lightSample;
    }

    bool Eval(const float3 rayPos, const float3 rayDir, inout float3 outRadiance, inout float3 outLightSamplePosition)
    {
        // TODO: pretty sure this can be optimized
         const float3 lightVector = position - rayPos;
         const float lightDistance2 = dot(lightVector, lightVector);
         const float radius2 = sq(radius);
         if (lightDistance2 < radius2)
             return false;

        if (!IntersectRaySphere(rayPos, rayDir, position, radius, outLightSamplePosition))
            return false;

        outRadiance = radiance * evaluateLightShaping(shaping, rayPos, position); // not sure this is correct/wanted

        return true;
    }

    float CalcSolidAnglePdfForMIS(in const float3 viewerPosition, in const float3 lightSamplePosition)
    {
        const float3 lightVector = position - viewerPosition;
        const float lightDistance2 = dot(lightVector, lightVector);
        const float radius2 = sq(radius);

        // Compute theta and phi for cone sampling

        const float sinThetaMax2 = radius2 / lightDistance2;
        const float cosThetaMax = sqrt(max(0.0f, 1.0f - sinThetaMax2));

        // Note: The cone already represents a solid angle effectively so its pdf is already a solid angle pdf
        const float solidAnglePdf = 1.0f / (2.0f * K_PI * (1.0f - cosThetaMax));
        return solidAnglePdf;
    }


    float getSurfaceArea()
    {
        return 4 * K_PI * sq(radius);
    }

    float GetPower()
    {
        return getSurfaceArea() * K_PI * Luminance(radiance) * getShapingFluxFactor(shaping);
    }

    float GetWeightForVolume(in const float3 volumeCenter, in const float volumeRadius)
    {
        if (!testSphereIntersectionForShapedLight(position, radius, shaping, volumeCenter, volumeRadius))
            return 0.0;

        float distance = length(volumeCenter - position);
        distance = getAverageDistanceToVolume(distance, volumeRadius);

        float sinHalfAngle = radius / distance;
        float solidAngle = 2 * K_PI * (1.0 - sqrt(1.0 - sq(sinHalfAngle)));

        return solidAngle * Luminance(radiance);
    }

    static SphereLight Create(in const PolymorphicLightInfoFull lightInfo)
    {
        SphereLight sphereLight;

        sphereLight.position = lightInfo.Base.Center;
        sphereLight.radius = f16tof32(lightInfo.Base.Scalars);
        sphereLight.radiance = PolymorphicLight::UnpackColor(lightInfo.Base);
        sphereLight.shaping = unpackLightShaping(lightInfo);

        return sphereLight;
    }
};
#endif // #if POLYLIGHT_SPHERE_ENABLE

#if POLYLIGHT_POINT_ENABLE || defined(__INTELLISENSE__)
// Point light is a sphere light with zero radius.
// On the host side, they are both created from LightType_Point, depending on the radius.
// The values returned from all interface methods of PointLight are the same as SphereLight
// would produce in the limit when radius approaches zero, with some exceptions in CalcSample.
struct PointLight
{
    float3 position;
    float3 flux;
    float3 direction;
    float outerAngle;
    float innerAngle;
    LightShaping shaping;

    // Interface methods

    PolymorphicLightSample CalcSample(in const float3 viewerPosition)
    {
        const float3 lightVector = position - viewerPosition;

        // We cannot compute finite values for radiance and solidAnglePdf for a point light,
        // so return the limit of (radiance / solidAnglePdf) with radius --> 0 as radiance.
        PolymorphicLightSample lightSample;
        lightSample.Position = position;
        lightSample.Normal = normalize(-lightVector);
        lightSample.Radiance = flux / dot(lightVector, lightVector);
        lightSample.SolidAnglePdf = 1.0;
        lightSample.LightSampleableByBSDF = false;

        return lightSample;
    }

    float CalcSolidAnglePdfForMIS(in const float3 viewerPosition, in const float3 lightSamplePosition)
    {
        return 0;
    }

    float GetPower()
    {
        return 4.0 * K_PI * Luminance(flux) * getShapingFluxFactor(shaping);
    }

    float GetWeightForVolume(in const float3 volumeCenter, in const float volumeRadius)
    {
        if (!testSphereIntersectionForShapedLight(position, 0, shaping, volumeCenter, volumeRadius))
            return 0.0;

        float distance = length(volumeCenter - position);
        distance = getAverageDistanceToVolume(distance, volumeRadius);

        return Luminance(flux) / sq(distance);
    }

    static PointLight Create(in const PolymorphicLightInfoFull lightInfo)
    {
        PointLight pointLight;

        pointLight.position = lightInfo.Base.Center;
        pointLight.flux = PolymorphicLight::UnpackColor(lightInfo.Base);
        pointLight.direction = OctToNDirUnorm32(lightInfo.Base.Direction1);
        pointLight.outerAngle = f16tof32(lightInfo.Base.Direction2);
        pointLight.innerAngle = f16tof32(lightInfo.Base.Direction2 >> 16);
        pointLight.shaping = unpackLightShaping(lightInfo);

        return pointLight;
    }
};
#endif // #if POLYLIGHT_POINT_ENABLE

#if POLYLIGHT_DIRECTIONAL_ENABLE || defined(__INTELLISENSE__)
struct DirectionalLight
{
    float3 direction;
    float cosHalfAngle; // Note: Assumed to be != 1 to avoid delta light special case
    float sinHalfAngle;
    float angularSize;
    float solidAngle;
    float3 radiance;

    // Interface methods

    PolymorphicLightSample CalcSample(in const float2 random, in const float3 viewerPosition)
    {
        const float2 diskSample = SampleDiskUniform(random);

        float3 tangent, bitangent;
        BranchlessONB(direction, tangent, bitangent);

        const float3 distantDirectionSample = direction
            + tangent * diskSample.x * sinHalfAngle
            + bitangent * diskSample.y * sinHalfAngle;

        // Special convention for directional lights: since there is no physical position (the
        // light is at infinity), store the unit vector that points FROM the receiver TO the
        // sampled point on the sun disk directly in Position. Consumers can then read it as a
        // direction without any subtract/normalize round trip against viewerPosition.
        // `direction` is the light's travel direction, so toLight = -distantDirectionSample.
        const float3 toLightSample = -distantDirectionSample;

        PolymorphicLightSample lightSample;

        lightSample.Position = toLightSample;
        lightSample.Normal = direction;
        lightSample.Radiance = radiance;
        lightSample.SolidAnglePdf = 1.0 / solidAngle;
        lightSample.LightSampleableByBSDF = false;

        return lightSample;
    }

    float CalcSolidAnglePdfForMIS(in const float3 viewerPosition, in const float3 lightSamplePosition)
    {
        return 0;
    }

    // Helper methods

    static DirectionalLight Create(in const PolymorphicLightInfoFull lightInfo)
    {
        DirectionalLight directionalLight;

        directionalLight.direction = OctToNDirUnorm32(lightInfo.Base.Direction1);

        float halfAngle = f16tof32(lightInfo.Base.Scalars);
        directionalLight.angularSize = 2 * halfAngle;
        sincos(halfAngle, directionalLight.sinHalfAngle, directionalLight.cosHalfAngle);
        directionalLight.solidAngle = f16tof32(lightInfo.Base.Scalars >> 16);
        directionalLight.radiance = PolymorphicLight::UnpackColor(lightInfo.Base);

        return directionalLight;
    }
};
#endif // #if POLYLIGHT_DIRECTIONAL_ENABLE

#if POLYLIGHT_TRIANGLE_ENABLE || defined(__INTELLISENSE__)
struct TriangleLight
{
    float3 base;
    float3 edge1;
    float3 edge2;
    float3 radiance;
    float3 normal;
    float surfaceArea;
    
    // Interface methods
    PolymorphicLightSample CalcSample(in const float2 random, in const float3 viewerPosition)
    {
        // consider https://www.graphics.cornell.edu/pubs/1995/Arv95c.pdf for ugprade - thanks to Johannes 

        PolymorphicLightSample result = (PolymorphicLightSample)0;

        float3 bary = SampleTriangleUniform(random);
        result.Position = base + edge1 * bary.y + edge2 * bary.z;
        result.Position = ComputeRayOrigin(result.Position, normal);
        result.Normal = normal;

        const float3 toLight = result.Position - viewerPosition;
        const float distSqr = max(FLT_EPSILON_MINI, dot(toLight, toLight));
        const float distance = sqrt(distSqr);
        const float3 dir = toLight / distance;
        float cosTheta = dot(normal, -dir);
        
        result.SolidAnglePdf = 0.f;
        result.Radiance = 0.f;
        if (cosTheta <= 0.f) return result;

        const float areaPdf = max(FLT_EPSILON_MINI, 1.0 / surfaceArea);
        result.SolidAnglePdf = min( MAX_SOLID_ANGLE_PDF, pdfAtoW(areaPdf, distance, cosTheta) );
        result.Radiance = radiance;
        result.LightSampleableByBSDF = true;

        #if 0 && defined(DEBUG_PRINT_DEFINED)
        float solidAnglePdfTest = CalcSolidAnglePdfForMIS(viewerPosition, result.Position);
        if( !RelativelyEqual(result.SolidAnglePdf, solidAnglePdfTest ))
            DebugPrint( "ERROR: solidAngle {0} solidAngleTest {1}", result.SolidAnglePdf, solidAnglePdfTest );
        #endif

        return result;
    }

    float CalcSolidAnglePdfForMIS(in const float3 viewerPosition, in const float3 lightSamplePosition)
    {
        const float3 toLight = lightSamplePosition - viewerPosition;
        const float distSqr = max(FLT_EPSILON_MINI, dot(toLight, toLight));
        const float distance = sqrt(distSqr);
        const float3 dir = toLight / distance;
        float cosTheta = dot(normal, -dir);
        // Like CalcSample, which never produces these directions (e.g. a triangle that is degenerate or flipped after fp16 packing) - so
        // BSDF hits on them must get MIS weight 1
        if (cosTheta <= 0.f) return 0.f;

        const float areaPdf = max(FLT_EPSILON_MINI, 1.0 / surfaceArea);
        return min( MAX_SOLID_ANGLE_PDF, pdfAtoW(areaPdf, distance, cosTheta) );
    }

    float GetPower()
    {
        return surfaceArea * K_PI * Luminance(radiance);
    }

    float GetWeightForVolume(in const float3 volumeCenter, in const float volumeRadius)
    {
        float distanceToPlane = dot(volumeCenter - base, normal);
        if (distanceToPlane < -volumeRadius)
            return 0; // Cull - the entire volume is below the light's horizon

        float3 barycenter = base + ((edge1 + edge2) / 3.0);
        float distance = length(barycenter - volumeCenter);
        distance = getAverageDistanceToVolume(distance, volumeRadius);

        float approximateSolidAngle = surfaceArea / sq(distance);
        approximateSolidAngle = min(approximateSolidAngle, 2 * K_PI);

        return approximateSolidAngle * Luminance(radiance);
    }

    // Helper methods
    static TriangleLight Create(in const PolymorphicLightInfoFull lightInfo)
    {
        TriangleLight triLight;

        triLight.edge1 = f16tof32(uint3(lightInfo.Base.Direction1, lightInfo.Base.Direction2, lightInfo.Base.Scalars) & 0xffff);
        triLight.edge2 = f16tof32(uint3(lightInfo.Base.Direction1, lightInfo.Base.Direction2, lightInfo.Base.Scalars) >> 16);
        triLight.base = lightInfo.Base.Center - ((triLight.edge1 + triLight.edge2) / 3.0);
        triLight.radiance = PolymorphicLight::UnpackColor(lightInfo.Base);

        float3 lightNormal = cross(triLight.edge1, triLight.edge2);
        float lightNormalLength = length(lightNormal);
        
        //Check for tiny triangles
        if(lightNormalLength > 0.0)
        {
            triLight.surfaceArea = 0.5 * lightNormalLength;
            triLight.normal = lightNormal / lightNormalLength;
        }
        else
        {
           triLight.surfaceArea = 0.0;
           triLight.normal = 0.0; 
        }

        return triLight;
    }

    PolymorphicLightInfoFull Store()
    {
        PolymorphicLightInfoFull lightInfo = (PolymorphicLightInfoFull)0;

        PolymorphicLight::PackColor(radiance, lightInfo.Base);
        lightInfo.Base.Center = base + ((edge1 + edge2) / 3.0); 
        // uint, not float: each packed pair exceeds 2^24, so a float round trip would round off edge1's low mantissa bits.
        uint3 edges = (f32tof16(edge1) & 0xffff) | (f32tof16(edge2) << 16);
        lightInfo.Base.Direction1 = edges.x;
        lightInfo.Base.Direction2 = edges.y;
        lightInfo.Base.Scalars = edges.z;
        lightInfo.Base.ColorTypeAndFlags |= uint(PolymorphicLightType::kTriangle) << kPolymorphicLightTypeShift;
        //lightInfo.Base.logRadiance |= f32tof16((uint) empty slot) << 16; //unused
        lightInfo.Extended = PolymorphicLightInfoEx::empty();
        return lightInfo;
    }
};
#endif // #if POLYLIGHT_TRIANGLE_ENABLE

#if POLYLIGHT_ENV_ENABLE || defined(__INTELLISENSE__)
struct EnvironmentLight
{
    // Interface methods

    // User needs to implement these themselves - needs access to environment sampling which isn't packed with the light
    PolymorphicLightSample CalcSample(in const float2 random, in const float3 viewerPosition);
    // {
    //     PolymorphicLightSample pls;
    //    
    //     Buffer<uint2> unused;
    //     EnvMapSampler envMapSampler = EnvMapSampler::make( s_EnvironmentMapImportanceSampler name as define or somt'n, t_EnvironmentMapImportanceMap, g_Const.envMapImportanceSamplingParams,
    //                                                         t_EnvironmentMap, s_EnvironmentMapSampler, g_Const.envMapSceneParams, unused );
    // 
    //     float3 worldDir = Decode_Oct( random );
    //     pls.Position = viewerPosition + worldDir * DISTANT_LIGHT_DISTANCE;
    //     pls.Normal = -worldDir;
    //     pls.Radiance = envMapSampler.Eval(worldDir);
    //     pls.SolidAnglePdf = envMapSampler.MIPDescentEvalPdf(worldDir);
    //     
    //     return pls;
    // }    

    float CalcSolidAnglePdfForMIS(in const float3 viewerPosition, in const float3 lightSamplePosition);

    // Helper methods

    static EnvironmentLight Create(in const PolymorphicLightInfoFull lightInfo);
    // {
    //     EnvironmentLight envLight;
    // 
    //     return envLight;
    // }
};
#endif // #if POLYLIGHT_ENV_ENABLE

#if POLYLIGHT_QT_ENV_ENABLE || defined(__INTELLISENSE__)

struct EnvironmentQuadLight
{
    uint        NodeX;          // x coord in the [0, NodeDim-1]
    uint        NodeY;          // y coord in the [0, NodeDim-1]
    uint        NodeDim;        // this is the resolution of the node layer in nodes (which exactly correspond to the ea-octahedral mapped environment map's MIP level of the same resolution)
    float       Weight;
    float3      Radiance;

    // User needs to implement these themselves - needs access to environment sampling which isn't packed with the light
    static float3 ToWorld(float3 localDir);
    static float3 ToLocal(float3 worldDir);
    static float3 SampleLocalSpace(float3 localDir);

    // Interface methods
    PolymorphicLightSample CalcSample(in const float2 random, in const float3 viewerPosition)
    {
        PolymorphicLightSample pls;

        float2 subTexelPos = float2( ((float)NodeX+random.x) / (float)NodeDim, ((float)NodeY+random.y) / (float)NodeDim );

        float3 localDir = oct_to_ndir_equal_area_unorm(subTexelPos);
        float3 worldDir = ToWorld(localDir);

        pls.Position = viewerPosition + worldDir * DISTANT_LIGHT_DISTANCE;
        pls.Normal = -worldDir;
#if !defined(NEE_AT_SAMPLE_BAKED_ENVIRONMENT)
#define NEE_AT_SAMPLE_BAKED_ENVIRONMENT 1
#endif
#if NEE_AT_SAMPLE_BAKED_ENVIRONMENT
        pls.Radiance = Radiance;
#else
        pls.Radiance = SampleLocalSpace(localDir);
#endif
        pls.SolidAnglePdf = (NodeDim*NodeDim) / (4.0f * K_PI);
        pls.LightSampleableByBSDF = true;
        
        return pls;
    }    

    float CalcSolidAnglePdfForMIS(in const float3 viewerPosition, in const float3 lightSamplePosition)
    {
        return (NodeDim*NodeDim) / (4.0f * K_PI);
    }

    // Helper methods

    static EnvironmentQuadLight Create(in const PolymorphicLightInfoFull lightInfo)
    {
        EnvironmentQuadLight envQuadLight;

        envQuadLight.NodeX      = lightInfo.Base.Direction1 >> 16;
        envQuadLight.NodeY      = lightInfo.Base.Direction1 & 0xFFFF;
        envQuadLight.NodeDim    = lightInfo.Base.Direction2 >> 16;
        envQuadLight.Weight     = asfloat(lightInfo.Base.Scalars);
        envQuadLight.Radiance   = PolymorphicLight::UnpackColor(lightInfo.Base);
    
        return envQuadLight;
    }

    float GetPower()
    {
        return Weight;
    }

    PolymorphicLightInfoFull Store()
    {
        PolymorphicLightInfoFull lightInfo = (PolymorphicLightInfoFull)0;

        PolymorphicLight::PackColor(Radiance, lightInfo.Base);
        lightInfo.Base.Direction1    = (NodeX << 16) | NodeY;
        lightInfo.Base.Direction2    = (NodeDim << 16);
        lightInfo.Base.Scalars       = asuint(Weight);
        lightInfo.Base.ColorTypeAndFlags |= uint(PolymorphicLightType::kEnvironmentQuad) << kPolymorphicLightTypeShift;
        //PolymorphicLight::PackColor(radiance, lightInfo.Base);
        lightInfo.Extended = PolymorphicLightInfoEx::empty();
        return lightInfo;
    }
};
#endif // #if POLYLIGHT_ENV_ENABLE

#if POLYLIGHT_LIGHT_CLUSTER || defined(__INTELLISENSE__)

struct LightCluster
{
    uint        IndexFrom;          // starting index of the light we're representing; it should always be valid (not RTXPT_INVALID_LIGHT_INDEX) even if not active
    uint        IndexCount;         // number of sequential lights we're representing; it should always be valid (>0) even if not active

    float3      AABBCenter;         // covers all triangles (or volumes of whatever lights we're representing from IndexFrom to but not including IndexFrom+IndexCount)
    float3      AABBExtents;        // covers all triangles (or volumes of whatever lights we're representing from IndexFrom to but not including IndexFrom+IndexCount)

    bool        Active;            // whether it's currently representing any triangles

    // Interface methods

    PolymorphicLightSample CalcSample(in const float3 viewerPosition)
    {
        // Note: this should never get called. LightCluster is a proxy for other light types and as such this call should be handled on
        // the caller side by picking one random index between [IndexFrom, IndexFrom+Count) and calling its CalcSample
        const float3 lightVector = AABBCenter - viewerPosition;
        PolymorphicLightSample lightSample;
        lightSample.Position = AABBCenter;
        lightSample.Normal = normalize(-lightVector);
        lightSample.Radiance = float3(10,1,0); // pink-ish?
        lightSample.SolidAnglePdf = 1.0;
        lightSample.LightSampleableByBSDF = false;

        return lightSample;
    }

    float CalcSolidAnglePdfForMIS(in const float3 viewerPosition, in const float3 lightSamplePosition)
    {
        // Note: this should never get called. LightCluster is a proxy for other light types and as such this call should be handled on
        // the caller side by picking one random index between [IndexFrom, IndexFrom+Count) and calling its CalcSample
        return 0;
    }

    float GetPower()
    {
        return 0; // not implemented
    }

    float GetWeightForVolume(in const float3 volumeCenter, in const float volumeRadius)
    {
        return 0; // not implemented
    }

    // Packing: Center = AABBCenter; Direction1 = IndexFrom (23 bits) | (IndexCount-1) << 23 (6 bits, so max 64) | Active << 31;
    // Direction2 unused; Scalars = fp16 AABBExtents.xy; LogRadiance = fp16 AABBExtents.z (upper 16 bits unused)
    static LightCluster Create(in const PolymorphicLightInfoFull lightInfo)
    {
        LightCluster lightCluster;
        lightCluster.IndexFrom      = lightInfo.Base.Direction1 & 0x7FFFFF;
        lightCluster.IndexCount     = ((lightInfo.Base.Direction1 >> 23) & 0x3F) + 1;
        lightCluster.Active         = (lightInfo.Base.Direction1 >> 31) != 0;
        lightCluster.AABBCenter     = lightInfo.Base.Center;
        lightCluster.AABBExtents    = f16tof32(uint3(lightInfo.Base.Scalars, lightInfo.Base.Scalars >> 16, lightInfo.Base.LogRadiance));
        return lightCluster;
    }

    PolymorphicLightInfoFull Store()
    {
        PolymorphicLightInfoFull lightInfo = (PolymorphicLightInfoFull)0;
        lightInfo.Base.Center = AABBCenter;
        lightInfo.Base.ColorTypeAndFlags = uint(PolymorphicLightType::kLightCluster) << kPolymorphicLightTypeShift;
        lightInfo.Base.Direction1 = (IndexFrom & 0x7FFFFF) | (((IndexCount - 1) & 0x3F) << 23) | (Active ? (1u << 31) : 0);
        uint3 extents = f32tof16(AABBExtents);
        extents += uint3(f16tof32(extents) < AABBExtents);  // round up so the fp16 box never shrinks
        lightInfo.Base.Scalars = extents.x | (extents.y << 16);
        lightInfo.Base.LogRadiance = extents.z;
        lightInfo.Extended = PolymorphicLightInfoEx::empty();
        return lightInfo;
    }

    // Whether the global/local samplers may hand this light out directly: a light in an active cluster is only reachable through
    // its cluster, and an inactive cluster not at all (its lights are sampled individually)
    static bool IsDirectlySampleable(PolymorphicLightInfo lightInfo)
    {
        if( lightInfo.IsPartOfLightCluster() )
            return false;
        return PolymorphicLight::DecodeType(lightInfo) != PolymorphicLightType::kLightCluster || LightCluster::Create(PolymorphicLightInfoFull::make(lightInfo)).Active;
    }

    // This takes a light index, checks the type, and if it's a cluster then it will randomly pick a single contained light and adjust pdf accordingly.
    // If it's not a light cluster, it does nothing.
    static void MaybeUnwrap(in const StructuredBuffer<PolymorphicLightInfo> lightsBuffer, inout uint lightIndex, inout float selectionPdf, in const float random)
    {
        PolymorphicLightInfo infoBase = lightsBuffer[lightIndex];  // no bounds checking here

        // if this is a light cluster, pick one index from it and re-load it
        if (PolymorphicLight::DecodeType(infoBase) == PolymorphicLightType::kLightCluster)
        {
            LightCluster lightCluster = LightCluster::Create(PolymorphicLightInfoFull::make(infoBase, PolymorphicLightInfoEx::empty()));
            lightIndex = lightCluster.IndexFrom + clamp( uint(random * lightCluster.IndexCount), 0, lightCluster.IndexCount-1 );
            infoBase = lightsBuffer[lightIndex];
            selectionPdf *= 1.0 / (float)lightCluster.IndexCount;

            // we should assert on
            // if (DecodeType(infoBase) == PolymorphicLightType::kLightCluster) assert();
        }
    }
    // Opposite of MaybeUnwrap - this takes a light index, checks the flag, and if it's >in a cluster< then it will replace its index with cluster index
    // and adjust pdf accordingly.
    // If it's not a light cluster, it does nothing.
    static void MaybeWrap(in const StructuredBuffer<PolymorphicLightInfo> lightsBuffer, in const StructuredBuffer<PolymorphicLightInfoEx> lightsExBuffer, inout uint lightIndex, inout float selectionPdf, in const float random)
    {
        PolymorphicLightInfo infoBase = lightsBuffer[lightIndex];  // no bounds checking here

        // if this is a light cluster, pick one index from it and re-load it
        if ( infoBase.IsPartOfLightCluster() )
        {
            const uint clusterIndexAndCount = lightsExBuffer[lightIndex].ClusterIndexAndCount;
            lightIndex = PolymorphicLightInfoEx::UnpackClusterIndex( clusterIndexAndCount );
            selectionPdf *= (float)PolymorphicLightInfoEx::UnpackClusterLightCount( clusterIndexAndCount );
        }        
    }
};
#endif // #if POLYLIGHT_LIGHT_CLUSTER

PolymorphicLightSample PolymorphicLight::CalcSample(in const PolymorphicLightInfoFull lightInfo, in const float2 random, in const float3 viewerPosition)
{
    PolymorphicLightSample lightSample = (PolymorphicLightSample)0;

    switch (DecodeType(lightInfo))
    {
#if POLYLIGHT_SPHERE_ENABLE
    case PolymorphicLightType::kSphere:         lightSample = SphereLight::Create(lightInfo).CalcSample(random, viewerPosition); break;
#endif
#if POLYLIGHT_POINT_ENABLE
    case PolymorphicLightType::kPoint:          lightSample = PointLight::Create(lightInfo).CalcSample(viewerPosition); break;
#endif
#if POLYLIGHT_TRIANGLE_ENABLE
#ifndef OVERRIDE_POLYLIGHT_TRIANGLE_ENABLE
    case PolymorphicLightType::kTriangle:       lightSample = TriangleLight::Create(lightInfo).CalcSample(random, viewerPosition); break;
#endif
#endif
#if POLYLIGHT_DIRECTIONAL_ENABLE
    case PolymorphicLightType::kDirectional:    lightSample = DirectionalLight::Create(lightInfo).CalcSample(random, viewerPosition); break;
#endif
#if POLYLIGHT_ENV_ENABLE
    case PolymorphicLightType::kEnvironment:    lightSample = EnvironmentLight::Create(lightInfo).CalcSample(random, viewerPosition); break;
#endif
#if POLYLIGHT_QT_ENV_ENABLE
    case PolymorphicLightType::kEnvironmentQuad:lightSample = EnvironmentQuadLight::Create(lightInfo).CalcSample(random, viewerPosition); break;
#endif
#if POLYLIGHT_LIGHT_CLUSTER
    case PolymorphicLightType::kLightCluster:   lightSample = LightCluster::Create(lightInfo).CalcSample(viewerPosition); break;
#endif
    default: break;
    }

    if (lightSample.SolidAnglePdf > 0)
    {
        lightSample.Radiance *= evaluateLightShaping(unpackLightShaping(lightInfo), viewerPosition, lightSample.Position); // not sure this is correct/wanted - might be better placed in individual lights
    }

    return lightSample;
}

float PolymorphicLight::GetPower(in const PolymorphicLightInfoFull lightInfo)
{
    switch (DecodeType(lightInfo))
    {
#if POLYLIGHT_SPHERE_ENABLE
    case PolymorphicLightType::kSphere:         return SphereLight::Create(lightInfo).GetPower();
#endif
#if POLYLIGHT_POINT_ENABLE
    case PolymorphicLightType::kPoint:          return PointLight::Create(lightInfo).GetPower();
#endif
#if POLYLIGHT_TRIANGLE_ENABLE
    case PolymorphicLightType::kTriangle:       return TriangleLight::Create(lightInfo).GetPower();
#endif
#if POLYLIGHT_DIRECTIONAL_ENABLE
    case PolymorphicLightType::kDirectional:    return 0; // infinite lights don't go into the local light PDF map
#endif
#if POLYLIGHT_ENV_ENABLE
    case PolymorphicLightType::kEnvironment:    return 0;
#endif
#if POLYLIGHT_QT_ENV_ENABLE
    case PolymorphicLightType::kEnvironmentQuad:return EnvironmentQuadLight::Create(lightInfo).GetPower();
#endif
    default: return 0;
    }
}

float PolymorphicLight::GetWeightForVolume(in const PolymorphicLightInfoFull lightInfo, in const float3 volumeCenter, in const float volumeRadius)
{
    switch (DecodeType(lightInfo))
    {
#if POLYLIGHT_SPHERE_ENABLE
    case PolymorphicLightType::kSphere:         return SphereLight::Create(lightInfo).GetWeightForVolume(volumeCenter, volumeRadius);
#endif
#if POLYLIGHT_POINT_ENABLE
    case PolymorphicLightType::kPoint:          return PointLight::Create(lightInfo).GetWeightForVolume(volumeCenter, volumeRadius);
#endif
#if POLYLIGHT_TRIANGLE_ENABLE
    case PolymorphicLightType::kTriangle:       return TriangleLight::Create(lightInfo).GetWeightForVolume(volumeCenter, volumeRadius);
#endif
#if POLYLIGHT_DIRECTIONAL_ENABLE    
    case PolymorphicLightType::kDirectional:    return 0; // infinite lights do not affect volume sampling
#endif
#if POLYLIGHT_ENV_ENABLE
    case PolymorphicLightType::kEnvironment:    return 0; // infinite lights do not affect volume sampling
#endif
#if POLYLIGHT_QT_ENV_ENABLE
    case PolymorphicLightType::kEnvironmentQuad:return 0; // infinite lights do not affect volume sampling
#endif
#if POLYLIGHT_LIGHT_CLUSTER
    case PolymorphicLightType::kLightCluster:   return LightCluster::Create(lightInfo).GetWeightForVolume(volumeCenter, volumeRadius);
#endif
    default: return 0;
    }
}

PolymorphicLightType PolymorphicLight::DecodeType(PolymorphicLightInfo lightInfo)
{
    uint typeCode = (lightInfo.ColorTypeAndFlags >> kPolymorphicLightTypeShift) & kPolymorphicLightTypeMask;
    return (PolymorphicLightType)typeCode;
}

PolymorphicLightType PolymorphicLight::DecodeType(PolymorphicLightInfoFull lightInfo)
{
    return PolymorphicLight::DecodeType(lightInfo.Base);
}

float PolymorphicLight::UnpackRadiance(uint logRadiance)
{
    return (logRadiance == 0) ? 0 : exp2((float(logRadiance - 1) / 65534.0) * (kPolymorphicLightMaxLog2Radiance - kPolymorphicLightMinLog2Radiance) + kPolymorphicLightMinLog2Radiance);
}

float3 PolymorphicLight::UnpackColor(PolymorphicLightInfo lightInfoBase)
{
    float3 color = Unpack_R8G8B8_UFLOAT(lightInfoBase.ColorTypeAndFlags);
    float radiance = UnpackRadiance(lightInfoBase.LogRadiance & 0xffff);
    return color * radiance.xxx;
}

void PolymorphicLight::PackColor(float3 radiance, inout PolymorphicLightInfo lightInfoBase)
{   
    float intensity = max(radiance.r, max(radiance.g, radiance.b));

    if (intensity > 0.0)
    {
        float logRadiance = saturate((log2(intensity) - kPolymorphicLightMinLog2Radiance) 
            / (kPolymorphicLightMaxLog2Radiance - kPolymorphicLightMinLog2Radiance));
        uint packedRadiance = min(uint32_t(ceil(logRadiance * 65534.0)) + 1, 0xffffu);
        float unpackedRadiance = UnpackRadiance(packedRadiance);

        float3 normalizedRadiance = saturate(radiance.rgb / unpackedRadiance.xxx);

        lightInfoBase.LogRadiance |= packedRadiance;
        lightInfoBase.ColorTypeAndFlags |= Pack_R8G8B8_UFLOAT(normalizedRadiance);
    }
}

bool PolymorphicLight::PackCompactInfo(PolymorphicLightInfoFull lightInfo, out uint4 res1, out uint4 res2)
{
    if (unpackLightShaping(lightInfo).isSpot)
    {
        res1 = 0;
        res2 = 0;
        return false;
    }

    res1.xyz = asuint(lightInfo.Base.Center.xyz);
    res1.w = lightInfo.Base.ColorTypeAndFlags;

    res2.x = lightInfo.Base.Direction1;
    res2.y = lightInfo.Base.Direction2;
    res2.z = lightInfo.Base.Scalars;
    res2.w = lightInfo.Base.LogRadiance;
    return true;
}

PolymorphicLightInfoFull PolymorphicLight::UnpackCompactInfo(const uint4 data1, const uint4 data2)
{
    PolymorphicLightInfoFull lightInfo = (PolymorphicLightInfoFull)0;
    lightInfo.Extended = PolymorphicLightInfoEx::empty();
    lightInfo.Base.Center.xyz = asfloat(data1.xyz);
    lightInfo.Base.ColorTypeAndFlags = data1.w;
    lightInfo.Base.Direction1 = data2.x;
    lightInfo.Base.Direction2 = data2.y;
    lightInfo.Base.Scalars = data2.z;
    lightInfo.Base.LogRadiance = data2.w;
    return lightInfo;
}

#endif // __POLYMORPHIC_LIGHT_HLSLI__