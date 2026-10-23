/*
* Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#ifndef __POLYMORPHIC_LIGHT_H__
#define __POLYMORPHIC_LIGHT_H__

#define RTXPT_INVALID_LIGHT_INDEX       0xFFFFFFFF    // independent macro from NEE-AT but must match the value (static assert ensured)

#define DISTANT_LIGHT_DISTANCE          100000.0

static const float kMinSpotlightFalloff = 0.0001f;

static const uint   kPolymorphicLightTypeShift = 24;
static const uint   kPolymorphicLightTypeMask = 0xf;
static const uint   kPolymorphicLightShapingEnableBit = 1 << 28;
static const uint   kPolymorphicLightIesProfileEnableBit = 1 << 29;
static const uint   kPolymorphicLightShapingUseMinFalloff = 1 << 30;  // see kMinSpotlightFalloff
static const uint   kPolymorphicLightIsInCluster = 1u << 31;
static const float  kPolymorphicLightMinLog2Radiance = -8.f;
static const float  kPolymorphicLightMaxLog2Radiance = 40.f;

// NOTE: see kPolymorphicLightTypeMask for the available number of bits to store the type
#ifdef __cplusplus
enum class PolymorphicLightType
#else
enum PolymorphicLightType
#endif
{
    kSphere = 0,
    kTriangle,
    kDirectional,
    kEnvironment,
    kPoint,
    kEnvironmentQuad,
    kLightCluster
};

// Stores shared light information (type) and specific light information; see PolymorphicLight.hlsli for encoding format
struct PolymorphicLightInfo
{
    // uint4[0]
    float3  Center;
    uint    ColorTypeAndFlags;  // RGB8 + uint8 (see the kPolymorphicLight... constants above)

    // uint4[1] - Direction1/Direction2/Scalars depend on the light type; its Create() in PolymorphicLight.hlsli decodes them
    uint    Direction1;         // e.g. oct-encoded direction (spot, directional); fp16 edge1.x | edge2.x (triangle)
    uint    Direction2;         // e.g. fp16 outer | inner cone angle (spot, point); fp16 edge1.y | edge2.y (triangle)
    uint    Scalars;            // e.g. fp16 radius (sphere, spot); fp16 edge1.z | edge2.z (triangle)
    uint    LogRadiance;        // uint16 | empty slot

    bool    HasLightShaping()                   { return (ColorTypeAndFlags & kPolymorphicLightShapingEnableBit) != 0; }
    bool    IsPartOfLightCluster()              { return (ColorTypeAndFlags & kPolymorphicLightIsInCluster) != 0; }
};

struct PolymorphicLightInfoEx
{
    // uint4[2] shaping data
    uint    IesProfileIndex;
    uint    PrimaryAxis;                // oct-encoded
    uint    CosConeAngleAndSoftness;    // 2x float16
    uint    ClusterIndexAndCount;       // cluster light index (bits 9..31) and its light count - 1 (bits 0..8), same layout as NEEAT::PackMiniListLightAndCount;
                                        // RTXPT_INVALID_LIGHT_INDEX if not in a cluster. The count is final only while IsPartOfLightCluster() (see ProcessClusters)

    static uint PackClusterIndexAndCount(uint clusterIndex, uint lightCount)    { return ((clusterIndex & 0x007FFFFF) << 9) | ((lightCount - 1) & 0x1FF); }
    static uint UnpackClusterIndex(uint clusterIndexAndCount)                  { return clusterIndexAndCount >> 9; }
    static uint UnpackClusterLightCount(uint clusterIndexAndCount)             { return (clusterIndexAndCount & 0x1FF) + 1; }

    static PolymorphicLightInfoEx empty() { PolymorphicLightInfoEx ret; ret.IesProfileIndex = 0; ret.PrimaryAxis = 0; ret.CosConeAngleAndSoftness = 0; ret.ClusterIndexAndCount = RTXPT_INVALID_LIGHT_INDEX; return ret; }
};

struct PolymorphicLightInfoFull
{
    PolymorphicLightInfo    Base;
    PolymorphicLightInfoEx  Extended;
    static PolymorphicLightInfoFull make(PolymorphicLightInfo base) { PolymorphicLightInfoFull ret; ret.Base = base; ret.Extended = PolymorphicLightInfoEx::empty(); return ret; }
    static PolymorphicLightInfoFull make(PolymorphicLightInfo base, PolymorphicLightInfoEx extended) { PolymorphicLightInfoFull ret; ret.Base = base; ret.Extended = extended; return ret; }
};

// Describes a light sample, mainly for use in NEE (but could be just a standalone direct illumination light sampler).
// It is considered in the context of a shaded surface point, from which Distance and Direction to light sample are computed.
// In case of emissive triangle light source, it is advisable to compute anti-self-intersection offset before computing
// distance and direction, even though distance shortening is needed anyways for shadow rays due to precision issues.
struct LightSample
{
    float3  Li;                     ///< Incident radiance at the shading point (unshadowed). This is already divided by the pdf.
    //float   Pdf;                  ///< Pdf with respect to solid angle at the shading point with selected light (selectionPDF*solidAnglePdf).
    float   Distance;               ///< Ray distance for visibility evaluation (NOT shortened or offset to avoid self-intersection). Ray starts at shading surface.
    float3  Direction;              ///< Ray direction (normalized). Ray starts at shading surface.
    uint    LightIndex;             ///< Identifier of the source light (index in the light list), 0xFFFFFFFF if not available.
    float   SelectionPdf;           ///< Pdf of just the source light (LightIndex) selection; In contrast to 'LightSample::Pdf' which is a 'selectionPDF * solidAnglePdf'.
    float   SolidAnglePdf;          ///< Pdf with respect to solid angle at the shading point
    bool    LightSampleableByBSDF;  ///< Required for MIS vs BSDF; typically emissive and environment samples can be "seen" by BSDF, while analytic Sphere/Point/Spotlight/etc. are virtual, with non-scene representation
    bool    FromLocalDistribution;  ///< Was drawn from NEE-AT local distribution; false if global or NEE-AT not used

    static LightSample make()
    {
        LightSample ret;
        ret.Li = float3(0, 0, 0);
        // ret.Pdf = 0; 
        ret.Distance = 0;
        ret.Direction = float3(0, 0, 0);
        ret.LightIndex = 0xFFFFFFFF;
        ret.LightSampleableByBSDF = false;
        ret.SolidAnglePdf = 0;
        ret.SelectionPdf = 0;
        ret.FromLocalDistribution = false;
        return ret;
    }

    bool Valid()
    {
        return Li.x > 0 || Li.y > 0 || Li.z > 0;
    }
};

#endif // __POLYMORPHIC_LIGHT_H__