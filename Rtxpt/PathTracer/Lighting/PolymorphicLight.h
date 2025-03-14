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

static const uint kPolymorphicLightTypeShift = 24;
static const uint kPolymorphicLightTypeMask = 0xf;
static const uint kPolymorphicLightShapingEnableBit = 1 << 28;
static const uint kPolymorphicLightIesProfileEnableBit = 1 << 29;
static const float kPolymorphicLightMinLog2Radiance = -8.f;
static const float kPolymorphicLightMaxLog2Radiance = 40.f;

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
};

// Note: two potential optimizations w.r.t. to memory access; at the moment PolymorphicLightInfo is 48 bytes which makes
// it misaligned with cache line size of 64 bytes.
// We could a.) enlarge to 64 bytes, and use the space for less packing/unpacking arithmetic or b.) store light shaping
// data in a separate list as it's rarely used anyway (and already branched out on flag stored in ColorTypeAndFlags)

// Stores shared light information (type) and specific light information; see PolymorphicLight.hlsli for encoding format
struct PolymorphicLightInfo
{
    // uint4[0]
    float3  Center;
    uint    ColorTypeAndFlags;  // RGB8 + uint8 (see the kPolymorphicLight... constants above)

    // uint4[1]
    uint    Direction1;         // oct-encoded
    uint    Direction2;         // oct-encoded
    uint    Scalars;            // 2x float16
    uint    LogRadiance;        // uint16 | empty slot

    bool    HasLightShaping()                   { return (ColorTypeAndFlags & kPolymorphicLightShapingEnableBit) != 0; }
};

struct PolymorphicLightInfoEx
{
    // uint4[2] shaping data
    uint    IesProfileIndex;
    uint    PrimaryAxis;                // oct-encoded
    uint    CosConeAngleAndSoftness;    // 2x float16
    uint    UniqueID;                   // light hash IDs - used only for debug view coloring and validation

    static PolymorphicLightInfoEx empty() { PolymorphicLightInfoEx ret; ret.IesProfileIndex = 0; ret.PrimaryAxis = 0; ret.CosConeAngleAndSoftness = 0; ret.UniqueID = 0; return ret; }
};

struct PolymorphicLightInfoFull
{
    PolymorphicLightInfo    Base;
    PolymorphicLightInfoEx  Extended;
    static PolymorphicLightInfoFull make(PolymorphicLightInfo base) { PolymorphicLightInfoFull ret; ret.Base = base; ret.Extended = PolymorphicLightInfoEx::empty(); return ret; }
    static PolymorphicLightInfoFull make(PolymorphicLightInfo base, PolymorphicLightInfoEx extended) { PolymorphicLightInfoFull ret; ret.Base = base; ret.Extended = extended; return ret; }
};

#endif // __POLYMORPHIC_LIGHT_H__