/*
* Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#ifndef __HIT_INFO_HLSLI__ // using instead of "#pragma once" due to https://github.com/microsoft/DirectXShaderCompiler/issues/3943
#define __HIT_INFO_HLSLI__

#include "../Config.h"

#include "HitInfoType.hlsli"
#include "SceneTypes.hlsli"

/** Ray hit information.

    'HitInfo' is a polymorphic container for storing ray hit information.
    'PackedHitInfo' holds the data in 'HitInfo'.
    The upper bits of the first uint are used to store the type of the hit.
    A zero initialized value represent hit type 'None'.

    HitInfo stores data for one of the following type specific structs:
    TriangleHit, DisplacedTriangleHit, CurveHit, SDFGridHit, VolumeHit

    TriangleHit, DisplacedTriangleHit and CurveHit all store the hit point in terms of
    an instance ID and a primitive index, together with barycentrics.

    DisplacedTriangleHit additionally stores the displacement offset.

    To use HitInfo, the host needs to set the following defines:

    - HIT_INFO_DEFINES              Marks that the rest of the defines are available.
    - HIT_INFO_USE_COMPRESSION      Use compressed format (64 bits instead of 128 bits).
    - HIT_INFO_TYPE_BITS            Bits needed to encode the hit type.
    - HIT_INFO_INSTANCE_ID_BITS     Bits needed to encode the instance ID of the hit.
    - HIT_INFO_PRIMITIVE_INDEX_BITS Bits needed to encode the primitive index of the hit.

    If a bit size define is zero, no bits are needed (the field has only one possible value = 0).
*/

#if HIT_INFO_USE_COMPRESSION
typedef uint2 PackedHitInfo;
#define PACKED_HIT_INFO_ZERO uint2(0, 0)
#else
typedef uint4 PackedHitInfo;
#define PACKED_HIT_INFO_ZERO uint4(0, 0, 0, 0)
#endif

bool IsValid(const PackedHitInfo packedHitInfo) { return packedHitInfo[0] != 0; }

struct GeometryHit;
struct TriangleHit;

// Since HitInfo is used by types below, has to be declared first in hlsl.
/** Polymorphic hit information type.
*/
struct HitInfo
{
#ifdef HIT_INFO_DEFINES
    static const uint kTypeBits = HIT_INFO_TYPE_BITS;
    static const uint kInstanceIDBits = HIT_INFO_INSTANCE_ID_BITS;
    static const uint kPrimitiveIndexBits = HIT_INFO_PRIMITIVE_INDEX_BITS;
#else
    static const uint kTypeBits = 1;
    static const uint kInstanceIDBits = 1;
    static const uint kPrimitiveIndexBits = 1;
#endif

    static const uint kTypeOffset = 32u - kTypeBits;
    static const uint kInstanceIDOffset = kPrimitiveIndexBits;

    static const uint kHeaderBits = kTypeBits + kInstanceIDBits + kPrimitiveIndexBits;

    PackedHitInfo data;

    /** Initialize empty (invalid) hit info.
    */
    void __init();
    static HitInfo make();

    /** Initialize hit info from a packed hit info.
        \param[in] packed Packed hit information.
    */
    void __init(const PackedHitInfo packed);
    static HitInfo make(const PackedHitInfo packed);

    /** Initialize hit info from a triangle hit.
        \param[in] triangleHit Triangle hit information.
    */
    void __init(const TriangleHit triangleHit);
    static HitInfo make(const TriangleHit triangleHit);

    /** Return true if object represents a valid hit.
    */
    bool isValid();

    /** Return hit type.
    */
    HitType getType();

    /** Return the triangle hit.
        Only valid if type is HitType::Triangle.
    */
    TriangleHit getTriangleHit();

    /** Return the packed hit info.
        \return Packed hit info.
    */
    PackedHitInfo getData();

    static void packHeader(inout PackedHitInfo packed, const HitType type);

    static void packHeader(inout PackedHitInfo packed, const HitType type, const GeometryInstanceID instanceID, const uint primitiveIndex);
    static void unpackHeader(const PackedHitInfo packed, out GeometryInstanceID instanceID, out uint primitiveIndex);
};

/** Geometry hit information (base class).
*/
struct GeometryHit
{
    GeometryInstanceID instanceID;
    uint primitiveIndex;
    float2 barycentrics;

    /** Return the barycentric weights.
    */
    float3 getBarycentricWeights()
    {
        return float3(1.f - barycentrics.x - barycentrics.y, barycentrics.x, barycentrics.y);
    }
};

/** Triangle hit information.

    Encoding (without compression):
    | header  | barycentrics.x | barycentrics.y |
    | 64 bits | 32 bit float   | 32 bit float   |

    Encoding (with compression):
    | header     | barycentrics.x | barycentrics.y |
    | 32 bits    | 16 bit unorm   | 16 bit unorm   |
*/
struct TriangleHit : GeometryHit
{
    void __init(const PackedHitInfo packed)
    {
        HitInfo::unpackHeader(packed, instanceID, primitiveIndex);
#if HIT_INFO_USE_COMPRESSION
        barycentrics = unpackUnorm2x16(packed[1]);
#else
        barycentrics.x = asfloat(packed[2]);
        barycentrics.y = asfloat(packed[3]);
#endif
    }
    static TriangleHit make(const PackedHitInfo packed)  { TriangleHit ret; ret.__init(packed); return ret; }
    static TriangleHit make(const uint instanceIndex, const uint geometryIndex, const uint primitiveIndex, float2 barycentrics) 
    { 
        TriangleHit ret; 
        ret.instanceID      = GeometryInstanceID::make( InstanceIndex(), GeometryIndex() );
        ret.primitiveIndex  = primitiveIndex;
        ret.barycentrics    = barycentrics;
        return ret;
    }

    PackedHitInfo pack()
    {
        PackedHitInfo packed = PACKED_HIT_INFO_ZERO;
        HitInfo::packHeader(packed, HitType::Triangle, instanceID, primitiveIndex);
#if HIT_INFO_USE_COMPRESSION
        packed[1] = packUnorm2x16_unsafe(barycentrics);
#else
        packed[2] = asuint(barycentrics.x);
        packed[3] = asuint(barycentrics.y);
#endif
        return packed;
    }
};

// Definition for HitInfo functions
void HitInfo::__init()
{
    data = PACKED_HIT_INFO_ZERO;
}
static HitInfo HitInfo::make()                                  
{ 
    HitInfo ret; ret.__init(); return ret; 
}

/** Initialize hit info from a packed hit info.
    \param[in] packed Packed hit information.
*/
void HitInfo::__init(const PackedHitInfo packed)
{
    data = packed;
}
static HitInfo HitInfo::make(const PackedHitInfo packed)
{ 
    HitInfo ret; ret.__init(packed); return ret; 
}

/** Initialize hit info from a triangle hit.
    \param[in] triangleHit Triangle hit information.
*/
void HitInfo::__init(const TriangleHit triangleHit)
{
    data = triangleHit.pack();
}
static HitInfo HitInfo::make(const TriangleHit triangleHit)
{ 
    HitInfo ret; ret.__init(triangleHit); return ret; 
}

/** Return true if object represents a valid hit.
*/
bool HitInfo::isValid()
{
    return getType() != HitType::None;
}

/** Return hit type.
*/
HitType HitInfo::getType()
{
    return (HitType)(data[0] >> kTypeOffset);
}

/** Return the triangle hit.
    Only valid if type is HitType::Triangle.
*/
TriangleHit HitInfo::getTriangleHit()
{
    return TriangleHit::make(data);
}

/** Return the packed hit info.
    \return Packed hit info.
*/
PackedHitInfo HitInfo::getData()
{
    return data;
}

static void HitInfo::packHeader(inout PackedHitInfo packed, const HitType type)
{
    packed[0] = ((uint)type) << kTypeOffset;
}

static void HitInfo::packHeader(inout PackedHitInfo packed, const HitType type, const GeometryInstanceID instanceID, const uint primitiveIndex)
{
    if (kHeaderBits <= 32)
    {
        packed[0] = (((uint)type) << kTypeOffset) | (instanceID.data << kInstanceIDOffset) | primitiveIndex;
    }
    else
    {
        packed[0] = (((uint)type) << kTypeOffset) | instanceID.data;
        packed[1] = primitiveIndex;
    }
}

static void HitInfo::unpackHeader(const PackedHitInfo packed, out GeometryInstanceID instanceID, out uint primitiveIndex)
{
    if (kHeaderBits <= 32)
    {
        instanceID.data = (packed[0] >> kInstanceIDOffset) & ((1u << kInstanceIDBits) - 1u);
        primitiveIndex = packed[0] & ((1u << kPrimitiveIndexBits) - 1u);
    }
    else
    {
        instanceID.data = packed[0] & ((1u << kInstanceIDBits) - 1u);
        primitiveIndex = packed[1];
    }
}


#endif // __HIT_INFO_HLSLI__