/*
* Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#ifndef __MATERIAL_DATA_HLSLI__ // using instead of "#pragma once" due to https://github.com/microsoft/DirectXShaderCompiler/issues/3943
#define __MATERIAL_DATA_HLSLI__

#include "../Config.h"    

static uint bitExtract(uint val, uint offset, uint bits) { return (val >> offset) & ((1u << bits) - 1u); }
static uint bitInsert(uint dst, uint offset, uint bits, uint val) { uint mask = ((1u << bits) - 1u) << offset; return (dst & ~mask) | ((val << offset) & mask); }

/** This is a host/device structure for material header data for all material types (8B).
*/
struct ShadingFlags
{
    uint packedData; // = {}; <- not supported in .hlsl ! use ShadingFlags::make()

    static const uint kNestedPriorityBits = 4;
    static const uint kLobeTypeBits = 8; // Only 6 bits needed if packing LobeType
    static const uint kPSDDominantDeltaLobeP1Bits = 4;

    // packedData bit layout
    static const uint kNestedPriorityOffset = 0;
    static const uint kLobeTypeOffset = kNestedPriorityOffset + kNestedPriorityBits;
    static const uint kThinSurfaceFlagOffset = kLobeTypeOffset + kLobeTypeBits;
    static const uint kPSDExcludeFlagOffset = kThinSurfaceFlagOffset + 1;
    static const uint kPSDBlockMotionVectorsAtSurfaceFlagOffset = kPSDExcludeFlagOffset + 1;
    static const uint kPSDDominantDeltaLobeP1Offset = kPSDBlockMotionVectorsAtSurfaceFlagOffset + 1;

    static const uint kTotalHeaderBitsX = kPSDDominantDeltaLobeP1Offset + kPSDDominantDeltaLobeP1Bits;

    /** Set the nested priority used for nested dielectrics.
    */
    void setNestedPriority(uint priority) { packedData = bitInsert(packedData, kNestedPriorityOffset, kNestedPriorityBits, priority); }

    /** Get the nested priority used for nested dielectrics.
        \return Nested priority, with 0 reserved for the highest possible priority.
    */
    uint getNestedPriority() { return bitExtract(packedData, kNestedPriorityOffset, kNestedPriorityBits); }

    /** Set active BxDF lobes.
        \param[in] activeLobes Bit mask of active lobes. See LobeType.
    */
    void setActiveLobes(uint activeLobes) { packedData = bitInsert(packedData, kLobeTypeOffset, kLobeTypeBits, activeLobes); }

    /** Get active BxDF lobes.
        \return Bit mask of active lobes. See LobeType.
    */
    uint getActiveLobes() { return bitExtract(packedData, kLobeTypeOffset, kLobeTypeBits); }

    /** Set thin surface flag.
    */
#ifdef RTXPT_MATERIAL_THIN_SURFACE
    void setThinSurface(bool thinSurface) { }
#else
    void setThinSurface(bool thinSurface) { packedData = bitInsert(packedData, kThinSurfaceFlagOffset, 1, uint(thinSurface)); }
#endif

    /** Get thin surface flag.
    */
#ifdef RTXPT_MATERIAL_THIN_SURFACE
    bool isThinSurface() { return RTXPT_MATERIAL_THIN_SURFACE; }
#else
    bool isThinSurface() { return packedData & (1u << kThinSurfaceFlagOffset); }
#endif

    void setPSDExclude(bool psdExclude) { packedData = bitInsert(packedData, kPSDExcludeFlagOffset, 1, uint(psdExclude)); }
    bool isPSDExclude()  { return packedData & (1u << kPSDExcludeFlagOffset); }

    void setPSDBlockMotionVectorsAtSurface(bool PSDBlockMotionVectorsAtSurface) { packedData = bitInsert(packedData, kPSDBlockMotionVectorsAtSurfaceFlagOffset, 1, uint(PSDBlockMotionVectorsAtSurface)); }
    bool isPSDBlockMotionVectorsAtSurface()  { return packedData & (1u << kPSDBlockMotionVectorsAtSurfaceFlagOffset); }

    void setPSDDominantDeltaLobeP1(uint psdDominantDeltaLobeP1) { packedData = bitInsert(packedData, kPSDDominantDeltaLobeP1Offset, kPSDDominantDeltaLobeP1Bits, psdDominantDeltaLobeP1); }
    uint getPSDDominantDeltaLobeP1() { return bitExtract(packedData, kPSDDominantDeltaLobeP1Offset, kPSDDominantDeltaLobeP1Bits); }

    static ShadingFlags make( ) { ShadingFlags header; header.packedData = 0; return header; }
};

#endif // __MATERIAL_DATA_HLSLI__