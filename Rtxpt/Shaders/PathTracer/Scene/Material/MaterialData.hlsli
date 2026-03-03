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

#include "../../Config.h"    

// TODO: Replace by bit packing functions
#define EXTRACT_BITS(bits, offset, value) (((value) >> (offset)) & ((1u << (bits)) - 1u))
#define PACK_BITS(bits, offset, flags, value) ((((value) & ((1u << (bits)) - 1u)) << (offset)) | ((flags) & (~(((1u << (bits)) - 1u) << (offset)))))
#define PACK_BITS_UNSAFE(bits, offset, flags, value) (((value) << (offset)) | ((flags) & (~(((1u << (bits)) - 1u) << (offset)))))

/** This is a host/device structure for material header data for all material types (8B).
*/
struct MaterialHeader
{
    uint packedData; // = {}; <- not supported in .hlsl ! use MaterialHeader::make()

    static const uint kNestedPriorityBits = 4;
    static const uint kLobeTypeBits = 8; // Only 6 bits needed if packing LobeType
    static const uint kPSDDominantDeltaLobeP1Bits = 4;

    // packedData.x bit layout
    static const uint kNestedPriorityOffset = 0;
    static const uint kLobeTypeOffset = kNestedPriorityOffset + kNestedPriorityBits;
    static const uint kThinSurfaceFlagOffset = kLobeTypeOffset + kLobeTypeBits;
    static const uint kPSDExcludeFlagOffset = kThinSurfaceFlagOffset + 1;
    static const uint kPSDBlockMotionVectorsAtSurfaceFlagOffset = kPSDExcludeFlagOffset + 1;
    static const uint kPSDDominantDeltaLobeP1Offset = kPSDBlockMotionVectorsAtSurfaceFlagOffset + 1;

    static const uint kTotalHeaderBitsX = kPSDDominantDeltaLobeP1Offset + kPSDDominantDeltaLobeP1Bits;

    /** Set the nested priority used for nested dielectrics.
    */
    void setNestedPriority(uint32_t priority) { packedData.x = PACK_BITS(kNestedPriorityBits, kNestedPriorityOffset, packedData.x, priority); }

    /** Get the nested priority used for nested dielectrics.
        \return Nested priority, with 0 reserved for the highest possible priority.
    */
    uint getNestedPriority() { return EXTRACT_BITS(kNestedPriorityBits, kNestedPriorityOffset, packedData.x); }

    /** Set active BxDF lobes.
        \param[in] activeLobes Bit mask of active lobes. See LobeType.
    */
    void setActiveLobes(uint activeLobes) { packedData.x = PACK_BITS(kLobeTypeBits, kLobeTypeOffset, packedData.x, activeLobes); }

    /** Get active BxDF lobes.
        \return Bit mask of active lobes. See LobeType.
    */
    uint getActiveLobes() { return EXTRACT_BITS(kLobeTypeBits, kLobeTypeOffset, packedData.x); }

    /** Set thin surface flag.
    */
#ifdef RTXPT_MATERIAL_THIN_SURFACE
    void setThinSurface(bool thinSurface) { }
#else
    void setThinSurface(bool thinSurface) { packedData.x = PACK_BITS(1, kThinSurfaceFlagOffset, packedData.x, thinSurface ? 1 : 0); }
#endif

    /** Get thin surface flag.
    */
#ifdef RTXPT_MATERIAL_THIN_SURFACE
    bool isThinSurface() { return RTXPT_MATERIAL_THIN_SURFACE; }
#else
    bool isThinSurface() { return packedData.x & (1u << kThinSurfaceFlagOffset); }
#endif

    void setPSDExclude(bool psdExclude) { packedData.x = PACK_BITS(1, kPSDExcludeFlagOffset, packedData.x, psdExclude ? 1 : 0); }
    bool isPSDExclude()  { return packedData.x & (1u << kPSDExcludeFlagOffset); }
    
    void setPSDBlockMotionVectorsAtSurface(bool PSDBlockMotionVectorsAtSurface) { packedData.x = PACK_BITS(1, kPSDBlockMotionVectorsAtSurfaceFlagOffset, packedData.x, PSDBlockMotionVectorsAtSurface ? 1 : 0); }
    bool isPSDBlockMotionVectorsAtSurface()  { return packedData.x & (1u << kPSDBlockMotionVectorsAtSurfaceFlagOffset); }

    void setPSDDominantDeltaLobeP1(uint psdDominantDeltaLobeP1) { packedData.x = PACK_BITS(kPSDDominantDeltaLobeP1Bits, kPSDDominantDeltaLobeP1Offset, packedData.x, (uint)psdDominantDeltaLobeP1); }
    uint getPSDDominantDeltaLobeP1()             { return EXTRACT_BITS(kPSDDominantDeltaLobeP1Bits, kPSDDominantDeltaLobeP1Offset, packedData.x); }

    static MaterialHeader make( ) { MaterialHeader header; header.packedData = 0; return header; }
};

#endif // __MATERIAL_DATA_HLSLI__