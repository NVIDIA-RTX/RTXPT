/*
* Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#ifndef __SUB_INSTANCE_DATA_H__
#define __SUB_INSTANCE_DATA_H__

#if !defined(__cplusplus) // not needed in the port so far
#pragma pack_matrix(row_major) // matrices below are expected in row_major
#else
using namespace donut::math;
#endif

// per-instance-geometry data (avoids 1 layer of indirection that requires reading from instance and geometry buffers)
struct SubInstanceData  // could have been called GeometryInstanceData but that's already used in Falcor codebase
{
    static const int Flags_AlphaTested      	= (1<<16);
    static const int Flags_ExcludeFromNEE    	= (1<<17);

    static const int Flags_AlphaOffsetMask      = (0xFF000000);
    static const int Flags_AlphaOffsetOffset    = (24);

    uint FlagsAndSERSortKey;
    uint GlobalGeometryIndex_MaterialPTDataIndex;    // index into t_GeometryData and t_GeometryDebugData in higher 16 bits, index in materialPT list in lower 16 bits
    uint AlphaTextureIndex;                     // index into t_BindlessTextures
    //float AlphaCutoff;                  // could be packed into 8 bits and kept in FlagsAndSERSortKey
    uint EmissiveLightMappingOffset;            // if emissive mesh, index of the first light (see LightsBaker); will be 0xFFFFFFFF if triangle not emissive!

    float AlphaCutoff()                         { return (FlagsAndSERSortKey>>Flags_AlphaOffsetOffset) / 255.0f; }
};

#endif // __SUB_INSTANCE_DATA_H__