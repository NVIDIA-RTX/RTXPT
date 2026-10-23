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

#define SUBINSTANCEDATA_EXTENDED 1

// Per-instance-geometry data (avoids 1 layer of indirection that requires reading from instance and geometry buffers)
// Note: the size of this structure has measurable impact on CPU-side performance with many (100k+) (sub)instances
struct SubInstanceData  // could have been called GeometryInstanceData but that's already used in Falcor codebase
{
    static const int MaxGlobalGeometryCount     = (1<<18);
    static const int GlobalGeometryIndexMask    = MaxGlobalGeometryCount - 1;

    static const int Flags_AlphaTested      	= (1<<18);
    static const int Flags_ExcludeFromNEE    	= (1<<19);
    static const int Flags_Dummy0               = (1<<20); // free up to (1<<23)

    static const int Flags_AlphaOffsetMask      = (0xFF000000);
    static const int Flags_AlphaOffsetOffset    = (24);

    uint GlobalGeometryIndex_FlagsAndAlphaCutoff;   // index into t_GeometryData and t_GeometryDebugData in GlobalGeometryIndexMask, then Flags_*, then the alpha cutoff in the top 8 bits
    uint AlphaTextureIndex_PTMaterialDataIndex;     // alpha texture bindless index in higher 16 bits, index in PTMaterial list in lower 16 bits
    uint EmissiveLightMappingOffset;                // if emissive mesh, index of the first light (see LightsBaker); will be 0xFFFFFFFF if triangles are not emissive!
    uint AnalyticProxyLightIndex;                   // if standing in as an analytic light proxy
    
#if SUBINSTANCEDATA_EXTENDED
    uint IndexBufferIndex_VertexBufferIndex;        // (allows skipping loading of t_GeometryData for alpha testing) packed 16 bits each
    uint IndexOffset;                               // (allows skipping loading of t_GeometryData for alpha testing) 
    uint TexCoord1Offset;                           // (allows skipping loading of t_GeometryData for alpha testing) 
    uint padding0;
#endif

    static float DecodeAlphaCutoff(uint quantizedCutoff)
    {
        // The rounded float reciprocal is shared by CPU bake inputs and runtime shaders.
        return float(quantizedCutoff) * 0.003921568859368563f;
    }

    uint  GlobalGeometryIndex()                 { return GlobalGeometryIndex_FlagsAndAlphaCutoff & GlobalGeometryIndexMask; }
    float AlphaCutoff()                         { return DecodeAlphaCutoff(GlobalGeometryIndex_FlagsAndAlphaCutoff >> Flags_AlphaOffsetOffset); }
    uint  AlphaTextureIndex()                   { return AlphaTextureIndex_PTMaterialDataIndex >> 16; }

#if defined(__cplusplus) // not needed in the port so far
    static uint QuantizeAlphaCutoff(float alphaCutoff)
    {
        return (uint)(donut::math::clamp(alphaCutoff, 0.0f, 1.0f) * 255.0f + 0.5f);
    }

    static SubInstanceData Default()
    {
        SubInstanceData ret{};
        ret.AnalyticProxyLightIndex     = 0xFFFFFFFF; //NEEAT_INVALID_LIGHT_INDEX;
        ret.EmissiveLightMappingOffset  = 0xFFFFFFFF; //NEEAT_INVALID_LIGHT_INDEX;
        return ret;
    }
#endif
};

#endif // __SUB_INSTANCE_DATA_H__