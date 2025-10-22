/*
* Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#ifndef __OMM_GEOMETRY_DEBUG_DATA_HLSLI__ // using instead of "#pragma once" due to https://github.com/microsoft/DirectXShaderCompiler/issues/3943
#define __OMM_GEOMETRY_DEBUG_DATA_HLSLI__


struct GeometryDebugData
{
    unsigned int ommArrayDataBufferIndex;
    unsigned int ommArrayDataBufferOffset;
    unsigned int ommDescArrayBufferIndex;
    unsigned int ommDescArrayBufferOffset;

    unsigned int ommIndexBufferIndex;
    unsigned int ommIndexBufferOffset;
    unsigned int ommIndexBuffer16Bit; // (bool) 16 or 32 bit indices.
    unsigned int _pad0;
};

#endif // __OMM_GEOMETRY_DEBUG_DATA_HLSLI__
