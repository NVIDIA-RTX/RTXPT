/*
* Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#ifndef __BINDING_DATA_TYPES_HLSLI__ // using instead of "#pragma once" due to https://github.com/microsoft/DirectXShaderCompiler/issues/3943
#define __BINDING_DATA_TYPES_HLSLI__

struct PackedPathTracerSurfaceData
{
    float3 _posW;
    uint _faceNCorrected;                   // fp16[3]
    uint2 _mtl;                             // Falcor::MaterialDefinition
    uint2 _V;                               // fp16[3]

    // misc (mostly subset of struct ShadingData)
    uint _T;                                // octFp16
    uint _N;                                // octFp16
    uint _viewDepth_planeHash_isEmpty_frontFacing;	// (fp16) | u15 | u1

    // StandardBSDFData (All fields nessesary)
    uint _diffuse;							// R11G11B10_FLOAT
    uint _specular;							// R11G11B10_FLOAT
    uint _roughnessMetallicEta;				// R11G11B10_FLOAT
    uint _transmission;						// R11G11B10_FLOAT
    uint _diffuseSpecularTransmission;		// fp16 | fp16
};

#endif // __BINDING_DATA_TYPES_HLSLI__