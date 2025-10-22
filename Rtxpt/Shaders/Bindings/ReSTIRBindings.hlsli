/*
* Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#ifndef __RESTIR_GI_BINDINGS_HLSLI__ // using instead of "#pragma once" due to https://github.com/microsoft/DirectXShaderCompiler/issues/3943
#define __RESTIR_GI_BINDINGS_HLSLI__

#include "BindingDataTypes.hlsli"
#include <donut/shaders/binding_helpers.hlsli>

// ReSTIR GI resources
RWTexture2D<float4>                     u_SecondarySurfacePositionNormal: register(u60);
RWTexture2D<float4>                     u_SecondarySurfaceRadiance      : register(u61);


void ReSTIRGI_StoreSecondarySurfacePositionAndNormal(uint2 pixPos, float3 worldPos, float3 normal)
{
    u_SecondarySurfacePositionNormal[pixPos] = float4(worldPos, asfloat(NDirToOctUnorm32(normal)));
}

void ReSTIRGI_StorePrimarySurfaceScatterPdf(uint2 pixPos, float scatterPdf)
{
    u_SecondarySurfaceRadiance[pixPos].a = scatterPdf;
}

void ReSTIRGI_AddSecondarySurfaceRadiance(uint2 pixPos, float3 secondaryRadiance)
{
    u_SecondarySurfaceRadiance[pixPos].rgb += secondaryRadiance;
}

void ReSTIRGI_Clear( uint2 pixPos )
{
    u_SecondarySurfaceRadiance[pixPos].rgba = float4(0,0,0,0);
    ReSTIRGI_StoreSecondarySurfacePositionAndNormal( pixPos, float3(0,0,0), float3(0,0,0) );
}

bool ReSTIRGI_IsEmpty( uint2 pixPos )
{
    return u_SecondarySurfaceRadiance[pixPos].a == 0;
}



#endif // #ifndef __RESTIR_GI_BINDINGS_HLSLI__
