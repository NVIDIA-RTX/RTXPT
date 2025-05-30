/*
* Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "PathTracer/PathTracerTypes.hlsli"

#include "Bindings/ShaderResourceBindings.hlsli"

// These are used for performance testing and development; actual post-process stuff is in PostProcess.h/.cpp/.hlsl

#ifdef PP_TEST_HDR
[shader("raygeneration")]
void RayGen()
{
    uint2 pixelPos = DispatchRaysIndex().xy;

    float3 existingColor = u_ProcessedOutputColor[pixelPos].rgb;

    if ( length(float2(pixelPos.xy) - float2(800, 500)) < 100 ) // draw a circle for testing
        existingColor.z += 10;

    u_ProcessedOutputColor[pixelPos] = float4( existingColor, 1 );
}
#endif

#ifdef PP_EDGE_DETECTION
float3 LoadLDR(uint2 pixelPos)
{
    return t_LdrColorScratch[pixelPos].rgb;
}
void SaveLDR(uint2 pixelPos, float3 linearColor)
{
    u_PostTonemapOutputColor[pixelPos].rgb = LinearToSRGB(linearColor);
}
[shader("raygeneration")]
void RayGen()
{
    uint2 pixelPos = DispatchRaysIndex().xy;
    int offX = 1; int offY = 1;

	float3 s00 = LoadLDR(pixelPos + int2( -offX, -offY ));
	float3 s01 = LoadLDR(pixelPos + int2(     0, -offY ));
	float3 s02 = LoadLDR(pixelPos + int2(  offX, -offY ));
	float3 s10 = LoadLDR(pixelPos + int2( -offX,  0    ));
	float3 s12 = LoadLDR(pixelPos + int2(  offX,  0    ));
	float3 s20 = LoadLDR(pixelPos + int2( -offX,  offY ));
	float3 s21 = LoadLDR(pixelPos + int2(     0,  offY ));
	float3 s22 = LoadLDR(pixelPos + int2(  offX,  offY ));

// add reorder threads here? convert to lpfloat?
	
	float3 sobelX = s00 + 2 * s10 + s20 - s02 - 2 * s12 - s22;
	float3 sobelY = s00 + 2 * s01 + s02 - s20 - 2 * s21 - s22;

	float3 edgeSqr = (sobelX * sobelX + sobelY * sobelY);
	
    const float kThreshold = asfloat(g_MiniConst.params[0]);

	float3 edgeColor = 1.xxx-(edgeSqr > kThreshold.xxx * kThreshold.xxx);
    SaveLDR( pixelPos, saturate(edgeColor) );
}
#endif

#define CLOSEST_HIT_VARIANT( name, NoTextures, NoTransmission, OnlyDeltaLobes )     \
[shader("closesthit")] void ClosestHit##name(inout PathPayload payload : SV_RayPayload, in BuiltInTriangleIntersectionAttributes attrib) \
{   \
}

//hints: NoTextures, NoTransmission, OnlyDeltaLobes
#if 1 // 3bit 8-variant version
CLOSEST_HIT_VARIANT( 000, false, false, false );
CLOSEST_HIT_VARIANT( 001, false, false, true  );
CLOSEST_HIT_VARIANT( 010, false, true,  false );
CLOSEST_HIT_VARIANT( 011, false, true,  true  );
CLOSEST_HIT_VARIANT( 100, true,  false, false );
CLOSEST_HIT_VARIANT( 101, true,  false, true  );
CLOSEST_HIT_VARIANT( 110, true,  true,  false );
CLOSEST_HIT_VARIANT( 111, true,  true,  true  );
#endif

[shader("miss")]
void Miss(inout PathPayload payload : SV_RayPayload)
{
}

[shader("anyhit")]
void AnyHit(inout PathPayload payload, in BuiltInTriangleIntersectionAttributes attrib/* : SV_IntersectionAttributes*/)
{
}
