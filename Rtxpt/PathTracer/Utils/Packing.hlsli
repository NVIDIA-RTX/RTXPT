/*
* Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

// Partially based on "donut/include/donut/shaders/packing.hlsli"

#ifndef __PACKING_HLSLI__
#define __PACKING_HLSLI__

// Pack [0.0, 1.0] float to a uint of a given bit depth
#define PACK_UFLOAT_TEMPLATE(size)                      \
uint Pack_R ## size ## _UFLOAT(float r, float d = 0.5f) \
{                                                       \
    const uint mask = (1U << size) - 1U;                \
                                                        \
    return (uint)floor(r * mask + d) & mask;            \
}                                                       \
                                                        \
float Unpack_R ## size ## _UFLOAT(uint r)               \
{                                                       \
    const uint mask = (1U << size) - 1U;                \
                                                        \
    return (float)(r & mask) / (float)mask;             \
}

PACK_UFLOAT_TEMPLATE(8)
PACK_UFLOAT_TEMPLATE(10)
PACK_UFLOAT_TEMPLATE(11)
PACK_UFLOAT_TEMPLATE(16)

uint Pack_R8G8B8_UFLOAT(float3 rgb, float3 d = float3(0.5f, 0.5f, 0.5f))
{
    uint r = Pack_R8_UFLOAT(rgb.r, d.r);
    uint g = Pack_R8_UFLOAT(rgb.g, d.g) << 8;
    uint b = Pack_R8_UFLOAT(rgb.b, d.b) << 16;
    return r | g | b;
}

float3 Unpack_R8G8B8_UFLOAT(uint rgb)
{
    float r = Unpack_R8_UFLOAT(rgb);
    float g = Unpack_R8_UFLOAT(rgb >> 8);
    float b = Unpack_R8_UFLOAT(rgb >> 16);
    return float3(r, g, b);
}

uint Pack_R8G8B8A8_Gamma_UFLOAT(float4 rgba, float gamma = 2.2, float4 d = float4(0.5f, 0.5f, 0.5f, 0.5f))
{
    rgba = pow(saturate(rgba), 1.0 / gamma);
    uint r = Pack_R8_UFLOAT(rgba.r, d.r);
    uint g = Pack_R8_UFLOAT(rgba.g, d.g) << 8;
    uint b = Pack_R8_UFLOAT(rgba.b, d.b) << 16;
    uint a = Pack_R8_UFLOAT(rgba.a, d.a) << 24;
    return r | g | b | a;
}

float4 Unpack_R8G8B8A8_Gamma_UFLOAT(uint rgba, float gamma = 2.2)
{
    float r = Unpack_R8_UFLOAT(rgba);
    float g = Unpack_R8_UFLOAT(rgba >> 8);
    float b = Unpack_R8_UFLOAT(rgba >> 16);
    float a = Unpack_R8_UFLOAT(rgba >> 24);
    float4 v = float4(r, g, b, a);
    v = pow(saturate(v), gamma);
    return v;
}

// uint Pack_R11G11B10_UFLOAT(float3 rgb, float3 d = float3(0.5f, 0.5f, 0.5f))
// {
//     uint r = Pack_R11_UFLOAT(rgb.r, d.r);
//     uint g = Pack_R11_UFLOAT(rgb.g, d.g) << 11;
//     uint b = Pack_R10_UFLOAT(rgb.b, d.b) << 22;
//     return r | g | b;
// }
// 
// float3 Unpack_R11G11B10_UFLOAT(uint rgb)
// {
//     float r = Unpack_R11_UFLOAT(rgb);
//     float g = Unpack_R11_UFLOAT(rgb >> 11);
//     float b = Unpack_R10_UFLOAT(rgb >> 22);
//     return float3(r, g, b);
// }

uint Pack_R8G8B8A8_UFLOAT(float4 rgba, float4 d = float4(0.5f, 0.5f, 0.5f, 0.5f))
{
    uint r = Pack_R8_UFLOAT(rgba.r, d.r);
    uint g = Pack_R8_UFLOAT(rgba.g, d.g) << 8;
    uint b = Pack_R8_UFLOAT(rgba.b, d.b) << 16;
    uint a = Pack_R8_UFLOAT(rgba.a, d.a) << 24;
    return r | g | b | a;
}

float4 Unpack_R8G8B8A8_UFLOAT(uint rgba)
{
    float r = Unpack_R8_UFLOAT(rgba);
    float g = Unpack_R8_UFLOAT(rgba >> 8);
    float b = Unpack_R8_UFLOAT(rgba >> 16);
    float a = Unpack_R8_UFLOAT(rgba >> 24);
    return float4(r, g, b, a);
}

uint Pack_R16G16_UFLOAT(float2 rg, float2 d = float2(0.5f, 0.5f))
{
    uint r = Pack_R16_UFLOAT(rg.r, d.r);
    uint g = Pack_R16_UFLOAT(rg.g, d.g) << 16;
    return r | g;
}

float2 Unpack_R16G16_UFLOAT(uint rg)
{
    float r = Unpack_R16_UFLOAT(rg);
    float g = Unpack_R16_UFLOAT(rg >> 16);
    return float2(r, g);
}

uint Pack_R8_SNORM(float value)
{
    return int(clamp(value, -1.0, 1.0) * 127.0) & 0xff;
}

float Unpack_R8_SNORM(uint value)
{
    int signedValue = int(value << 24) >> 24;
    return clamp(float(signedValue) / 127.0, -1.0, 1.0);
}

uint Pack_RGB8_SNORM(float3 rgb)
{
    uint r = Pack_R8_SNORM(rgb.r);
    uint g = Pack_R8_SNORM(rgb.g) << 8;
    uint b = Pack_R8_SNORM(rgb.b) << 16;
    return r | g | b;
}

float3 Unpack_RGB8_SNORM(uint value)
{
    return float3(
        Unpack_R8_SNORM(value),
        Unpack_R8_SNORM(value >> 8),
        Unpack_R8_SNORM(value >> 16)
    );
}

uint Pack_RGBA8_SNORM(float4 rgb)
{
    uint r = Pack_R8_SNORM(rgb.r);
    uint g = Pack_R8_SNORM(rgb.g) << 8;
    uint b = Pack_R8_SNORM(rgb.b) << 16;
    uint a = Pack_R8_SNORM(rgb.a) << 24;
    return r | g | b | a;
}

float4 Unpack_RGBA8_SNORM(uint value)
{
    return float4(
        Unpack_R8_SNORM(value),
        Unpack_R8_SNORM(value >> 8),
        Unpack_R8_SNORM(value >> 16),
        Unpack_R8_SNORM(value >> 24)
    );
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// (R11G11B10 conversion code below taken from Miniengine's PixelPacking_R11G11B10.hlsli,  
// Copyright (c) Microsoft, MIT license, Developed by Minigraph, Author:  James Stanard; original file link:
// https://github.com/Microsoft/DirectX-Graphics-Samples/blob/master/MiniEngine/Core/Shaders/PixelPacking_R11G11B10.hlsli )
//
// The standard 32-bit HDR color format.  Each float has a 5-bit exponent and no sign bit.
uint Pack_R11G11B10_FLOAT( float3 rgb )
{
    // Clamp upper bound so that it doesn't accidentally round up to INF 
    // Exponent=15, Mantissa=1.11111
    rgb = min(rgb, asfloat(0x477C0000));  
    uint r = ((f32tof16(rgb.x) + 8) >> 4) & 0x000007FF;
    uint g = ((f32tof16(rgb.y) + 8) << 7) & 0x003FF800;
    uint b = ((f32tof16(rgb.z) + 16) << 17) & 0xFFC00000;
    return r | g | b;
}

float3 Unpack_R11G11B10_FLOAT( uint rgb )
{
    float r = f16tof32((rgb << 4 ) & 0x7FF0);
    float g = f16tof32((rgb >> 7 ) & 0x7FF0);
    float b = f16tof32((rgb >> 17) & 0x7FE0);
    return float3(r, g, b);
}

uint4   PackTwoFp32ToFp16(float4 a, float4 b)                               { return (f32tof16(clamp(a, -HLF_MAX, HLF_MAX)) << 16) | f32tof16(clamp(b, -HLF_MAX, HLF_MAX)); }
void    UnpackTwoFp32ToFp16(uint4 packed, out float4 a, out float4 b)       { a = f16tof32(packed >> 16); b = f16tof32(packed & 0xFFFF); }
uint3   PackTwoFp32ToFp16(float3 a, float3 b)                               { return (f32tof16(clamp(a, -HLF_MAX, HLF_MAX)) << 16) | f32tof16(clamp(b, -HLF_MAX, HLF_MAX)); }
void    UnpackTwoFp32ToFp16(uint3 packed, out float3 a, out float3 b)       { a = f16tof32(packed >> 16); b = f16tof32(packed & 0xFFFF); }
uint    PackTwoFp32ToFp16(float  a, float  b)                               { return (f32tof16(clamp(a, -HLF_MAX, HLF_MAX)) << 16) | f32tof16(clamp(b, -HLF_MAX, HLF_MAX)); }
void    UnpackTwoFp32ToFp16(uint packed, out float a, out float b)          { a = f16tof32(packed >> 16); b = f16tof32(packed & 0xFFFF); }

uint    PackTwoFp32ToFp16(float2 v)                                         { return PackTwoFp32ToFp16(v.x, v.y); }
float2  UnpackTwoFp32ToFp16(uint packed)                                    { float a, b; UnpackTwoFp32ToFp16(packed, a, b); return float2(a, b); }

// TODO: unify with above & cleanup

uint Fp32ToFp16(float2 v)
{
	const uint2 r = f32tof16(v);
	return (r.y << 16) | (r.x & 0xFFFF);
}

float2 Fp16ToFp32(uint r)
{
	uint2 v;
	v.x = (r & 0xFFFF);
	v.y = (r >> 16);
	return f16tof32(v);
}

uint2 Fp32ToFp16(float4 v)
{
	const uint d0 = Fp32ToFp16(v.xy);
	const uint d1 = Fp32ToFp16(v.zw);
	return uint2(d0, d1);
}

float4 Fp16ToFp32(uint2 d)
{
	const float2 d0 = Fp16ToFp32(d.x);
	const float2 d1 = Fp16ToFp32(d.y);
	return float4(d0.xy, d1.xy);
}

uint3 Fp32ToFp16(float3 a, float3 b)
{
	const uint d0 = Fp32ToFp16(a.xy);
	const uint d1 = Fp32ToFp16(float2(a.z, b.x));
	const uint d2 = Fp32ToFp16(b.yz);

	return uint3(d0, d1, d2);
}

void Fp16ToFp32(uint3 d, out float3 a, out float3 b)
{
	const float2 d0 = Fp16ToFp32(d.x);
	const float2 d1 = Fp16ToFp32(d.y);
	const float2 d2 = Fp16ToFp32(d.z);

	a = float3(d0, d1.x);
	b = float3(d1.y, d2);
}

#endif // __PACKING_HLSLI__