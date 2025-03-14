/*
* Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#ifndef __UTILS_HLSLI__ // using instead of "#pragma once" due to https://github.com/microsoft/DirectXShaderCompiler/issues/3943
#define __UTILS_HLSLI__

#if !defined(__cplusplus)
#include "Math/MathConstants.hlsli"

// TODO: remove these from here and include directly, only where needed
#include "Packing.hlsli"

// Returns a relative luminance of an input linear RGB color in the ITU-R BT.709 color space
inline float Luminance(float3 rgb)
{
    return dot(rgb, float3(0.2126f, 0.7152f, 0.0722f));
}

// Clamp .rgb by luminance
float3 LuminanceClamp(float3 signalIn, const float luminanceThreshold)
{
    float lumSig = max( 1e-7, Luminance( signalIn ) );
    if( lumSig > luminanceThreshold )
        signalIn = signalIn / lumSig * luminanceThreshold;
    return signalIn;
}

float3 Reinhard(float3 color)
{
    float luminance = max( 1e-7, Luminance(color) );
    float reinhard = luminance / (luminance + 1);
    return color * (reinhard / luminance);
}

// used for debugging, from https://www.shadertoy.com/view/llKGWG - Heat map, Created by joshliebe in 2016-Oct-15
float3 GradientHeatMap( float greyValue )
{
    greyValue = saturate(greyValue);
    float3 heat; heat.r = smoothstep(0.5, 0.8, greyValue);
    if(greyValue >= 0.90)
    	heat.r *= (1.1 - greyValue) * 5.0;
	if(greyValue > 0.7)
		heat.g = smoothstep(1.0, 0.7, greyValue);
	else
		heat.g = smoothstep(0.0, 0.7, greyValue);
	heat.b = smoothstep(1.0, 0.0, greyValue);          
    if(greyValue <= 0.3)
    	heat.b *= greyValue / 0.3;     
    return heat;
}

float3 ColorFromHash( uint hash )
{
    return Unpack_R11G11B10_FLOAT(hash);
}

// *************************************************************************************************************************************
// Octahedral normal encoding/decoding - see https://knarkowicz.wordpress.com/2014/04/16/octahedron-normal-vector-encoding/
float2 OctWrap(float2 v)
{
	return (1.0 - abs(v.yx)) * select(v.xy >= 0.0, 1.0, -1.0);
}
float2 Encode_Oct(float3 n)
{
	n /= (abs(n.x) + abs(n.y) + abs(n.z));
	n.xy = n.z >= 0.0 ? n.xy : OctWrap(n.xy);
	n.xy = n.xy * 0.5 + 0.5;
	return n.xy;
}
float3 Decode_Oct(float2 f)
{
	f = f * 2.0 - 1.0;

	// https://twitter.com/Stubbesaurus/status/937994790553227264
	float3 n = float3(f.x, f.y, 1.0 - abs(f.x) - abs(f.y));
	float t = saturate(-n.z);
	n.xy += select(n.xy >= 0.0, -t, t);
	return normalize(n);
}
// Same as Encode_Oct but with encoding into two uint16-s packed into one uint32
uint NDirToOctUnorm32(float3 n)
{
    float2 p = Encode_Oct(n);
    p = saturate(p.xy * 0.5 + 0.5);
    return uint(p.x * 0xfffe) | (uint(p.y * 0xfffe) << 16);
}
// Same as Decode_Oct but with decoding from two uint16-s packed into one uint32
float3 OctToNDirUnorm32(uint pUnorm)
{
    float2 p;
    p.x = saturate(float(pUnorm & 0xffff) / 0xfffe);
    p.y = saturate(float(pUnorm >> 16) / 0xfffe);
    p = p * 2.0 - 1.0;
    return Decode_Oct(p);
}
// *************************************************************************************************************************************

#if (RTXPT_LP_TYPES_USE_16BIT_PRECISION != 0)
    typedef float16_t       lpfloat; 
    typedef float16_t2      lpfloat2;
    typedef float16_t3      lpfloat3;
    typedef float16_t4      lpfloat4;
    typedef float16_t3x3    lpfloat3x3;
    typedef uint16_t        lpuint; 
    typedef uint16_t2       lpuint2;
    typedef uint16_t3       lpuint3;
    typedef uint16_t4       lpuint4;
#else
    typedef float           lpfloat;
    typedef float2          lpfloat2;
    typedef float3          lpfloat3;
    typedef float4          lpfloat4;
    typedef float3x3        lpfloat3x3;
    typedef uint            lpuint; 
    typedef uint2           lpuint2;
    typedef uint3           lpuint3;
    typedef uint4           lpuint4;
#endif

float   sq(float x)     { return x * x; }
float2  sq(float2 x)    { return x * x; }
float3  sq(float3 x)    { return x * x; }
float4  sq(float4 x)    { return x * x; }

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// ---- 8< ---- GLSL Number Printing - @P_Malin ---- 8< ----
// Smaller Number Printing - @P_Malin
// Creative Commons CC0 1.0 Universal (CC-0)

// Feel free to modify, distribute or use in commercial code, just don't hold me liable for anything bad that happens!
// If you use this code and want to give credit, that would be nice but you don't have to.

// I first made this number printing code in https://www.shadertoy.com/view/4sf3RN
// It started as a silly way of representing digits with rectangles.
// As people started actually using this in a number of places I thought I would try to condense the 
// useful function a little so that it can be dropped into other shaders more easily,
// just snip between the perforations below.
// Also, the licence on the previous shader was a bit restrictive for utility code.
//
// Disclaimer: The values printed may not be accurate!
// Accuracy improvement for fractional values taken from TimoKinnunen https://www.shadertoy.com/view/lt3GRj

// ---- 8< ---- GLSL Number Printing - @P_Malin ---- 8< ----
// Creative Commons CC0 1.0 Universal (CC-0) 
// https://www.shadertoy.com/view/4sBSWW
float _DigitBin( const int x )
{
    return x==0?480599.0:x==1?139810.0:x==2?476951.0:x==3?476999.0:x==4?350020.0:x==5?464711.0:x==6?464727.0:x==7?476228.0:x==8?481111.0:x==9?481095.0:0.0;
}
float glsl_mod(float x, float y)
{
    return x - y * floor(x/y);
}
float ShaderDrawFloat( float2 vStringCoords, float fValue, float fMaxDigits, float fDecimalPlaces )
{       
    if ((vStringCoords.y < 0.0) || (vStringCoords.y >= 1.0)) return 0.0;
    
    bool bNeg = ( fValue < 0.0 );
	fValue = abs(fValue);
    
	float fLog10Value = log2(abs(fValue)) / log2(10.0);
	float fBiggestIndex = max(floor(fLog10Value), 0.0);
	float fDigitIndex = fMaxDigits - floor(vStringCoords.x);
	float fCharBin = 0.0;
	if(fDigitIndex > (-fDecimalPlaces - 1.01)) {
		if(fDigitIndex > fBiggestIndex) {
			if((bNeg) && (fDigitIndex < (fBiggestIndex+1.5))) fCharBin = 1792.0;
		} else {		
			if(fDigitIndex == -1.0) {
				if(fDecimalPlaces > 0.0) fCharBin = 2.0;
			} else {
                float fReducedRangeValue = fValue;
                if(fDigitIndex < 0.0) { fReducedRangeValue = frac( fValue ); fDigitIndex += 1.0; }
				float fDigitValue = (abs(fReducedRangeValue / (pow(10.0, fDigitIndex))));
                fCharBin = _DigitBin(int(floor(glsl_mod(fDigitValue, 10.0))));
			}
        }
	}
    return floor(glsl_mod((fCharBin / pow(2.0, floor(frac(vStringCoords.x) * 4.0) + (floor(vStringCoords.y * 5.0) * 4.0))), 2.0));
}
float ShaderDrawFloat(const in float2 pixelCoord, const in float2 textCoord, const in float2 vFontSize, const in float fValue, const in float fMaxDigits, const in float fDecimalPlaces)
{
    float2 vStringCharCoords = (pixelCoord.xy - textCoord) / vFontSize;
    vStringCharCoords.y = 1-vStringCharCoords.y;
    
    return ShaderDrawFloat( vStringCharCoords, fValue, fMaxDigits, fDecimalPlaces );
}
// ---- 8< -------- 8< -------- 8< -------- 8< ----

float ShaderDrawCrosshair(const in float2 pixelCoord, const in float2 crossCoord, const in float size, const in float thickness)
{
    float2 incoords = abs( pixelCoord.xy - crossCoord );
    return all(incoords<size) && any(incoords<thickness) && !all(incoords<thickness);
}

#endif // !defined(__cplusplus)

// Morton coding/decoding based on https://fgiesen.wordpress.com/2022/09/09/morton-codes-addendum/ 
inline uint Morton16BitEncode(uint x, uint y) // x and y are expected to be max 8 bit
{
  uint temp = (x & 0xff) | ((y & 0xff)<<16);
  temp = (temp ^ (temp <<  4)) & 0x0f0f0f0f;
  temp = (temp ^ (temp <<  2)) & 0x33333333;
  temp = (temp ^ (temp <<  1)) & 0x55555555;
  return ((temp >> 15) | temp) & 0xffff;
}
inline uint2 Morton16BitDecode(uint morton) // morton is expected to be max 16 bit
{
  uint temp=(morton&0x5555)|((morton&0xaaaa)<<15);
  temp=(temp ^ (temp>>1))&0x33333333;
  temp=(temp ^ (temp>>2))&0x0f0f0f0f;
  temp^=temp>>4;
  return uint2( 0xff & temp, 0xff & (temp >> 16) );
}

// Generic Tiled Swizzled Addressing
#if 0 // use linear storage (change requires cpp and shader recompile)
inline uint GenericTSComputeLineStride(const uint imageWidth, const uint imageHeight)
{
    return imageWidth;
}
inline uint GenericTSComputePlaneStride(const uint imageWidth, const uint imageHeight)
{
    return imageWidth * imageHeight;
}
inline uint GenericTSComputeStorageElementCount(const uint imageWidth, const uint imageHeight, const uint imageDepth)
{ 
    return imageDepth * GenericTSComputePlaneStride(imageWidth, imageHeight);
}
inline uint GenericTSPixelToAddress(const uint2 pixelPos, const uint planeIndex, const uint lineStride, const uint planeStride)
{
    uint yInPlane = pixelPos.y;
    return yInPlane * lineStride + pixelPos.x + planeIndex * planeStride;
}
inline uint3 GenericTSAddressToPixel(const uint address, const uint lineStride, const uint planeStride)
{
    uint planeIndex = address / planeStride;
    uint localAddress = address % planeStride;
    uint2 pixelPos;
    pixelPos.x = localAddress % lineStride;
    pixelPos.y = localAddress / lineStride;
    return uint3(pixelPos, planeIndex);
}
#else // use tiled swizzled storage - this is not yet completely optimized but noticeably faster than scanline addressing
#define TS_USE_MORTON 1
#define TS_TILE_SIZE 8 // seems to be the sweet spot
#define TS_TILE_MASK (TS_TILE_SIZE*TS_TILE_SIZE-1)
inline uint GenericTSComputeLineStride(const uint imageWidth, const uint imageHeight)
{
    uint tileCountX = (imageWidth + TS_TILE_SIZE - 1) / TS_TILE_SIZE;
    return tileCountX * TS_TILE_SIZE;
}
inline uint GenericTSComputePlaneStride(const uint imageWidth, const uint imageHeight)
{
    uint tileCountY = (imageHeight + TS_TILE_SIZE - 1) / TS_TILE_SIZE;
    return GenericTSComputeLineStride(imageWidth, imageHeight) * tileCountY * TS_TILE_SIZE;
}
inline uint GenericTSComputeStorageElementCount(const uint imageWidth, const uint imageHeight, const uint imageDepth)
{ 
    return imageDepth * GenericTSComputePlaneStride(imageWidth, imageHeight);
}
inline uint GenericTSPixelToAddress(const uint2 pixelPos, const uint planeIndex, const uint lineStride, const uint planeStride) // <- pass ptConstants or StablePlane constants in...
{
    // coords within tile
    uint xInTile = pixelPos.x % TS_TILE_SIZE;
    uint yInTile = pixelPos.y % TS_TILE_SIZE;

#if TS_USE_MORTON
    uint tilePixelIndex = Morton16BitEncode(xInTile, yInTile);
#else // else simple scanline
    uint tilePixelIndex = xInTile + TS_TILE_SIZE * yInTile;
#endif
    uint tileBaseX = pixelPos.x - xInTile;
    uint tileBaseY = pixelPos.y - yInTile;
    return tileBaseX * TS_TILE_SIZE + tileBaseY * lineStride + tilePixelIndex + planeIndex * planeStride;
}
inline uint3 GenericTSAddressToPixel(const uint address, const uint lineStride, const uint planeStride) // <- pass ptConstants or StablePlane constants in...
{
    const uint planeIndex = address / planeStride;
    const uint localAddress = address % planeStride;
#if TS_USE_MORTON
    uint2 pixelPos = Morton16BitDecode(localAddress & TS_TILE_MASK);
#else // else simple scanline
    uint tilePixelIndex = localAddress % (TS_TILE_SIZE*TS_TILE_SIZE);
    uint2 pixelPos = uint2( tilePixelIndex % TS_TILE_SIZE, tilePixelIndex / TS_TILE_SIZE ); // linear
#endif
    uint maskedLocalAddressBase = (localAddress & ~TS_TILE_MASK)/TS_TILE_SIZE;
    pixelPos += uint2( maskedLocalAddressBase % lineStride, (maskedLocalAddressBase / lineStride) * TS_TILE_SIZE );
    return uint3(pixelPos.x, pixelPos.y, planeIndex);
}
#undef TS_USE_MORTON
#undef TS_TILE_SIZE
#undef TS_TILE_MASK
#endif

#if defined(__cplusplus) && defined(_DEBUG)
// useful test for custom addressing with random texture sizes
inline bool RunTSAddressingTest()
{
    bool res = true;
    for (int i = 0; i < 1000; i++)
    {
        uint2 randsiz = { (uint)(std::rand() % 2000) + 1, (uint)(std::rand() % 2000) + 1 };
        uint3 randp = { (uint)std::rand() % randsiz.x, (uint)std::rand() % randsiz.y, (uint)std::rand() % 3 };
        uint lineStride = GenericTSComputeLineStride(randsiz.x, randsiz.y);
        uint planeStride = GenericTSComputePlaneStride(randsiz.x, randsiz.y);
        uint addr = GenericTSPixelToAddress(randp.xy(), randp.z, lineStride, planeStride);
        uint3 randpt = GenericTSAddressToPixel(addr, lineStride, planeStride);
        assert(randp.x == randpt.x && randp.y == randpt.y && randp.z == randpt.z);
        res &= randp.x == randpt.x && randp.y == randpt.y && randp.z == randpt.z;
    }
    return res;
}
inline static bool g_TSAddressingTest = RunTSAddressingTest();
#endif

enum class MISHeuristic : uint32_t
{
    Balance     = 0,    ///< Balance heuristic.
    PowerTwo    = 1,    ///< Power heuristic (exponent = 2.0).
    PowerExp    = 2,    ///< Power heuristic (variable exponent).
    Disabled    = 3,    ///< Fixed constant (0.5 for two way MIS, 1/3.0 for 3-way MIS, etc.)
};

/** Evaluates the currently configured heuristic for multiple importance sampling (MIS).
    \param[in] n0 Number of samples taken from the first sampling strategy.
    \param[in] p0 Pdf for the first sampling strategy.
    \param[in] n1 Number of samples taken from the second sampling strategy.
    \param[in] p1 Pdf for the second sampling strategy.
    \return Weight for the contribution from the first strategy (p0).
*/
inline float EvalMIS(MISHeuristic heuristic, float n0, float p0, float n1, float p1)
{
    float retVal = 0.0;
    switch (heuristic)
    {
    case MISHeuristic::Balance:
    {
        // Balance heuristic
        float q0 = n0 * p0;
        float q1 = n1 * p1;
        retVal = q0 / (q0 + q1);
    } break;
    case MISHeuristic::PowerTwo:
    {
        // Power two heuristic
        float q0 = (n0 * p0) * (n0 * p0);
        float q1 = (n1 * p1) * (n1 * p1);
        retVal = q0 / (q0 + q1);
    } break;
    case MISHeuristic::PowerExp:
    {
        const float kMISPowerExponent = 1.5;    // <- TODO: get it from PathTracerParams
        // Power exp heuristic
        float q0 = pow(n0 * p0, kMISPowerExponent);
        float q1 = pow(n1 * p1, kMISPowerExponent);
        retVal = q0 / (q0 + q1);
    } break;
    case MISHeuristic::Disabled: retVal = 1.0/2.0; break;
    }
    return saturate(retVal); // only [0, 1] is valid
}

inline float EvalMIS(MISHeuristic heuristic, float n0, float p0, float n1, float p1, float n2, float p2)
{
    float retVal = 0.0;
    switch (heuristic)
    {
    case MISHeuristic::Balance:
    {
        // Balance heuristic
        float q0 = n0 * p0;
        float q1 = n1 * p1;
        float q2 = n2 * p2;
        retVal = q0 / (q0 + q1 + q2);
    } break;
    case MISHeuristic::PowerTwo:
    {
        // Power two heuristic
        float q0 = (n0 * p0) * (n0 * p0);
        float q1 = (n1 * p1) * (n1 * p1);
        float q2 = (n2 * p2) * (n2 * p2);
        retVal = q0 / (q0 + q1 + q2);
    } break;
    case MISHeuristic::PowerExp:
    {
        const float kMISPowerExponent = 1.5;    // <- TODO: get it from PathTracerParams
        // Power exp heuristic
        float q0 = pow(n0 * p0, kMISPowerExponent);
        float q1 = pow(n1 * p1, kMISPowerExponent);
        float q2 = pow(n2 * p2, kMISPowerExponent);
        retVal = q0 / (q0 + q1 + q2);
    } break;
    case MISHeuristic::Disabled: retVal = 1.0/3.0; break;
    }
    return saturate(retVal); // only [0, 1] is valid
}

inline bool RelativelyEqual( float a, float b, float epsilon = 5e-5f )
{
    return (abs(a-b) / (abs(a)+abs(b)+epsilon)) < epsilon;
}

inline uint AbsDiffUINT( uint a, uint b )
{
    return ( a < b )?( b - a ):( a - b );
}

#if !defined(__cplusplus)
// http://h14s.p5r.org/2012/09/0x5f3759df.html, [Drobot2014a] Low Level Optimizations for GCN, https://blog.selfshadow.com/publications/s2016-shading-course/activision/s2016_pbs_activision_occlusion.pdf slide 63
float FastSqrt( float x )
{
    return asfloat( 0x1fbd1df5 + ( asint( x ) >> 1 ) );
}
// input [-1, 1] and output [0, PI], from https://seblagarde.wordpress.com/2014/12/01/inverse-trigonometric-functions-gpu-optimization-for-amd-gcn-architecture/
float FastACos( float inX )
{ 
    const float PI = 3.141593;
    const float HALF_PI = 1.570796;
    float x = abs(inX); 
    float res = -0.156583 * x + HALF_PI; 
    res *= FastSqrt(1.0 - x); 
    return (inX >= 0) ? res : PI - res; 
}
// input [-1, 1] and output [0, PI], from https://seblagarde.wordpress.com/2014/12/01/inverse-trigonometric-functions-gpu-optimization-for-amd-gcn-architecture/
lpfloat FastACosLp( lpfloat inX )
{ 
    const lpfloat PI = 3.141593;
    const lpfloat HALF_PI = 1.570796;
    lpfloat x = abs(inX); 
    lpfloat res = -0.156583 * x + HALF_PI; 
    res *= (lpfloat)FastSqrt(1.0 - (float)x); // TODO: do a lpfloat version of a FastSqrt?
    return (inX >= 0) ? res : PI - res; 
}
#endif

#endif // __UTILS_HLSLI__
