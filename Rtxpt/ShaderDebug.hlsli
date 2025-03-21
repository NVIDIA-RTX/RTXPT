/*
* Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#ifndef __SHADER_DEBUG_HLSLI__
#define __SHADER_DEBUG_HLSLI__

#if !defined(__cplusplus) // not needed in the port so far
#pragma pack_matrix(row_major)
#include "PathTracer/Utils/Math/MathHelpers.hlsli"
#endif

// text print based on https://therealmjp.github.io/posts/hlsl-printf/
// NOTE: at the moment, DXC SPIRV compile errors make formatting for DebugPrint impossible, but arguments will still be displayed

struct ShaderDebugHeader
{
    int         PrintItemsCounterInBytes;
    int         TrianglesCounterInBytes;
    int         Padding0;
    int         Padding1;

    // Draw Indirect arguments (for drawing triangles)
    int         VertexCountPerInstance; // should be fixed to 3
    int         InstanceCount;          // increasing by 1 with every triangle
    int         StartVertexLocation;    // fixed to 0
    int         StartInstanceLocation;  // fixed to 0

    // Last view matrix
#if !defined(__cplusplus)
    float4x4    WorldViewProjectionMatrix;
#else
    float       WorldViewProjectionMatrix[16];
#endif
};

#define SHADER_DEBUG_BUFFER_UAV_INDEX               (125)               // make sure it matches u_ShaderDebugBuffer register below...
#define SHADER_DEBUG_VIZ_TEXTURE_UAV_INDEX          (126)               // make sure it matches u_ShaderDebugBuffer register below
#define SHADER_DEBUG_HEADER_IN_BYTES                (96)                // header containing counter for debug prints and triangles and etc
#define SHADER_DEBUG_PRINT_BUFFER_IN_BYTES          (256*1024)          // going above this will make Debug Output in debugger unresponsive
#define SHADER_PRINTF_MAX_DEBUG_PRINT_ARGS          (8)                 // if changing, make sure to add more AppendArgs
#define SHADER_DEBUG_TRIANGLE_BUFFER_IN_BYTES       (200*1024*1024)     // <shrug> this is a lot - consider compressing ShaderDebugTriangle
#define SHADER_DEBUG_BUFFER_IN_BYTES                (SHADER_DEBUG_HEADER_IN_BYTES+SHADER_DEBUG_PRINT_BUFFER_IN_BYTES+SHADER_DEBUG_TRIANGLE_BUFFER_IN_BYTES)
#define SHADER_DEBUG_BUFFER_IN_BYTES_NO_TRIANGLES   (SHADER_DEBUG_HEADER_IN_BYTES+SHADER_DEBUG_PRINT_BUFFER_IN_BYTES)
//#define SHADER_DEBUG_MAX_TRIANGLES              (SHADER_DEBUG_TRIANGLE_BUFFER_IN_BYTES / (4*4*4) ) //  SHADER_DEBUG_TRIANGLE_BUFFER_IN_BYTES / sizeof(ShaderDebugTriangle))


#if defined(__cplusplus)
static_assert( sizeof(ShaderDebugHeader) == SHADER_DEBUG_HEADER_IN_BYTES );
#endif

// Buffer layout in memory is:
//     ShaderDebugHeader with counters and etc. (total SHADER_DEBUG_HEADER_IN_BYTES)
//     debug prints (SHADER_DEBUG_PRINT_BUFFER_IN_BYTES)
//     debug triangles SHADER_DEBUG_TRIANGLE_BUFFER_IN_BYTES

struct ShaderDebugPrintItem
{
    int     NumBytes;
    int     StringSize;
    int     NumArgs;
};

enum class ShaderDebugArgCode
{
    DebugPrint_ErrorType = 0,
    DebugPrint_Uint = 1,
    DebugPrint_Uint2,
    DebugPrint_Uint3,
    DebugPrint_Uint4,
    DebugPrint_Int,
    DebugPrint_Int2,
    DebugPrint_Int3,
    DebugPrint_Int4,
    DebugPrint_Float,
    DebugPrint_Float2,
    DebugPrint_Float3,
    DebugPrint_Float4,

    NumDebugPrintArgCodes,
};

#if !defined(__cplusplus)

struct ShaderDebugTriangle
{
    float4  A;
    float4  B;
    float4  C;
    float4  Color;
};

// see SampleNull.hlsl for explanation of below
#ifdef register
#undef register
#define REGISTER_WAS_DEFINED
#endif
RWByteAddressBuffer u_ShaderDebugBuffer : register(u125);      // make sure it matches SHADER_DEBUG_BUFFER_UAV_INDEX
RWTexture2D<float4> u_ShaderDebugVizTextureBuffer : register(u126);      // make sure it matches SHADER_DEBUG_VIZ_TEXTURE_UAV_INDEX
#ifdef REGISTER_WAS_DEFINED
#define register
#endif

#if defined(__HLSL_VERSION) && (__HLSL_VERSION >= 2021) 

namespace ShaderDebug
{
    template<typename T, uint N> uint StrLen(T str[N])
    {
        // Includes the null terminator
        return N;
    }

    template<typename T> uint CharToUint(in T c)
    {
        if(c == " "[0]) return 32; 
        if(c == "!"[0]) return 33; 
        if(c == "\""[0]) return 34; 
        if(c == "#"[0]) return 35; 
        if(c == "$"[0]) return 36; 
        if(c == "%"[0]) return 37; 
        if(c == "&"[0]) return 38; 
        if(c == "'"[0]) return 39; 
        if(c == "("[0]) return 40; 
        if(c == ")"[0]) return 41; 
        if(c == "*"[0]) return 42; 
        if(c == "+"[0]) return 43; 
        if(c == ","[0]) return 44; 
        if(c == "-"[0]) return 45; 
        if(c == "."[0]) return 46; 
        if(c == "/"[0]) return 47; 
        if(c == "0"[0]) return 48; 
        if(c == "1"[0]) return 49; 
        if(c == "2"[0]) return 50; 
        if(c == "3"[0]) return 51; 
        if(c == "4"[0]) return 52; 
        if(c == "5"[0]) return 53; 
        if(c == "6"[0]) return 54; 
        if(c == "7"[0]) return 55; 
        if(c == "8"[0]) return 56; 
        if(c == "9"[0]) return 57; 
        if(c == ":"[0]) return 58; 
        if(c == ";"[0]) return 59; 
        if(c == "<"[0]) return 60; 
        if(c == "="[0]) return 61; 
        if(c == ">"[0]) return 62; 
        if(c == "?"[0]) return 63; 
        if(c == "@"[0]) return 64; 
        if(c == "A"[0]) return 65; 
        if(c == "B"[0]) return 66; 
        if(c == "C"[0]) return 67; 
        if(c == "D"[0]) return 68; 
        if(c == "E"[0]) return 69; 
        if(c == "F"[0]) return 70; 
        if(c == "G"[0]) return 71; 
        if(c == "H"[0]) return 72; 
        if(c == "I"[0]) return 73; 
        if(c == "J"[0]) return 74; 
        if(c == "K"[0]) return 75; 
        if(c == "L"[0]) return 76; 
        if(c == "M"[0]) return 77; 
        if(c == "N"[0]) return 78; 
        if(c == "O"[0]) return 79; 
        if(c == "P"[0]) return 80; 
        if(c == "Q"[0]) return 81; 
        if(c == "R"[0]) return 82; 
        if(c == "S"[0]) return 83; 
        if(c == "T"[0]) return 84; 
        if(c == "U"[0]) return 85; 
        if(c == "V"[0]) return 86; 
        if(c == "W"[0]) return 87; 
        if(c == "X"[0]) return 88; 
        if(c == "Y"[0]) return 89; 
        if(c == "Z"[0]) return 90; 
        if(c == "["[0]) return 91; 
        if(c == "\\"[0]) return 92;
        if(c == "]"[0]) return 93; 
        if(c == "^"[0]) return 94; 
        if(c == "_"[0]) return 95; 
        if(c == "`"[0]) return 96; 
        if(c == "a"[0]) return 97; 
        if(c == "b"[0]) return 98; 
        if(c == "c"[0]) return 99; 
        if(c == "d"[0]) return 100;
        if(c == "e"[0]) return 101;
        if(c == "f"[0]) return 102;
        if(c == "g"[0]) return 103;
        if(c == "h"[0]) return 104;
        if(c == "i"[0]) return 105;
        if(c == "j"[0]) return 106;
        if(c == "k"[0]) return 107;
        if(c == "l"[0]) return 108;
        if(c == "m"[0]) return 109;
        if(c == "n"[0]) return 110;
        if(c == "o"[0]) return 111;
        if(c == "p"[0]) return 112;
        if(c == "q"[0]) return 113;
        if(c == "r"[0]) return 114;
        if(c == "s"[0]) return 115;
        if(c == "t"[0]) return 116;
        if(c == "u"[0]) return 117;
        if(c == "v"[0]) return 118;
        if(c == "w"[0]) return 119;
        if(c == "x"[0]) return 120;
        if(c == "y"[0]) return 121;
        if(c == "z"[0]) return 122;
        if(c == "{"[0]) return 123;
        if(c == "|"[0]) return 124;
        if(c == "}"[0]) return 125;
        if(c == "~"[0]) return 126;
        return 63; // return '?' if unrecognized
    }

    struct DebugPrinter
    {
        static const uint BufferSize = 256;
        static const uint BufferSizeInBytes = BufferSize * sizeof(uint);
        uint InternalBuffer[BufferSize];
        uint ByteCount;
        uint StringSize;
        uint ArgCount;

        void Init()
        {
            for(uint i = 0; i < BufferSize; ++i)
                InternalBuffer[i] = 0;
            ByteCount = 0;
            StringSize = 0;
            ArgCount = 0;
        }

        uint CurrBufferIndex()
        {
            return ByteCount / 4;
        }

        uint CurrBufferShift()
        {
            return (ByteCount % 4) * 8;
        }

        void AppendChar(uint c)
        {
            if(ByteCount < BufferSizeInBytes)
            {
                InternalBuffer[CurrBufferIndex()] |= ((c & 0xFF) << CurrBufferShift());
                ByteCount += 1;
            }
        }

        template<typename T, uint N> void AppendArgWithCode(ShaderDebugArgCode code, T arg[N])
        {
            if(ByteCount + sizeof(arg) > BufferSizeInBytes)
                return;

            if(ArgCount >= SHADER_PRINTF_MAX_DEBUG_PRINT_ARGS)
                return;

            AppendChar((uint)code);
            for(uint elem = 0; elem < N; ++elem)
            {
                for(uint b = 0; b < sizeof(T); ++b)
                {
                    AppendChar(asuint(arg[elem]) >> (b * 8));
                }
            }

            ArgCount += 1;
        }

        void AppendArg(uint x)
        {
            uint a[1] = { x };
            AppendArgWithCode(ShaderDebugArgCode::DebugPrint_Uint, a);
        }

        void AppendArg(uint2 x)
        {
            uint a[2] = { x.x, x.y };
            AppendArgWithCode(ShaderDebugArgCode::DebugPrint_Uint2, a);
        }

        void AppendArg(uint3 x)
        {
            uint a[3] = { x.x, x.y, x.z };
            AppendArgWithCode(ShaderDebugArgCode::DebugPrint_Uint3, a);
        }

        void AppendArg(uint4 x)
        {
            uint a[4] = { x.x, x.y, x.z, x.w };
            AppendArgWithCode(ShaderDebugArgCode::DebugPrint_Uint4, a);
        }

        void AppendArg(int x)
        {
            int a[1] = { x };
            AppendArgWithCode(ShaderDebugArgCode::DebugPrint_Int, a);
        }

        void AppendArg(int2 x)
        {
            int a[2] = { x.x, x.y };
            AppendArgWithCode(ShaderDebugArgCode::DebugPrint_Int2, a);
        }

        void AppendArg(int3 x)
        {
            int a[3] = { x.x, x.y, x.z };
            AppendArgWithCode(ShaderDebugArgCode::DebugPrint_Int3, a);
        }

        void AppendArg(int4 x)
        {
            int a[4] = { x.x, x.y, x.z, x.w };
            AppendArgWithCode(ShaderDebugArgCode::DebugPrint_Int4, a);
        }


        void AppendArg(float x)
        {
            float a[1] = { x };
            AppendArgWithCode(ShaderDebugArgCode::DebugPrint_Float, a);
        }

        void AppendArg(float2 x)
        {
            float a[2] = { x.x, x.y };
            AppendArgWithCode(ShaderDebugArgCode::DebugPrint_Float2, a);
        }

        void AppendArg(float3 x)
        {
            float a[3] = { x.x, x.y, x.z };
            AppendArgWithCode(ShaderDebugArgCode::DebugPrint_Float3, a);
        }

        void AppendArg(float4 x)
        {
            float a[4] = { x.x, x.y, x.z, x.w };
            AppendArgWithCode(ShaderDebugArgCode::DebugPrint_Float4, a);
        }

        void AppendArgs()
        {
        }

        template<typename T0> void AppendArgs(T0 arg0)
        {
            AppendArg(arg0);
        }

        template<typename T0, typename T1> void AppendArgs(T0 arg0, T1 arg1)
        {
            AppendArg(arg0);
            AppendArg(arg1);
        }

        template<typename T0, typename T1, typename T2> void AppendArgs(T0 arg0, T1 arg1, T2 arg2)
        {
            AppendArg(arg0);
            AppendArg(arg1);
            AppendArg(arg2);
        }

        template<typename T0, typename T1, typename T2, typename T3> void AppendArgs(T0 arg0, T1 arg1, T2 arg2, T3 arg3)
        {
            AppendArg(arg0);
            AppendArg(arg1);
            AppendArg(arg2);
            AppendArg(arg3);
        }

        template<typename T0, typename T1, typename T2, typename T3, typename T4> void AppendArgs(T0 arg0, T1 arg1, T2 arg2, T3 arg3, T4 arg4)
        {
            AppendArg(arg0);
            AppendArg(arg1);
            AppendArg(arg2);
            AppendArg(arg3);
            AppendArg(arg4);
        }

        template<typename T0, typename T1, typename T2, typename T3, typename T4, typename T5> void AppendArgs(T0 arg0, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5)
        {
            AppendArg(arg0);
            AppendArg(arg1);
            AppendArg(arg2);
            AppendArg(arg3);
            AppendArg(arg4);
            AppendArg(arg5);
        }

        template<typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename T6> void AppendArgs(T0 arg0, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6)
        {
            AppendArg(arg0);
            AppendArg(arg1);
            AppendArg(arg2);
            AppendArg(arg3);
            AppendArg(arg4);
            AppendArg(arg5);
            AppendArg(arg6);
        }

        // note: if extending beyond 6 args, SHADER_PRINTF_MAX_DEBUG_PRINT_ARGS has to be raised as well
        template<typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7> void AppendArgs(T0 arg0, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7)
        {
            AppendArg(arg0);
            AppendArg(arg1);
            AppendArg(arg2);
            AppendArg(arg3);
            AppendArg(arg4);
            AppendArg(arg5);
            AppendArg(arg6);
            AppendArg(arg7);
        }


        void Commit()
        {
            // Round up to the next multiple of 4 since we work with 4-byte alignment for each print
            ByteCount = ((ByteCount + 3) / 4) * 4;

            RWByteAddressBuffer printBuffer = u_ShaderDebugBuffer; // ResourceDescriptorHeap[debugInfo.PrintBuffer];

            // Increment the atomic counter to allocate space to store the bytes
            const uint numBytesToWrite = ByteCount + sizeof(ShaderDebugPrintItem);
            uint offset = 0;
            const uint counterOffset = 0;
            const uint dataOffset = SHADER_DEBUG_HEADER_IN_BYTES;
            printBuffer.InterlockedAdd(counterOffset, numBytesToWrite, offset);

            // Account for the atomic counter at the beginning of the buffer
            offset += dataOffset;

            if((offset + numBytesToWrite) > (SHADER_DEBUG_HEADER_IN_BYTES+SHADER_DEBUG_PRINT_BUFFER_IN_BYTES)) // debugInfo.PrintBufferSize)
                return;

            // Store the header
            ShaderDebugPrintItem header;
            header.NumBytes = ByteCount;
            header.StringSize = StringSize;
            header.NumArgs = ArgCount;

            printBuffer.Store<ShaderDebugPrintItem>(offset, header);
            offset += sizeof(ShaderDebugPrintItem);

            // Store the buffer data
            for(uint i = 0; i < ByteCount / 4; ++i)
                printBuffer.Store(offset + (i * sizeof(uint)), InternalBuffer[i]);
        }

    };

} // namespace ShaderDebug

#ifdef ENABLE_DEBUG_PRINT

#define DEBUG_PRINT_DEFINED         1

// strings disabled for now, see https://github.com/microsoft/hlsl-specs/issues/279 and https://github.com/microsoft/hlsl-specs/issues/245
#if 1 // defined(SPIRV)
#define DebugPrint(str, ...) do {                            \
    ShaderDebug::DebugPrinter printer;                       \
    printer.Init();                                          \
    printer.StringSize = printer.ByteCount;                  \
    printer.AppendArgs(__VA_ARGS__);                         \
    printer.Commit();                                        \
} while(false)
#else
#define DebugPrint(str, ...) do {                            \
    ShaderDebug::DebugPrinter printer;                       \
    printer.Init();                                          \
    const uint strLen = ShaderDebug::StrLen(str)-1;          \
    for(uint i = 0; i < strLen; ++i)                         \
        printer.AppendChar(ShaderDebug::CharToUint(str[i])); \
    printer.StringSize = printer.ByteCount;                  \
    printer.AppendArgs(__VA_ARGS__);                         \
    printer.Commit();                                        \
} while(false)
#endif
#endif

static const float cMagicTriangleMarker   = 42.0f;
static const float cMagicLineMarker       = 7.0f;

void DebugTriangle(float3 a, float3 b, float3 c, float4 color)
{
    RWByteAddressBuffer debugBuffer = u_ShaderDebugBuffer; // ResourceDescriptorHeap[debugInfo.PrintBuffer];

    // Increment the atomic counter to allocate space to store the bytes
    const uint numBytesToWrite = sizeof(ShaderDebugTriangle);
    uint offset = 0;
    const uint counterOffset = sizeof(uint);
    const uint dataOffset = SHADER_DEBUG_HEADER_IN_BYTES+SHADER_DEBUG_PRINT_BUFFER_IN_BYTES;
    debugBuffer.InterlockedAdd(counterOffset, numBytesToWrite, offset);

    if((offset + numBytesToWrite) > SHADER_DEBUG_TRIANGLE_BUFFER_IN_BYTES)
        return;

    // if enough space in buffer, increase the number of instances
    const uint instanceCountOffset = sizeof(uint) * 5;
    debugBuffer.InterlockedAdd(instanceCountOffset, 1);

    offset += dataOffset;

    ShaderDebugTriangle tri;
    tri.A = float4(a, cMagicTriangleMarker);
    tri.B = float4(b, cMagicTriangleMarker);
    tri.C = float4(c, cMagicTriangleMarker);
    tri.Color = color;

    debugBuffer.Store<ShaderDebugTriangle>(offset, tri);
}

void DebugLine(float3 a, float3 b, float4 color)
{
    RWByteAddressBuffer debugBuffer = u_ShaderDebugBuffer; // ResourceDescriptorHeap[debugInfo.PrintBuffer];

    // Increment the atomic counter to allocate space to store the bytes
    const uint numBytesToWrite = sizeof(ShaderDebugTriangle);
    uint offset = 0;
    const uint counterOffset = sizeof(uint);
    const uint dataOffset = SHADER_DEBUG_HEADER_IN_BYTES+SHADER_DEBUG_PRINT_BUFFER_IN_BYTES;
    debugBuffer.InterlockedAdd(counterOffset, numBytesToWrite, offset);

    if((offset + numBytesToWrite) > SHADER_DEBUG_TRIANGLE_BUFFER_IN_BYTES)
        return;

    // if enough space in buffer, increase the number of instances
    const uint instanceCountOffset = sizeof(uint) * 5;
    debugBuffer.InterlockedAdd(instanceCountOffset, 1);

    offset += dataOffset;

    ShaderDebugTriangle tri;
    tri.A = float4(a, cMagicLineMarker);
    tri.B = float4(b, cMagicLineMarker);
    tri.C = float4(b, cMagicLineMarker);
    tri.Color = color;

    debugBuffer.Store<ShaderDebugTriangle>(offset, tri);
}

// 2D version - note that FLT_MAX indicates that coords are in screen space; also possible to mix one in world and another in screen
void DebugLine(float2 a, float2 b, float4 color)
{
    DebugLine( float3(a, FLT_MAX), float3(b, FLT_MAX), color );
}

void DebugCross(float3 a, float size, float4 color)  
{
    float3 tl = {-size, -size, 0};
    float3 tr = {+size, -size, 0};
    float3 bl = {-size, +size, 0};
    float3 br = {+size, +size, 0};

    DebugTriangle( a + tl, a + tr, a + bl, color ); 
    DebugTriangle( a + tr, a + br, a + bl, color ); 

    DebugTriangle( a + tl.yzx, a + tr.yzx, a + bl.yzx, color ); 
    DebugTriangle( a + tr.yzx, a + br.yzx, a + bl.yzx, color ); 
                                                             
    DebugTriangle( a + tl.zxy, a + tr.zxy, a + bl.zxy, color ); 
    DebugTriangle( a + tr.zxy, a + br.zxy, a + bl.zxy, color ); 
}

void DebugSphere(float3 a, float size, float4 fillColor, float4 lineColor, int tess = 7)
{
    // draw a tessellated grid in equal area octahedral and convert to dir
    // TODO: this should be faster: https://mathproofs.blogspot.com/2005/07/mapping-cube-to-sphere.html
    for( int x = 0; x < tess; x++ )
        for( int y = 0; y < tess; y++ )
        {
            float2 c00 = float2(x, y) / float2(tess.xx);
            float2 c10 = float2(x+1, y) / float2(tess.xx);
            float2 c01 = float2(x, y+1) / float2(tess.xx);
            float2 c11 = float2(x+1, y+1) / float2(tess.xx);

            float3 tl = size * oct_to_ndir_equal_area_unorm(c00);
            float3 tr = size * oct_to_ndir_equal_area_unorm(c10);
            float3 bl = size * oct_to_ndir_equal_area_unorm(c01);
            float3 br = size * oct_to_ndir_equal_area_unorm(c11);
            
            if( length(tl-br) > length(tr-bl) )
            {
                if ( fillColor.a > 0 )
                {
                    DebugTriangle( a+tl, a+tr, a+bl, fillColor ); 
                    DebugTriangle( a+tr, a+br, a+bl, fillColor );
                }
                if ( lineColor.a > 0 )
                {
                    DebugLine( a+tl, a+tr, lineColor ); 
                    DebugLine( a+tr, a+bl, lineColor ); 
                    DebugLine( a+tl, a+bl, lineColor ); 
                    DebugLine( a+tr, a+br, lineColor ); 
                }
            }
            else
            {
                if ( fillColor.a > 0 )
                {
                    DebugTriangle( a+tl, a+br, a+bl, fillColor ); 
                    DebugTriangle( a+tl, a+br, a+tr, fillColor );
                }
                if ( lineColor.a > 0 )
                {
                    DebugLine( a+tl, a+br, lineColor ); 
                    DebugLine( a+br, a+bl, lineColor ); 
                    DebugLine( a+tl, a+bl, lineColor ); 
                    DebugLine( a+tl, a+tr, lineColor ); 
                }
            }
        }
}

void DebugPixel(uint2 screenPos, float4 color)
{
    u_ShaderDebugVizTextureBuffer[screenPos] = color;
}


// these could go into a separate file
#ifdef DRAW_TRIANGLES_SHADERS

void main_vs( in uint vertexID : SV_VertexID, in uint instanceID : SV_InstanceID, out float4 outPos : SV_Position, out float4 outCol : COLOR, out float4 outProjPos : TEXCOORD0, out float3 outBarys : TEXCOORD1 )
{
    uint triangleIndex = instanceID;
    uint vertexIndex = vertexID;

    const uint dataOffset = SHADER_DEBUG_HEADER_IN_BYTES+SHADER_DEBUG_PRINT_BUFFER_IN_BYTES + triangleIndex * sizeof(ShaderDebugTriangle);

    float4 worldPos = u_ShaderDebugBuffer.Load<float4>(dataOffset + vertexIndex*4*4); 
    if( worldPos.w != cMagicTriangleMarker )
    {
        outPos = outCol = outProjPos = float4(0,0,0,0); outBarys = float3(0,0,0);
        return;
    }

    worldPos.w = 1.0;
    outCol = u_ShaderDebugBuffer.Load<float4>(dataOffset + 3*4*4);

    float4x4 worldToClip; // = u_ShaderDebugBuffer.Load<float4x4>(8*4); // <- ignores #pragma row_major in Vulkan
    worldToClip[0] = u_ShaderDebugBuffer.Load<float4>(8*4+0*16);
    worldToClip[1] = u_ShaderDebugBuffer.Load<float4>(8*4+1*16);
    worldToClip[2] = u_ShaderDebugBuffer.Load<float4>(8*4+2*16);
    worldToClip[3] = u_ShaderDebugBuffer.Load<float4>(8*4+3*16);

	outPos = mul(worldPos, worldToClip);
    outProjPos = outPos;

    outBarys = float3( vertexIndex == 0, vertexIndex == 1, vertexIndex == 2 );
}

Texture2D<float> t_Depth        : register(t0);
void main_ps( in float4 i_pos : SV_Position, in float4 i_col : COLOR, in float4 projPos: TEXCOORD0, in float3 barys : TEXCOORD1, out float4 o_color : SV_Target0 )
{
    uint2 upos = uint2(i_pos.xy);
    float depth = t_Depth[upos];

    bool behind = depth > (projPos.z/projPos.w*1.00001);
    //bool checkerboard = (upos.x%2)==(upos.y%2);
    //if (behind && checkerboard)
    //    discard;

    float4 color = float4( lerp(i_col.rgb, i_col.rgb*0.7+0.2, behind), i_col.a * ((behind)?(0.25):(1)));

    float3 onEdge3 = min( barys, 1-barys );

    float onEdge = min( onEdge3.x, min( onEdge3.y, onEdge3.z ) );

    float delta = abs(ddx(onEdge)) + abs(ddy(onEdge));
    
    float wireframeThickness = 0.7;

    onEdge = smoothstep(0, delta*wireframeThickness, onEdge);

    float wireframe = (1-onEdge);

    // add checkerboard to wireframe
    wireframe *= ((uint(i_pos.x)+uint(i_pos.y)) % 2 == 0);

	o_color = lerp( color, float4(0.8,1.0,0.8,1), wireframe );
}

#endif // #ifdef DRAW_TRIANGLES_SHADERS

#ifdef DRAW_LINES_SHADERS
Texture2D<float>    t_Depth             : register(t0);
Texture2D<float4>   t_DebugVizOutput    : register(t1);

void main_vs( in uint vertexID : SV_VertexID, in uint instanceID : SV_InstanceID, out float4 outPos : SV_Position, out float4 outCol : COLOR, out float4 outProjPos : TEXCOORD0 )
{
    uint triangleIndex = instanceID;
    uint vertexIndex = vertexID;

    const uint dataOffset = SHADER_DEBUG_HEADER_IN_BYTES+SHADER_DEBUG_PRINT_BUFFER_IN_BYTES + triangleIndex * sizeof(ShaderDebugTriangle);

    float4 worldPos = u_ShaderDebugBuffer.Load<float4>(dataOffset + vertexIndex*4*4); 
    if( worldPos.w != cMagicLineMarker )
    {
        outPos = outCol = outProjPos = float4(0,0,0,0);
        return;
    }
    worldPos.w = 1.0;
    outCol = u_ShaderDebugBuffer.Load<float4>(dataOffset + 3*4*4);

    if( worldPos.z == FLT_MAX )
    {
        uint width, height;
        t_DebugVizOutput.GetDimensions(width, height);
        outPos = float4( ((worldPos.x + 0.5) / float(width))*2.0 - 1.0, ((worldPos.y + 0.5) / float(height))*(-2.0) + 1.0, 0.5, 1 );
    }
    else
    {
        float4x4 worldToClip; // = u_ShaderDebugBuffer.Load<float4x4>(8*4); // <- ignores #pragma row_major in Vulkan
        worldToClip[0] = u_ShaderDebugBuffer.Load<float4>(8*4+0*16);
        worldToClip[1] = u_ShaderDebugBuffer.Load<float4>(8*4+1*16);
        worldToClip[2] = u_ShaderDebugBuffer.Load<float4>(8*4+2*16);
        worldToClip[3] = u_ShaderDebugBuffer.Load<float4>(8*4+3*16);
	    outPos = mul(worldPos, worldToClip);
    }

    outProjPos = outPos;

    //outBarys = float3( vertexIndex == 0, vertexIndex == 1, vertexIndex == 2 );
}

void main_ps( in float4 i_pos : SV_Position, in float4 i_col : COLOR, in float4 projPos: TEXCOORD0, out float4 o_color : SV_Target0 )
{
    uint2 upos = uint2(i_pos.xy);
    float depth = t_Depth[upos];

    bool behind = depth > (projPos.z/projPos.w*1.00001);
    //bool checkerboard = (upos.x%2)==(upos.y%2);
    //if (behind && checkerboard)
    //    discard;

	o_color = float4( lerp(i_col.rgb, i_col.rgb*0.7+0.2, behind), i_col.a * ((behind)?(0.25):(1)));
}

#endif // #ifdef DRAW_LINES_SHADERS

#if defined(BLEND_DEBUG_BUFFER)
//#include "PathTracer/SampleGenerators.hlsli"
//#include "PathTracer/Utils/Utils.hlsli"
Texture2D<float>    t_Depth             : register(t0);
Texture2D<float4>   t_DebugVizOutput    : register(t1);
float4 main( in float4 pos : SV_Position, in float2 uv : UV ) : SV_Target0
{
#if 0 // example of full screen test for something! (in this case testing the precision of Encode_Oct/Decode_Oct with fp16)
    SampleGenerator sampleGenerator = SampleGenerator::make( uint2(pos.xy), 0, 0 );
    float3 bah = sampleNext3D(sampleGenerator);
    bah = normalize(bah);
    float2 enc = Encode_Oct(bah);
    uint encui = PackTwoFp32ToFp16( enc.x, enc.y );
    UnpackTwoFp32ToFp16( encui, enc.x, enc.y );
    float3 bahd = Decode_Oct(enc);
    return float4( abs(bah-bahd) * float3(1, 100, 10000), 1 );
#endif
    return t_DebugVizOutput[uint2(pos.xy)].rgba;
}
#endif

#endif // #if defined(__HLSL_VERSION) && (__HLSL_VERSION >= 2021) 

#if !defined(__HLSL_VERSION) || (__HLSL_VERSION < 2021) || !defined(ENABLE_DEBUG_PRINT)
#define DebugPrint(str, ...) do { } while(false)
#endif

#endif // !defined(__cplusplus)

#endif // __SHADER_DEBUG_HLSLI__