/*
* Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#ifndef __SHADER_OPACITY_MICRO_MAP_DEBUG_HLSLI__ // using instead of "#pragma once" due to https://github.com/microsoft/DirectXShaderCompiler/issues/3943
#define __SHADER_OPACITY_MICRO_MAP_DEBUG_HLSLI__

struct OpacityMicroMapDebugInfo
{
    bool hasOmmAttachment;
    float3 opacityStateDebugColor;

    static OpacityMicroMapDebugInfo initDefault()
    {
        OpacityMicroMapDebugInfo dbg;
        dbg.hasOmmAttachment = false;
        dbg.opacityStateDebugColor = 0;
        return dbg;
    }
};

struct OpacityMicroMapContext
{
    ByteAddressBuffer OmmIndexBuffer;
    uint ommIndexBufferOffset;
    bool ommIndexBuffer16Bit;
    ByteAddressBuffer ommDescArrayBuffer;
    uint ommDescArrayBufferOffset;
    ByteAddressBuffer OmmArrayDataBuffer;
    uint ommArrayDataBufferOffset;

    uint primitiveIndex;
    float2 barycentrics;

    static OpacityMicroMapContext make(
        ByteAddressBuffer _ommIndexBuffer,
        uint _ommIndexBufferOffset,
        bool _ommIndexBuffer16Bit,
        ByteAddressBuffer _ommDescArrayBuffer,
        uint _ommDescArrayBufferOffset,
        ByteAddressBuffer _ommArrayDataBuffer,
        uint _ommArrayDataBufferOffset,
        uint _primitiveIndex,
        float2 _barycentrics)
    {
        OpacityMicroMapContext ctx;
        ctx.OmmIndexBuffer = _ommIndexBuffer;
        ctx.ommIndexBufferOffset = _ommIndexBufferOffset;
        ctx.ommIndexBuffer16Bit = _ommIndexBuffer16Bit;
        ctx.ommDescArrayBuffer = _ommDescArrayBuffer;
        ctx.ommDescArrayBufferOffset = _ommDescArrayBufferOffset;
        ctx.OmmArrayDataBuffer = _ommArrayDataBuffer;
        ctx.ommArrayDataBufferOffset = _ommArrayDataBufferOffset;

        ctx.primitiveIndex = _primitiveIndex;
        ctx.barycentrics = _barycentrics;
        return ctx;
    }
};

namespace bird // birc-curve indexing helpers
{
    static inline uint prefixEor2(uint x)
    {
        x ^= (x >> 1) & 0x7fff7fff;
        x ^= (x >> 2) & 0x3fff3fff;
        x ^= (x >> 4) & 0x0fff0fff;
        x ^= (x >> 8) & 0x00ff00ff;
        return x;
    }

    static inline uint interleaveBits2(uint x, uint y)
    {
        x = (x & 0xffff) | (y << 16);
        x = ((x >> 8) & 0x0000ff00) | ((x << 8) & 0x00ff0000) | (x & 0xff0000ff);
        x = ((x >> 4) & 0x00f000f0) | ((x << 4) & 0x0f000f00) | (x & 0xf00ff00f);
        x = ((x >> 2) & 0x0c0c0c0c) | ((x << 2) & 0x30303030) | (x & 0xc3c3c3c3);
        x = ((x >> 1) & 0x22222222) | ((x << 1) & 0x44444444) | (x & 0x99999999);
        return x;
    }

    static uint dbary2index(uint u, uint v, uint w, uint level)
    {
        const uint32_t coordMask = ((1U << level) - 1);

        uint b0 = ~(u ^ w) & coordMask;
        uint t = (u ^ v) & b0;
        uint c = (((u & v & w) | (~u & ~v & ~w)) & coordMask) << 16;
        uint f = prefixEor2(t | c) ^ u;
        uint b1 = (f & ~b0) | t; 

        return interleaveBits2(b0, b1);
    }

    static uint bary2index(float2 bc, uint level, out bool isUpright)
    {
        float numSteps = float(1u << level);
        uint iu = uint(numSteps * bc.x);
        uint iv = uint(numSteps * bc.y);
        uint iw = uint(numSteps * (1.f - bc.x - bc.y));
        isUpright = (iu & 1) ^ (iv & 1) ^ (iw & 1);
        return dbary2index(iu, iv, iw, level);
    }
}

static int GetOmmDescOffset(ByteAddressBuffer OmmIndexBuffer, uint ommIndexBufferOffset, uint primitiveIndex, bool is16bit)
{
    if (is16bit)
    {
        const uint dwOffset = primitiveIndex.x >> 1u;
        const uint shift = (primitiveIndex.x & 1u) << 4u; // 0 or 16
        const uint raw = OmmIndexBuffer.Load(4 * dwOffset + ommIndexBufferOffset);
        const uint raw16 = (raw >> shift) & 0xFFFFu;

        if (raw16 > 0xFFFB) // e.g special index
        {
            return (raw16 - 0xFFFF) - 1; // -1, -2, -3 or -4
        }

        return raw16;
    }
    else
    {
        return OmmIndexBuffer.Load(4 * primitiveIndex + ommIndexBufferOffset);
    }
}

float3 OpacityMicroMapDebugViz(OpacityMicroMapContext c)
{
    const int ommIndex = GetOmmDescOffset(c.OmmIndexBuffer, c.ommIndexBufferOffset, c.primitiveIndex, c.ommIndexBuffer16Bit);

	// dynamic register indexing is not recommended generally, but since this is debug code we allow it.
    float3 debugColors[4];
    debugColors[0] = float3(0, 0, 1); // Transparent
    debugColors[1] = float3(0, 1, 0); // Opaque
    debugColors[2] = float3(1, 0, 1); // UnknownTransparent
    debugColors[3] = float3(1, 1, 0); // UnknownOpaque

    if (ommIndex < 0)
    {
        return debugColors[-ommIndex - 1]; // special index... perhaps come up with a better way to distinguish?
    }
    else
    {
        const uint2 ommDescData = c.ommDescArrayBuffer.Load2(c.ommDescArrayBufferOffset + 8 * ommIndex);

        // PrimitiveIndex -> OMM Desc.
        const uint baseOffset           = ommDescData.x;
        const uint ommFormat            = (ommDescData.y >> 16u) & 0x0000FFFF; // 1 = 0C2, 2 = OC4
        const uint subdivisionLevels    = ommDescData.y & 0x0000FFFF;

        // BC -> micro-triangle index.
        bool isUpright;
        const uint microTriIndex = bird::bary2index(c.barycentrics, subdivisionLevels, isUpright);

        // micro-tri index -> state
        const uint rShift           = ommFormat == 2 ?   4 : 5;
        const uint idxMask          = ommFormat == 2 ? 0xF : 0x1F;
        const uint bitsPerState     = ommFormat == 2 ?   2 : 1;
        const uint stateMask        = ommFormat == 2 ? 0x3 : 0x1;

        const uint wordIdx      = microTriIndex >> rShift; // div by 16 or div by 32
        const uint stateIdx     = microTriIndex & idxMask; // mod 16 or 32

        const uint ommStateWord = c.OmmArrayDataBuffer.Load(c.ommArrayDataBufferOffset + baseOffset + 4 * wordIdx);

        const uint state = (ommStateWord >> (bitsPerState * stateIdx)) & stateMask;

        float3 opacityStateDebugColor = debugColors[state];
        if (isUpright)
            opacityStateDebugColor *= 0.5f;

        return opacityStateDebugColor;
    }
} 

#endif // __SHADER_OPACITY_MICRO_MAP_DEBUG_HLSLI__
