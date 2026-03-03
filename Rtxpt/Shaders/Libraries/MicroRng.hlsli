// Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
// 
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef __MICRO_RNG_HLSLI__
#define __MICRO_RNG_HLSLI__

struct MicroRng
{
    uint    N;
    
    static MicroRng make( uint seed )
    {
        MicroRng ret;
        ret.N = seed ^ 0x9e3779b9;    // with our hash function, 0 will loop so please don't seed with 0x9e3779b9 :D
        return ret;
    }
    
    static MicroRng make( uint2 seedCoord, uint seedValue )
    {
        MicroRng ret;
        ret.N = ((seedCoord.x << 16) | seedCoord.y) ^ 0x9e3779b9;
        ret.N = ret.N ^ (seedValue + (ret.N << 6) + (ret.N >> 2));
        return ret;
    }

    static MicroRng make( uint2 seedCoord, uint seedValueA, uint seedValueB )
    {
        MicroRng ret;
        ret.N = ((seedCoord.x << 16) | seedCoord.y) ^ 0x9e3779b9;
        ret.N = ret.N ^ (seedValueA + (ret.N << 6) + (ret.N >> 2));
        ret.N = ret.N ^ (seedValueB + (ret.N << 6) + (ret.N >> 2));
        return ret;
    }
        
    uint Next()
    {
        // See https://nullprogram.com/blog/2018/07/31/, "Prospecting for Hash Functions" by Chris Wellons
        // Latest update: https://github.com/skeeto/hash-prospector/issues/19
        N ^= N >> 16;
        N *= 0x21f0aaad;
        N ^= N >> 15;
        N *= 0xf35a2d97;
        N ^= N >> 15;
        return N;
    }
    
    float NextFloat()
    {
        // a.) converting the upper 24bits to [0, 1) because the higher bits have better distribution in some hash algorithms (like sobol)
        // b.) this is a good way to guarantee [0, 1) since float32 mantissa is only 23 bits
        return (Next() >> 8) / float(1 << 24); // same as '/ 16777216.0'
    }

    float2 NextFloat2()  { return float2(NextFloat(), NextFloat()); } 
};

#endif //__MICRO_RNG_HLSLI__