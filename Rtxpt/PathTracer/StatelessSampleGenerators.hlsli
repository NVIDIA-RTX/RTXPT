/*
* Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#ifndef __INLINE_SAMPLE_GENERATORS_HLSLI__ // using instead of "#pragma once" due to https://github.com/microsoft/DirectXShaderCompiler/issues/3943
#define __INLINE_SAMPLE_GENERATORS_HLSLI__

#if USE_PRECOMPUTED_SOBOL_BUFFER
// there should be a better place for this, but it has to be defined before "NoiseAndSequences.hlsli" is included
Buffer<uint> t_precomputedSobol : register(t42);
#define SOBOL_PRECOMPUTED_BUFFER t_precomputedSobol
#endif

#include "Utils/NoiseAndSequences.hlsli"

// Each path tracing vertex initializes this once and spawns sample sequences using it
struct SampleGeneratorVertexBase
{
    uint    m_baseHash;
    uint    m_sampleIndex;

    static SampleGeneratorVertexBase    make(uint2 pixelCoord, uint vertexIndex, uint sampleIndex)
    {
        SampleGeneratorVertexBase ret;
        ret.m_sampleIndex = sampleIndex;
        ret.m_baseHash = Hash32Combine(Hash32(vertexIndex + 0x035F9F29), (pixelCoord.x << 16) | pixelCoord.y);
        return ret;
    }

    static SampleGeneratorVertexBase    make(uint baseHash, uint sampleIndex)
    {
        SampleGeneratorVertexBase ret;
        ret.m_sampleIndex = sampleIndex;
        ret.m_baseHash = Hash32(baseHash+0x035F9F29);
        return ret;
    }
};

/** Inline quasi-random sample generator.

    Current implementation is based on "Practical Hash-based Owen Scrambling", Brent Burley 2020, https://jcgt.org/published/0009/04/01/paper.pdf
    Shader implementation borrows from Andrew Helmer's Shadertoy implementation (https://www.reddit.com/r/GraphicsProgramming/comments/l1go2r/owenscrambled_sobol_02_sequences_shadertoy/)

    It supports up to >6< dimensions and reverts back to pseudo-random for subsequent samples.

*/
struct SampleSequenceGenerator
{
    static const uint cLDDisabled                       = 0xFFFFFFFE;
    static const uint cLDDisabled_RanOutOfDimensions    = 0xFFFFFFFF;

    uint m_startingHash;
    uint m_currentHash;

    uint m_sampleIndex;
    uint m_dimension;   // cLDDisabled and cLDDisabled_RanOutOfDimensions mean it's in non-LD mode
    uint m_activeIndex; // in case subIndexing is used

    // Creates an effect-specific sequence sampler state that is decorrelated from other effects.
    // The state is also deterministic for the specific path vertex, regardless of shader code ordering / branching.
    // This either activates the non-low discrepancy path (faster) or a low-discrepancy (slower) sampler where
    // each subsequent 'Next()' call advances dimension and provide a low discrepancy sample for the index.
    // Once all available dimensions provided by the LD sampler are used up (4, 5, 6, etc.), it reverts to the faster non-low discrepancy path.
    //
    // Subindex-ing feature allows sampling multiple sequence samples in case of inner loops (i.e. sampling multiple lights in a loop per bounce).
    static SampleSequenceGenerator      make(const SampleGeneratorVertexBase base, const SampleGeneratorEffectSeed effectSeed = SampleGeneratorEffectSeed::Base, bool lowDiscrepancy = false, int subSampleCount = 1)
    {
        SampleSequenceGenerator ret;
        ret.m_sampleIndex = base.m_sampleIndex;
        ret.m_activeIndex = ret.m_sampleIndex * subSampleCount;
        ret.m_currentHash = Hash32Combine(base.m_baseHash, (uint)effectSeed);
        ret.m_startingHash = ret.m_currentHash;
        if (lowDiscrepancy)
        {
            ret.m_dimension = 0;    // set LD mode
        }
        else
        {
            ret.m_currentHash = Hash32Combine(ret.m_currentHash, ret.m_activeIndex); // unlike the StatelessUniformSampleGenerator, we combine the sampleIndex here in uniform mode; a bit slower but allows for LD mode that needs it not combined into m_baseHash
            ret.m_dimension = cLDDisabled; // set non-LD mode
        }
        return ret;
    }

    // Returns the next sample value. This function updates the state by advancing Dimension if possible, otherwise it just advances the uniform random sequence. 
    uint Next()
    {
        const uint maxSupportedDimensionIndex = 5;
        
        if (m_dimension >= cLDDisabled)
        {
            m_currentHash = Hash32(m_currentHash);
            return m_currentHash;
        }
        else
        {
            uint shuffle_seed = Hash32Combine( m_currentHash, 0 );   // this might not need combine, just use raw and then remove '1+' from '1+m_dimension' below
            uint dim_seed = Hash32Combine( m_currentHash, 1+m_dimension );
            uint shuffled_index = bhos_owen_scramble(m_activeIndex, shuffle_seed);
            #if 1
            // Sobol' sequence is expensive but we can use Laine-Kerras permutation for the 1st dimension and Sobol' only for second and subsequent. See 'bhos_sobol' function comments for more detail.
            uint dim_sample;
            [branch] if (m_dimension==0) 
                dim_sample = bhos_reverse_bits(shuffled_index);
            else
                dim_sample = bhos_sobol(shuffled_index, m_dimension);
            #else
            uint dim_sample = bhos_sobol(shuffled_index, m_dimension);
            #endif
            dim_sample = bhos_owen_scramble(dim_sample, dim_seed);

            // step dimension for next sample!
            m_dimension++;
            
            // for all subsequent samples fall back to pseudo-random
            if(m_dimension >= maxSupportedDimensionIndex)
            {
                m_currentHash = Hash32Combine(m_currentHash, m_activeIndex);
                m_dimension = cLDDisabled_RanOutOfDimensions;    // set non-LD mode
            }
            return dim_sample;
        }
    }

    // Start next sample, reset the dimension to 0
    void AdvanceSampleIndex()
    {
        if (m_dimension != cLDDisabled)  // if not uniform
        {
            m_dimension = 0;
            m_currentHash = m_startingHash;
            m_activeIndex++;
        }
    }

    // Note - this is limited to generating up to 4 values
    static float4 Generate(uint count, const SampleGeneratorVertexBase base, const SampleGeneratorEffectSeed effectSeed, const int subSampleIndex = 0, const int subSampleCount = 1)
    {
        count = min( count, 4 );
        float4 retVal;

        SampleSequenceGenerator context;
        context.m_sampleIndex = base.m_sampleIndex;
        context.m_activeIndex = context.m_sampleIndex * subSampleCount + subSampleIndex;
        context.m_currentHash = Hash32Combine(base.m_baseHash, (uint)effectSeed);
        context.m_startingHash = context.m_currentHash;
        
        [unroll]for( context.m_dimension = 0; context.m_dimension < count; context.m_dimension++ )
        {
            uint shuffle_seed = Hash32Combine( context.m_currentHash, 0 );   // this might not need combine, just use raw and then remove '1+' from '1+m_dimension' below
            uint dim_seed = Hash32Combine( context.m_currentHash, 1+context.m_dimension );
            uint shuffled_index = bhos_owen_scramble(context.m_activeIndex, shuffle_seed);
            
            uint dim_sample;
            // Sobol' sequence is expensive but we can use Laine-Kerras permutation for the 1st dimension and Sobol' only for second and subsequent. See 'bhos_sobol' function comments for more detail.
            if (context.m_dimension==0) 
                dim_sample = bhos_reverse_bits(shuffled_index);
            else
                dim_sample = bhos_sobol(shuffled_index, context.m_dimension);
            dim_sample = bhos_owen_scramble(dim_sample, dim_seed);
            retVal[context.m_dimension] = Hash32ToFloat(dim_sample);
        }
        return retVal;
    }

};

/** Inline uniform random sample generator.

    This generator has only 32 bit state and sub-optimal statistical properties, however it's 'mostly fine' for up to millions of samples.
    TODO: try using LCG
    TODO: read & try out: https://pharr.org/matt/blog/2022/03/05/sampling-fp-unit-interval
*/
struct UniformSampleSequenceGenerator
{
    uint m_currentHash;

    static UniformSampleSequenceGenerator make(const SampleGeneratorVertexBase base, const SampleGeneratorEffectSeed effectSeed = SampleGeneratorEffectSeed::Base, bool lowDiscrepancy = false, int subSampleCount = 1)
    {
        UniformSampleSequenceGenerator ret;

        uint activeIndex = base.m_sampleIndex * subSampleCount;
        ret.m_currentHash = Hash32Combine(base.m_baseHash, (uint)effectSeed);
        ret.m_currentHash = Hash32Combine(ret.m_currentHash, activeIndex); // haven't seen any instances where it's necessary
        return ret;
    }

    // Returns the next sample value. This function updates the state.
    uint Next()
    {
        m_currentHash = Hash32(m_currentHash);
        return m_currentHash;
    }

    void AdvanceSampleIndex()
    {
        // nothing to do here
    }

    // Note - this is limited to generating up to 4 values
    static float4 Generate(uint count, const SampleGeneratorVertexBase base, const SampleGeneratorEffectSeed effectSeed, const int subSampleIndex = 0, const int subSampleCount = 1)
    {
        count = min( count, 4 );
        float4 retVal;

        UniformSampleSequenceGenerator context;
        uint activeIndex = base.m_sampleIndex * subSampleCount + subSampleIndex;
        context.m_currentHash = Hash32Combine(base.m_baseHash, (uint)effectSeed);
        context.m_currentHash = Hash32Combine(context.m_currentHash, activeIndex); // haven't seen any instances where it's necessary
        
        [unroll]for( uint counter = 0; counter < count; counter++ )
        {
            context.m_currentHash = Hash32(context.m_currentHash);
            retVal[counter] = Hash32ToFloat(context.m_currentHash);
        }
        return retVal;
    }

};


#if 0 // needs porting to new interface format

#include "Utils/Math/BitTricks.hlsli"
#include "Utils/Sampling/Pseudorandom/Xoshiro.hlsli"
#include "Utils/Sampling/Pseudorandom/SplitMix64.hlsli"

/** Inline uniform random sample generator using high quality xoshiro128 RTG (http://xoshiro.di.unimi.it/xoshiro128starstar.c)

    This implementation is intended for collecting high quality reference. It is slower and ignores startEffect interface.
*/
struct HQUniformSampleSequenceGenerator
{
    Xoshiro128StarStar rng;

//public:
    static HQUniformSampleSequenceGenerator make(uint2 pixelCoord, uint vertexIndex, uint sampleIndex)
    {
        HQUniformSampleSequenceGenerator ret;
        
        // This is an inline variant of Falcor's UniformSampleGenerator; due to it being inline and not
        // keeping state along whole path, it means we have to re-seed per vertex (bounce) and not per pixel, 
        // increasing chances of overlapping sequences.
        // See original implementation for more info on seed generation:
        // (https://github.com/NVIDIAGameWorks/Falcor/blob/9fdfdbb37516f4273e952a5e30b85af8ccfe171d/Source/Falcor/Utils/Sampling/UniformSampleGenerator.slang#L57).

        SplitMix64 rng = createSplitMix64(interleave_32bit(pixelCoord), sampleIndex + (vertexIndex<<24)); // 24 bits for sample index guarantees first 16 mil samples without overlapping vertexIndex, and 8 bits for vertex index is plenty
        uint64_t s0 = nextRandom64(rng);
        uint64_t s1 = nextRandom64(rng);
        uint seed[4] = { uint(s0), uint(s0 >> 32), uint(s1), uint(s1 >> 32) };

        // Create xoshiro128** pseudorandom generator.
        ret.rng = createXoshiro128StarStar(seed);        
        
        return ret;
    }

    // Sets an effect-specific sampler state that is decorrelated from other effects.
    // The state is also deterministic for the specific path vertex, regardless of shader code ordering / branching.
    void startEffect(SampleGeneratorEffectSeed effectSeed, bool lowDiscrepancy = false, int subSampleCount = 1)
    {
    }
    
    // Returns the next sample value. This function updates the state.
    uint next()
    {
        return nextRandom(rng);
    }
};

#endif

#endif // __INLINE_SAMPLE_GENERATORS_HLSLI__