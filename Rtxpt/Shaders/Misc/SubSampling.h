/*
* Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#ifndef __SUB_SAMPLING_H__  // using instead of "#pragma once" due to https://github.com/microsoft/DirectXShaderCompiler/issues/3943
#define __SUB_SAMPLING_H__

// Realtime sub-sampling (sub-1spp): which pixels of the denoising layers fill pass get a path this frame. The rate is
// RTXPT_SUB_SAMPLING_RATE_NUM / (RTXPT_SUB_SAMPLING_SCALE_X * RTXPT_SUB_SAMPLING_SCALE_Y) (half, quarter or three
// quarters); a traced pixel's throughput is scaled by the inverse of the rate and the rest are left with zero noisy
// radiance, so every pattern is unbiased in expectation.
//
// Patterns - must match SubSamplingPattern in SampleUI.h:
#define RTXPT_SUB_SAMPLING_PATTERN_BLOCK_JITTER             0   // RATE_NUM uniformly random pixels per SCALE_X x SCALE_Y block
#define RTXPT_SUB_SAMPLING_PATTERN_STOCHASTIC_COMPACTED     1   // per-pixel white noise coin flip, stream-compacted list
#define RTXPT_SUB_SAMPLING_PATTERN_BLOCK_STRATIFIED         2   // exactly N*N*rate pixels at random distinct positions per NxN block
#define RTXPT_SUB_SAMPLING_PATTERN_SOBOL_BLOCK              3   // Owen-scrambled Sobol (0,2) sequence per NxN block, see SubSampling_ThreadToPixel

// The fill pass raygen gets the pattern and N (block size for BLOCK_STRATIFIED and SOBOL_BLOCK, power of two) as the
// RTXPT_SUB_SAMPLING_PATTERN and RTXPT_SUB_SAMPLING_BLOCK_SIZE macros. Push constants (SampleMiniConstants), when
// sub-sampling is active:
//      params.x    subSampleIndex (always 0, sub-sampling is only available at 1spp)
//      params1.z   SCALE_X * SCALE_Y   } rate is w / z, for the compaction pass, which has no SCALE / RATE_NUM macros
//      params1.w   RATE_NUM            }

// Compacted pixel list (u_SubSamplingPixelList, RenderTargets::SubSamplingPixelList) layout:
//      +0                                      uint count, reset to 0 every frame before compaction
//      +RTXPT_SUB_SAMPLING_LIST_HEADER_BYTES   uint pixel[count], packed as x | (y << 16)
#define RTXPT_SUB_SAMPLING_LIST_HEADER_BYTES                16

// Compaction pass group is a TILE x TILE screen tile, so the list - and the 1D ray launch that consumes it - stays
// spatially coherent: consecutive entries come from the same tile.
#define RTXPT_SUB_SAMPLING_COMPACTION_TILE                  8

#if !defined(__cplusplus)

#ifndef RTXPT_SUB_SAMPLING_RATE_NUM
#define RTXPT_SUB_SAMPLING_RATE_NUM 1   // traced pixels per SCALE_X x SCALE_Y block; 3 (of 2x2) for three quarters
#endif

#include "../Libraries/MicroRng.hlsli"

// Same seed for the compaction pass and the fill pass raygen; equal to Bridge::getSampleIndex() at 1spp.
uint SubSampling_FrameSeed()
{
    return g_Const.ptConsts.sampleBaseIndex + g_MiniConst.params.x;
}

// White noise coin flip for RTXPT_SUB_SAMPLING_PATTERN_STOCHASTIC_COMPACTED: independent per pixel and per frame.
bool SubSampling_IsPixelSelected(const uint2 pixelPos, const uint frameSeed, const uint rateNum, const uint rateDen)
{
    MicroRng rng = MicroRng::make(pixelPos, frameSeed, 0x53554253 /*'SUBS'*/);
    return rng.NextFloat() * float(rateDen) < float(rateNum);
}

// Brent Burley, "Practical Hash-based Owen Scrambling" (JCGT 2020): nested uniform scramble of a 32-bit fixed point
// coordinate. Preserves the (t,m,s)-net properties of the points it is applied to.
uint SubSampling_OwenScramble(uint v, const uint seed)
{
    v = reversebits(v);
    v ^= v * 0x3d20adea;
    v += seed;
    v *= (seed >> 16) | 1;
    v ^= v * 0x05526c56;
    v ^= v * 0x53a22864;
    return reversebits(v);
}

// Second dimension of the Sobol sequence (the first one is the van der Corput sequence, reversebits(index)); together
// they form the (0,2) sequence in base 2.
uint SubSampling_SobolDim1(uint index)
{
    uint result = 0;
    for (uint v = 1u << 31; index != 0; index >>= 1, v ^= v >> 1)
        if (index & 1)
            result ^= v;
    return result;
}

// Andrew Kensler, "Correlated Multi-Jittered Sampling" (Pixar Technical Memo 13-01): a permutation of [0, l) keyed
// by p. For a power of two l the cycle-walking loop runs exactly once.
uint SubSampling_Permute(uint i, const uint l, const uint p)
{
    uint w = l - 1;
    w |= w >> 1; w |= w >> 2; w |= w >> 4; w |= w >> 8; w |= w >> 16;
    do
    {
        i ^= p;             i *= 0xe170893d;
        i ^= p >> 16;
        i ^= (i & w) >> 4;
        i ^= p >> 8;        i *= 0x0929eb3f;
        i ^= p >> 23;
        i ^= (i & w) >> 1;  i *= 1 | p >> 27;
                            i *= 0x6935fa69;
        i ^= (i & w) >> 11; i *= 0x74dcb303;
        i ^= (i & w) >> 2;  i *= 0x9e501cc3;
        i ^= (i & w) >> 2;  i *= 0xc860a3df;
        i &= w;
        i ^= i >> 5;
    } while (i >= l);
    return (i + p) % l;
}

// Maps a fill pass raygen thread onto the pixel it traces. Returns false if the thread has nothing to do: its pixel
// is outside the viewport (thread grids are rounded up to whole blocks), or it is past the end of the compacted list
// (that launch is sized conservatively on the CPU, which does not know the count). The thread grid per pattern:
//      BLOCK_JITTER                    (ceil(viewport.x / scale.x) * rateNum, ceil(viewport.y / scale.y)), rateNum threads per block
//      STOCHASTIC_COMPACTED            (conservative upper bound on the list size, 1)
//      BLOCK_STRATIFIED, SOBOL_BLOCK   (ceil(viewport.x / N) * samplesPerBlock, ceil(viewport.y / N))
// rateNum is RTXPT_SUB_SAMPLING_RATE_NUM: 1, or scale.x * scale.y - 1 (three quarters).
bool SubSampling_ThreadToPixel(const uint2 threadID, const uint2 scale, const uint rateNum, const uint pattern, const uint blockSize, const uint frameSeed, const uint2 viewportSize, out uint2 pixelPos)
{
    if (pattern == RTXPT_SUB_SAMPLING_PATTERN_STOCHASTIC_COMPACTED)
    {
        const uint index = threadID.x;
        if (index >= u_SubSamplingPixelList.Load(0))
        {
            pixelPos = uint2(0, 0);
            return false;
        }
        const uint packed = u_SubSamplingPixelList.Load(RTXPT_SUB_SAMPLING_LIST_HEADER_BYTES + index * 4);
        pixelPos = uint2(packed & 0xFFFF, packed >> 16);
        return true;    // the list only holds in-viewport pixels
    }
    else if (pattern == RTXPT_SUB_SAMPLING_PATTERN_BLOCK_STRATIFIED)
    {
        // Each block draws its own random permutation of its N*N slots every frame and takes the first
        // samplesPerBlock of them, so a block always gets exactly that many distinct pixels.
        const uint blockPixels      = blockSize * blockSize;
        const uint samplesPerBlock  = blockPixels * rateNum / (scale.x * scale.y);
        const uint2 block           = uint2(threadID.x / samplesPerBlock, threadID.y);
        const uint sampleInBlock    = threadID.x % samplesPerBlock;
        MicroRng rng = MicroRng::make(block, frameSeed, 0x424c4b53 /*'BLKS'*/);
        const uint slot = SubSampling_Permute(sampleInBlock, blockPixels, rng.Next());
        pixelPos = block * blockSize + uint2(slot % blockSize, slot / blockSize);
    }
    else if (pattern == RTXPT_SUB_SAMPLING_PATTERN_SOBOL_BLOCK)
    {
        // The block (N = 2^b) walks the Owen-scrambled (0,2) sequence samplesPerBlock = 2^m points per frame. Any
        // aligned run of 2^m points is a (0,m,2)-net: one point in every elementary box of area 2^-m, so the frame's
        // pixels are distinct (m >= b, so a pixel always fits in one box) and stratified at every power of two aspect.
        // A cycle of cycleFrames frames covers indices [0, N*N), a (0,2b,2)-net, i.e. every pixel exactly once. The
        // scramble, independent per block, only changes between cycles so that property holds.
        // Three quarters: 3 of the 4 aligned quarters of [0, N*N) per frame (wrapping), each pixel traced 3x per cycle.
        const uint cycleFrames      = scale.x * scale.y;
        const uint samplesPerBlock  = blockSize * blockSize * rateNum / cycleFrames;
        const uint2 block           = uint2(threadID.x / samplesPerBlock, threadID.y);
        const uint index            = ((frameSeed % cycleFrames) * samplesPerBlock + (threadID.x % samplesPerBlock)) % (blockSize * blockSize);
        MicroRng rng = MicroRng::make(block, frameSeed / cycleFrames, 0x534f424c /*'SOBL'*/);
        const uint seedX = rng.Next();
        const uint seedY = rng.Next();
        const uint shift = 32 - firstbithigh(blockSize);    // keep the top log2(N) bits
        pixelPos = block * blockSize + uint2(SubSampling_OwenScramble(reversebits(index), seedX) >> shift,
                                             SubSampling_OwenScramble(SubSampling_SobolDim1(index), seedY) >> shift);
    }
    else if (rateNum == 1) // RTXPT_SUB_SAMPLING_PATTERN_BLOCK_JITTER, the original method
    {
        MicroRng mrng = MicroRng::make(threadID, frameSeed);
        // uniform over [0, scale-1]; NextFloat() is [0,1) so the product can never reach scale
        pixelPos.x = threadID.x * scale.x + uint(mrng.NextFloat() * scale.x);
        pixelPos.y = threadID.y * scale.y + uint(mrng.NextFloat() * scale.y);
    }
    else // RTXPT_SUB_SAMPLING_PATTERN_BLOCK_JITTER at three quarters
    {
        // The block leaves out one uniformly random pixel and its rateNum threads take the others in order, so the
        // random number is seeded by the block rather than the thread.
        const uint2 block           = uint2(threadID.x / rateNum, threadID.y);
        const uint sampleInBlock    = threadID.x % rateNum;
        MicroRng mrng = MicroRng::make(block, frameSeed);
        const uint skipped = uint(mrng.NextFloat() * (scale.x * scale.y));
        const uint slot = sampleInBlock + ((sampleInBlock >= skipped) ? 1 : 0);
        pixelPos = block * scale + uint2(slot % scale.x, slot / scale.x);
    }

    // Thread grids round up to whole blocks, so a block on the right/bottom edge can be only partially inside the
    // viewport and the pixel can land outside it. Drop those threads as bad things can happen.
    return all(pixelPos < viewportSize);
}

#endif // #if !defined(__cplusplus)

#endif // __SUB_SAMPLING_H__
