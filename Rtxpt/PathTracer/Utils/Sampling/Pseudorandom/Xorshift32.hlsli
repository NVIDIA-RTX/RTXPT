/*
* Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

/** Xorshift32 is a tiny pseudorandom generator with reasonably good properties for its size.
    It's still not good enough for many use cases due to its small state.

    Prefer pseudorandom generators with a larger state.
*/
struct Xorshift32
{
    /** Initializes a Xorshift32 pseudorandom generator.
        \param[in] seed The random seed.
        \return the new pseudorandom generator.
    */
    __init(uint seed)
    {
        this.state = seed;
    }

    /** Generates the next pseudorandom number in the sequence (32 bits).
        This function updates the state.
    */
    [mutating] uint next()
    {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        return state;
    }

    uint state;
};
