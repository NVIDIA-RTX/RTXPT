/*
* Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#pragma once

#include <omm-gpu-nvrhi.h>
#include <bit>
#include <cstdint>
#include <iterator>
#include <vector>

namespace omm
{
    // The geometry ID comes from exact, collision-safe CPU UV equality. Texture and
    // bake settings remain part of every lookup; hash collisions never imply equality.
    using BakeInputKey = std::vector<uint32_t>;

    inline void AppendBakeInputKey(BakeInputKey& key, const GpuBakeNvrhi::Input& input,
        nvrhi::rt::OpacityMicromapBuildFlags flags, uint64_t geometryId)
    {
        const uint64_t texture = reinterpret_cast<uintptr_t>(input.alphaTexture.Get());
        const uint32_t settings[] = {
            uint32_t(texture), uint32_t(texture >> 32), input.alphaTextureChannel,
            std::bit_cast<uint32_t>(input.alphaCutoff), uint32_t(input.alphaCutoffGreater),
            uint32_t(input.alphaCutoffLessEqual), uint32_t(input.bilinearFilter),
            uint32_t(input.enableLevelLineIntersection), uint32_t(input.sampleMode),
            uint32_t(input.texCoordFormat), input.numIndices, input.maxSubdivisionLevel,
            input.maxOutOmmArraySize, uint32_t(input.format),
            std::bit_cast<uint32_t>(input.dynamicSubdivisionScale), uint32_t(input.minimalMemoryMode),
            uint32_t(input.enableStats), uint32_t(input.enableSpecialIndices),
            uint32_t(input.force32BitIndices), uint32_t(input.enableTexCoordDeduplication),
            uint32_t(input.computeOnly), uint32_t(input.enableNsightDebugMode), uint32_t(flags)
        };
        key.insert(key.end(), std::begin(settings), std::end(settings));
        key.push_back(uint32_t(geometryId));
        key.push_back(uint32_t(geometryId >> 32));
    }

    struct BakeInputKeyHash
    {
        size_t operator()(const BakeInputKey& key) const
        {
            uint64_t hash = 14695981039346656037ull;
            for (uint32_t word : key)
            {
                hash ^= word;
                hash *= 1099511628211ull;
            }
            return size_t(hash);
        }
    };
}
