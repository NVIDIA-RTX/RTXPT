// Standalone CPU contract checks; uses the production key and equality.
#include "../../Rtxpt/OpacityMicroMap/OmmBakeInputKey.h"
#include <cstdio>
#include <cstdlib>
#include <unordered_map>

namespace
{
    void Require(bool condition)
    {
        if (!condition)
            std::abort();
    }
}

void RunCpuIdentityChecks();

int main()
{
    using Input = omm::GpuBakeNvrhi::Input;
    using Flags = nvrhi::rt::OpacityMicromapBuildFlags;
    Input input;
    input.numIndices = 3;
    const uint64_t geometryId = 0x1234567800000001ull;
    auto keyFor = [&](const Input& value, Flags flags = Flags::FastTrace) {
        omm::BakeInputKey key;
        omm::AppendBakeInputKey(key, value, flags, geometryId);
        return key;
    };
    const auto key = keyFor(input);
    Require(key == keyFor(input));
    auto changed = [&](auto change) {
        Input other = input;
        change(other);
        Require(key != keyFor(other));
    };
    changed([](Input& v) { v.alphaTextureChannel++; });
    changed([](Input& v) { v.alphaCutoff = 0.6f; });
    changed([](Input& v) { v.alphaCutoffGreater = omm::OpacityState::UnknownOpaque; });
    changed([](Input& v) { v.alphaCutoffLessEqual = omm::OpacityState::UnknownTransparent; });
    changed([](Input& v) { v.bilinearFilter = !v.bilinearFilter; });
    changed([](Input& v) { v.enableLevelLineIntersection = !v.enableLevelLineIntersection; });
    changed([](Input& v) { v.sampleMode = nvrhi::SamplerAddressMode::Wrap; });
    changed([](Input& v) { v.texCoordFormat = nvrhi::Format::RG32_FLOAT; });
    changed([](Input& v) { v.numIndices += 3; });
    changed([](Input& v) { v.maxSubdivisionLevel++; });
    changed([](Input& v) { v.maxOutOmmArraySize--; });
    changed([](Input& v) { v.format = nvrhi::rt::OpacityMicromapFormat::OC1_2_State; });
    changed([](Input& v) { v.dynamicSubdivisionScale = 2.f; });
    changed([](Input& v) { v.minimalMemoryMode = !v.minimalMemoryMode; });
    changed([](Input& v) { v.enableStats = !v.enableStats; });
    changed([](Input& v) { v.enableSpecialIndices = !v.enableSpecialIndices; });
    changed([](Input& v) { v.force32BitIndices = !v.force32BitIndices; });
    changed([](Input& v) { v.enableTexCoordDeduplication = !v.enableTexCoordDeduplication; });
    changed([](Input& v) { v.computeOnly = !v.computeOnly; });
    changed([](Input& v) { v.enableNsightDebugMode = !v.enableNsightDebugMode; });
    Require(key != keyFor(input, Flags::FastBuild));

    // Source addresses/offsets and vertex IDs are absent: resolved UV order is the input.
    Input relocated = input;
    relocated.indexBufferOffsetInBytes = 64;
    relocated.texCoordBufferOffsetInBytes = 128;
    Require(key == keyFor(relocated));
    auto keyForId = [&](uint64_t id) {
        omm::BakeInputKey result;
        omm::AppendBakeInputKey(result, input, Flags::FastTrace, id);
        return result;
    };
    Require(key != keyForId(geometryId ^ 1));
    const auto differentGeometry = keyForId(geometryId ^ (1ull << 32));
    Require(key != differentGeometry);

    struct CollidingHash { size_t operator()(const omm::BakeInputKey&) const { return 0; } };
    std::unordered_map<omm::BakeInputKey, int, CollidingHash> cache;
    cache.emplace(key, 1);
    cache.emplace(differentGeometry, 2);
    Require(cache.size() == 2 && cache.at(key) == 1 && cache.at(differentGeometry) == 2);
    RunCpuIdentityChecks();
    std::puts("OMM bake input key checks passed");
}
