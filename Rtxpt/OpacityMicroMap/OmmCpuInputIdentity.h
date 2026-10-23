/*
* Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
*/
#pragma once

#include <donut/engine/SceneTypes.h>
#include <atomic>
#include <cstring>
#include <optional>
#include <span>
#include <unordered_map>

namespace omm
{
    // Borrowed only until Scene::CreateMeshBuffers uploads and discards its arrays.
    struct CpuInputView
    {
        std::span<const uint32_t> indices;
        std::span<const donut::math::float2> texcoords;

        static std::optional<CpuInputView> FromGeometry(
            const donut::engine::MeshInfo& mesh, const donut::engine::MeshGeometry& geometry)
        {
            if (!mesh.buffers || geometry.type != donut::engine::MeshGeometryPrimitiveType::Triangles
                || !geometry.numIndices || geometry.numIndices % 3)
                return {};
            const uint64_t firstIndex = uint64_t(mesh.indexOffset) + geometry.indexOffsetInMesh;
            const uint64_t firstVertex = uint64_t(mesh.vertexOffset) + geometry.vertexOffsetInMesh;
            const auto& buffers = *mesh.buffers;
            if (firstIndex + geometry.numIndices > buffers.indexData.size()
                || firstVertex + geometry.numVertices > buffers.texcoord1Data.size())
                return {};
            return CpuInputView{
                std::span(buffers.indexData).subspan(size_t(firstIndex), geometry.numIndices),
                std::span(buffers.texcoord1Data).subspan(size_t(firstVertex), geometry.numVertices) };
        }

        uint64_t UvBits(size_t corner) const
        {
            uint64_t bits;
            static_assert(sizeof(donut::math::float2) == sizeof(bits));
            std::memcpy(&bits, &texcoords[indices[corner]], sizeof(bits));
            return bits;
        }

        // Validation and hashing share one pass; float bits, including signed zero
        // and NaN payloads, have exactly the same equality as the GPU input stream.
        std::optional<uint64_t> Hash() const
        {
            if (indices.empty() || indices.size() % 3) return {};
            uint64_t hash = 14695981039346656037ull;
            for (size_t i = 0; i < indices.size(); ++i)
            {
                if (indices[i] >= texcoords.size()) return {};
                const uint64_t bits = UvBits(i);
                hash = (hash ^ uint32_t(bits)) * 1099511628211ull;
                hash = (hash ^ uint32_t(bits >> 32)) * 1099511628211ull;
            }
            return hash;
        }

        bool operator==(const CpuInputView& other) const
        {
            if (indices.size() != other.indices.size()) return false;
            for (size_t i = 0; i < indices.size(); ++i)
                if (UvBits(i) != other.UvBits(i)) return false;
            return true;
        }
    };

    // Stack-local for one upload batch. Destroy before any borrowed arrays change.
    class CpuInputIdentityRegistry
    {
        struct Entry { CpuInputView input; uint64_t id; };
        std::unordered_multimap<uint64_t, Entry> m_inputs;

    public:
        // Both views must have passed Hash() validation. The hash selects candidates;
        // exact indexed-UV comparison decides equality, even if hashes collide.
        uint64_t Intern(const CpuInputView& input, uint64_t hash)
        {
            const auto [begin, end] = m_inputs.equal_range(hash);
            for (auto it = begin; it != end; ++it)
                if (it->second.input == input)
                    return it->second.id;
            static std::atomic<uint64_t> nextId{1};
            const uint64_t id = nextId.fetch_add(1, std::memory_order_relaxed);
            m_inputs.emplace(hash, Entry{input, id});
            return id;
        }

        uint64_t Intern(const donut::engine::MeshInfo& mesh, const donut::engine::MeshGeometry& geometry)
        {
            const auto input = CpuInputView::FromGeometry(mesh, geometry);
            if (!input) return 0;
            const auto hash = input->Hash();
            return hash ? Intern(*input, *hash) : 0;
        }
    };

    struct CpuMeshInputIdentity
    {
        struct Geometry
        {
            uint64_t id = 0;
            uint32_t indexOffset = 0, vertexOffset = 0, numIndices = 0, numVertices = 0;
            donut::engine::MeshGeometryPrimitiveType type = donut::engine::MeshGeometryPrimitiveType::Triangles;
        };
        std::weak_ptr<donut::engine::BufferGroup> bufferGroup;
        // Strong handles prevent a destroyed buffer's address being reused by a
        // replacement that would otherwise appear to match this snapshot.
        nvrhi::BufferHandle indexBuffer, vertexBuffer;
        uint32_t indexOffset = 0, vertexOffset = 0, totalIndices = 0, totalVertices = 0;
        nvrhi::BufferRange texCoordRange;
        std::vector<Geometry> geometries;

        static bool SupportsStaticMesh(const donut::engine::MeshInfo& mesh)
        {
            return mesh.type == donut::engine::MeshType::Triangles && !mesh.isSkinPrototype
                && !mesh.skinPrototype && !mesh.isMorphTargetAnimationMesh;
        }

        explicit CpuMeshInputIdentity(const donut::engine::MeshInfo& mesh)
            : bufferGroup(mesh.buffers), indexOffset(mesh.indexOffset), vertexOffset(mesh.vertexOffset)
            , totalIndices(mesh.totalIndices), totalVertices(mesh.totalVertices), geometries(mesh.geometries.size())
        {
            for (size_t i = 0; i < geometries.size(); ++i)
                if (const auto& source = mesh.geometries[i])
                    geometries[i] = {0, source->indexOffsetInMesh, source->vertexOffsetInMesh,
                        source->numIndices, source->numVertices, source->type};
        }

        bool BindUploadedBuffers(const donut::engine::MeshInfo& mesh)
        {
            const auto buffers = bufferGroup.lock();
            if (!buffers || buffers != mesh.buffers) return false;
            indexBuffer = buffers->indexBuffer;
            vertexBuffer = buffers->vertexBuffer;
            texCoordRange = buffers->getVertexBufferRange(donut::engine::VertexAttribute::TexCoord1);
            return IsValidFor(mesh);
        }

        bool IsValidFor(const donut::engine::MeshInfo& mesh) const
        {
            if (!SupportsStaticMesh(mesh) || !mesh.buffers || bufferGroup.lock() != mesh.buffers
                || !indexBuffer || !vertexBuffer || indexBuffer != mesh.buffers->indexBuffer
                || vertexBuffer != mesh.buffers->vertexBuffer || indexBuffer->getDesc().canHaveUAVs
                || vertexBuffer->getDesc().canHaveUAVs || mesh.indexOffset != indexOffset
                || mesh.vertexOffset != vertexOffset || mesh.totalIndices != totalIndices
                || mesh.totalVertices != totalVertices || mesh.geometries.size() != geometries.size())
                return false;
            const auto& range = mesh.buffers->getVertexBufferRange(donut::engine::VertexAttribute::TexCoord1);
            if (range.byteOffset != texCoordRange.byteOffset || range.byteSize != texCoordRange.byteSize)
                return false;
            for (size_t i = 0; i < geometries.size(); ++i)
            {
                if (!mesh.geometries[i]) return false;
                const auto& current = *mesh.geometries[i];
                const auto& saved = geometries[i];
                if (current.indexOffsetInMesh != saved.indexOffset || current.vertexOffsetInMesh != saved.vertexOffset
                    || current.numIndices != saved.numIndices || current.numVertices != saved.numVertices
                    || current.type != saved.type)
                    return false;
                if (saved.id)
                {
                    const uint64_t indexEnd = (uint64_t(indexOffset) + saved.indexOffset + saved.numIndices) * sizeof(uint32_t);
                    const uint64_t uvEnd = (uint64_t(vertexOffset) + saved.vertexOffset + saved.numVertices) * sizeof(donut::math::float2);
                    if (indexEnd > indexBuffer->getDesc().byteSize || uvEnd > range.byteSize
                        || range.byteOffset > vertexBuffer->getDesc().byteSize
                        || uvEnd > vertexBuffer->getDesc().byteSize - range.byteOffset)
                        return false;
                }
            }
            return true;
        }

        uint64_t GetGeometryId(size_t geometryIndex) const
        {
            return geometryIndex < geometries.size() ? geometries[geometryIndex].id : 0;
        }
    };
}
