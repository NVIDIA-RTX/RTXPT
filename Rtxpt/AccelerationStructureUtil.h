/*
* Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#pragma once

#include <donut/engine/SceneTypes.h>
#include <nvrhi/nvrhi.h>
#include "Materials/MaterialsBaker.h"

namespace bvh
{
    struct Config
    {
        bool excludeTransmissive = false;
    };

    struct OmmAttachment
    {
        nvrhi::rt::OpacityMicromapHandle ommBuffer;
        nvrhi::Format ommIndexFormat = nvrhi::Format::UNKNOWN;
        std::vector<nvrhi::rt::OpacityMicromapUsageCount> ommIndexHistogram;
        nvrhi::BufferHandle ommIndexBuffer;
        uint32_t ommIndexBufferOffset = 0;
        nvrhi::BufferHandle ommArrayDataBuffer;
        uint32_t ommArrayDataBufferOffset = 0;
    };

    static nvrhi::rt::AccelStructDesc GetMeshBlasDesc(
        const Config& cfg,
        const donut::engine::MeshInfo& mesh,
        const std::vector<OmmAttachment>* ommAttachment)
    {
        nvrhi::rt::AccelStructDesc blasDesc;
        blasDesc.isTopLevel = false;
        blasDesc.debugName = mesh.name;

        for (uint32_t geomIt = 0; geomIt < mesh.geometries.size(); ++geomIt)
        {
            const donut::engine::MeshGeometry* geometry = mesh.geometries[geomIt].get();

            nvrhi::rt::GeometryDesc geometryDesc;
            auto& triangles = geometryDesc.geometryData.triangles;
            triangles.indexBuffer = mesh.buffers->indexBuffer;
            triangles.indexOffset = (mesh.indexOffset + geometry->indexOffsetInMesh) * sizeof(uint32_t);
            triangles.indexFormat = nvrhi::Format::R32_UINT;
            triangles.indexCount = geometry->numIndices;
            triangles.vertexBuffer = mesh.buffers->vertexBuffer;
            triangles.vertexOffset = (mesh.vertexOffset + geometry->vertexOffsetInMesh) * sizeof(donut::math::float3) + mesh.buffers->getVertexBufferRange(donut::engine::VertexAttribute::Position).byteOffset;
            triangles.vertexFormat = nvrhi::Format::RGB32_FLOAT;
            triangles.vertexStride = sizeof(donut::math::float3);
            triangles.vertexCount = geometry->numVertices;

            MaterialPT & materialPT = *MaterialPT::FromDonut(geometry->material);

            if (cfg.excludeTransmissive &&
                geometry->material->domain == donut::engine::MaterialDomain::Transmissive)
            {
                constexpr float nan = std::numeric_limits<float>::quiet_NaN();
                constexpr nvrhi::rt::AffineTransform c_NanTransform = {
                        nan, nan, nan, nan,
                        nan, nan, nan, nan,
                        nan, nan, nan, nan
                };
                geometryDesc.setTransform(c_NanTransform);
            }

            if (ommAttachment)
            {
                const OmmAttachment& omm = (*ommAttachment)[geomIt];
                triangles.opacityMicromap = omm.ommBuffer;
                triangles.ommIndexBuffer = omm.ommIndexBuffer;
                triangles.ommIndexBufferOffset = omm.ommIndexBufferOffset;
                triangles.ommIndexFormat = omm.ommIndexFormat;
                triangles.pOmmUsageCounts = omm.ommIndexHistogram.data();
                triangles.numOmmUsageCounts = (uint32_t)omm.ommIndexHistogram.size();
            }

            geometryDesc.geometryType = nvrhi::rt::GeometryType::Triangles;
            
            // both EnableAlphaTesting and ExcludeFromNEE require custom shader handling
            geometryDesc.flags = (materialPT.EnableAlphaTesting || materialPT.ExcludeFromNEE) ? nvrhi::rt::GeometryFlags::None : nvrhi::rt::GeometryFlags::Opaque;
            blasDesc.bottomLevelGeometries.push_back(geometryDesc);
        }

        // don't compact acceleration structures that are built per frame
        if (mesh.skinPrototype.use_count() != 0)
        {
            blasDesc.buildFlags = nvrhi::rt::AccelStructBuildFlags::PreferFastBuild;
        }
        else
        {
            blasDesc.buildFlags = nvrhi::rt::AccelStructBuildFlags::PreferFastTrace | nvrhi::rt::AccelStructBuildFlags::AllowCompaction;
        }

        return blasDesc;
    }
} // namespace util
