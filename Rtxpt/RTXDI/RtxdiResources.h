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

#include <nvrhi/nvrhi.h>

namespace rtxdi
{
    class ReSTIRDIContext;
    class RISBufferSegmentAllocator;
}

class RtxdiResources
{
private:
    bool m_NeighborOffsetsInitialized = false;
    uint32_t m_MaxEmissiveMeshes = 0;
    uint32_t m_MaxEmissiveTriangles = 0;
    uint32_t m_MaxPrimitiveLights = 0;
    uint32_t m_MaxGeometryInstances = 0;

public:
    nvrhi::BufferHandle TaskBuffer;
    nvrhi::BufferHandle PrimitiveLightBuffer;
    nvrhi::BufferHandle LightDataBuffer;
    nvrhi::BufferHandle GeometryInstanceToLightBuffer;
    nvrhi::BufferHandle LightIndexMappingBuffer;
    nvrhi::BufferHandle RisBuffer;
    nvrhi::BufferHandle RisLightDataBuffer;
    nvrhi::BufferHandle NeighborOffsetsBuffer;
    nvrhi::BufferHandle LightReservoirBuffer;
    nvrhi::BufferHandle GIReservoirBuffer;
    nvrhi::TextureHandle LocalLightPdfTexture;

    RtxdiResources(
        nvrhi::IDevice* device, 
        const rtxdi::ReSTIRDIContext& context,
        const rtxdi::RISBufferSegmentAllocator& risBufferSegmentAllocator,
        uint32_t maxEmissiveMeshes,
        uint32_t maxEmissiveTriangles,
        uint32_t maxPrimitiveLights,
        uint32_t maxGeometryInstances);

    void InitializeNeighborOffsets(nvrhi::ICommandList* commandList, uint32_t neighborOffsetCount);

    uint32_t GetMaxEmissiveMeshes() const { return m_MaxEmissiveMeshes; }
    uint32_t GetMaxEmissiveTriangles() const { return m_MaxEmissiveTriangles; }
    uint32_t GetMaxPrimitiveLights() const { return m_MaxPrimitiveLights; }
    uint32_t GetMaxGeometryInstances() const { return m_MaxGeometryInstances; }

    nvrhi::BufferHandle GetRisLightDataBuffer() const { return RisLightDataBuffer; }
    nvrhi::BufferHandle GetLightDataBuffer() const { return LightDataBuffer; }
    nvrhi::BufferHandle GetRisBuffer() const { return RisBuffer; }

    static constexpr uint32_t c_NumReservoirBuffers = 3;
    static constexpr uint32_t c_NumGIReservoirBuffers = 2;

};