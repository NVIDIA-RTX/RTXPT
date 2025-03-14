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
#include <donut/engine/TextureCache.h>
#include "../AccelerationStructureUtil.h"

#include <nvrhi/nvrhi.h>
#include <omm-gpu-nvrhi.h>
#include <unordered_map>
#include <list>

namespace donut
{
	namespace engine
	{
		class ShaderFactory;
		class DescriptorTableManager;
	}
}

class OmmBuildQueue
{
public:
	struct BuildInput
	{
		struct Geometry
		{
			int geometryIndexInMesh = -1;
			std::shared_ptr < donut::engine::TextureData > alphaTexture;

			// Settings
			uint32_t maxSubdivisionLevel = 5;
			float dynamicSubdivisionScale = 2.f;
			nvrhi::rt::OpacityMicromapFormat format = nvrhi::rt::OpacityMicromapFormat::OC1_4_State;
			nvrhi::rt::OpacityMicromapBuildFlags flags = nvrhi::rt::OpacityMicromapBuildFlags::FastTrace;
			uint32_t maxOmmArrayDataSizeInMB; // Limit OMM memory footprint to this value.
			omm::OpacityState alphaCutoffGT = omm::OpacityState::Opaque;
			omm::OpacityState alphaCutoffLE = omm::OpacityState::Transparent;

			// Debug settings
			bool computeOnly = false;
			bool enableLevelLineIntersection = true;
			bool enableTexCoordDeduplication = true;
			bool force32BitIndices = false;
			bool enableSpecialIndices = true;
			bool enableNsightDebugMode = false;
		};

		std::shared_ptr < donut::engine::MeshInfo > mesh;
		std::vector < Geometry > geometries;
		bvh::Config bvhCfg;
	};

	OmmBuildQueue(
		nvrhi::DeviceHandle device, 
		std::shared_ptr<donut::engine::DescriptorTableManager>,
		std::shared_ptr<donut::engine::ShaderFactory> shaderFactory
	);
	~OmmBuildQueue();

	void Initialize(nvrhi::CommandListHandle commandList);
	void Update(nvrhi::CommandListHandle commandList);
	void CancelPendingBuilds();
	void QueueBuild(const BuildInput& inputs);
	uint32_t NumPendingBuilds() const;

private:

	enum BuildState
	{
		None,
		Setup,
		BakeAndBuild,
	};

	struct BufferInfo
	{
		nvrhi::Format	ommIndexFormat;
		uint32_t		ommIndexCount;
		size_t			ommIndexOffset;
		size_t			ommDescArrayOffset;
		size_t			ommDescArrayHistogramOffset;
		size_t			ommDescArrayHistogramSize;
		size_t			ommDescArrayHistogramReadbackOffset;
		size_t			ommIndexHistogramOffset;
		size_t			ommIndexHistogramSize;
		size_t			ommIndexHistogramReadbackOffset;
		size_t			ommPostDispatchInfoOffset;
		size_t			ommPostDispatchInfoReadbackOffset;

		// below will be populated after Setup pass has finished.
		uint32_t		ommArrayDataOffset;
		std::vector<nvrhi::rt::OpacityMicromapUsageCount> ommIndexHistogram;
		std::vector<nvrhi::rt::OpacityMicromapUsageCount> ommArrayHistogram;
	};           

	struct Buffers
	{
		nvrhi::BufferHandle ommArrayDataBuffer;
		nvrhi::BufferHandle ommIndexBuffer;
		nvrhi::BufferHandle ommDescBuffer;
		nvrhi::BufferHandle ommDescArrayHistogramBuffer;
		nvrhi::BufferHandle ommIndexArrayHistogramBuffer;
		nvrhi::BufferHandle ommPostDispatchInfoBuffer;
		nvrhi::BufferHandle ommReadbackBuffer;
	};

	struct BuildTask
	{
		BuildInput input;
		BuildState state = BuildState::None;
		nvrhi::EventQueryHandle query;

		Buffers buffers;
		std::vector<BufferInfo> bufferInfos;

		BuildTask(const BuildInput& input) : input(input) {}
	};

	void RunSetup(nvrhi::CommandListHandle commandList, BuildTask& task);
	void RunBakeAndBuild(nvrhi::CommandListHandle commandList, BuildTask& task);
	void Finalize(nvrhi::CommandListHandle commandList, BuildTask& task);

	std::list< BuildTask > m_pending;

	nvrhi::DeviceHandle m_device;
	std::shared_ptr<donut::engine::DescriptorTableManager> m_descriptorTable;
	std::shared_ptr<donut::engine::ShaderFactory> m_shaderFactory;
	std::unique_ptr<omm::GpuBakeNvrhi> m_baker;
};
