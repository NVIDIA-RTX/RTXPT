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
#include "../SampleCommon/AccelerationStructureUtil.h"
#include "OmmBakeInputKey.h"

#include <nvrhi/nvrhi.h>

namespace bvh { class BlasCompactor; }
struct MeshDebugData;
struct MeshGeometryDebugData;
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
			uint32_t maxSubdivisionLevel = 6;
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
		nvrhi::DeviceHandle& device, 
		std::shared_ptr<donut::engine::DescriptorTableManager>,
		std::shared_ptr<donut::engine::ShaderFactory> shaderFactory
	);
	~OmmBuildQueue();

	void Update(nvrhi::ICommandList& commandList);
	void CancelPendingBuilds();
	void QueueBuild(const BuildInput& inputs);
	uint32_t NumPendingBuilds() const;
	void SetRetainDebugData(bool retain);

	// When set, OMM BLASes are compacted into the shared pool rather than each keeping the buffer its
	// build reserved. Null leaves the original per-BLAS allocation path in place.
	void SetBlasCompactor(bvh::BlasCompactor* compactor) { m_blasCompactor = compactor; }

private:

	enum BuildState
	{
		None,
		WaitingForSharedBake,
		SharedBuild,
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

	struct SharedBake
	{
		std::vector<nvrhi::TextureHandle> alphaTextures;
		std::vector<bvh::OmmAttachment> attachments;
		std::shared_ptr<MeshDebugData> debugData;
		std::vector<MeshGeometryDebugData> geometryDebugData;
		bool ready = false;
	};

	struct BuildTask
	{
		BuildInput input;
		BuildState state = BuildState::None;
		std::vector<omm::GpuBakeNvrhi::Input> bakeInputs;
		std::shared_ptr<SharedBake> sharedBake;

		Buffers buffers;
		std::vector<BufferInfo> bufferInfos;

		BuildTask(const BuildInput& input) : input(input) {}
		void Reset();
	};

	void ConsumeOneTask(nvrhi::ICommandList& commandList, BuildState taskState);
	bool ExecuteTask(nvrhi::ICommandList& commandList, BuildTask& taskState); // Returns whether the task is finished and can be removed from the queue

	bool MakeInputKey(const BuildTask& task, omm::BakeInputKey& key) const;
	void FindSharedBake(BuildTask& task);
	void RunSharedBuild(nvrhi::ICommandList& commandList, BuildTask& task);
	void FinalizeSharedBuild(BuildTask& task);
	void RunSetup(nvrhi::ICommandList& commandList, BuildTask& task);
	void RunBakeAndBuild(nvrhi::ICommandList& commandList, BuildTask& task);
	void Finalize(nvrhi::ICommandList& commandList, BuildTask& task);
	
	void AllocateOMMArrayDataBuffer(BuildTask& task);
	void BakeOmmArrayData(nvrhi::ICommandList& commandList, BuildTask& task);
	std::vector<bvh::OmmAttachment> BuildOMMAttachments(nvrhi::ICommandList& commandList, BuildTask& task);
	void BuildBLASWithOMM(nvrhi::ICommandList& commandList, BuildTask& task, const std::vector<bvh::OmmAttachment>& ommAttachment);

	bool ReadyToRecordWork();
	void SubmitAndSubscribeQuery(nvrhi::ICommandList& commandList);
	void CreateBaker(nvrhi::ICommandList& commandList);

	std::vector<BuildTask> m_pending;
	// Build-only ownership keeps triangle mappings alive through every consumer BLAS.
	// Meshes own the shared hardware arrays after this cache is cleared.
	std::unordered_map<omm::BakeInputKey, std::shared_ptr<SharedBake>, omm::BakeInputKeyHash> m_sharedBakes;
	bool m_retainDebugData = false;
	nvrhi::EventQueryHandle m_InFlightQuery;

	nvrhi::DeviceHandle m_device;
	bvh::BlasCompactor* m_blasCompactor = nullptr;
	std::shared_ptr<donut::engine::DescriptorTableManager> m_descriptorTable;
	std::shared_ptr<donut::engine::ShaderFactory> m_shaderFactory;
	// Only alive while builds are queued: it owns a scratch pool (up to 256 MB) that it never shrinks, so it is released
	// whenever the queue empties and Update recreates it for the next batch.
	std::unique_ptr<omm::GpuBakeNvrhi> m_baker;
};
