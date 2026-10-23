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

#include <nvrhi/nvrhi.h>
#include <tracy/Tracy.hpp>

namespace tracy { struct SourceLocationData; }

// Tracy GPU zones (D3D12 only for now). Begin/End are drop-ins for nvrhi beginMarker/endMarker and still emit them.
// A zone may span a close/execute/open of its command list, but every list holding zones must execute before Present.
namespace GpuProfiler
{
    void Init(nvrhi::IDevice* device);
    void Shutdown();        // GPU must be idle
    void AfterPresent();    // once per frame, after the frame's last ExecuteCommandLists and right before FrameMark

    // srcloc == nullptr interns by name (for dynamic names).
    void Begin(nvrhi::ICommandList* commandList, const char* name, const tracy::SourceLocationData* srcloc = nullptr);
    void End(nvrhi::ICommandList* commandList);

    // Tracy-only "Frame" zone (no marker) annotated with FrameNumber(); closed with End. The viewer keys notes by query id, so
    // only the last 16K frames of a capture keep theirs; a handshake racing the first Present can offset labels by one.
    void BeginFrameZone(nvrhi::ICommandList* commandList);

    // The number Tracy's frame bar shows for the frame being recorded.
    uint64_t FrameNumber();

    class Scope
    {
    public:
        Scope(nvrhi::ICommandList* commandList, const char* name, const tracy::SourceLocationData* srcloc = nullptr) : m_commandList(commandList) { Begin(commandList, name, srcloc); }
        ~Scope() { End(m_commandList); }
        Scope(const Scope&) = delete;
        Scope& operator=(const Scope&) = delete;
    private:
        nvrhi::ICommandList* m_commandList;
    };
}

#define RTXPT_GPU_CONCAT_IMPL(a, b) a##b
#define RTXPT_GPU_CONCAT(a, b) RTXPT_GPU_CONCAT_IMPL(a, b)

#ifdef TRACY_ENABLE
#define RTXPT_GPU_SCOPE_IMPL(commandList, name, var) \
    static constexpr tracy::SourceLocationData RTXPT_GPU_CONCAT(var, _srcloc) { name, TracyFunction, TracyFile, (uint32_t)TracyLine, 0 }; \
    GpuProfiler::Scope var(commandList, name, &RTXPT_GPU_CONCAT(var, _srcloc))
#else
#define RTXPT_GPU_SCOPE_IMPL(commandList, name, var) GpuProfiler::Scope var(commandList, name)
#endif

// Scoped marker + Tracy GPU zone; name must be a string literal (use Begin/End for dynamic names).
#define RTXPT_GPU_SCOPE(commandList, name) RTXPT_GPU_SCOPE_IMPL(commandList, name, RTXPT_GPU_CONCAT(_rtxpt_gpu_scope_, __COUNTER__))
