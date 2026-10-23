/*
* Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "GpuProfiler.h"

#include <cassert>
#include <vector>

#if defined(TRACY_ENABLE) && DONUT_WITH_DX12
#define RTXPT_GPU_PROFILER_D3D12 1
#else
#define RTXPT_GPU_PROFILER_D3D12 0
#endif

#if RTXPT_GPU_PROFILER_D3D12
#include <nvrhi/d3d12.h>
#include <client/TracyProfiler.hpp>     // internal API (QueueSerial, Gpu* items) - the same one tracy/TracyD3D12.hpp uses

#include <algorithm>
#include <atomic>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#endif

namespace
{
#if RTXPT_GPU_PROFILER_D3D12
    constexpr uint32_t c_invalidQuery = ~0u;
    constexpr uint64_t c_noConnection = ~0ull;

    // Tracy query ids are 16 bit: zones use their heap slot, in [0, 32K), and frame zones get [32K, 64K) to themselves so
    // their annotation, which the viewer keys by query id, survives 16K frames before being overwritten.
    constexpr uint32_t c_frameZoneIdBase = 32 * 1024;
    constexpr uint32_t c_frameZoneIdCount = 32 * 1024;
    constexpr int64_t c_frameNoteId = 0;

    bool IsConnected()
    {
#ifdef TRACY_ON_DEMAND
        return tracy::GetProfiler().IsConnected();
#else
        return true;
#endif
    }

    uint64_t ConnectionId()
    {
#ifdef TRACY_ON_DEMAND
        return tracy::GetProfiler().ConnectionId();
#else
        return 0;
#endif
    }

    // Sends the item now and, with on-demand profiling, replays it to every later connection.
    void QueueSerialFinishPersistent(tracy::QueueItem* item)
    {
#ifdef TRACY_ON_DEMAND
        tracy::GetProfiler().DeferItem(*item);
#endif
        tracy::Profiler::QueueSerialFinish();
    }

    // Brackets the call with Tracy timestamps, since GetClockCalibration returns QPC rather than Tracy time.
    bool ClockCalibration(ID3D12CommandQueue* queue, uint64_t& gpuTimestamp, uint64_t& qpcTimestamp, int64_t& tracyTime)
    {
        const int64_t before = tracy::Profiler::GetTime();
        if (FAILED(queue->GetClockCalibration(&gpuTimestamp, &qpcTimestamp)))
            return false;
        tracyTime = before + (tracy::Profiler::GetTime() - before) / 2;
        return true;
    }

    // Per-queue Tracy GPU context, after tracy/TracyD3D12.hpp's D3D12QueueCtx. Differences: zones don't hold on to a
    // native list (they may span a list split), one resolve per frame, and connection changes are handled.
    class QueueContext
    {
    public:
        static std::unique_ptr<QueueContext> Create(nvrhi::IDevice* device, nvrhi::CommandQueue queueType, const char* name);

        uint8_t     Id() const              { return m_id; }
        uint32_t    AllocQueryPair(bool frameZone);    // returns the begin query's heap slot
        uint16_t    TracyId(uint32_t query) const   { return uint16_t(m_pairTracyId[query / 2] + (query & 1)); }
        void        WriteTimestamp(ID3D12GraphicsCommandList* nativeList, uint32_t query) { nativeList->EndQuery(m_queryHeap, D3D12_QUERY_TYPE_TIMESTAMP, query); }
        // The connection that got both the pair's begin and end items; only such pairs get their times sent.
        void        SetPairConnection(uint32_t query, uint64_t connection) { m_pairConnection[query / 2] = connection; }
        void        AnnotateFrame(uint32_t query, uint64_t connection, double frameNumber);
        void        Collect();
        void        SubmitFrame();
        void        Drain();

    private:
        struct Payload
        {
            uint32_t    Start;
            uint32_t    Count;
            uint64_t    FenceValue;
        };

        void        Recalibrate(uint64_t connection);

        nvrhi::IDevice*                     m_device = nullptr;
        nvrhi::CommandQueue                 m_queueType = nvrhi::CommandQueue::Graphics;
        ID3D12CommandQueue*                 m_queue = nullptr;          // owned by nvrhi
        nvrhi::RefCountPtr<ID3D12QueryHeap> m_queryHeap;
        nvrhi::RefCountPtr<ID3D12Resource>  m_readback;
        nvrhi::RefCountPtr<ID3D12Fence>     m_fence;
        nvrhi::CommandListHandle            m_resolveList;
        uint32_t                            m_queryLimit = 0;           // ring size, even so a begin/end pair never wraps
        uint8_t                             m_id = 0;

        // Current frame's queries are [m_frameStart, m_frameStart + m_frameCount) in the ring.
        uint32_t                            m_frameStart = 0;
        uint32_t                            m_frameBudget = 0;          // ring space not held by uncollected frames
        std::atomic<uint32_t>               m_frameCount = 0;
        std::vector<uint64_t>               m_pairConnection;           // per begin/end pair: query / 2
        std::vector<uint16_t>               m_pairTracyId;              // per pair: Tracy id of the begin query
        uint32_t                            m_frameZoneCount = 0;
        uint64_t                            m_noteNameConnection = c_noConnection;

        std::deque<Payload>                 m_pending;
        std::vector<std::pair<uint64_t, uint16_t>> m_sendOrder;         // (timestamp, Tracy id), reused by Collect
        uint32_t                            m_pendingQueries = 0;
        uint64_t                            m_fenceValue = 0;
        bool                                m_deviceLost = false;

        uint64_t                            m_creationQpc = 0;
        uint64_t                            m_lastCalibrationQpc = 0;
        uint64_t                            m_calibrationConnection = c_noConnection;
        double                              m_nsPerQpcTick = 0.0;
    };

    std::unique_ptr<QueueContext> QueueContext::Create(nvrhi::IDevice* device, nvrhi::CommandQueue queueType, const char* name)
    {
        ID3D12Device* d3dDevice = device->getNativeObject(nvrhi::ObjectTypes::D3D12_Device);
        ID3D12CommandQueue* queue = device->getNativeQueue(nvrhi::ObjectTypes::D3D12_CommandQueue, queueType);
        if (d3dDevice == nullptr || queue == nullptr)
            return nullptr;

        std::unique_ptr<QueueContext> ctx = std::make_unique<QueueContext>();
        ctx->m_device = device;
        ctx->m_queueType = queueType;
        ctx->m_queue = queue;

        D3D12_QUERY_HEAP_DESC heapDesc = {};
        heapDesc.Type = D3D12_QUERY_HEAP_TYPE_TIMESTAMP;
        for (ctx->m_queryLimit = c_frameZoneIdBase; ctx->m_queryLimit >= 1024; ctx->m_queryLimit /= 2)
        {
            heapDesc.Count = ctx->m_queryLimit;
            if (SUCCEEDED(d3dDevice->CreateQueryHeap(&heapDesc, IID_PPV_ARGS(&ctx->m_queryHeap))))
                break;
        }
        if (ctx->m_queryHeap.Get() == nullptr)
            return nullptr;

        D3D12_HEAP_PROPERTIES heapProps = {};
        heapProps.Type = D3D12_HEAP_TYPE_READBACK;
        D3D12_RESOURCE_DESC bufferDesc = {};
        bufferDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        bufferDesc.Width = uint64_t(ctx->m_queryLimit) * sizeof(uint64_t);
        bufferDesc.Height = 1;
        bufferDesc.DepthOrArraySize = 1;
        bufferDesc.MipLevels = 1;
        bufferDesc.SampleDesc.Count = 1;
        bufferDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        if (FAILED(d3dDevice->CreateCommittedResource(&heapProps, D3D12_HEAP_FLAG_NONE, &bufferDesc, D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&ctx->m_readback))))
            return nullptr;
        if (FAILED(d3dDevice->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&ctx->m_fence))))
            return nullptr;

        uint64_t frequency = 0;
        uint64_t gpuTimestamp = 0;
        int64_t tracyTime = 0;
        LARGE_INTEGER qpcFrequency = {};
        if (FAILED(queue->GetTimestampFrequency(&frequency)) || frequency == 0 || !QueryPerformanceFrequency(&qpcFrequency)
            || !ClockCalibration(queue, gpuTimestamp, ctx->m_creationQpc, tracyTime))
            return nullptr;
        ctx->m_nsPerQpcTick = 1e9 / double(qpcFrequency.QuadPart);
        ctx->m_lastCalibrationQpc = ctx->m_creationQpc;
        ctx->m_calibrationConnection = ConnectionId();
        ctx->m_frameBudget = ctx->m_queryLimit;
        ctx->m_pairConnection.assign(ctx->m_queryLimit / 2, c_noConnection);
        ctx->m_pairTracyId.assign(ctx->m_queryLimit / 2, 0);

        ctx->m_resolveList = device->createCommandList(nvrhi::CommandListParameters().setQueueType(queueType).setEnableImmediateExecution(false));
        if (ctx->m_resolveList == nullptr)
            return nullptr;

        const int32_t contextId = tracy::NextGpuContextId();
        if (contextId == tracy::InvalidGpuContextId)
            return nullptr;
        ctx->m_id = uint8_t(contextId);

        tracy::QueueItem* item = tracy::Profiler::QueueSerial();
        tracy::MemWrite(&item->hdr.type, tracy::QueueType::GpuNewContext);
        tracy::MemWrite(&item->gpuNewContext.cpuTime, tracyTime);
        tracy::MemWrite(&item->gpuNewContext.gpuTime, int64_t(gpuTimestamp));
        tracy::MemWrite(&item->gpuNewContext.thread, uint32_t(0));     // not bound to a thread, like tracy's D3D12 contexts
        tracy::MemWrite(&item->gpuNewContext.period, float(1e9 / double(frequency)));
        tracy::MemWrite(&item->gpuNewContext.context, ctx->m_id);
        tracy::MemWrite(&item->gpuNewContext.flags, tracy::GpuContextCalibration);
        tracy::MemWrite(&item->gpuNewContext.type, tracy::GpuContextType::Direct3D12);
        QueueSerialFinishPersistent(item);

        const size_t nameLength = strlen(name);
        char* nameCopy = (char*)tracy::tracy_malloc(nameLength);
        memcpy(nameCopy, name, nameLength);
        item = tracy::Profiler::QueueSerial();
        tracy::MemWrite(&item->hdr.type, tracy::QueueType::GpuContextName);
        tracy::MemWrite(&item->gpuContextNameFat.context, ctx->m_id);
        tracy::MemWrite(&item->gpuContextNameFat.ptr, reinterpret_cast<uint64_t>(nameCopy));
        tracy::MemWrite(&item->gpuContextNameFat.size, uint16_t(nameLength));
        QueueSerialFinishPersistent(item);

        return ctx;
    }

    uint32_t QueueContext::AllocQueryPair(bool frameZone)
    {
        uint32_t count = m_frameCount.load(std::memory_order_relaxed);
        do
        {
            if (count + 2 > m_frameBudget)
                return c_invalidQuery;  // zone goes untimed rather than overwrite a frame still waiting to be collected
        } while (!m_frameCount.compare_exchange_weak(count, count + 2, std::memory_order_relaxed));
        const uint32_t query = (m_frameStart + count) % m_queryLimit;
        m_pairTracyId[query / 2] = uint16_t(frameZone ? c_frameZoneIdBase + (m_frameZoneCount++ * 2) % c_frameZoneIdCount : query);
        return query;
    }

    void QueueContext::AnnotateFrame(uint32_t query, uint64_t connection, double frameNumber)
    {
        // Tracy frees annotation names after sending and doesn't replay them on reconnect, so send a fresh copy per connection.
        tracy::QueueItem* item;
        if (m_noteNameConnection != connection)
        {
            m_noteNameConnection = connection;
            static const char c_noteName[] = "CPU frame";
            char* nameCopy = (char*)tracy::tracy_malloc(sizeof(c_noteName) - 1);
            memcpy(nameCopy, c_noteName, sizeof(c_noteName) - 1);
            item = tracy::Profiler::QueueSerial();
            tracy::MemWrite(&item->hdr.type, tracy::QueueType::GpuAnnotationName);
            tracy::MemWrite(&item->gpuAnnotationNameFat.noteId, c_frameNoteId);
            tracy::MemWrite(&item->gpuAnnotationNameFat.context, m_id);
            tracy::MemWrite(&item->gpuAnnotationNameFat.ptr, reinterpret_cast<uint64_t>(nameCopy));
            tracy::MemWrite(&item->gpuAnnotationNameFat.size, uint16_t(sizeof(c_noteName) - 1));
            tracy::Profiler::QueueSerialFinish();
        }

        item = tracy::Profiler::QueueSerial();
        tracy::MemWrite(&item->hdr.type, tracy::QueueType::GpuZoneAnnotation);
        tracy::MemWrite(&item->zoneAnnotation.noteId, c_frameNoteId);
        tracy::MemWrite(&item->zoneAnnotation.value, frameNumber);
        tracy::MemWrite(&item->zoneAnnotation.thread, tracy::GetThreadHandle());
        tracy::MemWrite(&item->zoneAnnotation.queryId, TracyId(query));
        tracy::MemWrite(&item->zoneAnnotation.context, m_id);
        tracy::Profiler::QueueSerialFinish();
    }

    void QueueContext::Recalibrate(uint64_t connection)
    {
        // A new connection only knows the creation calibration (replayed from the deferred GpuNewContext), so measure from it.
        if (connection != m_calibrationConnection)
        {
            m_calibrationConnection = connection;
            m_lastCalibrationQpc = m_creationQpc;
        }

        uint64_t gpuTimestamp = 0;
        uint64_t qpcTimestamp = 0;
        int64_t tracyTime = 0;
        if (!ClockCalibration(m_queue, gpuTimestamp, qpcTimestamp, tracyTime) || qpcTimestamp <= m_lastCalibrationQpc)
            return;
        const int64_t cpuDelta = int64_t(double(qpcTimestamp - m_lastCalibrationQpc) * m_nsPerQpcTick);
        m_lastCalibrationQpc = qpcTimestamp;

        tracy::QueueItem* item = tracy::Profiler::QueueSerial();
        tracy::MemWrite(&item->hdr.type, tracy::QueueType::GpuCalibration);
        tracy::MemWrite(&item->gpuCalibration.gpuTime, int64_t(gpuTimestamp));
        tracy::MemWrite(&item->gpuCalibration.cpuTime, tracyTime);
        tracy::MemWrite(&item->gpuCalibration.cpuDelta, cpuDelta);
        tracy::MemWrite(&item->gpuCalibration.context, m_id);
        tracy::Profiler::QueueSerialFinish();
    }

    void QueueContext::Collect()
    {
        if (m_deviceLost || m_pending.empty())
            return;

        const uint64_t completed = m_fence->GetCompletedValue();
        if (completed == UINT64_MAX)    // device removed
        {
            m_deviceLost = true;
            return;
        }

        // Flag before id (Tracy bumps the id before setting the flag), so a reconnect in between can't pair an old id with 'connected'.
        const bool connected = IsConnected();
        const uint64_t connection = ConnectionId();
        size_t readyCount = 0;
        while (readyCount < m_pending.size() && m_pending[readyCount].FenceValue <= completed)
            readyCount++;

        const uint64_t* timestamps = nullptr;
        if (readyCount > 0 && connected)
        {
            Recalibrate(connection);    // before the times, so a fresh connection maps them with current clocks
            D3D12_RANGE readRange = { 0, size_t(m_queryLimit) * sizeof(uint64_t) };
            void* mapped = nullptr;
            if (SUCCEEDED(m_readback->Map(0, &readRange, &mapped)))
                timestamps = static_cast<const uint64_t*>(mapped);
        }

        for (; readyCount > 0; readyCount--, m_pending.pop_front())
        {
            const Payload& payload = m_pending.front();
            m_pendingQueries -= payload.Count;

            // Only pairs whose begin and end both went to this connection - the viewer asserts on queries it never saw.
            m_sendOrder.clear();
            for (uint32_t i = 0; i < payload.Count; i += 2)
            {
                const uint32_t query = (payload.Start + i) % m_queryLimit;
                if (timestamps != nullptr && m_pairConnection[query / 2] == connection)
                {
                    m_sendOrder.push_back({ timestamps[query], TracyId(query) });
                    m_sendOrder.push_back({ timestamps[query + 1], TracyId(query + 1) });
                }
                m_pairConnection[query / 2] = c_noConnection;
            }

            // Time order, not query order: a parent's end slot precedes its children's begin slots, and a backward step
            // of over 2^31 ticks (a parent longer than ~2s) trips the viewer's timer-wrap detection for the rest of the session.
            std::stable_sort(m_sendOrder.begin(), m_sendOrder.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
            for (const auto& [timestamp, tracyId] : m_sendOrder)
            {
                tracy::QueueItem* item = tracy::Profiler::QueueSerial();
                tracy::MemWrite(&item->hdr.type, tracy::QueueType::GpuTime);
                tracy::MemWrite(&item->gpuTime.gpuTime, int64_t(timestamp));
                tracy::MemWrite(&item->gpuTime.queryId, tracyId);
                tracy::MemWrite(&item->gpuTime.context, m_id);
                tracy::Profiler::QueueSerialFinish();
            }
        }

        if (timestamps != nullptr)
        {
            D3D12_RANGE writtenRange = { 0, 0 };
            m_readback->Unmap(0, &writtenRange);
        }
    }

    void QueueContext::SubmitFrame()
    {
        const uint32_t count = m_frameCount.exchange(0, std::memory_order_relaxed);

        // One resolve for the whole frame, on its own list after everything else this frame put on the queue.
        if (count > 0 && !m_deviceLost)
        {
            m_resolveList->open();
            ID3D12GraphicsCommandList* nativeList = m_resolveList->getNativeObject(nvrhi::ObjectTypes::D3D12_GraphicsCommandList);
            const uint32_t firstCount = std::min(count, m_queryLimit - m_frameStart);
            nativeList->ResolveQueryData(m_queryHeap, D3D12_QUERY_TYPE_TIMESTAMP, m_frameStart, firstCount, m_readback, uint64_t(m_frameStart) * sizeof(uint64_t));
            if (firstCount < count)
                nativeList->ResolveQueryData(m_queryHeap, D3D12_QUERY_TYPE_TIMESTAMP, 0, count - firstCount, m_readback, 0);
            m_resolveList->close();
            m_device->executeCommandList(m_resolveList, m_queueType);
            if (FAILED(m_queue->Signal(m_fence, m_fenceValue + 1)))
                m_deviceLost = true;    // the payload could never complete, so it isn't queued
            else
            {
                m_pending.push_back({ m_frameStart, count, ++m_fenceValue });
                m_pendingQueries += count;
            }
            m_frameStart = (m_frameStart + count) % m_queryLimit;
        }
        m_frameBudget = m_deviceLost ? 0 : m_queryLimit - m_pendingQueries;
    }

    void QueueContext::Drain()
    {
        // The last frames' payloads are still pending after the final AfterPresent. NVRHI's waitForIdle covers only its own
        // fence, not ours signaled after it, so wait for ours (bounded, in case the device hangs) and publish them.
        if (m_deviceLost || m_pending.empty())
            return;
        const uint64_t lastSignaled = m_pending.back().FenceValue;
        if (m_fence->GetCompletedValue() < lastSignaled)   // UINT64_MAX once the device is removed
        {
            HANDLE event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
            if (event != nullptr)
            {
                if (SUCCEEDED(m_fence->SetEventOnCompletion(lastSignaled, event)))
                    WaitForSingleObject(event, 5000);
                CloseHandle(event);
            }
        }
        Collect();
    }

    nvrhi::IDevice*                 g_device = nullptr;
    std::unique_ptr<QueueContext>   g_contexts[2];                  // Graphics, Compute
    bool                            g_contextFailed[2] = {};
    std::atomic<bool>               g_computeUsed = false;
    uint64_t                        g_seenConnection = c_noConnection;

    ID3D12GraphicsCommandList* NativeList(nvrhi::ICommandList* commandList)
    {
        return commandList->getNativeObject(nvrhi::ObjectTypes::D3D12_GraphicsCommandList);  // nullptr unless open
    }

    QueueContext* ContextFor(nvrhi::ICommandList* commandList)
    {
        switch (commandList->getDesc().queueType)
        {
        case nvrhi::CommandQueue::Graphics:
            return g_contexts[0].get();
        case nvrhi::CommandQueue::Compute:
            g_computeUsed.store(true, std::memory_order_relaxed);
            return g_contexts[1].get();
        default:
            return nullptr;     // copy queues would need a COPY_QUEUE_TIMESTAMP heap
        }
    }

    // Never freed: Tracy reads source locations through these pointers, possibly late.
    const tracy::SourceLocationData* InternName(const char* name)
    {
        struct NameHash
        {
            using is_transparent = void;
            size_t operator()(std::string_view s) const { return std::hash<std::string_view>()(s); }
        };
        static auto* names = new std::unordered_map<std::string, tracy::SourceLocationData, NameHash, std::equal_to<>>();
        static std::mutex mutex;

        std::lock_guard lock(mutex);
        auto it = names->find(std::string_view(name));
        if (it == names->end())
        {
            it = names->emplace(name, tracy::SourceLocationData{}).first;
            it->second = { it->first.c_str(), "", "", 0, 0 };    // no call site known; better none than this file
        }
        return &it->second;
    }
#endif // RTXPT_GPU_PROFILER_D3D12

    struct OpenZone
    {
        nvrhi::ICommandList*    CommandList;
        bool                    EmitMarker;
#if RTXPT_GPU_PROFILER_D3D12
        QueueContext*           Context;        // nullptr when not timed
        uint32_t                Query;
        uint64_t                Connection;
#endif
    };
    thread_local std::vector<OpenZone> t_openZones;

    uint64_t g_frameMarks = 0;      // FrameMark calls so far; AfterPresent runs right before each one
#if defined(TRACY_ENABLE) && defined(TRACY_ON_DEMAND)
    uint64_t g_frameLabelConnection = ~0ull;
    bool     g_connectedBeforeFirstFrame = false;

    void NoteFrameLabelConnection()
    {
        const uint64_t connection = tracy::GetProfiler().ConnectionId();
        if (connection != g_frameLabelConnection)
        {
            g_frameLabelConnection = connection;
            g_connectedBeforeFirstFrame = g_frameMarks == 0;
        }
    }
#endif

    void BeginZone(nvrhi::ICommandList* commandList, const char* name, const tracy::SourceLocationData* srcloc, bool emitMarker, bool frameZone)
    {
        if (emitMarker)
            commandList->beginMarker(name);

        OpenZone zone = { commandList, emitMarker };
#if RTXPT_GPU_PROFILER_D3D12
        QueueContext* context = ContextFor(commandList);
        if (context != nullptr && IsConnected())
        {
            ID3D12GraphicsCommandList* nativeList = NativeList(commandList);
            assert(nativeList != nullptr);  // zone begun on a closed command list
            const uint32_t query = nativeList != nullptr ? context->AllocQueryPair(frameZone) : c_invalidQuery;
            if (query != c_invalidQuery)
            {
                zone.Context = context;
                zone.Query = query;
                zone.Connection = ConnectionId();
                context->WriteTimestamp(nativeList, query);

                const tracy::SourceLocationData* location = srcloc != nullptr ? srcloc : InternName(name);   // not under Tracy's serial lock
                tracy::QueueItem* item = tracy::Profiler::QueueSerial();
                tracy::MemWrite(&item->hdr.type, tracy::QueueType::GpuZoneBeginSerial);
                tracy::MemWrite(&item->gpuZoneBegin.cpuTime, tracy::Profiler::GetTime());
                tracy::MemWrite(&item->gpuZoneBegin.srcloc, reinterpret_cast<uint64_t>(location));
                tracy::MemWrite(&item->gpuZoneBegin.thread, tracy::GetThreadHandle());
                tracy::MemWrite(&item->gpuZoneBegin.queryId, context->TracyId(query));
                tracy::MemWrite(&item->gpuZoneBegin.context, context->Id());
                tracy::Profiler::QueueSerialFinish();
                context->SetPairConnection(query, zone.Connection);
                if (frameZone)
                    context->AnnotateFrame(query, zone.Connection, double(GpuProfiler::FrameNumber()));
            }
        }
#else
        (void)srcloc;
        (void)frameZone;
#endif
        t_openZones.push_back(zone);
    }
}

void GpuProfiler::Init(nvrhi::IDevice* device)
{
#if RTXPT_GPU_PROFILER_D3D12
    if (device->getGraphicsAPI() == nvrhi::GraphicsAPI::D3D12)
        g_device = device;
#endif
}

void GpuProfiler::Shutdown()
{
#if RTXPT_GPU_PROFILER_D3D12
    for (std::unique_ptr<QueueContext>& context : g_contexts)
        if (context != nullptr)
        {
            context->Drain();
            context.reset();
        }
    g_device = nullptr;
#endif
}

void GpuProfiler::AfterPresent()
{
    assert(t_openZones.empty());    // an open zone would get resolved before its end timestamp is written
#if defined(TRACY_ENABLE) && defined(TRACY_ON_DEMAND)
    NoteFrameLabelConnection();     // a connection made during this frame saw g_frameMarks FrameMarks
#endif
    g_frameMarks++;
#if RTXPT_GPU_PROFILER_D3D12
    for (std::unique_ptr<QueueContext>& context : g_contexts)
        if (context != nullptr)
        {
            context->Collect();
            context->SubmitFrame();
        }

    // Created once a connection has been up for a frame (past Tracy's replay of deferred items, which would otherwise race
    // GpuNewContext into being sent twice), so an unprofiled run allocates nothing and adds no GPU work.
    const bool connectedLastFrame = IsConnected() && ConnectionId() == g_seenConnection;
    g_seenConnection = IsConnected() ? ConnectionId() : c_noConnection;
    if (g_device != nullptr && connectedLastFrame)
    {
        const nvrhi::CommandQueue queues[2] = { nvrhi::CommandQueue::Graphics, nvrhi::CommandQueue::Compute };
        const char* names[2] = { "Graphics", "Compute" };
        for (int i = 0; i < 2; i++)
            if (g_contexts[i] == nullptr && !g_contextFailed[i] && (i == 0 || g_computeUsed.load(std::memory_order_relaxed)))
            {
                g_contexts[i] = QueueContext::Create(g_device, queues[i], names[i]);
                g_contextFailed[i] = g_contexts[i] == nullptr;
            }
    }
#endif
}

void GpuProfiler::Begin(nvrhi::ICommandList* commandList, const char* name, const tracy::SourceLocationData* srcloc)
{
    BeginZone(commandList, name, srcloc, true, false);
}

void GpuProfiler::BeginFrameZone(nvrhi::ICommandList* commandList)
{
    BeginZone(commandList, "Frame", nullptr, false, true);
}

uint64_t GpuProfiler::FrameNumber()
{
    // Tracy's frame bar (View::GetFrameNumber, Worker's on-demand handshake) labels the frame recorded after k FrameMarks
    // as k+1, or k+2 when the viewer connected before the first FrameMark.
#if defined(TRACY_ENABLE) && defined(TRACY_ON_DEMAND)
    NoteFrameLabelConnection();
    return g_frameMarks + (g_connectedBeforeFirstFrame ? 2 : 1);
#else
    return g_frameMarks + 1;
#endif
}

void GpuProfiler::End(nvrhi::ICommandList* commandList)
{
    assert(!t_openZones.empty() && t_openZones.back().CommandList == commandList);
    if (t_openZones.empty())
        return;
    const OpenZone zone = t_openZones.back();
    t_openZones.pop_back();

#if RTXPT_GPU_PROFILER_D3D12
    if (zone.Context != nullptr)
    {
        // The end slot is written even if the connection changed, so the frame's single resolve never reads an unwritten query.
        ID3D12GraphicsCommandList* nativeList = NativeList(commandList);
        assert(nativeList != nullptr);  // zone ended on a closed command list
        if (nativeList != nullptr)
            zone.Context->WriteTimestamp(nativeList, zone.Query + 1);
        if (nativeList != nullptr && ConnectionId() == zone.Connection)
        {
            tracy::QueueItem* item = tracy::Profiler::QueueSerial();
            tracy::MemWrite(&item->hdr.type, tracy::QueueType::GpuZoneEndSerial);
            tracy::MemWrite(&item->gpuZoneEnd.cpuTime, tracy::Profiler::GetTime());
            tracy::MemWrite(&item->gpuZoneEnd.thread, tracy::GetThreadHandle());
            tracy::MemWrite(&item->gpuZoneEnd.queryId, zone.Context->TracyId(zone.Query + 1));
            tracy::MemWrite(&item->gpuZoneEnd.context, zone.Context->Id());
            tracy::Profiler::QueueSerialFinish();
        }
        else
            zone.Context->SetPairConnection(zone.Query, c_noConnection);  // its begin went to an earlier connection
    }
#endif

    if (zone.EmitMarker)
        commandList->endMarker();
}
