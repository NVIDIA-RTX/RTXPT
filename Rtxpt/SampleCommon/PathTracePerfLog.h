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

#include <donut/core/log.h>
#include <donut/core/math/math.h>
#include <nvrhi/nvrhi.h>

#include <algorithm>
#include <fstream>
#include <string>
#include <vector>

// Per-frame GPU timings of the path tracing passes, appended to a CSV (--perfLog). Each pass has a ring of timer
// queries covering the frames in flight; a frame's row is written once the ring comes back around to it, by which
// time its queries have resolved, and the rest by Flush on exit. Rows carry the scene time so a capture sequence's
// recorded frames can be picked out exactly, and are flushed one by one because the DLSS-RR shutdown path can crash.
class PathTracePerfLog
{
public:
    enum Pass { Build, Compaction, Fill, PassCount };

    ~PathTracePerfLog() { Flush(); }

    // Writes the rows still in the ring, oldest first. Only once their command lists have been submitted, and not per
    // frame: it waits for the GPU.
    void Flush()
    {
        if (!IsEnabled())
            return;
        std::vector<Slot*> pending;
        for (Slot& slot : m_slots)
            if (slot.Pending)
                pending.push_back(&slot);
        std::sort(pending.begin(), pending.end(), [](const Slot* a, const Slot* b) { return a->Frame < b->Frame; });
        for (Slot* slot : pending)
            WriteRow(*slot);
    }

    void Init(nvrhi::IDevice* device, const std::string& path)
    {
        m_device = device;
        m_file.open(path, std::ios::out | std::ios::app);
        if (!m_file.is_open())
        {
            donut::log::warning("Unable to open perf log '%s'", path.c_str());
            return;
        }
        if (m_file.tellp() == std::streampos(0))
            m_file << "frame,sceneTime,renderWidth,renderHeight,subSamplingMode,pattern,blockSize,buildMs,compactionMs,fillMs\n";
        for (Slot& slot : m_slots)
            for (nvrhi::TimerQueryHandle& query : slot.Queries)
                query = device->createTimerQuery();
    }
    bool IsEnabled() const { return m_file.is_open(); }

    void BeginFrame(uint64_t frame, double sceneTime, dm::uint2 renderSize, uint32_t subSamplingMode, uint32_t pattern, uint32_t blockSize)
    {
        if (!IsEnabled())
            return;
        m_current = &m_slots[frame % std::size(m_slots)];
        WriteRow(*m_current);
        m_current->Frame = frame;
        m_current->SceneTime = sceneTime;
        m_current->RenderSize = renderSize;
        m_current->Settings = dm::uint3(subSamplingMode, pattern, blockSize);
        for (bool& used : m_current->Used)
            used = false;
    }
    void Begin(nvrhi::ICommandList* commandList, Pass pass)
    {
        if (m_current == nullptr)
            return;
        commandList->beginTimerQuery(m_current->Queries[pass]);
        m_current->Used[pass] = true;
    }
    void End(nvrhi::ICommandList* commandList, Pass pass)
    {
        if (m_current != nullptr)
            commandList->endTimerQuery(m_current->Queries[pass]);
    }
    void EndFrame()
    {
        if (m_current != nullptr)
            m_current->Pending = true;
        m_current = nullptr;
    }

private:
    struct Slot
    {
        nvrhi::TimerQueryHandle Queries[PassCount];
        bool                    Used[PassCount] = {};
        bool                    Pending = false;
        uint64_t                Frame = 0;
        double                  SceneTime = 0;
        dm::uint2               RenderSize = dm::uint2(0, 0);
        dm::uint3               Settings = dm::uint3(0, 0, 0);
    };

    void WriteRow(Slot& slot)
    {
        if (!slot.Pending)
            return;
        m_file << slot.Frame << ',' << slot.SceneTime << ',' << slot.RenderSize.x << ',' << slot.RenderSize.y << ','
               << slot.Settings.x << ',' << slot.Settings.y << ',' << slot.Settings.z;
        for (int pass = 0; pass < PassCount; pass++)
        {
            m_file << ',';
            if (slot.Used[pass])
            {
                m_file << m_device->getTimerQueryTime(slot.Queries[pass]) * 1000.0f;   // blocks only if somehow still in flight
                m_device->resetTimerQuery(slot.Queries[pass]);
            }
        }
        m_file << '\n' << std::flush;
        slot.Pending = false;
    }

    nvrhi::DeviceHandle m_device;
    std::ofstream       m_file;
    Slot                m_slots[8];     // comfortably more than the frames in flight
    Slot*               m_current = nullptr;
};
