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

#include <vector>
#include <cassert>
#include <memory>

namespace donut::engine { class SceneGraphNode; class ThreadPool; struct MeshInfo; struct MeshGeometry; }
struct MeshInstanceExtension;
struct PTMaterial;
class ExtendedScene;

// Self-validation re-derives every cached value at Close() and compares, which defeats the point of the cache - so it is
// on in Debug (where it is cheap insurance and, importantly, keeps the code path compiled and exercised) and off in
// Release. Override the define if you want it in a Release build to chase a staleness bug.
#ifndef RTXPT_SCENE_RENDER_CACHE_VALIDATION
#ifdef _DEBUG
#define RTXPT_SCENE_RENDER_CACHE_VALIDATION 1
#else
#define RTXPT_SCENE_RENDER_CACHE_VALIDATION 0
#endif
#endif

// Per-frame cache of scene-derived data that is expensive to re-query but constant for the duration of a frame.
//
// Lifetime: Open() from Sample::Render() immediately after Scene::Refresh(), Close() from Sample::PostRender(). Nothing
// may be read outside that window. Open() is deliberately NOT in PreRender(): Scene::Refresh() is where pending scene graph
// structure changes get applied, so a cache opened before it would be stale on exactly the frames where the graph changed.
// Frames that exit before Open() simply never open, and Close() is a no-op for them.
//
// Entries are filled LAZILY, on first access, rather than eagerly in Open(), so each entry is built after whatever
// produced it, wherever consumers sit in the frame.
//
// NOT DONE, BUT POSSIBLE - cross-frame persistence:
// Node pointers only change when the scene graph restructures, and Donut exposes that signal cheaply via
// SceneGraph::HasPendingStructureChanges() (a single dirty-flag test on the root). The cache could therefore keep its
// contents across frames and only refill when that flag was set, turning a per-frame O(instances) refill into a one-off
// cost at load. Note the flag must be sampled BEFORE Scene::Refresh(), which clears the dirty bits; Scene latches it into
// m_SceneStructureChanged but exposes no accessor, so the sampling has to happen at our own call site.
//
// We deliberately do not do this: the scene is modified at runtime (game object spawning - see GameModel/GameProps), and
// every such change would force a full rebuild, converting a steady per-frame cost into unpredictable frame spikes. A
// smooth constant cost is worth more here than a lower average with hitches. Revisit if the runtime mutation pattern
// changes, or if the rebuild is made incremental.
//
// Not to be confused with what Invalidate() does: the arrays DO keep their bytes across frames, but purely so the
// refill's resize() has nothing to value-initialise. Every element is still recomputed from scratch every frame.
class SceneRenderCache
{
public:
    SceneRenderCache() = default;
    ~SceneRenderCache();

    // Begins the frame window. Discards anything cached for the previous frame.
    // The thread pool must be dedicated to per-frame work: ThreadPool::WaitForTasks() waits on a POOL-WIDE counter, so
    // sharing one with async scene/texture loading would stall the frame on unrelated tasks.
    void            Open( const std::shared_ptr<ExtendedScene> & scene, donut::engine::ThreadPool * threadPool );
    // Ends the frame window. With validation enabled, verifies every filled entry still matches a fresh derivation.
    void            Close();

    bool            IsOpen() const      { return m_open; }
    // The frame worker pool this cache was opened with - consumers parallelising their own per-instance work should use
    // the same dedicated pool rather than introducing another (see the warning on Open).
    donut::engine::ThreadPool & GetThreadPool() const   { assert( m_threadPool != nullptr ); return *m_threadPool; }

    // Raw SceneGraphNode pointers, parallel to and index-matched with SceneGraph::GetMeshInstances().
    //
    // Exists because MeshInstance::GetNode() is 'm_Node.lock().get()' - a weak_ptr lock, i.e. an atomic CAS loop plus a
    // decrement, per instance per frame. Profiling BuildTLAS in an instance-heavy scene put that single call at ~13% of
    // total CPU. The pointers are borrowed, not owned: the scene graph owns the nodes and outlives the frame window.
    const std::vector<donut::engine::SceneGraphNode*> & GetMeshInstanceNodes();

    // Per-instance data, index-matched with SceneGraph::GetMeshInstances().
    struct InstanceInfo
    {
        donut::engine::MeshInfo *   Mesh                        = nullptr;
        // Result of a dynamic_cast. MeshInstanceExtension is a SEPARATE base of MeshInstanceEx/SkinnedMeshInstanceEx, so
        // MeshInstance* -> MeshInstanceExtension* is a cross-cast that static_cast cannot express. Doing it once here
        // keeps it out of LightsBaker::ProcessEmissiveGeometry's serial loop.
        //
        // ONLY RESOLVED FOR LIGHT-RELEVANT INSTANCES - nullptr on every other instance, by design. The cast is the single
        // most expensive thing in the fill (MSVC's __RTDynamicCast walks the RTTI hierarchy descriptors), the only
        // consumer is ProcessEmissiveGeometry, and that only ever visits GetLightRelevantSubInstances(). An instance
        // qualifies iff at least one of its sub-instances appears in that list. New consumers wanting this for an
        // arbitrary instance must do their own cast - do NOT lift it back out of the branch.
        // With validation enabled the fill separately checks that EVERY instance would have cast successfully, so a
        // wrong type still asserts immediately rather than on the frame its material turns emissive.
        MeshInstanceExtension *     InstanceEx                  = nullptr;
        uint32_t                    FirstGeometryInstanceIndex  = 0;
        uint32_t                    GeometryCount               = 0;
    };

    // Per-sub-instance data (one entry per instance*geometry), indexed by FirstGeometryInstanceIndex + geometryIndex -
    // the same indexing subInstanceData uses. Flattens the instance->mesh->geometry->material pointer chase that both
    // MaterialsBaker::Update and LightsBaker::ProcessEmissiveGeometry would otherwise walk independently.
    struct SubInstanceInfo
    {
        donut::engine::MeshGeometry * Geometry   = nullptr;
        PTMaterial *                  MaterialPT = nullptr;
    };

    const std::vector<InstanceInfo> &    GetInstanceInfos();
    const std::vector<SubInstanceInfo> & GetSubInstanceInfos();

    // Sub-instances that LightsBaker::ProcessEmissiveGeometry must actually visit. Emissive geometry is very sparse, and
    // that loop cannot be parallelised (it carries a running light-index allocator), so shrinking what it walks is the
    // only lever available. Qualifies iff:
    //     EnableAsAnalyticLightProxy || (IsEmissiveSurfaceLightSampling() && !SkipRender)
    // which are exactly the gates inside that loop - all plain material state, so this cannot go stale mid-frame.
    //
    // ORDER IS LOAD-BEARING: entries are strictly ascending by SubInstanceIndex, i.e. exactly the order a serial scan
    // would produce. LightsBaker allocates light indices in iteration order, so any reordering changes output.
    struct LightRelevantSubInstance
    {
        uint32_t InstanceArrayIndex = 0;    // index into GetInstanceInfos() / SceneGraph::GetMeshInstances()
        uint32_t GeometryIndex      = 0;    // index within that instance's mesh
        uint32_t SubInstanceIndex   = 0;    // == InstanceInfo::FirstGeometryInstanceIndex + GeometryIndex
    };

    const std::vector<LightRelevantSubInstance> & GetLightRelevantSubInstances();

private:
    void            FillMeshInstanceNodes();
    void            FillInstanceInfos();
    // Marks every entry stale without discarding the storage or its contents - see the comment on the definition.
    void            Invalidate();

#if RTXPT_SCENE_RENDER_CACHE_VALIDATION
    void            ValidateMeshInstanceNodes() const;
    void            ValidateLightRelevantSubInstances() const;
#endif

    // Raw on purpose: the cache never outlives the frame, and Sample owns both it and the scene reference.
    ExtendedScene * m_scene                 = nullptr;
    donut::engine::ThreadPool * m_threadPool = nullptr;
    bool            m_open                  = false;

    std::vector<donut::engine::SceneGraphNode*> m_meshInstanceNodes;
    std::vector<InstanceInfo>    m_instanceInfos;
    std::vector<SubInstanceInfo> m_subInstanceInfos;
    std::vector<LightRelevantSubInstance> m_lightRelevantSubInstances;
    // per-chunk scratch for the parallel compaction; retained across frames so capacity is reused
    std::vector<std::vector<LightRelevantSubInstance>> m_lightRelevantPerChunk;
    bool            m_instanceInfosFilled     = false;
    bool            m_meshInstanceNodesFilled = false;
};
