/*
* Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "SceneRenderCache.h"
#include "SampleCommon/SampleCommon.h"

#include "SampleCommon/ExtendedScene.h"
#include "Materials/MaterialsBaker.h"

#include <donut/engine/SceneGraph.h>
#include <donut/engine/ThreadPool.h>

#include "SampleCommon/ParallelFor.h"

#include <cassert>
#include <algorithm>
#include <thread>

// The MeshInstance prefetch in FillInstanceInfos is x86/x64 only; ARM64EC also defines _M_X64, hence the explicit exclusion.
#if ( defined(_M_X64) || defined(_M_IX86) || defined(__x86_64__) || defined(__i386__) ) && !defined(_M_ARM64EC)
#define RTXPT_MESH_INSTANCE_PREFETCH 1
#include <xmmintrin.h>
#else
#define RTXPT_MESH_INSTANCE_PREFETCH 0
#endif

using namespace donut::engine;

SceneRenderCache::~SceneRenderCache()
{
    // Reading the cache outside the Open()/Close() window is the bug this is guarding against; if we are still open at
    // destruction then somebody's exit path skipped Close().
    assert( !m_open && "SceneRenderCache destroyed while open - a Close() was missed" );
}

void SceneRenderCache::Open( const std::shared_ptr<ExtendedScene> & scene, donut::engine::ThreadPool * threadPool )
{
    ZoneScoped;

    assert( !m_open && "SceneRenderCache::Open called twice without a Close" );

    m_scene      = scene.get();
    m_threadPool = threadPool;
    assert( m_threadPool != nullptr );
    m_open  = true;

    // Nothing is derived here on purpose - see the note in the header on lazy filling vs Scene::Refresh ordering.
    Invalidate();
}

// Marks every entry as needing a refill.
//
// The per-instance and per-sub-instance arrays deliberately keep their contents, not just their capacity. Every element
// they hand out is rewritten by the fill - sub-instance slots tile [0, GetGeometryInstancesCount()) exactly, so there is
// no gap - which makes clearing them strictly a cost: the following resize() then has to value-initialise the lot, over
// a megabyte on an instance-heavy scene, on the calling thread, immediately overwritten. Worse than a memset, in fact:
// the default member initialisers on InstanceInfo/SubInstanceInfo make them non-trivially-default-constructible, so the
// STL cannot take its memset path and emits a per-element constructor loop. Leaving the contents in place makes the
// resize() a no-op in steady state.
//
// The '...Filled' flags are what gates access, and a read through a getter after Close() dereferences a null m_scene
// immediately, so nothing here weakens that path. What it does weaken is a read through a reference RETAINED past
// Close(): that used to see an empty vector, and now sees last frame's pointers. Validation builds still clear, so that
// mistake degrades loudly where it is cheap to do so.
void SceneRenderCache::Invalidate()
{
    m_meshInstanceNodesFilled = false;
    m_instanceInfosFilled     = false;
    m_lightRelevantSubInstances.clear();    // rebuilt by concatenation, so it has to start empty regardless - and it is O(1) here

#if RTXPT_SCENE_RENDER_CACHE_VALIDATION
    m_meshInstanceNodes.clear();            // clear() keeps capacity, so steady-state frames do not reallocate
    m_instanceInfos.clear();
    m_subInstanceInfos.clear();
#endif
}

void SceneRenderCache::Close()
{
    ZoneScoped;

    // Tolerated on purpose: Open() happens partway into the frame (right after Scene::Refresh), so any earlier exit path
    // reaches PostRender without ever having opened. Nothing to do in that case.
    if( !m_open )
        return;

#if RTXPT_SCENE_RENDER_CACHE_VALIDATION
    if( m_meshInstanceNodesFilled )
        ValidateMeshInstanceNodes();
    if( m_instanceInfosFilled )
        ValidateLightRelevantSubInstances();
#endif

    // Invalidate at the frame boundary: past this point the scene graph is free to restructure, and anything we hold
    // becomes a dangling read waiting to happen. Storage - and, outside validation builds, its stale contents - is
    // retained for the next frame; see Invalidate() for why that is deliberate and what it costs.
    Invalidate();

    m_scene      = nullptr;
    m_threadPool = nullptr;
    m_open  = false;
}

const std::vector<SceneGraphNode*> & SceneRenderCache::GetMeshInstanceNodes()
{
    assert( m_open && "SceneRenderCache accessed outside the Open()/Close() frame window" );

    if( !m_meshInstanceNodesFilled )
        FillMeshInstanceNodes();

    return m_meshInstanceNodes;
}

void SceneRenderCache::FillMeshInstanceNodes()
{
    ZoneScoped;
    assert( m_scene != nullptr && m_threadPool != nullptr );

    const auto & meshInstances = m_scene->GetSceneGraph()->GetMeshInstances();
    const size_t count = meshInstances.size();

    // resize() rather than reserve()+push_back(): chunks then write disjoint indices and need no synchronisation at all.
    // Every slot in [0, count) is written below, so as with the arrays in FillInstanceInfos this is a no-op in steady
    // state - Invalidate() leaves the contents alone precisely so it can be.
    m_meshInstanceNodes.resize( count );

    // This is the one place we pay for MeshInstance::GetNode()'s weak_ptr lock - once per instance per frame, shared by
    // every consumer instead of once per instance per consumer.
    auto fillRange = [ this, &meshInstances ]( size_t begin, size_t end )
    {
        for( size_t i = begin; i < end; i++ )
        {
            // Concurrent weak_ptr::lock() is safe: control block refcounting is atomic, even if two instances happen to
            // share a node. Nodes are non-null here - a MeshInstance only appears in GetMeshInstances() while attached.
            donut::engine::SceneGraphNode * node = meshInstances[i]->GetNode();
            assert( node != nullptr && "mesh instance in the scene graph with no node" );
            m_meshInstanceNodes[i] = node;
        }
    };

    // Static partitioning via the shared helper - see ParallelFor.h for why chunks are coarse and why the calling
    // thread takes one.
    ParallelForChunked( *m_threadPool, count, fillRange );

    m_meshInstanceNodesFilled = true;
}

const std::vector<SceneRenderCache::InstanceInfo> & SceneRenderCache::GetInstanceInfos()
{
    assert( m_open && "SceneRenderCache accessed outside the Open()/Close() frame window" );
    if( !m_instanceInfosFilled )
        FillInstanceInfos();
    return m_instanceInfos;
}

const std::vector<SceneRenderCache::SubInstanceInfo> & SceneRenderCache::GetSubInstanceInfos()
{
    assert( m_open && "SceneRenderCache accessed outside the Open()/Close() frame window" );
    if( !m_instanceInfosFilled )
        FillInstanceInfos();
    return m_subInstanceInfos;
}

const std::vector<SceneRenderCache::LightRelevantSubInstance> & SceneRenderCache::GetLightRelevantSubInstances()
{
    assert( m_open && "SceneRenderCache accessed outside the Open()/Close() frame window" );
    if( !m_instanceInfosFilled )
        FillInstanceInfos();
    return m_lightRelevantSubInstances;
}

void SceneRenderCache::FillInstanceInfos()
{
    ZoneScoped;
    assert( m_scene != nullptr && m_threadPool != nullptr );

    const SceneGraph & sceneGraph = *m_scene->GetSceneGraph();
    const auto & meshInstances = sceneGraph.GetMeshInstances();
    const size_t count = meshInstances.size();

    // SceneGraph maintains m_GeometryInstanceIndex as a running prefix sum over mesh->geometries.size(), and
    // GetGeometryInstancesCount() is its total - so sub-instance slots are contiguous and we can size the array up front
    // without walking anything.
    const size_t subInstanceCount = sceneGraph.GetGeometryInstancesCount();

    // Both arrays are fully rewritten below - instance slots are [0, count), and sub-instance slots tile
    // [0, subInstanceCount) exactly because SceneGraph hands out m_GeometryInstanceIndex as a running prefix sum - so
    // whatever Invalidate() left behind is overwritten, not read. In steady state these two calls are no-ops; only a
    // grown scene pays to value-initialise the new tail.
    m_instanceInfos.resize( count );
    m_subInstanceInfos.resize( subInstanceCount );

    m_lightRelevantPerChunk.resize( ParallelForChunkCount() );
    for( auto & c : m_lightRelevantPerChunk ) c.clear();   // clear() keeps capacity - no per-frame reallocation

#if RTXPT_MESH_INSTANCE_PREFETCH
    static constexpr size_t c_prefetchAhead = 16;
#endif

    auto fillRange = [ this, &meshInstances, subInstanceCount ]( size_t begin, size_t end, uint32_t chunkIndex )
    {
        for( size_t i = begin; i < end; i++ )
        {
#if RTXPT_MESH_INSTANCE_PREFETCH
            // Each MeshInstance is its own heap object, so its first read below misses in a way the hardware prefetcher
            // cannot predict; prefetching a few ahead overlaps those misses. Offsets only affect speed - a prefetch is a hint.
            if( i + c_prefetchAhead < end )
            {
                const char * m = (const char *)&meshInstances[i + c_prefetchAhead]->GetMesh();
                _mm_prefetch( m - 8,  _MM_HINT_T0 );    // m_InstanceIndex / m_GeometryInstanceIndex, just before m_Mesh
                _mm_prefetch( m + 15, _MM_HINT_T0 );    // end of m_Mesh, in case it straddles a line
            }
#endif

            const auto & instance = meshInstances[i];
            MeshInfo * mesh = instance->GetMesh().get();    // GetMesh() returns a const shared_ptr&, so .get() is free

            InstanceInfo & info = m_instanceInfos[i];
            info.Mesh                       = mesh;
            info.FirstGeometryInstanceIndex = (uint32_t)instance->GetGeometryInstanceIndex();
            info.GeometryCount              = (uint32_t)mesh->geometries.size();
            info.InstanceEx                 = nullptr;   // resolved below, only if this instance turns out to be light-relevant

            // Sub-instance slots for this instance are [First, First+GeometryCount) - disjoint across instances, so
            // chunks never overlap even though they write into a shared array.
            assert( info.FirstGeometryInstanceIndex + info.GeometryCount <= subInstanceCount );
            for( uint32_t g = 0; g < info.GeometryCount; g++ )
            {
                MeshGeometry * geometry = mesh->geometries[g].get();
                const uint32_t subInstanceIndex = info.FirstGeometryInstanceIndex + g;
                SubInstanceInfo & sub = m_subInstanceInfos[ subInstanceIndex ];
                sub.Geometry   = geometry;
                sub.MaterialPT = PTMaterial::SafeCast( geometry->material );
                if ( sub.MaterialPT == nullptr )
                    continue;

                // Phase A of the compaction: append in-order within this chunk. This predicate MUST stay in lockstep
                // with the gates inside LightsBaker::ProcessEmissiveGeometry - a Debug guard there re-derives it over
                // every sub-instance and asserts nothing was wrongly excluded.
                const PTMaterial & mat = *sub.MaterialPT;
                if( mat.EnableAsAnalyticLightProxy || ( mat.IsEmissiveSurfaceLightSampling() && !mat.SkipRender ) )
                {
                    // The only place the MeshInstance* -> MeshInstanceExtension* cross-cast is paid, and only for the
                    // instances ProcessEmissiveGeometry will actually visit. MeshInstanceExtension is a SEPARATE base of
                    // MeshInstanceEx/SkinnedMeshInstanceEx, so this cannot be a static_cast, and MSVC's __RTDynamicCast
                    // walks the RTTI hierarchy descriptors - doing it for every instance cost more than the rest of this
                    // loop put together, to serve a set that is typically a fraction of a percent of it.
                    if( info.InstanceEx == nullptr )
                    {
                        info.InstanceEx = dynamic_cast<MeshInstanceExtension*>( instance.get() );
                        assert( info.InstanceEx != nullptr && "mesh instance is not a MeshInstanceEx/SkinnedMeshInstanceEx" );
                    }
                    m_lightRelevantPerChunk[chunkIndex].push_back( { (uint32_t)i, g, subInstanceIndex } );
                }
            }
        }
    };

    ParallelForChunkedIdx( *m_threadPool, count, fillRange );

    // InstanceEx is only resolved for light-relevant instances above, so a mesh instance of an unexpected type would
    // otherwise stay invisible until the frame its material happens to turn emissive. Check the type assumption over all
    // of them here instead - same guarantee the unconditional cast used to give, at no Release cost.
#if RTXPT_SCENE_RENDER_CACHE_VALIDATION
    for( size_t i = 0; i < count; i++ )
        assert( dynamic_cast<MeshInstanceExtension*>( meshInstances[i].get() ) != nullptr
                && "mesh instance is not a MeshInstanceEx/SkinnedMeshInstanceEx" );
#endif

    // Phase B: concatenate per-chunk lists in chunk order. Chunks are contiguous and ordered, so this reproduces a
    // serial scan exactly - which matters because LightsBaker allocates light indices in iteration order.
    m_lightRelevantSubInstances.clear();
    size_t lightRelevantTotal = 0;
    for( const auto & c : m_lightRelevantPerChunk ) lightRelevantTotal += c.size();
    m_lightRelevantSubInstances.reserve( lightRelevantTotal );
    for( const auto & c : m_lightRelevantPerChunk )
        m_lightRelevantSubInstances.insert( m_lightRelevantSubInstances.end(), c.begin(), c.end() );

    m_instanceInfosFilled = true;
}

#if RTXPT_SCENE_RENDER_CACHE_VALIDATION
void SceneRenderCache::ValidateMeshInstanceNodes() const
{
    assert( m_scene != nullptr );

    const auto & meshInstances = m_scene->GetSceneGraph()->GetMeshInstances();

    // A size change means the instance list itself was rebuilt mid-frame, which invalidates every index we handed out -
    // not just the node pointers.
    assert( meshInstances.size() == m_meshInstanceNodes.size()
            && "mesh instance count changed between SceneRenderCache Open and Close" );

    const size_t count = std::min( meshInstances.size(), m_meshInstanceNodes.size() );
    for( size_t i = 0; i < count; i++ )
    {
        assert( meshInstances[i]->GetNode() == m_meshInstanceNodes[i]
                && "cached scene graph node changed between SceneRenderCache Open and Close" );
    }
}
#endif

#if RTXPT_SCENE_RENDER_CACHE_VALIDATION
void SceneRenderCache::ValidateLightRelevantSubInstances() const
{
    // 1) strictly ascending - LightsBaker allocates light indices in iteration order, so order is part of the contract
    for( size_t i = 1; i < m_lightRelevantSubInstances.size(); i++ )
        assert( m_lightRelevantSubInstances[i-1].SubInstanceIndex < m_lightRelevantSubInstances[i].SubInstanceIndex
                && "light-relevant list is not strictly ascending - parallel compaction lost ordering" );

    // 2) the parallel compaction matches a plain serial re-derivation, entry for entry
    size_t next = 0;
    for( size_t i = 0; i < m_instanceInfos.size(); i++ )
    {
        const InstanceInfo & info = m_instanceInfos[i];
        for( uint32_t g = 0; g < info.GeometryCount; g++ )
        {
            const uint32_t subInstanceIndex = info.FirstGeometryInstanceIndex + g;
            const PTMaterial * mat = m_subInstanceInfos[subInstanceIndex].MaterialPT;
            if( mat == nullptr )
                continue;
            if( !( mat->EnableAsAnalyticLightProxy || ( mat->IsEmissiveSurfaceLightSampling() && !mat->SkipRender ) ) )
                continue;
            assert( next < m_lightRelevantSubInstances.size() && "light-relevant list is missing entries" );
            const LightRelevantSubInstance & e = m_lightRelevantSubInstances[next];
            assert( e.InstanceArrayIndex == (uint32_t)i && e.GeometryIndex == g && e.SubInstanceIndex == subInstanceIndex
                    && "light-relevant list disagrees with a serial re-derivation" );
            next++;
        }
    }
    assert( next == m_lightRelevantSubInstances.size() && "light-relevant list has extra entries" );
}
#endif
