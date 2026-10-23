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

// NOTE: deliberately NOT included from SampleCommon.h - donut/engine/ThreadPool.h has a file-scope
// 'using namespace std::chrono', and SampleCommon.h is pulled into ~26 translation units.

#include <donut/engine/ThreadPool.h>

#include <algorithm>
#include <cstdint>
#include <thread>

// Splits [0, count) into one chunk per hardware thread and invokes func(begin, end) on each.
//
// Intended for uniform, index-independent work over large arrays - static partitioning, so there is nothing to steal.
// Chunks are coarse on purpose: ThreadPool::AddTask allocates a shared_ptr and takes the queue lock, so per-item tasks
// would cost far more than the work itself.
//
// The calling thread runs the first chunk itself and only then waits, because ThreadPool::WaitForTasks() is a yield-spin
// - otherwise the caller would burn a core doing nothing while the workers run.
//
// IMPORTANT: threadPool must be dedicated to this kind of short, frame-local work. WaitForTasks() waits on a POOL-WIDE
// pending counter, so any unrelated long-running task queued on the same pool would stall this call until it finishes.
//
// func must be safe to invoke concurrently for disjoint ranges. It is passed by reference to the queued tasks, which is
// safe because we always WaitForTasks() before returning.
// Number of chunks ParallelForChunked* will use. Exposed so callers can size per-chunk output arrays before the call.
inline uint32_t ParallelForChunkCount()
{
    // hardware_concurrency() asks the OS every call (~0.6 us on Windows) and the answer cannot change.
    static const uint32_t s_chunkCount = std::max( 1u, std::thread::hardware_concurrency() );
    return s_chunkCount;
}

namespace ParallelForDetail
{
    // Enqueues chunk 'chunk', whose task first enqueues its children 2*chunk+1 and 2*chunk+2 - a binary tree rooted at the
    // caller's chunk 0 - so the caller pays for two AddTask calls instead of chunkCount-1 (each is a make_shared, a lock
    // and a notify). Safe with WaitForTasks(): children are enqueued before their parent returns, and the pool decrements
    // its pending count only after a task returns, so the count cannot reach zero while any descendant is outstanding.
    template< typename FuncT >
    inline void SpawnChunk( donut::engine::ThreadPool & threadPool, uint32_t chunk, uint32_t chunkCount, size_t chunkSize, size_t count, FuncT & func )
    {
        if( chunk >= chunkCount || (size_t)chunk * chunkSize >= count )
            return;     // ranges only grow with the index, so an empty chunk's descendants are empty too
        threadPool.AddTask( [&threadPool, &func, chunk, chunkCount, chunkSize, count]()
        {
            SpawnChunk( threadPool, 2 * chunk + 1, chunkCount, chunkSize, count, func );
            SpawnChunk( threadPool, 2 * chunk + 2, chunkCount, chunkSize, count, func );
            const size_t begin = (size_t)chunk * chunkSize;
            func( begin, std::min( count, begin + chunkSize ), chunk );
        } );
    }
}

// As ParallelForChunked, but func also receives its chunk index - use when each chunk needs its own output slot (e.g.
// building a per-chunk list that is concatenated afterwards). Chunks are contiguous and processed in index order, so
// concatenating their outputs in chunk order reproduces a serial traversal exactly.
template< typename FuncT >
inline void ParallelForChunkedIdx( donut::engine::ThreadPool & threadPool, size_t count, FuncT && func )
{
    const uint32_t chunkCount = ParallelForChunkCount();
    const size_t   chunkSize  = ( count + chunkCount - 1 ) / chunkCount;

    // chunks 1..chunkCount-1, spawned as a tree - see SpawnChunk
    ParallelForDetail::SpawnChunk( threadPool, 1, chunkCount, chunkSize, count, func );
    ParallelForDetail::SpawnChunk( threadPool, 2, chunkCount, chunkSize, count, func );

    // first chunk on the calling thread; also covers count==0 (chunkSize==0 -> empty range)
    func( 0, std::min( count, chunkSize ), 0u );

    threadPool.WaitForTasks();
}

template< typename FuncT >
inline void ParallelForChunked( donut::engine::ThreadPool & threadPool, size_t count, FuncT && func )
{
    ParallelForChunkedIdx( threadPool, count, [&func]( size_t begin, size_t end, uint32_t ) { func( begin, end ); } );
}
