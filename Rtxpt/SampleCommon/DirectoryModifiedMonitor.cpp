/*
* Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "DirectoryModifiedMonitor.h"
#include "SampleCommon.h"

#include <donut/core/log.h>

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <thread>

namespace fs = std::filesystem;

namespace
{
    // How often the worker re-walks every registered directory.
    constexpr auto c_pollInterval = std::chrono::milliseconds(500);
}

struct DirectoryModifiedMonitor::Impl
{
    // One per UNIQUE directory. Append-only, so indices held by groups stay valid and the worker can scan without
    // holding the lock across the walk.
    struct Directory
    {
        fs::path                            Path;
        std::optional<fs::file_time_type>   LatestWrite;
    };

    struct Group
    {
        std::vector<uint32_t>               DirectoryIndices;
    };

    std::mutex                  Mutex;
    std::vector<Directory>      Directories;
    std::vector<Group>          Groups;

    std::thread                 Worker;
    std::condition_variable     WakeUp;
    bool                        Stop = false;

    // Walks one directory. Must be called WITHOUT Mutex held - this is the ~1.4 ms part.
    static std::optional<fs::file_time_type> ScanDirectory( const fs::path & path )
    {
        return GetLatestModifiedTimeDirectoryRecursive( path );
    }

    // Scans directories [0, count) and writes the results back. 'count' is sampled by the caller under the lock so a
    // concurrent Register() that appends is simply picked up on the next pass.
    void ScanRange( size_t count )
    {
        for( size_t i = 0; i < count; i++ )
        {
            fs::path path;
            {
                std::lock_guard<std::mutex> lock( Mutex );
                if( i >= Directories.size() )
                    return;
                path = Directories[i].Path;     // copy: the walk below runs unlocked
            }

            std::optional<fs::file_time_type> latest = ScanDirectory( path );

            {
                std::lock_guard<std::mutex> lock( Mutex );
                if( i < Directories.size() )
                    Directories[i].LatestWrite = latest;
            }
        }
    }

    void WorkerProc()
    {
        for(;;)
        {
            {
                std::unique_lock<std::mutex> lock( Mutex );
                // wait_for on the stop flag rather than sleep_for, so Shutdown() does not have to wait out a full
                // interval before the process can exit.
                WakeUp.wait_for( lock, c_pollInterval, [this]{ return Stop; } );
                if( Stop )
                    return;
            }

            size_t count;
            {
                std::lock_guard<std::mutex> lock( Mutex );
                count = Directories.size();
            }
            ScanRange( count );
        }
    }

    // Folds the group's cached values. Caller must hold Mutex.
    std::optional<fs::file_time_type> FoldLocked( Handle handle ) const
    {
        if( handle >= Groups.size() )
            return std::nullopt;

        std::optional<fs::file_time_type> latest;
        for( uint32_t index : Groups[handle].DirectoryIndices )
        {
            const std::optional<fs::file_time_type> & entry = Directories[index].LatestWrite;
            if( entry.has_value() && ( !latest.has_value() || *entry > *latest ) )
                latest = entry;
        }
        return latest;
    }
};

DirectoryModifiedMonitor::Impl * DirectoryModifiedMonitor::s_impl = nullptr;

void DirectoryModifiedMonitor::Init()
{
    assert( s_impl == nullptr && "DirectoryModifiedMonitor::Init called twice" );
    if( s_impl != nullptr )
        return;

    s_impl = new Impl();
    s_impl->Worker = std::thread( [impl = s_impl]{ impl->WorkerProc(); } );
}

void DirectoryModifiedMonitor::Shutdown()
{
    // Tolerated on purpose so exit paths that never got as far as Init() can call this unconditionally.
    if( s_impl == nullptr )
        return;

    {
        std::lock_guard<std::mutex> lock( s_impl->Mutex );
        s_impl->Stop = true;
    }
    s_impl->WakeUp.notify_all();

    if( s_impl->Worker.joinable() )
        s_impl->Worker.join();      // bounded by one in-flight directory walk, not by the poll interval

    delete s_impl;
    s_impl = nullptr;
}

DirectoryModifiedMonitor::Handle DirectoryModifiedMonitor::Register( const std::vector<fs::path> & directories )
{
    if( s_impl == nullptr )
    {
        assert( false && "DirectoryModifiedMonitor::Register before Init" );
        return kInvalidHandle;
    }

    Impl::Group group;
    std::vector<size_t> freshIndices;

    for( const fs::path & directory : directories )
    {
        // Canonicalise so two groups naming the same root share one scan. weakly_canonical does not require the path to
        // exist, which matters because a monitored directory legitimately may not (the additional paths are built from a
        // relative path next to the executable).
        std::error_code ec;
        fs::path canonical = fs::weakly_canonical( directory, ec );
        if( ec )
            canonical = fs::absolute( directory, ec );
        if( ec )
            canonical = directory;

        std::lock_guard<std::mutex> lock( s_impl->Mutex );

        uint32_t index = kInvalidHandle;
        for( uint32_t i = 0; i < (uint32_t)s_impl->Directories.size(); i++ )
        {
            if( s_impl->Directories[i].Path == canonical )
                { index = i; break; }
        }

        if( index == kInvalidHandle )
        {
            index = (uint32_t)s_impl->Directories.size();
            s_impl->Directories.push_back( { canonical, std::nullopt } );
            freshIndices.push_back( index );
        }

        // Duplicates within one group would just make the fold do redundant work.
        if( std::find( group.DirectoryIndices.begin(), group.DirectoryIndices.end(), index ) == group.DirectoryIndices.end() )
            group.DirectoryIndices.push_back( index );
    }

    Handle handle;
    {
        std::lock_guard<std::mutex> lock( s_impl->Mutex );
        handle = (Handle)s_impl->Groups.size();
        s_impl->Groups.push_back( std::move( group ) );
    }

    // Prime only the directories this call actually introduced, so the group is queryable on return and no caller needs
    // a "no data yet" branch. Registration happens once at startup, so this is a startup cost.
    for( size_t index : freshIndices )
    {
        fs::path path;
        {
            std::lock_guard<std::mutex> lock( s_impl->Mutex );
            path = s_impl->Directories[index].Path;
        }
        std::optional<fs::file_time_type> latest = Impl::ScanDirectory( path );
        {
            std::lock_guard<std::mutex> lock( s_impl->Mutex );
            s_impl->Directories[index].LatestWrite = latest;
        }
    }

    return handle;
}

std::optional<fs::file_time_type> DirectoryModifiedMonitor::Query( Handle handle )
{
    if( s_impl == nullptr || handle == kInvalidHandle )
        return std::nullopt;

    std::lock_guard<std::mutex> lock( s_impl->Mutex );
    return s_impl->FoldLocked( handle );
}

std::optional<fs::file_time_type> DirectoryModifiedMonitor::ScanNow( Handle handle )
{
    if( s_impl == nullptr || handle == kInvalidHandle )
        return std::nullopt;

    // Snapshot the group's directory list, walk them unlocked, then publish - same discipline as the worker, so a
    // ScanNow racing the worker just means one of them wins per directory, and both wrote a freshly observed value.
    std::vector<uint32_t> indices;
    {
        std::lock_guard<std::mutex> lock( s_impl->Mutex );
        if( handle >= s_impl->Groups.size() )
            return std::nullopt;
        indices = s_impl->Groups[handle].DirectoryIndices;
    }

    for( uint32_t index : indices )
    {
        fs::path path;
        {
            std::lock_guard<std::mutex> lock( s_impl->Mutex );
            path = s_impl->Directories[index].Path;
        }
        std::optional<fs::file_time_type> latest = Impl::ScanDirectory( path );
        {
            std::lock_guard<std::mutex> lock( s_impl->Mutex );
            s_impl->Directories[index].LatestWrite = latest;
        }
    }

    std::lock_guard<std::mutex> lock( s_impl->Mutex );
    return s_impl->FoldLocked( handle );
}
