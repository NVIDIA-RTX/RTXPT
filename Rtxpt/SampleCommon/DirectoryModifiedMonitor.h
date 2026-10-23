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

#include <cstdint>
#include <filesystem>
#include <optional>
#include <vector>

// Watches directories for source changes so shader hot-reload does not have to walk them on the render thread.
//
// The walk it replaces (a recursive_directory_iterator over ~127 files) measures ~1.4 ms. Both pipeline bakers used to
// run it inline from their Update(), each on its own 0.5 s timer, over the SAME shader root - so it landed as two
// unrelated ~1.4 ms render-thread hitches per second. A sampling profiler averages that into nothing, which is exactly
// why it went unnoticed; it only shows up as per-frame outliers in Tracy.
//
// This moves the walk to a background thread and folds the duplicates: directories are canonicalised on registration,
// so two groups naming the same root are scanned once between them.
//
// The polling period is an implementation detail on purpose. Callers only ever ask "what is the newest write time in my
// directories", so switching to ReadDirectoryChangesW later is an internal change with no caller edits.
class DirectoryModifiedMonitor
{
public:
    // Opaque registration token. Groups are never unregistered - they live for the process, like the bakers that own them.
    using Handle = uint32_t;
    static constexpr Handle kInvalidHandle = ~0u;

    // Explicit lifetime rather than a file-scope global: a global would join its worker during static destruction, which
    // on Windows can run under the loader lock. Call these from Sample, around the bakers that use them.
    static void     Init();
    static void     Shutdown();     // safe to call without a matching Init, and safe to call twice

    // Registers a group of directories; queries return the newest write time across all of them.
    // Performs one synchronous scan so the group is immediately queryable - that cost lands at startup, not in a frame.
    // Missing or inaccessible directories are not an error; they simply contribute nothing.
    static Handle   Register( const std::vector<std::filesystem::path> & directories );

    // Newest write time across the group as of the last completed scan, so up to one poll interval stale BY DESIGN.
    // nullopt means no file was found in any of the group's directories (or the handle is invalid).
    // Cheap: a mutex and a fold over a handful of cached values. Safe to call every frame.
    static std::optional<std::filesystem::file_time_type> Query( Handle handle );

    // Synchronous fresh walk, bypassing the cache, also refreshing it.
    //
    // Use this where the timestamp gates actual compilation (see ShaderCompilerUtils::IsCompiledShaderUpToDate, which
    // does 'binaryTime >= lastSourceModified' per shader) rather than merely detecting change. A stale value there can
    // only ever SKIP a compile, and it is not worth reasoning about the window: these sites run only when a reload is
    // already happening, where 1.4 ms disappears next to seconds of shader compilation.
    static std::optional<std::filesystem::file_time_type> ScanNow( Handle handle );

private:
    struct Impl;
    static Impl *   s_impl;         // nullptr unless Init() has been called; all entry points tolerate that
};
