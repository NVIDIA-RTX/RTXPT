/*
* Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

// RTXPT Blender Live Link
// ------------------------
// A minimal, dependency-free TCP server that lets an external tool - primarily the
// "RTXPT Live Link" Blender add-on (see the rtxpt-exporter repository) - drive the
// free-fly camera live and trigger scene reloads while RTXPT is running.
//
// The wire protocol is intentionally a tiny, newline-delimited ASCII text protocol
// rather than JSON or a binary format, so that both this file and the Blender-side
// Python client can be implemented with zero third-party dependencies. See
// Docs/LiveLink.md for the full protocol description.

#pragma once

#include <donut/core/math/math.h>

#include <atomic>
#include <cstdint>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace rtxpt::livelink
{
    enum class CommandType
    {
        Hello,
        Camera,
        Reload,
    };

    // A single command decoded from the wire. Only the fields relevant to `type` are
    // populated; the rest keep harmless defaults.
    struct Command
    {
        CommandType type = CommandType::Hello;

        // valid when type == Camera; all values already in RTXPT/glTF (Y-up, right-handed)
        // world space - the Blender add-on performs the Z-up -> Y-up conversion.
        dm::float3  cameraPosition           = { 0, 0, 0 };
        dm::float3  cameraDirection          = { 0, 0, -1 };
        dm::float3  cameraUp                 = { 0, 1, 0 };
        float       cameraVerticalFovRadians = 0.0f;   // <= 0 means "leave FOV unchanged"
        float       cameraZNear              = 0.0f;   // <= 0 means "leave zNear unchanged"

        // valid when type == Reload: path of the .scene.json to load, relative to the
        // RTXPT Assets folder (i.e. exactly what --scene / SetCurrentScene() expects).
        std::string reloadScenePath;

        // valid when type == Hello: free-form client info string, logged for diagnostics.
        std::string helloInfo;
    };

    // Default TCP port used by the Blender "RTXPT Live Link" add-on.
    constexpr uint16_t c_DefaultPort = 42042;

    // A tiny, single-client TCP server implementing the RTXPT Live Link protocol.
    //
    // All socket I/O happens on a private background thread; received commands are
    // queued and must be drained once per frame via PopCommands() from the main/render
    // thread, which is the only thread allowed to touch scene/camera state.
    //
    // For safety, the server only ever binds to the loopback interface (127.0.0.1) -
    // Blender and RTXPT are expected to run on the same machine. Currently implemented
    // for Windows (Winsock2) only, matching RTXPT's current platform support; Start()
    // returns false (and logs a warning) on other platforms.
    class LiveLinkServer
    {
    public:
        LiveLinkServer();
        ~LiveLinkServer();

        LiveLinkServer(const LiveLinkServer&) = delete;
        LiveLinkServer& operator=(const LiveLinkServer&) = delete;

        // Starts listening on 127.0.0.1:<port> on a background thread. Safe to call if
        // already running (no-op, returns true). Returns false if the platform isn't
        // supported or the socket could not be created/bound.
        bool Start(uint16_t port = c_DefaultPort);

        // Stops the listener and disconnects any client. Safe to call multiple times /
        // even if never started. Blocks until the background thread has exited.
        void Stop();

        bool     IsRunning() const         { return m_running.load(std::memory_order_acquire); }
        bool     IsClientConnected() const { return m_clientConnected.load(std::memory_order_acquire); }
        uint16_t GetPort() const           { return m_port; }

        // Moves out all commands received since the last call. Call once per frame from
        // the main thread.
        std::vector<Command> PopCommands();

    private:
        void ThreadMain();
        void HandleLine(const std::string& line);
        void SendLineToClient(const std::string& line); // network thread only, best-effort

        std::atomic<bool> m_running{ false };
        std::atomic<bool> m_stopRequested{ false };
        std::atomic<bool> m_clientConnected{ false };
        uint16_t m_port = 0;
        std::thread m_thread;

        std::mutex m_queueMutex;
        std::deque<Command> m_queue;

        // Opaque, platform-specific socket handles (SOCKET on Windows). Owned and only
        // ever touched by the background thread - kept as intptr_t here so this header
        // doesn't have to pull in <winsock2.h>.
        intptr_t m_listenSocket = -1;
        intptr_t m_clientSocket = -1;
    };
}
