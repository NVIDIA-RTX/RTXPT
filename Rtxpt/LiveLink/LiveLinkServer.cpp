/*
* Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#if defined(_WIN32)
// Winsock2 must be included before <windows.h> ends up pulled in by anything else in
// this translation unit (including donut headers), so keep these as the very first
// includes of the file.
#define WIN32_LEAN_AND_MEAN
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "Ws2_32.lib")
#endif

#include "LiveLinkServer.h"

#include <donut/core/log.h>

#include <algorithm>
#include <sstream>

using namespace rtxpt::livelink;

#if defined(_WIN32)

namespace
{
    // RAII wrapper so WSAStartup/WSACleanup stay balanced even on early-return error paths.
    struct WinsockGuard
    {
        bool Ok = false;
        WinsockGuard()
        {
            WSADATA wsaData;
            Ok = (WSAStartup(MAKEWORD(2, 2), &wsaData) == 0);
        }
        ~WinsockGuard()
        {
            if (Ok)
                WSACleanup();
        }
    };

    bool ParseFloats(std::istringstream& iss, float* values, int count)
    {
        for (int i = 0; i < count; i++)
        {
            if (!(iss >> values[i]))
                return false;
        }
        return true;
    }
}

LiveLinkServer::LiveLinkServer() = default;

LiveLinkServer::~LiveLinkServer()
{
    Stop();
}

bool LiveLinkServer::Start(uint16_t port)
{
    if (m_running.load())
        return true;

    m_port = port;
    m_stopRequested.store(false);
    m_running.store(true);
    m_thread = std::thread(&LiveLinkServer::ThreadMain, this);
    return true;
}

void LiveLinkServer::Stop()
{
    if (!m_thread.joinable())
        return;

    m_stopRequested.store(true);
    m_thread.join(); // ThreadMain wakes up at most every 100ms to check m_stopRequested
    m_running.store(false);
    m_clientConnected.store(false);
}

std::vector<Command> LiveLinkServer::PopCommands()
{
    std::vector<Command> result;
    std::lock_guard<std::mutex> lock(m_queueMutex);
    result.reserve(m_queue.size());
    for (auto& cmd : m_queue)
        result.push_back(std::move(cmd));
    m_queue.clear();
    return result;
}

void LiveLinkServer::SendLineToClient(const std::string& line)
{
    if (m_clientSocket == -1)
        return;
    std::string withNewline = line + "\n";
    send((SOCKET)m_clientSocket, withNewline.c_str(), (int)withNewline.size(), 0);
}

void LiveLinkServer::HandleLine(const std::string& lineIn)
{
    std::string line = lineIn;
    while (!line.empty() && (line.back() == '\r' || line.back() == '\n'))
        line.pop_back();
    if (line.empty())
        return;

    std::istringstream iss(line);
    std::string verb;
    iss >> verb;

    if (verb == "HELLO")
    {
        Command cmd;
        cmd.type = CommandType::Hello;
        std::getline(iss, cmd.helloInfo);
        if (!cmd.helloInfo.empty() && cmd.helloInfo.front() == ' ')
            cmd.helloInfo.erase(0, 1);
        {
            std::lock_guard<std::mutex> lock(m_queueMutex);
            m_queue.push_back(std::move(cmd));
        }
        SendLineToClient("HELLO_OK 1 RTXPT");
    }
    else if (verb == "CAM")
    {
        float v[11];
        if (!ParseFloats(iss, v, 11))
        {
            SendLineToClient("ERR malformed CAM command, expected 11 floats");
            return;
        }
        Command cmd;
        cmd.type = CommandType::Camera;
        cmd.cameraPosition           = dm::float3(v[0], v[1], v[2]);
        cmd.cameraDirection          = dm::float3(v[3], v[4], v[5]);
        cmd.cameraUp                 = dm::float3(v[6], v[7], v[8]);
        cmd.cameraVerticalFovRadians = v[9];
        cmd.cameraZNear              = v[10];
        {
            std::lock_guard<std::mutex> lock(m_queueMutex);
            m_queue.push_back(std::move(cmd));
        }
        // Intentionally no ack sent here: the camera stream can run at 30-60Hz and an
        // ack round-trip per message isn't useful to the client.
    }
    else if (verb == "RELOAD")
    {
        Command cmd;
        cmd.type = CommandType::Reload;
        std::getline(iss, cmd.reloadScenePath);
        if (!cmd.reloadScenePath.empty() && cmd.reloadScenePath.front() == ' ')
            cmd.reloadScenePath.erase(0, 1);
        if (cmd.reloadScenePath.empty())
        {
            SendLineToClient("ERR RELOAD requires a scene path");
            return;
        }
        {
            std::lock_guard<std::mutex> lock(m_queueMutex);
            m_queue.push_back(std::move(cmd));
        }
        SendLineToClient("OK");
    }
    else if (verb == "PING")
    {
        SendLineToClient("PONG");
    }
    else
    {
        SendLineToClient("ERR unknown command");
    }
}

void LiveLinkServer::ThreadMain()
{
    WinsockGuard wsaGuard;
    if (!wsaGuard.Ok)
    {
        donut::log::error("LiveLink: WSAStartup failed");
        return;
    }

    SOCKET listenSocket = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (listenSocket == INVALID_SOCKET)
    {
        donut::log::error("LiveLink: failed to create listen socket");
        return;
    }

    BOOL reuse = TRUE;
    setsockopt(listenSocket, SOL_SOCKET, SO_REUSEADDR, (const char*)&reuse, sizeof(reuse));

    sockaddr_in addr = {};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(m_port);
    // Loopback-only by design: Live Link is meant to talk to a Blender instance on the
    // same machine. If you need to drive RTXPT from another machine on the LAN, change
    // this to INADDR_ANY and make sure the port is allowed through your firewall.
    inet_pton(AF_INET, "127.0.0.1", &addr.sin_addr);

    if (bind(listenSocket, (sockaddr*)&addr, sizeof(addr)) == SOCKET_ERROR)
    {
        donut::log::error("LiveLink: failed to bind 127.0.0.1:%d (port already in use?)", (int)m_port);
        closesocket(listenSocket);
        return;
    }

    if (listen(listenSocket, 1) == SOCKET_ERROR)
    {
        donut::log::error("LiveLink: listen() failed");
        closesocket(listenSocket);
        return;
    }

    m_listenSocket = (intptr_t)listenSocket;
    donut::log::info("LiveLink: listening on 127.0.0.1:%d for the Blender Live Link add-on", (int)m_port);

    std::string recvBuffer;
    char rawBuffer[4096];

    while (!m_stopRequested.load())
    {
        SOCKET clientSocket = (SOCKET)m_clientSocket;
        bool haveClient = m_clientConnected.load() && clientSocket != INVALID_SOCKET;

        fd_set readSet;
        FD_ZERO(&readSet);
        FD_SET(haveClient ? clientSocket : listenSocket, &readSet);

        timeval timeout = {};
        timeout.tv_sec = 0;
        timeout.tv_usec = 100 * 1000; // 100ms poll, keeps Stop() responsive without busy-looping

        int selectResult = select(0 /* ignored on Windows */, &readSet, nullptr, nullptr, &timeout);
        if (m_stopRequested.load())
            break;
        if (selectResult <= 0)
            continue;

        if (!haveClient)
        {
            sockaddr_in clientAddr = {};
            int clientAddrLen = sizeof(clientAddr);
            SOCKET newClient = accept(listenSocket, (sockaddr*)&clientAddr, &clientAddrLen);
            if (newClient != INVALID_SOCKET)
            {
                m_clientSocket = (intptr_t)newClient;
                m_clientConnected.store(true);
                recvBuffer.clear();
                donut::log::info("LiveLink: Blender client connected");
            }
        }
        else
        {
            int received = recv(clientSocket, rawBuffer, sizeof(rawBuffer), 0);
            if (received <= 0)
            {
                closesocket(clientSocket);
                m_clientSocket = -1;
                m_clientConnected.store(false);
                donut::log::info("LiveLink: Blender client disconnected");
                continue;
            }

            recvBuffer.append(rawBuffer, received);

            size_t newlinePos;
            while ((newlinePos = recvBuffer.find('\n')) != std::string::npos)
            {
                std::string line = recvBuffer.substr(0, newlinePos);
                recvBuffer.erase(0, newlinePos + 1);
                HandleLine(line);
            }
            // Safety valve: don't let a malformed/never-terminated line grow unbounded.
            if (recvBuffer.size() > (1u << 20))
                recvBuffer.clear();
        }
    }

    if (m_clientSocket != -1)
    {
        closesocket((SOCKET)m_clientSocket);
        m_clientSocket = -1;
    }
    closesocket(listenSocket);
    m_listenSocket = -1;
    m_clientConnected.store(false);
    donut::log::info("LiveLink: server stopped");
}

#else // !_WIN32

LiveLinkServer::LiveLinkServer() = default;
LiveLinkServer::~LiveLinkServer() = default;

bool LiveLinkServer::Start(uint16_t /*port*/)
{
    donut::log::warning("LiveLink: not supported on this platform yet (Windows only)");
    return false;
}

void LiveLinkServer::Stop() {}
std::vector<Command> LiveLinkServer::PopCommands() { return {}; }
void LiveLinkServer::SendLineToClient(const std::string&) {}
void LiveLinkServer::HandleLine(const std::string&) {}
void LiveLinkServer::ThreadMain() {}

#endif
