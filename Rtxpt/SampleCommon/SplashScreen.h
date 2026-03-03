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

#include "../Shaders/PathTracer/Config.h"

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <string>
#include <vector>

class SplashScreen
{
public:
    SplashScreen();
    ~SplashScreen();

    SplashScreen(const SplashScreen&) = delete;
    SplashScreen& operator=(const SplashScreen&) = delete;

    // Starts the splash UI thread and shows the PNG.
    // Returns true if the thread started (image load may still fail and show nothing).
    bool Start(const std::wstring& pngName);

    // Asks the splash to close and waits for the thread to exit.
    void Stop();

private:
    struct State;

    static DWORD WINAPI ThreadProc(void* param);
    static LRESULT CALLBACK WndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam);

    static bool LoadPngWic_BGRA32(const wchar_t* path, std::vector<uint8_t>& outBGRA, UINT& outW, UINT& outH);
    static void CenterOnPrimaryMonitorPx(UINT wPx, UINT hPx, int& outX, int& outY);
    static void SetLayeredWindowBitmap(HWND hwnd, const uint8_t* bgraStraightOrPremul, UINT w, UINT h);

private:
    State* m_state;
};


