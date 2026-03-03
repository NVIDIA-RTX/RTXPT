/*
* Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "SplashScreen.h"
#include "SampleCommon.h"


// SplashScreen.h
#pragma once

#include <wincodec.h>
#include <atomic>

#pragma comment(lib, "windowscodecs.lib")

namespace
{
    constexpr UINT WM_SPLASH_CLOSE = WM_APP + 100;

    struct SplashScreenState
    {
        std::wstring pngPath;

        HANDLE threadHandle = nullptr;
        DWORD threadId = 0;

        HANDLE readyEvent = nullptr;

        std::atomic<HWND> hwnd{ nullptr };

        HINSTANCE hInstance = nullptr;
    };

    bool PremultiplyIfNeeded(std::vector<uint8_t>& bgra)
    {
        // If the image is fully opaque (A==255 everywhere), premultiplication is a no-op.
        // Still safe for general PNGs.
        for (size_t i = 0; i < bgra.size(); i += 4)
        {
            uint8_t& b = bgra[i + 0];
            uint8_t& g = bgra[i + 1];
            uint8_t& r = bgra[i + 2];
            uint8_t& a = bgra[i + 3];

            if (a == 255)
            {
                continue;
            }

            b = (uint8_t)((b * a + 127) / 255);
            g = (uint8_t)((g * a + 127) / 255);
            r = (uint8_t)((r * a + 127) / 255);
        }
        return true;
    }
}

struct SplashScreen::State
{
    SplashScreenState s;
};

SplashScreen::SplashScreen()
{
    m_state = new State();
    m_state->s.hInstance = GetModuleHandleW(nullptr);
}

SplashScreen::~SplashScreen()
{
    Stop();
    delete m_state;
    m_state = nullptr;
}

#include <windows.h>
#include <string>

static std::wstring GetExeDirectory()
{
    wchar_t path[MAX_PATH];

    DWORD len = GetModuleFileNameW(nullptr, path, MAX_PATH);
    if (len == 0 || len == MAX_PATH)
    {
        return L"";
    }

    // Strip filename
    for (DWORD i = len; i > 0; --i)
    {
        if (path[i - 1] == L'\\' || path[i - 1] == L'/')
        {
            path[i - 1] = L'\0';
            break;
        }
    }

    return std::wstring(path);
}

static bool FileExists(const std::wstring& path)
{
    DWORD attrs = GetFileAttributesW(path.c_str());
    return (attrs != INVALID_FILE_ATTRIBUTES) &&
        !(attrs & FILE_ATTRIBUTE_DIRECTORY);
}

static std::wstring FindSplashPngPath(const std::wstring& pngName)
{
    const std::wstring exeDir = GetExeDirectory();
    if (exeDir.empty())
    {
        return L"";
    }

    std::wstring p1 = exeDir + L"\\" + pngName; 
    if (FileExists(p1))
    {
        return p1;
    }

    std::wstring p2 = exeDir + L"\\Assets\\" + pngName;
    if (FileExists(p2))
    {
        return p2;
    }

    std::wstring p3 = exeDir + L"\\..\\Assets\\" + pngName;
    if (FileExists(p3))
    {
        return p3;
    }

    return L""; // not found
}


bool SplashScreen::Start(const std::wstring& pngName)
{
    Stop();

    auto& s = m_state->s;
    s.pngPath = FindSplashPngPath(pngName);

    if (s.pngPath == L"")
        return false;

    s.readyEvent = CreateEventW(nullptr, TRUE, FALSE, nullptr);

    s.threadHandle = CreateThread(nullptr, 0, ThreadProc, m_state, 0, &s.threadId);
    if (!s.threadHandle)
    {
        CloseHandle(s.readyEvent);
        s.readyEvent = nullptr;
        return false;
    }

    // Wait until the splash thread has created the window (or decided it can't).
    WaitForSingleObject(s.readyEvent, INFINITE);
    CloseHandle(s.readyEvent);
    s.readyEvent = nullptr;

    return true;
}

void SplashScreen::Stop()
{
    auto& s = m_state->s;

    HWND hwnd = s.hwnd.load(std::memory_order_acquire);
    if (hwnd)
    {
        PostMessageW(hwnd, WM_SPLASH_CLOSE, 0, 0);
    }

    if (s.threadHandle)
    {
        WaitForSingleObject(s.threadHandle, INFINITE);
        CloseHandle(s.threadHandle);
        s.threadHandle = nullptr;
        s.threadId = 0;
    }

    s.hwnd.store(nullptr, std::memory_order_release);
}

DWORD WINAPI SplashScreen::ThreadProc(void* param)
{
    State* st = (State*)param;
    auto& s = st->s;

    // Your app is PerMonitorV2. Make this UI thread PerMonitorV2 as well.
    auto oldCtx = SetThreadDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2);

    CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED);

    // Register window class (per-process; safe to call multiple times with same name)
    const wchar_t* kClassName = L"SplashScreenWindowClass_PMv2";

    WNDCLASSW wc = {};
    wc.lpfnWndProc = &SplashScreen::WndProc;
    wc.hInstance = s.hInstance;
    wc.lpszClassName = kClassName;
    wc.hCursor = LoadCursor(nullptr, IDC_ARROW);
    RegisterClassW(&wc);

    // Create window: borderless, no taskbar button, optionally topmost.
    HWND hwnd = CreateWindowExW(
        WS_EX_TOOLWINDOW | WS_EX_LAYERED | WS_EX_TOPMOST,
        kClassName,
        L"",
        WS_POPUP,
        0, 0, 1, 1,
        nullptr, nullptr, s.hInstance,
        st);

    s.hwnd.store(hwnd, std::memory_order_release);

    // Load image
    std::vector<uint8_t> bgra;
    UINT w = 0;
    UINT h = 0;

    bool ok = LoadPngWic_BGRA32(s.pngPath.c_str(), bgra, w, h);
    if (ok)
    {
        // Premultiply for UpdateLayeredWindow (even if mostly opaque, this is fine).
        PremultiplyIfNeeded(bgra);

        ShowWindow(hwnd, SW_SHOWNORMAL);
        UpdateWindow(hwnd);

        SetLayeredWindowBitmap(hwnd, bgra.data(), w, h);
    }
    else
    {
        // If image fails, don't show an empty window.
        ShowWindow(hwnd, SW_HIDE);
    }

    if (s.readyEvent)
    {
        SetEvent(s.readyEvent);
    }

    MSG msg;
    while (GetMessageW(&msg, nullptr, 0, 0) > 0)
    {
        TranslateMessage(&msg);
        DispatchMessageW(&msg);
    }

    s.hwnd.store(nullptr, std::memory_order_release);

    CoUninitialize();

    if (oldCtx)
    {
        SetThreadDpiAwarenessContext(oldCtx);
    }

    return 0;
}

LRESULT CALLBACK SplashScreen::WndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
    switch (msg)
    {
    case WM_NCCREATE:
    {
        // Store pointer for potential future extension.
        auto* cs = (CREATESTRUCTW*)lParam;
        SetWindowLongPtrW(hwnd, GWLP_USERDATA, (LONG_PTR)cs->lpCreateParams);
        return TRUE;
    }

    case WM_SPLASH_CLOSE:
    case WM_CLOSE:
    {
        DestroyWindow(hwnd);
        return 0;
    }

    case WM_DESTROY:
    {
        PostQuitMessage(0);
        return 0;
    }
    }

    return DefWindowProcW(hwnd, msg, wParam, lParam);
}

bool SplashScreen::LoadPngWic_BGRA32(const wchar_t* path, std::vector<uint8_t>& outBGRA, UINT& outW, UINT& outH)
{
    IWICImagingFactory* factory = nullptr;
    IWICBitmapDecoder* decoder = nullptr;
    IWICBitmapFrameDecode* frame = nullptr;
    IWICFormatConverter* conv = nullptr;

    HRESULT hr = CoCreateInstance(CLSID_WICImagingFactory, nullptr, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&factory));
    if (FAILED(hr))
    {
        return false;
    }

    hr = factory->CreateDecoderFromFilename(path, nullptr, GENERIC_READ, WICDecodeMetadataCacheOnLoad, &decoder);
    if (FAILED(hr))
    {
        factory->Release();
        return false;
    }

    hr = decoder->GetFrame(0, &frame);
    if (FAILED(hr))
    {
        decoder->Release();
        factory->Release();
        return false;
    }

    hr = frame->GetSize(&outW, &outH);
    if (FAILED(hr) || outW == 0 || outH == 0)
    {
        frame->Release();
        decoder->Release();
        factory->Release();
        return false;
    }

    hr = factory->CreateFormatConverter(&conv);
    if (FAILED(hr))
    {
        frame->Release();
        decoder->Release();
        factory->Release();
        return false;
    }

    // 32bpp BGRA (straight alpha)
    hr = conv->Initialize(
        frame,
        GUID_WICPixelFormat32bppBGRA,
        WICBitmapDitherTypeNone,
        nullptr,
        0.0,
        WICBitmapPaletteTypeCustom);

    if (FAILED(hr))
    {
        conv->Release();
        frame->Release();
        decoder->Release();
        factory->Release();
        return false;
    }

    outBGRA.resize((size_t)outW * (size_t)outH * 4);
    hr = conv->CopyPixels(nullptr, outW * 4, (UINT)outBGRA.size(), outBGRA.data());

    conv->Release();
    frame->Release();
    decoder->Release();
    factory->Release();

    return SUCCEEDED(hr);
}

void SplashScreen::CenterOnPrimaryMonitorPx(UINT wPx, UINT hPx, int& outX, int& outY)
{
    // With thread DPI set to PerMonitorV2, these monitor rects are in physical pixels.
    POINT pt = { 0, 0 };
    HMONITOR mon = MonitorFromPoint(pt, MONITOR_DEFAULTTOPRIMARY);

    MONITORINFO mi = {};
    mi.cbSize = sizeof(mi);
    GetMonitorInfoW(mon, &mi);

    RECT rc = mi.rcMonitor;
    int monW = rc.right - rc.left;
    int monH = rc.bottom - rc.top;

    outX = rc.left + (monW - (int)wPx) / 2;
    outY = rc.top + (monH - (int)hPx) / 2;
}

void SplashScreen::SetLayeredWindowBitmap(HWND hwnd, const uint8_t* bgraPremul, UINT w, UINT h)
{
    HDC screenDC = GetDC(nullptr);
    HDC memDC = CreateCompatibleDC(screenDC);

    BITMAPINFO bmi = {};
    bmi.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
    bmi.bmiHeader.biWidth = (LONG)w;
    bmi.bmiHeader.biHeight = -(LONG)h; // top-down
    bmi.bmiHeader.biPlanes = 1;
    bmi.bmiHeader.biBitCount = 32;
    bmi.bmiHeader.biCompression = BI_RGB;

    void* dibBits = nullptr;
    HBITMAP dib = CreateDIBSection(screenDC, &bmi, DIB_RGB_COLORS, &dibBits, nullptr, 0);
    memcpy(dibBits, bgraPremul, (size_t)w * (size_t)h * 4);

    HGDIOBJ oldObj = SelectObject(memDC, dib);

    int x = 0;
    int y = 0;
    CenterOnPrimaryMonitorPx(w, h, x, y);

    POINT ptDst = { x, y };
    SIZE  szDst = { (LONG)w, (LONG)h };
    POINT ptSrc = { 0, 0 };

    BLENDFUNCTION bf = {};
    bf.BlendOp = AC_SRC_OVER;
    bf.SourceConstantAlpha = 255;
    bf.AlphaFormat = AC_SRC_ALPHA;

    UpdateLayeredWindow(hwnd, screenDC, &ptDst, &szDst, memDC, &ptSrc, 0, &bf, ULW_ALPHA);

    SelectObject(memDC, oldObj);
    DeleteObject(dib);
    DeleteDC(memDC);
    ReleaseDC(nullptr, screenDC);
}
