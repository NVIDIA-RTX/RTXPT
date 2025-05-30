/*
* Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "SampleCommon.h"
#include <donut/core/log.h>

#include <donut/core/json.h>
#include <donut/app/ApplicationBase.h>
#include <json/json.h>
#include <json/value.h>
#include <fstream>

using namespace donut;

#include <windows.h>
#include <commctrl.h>
#include <thread>
#include <atomic>
#include <regex>

#ifdef _WIN32
#pragma comment(lib, "Comctl32.lib")
#endif

bool EnsureDirectoryExists(const std::filesystem::path& dir)
{
    if (std::filesystem::exists(dir))
    {
        if (!std::filesystem::is_directory(dir))
        {
            log::error("Attempting ensure directory exists, but path '%s' is a file", dir.string().c_str());
            return false;
        }
        return true;
    }

    std::filesystem::path pathSoFar;
    for (const auto& part : std::filesystem::absolute(dir)) 
    {
        pathSoFar /= part;
        if (std::filesystem::exists(pathSoFar))
        {
            if (!std::filesystem::is_directory(pathSoFar))
            {
                log::error("Attempting ensure directory exists, but path '%s' is a file", pathSoFar.string().c_str());
                return false;
            }
        }
        else
        {
            if (!std::filesystem::create_directory(pathSoFar))
            {
                log::error("Attempting ensure directory exists, but unable to create or access '%s' directory", pathSoFar.string().c_str());
                return false;
            }
        }
    }
    return true;
}

std::vector<std::filesystem::path> EnumerateFilesWithWildcard( const std::filesystem::path& folder, const std::string& wildcard ) 
{
    // Convert wildcard to regex: ? -> ., * -> .*
    std::string regexPattern = "^";
    for (char c : wildcard) {
        if (c == '*')      regexPattern += ".*";
        else if (c == '?') regexPattern += '.';
        else if (c == '.') regexPattern += "\\.";
        else               regexPattern += c;
    }
    regexPattern += "$";
    std::regex pattern(regexPattern, std::regex::icase);

    std::vector<std::filesystem::path> result;
    for (const auto& entry : std::filesystem::directory_iterator(folder)) 
    {
        if (entry.is_regular_file()) 
        {
            if (std::regex_match(entry.path().filename().string(), pattern)) 
            {
                result.push_back(entry.path());
            }
        }
    }
    return result;
}

bool SaveJsonToFile( const std::filesystem::path & filePath, const class Json::Value & rootNode )
{
    std::ofstream outFile(filePath, std::ios::trunc);
    if (!outFile.is_open())
    {
        log::error("Error attempting to save json contents to file '%s'", filePath.string().c_str());
        return false;
    }
    Json::StreamWriterBuilder builder;
    std::unique_ptr<Json::StreamWriter> writer(builder.newStreamWriter());
    writer->write(rootNode, &outFile);
    outFile.close();
    return true;
}

bool LoadJsonFromFile( const std::filesystem::path & filePath, Json::Value & outRootNode )
{
    std::ifstream inFile;
    inFile.open(filePath);

    if (!inFile.is_open())
    {
        inFile.open(filePath);
        if (!inFile.is_open())
        {
            donut::log::warning("Error attempting to load json file '%s'", filePath.string().c_str());
            return false;
        }
    }

    inFile >> outRootNode;
    return true;
}

bool LoadJsonFromString(const std::string & jsonData, Json::Value& outRootNode)
{
    Json::CharReaderBuilder readerBuilder;
    std::string errs;

    std::istringstream iss(jsonData);
    return Json::parseFromStream(readerBuilder, iss, &outRootNode, &errs);
}

struct ProgressBarGlobals
{
    bool                    Active          = false;
    std::string             Title           = "";
    int                     Value           = 0;
#ifdef _WIN32
    HWND                    hMainWindow     = nullptr;
    HWND                    hProgressBar    = nullptr;
#endif
    std::recursive_mutex    Mutex;
    std::thread             Thread;

};

constexpr int               g_ProgressSlotCount = 8;
static ProgressBarGlobals   g_ProgressSlots[g_ProgressSlotCount];

#ifdef _WIN32
static std::atomic<HWND>    g_hActiveParentWindow(NULL);
void HelpersRegisterActiveWindow()
{
    g_hActiveParentWindow.store( GetActiveWindow() );
    assert( g_hActiveParentWindow.load() != NULL );
}
HWND HelpersGetActiveWindow()
{
    return g_hActiveParentWindow.load();
}

LRESULT CALLBACK ProgressWndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam) 
{
    switch (msg) 
    {
    case WM_CREATE:
    {
        CREATESTRUCT * cs = reinterpret_cast<CREATESTRUCT*>(lParam);
        ProgressBarGlobals & slot = *reinterpret_cast<ProgressBarGlobals*>(cs->lpCreateParams);
        std::lock_guard guard(slot.Mutex);
        assert (slot.Active); // && g_ProgressSlots[i].hMainWindow == hwnd) <- hwnd doesn't get set until CreateWindowEx finishes
        RECT clientRect;
        GetClientRect(hwnd, &clientRect);
        slot.hProgressBar = CreateWindowEx(0, PROGRESS_CLASS, NULL, WS_CHILD | WS_VISIBLE,
                    0, 0, clientRect.right, clientRect.bottom, hwnd, NULL, GetModuleHandle(NULL), NULL);

        SendMessage(slot.hProgressBar, PBM_SETRANGE, 0, MAKELPARAM(0, 100));
        SendMessage(slot.hProgressBar, PBM_SETPOS, 0, 0);
        return 0;
    }

    //  in case of resizing main window, something like this keeps progress ctrl proper size:
    //  case WM_SIZE: { RECT clientRect; GetClientRect(hwnd, &clientRect); MoveWindow(g_hProgressBar, 0, 0, clientRect.right, clientRect.bottom, TRUE); }

    case WM_CLOSE:      DestroyWindow(hwnd);    return 0;
    case WM_DESTROY:    PostQuitMessage(0);     return 0;
    }
    return DefWindowProc(hwnd, msg, wParam, lParam);
}

void ShowLastError()
{
    DWORD err = GetLastError(); wchar_t buf[4096];
    FormatMessageW(FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS, NULL, err, 0, buf, sizeof(buf) / sizeof(wchar_t), NULL);
    MessageBoxW(NULL, buf, L"LastError", MB_OK | MB_ICONERROR);
}

#else // _WIN32
void HelpersRegisterActiveWindow() {}
#endif

void ShowProgressWindow(int slotIndex)
{
#ifdef _WIN32
    ProgressBarGlobals & slot = g_ProgressSlots[slotIndex];

    const char* wClassName = "ProgressWindowClass";
    HINSTANCE hInstance = GetModuleHandle(NULL);
    static std::mutex   registerClassMtx;
    static bool         registerClassFlag = false;
    {
        std::lock_guard<std::mutex> lock(registerClassMtx);
        if (!registerClassFlag)
        {
            INITCOMMONCONTROLSEX icex = { sizeof(INITCOMMONCONTROLSEX), ICC_PROGRESS_CLASS };
            InitCommonControlsEx(&icex);
            WNDCLASS wc = {};
            wc.lpfnWndProc = ProgressWndProc;
            wc.hInstance = hInstance;
            wc.lpszClassName = wClassName;
            wc.hCursor = LoadCursor(NULL, IDC_ARROW);

            if (!RegisterClass(&wc))
                ShowLastError();
            registerClassFlag = true;
        }
    }

    int width = 600;
    int height = 100;

    // Default position if no parent
    int x = CW_USEDEFAULT;
    int y = CW_USEDEFAULT;

    HWND hParent = g_hActiveParentWindow.load();
    HWND hwnd = NULL;
    {
        std::lock_guard guard(slot.Mutex);
        if (!slot.Active) // already finished? don't even open the window
            return;

        if (hParent) 
        {
            RECT parentRect;
            GetWindowRect(hParent, &parentRect);

            x = parentRect.left + (parentRect.right - parentRect.left - width) / 2;
            y = parentRect.top + (parentRect.bottom - parentRect.top - height*3) / 2;
            y += height * slotIndex;
        }

        slot.hMainWindow = CreateWindowEx( WS_EX_TOOLWINDOW, wClassName, slot.Title.c_str(),
            WS_OVERLAPPED | WS_CAPTION | WS_SYSMENU, x, y, width, height,
            NULL/*hParent*/, NULL, hInstance, reinterpret_cast<LPVOID>(&slot));

        if (slot.hMainWindow == NULL)
        {
            ShowLastError();
            assert( false );
            return;
        }

        hwnd = slot.hMainWindow;
    }

    ShowWindow(hwnd, SW_SHOW);
    UpdateWindow(hwnd);

    while (true) 
    {
        MSG msg;
        while (PeekMessage(&msg, NULL, 0, 0, PM_REMOVE)) 
        {
            if (msg.message == WM_QUIT)
                return;
            TranslateMessage(&msg);
            DispatchMessage(&msg);
        }

        {
            std::lock_guard guard(slot.Mutex);
            if (!slot.Active)
                break;

            PostMessage(slot.hProgressBar, PBM_SETPOS, slot.Value, 0);
        }
        Sleep(16); // avoid busy-waiting
    }
    SendMessage(hwnd, WM_CLOSE, 0, 0);
#endif
}

bool IsWindowFullscreen()
{
#ifdef _WIN32
    HWND activeWindow = HelpersGetActiveWindow();
    // Check if window is valid
    if (!IsWindow(activeWindow)) return false;

    // Get window rect
    RECT windowRect;
    if (!GetWindowRect(activeWindow, &windowRect)) return false;

    // Get the monitor handle the window is on
    HMONITOR hMonitor = MonitorFromWindow(activeWindow, MONITOR_DEFAULTTONEAREST);
    if (!hMonitor) return false;

    // Get monitor info
    MONITORINFO mi = {};
    mi.cbSize = sizeof(mi);
    if (!GetMonitorInfo(hMonitor, &mi)) return false;

    RECT monitorRect = mi.rcMonitor;

    // Compare rects
    return EqualRect(&windowRect, &monitorRect);
#else
    return true;
#endif
}

int ProgressBarStart(const char * windowText)
{
    // progress bar currently interfering with fullscreen so disable
    if (IsWindowFullscreen())
        return -1;

    for (int i = 0; i < g_ProgressSlotCount; i++)
    {
        std::lock_guard guard(g_ProgressSlots[i].Mutex);
        ProgressBarGlobals & slot = g_ProgressSlots[i];
        if (!slot.Active)
        {
            slot.Active = true;
            slot.Title = windowText;
            slot.Value = 0;
            assert(!slot.Thread.joinable());

            slot.Thread = std::thread(ShowProgressWindow, i);
            return i;
        }
    }
    return -1;
}

void ProgressBarStop(int slotIndex)
{
    assert( slotIndex >= 0 && slotIndex < g_ProgressSlotCount );
    std::thread threadToJoin;
    {
        ProgressBarGlobals& slot = g_ProgressSlots[slotIndex];
        std::lock_guard guard(slot.Mutex);
        
        slot.Active          = false;
        slot.Title           = "";
        slot.Value           = 0;
#ifdef _WIN32
        slot.hMainWindow     = nullptr;
        slot.hProgressBar    = nullptr;
#endif
        threadToJoin.swap(slot.Thread);
    }
    if (threadToJoin.joinable())
        threadToJoin.join();
}

void ProgressBarUpdate(int slotIndex, int percentage)
{
    assert(slotIndex >= 0 && slotIndex < g_ProgressSlotCount);
    {
        ProgressBarGlobals& slot = g_ProgressSlots[slotIndex];
        std::lock_guard guard(slot.Mutex);
        slot.Value = percentage;
    }
}

std::filesystem::path GetLocalPath(std::string subfolder)
{
    static std::filesystem::path oneChoice;
    // if( oneChoice.empty() )
    {
        std::filesystem::path candidateA = donut::app::GetDirectoryWithExecutable() / subfolder;
        std::filesystem::path candidateB = donut::app::GetDirectoryWithExecutable().parent_path() / subfolder;
        if (std::filesystem::exists(candidateA))
            oneChoice = candidateA;
        else
            oneChoice = candidateB;
    }
    return oneChoice;
}

std::string StringLoadFromFile( const std::filesystem::path & filePath )
{
    std::ifstream file(filePath);
    if (!file) { return ""; }

    std::ostringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

std::tuple<int, std::string, std::string > SystemShell(const std::string & command, bool useCmd, bool blockOnExecution)
{
    assert(blockOnExecution); // non blocking not implemented

    int resultValue;
    std::string resultString;
    std::string resultErrorString;
    
#if defined(_WIN32)
    SECURITY_ATTRIBUTES sa = { sizeof(SECURITY_ATTRIBUTES), NULL, TRUE };
    HANDLE hReadOut = NULL, hWriteOut = NULL, hReadErr = NULL, hWriteErr = NULL;

    // Create pipes for stdout and stderr
    CreatePipe(&hReadOut, &hWriteOut, &sa, 0);
    CreatePipe(&hReadErr, &hWriteErr, &sa, 0);

    // Ensure the read handles are not inherited
    SetHandleInformation(hReadOut, HANDLE_FLAG_INHERIT, 0);
    SetHandleInformation(hReadErr, HANDLE_FLAG_INHERIT, 0);

    // useful for debugging
    bool showConsole = false;

    STARTUPINFOA si = { 0 };
    PROCESS_INFORMATION pi = { 0 };
    si.cb = sizeof(si);
    if (!showConsole)
    {
        si.dwFlags = STARTF_USESTDHANDLES | STARTF_USESHOWWINDOW;
        si.hStdOutput = hWriteOut;
        si.hStdError = hWriteErr;
        si.wShowWindow = SW_HIDE;
    }

    std::string cmd = command;
    if (useCmd)
        cmd = "cmd /C \"" + command + "\"";
    BOOL success = CreateProcessA( NULL, cmd.data(), NULL, &sa, TRUE, showConsole?CREATE_NEW_CONSOLE:CREATE_NO_WINDOW, NULL, NULL, &si, &pi );
    CloseHandle(hWriteOut);
    CloseHandle(hWriteErr);

    if (success)
    {
        std::string output;
        char buffer[4096];
        DWORD bytesRead;
        int loopCounter = 0;

        while(true)
        {
            DWORD waitResult = WaitForSingleObject(pi.hProcess, 50);

            DWORD available = 0;

            // Read stdout
            while (PeekNamedPipe(hReadOut, NULL, 0, NULL, &available, NULL) && available > 0)
            {
                if (ReadFile(hReadOut, buffer, sizeof(buffer) - 1, &bytesRead, NULL) && bytesRead > 0) 
                {
                    buffer[bytesRead] = '\0';
                    resultString += buffer;
                } else { assert( false ); break; }
            }

            while (PeekNamedPipe(hReadErr, NULL, 0, NULL, &available, NULL) && available > 0)
            {
                if (ReadFile(hReadErr, buffer, sizeof(buffer) - 1, &bytesRead, NULL) && bytesRead > 0) 
                {
                    buffer[bytesRead] = '\0';
                    resultErrorString += buffer;
                } else { assert( false ); break; }
            }

            if (waitResult == WAIT_OBJECT_0) 
                break;
            
            loopCounter++;
        }

        GetExitCodeProcess(pi.hProcess, (DWORD*)&resultValue);
    }


    if (success) 
    {

        CloseHandle(pi.hProcess);
        CloseHandle(pi.hThread);
    }
    else 
    {
        resultErrorString = "CreateProcess failed.";
        resultValue = 0;
    }

    CloseHandle(hReadOut);
    CloseHandle(hReadErr);

#else
    static std::atomic<int> fileLogIndexStorage(0);
    int uniqueIndex = fileLogIndexStorage.fetch_add(1);

#ifdef _WIN32
    DWORD pid = GetCurrentProcessId();
#else
    pid_t pid = getpid();
#endif
    std::filesystem::path tempLogFile = std::filesystem::temp_directory_path() / StringFormat("RTXPT_out_%d_%d.txt", pid, uniqueIndex);
    std::filesystem::path tempErrLogFile = std::filesystem::temp_directory_path() / StringFormat("RTXPT_err_%d_%d.txt", pid, uniqueIndex);
    std::string startCmd = "\"" + command + " > \"" + tempLogFile.string() + "\"" + " 2> \"" + tempErrLogFile.string() + "\"" + " \"";

    resultValue = std::system(startCmd.c_str());

    if (std::filesystem::exists(tempLogFile))
    {
        resultString = StringLoadFromFile(tempLogFile);
        std::filesystem::remove(tempLogFile);
    }
    if (std::filesystem::exists(tempErrLogFile))
    {
        resultErrorString = StringLoadFromFile(tempErrLogFile);
        std::filesystem::remove(tempErrLogFile);
    }
#endif

    return std::make_tuple(resultValue, resultString, resultErrorString);
}

namespace fs = std::filesystem;

std::optional<fs::file_time_type> GetLatestModifiedTimeDirectoryRecursive(const fs::path & directory) 
{
    std::error_code ec;

    if (!fs::exists(directory, ec) || !fs::is_directory(directory, ec)) 
    {
        //log::info("Invalid directory or access error: '%s'", directory.string().c_str());
        return std::nullopt;
    }

    std::optional<fs::file_time_type> latest_time;

    fs::recursive_directory_iterator dir_it(directory, fs::directory_options::skip_permission_denied, ec);
    fs::recursive_directory_iterator end_it;

    while (dir_it != end_it && !ec) 
    {
        const auto& entry = *dir_it;

        if (entry.is_regular_file(ec)) 
        {
            auto ftime = entry.last_write_time(ec);
            if (!ec) 
            {
                if (!latest_time || ftime > *latest_time)
                    latest_time = ftime;
            }
        }

        dir_it.increment(ec); // manually increment to pass error_code
    }

    if (ec) 
    {
        log::warning("Invalid directory or access error for '%s', error: ", directory.string().c_str(), ec.message().c_str());
        return std::nullopt;
    }

    return latest_time;
}

std::optional<std::filesystem::file_time_type> GetFileModifiedTime(const std::filesystem::path & file)
{
    std::error_code ec;

    if (!fs::exists(file, ec) || !fs::is_regular_file(file, ec))
    {
        //log::info("File does not exist or is not a regular file: '%s'", file.string().c_str());
        return std::nullopt;
    }

    auto ftime = fs::last_write_time(file, ec);
    if (ec)
    {
        log::warning("Failed to get last write time for file '%s', error: %s", file.string().c_str(), ec.message().c_str());
        return std::nullopt;
    }

    return ftime;
}