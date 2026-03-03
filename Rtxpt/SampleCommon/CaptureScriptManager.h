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

#include <memory>
#include <filesystem>

#include <donut/app/imgui_renderer.h>

class Sample;
class SampleUIData;
struct CommandLineOptions;

class CaptureScriptManager
{
public:
	CaptureScriptManager(Sample & sample, SampleUIData & sampleUI, const CommandLineOptions & cmdLine);
	~CaptureScriptManager();

    bool ScriptProgressUI();

    void ScriptMainUI(const ImVec4 & warnColor, const ImVec4 & categoryColor, float indent, float currentScale);

    void PreAnim(float & fElapsedTimeSeconds);
    void PostAnim();

    void PreRender();
    void PostRender(const std::function<bool(const char*)>& dumpScreenshotCallback);

    bool IsDoingWork() const    { return m_start || m_active; }


private:
    Sample &                    m_app;
    SampleUIData &              m_ui;
    const CommandLineOptions &  m_cmdLine;
    
    std::filesystem::path       m_screenshotFileName;

    // std::string                 m_screenshotSequencePath            = "D:/AnimSequence/";
    // bool                        m_screenshotSequenceCaptureActive   = false;
    // int                         m_screenshotSequenceCaptureIndex    = -64; // -x means x warmup frames for recording to stabilize denoiser and other subsystems

    bool                        m_resetAndWarmup                    = false;
    int                         m_resetAndWarmupFrames              = 64;
    int                         m_resetAndWarmupCounter             = -1;

    // bool                        m_screenshotMiniSequence            = false;
    // int                         m_screenshotMiniSequenceFrames      = 5;
    // int                         m_screenshotMiniSequenceCounter     = -1;

    //bool                        m_sequenceRecorder                  = false;
    double                      m_sequenceBeginTime                 = 0.0;     // based on sequenceWarmupStart cmdLine parameter 
    double                      m_sequenceRecordStartTime           = 0.0;
    double                      m_sequenceDeltaTime                 = 1.0 / 60.0;
    int                         m_sequenceRecordFrames              = 0;
    int                         m_sequenceRecordCounter             = -1;


    bool                        m_start                             = false;
    bool                        m_active                            = false;
    int                         m_type                              = 0;    // 0 - simple screenshot; 1 - advanced sequence

    bool                        m_captureSuccess                    = false;
    bool                        m_exitAfterCapture                  = false; // only active if command line parameters used
};
