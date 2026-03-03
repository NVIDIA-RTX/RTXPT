/*
* Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "CaptureScriptManager.h"

#include <string>
#include "../SampleUI.h"
#include "../Sample.h"
#include "SampleCommon.h"
#include "CommandLine.h"

#include <donut/app/UserInterfaceUtils.h>

#include <donut/core/math/math.h>

using namespace donut::app;
using namespace donut::engine;

CaptureScriptManager::CaptureScriptManager(Sample & sample, SampleUIData & sampleUI, const CommandLineOptions & cmdLine)
    : m_app(sample), m_ui(sampleUI), m_cmdLine(cmdLine)
{
    if (m_cmdLine.capturePath != "")
    {
        m_screenshotFileName = m_cmdLine.capturePath;

        if (m_cmdLine.captureSimple)
        {
            m_type = 0;
            m_start = true;
            m_resetAndWarmup = true;
            m_exitAfterCapture = true;
        }

        if (m_cmdLine.captureSequence && m_cmdLine.sequenceFPS > 0 && m_cmdLine.sequenceFrameCount > 0)
        {
            m_type = 1;
            m_start = true;
            assert( m_cmdLine.sequenceWarmupStart >= 0 );
            assert( m_cmdLine.sequenceWarmupStart < m_cmdLine.sequenceRecordStart );
            assert( m_cmdLine.sequenceFPS >= 1 && m_cmdLine.sequenceFPS <= 1000 );
            m_sequenceDeltaTime = 1.0 / m_cmdLine.sequenceFPS;
            m_sequenceBeginTime = m_cmdLine.sequenceWarmupStart;
            m_sequenceRecordStartTime = m_cmdLine.sequenceRecordStart;

            m_resetAndWarmupFrames = (int)((m_cmdLine.sequenceRecordStart - m_cmdLine.sequenceWarmupStart) / m_sequenceDeltaTime + 1e-6);
            m_resetAndWarmup = true;
            m_sequenceRecordFrames = m_cmdLine.sequenceFrameCount;
            m_exitAfterCapture = true;
            assert( !m_cmdLine.stopAnimations ); // doesn't really make sense
        }
    }
}

CaptureScriptManager::~CaptureScriptManager()
{
}

bool CaptureScriptManager::ScriptProgressUI()
{
    if (!m_active && !m_start)
        return false;

    if (m_resetAndWarmupCounter > 0)
    {
        ImGui::Text("");
        ImGui::TextWrapped("Running warm-up: %d out of %d", m_resetAndWarmupCounter, m_resetAndWarmupFrames);
    }
    if (!m_ui.RealtimeMode)
    {
        ImGui::TextWrapped("Accumulation mode, sample %d (out of %d target)", m_app.GetAccumulationSampleIndex(), m_ui.AccumulationTarget);
    }
    if (m_sequenceRecordCounter > 0)
    {
        ImGui::Text("");
        ImGui::TextWrapped("Running sequence export: %d", m_sequenceRecordCounter);
    }
    if (m_start)
    {
        ImGui::Text("");
        ImGui::TextWrapped("Starting (scene loading might still in progress)...");
    }

    return true;
}

void CaptureScriptManager::ScriptMainUI(const ImVec4 & warnColor, const ImVec4 & categoryColor, float indent, float currentScale)
{
    assert( !m_active && !m_start ); // this point should never be reached if m_active because ScritProgressUI should be up instead

    ImGui::TextColored(categoryColor, "Options");

    if (ImGui::Combo("Capture type", &m_type, "SimpleScreenshot\0SequenceCapture\0Benchmark\00" ))
        m_type = dm::clamp(m_type, 0, 0 );

    if( m_type == 0 )
    {
        ImGui::Checkbox("ResetAndWarmup", &m_resetAndWarmup);
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("When 'Save screenshot' is used, all subsystem's temporal histories will first be reset,\nfollowed by a selected number of frames before saving the screenshot.\nIf animation is running, it will keep running - if not, it won't.");
        {
            UI_SCOPED_DISABLE(!m_resetAndWarmup);
            ImGui::SameLine();
            ImGui::PushItemWidth(-90.0f * currentScale);
            ImGui::InputInt("delay frames", &m_resetAndWarmupFrames); m_resetAndWarmupFrames = dm::clamp(m_resetAndWarmupFrames, 0, 10000);
            ImGui::PopItemWidth();
        }
    }

    // ImGui::Checkbox("Sequence     ", &m_screenshotMiniSequence);
    // if (ImGui::IsItemHovered()) ImGui::SetTooltip("When 'Save screenshot' used, a sequence of screenshots will be recorded instead of saving\na single one. Can work together with ResetAndDelay.");
    // {
    //     UI_SCOPED_DISABLE(!m_screenshotMiniSequence);
    //     ImGui::SameLine();
    //     ImGui::PushItemWidth(-90.0f * currentScale);
    //     ImGui::InputInt("length", &m_screenshotMiniSequenceFrames); m_screenshotMiniSequenceFrames = dm::clamp(m_screenshotMiniSequenceFrames, 1, 999);
    //     ImGui::PopItemWidth();
    // }

    float w = ImGui::GetContentRegionAvail().x;

    ImGui::TextColored(categoryColor, "Output file(s) path:");
    ImGui::SameLine(w * 0.65f);
    if (ImGui::Button("[ select path ]"))
    {
        std::string fileName;
        if (FileDialog(false, "PNG files\0*.png\0BMP files\0*.bmp\0All files\0*.*\0\0", fileName))
            m_screenshotFileName = fileName;
    }

    {
        RAII_SCOPE(ImGui::Indent(indent);, ImGui::Unindent(indent); );
        RAII_SCOPE(ImGui::PushItemWidth(ImGui::GetContentRegionAvail().x); , ImGui::PopItemWidth(); );
        std::string pathStr = m_screenshotFileName.string();
        ImGui::InputText("##path", (char*)pathStr.c_str(), pathStr.length()+1, /*ImVec2(0, 0),*/ ImGuiInputTextFlags_ReadOnly);
    }

    {
        UI_SCOPED_DISABLE(m_screenshotFileName.empty());

        std::string cmd = "";
        if (m_type == 0)
            cmd += "--captureSimple ";
        cmd += "--capturePath " + (m_screenshotFileName.empty()?"<error, empty>":m_screenshotFileName.string());

        ImGui::TextColored(categoryColor, "CmdLine, active settings:");
        ImGui::SameLine(w * 0.65f);
        if (ImGui::Button("[cpy to clpbrd]"))
            ImGui::SetClipboardText(cmd.c_str());

        {
            RAII_SCOPE(ImGui::Indent(indent);, ImGui::Unindent(indent); );
            RAII_SCOPE(ImGui::PushItemWidth(ImGui::GetContentRegionAvail().x);, ImGui::PopItemWidth(); );
            ImGui::InputText( "##cmdline", (char*)cmd.c_str(), cmd.length()+1, /*ImVec2(0,0),*/ ImGuiInputTextFlags_ReadOnly);
        }

        if (ImGui::Button("=== START CAPTURE ===", ImVec2(-FLT_MIN, 0.0f)))
            m_start = true;
    }


#if 0
    ImGui::Separator();
    ImGui::TextColored(categoryColor, "[experimental] Save stable animation sequence, path:");
    ImGui::Text(" '%s'", m_screenshotSequencePath.c_str());
    if (ImGui::Checkbox("Save animation sequence", &m_screenshotSequenceCaptureActive))
        if (m_screenshotSequenceCaptureActive)
            m_ui.FPSLimiter = 60;
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Example to convert to movie: \nffmpeg -r 60 -i frame_%%05d.bmp -vcodec libx265 -crf 13 -vf scale=1920:1080  outputvideo-1080p-60fps.mp4\n"
        "60 FPS limiter will be automatically enabled for smooth recording!");
    if (!m_screenshotSequenceCaptureActive)
        m_screenshotSequenceCaptureIndex = -64; // -x means x warmup frames for recording to stabilize denoiser
    else
    {
        if (m_screenshotSequenceCaptureIndex < 0) // first x frames are warmup!
            m_app.ResetSceneTime();
        else
        {
            char windowName[1024];
            snprintf(windowName, sizeof(windowName), "%s/frame_%05d.bmp", m_screenshotSequencePath.c_str(), m_screenshotSequenceCaptureIndex);
            m_screenshotFileName = windowName;
        }
        m_screenshotSequenceCaptureIndex++;
    }
    // ImGui::Separator();
    // ImGui::Checkbox("Loop longest animation", &m_ui.LoopLongestAnimation);
    // if (ImGui::IsItemHovered()) ImGui::SetTooltip("If enabled, only restarts all animations when longest one played out. Otherwise loops them individually (and not in sync)!");
#endif
}

void CaptureScriptManager::PreAnim(float& fElapsedTimeSeconds)
{
    // don't start anything until everything is loaded
    if (m_app.HasAsyncLoadingInProgress())
        return;

    if (m_start && !m_screenshotFileName.empty())
    {
        m_app.SetSceneTime(m_sequenceBeginTime);
        m_start = false;
        m_active = true;
        m_resetAndWarmupCounter = -1;
    }
    if (m_active && m_type == 1)
        fElapsedTimeSeconds = m_sequenceDeltaTime;
}

void CaptureScriptManager::PostAnim()
{

}

void CaptureScriptManager::PreRender()
{
    // Update Screenshot counter
    if (m_active && m_resetAndWarmup)
    {
        if (m_resetAndWarmupCounter == -1) // we just started with delay, set it up
        {
            m_resetAndWarmupCounter = m_resetAndWarmupFrames;
            m_ui.ResetRealtimeCaches = true;
        }
        if (m_resetAndWarmupCounter)
            m_ui.ResetAccumulation = true;

        m_resetAndWarmupCounter = std::max(0, m_resetAndWarmupCounter-1);
    }
}

void CaptureScriptManager::PostRender(const std::function<bool(const char*)>& dumpScreenshotCallback)
{
    if (m_active && !m_screenshotFileName.empty() && !(m_resetAndWarmup && m_resetAndWarmupCounter > 0) && (m_ui.RealtimeMode || m_app.AccumulationCompleted()) )
    {
        std::filesystem::path screenshotFile = m_screenshotFileName;

        if (m_type == 1)
        {
            if (m_app.GetSceneTime() < m_sequenceRecordStartTime) // if not, we're still in warmup - exit
                return;

            if (m_sequenceRecordCounter == -1) // start sequence if in sequence mode
                m_sequenceRecordCounter = m_sequenceRecordFrames;
            assert(m_sequenceRecordCounter > 0);

            std::filesystem::path justName = screenshotFile.filename().stem();
            std::filesystem::path justExtension = screenshotFile.extension();
            screenshotFile.remove_filename();
            screenshotFile /= justName.string() + StringFormat("_%03d", m_sequenceRecordFrames - m_sequenceRecordCounter) + justExtension.string();
            m_sequenceRecordCounter--;
        }

        m_captureSuccess = dumpScreenshotCallback(screenshotFile.string().c_str());

        if (m_type == 0 || (m_type == 1 && m_sequenceRecordCounter == 0) || !m_captureSuccess)
        {
            m_active = false;
            m_resetAndWarmupCounter = -1;
            m_sequenceRecordCounter = -1;
        }
    }

    if (m_exitAfterCapture && !m_active && !m_start)
    {
        if (m_captureSuccess)
        {
            donut::log::info("Capture of '%s' finished successfully. Exiting.", m_screenshotFileName.string().c_str());
            std::exit(0);
        }
        else
        {
            donut::log::fatal("Unable capture '%s'. Exiting.", m_screenshotFileName.string().c_str());
            std::exit(1);
        }
    }
}
