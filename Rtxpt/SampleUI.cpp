/*
* Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "Sample.h"
#include "SampleCommon/SampleBaseApp.h"

#include <inttypes.h>

#include <donut/app/UserInterfaceUtils.h>
#include <donut/core/vfs/VFS.h>
#include <donut/engine/SceneTypes.h>
#include <iterator>
#include <imgui_internal.h>
#include "Materials\MaterialsBaker.h"

#include "ToneMapper/ToneMappingPasses.h"
#include "Misc/Korgi.h"

#include "OpacityMicroMap/OmmBaker.h"

#include "SampleGame/GameScene.h"
#include "Misc/ZoomTool.h"

#include "SampleCommon/CaptureScriptManager.h"

using namespace donut::app;
using namespace donut::engine;

std::filesystem::path GetLocalPath(std::string subfolder);

// Declare SampleUIData as a global, so that we can use the KorgI macros to enable
// Korg nanoKontrol support
SampleUIData g_sampleUIData;

// Declare how the Korg nanoKontrol 2 controls will interact with UI elements
KORGI_TOGGLE(g_sampleUIData.EnableAnimations, 0, Play )

KORGI_TOGGLE(g_sampleUIData.ToneMappingParams.autoExposure, 0, S1 )
KORGI_INT_TOGGLE(g_sampleUIData.ToneMappingParams.toneMapOperator, 0, M1, ToneMapperOperator::Linear, ToneMapperOperator::HableUc2)
KORGI_KNOB(g_sampleUIData.ToneMappingParams.exposureCompensation, 0, Slider1, -12.f, 12.f)

#define RESET_ON_CHANGE(code) do{if (code) m_ui.ResetAccumulation = true;} while(false)

const static ImVec4 warnColor = { 1,0.5f,0.5f,1 };
const static ImVec4 categoryColor = { 0.5f,1.0f,0.7f,1 };

static const PerformancePreset s_performancePresets[] = {
    //                                    SubSamp  NEECand  NEEFull  NEEMIS  SPP  Bounce  DiffBnc   TexLOD  NestDiel  EnvMIP  SPActive  PrimRepl  Bloom    LDSampl    FflyTrhld  Omm2St   DLSS (on separate line due to macros)
    { "Ultra Performance",                2,       3,       1,       1,      1,   6,      1,        0.0f,   1,        3,      2,        false,    false,   false,     0.01,      true,
#if DONUT_WITH_STREAMLINE
        SI::DLSSMode::eUltraPerformance,
#endif
    },
    { "Performance",                      2,       3,       1,       1,      1,   10,     1,       -0.5f,   1,        2,      3,        true,     true,    false,     0.05,      true,
#if DONUT_WITH_STREAMLINE                                                                                                                                                        
        SI::DLSSMode::eMaxPerformance,                                                                                                                                           
#endif                                                                                                                                                                           
    },                                                                                                                                                                           
    { "Balanced",                         1,       5,       1,       1,      1,   14,     2,       -1.0f,   1,        2,      3,        true,     true,    true,      0.1,       false,
#if DONUT_WITH_STREAMLINE                                                                                                                                                        
        SI::DLSSMode::eBalanced,                                                                                                                                                 
#endif                                                                                                                                                                           
    },                                                                                                                                                                           
    { "Quality",                          1,       3,       2,       1,      1,   24,     3,       -1.5f,   1,        2,      3,        true,     true,    true,      0.2,       false,
#if DONUT_WITH_STREAMLINE                                                                                                                                                        
        SI::DLSSMode::eMaxQuality,                                                                                                                                               
#endif                                                                                                                                                                           
    },                                                                                                                                                                           
    { "Ultra Quality",                    1,       3,       2,       0,      1,   48,     3,       -1.5f,   2,        1,      3,        true,     true,    true,      1.0,       false,
#if DONUT_WITH_STREAMLINE
        SI::DLSSMode::eDLAA,
#endif
    },
};
static bool MatchesPreset(const SampleUIData& ui, const PerformancePreset& p, OmmBaker* ommBaker)
{
    if (ui.RealtimeSubSamplingMode != p.SubSamplingMode)                        return false;
    if (ui.NEECandidateSamples != p.NEECandidateSamples)                        return false;
    if (ui.NEEFullSamples != p.NEEFullSamples)                                  return false;
    if (ui.NEEMISType != p.NEEMISType)                                          return false;
    if (ui.RealtimeSamplesPerPixel != p.RealtimeSamplesPerPixel)                return false;
    if (ui.BounceCount != p.BounceCount)                                        return false;
    if (ui.DiffuseBounceCount != p.DiffuseBounceCount)                          return false;
    if (ui.TexLODBias != p.TexLODBias)                                          return false;
    if (ui.NestedDielectricsQuality != p.NestedDielectricsQuality)              return false;
    if (ui.EnvironmentMapDiffuseSampleMIPLevel != p.EnvironmentMapDiffuseSampleMIPLevel) return false;
#if DONUT_WITH_STREAMLINE
    if (ui.DLSSMode != p.DLSSMode)                                              return false;
#endif
    if (ui.DenoisingLayersActiveCount != p.DenoisingLayersActiveCount)                return false;
    if (ui.AllowPrimarySurfaceReplacement != p.AllowPrimarySurfaceReplacement)  return false;
    if (ui.EnableBloom != p.EnableBloom)                                        return false;
    if (ui.EnableLDSamplerForBSDF != p.EnableLDSamplerForBSDF)                  return false;
    if (ui.RealtimeFireflyFilterThreshold != p.FireflyThreshold)                return false;
    if (ommBaker && ommBaker->UIData().Force2State != p.OmmForce2State)         return false;
    return true;
}

static void ApplyPreset(SampleUIData& ui, const PerformancePreset& p, OmmBaker* ommBaker)
{
    ui.RealtimeSubSamplingMode = p.SubSamplingMode;
    ui.NEECandidateSamples = p.NEECandidateSamples;
    ui.NEEFullSamples = p.NEEFullSamples;
    ui.NEEMISType = p.NEEMISType;
    ui.RealtimeSamplesPerPixel = p.RealtimeSamplesPerPixel;
    ui.BounceCount = p.BounceCount;
    ui.DiffuseBounceCount = p.DiffuseBounceCount;
    ui.TexLODBias = p.TexLODBias;
    ui.NestedDielectricsQuality = p.NestedDielectricsQuality;
    ui.EnvironmentMapDiffuseSampleMIPLevel = p.EnvironmentMapDiffuseSampleMIPLevel;
#if DONUT_WITH_STREAMLINE
    ui.DLSSMode = p.DLSSMode;
#endif
    ui.DenoisingLayersActiveCount = p.DenoisingLayersActiveCount;
    ui.AllowPrimarySurfaceReplacement = p.AllowPrimarySurfaceReplacement;
    ui.EnableBloom = p.EnableBloom;
    ui.EnableLDSamplerForBSDF = p.EnableLDSamplerForBSDF;
    ui.RealtimeFireflyFilterThreshold = p.FireflyThreshold;
    if (ommBaker)
        ommBaker->UIData().Force2State = p.OmmForce2State;
    ui.ResetAccumulation = true;
}

void ApplyPerformancePreset(SampleUIData& ui, uint presetIndex, class OmmBaker* ommBaker)
{
    presetIndex = dm::clamp(presetIndex, 0u, (uint)_countof(s_performancePresets));
    ApplyPreset(ui, s_performancePresets[presetIndex], ommBaker);
}

SampleUI::SampleUI(DeviceManager* deviceManager, SampleBaseApp & baseApp, Sample& app, SampleUIData& ui, bool NVAPI_SERSupported, const CommandLineOptions& cmdLine)
        : ImGui_Renderer(deviceManager)
        , m_baseApp(baseApp)
        , m_app(app)
        , m_ui(ui)
        , m_NVAPI_SERSupported(/*NVAPI_SERSupported*/false)
{
    m_commandList = GetDevice()->createCommandList();

    auto nativeFS = std::make_shared<donut::vfs::NativeFileSystem>(); // *(app.GetRootFs())

    // // auto fontPath = GetLocalPath(c_AssetsFolder) / "fonts/OpenSans/OpenSans-Regular.ttf";
    auto fontPath = GetLocalPath(c_AssetsFolder) / "fonts/DroidSans/DroidSans-Mono.ttf";
    // float baseFontSize = 15.0f;
    // 
    m_defaultFont = this->CreateFontFromFile(*nativeFS, GetLocalPath(c_AssetsFolder) / "fonts/DroidSans/DroidSans-Mono.ttf", 16.0f);

    ImGui::GetIO().IniFilename = nullptr;

    // Choose which, if any, hit object extension we can use
#if RTXPT_D3D_AGILITY_SDK_VERSION >= 619
    m_ui.DXHitObjectExtension = (GetDevice()->getGraphicsAPI() == nvrhi::GraphicsAPI::D3D12) && !cmdLine.disableSER;   // 6.9 support guarantees SER; --noSER falls back to TraceRay
    m_ui.NVAPIHitObjectExtension &= !m_ui.DXHitObjectExtension && NVAPI_SERSupported;
#else
    m_ui.NVAPIHitObjectExtension &= NVAPI_SERSupported;  // no need to check for or attempt using HitObjectExtension if SER not supported
#endif


#if ENABLE_DEBUG_DELTA_TREE_VIZUALISATION
    m_ImNodesContext = ImNodes::Ez::CreateContext();
#endif

    m_ui.RelaxSettings = NrdConfig::getDefaultRELAXSettings();
    m_ui.ReblurSettings = NrdConfig::getDefaultREBLURSettings();

    m_ui.TemporalAntiAliasingParams.useHistoryClampRelax = true;

    m_ui.ToneMappingParams.toneMapOperator = ToneMapperOperator::HableUc2;

    // enable by default for now
    m_ui.RTXDI.regir.regirStaticParams.Mode = rtxdi::ReGIRMode::Grid;

    // load core settings
    m_ui.UseNEE                     = cmdLine.UseNEE != 0;
    m_ui.NEEType                    = cmdLine.NEEType;
    m_ui.UseReSTIRDI                = cmdLine.UseReSTIRDI != 0;
    m_ui.UseReSTIRGI                = cmdLine.UseReSTIRGI != 0;
    m_ui.RealtimeSamplesPerPixel    = cmdLine.RealtimeSamplesPerPixel;
    m_ui.AccumulationTarget         = cmdLine.ReferenceSamplesPerPixel;
    m_ui.StandaloneDenoiser         = cmdLine.StandaloneDenoiser != 0;
    m_ui.RealtimeAA                 = cmdLine.RealtimeAA;
    m_ui.RealtimeSubSamplingPattern = dm::clamp(cmdLine.SubSamplingPattern, 0, (int)SubSamplingPattern::Count - 1);
    m_ui.RealtimeSubSamplingStratifiedBlockSize = cmdLine.SubSamplingBlockSize;

    m_ui.EnableBloom &= !cmdLine.DisablePostProcessFilters;
}

SampleUI::~SampleUI()
{
#if ENABLE_DEBUG_DELTA_TREE_VIZUALISATION
    ImNodes::Ez::FreeContext(m_ImNodesContext);
#endif
}

bool SampleUI::MousePosUpdate(double xpos, double ypos)
{
    return ImGui_Renderer::MousePosUpdate(xpos, ypos);
}

std::string TrimTogglable(const std::string text)
{
    size_t tog = text.rfind("_togglable");
    if (tog != std::string::npos)
        return text.substr(0, tog);
    return text;
}
std::string TrimSkyDisplayName(std::string text)
{
    if (text == c_EnvMapSceneDefault)
        return "default";
    else if (text == c_EnvMapProcSky)
        return "procedural";
    else if (text == c_EnvMapProcSky_Morning)
        return "morning";
    else if (text == c_EnvMapProcSky_Midday)
        return "midday";
    else if (text == c_EnvMapProcSky_Evening)
        return "evening";
    else if (text == c_EnvMapProcSky_Dawn)
        return "dawn";
    else if (text == c_EnvMapProcSky_PitchBlack)
        return "pitch black";
    return "unknown";
}

void SampleUI::Animate(float elapsedTimeSeconds)
{
    donut::app::ImGui_Renderer::Animate(elapsedTimeSeconds);

    int w, h;
    GetDeviceManager()->GetWindowDimensions(w, h);
    ImGuiIO& io = ImGui::GetIO();

    m_showSceneWidgets = dm::clamp(m_showSceneWidgets + elapsedTimeSeconds * 8.0f * ((io.MousePos.y >= 0 && io.MousePos.y < h * 0.1f) ? (1) : (-1)), 0.0f, 1.0f);
}

#if DONUT_WITH_STREAMLINE
SI::DLSSMode SampleUI::DLSSModeUI(SI::DLSSMode dlssModeCurrent)
{
    int current = -1;
    switch (dlssModeCurrent)
    {
    case donut::app::StreamlineInterface::DLSSMode::eMaxPerformance:    current = 1; break;
    case donut::app::StreamlineInterface::DLSSMode::eBalanced:          current = 2; break;
    case donut::app::StreamlineInterface::DLSSMode::eMaxQuality:        current = 3; break;
    case donut::app::StreamlineInterface::DLSSMode::eUltraPerformance:  current = 0; break;
    case donut::app::StreamlineInterface::DLSSMode::eDLAA:              current = 4; break;
    default: assert(false); return donut::app::StreamlineInterface::DLSSMode::eBalanced;
    }

    ImGui::Combo("DLSS Resolution Scale", (int*)&current, "UltraPerformance\0Performance\0Balanced\0Quality\0DLAA\0");

    switch (current)
    {
    case 0 : return donut::app::StreamlineInterface::DLSSMode::eUltraPerformance;
    case 1 : return donut::app::StreamlineInterface::DLSSMode::eMaxPerformance;
    case 2 : return donut::app::StreamlineInterface::DLSSMode::eBalanced;
    case 3 : return donut::app::StreamlineInterface::DLSSMode::eMaxQuality;
    case 4 : return donut::app::StreamlineInterface::DLSSMode::eDLAA;
    default: assert(false); return donut::app::StreamlineInterface::DLSSMode::eBalanced;

    }
    ImGui::Text("(DLSS setting also apply to Ray Reconstruction)");
}
#endif

bool SampleUI::BuildUIScriptsAndEtc(void)
{
    bool scriptsActive = false;
    if (m_app.GetCaptureScriptManager()->ScriptProgressUI())
        scriptsActive = true;

    if (scriptsActive)
        ImGui::Text("=================================================");

    return scriptsActive;
}

void SampleUI::BuildUIResolutionPicker()
{
    if (!ImGui::BeginPopupModal("Resolution Picker", nullptr, ImGuiWindowFlags_AlwaysAutoResize))
        return;

    struct Resolution { int w, h; const char* label; };
    static const Resolution standardResolutions[] = {
        { 1280,  720, "1280 x 720   (16:9)" },
        { 1280,  800, "1280 x 800   (16:10)" },
        { 1366,  768, "1366 x 768   (16:9)" },
        { 1440,  900, "1440 x 900   (16:10)" },
        { 1600,  900, "1600 x 900   (16:9)" },
        { 1680, 1050, "1680 x 1050  (16:10)" },
        { 1920, 1080, "1920 x 1080  (16:9)" },
        { 1920, 1200, "1920 x 1200  (16:10)" },
        { 2560, 1440, "2560 x 1440  (16:9)" },
        { 2560, 1600, "2560 x 1600  (16:10)" },
        { 3840, 2160, "3840 x 2160  (16:9)" },
        { 3840, 2400, "3840 x 2400  (16:10)" },
    };

    const GLFWvidmode* monitorMode = glfwGetVideoMode(glfwGetPrimaryMonitor());
    int currentW = (int)m_app.GetDisplaySize().x;
    int currentH = (int)m_app.GetDisplaySize().y;

    ImGui::Text("Click to change resolution:");
    ImGui::Separator();

    for (const auto& res : standardResolutions)
    {
        if (res.w > monitorMode->width || res.h > monitorMode->height)
            continue;

        bool isCurrent = (res.w == currentW && res.h == currentH);
        std::string displayLabel = std::string(res.label) + (isCurrent ? "  [current]" : "");
        if (ImGui::Selectable(displayLabel.c_str(), isCurrent))
        {
            glfwSetWindowSize(GetDeviceManager()->GetWindow(), res.w, res.h);
            ImGui::CloseCurrentPopup();
        }
    }

    ImGui::Separator();
    if (ImGui::Button("Cancel", { -1, 0 }) || ImGui::IsKeyPressed(ImGuiKey_Escape))
        ImGui::CloseCurrentPopup();

    ImGui::EndPopup();
}

void SampleUI::BuildUIPerformancePresets()
{
    if (!ImGui::BeginPopupModal("Performance Preset", nullptr, ImGuiWindowFlags_AlwaysAutoResize))
        return;

    ImGui::Text("Click to select a performance preset:");
    ImGui::Separator();

    OmmBaker* ommBaker = (m_app.GetOMMBaker()!=nullptr)?(m_app.GetOMMBaker().get()):(nullptr);
    for (const auto& preset : s_performancePresets)
    {
        bool isCurrent = MatchesPreset(m_ui, preset, ommBaker);
        std::string displayLabel = std::string(preset.Name) + (isCurrent ? "  [current]" : "");
        if (ImGui::Selectable(displayLabel.c_str(), isCurrent))
        {
            ApplyPreset(m_ui, preset, ommBaker);
            ImGui::CloseCurrentPopup();
        }
    }

    ImGui::Separator();
    if (ImGui::Button("Cancel", { -1, 0 }) || ImGui::IsKeyPressed(ImGuiKey_Escape))
        ImGui::CloseCurrentPopup();

    ImGui::EndPopup();
}

void SampleUI::DLSSFGSelectorUI()
{
    const char* items[] = { "Off", "2x", "3x", "4x" };
    const int itemCount = IM_ARRAYSIZE(items);

    static int currentItem = 0;
    if (ImGui::BeginCombo("Frame Generation", items[currentItem]))
    {
        for (int itemId = 0; itemId < itemCount; itemId++)
        {
#if DONUT_WITH_STREAMLINE
            UI_SCOPED_DISABLE(itemId > m_ui.DLSSFGMaxNumFramesToGenerate);
#endif
            bool isSelected = (currentItem == itemId);
            if (ImGui::Selectable(items[itemId], isSelected))
                currentItem = itemId;
            if (isSelected)
                ImGui::SetItemDefaultFocus();
        }
        ImGui::EndCombo();
    }

#if DONUT_WITH_STREAMLINE
    m_ui.DLSSFGMode = (currentItem > 0)
        ? donut::app::StreamlineInterface::DLSSGMode::eOn
        : donut::app::StreamlineInterface::DLSSGMode::eOff;

    m_ui.DLSSFGNumFramesToGenerate = (m_ui.DLSSFGMode == donut::app::StreamlineInterface::DLSSGMode::eOn) ? currentItem : 1;
#endif
    if (!m_ui.RealtimeMode)
        ImGui::TextColored(warnColor, "Note: DLSS-G is DISABLED in Reference PT mode");
};



void SampleUI::buildUI(void)
{
    if (!m_ui.ShowUI)
        return;

    RAII_SCOPE( ImGui::PushFont(m_defaultFont->GetScaledFont());, ImGui::PopFont(); );

    // Ideally we'd want to rework UI scaling so that it is not based on m_currentScale but on ImGui::GetFontSize() so we can freely change fonts
    auto& io = ImGui::GetIO();
    float scaledWidth = io.DisplaySize.x; 
    float scaledHeight = io.DisplaySize.y;

    const float defWindowWidth = 335.0f * m_currentScale;
    const float defItemWidth = defWindowWidth * 0.3f * m_currentScale;

    auto imGuiCheckboxUInt32 = [ & ](const char* label, uint32_t* v)
    {
        bool pv = (*v) != 0;
        bool ret = ImGui::Checkbox(label, &pv);
        *v = pv ? (1) : (0);
        return ret;
    };

    {
        ImGui::SetNextWindowPos(ImVec2(10.f, 10.f), ImGuiCond_Appearing);
        ImGui::SetNextWindowSize(ImVec2(defWindowWidth, scaledHeight - 20), ImGuiCond_Appearing);

        RAII_SCOPE( ImGui::Begin("Settings", 0, ImGuiWindowFlags_None /*AlwaysAutoResize*/); , ImGui::End(); );
        RAII_SCOPE( ImGui::PushItemWidth(defItemWidth); , ImGui::PopItemWidth(); );
            
        const float indent = (int)ImGui::GetStyle().IndentSpacing*0.4f;

        ImGui::Text("%s, %s", GetDeviceManager()->GetRendererString(), m_app.GetResolutionInfo().c_str() );
        ImGui::Text(m_app.GetFPSInfo().c_str());

        if (BuildUIScriptsAndEtc())
        {
            return;
        }

        if (ImGui::CollapsingHeader("Display and performance")) //, ImGuiTreeNodeFlags_DefaultOpen))
        {
            RAII_SCOPE(ImGui::Indent(indent);, ImGui::Unindent(indent); );
            
            {
                if (ImGui::Button(StringFormat("Resolution:  %dx%d (click to change)", m_app.GetDisplaySize().x, m_app.GetDisplaySize().y, m_app.GetRenderSize().x, m_app.GetRenderSize().y).c_str(), { -1, 0 }))
                    ImGui::OpenPopup("Resolution Picker");
                BuildUIResolutionPicker();
            }

            {
                const char* currentPresetName = "Custom";
                OmmBaker* ommBaker = (m_app.GetOMMBaker()!=nullptr)?(m_app.GetOMMBaker().get()):(nullptr);
                for (const auto& preset : s_performancePresets)
                    if (MatchesPreset(m_ui, preset, ommBaker)) { currentPresetName = preset.Name; break; }
                if (ImGui::Button(StringFormat("Perf. preset: %s (click to change)", currentPresetName).c_str(), { -1, 0 }))
                    ImGui::OpenPopup("Performance Preset");
                BuildUIPerformancePresets();
            }

            {
                RAII_SCOPE(ImGui::PushID("DispAndPerf"); , ImGui::PopID(); );
                DLSSFGSelectorUI();
            }

            {
#if DONUT_WITH_STREAMLINE
                UI_SCOPED_DISABLE(m_ui.ActualDLSSFGMode() != SI::DLSSGMode::eOff);
#endif
                ImGui::Checkbox("VSync", &m_ui.EnableVsync);
                bool fpsLimiter = m_ui.FPSLimiter != 0;
                ImGui::SameLine();
                ImGui::SeparatorEx(ImGuiSeparatorFlags_Vertical);
                ImGui::SameLine();
                ImGui::Text("Cap fps to ");
                ImGui::SameLine();
                std::array<int, 8> fpsOptions{ 0, /*1,*/ 2, 5, 10, 15, 30, 60, 120 }; auto curr = std::find(fpsOptions.begin(), fpsOptions.end(), m_ui.FPSLimiter);
                int fpsLimitIndex = (curr != fpsOptions.end()) ? (int(curr - fpsOptions.begin())) : (0);
                if (ImGui::Combo("##FPSLIMITER", &fpsLimitIndex, "disabled\0" /* " 1 \0" */ " 2 \0 5 \0 10 \0 15 \0 30 \0 60 \0 120 \0\0"))
                    m_ui.FPSLimiter = fpsOptions[dm::clamp(fpsLimitIndex, 0, (int)fpsOptions.size() - 1)];
            }

            {
                UI_SCOPED_DISABLE(!m_app.IsAsyncPrefixAvailable());
                ImGui::Checkbox("AsyncPrefix", &m_ui.EnableAsyncPrefix);
                if (ImGui::IsItemHovered())
                    ImGui::SetTooltip("Steady-state DLSS-RR frames: everything up to the DLSS-RR inputs on the compute queue,\noverlapping the previous frame's DLSS-RR and post-processing. Off: everything on the graphics queue.");
            }
        }

        if (ImGui::CollapsingHeader("System")) //, ImGuiTreeNodeFlags_DefaultOpen))
        {
            RAII_SCOPE(ImGui::Indent(indent); , ImGui::Unindent(indent); );
            if (ImGui::Button("Reload Shaders (requires VS .hlsl->.bin build)"))
                m_ui.ShaderReloadRequested = true;

            ImGui::Checkbox("Render when out of focus", &m_ui.RenderWhenOutOfFocus);
            if (ImGui::IsItemHovered()) 
                ImGui::SetTooltip("Render loop will pause when app window is out of focus. Note: Reference mode will accumulate until all frames are done.");
        
        
            {
                //RAII_SCOPE(ImGui::Indent(indent); , ImGui::Unindent(indent););

                if (ImGui::CollapsingHeader("Capture scripts and tools"))
                {
                    RAII_SCOPE(ImGui::Indent(indent); , ImGui::Unindent(indent); );

                    m_app.GetCaptureScriptManager()->ScriptMainUI(warnColor, categoryColor, indent, m_currentScale);
                }
            }

            if (ImGui::CollapsingHeader("Info")) //, ImGuiTreeNodeFlags_DefaultOpen))
            {
                uint64_t budget, currentUsage, availableForReservation, currentReservation;
                if (m_baseApp.QueryVideoMemoryInfo( budget, currentUsage, availableForReservation, currentReservation ) )
                {
                    ImGui::TextColored(categoryColor, "QueryVideoMemoryInfo:");
                    budget /= 1024*1024; currentUsage /= 1024*1024, availableForReservation /= 1024*1024, currentReservation /= 1024*1024;
                    RAII_SCOPE(ImGui::Indent(indent); , ImGui::Unindent(indent););
                    ImGui::Text("Budget:             %7" PRIu64 "MB", budget );
                    ImGui::Text("CurrentUsage:       %7" PRIu64 "MB", currentUsage);
                    ImGui::Text("AvailableForRes.:   %7" PRIu64 "MB", availableForReservation);
                    ImGui::Text("CurrentReservation: %7" PRIu64 "MB", currentReservation);
                }
                ImGui::TextColored(categoryColor, "Timings:");
                RAII_SCOPE(ImGui::Indent(indent); , ImGui::Unindent(indent););
                ImGui::Text("Avg render time:   %.3f ms", m_app.GetAvgTimeInRender() * 1000.0);
                ImGui::Text("Avg present time:  %.3f ms", m_app.GetAvgTimeInPresent() * 1000.0);
                ImGui::Text("Avg frame gap:     %.3f ms", m_app.GetAvgTimeInFrameGap() * 1000.0);
                ImGui::TextWrapped("Note: when GPU limited (should be most common case), majority of CPU time should in 'present' or 'frame gap'. If 'render time' is larger than the other two, there's probably a sync bug.");
            }
        }

        {
            const std::string currentScene = m_app.GetCurrentSceneName();
            RAII_SCOPE(ImGui::PushItemWidth(-60.0f * m_currentScale); , ImGui::PopItemWidth(); );
            RAII_SCOPE(ImGui::PushID("SceneComboID"); , ImGui::PopID(); );
            if (ImGui::BeginCombo("Scene", currentScene.c_str()))
            {
                const std::vector<std::string>& scenes = m_app.GetAvailableScenes();
                for (const std::string& scene : scenes)
                {
                    bool is_selected = scene == currentScene;
                    if (ImGui::Selectable(scene.c_str(), is_selected))
                        m_app.SetCurrentScene(scene);
                    if (is_selected)
                        ImGui::SetItemDefaultFocus();
                }
                ImGui::EndCombo();
            }
        }

        if (ImGui::CollapsingHeader("Scene"/*, ImGuiTreeNodeFlags_DefaultOpen*/))
        {
            RAII_SCOPE(ImGui::Indent(indent); , ImGui::Unindent(indent); );
            uint uncompressedTextureCount = (uint)m_app.GetUncompressedTextures().size();
            if (uncompressedTextureCount > 0)
            {
                ImGui::TextColored(warnColor, "Scene has %d uncompressed textures", uncompressedTextureCount);
                if (ImGui::Button("Batch compress with nvtt_export.exe", { -1, 0 }))
                    if (CompressTextures(m_app.GetUncompressedTextures()))
                    {   // reload scene
                        m_app.SetCurrentScene(m_app.GetCurrentSceneName(), true);
                    }
            }

            {
                UI_SCOPED_DISABLE(!m_ui.RealtimeMode);
                ImGui::Checkbox("Enable animations", &m_ui.EnableAnimations);
                if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) ImGui::SetTooltip("Animations are not available in reference mode");
            }
            ImGui::SameLine();
            if (ImGui::Button("Reset animation time"))
            {
                m_app.SetSceneTime(0);
                m_ui.ResetAccumulation = true;
            }

            if (m_app.GetGame() && m_app.GetGame()->IsInitialized())
            {
                if (ImGui::CollapsingHeader("Interactive elements"/*, ImGuiTreeNodeFlags_DefaultOpen*/))
                {
                    RAII_SCOPE(ImGui::Indent(indent);, ImGui::Unindent(indent); );
                    m_app.GetGame()->DebugGUI(indent);
                }
            }

            if (m_ui.TogglableNodes != nullptr && m_ui.TogglableNodes->size() > 0 && ImGui::CollapsingHeader("Togglables"))
            {
                for (int i = 0; i < m_ui.TogglableNodes->size(); i++)
                {
                    auto& node = (*m_ui.TogglableNodes)[i];
                    bool selected = node.IsSelected();
                    if (ImGui::Checkbox(node.UIName.c_str(), &selected))
                    {
                        node.SetSelected(selected);
                        m_ui.ResetAccumulation = true;
                    }
                }
            }

            if (ImGui::CollapsingHeader("Environment Map"))
            {
                RAII_SCOPE(ImGui::Indent(indent); , ImGui::Unindent(indent); );

                RESET_ON_CHANGE(ImGui::Checkbox("Enabled", &m_ui.EnvironmentMapParams.Enabled));

                if (m_app.GetEnvMapLocalPath() != "==PROCEDURAL_SKY==")
                    ImGui::TextWrapped("Source: `%s`", m_app.GetEnvMapLocalPath().c_str());
                else
                    ImGui::TextWrapped("Source: Procedural Sky");

                std::string overrideSource = m_app.GetEnvMapOverrideSource();
                const std::vector<std::filesystem::path> & envMapMediaList = m_app.GetEnvMapMediaList();

                RAII_SCOPE( ImGui::PushItemWidth(-65.0f*m_currentScale);, ImGui::PopItemWidth(); );
                if (ImGui::BeginCombo("Override", overrideSource.c_str()))
                {
                    for (int i = -7; i < (int)envMapMediaList.size(); i++)
                    {
                        std::string itemName;
                        if (i == -7)
                            itemName = c_EnvMapSceneDefault;
                        else if (i == -6)
                            itemName = c_EnvMapProcSky;
                        else if (i == -5)
                            itemName = c_EnvMapProcSky_Morning;
                        else if (i == -4)
                            itemName = c_EnvMapProcSky_Midday;
                        else if (i == -3)
                            itemName = c_EnvMapProcSky_Evening;
                        else if (i == -2)
                            itemName = c_EnvMapProcSky_Dawn;
                        else if (i == -1)
                            itemName = c_EnvMapProcSky_PitchBlack;
                        else
                            itemName = envMapMediaList[i].filename().string();

                        bool is_selected = itemName == overrideSource;
                        if (ImGui::Selectable(itemName.c_str(), is_selected))
                            overrideSource = itemName;
                        if (is_selected)
                            ImGui::SetItemDefaultFocus();
                    }
                    ImGui::EndCombo();
                }
                if (ImGui::IsItemHovered()) ImGui::SetTooltip("Overrides scene's default environment map");
                if (m_app.GetEnvMapOverrideSource() != overrideSource)
                {
                    m_ui.ResetAccumulation = true;
                    m_app.SetEnvMapOverrideSource(overrideSource);
                }

                ImGui::Separator();
                RESET_ON_CHANGE( ImGui::InputFloat3("Tint Color", (float*)&m_ui.EnvironmentMapParams.TintColor.x) );
                RESET_ON_CHANGE( ImGui::InputFloat("Intensity", &m_ui.EnvironmentMapParams.Intensity) );
                RESET_ON_CHANGE( ImGui::InputFloat3("Rotation XYZ", (float*)&m_ui.EnvironmentMapParams.RotationXYZ.x) );
                ImGui::Separator();

                if (m_app.GetEnvMapBaker() != nullptr && m_app.GetEnvMapBaker()->IsProcedural() && m_app.GetEnvMapBaker()->GetProceduralSky() != nullptr) // one frame delay for these settings
                {
                    ImGui::TextColored(categoryColor, "Procedural Sky settings:");
                    RAII_SCOPE(ImGui::Indent(indent); , ImGui::Unindent(indent););
                    m_app.GetEnvMapBaker()->GetProceduralSky()->DebugGUI(indent);
                }
            }

            if (ImGui::CollapsingHeader("Materials"))
            {
                RAII_SCOPE( ImGui::Indent(indent);, ImGui::Unindent(indent); );
                if ( m_app.GetMaterialsBaker() != nullptr )
                    m_app.GetMaterialsBaker()->DebugGUI(indent);
            }
        }

        DrawSampleUI(indent, defItemWidth, warnColor, categoryColor);

        if (m_ui.RealtimeMode && m_ui.RealtimeAA > 1 && ImGui::CollapsingHeader("DLSS & Reflex settings"))
        {
            ImGui::TextColored(categoryColor, "Anti-aliasing and super-resolution");
            {
                RAII_SCOPE(ImGui::Indent(indent); , ImGui::Unindent(indent); );

    #if DONUT_WITH_STREAMLINE
                if (m_ui.RealtimeAA == 2 || m_ui.RealtimeAA == 3)
                    m_ui.DLSSMode = DLSSModeUI(m_ui.DLSSMode);
    
                if (m_ui.RealtimeAA == 3)
                {
                    ImGui::SliderFloat("DLSS-RR micro jitter", &m_ui.DLSSRRMicroJitter, 0.0f, 1.0f);
					int presetIndex = 0;
					switch (m_ui.DLSRRPreset) {
						case SI::DLSSRRPreset::eDefault: presetIndex = 0; break;
						case SI::DLSSRRPreset::ePresetD: presetIndex = 1; break;
						case SI::DLSSRRPreset::ePresetE: presetIndex = 2; break;
						case SI::DLSSRRPreset::ePresetF: presetIndex = 3; break;
					}
                    ImGui::Combo("DLSS-RR Preset", &presetIndex, "Default\0PresetD\0PresetE\0PresetF\0");
					const SI::DLSSRRPreset DLSSRR_PRESETS[] = { SI::DLSSRRPreset::eDefault, SI::DLSSRRPreset::ePresetD, SI::DLSSRRPreset::ePresetE, SI::DLSSRRPreset::ePresetF };  // Maps combo index to enum value
					m_ui.DLSRRPreset = DLSSRR_PRESETS[presetIndex];
					m_ui.DLSRRPreset = clamp(m_ui.DLSRRPreset, SI::DLSSRRPreset::eDefault, SI::DLSSRRPreset::ePresetF);
                }
    #endif    
                ImGui::Combo("AA Camera Jitter", (int*)&m_ui.TemporalAntiAliasingJitter, "MSAA\0Halton\0R2\0White Noise\0");

            }

            RAII_SCOPE(ImGui::Indent(indent);, ImGui::Unindent(indent); );

            if (ImGui::CollapsingHeader("Reflex", 0))
            {
    #if DONUT_WITH_STREAMLINE
                ImGui::Text("Reflex LowLatency Supported: %s", m_ui.IsReflexSupported && m_ui.IsReflexLowLatencyAvailable ? "yes" : "no");
                if (m_ui.IsReflexSupported && m_ui.IsReflexLowLatencyAvailable)
                {
                    ImGui::Combo("Reflex Low Latency", (int*)&m_ui.ReflexMode, "Off\0On\0On + Boost\0");

                    bool useFrameCap = m_ui.ReflexCappedFps != 0;
                    if (ImGui::Checkbox("Reflex FPS Capping", &useFrameCap))
                    {
                        if (useFrameCap) { m_ui.FpsCap = 0; }
                    }
                    else if (m_ui.FpsCap != 0)
                    {
                        useFrameCap = false;
                        m_ui.ReflexCappedFps = 0;
                    }

                    if (useFrameCap)
                    {
                        if (m_ui.ReflexCappedFps == 0) { m_ui.ReflexCappedFps = 60; }
                        ImGui::SameLine();
                        ImGui::DragInt("##FPSReflexCap", &m_ui.ReflexCappedFps, 1.f, 20, 240);
                        m_ui.FpsCap = 0;
                    }
                    else
                    {
                        m_ui.ReflexCappedFps = 0;
                    }

                    ImGui::Checkbox("Show Stats Report", &m_ui.ReflexShowStats);
                    if (m_ui.ReflexShowStats)
                    {
                        RAII_SCOPE(ImGui::Indent(indent);, ImGui::Unindent(indent); );
                        ImGui::Text(m_ui.ReflexStats.c_str());
                    }

                    if (!m_ui.RealtimeMode)
                        ImGui::TextColored(warnColor, "Note: Reflex is DISABLED in Reference PT mode");
                }
    #else
                ImGui::Text("Compiled without REFLEX enabled");
    #endif
            }

            if (ImGui::CollapsingHeader("DLSS-G", 0))
            {
    #if DONUT_WITH_STREAMLINE
                ImGui::Text("DLSS-G Supported: %s", m_ui.IsDLSSFGSupported ? "yes" : "no");
                if (m_ui.IsDLSSFGSupported)
                {

                    if (m_ui.ReflexMode == donut::app::StreamlineInterface::ReflexMode::eOff)
                        ImGui::Text("Please note: Reflex is currently off and will be\nautomatically enabled if DLSS-FG is enabled");

                    DLSSFGSelectorUI();
                }
    #else
                ImGui::Text("Compiled without DLSS-G enabled");
    #endif
            }
        }

        if( m_ui.RealtimeMode && m_ui.RealtimeAA == 1 && ImGui::CollapsingHeader("TAA settings") )
        {
            ImGui::Checkbox("TAA History Clamping", &m_ui.TemporalAntiAliasingParams.enableHistoryClamping);
            ImGui::SliderFloat("TAA New Frame Weight", &m_ui.TemporalAntiAliasingParams.newFrameWeight, 0.001f, 1.0f);
            ImGui::Checkbox("TAA Use Clamp Relax", &m_ui.TemporalAntiAliasingParams.useHistoryClampRelax);
            ImGui::Combo("AA Camera Jitter", (int*)&m_ui.TemporalAntiAliasingJitter, "MSAA\0Halton\0R2\0White Noise\0");
        }

        if (ImGui::CollapsingHeader("Opacity Micro-Maps"))
        {
            UI_SCOPED_INDENT(indent);

            if (m_app.GetOMMBaker())
            {
                m_app.GetOMMBaker()->DebugGUI(indent, *m_app.GetScene());
            }
            else
                ImGui::Text("<Opacity Micro-Maps not supported on the current device>");
        }

        if (ImGui::CollapsingHeader("Acceleration Structure"))
        {
            UI_SCOPED_INDENT(indent);

            {
                if (ImGui::Checkbox("Force Opaque", &m_ui.AS.ForceOpaque))
                {
                    m_ui.ResetAccumulation = true;
                }

                if (ImGui::IsItemHovered())
                    ImGui::SetTooltip("Will set the instance flag ForceOpaque on all instances");
            }

            {
                ImGui::Checkbox("GPU TLAS build", &m_ui.AS.GPUTLASBuild);
                if (ImGui::IsItemHovered())
                    ImGui::SetTooltip(
                        "Build the TLAS instance descriptors in a compute pass instead of the CPU loop.\n"
                        "Off falls back to the multithreaded CPU builder, for A/B comparison.");

                UI_SCOPED_DISABLE(!m_ui.AS.GPUTLASBuild);
                UI_SCOPED_INDENT(indent);

                if (ImGui::Checkbox("Frustum + distance cull", &m_ui.AS.GPUTLASCull))
                    m_ui.ResetAccumulation = true;
                if (ImGui::IsItemHovered())
                    ImGui::SetTooltip(
                        "Drop instances outside a loose camera frustum and beyond a distance cutoff.\n"
                        "\n"
                        "Rays are not bounded by the view frustum, so this removes off-screen shadow\n"
                        "casters and reflection sources, and the retained set changes as the camera\n"
                        "turns. Ignored outside realtime mode - reference accumulation always builds\n"
                        "the full TLAS.");

                UI_SCOPED_DISABLE(!m_ui.AS.GPUTLASCull || !m_ui.RealtimeMode);

                if (ImGui::SliderFloat("Frustum margin (m)", &m_ui.AS.GPUTLASCullFrustumMargin, 0.0f, 2000.0f, "%.0f"))
                    m_ui.ResetAccumulation = true;
                if (ImGui::IsItemHovered())
                    ImGui::SetTooltip("Metres the side planes are pushed outward. This skirt is what keeps off-screen shadow casters and reflection sources in the TLAS.");

                if (ImGui::SliderFloat("Distance cutoff (m)", &m_ui.AS.GPUTLASCullDistanceCutoff, 10.0f, 100000.0f, "%.0f", ImGuiSliderFlags_Logarithmic))
                    m_ui.ResetAccumulation = true;
                if (ImGui::IsItemHovered())
                    ImGui::SetTooltip("Metres beyond which an instance is culled regardless of the frustum. Depends only on camera position, so unlike the margin it is invariant under pure rotation.");
            }

            ImGui::Separator();
            ImGui::Text("Settings below require AS rebuild");

            {
                if (ImGui::Checkbox("Exclude Transmissive", &m_ui.AS.ExcludeTransmissive))
                {
                    m_ui.AccelerationStructRebuildRequested = true;
                }

                if (ImGui::IsItemHovered())
                    ImGui::SetTooltip("Will exclude all transmissive geometries from the BVH");
            }
        }

        if (ImGui::CollapsingHeader("Post-process"))
        {
            RAII_SCOPE(ImGui::Indent(indent);, ImGui::Unindent(indent); );

            if (ImGui::CollapsingHeader("Early (HDR) post-process"))
            {
                RAII_SCOPE(ImGui::Indent(indent); , ImGui::Unindent(indent); );
                ImGui::Checkbox("PostProcessTestPass", &m_ui.PostProcessTestPassHDR );
                
                ImGui::Separator();

                if (ImGui::CollapsingHeader("Bloom"))
                {
                    ImGui::Checkbox("Enable Bloom", &m_ui.EnableBloom);
                    ImGui::SliderFloat("Bloom Width (Pixels)", &m_ui.BloomRadius, 0.f, 64.f);
                    ImGui::SliderFloat("Bloom Intensity", &m_ui.BloomIntensity, 0.f, 0.1f);
                }
            }

            if (ImGui::CollapsingHeader("Tone Mapping"))
            {
                RAII_SCOPE(ImGui::Indent(indent); , ImGui::Unindent(indent); );
                ImGui::Checkbox("Enable", &m_ui.EnableToneMapping);

                const std::string currentOperator = tonemapOperatorToString.at(m_ui.ToneMappingParams.toneMapOperator);
                if (ImGui::BeginCombo("Operator", currentOperator.c_str()))
                {
                    for (auto it = tonemapOperatorToString.begin(); it != tonemapOperatorToString.end(); it++)
                    {
                        bool is_selected = it->first == m_ui.ToneMappingParams.toneMapOperator;
                        if (ImGui::Selectable(it->second.c_str(), is_selected))
                            m_ui.ToneMappingParams.toneMapOperator = it->first;
                    }
                    ImGui::EndCombo();
                }

                ImGui::Checkbox("Auto Exposure", &m_ui.ToneMappingParams.autoExposure);

                if (m_ui.ToneMappingParams.autoExposure)
                {
                    ImGui::InputFloat("Auto Exposure Min", &m_ui.ToneMappingParams.exposureValueMin);
                    m_ui.ToneMappingParams.exposureValueMin = dm::min(m_ui.ToneMappingParams.exposureValueMax, m_ui.ToneMappingParams.exposureValueMin);
                    ImGui::InputFloat("Auto Exposure Max", &m_ui.ToneMappingParams.exposureValueMax);
                    m_ui.ToneMappingParams.exposureValueMax = dm::max(m_ui.ToneMappingParams.exposureValueMin, m_ui.ToneMappingParams.exposureValueMax);
                }

                if (const ToneMappingPass * toneMappingPass = m_app.GetToneMappingPass())
                {
                    const ToneMappingExposureInfo & info = toneMappingPass->GetExposureInfo(0);
                    ImGui::TextColored(info.Clamped ? warnColor : ImGui::GetStyleColorVec4(ImGuiCol_Text), "Exposure (log2): auto %+.2f, user %+.2f, total %+.2f%s",
                        info.AutoLog2, info.UserLog2, info.TotalLog2, info.Clamped ? " (clamped)" : "");
                    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Total exposure is clamped to [%+.0f, %+.0f]", TONEMAPPING_EXPOSURE_LOG2_MIN, TONEMAPPING_EXPOSURE_LOG2_MAX);
                }

                const std::string currentMode = ExposureModeToString.at(m_ui.ToneMappingParams.exposureMode);
                if (ImGui::BeginCombo("Exposure Mode", currentMode.c_str()))
                {
                    for (auto it = ExposureModeToString.begin(); it != ExposureModeToString.end(); it++)
                    {
                        bool is_selected = it->first == m_ui.ToneMappingParams.exposureMode;
                        if (ImGui::Selectable(it->second.c_str(), is_selected))
                            m_ui.ToneMappingParams.exposureMode = it->first;
                    }
                    ImGui::EndCombo();
                }

                ImGui::InputFloat("Exposure Compensation", &m_ui.ToneMappingParams.exposureCompensation);
                m_ui.ToneMappingParams.exposureCompensation = dm::clamp(m_ui.ToneMappingParams.exposureCompensation, -12.0f, 12.0f);

                ImGui::InputFloat("Exposure Value", &m_ui.ToneMappingParams.exposureValue);
                m_ui.ToneMappingParams.exposureValue = dm::clamp(m_ui.ToneMappingParams.exposureValue, dm::log2f(0.1f * 0.1f * 0.1f), dm::log2f(100000.f * 100.f * 100.f));

                ImGui::InputFloat("Film Speed", &m_ui.ToneMappingParams.filmSpeed);
                m_ui.ToneMappingParams.filmSpeed = dm::clamp(m_ui.ToneMappingParams.filmSpeed, 1.0f, 6400.0f);

                ImGui::InputFloat("fNumber", &m_ui.ToneMappingParams.fNumber);
                m_ui.ToneMappingParams.fNumber = dm::clamp(m_ui.ToneMappingParams.fNumber, 0.1f, 100.0f);

                ImGui::InputFloat("Shutter", &m_ui.ToneMappingParams.shutter);
                m_ui.ToneMappingParams.shutter = dm::clamp(m_ui.ToneMappingParams.shutter, 0.1f, 10000.0f);

                ImGui::Checkbox("Enable White Balance", &m_ui.ToneMappingParams.whiteBalance);

                ImGui::InputFloat("White Point", &m_ui.ToneMappingParams.whitePoint);
                m_ui.ToneMappingParams.whitePoint = dm::clamp(m_ui.ToneMappingParams.whitePoint, 1905.0f, 25000.0f);

                ImGui::InputFloat("White Max Luminance", &m_ui.ToneMappingParams.whiteMaxLuminance);
                m_ui.ToneMappingParams.whiteMaxLuminance = dm::clamp(m_ui.ToneMappingParams.whiteMaxLuminance, 0.1f, FLT_MAX);

                ImGui::InputFloat("White Scale", &m_ui.ToneMappingParams.whiteScale);
                m_ui.ToneMappingParams.whiteScale = dm::clamp(m_ui.ToneMappingParams.whiteScale, 0.f, 100.f);

                ImGui::Checkbox("Enable Clamp", &m_ui.ToneMappingParams.clamped);
            }

            if (ImGui::CollapsingHeader("Late (LDR) post-process"))
            {
                RAII_SCOPE(ImGui::Indent(indent); , ImGui::Unindent(indent); );

                ImGui::Checkbox("EdgeDetection", &m_ui.PostProcessEdgeDetection);
                ImGui::SliderFloat("EdgeDetectionThreshold", &m_ui.PostProcessEdgeDetectionThreshold, 0.0f, 1.0f );
                ImGui::Separator();
            }
        }

        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.8f, 0.5, 1.0f));
        bool debuggingIsOpen = ImGui::CollapsingHeader("Debugging"); //, ImGuiTreeNodeFlags_DefaultOpen ) )
        ImGui::PopStyleColor(1);
        if (debuggingIsOpen)
        {
            RAII_SCOPE(ImGui::Indent(indent);, ImGui::Unindent(indent); );

            if (ImGui::CollapsingHeader("Debug switches"))
            {
                RAII_SCOPE(ImGui::Indent(indent); , ImGui::Unindent(indent); );

                if (m_ui.RealtimeMode)
                {
                    ImGui::Checkbox("Freeze realtime noise seed", &m_ui.DbgFreezeRealtimeNoiseSeed);
                    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Freeze global noise seed will not change per frame. Useful for \ndebugging transient issues hidden by noise, or for before/after comparison");
                }
                ImGui::Checkbox("Disable SER path termination hint", &m_ui.DbgDisableSERTerminationHint);
                if (ImGui::IsItemHovered()) ImGui::SetTooltip("Disable SER ReorderThread getting receive additional hint about path termination.");

                ImGui::Checkbox("Discard path (non-NEE) lighting", &m_ui.DbgDiscardNonNEELighting);
                ImGui::Checkbox("Discard NEE lighting", &m_ui.DbgDiscardNEELighting);
            }


#if ENABLE_DEBUG_VIZUALISATIONS
            if (ImGui::Combo("Debug view", (int*)&m_ui.DebugView,
                "Disabled\0"
                "DominantDenoisingLayerIndex\0DenoisingLayer_VirtualRayLength\0DenoisingLayer_WorldMotionVectors\0"
                "DenoisingLayer_Normals\0DenoisingLayer_Roughness\0DenoisingLayer_SpecAvg\0DenoisingLayer_DiffBSDFEstimate\0DenoisingLayer_DiffRadiance\0DenoisingLayer_SpecBSDFEstimate\0DenoisingLayer_SpecRadiance\0"
                "DenoisingLayer_RelaxedDisocclusion\0DenoisingLayer_DiffRadianceDenoised\0DenoisingLayer_SpecRadianceDenoised\0DenoisingLayer_CombinedRadianceDenoised\0DenoisingLayer_ViewZ\0DenoisingLayer_Throughput\0DenoisingLayer_DenoiserValidation\0"
                "StableRadiance\0"
                "DenoiserGuide_Depth\0" "DenoiserGuide_Roughness\0" "DenoiserGuide_Albedo\0" "DenoiserGuide_SpecAlbedo\0"
                "DenoiserGuide_Normal\0" "DenoiserGuide_MotionVectors\0" "DenoiserGuide_SpecMotionVectors\0" "DenoiserGuide_SpecHitT\0" "DenoiserGuide_LayerWeights\0" "DenoiserGuide_PrimaryLayer\0"
                "FirstHit_Barycentrics\0FirstHit_FaceNormal\0FirstHit_GeometryNormal\0FirstHit_ShadingNormal\0FirstHit_ShadingTangent\0FirstHit_ShadingBitangent\0FirstHit_FrontFacing\0FirstHit_ThinSurface\0"
                "FirstHit_Diffuse\0FirstHit_Specular\0FirstHit_Roughness\0FirstHit_Metallic\0"
                "FirstHit_ShaderID\0FirstHit_MaterialID\0"
                "VBufferMotionVectors\0VBufferDepth\0"
                "SecondarySurfacePosition\0SecondarySurfaceRadiance\0ReSTIRGIOutput\0"
                "ReSTIRDIInitialOutput\0ReSTIRDITemporalOutput\0ReSTIRDISpatialOutput\0ReSTIRDIFinalOutput\0ReSTIRDIFinalContribution\0"
                "ReGIRIndirectOutput\0"
                "\0\0"))
                m_ui.ResetAccumulation = true;
            m_ui.DebugView = dm::clamp(m_ui.DebugView, (DebugViewType)0, DebugViewType::MaxCount);

            if (m_ui.DebugView >= DebugViewType::DenoisingLayer_VirtualRayLength && m_ui.DebugView <= DebugViewType::DenoisingLayer_DenoiserValidation)
            {
                m_ui.DebugViewDenoisingLayerIndex = dm::clamp(m_ui.DebugViewDenoisingLayerIndex, -1, (int)m_ui.ActualDenoisingLayersActiveCount() - 1);
                RAII_SCOPE(ImGui::Indent(indent);, ImGui::Unindent(indent); );
                float3 spcolor = (m_ui.DebugViewDenoisingLayerIndex >= 0) ? (DenoisingLayerDebugVizColor(m_ui.DebugViewDenoisingLayerIndex)) : (float3(1, 1, 0)); spcolor = spcolor * 0.7f + float3(0.2f, 0.2f, 0.2f);
                ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(spcolor.x, spcolor.y, spcolor.z, 1.0f));
                ImGui::InputInt("Denoising Layer index", &m_ui.DebugViewDenoisingLayerIndex);
                ImGui::PopStyleColor(1);
                m_ui.DebugViewDenoisingLayerIndex = dm::clamp(m_ui.DebugViewDenoisingLayerIndex, -1, (int)m_ui.ActualDenoisingLayersActiveCount() - 1);
            }

            const DebugFeedbackStruct& feedback = m_app.GetFeedbackData();
            if (ImGui::InputInt2("Debug pixel", (int*)&m_ui.DebugPixel.x))
                m_app.SetUIPick();

            ImGui::Checkbox("Continuous feedback", &m_ui.ContinuousDebugFeedback);

            ImGui::Checkbox("Show debug lines", &m_ui.ShowDebugLines);

            if (ImGui::Checkbox("Show material editor", &m_ui.ShowMaterialEditor) && m_ui.ShowMaterialEditor)
            {
#if ENABLE_DEBUG_DELTA_TREE_VIZUALISATION
                m_ui.ShowDeltaTree = false; // no space for both
#endif
                //m_app.SetUIPick();
            }

#if ENABLE_DEBUG_DELTA_TREE_VIZUALISATION
            if (!m_ui.ActualUseDenoisingLayers())
            {
                ImGui::Text("Enable Denoising Layers for delta tree viz!");
                m_ui.ShowDeltaTree = false;
            }
            else
            {
                if (ImGui::Checkbox("Show delta tree window", &m_ui.ShowDeltaTree) && m_ui.ShowDeltaTree)
                {
                    m_ui.ShowMaterialEditor = false; // no space for both
                    m_app.SetUIPick();
                }
            }
#else
            ImGui::Text("Delta tree debug viz disabled; to enable set ENABLE_DEBUG_DELTA_TREE_VIZUALISATION to 1");
#endif
            ImGui::Separator();

            for (int i = 0; i < MAX_DEBUG_PRINT_SLOTS; i++)
                ImGui::Text("debugPrint %d: %f, %f, %f, %f", i, feedback.debugPrint[i].x, feedback.debugPrint[i].y, feedback.debugPrint[i].z, feedback.debugPrint[i].w);
            ImGui::Text("Debug line count: %d", feedback.lineVertexCount / 2);
            ImGui::InputFloat("Debug Line Scale", &m_ui.DebugLineScale);
#else
            ImGui::TextWrapped("Debug visualization disabled; to enable set ENABLE_DEBUG_VIZUALISATIONS to 1");
#endif 

            if (m_app.GetZoomTool() != nullptr && ImGui::CollapsingHeader("Zoom Tool"))
                m_app.GetZoomTool()->DebugGUI(indent);
        }

        {
            // quick tonemapping settings
            ImGui::PushItemWidth(defItemWidth * 0.7f);
            const char* tooltipInfo = "Detailed exposure settings are in Tone Mapping section";
            ImGui::PushID("QS");
            ImGui::Checkbox("AutoExposure", &m_ui.ToneMappingParams.autoExposure); if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", tooltipInfo);
            ImGui::SameLine();
            ImGui::SeparatorEx(ImGuiSeparatorFlags_Vertical);
            ImGui::SameLine();
            ImGui::SliderFloat("Brightness", &m_ui.ToneMappingParams.exposureCompensation, -12.0f, 12.0f, "%.2f");  if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", tooltipInfo);
            ImGui::SameLine();
            if (ImGui::Button("0"))
                m_ui.ToneMappingParams.exposureCompensation = 0;
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", tooltipInfo);
            const ToneMappingPass * toneMappingPass = m_app.GetToneMappingPass();
            if (m_ui.EnableToneMapping && toneMappingPass != nullptr && toneMappingPass->GetExposureInfo(0).Clamped)
            {
                ImGui::SameLine();
                ImGui::TextColored(warnColor, "(clamped)");
                if (ImGui::IsItemHovered()) ImGui::SetTooltip("Total exposure (auto or manual camera + brightness) is clamped to [%+.0f, %+.0f] stops", TONEMAPPING_EXPOSURE_LOG2_MIN, TONEMAPPING_EXPOSURE_LOG2_MAX);
            }
            ImGui::PopID();
            ImGui::PopItemWidth();
        }
    }

    PTMaterial* material = PTMaterial::SafeCast(m_ui.SelectedMaterial);
    if (material != nullptr && m_ui.ShowMaterialEditor && m_app.GetMaterialsBaker() != nullptr)
    {
        // TODO: move all this to MaterialBaker

        ImGui::SetNextWindowPos(ImVec2(float(scaledWidth) - 10.f, 10.f), 0, ImVec2(1.f, 0.f));
        ImGui::SetNextWindowSize(ImVec2(defWindowWidth, 0), ImGuiCond_Appearing);
        ImGui::Begin("Material Editor");
        ImGui::PushItemWidth(defItemWidth);
        ImGui::Text("Material %d: %s.%s ", material->GPUDataIndex, material->ModelName.c_str(), material->Name.c_str());

        const bool wasAlphaTestedEnabled = material->EnableAlphaTesting;
        const bool wasTransmissionEnabled = material->EnableTransmission;
        const bool wasExcludedFromNEE = material->ExcludeFromNEE;
        const float alphaCutoffBefore = material->AlphaCutoff;
        const bool wasSkipRender = material->SkipRender;
        
        MaterialShaderPermutationKey mspBefore = MaterialShaderPermutationKey(material->ComputeShaderPermutation(""));

        bool dirty = material->EditorGUI(*m_app.GetMaterialsBaker());

        MaterialShaderPermutationKey mspAfter = MaterialShaderPermutationKey(material->ComputeShaderPermutation(""));

        const float alphaCutoffAfter = material->AlphaCutoff;

        if (mspBefore != mspAfter || 
            wasAlphaTestedEnabled != material->EnableAlphaTesting || 
            wasTransmissionEnabled != material->EnableTransmission || 
            wasExcludedFromNEE != material->ExcludeFromNEE ||
            wasSkipRender != material->SkipRender ||
            dirty)
        {
            m_app.GetScene()->GetSceneGraph()->GetRootNode()->InvalidateContent();
            m_ui.ResetAccumulation = 1;
        }

		// The domain change might require a rebuild without the Opaque flag
        if (wasAlphaTestedEnabled != material->EnableAlphaTesting || alphaCutoffBefore != alphaCutoffAfter ||
            wasExcludedFromNEE != material->ExcludeFromNEE || mspBefore != mspAfter || wasSkipRender != material->SkipRender)
            m_ui.ShaderAndACRefreshDelayedRequest = 1.0f;

        if( m_ui.ShaderAndACRefreshDelayedRequest > 0 )
            ImGui::TextColored( ImVec4(1,0.5f,0.5f,1), "PLEASE NOTE: shader and AC rebuild scheduled!\nUI might freeze for a bit." );
        else
            ImGui::Text(" ");

        ImGui::PopItemWidth();
        ImGui::End();
    }

#if ENABLE_DEBUG_DELTA_TREE_VIZUALISATION
    if (m_ui.ShowDeltaTree)
    {
        float scaledWindowWidth = scaledWidth - defWindowWidth - 20;
        ImGui::SetNextWindowPos(ImVec2(scaledWidth - float(scaledWindowWidth) - 10, 10.f), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(scaledWindowWidth, scaledWindowWidth * 0.5f), ImGuiCond_FirstUseEver);
        const DeltaTreeVizHeader& DeltaTreeVizHeader = m_app.GetFeedbackData().deltaPathTree;
        char windowName[1024];
        snprintf(windowName, sizeof(windowName), "Delta Tree Explorer, pixel (%d, %d), sampleIndex: %d, nodes: %d###DeltaExplorer", DeltaTreeVizHeader.pixelPos.x, DeltaTreeVizHeader.pixelPos.y, DeltaTreeVizHeader.sampleIndex, DeltaTreeVizHeader.nodeCount);

        if (ImGui::Begin(windowName, nullptr, ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse))
        {
            ImGui::PushItemWidth(defItemWidth);
            buildDeltaTreeViz();
            ImGui::PopItemWidth();
        }
        ImGui::End();
    }
#endif

    if (m_showSceneWidgets > 0.0f 
#if ENABLE_DEBUG_DELTA_TREE_VIZUALISATION
        && !m_ui.ShowDeltaTree
#endif
        )
    {

        std::string envMapOverrideSource = m_app.GetEnvMapOverrideSource();
        std::vector<std::string> envOptions;
        envOptions.push_back( c_EnvMapSceneDefault );
        //envOptions.push_back( c_EnvMapProcSky );
        envOptions.push_back( c_EnvMapProcSky_Morning );
        envOptions.push_back( c_EnvMapProcSky_Midday );
        envOptions.push_back( c_EnvMapProcSky_Evening );
        envOptions.push_back( c_EnvMapProcSky_Dawn );
        envOptions.push_back( c_EnvMapProcSky_PitchBlack );
        int envOptionsCurrentIndex = -1; for (int i = 0; i < envOptions.size(); i++) if (envOptions[i]==envMapOverrideSource) envOptionsCurrentIndex = i;

        std::vector<std::string> materialVariants;
        if (FindSubStringIgnoreCase(m_app.GetCurrentSceneName(), "bistro") != std::string::npos)
            materialVariants = {"dry", "wet", "silly"};
        int materialVariantIndexPrev = m_ui.MaterialVariantIndex;

        // collect toggles
        struct BigButton
        {
            std::string                 Name;
            std::optional<std::string>  HoverText;

            bool *                      PropVar         = nullptr; // type 1
            TogglableNode *             PropNode        = nullptr; // type 2
            std::vector<std::string> *  PropOptions     = nullptr; // type 3
            int *                       PropOptionIndex = nullptr; // type 3
            std::function<std::string(std::string)>
                                        GetItemName     = nullptr;

            bool                        Enabled;

            BigButton( const std::string & name, bool & prop ) : Name(name), PropVar(&prop), PropNode(nullptr), Enabled(true) {}
            BigButton( const std::string & name, bool & prop, const std::string& hoverText, bool enabled ) : Name(name), PropVar(&prop), PropNode(nullptr), HoverText(hoverText), Enabled(enabled) {}
            BigButton( const std::string & name, TogglableNode * prop ) : Name(TrimTogglable(name)), PropVar(nullptr), PropNode(prop), Enabled(true) {}
            BigButton( const std::string & name, std::vector<std::string>* propOptions, int* propOptionIndex, const std::string& hoverText, const std::function<std::string(std::string)> & getItemName) : Name(name), PropOptions(propOptions), PropOptionIndex(propOptionIndex), HoverText(hoverText), Enabled(true), GetItemName(getItemName) { assert(PropOptions->size()>0); }
            bool                IsSelected() const            { return (PropOptions != nullptr)?(true):((PropVar != nullptr)?(*PropVar):(PropNode->IsSelected())); }
            void                SetSelected( bool selected )  { if( PropVar != nullptr ) *PropVar = selected; else if (PropNode != nullptr ) PropNode->SetSelected(selected); else *PropOptionIndex = ( ((*PropOptionIndex)+1) % PropOptions->size() ); }
            std::string         GetText() const 
            {
                if (PropOptions != nullptr)
                {
                    if (GetItemName!=nullptr)
                        return Name + (((*PropOptionIndex) >= 0) ? (GetItemName((*PropOptions)[*PropOptionIndex])) : (std::string("other")));
                    else
                        return Name + (((*PropOptionIndex)>=0)?((*PropOptions)[*PropOptionIndex]):(std::string("other")));
                }
                else
                    return Name;
            }

        };
        std::vector<BigButton> buttons;
        buttons.push_back(BigButton("Animations", m_ui.EnableAnimations, "Animations are not available in reference mode", m_ui.RealtimeMode));
        buttons.push_back(BigButton("AutoExposure", m_ui.ToneMappingParams.autoExposure ) );
        buttons.push_back(BigButton("Sky: ", &envOptions, &envOptionsCurrentIndex, "For more options see Scene/Environment in the main UI", std::function<std::string(std::string)>(TrimSkyDisplayName) ));
        if (materialVariants.size()>0) buttons.push_back(BigButton("Variant: ", &materialVariants, &m_ui.MaterialVariantIndex, "Material or other scene variants", nullptr));
        for (int i = 0; m_ui.TogglableNodes != nullptr && i < m_ui.TogglableNodes->size(); i++)
            buttons.push_back(BigButton((*m_ui.TogglableNodes)[i].SceneNode->GetName(), &(*m_ui.TogglableNodes)[i]));

        if( buttons.size() > 0 )
        {
            // show & 
            ImVec2 texSizeA = ImGui::CalcTextSize("A");
            float buttonWidth = texSizeA.x * 16;
            float windowHeight = texSizeA.y * 3.0f;
            float windowWidth = buttonWidth * buttons.size() + ImGui::GetStyle().ItemSpacing.x * (buttons.size()+1);
            ImGui::SetNextWindowPos(ImVec2(0.5f * (scaledWidth - windowWidth), 10.0f), ImGuiCond_Always);
            ImGui::SetNextWindowSize(ImVec2(windowWidth, windowHeight), ImGuiCond_Always);
            ImGui::SetNextWindowBgAlpha(0.0f);
            if (ImGui::Begin("Widgets", nullptr, ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoScrollWithMouse | ImGuiWindowFlags_NoBackground | ImGuiWindowFlags_NoNav))
            {
                for (int i = 0; i < buttons.size(); i++)
                {
                    if (i > 0)
                        ImGui::SameLine();
                    
                    UI_SCOPED_DISABLE(!buttons[i].Enabled);

                    bool selected = buttons[i].IsSelected();

                    ImGui::PushID(i);
                    float h = 0.33f; 
                    float b = selected ? 1.0f : 0.1f;
                    ImGui::PushStyleColor(ImGuiCol_Button, (ImVec4)ImColor::HSV(h, 0.6f * b, 0.6f * b));
                    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, (ImVec4)ImColor::HSV(h, 0.7f * b, 0.7f * b));
                    ImGui::PushStyleColor(ImGuiCol_ButtonActive, (ImVec4)ImColor::HSV(h, 0.8f * b, 0.8f * b));
                    if (ImGui::Button(buttons[i].GetText().c_str(), ImVec2(buttonWidth, texSizeA.y * 2)))
                    {
                        buttons[i].SetSelected(!selected);
                        m_ui.ResetAccumulation = true;
                    }
                    ImGui::PopStyleColor(3);
                    ImGui::PopID();

                    if (buttons[i].HoverText.has_value())
                    {
                        if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) 
                            ImGui::SetTooltip(buttons[i].HoverText.value().c_str());
                    }
                }
            }
            ImGui::End();
        }

        if (envOptionsCurrentIndex >= 0 && envOptionsCurrentIndex < envOptions.size() && envOptions[envOptionsCurrentIndex] != envMapOverrideSource )
        {
            m_app.SetEnvMapOverrideSource(envOptions[envOptionsCurrentIndex]);
        }

        if (m_ui.MaterialVariantIndex != materialVariantIndexPrev)
        {
            if (FindSubStringIgnoreCase(m_app.GetCurrentSceneName(), "bistro") != std::string::npos)
            {
                // bistro dry-wet test
                std::vector<std::string> pavementList = { "LMBR0000163Cobbl_a1d987f5", "LMBR000016bCobbl_8652c51e", "LMBR0000162Paris_c30c71f1", "LMBR0000162Paris_c30c71f1", "LMBR000016cCobbl_f202ecfa", "LMBR0000161Pavem_e2e87964", "LMBR0000168Cobbl_a5a7f4b4", "LMBR0000160Pavem_613287fe", "LMBR000016aCobbl_e1c68d26" };
                for (std::string& id : pavementList)
                    if (auto m = m_app.GetMaterialsBaker()->FindByUniqueID(id))
                    {
                        if (m_ui.MaterialVariantIndex == 0) // reset to default
                            m_app.GetMaterialsBaker()->LoadSingle(*m);
                        else
                        {   // make wet-looking
                            m->Roughness = 0.0f;
                            //m->SpecularColor = float3(0.08f, 0.08f, 0.08f);
                        }
                        m->GPUDataDirty = true;
                    }

                std::vector<std::string> emissivesList = { "LMBR0000172Paris_1d83765c" /*bollards*/, "LMBR00000aeGreen_04f5ae02" /*green leaves*/, "LMBR00000afOrang_a907f305" /*yellow leaves*/, "LMBR00000b0Branc_5990161e" /*branches*/ };
                for (std::string& id : emissivesList)
                    if (auto m = m_app.GetMaterialsBaker()->FindByUniqueID(id))
                    {
                        if (m_ui.MaterialVariantIndex == 0 || m_ui.MaterialVariantIndex == 1) // reset to default
                            m_app.GetMaterialsBaker()->LoadSingle(*m);
                        else
                        {   // silly stuff
                            if (id == "LMBR0000172Paris_1d83765c")
                            {
                                m->EmissiveColor = float3( 0.01f, 1.0f, 0.1f );
                                m->EmissiveIntensity = 0.5f;
                            }
                            if (id == "LMBR00000aeGreen_04f5ae02" || id == "LMBR00000afOrang_a907f305")
                            {
                                m->EmissiveColor = (id == "LMBR00000aeGreen_04f5ae02")?float3(0.9f, 0.3f, 0.01f):float3(0.001f, 1.0f, 0.01f);
                                m->EmissiveIntensity = 0.6f;
                            }
                            if (id == "LMBR00000b0Branc_5990161e")
                            {
                                m->EmissiveColor = float3(1.0f, 0.001f, 0.005f);
                                m->EmissiveIntensity = 1.0f;
                            }
                        }
                        m->GPUDataDirty = true;
                    }

                auto m = m_app.GetMaterialsBaker()->FindByUniqueID("RX6Fighter_09479c96");
                if (m != nullptr)
                {
                    if (m_ui.MaterialVariantIndex == 0 || m_ui.MaterialVariantIndex == 1) // reset to default
                        m_app.GetMaterialsBaker()->LoadSingle(*m);
                    else
                        m->EmissiveIntensity = 5000.0f;
                    m->GPUDataDirty = true;
                }

                // emissives no longer require TLAS/BLAS update
                // if (m_ui.MaterialVariantIndex != 1 && materialVariantIndexPrev != 0) // this one doesn't change emissives so no TLAS/BLAS update needed
                //     m_ui.ShaderAndACRefreshDelayedRequest = 0.01f;
            }
        }
        
    }

    if ( m_app.GetGame() != nullptr && m_app.GetGame()->IsInitialized() )
    {
        m_app.GetGame()->StandaloneGUI(m_app.GetCurrentView(), float2(m_app.GetDisplaySize()));
    }

    // ImGui::ShowDemoWindow();
}

void SampleUI::buildDeltaTreeViz()
{
#if ENABLE_DEBUG_DELTA_TREE_VIZUALISATION
    // make tiny scaling
    int localScaleIndex = FindBestScaleFontIndex(m_currentScale*0.75f);
    float localScale = m_scaledFonts[localScaleIndex].second;
    ImGui::PushFont(m_scaledFonts[localScaleIndex].first);
    ImGuiStyle& style = ImGui::GetStyle(); 
    style = m_defaultStyle;
    style.ScaleAllSizes(localScale);

    // fixed a lot of stability issues so this no longer needed - probably, leaving in just for a bit longer
    // // Unfortunately, the ImNodes are unstable when changed every frame. At some point they can be dropped and all drawing done ourselves, since we do the layout anyway and only use it for drawing connections which we can do.
    // // Until that's done, we have to cache and only update once every few frames.
    // static DeltaTreeVizHeader cachedHeader = DeltaTreeVizHeader::make();
    // static DeltaTreeVizPathVertex cachedVertices[cDeltaTreeVizMaxVertices];
    // {
    //     static int frameCounter = 0; frameCounter++;
    //     static int lastUpdated = -10;
    //     if ((frameCounter - lastUpdated) > 0)
    //     {
    //         lastUpdated = frameCounter;
    //         cachedHeader = m_app.GetFeedbackData().deltaPathTree;
    //         memcpy( cachedVertices, m_app.GetDebugDeltaPathTree(), sizeof(DeltaTreeVizPathVertex)*cDeltaTreeVizMaxVertices );
    //     }
    // }
    const DeltaTreeVizHeader& DeltaTreeVizHeader   = m_app.GetFeedbackData().deltaPathTree; // cachedHeader;
    const DeltaTreeVizPathVertex* deltaPathTreeVertices = m_app.GetDebugDeltaPathTree(); // cachedVertices;
    const int nodeCount = DeltaTreeVizHeader.nodeCount;

    ImGui::NewLine(); ImGui::NewLine(); ImGui::NewLine(); ImGui::NewLine(); ImGui::NewLine(); ImGui::NewLine(); ImGui::NewLine(); ImGui::NewLine(); ImGui::NewLine(); ImGui::NewLine();
    ImGui::Text( "Denoising layers branch IDs:" );
    for (int i = 0; i < cDenoisingLayerCount; i++)
    {
        ImGui::Text( " %d: 0x%08x (%d dec)", i, DeltaTreeVizHeader.stableBranchIDs[i], DeltaTreeVizHeader.stableBranchIDs[i] );
        if (i == DeltaTreeVizHeader.dominantDenoisingLayerIndex)
        {
            ImGui::SameLine();
            ImGui::Text( " <DOMINANT>");
        }
    }

    ImNodes::Ez::BeginCanvas();

    ImVec2 topLeft = { ImGui::GetStyle().ItemSpacing.x * 8.0f, ImGui::GetStyle().ItemSpacing.y * 12.0f };
    ImVec2 nodeSize = {};
    const int nodeWidthInChars  = 28;
    const int nodeHeightInLines = 40;
    nodeSize.x = ImGui::CalcTextSize(std::string(' ', (size_t)nodeWidthInChars).c_str()).x;
    nodeSize.y = ImGui::GetStyle().ItemSpacing.y * nodeHeightInLines;
    ImVec2 nodePadding = ImVec2(nodeSize.x * 0.5f, nodeSize.y * 0.1f);

    struct UITreeNode
    {
        ImVec2                      pos;
        bool                        selected;
        std::string                 title;
        DeltaTreeVizPathVertex      deltaVertex;
        uint                        parentLobe;
        uint                        vertexIndex;
        std::shared_ptr<donut::engine::Material> material;  // nullptr for sky
        UITreeNode *                parent = nullptr;
        std::vector<UITreeNode *>   children;

        void Init(const DeltaTreeVizPathVertex& deltaVertex, Sample & app, const ImVec2 & nodeSize, const ImVec2 & nodePadding, const ImVec2 & topLeft)
        {   app;
            this->deltaVertex = deltaVertex;
            selected = false;
            vertexIndex = deltaVertex.vertexIndex ;
            parentLobe = deltaVertex.getParentLobe();
            
            float thpLum = dm::luminance(deltaVertex.throughput);

            char info[1024];
            snprintf(info, sizeof(info), "Vertex: %d, Throughput: %.1f%%", vertexIndex, thpLum*100.0f );
            title = info;
            if(deltaVertex.isDominant)
                title += " DOM";
            int padding = max( 0, nodeWidthInChars - (int)title.size() );
            title.append((size_t)padding, ' ');
            pos = topLeft;
            pos.x += (vertexIndex-1) * (nodeSize.x + nodePadding.x);
        }
    };

    UITreeNode treeNodes[cDeltaTreeVizMaxVertices];
    std::vector<std::vector<UITreeNode*>> nodeLevels;
    nodeLevels.resize( MAX_BOUNCE_COUNT+2 );
    int longestLevelCount = 0;
    for (int i = 0; i < nodeCount; i++)
    {
        UITreeNode & node = treeNodes[i];
        node.Init(deltaPathTreeVertices[i], m_app, nodeSize, nodePadding, topLeft);
        assert(node.vertexIndex < nodeLevels.size());
        nodeLevels[node.vertexIndex].push_back(&node);
        longestLevelCount = std::max(longestLevelCount, (int)nodeLevels[node.vertexIndex].size());
        // find parent - which is the last node with lower vertex index
        if (node.vertexIndex > 1) // vertex index 0 is camera, vertex index 1 is primary hit
        {
            assert( i>0 );
            for( int j = i-1; j >= 0; j-- )
                if (treeNodes[j].vertexIndex == node.vertexIndex - 1)
                {
                    node.parent = &treeNodes[j];
                    node.parent->children.push_back(&node);
                    break;
                }
            assert( node.parent != nullptr );
        }
    }

    // update Y positions, including parents
    for (int i = (int)nodeLevels.size() - 1; i >= 0; i--)
    {
        auto& level = nodeLevels[i];
        for (int npl = 0; npl < level.size(); npl++)
        {
            auto& node = level[npl];
            node->pos.y = topLeft.y + std::max(0, npl) * (nodeSize.y + nodePadding.y);
            // just make aligned to the top child if any - easier to see
            if (node->children.size() > 0)
            {
                float topChild = FLT_MAX;
                for (auto& child : node->children)
                    topChild = std::min(topChild, child->pos.y);
                node->pos.y = std::max(topChild, node->pos.y);
            }
        }
    }
    
    auto outSlotName = [](int lobeIndex){ return "D" + std::to_string(lobeIndex); };
    ImNodes::Ez::SlotInfo inS; inS.kind = 1; inS.title = "in";

    auto ImGuiColorInfo = [&]( const char * text, ImVec4 color, const char * tooltipText, auto... tooltipParams ) -> bool
    {
        char info[1024];
        snprintf(info, sizeof(info), "%.2f, %.2f, %.2f###%s", color.x, color.y, color.z, text);
        bool selected = true;
        ImGui::PushStyleColor(ImGuiCol_HeaderActive, color);
        ImGui::PushStyleColor(ImGuiCol_HeaderHovered, color);
        ImGui::PushStyleColor(ImGuiCol_Header, color);
        ImGui::Text("%s",text); ImGui::SameLine();
        ImGui::Selectable(info, true, 0, ImVec2(nodeSize.x*0.7f, 0) ); /*, ImGuiSelectableFlags_Disabled*/
        ImGui::PopStyleColor(3);
        if( ImGui::IsItemHovered() )
        {
            ImGui::SetTooltip(tooltipText, tooltipParams...);
            return true;
        }
        return false;
    };

    for (int i = 0; i < nodeCount; i++)
    {
        UITreeNode & treeNode = treeNodes[i];

        int onPlaneIndex = -1; bool onStablePath = false;
        for (int spi = 0; spi < cDenoisingLayerCount; spi++)
        {
            if (DenoisingLayerIsOnPlane(DeltaTreeVizHeader.stableBranchIDs[spi], treeNode.deltaVertex.stableBranchID))
            {
                onPlaneIndex = spi;
                onStablePath = true;
                break;
            }
            onStablePath |= DenoisingLayerIsOnStablePath(DeltaTreeVizHeader.stableBranchIDs[spi], treeNode.deltaVertex.stableBranchID);
        }
        auto mergeColor = [](ImVec4 & inout, ImVec4 ref) { inout = ImVec4( min(1.0f, inout.x + ref.x), min(1.0f, inout.y + ref.y), min(1.0f, inout.z + ref.z), inout.w ); };
        ImVec4 colorAdd = { 0,0.0f,0.0f,0.0f };
        if (onPlaneIndex >= 0)
            colorAdd = ImVec4((onPlaneIndex == 0) ? 0.5f : 0.0f, (onPlaneIndex == 1) ? 0.5f : 0.0f, (onPlaneIndex == 2) ? 0.5f : 0.0f, 1);
        else if (onStablePath)
            colorAdd = ImVec4(0.3f, 0.3f, 0.0f, 1);

        ImVec4 cola{ 0.22f, 0.22f, 0.22f, 1.0f };   mergeColor(cola, colorAdd);
        ImVec4 colb{ 0.32f, 0.32f, 0.32f, 1.0f };   mergeColor(colb, colorAdd);
        ImVec4 colc{ 0.5f, 0.5f, 0.5f, 1.0f };      mergeColor(colc, colorAdd);
        ImNodes::Ez::PushStyleColor(ImNodesStyleCol_NodeTitleBarBg, cola);
        ImNodes::Ez::PushStyleColor(ImNodesStyleCol_NodeTitleBarBgHovered, colb);
        ImNodes::Ez::PushStyleColor(ImNodesStyleCol_NodeTitleBarBgActive, colc);

        if (ImNodes::Ez::BeginNode(&treeNode, treeNode.title.c_str(), &treeNode.pos, &treeNode.selected))
        {
            bool isAnyHovered = ImGui::IsItemHovered();
            if (isAnyHovered)
                ImGui::SetTooltip("Stable delta tree branch ID: 0x%08x (%d dec)", treeNode.deltaVertex.stableBranchID, treeNode.deltaVertex.stableBranchID);

            ImNodes::Ez::InputSlots(&inS, 1);

            isAnyHovered |= ImGuiColorInfo("Thp:", ImVec4(treeNode.deltaVertex.throughput.x, treeNode.deltaVertex.throughput.y, treeNode.deltaVertex.throughput.z, 1.0f),
                "Throughput at current vertex: %.4f, %.4f, %.4f\nLast segment volume absorption was %.1f%%\n", treeNode.deltaVertex.throughput.x, treeNode.deltaVertex.throughput.y, treeNode.deltaVertex.throughput.z, treeNode.deltaVertex.volumeAbsorption*100.0f );

            std::string matName = ">>SKY<<";
            if( treeNode.deltaVertex.materialID != 0xFFFFFFFF )
            {
                treeNode.material = m_app.FindMaterial((int)treeNode.deltaVertex.materialID);
                if( treeNode.material != nullptr )
                    matName = treeNode.material->name; 
            }
            std::string matNameFull = matName;
            if( matName.length() > 30 ) matName = matName.substr(0, 30) + "...";

            ImGui::Text("Surface: %s", matName.c_str());
            if (ImGui::IsItemHovered())
            {
                ImGui::SetTooltip("Surface info: %s", matNameFull.c_str());
                isAnyHovered = true;
            }

            ImGui::Text("Lobes: %d", treeNode.deltaVertex.deltaLobeCount);

            //ImGui::Col
            ImNodes::Ez::SlotInfo outS[cMaxDeltaLobes+1+3];
            int outSN = 0;
            outS[outSN++] = ImNodes::Ez::SlotInfo{ "", 0 }; // empty text to align with ^ text
            outS[outSN++] = ImNodes::Ez::SlotInfo{ "", 0 }; // empty text to align with ^ text
            outS[outSN++] = ImNodes::Ez::SlotInfo{ "", 0 }; // empty text to align with ^ text
            for (int j = 0; j < (int)treeNode.deltaVertex.deltaLobeCount; j++ )
            {
                auto lobe = treeNode.deltaVertex.deltaLobes[j];
                if( lobe.probability > 0 )
                    outS[outSN++] = ImNodes::Ez::SlotInfo{ outSlotName(j), 1 };
                isAnyHovered |= ImGuiColorInfo( (std::string(" D")+std::to_string(j) + ":").c_str(), ImVec4(lobe.thp.x, lobe.thp.y, lobe.thp.z, 1.0f),
                    "Delta lobe %d throughput: %.4f, %.4f, %.4f\nType: %s", j, lobe.thp.x, lobe.thp.y, lobe.thp.z, lobe.transmission?("transmission"):("reflection") );
            }

            ImGui::Text(" Non-delta: %.1f%%", treeNode.deltaVertex.nonDeltaPart*100.0f);
            if (ImGui::IsItemHovered())
            {
                ImGui::SetTooltip("This is the amount of throughput that gets handled by diffuse and rough specular lobes");
                isAnyHovered = true;
            }

            ImNodes::Ez::OutputSlots(outS, outSN);
            if (ImGui::IsItemHovered())
                isAnyHovered = true;
            ImNodes::Ez::EndNode();
            if (ImGui::IsItemHovered())
                isAnyHovered = true;

            if (isAnyHovered)
            {
                float3 worldPos = treeNode.deltaVertex.worldPos;
                float3 viewVec = worldPos - m_app.GetCurrentCamera().GetPosition();
                float sphereSize = 0.006f + 0.004f * dm::length(viewVec);
                float step = 0.15f;
                viewVec = dm::normalize(viewVec);
                float3 right = dm::cross(viewVec, m_app.GetCurrentCamera().GetUp());
                float3 up = dm::cross(right, viewVec);
                float3 prev0 = worldPos;
                float3 prev1 = worldPos;
                float3 prev2 = worldPos;
                for (float s = 0.0f; s < 2.06f; s += step)
                {
                    float px = cos(s * dm::PI_f);
                    float py = sin(s * dm::PI_f);
                    float3 sp0 = worldPos + up * py * sphereSize + right * px * sphereSize;
                    float3 sp1 = worldPos + up * py * sphereSize * 0.8f + right * px * sphereSize * 0.8f;
                    float3 sp2 = worldPos + up * py * sphereSize * 0.6f + right * px * sphereSize * 0.6f;
                    float4 col1 = float4(colorAdd.x, colorAdd.y, colorAdd.z, 1);//float4(1,1,1,1); //float3( fmodf((s+1)*13.33f,1), fmodf((s+1)*17.55f,1), fmodf((s+1)*23.77f,1));
                    float4 col0 = float4(0,0,0,1);
                    if( s > 0.0f )
                    {
                        m_app.DebugDrawLine(prev0, sp0, col1, col1); 
                        m_app.DebugDrawLine(prev1, sp1, col0, col0); 
                        m_app.DebugDrawLine(prev0, sp1, col1, col0);
                        m_app.DebugDrawLine(prev2, sp0, col1, col0);
                        m_app.DebugDrawLine(prev2, sp2, col1, col1);
                    }
                    prev0 = sp0; prev1 = sp1; prev2 = sp2;
                }
            }
        }
        ImNodes::Ez::PopStyleColor(3);
    }

    // update connections
    for (auto& level : nodeLevels)
        for (int npl = 0; npl < level.size(); npl++)
        {
            auto& node = level[npl];
            if (node->parent != nullptr)
                ImNodes::Connection(node, inS.title.c_str(), node->parent, outSlotName(node->parentLobe).c_str());
        }

    ImNodes::Ez::EndCanvas();

    // reset scaling
    style = m_defaultStyle;
    style.ScaleAllSizes(m_currentScale);
    ImGui::PopFont();
#endif
}

bool TogglableNode::IsSelected() const
{
    return all( SceneNode->GetTranslation() == OriginalTranslation );
}

void TogglableNode::SetSelected(bool selected)
{
    if( selected )
        SceneNode->SetTranslation( OriginalTranslation );
    else
        SceneNode->SetTranslation( {-10000.0,-10000.0,-10000.0} );
}

void UpdateTogglableNodes(std::vector<TogglableNode>& togglableNodes, donut::engine::SceneGraphNode* node)
{
    auto addIfTogglable = [ & ](const std::string & token, SceneGraphNode* node) -> TogglableNode *
    {
        const size_t tokenLen = token.length();
        const std::string name = node->GetName();   const size_t nameLen = name.length();
        if (nameLen > tokenLen && name.substr(nameLen - tokenLen) == token)
        {
            TogglableNode tn;
            tn.SceneNode = node;
            tn.UIName = name.substr(0, nameLen - tokenLen);
            tn.OriginalTranslation = node->GetTranslation();
            togglableNodes.push_back(tn);
            return &togglableNodes.back();
        }
        return nullptr;
    };
    TogglableNode * justAdded = addIfTogglable("_togglable", node);
    if (justAdded==nullptr)
    {
        justAdded = addIfTogglable("_togglable_off", node);
        if( justAdded != nullptr )
            justAdded->SetSelected(false);
    }

    for (int i = (int)node->GetNumChildren() - 1; i >= 0; i--)
        UpdateTogglableNodes( togglableNodes, node->GetChild(i) );
}
