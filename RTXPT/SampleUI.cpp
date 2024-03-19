/*
* Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "Sample.h"

#include <donut/app/UserInterfaceUtils.h>
#include <donut/core/vfs/VFS.h>
#include <donut/engine/SceneTypes.h>
#include <iterator>

#include "ToneMapper/ToneMappingPasses.h"
#include "PathTracer/ShaderDebug.hlsli"

using namespace donut::app;
using namespace donut::engine;

std::filesystem::path GetLocalPath(std::string subfolder);

struct ImGUIScopedIndent
{
    ImGUIScopedIndent(float indent) :m_indent(indent)
    {
        ImGui::Indent(m_indent);
    }
    ~ImGUIScopedIndent()
    {
        ImGui::Unindent(m_indent);
    }
private:
    float m_indent;
};

struct ImGUIScopedDisable
{
    ImGUIScopedDisable(bool condition)
    {
        ImGui::BeginDisabled(condition);
    }
    ~ImGUIScopedDisable()
    {
        ImGui::EndDisabled();
    }
};

#define UI_SCOPED_INDENT(indent) ImGUIScopedIndent scopedIndent__##__LINE__(indent)
#define UI_SCOPED_DISABLE(cond) ImGUIScopedDisable scopedDisable__##__LINE__(cond)

#define IMAGE_QUALITY_OPTION(code) do{if (code) m_ui.ResetAccumulation = true;} while(false)

SampleUI::SampleUI(DeviceManager* deviceManager, Sample& app, SampleUIData& ui, bool SERSupported, bool OMMSupported)
        : ImGui_Renderer(deviceManager)
        , m_app(app)
        , m_ui(ui)
        , m_SERSupported(SERSupported)
        , m_OMMSupported(OMMSupported)
{
    m_CommandList = GetDevice()->createCommandList();

    auto nativeFS = std::make_shared<donut::vfs::NativeFileSystem>(); // *(app.GetRootFs())
    //auto fontPath = GetLocalPath("media") / "fonts/OpenSans/OpenSans-Regular.ttf";
    auto fontPath = GetLocalPath("media") / "fonts/DroidSans/DroidSans-Mono.ttf";

    float baseFontSize = 15.0f;
    for (int i = 0; i < std::size(m_scaledFonts); i++)
    {
        float scale = (i+2.0f) / 4.0f;
        m_scaledFonts[i] = std::make_pair(this->LoadFont(*nativeFS, fontPath, baseFontSize * scale), scale);
    }

    m_FontDroidMono = this->LoadFont(*nativeFS, GetLocalPath("media") / "fonts/DroidSans/DroidSans-Mono.ttf", 14.f);
    
    ImGui::GetIO().IniFilename = nullptr;

    m_ui.DXRHitObjectExtension = SERSupported;  // no need to check for or attempt using HitObjectExtension if SER not supported
    m_ui.ShaderExecutionReordering = SERSupported;

#if ENABLE_DEBUG_DELTA_TREE_VIZUALISATION
    m_ImNodesContext = ImNodes::Ez::CreateContext();
#endif

    m_ui.RelaxSettings = NrdConfig::getDefaultRELAXSettings();
    m_ui.ReblurSettings = NrdConfig::getDefaultREBLURSettings();

    m_ui.TemporalAntiAliasingParams.useHistoryClampRelax = true;

    m_ui.ToneMappingParams.toneMapOperator = ToneMapperOperator::HableUc2;

    // enable by default for now
    m_ui.RTXDI.regir.regirStaticParams.Mode = rtxdi::ReGIRMode::Grid;
}

SampleUI::~SampleUI()
{
#if ENABLE_DEBUG_DELTA_TREE_VIZUALISATION
    ImNodes::Ez::FreeContext(m_ImNodesContext);
#endif
}

bool SampleUI::MousePosUpdate(double xpos, double ypos)
{
    float scaleX, scaleY;
    GetDeviceManager()->GetDPIScaleInfo(scaleX, scaleY);
    xpos *= scaleX;
    ypos *= scaleY;
    return ImGui_Renderer::MousePosUpdate(xpos, ypos);
}

int SampleUI::FindBestScaleFontIndex(float scale)
{
    int bestScaleIndex = -1; float bestScaleDiff = FLT_MAX;
    for (int i = 0; i < std::size(m_scaledFonts); i++)
        if (std::abs(m_scaledFonts[i].second - scale) < bestScaleDiff)
        {
            bestScaleIndex = i; bestScaleDiff = std::abs(m_scaledFonts[i].second - scale);
        }
    return bestScaleIndex;
}

void SampleUI::Animate(float elapsedTimeSeconds)
{
    // overriding ImGui_Renderer::Animate(elapsedTimeSeconds) to handle scaling ourselves
    if (!imgui_nvrhi) return;

    int w, h;
    float scaleX, scaleY;
    GetDeviceManager()->GetWindowDimensions(w, h);
    GetDeviceManager()->GetDPIScaleInfo(scaleX, scaleY);
    assert(scaleX == scaleY);
    float scale = scaleX;

    ImGuiIO& io = ImGui::GetIO();
    io.DisplaySize = ImVec2(float(w), float(h));
    io.DisplayFramebufferScale.x = 1.0f;
    io.DisplayFramebufferScale.y = 1.0f;

    io.KeyCtrl = io.KeysDown[GLFW_KEY_LEFT_CONTROL] || io.KeysDown[GLFW_KEY_RIGHT_CONTROL];
    io.KeyShift = io.KeysDown[GLFW_KEY_LEFT_SHIFT] || io.KeysDown[GLFW_KEY_RIGHT_SHIFT];
    io.KeyAlt = io.KeysDown[GLFW_KEY_LEFT_ALT] || io.KeysDown[GLFW_KEY_RIGHT_ALT];
    io.KeySuper = io.KeysDown[GLFW_KEY_LEFT_SUPER] || io.KeysDown[GLFW_KEY_RIGHT_SUPER];

    // find the best scale
    int bestScaleIndex = FindBestScaleFontIndex(scale);
    if( m_currentFontScaleIndex != bestScaleIndex )
    {
        // rescale!
        m_currentFontScaleIndex = bestScaleIndex;
        io.FontDefault = m_scaledFonts[m_currentFontScaleIndex].first;
        ImGuiStyle& style = ImGui::GetStyle();
        style = m_defaultStyle;
        m_currentScale = m_scaledFonts[m_currentFontScaleIndex].second;
        style.ScaleAllSizes(m_currentScale);
    }
    //ImGui::SetCurrentFont(  );

    m_showSceneWidgets = dm::clamp( m_showSceneWidgets + elapsedTimeSeconds * 8.0f * ((io.MousePos.y >= 0 && io.MousePos.y < h * 0.1f)?(1):(-1)), 0.0f, 1.0f );

    imgui_nvrhi->beginFrame(elapsedTimeSeconds);
}

std::string TrimTogglable(const std::string text)
{
    size_t tog = text.rfind("_togglable");
    if (tog != std::string::npos)
        return text.substr(0, tog);
    return text;
}

void SampleUI::buildUI(void)
{
    if (!m_ui.ShowUI)
        return;

    auto& io = ImGui::GetIO();
    float scaledWidth = io.DisplaySize.x; 
    float scaledHeight = io.DisplaySize.y;

    const float defWindowWidth = 320.0f * m_currentScale;
    const float defItemWidth = defWindowWidth * 0.3f * m_currentScale;

    ImGui::SetNextWindowPos(ImVec2(10.f, 10.f), ImGuiCond_Appearing);
    ImGui::SetNextWindowSize(ImVec2(defWindowWidth, scaledHeight-20), ImGuiCond_Appearing);
    ImGui::Begin("Settings", 0, ImGuiWindowFlags_None /*AlwaysAutoResize*/);
    ImGui::PushItemWidth(defItemWidth);

    const float indent = (int)ImGui::GetStyle().IndentSpacing*0.4f;
    ImVec4 warnColor = { 1,0.5f,0.5f,1 };

    ImGui::Text("%s, %s", GetDeviceManager()->GetRendererString(), m_app.GetResolutionInfo().c_str() );
    ImGui::Text(m_app.GetFPSInfo().c_str());

    if (ImGui::CollapsingHeader("System")) //, ImGuiTreeNodeFlags_DefaultOpen))
    {
        ImGui::Indent(indent);
        if (ImGui::Button("Reload Shaders (requires VS .hlsl->.bin build)"))
            m_ui.ShaderReloadRequested = true;
        ImGui::Checkbox("VSync", &m_ui.EnableVsync); 
        bool fpsLimiter = m_ui.FPSLimiter != 0;
        ImGui::SameLine(); ImGui::Checkbox("Cap FPS to 60", &fpsLimiter);
        m_ui.FPSLimiter = fpsLimiter?60:0;
        ImGui::SameLine(); 
        if (ImGui::Button("Save screenshot"))
        {
            std::string fileName;
            if (FileDialog(false, "BMP files\0*.bmp\0All files\0*.*\0\0", fileName))
            {
                m_ui.ScreenshotFileName = fileName;
            }
        }

        if (ImGui::CollapsingHeader("Advanced"))
        {
            ImGui::Indent(indent);
            ImGui::Text("Screenshot sequence path:");
            ImGui::Text(" '%s'", m_ui.ScreenshotSequencePath.c_str()); 
            if (ImGui::Checkbox("Save screenshot sequence", &m_ui.ScreenshotSequenceCaptureActive))
                if (m_ui.ScreenshotSequenceCaptureActive)
                    m_ui.FPSLimiter = 60;
            if (ImGui::IsItemHovered()) ImGui::SetTooltip(  "Example to convert to movie: \nffmpeg -r 60 -i frame_%%05d.bmp -vcodec libx265 -crf 13 -vf scale=1920:1080  outputvideo-1080p-60fps.mp4\n"
                                                            "60 FPS limiter will be automatically enabled for smooth recording!");
            if (!m_ui.ScreenshotSequenceCaptureActive)
                m_ui.ScreenshotSequenceCaptureIndex = -64; // -x means x warmup frames for recording to stabilize denoiser
            else
            {
                if (m_ui.ScreenshotSequenceCaptureIndex < 0) // first x frames are warmup!
                    m_app.ResetSceneTime();
                else
                {
                    char windowName[1024];
                    snprintf(windowName, sizeof(windowName), "%s/frame_%05d.bmp", m_ui.ScreenshotSequencePath.c_str(), m_ui.ScreenshotSequenceCaptureIndex);
                    m_ui.ScreenshotFileName = windowName;
                }
                m_ui.ScreenshotSequenceCaptureIndex++;
            }
            ImGui::Separator();
            ImGui::Checkbox("Loop longest animation", &m_ui.LoopLongestAnimation);
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("If enabled, only restarts all animations when longest one played out. Otherwise loops them individually (and not in sync)!");
            ImGui::Unindent(indent);
        }

        ImGui::Unindent(indent);
    }

    const std::string currentScene = m_app.GetCurrentSceneName();
    ImGui::PushItemWidth(-60.0f);
    if (ImGui::BeginCombo("Scene selection", currentScene.c_str()))
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
    ImGui::PopItemWidth();

    if (ImGui::CollapsingHeader("Scene"/*, ImGuiTreeNodeFlags_DefaultOpen*/))
    {
        ImGui::Indent(indent);
        if (m_app.UncompressedTextureCount() > 0)
        {
            ImGui::TextColored(warnColor, "Scene has %d uncompressed textures", (uint)m_app.UncompressedTextureCount());
            if (ImGui::Button("Batch compress with nvtt_export.exe", { -1, 0 }))
                if (m_app.CompressTextures())
                {   // reload scene
                    m_app.SetCurrentScene(m_app.GetCurrentSceneName(), true);
                }
        }
        
        {
            UI_SCOPED_DISABLE(!m_ui.RealtimeMode);
            ImGui::Checkbox("Enable animations", &m_ui.EnableAnimations);
            if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) ImGui::SetTooltip("Animations are not available in reference mode");
        }
        if (ImGui::Button("Reset animation time"))
        {
            m_app.ResetSceneTime();
            m_ui.ResetAccumulation = true;
        }

        if (m_ui.TogglableNodes != nullptr && ImGui::CollapsingHeader("Togglables"))
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
            ImGui::Indent(indent);
            if (m_app.IsEnvMapLoaded())
            {
                if (ImGui::InputFloat3("Tint Color", (float*)&m_ui.EnvironmentMapParams.TintColor.x))
                    m_ui.ResetAccumulation = true;
                if (ImGui::InputFloat("Intensity", &m_ui.EnvironmentMapParams.Intensity))
                    m_ui.ResetAccumulation = true;
                if (ImGui::InputFloat3("Rotation XYZ", (float*)&m_ui.EnvironmentMapParams.RotationXYZ.x))
                    m_ui.ResetAccumulation = true;
                if (ImGui::Checkbox("Enabled", &m_ui.EnvironmentMapParams.Enabled))
                    m_ui.ResetAccumulation = true;
            }
            else
            {
                ImGui::Text("No envmap loaded");
            }
            ImGui::Unindent(indent);
        }

        ImGui::Unindent(indent);
    }

    if (ImGui::CollapsingHeader("Camera", 0/*ImGuiTreeNodeFlags_DefaultOpen*/))
    {
        ImGui::Indent(indent);
        std::vector<std::string> options; options.push_back("Free flight");
        for( uint i = 0; i < m_app.GetSceneCameraCount(); i++ )
            options.push_back( "Scene cam " + std::to_string(i) );
        uint & currentlySelected = m_app.SelectedCameraIndex();
        currentlySelected = std::min( currentlySelected, (uint)m_app.GetSceneCameraCount()-1 );
        if (ImGui::BeginCombo("Motion", options[currentlySelected].c_str()))
        {
            for (uint i = 0; i < m_app.GetSceneCameraCount(); i++)
            {
                bool is_selected = i == currentlySelected;
                if (ImGui::Selectable(options[i].c_str(), is_selected))
                    currentlySelected = i;
                if (is_selected)
                    ImGui::SetItemDefaultFocus();
            }
            ImGui::EndCombo();
        }

        if (currentlySelected == 0)
        {
            ImGui::Text("Camera position: "); ImGui::SameLine();
            if (ImGui::Button("Save", ImVec2(ImGui::GetFontSize() * 5.0f, ImGui::GetTextLineHeightWithSpacing())) ) m_app.SaveCurrentCamera(); ImGui::SameLine();
            if (ImGui::Button("Load", ImVec2(ImGui::GetFontSize() * 5.0f, ImGui::GetTextLineHeightWithSpacing())) ) m_app.LoadCurrentCamera();
        }

#if 1
        if (ImGui::InputFloat("Aperture", &m_ui.CameraAperture, 0.001f, 0.01f, "%.4f"))
            m_ui.ResetAccumulation = true;
        m_ui.CameraAperture = dm::clamp(m_ui.CameraAperture, 0.0f, 1.0f);

        if (ImGui::InputFloat("Focal Distance", &m_ui.CameraFocalDistance, 0.1f))
            m_ui.ResetAccumulation = true;
        m_ui.CameraFocalDistance = dm::clamp(m_ui.CameraFocalDistance, 0.001f, 1e16f);
        ImGui::SliderFloat("Keyboard move speed", &m_ui.CameraMoveSpeed, 0.1f, 10.0f);

        float cameraFOV = 2.0f * dm::degrees(m_app.GetCameraVerticalFOV());
        if (ImGui::InputFloat("Vertical FOV", &cameraFOV, 0.1f))
        {
            cameraFOV = dm::clamp(cameraFOV, 1.0f, 360.0f);
            m_ui.ResetAccumulation = true;
            m_app.SetCameraVerticalFOV(dm::radians(cameraFOV/2.0f));
        }
#endif
        ImGui::Unindent(indent);
    }

    if (ImGui::CollapsingHeader("Lighting", 0/*ImGuiTreeNodeFlags_DefaultOpen*/))
    {
        RAII_SCOPE(ImGui::Indent(indent);, ImGui::Unindent(indent););

        if (ImGui::CollapsingHeader("Distant", ImGuiTreeNodeFlags_DefaultOpen))
        {
            RAII_SCOPE(ImGui::Indent(indent); , ImGui::Unindent(indent););
            if (m_app.GetEnvMapBaker()!=nullptr) // envmap baker can legally be nullptr
                m_ui.ResetAccumulation |= m_app.GetEnvMapBaker()->DebugGUI(indent);
        }

        if (ImGui::CollapsingHeader("Local", ImGuiTreeNodeFlags_DefaultOpen))
        {
            RAII_SCOPE(ImGui::Indent(indent);, ImGui::Unindent(indent););
            // local lighting UI
        }
    }

    if (ImGui::CollapsingHeader("Path tracer", ImGuiTreeNodeFlags_DefaultOpen))
    {
        ImGui::Indent(indent);

        int modeIndex = (m_ui.RealtimeMode)?(1):(0);
        if (ImGui::Combo("Mode", &modeIndex, "Reference\0Realtime\0\0"))
        {
            m_ui.RealtimeMode = (modeIndex!=0);
            m_ui.ResetAccumulation = true;
        }
        ImGui::Indent(indent);
        if (m_ui.RealtimeMode)
        {
            {
                UI_SCOPED_DISABLE( (m_ui.UseReSTIRDI || m_ui.UseReSTIRGI) );
                ImGui::InputInt("Samples per pixel", &m_ui.RealtimeSamplesPerPixel); 
                m_ui.RealtimeSamplesPerPixel = dm::clamp(m_ui.RealtimeSamplesPerPixel, 1, 128);
                if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) 
                    ImGui::SetTooltip("How many full paths to trace per pixel from the primary surface\n(camera ray is not re-cast so there is no added AA)\n(currently incompatible with ReSTIR DI & ReSTIR GI)");
            }

            {
#ifdef STREAMLINE_INTEGRATION
                const bool DLSSAvailable = SLWrapper::Get().GetDLSSAvailable();
#else
                const bool DLSSAvailable = false;
#endif
                const char* items[] = { "No AA", "TAA", "DLSS", "DLAA" };

                const int itemCount = IM_ARRAYSIZE(items);

                m_ui.RealtimeAA = dm::clamp(m_ui.RealtimeAA, 0, DLSSAvailable ? itemCount : 1);

                if (ImGui::BeginCombo("Anti-aliasing", items[m_ui.RealtimeAA]))
                {
                    for (int i = 0; i < itemCount; i++)
                    {
                        UI_SCOPED_DISABLE(!DLSSAvailable && i > 1);

                        bool is_selected = (m_ui.RealtimeAA == i);
                        if (ImGui::Selectable(items[i], is_selected))
                            m_ui.RealtimeAA = i;
                        if (is_selected)
                            ImGui::SetItemDefaultFocus();
                    }
                    ImGui::EndCombo();
                }
            }

            ImGui::Checkbox("Realtime noise", &m_ui.RealtimeNoise);
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("If disabled, global noise seed will not change per frame. Useful for \ndebugging transient issues hidden by noise, or for before/after comparison");
        }
        else // reference mode
        {
            if (ImGui::Button("Reset"))
                m_ui.ResetAccumulation = true;
            ImGui::SameLine();
            ImGui::InputInt("Sample count", &m_ui.AccumulationTarget);
            m_ui.AccumulationTarget = dm::clamp(m_ui.AccumulationTarget, 1, 4 * 1024 * 1024); // this max is beyond float32 precision threshold; expect some banding creeping in when using more than 500k samples
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("Number of path samples per pixel to collect");
            ImGui::Text("Accumulated samples: %d (out of %d target)", m_ui.AccumulationIndex, m_ui.AccumulationTarget);
            ImGui::Text("(avg frame time: %.3fms)", m_app.GetAvgTimePerFrame() * 1000.0f);
            //if (m_ui.AccumulationIndex == m_ui.AccumulationTarget)
            {
                if (ImGui::Button("Photo mode screenshot"))
                    m_ui.ExperimentalPhotoModeScreenshot = true;
                if (ImGui::IsItemHovered()) ImGui::SetTooltip(  "Experimental: Saves a photo.bmp next to where .exe is and applies\n"
                                                                "denoising using command line tool that wraps OptiX and OIDN denoisers.\n"
                                                                "No guidance buffers are used and color is in LDR (so not as high quality\n"
                                                                "as it could be - will get improved in the future). \n"
                                                                "Command line denoiser wrapper tools by Declan Russel, available at:\n"
                                                                "https://github.com/DeclanRussell/NvidiaAIDenoiser\n"
                                                                "https://github.com/DeclanRussell/IntelOIDenoiser");
            }

            IMAGE_QUALITY_OPTION(ImGui::Checkbox("Use StablePlanes (*)", &m_ui.UseStablePlanes));
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("Use to test (should be identical before/after)\nUseStablePlanes is always on when RTXDI is enabled or in realtime mode");
            IMAGE_QUALITY_OPTION(ImGui::Checkbox("Anti-aliasing", &m_ui.AccumulationAA));
            IMAGE_QUALITY_OPTION(ImGui::Checkbox("Allow RTXDI in reference mode", &m_ui.AllowRTXDIInReferenceMode));
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("Note: RTXDI history isn't currently being reset with accumulation reset, so expect non-determinism if RTXDI enabled in reference mode");

        }
        ImGui::Unindent(indent);

        {
            UI_SCOPED_DISABLE(!m_ui.RealtimeMode);
            ImGui::Checkbox("Enable denoiser", &m_ui.RealtimeDenoiser);
            if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) ImGui::SetTooltip("Note: no built-in denoiser available in 'Reference' \nmode, however 'Photo mode screenshot' button launches\nexternal denoiser!");
        }

        IMAGE_QUALITY_OPTION(ImGui::Checkbox("Use Russian Roulette", &m_ui.EnableRussianRoulette));
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("This enables stochastic path termination for low throughput diffuse paths");

        if ( m_ui.RealtimeMode || m_ui.AllowRTXDIInReferenceMode )
        {
            {
                UI_SCOPED_DISABLE(!m_ui.UseNEE);
                IMAGE_QUALITY_OPTION(ImGui::Checkbox("Use ReSTIR DI (RTXDI)", &m_ui.UseReSTIRDI));
            }
            if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) ImGui::SetTooltip("ReSTIR DI (RTXDI) requires Next Event Estimation to be enabled");

            IMAGE_QUALITY_OPTION(ImGui::Checkbox("Use ReSTIR GI (RTXDI)", &m_ui.UseReSTIRGI));
        }

        IMAGE_QUALITY_OPTION(ImGui::InputInt("Max bounces", &m_ui.BounceCount));
        m_ui.BounceCount = dm::clamp(m_ui.BounceCount, 0, MAX_BOUNCE_COUNT);
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Max number of all bounces (including NEE and diffuse bounces)");
        if (m_ui.RealtimeMode)
            IMAGE_QUALITY_OPTION(ImGui::InputInt("Max diffuse bounces (realtime)", &m_ui.RealtimeDiffuseBounceCount));
        else
            IMAGE_QUALITY_OPTION(ImGui::InputInt("Max diffuse bounces (reference)", &m_ui.ReferenceDiffuseBounceCount));
        m_ui.RealtimeDiffuseBounceCount = dm::clamp(m_ui.RealtimeDiffuseBounceCount, 0, MAX_BOUNCE_COUNT);
        m_ui.ReferenceDiffuseBounceCount = dm::clamp(m_ui.ReferenceDiffuseBounceCount, 0, MAX_BOUNCE_COUNT);
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Max number of diffuse bounces (diffuse lobe and specular with roughness > 0.25 or similar depending on settings)");

        if (ImGui::InputFloat("Texture MIP bias", &m_ui.TexLODBias))
            m_ui.ResetAccumulation = true;
        if (m_ui.RealtimeMode)
        {
            if (ImGui::Checkbox("FireflyFilter (realtime)", &m_ui.RealtimeFireflyFilterEnabled))
                m_ui.ResetAccumulation = true;
            if (m_ui.RealtimeFireflyFilterEnabled && ImGui::InputFloat("FireflyFilter Threshold", &m_ui.RealtimeFireflyFilterThreshold, 0.01f, 0.1f, "%.5f") )
                m_ui.ResetAccumulation = true;
            m_ui.RealtimeFireflyFilterThreshold = dm::clamp( m_ui.RealtimeFireflyFilterThreshold, 0.00001f, 1000.0f );
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("Better light importance sampling allows for setting higher firefly filter threshold and conversely.");
        }
        else
        {
            if (ImGui::Checkbox("FireflyFilter (reference *)", &m_ui.ReferenceFireflyFilterEnabled))
                m_ui.ResetAccumulation = true;
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("* when both tonemapping autoexposure and firefly filter are enabled\nin reference mode, results are no longer deterministic!");
            if (m_ui.ReferenceFireflyFilterEnabled && ImGui::InputFloat("FireflyFilter Threshold", &m_ui.ReferenceFireflyFilterThreshold, 0.1f, 0.2f, "%.5f"))
                m_ui.ResetAccumulation = true;
            m_ui.ReferenceFireflyFilterThreshold = dm::clamp(m_ui.ReferenceFireflyFilterThreshold, 0.01f, 1000.0f);
        }

        if (m_SERSupported)
        {
            if (ImGui::Checkbox("DXR HitObject Extension codepath", &m_ui.DXRHitObjectExtension))
                m_ui.ResetAccumulation = true; // <- while there's no need to reset accumulation since this is a performance only feature, leaving the reset in for testing correctness
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("If disabled, traditional TraceRay path is used.\nIf enabled, TraceRayInline->MakeHit->ReorderThread->InvokeHit approach is used!");
            if (m_ui.DXRHitObjectExtension)
            {
                ImGui::Indent(indent);
                ImGui::Checkbox("Shader Execution Reordering", &m_ui.ShaderExecutionReordering);
                if (ImGui::IsItemHovered()) ImGui::SetTooltip("This enables/disables the actual ReorderThread call in the shader.");
                ImGui::Unindent(indent);
            }
        }
        else
        {
            ImGui::Text( "<DXR Hit Object Extension not supported>" );
            m_ui.DXRHitObjectExtension = false;
        }

        if (ImGui::CollapsingHeader("Next Event Estimation"))
        {
            IMAGE_QUALITY_OPTION(ImGui::Checkbox("Use Next Event Estimation", &m_ui.UseNEE));
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("This enables NEE a.k.a. direct light importance sampling (this includes ReSTIR DI but not ReSTIR GI)\nNote: analytic lights currently only come out of NEE so they will be missing when NEE is disabled");

            if (m_ui.UseNEE)
            {
                ImGui::Separator();

                ImGui::Text("Distant light importance sampling:");
                ImGui::Indent(indent); 
                ImGui::PushID("Distant");
#ifndef ENVMAP_IMPORTANCE_SAMPLING_TYPE
                IMAGE_QUALITY_OPTION(ImGui::Combo("Approach", (int*)&m_ui.NEEDistantType, "Uniform\0MIP descent\0Pre-sampling\0\0"));          
                m_ui.NEEDistantType = dm::clamp(m_ui.NEEDistantType, 0, 2);
                if (ImGui::IsItemHovered()) ImGui::SetTooltip("Note: all distant (directional) analytic lights will get baked into environment map. Currently not implemented.");
#else
                ImGui::TextWrapped("Undefine ENVMAP_IMPORTANCE_SAMPLING_TYPE for dynamic control over environment map approach type");
#endif
                IMAGE_QUALITY_OPTION(ImGui::InputInt("Candidate samples", &m_ui.NEEDistantCandidateSamples, 1));                
                m_ui.NEEDistantCandidateSamples = dm::clamp(m_ui.NEEDistantCandidateSamples, 1, 1);
                IMAGE_QUALITY_OPTION(ImGui::InputInt("Full samples", &m_ui.NEEDistantFullSamples, 1));                          
                m_ui.NEEDistantFullSamples = dm::clamp(m_ui.NEEDistantFullSamples, 0, 256);
                ImGui::Unindent(indent);
                ImGui::PopID();

                ImGui::Separator();

                ImGui::Text("Local light importance sampling:");
                ImGui::Indent(indent);
                ImGui::PushID("Local");
                IMAGE_QUALITY_OPTION(ImGui::Combo("Approach", (int*)&m_ui.NEELocalType, "Uniform\0Power (pre-sampling)\0ReGIR\0\0"));   
                m_ui.NEELocalType = dm::clamp(m_ui.NEELocalType, 0, 2);
                if (ImGui::IsItemHovered()) ImGui::SetTooltip("These are combined emissive triangles and analytic lights.\nNote: Some ReGIR internal settings are still controlled by ReSTIR DI settings");
                IMAGE_QUALITY_OPTION(ImGui::InputInt("Candidate samples", &m_ui.NEELocalCandidateSamples, 1));                  
                m_ui.NEELocalCandidateSamples = dm::clamp(m_ui.NEELocalCandidateSamples, 1, 16);
                IMAGE_QUALITY_OPTION(ImGui::InputInt("Full samples", &m_ui.NEELocalFullSamples, 1));                            
                m_ui.NEELocalFullSamples = dm::clamp(m_ui.NEELocalFullSamples, 0, 256);
                ImGui::Unindent(indent);
                ImGui::PopID();

                ImGui::Separator();

                IMAGE_QUALITY_OPTION(ImGui::InputInt("Boost samples on dominant surface", &m_ui.NEEBoostSamplingOnDominantPlane, 1));                            
                if (ImGui::IsItemHovered()) ImGui::SetTooltip("Multiplier for the number of distant and local samples on\nthe dominant denoising surface (usually primary surface).\nThis setting is ignored when RTXDI is enabled.\nThis setting is ignored in Reference mode.\nThis approach could be extended to allow per-material boosting for tricky surfaces.");
                m_ui.NEEBoostSamplingOnDominantPlane = std::clamp(m_ui.NEEBoostSamplingOnDominantPlane, 1, 8);

                IMAGE_QUALITY_OPTION(ImGui::InputFloat("Minimum radiance threshold", &m_ui.NEEMinRadianceThresholdMul, 0.0f, 0.0f, "%.4f"));
                if (ImGui::IsItemHovered()) ImGui::SetTooltip("This will skip samples under certain radiance threshold (in multiples \nof pre-exposed gray luminance). Helps performance.");
                m_ui.NEEMinRadianceThresholdMul = dm::clamp(m_ui.NEEMinRadianceThresholdMul, 0.0f, 1000.0f);

                ImGui::Separator();

                IMAGE_QUALITY_OPTION(ImGui::Checkbox("Suppress Primary NEE", &m_ui.SuppressPrimaryNEE));
                if (ImGui::IsItemHovered()) ImGui::SetTooltip("NOTE: works only in realtime mode at the moment!");
            }
        }

        ImGui::Unindent(indent);
    }

    if( m_ui.RealtimeMode && m_ui.RealtimeAA != 0 && ImGui::CollapsingHeader("Anti-Aliasing and upscaling") )
    {
        ImGui::Combo("AA Camera Jitter", (int*)&m_ui.TemporalAntiAliasingJitter, "MSAA\0Halton\0R2\0White Noise\0");
        ImGui::Separator();
        if (m_ui.RealtimeAA == 1)
        {
            ImGui::Text("Basic TAA settings:");
            ImGui::Checkbox("TAA History Clamping", &m_ui.TemporalAntiAliasingParams.enableHistoryClamping);
            ImGui::SliderFloat("TAA New Frame Weight", &m_ui.TemporalAntiAliasingParams.newFrameWeight, 0.001f, 1.0f);
            ImGui::Checkbox("TAA Use Clamp Relax", &m_ui.TemporalAntiAliasingParams.useHistoryClampRelax);
        }
#ifdef STREAMLINE_INTEGRATION
        if (m_ui.RealtimeAA == 2)
        {
            ImGui::Text("DLSS settings:");
            ImGui::Combo("DLSS Mode", (int*)&m_ui.DLSS_Mode, "Off\0Performance\0Balanced\0Quality\0Ultra-Performance\0");
            m_ui.DLSS_Mode = dm::clamp(m_ui.DLSS_Mode, (sl::DLSSMode)0, (sl::DLSSMode)4);
        }
        if (m_ui.RealtimeAA == 3)
        {
            ImGui::Text("DLAA settings (no settings)");
            m_ui.DLSS_Mode = sl::DLSSMode::eDLAA;
        }
#endif    
    }

    if ((m_ui.ActualUseReSTIRDI() || m_ui.NEELocalType == 2)  && ImGui::CollapsingHeader("ReGIR"))
    {
		ImGui::Indent(indent);
		ImGui::PushItemWidth(defItemWidth);
       
		IMAGE_QUALITY_OPTION(ImGui::InputInt("Number of Build Samples", (int*)&m_ui.RTXDI.regir.regirDynamicParameters.regirNumBuildSamples));
		m_ui.RTXDI.regir.regirDynamicParameters.regirNumBuildSamples = dm::clamp(m_ui.RTXDI.regir.regirDynamicParameters.regirNumBuildSamples, 0u, 128u);
        IMAGE_QUALITY_OPTION(ImGui::SliderFloat("Cell Size", &m_ui.RTXDI.regir.regirDynamicParameters.regirCellSize, 0.1f, 2.f));
        IMAGE_QUALITY_OPTION(ImGui::SliderFloat("Sampling Jitter", &m_ui.RTXDI.regir.regirDynamicParameters.regirSamplingJitter, 0.f, 1.f));

        ImGui::PopItemWidth();
        ImGui::Unindent(indent);
    }

    if( m_ui.ActualUseReSTIRDI() && ImGui::CollapsingHeader("ReSTIR DI")) //, ImGuiTreeNodeFlags_DefaultOpen))
    {
        ImGui::Indent(indent);
        ImGui::PushItemWidth(defItemWidth);

        IMAGE_QUALITY_OPTION(ImGui::Combo("Resampling Mode", (int*)&m_ui.RTXDI.restirDI.resamplingMode,
            "Disabled\0Temporal\0Spatial\0Temporal & Spatial\0Fused\0\0"));
       
        IMAGE_QUALITY_OPTION(ImGui::Combo("Spatial Bias Correction", (int*)&m_ui.RTXDI.restirDI.spatialResamplingParams.spatialBiasCorrection,
            "Off\0Basic\0Pairwise\0Ray Traced\0\0"));
		
        IMAGE_QUALITY_OPTION(ImGui::Combo("Temporal Bias Correction", (int*)&m_ui.RTXDI.restirDI.temporalResamplingParams.temporalBiasCorrection,
            "Off\0Basic\0Pairwise\0Ray Traced\0\0"));
		
		IMAGE_QUALITY_OPTION(ImGui::Combo("Local Light Sampling Mode", (int*)&m_ui.RTXDI.restirDI.initialSamplingParams.localLightSamplingMode,
			"Uniform\0Power RIS\0ReGIR RIS\0\0"));

        if (m_ui.RTXDI.restirDI.initialSamplingParams.localLightSamplingMode == ReSTIRDI_LocalLightSamplingMode::ReGIR_RIS)
        {
            IMAGE_QUALITY_OPTION(ImGui::Combo("ReGIR Mode", (int*)&m_ui.RTXDI.regir.regirStaticParams.Mode,
                "Disabled\0Grid\0Onion\0\0"));
        }
        
        ImGui::PopItemWidth();

        ImGui::PushItemWidth(defItemWidth*0.8f);
            
        ImGui::Text("Number of Primary Samples: ");
        ImGui::Indent(indent);

        IMAGE_QUALITY_OPTION(ImGui::InputInt("ReGir", (int*)&m_ui.RTXDI.regir.regirDynamicParameters.regirNumBuildSamples));
        m_ui.RTXDI.regir.regirDynamicParameters.regirNumBuildSamples = dm::clamp(m_ui.RTXDI.regir.regirDynamicParameters.regirNumBuildSamples, 0u, 32u);
        IMAGE_QUALITY_OPTION(ImGui::InputInt("Local Light", (int*)&m_ui.RTXDI.restirDI.initialSamplingParams.numPrimaryLocalLightSamples));
		m_ui.RTXDI.restirDI.initialSamplingParams.numPrimaryLocalLightSamples = dm::clamp(m_ui.RTXDI.restirDI.initialSamplingParams.numPrimaryLocalLightSamples, 0u, 32u);
        IMAGE_QUALITY_OPTION(ImGui::InputInt("BRDF", (int*)&m_ui.RTXDI.restirDI.initialSamplingParams.numPrimaryBrdfSamples));
		m_ui.RTXDI.restirDI.initialSamplingParams.numPrimaryBrdfSamples = dm::clamp(m_ui.RTXDI.restirDI.initialSamplingParams.numPrimaryBrdfSamples, 0u, 32u);
        IMAGE_QUALITY_OPTION(ImGui::InputInt("Infinite Light", (int*)&m_ui.RTXDI.restirDI.initialSamplingParams.numPrimaryInfiniteLightSamples));
		m_ui.RTXDI.restirDI.initialSamplingParams.numPrimaryInfiniteLightSamples = dm::clamp(m_ui.RTXDI.restirDI.initialSamplingParams.numPrimaryInfiniteLightSamples, 0u, 32u);
        IMAGE_QUALITY_OPTION(ImGui::InputInt("Environment Light", (int*)&m_ui.RTXDI.restirDI.initialSamplingParams.numPrimaryEnvironmentSamples));
		m_ui.RTXDI.restirDI.initialSamplingParams.numPrimaryEnvironmentSamples = dm::clamp(m_ui.RTXDI.restirDI.initialSamplingParams.numPrimaryEnvironmentSamples, 0u, 32u);

        ImGui::Unindent(indent);
    
        if (ImGui::CollapsingHeader("Fine Tuning"))
        {
            ImGui::Indent(indent);
            ImGui::PushItemWidth(defItemWidth);
            IMAGE_QUALITY_OPTION(ImGui::SliderFloat("BRDF Cut-off", &m_ui.RTXDI.restirDI.initialSamplingParams.brdfCutoff, 0.0f, 1.0f));
            ImGui::Separator();
            IMAGE_QUALITY_OPTION(ImGui::Checkbox("Use Permutation Sampling", (bool*)&m_ui.RTXDI.restirDI.temporalResamplingParams.enablePermutationSampling));
            IMAGE_QUALITY_OPTION(ImGui::SliderFloat("Temporal Depth Threshold", &m_ui.RTXDI.restirDI.temporalResamplingParams.temporalDepthThreshold, 0.f, 1.f));
            IMAGE_QUALITY_OPTION(ImGui::SliderFloat("Temporal Normal Threshold", &m_ui.RTXDI.restirDI.temporalResamplingParams.temporalNormalThreshold, 0.f, 1.f));
			IMAGE_QUALITY_OPTION(ImGui::SliderFloat("Boling Filter Strength", &m_ui.RTXDI.restirDI.temporalResamplingParams.boilingFilterStrength, 0.f, 1.f));
            ImGui::Separator();
            IMAGE_QUALITY_OPTION(ImGui::SliderInt("Spatial Samples", (int*)&m_ui.RTXDI.restirDI.spatialResamplingParams.numSpatialSamples, 0, 8));
			IMAGE_QUALITY_OPTION(ImGui::SliderInt("Disocclusion Samples", (int*)&m_ui.RTXDI.restirDI.spatialResamplingParams.numDisocclusionBoostSamples, 0, 8));
            IMAGE_QUALITY_OPTION(ImGui::SliderFloat("Spatial Sampling Radius", &m_ui.RTXDI.restirDI.spatialResamplingParams.spatialSamplingRadius, 0.f, 64.f));
            IMAGE_QUALITY_OPTION(ImGui::SliderFloat("Spatial Depth Threshold", &m_ui.RTXDI.restirDI.spatialResamplingParams.spatialDepthThreshold, 0.f, 1.f));
            IMAGE_QUALITY_OPTION(ImGui::SliderFloat("Spatial Normal Threshold", &m_ui.RTXDI.restirDI.spatialResamplingParams.spatialNormalThreshold, 0.f, 1.f));
			IMAGE_QUALITY_OPTION(ImGui::Checkbox("Discount Naive Samples", (bool*)&m_ui.RTXDI.restirDI.spatialResamplingParams.discountNaiveSamples));
			if (ImGui::IsItemHovered()) ImGui::SetTooltip("Prevents samples which are from the current frame or have no reasonable temporal history merged being spread to neighbors");
            ImGui::Separator();
            IMAGE_QUALITY_OPTION(ImGui::DragFloat("Ray Epsilon", &m_ui.RTXDI.rayEpsilon, 0.0001f, 0.0001f, 0.01f, "%.4f"));
            ImGui::PopItemWidth();
            ImGui::Unindent(indent);
        }

        ImGui::PopItemWidth();
        ImGui::Unindent(indent);
    }

    if (m_ui.ActualUseReSTIRGI() && ImGui::CollapsingHeader("ReSTIR GI"))
    {
        ImGui::Indent(indent);
        ImGui::PushItemWidth(defItemWidth);
		IMAGE_QUALITY_OPTION(ImGui::Combo("Resampling Mode", (int*)&m_ui.RTXDI.restirGI.resamplingMode,
			"Disabled\0Temporal\0Spatial\0Temporal & Spatial\0Fused\0\0"));
        ImGui::Separator();
        IMAGE_QUALITY_OPTION(ImGui::SliderInt("History Length ##GI", (int*)&m_ui.RTXDI.restirGI.temporalResamplingParams.maxHistoryLength, 0, 64));
        IMAGE_QUALITY_OPTION(ImGui::SliderInt("Max Reservoir Age ##GI", (int*)&m_ui.RTXDI.restirGI.temporalResamplingParams.maxReservoirAge, 0, 100));
        IMAGE_QUALITY_OPTION(ImGui::Checkbox("Permutation Sampling ##GI", (bool*)&m_ui.RTXDI.restirGI.temporalResamplingParams.enablePermutationSampling));
        IMAGE_QUALITY_OPTION(ImGui::Checkbox("Fallback Sampling ##GI", (bool*)&m_ui.RTXDI.restirGI.temporalResamplingParams.enableFallbackSampling));
        IMAGE_QUALITY_OPTION(ImGui::SliderFloat("Boling Filter Strength##GI", &m_ui.RTXDI.restirGI.temporalResamplingParams.boilingFilterStrength, 0.f, 1.f));
        IMAGE_QUALITY_OPTION(ImGui::Combo("Temporal Bias Correction ##GI", (int*)&m_ui.RTXDI.restirGI.temporalResamplingParams.temporalBiasCorrectionMode,
            "Off\0Basic\0Ray Traced\0"));
        ImGui::Separator();
        IMAGE_QUALITY_OPTION(ImGui::SliderInt("Spatial Samples ##GI", (int*)&m_ui.RTXDI.restirGI.spatialResamplingParams.numSpatialSamples, 0, 8));
        IMAGE_QUALITY_OPTION(ImGui::SliderFloat("Spatial Sampling Radius ##GI", &m_ui.RTXDI.restirGI.spatialResamplingParams.spatialSamplingRadius, 1.f, 64.f));
        IMAGE_QUALITY_OPTION(ImGui::Combo("Spatial Bias Correction ##GI", (int*)&m_ui.RTXDI.restirGI.spatialResamplingParams.spatialBiasCorrectionMode, "Off\0Basic\0Ray Traced\0"));
        ImGui::Separator();
        IMAGE_QUALITY_OPTION(ImGui::Checkbox("Final Visibility ##GI", (bool*)&m_ui.RTXDI.restirGI.finalShadingParams.enableFinalVisibility));
        IMAGE_QUALITY_OPTION(ImGui::Checkbox("Final MIS ##GI", (bool*)&m_ui.RTXDI.restirGI.finalShadingParams.enableFinalMIS));

        ImGui::PopItemWidth();
        ImGui::Unindent(indent);
    }

    if (m_ui.ActualUseStablePlanes() && ImGui::CollapsingHeader("Stable Planes"))
    {
        ImGui::InputInt("Active stable planes", &m_ui.StablePlanesActiveCount);
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("How many stable planes to allow - 1 is just standard denoising");
        m_ui.StablePlanesActiveCount = dm::clamp(m_ui.StablePlanesActiveCount, 1, (int)cStablePlaneCount);
        ImGui::InputInt("Max stable plane vertex depth", &m_ui.StablePlanesMaxVertexDepth);
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("How deep the stable part of path tracing can go");
        m_ui.StablePlanesMaxVertexDepth = dm::clamp(m_ui.StablePlanesMaxVertexDepth, 2, (int)cStablePlaneMaxVertexIndex);
        ImGui::SliderFloat("Path split stop threshold", &m_ui.StablePlanesSplitStopThreshold, 0.0f, 2.0f);
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Stops splitting if more than this threshold throughput will be on a non-taken branch.\nActual threshold is this value divided by vertexIndex.");
        ImGui::SliderFloat("Min denoising roughness", &m_ui.StablePlanesMinRoughness, 0.0f, 0.3f);
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Artificially increase roughness that denoiser sees, for radiance that \nfalls through on delta surfaces. Attenuates the noise at the expense of sharpness.");
        ImGui::Checkbox("Primary Surface Replacement", &m_ui.AllowPrimarySurfaceReplacement);
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("When stable planes enabled, whether we can use PSR for the first (base) plane");
        ImGui::Checkbox("Suppress primary plane noisy specular", &m_ui.StablePlanesSuppressPrimaryIndirectSpecular);
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("This will suppress noisy specular to primary stable plane by specified amount\nbut only if at least 1 stable plane is also used on the same pixel.\nThis for ex. reduces secondary internal smudgy reflections from internal many bounces in a window.");
        ImGui::SliderFloat("Suppress primary plane noisy specular amount", &m_ui.StablePlanesSuppressPrimaryIndirectSpecularK, 0.0f, 1.0f);
        ImGui::SliderFloat("Non-primary plane anti-aliasing fallthrough", &m_ui.StablePlanesAntiAliasingFallthrough, 0.0f, 1.0f);
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Divert some radiance on highly curved and edge areas from non-0 plane back\nto plane 0. This reduces aliasing on complex boundary bounces.");
    }

    if (m_ui.RealtimeMode && m_ui.RealtimeDenoiser && ImGui::CollapsingHeader("Denoising"))
    {
        ImGui::Indent(indent);

        ImGui::InputFloat("Disocclusion Threshold", &m_ui.NRDDisocclusionThreshold);
        ImGui::Checkbox("Use Alternate Disocclusion Threshold Mix", &m_ui.NRDUseAlternateDisocclusionThresholdMix);
        ImGui::InputFloat("Disocclusion Threshold Alt", &m_ui.NRDDisocclusionThresholdAlternate);
        ImGui::InputFloat("Radiance clamping", &m_ui.DenoiserRadianceClampK);

        ImGui::Separator();

        m_ui.NRDModeChanged = ImGui::Combo("Denoiser Mode", (int*)&m_ui.NRDMethod, "REBLUR\0RELAX\0\0");
        m_ui.NRDMethod = dm::clamp(m_ui.NRDMethod, (NrdConfig::DenoiserMethod)0, (NrdConfig::DenoiserMethod)1);

        if (ImGui::CollapsingHeader("Advanced Settings"))
        {
            if (m_ui.NRDMethod == NrdConfig::DenoiserMethod::REBLUR)
            {
                // TODO: make sure these are updated to constants
                ImGui::SliderFloat("Hit Distance A", &m_ui.ReblurSettings.hitDistanceParameters.A, 0.0f, 10.0f);
                ImGui::SliderFloat("Hit Distance B", &m_ui.ReblurSettings.hitDistanceParameters.B, 0.0f, 10.0f);
                ImGui::SliderFloat("Hit Distance C", &m_ui.ReblurSettings.hitDistanceParameters.C, 0.0f, 50.0f);
                ImGui::SliderFloat("Hit Distance D", &m_ui.ReblurSettings.hitDistanceParameters.D, -50.0f, 0.0f);

                ImGui::SliderFloat("Antilag Luminance Sigma Scale", &m_ui.ReblurSettings.antilagSettings.luminanceSigmaScale, 1.0f, 3.0f);
                ImGui::SliderFloat("Antilag Hit Distance Sigma Scale", &m_ui.ReblurSettings.antilagSettings.hitDistanceSigmaScale, 1.0f, 3.0f);
                ImGui::SliderFloat("Antilag Luminance Antilag Power", &m_ui.ReblurSettings.antilagSettings.luminanceAntilagPower, 0.001f, 1.0f);
                ImGui::SliderFloat("Antilag Hit Distance Antilag Power", &m_ui.ReblurSettings.antilagSettings.hitDistanceAntilagPower, 0.001f, 1.0f);

                ImGui::SliderInt("Max Accumulated Frames", (int*)&m_ui.ReblurSettings.maxAccumulatedFrameNum, 0, nrd::REBLUR_MAX_HISTORY_FRAME_NUM);
                ImGui::SliderInt("Fast Max Accumulated Frames", (int*)&m_ui.ReblurSettings.maxFastAccumulatedFrameNum, 0, nrd::REBLUR_MAX_HISTORY_FRAME_NUM);
                ImGui::SliderInt("History Fix Frames", (int*)&m_ui.ReblurSettings.historyFixFrameNum, 0, nrd::REBLUR_MAX_HISTORY_FRAME_NUM);

                ImGui::SliderFloat("Diffuse Prepass Blur Radius (pixels)", &m_ui.ReblurSettings.diffusePrepassBlurRadius, 0.0f, 100.0f);
                ImGui::SliderFloat("Specular Prepass Blur Radius (pixels)", &m_ui.ReblurSettings.specularPrepassBlurRadius, 0.0f, 100.0f);
                ImGui::SliderFloat("Blur Radius (pixels)", &m_ui.ReblurSettings.blurRadius, 0.0f, 100.0f);

                ImGui::SliderFloat("Base Stride Between Samples (pixels)", &m_ui.ReblurSettings.historyFixStrideBetweenSamples, 0.0f, 30.0f);

                ImGui::SliderFloat("Lobe Angle Fraction", &m_ui.ReblurSettings.lobeAngleFraction, 0.0f, 1.0f);
                ImGui::SliderFloat("Roughness Fraction", &m_ui.ReblurSettings.roughnessFraction, 0.0f, 1.0f);

                ImGui::SliderFloat("Accumulation Roughness Threshold", &m_ui.ReblurSettings.responsiveAccumulationRoughnessThreshold, 0.0f, 1.0f);

                ImGui::SliderFloat("Stabilization Strength", &m_ui.ReblurSettings.stabilizationStrength, 0.0f, 1.0f);

                ImGui::SliderFloat("Plane Distance Sensitivity", &m_ui.ReblurSettings.planeDistanceSensitivity, 0.0f, 1.0f);

                // ImGui::Combo("Checkerboard Mode", (int*)&m_ui.ReblurSettings.checkerboardMode, "Off\0Black\0White\0\0");

                // these are uint8_t and ImGUI takes a ptr to int32_t :(
                int hitDistanceReconstructionMode = (int)m_ui.ReblurSettings.hitDistanceReconstructionMode;
                ImGui::Combo("Hit Distance Reconstruction Mode", &hitDistanceReconstructionMode, "Off\0AREA_3X3\0AREA_5X5\0\0");
                m_ui.ReblurSettings.hitDistanceReconstructionMode = (nrd::HitDistanceReconstructionMode)hitDistanceReconstructionMode;

                ImGui::Checkbox("Enable Firefly Filter", &m_ui.ReblurSettings.enableAntiFirefly);

                ImGui::Checkbox("Enable Performance Mode", &m_ui.ReblurSettings.enablePerformanceMode);

                ImGui::Checkbox("Enable Diffuse Material Test", &m_ui.ReblurSettings.enableMaterialTestForDiffuse);
                ImGui::Checkbox("Enable Specular Material Test", &m_ui.ReblurSettings.enableMaterialTestForSpecular);
            }
            else // m_ui.NRDMethod == NrdConfig::DenoiserMethod::RELAX
            {
                ImGui::SliderFloat("Diffuse Prepass Blur Radius", &m_ui.RelaxSettings.diffusePrepassBlurRadius, 0.0f, 100.0f);
                ImGui::SliderFloat("Specular Prepass Blur Radius", &m_ui.RelaxSettings.specularPrepassBlurRadius, 0.0f, 100.0f);

                ImGui::SliderInt("Diffuse Max Accumulated Frames", (int*)&m_ui.RelaxSettings.diffuseMaxAccumulatedFrameNum, 0, nrd::RELAX_MAX_HISTORY_FRAME_NUM);
                ImGui::SliderInt("Specular Max Accumulated Frames", (int*)&m_ui.RelaxSettings.specularMaxAccumulatedFrameNum, 0, nrd::RELAX_MAX_HISTORY_FRAME_NUM);

                ImGui::SliderInt("Diffuse Fast Max Accumulated Frames", (int*)&m_ui.RelaxSettings.diffuseMaxFastAccumulatedFrameNum, 0, nrd::RELAX_MAX_HISTORY_FRAME_NUM);
                ImGui::SliderInt("Specular Fast Max Accumulated Frames", (int*)&m_ui.RelaxSettings.specularMaxFastAccumulatedFrameNum, 0, nrd::RELAX_MAX_HISTORY_FRAME_NUM);

                ImGui::SliderInt("History Fix Frame Num", (int*)&m_ui.RelaxSettings.historyFixFrameNum, 0, nrd::RELAX_MAX_HISTORY_FRAME_NUM);

                ImGui::SliderFloat("Diffuse Edge Stopping Sensitivity", &m_ui.RelaxSettings.diffusePhiLuminance, 0.0f, 10.0f);
                ImGui::SliderFloat("Specular Edge Stopping Sensitivity", &m_ui.RelaxSettings.specularPhiLuminance, 0.0f, 10.0f);

                ImGui::SliderFloat("Diffuse Lobe Angle Fraction", &m_ui.RelaxSettings.diffuseLobeAngleFraction, 0.0f, 1.0f);
                ImGui::SliderFloat("Specular Lobe Angle Fraction", &m_ui.RelaxSettings.specularLobeAngleFraction, 0.0f, 1.0f);

                ImGui::SliderFloat("Roughness Fraction", &m_ui.RelaxSettings.roughnessFraction, 0.0f, 1.0f);

                ImGui::SliderFloat("Specular Variance Boost", &m_ui.RelaxSettings.specularVarianceBoost, 0.0f, 1.0f);

                ImGui::SliderFloat("Specular Lobe Angle Slack", &m_ui.RelaxSettings.specularLobeAngleSlack, 0.0f, 1.0f);

                ImGui::SliderFloat("Base Stride Between Samples (pixels)", &m_ui.RelaxSettings.historyFixStrideBetweenSamples, 0.0f, 30.0f);

                ImGui::SliderFloat("Normal Edge Stopping Power", &m_ui.RelaxSettings.historyFixEdgeStoppingNormalPower, 0.0f, 30.0f);

                ImGui::SliderFloat("Clamping Color Box Sigma Scale", &m_ui.RelaxSettings.historyClampingColorBoxSigmaScale, 0.0f, 3.0f);

                ImGui::SliderInt("Spatial Variance Estimation History Threshold", (int*)&m_ui.RelaxSettings.spatialVarianceEstimationHistoryThreshold, 0, nrd::RELAX_MAX_HISTORY_FRAME_NUM);

                ImGui::SliderInt("Number of Atrous iterations", (int*)&m_ui.RelaxSettings.atrousIterationNum, 2, 8);

                ImGui::SliderFloat("Diffuse Min Luminance Weight", &m_ui.RelaxSettings.diffuseMinLuminanceWeight, 0.0f, 1.0f);
                ImGui::SliderFloat("Specular Min Luminance Weight", &m_ui.RelaxSettings.specularMinLuminanceWeight, 0.0f, 1.0f);

                ImGui::SliderFloat("Edge Stopping Threshold", &m_ui.RelaxSettings.depthThreshold, 0.0f, 0.1f);

                ImGui::SliderFloat("Confidence: Relaxation Multiplier", &m_ui.RelaxSettings.confidenceDrivenRelaxationMultiplier, 0.0f, 1.0f);
                ImGui::SliderFloat("Confidence: Luminance Edge Stopping Relaxation", &m_ui.RelaxSettings.confidenceDrivenLuminanceEdgeStoppingRelaxation, 0.0f, 1.0f);
                ImGui::SliderFloat("Confidence: Normal Edge Stopping Relaxation", &m_ui.RelaxSettings.confidenceDrivenNormalEdgeStoppingRelaxation, 0.0f, 1.0f);

                ImGui::SliderFloat("Luminance Edge Stopping Relaxation", &m_ui.RelaxSettings.luminanceEdgeStoppingRelaxation, 0.0f, 1.0f);
                ImGui::SliderFloat("Normal Edge Stopping Relaxation", &m_ui.RelaxSettings.normalEdgeStoppingRelaxation, 0.0f, 1.0f);

                ImGui::SliderFloat("Roughness Edge Stopping Relaxation", &m_ui.RelaxSettings.roughnessEdgeStoppingRelaxation, 0.0f, 5.0f);

                ImGui::SliderFloat("Antilag Acceleration Amount", &m_ui.RelaxSettings.antilagSettings.accelerationAmount, 0.0f, 1.0f);
                ImGui::SliderFloat("Antilag Spatial Sigma Scale", &m_ui.RelaxSettings.antilagSettings.spatialSigmaScale, 0.0f, 5.0f);
                ImGui::SliderFloat("Antilag Temporal Sigma Scale", &m_ui.RelaxSettings.antilagSettings.temporalSigmaScale, 0.0f, 5.0f);
                ImGui::SliderFloat("Antilag Reset Amount", &m_ui.RelaxSettings.antilagSettings.resetAmount, 0.0f, 1.0f);

                // ImGui::Combo("Checkerboard Mode", (int*)&m_ui.RelaxSettings.checkerboardMode, "Off\0Black\0White\0\0");

                int hitDistanceReconstructionMode = (int)m_ui.RelaxSettings.hitDistanceReconstructionMode;  // these are uint8_t and ImGUI takes a ptr to int32_t :(
                ImGui::Combo("Hit Distance Reconstruction Mode", &hitDistanceReconstructionMode, "Off\0AREA_3X3\0AREA_5X5\0\0");
                m_ui.RelaxSettings.hitDistanceReconstructionMode = (nrd::HitDistanceReconstructionMode)hitDistanceReconstructionMode;

                ImGui::Checkbox("Enable Firefly Filter", &m_ui.RelaxSettings.enableAntiFirefly);

                ImGui::Checkbox("Enable Reprojection Test Skipping Without Motion", &m_ui.RelaxSettings.enableReprojectionTestSkippingWithoutMotion);

                ImGui::Checkbox("Roughness Edge Stopping", &m_ui.RelaxSettings.enableRoughnessEdgeStopping);

                ImGui::Checkbox("Enable Diffuse Material Test", &m_ui.RelaxSettings.enableMaterialTestForDiffuse);
                ImGui::Checkbox("Enable Specular Material Test", &m_ui.RelaxSettings.enableMaterialTestForSpecular);
            }

            // Not really needed for now since we have reference codepath, but it could be used to debug some of the NRD codepaths so leaving in as a reminder
            // ImGui::Checkbox("Reference Accumulation", &m_ui.NRDReferenceSettings.maxAccumulatedFrameNum);
        }

        ImGui::Unindent(indent);
    }

    if (ImGui::CollapsingHeader("Opacity Micro-Maps"))
    {
        UI_SCOPED_INDENT(indent);

        if (!m_OMMSupported)
        {
            ImGui::Text("<Opacity Micro-Maps not supported on the current device>");
        }

        {
            UI_SCOPED_DISABLE(!m_OMMSupported);

            if (ImGui::Checkbox("Enable", &m_ui.OpacityMicroMaps.Enable))
            {
                m_ui.ResetAccumulation = true;
            }

            {
                {
                    UI_SCOPED_DISABLE(m_ui.OpacityMicroMaps.ActiveState.has_value() && m_ui.OpacityMicroMaps.ActiveState->Format != nvrhi::rt::OpacityMicromapFormat::OC1_4_State);
                    if (ImGui::Checkbox("Force 2 State", &m_ui.OpacityMicroMaps.Force2State))
                    {
                        m_ui.ResetAccumulation = true;
                    }
                    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Will force 2-State via TLAS instance mask.");
                }

                {
                    if (ImGui::Checkbox("Render ONLY OMMs", &m_ui.OpacityMicroMaps.OnlyOMMs))
                    {
                        m_ui.ResetAccumulation = true;
                    }
                }

                ImGui::Separator();
                ImGui::Text("Bake Settings (Require Rebuild to take effect)");

                if (m_ui.OpacityMicroMaps.BuildsLeftInQueue != 0)
                {
                    const float progress = (1.f - (float)m_ui.OpacityMicroMaps.BuildsLeftInQueue / m_ui.OpacityMicroMaps.BuildsQueued);
                    std::stringstream ss;
                    ss << "Build progress: " << (uint32_t)(100.f * progress) << "%";
                    std::string str = ss.str();
                    ImGui::ProgressBar(progress, ImVec2(-FLT_MIN, 0), str.c_str());
    }

                {
                    UI_SCOPED_DISABLE(m_ui.OpacityMicroMaps.ActiveState.has_value() && m_ui.OpacityMicroMaps.ActiveState == m_ui.OpacityMicroMaps.DesiredState);
                    if (ImGui::Button("Trigger Rebuild"))
                    {
                        m_ui.OpacityMicroMaps.TriggerRebuild = true;
                    }
                }

                {
                    ImGui::Checkbox("Dynamic subdivision level", &m_ui.OpacityMicroMaps.DesiredState.EnableDynamicSubdivision);
                }

                {
                    UI_SCOPED_DISABLE(!m_ui.OpacityMicroMaps.DesiredState.EnableDynamicSubdivision);
                    ImGui::SliderFloat("Dynamic subdivision scale", &m_ui.OpacityMicroMaps.DesiredState.DynamicSubdivisionScale, 0.01f, 20.f, "%.1f", ImGuiSliderFlags_Logarithmic);
                }

                {
                    const int MaxSubdivisionLevel = m_ui.OpacityMicroMaps.DesiredState.ComputeOnly ? 12 : 10;
                    m_ui.OpacityMicroMaps.DesiredState.MaxSubdivisionLevel = std::clamp(m_ui.OpacityMicroMaps.DesiredState.MaxSubdivisionLevel, 1, MaxSubdivisionLevel);
                    ImGui::SliderInt("Max subdivision level", &m_ui.OpacityMicroMaps.DesiredState.MaxSubdivisionLevel, 1, MaxSubdivisionLevel, "%.3f", ImGuiSliderFlags_AlwaysClamp);
                }

                {
                    std::array<const char*, 3> formatNames =
                    {
                        "None",
                        "Fast Trace",
                        "Fast Build"
                    };

                    std::array<nvrhi::rt::OpacityMicromapBuildFlags, 3> formats =
                    {
                        nvrhi::rt::OpacityMicromapBuildFlags::None,
                        nvrhi::rt::OpacityMicromapBuildFlags::FastTrace,
                        nvrhi::rt::OpacityMicromapBuildFlags::FastBuild
                    };

                    if (ImGui::BeginCombo("Flag", formatNames[(uint32_t)m_ui.OpacityMicroMaps.DesiredState.Flag]))
                    {
                        for (uint i = 0; i < formats.size(); i++)
                        {
                            bool is_selected = formats[i] == m_ui.OpacityMicroMaps.DesiredState.Flag;
                            if (ImGui::Selectable(formatNames[i], is_selected))
                                m_ui.OpacityMicroMaps.DesiredState.Flag = formats[i];
                            if (is_selected)
                                ImGui::SetItemDefaultFocus();
                        }
                        ImGui::EndCombo();
                    }
                }

                {
                    auto FormatToString = [ ](nvrhi::rt::OpacityMicromapFormat format) {
                        assert(format == nvrhi::rt::OpacityMicromapFormat::OC1_2_State || format == nvrhi::rt::OpacityMicromapFormat::OC1_4_State);
                        return format == nvrhi::rt::OpacityMicromapFormat::OC1_2_State ? "2-State" : "4-State";
                    };
                    std::array<nvrhi::rt::OpacityMicromapFormat, 2> formats = { nvrhi::rt::OpacityMicromapFormat::OC1_2_State, nvrhi::rt::OpacityMicromapFormat::OC1_4_State };
                    if (ImGui::BeginCombo("Format", FormatToString(m_ui.OpacityMicroMaps.DesiredState.Format)))
                    {
                        for (uint i = 0; i < formats.size(); i++)
                        {
                            bool is_selected = formats[i] == m_ui.OpacityMicroMaps.DesiredState.Format;
                            if (ImGui::Selectable(FormatToString(formats[i]), is_selected))
                                m_ui.OpacityMicroMaps.DesiredState.Format = formats[i];
                            if (is_selected)
                                ImGui::SetItemDefaultFocus();
                        }
                        ImGui::EndCombo();
                    }
                }

                if (ImGui::CollapsingHeader("Debug Settings"))
                {
                    UI_SCOPED_INDENT(indent);

                    {
                        ImGui::Checkbox("Compute Only", &m_ui.OpacityMicroMaps.DesiredState.ComputeOnly);
                    }

                    {
                        ImGui::Checkbox("Enable \"Level Line Intersection\"", &m_ui.OpacityMicroMaps.DesiredState.LevelLineIntersection);
                    }

                    {
                        ImGui::Checkbox("Enable TexCoord deduplication", &m_ui.OpacityMicroMaps.DesiredState.EnableTexCoordDeduplication);
                    }

                    {
                        ImGui::Checkbox("Force 32-bit indices", &m_ui.OpacityMicroMaps.DesiredState.Force32BitIndices);
                    }

                    {
                        ImGui::Checkbox("Enable Special Indices", &m_ui.OpacityMicroMaps.DesiredState.EnableSpecialIndices);
                    }

                    {
                        ImGui::SliderInt("Max memory per OMM", &m_ui.OpacityMicroMaps.DesiredState.MaxOmmArrayDataSizeInMB, 1, 1000, "%dMB", ImGuiSliderFlags_Logarithmic);
                    }

                    {
                        ImGui::Checkbox("Enable NSight debug mode", &m_ui.OpacityMicroMaps.DesiredState.EnableNsightDebugMode);
                    }
                }

                ImGui::Separator();
                ImGui::Text("Stats");

                {
                    std::stringstream ss;
                    ss << m_ui.OpacityMicroMaps.BuildsQueued << " active OMMs";
                    std::string str = ss.str();
                    ImGui::Text(str.c_str());

                    if (ImGui::CollapsingHeader("Bake Stats"))
                    {
                        UI_SCOPED_INDENT(indent);

                        for (const std::shared_ptr<donut::engine::MeshInfo>& mesh : m_app.GetScene()->GetSceneGraph()->GetMeshes())
                        {
                            bool meshHasOmms = false;
                            for (uint32_t i = 0; i < mesh->geometries.size(); ++i)
                            {
                                if (mesh->geometries[i]->debugData.ommIndexBufferOffset != 0xFFFFFFFF)
                                {
                                    meshHasOmms = true;
                                    break;
                                }
                            }

                            if (!meshHasOmms)
                                continue;

                            ImGui::Text(mesh->name.c_str());

                            {
                                UI_SCOPED_INDENT(indent);

                                for (uint32_t i = 0; i < mesh->geometries.size(); ++i)
                                {
                                    if (mesh->geometries[i]->debugData.ommIndexBufferOffset == 0xFFFFFFFF)
                                        continue;

                                    const uint64_t known = mesh->geometries[i]->debugData.ommStatsTotalKnown;
                                    const uint64_t unknown = mesh->geometries[i]->debugData.ommStatsTotalUnknown;
                                    const uint64_t total = known + unknown;
                                    const float ratio = total == 0 ? -1.f : 100.f * float(known) / float(total);

                                    std::stringstream ss;
                                    ss << ratio << "%% (" << known << " known, " << unknown << " unknown" << ")";

                                    std::string str = ss.str();
                                    ImGui::Text(str.c_str());
                                }
                            }
                        }
                    }
                }
            }
        }
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

        ImGui::Separator();
        ImGui::Text("Settings below require AS rebuild");

        {
            if (ImGui::Checkbox("Exclude Transmissive", &m_ui.AS.ExcludeTransmissive))
            {
                m_ui.AS.IsDirty = true;
            }

            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("Will exclude all transmissive geometries from the BVH");
        }
    }

    if (ImGui::CollapsingHeader("Reflex", 0))
    {
#ifdef STREAMLINE_INTEGRATION
        ImGui::Text("Reflex LowLatency Supported: %s", m_ui.REFLEX_Supported && m_ui.REFLEX_LowLatencyAvailable ? "yes" : "no");
        if (m_ui.REFLEX_Supported && m_ui.REFLEX_LowLatencyAvailable)
        {
            ImGui::Combo("Reflex Low Latency", (int*)&m_ui.REFLEX_Mode, "Off\0On\0On + Boost\0");

            bool useFrameCap = m_ui.REFLEX_CapedFPS != 0;
            if (ImGui::Checkbox("Reflex FPS Capping", &useFrameCap)) {
                if (useFrameCap) m_ui.FpsCap = 0;
            }
            else if (m_ui.FpsCap != 0) {
                useFrameCap = false;
                m_ui.REFLEX_CapedFPS = 0;
            }

            if (useFrameCap) {
                if (m_ui.REFLEX_CapedFPS == 0) { m_ui.REFLEX_CapedFPS = 60; }
                ImGui::SameLine();
                ImGui::DragInt("##FPSReflexCap", &m_ui.REFLEX_CapedFPS, 1.f, 20, 240);
                m_ui.FpsCap = 0;
            }
            else {
                m_ui.REFLEX_CapedFPS = 0;
            }

            ImGui::Checkbox("Show Stats Report", &m_ui.REFLEX_ShowStats);
            if (m_ui.REFLEX_ShowStats)
            {
                ImGui::Indent();
                ImGui::Text(m_ui.REFLEX_Stats.c_str());
                ImGui::Unindent();
            }
        }
#else
        ImGui::Text("Compiled without STREAMLINE_INTEGRATION");
#endif
    }

    if (ImGui::CollapsingHeader("DLSS-G", 0))
    {
#ifdef STREAMLINE_INTEGRATION
        ImGui::Text("DLSS-G Supported: %s", m_ui.DLSSG_Supported ? "yes" : "no");
        if (m_ui.DLSSG_Supported) {

            if (m_ui.REFLEX_Mode == sl::ReflexMode::eOff) {
                ImGui::Text("Reflex needs to be enabled for DLSSG to be enabled");
                m_ui.DLSSG_mode = sl::DLSSGMode::eOff;
            }
            else {
                ImGui::Combo("DLSS-G Mode", (int*)&m_ui.DLSSG_mode, "OFF\0ON");
            }
        }
#else
        ImGui::Text("Compiled without STREAMLINE_INTEGRATION");
#endif
    }

    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.8f, 0.5, 1.0f));
    bool debuggingIsOpen = ImGui::CollapsingHeader("Debugging"); //, ImGuiTreeNodeFlags_DefaultOpen ) )
    ImGui::PopStyleColor(1);
    if (debuggingIsOpen)
    {
        ImGui::Indent(indent);
#if ENABLE_DEBUG_VIZUALISATION
        if( ImGui::Combo( "Debug view", (int*)&m_ui.DebugView,
            "Disabled\0"
            "ImagePlaneRayLength\0DominantStablePlaneIndex\0"
            "StablePlaneVirtualRayLength\0StablePlaneMotionVectors\0"
            "StablePlaneNormals\0StablePlaneRoughness\0StablePlaneDiffBSDFEstimate\0StablePlaneDiffRadiance\0StablePlaneDiffHitDist\0StablePlaneSpecBSDFEstimate\0StablePlaneSpecRadiance\0StablePlaneSpecHitDist\0"
            "StablePlaneRelaxedDisocclusion\0StablePlaneDiffRadianceDenoised\0StablePlaneSpecRadianceDenoised\0StablePlaneCombinedRadianceDenoised\0StablePlaneViewZ\0StablePlaneDenoiserValidation\0"
            "StableRadiance\0"
            "FirstHitBarycentrics\0FirstHitFaceNormal\0FirstHitShadingNormal\0FirstHitShadingTangent\0FirstHitShadingBitangent\0FirstHitFrontFacing\0FirstHitThinSurface\0FirstHitShaderPermutation\0"
            "FirstHitDiffuse\0FirstHitSpecular\0FirstHitRoughness\0FirstHitMetallic\0"
            "VBufferMotionVectors\0VBufferDepth\0"
            "FirstHitOpacityMicroMapInWorld\0FirstHitOpacityMicroMapOverlay\0"
            "SecondarySurfacePosition\0SecondarySurfaceRadiance\0ReSTIRGIOutput\0"
            "ReSTIRDIInitialOutput\0ReSTIRDITemporalOutput\0ReSTIRDISpatialOutput\0ReSTIRDIFinalOutput\0ReSTIRDIFinalContribution\0"
			"ReGIRIndirectOutput\0"
            "\0\0") )
            m_ui.ResetAccumulation = true;
        m_ui.DebugView = dm::clamp( m_ui.DebugView, (DebugViewType)0, DebugViewType::MaxCount );

        if( m_ui.DebugView >= DebugViewType::StablePlaneVirtualRayLength && m_ui.DebugView <= DebugViewType::StablePlaneDenoiserValidation )
        {
            m_ui.DebugViewStablePlaneIndex = dm::clamp( m_ui.DebugViewStablePlaneIndex, -1, (int)m_ui.StablePlanesActiveCount-1 );
            ImGui::Indent();
            float3 spcolor = (m_ui.DebugViewStablePlaneIndex>=0)?(StablePlaneDebugVizColor(m_ui.DebugViewStablePlaneIndex)):(float3(1,1,0)); spcolor = spcolor * 0.7f + float3(0.2f, 0.2f, 0.2f);
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(spcolor.x, spcolor.y, spcolor.z, 1.0f));
            ImGui::InputInt("Stable Plane index", &m_ui.DebugViewStablePlaneIndex);
            ImGui::PopStyleColor(1);
            ImGui::Unindent();
            m_ui.DebugViewStablePlaneIndex = dm::clamp(m_ui.DebugViewStablePlaneIndex, -1, (int)m_ui.StablePlanesActiveCount - 1);
        }

        const DebugFeedbackStruct& feedback = m_app.GetFeedbackData();
        if (ImGui::InputInt2("Debug pixel", (int*)&m_ui.DebugPixel.x))
            m_app.SetUIPick();

        ImGui::Checkbox("Continuous feedback", &m_ui.ContinuousDebugFeedback);

        {
            UI_SCOPED_DISABLE(ENABLE_DEBUG_LINES_VIZUALISATION == 0);
            ImGui::Checkbox("Show debug lines", &m_ui.ShowDebugLines);
        }
        if (ENABLE_DEBUG_LINES_VIZUALISATION == 0 && ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) 
            ImGui::SetTooltip("To enable debug lines, set ENABLE_DEBUG_LINES_VIZUALISATION to 1 in Config.h");
        
        if (ImGui::Checkbox("Show material editor", &m_ui.ShowMaterialEditor) && m_ui.ShowMaterialEditor)
        {
#if ENABLE_DEBUG_DELTA_TREE_VIZUALISATION
            m_ui.ShowDeltaTree = false; // no space for both
#endif
            //m_app.SetUIPick();
        }

#if ENABLE_DEBUG_DELTA_TREE_VIZUALISATION
        if( !m_ui.ActualUseStablePlanes() )
        {
            ImGui::Text("Enable Stable Planes for delta tree viz!");
            m_ui.ShowDeltaTree = false;
        }
        else
        {
            if( ImGui::Checkbox("Show delta tree window", &m_ui.ShowDeltaTree) && m_ui.ShowDeltaTree )
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
        ImGui::TextWrapped("Debug visualization disabled; to enable set ENABLE_DEBUG_VIZUALISATION to 1");
#endif 
        ImGui::Unindent(indent);
    }

    if (ImGui::CollapsingHeader("Tone Mapping"))
    {
        ImGui::Indent(indent);
        ImGui::Checkbox("Enable Tone Mapping", &m_ui.EnableToneMapping);

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
        ImGui::Unindent(indent);
    }
    else
    {
        // quick tonemapping settings
        ImGui::PushItemWidth(defItemWidth*0.8f);
        const char * tooltipInfo = "Detailed exposure settings are in Tone Mapping section";
        ImGui::PushID("QS");
        ImGui::Checkbox("AutoExposure", &m_ui.ToneMappingParams.autoExposure); if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s",tooltipInfo);
        ImGui::SameLine();
        ImGui::SeparatorEx(ImGuiSeparatorFlags_Vertical); 
        ImGui::SameLine();
        ImGui::SliderFloat("Brightness", &m_ui.ToneMappingParams.exposureCompensation, -18.0f, 8.0f, "%.2f");  if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s",tooltipInfo);
        ImGui::SameLine();
        if (ImGui::Button("0"))
            m_ui.ToneMappingParams.exposureCompensation = 0;
         if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s",tooltipInfo);
        ImGui::PopID();
        ImGui::PopItemWidth();
    }

    ImGui::PopItemWidth();
    ImGui::End();

    auto material = m_ui.SelectedMaterial;
    if (material && m_ui.ShowMaterialEditor)
    {
        ImGui::SetNextWindowPos(ImVec2(float(scaledWidth) - 10.f, 10.f), 0, ImVec2(1.f, 0.f));
        ImGui::SetNextWindowSize(ImVec2(defWindowWidth, 0), ImGuiCond_Appearing);
        ImGui::Begin("Material Editor");
        ImGui::PushItemWidth(defItemWidth);
        ImGui::Text("Material %d: %s", material->materialID, material->name.c_str());

        MaterialDomain previousDomain = material->domain;
        const bool excludeFromNEEBefore = material->excludeFromNEE;
        const float alphaCutoffBefore = material->alphaCutoff;
        MaterialShadingProperties matPropsBefore = MaterialShadingProperties::Compute(*material);
        material->dirty = donut::app::MaterialEditor(material.get(), true);
        MaterialShadingProperties matPropsAfter = MaterialShadingProperties::Compute(*material);
        const bool excludeFromNEEAfter = material->excludeFromNEE;
        const float alphaCutoffAfter = material->alphaCutoff;

        if (matPropsBefore != matPropsAfter || 
            previousDomain != material->domain || 
            excludeFromNEEBefore != excludeFromNEEAfter ||
            material->dirty)
        {
            m_app.GetScene()->GetSceneGraph()->GetRootNode()->InvalidateContent();
            m_ui.ResetAccumulation = 1;
        }

		// The domain change might require a rebuild without the Opaque flag
        if (previousDomain != material->domain ||
            excludeFromNEEBefore != excludeFromNEEAfter ||
            alphaCutoffBefore != alphaCutoffAfter
            )
        {
            m_ui.AS.IsDirty = true;
        }

        if (matPropsBefore != matPropsAfter)
            m_ui.ShaderReloadDelayedRequest = 1.0f;

        if( m_ui.ShaderReloadDelayedRequest > 0 )
            ImGui::TextColored( ImVec4(1,0.5f,0.5f,1), "Please note: shader reload scheduled - UI might freeze for a bit." );
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
        // collect toggles
        struct LocalToggle
        {
            std::string                 Name;
            std::optional<std::string>  HoverText;
            bool *                      PropVar;
            TogglableNode *             PropNode;
            bool                        Enabled;
            LocalToggle( const std::string & name, bool & prop ) : Name(name), PropVar(&prop), PropNode(nullptr), Enabled(true) {}
            LocalToggle( const std::string & name, bool & prop, const std::string& hoverText, bool enabled ) : Name(name), PropVar(&prop), PropNode(nullptr), HoverText(hoverText), Enabled(enabled) {}
            LocalToggle( const std::string & name, TogglableNode * prop ) : Name(TrimTogglable(name)), PropVar(nullptr), PropNode(prop), Enabled(true) {}
            bool                IsSelected() const            { return (PropVar != nullptr)?(*PropVar):(PropNode->IsSelected()); }
            void                SetSelected( bool selected )  { if( PropVar != nullptr ) *PropVar = selected; else PropNode->SetSelected(selected); }
        };
        std::vector<LocalToggle> buttons;
        buttons.push_back(LocalToggle("Animations", m_ui.EnableAnimations, "Animations are not available in reference mode", m_ui.RealtimeMode));
        buttons.push_back(LocalToggle("AutoExposure", m_ui.ToneMappingParams.autoExposure ) );
        for (int i = 0; m_ui.TogglableNodes != nullptr && i < m_ui.TogglableNodes->size(); i++)
            buttons.push_back(LocalToggle((*m_ui.TogglableNodes)[i].SceneNode->GetName(), &(*m_ui.TogglableNodes)[i]));

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
                    if (ImGui::Button(buttons[i].Name.c_str(), ImVec2(buttonWidth, texSizeA.y * 2)))
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
    ImGui::Text( "Stable planes branch IDs:" );
    for (int i = 0; i < cStablePlaneCount; i++)
    {
        ImGui::Text( " %d: 0x%08x (%d dec)", i, DeltaTreeVizHeader.stableBranchIDs[i], DeltaTreeVizHeader.stableBranchIDs[i] );
        if (i == DeltaTreeVizHeader.dominantStablePlaneIndex)
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
        for (int spi = 0; spi < cStablePlaneCount; spi++)
        {
            if (StablePlaneIsOnPlane(DeltaTreeVizHeader.stableBranchIDs[spi], treeNode.deltaVertex.stableBranchID))
            {
                onPlaneIndex = spi;
                onStablePath = true;
                break;
            }
            onStablePath |= StablePlaneIsOnStablePath(DeltaTreeVizHeader.stableBranchIDs[spi], treeNode.deltaVertex.stableBranchID);
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

void UpdateTogglableNodes(std::vector<TogglableNode>& TogglableNodes, donut::engine::SceneGraphNode* node)
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
            TogglableNodes.push_back(tn);
            return &TogglableNodes.back();
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

    if( node->GetNextSibling() != nullptr )
        UpdateTogglableNodes( TogglableNodes, node->GetNextSibling() );
    if( node->GetFirstChild() != nullptr )
        UpdateTogglableNodes( TogglableNodes, node->GetFirstChild() );
}
