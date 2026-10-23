/*
* Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "OmmBaker.h"

#include <donut/engine/ShaderFactory.h>
#include <donut/engine/FramebufferFactory.h>
#include <donut/engine/CommonRenderPasses.h>
#include <donut/engine/TextureCache.h>

#include <donut/app/UserInterfaceUtils.h>
#include <donut/engine/Scene.h>

#include <nvrhi/utils.h>
#include <nvrhi/common/misc.h>

#include <donut/app/imgui_renderer.h>

#include "../SampleCommon/SampleCommon.h"

#include "OmmBuildQueue.h"

using namespace donut;
using namespace donut::math;
using namespace donut::engine;

#include "../Shaders/Misc/OmmGeometryDebugData.hlsli"

OmmBaker::OmmBaker(nvrhi::DeviceHandle device,
    std::shared_ptr<donut::engine::DescriptorTableManager> descriptorTableManager,
    std::shared_ptr<donut::engine::TextureCache> textureCache,
    std::shared_ptr<donut::engine::ShaderFactory> shaderFactory)
    : m_device(device)
    , m_textureCache(std::move(textureCache))
    , m_shaderFactory(std::move(shaderFactory))
    , m_descriptorTableManager(std::move(descriptorTableManager))
{
    // Setup OMM baker.
    m_ommBuildQueue = std::make_unique<OmmBuildQueue>(device, m_descriptorTableManager, m_shaderFactory);

    // allocate dummy buffer that works even if not enabled
    nvrhi::BufferDesc bufferDesc;
    bufferDesc.byteSize = sizeof(GeometryDebugData) * 1024;
    bufferDesc.debugName = "BindlessGeometryDebug";
    bufferDesc.structStride = sizeof(GeometryDebugData);
    bufferDesc.canHaveRawViews = true;
    bufferDesc.canHaveUAVs = true;
    bufferDesc.initialState = nvrhi::ResourceStates::Common;
    bufferDesc.keepInitialState = true;
    m_geometryDebugBuffer = m_device->createBuffer(bufferDesc);
}

OmmBaker::~OmmBaker()
{
}

void OmmBaker::SceneLoaded(const donut::engine::Scene& scene)
{
    const size_t allocationGranularity = 1024;
    const size_t geometryCount = scene.GetSceneGraph()->GetGeometryCount();
    if (scene.GetSceneGraph()->GetGeometryCount() > m_geometryDebugDataPtr.size())
    {
        m_geometryDebugDataPtr.resize(nvrhi::align<size_t>(geometryCount, allocationGranularity));
        
        nvrhi::BufferDesc bufferDesc;
        bufferDesc.byteSize = sizeof(GeometryDebugData) * m_geometryDebugDataPtr.size();
        bufferDesc.debugName = "BindlessGeometryDebug";
        bufferDesc.structStride = sizeof(GeometryDebugData);
        bufferDesc.canHaveRawViews = true;
        bufferDesc.canHaveUAVs = true;
        bufferDesc.initialState = nvrhi::ResourceStates::Common;
        bufferDesc.keepInitialState = true;

        m_geometryDebugBuffer = m_device->createBuffer(bufferDesc);
    }
}

void OmmBaker::SceneUnloading()
{
    m_ommBuildQueue->CancelPendingBuilds();
}

void OmmBaker::CreateRenderPasses(nvrhi::BindingLayoutHandle bindlessLayout, std::shared_ptr<engine::CommonRenderPasses> commonPasses)
{
    m_bindlessLayout = std::move(bindlessLayout);
    m_commonPasses = std::move(commonPasses);
}

void OmmBaker::SetBlasCompactor(bvh::BlasCompactor* compactor)
{
    m_ommBuildQueue->SetBlasCompactor(compactor);
}

void OmmBaker::CreateOpacityMicromaps(const donut::engine::Scene& scene)
{
    m_ommBuildQueue->CancelPendingBuilds();

    m_uiData.ActiveState = m_uiData.DesiredState;
    m_uiData.BuildsLeftInQueue = 0;
    m_uiData.BuildsQueued = 0;
    m_uiData.TriggerRebuild = false;
    m_retainDebugData = m_uiData.DebugView != OpacityMicroMapDebugView::Disabled;
    m_ommBuildQueue->SetRetainDebugData(m_retainDebugData);
    ++m_memoryGeneration;

    for (auto& mesh : scene.GetSceneGraph()->GetMeshes())
    {
        if (mesh->isSkinPrototype) //buffers->hasAttribute(engine::VertexAttribute::JointWeights))
            continue; // skip the skinning prototypes
        if (mesh->skinPrototype)
            continue;

        for (const auto& geometry : mesh->geometries)
            static_cast<MeshGeometryEx&>(*geometry).DebugData = {};

        OmmBuildQueue::BuildInput input;
        input.mesh = mesh;

        for (size_t i = 0; i < mesh->geometries.size(); ++i)
        {
            const donut::engine::MeshGeometry& geometry = *mesh->geometries[i];
            const auto material = static_cast<const MaterialEx*>(geometry.material.get())->PTMaterial;
            if (material == nullptr)
                continue;
            if (!(material->EnableBaseTexture && material->BaseTexture.Loaded != nullptr && material->BaseTexture.Loaded->texture != nullptr))
                continue;
            if (!material->EnableAlphaTesting)
                continue;

            // The texture cache key is {path, TextureLoadOptions}, so look the texture up under the options it was
            // actually loaded with (glTF base colour comes in as SRGBMode::ForceSRGB). The default options would not
            // match, and GetLoadedTexture indexes with operator[], so a miss both returns null and leaves a null
            // placeholder behind in the cache.
            // Nothing renders differently today: geom.alphaTexture below is currently write-only, as the bake reads
            // materialPT.BaseTexture.Loaded->texture instead (see OmmBuildQueue.cpp). This keeps the field honest for
            // whenever it does get read, and keeps the junk entry out of the cache.
            // Take it from PTMaterial (null-checked above, and what the bake uses): a material JSON can supply a
            // BaseTexture the glTF material lacks, leaving Donut's baseOrDiffuseTexture null.
            const auto & baseTexture = material->BaseTexture.Loaded;
            std::shared_ptr<TextureData> alphaTexture = m_textureCache->GetLoadedTexture(baseTexture->path, baseTexture->loadOptions);

            OmmBuildQueue::BuildInput::Geometry geom;
            geom.geometryIndexInMesh = i;
            geom.alphaTexture = alphaTexture;
            geom.maxSubdivisionLevel = m_uiData.ActiveState->MaxSubdivisionLevel;
            geom.dynamicSubdivisionScale = m_uiData.ActiveState->EnableDynamicSubdivision ? m_uiData.ActiveState->DynamicSubdivisionScale : 0.f;
            geom.format = m_uiData.ActiveState->Format;
            geom.flags = m_uiData.ActiveState->Flag;
            geom.alphaCutoffGT = (omm::OpacityState)m_uiData.ActiveState->AlphaCutoffGT;
            geom.alphaCutoffLE = (omm::OpacityState)m_uiData.ActiveState->AlphaCutoffLE;
            geom.maxOmmArrayDataSizeInMB = m_uiData.ActiveState->MaxOmmArrayDataSizeInMB;
            geom.computeOnly = m_uiData.ActiveState->ComputeOnly;
            geom.enableLevelLineIntersection = m_uiData.ActiveState->LevelLineIntersection;
            geom.enableTexCoordDeduplication = m_uiData.ActiveState->EnableTexCoordDeduplication;
            geom.force32BitIndices = m_uiData.ActiveState->Force32BitIndices;
            geom.enableSpecialIndices = m_uiData.ActiveState->EnableSpecialIndices;
            geom.enableNsightDebugMode = m_uiData.ActiveState->EnableNsightDebugMode;

            input.geometries.push_back(geom);
        }

        if (input.geometries.size() != 0ull)
        {
            ++m_uiData.BuildsQueued;
            m_ommBuildQueue->QueueBuild(input);
        }
    }
    m_uiData.BuildsLeftInQueue = m_ommBuildQueue->NumPendingBuilds();
}

void OmmBaker::DestroyOpacityMicromaps(nvrhi::ICommandList& commandList, const donut::engine::Scene& scene)
{
    commandList.close();
    m_device->executeCommandList(&commandList);
    m_device->waitForIdle();
    commandList.open();

    for (const std::shared_ptr<MeshInfo>& _mesh : scene.GetSceneGraph()->GetMeshes())
    {
        assert(std::dynamic_pointer_cast<MeshInfoEx>(_mesh) != nullptr);
        const std::shared_ptr<MeshInfoEx>& mesh = std::static_pointer_cast<MeshInfoEx>(_mesh);
        mesh->AccelStructOMM = nullptr;
        mesh->OpacityMicroMaps.clear();
        mesh->DebugData = nullptr;
        mesh->MarkDebugDataDirty();
    }
}

void OmmBaker::ReleaseDebugData(nvrhi::ICommandList& commandList, const donut::engine::Scene& scene)
{
    // Bindless descriptors can still be read by the previous frame's debug shaders.
    commandList.close();
    m_device->executeCommandList(&commandList);
    m_device->waitForIdle();
    commandList.open();

    for (const auto& mesh : scene.GetSceneGraph()->GetMeshes())
    {
        MeshInfoEx& meshEx = static_cast<MeshInfoEx&>(*mesh);
        if (meshEx.DebugData)
        {
            meshEx.DebugData.reset();
            meshEx.MarkDebugDataDirty();
        }
    }
    ++m_memoryGeneration;
}

bool OmmBaker::NeedsGraphicsQueue() const
{
    const bool retainDebugData = m_uiData.DebugView != OpacityMicroMapDebugView::Disabled;
    return m_ommBuildQueue->NumPendingBuilds() > 0 || m_uiData.TriggerRebuild || retainDebugData != m_retainDebugData;
}

void OmmBaker::BuildOpacityMicromaps(nvrhi::ICommandList& commandList, const donut::engine::Scene& scene)
{
    ZoneScoped;

    GpuProfiler::Begin(&commandList, "OMM Updates");

    const bool shouldRetainDebugData = m_uiData.DebugView != OpacityMicroMapDebugView::Disabled;
    if (shouldRetainDebugData != m_retainDebugData)
    {
        m_retainDebugData = shouldRetainDebugData;
        m_ommBuildQueue->SetRetainDebugData(shouldRetainDebugData);
        if (shouldRetainDebugData)
            m_uiData.TriggerRebuild = true;
        else
            ReleaseDebugData(commandList, scene);
    }

    if (m_uiData.TriggerRebuild)
    {
        DestroyOpacityMicromaps(commandList, scene);

        m_ommBuildQueue->CancelPendingBuilds();

        CreateOpacityMicromaps(scene);

        m_uiData.TriggerRebuild = false;
    }

    m_ommBuildQueue->Update(commandList);

    m_uiData.BuildsLeftInQueue = m_ommBuildQueue->NumPendingBuilds();

    GpuProfiler::End(&commandList);
}

void OmmBaker::WriteGeometryDebugBuffer(nvrhi::ICommandList& commandList)
{
    commandList.writeBuffer(m_geometryDebugBuffer, m_geometryDebugDataPtr.data(), m_geometryDebugDataPtr.size() * sizeof(GeometryDebugData));
}

bool MeshInfoEx::s_anyDebugDataDirty = true;   // start dirty so the first Update() always scans

void OmmBaker::UpdateDebugGeometry(const MeshInfo& _mesh)
{
    const MeshInfoEx& mesh = static_cast<const MeshInfoEx&>(_mesh);
    assert(&mesh != nullptr);

    for (const auto& _geometry : mesh.geometries)
    {
        assert(std::dynamic_pointer_cast<MeshGeometryEx>(_geometry) != nullptr);
        const std::shared_ptr<MeshGeometryEx>& geometry = std::static_pointer_cast<MeshGeometryEx>(_geometry);

        if (MeshDebugData* debugData = mesh.DebugData.get())
        {
            GeometryDebugData& dgdata = m_geometryDebugDataPtr[geometry->globalGeometryIndex];
            dgdata.ommArrayDataBufferIndex = debugData->ommArrayDataBufferDescriptor ? debugData->ommArrayDataBufferDescriptor->Get() : -1;
            dgdata.ommArrayDataBufferOffset = geometry->DebugData.ommArrayDataOffset;

            dgdata.ommDescArrayBufferIndex = debugData->ommDescBufferDescriptor ? debugData->ommDescBufferDescriptor->Get() : -1;
            dgdata.ommDescArrayBufferOffset = geometry->DebugData.ommDescBufferOffset;

            dgdata.ommIndexBufferIndex = debugData->ommIndexBufferDescriptor ? debugData->ommIndexBufferDescriptor->Get() : -1;
            dgdata.ommIndexBufferOffset = geometry->DebugData.ommIndexBufferOffset;
            dgdata.ommIndexBuffer16Bit = geometry->DebugData.ommIndexBufferFormat == nvrhi::Format::R16_UINT;
        }
        else
        {
            GeometryDebugData& dgdata = m_geometryDebugDataPtr[geometry->globalGeometryIndex];
            dgdata.ommArrayDataBufferIndex = -1;
            dgdata.ommArrayDataBufferOffset = 0xFFFFFFFF;
            dgdata.ommDescArrayBufferIndex = -1;
            dgdata.ommDescArrayBufferOffset = 0xFFFFFFFF;
            dgdata.ommIndexBufferIndex = -1;
            dgdata.ommIndexBufferOffset = 0xFFFFFFFF;
            dgdata.ommIndexBuffer16Bit = 0;
        }
    }
}

bool OmmBaker::Update(nvrhi::ICommandList& commandList, const donut::engine::Scene& scene)
{
    ZoneScoped;

    // Skip the full-scene scan entirely when nothing has been marked dirty. Without this we dereference every mesh's
    // shared_ptr every frame just to read a bool that is almost always false. The flag is push-based - raised only by
    // MeshInfoEx::MarkDebugDataDirty() and by MeshInfoEx construction - so this cannot miss an update.
    if( !MeshInfoEx::AnyDebugDataDirty() )
        return false;

    RTXPT_GPU_SCOPE(&commandList, "OmmBaker");

    bool anyDirty = false;
    for (auto& _mesh : scene.GetSceneGraph()->GetMeshes())
    {
        MeshInfoEx& mesh = static_cast<MeshInfoEx&>(*_mesh);
        assert(&mesh != nullptr);

        if (mesh.IsDebugDataDirty())
        {
            mesh.ClearDebugDataDirty();
            anyDirty = true;
            UpdateDebugGeometry(mesh);
        }
    }
    // Cleared after the scan, which has just visited every mesh. Nothing in the loop marks anything dirty, so there is
    // no reentrancy window here.
    MeshInfoEx::ClearAnyDebugDataDirty();

    if (anyDirty)
        WriteGeometryDebugBuffer(commandList);
    return anyDirty;
}

void OmmBaker::SetGlobalShaderMacros(std::vector<donut::engine::ShaderMacro>& macros)
{
    if (m_uiData.DebugView == OpacityMicroMapDebugView::InWorld)
        macros.push_back( { "OMM_DEBUG_VIEW_IN_WORLD", "1" } );
    if (m_uiData.DebugView == OpacityMicroMapDebugView::Overlay)
        macros.push_back( { "OMM_DEBUG_VIEW_OVERLAY", "1" } );
}

bool OmmBaker::DebugGUI(float indent, const donut::engine::Scene& scene)
{
    RAII_SCOPE(ImGui::PushID("OmmBakerDebugGUI"); , ImGui::PopID(); );
    
    bool resetAccumulation = false;
    #define RESET_ON_CHANGE(code) do{if (code) resetAccumulation = true;} while(false)
    #define UI_SCOPED_INDENT(indent) RAII_SCOPE(ImGui::Indent(indent); , ImGui::Unindent(indent); )
    #define UI_SCOPED_DISABLE(cond) RAII_SCOPE(ImGui::BeginDisabled(cond); , ImGui::EndDisabled(); )

    if (ImGui::Checkbox("Enable", &m_uiData.Enable))
        resetAccumulation = true;

    {
        {
            UI_SCOPED_DISABLE(m_uiData.ActiveState.has_value() && m_uiData.ActiveState->Format != nvrhi::rt::OpacityMicromapFormat::OC1_4_State);
            if (ImGui::Checkbox("Force 2 State", &m_uiData.Force2State))
                resetAccumulation = true;
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("Will force 2-State via TLAS instance mask.");
        }

        {
            if (ImGui::Checkbox("Render ONLY OMMs", &m_uiData.OnlyOMMs))
                resetAccumulation = true;
        }

        ImGui::Separator();
        ImGui::Text("Bake Settings (Require Rebuild to take effect)");

        if (m_uiData.BuildsLeftInQueue != 0)
        {
            const float progress = (1.f - (float)m_uiData.BuildsLeftInQueue / m_uiData.BuildsQueued);
            std::stringstream ss;
            ss << "Build progress: " << (uint32_t)(100.f * progress) << "%";
            std::string str = ss.str();
            ImGui::ProgressBar(progress, ImVec2(-FLT_MIN, 0), str.c_str());
        }

        {
            UI_SCOPED_DISABLE(m_uiData.ActiveState.has_value() && m_uiData.ActiveState == m_uiData.DesiredState);
            if (ImGui::Button("Trigger Rebuild"))
            {
                m_uiData.TriggerRebuild = true;
            }
        }

        {
            ImGui::Checkbox("Dynamic subdivision level", &m_uiData.DesiredState.EnableDynamicSubdivision);
        }

        {
            UI_SCOPED_DISABLE(!m_uiData.DesiredState.EnableDynamicSubdivision);
            ImGui::SliderFloat("Dynamic subdivision scale", &m_uiData.DesiredState.DynamicSubdivisionScale, 0.01f, 20.f, "%.1f", ImGuiSliderFlags_Logarithmic);
        }

        {
            const int MaxSubdivisionLevel = m_uiData.DesiredState.ComputeOnly ? 12 : 10;
            m_uiData.DesiredState.MaxSubdivisionLevel = std::clamp(m_uiData.DesiredState.MaxSubdivisionLevel, 1, MaxSubdivisionLevel);
            ImGui::SliderInt("Max subdivision level", &m_uiData.DesiredState.MaxSubdivisionLevel, 1, MaxSubdivisionLevel, "%d", ImGuiSliderFlags_AlwaysClamp);
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

            if (ImGui::BeginCombo("Flag", formatNames[(uint32_t)m_uiData.DesiredState.Flag]))
            {
                for (uint i = 0; i < formats.size(); i++)
                {
                    bool is_selected = formats[i] == m_uiData.DesiredState.Flag;
                    if (ImGui::Selectable(formatNames[i], is_selected))
                        m_uiData.DesiredState.Flag = formats[i];
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
            if (ImGui::BeginCombo("Format", FormatToString(m_uiData.DesiredState.Format)))
            {
                for (uint i = 0; i < formats.size(); i++)
                {
                    bool is_selected = formats[i] == m_uiData.DesiredState.Format;
                    if (ImGui::Selectable(FormatToString(formats[i]), is_selected))
                        m_uiData.DesiredState.Format = formats[i];
                    if (is_selected)
                        ImGui::SetItemDefaultFocus();
                }
                ImGui::EndCombo();
            }
        }

        {
            auto StateToString = [ ](omm::OpacityState state) {
                const char* strings[] = { "Transparent", "Opaque", "UnknownTransparent", "UnknownOpaque" };
                assert((int)state >= 0 && (int)state < IM_ARRAYSIZE(strings));
                return strings[(int)state];
            };
            const std::array<omm::OpacityState, 4> states = { omm::OpacityState::Transparent, omm::OpacityState::Opaque, omm::OpacityState::UnknownTransparent, omm::OpacityState::UnknownOpaque };

            if (ImGui::BeginCombo("AlphaCutoffGT", StateToString((omm::OpacityState)m_uiData.DesiredState.AlphaCutoffGT)))
            {
                for (uint i = 0; i < states.size(); i++)
                {
                    bool is_selected = states[i] == (omm::OpacityState)m_uiData.DesiredState.AlphaCutoffGT;
                    if (ImGui::Selectable(StateToString(states[i]), is_selected))
                        m_uiData.DesiredState.AlphaCutoffGT = (int)states[i];
                    if (is_selected)
                        ImGui::SetItemDefaultFocus();
                }
                ImGui::EndCombo();
            }

            if (ImGui::BeginCombo("AlphaCutoffLE", StateToString((omm::OpacityState)m_uiData.DesiredState.AlphaCutoffLE)))
            {
                for (uint i = 0; i < states.size(); i++)
                {
                    bool is_selected = states[i] == (omm::OpacityState)m_uiData.DesiredState.AlphaCutoffLE;
                    if (ImGui::Selectable(StateToString(states[i]), is_selected))
                        m_uiData.DesiredState.AlphaCutoffLE = (int)states[i];
                    if (is_selected)
                        ImGui::SetItemDefaultFocus();
                }
                ImGui::EndCombo();
            }
        }

        if (ImGui::CollapsingHeader("Debug Settings"))
        {
            UI_SCOPED_INDENT(indent);

#if ENABLE_DEBUG_VIZUALISATIONS
            RESET_ON_CHANGE( ImGui::Combo("Debug View", (int*)&m_uiData.DebugView, "Disabled\0InWorld\0Overlay\0\0") );
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("Enabling visualization rebakes OMMs and retains their debug buffers. Disabling it releases those buffers.");
#else
            ImGui::Text("Please enable ENABLE_DEBUG_VIZUALISATIONS for debug viz");
            m_uiData.DebugView = OpacityMicroMapDebugView::Disabled;
#endif

            ImGui::Checkbox("Compute Only", &m_uiData.DesiredState.ComputeOnly);

            ImGui::Checkbox("Enable \"Level Line Intersection\"", &m_uiData.DesiredState.LevelLineIntersection);

            ImGui::Checkbox("Enable TexCoord deduplication", &m_uiData.DesiredState.EnableTexCoordDeduplication);

            ImGui::Checkbox("Force 32-bit indices", &m_uiData.DesiredState.Force32BitIndices);

            ImGui::Checkbox("Enable Special Indices", &m_uiData.DesiredState.EnableSpecialIndices);

            ImGui::SliderInt("Max memory per OMM", &m_uiData.DesiredState.MaxOmmArrayDataSizeInMB, 1, 1000, "%dMB", ImGuiSliderFlags_Logarithmic);

            ImGui::Checkbox("Enable NSight debug mode", &m_uiData.DesiredState.EnableNsightDebugMode);
        }

        ImGui::Separator();
        ImGui::Text("Stats");

        {
            std::stringstream ss;
            ss << (m_uiData.BuildsQueued - m_uiData.BuildsLeftInQueue) << " / " << m_uiData.BuildsQueued << " OMM meshes built";
            std::string str = ss.str();
            ImGui::Text(str.c_str());

            if (ImGui::CollapsingHeader("Bake Stats"))
            {
                UI_SCOPED_INDENT(indent);

                for (const std::shared_ptr<donut::engine::MeshInfo>& mesh : scene.GetSceneGraph()->GetMeshes())
                {
                    bool meshHasOmms = false;
                    for (uint32_t i = 0; i < mesh->geometries.size(); ++i)
                    {
                        if (std::static_pointer_cast<MeshGeometryEx>(mesh->geometries[i])->DebugData.ommIndexBufferOffset != 0xFFFFFFFF)
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
                            auto geometry = std::static_pointer_cast<MeshGeometryEx>(mesh->geometries[i]);

                            if (geometry->DebugData.ommIndexBufferOffset == 0xFFFFFFFF)
                                continue;

                            const uint64_t known = geometry->DebugData.ommStatsTotalKnown;
                            const uint64_t unknown = geometry->DebugData.ommStatsTotalUnknown;
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

    return resetAccumulation;
}

