/*
* Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "MaterialsBaker.h"

#include <donut/engine/ShaderFactory.h>
#include <donut/engine/FramebufferFactory.h>
#include <donut/engine/CommonRenderPasses.h>
#include <donut/engine/TextureCache.h>

#include <donut/app/UserInterfaceUtils.h>

#include <nvrhi/utils.h>

#include <donut/app/imgui_renderer.h>

#include "../SampleCommon.h"
#include "../ExtendedScene.h"

#include <filesystem>

#include <donut/core/json.h>
#include <json/json.h>
#include <json/value.h>

#include <fstream>

#include <unordered_set>

#include <cctype>      // std::tolower

#include "../SampleUI.h"

using namespace donut;
using namespace donut::math;
using namespace donut::engine;

void MaterialPT::Texture::InitFromLoadedTexture(std::shared_ptr<donut::engine::LoadedTexture> & loaded, bool _sRGB, bool _normalMap, const std::filesystem::path & mediaPath)
{
    if (loaded == nullptr)
    { LocalPath = ""; sRGB = false; Loaded = nullptr; return; }

    LocalPath = std::filesystem::relative(loaded->path, mediaPath);
    sRGB = _sRGB;
    Loaded = loaded;
    NormalMap = _normalMap;
}

std::shared_ptr<MaterialPT> MaterialPT::FromDonut(const std::shared_ptr<Material>& donutMaterial)
{
    if (donutMaterial == nullptr)
        return nullptr;
    assert(std::dynamic_pointer_cast<MaterialEx>(donutMaterial) != nullptr);
    return std::static_pointer_cast<MaterialEx>(donutMaterial)->MaterialPT;
}

void MaterialPT::Write(Json::Value& output, const std::filesystem::path& mediaPath)
{
    auto saveTexture = [ ](Json::Value& output, const MaterialPT::Texture & texture, const std::string& name)
    {
        if (texture.Loaded == nullptr)
            return;
        Json::Value texJ;
        texJ["sRGB"] = texture.sRGB;
        texJ["NormalMap"] = texture.NormalMap;
        texJ["path"] = texture.LocalPath.string();
        output[name] = texJ;
    };

    output["name"] = Name;
    output["version"] = 1;

    saveTexture(output, BaseTexture, "BaseTexture");
    saveTexture(output, OcclusionRoughnessMetallicTexture, "OcclusionRoughnessMetallicTexture");
    saveTexture(output, NormalTexture, "NormalTexture");
    saveTexture(output, EmissiveTexture, "EmissiveTexture");
    saveTexture(output, TransmissionTexture, "TransmissionTexture");

#define STORE_FIELD(NAME) output[#NAME] << NAME;

    STORE_FIELD(BaseOrDiffuseColor);
    STORE_FIELD(SpecularColor);
    STORE_FIELD(EmissiveColor);

    STORE_FIELD(EmissiveIntensity);
    STORE_FIELD(Metalness);
    STORE_FIELD(Roughness);
    STORE_FIELD(Opacity);
    STORE_FIELD(TransmissionFactor);
    STORE_FIELD(DiffuseTransmissionFactor);
    STORE_FIELD(NormalTextureScale);
    STORE_FIELD(IoR);

    STORE_FIELD(UseSpecularGlossModel);
    STORE_FIELD(EnableBaseTexture);
    STORE_FIELD(EnableOcclusionRoughnessMetallicTexture);
    STORE_FIELD(EnableNormalTexture);
    STORE_FIELD(EnableEmissiveTexture);
    STORE_FIELD(EnableTransmissionTexture);
    STORE_FIELD(EnableAlphaTesting);
    STORE_FIELD(AlphaCutoff);
    STORE_FIELD(EnableTransmission);
    STORE_FIELD(MetalnessInRedChannel);
    STORE_FIELD(ThinSurface);
    STORE_FIELD(ExcludeFromNEE);
    STORE_FIELD(PSDExclude);

    STORE_FIELD(PSDDominantDeltaLobe);
    STORE_FIELD(NestedPriority);

    STORE_FIELD(VolumeAttenuationDistance);
    STORE_FIELD(VolumeAttenuationColor);

    STORE_FIELD(ShadowNoLFadeout);
}

std::shared_ptr<MaterialPT> MaterialPT::FromJson(Json::Value& input, const std::filesystem::path& mediaPath, const std::shared_ptr<donut::engine::TextureCache>& textureCache)
{
    std::shared_ptr<MaterialPT> material = std::make_shared<MaterialPT>();

    int version = -1;
    input["version"] >> version;
    if (version != 1)
        { donut::log::warning("Unsupported/missing material version"); return nullptr; }

    auto loadTexture = [ & ](Json::Value& input, MaterialPT::Texture & output, const std::string & name)
    {
        output = MaterialPT::Texture();
        Json::Value texJ = input[name];

        if (texJ.empty())
            return;

        std::string localPath;
        texJ["path"] >> localPath; output.LocalPath = localPath;
        if (output.LocalPath == "")
            { donut::log::warning("Path for texture is empty"); return; }
        texJ["sRGB"] >> output.sRGB;
        texJ["NormalMap"] >> output.NormalMap;

        std::filesystem::path fullPath = mediaPath;
        fullPath /= output.LocalPath;

        bool cSearchForDDS = true;
        std::string extension = fullPath.extension().string();
        std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);
        if ( cSearchForDDS && extension == ".png" )
        {
            std::filesystem::path filePathDDS = fullPath;
            filePathDDS.replace_extension(".dds");

            if ( std::filesystem::exists(filePathDDS) )
            {
                fullPath = filePathDDS;
                output.LocalPath.replace_extension(".dds");
            }
        }

        output.Loaded = textureCache->LoadTextureFromFileDeferred(fullPath, output.sRGB);
    };

    input["name"] >> material->Name;
    if (material->Name == "")
        { donut::log::warning("Unsupported/missing material name"); return nullptr; }

    loadTexture(input, material->BaseTexture, "BaseTexture");
    loadTexture(input, material->OcclusionRoughnessMetallicTexture, "OcclusionRoughnessMetallicTexture");
    loadTexture(input, material->NormalTexture, "NormalTexture");
    loadTexture(input, material->EmissiveTexture, "EmissiveTexture");
    loadTexture(input, material->TransmissionTexture, "TransmissionTexture");

#define LOAD_FIELD(NAME) input[#NAME] >> material->NAME;

    LOAD_FIELD(BaseOrDiffuseColor);
    LOAD_FIELD(SpecularColor);
    LOAD_FIELD(EmissiveColor);

    LOAD_FIELD(EmissiveIntensity);
    LOAD_FIELD(Metalness);
    LOAD_FIELD(Roughness);
    LOAD_FIELD(Opacity);
    LOAD_FIELD(TransmissionFactor);
    LOAD_FIELD(DiffuseTransmissionFactor);
    LOAD_FIELD(NormalTextureScale);
    LOAD_FIELD(IoR);

    LOAD_FIELD(UseSpecularGlossModel);
    LOAD_FIELD(EnableBaseTexture);
    LOAD_FIELD(EnableOcclusionRoughnessMetallicTexture);
    LOAD_FIELD(EnableNormalTexture);
    LOAD_FIELD(EnableEmissiveTexture);
    LOAD_FIELD(EnableTransmissionTexture);
    LOAD_FIELD(EnableAlphaTesting);
    LOAD_FIELD(AlphaCutoff);
    LOAD_FIELD(EnableTransmission);
    LOAD_FIELD(MetalnessInRedChannel);
    LOAD_FIELD(ThinSurface);
    LOAD_FIELD(ExcludeFromNEE);
    LOAD_FIELD(PSDExclude);

    LOAD_FIELD(PSDDominantDeltaLobe);
    LOAD_FIELD(NestedPriority);

    LOAD_FIELD(VolumeAttenuationDistance);
    LOAD_FIELD(VolumeAttenuationColor);

    LOAD_FIELD(ShadowNoLFadeout);

    return material;
}


bool MaterialPT::EditorGUI()
{
    bool update = false;

    float itemWidth = ImGui::CalcItemWidth();

    auto getShortTexturePath = [ ](const MaterialPT::Texture & texture) -> std::string
    {
        if( texture.Loaded == nullptr ) return "<nullptr>";
        return texture.LocalPath.string();
    };

    const ImVec4 filenameColor = ImVec4(0.474f, 0.722f, 0.176f, 1.0f);

    update |= ImGui::Checkbox("Thin surface", &ThinSurface);
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Material has no volumetric properties - used for double sided thin surfaces like leafs.");
    update |= ImGui::Checkbox("Ignore by NEE shadow ray", &ExcludeFromNEE);
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Ignored for shadow rays during Next Event Estimation");

    if (UseSpecularGlossModel)
    {
        if (BaseTexture.Loaded != nullptr)
        {
            update |= ImGui::Checkbox("Use Base (Diffuse) Texture", &EnableBaseTexture);
            ImGui::SameLine();
            ImGui::TextColored(filenameColor, "%s", getShortTexturePath(BaseTexture).c_str());
        }

        update |= ImGui::ColorEdit3(EnableBaseTexture ? "Diffuse Factor" : "Diffuse Color", BaseOrDiffuseColor.data(), ImGuiColorEditFlags_Float);

        if (OcclusionRoughnessMetallicTexture.Loaded != nullptr)
        {
            update |= ImGui::Checkbox("Use Specular Texture", &EnableOcclusionRoughnessMetallicTexture);
            ImGui::SameLine();
            ImGui::TextColored(filenameColor, "%s", getShortTexturePath(OcclusionRoughnessMetallicTexture).c_str());
        }

        update |= ImGui::ColorEdit3(EnableOcclusionRoughnessMetallicTexture ? "Specular Factor" : "Specular Color", SpecularColor.data(), ImGuiColorEditFlags_Float);

        float glossiness = 1.0f - Roughness;
        update |= ImGui::SliderFloat(EnableOcclusionRoughnessMetallicTexture ? "Glossiness Factor" : "Glossiness", &glossiness, 0.f, 1.f);
        Roughness = 1.0f - glossiness;
    }
    else
    {
        if (BaseTexture.Loaded)
        {
            update |= ImGui::Checkbox("Use Base (Diffuse) Texture", &EnableBaseTexture);
            ImGui::SameLine();
            ImGui::TextColored(filenameColor, "%s", getShortTexturePath(BaseTexture).c_str());
        }

        update |= ImGui::ColorEdit3(EnableBaseTexture ? "Base Color Factor" : "Base Color", BaseOrDiffuseColor.data(), ImGuiColorEditFlags_Float);

        if (OcclusionRoughnessMetallicTexture.Loaded)
        {
            update |= ImGui::Checkbox("Use Metal-Rough Texture", &EnableOcclusionRoughnessMetallicTexture);
            ImGui::SameLine();
            ImGui::TextColored(filenameColor, "%s", getShortTexturePath(OcclusionRoughnessMetallicTexture).c_str());
        }

        update |= ImGui::SliderFloat(EnableOcclusionRoughnessMetallicTexture ? "Metalness Factor" : "Metalness", &Metalness, 0.f, 1.f);
        update |= ImGui::SliderFloat(EnableOcclusionRoughnessMetallicTexture ? "Roughness Factor" : "Roughness", &Roughness, 0.f, 1.f);
    }

    update |= ImGui::Checkbox("Enable Alpha Testing", &EnableAlphaTesting);

    if (EnableAlphaTesting && BaseTexture.Loaded)
    {
        update |= ImGui::SliderFloat("Alpha Cutoff", &AlphaCutoff, 0.f, 1.f);
    }

    if (NormalTexture.Loaded != nullptr)
    {
        update |= ImGui::Checkbox("Use Normal Texture", &EnableNormalTexture);
        ImGui::SameLine();
        ImGui::TextColored(filenameColor, "%s", getShortTexturePath(NormalTexture).c_str());
    }

    if (EnableNormalTexture)
    {
        ImGui::SetNextItemWidth(itemWidth - 31.f);
        update |= ImGui::SliderFloat("###normtexscale", &NormalTextureScale, -2.f, 2.f);
        ImGui::SameLine(0.f, 5.f);
        ImGui::SetNextItemWidth(26.f);
        if (ImGui::Button("1.0"))
        {
            NormalTextureScale = 1.f;
            update = true;
        }
        ImGui::SameLine();
        ImGui::Text("Normal Scale");
    }

    if (EmissiveTexture.Loaded)
    {
        update |= ImGui::Checkbox("Use Emissive Texture", &EnableEmissiveTexture);
        ImGui::SameLine();
        ImGui::TextColored(filenameColor, "%s", getShortTexturePath(EmissiveTexture).c_str());
    }

    update |= ImGui::ColorEdit3("Emissive Color", EmissiveColor.data(), ImGuiColorEditFlags_Float);
    update |= ImGui::SliderFloat("Emissive Intensity", &EmissiveIntensity, 0.f, 100000.f, "%.3f", ImGuiSliderFlags_Logarithmic);

    update |= ImGui::Checkbox("Enable Transmission", &EnableTransmission);

    if (EnableTransmission)   // transmissive
    {
        update |= ImGui::InputFloat("Index of Refraction", &IoR);
        if (IoR < 1.0f) { IoR = 1.0f; update = true; }

        if (TransmissionTexture.Loaded)
        {
            update |= ImGui::Checkbox("Use Transmission Texture", &EnableTransmissionTexture);
            ImGui::SameLine();
            ImGui::TextColored(filenameColor, "%s", getShortTexturePath(TransmissionTexture).c_str());
        }

        update |= ImGui::SliderFloat("Transmission Factor", &TransmissionFactor, 0.f, 1.f);
        update |= ImGui::SliderFloat("Diff Transmission Factor", &DiffuseTransmissionFactor, 0.f, 1.f);

        if (ThinSurface)
        {
            update |= ImGui::InputFloat("Attenuation Distance", &VolumeAttenuationDistance);
            if (VolumeAttenuationDistance < 0.0f) { VolumeAttenuationDistance = 0.0f; update = true; }

            update |= ImGui::ColorEdit3("Attenuation Color", VolumeAttenuationColor.data(), ImGuiColorEditFlags_Float);

            update |= ImGui::InputInt("Nested Priority", &NestedPriority);
            if (NestedPriority < 0 || NestedPriority > 14) { NestedPriority = dm::clamp(NestedPriority, 0, 14); update = true; }
        }
        else
        {
            ImGui::Text("Thin surface transmissive materials have no volume properties");
        }
    }

    update |= ImGui::SliderFloat("Shadow NoL Fadeout", &ShadowNoLFadeout, 0.0f, 0.2f);
    if (ImGui::IsItemHovered()) ImGui::SetTooltip(
        "Low tessellation geometry often has triangle (flat) normals that differ significantly from shading normals. \n"
        "This causes shading vs shadow discrepancy that exposes triangle edges. One way to mitigate this (other than \n"
        "having more detailed mesh) is to add additional shadowing falloff to hide the seam. This setting is not \n"
        "physically correct and adds bias. Setting of 0 means no fadeout (default).");


    if (ImGui::CollapsingHeader("Path Decomposition"))
    {
        ImGui::Indent();

        bool psdEnable = !PSDExclude; // makes more sense from UI perspective - avoids double negative
        update |= ImGui::Checkbox("Enable delta lobe decomposition", &psdEnable);
        PSDExclude = !psdEnable;
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Some materials/meshes look best without decomposition.");

        {
            UI_SCOPED_DISABLE(PSDExclude);
            int dominantDeltaLobeP1 = dm::clamp(PSDDominantDeltaLobe, -1, 1) + 1;
            update |= ImGui::Combo("Dominant bounce", &dominantDeltaLobeP1, "None (surface)\0Transparency\0Reflection\0\0");
            PSDDominantDeltaLobe = dm::clamp(dominantDeltaLobeP1 - 1, -1, 1);
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("Determines which surface will:\n * provide motion vectors for denoising\n * get ReSTIR DI lighting\n * get 'boost samples' for NEE lighting");
        }
        ImGui::Unindent();
    }

    // mark for update
    GPUDataDirty |= update;

    return update;
}

static void GetBindlessTextureIndex(const std::shared_ptr<LoadedTexture>& texture, uint& outEncodedInfo, unsigned int& flags, unsigned int textureBit)
{
    // if bit not set, don't set the texture; if texture unavailable - remove the texture bit!
    if ((flags & textureBit) == 0 || texture == nullptr || texture->texture == nullptr)
    {
        outEncodedInfo = 0xFFFFFFFF;
        flags &= ~textureBit; // remove flag
        return;
    }

    uint bindlessDescIndex = texture->bindlessDescriptor.Get();
    assert(bindlessDescIndex <= 0xFFFF);
    bindlessDescIndex &= 0xFFFF;

    const auto desc = texture->texture->getDesc();
    float baseLODf = donut::math::log2f((float)desc.width * desc.height);
    uint baseLOD = (uint)(baseLODf + 0.5f);
    uint mipLevels = desc.mipLevels;
    assert(baseLOD >= 0 && baseLOD <= 255);
    assert(mipLevels >= 0 && mipLevels <= 255);

    outEncodedInfo = (baseLOD << 24) | (mipLevels << 16) | bindlessDescIndex;
}

bool MaterialPT::IsEmissive() const
{
    return (EmissiveIntensity > 0) && (donut::math::any(EmissiveColor>0.0f));
}

void MaterialPT::FillData(MaterialPTData & data)
{
    // flags

    data.Flags = 0;

    if (UseSpecularGlossModel)
        data.Flags |= MaterialPTFlags_UseSpecularGlossModel;

    if (BaseTexture.Loaded && EnableBaseTexture)
        data.Flags |= MaterialPTFlags_UseBaseOrDiffuseTexture;

    if (OcclusionRoughnessMetallicTexture.Loaded && EnableOcclusionRoughnessMetallicTexture)
        data.Flags |= MaterialPTFlags_UseMetalRoughOrSpecularTexture;

    if (EmissiveTexture.Loaded && EnableEmissiveTexture)
        data.Flags |= MaterialPTFlags_UseEmissiveTexture;

    if (NormalTexture.Loaded && EnableNormalTexture)
        data.Flags |= MaterialPTFlags_UseNormalTexture;

    if (TransmissionTexture.Loaded && EnableTransmissionTexture && EnableTransmission)
        data.Flags |= MaterialPTFlags_UseTransmissionTexture;

    if (MetalnessInRedChannel)
        data.Flags |= MaterialPTFlags_MetalnessInRedChannel;

    if (ThinSurface)
        data.Flags |= MaterialPTFlags_ThinSurface;

    if (PSDExclude)
        data.Flags |= MaterialPTFlags_PSDExclude;

    // free parameters

    data.BaseOrDiffuseColor = BaseOrDiffuseColor;
    data.SpecularColor = SpecularColor;
    data.EmissiveColor = EmissiveColor * EmissiveIntensity;
    data.Roughness = Roughness;
    data.Metalness = Metalness;
    data.NormalTextureScale = NormalTextureScale;
    data.TransmissionFactor = (EnableTransmission)?(TransmissionFactor):(0);
    data.DiffuseTransmissionFactor = (EnableTransmission)?(DiffuseTransmissionFactor):(0);
    data.Opacity = Opacity;
    data.AlphaCutoff = AlphaCutoff;
    data.IoR = IoR;
    data.Volume.AttenuationColor    = VolumeAttenuationColor;
    data.Volume.AttenuationDistance = VolumeAttenuationDistance;

    // bindless textures

    GetBindlessTextureIndex(BaseTexture.Loaded, data.BaseOrDiffuseTextureIndex, data.Flags, MaterialPTFlags_UseBaseOrDiffuseTexture);
    GetBindlessTextureIndex(OcclusionRoughnessMetallicTexture.Loaded, data.MetalRoughOrSpecularTextureIndex, data.Flags, MaterialPTFlags_UseMetalRoughOrSpecularTexture);
    GetBindlessTextureIndex(EmissiveTexture.Loaded, data.EmissiveTextureIndex, data.Flags, MaterialPTFlags_UseEmissiveTexture);
    GetBindlessTextureIndex(NormalTexture.Loaded, data.NormalTextureIndex, data.Flags, MaterialPTFlags_UseNormalTexture);
    GetBindlessTextureIndex(TransmissionTexture.Loaded, data.TransmissionTextureIndex, data.Flags, MaterialPTFlags_UseTransmissionTexture);

    data.Flags |= (uint)(min(NestedPriority, kMaterialMaxNestedPriority)) << MaterialPTFlags_NestedPriorityShift;
    data.Flags |= (uint)(clamp(PSDDominantDeltaLobe + 1, 0, 7)) << MaterialPTFlags_PSDDominantDeltaLobeP1Shift;

    data.ShadowNoLFadeout = std::clamp(ShadowNoLFadeout, 0.0f, 0.25f);

    data._padding0 = data._padding1 = 42;
}

MaterialsBaker::MaterialsBaker(nvrhi::IDevice* device, std::shared_ptr<donut::engine::TextureCache> textureCache, std::shared_ptr<donut::engine::ShaderFactory> shaderFactory)
    : m_device(device)
    , m_textureCache(textureCache)
    , m_bindingCache(device)
    , m_shaderFactory(shaderFactory)
{
}

void MaterialsBaker::Clear() 
{
    m_materialDataWasReset = true;
    m_materials.clear();
    m_materialsGPU.clear();
    m_textures.clear();
    m_sceneMaterialsFilePath = std::filesystem::path();
    m_mediaPath = std::filesystem::path();
}

MaterialsBaker::~MaterialsBaker()
{
}

std::shared_ptr<MaterialPT> MaterialsBaker::ImportFromDonut(donut::engine::Material& material)
{
    std::shared_ptr<MaterialPT> materialPT = std::make_shared<MaterialPT>();

    materialPT->Name = material.name;

    materialPT->BaseTexture.InitFromLoadedTexture(material.baseOrDiffuseTexture, true, false, m_mediaPath);

    if( material.useSpecularGlossModel ) // spec-gloss model is a special case hack where we use metalRoughOrSpecularTexture to store specular color, which is handled as sRGB
        materialPT->OcclusionRoughnessMetallicTexture.InitFromLoadedTexture(material.metalRoughOrSpecularTexture, true, false, m_mediaPath);
    else
        materialPT->OcclusionRoughnessMetallicTexture.InitFromLoadedTexture(material.metalRoughOrSpecularTexture, false, false, m_mediaPath);

    materialPT->NormalTexture.InitFromLoadedTexture(material.normalTexture, false, true, m_mediaPath);
    materialPT->EmissiveTexture.InitFromLoadedTexture(material.emissiveTexture, true, false, m_mediaPath);
    materialPT->TransmissionTexture.InitFromLoadedTexture(material.transmissionTexture, false, false, m_mediaPath);

    // Toggles for the textures. Only effective if the corresponding texture is non-null.
    materialPT->EnableBaseTexture = material.enableBaseOrDiffuseTexture;
    materialPT->EnableOcclusionRoughnessMetallicTexture = material.enableMetalRoughOrSpecularTexture;
    materialPT->EnableNormalTexture = material.enableNormalTexture;
    materialPT->EnableEmissiveTexture = material.enableEmissiveTexture;
    materialPT->EnableTransmissionTexture = material.enableTransmissionTexture;

    materialPT->BaseOrDiffuseColor = material.baseOrDiffuseColor;
    materialPT->SpecularColor = material.specularColor;
    materialPT->EmissiveColor = material.emissiveColor;

    materialPT->EmissiveIntensity = material.emissiveIntensity;
    materialPT->Metalness = material.metalness;
    materialPT->Roughness = material.roughness;
    materialPT->Opacity = material.opacity;
    materialPT->AlphaCutoff = material.alphaCutoff;
    materialPT->TransmissionFactor = material.transmissionFactor;
    //materialPT->DiffuseTransmissionFactor = material.diffuseTransmissionFactor;
    materialPT->NormalTextureScale = material.normalTextureScale;
    //materialPT->IoR = material.ior;
    materialPT->UseSpecularGlossModel = material.useSpecularGlossModel;
    materialPT->MetalnessInRedChannel = material.metalnessInRedChannel;
    //materialPT->ThinSurface = material.thinSurface;
    //materialPT->ExcludeFromNEE = material.excludeFromNEE;
    //materialPT->PSDExclude = material.psdExclude;
    //materialPT->PSDDominantDeltaLobe = material.psdDominantDeltaLobe;
    //materialPT->NestedPriority = material.nestedPriority;
    //materialPT->VolumeAttenuationDistance = material.volumeAttenuationDistance;
    //materialPT->VolumeAttenuationColor = material.volumeAttenuationColor;
    //materialPT->ShadowNoLFadeout = material.shadowNoLFadeout;

    materialPT->EnableAlphaTesting = (material.domain == MaterialDomain::AlphaTested || material.domain == MaterialDomain::TransmissiveAlphaTested);
    materialPT->EnableTransmission = (material.domain == MaterialDomain::Transmissive || material.domain == MaterialDomain::TransmissiveAlphaBlended || material.domain == MaterialDomain::TransmissiveAlphaTested);

    return materialPT;
}

void MaterialsBaker::SceneReloaded()
{
    Clear();
}

void MaterialsBaker::CreateRenderPassesAndLoadMaterials(nvrhi::IBindingLayout* bindlessLayout, std::shared_ptr<engine::CommonRenderPasses> commonPasses, const std::shared_ptr<donut::engine::ExtendedScene>& scene, const std::filesystem::path& sceneFilePath, const std::filesystem::path & mediaPath )
{
    assert(!mediaPath.empty());
    //m_bindlessLayout = bindlessLayout;
    m_commonPasses = commonPasses;

    {
        nvrhi::BufferDesc bufferDesc;
        bufferDesc.initialState = nvrhi::ResourceStates::ShaderResource;
        bufferDesc.keepInitialState = true;
        bufferDesc.canHaveUAVs = true;
        bufferDesc.byteSize = sizeof(MaterialPTData) * RTXPT_MATERIAL_MAX_COUNT;
        bufferDesc.structStride = sizeof(MaterialPTData);
        bufferDesc.debugName = "MaterialPTDataStorage";
        m_materialData = m_device->createBuffer(bufferDesc);
        m_materialDataWasReset = true;
    }

    // if already initialized for this scene, do not re-load materials
    if( !m_mediaPath.empty() )
        return;

    m_sceneMaterialsFilePath = sceneFilePath;
    std::filesystem::path justName = sceneFilePath.filename().stem();
    justName += ".materials.json";
    m_sceneMaterialsFilePath.remove_filename();
    m_sceneMaterialsFilePath /= justName;
    m_mediaPath = mediaPath;
  
    std::unordered_map<std::string, std::shared_ptr<MaterialPT>> loadedMaterialsPT;

    std::unordered_set<std::string> materialsPTUniqueNames;
    LoadAll(loadedMaterialsPT);

    int initializedFromDonutCount = 0;

    std::shared_ptr<SceneGraph> sceneGraph = scene->GetSceneGraph();
    auto& materials = sceneGraph->GetMaterials();
    for (auto& material : materials)
    {
        std::shared_ptr<MaterialEx> materialEx = std::dynamic_pointer_cast<MaterialEx>(material);
        if (materialEx == nullptr)
        {
            assert(false && "Is there something wrong with ExtendedSceneTypeFactory::CreateMaterial()?");
            continue;
        }
        else
        {
            auto it = loadedMaterialsPT.find(material->name);
            if (it != loadedMaterialsPT.end())
            {
                materialEx->MaterialPT = it->second;
                loadedMaterialsPT.erase(it);
            }
            else // ...and if we didn't find it in our .scene.materials.json, then import from Donut!
            {
                std::shared_ptr<MaterialPT> materialPT = ImportFromDonut(*material);
                materialEx->MaterialPT = materialPT;
                initializedFromDonutCount++;
            }

            auto existing = materialsPTUniqueNames.find(materialEx->MaterialPT->Name);
            if (existing != materialsPTUniqueNames.end() )
            {
                donut::log::error("Error while loading/converting materials for scene '%s' - there are at least two materials with the same name '%s'.\nThis is not supported and will result in errors.",
                    sceneFilePath.string().c_str(), materialEx->MaterialPT->Name.c_str());
            }
            else
                materialsPTUniqueNames.insert(materialEx->MaterialPT->Name);

            m_materials.push_back(materialEx->MaterialPT);

            m_materialsGPU.push_back(MaterialPTData{});
            materialEx->MaterialPT->GPUDataIndex = uint(m_materialsGPU.size() - 1);
            materialEx->MaterialPT->GPUDataDirty = true;
            assert(m_materialsGPU.size() <= RTXPT_MATERIAL_MAX_COUNT);
        }
    }

    // sort by name so when we're saving it's consistent
    std::sort(m_materials.begin(), m_materials.end(), [](const auto & a, const auto & b) { return a->Name < b->Name; } );

    if (loadedMaterialsPT.size() > 0)
        donut::log::warning("There were %d materials in RTXPT material definition file '%s' that were not used by the scene meshes; consider doing Scene->Materials->Advanced->Save", (int)loadedMaterialsPT.size(), m_sceneMaterialsFilePath.string().c_str()); 
    if (initializedFromDonutCount > 0)
        donut::log::warning("There were %d materials not found in RTXPT material definition file '%s'; consider doing Scene->Materials->Advanced->Save", initializedFromDonutCount, m_sceneMaterialsFilePath.string().c_str());

    // In case new textures were loaded, we need to make sure they were uploaded properly
    m_textureCache->ProcessRenderingThreadCommands(*m_commonPasses, 0.f);
    m_textureCache->LoadingFinished();

    // currently unused
    auto recordTexture = [&]( const MaterialPT::Texture & texture )
    {
        if( texture.Loaded == nullptr ) return;
        
        assert( texture.LocalPath != "" );
    
        auto existing = m_textures.find( texture.LocalPath.generic_string() );

        if (existing != m_textures.end())
        {
            if( existing->second.NormalMap != texture.NormalMap )
            { donut::log::warning("Texture with path '%s' is used as a NormalMap and not a NormalMap - this is not supported, expect errors.", texture.LocalPath.string().c_str()); assert( false ); }
            if( existing->second.sRGB != texture.sRGB )
            { donut::log::warning("Texture with path '%s' is marked as both sRGB and not sRGB in different places - this is not supported, expect errors.", texture.LocalPath.string().c_str()); assert( false ); }
        }
        else
            m_textures.insert( std::make_pair(texture.LocalPath.generic_string(), texture) );
    };
    for (auto& materialPT : m_materials)
    {
        recordTexture(materialPT->BaseTexture);
        recordTexture(materialPT->OcclusionRoughnessMetallicTexture);
        recordTexture(materialPT->NormalTexture);
        recordTexture(materialPT->EmissiveTexture);
        recordTexture(materialPT->TransmissionTexture);
    }
}

void MaterialsBaker::Update(nvrhi::ICommandList* commandList, const std::shared_ptr<donut::engine::ExtendedScene>& scene, std::vector<SubInstanceData>& subInstanceData)
{
    RAII_SCOPE( commandList->beginMarker("MaterialsBaker");, commandList->endMarker(); );

    bool needsUpload = false;
    for (auto& materialPT : m_materials)
    {
        if (!materialPT->GPUDataDirty && !m_materialDataWasReset)
            continue;

        materialPT->FillData(m_materialsGPU[materialPT->GPUDataIndex]);
        materialPT->GPUDataDirty = false;
        needsUpload = true;
    }

    if ( needsUpload )
    {
        commandList->writeBuffer( m_materialData, m_materialsGPU.data(), m_materialsGPU.size() * sizeof(MaterialPTData), 0 );
        m_materialDataWasReset = false;
    }

    uint subInstanceIndex = 0;
    const auto& instances = scene->GetSceneGraph()->GetMeshInstances();
    for (const auto& instance : instances)
    {
        const auto& mesh = instance->GetMesh();
        uint32_t firstGeometryInstanceIndex = instance->GetGeometryInstanceIndex();
        for (size_t geometryIndex = 0; geometryIndex < mesh->geometries.size(); ++geometryIndex, subInstanceIndex++)
        {
            const auto& geometry = mesh->geometries[geometryIndex];
            assert( geometry->material != nullptr && "No handling for null materials!" );
            std::shared_ptr<MaterialPT> materialPT = MaterialPT::FromDonut(geometry->material);
            assert(materialPT != nullptr && "Unknown error - should never have happened" );
            uint globalMaterialIndex = materialPT->GPUDataIndex;
            subInstanceData[subInstanceIndex].GlobalGeometryIndex_MaterialPTDataIndex = (subInstanceData[subInstanceIndex].GlobalGeometryIndex_MaterialPTDataIndex & 0xFFFF0000) | globalMaterialIndex;
        }
    }
}

void MaterialsBaker::LoadAll(std::unordered_map<std::string, std::shared_ptr<MaterialPT>>& container)
{
    std::ifstream inFile(m_sceneMaterialsFilePath);

    if (!inFile.is_open())
        { donut::log::warning("No RTXPT material definition file found at '%s' - consider doing Scene->Materials->Advanced->Save", m_sceneMaterialsFilePath.string().c_str()); return; }

    Json::Value rootJ;
    inFile >> rootJ;

    int version = -1;
    rootJ["RTXPTMaterials"]["version"] >> version;
    if (version != 1)
        { donut::log::warning("Malformed or unsupported RTXPT material definition file version '%s' - consider doing Scene->Materials->Advanced->Save", m_sceneMaterialsFilePath.string().c_str()); return; }

    Json::Value materialsJ;

    materialsJ = rootJ["materials"];
    if (materialsJ.empty() || !materialsJ.isArray())
        { donut::log::warning("Malformed or empty material definition file '%s' - consider doing Scene->Materials->Advanced->Save", m_sceneMaterialsFilePath.string().c_str()); return; }

    for ( Json::Value materialJ : materialsJ )
    {
        std::shared_ptr<MaterialPT> materialPT = materialPT->FromJson(materialJ, m_mediaPath, m_textureCache);
        if (materialPT == nullptr)
            { donut::log::warning("Error while reading material in material definition file '%s'", m_sceneMaterialsFilePath.string().c_str()); continue; }
        
        auto existing = container.find(materialPT->Name);
        if (existing != container.end())
            { donut::log::warning("Duplicated materials with name '%s' found in material definition file '%s' - subsequent instances ignored.", materialPT->Name.c_str(), m_sceneMaterialsFilePath.string().c_str()); assert( false ); continue; }
        else
            container.insert( make_pair(materialPT->Name, materialPT) );
    }
}

void MaterialsBaker::SaveAll()
{
    Json::Value rootJ;

    rootJ["RTXPTMaterials"]["version"] = 1;
    Json::Value materialsJ;
    auto saveTexture = [&](Json::Value & materialJ, const std::shared_ptr<donut::engine::LoadedTexture>& texture, const std::string & name)
    {
        if( texture == nullptr )
            return;
        std::filesystem::path localPath = std::filesystem::relative(texture->path, m_mediaPath);
        materialJ[name] = localPath.string();
    };
    for (auto& materialPT : m_materials)
    {
        Json::Value materialJ;
        materialPT->Write(materialJ, m_mediaPath);
        materialsJ.append(materialJ);
    }

    rootJ["materials"] = materialsJ;

    std::ofstream outFile(m_sceneMaterialsFilePath, std::ios::trunc);

    Json::StreamWriterBuilder builder;
    std::unique_ptr<Json::StreamWriter> writer(builder.newStreamWriter());
    
    writer->write(rootJ, &outFile);
    outFile.close();
}

bool MaterialsBaker::DebugGUI(float indent)
{
    RAII_SCOPE(ImGui::PushID("MaterialsBakerDebugGUI"); , ImGui::PopID(); );
    
    bool resetAccumulation = false;
    #define IMAGE_QUALITY_OPTION(code) do{if (code) resetAccumulation = true;} while(false)

    ImGui::Text("Scene material count: %d", (int)m_materials.size());
    ImGui::Text("Material texture use count: %d", (int)m_textures.size());

    // ImGui::Separator();
    // if (ImGui::CollapsingHeader("Debugging", ImGuiTreeNodeFlags_DefaultOpen))
    // {
    //     RAII_SCOPE(ImGui::Indent(indent); , ImGui::Unindent(indent););
    // 
    //     ImGui::Text("<shrug>");
    // }
    // ImGui::Separator();

    if (ImGui::CollapsingHeader("Advanced", 0/*ImGuiTreeNodeFlags_DefaultOpen*/))
    {
        ImGui::Text("Materials file path:");
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0, 1.0, 0.5, 1.0));
        ImGui::TextWrapped("%s", m_sceneMaterialsFilePath.string().c_str());
        ImGui::PopStyleColor();

        if( ImGui::Button("Save to file") )
            SaveAll();
    }

    return resetAccumulation;
}

