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

#include <filesystem>
#include <memory>

#include <donut/engine/BindingCache.h>
#include <nvrhi/nvrhi.h>
#include <donut/core/math/math.h>
#include <donut\engine\SceneTypes.h>

#include "../ComputePass.h"
#include "../Shaders/SubInstanceData.h"
#include "../Shaders/PathTracer/Materials/MaterialPT.h"

#include <unordered_map>

using namespace donut::math;

namespace donut::engine
{
    class FramebufferFactory;
    class TextureCache;
    class TextureHandle;
    class ShaderFactory;
    class CommonRenderPasses;
    struct TextureData;
    struct LoadedTexture;
}

class ShaderDebug;
class ExtendedScene;

struct MaterialShadingProperties
{
    bool AlphaTest;
    bool HasTransmission;
    bool NoTransmission;
    bool OnlyDeltaLobes;
    bool NoTextures;
    bool ExcludeFromNEE;

    bool operator==(const MaterialShadingProperties& other) const { return AlphaTest == other.AlphaTest && HasTransmission == other.HasTransmission && NoTransmission == other.NoTransmission && OnlyDeltaLobes == other.OnlyDeltaLobes && NoTextures == other.NoTextures && ExcludeFromNEE == other.ExcludeFromNEE; };
    bool operator!=(const MaterialShadingProperties& other) const { return !(*this == other); }

    static MaterialShadingProperties Compute(const struct PTMaterial& material);
};

struct PTTexture
{
    std::filesystem::path   LocalPath;
    bool                    sRGB;           // whether to assume that, when loading from sRGB agnostic formats, the texture's .rgb channels are in sRGB (.a is always linear)
    std::shared_ptr<donut::engine::LoadedTexture>
        Loaded;
    bool                    NormalMap;      // determines unpacking (not actually used as a flag now by shading, but normalmaps are marked as so for future use)

    bool                    Enabled;        // an easy way to disable/enable texture slot without actually disconnecting a texture

    // float4                  ValueDefault;   // when texture is not enabled or can't be loaded
    // float4                  ValueMultiply;
    // float4                  ValueAdd;

    void                    InitFromLoadedTexture(std::shared_ptr<donut::engine::LoadedTexture>& loaded, bool sRGB, bool normalMap, const std::filesystem::path& mediaPath);
};

// All materials share these base properties and some of them have tight integration with the rest of the renderer
struct PTMaterialBase
{
    std::string             Name;

    // base texture? & alpha test - opacity?
    // emissive texture?

    bool                    SharedWithAllScenes                 = true;    // if 'true', will be saved to MaterialsBaker::m_materialsPath; otherwise in MaterialsBaker::m_materialsSceneSpecializedPath

    virtual void            Write(Json::Value & output) = 0;
    virtual bool            Read(Json::Value & output, const std::filesystem::path& mediaPath, const std::shared_ptr<donut::engine::TextureCache>& textureCache) = 0;
};                                                                                                                      

struct PTMaterial : public PTMaterialBase
{
    PTTexture               BaseTexture;                        // .rgb base color; .a = opacity (both modes)
    PTTexture               OcclusionRoughnessMetallicTexture;  // .rgb ORM; (spec-gloss fallback: specular color, .a = glossiness)
    PTTexture               NormalTexture;
    PTTexture               EmissiveTexture;
    PTTexture               TransmissionTexture;                // see KHR_materials_transmission; undefined on specular-gloss materials

    dm::float3              BaseOrDiffuseColor                  = 1.f; // metal-rough: base color, spec-gloss: diffuse color (if no texture present)
    dm::float3              SpecularColor                       = 0.f; // spec-gloss: specular color
    dm::float3              EmissiveColor                       = 0.f;
    
    float                   EmissiveIntensity                   = 1.f; // additional multiplier for emissiveColor
    float                   Metalness                           = 0.f; // metal-rough only
    float                   Roughness                           = 0.f; // both metal-rough and spec-gloss
    float                   Opacity                             = 1.f; // for transparent materials; multiplied by diffuse.a if present
    float                   TransmissionFactor                  = 0.f; // see KHR_materials_transmission; undefined on specular-gloss materials
    float                   DiffuseTransmissionFactor           = 0.f; // like specularTransmissionFactor, except using diffuse transmission lobe (roughness ignored)
    float                   NormalTextureScale                  = 1.f;
    float                   IoR                                 = 1.5f; // index of refraction, see KHR_materials_ior

    // Toggle between two PBR models: metal-rough and specular-gloss.
    // See the comments on the other fields here.
    bool                    UseSpecularGlossModel = false;

    // Toggles for the textures. Only effective if the corresponding texture is non-null.
    bool                    EnableBaseTexture                   = true;
    bool                    EnableOcclusionRoughnessMetallicTexture   = true;
    bool                    EnableNormalTexture                 = true;
    bool                    EnableEmissiveTexture               = true;
    bool                    EnableTransmissionTexture           = true;

    bool                    EnableAlphaTesting                  = false;
    float                   AlphaCutoff                         = 0.5f; // for alpha tested materials

    bool                    EnableTransmission                  = false;

    // Useful when metalness and roughness are packed into a 2-channel texture for BC5 encoding.
    bool                    MetalnessInRedChannel               = false;

    // As per Falcor/RTXPT convention, ray hitting a material with the thin surface is assumed to enter and leave surface in the same bounce and it makes most sense when used with doubleSided; it skips all volume logic.
    bool                    ThinSurface                         = false;

    // The mesh will not be part of NEE.
    bool                    ExcludeFromNEE                      = false;

    // will not propagate dominant stable plane when doing path space decomposition
    bool                    PSDExclude                          = false;
    // for path space decomposition: -1 means no dominant; 0 usually means transmission, 1 usually means reflection, 2 usually means clearcoat reflection - must match corresponding BSDFSample::getDeltaLobeIndex()!
    int                     PSDDominantDeltaLobe                = -1;

    // When volume meshes overlap, will cause higher nestedPriority mesh to 'carve out' the volumes with lower nestedPriority (see https://www.sidefx.com/docs/houdini/render/nested.html)
    static constexpr int kMaterialMaxNestedPriority = 14;
    int                     NestedPriority                      = kMaterialMaxNestedPriority;

    // KHR_materials_volume - see https://github.com/KhronosGroup/glTF/tree/main/extensions/2.0/Khronos/KHR_materials_volume#properties
    float                   VolumeAttenuationDistance           = FLT_MAX;
    dm::float3              VolumeAttenuationColor              = 1.0f;

    // Low tessellation geometry often has triangle (flat) normals that differ significantly from shading normals. This causes shading vs shadow discrepancy that exposes triangle edges. 
    // One way to mitigate this (other than having more detailed mesh) is to add additional shadowing falloff to hide the seam. 
    // This setting is not physically correct and adds bias. Setting of 0 means no fadeout (default).
    float                   ShadowNoLFadeout                    = 0.0f;

    bool                    GPUDataDirty                        = true;         // params changed, GPU data needs update
    uint                    GPUDataIndex                        = 0xFFFFFFFF;   // 0xFFFFFFFF if no GPU buffer slot allocated

    bool                    EnableAsAnalyticLightProxy          = false;

    void                    FillData(PTMaterialData & data);
    bool                    EditorGUI(class MaterialsBaker & baker);
    bool                    IsEmissive() const;

    static std::shared_ptr<PTMaterial> FromDonut(const std::shared_ptr<donut::engine::Material>& donutMaterial);
    static std::shared_ptr<PTMaterial> FromJson(Json::Value& input, const std::filesystem::path& mediaPath, const std::shared_ptr<donut::engine::TextureCache>& textureCache);

    virtual void            Write(Json::Value & output) override;
    virtual bool            Read(Json::Value & output, const std::filesystem::path& mediaPath, const std::shared_ptr<donut::engine::TextureCache>& textureCache) override;
};

struct MaterialEx : donut::engine::Material
{
    std::shared_ptr<PTMaterial> PTMaterial;              
};

class MaterialsBaker
{
public:
    MaterialsBaker(nvrhi::IDevice* device, std::shared_ptr<donut::engine::TextureCache> textureCache, std::shared_ptr<donut::engine::ShaderFactory> shaderFactory);
    ~MaterialsBaker();

    void                            CreateRenderPassesAndLoadMaterials(nvrhi::IBindingLayout* bindlessLayout, std::shared_ptr<donut::engine::CommonRenderPasses> commonPasses, const std::shared_ptr<ExtendedScene>& scene, const std::filesystem::path & sceneFilePath, const std::filesystem::path & mediaPath);

    // this update can happen in parallel with any other ray preparatory tracing work - anything from BVH building to laying down denoising layers
    void                            Update(nvrhi::ICommandList * commandList, const std::shared_ptr<ExtendedScene> & scene, std::vector<SubInstanceData> & subInstanceData);

    nvrhi::BufferHandle             GetMaterialDataBuffer() const           { return m_materialData; }
    uint                            GetMaterialDataCount() const            { return m_materialsGPU.size(); }

    const std::unordered_map<std::string, PTTexture> &
                                    GetUsedTextures() const                 { return m_textures; }

    bool                            DebugGUI(float indent);

    void                            SceneReloaded();

    std::filesystem::path           GetMaterialStoragePath(PTMaterialBase& material);

    bool                            SaveSingle(PTMaterialBase& material);
    bool                            LoadSingle(PTMaterialBase& material);

private:
    void                            Clear();

    std::shared_ptr<PTMaterial>     Load(const std::string& name);
    std::shared_ptr<PTMaterial>     ImportFromDonut(donut::engine::Material & material);
    void                            SaveAll();

    void                            CompleteDeferredTexturesLoad(nvrhi::ICommandList* commandList);

private:
    nvrhi::DeviceHandle             m_device;
    std::shared_ptr<donut::engine::TextureCache> m_textureCache;
    std::shared_ptr<donut::engine::CommonRenderPasses> m_commonPasses;
    std::shared_ptr<donut::engine::FramebufferFactory> m_framebufferFactory;
    std::shared_ptr<donut::engine::ShaderFactory> m_shaderFactory;

    donut::engine::BindingCache     m_bindingCache;

    nvrhi::BufferHandle             m_materialData;
    bool                            m_materialDataWasReset = true;
    bool                            m_deferredTextureLoadInProgress = false;

    std::vector<std::shared_ptr<PTMaterial>>    
                                    m_materials;
    std::vector<PTMaterialData>     m_materialsGPU;

    std::unordered_map<std::string, PTTexture> m_textures;

    std::filesystem::path           m_mediaPath;
    std::filesystem::path           m_materialsPath;                    // usually "Assets/Materials/"              <- used for materials shared between all scenes
    std::filesystem::path           m_materialsSceneSpecializedPath;    // usually "Assets/Materials/SceneName/"    <- used for materials specific to scene (not shared between scenes)
};
