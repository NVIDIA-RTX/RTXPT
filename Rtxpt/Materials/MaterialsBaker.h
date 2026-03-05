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

#include "../SampleCommon/ComputePass.h"
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

struct MaterialShaderPermutation
{
    const std::string   ShaderFilePath;
    // const std::string   ClosestHitName;
    // const std::string   AnyHitName;

    const std::vector<donut::engine::ShaderMacro> 
                        Macros;

    // runtime data
    int                 IndexInTable            = -1;
    std::string         UniqueMaterialName      = "";               // not deterministic! but useful for debugging anyway

    //MaterialShaderPermutation(const std::string & shaderFilePath, const std::string & closestHitName, const std::string & anyHitName, const std::vector<std::pair<std::string, std::string>> & macros );
};

struct MaterialShaderPermutationKey
{
    //    const std::array<unsigned char, 32/*picosha2::k_digest_size*/>  Hash;
    std::size_t                                                     Hash;

    const std::string                                               FullKey;

    MaterialShaderPermutationKey( const MaterialShaderPermutation & msp );

    bool operator == (const MaterialShaderPermutationKey & other) const = default;    // technically we only need to compare FullKey
};

// Custom specialization of std::hash can be injected in namespace std.
struct MaterialShaderPermutationKeyHash
{
    // long hash to short hash
    std::size_t operator()(const MaterialShaderPermutationKey & s) const noexcept   { return s.Hash; }
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
    donut::engine::Material * DonutCounterpart = nullptr;

    // ModelName + Name is unique identifier for the material; there cannot be two materials with the same ModelName and Name - hopefully.
    std::string             Name;
    std::string             ModelName;                                  // material can be saved

    std::string             UniqueName;                                 // mix of Name and ModelName but shortened and hashed - or whatever makes sense and is short

    // base texture? & alpha test - opacity?
    // emissive texture?

    bool                    SharedWithAllScenes                 = true;     // if 'true', will be saved to MaterialsBaker::m_materialsPath; otherwise in MaterialsBaker::m_materialsSceneSpecializedPath

    // this gets set internally by MaterialsBaker::BakeShaderPermutations - it will be shared with other materials when possible
    std::shared_ptr<MaterialShaderPermutation>
                            BakedShaderPermutation;

    virtual void            Write(Json::Value & output) = 0;
    virtual bool            Read(Json::Value & output, const std::filesystem::path& mediaPath, const std::shared_ptr<donut::engine::TextureCache>& textureCache) = 0;

    virtual bool            HasAlphaTest() const = 0;

    /*virtual*/ std::string     UniqueFullName()                            { return UniqueName; }

    virtual MaterialShaderPermutation ComputeShaderPermutation(const std::string & defaultShaderPath) = 0;
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
    bool                    PSDExclude                          = true;
    // for path space decomposition: -1 means no dominant; 0 usually means transmission, 1 usually means reflection, 2 usually means clearcoat reflection - must match corresponding BSDFSample::getDeltaLobeIndex()!
    int                     PSDDominantDeltaLobe                = -1;
    // this surface is too curved or otherwise problematic to pass motion vectors; other decomposition can continue though
    int                     PSDBlockMotionVectorsAtSurfaceType  = 0;        // 0 - Off; 1 - AutoLow; 2 - AutoHigh; 3 - Full

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

    bool                    IgnoreMeshTangentSpace              = false;

    bool                    UseDonutEmissiveIntensity           = false;        // for being able to use Donut animations
    
    bool                    SkipRender                          = false;        // if 'true', we just skip drawing all geometries with this material; sometimes we can't edit a specific mesh but we can remove it this way; note: it can also be used for hidden emissives

    void                    FillData(PTMaterialData & data);
    bool                    EditorGUI(class MaterialsBaker & baker);
    bool                    IsEmissive() const;

    static std::shared_ptr<PTMaterial> SafeCast(const std::shared_ptr<donut::engine::Material>& donutMaterial);

    static std::shared_ptr<PTMaterial> FromJson(Json::Value& input, const std::filesystem::path& mediaPath, const std::shared_ptr<donut::engine::TextureCache>& textureCache, const std::string & modelName, const std::string & name);

    virtual void            Write(Json::Value & output) override;
    virtual bool            Read(Json::Value & output, const std::filesystem::path& mediaPath, const std::shared_ptr<donut::engine::TextureCache>& textureCache) override;

    virtual MaterialShaderPermutation 
                            ComputeShaderPermutation(const std::string & defaultShaderPath) override;

    virtual bool            HasAlphaTest() const override       { return EnableAlphaTesting; }
};

struct MaterialEx : donut::engine::Material
{
    std::shared_ptr<PTMaterial> PTMaterial;              
};

class MaterialsBaker
{
public:
    MaterialsBaker(const std::string & relativeShaderSourcePath, nvrhi::IDevice* device, std::shared_ptr<donut::engine::TextureCache> textureCache, std::shared_ptr<donut::engine::ShaderFactory> shaderFactory);
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

    const std::shared_ptr<MaterialShaderPermutation> & 
                                    GetUbershader() const                   { return m_ubershader; }
    const std::vector<std::shared_ptr<MaterialShaderPermutation>> & 
                                    GetShaderPermutationTable() const       { return m_shaderPermutationTable; }

    bool                            SaveSingle(PTMaterialBase& material);
    bool                            LoadSingle(PTMaterialBase& material);

    std::shared_ptr<PTMaterial>     FindByUniqueID(const std::string & name);

private:
    void                            Clear();

    std::shared_ptr<PTMaterial>     Load(const std::string & modelFileName, const std::string& name);
    std::shared_ptr<PTMaterial>     ImportFromDonut(donut::engine::Material & material);
    void                            SaveAll();

    void                            CompleteDeferredTexturesLoad(nvrhi::ICommandList* commandList);

    void                            BakeShaderPermutations();

    void                            InitializeUniqueDeterministicName(const std::shared_ptr<PTMaterialBase> & material);

private:
    nvrhi::DeviceHandle             m_device;
    std::string                     m_relativeShaderSourcePath;         // this is the path for the shader file containing material specializations for ClosestHit and (if enabled) AnyHit; it is currently 1 for all materials, but could be per-material
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

    // this ClosestHit/AnyHit should work for all materials (ubershader, not efficient) - it will likely be removed in the future, with only the specialized supported
    std::shared_ptr<MaterialShaderPermutation>
                                    m_ubershader;
    
    // these are per-material ClosestHit/AnyHit shaders (some materials share same); m_shaderPermutations and m_shaderPermutationTable are always updated together and point to same
    std::unordered_map<MaterialShaderPermutationKey, std::shared_ptr<MaterialShaderPermutation>, MaterialShaderPermutationKeyHash>
                                    m_shaderPermutations;
    std::vector<std::shared_ptr<MaterialShaderPermutation>> m_shaderPermutationTable;

    std::unordered_map<std::string, std::weak_ptr<PTMaterialBase> > m_uniqueNames;
};
