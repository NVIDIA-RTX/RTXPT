/*
* Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#ifndef __SAMPLE_COMMON_H__ // using instead of "#pragma once" due to https://github.com/microsoft/DirectXShaderCompiler/issues/3943
#define __SAMPLE_COMMON_H__

#include "Shaders/PathTracer/Config.h"

#include <utility>
#include <filesystem>
#include <assert.h>
#include <mutex>


#define TOKEN_COMBINE1(X,Y) X##Y  // helper macro
#define TOKEN_COMBINE(X,Y) TOKEN_COMBINE1(X,Y)

////////////////////////////////////////////////////////////////////////////////////////////////
// Custom generic RAII helper
template< typename AcquireType, typename FinalizeType >
class GenericScope
{
    FinalizeType            m_finalize;
public:
    GenericScope(AcquireType&& acquire, FinalizeType&& finalize) : m_finalize(std::move(finalize)) { acquire(); }
    ~GenericScope() { m_finalize(); }
};
// Should expand to something like: GenericScope scopevar_1( [ & ]( ) { ImGui::PushID( Scene::Components::TypeName( i ).c_str( ) ); }, [ & ]( ) { ImGui::PopID( ); } );
#define RAII_SCOPE( enter, leave ) GenericScope TOKEN_COMBINE( _generic_raii_scopevar_, __COUNTER__ ) ( [&](){ enter }, [&](){ leave } );
// Usage example: RAII_SCOPE( ImGui::PushID( keyID );, ImGui::PopID( ); )
////////////////////////////////////////////////////////////////////////////////////////////////

constexpr static const char * c_EnvMapProcSky           = "==PROCEDURAL_SKY==";
constexpr static const char * c_EnvMapProcSky_Morning   = "==PROCEDURAL_SKY_MORNING==";
constexpr static const char * c_EnvMapProcSky_Midday    = "==PROCEDURAL_SKY_MIDDAY==";
constexpr static const char * c_EnvMapProcSky_Evening   = "==PROCEDURAL_SKY_EVENING==";
constexpr static const char * c_EnvMapProcSky_Dawn      = "==PROCEDURAL_SKY_DAWN==";
constexpr static const char * c_EnvMapProcSky_PitchBlack= "==PROCEDURAL_SKY_PITCHBLACK==";
constexpr static const char * c_EnvMapSceneDefault      = "==SCENE_DEFAULT==";
constexpr static const char * c_AssetsFolder            = "Assets";
constexpr static const char * c_EnvMapSubFolder         = "EnvironmentMaps";
constexpr static const char * c_MaterialsSubFolder      = "Materials";
constexpr static const char * c_MaterialsExtension      = ".material.json";
constexpr static const char * c_SampleGameSubFolder     = "SampleGame";

inline bool IsProceduralSky( const char * str )         { if (str == nullptr) return false; for (int i = 0; i < 12; i++ ) if (str[i] != c_EnvMapProcSky[i]) return false; return true; }

bool EnsureDirectoryExists( const std::filesystem::path & dir );
std::vector<std::filesystem::path> EnumerateFilesWithWildcard( const std::filesystem::path& folder, const std::string& wildcard );
std::optional<std::filesystem::file_time_type> GetLatestModifiedTimeDirectoryRecursive(const std::filesystem::path & directory);
std::optional<std::filesystem::file_time_type> GetFileModifiedTime(const std::filesystem::path & file);

namespace Json { class Value; }
bool SaveJsonToFile( const std::filesystem::path & filePath, const Json::Value & rootNode );
bool LoadJsonFromFile( const std::filesystem::path & filePath, Json::Value & outRootNode );
bool LoadJsonFromString(const std::string & jsonData, Json::Value& outRootNode);
//inline Json::Value LoadJsonFromFile(const std::filesystem::path& filePath)                      { Json::Value ret; LoadJsonFromFile(filePath, ret); return ret; }
//inline Json::Value LoadJsonFromString(const std::string& jsonData)                              { Json::Value ret; LoadJsonFromString(jsonData, ret); return ret; }

std::string StringLoadFromFile( const std::filesystem::path & filePath );

template<typename ... Args> std::string StringFormat(const std::string& format, Args ... args)
{
    int size_s = std::snprintf(nullptr, 0, format.c_str(), args ...) + 1; // Extra space for '\0'
    if (size_s <= 0) { throw std::runtime_error("Error during formatting."); }
    auto size = static_cast<size_t>(size_s);
    std::unique_ptr<char[]> buf(new char[size]);
    std::snprintf(buf.get(), size, format.c_str(), args ...);
    return std::string(buf.get(), buf.get() + size - 1); // We don't want the '\0' inside
}
std::filesystem::path GetLocalPath(std::string subfolder);

void HelpersRegisterActiveWindow(); // call this from process main thread to grab current main window; this is optional and only used for progress bar to appear centered in the main window
int  ProgressBarStart(const char * windowText);
void ProgressBarStop(int slotIndex);
void ProgressBarUpdate(int slotIndex, int percentage);

class ProgressBar
{
public:
    ProgressBar()                               { }
    ProgressBar(const char * windowText)        { Start(windowText); }
    ~ProgressBar()                              { Stop(); }

    bool        Start(const char * windowText)  { std::lock_guard lock(m_mtx); assert( !Active() ); m_slot = ProgressBarStart(windowText); return Active(); }
    void        Set(int percentage)             { std::lock_guard lock(m_mtx); if (percentage<0) percentage = 0; if (percentage>100) percentage = 100; if (m_slot!=-1) ProgressBarUpdate(m_slot, percentage); }
    void        Stop()                          { std::lock_guard lock(m_mtx); if (Active()) ProgressBarStop(m_slot); m_slot = -1; }
    bool        Active() const                  { std::lock_guard lock(m_mtx); return m_slot != -1; }

private:
    mutable std::recursive_mutex m_mtx;
    int         m_slot = -1;
};

// result, outputText, errorText
std::tuple<int, std::string, std::string > SystemShell(const std::string & command, bool useCmd = false, bool blockOnExecution = true);

#endif // __SAMPLE_COMMON_H__
