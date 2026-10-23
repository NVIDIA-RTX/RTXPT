/*
* Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "ComputePipelineBaker.h"
#include "ShaderCompilerUtils.h"
#include "SampleCommon.h"
#include "DirectoryModifiedMonitor.h"

#include <donut/app/ApplicationBase.h>
#include <donut/core/log.h>
#include <donut/core/vfs/VFS.h>

#define COMPUTE_BAKER_ENABLE_MULTITHREADED_COMPILE 1
#define COMPUTE_BAKER_EMBED_PDBS 1
#define COMPUTE_BAKER_USE_OPTIMIZATIONS 1

using namespace donut;
using namespace donut::engine;

static const std::string c_ComputeShaderBinariesRoot = "ShaderDynamic/Bin";

//////////////////////////////////////////////////////////////////////////
// ComputePipelineBaker implementation
//////////////////////////////////////////////////////////////////////////

ComputePipelineBaker::ComputePipelineBaker(nvrhi::IDevice* device, const std::vector<std::filesystem::path>& additionalMonitorPaths)
    : m_device(device)
{
    if (!m_compilerConfig.Initialize(device, c_ComputeShaderBinariesRoot))
        log::fatal("Failed to initialize compute shader compiler configuration");

    m_shadersFS = std::make_shared<vfs::RootFileSystem>();
    m_shadersFS->mount("/" + c_ComputeShaderBinariesRoot, m_compilerConfig.ShaderBinariesPath);
    
    // Store additional paths to monitor (converted to absolute)
    for (const auto& path : additionalMonitorPaths)
    {
        m_additionalMonitorPaths.push_back(std::filesystem::absolute(path));
    }
    
    // One group covering ShadersPath plus every additional path, so a query is a single fold. ShadersPath is shared with
    // PTPipelineBaker and the monitor canonicalises, so that root ends up scanned once for both.
    std::vector<std::filesystem::path> monitored;
    monitored.push_back(m_compilerConfig.ShadersPath);
    monitored.insert(monitored.end(), m_additionalMonitorPaths.begin(), m_additionalMonitorPaths.end());
    m_sourceMonitor = DirectoryModifiedMonitor::Register(monitored);

    log::info("ComputePipelineBaker initialized (monitoring %d additional paths)", (int)m_additionalMonitorPaths.size());
}

ComputePipelineBaker::~ComputePipelineBaker()
{
}

std::shared_ptr<ComputeShaderVariant> ComputePipelineBaker::CreateVariant(
    const std::string& shaderSourcePath,
    const std::string& entryPoint,
    const std::vector<ShaderMacro>& macros,
    const nvrhi::BindingLayoutVector& bindingLayouts,
    const std::string& debugName)
{
    std::shared_ptr<ComputeShaderVariant> variant = std::shared_ptr<ComputeShaderVariant>(
        new ComputeShaderVariant(shaderSourcePath, entryPoint, macros, bindingLayouts, debugName, shared_from_this()));
    m_variants.push_back(variant);
    return variant;
}

void ComputePipelineBaker::ReleaseVariant(std::shared_ptr<ComputeShaderVariant>& variant)
{
    if (variant == nullptr)
        return;

    for (int i = int(m_variants.size()) - 1; i >= 0; i--)
    {
        if (m_variants[i] == variant)
        {
            m_variants.erase(m_variants.begin() + i);
            variant = nullptr;
            return;
        }
    }
    assert(false); // Variant not found
}

void ComputePipelineBaker::EnqueueShaderForCompilation(ComputeShaderVariant* variant)
{
    if (variant->m_compileCmdLine != "")
    {
        auto [it, inserted] = m_parallelCompileListUnique.try_emplace(variant->m_compiledFileNameNoExt, variant);
        // If not inserted, it means another variant with the same output already exists
        // (unlikely for compute shaders with unique debug names, but handled for safety)
    }
    m_parallelCompileListAll.push_back(variant);
}

void ComputePipelineBaker::Update(bool forceReload)
{
    // Auto-reload: check for source file changes. The monitor's worker does the walking; the registered group already
    // folds ShadersPath and every additional path, so this is one query.
    if (!forceReload && !m_variants.empty())
    {
        ZoneScopedN("ComputePipelineHotReloadCheck");
        auto currentTimestamp = DirectoryModifiedMonitor::Query(m_sourceMonitor);

        if (currentTimestamp.has_value())
        {
            if (!m_cachedSourceTimestamp.has_value())
            {
                // First observation - just cache the timestamp
                m_cachedSourceTimestamp = currentTimestamp;
            }
            else if (*currentTimestamp != *m_cachedSourceTimestamp)
            {
                // Source files changed - trigger reload
                m_cachedSourceTimestamp = currentTimestamp;
                forceReload = true;
                log::info("Compute shader source changes detected - triggering hot reload...");
            }
        }
    }

    // Check if any variants need updating
    bool needsUpdate = forceReload;
    
    for (const auto& variant : m_variants)
    {
        if (variant.use_count() == 1)
        {
            assert(false); // Dangling variant - forgotten a call to ReleaseVariant?
        }
        if (variant->NeedsUpdate())
        {
            needsUpdate = true;
            break;
        }
    }

    if (!needsUpdate && !forceReload)
        return;

    // Get latest source modification time (from all monitored paths)
    EnsureDirectoryExists(m_compilerConfig.ShaderBinariesPath);
    
    // ScanNow, not Query: this feeds ShaderCompilerUtils::IsCompiledShaderUpToDate ('binaryTime >= lastSourceModified'
    // per shader), so a cached timestamp that is up to one poll interval stale could silently skip a compile. Only runs
    // when a reload is already happening, where the walk is free next to the compilation it precedes. The registered
    // group already covers ShadersPath plus every additional path.
    m_lastUpdatedSourceTimestamp = DirectoryModifiedMonitor::ScanNow(m_sourceMonitor);
    
    if (!m_lastUpdatedSourceTimestamp.has_value())
    {
        log::error("Unable to get source timestamp - cannot dynamically compile compute shaders");
        return;
    }

    // Sync the watcher's baseline with what this batch compiles; otherwise an edit it hasn't polled yet (e.g. saved just
    // before Ctrl+R) triggers a second, redundant reload.
    m_cachedSourceTimestamp = m_lastUpdatedSourceTimestamp;

    m_version++;

    // Compilation loop with retry on error
    do
    {
        assert(m_parallelCompileListAll.empty() && m_parallelCompileListUnique.empty());

        std::vector<std::shared_ptr<ComputeShaderVariant>> updateQueue;
        
        for (auto& variant : m_variants)
        {
            if (variant->m_localVersion != m_version || forceReload)
            {
                updateQueue.push_back(variant);
                variant->m_lockedBaker = shared_from_this();
                variant->PrepareCompilation(*m_lastUpdatedSourceTimestamp);
                EnqueueShaderForCompilation(variant.get());
            }
        }

        if (updateQueue.empty())
            break;

        // Progress tracking
        std::atomic_int progressCounterCompleted;
        int progressTotal;

        ProgressBar progressCompilingShaders;
        progressCounterCompleted = 0;
        progressTotal = (int)m_parallelCompileListUnique.size();
        
        if (progressTotal > 0)
            progressCompilingShaders.Start(StringFormat("Compiling compute shaders (%d)...", progressTotal).c_str());

        // Compile shaders (potentially in parallel)
        for (auto& [name, variant] : m_parallelCompileListUnique)
        {
            if (variant->m_compileCmdLine == "")
            {
                assert(false);
                continue;
            }

#if COMPUTE_BAKER_ENABLE_MULTITHREADED_COMPILE
            m_threadPool.AddTask([this, variant, &progressCompilingShaders, &progressCounterCompleted, progressTotal]() {
#endif
                variant->CompileIfNeeded();
                int completed = progressCounterCompleted.fetch_add(1) + 1;
                progressCompilingShaders.Set(100 * completed / progressTotal);
#if COMPUTE_BAKER_ENABLE_MULTITHREADED_COMPILE
            });
#endif
        }

#if COMPUTE_BAKER_ENABLE_MULTITHREADED_COMPILE
        m_threadPool.WaitForTasks();
#endif

        // Check for errors and load shaders
        std::string firstError = "";

        for (ComputeShaderVariant* variant : m_parallelCompileListAll)
        {
            if (variant->m_compileError != "")
            {
                firstError = variant->m_compileError;
                break;
            }

            variant->LoadShaderAndCreatePipeline();
            
            if (variant->m_compileError != "")
            {
                firstError = variant->m_compileError;
                break;
            }
        }

        progressCompilingShaders.Set(100);
        m_parallelCompileListAll.clear();
        m_parallelCompileListUnique.clear();

        // All or nothing: callers chain variants over shared data, so a partial swap would silently mix old and new code.
        // On failure every variant keeps its previous pipeline until Retry recompiles it; Cancel closes the app.
        for (auto& variant : updateQueue)
        {
            if (firstError.empty())
            {
                variant->CommitPending();
                variant->m_localVersion = m_version;
            }
            else
                variant->DiscardPending();
            variant->m_lockedBaker = nullptr;
        }

        if (!firstError.empty())
        {
            firstError += m_compilerConfig.LongPathErrorHint();

            // Nobody to click Retry: log and take the Cancel path
            if (HelpersIsNonInteractive())
            {
                log::error("%s", firstError.c_str());
                HelpersExitProcess(1);
            }

#if _WIN32
            extern HWND HelpersGetActiveWindow();
            int result = MessageBoxA(HelpersGetActiveWindow(), firstError.c_str(),
                "Compute Shader Compile Error", MB_RETRYCANCEL | MB_ICONWARNING | MB_SETFOREGROUND | MB_TASKMODAL);
            if (result == IDCANCEL)
#endif
                HelpersExitProcess(1); // the renderer can't work with broken shaders

            // Retry: rescan, as the source may have been fixed while the error box was up
            if (std::optional<std::filesystem::file_time_type> rescan = DirectoryModifiedMonitor::ScanNow(m_sourceMonitor); rescan.has_value())
                m_lastUpdatedSourceTimestamp = m_cachedSourceTimestamp = rescan;
        }
        else
        {
            break;
        }

    } while (true);
}

//////////////////////////////////////////////////////////////////////////
// ComputeShaderVariant implementation
//////////////////////////////////////////////////////////////////////////

ComputeShaderVariant::ComputeShaderVariant(
    const std::string& shaderSourcePath,
    const std::string& entryPoint,
    const std::vector<ShaderMacro>& macros,
    const nvrhi::BindingLayoutVector& bindingLayouts,
    const std::string& debugName,
    const std::shared_ptr<ComputePipelineBaker>& baker)
    : m_baker(baker)
    , m_shaderSrcFileName(shaderSourcePath)
    , m_entryPoint(entryPoint)
    , m_macros(macros)
    , m_bindingLayouts(bindingLayouts)
    , m_debugName(debugName)
{
}

ComputeShaderVariant::~ComputeShaderVariant()
{
}

bool ComputeShaderVariant::NeedsUpdate() const
{
    auto baker = m_baker.lock();
    if (!baker)
        return false;
    return m_localVersion != baker->GetVersion() || m_pipeline == nullptr;
}

void ComputeShaderVariant::CommitPending()
{
    assert(m_pendingPipeline != nullptr);
    m_shader = m_pendingShader;
    m_pipeline = m_pendingPipeline;
    m_pendingShader = nullptr;
    m_pendingPipeline = nullptr;
}

void ComputeShaderVariant::DiscardPending()
{
    m_pendingShader = nullptr;
    m_pendingPipeline = nullptr;
}

bool ComputeShaderVariant::Dispatch(nvrhi::ICommandList* commandList, int width, int height, int depth, const nvrhi::BindingSetVector& bindings) const
{
    if (m_pipeline == nullptr)
        return false;

    nvrhi::ComputeState state;
    state.pipeline = m_pipeline;
    state.bindings = bindings;
    commandList->setComputeState(state);
    commandList->dispatch(width, height, depth);
    return true;
}

void ComputeShaderVariant::PrepareCompilation(std::filesystem::file_time_type lastModifiedSourceCode)
{
    auto baker = m_lockedBaker;
    assert(baker != nullptr);

    // Clear any previous error
    m_compileError = "";
    m_compileCmdLine = "";

    // Build full source path
    auto srcFullPath = std::filesystem::absolute(baker->GetShadersPath() / m_shaderSrcFileName);

    // Build DXC command using shared utilities
    ShaderCompilerUtils::DxcCommandOptions options;
    options.SourceFilePath = srcFullPath;
#if RTXPT_D3D_AGILITY_SDK_VERSION >= 619
    options.Profile = ShaderCompilerUtils::ShaderProfile::Compute_6_9;
#else
    options.Profile = ShaderCompilerUtils::ShaderProfile::Compute_6_6;
#endif
    options.EntryPoint = m_entryPoint;
    options.EnableDebugInfo = true;
    options.EmbedPdb = COMPUTE_BAKER_EMBED_PDBS != 0;
    options.UseOptimizations = COMPUTE_BAKER_USE_OPTIMIZATIONS != 0;
    options.Enable16BitTypes = true;
    options.WarningsAsErrors = true;
    options.AllResourcesBound = true;
    options.EnableDebugPrint = true;
    options.Macros = m_macros;

    auto cmdResult = ShaderCompilerUtils::BuildDxcCommand(baker->GetCompilerConfig(), options);

    // Check if hash changed
    std::string previousHashHex = m_compiledHashHex;
    if (cmdResult.HashHex != m_compiledHashHex)
    {
        m_compiledHashHex = cmdResult.HashHex;
    }

    m_compiledFileNameNoExt = m_shaderSrcFileName.stem().string() + "_" + m_entryPoint + "_" + m_compiledHashHex.substr(0, ShaderCompilerUtils::k_CacheFileHashHexChars);
    m_compiledFullPath = std::filesystem::absolute(
        baker->GetShaderBinariesPath() / m_compiledFileNameNoExt).string() + ".bin";
#if !COMPUTE_BAKER_EMBED_PDBS
    std::string compiledFullPathPdb = std::filesystem::absolute(
        baker->GetShaderBinariesPath() / m_compiledFileNameNoExt).string() + ".pdb";
#endif

    // Check if compiled file is up to date
    if (ShaderCompilerUtils::IsCompiledShaderUpToDate(m_compiledFullPath, lastModifiedSourceCode))
    {
        if (baker->IsVerbose())
            log::info("No need to compile compute shader '%s', up-to-date file exists", m_debugName.c_str());
        m_compileCmdLine = "";
    }
    else
    {
        // Need to recompile
        std::string command = baker->GetCompilerConfig().GetCompilerPathQuoted();
        command += cmdResult.CommandBase;
        
#if !COMPUTE_BAKER_EMBED_PDBS
        command += " /Fd \"" + compiledFullPathPdb + "\"";
#endif
        command += " -Fo \"" + m_compiledFullPath + "\"";

        if (baker->IsVerbose())
            log::info("Enqueuing compute shader '%s' for compilation...", m_debugName.c_str());
        
        m_compileCmdLine = command;
        m_sourceSnapshot = lastModifiedSourceCode;
    }
}

void ComputeShaderVariant::CompileIfNeeded()
{
    if (m_compileCmdLine.empty())
        return;

    log::info("Compiling compute shader '%s'...", m_debugName.c_str());

    // Ensure the output directory exists (shader paths may include subdirectories)
    std::filesystem::path outputDir = std::filesystem::path(m_compiledFullPath).parent_path();
    EnsureDirectoryExists(outputDir);

    auto [resNum, resString, resErrorString] = SystemShell(m_compileCmdLine, false);

    m_compileError = "";

    if (!resErrorString.empty())
    {
        m_compileError = StringFormat("ERROR compiling compute shader '%s':\nCommand: %s\nError: %s",
            m_debugName.c_str(), m_compileCmdLine.c_str(), resErrorString.c_str());
    }

    if (!m_compileError.empty())
        log::warning("%s", m_compileError.c_str());
    else
    {   // Stamped with the source snapshot, not the compile end time, so a source saved mid-compile still counts as newer
        std::error_code ec;
        std::filesystem::last_write_time(m_compiledFullPath, m_sourceSnapshot, ec);
    }
}

void ComputeShaderVariant::LoadShaderAndCreatePipeline()
{
    if (!m_compileError.empty())
        return;

    auto baker = m_lockedBaker;
    assert(baker != nullptr);

    // Load the compiled shader blob
    std::string blobPath = "/" + c_ComputeShaderBinariesRoot + "/" + m_compiledFileNameNoExt + ".bin";
    std::shared_ptr<donut::vfs::IBlob> data = baker->GetFS()->readFile(blobPath.c_str());

    if (!data)
    {
        m_compileError = StringFormat("ERROR loading compiled compute shader '%s' from '%s'",
            m_debugName.c_str(), blobPath.c_str());
        return;
    }

    // Create shader
    nvrhi::ShaderDesc shaderDesc;
    shaderDesc.shaderType = nvrhi::ShaderType::Compute;
    shaderDesc.debugName = m_debugName;
    shaderDesc.entryName = m_entryPoint.c_str();

    m_pendingShader = baker->GetDevice()->createShader(shaderDesc, data->data(), data->size());

    if (!m_pendingShader)
    {
        m_compileError = StringFormat("ERROR creating compute shader handle for '%s'", m_debugName.c_str());
        return;
    }

    // Create pipeline
    nvrhi::ComputePipelineDesc pipelineDesc;
    pipelineDesc.CS = m_pendingShader;
    pipelineDesc.bindingLayouts = m_bindingLayouts;

    m_pendingPipeline = baker->GetDevice()->createComputePipeline(pipelineDesc);

    if (!m_pendingPipeline)
    {
        m_compileError = StringFormat("ERROR creating compute pipeline for '%s'", m_debugName.c_str());
        m_pendingShader = nullptr;
        return;
    }

    log::info("Successfully created compute pipeline for '%s'", m_debugName.c_str());
}
