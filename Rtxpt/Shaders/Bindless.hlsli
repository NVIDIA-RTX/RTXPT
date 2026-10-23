/*
* Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#ifndef __BINDLESS_HLSLI__    // using instead of "#pragma once" due to https://github.com/microsoft/DirectXShaderCompiler/issues/3943
#define __BINDLESS_HLSLI__

#include "Libraries/ShaderDebug/ShaderDebug.hlsl"     // for RTXPT_SHADER_DEBUG, BREAKPOINT() and u_ShaderDebugBuffer

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////// WORKING NOTES, INFO, ETC ////////////////////////////////////////////////////////////////////
// Bindless requires SM6.6 and relies on ResourceDescriptorHeap
// https://rtarun9.github.io/blogs/bindless_rendering/#using-sm66s-resourcedescriptorheap--samplerdescriptorheap
// 
// Works on DX12 and Vulkan. On Vulkan it makes VK_EXT_mutable_descriptor_type a REQUIRED device extension
// (Donut promotes it), so --vk refuses to start on hardware without it. Also note that on Vulkan
// ResourceDescriptorHeap maps to the mutable descriptor set, which is NOT the set m_DescriptorTable owns -
// that is why bindless descriptors live in their own table (below) rather than the scene one.
// 
// We've got new m_bindlessHeapDescriptorTable that needs to be bound when using bindless. It must be the LAST
// entry in the pipeline's layout list - nvrhi pairs binding set i with layout i by position, so callers that
// do not use bindless can keep passing a shorter list only because the extra layout is trailing. On Vulkan it
// must also be the THIRD entry (set 2): both bakers compile with -fvk-bind-resource-heap 0 2.
// 
// DXC SPIR-V bug: assigning a fetched RWStructuredBuffer straight to a struct member fails with "cannot handle
// associated counter variable assignment". Fetch into a local first, then assign the local - see SharcBindings.hlsli.
// 
// There's built-in validation, active in Debug builds. It does not crash or break - it reports. A failure
// writes a sentinel that ShaderDebug reads back, and you get:
//     "Shader BREAKPOINT() was hit - most likely a bindless descriptor index that does not match the
//      fetched type..."
// in the log (rate-limited to 8). If you want it to stop on the GPU instead, set RTXPT_BREAKPOINT_HANG to 1
// in Shaders/Libraries/ShaderDebug/ShaderDebug.hlsl - that hangs the offending thread so a GPU debugger can catch it.
// 
// What it catches: invalid/unallocated indices (including 0xFFFFFFFF from a -1), SRV vs UAV mixups,
// texture vs buffer vs structured vs raw mixups, and - for structured buffers - a wrong element stride.
// What it does NOT catch: StructuredBuffer<A> vs StructuredBuffer<B> where sizeof(A)==sizeof(B). Their
// descriptors are genuinely identical, nothing observable separates them.
// 
// We've got new m_bindlessValidationBuffer that needs to be bound when using bindless (t50, added
// unconditionally so the binding layout is the same in Debug and Release).
// 
// Validation only works with ComputePipelineBaker and PTPipelineBaker - they inject RTXPT_SHADER_DEBUG, and
// shaders precompiled offline by ShaderMake (shaders.cfg) never get it and always take the disabled path.
// Requires ShaderDebug to be bound/working.
// 
// Main header to include is Bindless.hlsli
// 
// If your types are already registered at the bottom of that file - you're good. If not, add there if generic
// type but add locally if custom type (see Bindings/SharcBindings.hlsli). Registration is keyed on the TYPE
// and is program-wide, so a type may only be registered ONCE anywhere; a second one is a hard compile error
// ("redefinition of 'GetBindlessUniform'").
// 
//     RTXPT_BINDLESS_REGISTER_TYPE( Texture2D, RTXPT_BINDLESS_VIEW_TEXTURE, false )
//     RTXPT_BINDLESS_REGISTER_STRUCTURED_TYPE( RWStructuredBuffer<MyStruct>, true, 16 )   // 16 == sizeof(MyStruct)
// 
// Use the STRUCTURED variant for anything StructuredBuffer/RWStructuredBuffer - the extra argument is the
// element size, which is what the stride check compares against. HLSL has no sizeof so it is spelled out; it
// is self-validating, since getting it wrong trips the trap on the first correct fetch.
// 
// PREFER RWByteAddressBuffer for new buffers. It is untyped, already registered, supports templated
// Load<T>()/Store<T>() of arbitrary structs plus 32- and 64-bit atomics, so a new buffer of any layout costs
// no new registration at all.
// 
// To use in shader, do
// 
//     #include "../Bindless.hlsli"        // path relative to your file
// 
//     RWStructuredBuffer<MyStruct> myBuffer = GetBindlessUniform< RWStructuredBuffer<MyStruct> >( g_Const.MyBufferIndex );
//     myBuffer[i] = ...;
// 
// Use GetBindlessUniform when the index is the same across the wave - anything from a constant buffer is.
// Use GetBindlessNonUniform when it is not (per-material/per-instance index in a ray tracing shader, where
// threads in a wave shade different surfaces). Picking the wrong one is NOT a compile error: the uniform
// version with a divergent index quietly reads the wrong descriptor for most of the wave.
// 
// On the C++ side:
// 
//     m_handle = bindlessHeapTable->CreateDescriptorHandle( nvrhi::BindingSetItem::StructuredBuffer_UAV(0, buffer) );
//     constants.MyBufferIndex = m_handle.GetIndexInHeap();
// 
// GetIndexInHeap(), NOT Get(). And re-read it EVERY FRAME - growing the descriptor table relocates it and
// invalidates every index handed out before that point, and the table grows whenever a scene loads more
// textures. Keep the DescriptorHandle alive as long as the shader might use the index; it releases on destruct.
// 
// TWO LANDMINES, both cost a day each:
// 
// 1. NEVER zero-initialise a struct that holds a bindlessly fetched resource:
// 
//        MyParams p = (MyParams)0;        // <-- do not
//        p.buffer = GetBindlessUniform<...>( idx );
// 
//    That makes DXC emit createHandleForLib(createHandleFromHeap(...)), and the NVIDIA driver
//    ACCESS-VIOLATES on that DXIL shape while creating a DXR state object - a hard crash at pipeline
//    creation with no D3D12 error first. Assign every field explicitly instead. Compute is unaffected;
//    this is DXR only. See GetSharcParameters in Bindings/SharcBindings.hlsli for the worked example.
// 
// 2. Do not compile these shaders with -Od. A bindless resource stored in a struct that crosses a function
//    boundary only survives because the optimizer scalarises the struct away; at -Od DXC dies with an
//    internal compiler error. -O1 is enough, and the bakers use -O3.
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


// ===================================================================================================================
// HOW TO USE
// ===================================================================================================================
//
// The engine keeps a single descriptor table (donut::engine::DescriptorTableManager, created in Sample::Init) holding
// every bindless descriptor - scene textures, geometry buffers, and anything else we register. A shader reaches into
// it with a plain integer index, usually carried in a constant buffer, instead of having a named slot reserved for it
// in the global binding layout.
//
// This is built on Shader Model 6.6 dynamic resources (ResourceDescriptorHeap), so there are no descriptor arrays to
// declare, no register spaces to allocate and nothing in the binding layout to keep in sync. The only per-type cost
// is one RTXPT_BINDLESS_REGISTER_TYPE line, and that exists purely so validation knows what the type should be.
//
// -------------------------------------------------------------------------------------------------------------------
// 1. Fetch a resource
// -------------------------------------------------------------------------------------------------------------------
//
//      RWStructuredBuffer<uint64_t> hashEntries = GetBindlessUniform< RWStructuredBuffer<uint64_t> >( index );
//
//   Use GetBindlessUniform when the index is the same across the wave - anything coming out of a constant buffer is.
//   Use GetBindlessNonUniform when it is not: a per-material or per-instance index in a ray tracing shader, where
//   threads in a wave can be shading different surfaces. Picking the wrong one is NOT a compile error - the uniform
//   version with a divergent index quietly reads the wrong descriptor for most of the wave.
//
// -------------------------------------------------------------------------------------------------------------------
// 2. Register the type, once, in the "Registered types" section at the bottom of THIS file
// -------------------------------------------------------------------------------------------------------------------
//
//      RTXPT_BINDLESS_REGISTER_TYPE( Texture2D, RTXPT_BINDLESS_VIEW_TEXTURE, false )
//      RTXPT_BINDLESS_REGISTER_STRUCTURED_TYPE( RWStructuredBuffer<uint64_t>, true, 8 )   // structured: + stride check
//
//   Registration is keyed on the TYPE and is program-wide: two subsystems that both want RWStructuredBuffer<uint4>
//   share one registration and are distinguished only by their descriptor index. Registering the same type twice is
//   a hard error ("redefinition of 'GetBindlessUniform'"), which is why these all live in one place.
//
//   PREFER RWByteAddressBuffer FOR NEW BUFFER RESOURCES. It is untyped, supports templated Load<T>() / Store<T>() of
//   arbitrary structs plus 32- and 64-bit atomics, and is already registered below - so a new buffer of any layout
//   costs no new registration at all. The typed RWStructuredBuffer entries below exist only because the SHaRC SDK
//   declares its parameter struct in terms of them and takes lvalue references into them for atomics, which a
//   ByteAddressBuffer cannot provide.
//
// -------------------------------------------------------------------------------------------------------------------
// 3. What the C++ side has to do
// -------------------------------------------------------------------------------------------------------------------
//
//   a) Enable dynamic resources at device creation (SampleBaseApp::GetDefaultDeviceParams):
//          deviceParams.enableHeapDirectlyIndexed = true;
//      NVRHI re-checks Resource Binding Tier 3 and SM 6.6 at runtime and SILENTLY leaves the feature off if either
//      is missing, which would leave shaders reading from a heap the root signature never exposed. Verify with
//      GetDevice()->queryFeatureSupport(nvrhi::Feature::HeapDirectlyIndexed) and fail loudly.
//
//   b) Include a bindless layout with layoutType = MutableSrvUavCbv as the LAST entry in the layout list of every
//      pipeline that fetches this way. It consumes no root parameter - it only sets
//      D3D12_ROOT_SIGNATURE_FLAG_CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED. Pass the descriptor table built on that layout
//      in the matching slot. Do NOT pass nullptr: the backends would tolerate it, but nvrhi's validation layer
//      (on in Debug) rejects both a null binding set and a set/layout count mismatch on compute and graphics
//      dispatches. Ray tracing is not count-validated, which is the only reason callers that do not use this at
//      all can keep passing a shorter list - and only because the layout is trailing.
//
//   c) Allocate a descriptor and hand the index to the shader:
//          m_handle = bindlessHeapTable->CreateDescriptorHandle( nvrhi::BindingSetItem::StructuredBuffer_UAV(0, buffer) );
//          constants.MyBufferIndex = m_handle.GetIndexInHeap();
//
//      GetIndexInHeap(), NOT Get(). Allocate from the table built on the MUTABLE layout, not the scene descriptor
//      table: on Vulkan ResourceDescriptorHeap maps to the mutable descriptor set, and getFirstDescriptorIndexInHeap()
//      returns 0 there, so an index taken from any other table points into a set nothing ever wrote.
//
//   d) RE-READ GetIndexInHeap() EVERY FRAME. Growing the descriptor table reallocates it and moves its base, which
//      invalidates every heap index handed out before that point - and the table grows whenever a scene loads more
//      textures, i.e. long after SHaRC allocated its descriptors. Recomputing it during the constants update is a
//      single add; caching it is a silent corruption waiting for the next scene load.
//
//   e) Keep the DescriptorHandle alive for as long as the shader might use the index - it releases on destruction.
//
// -------------------------------------------------------------------------------------------------------------------
// GOTCHAS
// -------------------------------------------------------------------------------------------------------------------
//
//   * Fetching a type with no RTXPT_BINDLESS_REGISTER_TYPE is a compile error, not a silent fallback, because the
//     primary templates below are declared and never defined. The message is ugly (an undefined/exported-function
//     complaint naming the mangled template) but the build does fail.
//
//   * On Vulkan this requires VK_EXT_mutable_descriptor_type, which Donut promotes to a REQUIRED extension as soon
//     as enableHeapDirectlyIndexed is set - a device without it is rejected during selection rather than falling
//     back. Both runtime bakers pass -fvk-bind-resource-heap 0 2, which is what pins the heap layout to set 2.
//
//   * A macro argument cannot contain a top-level comma, so a type like RWStructuredBuffer<Some<a,b>> needs a
//     typedef first. Angle brackets on their own are fine.
//
//   * NEVER ZERO-INITIALISE A STRUCT THAT HOLDS A BINDLESSLY FETCHED RESOURCE. Writing `MyParams p = (MyParams)0;`
//     and then assigning handles into it makes DXC emit
//         dx.op.createHandleForLib( dx.op.createHandleFromHeap(...) )
//     and the NVIDIA driver ACCESS-VIOLATES on that DXIL shape while creating a DXR state object. There is no
//     D3D12 error first - the pipeline simply never gets created and the process dies inside nvwgf2umx.dll.
//     Assign every field explicitly instead. Established by bisecting DXIL against a standalone CreateStateObject
//     harness: every blob containing that shape faults, every blob where the heap handle feeds annotateHandle
//     directly creates fine. Plain ResourceDescriptorHeap use in a ray tracing library is NOT the problem, and
//     compute pipelines are unaffected. Worked example: GetSharcParameters in Bindings/SharcBindings.hlsli.
//
//   * DO NOT COMPILE SHADERS THAT DO THIS WITH -Od. Storing a bindlessly fetched resource in a struct that then
//     crosses a function boundary - which is exactly what the SHaRC SDK's SharcParameters does - only survives
//     because the optimizer scalarises the struct away. At -Od the handles genuinely live in memory, which DXIL
//     cannot represent, and DXC 1.9 dies with an internal compiler error ('access violation') or
//     'llvm::cast<X>() argument of incompatible type!'. -O1 is enough; the bakers use -O3.
//     This is not specific to ResourceDescriptorHeap - unbounded descriptor arrays fail identically.
//
// ===================================================================================================================

// ===================================================================================================================
// Validation
// ===================================================================================================================
//
// D3D12-only by design: it is a debugging aid, and Vulkan would need the metadata plumbing duplicated for no extra
// coverage. It matters more here than it would with declared descriptor arrays - on D3D12 nothing else diagnoses a
// wrong index. It reads a real descriptor belonging to something else and reinterprets it.
//
#ifndef RTXPT_BINDLESS_VALIDATION
#if RTXPT_SHADER_DEBUG && !defined(SPIRV) && !defined(TARGET_VULKAN)
#define RTXPT_BINDLESS_VALIDATION 1
#else
#define RTXPT_BINDLESS_VALIDATION 0
#endif
#endif

// View types - the C++ side must produce exactly these values when translating nvrhi::ResourceType.
#define RTXPT_BINDLESS_VIEW_NONE                    (0)
#define RTXPT_BINDLESS_VIEW_TEXTURE                 (1)
#define RTXPT_BINDLESS_VIEW_TYPED_BUFFER            (2)
#define RTXPT_BINDLESS_VIEW_STRUCTURED_BUFFER       (3)
#define RTXPT_BINDLESS_VIEW_RAW_BUFFER              (4)

// One uint per descriptor, indexed by the SAME heap index the shader passes to GetBindless* - so the C++ side writes
// entry [firstDescriptorInHeap + i] when describing table-relative slot i. Built by walking
// DescriptorTableManager::GetDescriptor(i) across the table capacity and translating the stored
// nvrhi::BindingSetItem::type, which means descriptors created by the engine itself (TextureCache, Scene,
// OmmBuildQueue) end up described correctly too, not just the ones we register.
//
// An index past the end of the buffer needs no explicit range check: D3D12 returns 0 for an out-of-bounds
// StructuredBuffer read, which reads back as "not allocated" and trips the trap - exactly the desired behaviour.
#define RTXPT_BINDLESS_META_ALLOCATED_BIT           (1u << 0)
#define RTXPT_BINDLESS_META_UAV_BIT                 (1u << 1)
#define RTXPT_BINDLESS_META_VIEW_SHIFT              (8)
#define RTXPT_BINDLESS_META_VIEW_MASK               (0xFFu)
// Structured-buffer element stride, 0 for everything else. Taken from the BUFFER DESC on the CPU, which is why it
// can catch a wrong index: it describes the descriptor actually sitting in that slot, not the type the shader
// declared. (GetDimensions cannot do this - DXC folds the stride from the static type annotation, so comparing it
// against a literal optimises away to nothing.)
#define RTXPT_BINDLESS_META_STRIDE_SHIFT            (16)
#define RTXPT_BINDLESS_META_STRIDE_MASK             (0xFFFFu)

#define RTXPT_BINDLESS_VALIDATION_SRV_INDEX         (50)    // make sure it matches t_BindlessValidationData below

#if RTXPT_BINDLESS_VALIDATION

// see SampleNull.hlsl for explanation of below
#ifdef register
#undef register
#define RTXPT_BINDLESS_REGISTER_WAS_DEFINED
#endif
StructuredBuffer<uint>  t_BindlessValidationData    : register(t50);
#ifdef RTXPT_BINDLESS_REGISTER_WAS_DEFINED
#define register
#undef RTXPT_BINDLESS_REGISTER_WAS_DEFINED
#endif

#endif // RTXPT_BINDLESS_VALIDATION

// Traps if the descriptor at 'index' is not the kind of view the caller is about to interpret it as. Compiles away
// entirely when validation is off.
//
// Deliberately validates the INDEX and returns nothing, rather than taking the resource and handing it back. The
// wrapper form - T Validate(T resource, ...) - is a fatal SPIR-V error for an RWStructuredBuffer fetched out of a
// descriptor array ("cannot find the associated counter variable") while compiling perfectly well on DXIL, so the
// nicer-looking version would have gone unnoticed until somebody built for Vulkan.
// expectedStride is the structured-buffer element size, or 0 for types where it does not apply (textures, raw
// buffers). It is what distinguishes two buffers that share a view type - swapping one StructuredBuffer_UAV index
// for another is otherwise invisible, since their metadata is identical.
void ValidateBindlessIndex( uint index, uint expectedViewType, bool expectedUAV, uint expectedStride )
{
#if RTXPT_BINDLESS_VALIDATION
    const uint metadata     = t_BindlessValidationData[index];
    const bool allocated    = (metadata & RTXPT_BINDLESS_META_ALLOCATED_BIT) != 0;
    const bool isUAV        = (metadata & RTXPT_BINDLESS_META_UAV_BIT) != 0;
    const uint viewType     = (metadata >> RTXPT_BINDLESS_META_VIEW_SHIFT) & RTXPT_BINDLESS_META_VIEW_MASK;
    const uint stride       = (metadata >> RTXPT_BINDLESS_META_STRIDE_SHIFT) & RTXPT_BINDLESS_META_STRIDE_MASK;

    if ( !allocated || isUAV != expectedUAV || viewType != expectedViewType || stride != expectedStride )
        BREAKPOINT();
#endif
}

// ===================================================================================================================
// Accessors
// ===================================================================================================================
//
// Declared but never defined - the specializations emitted by RTXPT_BINDLESS_REGISTER_TYPE are the only
// implementations, so fetching an unregistered type fails to compile instead of silently skipping validation.
//
template<typename T> T GetBindlessUniform( uint index );
template<typename T> T GetBindlessNonUniform( uint index );

#define RTXPT_BINDLESS_REGISTER_TYPE(TYPE, VIEW_TYPE, IS_UAV)                                   \
    template<> TYPE GetBindlessUniform<TYPE>( uint index )                                      \
    {                                                                                           \
        ValidateBindlessIndex( index, VIEW_TYPE, IS_UAV, 0 );                                   \
        return ResourceDescriptorHeap[index];                                                   \
    }                                                                                           \
    template<> TYPE GetBindlessNonUniform<TYPE>( uint index )                                   \
    {                                                                                           \
        ValidateBindlessIndex( index, VIEW_TYPE, IS_UAV, 0 );                                   \
        return ResourceDescriptorHeap[NonUniformResourceIndex(index)];                          \
    }

// Same, plus the element-stride check - use this for every StructuredBuffer / RWStructuredBuffer.
#define RTXPT_BINDLESS_REGISTER_STRUCTURED_TYPE(TYPE, IS_UAV, EXPECTED_STRIDE)                  \
    template<> TYPE GetBindlessUniform<TYPE>( uint index )                                      \
    {                                                                                           \
        ValidateBindlessIndex( index, RTXPT_BINDLESS_VIEW_STRUCTURED_BUFFER, IS_UAV, EXPECTED_STRIDE ); \
        return ResourceDescriptorHeap[index];                                                   \
    }                                                                                           \
    template<> TYPE GetBindlessNonUniform<TYPE>( uint index )                                   \
    {                                                                                           \
        ValidateBindlessIndex( index, RTXPT_BINDLESS_VIEW_STRUCTURED_BUFFER, IS_UAV, EXPECTED_STRIDE ); \
        return ResourceDescriptorHeap[NonUniformResourceIndex(index)];                          \
    }

// ===================================================================================================================
// Registered types
// ===================================================================================================================

// General purpose - add any additional general purpose (built-in types) here
RTXPT_BINDLESS_REGISTER_TYPE( RWByteAddressBuffer,                       RTXPT_BINDLESS_VIEW_RAW_BUFFER,        true  )
RTXPT_BINDLESS_REGISTER_TYPE( ByteAddressBuffer,                         RTXPT_BINDLESS_VIEW_RAW_BUFFER,        false )
RTXPT_BINDLESS_REGISTER_TYPE( Texture2D,                                 RTXPT_BINDLESS_VIEW_TEXTURE,           false )
RTXPT_BINDLESS_REGISTER_STRUCTURED_TYPE( RWStructuredBuffer<uint64_t>, true, 8 )

#endif // __BINDLESS_HLSLI__
