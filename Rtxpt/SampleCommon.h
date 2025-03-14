/*
* Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#ifndef __COMMON_H__ // using instead of "#pragma once" due to https://github.com/microsoft/DirectXShaderCompiler/issues/3943
#define __COMMON_H__

#include "PathTracer/Config.h"

#include <utility>

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

#endif // __COMMON_H__
