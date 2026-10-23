/*
* Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#ifndef __NEEAT_CONFIG_HLSL__
#define __NEEAT_CONFIG_HLSL__

// general settings


// number of light supported by NEE-AT - it is advisable to rely on some kind of LODding/aggregation to keep under 500k, but up to 2 million is supported
#define NEEAT_MAX_LIGHTS                                (2 * 1024 * 1024)

#define NEEAT_ENABLE_ENVMAP_PROCESSING                  1
//#define NEEAT_ENABLE_EMISSIVE_TRIANGLE_PROCESSING       1 not yet implemented
//#define NEEAT_ENABLE_ANALYTIC_LIGHT_PROCESSING          1 not yet implemented

#ifndef NEEAT_QUALITY_PRESET                             // runtime-compiled shaders get it from LightsBaker::GetQualityPreset
#define NEEAT_QUALITY_PRESET                            0
#endif


#define RTXPT_LIGHTING_OPTIMAL_PROXIES_PER_LIGHT        16                              // proxy budget per light until the cap below is hit; one per light is reserved for ceil() rounding
#define RTXPT_LIGHTING_MAX_SAMPLING_PROXIES             (3 * 1024 * 1024)               // proxy buffer size and cap on the total budget (hit at ~256K lights)
#define RTXPT_LIGHTING_MAX_SAMPLING_PROXIES_PER_LIGHT   256*1024                        // one light can have no more than this many (global sampling) proxies (puts bounds on power-based importance sampling component) - this limit is purely due to performance of the compute shader
#define RTXPT_LIGHTING_MIN_WEIGHT_THRESHOLD             1e-8                            // ignore lights under this weight threshold


// tile (local) sampling settings
#if NEEAT_QUALITY_PRESET == 0   // low
#define RTXPT_LIGHTING_SAMPLING_BUFFER_TILE_SIZE        (8)     // 6x6 is good; 8x8 is acceptable and will reduce post-processing (P3 pass) cost over 6x6 by around 1.8x; the loss in quality is on small detail and shadows
#define RTXPT_LIGHTING_SAMPLING_BUFFER_WINDOW_SIZE      (8)    // has to be same as RTXPT_LIGHTING_SAMPLING_BUFFER_TILE_SIZE or n*2+RTXPT_LIGHTING_SAMPLING_BUFFER_TILE_SIZE
#define RTXPT_LIGHTING_LOCAL_PROXY_COUNT                (128)   // how big is each local sampler instance
#define RTXPT_LIGHTING_LOCAL_PROXY_BINARY_SEARCH_STEPS  (8)
#elif NEEAT_QUALITY_PRESET == 1 // high
#define RTXPT_LIGHTING_SAMPLING_BUFFER_TILE_SIZE        (8)     // 6x6 is good; 8x8 is acceptable and will reduce post-processing (P3 pass) cost over 6x6 by around 1.8x; the loss in quality is on small detail and shadows
#define RTXPT_LIGHTING_SAMPLING_BUFFER_WINDOW_SIZE      (10)    // has to be same as RTXPT_LIGHTING_SAMPLING_BUFFER_TILE_SIZE or n*2+RTXPT_LIGHTING_SAMPLING_BUFFER_TILE_SIZE
#define RTXPT_LIGHTING_LOCAL_PROXY_COUNT                (256)   // how big is each local sampler instance
#define RTXPT_LIGHTING_LOCAL_PROXY_BINARY_SEARCH_STEPS  (9)
#elif NEEAT_QUALITY_PRESET == 2 // ultra
#define RTXPT_LIGHTING_SAMPLING_BUFFER_TILE_SIZE        (6)     // 6x6 is good; 8x8 is acceptable and will reduce post-processing (P3 pass) cost over 6x6 by around 1.8x; the loss in quality is on small detail and shadows
#define RTXPT_LIGHTING_SAMPLING_BUFFER_WINDOW_SIZE      (10)    // has to be same as RTXPT_LIGHTING_SAMPLING_BUFFER_TILE_SIZE or n*2+RTXPT_LIGHTING_SAMPLING_BUFFER_TILE_SIZE
#define RTXPT_LIGHTING_LOCAL_PROXY_COUNT                (512)   // how big is each local sampler instance
#define RTXPT_LIGHTING_LOCAL_PROXY_BINARY_SEARCH_STEPS  (10)
#else
#error NEEAT_QUALITY_PRESET invalid value
#endif
#define RTXPT_LIGHTING_TOP_UP_SAMPLES                   (RTXPT_LIGHTING_LOCAL_PROXY_COUNT-RTXPT_LIGHTING_SAMPLING_BUFFER_WINDOW_SIZE*RTXPT_LIGHTING_SAMPLING_BUFFER_WINDOW_SIZE) // these go from the main sampling tile size window and strengthen the "core"

// early feedback is a lag reduction approach using a pre-pass to handle disocclusion and help with just-appearing lights
#define RTXPT_NEEAT_EARLY_FEEDBACK_TILE_SIZE            (2)

// Provide NEE-AT feedback from BSDF rays hitting emissive surface/sky; helps primarily to speed up convergence / reduce lag but can increase converged noise
// Note: this is still last frame's data. However, this could be harvested in advance with a small, simple, 1 bounce only pre-pass (that also does NEE), done only on disocclusions.
#ifndef RTXPT_LIGHTING_ENABLE_BSDF_FEEDBACK
#define RTXPT_LIGHTING_ENABLE_BSDF_FEEDBACK             (1)
#endif

#if NEEAT_ENABLE_ENVMAP_PROCESSING
// environment map quad tree settings 
// Note 1: Current setup is a complete overkill for normal cloudy sky with one sun and is tuned to work with 'shanghai sky' high res HDR envmap.
//         One can drop RTXPT_NEEAT_ENVMAP_QT_SUBDIVISIONS to 16 for a much saner subdivision - can actually improve noise.
// Note 2: One potential future upgrade is to reduce subdivisions when child nodes are all equally lit and of same colour - e.g. there's no point having many nodes within the sun, 
//         only enough to precisely capture the sun outline and interior. This could be done by computing subdivision weight as delta/difference to 4 sub-MIPs in full color).
#define RTXPT_NEEAT_ENVMAP_QT_BASE_RESOLUTION        4   // first pass starting point - must be square of 2
#define RTXPT_NEEAT_ENVMAP_QT_SUBDIVISIONS           20  // first pass subdivision
#define RTXPT_NEEAT_ENVMAP_QT_ADDITIONAL_NODES       (3*RTXPT_NEEAT_ENVMAP_QT_SUBDIVISIONS) // for each subdivision, one goes out, 4 get added - net is 3 new nodes
#define RTXPT_NEEAT_ENVMAP_QT_UNBOOSTED_NODE_COUNT   (RTXPT_NEEAT_ENVMAP_QT_BASE_RESOLUTION*RTXPT_NEEAT_ENVMAP_QT_BASE_RESOLUTION + RTXPT_NEEAT_ENVMAP_QT_ADDITIONAL_NODES)
#define RTXPT_NEEAT_ENVMAP_QT_BOOST_SUBDIVISION_DPT  3   // need to stop above subdivision earlier to allow for enough depth to fully subdivide x required times in boost pass
#define RTXPT_NEEAT_ENVMAP_QT_BOOST_SUBDIVISION      12  // how many times to subdivide in the boost pass   
#define RTXPT_NEEAT_ENVMAP_QT_BOOST_NODES_MULT       (RTXPT_NEEAT_ENVMAP_QT_BOOST_SUBDIVISION*3+1)
#define RTXPT_NEEAT_ENVMAP_QT_TOTAL_NODE_COUNT       (RTXPT_NEEAT_ENVMAP_QT_UNBOOSTED_NODE_COUNT * RTXPT_NEEAT_ENVMAP_QT_BOOST_NODES_MULT)
#endif


#define RTXPT_LIGHTING_SCREEN_SPACE_COHERENT_FEEDBACK_BIAS      (1.00)  //< optionally leave more screen space samples
#define RTXPT_LIGHTING_DAMPEN_GLOBAL_FEEDBACK_EXP               (0.65)  //< it's not that important for lights to have high weight if they're already globally very likely to be picked (allows room for more locally important lights only)
#define RTXPT_LIGHTING_BOOST_FEEDBACK_EXP                       (0.40)  //< exponent applied to per-light boost ratio when pushing extra feedback for new/changed lights


#ifdef STATIC_ASSERT

// we can have 23 bits for the index (remaining 9 bits used for other things); minus 2 is for the index also has to have the special value NEEAT_INVALID_LIGHT_INDEX, and for it to always be unique even with local sampling tuples (which pack light index and counter)
STATIC_ASSERT(NEEAT_MAX_LIGHTS < ((1 << 23) - 2));
STATIC_ASSERT(NEEAT_MAX_LIGHTS <= 2 * 1024 * 1024);  // there are other places where 2 million is the max

STATIC_ASSERT(RTXPT_LIGHTING_MAX_SAMPLING_PROXIES >= 1.5 * NEEAT_MAX_LIGHTS);  // ceil() gives every lit light >= 1 proxy; below 1.5x-2x there's not much room left for power-proportional allocation (although light clustering avoids this in most cases)

// float random based sampling loses float32 exactness above 2^24
STATIC_ASSERT(RTXPT_LIGHTING_MAX_SAMPLING_PROXIES <= (1 << 24));

// must be power of two and max 512
STATIC_ASSERT(RTXPT_LIGHTING_LOCAL_PROXY_COUNT <= 512 && ((RTXPT_LIGHTING_LOCAL_PROXY_COUNT & (RTXPT_LIGHTING_LOCAL_PROXY_COUNT - 1)) == 0));
STATIC_ASSERT(RTXPT_LIGHTING_TOP_UP_SAMPLES >= 0);
STATIC_ASSERT((1 << (RTXPT_LIGHTING_LOCAL_PROXY_BINARY_SEARCH_STEPS - 1)) == RTXPT_LIGHTING_LOCAL_PROXY_COUNT);
//
#if NEEAT_ENABLE_ENVMAP_PROCESSING
STATIC_ASSERT((1u << (2 * RTXPT_NEEAT_ENVMAP_QT_BOOST_SUBDIVISION_DPT)) >= RTXPT_NEEAT_ENVMAP_QT_BOOST_NODES_MULT);
// Note: in theory can go higher, not fully tested and very likely unnecessary 
STATIC_ASSERT(RTXPT_NEEAT_ENVMAP_QT_TOTAL_NODE_COUNT <= 8192);
STATIC_ASSERT(RTXPT_NEEAT_ENVMAP_QT_TOTAL_NODE_COUNT < NEEAT_MAX_LIGHTS);
#endif
#endif // #ifdef STATIC_ASSERT


#endif // __NEEAT_CONFIG_HLSL__