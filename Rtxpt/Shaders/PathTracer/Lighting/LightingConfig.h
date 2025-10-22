/*
* Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#ifndef __LIGHTING_CONFIG_H__
#define __LIGHTING_CONFIG_H__

#include "../Config.h"

// general settings
#define RTXPT_LIGHTING_MAX_LIGHTS                       (2 * 1024 * 1024 - 2)           // number of PolymorphicLightInfo (currently 48 bytes each); 
STATIC_ASSERT(RTXPT_LIGHTING_MAX_LIGHTS < ((1 << 23) - 2));                             // we can have 23 bits for the index (remaining 9 bits used for other things); minus 2 is for the index also has to have the special value RTXPT_INVALID_LIGHT_INDEX, and for it to always be unique even with local sampling tuples (which pack light index and counter)

#define RTXPT_LIGHTING_SAMPLING_PROXY_RATIO             16                              // every light can have this many proxies on average 
#define RTXPT_LIGHTING_MAX_SAMPLING_PROXIES             RTXPT_LIGHTING_SAMPLING_PROXY_RATIO * RTXPT_LIGHTING_MAX_LIGHTS    // total buffer size required for proxies, worst case scenario
#define RTXPT_LIGHTING_MAX_SAMPLING_PROXIES_PER_LIGHT   512*1024                        // one light can have no more than this many (global sampling) proxies (puts bounds on power-based importance sampling component
#define RTXPT_LIGHTING_MIN_WEIGHT_THRESHOLD             1e-8                            // ignore lights under this weight threshold


// tile (local) sampling settings
#if 1   // low
#define RTXPT_LIGHTING_FEEDBACK_CANDIDATES_PER_PATH     (1)     //< the max number of light candidates to store per path (pixel) 
#define RTXPT_LIGHTING_SAMPLING_BUFFER_TILE_SIZE        (8)     // 6x6 is good; 8x8 is acceptable and will reduce post-processing (P3 pass) cost over 6x6 by around 1.8x; the loss in quality is on small detail and shadows
#define RTXPT_LIGHTING_SAMPLING_BUFFER_WINDOW_SIZE      (8)    // has to be same as RTXPT_LIGHTING_SAMPLING_BUFFER_TILE_SIZE or n*2+RTXPT_LIGHTING_SAMPLING_BUFFER_TILE_SIZE
#define RTXPT_LIGHTING_LOCAL_PROXY_COUNT                (128)   // how big is each local sampler instance
#define RTXPT_LIGHTING_LOCAL_PROXY_BINARY_SEARCH_STEPS  (8)
#elif 1 // medium
#define RTXPT_LIGHTING_FEEDBACK_CANDIDATES_PER_PATH     (1)     //< the max number of light candidates to store per path (pixel) 
#define RTXPT_LIGHTING_SAMPLING_BUFFER_TILE_SIZE        (8)     // 6x6 is good; 8x8 is acceptable and will reduce post-processing (P3 pass) cost over 6x6 by around 1.8x; the loss in quality is on small detail and shadows
#define RTXPT_LIGHTING_SAMPLING_BUFFER_WINDOW_SIZE      (10)    // has to be same as RTXPT_LIGHTING_SAMPLING_BUFFER_TILE_SIZE or n*2+RTXPT_LIGHTING_SAMPLING_BUFFER_TILE_SIZE
#define RTXPT_LIGHTING_LOCAL_PROXY_COUNT                (256)   // how big is each local sampler instance
#define RTXPT_LIGHTING_LOCAL_PROXY_BINARY_SEARCH_STEPS  (9)
#else   // high
#define RTXPT_LIGHTING_FEEDBACK_CANDIDATES_PER_PATH     (2)     //< the max number of light candidates to store per path (pixel) 
#define RTXPT_LIGHTING_SAMPLING_BUFFER_TILE_SIZE        (6)     // 6x6 is good; 8x8 is acceptable and will reduce post-processing (P3 pass) cost over 6x6 by around 1.8x; the loss in quality is on small detail and shadows
#define RTXPT_LIGHTING_SAMPLING_BUFFER_WINDOW_SIZE      (10)    // has to be same as RTXPT_LIGHTING_SAMPLING_BUFFER_TILE_SIZE or n*2+RTXPT_LIGHTING_SAMPLING_BUFFER_TILE_SIZE
#define RTXPT_LIGHTING_LOCAL_PROXY_COUNT                (512)   // how big is each local sampler instance
#define RTXPT_LIGHTING_LOCAL_PROXY_BINARY_SEARCH_STEPS  (10)
#endif
#define RTXPT_LIGHTING_TOP_UP_SAMPLES                   (RTXPT_LIGHTING_LOCAL_PROXY_COUNT-RTXPT_LIGHTING_SAMPLING_BUFFER_WINDOW_SIZE*RTXPT_LIGHTING_SAMPLING_BUFFER_WINDOW_SIZE*RTXPT_LIGHTING_FEEDBACK_CANDIDATES_PER_PATH) // these go from the main sampling tile size window and strengthen the "core"

// must be power of two and max 512
STATIC_ASSERT(RTXPT_LIGHTING_LOCAL_PROXY_COUNT <= 512 && ((RTXPT_LIGHTING_LOCAL_PROXY_COUNT & (RTXPT_LIGHTING_LOCAL_PROXY_COUNT-1))==0));
STATIC_ASSERT(RTXPT_LIGHTING_TOP_UP_SAMPLES>=0);
STATIC_ASSERT( (1 << (RTXPT_LIGHTING_LOCAL_PROXY_BINARY_SEARCH_STEPS-1)) == RTXPT_LIGHTING_LOCAL_PROXY_COUNT ); 

// early feedback is a lag reduction approach using a pre-pass to handle disocclusion and help with just-appearing lights
#define RTXPT_NEEAT_EARLY_FEEDBACK_TILE_SIZE            (2)

// environment map quad tree settings 
// Note 1: Current setup is a complete overkill for normal cloudy sky with one sun and is tuned to work with 'shanghai sky' high res HDR envmap.
//         One can drop RTXPT_NEEAT_ENVMAP_QT_SUBDIVISIONS to 16 for a much saner subdivision - can actually improve noise.
// Note 2: One potential future upgrade is to reduce subdivisions when child nodes are all equally lit and of same colour - e.g. there's no point having many nodes within the sun, 
//         only enough to precisely capture the sun outline and interior. This could be done by computing subdivision weight as delta/difference to 4 sub-MIPs in full color).
#define RTXPT_NEEAT_ENVMAP_QT_BASE_RESOLUTION        4   // first pass starting point - must be square of 2
#define RTXPT_NEEAT_ENVMAP_QT_SUBDIVISIONS           24  // first pass subdivision
#define RTXPT_NEEAT_ENVMAP_QT_ADDITIONAL_NODES       (3*RTXPT_NEEAT_ENVMAP_QT_SUBDIVISIONS) // for each subdivision, one goes out, 4 get added - net is 3 new nodes
#define RTXPT_NEEAT_ENVMAP_QT_UNBOOSTED_NODE_COUNT   (RTXPT_NEEAT_ENVMAP_QT_BASE_RESOLUTION*RTXPT_NEEAT_ENVMAP_QT_BASE_RESOLUTION + RTXPT_NEEAT_ENVMAP_QT_ADDITIONAL_NODES)
#define RTXPT_NEEAT_ENVMAP_QT_BOOST_SUBDIVISION_DPT  3   // need to stop above subdivision earlier to allow for enough depth to fully subdivide x required times in boost pass
#define RTXPT_NEEAT_ENVMAP_QT_BOOST_SUBDIVISION      20  // how many times to subdivide in the boost pass
#define RTXPT_NEEAT_ENVMAP_QT_BOOST_NODES_MULT       (RTXPT_NEEAT_ENVMAP_QT_BOOST_SUBDIVISION*3+1)
STATIC_ASSERT((1u << (2 * RTXPT_NEEAT_ENVMAP_QT_BOOST_SUBDIVISION_DPT)) >= RTXPT_NEEAT_ENVMAP_QT_BOOST_NODES_MULT);
#define RTXPT_NEEAT_ENVMAP_QT_TOTAL_NODE_COUNT       (RTXPT_NEEAT_ENVMAP_QT_UNBOOSTED_NODE_COUNT * RTXPT_NEEAT_ENVMAP_QT_BOOST_NODES_MULT)

// Note: in theory can go higher, not fully tested and very likely unnecessary 
STATIC_ASSERT(RTXPT_NEEAT_ENVMAP_QT_TOTAL_NODE_COUNT <= 8192);

// This will not fully clear reservoirs but diminish existing content to 5% (with reprojection). This will help focus on stronger lights at the expense of less influential ones.
// Note: this doubles the amount of memory used for reservoirs.
#define RTXPT_LIGHTING_ENABLE_BSDF_FEEDBACK       0           //< provide NEE-AT feedback from BSDF rays hitting emissive surface/sky; helps primarily to speed up convergence / reduce lag

#define RTXPT_LIGHTING_MAX_TOTAL_SAMPLE_COUNT     63

#define RTXPT_LIGHTING_COUNT_ONLY_ONE_GLOBAL_FEEDBACK   1   //< when counting lights for global feedback, process only one of the RTXPT_LIGHTING_FEEDBACK_CANDIDATES_PER_PATH - avoids the InterlockedAdd bottleneck

#define RTXPT_LIGHTING_INDIRECT_FEEDBACK_RATIO      (1.0)   //< optionally leave a bit less reservoir space for indirect feedback but still take it seriously 

#if RTXPT_LIGHTING_FEEDBACK_CANDIDATES_PER_PATH == 1
#define NEEAT_FEEDBACK_CANDIDATE_TYPE uint
#elif RTXPT_LIGHTING_FEEDBACK_CANDIDATES_PER_PATH == 2
#define NEEAT_FEEDBACK_CANDIDATE_TYPE uint2
#elif RTXPT_LIGHTING_FEEDBACK_CANDIDATES_PER_PATH == 4
#define NEEAT_FEEDBACK_CANDIDATE_TYPE uint4
#else
#error not implemented
#endif

#define RTXPT_LIGHTING_LOCAL_SAMPLING_BUFFER_IS_3D_TEXTURE  0

// these are for integration, testing & performance; enabling any of these will partially disable MIS, resulting in more noise but still unibased results
//#define RTXPT_NEEAT_MIS_OVERRIDE_BSDF_PDF               0.5       //< use constant value BSDF for MIS only; will result in more noise but still be unbiased (used to test BSDF pdf correctness on both MIS ends)
//#define RTXPT_NEEAT_MIS_OVERRIDE_LIGHT_SOLID_ANGLE_PDF  50        //< use constant value light solid angle pdf for MIS only; will result in more noise but still be unbiased (used to test whether solid angle pdf correctness on both MIS ends)
//#define RTXPT_NEEAT_MIS_OVERRIDE_LOCAL_SAMPLER_PDF      0.05      //< !this one's tricky! use constant value for local light selection pdf, !!!!but we still have to return 0 if not in local sampler!!!! so this doesn't give us any perf
//#define RTXPT_NEEAT_MIS_OVERRIDE_GLOBAL_SAMPLER_PDF     0.01      //< use constant value for global light selection pdf

#endif // #define __LIGHTING_CONFIG_H__