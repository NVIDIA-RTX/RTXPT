# RTX Path Tracing — Intro Sample

![Intro Sample — ReSTIR DI, unified indirect lighting and SHaRC](https://github.com/NVIDIA-RTX/RTXPT-Assets/raw/79edd32e02797d3624f56a153a768937e0432198/Docs/intro/step-8.jpg)

The Intro Sample is an introduction to real-time path tracing, organized as a sequence of steps that
build from a deferred baseline to resampled direct lighting and cached indirect lighting.

The progression begins with a conventional deferred renderer: a G-buffer pass, image-based lighting
(IBL), screen-space ambient occlusion and screen-space reflections, then replaces each
component in turn until the result is a real-time path tracer with resampled direct lighting, a
world-space radiance cache and multi-bounce indirect illumination.

Steps are selected from a dropdown list in the UI and can be changed at runtime, so any two can be compared directly
in the same scene from the same camera position.

The implementation favours clarity over performance and image quality: passes are kept separate and
readable rather than fused and optimized, and several steps use approximations that are documented
where they occur.

The [Advanced Sample](../README.md) in this repository is a separate, considerably more capable
renderer. It is not a tuned version of the progression described here: it is a state-of-the-art path
tracer that solves the same problems with different and more advanced techniques, covers a much wider
set of features, and is correspondingly more complex.

---

## Contents

- [Audience and prerequisites](#audience-and-prerequisites)
- [Running the sample](#running-the-sample)
- [The progression](#the-progression)
- [The steps](#the-steps)
- [Architecture](#architecture)
- [Reference mode](#reference-mode)
- [Denoising and upscaling](#denoising-and-upscaling)
- [Suggested experiments](#suggested-experiments)
- [References](#references)

---

## Audience and prerequisites

The sample is aimed at developers with an existing ray tracing integration in their engines (acceleration
structure builds, `TraceRay` or `RayQuery` calls, and possibly ray-traced shadows, ambient occlusion
or other RT effects), who want to understand the techniques that separate such an integration from a path tracer.

Practical experience with rasterization and deferred shading is enough to follow this sample. No prior
familiarity with the theory of light transport is required: the sampling techniques the sample relies
on are introduced by the step that first uses them, and every implementation is short and present in the sample.

Much of that material will already be familiar in practice. Randomly jittering a sample and averaging
the results over frames, weighting a sample by how likely it was to be chosen, or blurring a noisy
signal more aggressively where it is noisier are all routine in real-time rendering. What the
documentation adds is the vocabulary, the reason each technique is constructed the way it is, and the
context of how everything fits together.

### Scope

The sample is a learning path, not a production integration:

- Performance is not the objective. Passes are kept separate and readable rather than fused and
  optimized. While code is kept reasonably optimized, clarity is always favored.
- Image quality is not the objective either. Several steps use approximations, each documented where it
  is introduced, with the goal of presenting the core technique.
- The steps are cumulative and intended to be read in order.

---

## Running the sample

The repository builds **two** executables from a shared static library (`RtxptCore`):

| Target | Entry point | What it is |
| - | - | - |
| `IntroSample` | [`Rtxpt/IntroSample/IntroSample.cpp`](../Rtxpt/IntroSample/IntroSample.cpp) | This sample — the step-by-step learning path |
| `Rtxpt` | [`Rtxpt/AdvancedSample.cpp`](../Rtxpt/AdvancedSample.cpp) | The Advanced Sample — the full RTX Path Tracing renderer |

Follow the [build instructions](../README.md#build) in the main README, then select and run the
`IntroSample` project. Command-line options (`--scene`, `--width`, `--height`, `--fullscreen`,
`--debug`, `--vk`) are shared with the main sample.

Launching **without arguments** loads the scene and camera used throughout these documents. Some
screenshots isolate a lighting contribution or use the **Pitch Black** environment; their captions
and preset sections describe those changes. Relaunching restores the starting camera.

### The UI

All Intro-specific controls live under the **Intro Sample** header in the left-hand panel:

- **Mode** — `Realtime` (the step progression) or `Reference` (an accumulating path tracer; see
  [Reference mode](#reference-mode)).
- **Step** — the progression dropdown. Selecting a step applies a full preset: every ray tracing
  toggle plus the recommended anti-aliasing / denoising mode. The sample launches on Step 8.
- The individual toggles below the dropdown (`Direct Lighting Mode`, `MIS Indirect Light`,
  `Use RT Reflections`, `Use RT Global Illumination`, `Use SHaRC`, `Use RTAO`) are what each preset
  actually sets. Change any of them by hand and the dropdown flips to **Custom** — this is the
  intended way to isolate a single technique.

Environment, animation, debug views and ReSTIR tuning are separate from the step presets and persist
when switching steps. For comparisons, keep the camera, animation time, exposure and AA mode fixed.

Camera: `W`/`A`/`S`/`D` to move, left-mouse-drag to look.

The step presets are defined in one table, `kIntroStepPresets` in
[`Rtxpt/IntroSample/IntroSampleUI.cpp`](../Rtxpt/IntroSample/IntroSampleUI.cpp), and the launch step is
the single constant `kIntroDefaultStep` in
[`Rtxpt/IntroSample/IntroSampleUI.h`](../Rtxpt/IntroSample/IntroSampleUI.h).

---

## The progression

| Step | Technique introduced |
| - | - |
| [Step 0](intro/step-0.md) | Baseline: image-based lighting, screen-space AO and reflections; ray tracing setup |
| [Step 1](intro/step-1.md) | Ray-traced ambient occlusion |
| [Step 2](intro/step-2.md) | Ray-traced global illumination |
| [Step 3](intro/step-3.md) | Ray-traced reflections |
| [Step 4](intro/step-4.md) | Stochastic direct lighting (next-event estimation) |
| [Step 5](intro/step-5.md) | Multi-bounce indirect lighting via the SHaRC radiance cache |
| [Step 6](intro/step-6.md) | A unified indirect path tracer |
| [Step 7](intro/step-7.md) | Streaming RIS for direct lighting |
| [Step 8](intro/step-8.md) | ReSTIR DI |

The steps are cumulative: each replaces one component and carries the rest forward. Each step document
states the preset it applies, the technique it introduces, and how it is implemented.

---

## The steps

### [Step 0 — Baseline (IBL + SSAO)](intro/step-0.md)

The only step that does not introduce a technique. A deferred renderer with a split-sum image-based
lighting model: the environment is prefiltered into a GGX specular mip chain and a spherical-harmonics
irradiance cube, with a BRDF LUT supplying the split-sum's second term. Ambient occlusion is computed by
screen-space GTAO, reflections by a Hi-Z screen-space ray march, and local analytic lights are evaluated
without visibility — a configuration equivalent to what a conventional raster engine produces.

It also serves as the check that the ray tracing integration works. Directional lights cast hard
ray-traced shadows, and the two ray types the step uses between them exercise the acceleration
structure, the ray tracing pipeline and hit groups, alpha-tested any-hit shaders and inline ray
queries — everything the later steps build on.

### [Step 1 — Ray-traced ambient occlusion](intro/step-1.md)

The horizon-search ambient occlusion is replaced by one cosine-weighted hemisphere ray per
half-resolution pixel, traced with an inline `RayQuery`. Occlusion now includes off-screen geometry,
although alpha cutouts are treated as opaque. The step also establishes the pattern the remainder of
the progression relies on: a small number of rays per pixel, reconstructed by a filter.

### [Step 2 — Ray-traced global illumination](intro/step-2.md)

The occlusion ray is replaced by a ray that returns radiance. One cosine-weighted diffuse ray per pixel,
shaded at its hit point with analytic-light and environment samples, supersedes the diffuse ambient
occlusion term and environment-irradiance fallback, introducing colour bleeding and indirect shadows.
RTAO still supplies occlusion for the screen-space reflection fallback in this step.
Denoising is delegated to DLSS Ray Reconstruction from this step onwards.

### [Step 3 — Ray-traced reflections](intro/step-3.md)

The screen-space ray march is replaced by a reflection ray importance-sampled from the GGX distribution
of visible normals, with its hit shaded by the same function the GI ray uses. The roughness blur chain
and prefiltered specular fallback are removed. Reflection radiance is sampled directly and denoised
by DLSS-RR; the existing BRDF-LUT energy compensation remains in the combine pass.

### [Step 4 — Stochastic direct lighting](intro/step-4.md)

Primary analytic lighting becomes a single-sample Monte Carlo estimate: one light is selected,
visibility is tested and the result is divided by the selection probability. Local lights gain
shadows and directional lights gain sampled penumbrae, with at most one shadow ray per surface pixel
in this pass. Uniform selection can be noisy when a few lights dominate — the problem addressed by
Steps 7 and 8.

### [Step 5 — SHaRC: a world-space radiance cache](intro/step-5.md)

Rather than extending path length per pixel, a sparse set of multi-bounce paths (one per 4×4 pixel
block) deposits radiance into a world-space hash grid. GI and reflection rays can use resolved cache
entries at their hits, falling back to direct shading when no acceptable entry exists. This adds
multi-bounce illumination while amortizing its training cost over pixels and frames.

### [Step 6 — A unified indirect path tracer](intro/step-6.md)

The separate diffuse GI and specular reflection passes become one indirect pass. Each pixel selects
a BSDF lobe stochastically and weights its contribution by the inverse selection probability.
Smooth reflections and glass can trace additional bounded continuations, with SHaRC queries guarded
by the last scattering event and the hit's voxel size. The diffuse and specular outputs are combined
before DLSS-RR reconstructs the image.

### [Step 7 — Streaming RIS for direct lighting](intro/step-7.md)

Uniform light selection is replaced by Resampled Importance Sampling: several inexpensive candidates
are drawn, weighted by an unshadowed target function, and one is retained for the single shadow ray.
This is the first step to call into the [RTXDI](https://github.com/NVIDIA-RTX/RTXDI) SDK directly.

### [Step 8 — ReSTIR DI](intro/step-8.md)

The weighted RIS selection is stored in a reservoir and reused across time (the previous frame,
reprojected) and space (neighbouring pixels). Reuse re-evaluates candidates at the receiving surface
and spends additional rays on visibility correction. Explicit emissive-triangle sampling is enabled;
importance-sampled environment candidates are optional and off by default. This is the launch configuration.

---

## Architecture

The Intro Sample implements the shared `Sample` base class ([`Rtxpt/Sample.h`](../Rtxpt/Sample.h)),
which provides scene loading, the camera, the global binding set, the shader and pipeline bakers, and
post-processing. The sample overrides `SampleRenderCode` and a small number of resource hooks;
everything specific to it lives in two directories:

| Path | Contents |
| - | - |
| [`Rtxpt/IntroSample/`](../Rtxpt/IntroSample) | Host code: `IntroPathTracer` and one helper class per technique |
| [`Rtxpt/Shaders/IntroSample/`](../Rtxpt/Shaders/IntroSample) | Intro shaders, using shared material, lighting and SDK includes |

Host-side, `IntroSample.cpp` owns the frame graph and the light bake; the per-technique renderers
(`AORenderer`, `ReflectionRenderer`, `RTGIRenderer`, `SHaRCRenderer`, `IndirectPathTracer`,
`DepthHierarchyRenderer`) own their own pipelines and render targets. Each step document names the
shaders and resources the technique it describes actually uses.

### Frame structure

`IntroPathTracer::SampleRenderCode` contains the entire frame. In real-time mode it always has the same
shape, regardless of which step is selected:

```
  PrepareLights            pack analytic lights; bake emissive triangles
        |
  PopulateGBuffer          primary rays -> base colour, normal, roughness/metal, depth, motion
        |
  Environment prefilter    GGX specular cube + irradiance cube (only when the environment changes)
        |
  Direct lighting          -> u_DirectLighting
        |
  Indirect lighting        -> u_RTGIOutput (diffuse) and u_SSRResult (specular)
        |
  DeferredLightingCombine  direct + indirect + emissive -> u_OutputColor
        |
  (shared post-processing) TAA / DLSS / DLSS-RR, tone mapping
```

Two structural properties determine how the progression is organized:

1. **Direct and indirect lighting are accumulated into separate buffers and combined at the end.** A
   step replaces the pass that produces one of those buffers without modifying the consumer;
   `DeferredLightingCombine` is the only pass that determines how the components are summed.
2. **Techniques are organized as separate passes with different workloads.** AO runs at half
   resolution, SHaRC trains sparse paths and resolves a hash table, and ReSTIR DI uses up to four
   passes. The unified indirect pass introduced in Step 6 can trace bounded continuations inside
   one ray-generation shader.

Which passes fill those buffers, and any auxiliary passes they require, is what each step changes.

### Lighting inputs

`IntroPathTracer::PrepareLights` runs every frame, before any pass that reads lights. Analytic lights
from the scene graph are packed on the CPU; emissive triangles are baked on the GPU by
[`IntroPrepareLights.hlsl`](../Rtxpt/Shaders/IntroSample/IntroPrepareLights.hlsl). The result is a pair
of parallel structure-of-arrays buffers, `Base` (`PolymorphicLightInfo`) and `Ex`
(`PolymorphicLightInfoEx`), mirroring the layout the Advanced Sample's `LightsBaker` produces, so that
passes needing only `Base` do not pay for `Ex` cache traffic.

Within each buffer, lights are grouped by type — directional, local analytic, emissive triangles — and
the offset and count of each group are published to shaders through `g_Const.lightControl`. Passes
select the groups they sample from those ranges. The buffer is double-sized and alternates halves each
frame. ReSTIR translates absolute indices between those halves before reusing a light. It resets
history when the light layout changes; preserving history across insertion or reordering would
require a per-light identity mapping.

### Materials at secondary hits

Primary hits use the full material model in
[`IntroCommon.hlsli`](../Rtxpt/Shaders/IntroSample/IntroCommon.hlsli): normal mapping, metal-rough and
specular-gloss workflows, transmission and occlusion textures. Secondary hits use
[`MaterialEvalSimplified.hlsli`](../Rtxpt/Shaders/IntroSample/MaterialEvalSimplified.hlsli) instead,
which reads base colour and emissive only, at a fixed texture LOD, and classifies the surface as
standard GGX, glass or emissive. The ray payload is reduced to match.

The simplification is a standard production technique rather than a concession made for the sample:
secondary rays are incoherent, their shading is low-frequency once denoised, and payload size and
texture bandwidth are the limiting factors on throughput. These shaders are compiled into a dedicated
pipeline, separate from the primary-visibility one.

RTGI, RT reflections and the unified indirect pass share `ComputeHitLighting` in
[`IntroHitLighting.hlsli`](../Rtxpt/Shaders/IntroSample/IntroHitLighting.hlsli) for direct shading at
secondary hits. SHaRC training and the reference tracer have their own path loops and lighting
bookkeeping. Steps 7 and 8 improve primary direct-light sampling; analytic-light selection at
secondary hits remains uniform.

---

## Reference mode

Setting **Mode** to `Reference` bypasses the real-time pipeline in favour of
[`IntroPathTracer.hlsl`](../Rtxpt/Shaders/IntroSample/IntroPathTracer.hlsl), a unidirectional path
tracer that accumulates over frames while the camera is stationary:

- up to 5 bounces (extended when passing through transparency),
- next-event estimation against all three light groups (one analytic, one directional, one emissive
  triangle per bounce, each shadow-tested and divided by its uniform selection pdf),
- stochastic BSDF lobe selection with the same `specMISProbability` heuristic the real-time MIS pass
  uses,
- a firefly clamp (`ReferenceFireflyFilterThreshold`) that trades a small amount of bias for a
  substantial reduction in variance.

It uses the same scene and light records, making it useful for checking the effect of finite-sample
noise and real-time approximations: allow it to converge, then switch back to Realtime and compare.
Its path construction and shading differ, so a mismatch alone does not identify which approximation
caused it.

The reference mode is not unbiased ground truth — it limits bounce count, clamps fireflies and
approximates refraction. For ground-truth comparisons, use the Advanced Sample's reference mode.

---

## Denoising and upscaling

Once a step produces roughly one sample per pixel of a high-variance signal, reconstruction becomes
part of the algorithm rather than a post-process. The sample implements no denoiser of its own for
these signals and delegates reconstruction to **DLSS-RR** (Ray Reconstruction), which is what keeps
each pass short enough to read. Each step's preset selects the anti-aliasing and denoising mode
appropriate to the signal it produces.

DLSS-RR receives the **combined noisy colour**, depth and motion, plus guide buffers supplied by
[`PrepareGuideBuffers.hlsl`](../Rtxpt/Shaders/IntroSample/PrepareGuideBuffers.hlsl): diffuse albedo,
specular albedo, normals and roughness, and specular motion vectors. The sample currently copies
screen motion into the specular-motion guide, which is approximate for reflected features.

Keeping diffuse indirect in `u_RTGIOutput` without primary albedo, and specular indirect in
`u_SSRResult`, is an internal shading convention. The combine pass applies the material response
and sums the lighting before reconstruction; these are not separate radiance layers supplied to DLSS-RR.

Where DLSS-RR is unavailable, the sample falls back to DLSS or TAA, and the image is correspondingly
noisier.

---

## Suggested experiments

The sample is built around comparison, and a few habits make that productive:

- **Advance through the steps from a fixed camera position.** The presets exist for this comparison,
  but some steps also change reconstruction or the candidate light set. Read the preset's change
  column before attributing every visible difference to one technique.
- **Select a step, then change one of its options.** The dropdown switches to `Custom`, isolating a
  single technique against the rest of that step's configuration.
- **Switch to Reference mode after a change** to distinguish bias from noise.

Each step document lists the observations and parameter changes specific to the technique it
introduces. Some `Custom` combinations reach outside what either step's document covers — an early
step's indirect technique paired with a later step's direct-lighting mode, for instance, can trigger
double-counting bookkeeping that is only explained where that interaction is introduced. If a change
under `Custom` looks unexplained, check whether a later step's document addresses it.

---

## References

Papers and resources the sample's techniques come from, collected here; each step document cites the
subset relevant to it.

**Ambient occlusion**
- Jimenez, Wu, Pesce, Jarabo. *Practical Real-Time Strategies for Accurate Indirect Occlusion.*
  SIGGRAPH 2016 Courses. (GTAO)

**Image-based lighting**
- Karis. *Real Shading in Unreal Engine 4.* SIGGRAPH 2013 Courses. (split-sum approximation)
- Fdez-Agüera. *A Multiple-Scattering Microfacet Model for Real-Time Image-Based Lighting.* JCGT 8(1),
  2019. (the multi-scatter energy compensation term)

**Microfacet sampling**
- Heitz. *Sampling the GGX Distribution of Visible Normals.* JCGT 7(4), 2018.
- Dupuy, Benyoub. *Sampling Visible GGX Normals with Spherical Caps.* HPG 2023.

**Screen-space reflections**
- Uludag. *Hi-Z Screen-Space Cone-Traced Reflections.* GPU Pro 5, 2014.

**Light sampling and resampling**
- Talbot, Cline, Egbert. *Importance Resampling for Global Illumination.* EGSR 2005. (RIS)
- Bitterli, Wyman, Pharr, Shirley, Lefohn, Jarosz. *Spatiotemporal Reservoir Resampling for Real-Time
  Ray Tracing with Dynamic Direct Lighting.* SIGGRAPH 2020. (ReSTIR DI)
- Wyman, Panteleev, Marrs, Kettunen, Bitterli. *A Gentle Introduction to ReSTIR.* SIGGRAPH 2023 Courses.

**Radiance caching**
- [SHARC](https://github.com/NVIDIA-RTX/SHARC) — Spatial Hash Radiance Cache SDK.

**SDKs used by this sample**
- [RTXDI](https://github.com/NVIDIA-RTX/RTXDI) — ReSTIR DI/GI. Its documentation is the deeper
  reference for Steps 7 and 8.
- [SHARC](https://github.com/NVIDIA-RTX/SHARC) — radiance cache, Step 5.
- [Streamline](https://github.com/NVIDIAGameWorks/Streamline) — DLSS integration.
- [Donut](https://github.com/NVIDIA-RTX/Donut) — the underlying rendering framework.

**Related samples**
- [`vk_raytracing_tutorial_KHR`](https://nvpro-samples.github.io/vk_raytracing_tutorial_KHR/) — a
  step-by-step Vulkan ray tracing tutorial, and a suitable starting point for readers without an
  existing ray tracing integration.
- The [Advanced Sample](../README.md) in this repository — a state-of-the-art real-time path tracer,
  and the reference for the techniques this sample does not cover.

---

Questions, corrections and suggestions are welcome; please open an issue or contact
`pathtracing-sdk-support@nvidia.com`.
