# RTX Path Tracing Design
Last updated for v1.9.0

## Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Real-Time PT Pipeline Overview](#real-time-pt-pipeline-overview)
  - [Vertex Processing and BLAS/TLAS](#vertex-processing-and-blastlas)
  - [Early Light Update](#early-light-update)
  - [Path Trace Pre-Pass: V-Buffer and Path Decomposition](#path-trace-pre-pass-v-buffer-and-path-decomposition)
  - [Late Light Update](#late-light-update)
  - [SHaRC Update](#sharc-update)
  - [Main Path Tracing Pass](#main-path-tracing-pass)
  - [Denoiser Guide Generation](#denoiser-guide-generation)
  - [DLSS-RR Denoising](#dlss-rr-denoising)
  - [Tonemapping and Post-Process](#tonemapping-and-post-process)

## Overview
The specific architecture and design choices of RTXPT's real-time path tracer (`Rtxpt` project, `Sample.cpp`/`.h` and `AdvancedSample.cpp`) are influenced by:
 * Requirements and use cases from game engines
 * Battle-tested ray tracing and neural graphics research
 * Target hardware: NVIDIA Ada (GeForce RTX 40 Series) and Blackwell (GeForce RTX 50 Series) GPUs or equivalent, scaling from low-power laptop GPUs up to high-end desktop graphics cards
 * Forward-looking "universal" path tracing architecture (see [Falcor](https://github.com/NVIDIAGameWorks/Falcor)) compatible with popular path-tracing-based 3D creation tools and pipelines

There are many different real-time path tracing (PT) implementations and no universal best solution that fits every problem. RTXPT should therefore not be considered "the" way to do real-time PT, but a relatively generic, forward-looking implementation that tries to satisfy the most common use cases without specializing for any one application.

Sometimes, it can be "the" solution. Other times, it will be a useful repository for testing ideas and picking individual code components.

## Architecture

The `Rtxpt` project is built around a unified forward path tracing approach with next event estimation (NEE).

This means that a single instance of a ray tracing shader (raygen<->closesthit<->miss) can integrate over the area of a pixel and produce the final pixel value, including most light transport effects such as depth of field. This is followed by post-processing for tone mapping and certain effects such as bloom. Each material can compile its own shader permutation of the closesthit shader, and the design allows materials with custom inputs and a custom BSDF, although all materials currently use the same standard BSDF.

It follows that implementing a reference renderer is trivial: it uses the same light transport algorithms as the real-time one, providing (mostly) unbiased ground truth for the real-time mode to compare against.

The real-time path tracer can deviate from the reference implementation and indeed will "cheat" whenever possible, but the design remains fundamentally unified.

Guided light sampling, provided by NEE-AT (our feedback-based, temporally adaptive light importance sampling), is essential for high baseline signal quality. We occasionally break the model's simplicity where specialized techniques handle certain parts of the light transport, such as radiance caching (SHaRC) or ReSTIR DI/GI (now used only as a fallback when neural denoising with DLSS Ray Reconstruction (DLSS-RR) is not available). While these are often necessary, we try to maintain the simplicity of the unified architecture in order to keep code optimization and maintenance easy in the long run.

## Real-Time PT Pipeline Overview

![Real-time path tracing pipeline](https://github.com/NVIDIA-RTX/RTXPT-Assets/raw/79edd32e02797d3624f56a153a768937e0432198/Docs/Design/r-pipeline.png)

Let's examine the sequential steps that the current path tracing pipeline takes, as configured with the DLSS-RR denoiser, and the individual problems they solve:

### Vertex Processing and BLAS/TLAS

Animated (skinned) characters are posed on the GPU by a skinning compute pass, which also keeps the previous frame's vertex positions for motion vectors. Their BLASes are refit in place on frames where the skeleton moved, while static BLASes are built once at load; alpha-tested static meshes additionally get Opacity Micromaps, baked asynchronously over several frames and swapped in once ready. The TLAS is rebuilt from scratch on the GPU every frame.

### Early Light Update

This stage prepares the scene lights for sampling in NEE. We rely on the NEE-AT guided light sampling algorithm, which will be explained in detail separately; here we only describe how its work is split between this stage and the Late Light Update.

The early update collects all lights into a single list: the environment map, point and spot lights, and emissive triangles extracted on the GPU from the current (possibly skinned) geometry. The environment map, into which directional lights are baked, is re-baked when it changes and is represented as a quadtree of quad lights subdivided by importance. It then builds the global light sampler, whose per-light weights blend each light's emitted power with the previous frame's per-pixel sampling feedback (one contribution-weighted light pick per pixel).

None of this requires this frame's screen-space information, which is only produced by the following Path Trace Pre-Pass.

### Path Trace Pre-Pass: V-Buffer and Path Decomposition

This step traces the camera ray and its pure specular (delta) continuations, without NEE, and outputs the V-Buffer that the main pass restarts from, per-layer denoiser guides, and the emissive and sky radiance reached through delta bounces. Along the way it decomposes the path at delta lobes, inspired by [Path-space motion estimation and decomposition for robust animation filtering](https://cs.dartmouth.edu/~wjarosz/publications/zimmer15path.html).

Its first step is [Primary Surface Replacement](https://developer.nvidia.com/blog/rendering-perfect-reflections-and-refractions-in-path-traced-games/) (PSR), which provides correct motion vectors and denoiser guides for surfaces seen through chains of planar mirrors.

All initial mirror bounces are traced, accumulating the mirrors' 3x3 image transform (no translation or curvature), until the first non-mirror surface is reached...

![Primary Surface Replacement: scene with a wall mirror](https://github.com/NVIDIA-RTX/RTXPT-Assets/raw/79edd32e02797d3624f56a153a768937e0432198/Docs/Design/PSR-main.jpg)

...and the denoiser guides of that first non-mirror surface are output.

![Primary Surface Replacement: denoiser guides of the surfaces seen in the mirror](https://github.com/NVIDIA-RTX/RTXPT-Assets/raw/79edd32e02797d3624f56a153a768937e0432198/Docs/Design/PSR-guides.jpg)

This allows seamless motion vector tracking, and therefore stable denoising, of content seen through multiple stacked reflections:

https://github.com/user-attachments/assets/bf4cbff7-8de5-41eb-bd77-ff4ffa5d5665

https://github.com/user-attachments/assets/c2a3307b-ff63-4769-94b4-b818b1cf14e3

More complex cases, such as surfaces with both a delta reflection and a delta transmission lobe, are handled by decomposing the path into up to three layers, which allows (mostly) out-of-the-box handling of various pure specular setups. Decomposition is enabled per material.

For example, in this scene...

![Path decomposition example: glass sphere in front of a window](https://github.com/NVIDIA-RTX/RTXPT-Assets/raw/79edd32e02797d3624f56a153a768937e0432198/Docs/Design/PathDecomposition..jpg)

...we can help the denoiser by providing guides for the primary surface as well as for the first non-pure-specular surfaces seen through the first reflection and the first transmission:

![Path decomposition: normals of each layer](https://github.com/NVIDIA-RTX/RTXPT-Assets/raw/79edd32e02797d3624f56a153a768937e0432198/Docs/Design/PathDecompositionNormals.jpg)

The top-left quadrant shows the normals of the primary surface, the top-right the normals found by following the first transmission, and the bottom-left those found by following the first reflection. When NRD (NVIDIA Real-Time Denoisers) is used instead, each layer is denoised by its own NRD instance and the results are composited.

For DLSS-RR, however, we run a single denoising pass, as three would be prohibitively costly. Somewhat counter-intuitively, DLSS-RR denoises better when the layers' guides are blended into a single set than when it is given only one layer's guides: normals, roughness and diffuse/specular albedo are blended, weighted mostly by each layer's throughput. Depth and motion vectors cannot be blended; they come from the dominant layer, selected per material with the "Dominant bounce" property.

In this case...

https://github.com/user-attachments/assets/c3831ae0-2bbe-4dec-88b5-d2d47bd4a27a

...the glass material's "Dominant bounce" is set to "Transparency" (its transmission lobe), so DLSS-RR receives the motion vectors of what is seen through the window and refracted through the glass sphere, and the transmitted image is denoised better than the reflection. This is a field of active research.

### Late Light Update

This pass completes the NEE-AT pre-processing by building the local light samplers, one per screen tile, starting with reprojection of the previous frame's per-pixel light sampling feedback. It follows the Path Trace Pre-Pass because it needs that pass's depth and motion vectors for reprojection and history rejection.

### SHaRC Update

RTXPT relies on SHaRC, a spatially hashed radiance cache, to provide the indirect (mostly diffuse) signal, which is otherwise too noisy when produced by our path tracer. For example, in this scene most of the light entering the room comes indirectly from the environment map, bouncing off the walls of the building outside the window:

![SHaRC off (left) and on (right)](https://github.com/NVIDIA-RTX/RTXPT-Assets/raw/79edd32e02797d3624f56a153a768937e0432198/Docs/Design/SHaRC-off-on.jpg)

Without SHaRC (and with no path guiding), the indirect path-traced signal is too noisy and is mostly filtered out by the denoiser (left). With SHaRC we are able to provide a significant amount of this signal (right).

Perhaps surprisingly, this comes with a small performance improvement, as the SHaRC Update pass traces only one path per 4x4 pixel block (~6% of the pixels at render resolution) and the main path tracer can "terminate into cache" earlier, fully amortizing the added cost.

The downsides are that the indirect signal lags behind lighting changes, the approach is inherently biased, and there are specific minor visual artifacts.

For more detail, please refer to the [SHaRC documentation](https://github.com/NVIDIA-RTX/SHARC).

### Main Path Tracing Pass

This is the most resource-intensive part of the pipeline. Full path tracing starts from the primary surface stored in the V-Buffer by the pre-pass; each bounce computes the surface's material shading, BSDF sampling and throughput, and NEE using the light importance sampling structures prepared by the light updates.

In practice, this is a single `DispatchRays` with a thread per traced pixel. The raygen shader starts from the primary surface loaded from the V-Buffer and then loops over bounces. Each scatter ray is traced inline with a `RayQuery`, which handles alpha testing and Opacity Micromaps during traversal; a miss evaluates the environment map inline, while a hit becomes a hit object that is reordered with Shader Execution Reordering and then invokes the closest hit shader of the hit material's permutation (without hit object support, a regular `TraceRay` invokes it directly). The closest hit shader loads the surface (vertex attributes, material data and textures, with mip levels selected by ray cones), resolves nested dielectrics, collects emission, queries the SHaRC cache, samples the BSDF for the next ray and performs NEE, testing each light sample's visibility with an inline shadow `RayQuery`. The updated path state is packed back into the ray payload and returned to the raygen loop, which continues until the path terminates (bounce limit, Russian roulette or termination into the cache) and commits its radiance to the denoising layers.

The output radiance is split into separate denoising layers to support per-layer denoising with NRD; this is achieved by tracking the layer the path last passed through, as recorded by the pre-pass. Other than that, this is a classic forward path tracer that can be modified to suit any purpose or integrate common techniques.

![Path traced frame](https://github.com/NVIDIA-RTX/RTXPT-Assets/raw/79edd32e02797d3624f56a153a768937e0432198/Docs/Design/PT.png)

The main pass currently supports fixed-rate sparse rendering ("Sub-sampling" in the UI), tracing three quarters (default), half or a quarter of the pixels per frame, as a first step towards adaptive sampling.

### Denoiser Guide Generation

The per-layer guides are converted into the single set of inputs that DLSS-RR expects: normals, roughness and albedo are blended across layers as described above, the delta-path radiance from the pre-pass and the per-layer noisy radiance from the main pass are summed into the input color, and specular motion vectors are derived from the specular hit distance recorded by the main pass.

### DLSS-RR Denoising

DLSS-RR, evaluated through Streamline, denoises and upscales in a single pass, from the render resolution set by the DLSS quality mode to the display resolution. On D3D12 with enhanced barriers, the work leading up to it is by default recorded on the async compute queue, where everything before guide generation overlaps the previous frame's DLSS-RR (guide generation waits for it, as it rewrites DLSS-RR's inputs).

### Tonemapping and Post-Process

Post-processing runs at display resolution: bloom is added to the HDR image, then exposure (manual or automatic) and a tone mapping operator produce the final LDR image, before debug overlays and the UI are drawn on top.

---

*More complete documentation will follow in upcoming versions.*
