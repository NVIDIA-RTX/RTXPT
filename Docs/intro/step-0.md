# Step 0 — Baseline (IBL + SSAO)

[← Index](../intro-sample.md) · [Step 1: Ray-traced ambient occlusion →](step-1.md)

![Step 0 — Baseline](https://github.com/NVIDIA-RTX/RTXPT-Assets/raw/79edd32e02797d3624f56a153a768937e0432198/Docs/intro/step-0.jpg)

This step has two goals:

**It establishes the baseline image.** A conventional deferred renderer: a G-buffer, an image-based
lighting model built on the split-sum approximation, screen-space ambient occlusion, screen-space
reflections, and local analytic lights evaluated without visibility. None of it is specific to path
tracing, and all of it corresponds to what a modern raster engine already produces. It is the
before-picture that every later step is measured against, and the pass structure they plug into.

**It verifies the ray tracing machinery.** Nothing here is ray traced in a way that changes how the
image looks, with one exception: directional lights cast hard ray-traced shadows. That exception is not
arbitrary — hard sun shadows are the first thing many engines ray trace, and they are the simplest
end-to-end check that an integration works. The G-buffer pass is also raytraced to avoid complicating the
sample with a full rasterization pipeline, but the resulting G-Buffer could just as well have been generated
by rasterization. Between the two ray types this step does use, it exercises an acceleration structure,
a full ray tracing pipeline with per-material hit groups, alpha-tested any-hit shaders, a hand-designed
payload, and inline ray queries. If your renderer can produce this image, the machinery the rest of the
progression depends on is sound, and everything that follows is lighting.

---

## Preset

| Toggle | Value |
| - | - |
| Direct Lighting Mode | `Unshadowed` |
| Use RT Global Illumination | off |
| Use RT Reflections | off (→ screen-space reflections) |
| Use RTAO | off (→ GTAO) |
| Use SHaRC | off |
| MIS Indirect Light | off |
| AA / denoiser | TAA |

---

## What runs this frame

```
PopulateGBuffer  ->  ProcessEnvIBL  ->  DepthHierarchy  ->  GTAO  ->  DirectLighting
                                                                          |
                                              SSR (march + blur)  <-------+
                                                                          |
                                                   DeferredLightingCombine
```

### What the two rays are standing in for

Both places where this step traces a ray stand in for functionality an engine would normally already
have. Neither is advice about how to build a renderer; they exist so the sample does not have to ship
subsystems that are not the subject of the progression.

- **Primary visibility.** [`PopulateGBuffer.hlsl`](../../Rtxpt/Shaders/IntroSample/PopulateGBuffer.hlsl)
  fills the G-buffer from a ray-generation shader rather than a rasterizer. This corresponds to an
  existing depth pre-pass and G-buffer fill; rays are used only so the sample does not have to carry a
  second, raster geometry pipeline. Substituting a raster G-buffer has no effect on any later pass.
- **Directional light shadows.** Directional lights in
  [`DirectLightingPass.hlsl`](../../Rtxpt/Shaders/IntroSample/DirectLightingPass.hlsl) trace a
  visibility ray through `TraceVisibilityRay` — an inline `RayQuery` with
  `RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH`. This corresponds to a shadow map or a dedicated
  ray-traced shadow pass; a ray is used here to avoid including a shadow-map renderer in the sample.
  The shadows are **hard**: the ray is traced along the light's central direction, with no sampling of
  its solid angle. [Step 4](step-4.md) replaces this with a disk-sampled estimate and the shadows gain
  a penumbra. Scene-dependent — this only runs where the scene has a directional light, though
  `AddProceduralSunLight` synthesizes one from the procedural sky's sun when there is one.

All remaining lighting — ambient occlusion, indirect diffuse, reflections and local light shadows — is
either screen-space or unshadowed. Closing that gap is the subject of the remaining steps.

---

## The techniques

### Image-based lighting with the split-sum approximation

The specular part of the rendering equation at a surface point is

$$
L_o(\omega_o) \;=\; \int_{\Omega} L_i(\omega_i)\, f_s(\omega_i, \omega_o)\, (n \cdot \omega_i)\, d\omega_i
$$

which is far too expensive to evaluate per pixel against an environment map. The *split-sum*
approximation (Karis 2013) factors it into two independently precomputable pieces:

$$
L_o(\omega_o) \;\approx\;
\underbrace{\Big( \textstyle\int_{\Omega} L_i(\omega_i)\, D(\omega_i)\, d\omega_i \Big)}_{\text{prefiltered environment}}
\;\cdot\;
\underbrace{\Big( \textstyle\int_{\Omega} f_s(\omega_i, \omega_o)\, (n \cdot \omega_i)\, d\omega_i \Big)}_{\text{BRDF integration}}
$$

- The **first term** is a GGX-prefiltered cubemap: convolve the environment with the GGX lobe for a
  set of roughness values and store the results in a mip chain. `IntroPathTracer::ProcessEnvIBL` builds
  a 256² cube sampled at mip levels 0–7 (`kEnvGGXCubeSize`, `kEnvGGXMaxMip`), and only re-runs when the
  environment content actually changes. Roughness selects the mip: `ggxMip = roughness * kEnvGGXMaxMip`.
- The **second term** depends only on `(n·v)` and roughness, so it is a 2D lookup table — `t_BRDFLUT`,
  storing a scale and bias applied to `F0`: `specular = prefiltered * (F0 * A + B)`.

The diffuse term gets the same treatment, but the lobe is a cosine, so a low-order spherical-harmonics
projection is enough — that is `t_EnvIrradianceCube`, a small irradiance cubemap indexed by the shading
normal.

Both are consumed by `CombineIndirectLight` in
[`DeferredLightingCombine.hlsl`](../../Rtxpt/Shaders/IntroSample/DeferredLightingCombine.hlsl).

### Multiple-scattering energy compensation

The split-sum model is single-scattering: a microfacet BRDF that only accounts for one bounce inside
the microsurface loses energy as roughness rises — rough metals go visibly dark. The combine pass adds
the Fdez-Agüera compensation term, reconstructed entirely from values already in hand:

```hlsl
float3 FssEss = F0 * brdfLUT.x + brdfLUT.y;     // single-scatter energy
float  Ess    = brdfLUT.x + brdfLUT.y;          // directional albedo
float  Ems    = 1.0 - Ess;                      // energy lost to multiple scattering
float3 Favg   = F0 + (1.0 - F0) / 21.0;         // average Fresnel over the hemisphere
float3 Fms    = Favg / (1.0 - Ems * Favg);      // multi-scatter Fresnel (geometric series)

specular *= (1 + Fms * Ems);
diffuse  *= (1 - FssEss + Fms * Ems);
```

This compensation is applied in every later step as well, including where the specular
term comes from real reflection rays.

### GTAO — ground-truth ambient occlusion

Ambient occlusion approximates the visibility factor of the ambient (environment) term. GTAO computes
it in screen space by, per pixel, slicing the hemisphere with a plane containing the view vector,
searching along that slice for the maximum *horizon angle* on each side, and integrating the cosine-
weighted visible arc analytically.

[`GTAOPass.hlsl`](../../Rtxpt/Shaders/IntroSample/GTAOPass.hlsl) runs at half resolution and takes 12
steps per direction, reading depth from mip 1 of the Hi-Z pyramid to keep the sampling bandwidth-
friendly. For each slice it produces two horizon cosines, converts them to angles θ₁, θ₂,
clamps them to the normal's hemisphere, and evaluates the closed-form inner integral
(`AO_IntegrateArc`, Eq. 7 of the paper):

$$
\frac{1}{4}\Big[
\left(2\theta_1 \sin\gamma + \cos\gamma - \cos(2\theta_1 - \gamma)\right)
+ \left(2\theta_2 \sin\gamma + \cos\gamma - \cos(2\theta_2 - \gamma)\right)
\Big]
$$

where γ is the angle between the view direction and the normal projected onto the slice plane.
A thickness heuristic (`GTAO_THICKNESS_BETA`) decays the running horizon when a sample fails to raise
it, so thin foreground geometry does not occlude the whole background behind it.

One slice per pixel per frame would be far too noisy, so directions are distributed with a 4×4 spatial
tile (16 azimuths) rotated over 6 frames (`AO_SpatialDirection`), and the result is denoised by two
shared passes:

| Pass | What it does |
| - | - |
| [`AOSpatialPass.hlsl`](../../Rtxpt/Shaders/IntroSample/AOSpatialPass.hlsl) | 4×4 depth-aware bilateral filter matching the noise tile |
| [`AOTemporalUpscale.hlsl`](../../Rtxpt/Shaders/IntroSample/AOTemporalUpscale.hlsl) | Reprojected temporal accumulation, then bilateral upscale to full resolution |

The final AO is folded straight into the low byte of `u_GBufferMaterialInfo`, combined with the
material's own baked occlusion texture. The combine pass then applies a multi-bounce approximation
rather than a raw multiply, so albedo-heavy surfaces do not over-darken:

```hlsl
float3 multiBounceAO = ao / (1.0 - saturate(diffuseAlbedo) * (1.0 - ao));
```

### Screen-space reflections

[`SSRPasses.hlsl`](../../Rtxpt/Shaders/IntroSample/SSRPasses.hlsl) holds three entry points:

1. **`DepthHierarchyCS`** — builds the Hi-Z pyramid in a single dispatch. Each 256-thread group reduces
   a 64×64 tile with a max filter (reverse-Z, so max = nearest), the first 2×2 reduction folded into a
   max-sampler fetch, the rest through groupshared memory. Morton-ordered thread indexing keeps the
   reduction's memory access coherent. Used by both GTAO and SSR.

2. **`SSRCS`** — the ray march. The reflection direction is *not* the mirror direction: it is drawn
   from the GGX distribution of visible normals (`SampleVNDF`), giving glossy reflections a properly
   distributed ray and a `G2/G1` weight. The march then walks the Hi-Z pyramid: at each step, if the
   ray clears the current cell it advances to the cell edge and moves *up* a mip (bigger strides); if
   the ray passes behind the cell's depth it moves *down* a mip (finer test). Reaching mip −1 is a hit.
   This is the standard hierarchical traversal — logarithmic in screen distance rather than linear.

3. **`SSRBlurCS`** — builds a mip chain of the SSR result. Roughness selects a mip in the combine pass,
   faking a wider GGX lobe with a cheaper blur.

Two details are worth calling out because they change in Step 3:

- SSR does not read the lit colour buffer. It reads the *G-buffer* at the hit pixel and re-shades it
  with the same IBL model — which means screen-space reflections here carry no direct lighting.
- The result is written as `float4(colour, confidence)`, with confidence derived from how cleanly the
  ray landed and how far from grazing it was. The combine pass blends it over the prefiltered
  environment cube: `lerp(prefilteredColor, ssrResult.rgb, ssrResult.a)`. When SSR fails — off-screen,
  behind geometry, facing the camera — the image falls back gracefully to the environment.

### Direct lighting

[`DirectLightingPass.hlsl`](../../Rtxpt/Shaders/IntroSample/DirectLightingPass.hlsl) loops over *every*
analytic light in the scene:

- **Local lights** (point / spot / sphere) — full Lambert + Cook-Torrance evaluation, **no shadow ray**.
- **Directional lights** — evaluated along the light's central direction (no disk sampling), but *with*
  a visibility ray, so the sun casts shadows.

A deterministic loop over all lights has two limitations that the later steps address: its cost is
linear in the light count, and it omits local-light shadows entirely. Steps 4, 7 and 8 address both, in
that order.

---

## What to look for

- **Reflections disappear at screen edges** and wherever the reflected geometry is not in the
  G-buffer. Panning the camera makes reflections appear and disappear accordingly.
- **Ambient occlusion is limited to on-screen geometry.** Contact shadows are reasonable; large-scale
  occlusion from geometry outside the frustum is absent.
- **Local lights cast no shadows**, so light passes through geometry.
- **The ambient term is spatially flat.** The irradiance cube supplies the same environment lighting
  everywhere; there is no bounce light from nearby surfaces and no indirect colour bleeding.

The frame time recorded here serves as the baseline for the remaining steps.

---

## What is missing

| Problem | Fixed in |
| - | - |
| AO only sees what is on screen | [Step 1](step-1.md) |
| Ambient light is environment-only; no bounce, no colour bleed | [Step 2](step-2.md) |
| Reflections only see what is on screen | [Step 3](step-3.md) |
| Local lights cast no shadows | [Step 4](step-4.md) |
| No multi-bounce indirect lighting | [Step 5](step-5.md), after the single-bounce step |
| Primary direct lighting loops over every analytic light | [Step 4](step-4.md); Steps [7](step-7.md)–[8](step-8.md) improve selection and reuse |

---

## References

- Karis. *Real Shading in Unreal Engine 4.* SIGGRAPH 2013 Courses. — split-sum approximation, BRDF LUT.
- Fdez-Agüera. *A Multiple-Scattering Microfacet Model for Real-Time Image-Based Lighting.* JCGT 8(1),
  2019. — the energy compensation term.
- Jimenez, Wu, Pesce, Jarabo. *Practical Real-Time Strategies for Accurate Indirect Occlusion.*
  SIGGRAPH 2016 Courses. — GTAO; the sample's Eq. 7 / Eq. 13 references are to this paper.
- Uludag. *Hi-Z Screen-Space Cone-Traced Reflections.* GPU Pro 5, 2014. — hierarchical SSR traversal.
- Heitz. *Sampling the GGX Distribution of Visible Normals.* JCGT 7(4), 2018. — VNDF sampling.

---

[← Index](../intro-sample.md) · [Step 1: Ray-traced ambient occlusion →](step-1.md)
