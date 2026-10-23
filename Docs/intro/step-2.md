# Step 2 — Ray-traced global illumination

[← Step 1: Ray-traced AO](step-1.md) · [Index](../intro-sample.md) · [Step 3: Ray-traced reflections →](step-3.md)

![Step 2 — GI debug view](https://github.com/NVIDIA-RTX/RTXPT-Assets/raw/79edd32e02797d3624f56a153a768937e0432198/Docs/intro/step-2.jpg)

*GI debug view: **Debug RTGI (irradiance only)** is enabled. This isolates the GI buffer before the
primary surface's albedo is applied; the normal Step 2 view also includes direct lighting, reflections
and emission.*

**What's new:** the ray that returned a bit now returns *radiance*.

Ambient occlusion is replaced by one bounce of ray traced diffuse global illumination. A single
cosine-weighted diffuse ray per pixel, shaded at its hit point. The global environment map sample is now delayed
to tertiary bounces. As a result, we get more correct colour bleeding, indirect shadows and illumination.
Since environment map sampling is sampled on ray miss, we also get sky shadowing.

This is also the step at which the sample stops implementing its own denoiser and delegates
reconstruction to **DLSS Ray Reconstruction**, which is what makes one sample per pixel a workable
budget for the remaining steps.

---

## Preset

| Toggle | Value | Change from Step 1 |
| - | - | - |
| Use RT Global Illumination | **on** | ← this step |
| Use RTAO | on (unchanged) | still feeds the specular fallback — see below |
| AA / denoiser | **DLSS-RR** | ← this step |
| Direct Lighting Mode | `Unshadowed` | |
| Use RT Reflections | off (SSR) | |
| Use SHaRC | off | |
| MIS Indirect Light | off | |

---

## The technique

### From occlusion to radiance

The diffuse outgoing radiance at position x is

$$
L_o(x) \;=\; \frac{\rho(x)}{\pi} \int_{\Omega} L_i(x, \omega)\, (n \cdot \omega)\, d\omega
$$

Here ρ is diffuse albedo, n is the surface normal, Ω is the hemisphere above it, and
Lᵢ is incoming radiance: the light arriving from direction ω.


Ambient occlusion approximated this by pulling Lᵢ out of the integral as a constant (the environment
irradiance) and keeping only a visibility average. RTGI instead samples the integral
directly, with the same cosine-weighted pdf Step 1 used:

$$
p(\omega) = \frac{n \cdot \omega}{\pi}
\qquad\Longrightarrow\qquad
\widehat{L_o} \;=\; \frac{\rho}{\pi} \cdot \frac{L_i(\omega)\,(n \cdot \omega)}{p(\omega)} \;=\; \rho \cdot L_i(\omega)
$$

Everything cancels except *albedo times the incoming radiance along the sampled direction*. Albedo is
independent of the ray result, so [`RTGI.hlsl`](../../Rtxpt/Shaders/IntroSample/RTGI.hlsl) writes Lᵢ
into `u_RTGIOutput`, and `DeferredLightingCombine` multiplies by the primary surface's diffuse albedo.


The shader calls this intermediate value `irradiance`, but its expected value technically is irradiance **divided
by π**. That π factor has already cancelled through cosine-weighted sampling; the combine pass must
not divide by π again.

### Shading the hit

At a hit, `ComputeHitLighting`
([`IntroHitLighting.hlsli`](../../Rtxpt/Shaders/IntroSample/IntroHitLighting.hlsli)) estimates the outgoing
radiance toward the primary surface. It shades the hit as a **purely diffuse (Lambertian) surface**,
even if the actual material is glossy or transmissive, and adds emission where present. This keeps
secondary shading small, but it cannot reproduce specular or refractive light transport at that hit.

The primary surface can now receive light reflected from one other surface. The hit is lit by
analytic lights and the environment; it does not launch another GI ray to shade a third surface.
Emissive geometry contributes when the GI ray hits it directly. We are not yet sampling emissive
triangles explicitly as lights.

The added cost is **one GI ray plus up to three visibility rays at its hit**, per render pixel with a
surface. This is the cost of the GI pass, not the whole frame: primary visibility, RTAO, directional
shadows and SSR still run as well.

| Term | What it does | Rays |
| - | - | - |
| Emission | if the hit is an emissive material, add its colour | — |
| Analytic local light | pick **one** point/spot/sphere light uniformly, shadow-test, divide by the pick pdf | 1 |
| Directional light | pick **one** uniformly, sample its solid-angle disk, shadow-test | 1 |
| Environment | one cosine-weighted direction, visibility-tested against the scene, environment evaluated at a configurable mip (default 5, see below) | 1 |

The local and directional groups each use a uniform light-selection probability, corrected by
multiplying by that group's light count:

```hlsl
uint k = min(uint(rng.getNextUniformRand() * numLocal), numLocal - 1);
...
// Uniform pick over numLocal => divide by pdf = 1 / numLocal.
irradiance += EvaluatePolymorphicLightIrradiance(lightBase, dist2, spotAtten) * NdotL * numLocal;
```

The `* numLocal` factor makes the estimate unbiased for the sum of these local-light contributions.
Uniform selection can be noisy when only a few of the lights contribute strongly at the hit. Steps 7 and 8 improve direct-light sampling
at the primary surface; this secondary-hit function continues to select analytic lights uniformly.

### Misses and fireflies

```hlsl
irradiance *= envMap.Eval(rayDir, envMip);   // note the explicit mip
```

A GI ray that escapes the scene samples the environment at a fixed, pre-blurred mip rather than mip 0.
Sampling a high-dynamic-range environment at full resolution with one sample per pixel can easily
introduce fireflies: a single pixel of a bright sun sampled by one ray produces an outlier orders of
magnitude above its neighbours, which the denoiser will have a hard time cleaning up. Pre-blurring trades a small
amount of bias for a large amount of variance. As a side effect, look ups into a lower resolution mip will also
result in reduced bandwidth and a small improvement in performance. Look for this trade off repeatedly in the later steps:
SHaRC's anti-firefly filter, the reference path tracer's `FireflyClamp`, DLSS-RR's brightness clamp.

The mip is exposed as **Env Map Mip** in the UI (defaults to **5**) rather than hardcoded, because where
the accuracy/noise trade-off lands is scene-dependent: a low mip keeps small, bright environment
features (a sun disk, a window) intact but reintroduces the fireflies they cause; a high mip removes
that noise at the cost of washing those features into a soft average. If the environment map or
procedural sky is already smooth, changing this setting may have little visible effect. This control
does not change the separately sampled directional sun or the background seen directly by the camera.

### Glass

Glass has no diffuse component, so the sample reuses its GI ray for approximate transmission. At a
primary glass surface, the shader starts with the camera-ray direction, nudges it toward the flipped
normal, offsets the origin through the surface, and culls back-facing and non-opaque geometry. It
stores the result in `u_RTGIOutput`, attenuated by `1 - F`.

The Fresnel calculation uses `F0 = 0.08` as a rough allowance for entering and leaving the glass.
Neither this factor nor the direction change models a pair of physical refractions: there is no
Snell's-law calculation, thickness or absorption. This is a way to give primary glass a transmitted
view within the existing pass. Glass encountered at a secondary hit still uses the simplified hit
shading described above.

### Ambient occlusion — gone from diffuse, still alive in specular

`DeferredLightingCombine` applies AO in two separate places, and RTGI only silences one of them. On the
diffuse side, AO multiplies the environment-irradiance *fallback* — the moment RTGI drives the diffuse
term instead, that fallback (and the AO multiply with it) is skipped entirely, as it should be:
multiplying a computed indirect term by an approximate occlusion factor would double-count occlusion.

But in this step, we haven't touched reflections yet. Specular is still the screen-space path, and its
miss fallback (`prefilteredColor * ao`) is untouched by RTGI. RTAO remains on to preserve the occlusion
used by that fallback. A more advanced implementation could
derive AO from GI's hit distance and store it in the alpha channel, then denoise it before passing it on to reflections.
However, since this will soon be replaced in later steps, we decided to keep the simpler implementation.

---

## Denoising: enter DLSS Ray Reconstruction

Steps 0 and 1 filtered a scalar visibility signal bounded by [0,1]. RTGI instead produces noisy HDR
radiance, with colour, texture and potentially very bright outliers. The sample uses **DLSS-RR** to
reconstruct that signal, while retaining the existing AO filter for the specular fallback.

The order matters: `DeferredLightingCombine` applies albedo and combines direct lighting, diffuse
indirect, specular and emission into `u_OutputColor`. DLSS-RR receives **that combined colour**, not
`u_RTGIOutput` and `u_SSRResult` as separate lighting layers. It also receives depth and motion vectors
through the shared post-processing path, and the following guides from
[`PrepareGuideBuffers.hlsl`](../../Rtxpt/Shaders/IntroSample/PrepareGuideBuffers.hlsl):

| Guide | Contents |
| - | - |
| `u_RRDiffuseAlbedo` | `baseColor * (1 - metal)` |
| `u_RRSpecAlbedo` | `F0 * brdfLUT.x + brdfLUT.y` — the split-sum directional albedo |
| `u_RRNormalsAndRoughness` | shading normal + roughness |
| `u_RRSpecMotionVectors` | screen motion vectors |

It also clamps output radiance to `DLSSRRBrightnessClampK`, one more firefly guard.

The albedo guides describe the material response separately from the noisy colour input, helping
reconstruction preserve surface detail. Keeping the GI result free of the primary albedo until the
combine pass is an internal shading convention; it is not a requirement to supply separate diffuse
and specular radiance inputs to DLSS-RR.

Where DLSS-RR is unavailable, the sample falls back to DLSS or TAA, and the image is correspondingly
noisier at one sample per pixel.

---

## What to look for

Start from a fresh launch with no arguments and keep the default camera fixed:

- **Compare Steps 1 and 2.** Look beneath the café awnings, around the entrance and under the arch.
  The environment-only diffuse term is replaced by light that depends on the geometry around
  each point. Nearby surfaces can both block the sky and reflect light into those areas.
- **Isolate GI from reconstruction.** The presets also change TAA to DLSS-RR. To isolate the lighting
  change, select Step 2 and toggle **Use RT Global Illumination** off and on while keeping DLSS-RR
  selected. The dropdown becomes `Custom`; RTAO and SSR remain enabled.
- **Inspect the GI contribution.** Enable **Debug RTGI (irradiance only)** to display the GI buffer
  without the primary surface's albedo or the other lighting terms. The UI uses the shader's
  `irradiance` name; the value is the estimate described above. Shared post-processing still runs.
- **Try Env Map Mip 0 and 5.** Compare noise and environment detail, then restore 5. The difference
  depends on the environment's contrast; it does not blur the analytic sun's shadows.
- **Observe the remaining limits.** Deep interiors have no long chains of reflected light, primary
  local lights still illuminate through occluders, and reflections still lose off-screen geometry.

After the fixed-camera comparison, move the camera to expose disocclusions and watch reconstruction
settle in the newly revealed areas. Relaunching restores the starting scene and camera for another
comparison.

---

## What is still missing

| Problem | Fixed in |
| - | - |
| Reflections are still screen-space | [Step 3](step-3.md) |
| Local lights still cast no shadows on the primary surface | [Step 4](step-4.md) |
| No additional diffuse bounces beyond the GI hit | [Step 5](step-5.md) |
| Diffuse and specular indirect are two separate passes | [Step 6](step-6.md) |

Uniform analytic-light selection at secondary hits and their Lambertian material approximation
remain limitations of the later steps too.

---

## References

- Kajiya. *The Rendering Equation.* SIGGRAPH 1986.
- Pharr, Jakob, Humphreys. *Physically Based Rendering*, 4th ed., ch. 13–14 (Monte Carlo path tracing,
  next-event estimation).
- NVIDIA. [DLSS Ray Reconstruction](https://developer.nvidia.com/rtx/ray-reconstruction) — the denoiser
  this step delegates to.
- [Streamline SDK](https://github.com/NVIDIAGameWorks/Streamline) — how DLSS is integrated here.

---

[← Step 1: Ray-traced AO](step-1.md) · [Index](../intro-sample.md) · [Step 3: Ray-traced reflections →](step-3.md)
