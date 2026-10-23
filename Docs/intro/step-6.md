# Step 6 — A unified indirect path tracer

[← Step 5: SHaRC](step-5.md) · [Index](../intro-sample.md) · [Step 7: Streaming RIS →](step-7.md)

![Step 6 — Unified indirect lighting](https://github.com/NVIDIA-RTX/RTXPT-Assets/raw/79edd32e02797d3624f56a153a768937e0432198/Docs/intro/step-6.jpg)

*Combined lighting with the Step 6 preset, SHaRC and DLSS-RR, using the default scene, camera and
daylight environment. Smooth reflections and glass can continue beyond their first intersection.*

**What's new:** one indirect pass chooses between specular reflection and diffuse or transmitted
light at each pixel, replacing the separate RT reflections and RTGI passes.

Step 5 traces both lobes every frame. Here, each pixel randomly selects one and divides its contribution
by the selection probability. Across pixels and frames, both lobes still contribute. This saves one
full-screen dispatch and reduces ordinary indirect tracing to one ray per valid primary pixel, but
smooth reflections and glass may need additional rays. Shadow tests and SHaRC training also remain.

---

## Preset

| Toggle | Value | Change from Step 5 |
| - | - | - |
| MIS Indirect Light | **on** | ← this step |
| Use RT Global Illumination | superseded | disabled; the unified pass handles it |
| Use RT Reflections | superseded | disabled; the unified pass handles it |
| Use SHaRC | on | |
| Direct Lighting Mode | `Uniform Sampling` | |
| Use RTAO | off | disabled; neither AO consumer is active |
| AA / denoiser | DLSS-RR | |

Select **6: + MIS Indirect Light**. To match the screenshot, keep the default camera, turn off debug
views and use **Scene → Environment Map → Override → `==SCENE_DEFAULT==`**. Disable animations,
press **Reset animation time**, and allow the cache and denoiser to settle. Environment and debug
settings persist across presets.

[`IndirectPathTracer.hlsl`](../../Rtxpt/Shaders/IntroSample/IndirectPathTracer.hlsl) contains the
ray-generation shader, from lobe selection through the continuation loop and cache lookup. It uses
`TraceRay` and the same simplified hit shaders as RTGI, RT reflections and SHaRC training. The closest-hit
shader returns material and normal data in `PayloadSimplified`; lighting is evaluated in the
ray-generation shader. The host selects a plain or SHaRC-enabled variant according to **Use SHaRC**.

---

## The technique

### Sample one lobe

For an opaque surface, the reflected-light integral contains both specular and diffuse terms:

$$
L_o = \int_\Omega \big(f_s(\omega) + f_d(\omega)\big) L_i(\omega)\,|n\cdot\omega|\,d\omega
$$

Choose a lobe ℓ with probability pℓ, then a direction ω with that lobe's sampling density qℓ.
The corresponding one-sample estimate is

$$
\widehat{L_o} = \frac{f_\ell(\omega)\,L_i(\omega)\,|n\cdot\omega|}{p_\ell\,q_\ell(\omega)}
$$

Averaging over the lobe choice recovers the sum of the two integrals when the weights and sampling
support match. The UI calls this **MIS Indirect Light**; this implementation samples a BSDF component
and evaluates that component's contribution. It does not apply a balance heuristic between a light
sample and a BSDF sample.

The primary specular selection probability depends only on metalness:

```hlsl
float specProbability = lerp(0.2, 1, metal);
bool specularLobe = rng.getNextUniformRand() < specProbability;
```

A dielectric selects specular 20% of the time and diffuse or transmission 80%. A fully metallic
surface always selects specular. This is a simple allocation rule, not an optimum based on expected
contribution: infrequent branches can still be noisy, especially on materials with intermediate
metalness or strong grazing-angle reflection.

For rough specular reflection, the shader uses the GGX visible-normal sampling from [Step 3](step-3.md).
Its direction-sampling weight reduces to Fresnel times G₂/G₁, then divides by the lobe probability.
For diffuse reflection, cosine-weighted sampling cancels the cosine and 1/π factors, as in
[Step 2](step-2.md). The shader applies a view-dependent 1 − F factor and divides by the diffuse
selection probability; primary diffuse albedo is left for the combine pass.

The lobe-probability correction does not make the entire renderer unbiased. The cache, finite tracing
budget, simplified materials and existing BRDF-LUT compensation in the combine pass remain
approximations.

### Continue smooth reflections and glass

When the selected specular lobe has roughness **at or below 0.1**, the sample treats it as an ideal
mirror: it reflects around the surface normal instead of sampling a GGX half-vector. Primary glass
transmission is also marked as a delta event. Here, *delta* means a deterministic scattering direction.

The path loop tracks the most recent scattering event. At each intersection:

1. **An emitter ends the path.** Add its emission if it is not already covered by direct-light
   sampling. This check happens before any cache lookup.
2. **Glass transmits again**, if tracing budget remains. Apply tint and a transmission Fresnel factor,
   offset through the interface, bend the direction and continue. Clear the direct-light coverage
   flags because the next connection starts at a different surface.
3. **The first opaque surface reached after a delta event gets its own lighting.** If budget remains,
   sample its analytic direct lighting, multiply throughput by its diffuse albedo and scatter a
   cosine-weighted ray. That new segment is diffuse, even if the path started at glass or a mirror.
4. **The terminal opaque hit uses SHaRC or direct shading.** A miss instead contributes environment
   radiance, unless that connection was already sampled by the primary direct-light pass.

A typical rough or diffuse path traces one ray. A smooth reflection that reaches an opaque surface
usually traces two, giving that reflected surface both direct lighting and an indirect continuation.
The loop has a two-vertex budget, with glass crossings charged one quarter of a vertex and allowed
only when room remains for a following vertex. This bounds it to **five continuation rays**. It can
stop earlier on an emitter or miss; glass reached after the budget runs out contributes no further light.

Glass still uses Intro's approximate bend rather than Snell's law. For example, at a secondary glass hit:

```hlsl
ray.Direction = normalize(ray.Direction - 0.25 * hitNormal);
```

Back faces are culled on transmitted segments, but other non-opaque geometry is still traversed and
alpha-tested. This is not a model of nested dielectrics, absorption through a volume or total internal
reflection. Secondary opaque surfaces are shaded as Lambertian, regardless of their actual roughness
or metalness.

### Decide when the cache is safe to use

A direction-independent radiance cache can blur detail that a mirror or short connection preserves.
The terminal-hit query therefore requires:

- **A non-delta incoming segment.** This is based on the last scatter, not the original pixel's
  roughness or material. A path that started at glass can use SHaRC after a later diffuse scatter.
- **A segment at least as long as the hit's voxel width.** The width comes from the same distance-based
  grid level and scale used by training and resolve.
- **A matching entry with more than two resolved samples.** `SHARC_SAMPLE_NUM_THRESHOLD = 2` is shared
  by the Intro update, resolve and query shaders.

```hlsl
uint gridLevel = HashGridGetLevel(hitPos, sharc.hashGridParameters);
float voxelSize = HashGridGetVoxelSize(gridLevel, sharc.hashGridParameters);
if (!cameThroughDelta && hitDistance >= voxelSize
    && SharcGetCachedRadiance(sharc, sharcHit, cachedRadiance, false))
{
    radiance += throughput * cachedRadiance;
}
```

The voxel-distance test is a heuristic: it does not prove that the two endpoints occupy different
cells. A rejected query, immature entry or disabled cache falls back to sampled analytic direct
lighting plus a visibility-tested environment sample. That fallback uses environment mip 5; path
misses use mip 0. Step 5's separate RTGI pass retains its fixed 0.5-world-unit distance guard.

At a terminal hit, cached radiance **replaces** direct shading; the shader does not add both. The cache
still has the spatial bias, finite training depth and temporal lag described in [Step 5](step-5.md).

### Count emission and the environment once

In Step 6, uniform direct-light sampling covers analytic lights only. An indirect ray that directly
hits an emissive triangle or misses into the environment must therefore keep that contribution.

The same indirect pass is also used in Step 8, where ReSTIR DI can explicitly sample emissive triangles
and the environment at the primary surface. The host passes flags for whichever of those streams are
active. For a non-delta primary scatter, the indirect shader omits the corresponding first-hit emission
or environment miss to avoid adding it twice. Delta directions retain those contributions.

After transmission or a secondary diffuse scatter, those flags are cleared. Intro's secondary direct
lighting samples only analytic lights, so emitter hits and environment misses at the new vertex still
supply lighting. This is simple path partitioning, not full MIS between NEE and BSDF sampling.

### Combine the two outputs

The selected primary lobe determines which buffer receives the result; the other is zeroed:

```hlsl
u_SSRResult[pixel] = specularLobe ? result : 0;
u_RTGIOutput[pixel] = specularLobe ? 0 : result;
```

The specular buffer contains weighted reflected radiance. The diffuse/transmission buffer leaves
primary albedo for [`DeferredLightingCombine`](../../Rtxpt/Shaders/IntroSample/DeferredLightingCombine.hlsl),
which also applies the existing BRDF-LUT compensation and adds primary direct lighting and emission.
Glass transmission follows this diffuse-buffer route in Intro.

**DLSS-RR receives the combined colour**, along with diffuse/specular albedo guides and the other
reconstruction inputs. These two indirect buffers are not submitted as independently denoised layers.
Lobe selection changes the noise in the combined image, which reconstruction then filters over space
and time.

---

## What to look for

- **Lobe sampling noise.** Toggle **MIS Indirect Light** off and on with the camera fixed. The separate
  passes sample both lobes per pixel; the unified pass samples one. Pure metals already received a
  specular sample in Step 5, so this change does not automatically give them more samples.
- **Reflections and glass.** Look at the scooter and café windows. Use **Debug indirect specular only**
  to isolate the reflected contribution. Glass transmission is in the other buffer and is omitted from
  this view. The RTGI-only debug control is unavailable while MIS indirect is active.
- **SHaRC on and off.** Look for changes in indirect lighting and fine reflected detail. Pause
  animations, disable **AutoExposure**, and wait for history to settle. The voxel guard limits some
  cache artefacts but cannot eliminate spatial bias.
- **Cost of the unified pass.** Compare frame time with **MIS Indirect Light** off and on, keeping the
  other settings fixed and allowing shader compilation and temporal history to settle. One indirect
  dispatch can save work, but extra delta rays, shadow rays and cache training still matter. Frame time
  and image parity with Step 5 are not guaranteed: the path construction and environment filtering differ.

---

## What is still missing

| Limitation | Next step |
| - | - |
| Primary direct lighting selects lights uniformly within each group | [Step 7](step-7.md) improves selection |
| Direct-light samples are not reused across pixels or frames | [Step 8](step-8.md) adds ReSTIR DI |
| Emissive triangles are found only by indirect rays, not explicitly sampled as lights | [Step 8](step-8.md) samples them at primary surfaces |

Finite path depth, approximate glass, Lambertian secondary shading and cache bias remain throughout
Intro. The [Advanced Sample](../../README.md) provides more complete material and transport handling.

---

## References

- Veach. [Robust Monte Carlo Methods for Light Transport Simulation](https://graphics.stanford.edu/papers/veach_thesis/).
  Stanford, 1997, chapters 2 and 9 — Monte Carlo estimators and multiple importance sampling.
- Heitz. [Sampling the GGX Distribution of Visible Normals](https://jcgt.org/published/0007/04/01/).
  JCGT 7(4), 2018 — background for the rough-specular sampler.
- [Bundled SHaRC implementation](https://github.com/NVIDIA-RTX/SHARC/blob/4e21b585c33c83d723ca9a1e11bbb1090d145793/include/SharcCommon.h) and
  [Intro configuration](../../Rtxpt/Shaders/IntroSample/IntroSampleCommon.hlsli) — cache acceptance and grid parameters.

---

[← Step 5: SHaRC](step-5.md) · [Index](../intro-sample.md) · [Step 7: Streaming RIS →](step-7.md)
