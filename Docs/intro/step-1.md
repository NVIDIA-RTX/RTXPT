# Step 1 — Ray-traced ambient occlusion

[← Step 0: Baseline](step-0.md) · [Index](../intro-sample.md) · [Step 2: Ray-traced GI →](step-2.md)

![Step 1 — Ray-traced ambient occlusion](https://github.com/NVIDIA-RTX/RTXPT-Assets/raw/79edd32e02797d3624f56a153a768937e0432198/Docs/intro/step-1.jpg)

**What's new:** the screen-space horizon search is replaced by one actual ray per half-resolution pixel.

The step serves two purposes. It isolates the effect of ray tracing in one of the simplest possible cases
and it introduces the pattern the remaining steps rely on: a small number of rays per pixel, with filtering
used to reconstruct a usable estimate.

---

## Preset

| Toggle | Value | Change from Step 0 |
| - | - | - |
| Use RTAO | **on** | ← this step |
| Direct Lighting Mode | `Unshadowed` | |
| Use RT Global Illumination | off | |
| Use RT Reflections | off (SSR) | |
| Use SHaRC | off | |
| MIS Indirect Light | off | |
| AA / denoiser | TAA | |

Only one thing changed: `AORenderer::RenderRTAO` runs instead of `RenderGTAO`. The spatial filter, the
temporal accumulation, the bilateral upscale and every consumer downstream are identical.

---

## The technique

### What ambient occlusion actually is

Ambient occlusion is the cosine-weighted fraction of the hemisphere that is *not* blocked within some
radius r:

$$
AO(x) \;=\; \frac{1}{\pi} \int_{\Omega} V_r(x, \omega)\, (n \cdot \omega)\, d\omega
$$

where Vᵣ is 1 if nothing is hit within distance r along ω, and 0 otherwise.

GTAO evaluates this by *inferring* Vᵣ from the depth buffer — which is why it cannot see anything
that is not in the depth buffer. Ray tracing evaluates Vᵣ by asking the acceleration structure
directly.

### Why importance sampling, and why cosine

The integral above is an average over the hemisphere, and we only get to fire one ray per pixel, so one
sample of that average. Where should that one ray point?

The naive answer is "pick a uniformly random direction on the hemisphere." That works, but it spends
just as many rays near the horizon, where (n · ω) is close to zero and the sample barely
matters, as it does near the normal, where (n · ω) is largest and the sample matters most.
Rays spent on directions that contribute almost nothing are, statistically, wasted — they add variance
(noise) without adding much signal. This is what a purely random / brute-force approach to Monte Carlo
integration looks like, and it is why a naive path tracer is so noisy: most of its samples are spent
somewhere that barely moves the final average.

**Importance sampling** is the fix: instead of sampling directions uniformly, sample them *in
proportion to how much they're expected to matter*, and correct for the bias this introduces by dividing
each sample by the probability of having picked it (its PDF, p(ω)). Skew the dice toward the
directions that matter, then divide out the skew. Do this correctly and the estimate stays unbiased —
it converges to the same answer — but with far less noise per ray, because every ray is more likely to
land somewhere that actually moves the average.

The catch is that "how much a direction matters" is generally the *whole integrand*, which for AO
includes the visibility term Vᵣ — and we don't know whether a direction is occluded until we've
already traced the ray. We can't importance-sample the part we don't know yet. What we *can* do is
importance-sample the parts of the integrand we know in advance, analytically, before tracing anything.
Here that's the cosine weighting (n · ω) — it's a fixed property of the surface normal, cheap
to sample from, and it happens to already be the dominant term shaping how much each direction
contributes. This is the general recipe used throughout the rest of this sample: importance-sample
whatever part of the shading equation you already know the shape of, and let the ray tracing resolve
only the part you don't (visibility, or later, incoming light).

Concretely, **cosine-weighted hemisphere sampling** draws directions so that the probability density
matches the cosine lobe itself:

$$
p(\omega) = \frac{n \cdot \omega}{\pi}
$$

more samples near the normal, fewer near grazing angles, in exactly the proportion the cosine term would
have weighted them anyway. `SampleHemisphereCosineWeighted` below implements this with a standard,
cheap trick (Malley's method): pick a point uniformly on a disk, then project it up onto the hemisphere.
The sample uses a polar disk mapping with sine and cosine, followed by a square root for the height;
it requires no rejection loop.

### The estimator

[`RTAOTrace.hlsl`](../../Rtxpt/Shaders/IntroSample/RTAOTrace.hlsl) evaluates the integral with a
single-sample Monte Carlo estimate, importance-sampling the cosine:

$$
\widehat{AO} \;=\; \frac{1}{\pi}\cdot\frac{V_r(x,\omega)\,(n\cdot\omega)}{p(\omega)} \;=\; V_r(x,\omega)
$$

Because we chose p(ω) to be shaped exactly like the cosine term (up to the constant 1/π), the
cosine and the 1/π cancel exactly against the pdf, so the whole estimator collapses to "did the ray
hit anything?" — one binary sample, with the importance sampling already baked into *where the ray
pointed* rather than into any extra math at shading time. This cancellation is the reason cosine-weighted
hemisphere sampling recurs throughout this sample: whenever a Lambertian cosine term shows up in an
integral, sampling directions this way makes it disappear from the estimator entirely.

```hlsl
float3 localDir = SampleHemisphereCosineWeighted(rng.getNextUniformRand2());
float3 tangent, bitangent;
BranchlessONB(worldNormal, tangent, bitangent);
float3 rayDir = localDir.x * tangent + localDir.y * bitangent + localDir.z * worldNormal;

RayDesc ray;
ray.Origin    = worldPos + worldNormal * RTAO_NORMAL_BIAS;
ray.Direction = rayDir;
ray.TMin      = 0;
ray.TMax      = g_GTAO.Radius;              // AO is a bounded-distance query

RayQuery<RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> query;
query.TraceRayInline(SceneBVH, RAY_FLAG_FORCE_OPAQUE, 0xff, ray);
query.Proceed();

float ao = (query.CommittedStatus() == COMMITTED_NOTHING) ? 1.0 : 0.0;
```

Three details in this code will show up repeatedly in a path tracing implementation:

- **`RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH`.** A visibility query does not care *which* surface
  blocked it, so the traversal can stop at the first hit rather than finding the nearest one. This is
  useful for binary shadow and occlusion tests; computing partial transmission can require more hits.
- **`TMax = Radius`.** AO is defined over a bounded radius; a shorter ray is a cheaper ray.
- **The normal bias** (`RTAO_NORMAL_BIAS`, and the distance-scaled biases used elsewhere in the
  sample) prevents the ray from self-intersecting its originating surface. Self-intersection artefacts
  are a common failure mode when adding ray-traced effects; the bias is scaled with distance from the
  camera because the precision of world positions reconstructed from depth degrades with distance.
  You can find a more detailed explanation at [`Ray Tracing Gems, Chapter 6`](https://www.realtimerendering.com/raytracinggems/rtg/index.html)

### Denoising

One binary sample per half-resolution pixel is extremely noisy. To make the signal usable, we run it through
the exact same denoiser GTAO uses:

1. `AOSpatialPass.hlsl` — 4×4 depth-aware bilateral filter.
2. `AOTemporalUpscale.hlsl` — reprojected accumulation with depth rejection, then bilateral upscale to
   full resolution, written into the AO byte of `u_GBufferMaterialInfo`.

Notice that the result with RTAO is noisier than the GTAO baseline. GTAO, and other horizon based AO techniques
solve part of the AO integral analytically, which reduces the overall variance resulting from stochastic integration,
but makes some simplifying assumptions in the process (like treating the depth buffer as a heightfield) that
result in visible artifacts. RTAO does away with these simplifications, but as a result requires a more modern
denoiser for production use, like ReBLUR from NRD. However, since our goal here is to build a full path tracer,
and AO will be replaced by more robust global illumination, we will not dedicate time to that.

---

## What to look for

- **Off-screen occlusion.** With a large occluder positioned just outside the frustum, GTAO loses its
  contribution while RTAO retains it.
- **Thin geometry.** RTAO intersects scene triangles instead of inferring thickness from a depth
  discontinuity. This shader forces geometry opaque, so alpha cutouts and transmission still occlude
  as solid triangles.
- **Geometry occluded from the camera.** GTAO has access only to the front-most depth layer, so
  surfaces hidden behind others cannot contribute occlusion.
- **Disocclusion behaviour.** Under rapid camera motion, temporal accumulation fails at newly revealed
  pixels and the AO term is briefly noisy or flat until history is rebuilt. This failure mode is common
  to all temporally accumulated ray tracing and becomes more visible in later steps.
- **Cost.** One occlusion ray per half-resolution pixel. The difference relative to Step 0 is a useful
  unit for comparing subsequent steps.

---

## What is still missing

The raw RTAO sample estimates the bounded visibility integral for opaque geometry. Applying its
filtered scalar result to environment lighting remains an approximation with these limits:

- It carries **no colour of the occluder**. The ray itself returns a single visibility bit, not
  radiance, so nothing about what blocked it — its albedo, its own lighting — reaches the shading pass.
  `DeferredLightingCombine` does tint the result, but only by feeding the scalar AO through the same
  multi-bounce geometric series GTAO already used in Step 0 (`ao / (1 - albedo * (1 - ao))`), which
  colours the darkening with the *shaded surface's own* albedo, not the occluder's. A red wall next to a
  white one still doesn't bleed red onto it — the white wall just darkens a little less than a
  saturated one would, in its own colour.
- It carries **no directionality**. Occluded environment light is attenuated uniformly, irrespective of
  which part of the environment was blocked.
- It is **bounded by a radius**, a tuning parameter with no physical meaning.
- It is applied to a **spatially flat irradiance cube**, which models no light arriving from nearby
  geometry.

All four limitations are removed if the ray returns radiance rather than a visibility bit, which is the
subject of [Step 2](step-2.md).

---

## References

- Jimenez, Wu, Pesce, Jarabo. *Practical Real-Time Strategies for Accurate Indirect Occlusion.*
  SIGGRAPH 2016 Courses. — for what RTAO is replacing, and the shared filter chain.
- Pharr, Jakob, Humphreys. *Physically Based Rendering*, 4th ed., ch. 2 (Monte Carlo integration) and
  ch. A.5 (cosine-weighted hemisphere sampling).
- Microsoft. [DirectX Raytracing 1.1 — inline raytracing](https://microsoft.github.io/DirectX-Specs/d3d/Raytracing.html#inline-raytracing)
  — the `RayQuery` model used here.

---

[← Step 0: Baseline](step-0.md) · [Index](../intro-sample.md) · [Step 2: Ray-traced GI →](step-2.md)
