# Step 5 — SHaRC: multi-bounce GI from a world-space radiance cache

[← Step 4: Stochastic direct lighting](step-4.md) · [Index](../intro-sample.md) · [Step 6: Unified indirect →](step-6.md)

![Step 5 — GI with SHaRC](https://github.com/NVIDIA-RTX/RTXPT-Assets/raw/79edd32e02797d3624f56a153a768937e0432198/Docs/intro/step-5.jpg)

*GI debug view with SHaRC enabled. This shows the GI buffer before primary-surface albedo is applied,
using the default scene, camera and daylight environment. Primary direct lighting, indirect specular
and primary emission are omitted from this view; DLSS-RR and post-processing still run.*

**What's new:** the GI ray can retrieve lighting from several diffuse bounces with one cache lookup.

Step 4 shades each GI hit with direct lighting and an environment sample. Step 5 adds **SHaRC
(Spatially Hashed Radiance Cache)**: a sparse set of longer paths estimates outgoing radiance at
surfaces and stores it in a world-space grid. The full-resolution GI pass can reuse those estimates
instead of shading each hit independently.

This shares tracing work across pixels and frames. The sample's training paths are bounded, and the
cache introduces spatial approximation and temporal lag. Its update and resolve passes also add work;
it is not a guarantee of arbitrary bounce depth at the cost of one bounce.

---

## Preset

| Toggle | Value | Change from Step 4 |
| - | - | - |
| Use SHaRC | **on** | ← this step |
| Direct Lighting Mode | `Uniform Sampling` | |
| Use RT Global Illumination | on | |
| Use RT Reflections | on | |
| Use RTAO | off | |
| MIS Indirect Light | off | |
| AA / denoiser | DLSS-RR | |

Select **5: + SHaRC (multi-bounce GI)**. The SHaRC toggle is available when RTGI or MIS indirect is
enabled. In this step, **only RTGI queries the cache**: the separate reflection pass continues to
shade its hits as in Step 3. Step 6 uses the cache in the unified indirect pass.

To match the main screenshot, enable **Debug RTGI (irradiance only)** and keep the default camera.
If coming from Step 4's night view, restore **Scene → Environment Map → Override** to
**`==SCENE_DEFAULT==`**. For a repeatable still, disable animations under **Scene**, press
**Reset animation time**, and let the cache and DLSS-RR settle. Debug views persist across presets;
turn them off to see the combined lighting.

---

## The technique

### What the cache stores

In the sample's Lambertian model, outgoing diffuse radiance does not depend on viewing direction.
Nearby points on a similar surface can often share an estimate, so it is useful to cache that radiance
in world space. The approximation is less reliable across shadow boundaries, thin geometry and
changes in material or illumination.

The hash key combines quantised **world position**, a **distance-based grid level**, and coarse
**normal orientation**. The normal is encoded by the signs of its three components, not a full
angular description. Voxels grow with distance from the camera; the Intro integration uses a
logarithm base of 2, `sceneScale = 25` and `levelBias = 0`.

Each entry holds RGB outgoing radiance, including the cached surface's diffuse albedo and emission.
When a GI ray hits that surface, the value becomes incoming radiance at the primary surface. The
combine pass then applies the *primary* surface's albedo, as in Step 2.

The grid has no material or object identity in its key. Surfaces sharing a cell and normal group
share an estimate even if their materials differ. World-space storage allows reuse across camera
motion and at off-screen hits, but coverage still depends on where the training paths go. Entries
can also change level or expire; this is not a persistent cache of the entire scene.

### Three buffers, three ordered passes

[`SHaRCRenderer`](../../Rtxpt/IntroSample/SHaRCRenderer.cpp) allocates 2²² entries in each buffer:

| Buffer | Bytes per entry | Role |
| - | - | - |
| `u_SHaRCHashEntries` | 8 | Spatial key for each occupied slot |
| `u_SHaRCAccumulation` | 16 | New radiance sums and sample counts |
| `u_SHaRCResolved` | 16 | Filtered radiance and history metadata read by queries |

Together these buffers occupy **160 MiB**, before other rendering resources. The host clears them
before their first update.

Each frame runs the following sequence after primary visibility and direct lighting:

1. **Update:** trace sparse paths and deposit their contributions into the accumulation buffer.
2. **Resolve:** blend the new samples with history, expire stale entries and clear accumulation.
3. **Query:** run RTGI using the newly resolved cache, with direct shading as a fallback.

UAV barriers separate these passes. Queries can therefore use **this frame's resolved result**;
update and query are not simultaneous readers and writers of last frame's cache.

### Training on sparse paths

[`SHaRCUpdate.hlsl`](../../Rtxpt/Shaders/IntroSample/SHaRCUpdate.hlsl) dispatches at one quarter of
render width and height, rounded up: approximately **one training path per 4×4 render-pixel block**.
This is relative to the internal render resolution, before DLSS upscaling. Each thread selects a
pixel within its block using a hash that changes every frame:

```hlsl
uint blockSeed = Hash32Combine(Hash32(frameIndex), (quarterPixel.x << 16) | quarterPixel.y);
uint2 pixelPos = min(quarterPixel * 4 + uint2(blockSeed & 3, (blockSeed >> 2) & 3), fullResSize - 1);
```

A valid G-buffer hit starts a training path. At each surface, the shader evaluates
[`ComputeHitLighting`](../../Rtxpt/Shaders/IntroSample/IntroHitLighting.hlsli) and calls
`SharcUpdateHit`. The lighting helper samples one local and one directional analytic light when those
groups exist, tests visibility and adds surface emission. It does **not** take an extra environment
sample here: the continuation ray supplies environment lighting if it misses the scene.

For a diffuse continuation, the shader samples a cosine-weighted hemisphere. With diffuse albedo ρ
and sampling density p(ω), the segment weight is

$$
w = \frac{f_d(\omega)\,(n\cdot\omega)}{p(\omega)}
  = \frac{(\rho/\pi)\,(n\cdot\omega)}{(n\cdot\omega)/\pi}
  = \rho
$$

```hlsl
pathContext.rayDirection = SampleHemisphereCosineWeighted(rng.getNextUniformRand2(), normal);
SharcSetThroughput(sharcState, saturate(baseColor));
```

The albedo is the weight for **one segment**. `SharcSetThroughput` multiplies it into the weights of
all earlier vertices retained by the SDK. Each subsequent `SharcUpdateHit` deposits its local
lighting at the new vertex and propagates it to those earlier entries. `SharcUpdateMiss` similarly
propagates environment radiance. The caller must stop if `SharcUpdateHit` returns false, for example
when it cannot insert an entry.

The Intro configuration disables cache resampling during training and uses the SDK's default
`SHARC_PROPAGATION_DEPTH = 4`. This shader shades the primary surface and traces **at most three
continuation rays**, so a training path visits at most four surfaces. It does not recursively reuse
cached radiance to extend that path indefinitely. Different entries receive contributions from
paths with different amounts of tracing remaining.

Misses evaluate the environment at mip 0. By comparison, RTGI's direct-shading fallback uses its
configurable environment mip, normally 5. Differences between the two modes can therefore reflect
both additional bounces and different environment filtering.

### Resolving history

[`SHaRCResolve.hlsl`](../../Rtxpt/Shaders/IntroSample/SHaRCResolve.hlsl) dispatches one thread per
cache slot and calls `SharcResolveEntry`. Empty slots return immediately. For occupied entries it:

- combines new and previous radiance using their sample counts;
- limits history using `accumulationFrameNum = 64`, reducing the weight of older samples as the
  accumulated-frame count reaches that limit;
- evicts entries after **16 frames without training samples** (`2 * SHARC_STALE_FRAME_NUM_MIN`);
- clears the accumulation entry for the next frame.

Camera motion can change a surface's grid level. The resolve receives the previous camera position
and can blend from an adjacent level to help new entries start with history. Update, resolve and query
must agree on grid scale and encoding for these lookups to refer to the same cells.

This is sample-weighted temporal filtering, not a fixed 64-sample average or a promise of convergence
in one second. There is no per-entry anti-firefly rejection in this resolve pass. Rare bright samples
can remain noisy, and rapid lighting or geometry changes can leave stale radiance until the cache
adapts.

### Querying from the GI ray

With `RTGI_SHARC_ENABLED`, [`RTGI.hlsl`](../../Rtxpt/Shaders/IntroSample/RTGI.hlsl) first tries the
cache at the secondary hit:

```hlsl
if (payload.hitDistance >= 0.5 && shaderId != ShaderIdGlassGGX
    && SharcGetCachedRadiance(sharcParams, sharcHitData, cachedRadiance, false))
{
    irradiance *= cachedRadiance;
}
else
{
    irradiance *= ComputeHitLighting(...);
}
```

The **0.5-world-unit distance threshold** avoids using the cache for very short connections. It is a
fixed heuristic, not a test that the two endpoints occupy different voxels. A successful query also
requires a matching entry with more than two resolved samples (`SHARC_SAMPLE_NUM_THRESHOLD = 2`);
a missing or immature entry uses Step 2's direct-shading fallback.
A GI ray that misses geometry still evaluates the environment.

Here `shaderId` belongs to the **primary surface**: its simplified glass path does not query SHaRC.
This does not constitute general glass rejection at every training or query hit. Training transmits
straight through primary glass, but later surfaces use diffuse scattering. Glossy and refractive
transport are not faithfully represented by this RGB cache and simplified training shader.

The cache replaces shading at the GI hit, so its radiance must not also be added to
`ComputeHitLighting` for that hit. The RTGI buffer remains an estimate of irradiance divided by π
for diffuse primary surfaces; primary albedo is still applied by the combine pass.

### Inspecting the cache itself

![Step 5 — Cached radiance at primary surfaces](https://github.com/NVIDIA-RTX/RTXPT-Assets/raw/79edd32e02797d3624f56a153a768937e0432198/Docs/intro/step-5-cache.jpg)

*Debug SHaRC cache: cached outgoing radiance queried at primary hits. The blocks expose the spatial
approximation; this is a different quantity from the GI-only view at the top of the page.*

Enable **Debug SHaRC cache** to replace the combine pass with
[`SHaRCDebugCache.hlsl`](../../Rtxpt/Shaders/IntroSample/SHaRCDebugCache.hlsl). It looks up radiance at
the visible G-buffer surfaces themselves, rather than tracing a GI ray and querying its hit. The UI
turns off the GI and indirect-specular debug views when this is enabled.

The shader writes zero for a missing entry and the environment for a sky pixel. A black region can
also be a valid dark entry, so this is not an occupancy mask. A coloured region only demonstrates
that radiance is available, not that it has converged. Denoising and post-processing still affect the
displayed image.

---

## What to look for

- **Extra bounces and colour bleeding.** Keep **Debug RTGI (irradiance only)** enabled and try
  toggling **Use SHaRC** off and on. Look around the café doorway and beneath the awnings for changes
  in indirect light. Pause animations, keep the camera fixed and disable **AutoExposure** for the
  comparison. Allow a few seconds for the cache and DLSS-RR to settle after enabling SHaRC.
- **Cache coverage and spatial detail.** Switch to the cache debug view to inspect its blocks and
  material boundaries. Nearby surfaces can share a cell, causing colour bleeding or light leaks.
- **Response to motion.** Resume animations or move into a new region. New samples update the cache,
  while old lighting fades through history. Waiting can reduce noise, but cannot remove spatial bias
  or recover transport the training shader does not trace.
- **Cost of the whole feature.** A cache hit avoids the GI hit's direct-light and environment shadow
  tests. Balance that saving against sparse multi-bounce training, its shadow rays, a resolve over
  roughly four million slots, and cache memory traffic. The separate reflection pass still runs.

---

## What is still missing

| Limitation | Next step |
| - | - |
| Diffuse and specular indirect use separate full-screen ray passes | [Step 6](step-6.md) unifies them |
| Simplified handling of smooth specular and glass paths | [Step 6](step-6.md) adds delta-path handling; the cache remains approximate |
| Primary direct lighting still uses uniform light selection within each group | [Steps 7](step-7.md)–[8](step-8.md) improve selection and reuse |

Finite training depth, voxel sharing, incomplete coverage and temporal lag remain accuracy limits.
SHaRC provides an approximation that can make multi-bounce diffuse lighting practical within a
real-time budget.

---

## References

- NVIDIA. [SHaRC](https://github.com/NVIDIA-RTX/SHARC) — the radiance-cache library.
- [Bundled integration guide](https://github.com/NVIDIA-RTX/SHARC/blob/4e21b585c33c83d723ca9a1e11bbb1090d145793/docs/Integration.md) and
  [SDK implementation](https://github.com/NVIDIA-RTX/SHARC/blob/4e21b585c33c83d723ca9a1e11bbb1090d145793/include/SharcCommon.h) — the update, resolve and query
  protocol used by this checkout. The Intro sample's configuration and path tracer determine the
  behaviour described above.

---

[← Step 4: Stochastic direct lighting](step-4.md) · [Index](../intro-sample.md) · [Step 6: Unified indirect →](step-6.md)
