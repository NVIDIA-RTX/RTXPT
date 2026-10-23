# Step 7 — Streaming RIS for direct lighting

[← Step 6: Unified indirect](step-6.md) · [Index](../intro-sample.md) · [Step 8: ReSTIR DI →](step-8.md)

![Step 7 — Streaming RIS direct lighting](https://github.com/NVIDIA-RTX/RTXPT-Assets/raw/79edd32e02797d3624f56a153a768937e0432198/Docs/intro/step-7.jpg)

*Combined lighting with Streaming RIS, SHaRC and DLSS-RR. The Pitch Black environment makes the
analytic lights easier to see. The camera is the default launch camera; animation is paused with
moving lights near the café entrance.*

**What's new:** draw several inexpensive light candidates, then trace a shadow ray for one selected
in proportion to its estimated contribution.

[Step 4](step-4.md) chooses one analytic light uniformly within a selected group. That can waste the
shadow ray on a distant or dim light while a nearby bright light is rarely selected. Step 7 spends
more arithmetic on the selection, while keeping **at most one direct-light shadow ray per surface
pixel**. The unified indirect pass from Step 6 is unchanged.

---

## Preset

| Toggle | Value | Change from Step 6 |
| - | - | - |
| Direct Lighting Mode | **`Streaming RIS`** | ← this step |
| MIS Indirect Light | on | |
| Use SHaRC | on | |
| AA / denoiser | DLSS-RR | |

Select **7: + Streaming RIS Direct Lighting**. To match the screenshot, start without command-line
options, keep the camera fixed, and select **Scene → Environment Map → Override → `Pitch Black`**.
Let the animation bring lights near the camera, then pause it and allow the cache and denoiser to
settle. Environment, animation and debug settings persist across presets.

[`RTXDIStreamingRISLightingPass.hlsl`](../../Rtxpt/Shaders/IntroSample/RTXDIStreamingRISLightingPass.hlsl)
replaces `StochasticLightingPass.hlsl`. It writes the same `u_DirectLighting` buffer and is the first
step to use the [RTXDI](https://github.com/NVIDIA-RTX/RTXDI) SDK's direct-light sampling functions.

---

## The technique

### Resampled importance sampling

We would like to choose lights in proportion to the radiance they contribute at this pixel. Evaluating
every light, including its visibility, would undo the saving from stochastic direct lighting.
**Resampled Importance Sampling (RIS)** instead selects from a small, cheaply generated candidate set:

1. Draw M candidates from a source distribution p. Here, light indices are drawn uniformly within
   each group; the light sampler also chooses a point or direction on the selected light.
2. Score each candidate with a nonnegative target function p̂ that approximates its contribution.
3. Select one candidate with probability proportional to its weight, p̂ divided by the source density.
4. Evaluate the selected candidate's actual contribution, including visibility, and apply the RIS weight.

For a single candidate stream, with selected sample y and contribution f:

$$
w_i = \frac{\hat p(x_i)}{p(x_i)}, \qquad
\widehat L = \frac{f(y)}{\hat p(y)}\,\frac{1}{M}\sum_{i=1}^{M} w_i
$$

The average candidate weight estimates the target's unknown normalization. With consistent weights
and nonzero source and target densities wherever f contributes, this estimator is unbiased. A better
target can reduce variance, but a larger candidate set does not guarantee a better result in every scene.

### Keep one candidate in a reservoir

A **reservoir** stores the selected sample, a running weight sum and a count. It does not need to keep
the full candidate list:

```
weightSum += candidateWeight
if (random() * weightSum < candidateWeight)
    selected = candidate
M += 1
```

Each positive-weight candidate replaces the survivor with the appropriate probability. This produces
the same weighted selection as storing the list and selecting from it afterwards, using constant storage.

RTXDI implements this with `RTXDI_DIReservoir`. Its weight field is normalized before shading, so
`RTXDI_GetDIReservoirInvPdf` supplies the estimator's final multiplier. Step 7 keeps the reservoir
inside the shader invocation and discards it after shading. [Step 8](step-8.md) stores it for reuse.

### Score candidates without tracing them

The shared target function in
[`IntroRTXDIBridge.hlsli`](../../Rtxpt/Shaders/IntroSample/IntroRTXDIBridge.hlsli) is:

```hlsl
return dot(lightSample.radiance, 1) * NdotL / lightSample.solidAnglePdf;
```

This sums the RGB radiance, applies the surface cosine and accounts for the light sampler's
solid-angle PDF. It omits both the surface BRDF and visibility. Distance, light orientation and
emission still influence the score through the sampled light data.

Omitting visibility is a cost choice in this implementation: testing every candidate would require
many shadow rays. Consequently, a bright light behind a wall can win repeatedly and then contribute
zero. The final visibility test makes those samples dark, but does not recover the rays spent on them.

The full diffuse and specular BRDF is evaluated only for the winner. A BRDF-weighted target is present
behind `#if 0` in the bridge, but enabling it also requires supplying the material fields used when
evaluating previous-frame surfaces in Step 8; the current history stores only depth and normal.

### Sample two groups and combine their reservoirs

The direct pass uses two candidate streams:

| Stream | Candidates per surface pixel | Source |
| - | - | - |
| Analytic local lights | 8 | Uniform point/spot/sphere light selection |
| Infinite lights | 1 | Uniform directional light selection |

Empty groups produce no candidate. The local stream excludes emissive triangles. Giving directional
lights their own stream prevents a small number of them from being lost among many local lights.

RTXDI builds and normalizes each stream's reservoir, then the shader combines the two:

```hlsl
RTXDI_DIReservoir reservoir = RTXDI_EmptyDIReservoir();
RTXDI_CombineDIReservoirs(reservoir, localReservoir, 0.5, localReservoir.targetPdf);
bool selectInfinite = RTXDI_CombineDIReservoirs(reservoir, infiniteReservoir,
    RTXDI_GetNextRandom(rng), infiniteReservoir.targetPdf);
RTXDI_FinalizeResampling(reservoir, 1.0, 1.0);
reservoir.M = 1;
```

The `0.5` is the random-selection argument. The first positive-weight reservoir combined into an
empty one wins for any random value in [0, 1), so no random draw is needed there. The second combine
must make a random choice between the weighted streams.

Setting M to 1 makes the combined initial result one unit of confidence for subsequent resampling.
It does not mean that only one candidate was evaluated. Step 8 uses the same reset in its Initial
pass, then combines that result with history and neighbours.

### Shade the winner

For a valid selected sample above the surface, the pass traces one visibility ray and evaluates:

```hlsl
float invPdf = RTXDI_GetDIReservoirInvPdf(reservoir);
float3 incidentRadiance = selectedSample.radiance / selectedSample.solidAnglePdf * invPdf;
return IntroRTXDIEvaluateBRDF(surface, lightDir, NdotL) * incidentRadiance * NdotL;
```

The candidate work is bounded by the configured counts, rather than a loop over every light. That
statement applies to this selection pass: preparing light records, scene traversal, indirect lighting
and SHaRC still have their own costs. RIS also leaves the renderer's material, cache and denoising
approximations in place.

---

## What to look for

- **Compare `Uniform Sampling` and `Streaming RIS` in the Direct Lighting Mode control.** Keep
  animation paused, the camera fixed and the same AA mode. Look at surfaces lit by the moving
  analytic lights, especially where a few lights dominate the contribution.
- **Inspect the input noise.** Temporarily select **Disabled** under **AA/SR/Denoising**, then switch
  between the two direct modes. Indirect lighting is also noisy without reconstruction, so inspect
  the directly lit patches. Restore DLSS-RR afterwards.
- **Compare visible and occluded lights.** RIS can make better selections where its unshadowed
  score predicts the result, while occluded bright lights still waste samples.
- **Keep exposure fixed for brightness comparisons.** AutoExposure can conceal a change in energy.
  A change in selection strategy should primarily change noise, not deliberately brighten the scene.

Emissive surfaces already contribute through indirect rays and SHaRC. What remains missing is
**explicit emissive-triangle sampling in the primary direct-light pass**, along with reuse of the
selected light across pixels and frames. Step 8 adds those capabilities.

---

## References

- Talbot, Cline, Egbert. *Importance Resampling for Global Illumination.* EGSR 2005.
- Bitterli et al. [*Spatiotemporal Reservoir Resampling for Real-Time Ray Tracing with Dynamic Direct
  Lighting.*](https://research.nvidia.com/publication/2020-07_spatiotemporal-reservoir-resampling-real-time-ray-tracing-dynamic-direct)
  SIGGRAPH 2020.
- Wyman et al. [*A Gentle Introduction to ReSTIR.*](https://intro-to-restir.cwyman.org/) SIGGRAPH 2023 Courses.
- [RTXDI SDK](https://github.com/NVIDIA-RTX/RTXDI) — reservoir structures and initial light sampling.

---

[← Step 6: Unified indirect](step-6.md) · [Index](../intro-sample.md) · [Step 8: ReSTIR DI →](step-8.md)
