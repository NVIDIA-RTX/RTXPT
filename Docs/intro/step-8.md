# Step 8 — ReSTIR DI

[← Step 7: Streaming RIS](step-7.md) · [Index](../intro-sample.md)

![Step 8 — ReSTIR DI](https://github.com/NVIDIA-RTX/RTXPT-Assets/raw/79edd32e02797d3624f56a153a768937e0432198/Docs/intro/step-8.jpg)

*Combined lighting with temporal and spatial reuse, SHaRC and DLSS-RR. The scene, camera, paused
animation and Pitch Black environment match Step 7. Explicit environment sampling is off.*

**What's new:** retain the selected light in a reservoir, reuse it from the previous frame and share
it with nearby pixels. The initial sampler also gains an emissive-triangle stream.

Step 7 starts its selection from scratch at every pixel, every frame. Yet neighbouring surfaces often
receive light from the same sources, and a useful light in one frame may still be useful in the next.
**ReSTIR DI** (Reservoir-based Spatio-Temporal Importance Resampling for Direct Illumination) uses
those existing selections as candidates, re-evaluating them at the receiving surface.

The indirect path tracer, SHaRC and DLSS-RR continue to run as before. This is the final step and the
configuration the Intro Sample launches in.

---

## Preset

| Toggle | Value | Change from Step 7 |
| - | - | - |
| Direct Lighting Mode | **`ReSTIR DI`** | ← this step |
| Temporal / Spatial Reuse | on / on at launch | controls exposed in this mode |
| MIS Indirect Light | on | |
| Use SHaRC | on | |
| AA / denoiser | DLSS-RR | |

Select **8: + ReSTIR DI (spatio-temporal)**. The default launch uses daylight; the screenshot uses
**Scene → Environment Map → Override → `Pitch Black`**, as in Step 7. Keep the default camera and
pause animation after lights move near the café entrance.

The step preset selects the main lighting and AA options. ReSTIR tuning, environment and debug
settings persist, so restore **Temporal Reuse** and **Spatial Reuse** if an earlier experiment changed
them. The launch defaults come from
[`IntroSampleUI.h`](../../Rtxpt/IntroSample/IntroSampleUI.h).

---

## Reuse a weighted selection

A stored reservoir contains a light index, sample coordinates on that light, its target value,
normalized weight and M. Reuse treats that weighted result as another candidate; it does not read
back all the candidates that originally produced it.

For reservoirs built for the same target, weighted reservoir merging can reproduce selection from
their combined candidate lists. Across pixels or frames, the target changes: the receiving surface
may face away from the light, lie farther from it, or be occluded. RTXDI evaluates the reused sample
at the receiving surface and corrects the normalization using the contributing surfaces.

M acts as a confidence weight during reuse. It is **not a count of independent rays or independent
lighting observations**: neighbours and successive frames can share the same ancestors. More reuse
can reduce noise, but also introduces correlation and can delay adaptation to changes.

---

## The four passes

[`IntroPathTracer::ReSTIRDIDirectLightingPass`](../../Rtxpt/IntroSample/IntroSample.cpp) dispatches:

```
Initial  ->  Temporal  ->  Spatial  ->  Final
  RIS        reuse t-1     reuse xy     shade
```

Temporal and Spatial are independently optional. The RTXDI context assigns the reservoir-buffer
indices for each configuration, and UAV barriers separate the passes. Final reads Initial when both
reuse passes are disabled.

### 1. Initial: select a fresh candidate

[`IntroReSTIRDI_Initial.hlsl`](../../Rtxpt/Shaders/IntroSample/IntroReSTIRDI_Initial.hlsl) extends
Step 7's two streams:

| Stream | Launch candidate count | Sampling |
| - | - | - |
| Analytic local lights | 8 | Uniform light selection |
| Emissive triangles | 8 | Uniform triangle selection, then a point on the triangle |
| Infinite lights | 1 | Uniform directional light selection |
| Environment map | disabled | Optional importance-sampled environment stream; 1 candidate when first enabled |

Counts apply only when the corresponding light group exists. Keeping triangles separate from
analytic lights gives each group its own budget even when the scene has thousands of emissive triangles.

The streams are combined and normalized as in Step 7, then **M is reset to 1**. The pass traces a
visibility ray for the winner. If blocked, it removes the selected light and sets its weight to zero,
while retaining M for resampling normalization. The resulting reservoir is stored per pixel.

### 2. Temporal: reuse the previous frame

[`IntroReSTIRDI_Temporal.hlsl`](../../Rtxpt/Shaders/IntroSample/IntroReSTIRDI_Temporal.hlsl) uses
`u_MotionVectors` to find the previous surface. RTXDI checks its depth and normal against the current
surface, loads its reservoir, translates the stored light index to the current frame and combines
that sample with the fresh one.

The **History Length** setting limits the previous reservoir's M relative to the initial M. With
initial M = 1 and the default limit of 20, temporal reuse contributes at most 20 units of previous
confidence plus the current unit. Spatial reuse can subsequently increase M again. Temporal reuse
alone does not make M grow geometrically.

On startup or when history is incompatible, this pass simply forwards the initial reservoir. The
host resets reuse after resource recreation, light-layout changes, changes to candidate counts or
reuse mode, and frames in which ReSTIR DI did not run. This also prevents a disabled emitter or
environment stream from surviving in history while indirect tracing resumes counting those paths.

### 3. Spatial: reuse neighbouring pixels

[`IntroReSTIRDI_Spatial.hlsl`](../../Rtxpt/Shaders/IntroSample/IntroReSTIRDI_Spatial.hlsl) draws
neighbours from a disk, using a precomputed offset buffer. Depth and normal checks reject incompatible
surfaces before combining their reservoirs at the current pixel.

The default is **2 neighbour candidates within 32 internal render pixels**. Low-confidence pixels,
including fresh disocclusions, use a budget of **8 total neighbour candidates**, not eight additional
ones. The SDK activates this boost when the input reservoir has M below 12. Rejected neighbours do
not contribute, so the number actually combined can be smaller.

### Correct the reuse weights

Both reuse passes need to account for where a sample could have been selected:

- **Basic** evaluates the selected sample's target at the contributing surfaces and uses those values
  to normalize the result. It is more than dividing by M, but does not test visibility for that correction.
- **Ray-traced** also tests visibility for the normalization terms. This matters because the initial
  pass discards blocked selections: a neighbour that could not retain a sample should not contribute
  as though it could.

Temporal correction defaults to Ray-traced and can be switched to Basic in the UI. Spatial correction
uses Ray-traced. These checks add shadow rays beyond the Initial and Final visibility tests, so
**Step 8 does not retain Step 7's one-shadow-ray budget**.

The Intro sample uses the current TLAS and surface for the temporal visibility callback, following
the shared RTXDI bridge. It does not retain a previous-frame acceleration structure, so visibility
correction under motion remains approximate. History checks, finite candidate budgets and correlation
also limit what reuse can recover.

### 4. Final: shade the selected light

[`IntroReSTIRDI_Final.hlsl`](../../Rtxpt/Shaders/IntroSample/IntroReSTIRDI_Final.hlsl) reconstructs
the light sample from its index and UV, traces visibility at the current surface, then evaluates the
full diffuse and specular BRDF with the reservoir's weight. It writes `u_DirectLighting`, just as Step 7 does.

If the selected light is blocked, the pass removes that sample and weight from the stored reservoir,
retaining M. That prevents the known-blocked selection from being propagated into the next frame.
Final shading still traces visibility for each valid sample above the surface; it does not reuse a
previous frame's shadow result in place of that ray.

---

## Light records and indirect-light bookkeeping

### Keep light identity consistent between frames

`PrepareLights` packs analytic lights and bakes emissive triangles into two parallel light buffers.
Each buffer contains two halves, written on alternating frames. Reservoirs contain **absolute**
indices, so the same light's index shifts by one half's capacity each frame.

`RAB_TranslateLightIndex` moves an index to the corresponding slot in the other half, in either
direction. The host checks the ordered analytic-light identities, emissive geometry and triangle
counts, environment-light presence and buffer capacity. If that layout changes, reuse restarts from
fresh samples. This keeps the Intro integration small; an integration that preserves history across
light insertion, removal or reordering needs an identity-based mapping table.

### Explicitly sample emissive triangles

Emissive surfaces already lit the scene through indirect hits and SHaRC. Step 8 adds direct connections
from the primary surface to sampled emissive triangles, which is useful when an indirect ray is
unlikely to find a small emitter.

The CPU builds one task per emissive geometry and
[`IntroPrepareLights.hlsl`](../../Rtxpt/Shaders/IntroSample/IntroPrepareLights.hlsl) bakes a light record
per triangle. The Initial pass samples that region with its own candidate budget.

Those connections overlap with first-hit emission in the unified indirect tracer. When the triangle
stream is active and has candidates, the host sets the corresponding primary coverage flag. The
indirect shader omits emission reached directly after a **non-delta primary scatter**. Delta paths
remain eligible, and the flags are cleared after transmission or a later scatter. See
[Step 6](step-6.md#count-emission-and-the-environment-once) for the path bookkeeping.

### Optionally sample the environment

**Sample Environment Map** adds an importance-sampled stream using a synthetic environment-light
record and the environment's importance map. Its directions can then be retained and reused like
other light samples.

The sky already contributes on indirect-ray misses and through secondary lighting. Enabling this
stream changes how the primary environment connection is sampled; it does not introduce sky
visibility for the first time. With a nonzero environment candidate count, the unified indirect
tracer suppresses the overlapping non-delta primary miss contribution. The background remains visible.

---

## What to look for

- **Isolate reuse within Step 8.** Keep all candidate counts fixed, pause animation and compare both
  reuse passes off, Temporal only, Spatial only, then both on. Temporarily use **Disabled** under
  **AA/SR/Denoising** to expose sampling noise that DLSS-RR otherwise hides. Restore DLSS-RR afterwards.
- **Compare with Step 7 carefully.** Reuse off still leaves the emissive-triangle stream and the
  initial visibility test. For matching analytic candidate budgets, also set Triangle Local Candidates
  to 0, keep Sample Environment Map off, and use 8 analytic plus 1 infinite candidate. Ray cost still differs.
- **Try History Length 1 and 40.** Keep the spatial settings fixed and watch noise and response to
  moving lights. A value of 1 still permits one unit of previous confidence; disable Temporal Reuse
  for a comparison with no temporal history.
- **Move the camera after the fixed-camera comparison.** Newly revealed surfaces cannot rely on
  their own history. Compare how spatial reuse helps them, and where depth/normal rejection leaves noise.
- **Vary the triangle candidate count.** Small emitters can be difficult to find with uniform
  triangle sampling. Keep exposure and AA fixed; the intended improvement is sampling quality.
- **Try the Boiling Filter only if bright outliers are distracting.** It runs in the Temporal pass
  and culls reservoirs whose weights greatly exceed the tile average. This can suppress fireflies,
  but removing valid bright samples introduces bias. It is off by default.

The tuning controls change a trade-off among cost, variance, correlation and responsiveness. Larger
history and radius values are not universally better, and turning off a correction does not guarantee
an obvious artifact from every camera position.

---

## Where to go next

The progression now has resampled direct lighting, temporal and spatial reuse, cached multi-bounce
indirect lighting and a unified indirect scatter pass, reconstructed with DLSS-RR.

Limits remain: uniform light candidates can miss important emitters; secondary materials and glass
are simplified; the cache loses directional detail; and reuse and reconstruction depend on history.
The [RTXDI SDK](https://github.com/NVIDIA-RTX/RTXDI) covers more sampling and reuse options, including
ReGIR and ReSTIR GI. The [Advanced Sample](../../README.md) demonstrates a broader renderer with
different architecture, material handling and path-tracing techniques.

---

## References

- Bitterli et al. [*Spatiotemporal Reservoir Resampling for Real-Time Ray Tracing with Dynamic Direct
  Lighting.*](https://research.nvidia.com/publication/2020-07_spatiotemporal-reservoir-resampling-real-time-ray-tracing-dynamic-direct)
  SIGGRAPH 2020.
- Wyman et al. [*A Gentle Introduction to ReSTIR.*](https://intro-to-restir.cwyman.org/) SIGGRAPH 2023 Courses.
- Wyman, Panteleev. *Rearchitecting Spatiotemporal Resampling for Production.* HPG 2021.
- [RTXDI SDK](https://github.com/NVIDIA-RTX/RTXDI) — the implementation called by these passes.

---

[← Step 7: Streaming RIS](step-7.md) · [Index](../intro-sample.md)
