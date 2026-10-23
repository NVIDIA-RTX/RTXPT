# Step 4 — Stochastic direct lighting

[← Step 3: Ray-traced reflections](step-3.md) · [Index](../intro-sample.md) · [Step 5: SHaRC →](step-5.md)

![Step 4 — Stochastic direct lighting](https://github.com/NVIDIA-RTX/RTXPT-Assets/raw/79edd32e02797d3624f56a153a768937e0432198/Docs/intro/step-4.jpg)

*Combined lighting with the Pitch Black sky, highlighting local lights and their shadows. One analytic
light is sampled per render pixel; the GI and indirect specular debug views are off.*

**What's new:** local analytic lights cast shadows on the primary surface.

Step 3 already traced shadows for directional lights and for lighting at GI and reflection hits.
Local lights at the primary surface were still evaluated without visibility. Step 4 replaces the
loop over all analytic lights with **one randomly selected light and at most one shadow ray** per
render pixel with a surface. Dividing by the light's selection probability accounts for the lights
we did not evaluate this frame.

This is next-event estimation: explicitly sample a light and test the connection to it. The cost of
evaluating every local light is replaced by sampling noise. DLSS-RR reconstructs the result; Steps 7
and 8 improve which light is selected.

---

## Preset

| Toggle | Value | Change from Step 3 |
| - | - | - |
| Direct Lighting Mode | **`Uniform Sampling`** | ← this step |
| Use RT Global Illumination | on | |
| Use RT Reflections | on | |
| Use RTAO | off | |
| Use SHaRC | off | |
| MIS Indirect Light | off | |
| AA / denoiser | DLSS-RR | |

Select **4: + Uniform Direct Lighting** in the default scene. Turn off **Debug indirect specular**
or **Debug RTGI (irradiance only)** if either is enabled: these persist across presets and hide the
primary direct-lighting contribution this step changes.

To match the screenshot, keep the default camera and set **Scene → Environment Map → Override** to
**`==PROCEDURAL_SKY_PITCHBLACK==`**. This removes sky and sun illumination so the local lights stand out.
With animations enabled, reset the animation time and let it run for about 36 seconds to bring the
moving lights near the camera.

The host switches from
[`DirectLightingPass.hlsl`](../../Rtxpt/Shaders/IntroSample/DirectLightingPass.hlsl) to
[`StochasticLightingPass.hlsl`](../../Rtxpt/Shaders/IntroSample/StochasticLightingPass.hlsl).
Both write `u_DirectLighting`. RTGI, reflections and the combine pass carry forward from Step 3.

---

## The technique

### Replacing the light loop with one sample

For a fixed surface and viewing direction, write direct lighting as a sum of contributions from N
analytic lights:

$$
L_d = \sum_{k=1}^{N} C_k
$$

Each contribution includes the diffuse and specular BRDFs, incident illumination, the cosine term
and visibility. Choose a light k with probability p(k), then evaluate

$$
\widehat{L_d} = \frac{C_k}{p(k)}
\qquad\Longrightarrow\qquad
\mathbb{E}[\widehat{L_d}] = \sum_{k=1}^{N} p(k)\,\frac{C_k}{p(k)} = L_d
$$

Every contributing light must have a nonzero selection probability. A blocked light contributes zero;
we do not keep drawing until we find an unblocked one. See
[PBRT's light-sampling derivation](https://www.pbr-book.org/4ed/Light_Sources/Light_Sampling).

The weighting makes **light selection** unbiased. It does not remove approximations in the BRDF,
light shape, shadow test or denoiser. Those determine what each contribution represents.

### Uniform within two groups

The UI calls this **Uniform Sampling**, but the selection is not uniform across the entire light
list. Local and directional lights form separate groups:

| Available lights | Group selection | Selection within the chosen group |
| - | - | - |
| Both groups | 50% local, 50% directional | Uniform |
| Only one group | Always use that group | Uniform |
| Neither group | Write zero direct lighting | No sample or shadow ray |

If the directional group has Nᵢ lights and receives probability pᵢ, each directional light has
probability pᵢ/Nᵢ. Its contribution is therefore multiplied by Nᵢ/pᵢ. Locals use the corresponding
local count and group probability.

For example, with one directional light and 100 locals, the directional light is chosen half the
time and weighted by 2. Each local is chosen with probability 1/200 and weighted by 200. The split
keeps directional lights sampled frequently even when there are many locals, but it ignores their
relative brightness and visibility.

The shader uses one uniform random number for both decisions:

```hlsl
float pInfinite = (numInfiniteLights > 0) ? ((numLocalLights > 0) ? 0.5 : 1.0) : 0.0;
float u = rng.getNextUniformRand();

if (u < pInfinite) {
    float v = u / pInfinite;
    uint k = min(uint(numInfiniteLights * v), numInfiniteLights - 1);
    ...
    totalLight = contrib * (float(numInfiniteLights) / pInfinite);
} else {
    float pLocal = 1.0 - pInfinite;
    float v = (u - pInfinite) / pLocal;
    uint candidateLight = analyticStart + min(uint(numLocalLights * v), numLocalLights - 1);
    ...
    totalLight = radiance * (float(numLocalLights) / pLocal);
}
```

Remapping the selected interval to [0, 1) gives a uniform sample within that group. Directional
lights then consume two more random numbers to sample their disk.

This fixed group split differs from Step 7: Streaming RIS builds a reservoir for each group and
combines them using contribution weights. Its final local-versus-directional choice is not fixed
at 50/50.

### Evaluating the selected light

**Local lights** use the direction to `lightBase.Center`. Points use inverse-square attenuation;
spheres use their subtended solid angle, and spot attenuation is applied where present. The shader
evaluates Lambertian diffuse and Cook–Torrance specular, then multiplies by illumination and the
surface cosine.

The shadow ray also aims at the center. A sphere's radius affects its illumination, but this pass
does not sample points across its surface, so local shadows remain hard. Uniform light selection
does not by itself produce area-light penumbrae.

**Directional lights** sample a disk around their central direction:

```hlsl
float2 diskSample = SampleDiskUniform(diskRand);
float3 t, b;
BranchlessONB(lightDir, t, b);
float3 distantSample = lightDir + t * diskSample.x * sinHalfAngle
                               + b * diskSample.y * sinHalfAngle;
float3 L = -distantSample;
...
return (diffuse + specular) * radiance * NdotL * solidAngle;
```

This changes Step 3's hard central-direction shadow into a sampled soft shadow. The packed light
stores irradiance divided by its solid angle Ω; the evaluator multiplies by Ω to account for the
directional sample's nominal PDF, 1/Ω. Light selection adds its separate Nᵢ/pᵢ weight afterwards.

The disk construction is a small-angle approximation, shared with the polymorphic directional-light
sampler. It samples a planar disk and uses an approximately unit direction for shading; the visibility
ray uses `normalize(L)`. It is not exact uniform sampling over a spherical cap for large angular sizes.

### Testing visibility

[`TraceVisibilityRay`](../../Rtxpt/Shaders/IntroSample/IntroSampleShadowRay.hlsli) uses inline
`RayQuery` with `RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH`. A sample below the surface hemisphere
returns zero before tracing. Otherwise, local rays stop at `dist * 0.995`, and directional rays use
a maximum distance of `1e6` world units.

The origin offset grows with distance from the camera:

```hlsl
float viewDist = length(worldPos - g_Const.ptConsts.camera.PosW);
float shadowBias = max(1e-4, 1e-4 * viewDist);
// Ray origin: worldPos + normal * shadowBias
```

Using the reconstructed world position makes this independent of the camera's near-plane setting.
The offset reduces self-intersection, while shortening local rays avoids hitting the light itself.
Both are compromises: offsets can detach shadows, and the shortened endpoint can miss blockers very
close to the light.

For non-opaque triangle candidates, the query checks `Flags_ExcludeFromNEE` and commits the hit
unless that flag is set. It does not evaluate alpha textures or transmissive materials in this loop.
Opacity micromaps can resolve opacity during traversal when enabled; unresolved cutouts and glass
do not receive full material-aware shadow handling here.

### Light buffers and reconstruction

This pass currently reads local analytic lights from the shared `LightsBaker` buffers, `t_Lights`
and `t_LightsEx`, using `AnalyticLightOffset` and `AnalyticLightCount`. Directional lights come from
the Intro bake's `t_IntroRTXDILightsBase`, using `InfiniteRegionFirstLightIndex` and
`InfiniteRegionNumLights`. The Intro path samples directional lights analytically instead of baking
them into the environment cubemap.

The pass does not sample emissive triangles or the environment as direct-light candidates. They can
still appear through emission, RTGI and reflections. Explicit emissive-triangle and environment
sampling at the primary surface arrives in [Step 8](step-8.md).

The result goes into `u_DirectLighting`, then the combine pass adds it to indirect diffuse, indirect
specular and emission. DLSS-RR receives that combined colour. The random seed changes each frame,
but Step 4 stores no light-selection reservoir: reconstruction reuses image history, not selected
lights. Step 8 adds reuse of the light samples themselves.

---

## What to look for

Start from the default scene and camera with the Pitch Black sky and debug views off:

- **Compare Steps 3 and 4.** Look beneath the café awnings and around the doorway, tables and chairs.
  Local lights can now be blocked by the scene. GI still illuminates shadowed areas, so they should
  not become uniformly black.
- **Inspect directional shadow edges in daylight.** Set the environment override back to
  **`==SCENE_DEFAULT==`**. Directional shadows already existed; their edges can now soften because
  the sampled direction varies across the sun disk.
- **Expose the sampling noise.** Change **AA/SR/Denoising** from **DLSS-RR** to **Disabled**. Each render
  pixel receives only one direct-light sample per frame. GI and reflections are also unfiltered in
  this view. Switch back to **DLSS-RR** afterwards.
- **Consider which lights matter.** Selection ignores brightness, distance, orientation and occlusion.
  A few dominant lights among many weak or blocked ones produce rare, heavily weighted samples.
  Uniform selection works well when contributions are similar.
- **Separate pass cost from frame cost.** The direct pass evaluates one light and traces at most one
  shadow ray per surface pixel. Light preparation, traversal cost, primary visibility, RTGI and
  reflections still contribute to the frame's cost.

---

## What is still missing

| Problem | Addressed in |
| - | - |
| Light selection ignores contribution | [Step 7](step-7.md) |
| Selected lights are not reused across pixels or frames | [Step 8](step-8.md) |
| No explicit emissive-triangle or environment sampling in the primary direct pass | [Step 8](step-8.md) |
| No additional diffuse bounces beyond the secondary hit | [Step 5](step-5.md) |

Local center sampling, the directional disk approximation and simplified visibility remain limitations
of this step. The light-selection estimator does not make the entire renderer an unbiased reference.

---

## References

- Pharr, Jakob, Humphreys. [*Physically Based Rendering*, 4th ed., §12.6: Light Sampling.](https://www.pbr-book.org/4ed/Light_Sources/Light_Sampling)
  — estimating a sum over lights by sampling and inverse-probability weighting.

---

[← Step 3: Ray-traced reflections](step-3.md) · [Index](../intro-sample.md) · [Step 5: SHaRC →](step-5.md)
