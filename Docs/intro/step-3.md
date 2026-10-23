# Step 3 — Ray-traced reflections

[← Step 2: Ray-traced GI](step-2.md) · [Index](../intro-sample.md) · [Step 4: Stochastic direct lighting →](step-4.md)

![Step 3 — Indirect specular from ray-traced reflections](https://github.com/NVIDIA-RTX/RTXPT-Assets/raw/79edd32e02797d3624f56a153a768937e0432198/Docs/intro/step-3.jpg)

*The indirect specular contribution, shown with **Debug indirect specular**.*

**What's new:** reflection rays can reach geometry outside the camera's view.

Step 2 traced a ray for diffuse indirect lighting; Step 3 does the same for specular. The ray is traced
against scene geometry instead of the screen-space depth buffer, and its hit is shaded with analytic
lights and the environment. Reflections can now show off-screen objects and surfaces hidden behind
the camera's first depth layer.

The SSR implementation already sampled GGX directions. What changes here is where those rays can go,
how their hits are lit, and how their contributions are combined: the roughness blur chain and
split-sum specular approximation are no longer needed on this path.

---

## Preset

| Toggle | Value | Change from Step 2 |
| - | - | - |
| Use RT Reflections | **on** | ← this step |
| Use RT Global Illumination | on | |
| Direct Lighting Mode | `Unshadowed` | |
| Use RTAO | **off** | ← neither indirect term needs the AO fallback |
| Use SHaRC | off | |
| MIS Indirect Light | off | |
| AA / denoiser | DLSS-RR | |

Select **3: + RT Reflections** in the default scene. Enable **Debug indirect specular**
to reproduce the screenshot; it turns off the GI debug view. This shows the weighted reflection
contribution, including Fresnel and energy compensation, without diffuse lighting, direct highlights,
primary emission or the background. Denoising and post-processing still apply.

The debug views persist across presets. Turn them off to restore combined lighting.

---

## The technique

### Sampling the specular lobe

The specular integral is the same rendering equation as before, with a microfacet BRDF:

$$
L_o(\omega_o) \;=\; \int_{\Omega} f_s(\omega_i, \omega_o)\, L_i(\omega_i)\, (n \cdot \omega_i)\, d\omega_i
$$

Here ωₒ points toward the viewer, ωᵢ points toward the incoming light, and fₛ is the
specular BRDF. For GGX, we sample a **microfacet normal** *m* from the *distribution of visible normals*
(VNDF), then reflect −ωₒ about it. If Dᵥ(m) is the PDF of that normal, the reflected
direction has PDF

$$
p(\omega_i) \;=\; \frac{D_v(m)}{4\,|\omega_o \cdot m|}
$$

The factor 4|ωₒ · m| converts from normal directions to reflection directions. For the
ideal VNDF, the one-sample estimator simplifies to

$$
\widehat{L_o} \;=\; \frac{f_s\,L_i\,(n \cdot \omega_i)}{p(\omega_i)}
\;=\; L_i\,F(\omega_o \cdot m)\,\frac{G_2(\omega_i, \omega_o)}{G_1(\omega_o)}
$$

The normal distribution term cancels. The remaining weight is Fresnel times a Smith masking-shadowing
ratio: G₁ accounts for visibility from the viewer, and G₂ for visibility from both directions.
See [Heitz's derivation](https://jcgt.org/published/0007/04/01/paper.pdf), Appendix B. In
[`RTReflections.hlsl`](../../Rtxpt/Shaders/IntroSample/RTReflections.hlsl):

```hlsl
float3 halfVector = SampleVNDF(u, viewDir, worldNormal, roughness * roughness);
float3 reflectDir = reflect(-viewDir, halfVector);
float  invPdfVNDF = VNDF_SpecularBRDF(roughness, worldNormal, halfVector, viewDir, reflectDir);
...
reflectionColor *= invPdfVNDF;                                  // G2 / G1
float3 F0 = lerp(0.04, packedSurface.GetBaseColor(), metal);
reflectionColor *= Schlick_Fresnel(F0, saturate(dot(viewDir, halfVector)));   // F
```

Despite its name, `invPdfVNDF` contains the BRDF-times-cosine weight after division by the PDF, excluding
Fresnel — not just an inverse PDF. `SampleVNDF` calls Donut's Heitz-style projected-disk sampler. The
implementation clamps the sampling roughness and trims the disk with `ndf_trim = 0.98`; these are
practical approximations to the ideal estimator above. Samples below the surface receive zero weight,
and the pass writes zero at grazing angles where `NdotV < 0.01`.

Roughness determines the spread of sampled directions: smooth surfaces give sharp reflections, while
rough surfaces integrate light over a wider lobe. Unlike the SSR path, this result receives no
additional roughness-dependent mip-chain blur. DLSS-RR reconstructs the noisy radiance instead.

### Shading the hit

Reflection hits go through the same `ComputeHitLighting` that Step 2's GI ray uses: one uniformly
selected local light, one directional light and one cosine-weighted environment sample, each
visibility-tested. This costs one reflection ray plus up to three visibility rays at its hit.

The hit is still shaded as diffuse, regardless of its material. We can reflect a lit object, but do
not yet trace another specular bounce to see a reflection within that reflection. The hit's environment
sample uses the helper's default **mip 5**, independently of the RTGI **Env Map Mip** control.

Emission is included only for sharp reflections:

```hlsl
bool countEmissive = roughness < 0.25;   // only sharp reflections keep the emitter's own emission
```

On a miss, the environment is sampled at **mip 0**:

```hlsl
reflectionColor = envMap.Eval(reflectDir, 0.f);
```

This preserves environment detail in smooth reflections. Roughness already spreads the ray directions;
pre-blurring the environment would broaden the result again. Bright, small features can still produce
fireflies at one sample per pixel.

The `countEmissive` threshold suppresses another source of fireflies, but also removes emitter radiance
from rough reflections. It is a biased shortcut, not MIS. [Step 6](step-6.md) combines diffuse and
specular lobe selection into one pass; it does not implement MIS between light sampling and BSDF
sampling to resolve this trade-off.

### The combine pass changes shape

`DeferredLightingCombine` is compiled into two variants, selected by the `DIRECT_SPECULAR_READ` macro:

| Variant | Specular source | Used when |
| - | - | - |
| `DIRECT_SPECULAR_READ=0` | `t_SSRBlurChain`, mip-selected by roughness, blended over the prefiltered environment cube, then multiplied by the split-sum term `F0*A + B` | Steps 0–2 (SSR) |
| `DIRECT_SPECULAR_READ=1` | `u_SSRResult` read straight from the UAV — no blur, no split-sum | Step 3+ (RT reflections, and MIS indirect) |

```hlsl
#if DIRECT_SPECULAR_READ
    float3 specular = u_SSRResult[pixelPos].xyz;   // already fully shaded, including F and G2/G1
#else
    ...prefiltered cube + SSR blend + split-sum...
#endif
```

The primary surface's specular weight — including the Fresnel tint of metals — is already in
`u_SSRResult`. Multiplying by `F0*A + B` again would apply that material response twice.

RTAO drops out because both indirect terms now obtain visibility from their rays. RTGI bypasses the
AO-modulated diffuse environment term, and RT reflections bypass SSR's `prefilteredColor * ao`
fallback. A reflection ray that misses samples the environment directly, without an AO multiplier.

What *does* survive is the multiple-scattering energy compensation term from Step 0. It is a property
of the microfacet model, not of how the reflection was obtained, so it applies to ray-traced specular
just as it did to prefiltered specular.

The SSR ray march, SSR blur chain, AO passes and Hi-Z depth pyramid are skipped in this preset. AO and
the depth pyramid are enabled again if a custom configuration uses SSR or disables ray-traced GI.

### Denoising

In the normal view, the combine pass adds diffuse, specular, direct lighting and emission into `u_OutputColor`.
DLSS-RR receives that combined colour with depth, motion vectors and material guides. The separate
`u_RTGIOutput` and `u_SSRResult` buffers organize the shading passes; they are not separate radiance
inputs to DLSS-RR.

The reflection debug view writes only indirect specular to `u_OutputColor`, so DLSS-RR reconstructs
that isolated contribution instead.

Reflected features can move differently from the primary surface. This sample nevertheless copies
the primary screen motion vectors into `u_RRSpecMotionVectors` in
[`PrepareGuideBuffers.hlsl`](../../Rtxpt/Shaders/IntroSample/PrepareGuideBuffers.hlsl). That approximation
can leave reflections slow to settle under camera or object motion.

---

## What to look for

Start from the default scene and camera:

- **Compare Steps 2 and 3.** Watch the scooter's body, its chrome trim and the metal posts. Both presets
  use DLSS-RR, so the change comes from the reflection path. SSR's broad, bright fallback gives way to
  reflections of the surrounding geometry. Keep **Debug indirect specular** on
  while switching presets to isolate this comparison; it works with SSR too.
- **Inspect reflected lighting.** SSR re-shades its screen-space hit with IBL only. RT hits include
  shadow-tested analytic lighting, so reflected objects can show lighting and shadows absent in SSR.
- **Check the AO controls.** **Use RTAO** is disabled in Step 3 because neither indirect term uses it.
  Turn off RT reflections or RTGI and it becomes available again for the remaining environment
  fallback. Reselect Step 3 afterwards.
- **Watch reconstruction.** Move the camera slowly and watch reflected details settle. RT reflections
  can retain off-screen and occluded geometry, but they still have sampling noise, a grazing-angle
  cutoff and approximate motion guides.

---

## What is still missing

| Problem | Fixed in |
| - | - |
| Local lights still cast no shadows on the primary surface | [Step 4](step-4.md) |
| No additional diffuse bounces beyond the secondary hit | [Step 5](step-5.md) |
| Diffuse and specular indirect are two full-screen ray passes | [Step 6](step-6.md) |

Secondary-hit materials remain Lambertian. Emission suppression, sampling clamps and approximate
specular motion guides are also limitations; switching to ray tracing does not make this a full
path-traced reference.

---

## References

- Heitz. [*Sampling the GGX Distribution of Visible Normals.*](https://jcgt.org/published/0007/04/01/)
  JCGT 7(4), 2018 — the projected-disk method used by Donut's sampler.
- Heitz. *Understanding the Masking-Shadowing Function in Microfacet-Based BRDFs.* JCGT 3(2), 2014. —
  where G₂/G₁ comes from.
- Karis. *Real Shading in Unreal Engine 4.* SIGGRAPH 2013 Courses. — the split-sum model this step
  stops using.

---

[← Step 2: Ray-traced GI](step-2.md) · [Index](../intro-sample.md) · [Step 4: Stochastic direct lighting →](step-4.md)
