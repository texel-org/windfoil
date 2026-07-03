# How it compares

Where Windfoil ([`ALGORITHM.md`](ALGORITHM.md)) sits among established methods, along two
axes: **exactness** (true box filter or an approximation?) and **execution model** — a per-pixel **gather**
(each fragment reads the curves near it), a **scatter** (edges accumulate into a buffer, resolved by a scan),
or a **bake** (an offline texture sampled at draw time). A gather composes with painter's-order drawing and
needs no compute pass; a scatter is exact but needs a prefix sum and a per-frame flatten; a bake is fixed at
its baked resolution. This method evaluates a closed-form signed-area / winding integral in a gather, bounded
by a **row-band** structure with only _one_ band axis because the integral sweeps along one axis.

---

## The families

### Exact signed-area accumulation — font-rs, Pathfinder, Vello

_font-rs_, _Pathfinder_, and _Vello_ accumulate each edge's signed area into a coverage buffer; the winding at
a pixel is a running sum along the scanline, giving the exact box filter — the correctness gold standard, very
fast on a compute pipeline. But the winding at `x` depends on every edge to its right, resolved by a **prefix
sum**: a scatter that wants a compute pass and a per-frame flatten, and does not compose with immediate-mode
draws. This method uses the same signed-area math as a gather — the "everything to the right" dependency folds
into the `clamp(x, xlo, xhi)` term, so a pixel needs only the curves that reach it, with no accumulation
buffer, prefix sum, or flatten.

### Dual-ray analytic coverage — Slug (Lengyel)

_Slug_ also keeps curves analytic and gathers per fragment. Its anti-aliasing casts a horizontal **and** a
vertical ray per curve and blends 1-pixel ramps from each — an approximation of the box filter, not the
filter. The second ray exists because one axis-aligned ray can't see winding from a curve parallel to it, and
it doubles the per-curve solve and the acceleration structure. Here, interior winding and boundary AA fall out
of **one** continuous integral (§2), so there is no per-axis blend and no second ray.

### Implicit quadratics — Loop–Blinn

Loop & Blinn (2005) render each quadratic with a per-triangle `u² − v` sign test — analytic, but a fill
primitive rather than a coverage method: it needs triangulation, anti-aliases only along the tested edge, and
resolves overlapping winding with stencil passes that can reintroduce conflation at shared edges. Windfoil
fills by winding directly, and adjacent faces sharing a curve sum to full coverage.

### Analytic prefiltering & vector textures — Manson–Schaefer, Nehab–Hoppe, Ganacim et al.

These compute exact or prefiltered coverage but not in an immediate-mode gather: Manson–Schaefer's convolution
is scatter-shaped, Nehab–Hoppe's vector textures use a lattice with an approximate in-cell test, and Ganacim
et al. (2014) build a per-frame tree. The row bands here are a lattice in the same spirit, but the in-cell test
is the _exact_ integral and nothing is rebuilt per frame.

### Baked coverage / SDF atlases — MSDF and friends

MSDF and coverage caches sample a precomputed field — cheap and great for UI text, but fixed at the baked
resolution, so strong magnification shows the approximation (rounded corners, softened thin features). Windfoil
bakes nothing; it evaluates coverage at the device resolution every frame.

### Supersampling / MSAA

Brute-force ground truth, and the basis for `deno task validate`'s box-filter reference. Correct in the limit
but costs N× the samples; Windfoil reaches the same answer with a single sample.

---

## Summary

| method                       | exact box filter?   | model                  | per-frame flatten? | analytic at any zoom? |
| ---------------------------- | ------------------- | ---------------------- | ------------------ | --------------------- |
| font-rs / Pathfinder / Vello | yes                 | scatter (+ prefix sum) | yes                | yes                   |
| Slug (dual-ray)              | approximate (blend) | gather                 | no                 | yes                   |
| Loop–Blinn                   | edge-only           | gather (+ stencil)     | no                 | yes                   |
| Nehab–Hoppe / Ganacim        | approximate / tree  | gather / hybrid        | (tree)             | partial               |
| MSDF / coverage atlas        | baked resolution    | bake                   | no                 | no                    |
| MSAA / supersampling         | in the limit        | gather (N×)            | no                 | yes                   |
| **windfoil (this)**          | **yes** (§4 limits) | **gather**             | **no**             | **yes**               |

The aim is the exactness of the accumulation renderers with the execution model of the analytic per-fragment
ones, in a single gather. Whether that combination is genuinely new I don't claim — see the note in the
[README](../README.md).

---

## Honest weaknesses

No method is free; these are the real limits.

- **The winding fold.** `min(|F|,1)` and `tri(F)` are exact when a pixel spans at most two adjacent winding
  levels. In a pixel with opposite-sign winding cancellation or three-plus winding levels at once — a
  self-intersection, or extreme minification of a self-overlapping shape — the fold deviates. Skia and Vello
  make the same trade.
- **Rotated / sheared transforms.** The integration box is axis-aligned in local space, so under a
  non-axis-aligned transform it is the pixel's _local_ preimage, not the device pixel — an approximation shared
  by every local-space analytic method (including Slug).
- **Deep-zoom float precision (edges only).** Interiors stay solid at any zoom (the slab is rc-relative, so it
  can't collapse to a zero-height row); what remains is AA _edge_ positions wobbling by ~`ULP(coordinate) ×
  zoom` pixels from the `curve − rc` cancellation far from the origin. The fix is de Casteljau localization
  near the pixel (or `f64`).
- **Cost.** A pixel crossed by a curve does up to a few `sqrt`s versus a fixed one or two for ramp methods. Far
  curves are cheaper (compares, not solves) and the single-axis bands halve storage traffic — but on an
  ALU-bound workload the exact integral does more arithmetic than a heuristic, by design.
