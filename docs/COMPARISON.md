# How it compares

The area-coverage algorithm ([`ALGORITHM.md`](ALGORITHM.md)) is best understood by where it sits among the
established families of vector-coverage methods. Two axes organize the space:

- **Exactness** — does the method compute the true box filter, or an approximation of it?
- **Execution model** — is coverage a per-pixel **gather** (each fragment reads the curves near it and is
  done), a **scatter** (edges deposit into an accumulation buffer, resolved by a scan), or a **bake** (an
  offline texture/atlas sampled at draw time)?

A gather composes with ordinary painter's-order drawing and needs no compute pass or scene-wide buffer; a
scatter is exact and cheap per edge but needs a prefix sum and a per-frame flatten; a bake is fast to
sample but fixed at its baked resolution. This method's contribution is a closed-form signed-area / winding-integral evaluation in a gather-shaped kernel — and keeping that kernel bounded by a **row-band** structure that, because the
area integral sweeps along one axis, needs only _one_ band axis where a dual-ray gather needs two
(see [ALGORITHM.md §6](ALGORITHM.md)). Exactness and execution model are separable choices, and this is the
combination that had been missing.

---

## The families

### Exact signed-area accumulation — font-rs, Pathfinder, Vello

The lineage of Raph Levien's _font-rs_, Patrick Walton's _Pathfinder_, and _Vello_ accumulates each edge's
signed trapezoid area into a coverage buffer; the winding at a pixel is a running sum along the scanline,
and the result is the exact box filter. This is the gold standard for correctness and is extremely fast on
a compute-capable pipeline.

The catch is structural: the winding at `x` depends on every edge to its right, which a scanline resolves
with a **prefix sum** — a scatter. It wants a compute pass and a per-frame flatten of curves into edges,
and it does not naturally compose with immediate-mode, painter's-order draws (each draw would need its own
accumulation target).

**This method is the same signed-area mathematics** (Green's theorem, the exact trapezoid/curve integral)
**re-derived as a per-pixel gather**: the "everything to the right" dependency is folded into the
`clamp(x, xlo, xhi)` term, so a pixel needs only the curves that reach it, evaluated analytically — no
accumulation buffer, no prefix sum, no flatten, curves stay curves at any zoom.

### Dual-ray analytic coverage — Slug (Lengyel)

Eric Lengyel's _Slug_ keeps curves analytic and evaluates coverage per fragment (a gather, GPU-native,
resolution-independent — the same virtues as here). But its anti-aliasing casts a horizontal **and** a
vertical ray per curve, turns each crossing into a 1-pixel linear ramp, and blends the two with a
proximity heuristic, recovering interior winding with a separate `min`/`max`. That blend is a _proxy_ for
the box filter, not the filter: it deviates on diagonal edges, thin
features, high curvature, and minified shapes. The second ray exists only because a single axis-aligned ray
cannot see winding when a curve runs parallel to it, and it doubles the per-curve solve and the
acceleration structure.

The area method removes the proxy. Interior winding and boundary AA fall out of **one** continuous integral
(§2 of the algorithm doc), so there is no per-axis blend and no second ray — a curve's contribution carries
a `y′` weight that fades a grazing crossing in smoothly instead of flipping a crossing count. One ray's
worth of structure becomes zero rays; the discontinuous crossing-count bookkeeping disappears.

### Implicit quadratics — Loop–Blinn

Loop & Blinn (2005) render each quadratic with a per-triangle `u² − v` sign test in the fragment shader.
It's elegant and analytic, but it is a _fill primitive_, not a coverage method: it needs interior
triangulation, gives anti-aliasing only along the tested edge (via the gradient), and handles overlapping
winding and shared edges with stencil passes — which reintroduces conflation artifacts at the seams between
adjacent faces. The area method fills by winding directly and is conflation-free (adjacent faces sharing a
curve sum to exactly full coverage).

### Analytic prefiltering & vector textures — Manson–Schaefer, Nehab–Hoppe, Ganacim et al.

These compute exact prefiltered coverage but in shapes that don't fit an immediate-mode gather: Manson &
Schaefer's polynomial-filter convolution is scatter-shaped; Nehab & Hoppe's vector textures specialize to a
lattice with an approximate in-cell test; Ganacim et al. (2014) build a per-frame acceleration tree. The
area method's row bands are a lattice in the same spirit, but the in-cell test is the _exact_ integral
rather than an approximation, and nothing is rebuilt per frame.

### Baked coverage / SDF atlases — MSDF and friends

Distance-field atlases (e.g. Chlumský's MSDF) and coverage-texture caches sample a precomputed field. They
are cheap and great for UI text, but coverage is fixed at the baked resolution: under strong magnification
or at print DPI the field's approximation shows (rounded corners, softened thin features), and arbitrary
per-shape transforms are not the true box filter. The area method bakes nothing — it evaluates coverage
analytically at the device resolution every frame.

### Supersampling / MSAA

Brute-force ground truth. `deno task validate` uses a zero-AA point sample of the raw curves as its
independent box-filter reference (alongside Skia). Correct in the limit but costs N× the samples, and MSAA
doesn't help interior coverage of complex overlaps. The area method reaches the same answer with a single
sample.

---

## Summary

| method                       | exact box filter?     | model                  | per-frame flatten? | analytic at any zoom? | conflation-free fills?                                    |
| ---------------------------- | --------------------- | ---------------------- | ------------------ | --------------------- | --------------------------------------------------------- |
| font-rs / Pathfinder / Vello | ✅                    | scatter (+ prefix sum) | yes                | ✅                    | ❓ [(#49)](https://github.com/linebender/vello/issues/49) |
| Slug (dual-ray)              | ❌ (heuristic blend)  | gather                 | no                 | ✅                    | ✅                                                        |
| Loop–Blinn                   | ⚠️ edge-only          | gather (+ stencil)     | no                 | ✅                    | ❌ (seams)                                                |
| Nehab–Hoppe / Ganacim        | ⚠️ approximate / tree | gather / hybrid        | (tree)             | ⚠️                    | ⚠️                                                        |
| MSDF / coverage atlas        | ❌ (baked res.)       | bake                   | no                 | ❌                    | ⚠️                                                        |
| MSAA / supersampling         | ✅ in the limit       | gather (N×)            | no                 | ✅                    | ✅                                                        |
| **area coverage (this)**     | ✅                    | **gather**             | **no**             | ✅                    | ✅                                                        |

The row that had every check — exact, gather, no flatten, analytic, conflation-free — did not exist before.
That combination is the point: the exactness of the accumulation renderers with the execution model of the
analytic per-fragment ones.

---

## Honest weaknesses

No method is free; these are the area method's real limits.

- **The winding fold.** `min(|F|,1)` (nonzero) and `tri(F)` (even-odd) are exact when a pixel spans at most
  two adjacent winding levels. Inside a pixel that contains **opposite-sign winding cancellation** or
  **three or more winding levels at once** — a self-intersection point, or extreme minification of a
  self-overlapping shape — the fold deviates. This is the same trade Skia and Vello make (they use the same
  saturating-area model). A second-moment disambiguator would fix it but costs O(n²) per pixel — not worth
  it.
- **Rotated / sheared transforms.** The integration box is axis-aligned in the shape's local space. Under a
  non-axis-aligned transform that box is the pixel's _local_ preimage, not the device pixel — an
  approximation shared by every local-space analytic method (including Slug). It stays far closer to the
  true filter than a heuristic blend, but it is not the exact device-box filter for rotated draws.
- **Deep-zoom float precision (edges only).** Interiors stay solid at any zoom: the integration slab is
  built rc-relative (`[−sy/2, +sy/2]`), so it never round-trips through a large coordinate and can't collapse
  to a zero-height row (a naïve absolute slab quantizes to zero on far-from-origin content at deep zoom).
  What remains is that AA _edge_ positions wobble by ~`ULP(coordinate) × zoom` pixels, from the `curve − rc`
  cancellation when evaluating a piece far from the origin: interiors stay solid, but zoom far enough on
  far-origin content and the edges themselves
  get slightly wavy. The escape hatch is de Casteljau localization of the pieces near the pixel (or `f64`).
- **Cost vs a proxy.** A pixel crossed by a curve does up to a few `sqrt`s (the y-window and x-edge solves),
  versus a fixed one or two for the ramp-based methods. Far curves are cheaper (compares, not solves), and
  the single-axis band structure halves the storage traffic — but on a purely ALU-bound workload the exact
  integral is doing genuinely more arithmetic than a heuristic, by design.
