# The area-coverage algorithm

This is a coverage / anti-aliasing method for filling 2D vector shapes built from line and quadratic-Bézier contours. For each pixel, it evaluates the box-filtered winding integral in closed form from the curves themselves, without supersampling or baked coverage textures. For common single-edge pixels this gives exact area coverage; for more complex pixels involving winding cancellation or multiple overlapping winding levels, the result follows the same saturating coverage model used by production renderers, with documented limits.

It is designed to run as a per-fragment **gather** on the GPU: each pixel independently reads the curves
near it and evaluates one formula. That makes it drop into an ordinary painter's-order draw (no compute
pass, no prefix sum, no scene-wide scatter), while still producing the same mathematical result that
scanline signed-area rasterizers produce. Two pieces make that work: the **closed-form winding integral**
(§2–§4), and the **row-band structure** (§6) that keeps the gather bounded and — unlike a dual-ray method —
single-axis. Both are in this demo.

The whole method is the fragment function in [`../src/area.wgsl`](../src/area.wgsl); this document derives
it.

---

## 1. What we want

Let a shape be defined by directed contours and a fill rule (nonzero or even-odd). Its **winding number**
`w(x, y)` counts the signed crossings of a ray cast from `(x, y)`. The filled region is `{ w ≠ 0 }`
(nonzero) or `{ w odd }` (even-odd).

The ideal antialiased value of a pixel is the average of the shape's indicator over the pixel's footprint
— a **box filter**. Working in the shape's local coordinate space, the footprint of a device pixel is an
axis-aligned box

```
B = [xlo, xhi] × [ylo, yhi],   centered at the pixel center rc,   size s = (sx, sy)
```

where `s` is _units per pixel_ — the local-space size of one device pixel. Under pure scale/translation
`B` is exactly the pixel's preimage; `s` comes straight from the screen-space gradient of the local
coordinate (`dpdx`/`dpdy`), so the method is resolution-independent by construction.

We want, per pixel,

```
F = (1 / (sx·sy)) · ∫∫_B w(x, y) dA
```

and then a small fold from the winding integral `F` to a `[0,1]` coverage.

Most methods approximate this. This one computes it.

---

## 2. The master formula

The key step turns the _area_ integral of the winding number into a sum of _boundary_ integrals, one per
curve, using Green's theorem (`∫∫_R dA = ∮_∂R x dy` for signed area). Clipping that identity to the box
`B` and folding the box-edge terms into a clamp gives an exact decomposition over the curves `e`:

```
∫∫_B w dA  =  Σ_e A_e ,

A_e = ∫  ( clamp( x_e(t), xlo, xhi ) − xlo ) · y_e′(t)  dt          (integrated over t where y_e(t) ∈ [ylo, yhi])
```

Read the integrand as: at each height `y` the curve sits at some `x_e`, and the rightward ray from any box
point below `x_e` is crossed — so the horizontal extent that this curve _covers inside the box_ is
`clamp(x_e, xlo, xhi) − xlo`. Multiplying by `y_e′ dt` sweeps that extent up the curve with the correct
sign.

The single `clamp` is what makes this one formula instead of a pile of cases:

| where the curve is, relative to the box | `clamp(x_e, xlo, xhi) − xlo` | contribution                                    |
| --------------------------------------- | ---------------------------- | ----------------------------------------------- |
| fully **left** (`x_e ≤ xlo`)            | `0`                          | nothing                                         |
| fully **right** (`x_e ≥ xhi`)           | `xhi − xlo = sx`             | full box width × Δy — **pure interior winding** |
| **crossing** the box                    | partial                      | the exact boundary area — **the anti-aliasing** |

Interior winding and edge anti-aliasing are handled by the same term. There is no separate inside/outside test and no separate edge ramp: a curve to the right of the pixel contributes interior winding, a curve passing through the pixel contributes partial area, and the transition between those cases is continuous in the ideal formulation because the clamp is continuous.

This also reduces the number of geometric special cases. Grazing rays, vertices near scanlines, and nearly degenerate controls are handled through the same integral rather than through separate rasterization rules, although the implementation still depends on robust monotone splitting and finite-precision root evaluation.

---

## 3. The per-curve closed form

`A_e` has a closed form once each curve piece is **monotone in both x and y**. Monotonicity guarantees the
piece crosses each box edge at most once, so the set `{ t : y_e(t) ∈ [ylo, yhi] }` is a single interval and
each edge-crossing is a single root with a known branch.

A quadratic Bézier has at most one interior extremum per axis, so splitting it at those (≤ 2) parameters
yields **1–3 monotone pieces**. Straight segments are already monotone. We do this split once per unique
glyph on the CPU ([`../src/geometry.js`](../src/geometry.js)); the shader only ever sees monotone pieces.

For one monotone piece with endpoints `q1, q3` and control `q2` (all relative to the pixel center `rc`),
write `q(t) = q1 + a1·t + a2·t²` with `a1 = 2(q2−q1)`, `a2 = q1 − 2q2 + q3`.

**Step 1 — the y-window.** Solve `y(t)` for the window edges to get the sub-interval `[t_lo, t_hi]` where
the piece lies in the vertical band `[ylo, yhi]`. Because the piece is y-monotone, this is one root each,
with the branch fixed by whether the piece rises or falls, and saturated by the endpoints when the piece
starts past an edge or never reaches it (`mono_root` in the shader — one `sqrt`, no branch-count logic).

**Step 2 — the x-zones.** Along `[t_lo, t_hi]` the x-clamp splits the interval, at the (≤ 2) crossings of
`x = xlo` and `x = xhi`, into three zones in a statically known order (set by the x direction):

```
   x rising →      t_lo ─── LEFT ───┤ t_left ─── INSIDE ───┤ t_right ─── RIGHT ─── t_hi
   (mirrored when the piece runs the other way)
```

- **LEFT** (`x < xlo`): contributes `0`.
- **RIGHT** (`x > xhi`): contributes `sx · Δy`, with `Δy = Δt · y′(midpoint)` — exact, since `y′` is linear.
- **INSIDE**: the integrand `(x(t) + hx)·y′(t)` is a cubic in `t`. On a symmetric interval `t̄ ± δ` the odd
  powers cancel, so the **midpoint rule is exact**:

  ```
  A_inside = 2δ · X · Y′  +  (2δ³/3) · ( a2x·Y′ + 2·a2y·X′ )
  ```

  with `X = x(t̄) + hx` (box-local, so `0 ≤ X ≤ sx` — no large-magnitude cancellation), `X′ = x′(t̄)`,
  `Y′ = y′(t̄)`, `hx = sx/2`.

For each relevant monotone piece, the shader performs a small fixed amount of work: single-branch root solves for the y-window and x-zone boundaries, followed by a few multiply-adds for the closed-form area terms. Pieces whose monotone hull is fully left or fully right of the box can often skip the root solves using endpoint extent tests.

---

## 4. From winding integral to coverage

Normalize the signed box integral as:

```text
F = (Σ_e A_e) / (sx·sy)
```

The value `F` is the pixel-averaged winding number over the box. To turn that winding integral into a display coverage, the shader applies the usual fill-rule fold:

- **nonzero:** `coverage = min(|F|, 1)`
- **even-odd:** `coverage = tri(F)`, where `tri` is the period-2 triangle wave.

For ordinary edge pixels, where the box transitions between two adjacent winding levels, this gives the exact box-filtered coverage. For more complex pixels — for example, pixels containing opposite-sign winding cancellation or three or more winding levels at once — the fold is the standard saturating-area approximation rather than a full decomposition of each winding region inside the pixel.

This is the same practical coverage model used by production signed-area renderers (Skia, Vello, see [`COMPARISON.md`](COMPARISON.md)), and it is the right comparison target for most vector and glyph rendering workloads. The limitation is worth stating explicitly: the closed-form integral is exact for the averaged winding number, while the final coverage fold is exact only under the usual local-winding assumptions.

---

## 5. Numerical validation against a box filter

`deno task validate` compares the shader's coverage against two references, **neither of which is the
shader itself**:

- **box** — the box filter estimated by a zero-AA point sample: for each pixel, the fraction of a 24×24 grid
  of sub-sample points that fall inside the shape, inside/outside decided by ray-casting the winding against
  the raw curves (a different code path from the area integral, so agreement is independent).
- **skia** — [@napi-rs/canvas](https://www.npmjs.com/package/@napi-rs/canvas) (Skia), a mature independent
  rasterizer.

Measured on this machine (mean / worst-pixel `|Δcoverage|`):

```
shape                     ours vs box        skia vs box
                          mean      max      mean      max
rotated square 30°        0.00003   0.004    0.00079   0.092
circle r=44 (64 arcs)     0.00003   0.006    0.00073   0.177
glyph 'o' (with hole)     0.00006   0.014    0.00231   0.340
star {5/2} even-odd       0.00017   0.100    0.00111   0.090
```

On ordinary fills the shader matches the point-sampled box filter to within the reference's own sampling
noise (max ≲ 0.02) — two unrelated code paths agreeing that closely is meaningful independent evidence, not
a self-comparison. Skia, a production rasterizer, sits a little further from the point-sampled filter. Two
reasons, both about Skia: it **flattens curves to line segments** before rasterizing (so its deviation on a
circle shrinks as the arc count rises, while ours — evaluating the exact quadratics — does not; the test
circle is 64 arcs to keep this negligible), and its edge AA is its own model, not the box filter. The other
exception is the self-intersecting star, where the winding fold (§4) deviates from the true box filter by
~0.1 at the crossing points — the documented limit (see [COMPARISON.md](COMPARISON.md)), comparable to Skia
there.

This is numerical validation, not a proof: the closed-form derivation above is what establishes the
integral; the test confirms the implementation agrees with independent references on representative cases.

---

## 6. Banding: the bounded, single-axis gather

The math above says _how to weigh one curve at one pixel_. On its own that is a per-pixel loop over every
curve of the shape — fine for a single glyph, useless for a scene. What turns it into an algorithm — and
into something meaningfully different from both the scatter renderers and from a dual-ray method like Slug —
is the acceleration structure it runs inside. This demo implements it
([`../src/bands.js`](../src/bands.js) builds it; `area_of_face` in `area.wgsl` reads it).

Each shape's monotone pieces are filed into horizontal **row bands** over its y-extent (`~6` pieces per
band). A fragment maps its pixel's y-slab to the band range it touches and reads only those bands. Three
properties make this the right structure, and all of them fall out of the integral sweeping along `x`
inside a horizontal slab:

1. **Rows only — one band axis, not two.** A row band holds everything a fragment needs, because the
   integration direction is horizontal. There is no vertical ray and therefore no column bands. A dual-ray
   method (Slug) casts an axis-aligned ray on _each_ axis — it cannot see winding from a curve running
   parallel to its ray, so it needs a second ray and a second band structure. Removing the second ray
   removes the second band axis: **half the acceleration structure and half the per-fragment storage
   reads**.

2. **Window additivity — exact, no dedupe.** A piece is filed into every band its y-extent touches. A pixel
   whose footprint straddles a boundary reads each band _clipped to that band's own y-range_. The windows
   tile the slab, so a duplicated piece is integrated over **disjoint** y-windows; summing bands is exact to
   machine epsilon, with no "did I already count this?" test. Filing and lookup use the identical
   `floor((y − y0)·invH)` mapping, so they always agree. (`deno task validate` confirms the banded result
   equals the box filter — banding changes cost, never pixels.)

3. **Early break.** Pieces in a long band are sorted by hull x-max descending, so the shader stops the
   moment a curve is fully left of the pixel — every later one is further left and contributes exactly `0`.

Compared to the exact scatter renderers, this replaces the scanline prefix sum with a spatial lookup: the
"everything to the right" dependency is already handled analytically by the `clamp` (§2), so a band only has
to _find_ the nearby curves, not accumulate through them. No compute pass, no per-frame flatten, geometry
stays analytic at any zoom.

One precision detail worth calling out, because it is easy to get wrong: the fragment builds its
integration slab **rc-relative** — `[−sy/2, +sy/2]` — never as `rc.y ± sy/2` in absolute local space. As
you zoom in, the footprint `sy` shrinks as `~1/zoom`; past the point where `sy/2` drops below one ULP of the
local coordinate, an absolute slab that is later localized (`− rc.y`) round-trips through `rc.y` and
quantizes to **zero height**, so whole pixel rows integrate nothing — hard horizontal banding, constant
along each row. Working rc-relative keeps the slab height stable at any zoom and any distance from the origin
(the one absolute subtraction, `y0 − rc.y`, feeds only band selection and the band-boundary clips, whose ULP
wobble just nudges the split between adjacent windows — which still tile). Interiors then stay solid at any
zoom; the residual limit is that AA _edges_ wobble by ~`ULP(coordinate)·zoom` from the `curve − rc`
evaluation — see [COMPARISON.md](COMPARISON.md).

---

## 7. Map to the code

| concept                                                               | where                                                       |
| --------------------------------------------------------------------- | ----------------------------------------------------------- |
| split curves into xy-monotone pieces                                  | [`src/geometry.js`](../src/geometry.js)                     |
| file pieces into row bands, build the deduped atlas (§6)              | [`src/bands.js`](../src/bands.js)                           |
| `mono_root` — single-branch monotone quadratic solve                  | [`src/area.wgsl`](../src/area.wgsl)                         |
| `area_inside` — exact midpoint rule for the INSIDE zone               | `src/area.wgsl`                                             |
| `area_piece` — the LEFT / INSIDE / RIGHT zone split                   | `src/area.wgsl`                                             |
| `area_accum` — sum `A_e` over one band's pieces, with the early break | `src/area.wgsl`                                             |
| `area_of_face` — select + read the row bands a pixel touches (§6)     | `src/area.wgsl`                                             |
| `fs` — normalize `F` and fold (nonzero / even-odd)                    | `src/area.wgsl`                                             |
| instanced quad + per-glyph band table                                 | `src/area.wgsl` (`vs`), [`src/layout.js`](../src/layout.js) |
