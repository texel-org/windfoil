# Bounding the cost under minification

> **Status: analysis + implementation plan, not yet in the code.** This note documents a real cost
> characteristic of the gather ([`ALGORITHM.md`](ALGORITHM.md) §6) and a fix that stays exact. The demo as
> shipped does *not* yet implement it.

The demo renders "area coverage" at a ladder of sizes from a few pixels up to 200 px
([`../src/main.js`](../src/main.js)). Framed so the whole ladder is visible — or in any interactive viewer
zoomed out so glyphs shrink to specks — the fragment stage gets **more** expensive, not less. This note
explains why that is inherent to a per-pixel gather, and how to bound it without giving up the exact box
filter.

The short version: **the per-pixel cost should scale with the curves *crossing* the pixel box, not the
curves *inside* it.** The current shader pays for every curve inside; under minification that's every curve
of the glyph. The interior is precomputable, so it doesn't have to.

---

## 1. Why minified is the worst case

Per-pixel cost is governed by the footprint `s` = units-per-pixel (`area.wgsl` `fs`), which is
`≈ unitsPerEm / onScreenPx`. A small on-screen glyph ⇒ a **large** footprint box in em-space, and a large
box defeats all three of the gather's accelerations at once:

1. **Band selection stops narrowing.** `area_of_face` maps the pixel's y-slab to a band range `ri0..ri1`.
   When the footprint is taller than the glyph, that range spans *every* band, so the pixel reads all of
   them.
2. **The early break and the fully-right shortcut stop firing.** In `area_accum`, `hx = sx/2` is huge, so no
   piece is ever "fully left" (`x_hull_max <= -hx`, the `break`) or "fully right"
   (`x_hull_min >= hx`, the cheap winding-only add). Almost every piece falls through to the full
   `area_piece` closed form — up to four `mono_root` solves, each a potential `sqrt`.
3. **The band structure inverts from asset to liability.** A piece is duplicated into every band its
   y-extent touches ([`../src/bands.js`](../src/bands.js)). Magnified, a pixel touches one band and pays for
   its ~6 pieces. Minified, it touches *all* bands and pays the full **banded** count — the dup-inflated
   one.

Measured on this demo's atlas (`deno task render`):

```
7 unique glyphs → 319 monotone pieces in 56 row bands (465 banded, 1.46× dup)
```

So a minified pixel evaluates ~66 pieces/glyph (465 / 7) on the full `sqrt` path, versus ~6 (mostly
early-outs) when magnified — a 10–20× per-pixel swing, and the extra 1.46× is overhead the band structure
*adds* under minification. A flat per-glyph curve list would actually be cheaper in this regime.

The cost is highest where the footprint just covers the glyph (roughly on-screen height ≲ 1–2 band-heights).
With a fixed viewport, zooming out holds the pixel count constant while pushing every fragment onto this
worst-case branch — which is exactly the "poor performance fully zoomed out" symptom. Note the *extreme*
(everything sub-pixel) is cheap again, because the covered-pixel count collapses; the pain is the
intermediate band where glyphs are small **and** still cover real screen area.

This is inherent to an analytic per-pixel gather with **no LOD**: it re-integrates the exact geometry no
matter how few pixels the glyph occupies.

---

## 2. The interior is precomputable

The fix falls out of the master formula ([`ALGORITHM.md`](ALGORITHM.md) §2):

```
A_e = ∫ ( clamp(x_e(t), xlo, xhi) − xlo ) · y_e′(t) dt
```

The `clamp` only differs from the identity for curves that cross the box's **x-edges**. Every curve
strictly *inside* the box in x contributes a plain area moment `∫ x_e·y′ dt − xlo·∫ y′ dt` — no clamp, no
per-pixel root solve. And that moment **telescopes**: it can be precomputed at any granularity and summed.

**The asymptote is a single constant.** When the box contains the whole glyph, every curve is x-interior and
the full contours are integrated in y, so `xlo·Σ∫y′dt = 0` (closed contours return to their start) and

```
Σ A_e = Σ ∫ x_e·y′ dt = signed area of the glyph          ← one precomputed scalar
```

`F = A_glyph / (sx·sy)`, coverage `= min(F, 1)`. Because a contained glyph has `A_glyph ≤ box area`, we have
`F ≤ 1`, so `min(|F|,1)` and `tri(F)` **both** reduce to `F` — bit-for-bit what the loop returns (and more
stable, since it skips summing hundreds of signed terms).

**The middle regime hoists into per-band moments.** One level down, precompute per band:

- `S[i] = Σ ∫_bandᵢ (x − xref)·y′ dt` — the band's x-area moment, relative to a per-glyph `xref`
- `D[i] = Σ ∫_bandᵢ y′ dt` — net Δy through the band (the telescoping form already used for fully-right
  pieces in `area_accum`)

Then, when the box contains the glyph in **x**, any band the footprint covers *fully in y* contributes
`S[i] + (xref − rc.x + hx)·D[i]` in O(1); only the ≤2 bands the footprint *edge* cuts through run the real
integral. It is one analytic hierarchy:

```
piece  →  band (S, D)  →  glyph (Σ S = A_glyph)
```

The fragment descends only as far as the box boundary actually cuts. Under full minification every band is
covered → zero real integration → cost O(bands). The blowup is gone, and it is **exact** for both fill rules
— the moment is the same integral pre-summed, not an approximation. Using `xref` = the glyph's bbox `loX`
keeps `S` combined with `rc.x` free of large-magnitude cancellation.

The framing worth keeping: **interior area is precomputable at any granularity; only boundary-crossing
curves need per-pixel work** — the same reason a summed-area table answers box queries in O(1), done
analytically through Green's theorem instead of through samples.

---

## 3. What changes in the code

Blast radius is small: **one new storage buffer + one branch in `area_of_face`**, and the magnified path is
untouched (its `xContained` gate is `false` for a 1-px box over a large glyph).

### Data model

One `vec2` per band (`S[i]`, `D[i]`), parallel to the existing `rows` table. No per-instance change and no
`layout.js` change — `xref`, the glyph x-extent, and the band header are all already in the `Instance`
struct (`I.bbox`, `I.band`).

### `bands.js` — compute the moments during filing

In `bandPieces`, after bucketing, sum this closed form over each band's pieces (band `b` spans
`[y0 + b/invH, y0 + (b+1)/invH]`) and push into a new `bandData` array alongside `rowOut.push(start, count)`:

```js
// piece = [q1x,q1y, q2x,q2y, q3x,q3y]; integrate (x−xref)·y′ over t where y ∈ [b0,b1]
function bandMoment(piece, b0, b1, xref) {
  const [x1, y1, x2, y2, x3, y3] = piece;
  const a1x = 2 * (x2 - x1), a2x = x1 - 2 * x2 + x3;
  const a1y = 2 * (y2 - y1), a2y = y1 - 2 * y2 + y3;
  const [ta, tb] = yWindow(y1, a1y, a2y, b0, b1);   // monotone quadratic solve, saturating to [0,1]
  if (tb <= ta) return [0, 0];
  const yAt = (t) => y1 + a1y * t + a2y * t * t;
  const X0 = x1 - xref, X1 = a1x, X2 = a2x, Y0 = a1y, Y1 = 2 * a2y;
  const c0 = X0 * Y0, c1 = X0 * Y1 + X1 * Y0, c2 = X1 * Y1 + X2 * Y0, c3 = X2 * Y1; // X·y′ cubic
  const I = (t) => ((((c3 / 4) * t + c2 / 3) * t + c1 / 2) * t + c0) * t;
  return [I(tb) - I(ta), yAt(tb) - yAt(ta)];         // [S contribution, D contribution]
}
```

`yWindow` is the CPU twin of the shader's `mono_root`; drop it in `geometry.js`. `D` telescopes to 0 over
each closed contour, so `Σ S = A_glyph` falls out for free.

### `area.wgsl` — one binding, one branch

```wgsl
@group(0) @binding(4) var<storage, read> bandData : array<vec2<f32>>;  // (S, D) per band
```

In `area_of_face`, compute the gate once and swap the band-loop body:

```wgsl
let hx = s.x * 0.5;
let xContained = I.bbox.x >= rc.x - hx && I.bbox.z <= rc.x + hx;   // box spans the glyph's x-extent?
let momCoef = I.bbox.x - rc.x + hx;                               // (xref − rc.x + hx) · D
for (var ri = ri0; ri <= ri1; ri = ri + 1u) {
  let bandLo = dy0 + f32(ri) / invH;
  let bandHi = dy0 + (f32(ri) + 1.0) / invH;
  let w_lo = max(-sy2, bandLo);
  let w_hi = min(sy2, bandHi);
  if (w_hi <= w_lo) { continue; }
  if (invH > 0.0 && xContained && bandLo >= -sy2 && bandHi <= sy2) {
    let m = bandData[rowBase + ri];          // band wholly inside the footprint AND x-contained
    f_int += m.x + momCoef * m.y;            // ∫(x+hx)·y′ over the whole band, O(1)
  } else {
    let rIdx = (rowBase + ri) * 2u;
    f_int += area_accum(rows[rIdx], rows[rIdx + 1u], rc, w_lo, w_hi, s.x);   // unchanged real path
  }
}
```

The algebra: `∫(x+hx)y′ = ∫(x−xref)y′ + (xref−rc.x+hx)∫y′ = S[i] + momCoef·D[i]`. When `xContained`, every
curve is inside the box in x, so `area_piece` would compute exactly this.

### `gpu.js` / `main.js` — thread one buffer

Mechanical: create `bandBuf = storage(device, bandData)`, add `{ binding: 4, resource: { buffer: bandBuf } }`
to the bind group, and forward `bandData` through `renderToRGBA`. The pipeline layout is `'auto'`, so it
picks up binding 4 from the shader — no explicit layout to edit.

### Optional hard guard (provable O(1))

The moment path is exact but only fires when the box contains the glyph *in x*. For a provable worst-case
ceiling regardless of shape, add a per-instance `A_glyph = Σ S[i]` (one float; `Instance` → 5×`vec4`, one
push in `layout.js`) and short-circuit before the loop when the box contains the glyph in **both** axes:

```wgsl
if (I.bbox.x >= rc.x - hx && I.bbox.z <= rc.x + hx &&
    I.bbox.y >= rc.y - s.y * 0.5 && I.bbox.w <= rc.y + s.y * 0.5) {
  return A_glyph;   // box ⊇ glyph → exact, O(1), skip everything
}
```

Ship the moment path first; add `A_glyph` only if profiling shows a residual (tall-narrow glyphs minified in
y but not yet contained in x still loop their pieces).

### Validation

The moment path is the same integral, so `deno task validate` should stay **pixel-identical** — the same
claim §6.2 already makes ("banding changes cost, never pixels"). Extend `tools/validate.js` with a
deliberately minified row and assert `|Δcoverage|` against the point-sampled box filter is at the noise
floor, and that the moment path matches the loop bit-for-bit, before trusting the fast path.

---

## 4. Why Slug doesn't hit this — and can't apply the fix

Slug ([COMPARISON.md](COMPARISON.md) §"Dual-ray analytic coverage") is also a per-pixel gather, but its cost
is roughly **zoom-invariant**: each ray reads the single band its center-line lands in — a fixed handful of
curves — regardless of footprint size, because Slug never forms a 2D footprint. So it has no minification
cost cliff.

It pays for that in *quality*, not milliseconds. Sampling one line per axis **point-samples the cross-axis**,
so under minification it misses sub-pixel detail between the two sample lines — shimmer and dropped thin
features, which is why Slug pipelines fall back to a baked atlas/SDF for small text. Its answer to
minification is "switch renderers"; this method's problem is purely performance, because it is still
computing the exact box filter.

The crossover is clean and goes both ways:

| | area-coverage | Slug |
| --- | --- | --- |
| normal / magnified | **cheaper** — one band axis, thin slab, ~6 curves | ~12 curves (two axes) |
| heavy minification | ~all banded curves (~66) — *before this fix* | **cheaper** — still ~12, flat |

The §2 fix is something Slug **structurally cannot do**: you can only hoist an interior integral out of the
per-pixel loop if you form one. Slug re-samples two lines per pixel — there is no interior term to
precompute, so it has no exact way to get both cheaper and correct. This method can keep the exact 2D box
filter *and* bound the minified cost, ending up better than Slug at both ends.

---

## 5. Honest caveats

- **The x-containment gate.** The exact moment path only fires when the box spans the glyph's x-extent. For
  a glyph minified in y but not yet contained in x (tall-narrow letters, or the left/right edge columns of
  any minified glyph), the pixel still loops its pieces. That's the perimeter, few pixels — but it means the
  fix bounds the *dominant* case (uniform-scale specks, contained both ways), not literally every pixel,
  unless the optional `A_glyph` guard is added.
- **Precision regime.** The moment path runs under zoom-*out*, where the deep-zoom `curve − rc` cancellation
  of §6 does not apply. Keeping `S` relative to `xref` avoids the one large-magnitude subtraction that
  `area_inside` deliberately avoids with its box-local `+hx`.
- **Not the winding fold.** This bounds *cost*; it does not touch the winding-fold limit
  ([COMPARISON.md](COMPARISON.md) "Honest weaknesses") on pixels with opposite-sign cancellation or 3+
  winding levels. A sub-pixel self-overlapping shape is still that documented approximation — the moment
  path just reaches the same answer faster.
