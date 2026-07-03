# Notes

## Curve Normals

Free-ish to compute tangent-space normals of the glyph outlines, to create effects like emboss/deboss. Rides the coverage gather: by Stokes, the gradient of the coverage integral _is_ the outline normal:

```
∇ ∫∫_box w dA  =  ∮_{outline ∩ box} n ds
```

For one xy-monotone quadratic piece, `n ds = (y', −x') dt`, so over its in-box sub-interval `[t1, t2]` (the
INSIDE zone `integrate_piece` already solves) the integral telescopes to **the chord, rotated 90°**: `(Δy, −Δx)`.
Accumulate that alongside the area, gated on a `want_grad` uniform flag so the plain fill path pays nothing.

### Example

```
# per piece, inside the same loop that sums coverage:
p1, p2 = curve(t1), curve(t2)      # endpoints of the in-box sub-curve (already computed for the area)
grad  += vec2(p2.y - p1.y, p1.x - p2.x)   # rotated chord = ∫ n ds

# fragment, once the gather returns (f_int, grad):
slope = grad / avg(s)                       # s = pixel footprint; slope ~1 at a full edge, →0 in flat regions
n     = normalize(vec3(TILT*slope.x, -TILT*slope.y, 1))   # TILT ≈ 1.5, +Y-up (font space is Y-down)
body  = coverage(f_int)                      # the usual even-odd / nonzero fold
alpha = max(body, length(slope))             # glyph fill OR the ~1px edge rim
out   = vec4((n*0.5 + 0.5) * alpha, alpha)   # premultiplied normal-in-RGB
```

To restore: re-add a `flags`/view-mode uniform, thread `want_grad` through `integrate_piece`/`integrate_band`/
`integrate_face` (returning `grad` next to the scalar area), and branch on it in `fs` as above.

## Band Moments

Minified, a pixel's footprint spans whole row bands, so `integrate_face` integrates every curve of the glyph
per pixel — its worst case. But a band's _interior_ is precomputable: by Green's theorem, for a band the
footprint fully covers in y with the glyph inside the box in x, that band's contribution is a constant plus a
term linear in `rc.x`. Precompute per band (`xref` = glyph bbox `loX`):

```
S[i] = Σ ∫_band (x − xref)·y′ dt     # area moment
D[i] = Σ ∫_band y′ dt                 # net Δy (the telescoping clamp form integrate_band already uses)
```

A fully-covered band then contributes `S[i] + (xref − rc.x + hx)·D[i]` in O(1) — exact, since
`∫(x+hx)·y′ = S + (xref − rc.x + hx)·D`. Only the ≤2 bands the slab's edge cuts run the real integral. It
fires only where the footprint ≥ one band (glyph ≲ `bandCount` px); larger glyphs read finer than a band and
skip it, so it accelerates **small/medium sizes only**. `Σ S = A_glyph`, `Σ D = 0`, so the all-covered case
collapses to the sub-pixel area constant.

### Example

```
# bands.js — per band [b0,b1] while filing (one vec2 per band, parallel to `rows`):
for piece in band:
  [ta,tb] = t where y(t) ∈ [b0,b1]           # monotone solve, saturating (CPU mono_root)
  S += ∫_{ta}^{tb} (x(t) − xref)·y′(t) dt     # closed-form cubic integral
  D += y(tb) − y(ta)

# integrate_face — per band ri the slab touches:
if xContained && bandLo >= -sy2 && bandHi <= sy2:   # s.x ⊇ glyph x-extent; band wholly in slab
  f_int += S[ri] + (xref - rc.x + hx) * D[ri]
else:
  f_int += integrate_band(...)                          # unchanged real path
```

Wire-up: one `bandData : array<vec2<f32>>` storage buffer (binding 4), built in `bands.js`, bound in
`gpu.js`; the magnified path is untouched (`xContained` is false for a 1px box). Cost: 2 floats/band.

## Backdrop

Complex scenes stack many shapes, and a pixel buried inside them shouldn't scan curves it can't see. The
winding integral already splits geometry the way a backdrop needs: §2's `clamp` collapses every edge **fully
right of the box** to `sx·Δy`, a shape-independent term — and summed over edges that also **span the box in y**
it telescopes to `sx·sy·(integer winding)`. Two properties make the backdrop fall out: (1) everything outside a
cell reduces to one signed **integer** winding, not a curve scan; (2) because the integral is a linear sum over
edges (§6), that integer is constant across the cell and composes across stacked shapes — so it is just an
additive offset.

Tile the scene into cells; file into each cell only the curves whose hull enters it, and precompute `W` = net
winding of everything fully right of and y-spanning the cell. A fragment integrates its cell's short list and
adds the constant:

```
F = W + integrate_cell(local curves) / (sx·sy)      # then the usual nonzero / even-odd fold
```

`W` is exact integer winding, so coverage is unchanged bit-for-bit — the win is purely that far geometry costs
zero fragment work: a pixel inside a hundred nested contours scans the few curves crossing its cell, not all
hundred. Cells are the scene-level analogue of the row bands — bands bound the gather in y within a glyph, cells
bound it in xy across shapes.

Wire-up: a coarse cell grid carrying a per-cell `[curveStart, count]` + integer `W` (CPU prepass or a binning
pass), read in `integrate_face` before the band loop; one add on the fragment path.

## Box Blur

The gather already computes an exact box filter over the pixel footprint `s` (units/px, from `dpdx`/`dpdy`), and
a box filter over a **wider** box _is_ a box blur. So scaling `s` up before the single gather blurs the glyph
with no second sample, no SDF, same code path: `k` device px wide → a `k`px blur, edges ramp over `k`px and
sub-`k` stems dim toward the ink average. Consistent px width at any zoom (`s` is per-pixel), and tiny glyphs
still fall through the minification guard as a blurred smudge.

Cost scales with the box: a wider `s.y` selects more row bands, a wider `s.x` breaks later in each band. The
cheap directional variant widens **only** `s.x` — same bands (selection is y-only), just a few more pieces before
the early break — a horizontal smear at near-zero structural cost.

To restore: `let sEff = s * blur;` (`blur` = diameter in device px, 1 = off) fed to the minification guard and
`integrate_face` in place of `s`; gate `blur` behind a uniform so the fill path stays `sEff == s` bit-for-bit.
