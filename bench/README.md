# windfoil vs Slug — Deno WebGPU benchmark

Per-frame GPU time for **windfoil** (this repo's single-ray winding integral) versus the **Slug algorithm**
(Eric Lengyel's dual-ray analytic coverage, [reference](https://github.com/EricLengyel/Slug)), swept across a
ladder of zoom levels. The two methods degrade differently with zoom, so a single number hides the story — the
sweep is the point.

```sh
deno run --unstable-webgpu -A bench/main.js                      # both scenes: text grid + complex shape
deno run --unstable-webgpu -A bench/main.js --scene glyphs --check   # one scene + correctness/quality PNGs
deno run --unstable-webgpu -A bench/main.js --scene shape --shape-fill evenodd
deno run --unstable-webgpu -A bench/main.js --levels 1,1.5,2,4,16,64,256   # custom ladder (auto-densifies)
```

Three scenes (`--scene glyphs,shape,tiger` in any combination, or `all` / `both`; default `all`):

- **glyphs** — a dense grid of real text: sparse curves, ~1 edge per pixel. Slug's sweet spot.
- **shape** — one self-crossing shape of ~240 whole quadratics (a starburst of rotated overlapping ellipses)
  that all span the extent and pile into a high-winding core: many edges per pixel, bands packed with FAR
  curves. windfoil's intended regime.
- **tiger** — an actual SVG drawing (the Ghostscript tiger, `fixtures/tiger-quadratics.json`): 304 overlapping
  shapes / ~14k quadratics, real painter's-order **overdraw** (many shapes stack over the same pixels). Each
  shape is one instance with its own bands; the drawing is tiled to fill the viewport.

Flags: `--size` (offscreen square, 720), `--levels` (comma px list — the on-screen height of the tiled unit;
ladders run from a few px up to 8192px = ~512×), `--target-ms` (batch sizing), `--em` (world units per em),
`--shape-fill nonzero|evenodd`, `--check` (+ `--check-px N` to render/diff at a specific size), `--images`
(dump a PNG per level — both algorithms — to `output/bench/levels/<scene>_<px>px_{windfoil,slug}.png`, zero-padded
so they sort by zoom; skips timing). At deep magnification the glyph scene centers on a real stem near the origin
(so you "just see the stem"); the shape and tiger center on a unit, so all three zoom into ink not whitespace.

> The bundled tiger is a **simplified** version, so its absolute atlas sizes / timings won't match a
> production tiger — it's here to show relative behavior across views.

## What it measures

Both algorithms render the **same** scene into the same offscreen target; only the coverage technique differs.
A "zoom level" is the on-screen size in device pixels (pixels-per-em): the camera scales the fixed world scene
so a unit is `emPx` tall. Each scene's grid is sized to fill the viewport at its smallest level, and every
larger level zooms in on the same dense content, so **every level renders a full screen** — no empty frames.
Off-screen units are culled per level (as a real renderer would), so the timing is fragment-bound rather than
dominated by vertex-processing hundreds of thousands of off-screen instances.

Timing: for each (algorithm, level) it encodes N identical render passes into one command buffer, submits once,
and times `submit → onSubmittedWorkDone`, reporting the median per-frame over a few batches. N is picked so each
batch runs ~250 ms. (Timestamp queries were tried but Deno/wgpu on Metal doesn't normalise the timestamp period,
so the robust submit→done wall time is used instead.)

## DRY with `src/`

The benchmark reuses the repo's building blocks rather than duplicating them:

- **Banding** — `bandPieces` from `src/bands.js` files curves into row bands for *both* algorithms. Windfoil's
  horizontal bands are its monotone pieces; Slug's are whole quads, filed twice: once by y (horizontal ray) and
  once as 90°-rotated `(y, −x)` quads (vertical ray), so the same filer produces both band sets and the same
  in-shader gather serves both rays.
- **Pipeline** — `createGlyphRenderer` from `src/gpu.js` is reused verbatim. Slug packs its two band sets into
  one curve buffer + one row table, so it stays a 4-binding pipeline; only the instance stride (20 floats vs 16)
  and the shader differ.
- **Font / geometry / PNG** — `src/font.js`, `src/png.js` as-is.
- The only `src` change is making `loadShaderCode(url)` take an optional URL (defaulting to windfoil), so Slug's
  shader loads through the same loader.

Slug uses **whole quadratics** (its two-root solver), not windfoil's xy-monotone split — an intrinsic difference
the benchmark preserves.

## Implementation notes (Slug)

`slug.wgsl` is a faithful port of Eric Lengyel's **reference** `SlugPixelShader.hlsl`
([EricLengyel/Slug](https://github.com/EricLengyel/Slug)), not the simplified GreenLightning version. Getting the
last part right mattered a lot for quality (see below). Per curve it uses `CalcRootCode` (the `0x2E74` sign
classification) for root eligibility, solves `y(t)=0`, and accumulates BOTH a winding ramp
`clamp(x·pixelsPerEm + 0.5)` and a **weight** `clamp(1 − |r|·2)` (how close the crossing is to the pixel centre).
The **dual ray** does this for a horizontal and a vertical ray (the latter on the pre-rotated `(y,−x)` curves),
then combines them by Lengyel's `CalcCoverage`:

```
coverage = max( |xcov·xwgt + ycov·ywgt| / (xwgt + ywgt),  min(|xcov|, |ycov|) )
```

— a *weighted* combine (down-weight the ray whose crossing is far/unreliable), floored by the min of the two so
solid interiors stay solid. **This is not an average.** An earlier draft here used a naïve `(xcov+ycov)/2` and a
`t∈[0,1)` root test, which sprayed grey spikes at every sharp corner / tapered tip (whisker tips, fur, the
starburst cusps). Porting the real combine + `CalcRootCode` fixed it — that was a bug in this harness, not a
limitation of Slug.

One adaptation over the reference: roots use the numerically stable `q = b.y + sign(b.y)·d` form (roots
`{q/a.y, c.y/q}`), not `(b.y ∓ d)/a.y`. Our curves are in **font units** (coords in the hundreds/thousands),
where the naïve form loses all f32 precision on flat curves; the reference is fine because it works in normalized
em space. Same trick windfoil's `mono_root` uses.

`--check` confirms it: windfoil (validated against Skia — `docs/ALGORITHM.md §5`) and this Slug agree to mean
**|Δrgb| ≈ 0.0014** on text and **≈ 0.0006** on the tiger — two different exact-ish AA models nearly coinciding.

## Findings (Apple GPU, Deno 2.x — your numbers will differ)

**Glyphs (text):**

| regime | glyph px | result |
| --- | --- | --- |
| sub-pixel | ≤ ~1.4px | **windfoil ~8× faster** — its `MINIFICATION_GUARD` fires (pixel bigger than the glyph → cheap coverage-from-area) |
| minification | ~1.5–2px | **slug up to ~7× faster** — windfoil's worst case: the footprint spans many row bands, integrating many curves per pixel |
| small text | 3–16px | slug ~2.5–6× faster |
| reading / display | 24–128px | slug ~1.05–2× faster |
| magnified | ≥ ~192px | windfoil edges ahead (~1 curve/pixel, no dual-ray tax); both fill-rate bound |

windfoil trades per-pixel cost for exactness and degrades at minification (many bands per footprint), where
Slug's one-band-per-ray cost stays flat — until windfoil's sub-pixel guard undercuts it again below a pixel.
(The sweep bottoms out near ~1px: filling a viewport with sub-pixel units needs an impractical instance count —
a property of instanced rendering, not the algorithms.)

**Complex shape (240 self-crossing quads):**

- **Quality** — with the faithful Slug both renders are clean; windfoil's exact area integral still edges it at
  the ultra-sharp ellipse **cusps** (Slug leaves a thin needle where two edges converge to a near-point), but the
  gross "spiky" artifacts were a bug in the earlier naïve combine, now fixed. Mean |Δrgb| ≈ 0.012, concentrated
  at the cusps.
- **Speed** — Slug is faster through the practical range; windfoil overtakes at high magnification (from
  ~256–512px on the tiled shape, up to ~1.7× at 4096px), where its footprint collapses to a single band and its
  compare-don't-solve far-curve handling + single band axis win. A dense shape makes many bands, so windfoil's
  footprint spans several of them until magnified — its structural advantage is latent across normal zoom.

**Tiger (real SVG, 304 overlapping shapes):**

- **Quality** — near-identical to windfoil (mean |Δrgb| ≈ 0.0006 even at 512× zoom). The whisker-tip / fur
  spikes visible in an earlier version were the naïve-combine bug, not Slug.
- **Speed** — Slug leads at thumbnail/normal views (64px ~3.4×, 512px ~1.1×), but windfoil pulls ahead zoomed in
  (**~1024px up, to ~1.2–1.4×** at 4096px) — earlier than the single blob, because the tiger's many smaller shapes
  each reach a single-band footprint sooner, and the layered overdraw (a pixel runs one fragment per covering
  shape) rewards windfoil's cheaper per-shape gather + single band axis.
- **Memory** — the clearest win: windfoil's atlas is ~half of Slug's (**~490 KB vs ~950 KB** for this simplified
  tiger; the repo README cites ~0.84 vs 1.54 MB for the full one). Same ~2× ratio.

**Deep magnification (zoomed into a stem / a piece of the tiger — up to ~512×):** windfoil **wins on all three
scenes** here (glyphs ~1.2–1.25×, tiger ~1.16–1.23×, shape up to ~1.44×), and holds exact AA with no precision
wobble — the footprint is a single band of ~1 curve, so its cheap far-curve handling + single band axis beat
Slug's dual-ray, dual-solve. This is the regime for graphics work zoomed all the way in.

**Memory / bandwidth** — orthogonal to per-frame time, windfoil's single band axis stores about **half** of
Slug's dual bands (glyph atlas 59 KB vs 124 KB; shape 67 KB vs 90 KB; tiger ~½), matching the repo README's
tiger-SVG figures. This is the "half the reads" advantage; on these GPUs the coverage math is ALU-bound, so it
doesn't show up in frame time at normal zoom, but it does in footprint and would in a bandwidth-bound scene.

The honest summary: on this hardware Slug's lighter per-crossing coverage (one root solve + a ramp) beats
windfoil's exact area integral (several solves + a polynomial) across most zoom levels; windfoil's wins are
**exactness/quality** (self-intersections, overlap, the exact box filter) and **memory**, plus raw speed at the
sub-pixel guard and (increasingly, on the shape and tiger) magnification.

## Applied optimization

This harness was used to tune one core-algorithm change: **`TARGET_PER_BAND` 6 → 10** in `src/bands.js`.
Coarser bands cost windfoil almost nothing per extra piece (early-break + clamp/subtract far curves, no solve)
while a footprint spans fewer of them — measured **~8–19% faster at small/medium sizes** with no large-size
regression and a ~15% smaller atlas, coverage bit-identical (`deno task validate` unchanged). It's
windfoil-specific, so the benchmark pins Slug's own bands at 6 (via `bandPieces`'s new optional argument) to keep
the comparison fair. Also rejected: a straight-piece fast path (`mono_root` already skips the `sqrt` for lines).

## Rejected: band-moments acceleration (two attempts, both net-negative)

`docs/NOTES.md` proposes accelerating minified windfoil with **Band Moments** (a band wholly inside the pixel's
y-slab, x-contained by the box, contributes the closed form `S + (xref−rc.x+hx)·D`) and a **Backdrop** (fold
far-right winding into a constant). I built both — the math is correct and **bit-exact** (`--check` |Δrgb|
0.00000) — but both were **net-negative on real content and reverted:**

1. **2D-cell fusion** (cells + moments + backdrop, commit `e259b59`): only ~1.6× at deep minification of dense
   art (the shape at 2px), and ~5–20% *slower* everywhere else. Wide curves x-split into many straddle
   sub-pieces, so only interior cells collapse to O(1).
2. **Analytic y-prefix-sum of moments** (interior bands collapse to one prefix subtraction, instance-gated so the
   average case skips it): **worse** — ~3.8× *slower* at 2px and still ~19% slower at 256px.

The root cause is the GPU execution model, not the math:
- **Shader bloat** — adding the moment path lowers occupancy of the whole shader, so the *plain* path pays ~15–20%
  even when the fast path is never taken (measured at magnified sizes where the gate is off).
- **Branch divergence** — the moment can only be *exact* where the box x-contains the glyph, a per-fragment
  condition. At the small sizes where it fires, a warp spans many tiny glyphs (narrow ⇒ moment, wide ⇒ plain) and
  center-vs-edge pixels, so warps run **both** paths. Divergence rises as glyphs shrink — exactly the regime the
  moment targets — which is why the y-prefix gets *worse* toward 2px.

Conclusion: an exact per-fragment moment cannot beat windfoil's plain gather inside one shader — the bloat +
divergence swamp the O(1) win. The only acceleration that would sidestep both is a **separate, cheaper shader**
selected per instance/zoom — i.e. a prefiltered coverage **mip** sampled below a crossover size (O(1), no
divergence, box-filtered = windfoil's own target). That trades away windfoil's atlas-free identity, so it's a
product decision, not a free win. Absent that, **minification is windfoil's inherent weak flank** (it wins
magnification, memory, and exact quality); the `TARGET_PER_BAND` tune above is the analytic ceiling that helped.
