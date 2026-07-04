// windfoil.wgsl — box-filter coverage as a per-pixel winding integral. Math + derivation: docs/ALGORITHM.md.
//
// One instanced draw renders every glyph. The vertex stage expands a unit quad to each glyph's padded ink
// box; the fragment stage integrates the glyph's winding number over the pixel's footprint in closed form
// (F = ∫∫_box w dA / area(box)) and folds it to coverage. Curves are xy-monotone quadratic pieces filed
// into row bands (see bands.js); each instance points at its glyph's bands via the row table below.

struct Uniforms {
  res : vec2<f32>,    // render-target size in pixels
  style : vec2<f32>,  // (gamma, sharp) coverage transform; (1, 1) = exact (identity). See main.js --gamma / --sharp.
  cam : vec4<f32>,    // camera: device px = worldPx·(scaleX, scaleY) + (transX, transY). (1,1,0,0) = identity.
};

struct Instance {
  place : vec4<f32>, // originX, originY (device px of the glyph's origin), unitsToPx, fillRule (0 = nonzero, 1 = even-odd)
  bbox  : vec4<f32>, // ink box loX, loY, hiX, hiY (font units, Y-down)
  color : vec4<f32>, // straight-alpha RGBA
  band  : vec4<f32>, // rowBase, bandCount, y0, invH  (the glyph's row-band table + its y-origin / bands-per-unit)
};

// Bands with count > SORT_MIN are x-sorted on the CPU (by hull x-max, descending), so once we reach a piece
// fully left of the box every remaining piece is too and we can break. MUST equal BAND_SORT_MIN in bands.js —
// see the tuning note there.
const SORT_MIN : u32 = 4u;

// Row-table layout — MUST match bands.js's rowOut.push(start, count, areaBits, xMinBits, xMaxBits).
// start/count index the curve atlas; the bit-punned f32s (the band strip's precomputed winding integral and
// its piece hull in x) are consumed only by the minification guard's banded ink profile (profile_face).
const ROW_STRIDE : u32 = 5u;
const ROW_AREA : u32 = 2u;
const ROW_XMIN : u32 = 3u;
const ROW_XMAX : u32 = 4u;

// When a glyph shrinks to a few pixels, every pixel's footprint spans most of its bands, so the gather
// integrates nearly all of its curves at every pixel and per-pixel cost peaks — while the text is far too
// small to read, so exactness buys nothing. Below GUARD_PX device pixels (whole glyph, both axes) the guard
// swaps the gather for a banded ink profile: each band's EXACT winding integral ∫∫_strip w dA is precomputed
// at atlas build (one f32 riding in the row table) and the pixel takes its y-overlap share of each band,
// spread uniformly across that band's ink hull in x. Sub-pixel glyphs resolve to their true average
// coverage; at 2–4px the vertical ink distribution (stems, bowls, baselines) survives and only the per-band
// x-profile flattens. The branch is per-instance-per-zoom (warp-coherent), and the path is a few band taps —
// no curve reads at all.
//
// 3.7 covers every Lato glyph at a 4px em (tallest ≈ 0.9em ≈ 3.6px) while leaving an 8px em untouched
// (lowercase x-height ≈ 0.54em ≈ 4.3px) — it fires only where text is illegible anyway. Dense vector art
// with lots of sub-8px features (drawing thumbnails) can raise it: on the bench's tiger at a 64px drawing
// height, 8.0 renders 1.9× faster with a mean error smaller than the windfoil-vs-Slug AA-model difference —
// but that trades exactness of small-but-visible features, so it is not the default.
const MINIFICATION_GUARD = true;
const GUARD_PX = 3.7;

@group(0) @binding(0) var<uniform> U : Uniforms;
@group(0) @binding(1) var<storage, read> instances : array<Instance>;
// The deduped, band-duplicated curve atlas: three consecutive vec2 per xy-monotone piece (endpoints + control).
@group(0) @binding(2) var<storage, read> curves : array<vec2<f32>>;
// Row-band table: ROW_STRIDE u32s per band, laid out per the ROW_* constants above. See bands.js.
@group(0) @binding(3) var<storage, read> rows : array<u32>;

struct VsOut {
  @builtin(position) pos : vec4<f32>,
  @location(0) rc : vec2<f32>,                       // em-space position of this fragment (Y-down)
  @location(1) @interpolate(flat) inst : u32,
};

@vertex
fn vs(@builtin(vertex_index) vi : u32, @builtin(instance_index) ii : u32) -> VsOut {
  let I = instances[ii];
  let unitsToPx = I.place.z;                          // device pixels per glyph coordinate unit
  let camScale = U.cam.xy;                            // world→device px scale (camera zoom); (1,1) = identity
  // Pad the quad outward by 1 device px so an edge's anti-aliased skirt is never clipped: coverage reaches
  // at most half a pixel past the ink (the box filter's half-width), so 1px is 2× margin. The pad is in
  // glyph units, so divide by the ON-SCREEN scale (unitsToPx·camScale) to keep it 1 device px at any zoom.
  // Keeping it tight matters at minification, where the pad ring is a large share of the fragments.
  let pad = 1.0 / (unitsToPx * max(camScale.x, 1e-6));
  let lo = I.bbox.xy - vec2<f32>(pad);
  let hi = I.bbox.zw + vec2<f32>(pad);
  // Unit-quad corners for a triangle-strip: (0,0) (1,0) (0,1) (1,1) — vi ∈ {0..3} (every caller draws 4).
  let uv = vec2<f32>(f32(vi & 1u), f32(vi >> 1u));
  let em = mix(lo, hi, uv);
  let worldPx = I.place.xy + em * unitsToPx;          // Y-down "world" pixels (the layout space)
  let devicePx = worldPx * camScale + U.cam.zw;       // apply the camera (pan + zoom) → device px
  let ndc = devicePx / U.res * 2.0 - 1.0;             // Y-down NDC; clip space flips y
  var o : VsOut;
  o.pos = vec4<f32>(ndc.x, -ndc.y, 0.0, 1.0);
  o.rc = em;
  o.inst = ii;
  return o;
}

// Period-2 triangle wave T(t) = 1 − |1 − (t mod 2)|: folds a signed winding MAGNITUDE to its even-odd
// coverage (a value of 2 — an enclosed hole — folds to 0; fractional edges ramp linearly in and out).
// The result is in [0, 1] by construction.
fn tri_wave(t : f32) -> f32 {
  let m = t - 2.0 * floor(t * 0.5);
  return 1.0 - abs(1.0 - m);
}

// Optional perceptual styling of the final coverage (--gamma / --sharp in main.js): a post-fold transfer curve
// on the EXACT coverage. `gamma` re-weights stems (<1 bolder/darker, >1 thinner); `sharp` sets the edge
// contrast about 0.5 (>1 crisper, <1 softer). (1, 1) is the identity, so the default "exact" path is bit-for-
// bit untouched. This departs from the true box filter by design — opt-in tuning, not the reference.
fn style_coverage(cov : f32, gamma : f32, sharp : f32) -> f32 {
  if (gamma == 1.0 && sharp == 1.0) { return cov; } // exact: reference path, untouched
  let g = pow(cov, gamma);
  var s : f32;
  if (g < 0.5) {
    s = 0.5 * pow(2.0 * g, sharp);
  } else {
    s = 1.0 - 0.5 * pow(2.0 * (1.0 - g), sharp);
  }
  return clamp(s, 0.0, 1.0);
}

// Straight-alpha color × coverage → premultiplied RGBA (the pipeline blends premultiplied-over).
fn shade(color : vec4<f32>, cov : f32) -> vec4<f32> {
  let a = color.a * cov;
  return vec4<f32>(color.rgb * a, a);
}

// The shared fragment tail: fold the pixel-averaged winding number `f` to coverage by fill rule, apply the
// opt-in --gamma / --sharp styling ((1, 1) exact ⇒ identity), and emit premultiplied color.
fn fold_shade(f : f32, fillRule : f32, color : vec4<f32>) -> vec4<f32> {
  var cov : f32;
  if (fillRule > 0.5) {
    cov = tri_wave(f);                // even-odd
  } else {
    cov = clamp(abs(f), 0.0, 1.0);    // nonzero (saturating winding integral)
  }
  return shade(color, style_coverage(cov, U.style.x, U.style.y));
}

// Solve the monotone quadratic component A2·t² + A1·t + A0 = v on [0,1]. `a0` is the component's value at
// t = 0 (the constant coefficient); `e1` its value at t = 1 (endpoint-exact for a monotone piece): saturate
// to 0 if the piece starts past `v`, to 1 if it never reaches `v`.
fn mono_root(a2 : f32, a1 : f32, a0 : f32, e1 : f32, v : f32, rising : bool) -> f32 {
  if (rising) {
    if (a0 >= v) { return 0.0; }
    if (e1 <= v) { return 1.0; }
  } else {
    if (a0 <= v) { return 0.0; }
    if (e1 >= v) { return 1.0; }
  }
  let c = a0 - v;
  // Near-linear fallback: |a2| this small and v strictly between the endpoints ⇒ |a1| ≈ |e1 − e0| > 0.
  if (abs(a2) < 1e-12 * max(abs(a1), 1.0)) {
    return clamp(-c / a1, 0.0, 1.0);
  }
  let disc = max(a1 * a1 - 4.0 * a2 * c, 0.0);
  let sq = sqrt(disc);
  let qq = -0.5 * (a1 + select(-sq, sq, a1 >= 0.0));   // numerically stable quadratic
  let r1 = qq / a2;
  let r2 = select(0.0, c / qq, qq != 0.0);
  // The derivative at r1 is 2·a2·r1 + a1 = 2·qq + a1 = −sign(a1)·sq, so "derivative sign matches `rising`"
  // reduces to a sign test on a1 — no derivative evaluation, and the pick no longer waits on the division.
  // (When sq = 0 the roots coincide, so either pick is the same value.)
  let t = select(r2, r1, (a1 < 0.0) == rising);
  return clamp(t, 0.0, 1.0);
}

// The INSIDE zone's exact integral of (x(t)+hx)·y′(t) over [ta,tb]: the midpoint rule on a symmetric
// interval, which is exact for this cubic integrand (odd powers about the midpoint vanish). `x0` is the
// piece's constant x coefficient; `x(t)+hx` is box-local (0..sx), so there is no large-magnitude cancellation.
fn integrate_inside(a2 : vec2<f32>, a1 : vec2<f32>, x0 : f32, ta : f32, tb : f32, hx : f32) -> f32 {
  if (tb <= ta) { return 0.0; }
  let tm = 0.5 * (ta + tb);
  let d = 0.5 * (tb - ta);
  let x_mid = (a2.x * tm + a1.x) * tm + x0 + hx;
  let dmid = 2.0 * a2 * tm + a1;                       // (x′, y′) at the midpoint
  return 2.0 * d * x_mid * dmid.y + (2.0 * d * d * d / 3.0) * (a2.x * dmid.y + 2.0 * a2.y * dmid.x);
}

// One xy-monotone piece's contribution to ∫∫_box w over the rc-relative y-window [lo, hi] (already
// intersected with the piece's y-span), box half-width hx. The winding integral integrates out x as
// clamp(x(t), −hx, hx) + hx, splitting the crossing t-interval into LEFT (0) / INSIDE (exact) / RIGHT
// (full box width) zones at the two x-edge crossings — in a statically known order set by the x direction.
fn integrate_piece(q1 : vec2<f32>, q2 : vec2<f32>, q3 : vec2<f32>, lo : f32, hi : f32, hx : f32) -> f32 {
  let a2 = q1 - 2.0 * q2 + q3;
  let a1 = 2.0 * (q2 - q1);
  let y_rising = q3.y >= q1.y;
  let t_lo = mono_root(a2.y, a1.y, q1.y, q3.y, select(hi, lo, y_rising), y_rising);
  let t_hi = mono_root(a2.y, a1.y, q1.y, q3.y, select(lo, hi, y_rising), y_rising);
  if (t_hi <= t_lo) { return 0.0; }
  let x_rising = q3.x >= q1.x;
  let t_left = clamp(mono_root(a2.x, a1.x, q1.x, q3.x, -hx, x_rising), t_lo, t_hi);
  let t_right = clamp(mono_root(a2.x, a1.x, q1.x, q3.x, hx, x_rising), t_lo, t_hi);
  // Zones in sweep order: x rising ⇒ LEFT [t_lo,t_left] · INSIDE · RIGHT [t_right,t_hi]; mirrored if not.
  let t1 = select(t_right, t_left, x_rising);
  let t2 = max(select(t_left, t_right, x_rising), t1);
  var acc = integrate_inside(a2, a1, q1.x, t1, t2, hx);
  let ra = select(t_lo, t2, x_rising);
  let rb = select(t1, t_hi, x_rising);
  if (rb > ra) {
    // RIGHT zone: full box width × Δy, with Δy = Δt · y′(midpoint) (exact for the quadratic).
    let tm = 0.5 * (ra + rb);
    acc += (rb - ra) * (2.0 * a2.y * tm + a1.y) * (2.0 * hx);
  }
  return acc;
}

// A piece's y-span clipped to the window, as a difference of clamped ENDPOINTS (not sign·overlap) so it
// telescopes over piece chains: shared endpoints cancel exactly, so a run of ULP-scale segments sums to its
// span rather than accumulating per-piece rounding. Signed: returns clamp(y3) − clamp(y1).
fn clipped_dy(y1 : f32, y3 : f32, wlo : f32, whi : f32) -> f32 {
  return clamp(y3, wlo, whi) - clamp(y1, wlo, whi);
}

// Accumulate one ROW BAND's pieces over the rc-relative y-window [wlo, whi] (the pixel box clipped to this
// band's y-range). Extent tests are endpoint-exact for monotone pieces: a piece fully left of the box adds
// 0, one fully right adds full box width × its signed clipped y-span. Long bands are x-sorted, so we break
// at the first fully-left piece; short bands run the plain loop.
fn integrate_band(start : u32, count : u32, rc : vec2<f32>, wlo : f32, whi : f32, sx : f32) -> f32 {
  var acc : f32 = 0.0;
  let hx = sx * 0.5;
  let sorted = count > SORT_MIN;
  // a piece whose whole hull spans no more than a few coordinate-ULPs is f32-degenerate (see below)
  let coord_ulp = max(abs(rc.x), abs(rc.y)) * 1.2e-7;
  for (var i : u32 = 0u; i < count; i = i + 1u) {
    let base = (start + i) * 3u;
    let q1 = curves[base] - rc;
    let q2 = curves[base + 1u] - rc;
    let q3 = curves[base + 2u] - rc;
    let x_hull_max = max(q1.x, max(q2.x, q3.x));
    if (x_hull_max <= -hx) {                              // fully LEFT of the box → no area
      if (sorted) { break; }                             // every remaining piece is further left
      continue;
    }
    let py_lo = min(q1.y, q3.y);                          // piece y-span (endpoint-exact when monotone)
    let py_hi = max(q1.y, q3.y);
    let lo = max(wlo, py_lo);                             // window ∩ piece y-span
    let hi = min(whi, py_hi);
    if (hi <= lo) { continue; }
    let x_hull_min = min(q1.x, min(q2.x, q3.x));
    if (x_hull_min >= hx) {
      // fully RIGHT of the box → full box width × the clipped y-span
      acc += sx * clipped_dy(q1.y, q3.y, wlo, whi);
      continue;
    }
    // f32-degenerate piece (whole hull within a few coordinate-ULPs — flattened content at deep zoom far
    // from the origin): integrate_piece's t-solves would divide ULP-scale coefficients into noise. The
    // midpoint-clamp form is exact to ~span² and telescopes like the fully-right path.
    if (x_hull_max - x_hull_min + (py_hi - py_lo) <= coord_ulp * 16.0) {
      let xm = clamp((q1.x + q3.x) * 0.5, -hx, hx) + hx;
      acc += xm * clipped_dy(q1.y, q3.y, wlo, whi);
      continue;
    }
    acc += integrate_piece(q1, q2, q3, lo, hi, hx);
  }
  return acc;
}

// The band index a y-offset (relative to the band origin) maps to: floor(dy·invH) clamped to [0, R−1] —
// the same mapping bands.js files with, so filing and lookup always agree.
fn band_index(dy : f32, invH : f32, R : u32) -> u32 {
  return u32(clamp(floor(dy * invH), 0.0, f32(R) - 1.0));
}

// Band ri's y-range relative to `base` (rc-relative dy0 in the exact gather, absolute y0 in the guard).
// R ≤ 64 (MAX_BANDS in bands.js), so f32(ri) + 1.0 is exact.
fn band_edges(base : f32, ri : u32, invH : f32) -> vec2<f32> {
  let r = f32(ri);
  return vec2<f32>(base + r / invH, base + (r + 1.0) / invH);
}

// Length of the overlap of intervals [a0, a1] and [b0, b1] (0 when disjoint).
fn overlap1d(a0 : f32, a1 : f32, b0 : f32, b1 : f32) -> f32 {
  return max(min(a1, b1) - max(a0, b0), 0.0);
}

// One glyph's winding integral over the pixel box (rc ± s/2), gathered through its ROW BANDS. The pixel's
// y-slab selects the band range it touches; each band is read clipped to its own y-range. The bands tile
// the slab, so a piece duplicated across adjacent bands integrates over disjoint windows and the sum is
// exact without a dedupe test.
fn integrate_face(band : vec4<f32>, rc : vec2<f32>, s : vec2<f32>) -> f32 {
  let rowBase = u32(band.x);
  let R = u32(band.y);
  let invH = band.w;

  // Build the slab rc-RELATIVE ([−sy2, +sy2]). At deep zoom on far-from-origin coordinates sy/2 drops below
  // ULP(rc.y), so an absolute slab (rc.y ± sy2, then − rc.y) would quantize to zero height and drop whole
  // pixel rows (horizontal banding). The one absolute term (dy0 = y0 − rc.y) feeds only band selection and
  // the boundary clips, whose ULP wobble just nudges the split between adjacent (still exactly tiling)
  // windows. See docs/ALGORITHM.md §6.
  let sy2 = s.y * 0.5;
  let dy0 = band.z - rc.y;      // band origin y0, relative to the pixel center
  var ri0 : u32 = 0u;
  var ri1 : u32 = 0u;
  if (invH > 0.0) {             // invH > 0 only for multi-band glyphs (bands.js stores 0 when R == 1)
    ri0 = band_index(-dy0 - sy2, invH, R);
    ri1 = band_index(-dy0 + sy2, invH, R);
  }
  var f_int : f32 = 0.0;
  for (var ri = ri0; ri <= ri1; ri = ri + 1u) {
    var w_lo = -sy2;             // rc-relative window edges — stable at any zoom
    var w_hi = sy2;
    if (invH > 0.0) {
      let e = band_edges(dy0, ri, invH);                // band ri's y-range, rc-relative
      w_lo = max(w_lo, e.x);
      w_hi = min(w_hi, e.y);
    }
    if (w_hi <= w_lo) { continue; }
    let rIdx = (rowBase + ri) * ROW_STRIDE;
    f_int += integrate_band(rows[rIdx], rows[rIdx + 1u], rc, w_lo, w_hi, s.x);
  }
  return f_int;
}

// The minification guard's twin of integrate_face: the same ∫∫_box w dA, but approximated from the banded
// ink profile precomputed at atlas build — each band contributes its strip integral × the pixel's y-overlap
// share of the strip × its x-overlap share of the band's ink hull. A few table taps, no curve reads; see the
// GUARD_PX comment for when this fires and what it preserves.
fn profile_face(band : vec4<f32>, bbox : vec4<f32>, rc : vec2<f32>, s : vec2<f32>) -> f32 {
  let pixLo = rc - s * 0.5;      // this pixel's box, in glyph units
  let pixHi = rc + s * 0.5;
  if (overlap1d(pixLo.x, pixHi.x, bbox.x, bbox.z) <= 0.0) { return 0.0; }
  let rowBase = u32(band.x);
  let R = u32(band.y);
  // header invH is 0 for a single band — the profile math wants the real 1/bandHeight
  let invH = select(band.w, 1.0 / max(bbox.w - bbox.y, 1e-30), band.w == 0.0);
  let y0 = band.z;
  var ri0 : u32 = 0u;
  var ri1 : u32 = 0u;
  if (R > 1u) {
    ri0 = band_index(pixLo.y - y0, invH, R);
    ri1 = band_index(pixHi.y - y0, invH, R);
  }
  var ink : f32 = 0.0;           // Σ band strip integral × y-share × x-share
  for (var ri = ri0; ri <= ri1; ri = ri + 1u) {
    let rIdx = (rowBase + ri) * ROW_STRIDE;
    let e = band_edges(y0, ri, invH);
    let ov = overlap1d(pixLo.y, pixHi.y, e.x, e.y);
    let hull0 = bitcast<f32>(rows[rIdx + ROW_XMIN]);
    let hull1 = bitcast<f32>(rows[rIdx + ROW_XMAX]);
    let fx = overlap1d(pixLo.x, pixHi.x, hull0, hull1) / max(hull1 - hull0, 1e-30);
    ink += bitcast<f32>(rows[rIdx + ROW_AREA]) * (ov * invH) * fx;
  }
  return ink;
}

@fragment
fn fs(in : VsOut) -> @location(0) vec4<f32> {
  let I = instances[in.inst];
  let rc = in.rc;
  // units_per_pixel from the screen-space gradients (fwidth = |∂/∂x| + |∂/∂y| per axis) — under pure
  // scale/translation one term is zero and this is exactly the device pixel's preimage, with no sqrt.
  let s = max(fwidth(rc), vec2<f32>(1e-9));

  // Minification guard: when the whole glyph spans ≤ GUARD_PX device pixels on both axes, render from the
  // banded ink profile instead of the exact gather (see the constant's comment above).
  if (MINIFICATION_GUARD && all(s * GUARD_PX >= I.bbox.zw - I.bbox.xy)) {
    return fold_shade(profile_face(I.band, I.bbox, rc, s) / (s.x * s.y), I.place.w, I.color);
  }

  // One gather → the winding integral ∫∫_box w over the pixel box, normalized to the pixel-averaged
  // winding number, then fold / style / shade.
  return fold_shade(integrate_face(I.band, rc, s) / (s.x * s.y), I.place.w, I.color);
}
