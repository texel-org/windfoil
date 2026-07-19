// windfoil.wgsl — box-filter coverage as a per-pixel winding integral. Math + derivation: docs/ALGORITHM.md.
// The vertex stage expands a unit quad per instance; the fragment stage integrates the winding number over
// the pixel's footprint in closed form (F = ∫∫_box w dA / area) and folds it to coverage.

struct Uniforms {
  res : vec2f,    // render-target size in pixels
  style : vec2f,  // (gamma, sharp) coverage transform; (1, 1) = exact
  cam : vec4f,    // camera: device px = worldPx·(scaleX, scaleY) + (transX, transY)
};

struct Instance {
  place : vec4f, // originX, originY (device px), unitsToPx, fillRule (0 = nonzero, 1 = even-odd)
  bbox  : vec4f, // ink box loX, loY, hiX, hiY (glyph units, Y-down)
  color : vec4f, // straight-alpha RGBA
  band  : vec4f, // rowBase, bandCount, bandH, invH
};

// Bands with count > SORT_MIN are x-sorted on the CPU so the gather can break at the first piece fully left
// of the box. MUST equal BAND_SORT_MIN in bands.js.
const SORT_MIN : u32 = 4u;

// Row-table layout — MUST match bands.js's rowOut.push(start, count, densityBits, xMinBits, xMaxBits).
const ROW_STRIDE : u32 = 5u;
const ROW_DENSITY : u32 = 2u;
const ROW_XMIN : u32 = 3u;
const ROW_XMAX : u32 = 4u;

// Below GUARD_PX device pixels (whole glyph, both axes — illegible sizes) coverage comes from the banded ink
// profile (profile_face) instead of the exact gather. Threshold rationale + the quality/perf trade of raising
// it: bench/README.md, bench/ACCEL-NOTES.md. Pipeline-overridable (specialized at pipeline creation, so the
// disabled branch still compiles out): the validation suite's exact mode sets it to false via the pipeline
// `constants` map so the ink-profile approximation never stands in for the integral it is measuring.
override MINIFICATION_GUARD : bool = true;
const GUARD_PX = 3.7;

override EXACT_MODE : bool = false; // offline: point-sample the true fill rule, no fold (ALGORITHM.md §4)
override EXACT_GRID : u32 = 8u;     // sub-samples per axis in exact mode

// Kernel support plus 0.125px derivative slack; adjust support per axis for other kernels.
const KERNEL_SUPPORT_PX = vec2f(0.5);
const KERNEL_SKIRT_PX = KERNEL_SUPPORT_PX + vec2f(0.125);

@group(0) @binding(0) var<uniform> U : Uniforms;
@group(0) @binding(1) var<storage, read> instances : array<Instance>;
// Curve atlas: three consecutive vec2 per xy-monotone piece (endpoints + control).
@group(0) @binding(2) var<storage, read> curves : array<vec2f>;
// Row-band table: ROW_STRIDE u32s per band (see the ROW_* constants and bands.js).
@group(0) @binding(3) var<storage, read> rows : array<u32>;

struct VsOut {
  @builtin(position) pos : vec4f,
  @location(0) rc : vec2f,                       // glyph-space position of this fragment (Y-down)
  @location(1) @interpolate(flat) inst : u32,
};

@vertex
fn vs(@builtin(vertex_index) vi : u32, @builtin(instance_index) ii : u32) -> VsOut {
  let I = instances[ii];
  let unitsToPx = I.place.z;
  let camScale = U.cam.xy;
  let pad = KERNEL_SKIRT_PX / (unitsToPx * max(abs(camScale), vec2f(1e-6)));
  // Unit-quad corners for a triangle-strip; vi ∈ {0..3}.
  let uv = vec2f(f32(vi & 1u), f32(vi >> 1u));
  let em = mix(I.bbox.xy - pad, I.bbox.zw + pad, uv);
  let worldPx = I.place.xy + em * unitsToPx;
  let devicePx = worldPx * camScale + U.cam.zw;
  let ndc = devicePx / U.res * 2.0 - 1.0;             // Y-down NDC; clip space flips y
  return VsOut(vec4f(ndc.x, -ndc.y, 0.0, 1.0), em, ii);
}

// Period-2 triangle wave 1 − |1 − (t mod 2)|: folds signed winding to even-odd coverage, range [0, 1].
fn tri_wave(t : f32) -> f32 {
  return 1.0 - abs(1.0 - (t - 2.0 * floor(t * 0.5)));
}

// Opt-in perceptual styling (--gamma / --sharp); (1, 1) leaves the exact coverage bit-for-bit untouched.
fn style_coverage(cov : f32, gamma : f32, sharp : f32) -> f32 {
  if (gamma == 1.0 && sharp == 1.0) { return cov; }
  let g = pow(cov, gamma);
  let lo = g < 0.5;
  let p = 0.5 * pow(select(2.0 * (1.0 - g), 2.0 * g, lo), sharp);
  return clamp(select(1.0 - p, p, lo), 0.0, 1.0);
}

// Straight-alpha color × coverage → premultiplied RGBA (the pipeline blends premultiplied-over).
fn shade(color : vec4f, cov : f32) -> vec4f {
  let a = color.a * cov;
  return vec4f(color.rgb * a, a);
}

// Shared fragment tail: fold the pixel-averaged winding number by fill rule, style, shade.
fn fold_shade(f : f32, fillRule : f32, color : vec4f) -> vec4f {
  let cov = select(min(abs(f), 1.0), tri_wave(f), fillRule > 0.5); // nonzero (saturating) | even-odd
  return shade(color, style_coverage(cov, U.style.x, U.style.y));
}

// Solve A2·t² + A1·t + A0 = v for t∈[0,1], saturating at a0=q(0) and e1=q(1).
fn mono_root(a2 : f32, a1 : f32, a0 : f32, e1 : f32, v : f32, rising : bool) -> f32 {
  if (rising) {
    if (a0 >= v) { return 0.0; }
    if (e1 <= v) { return 1.0; }
  } else {
    if (a0 <= v) { return 0.0; }
    if (e1 >= v) { return 1.0; }
  }
  let c = a0 - v;
  if (abs(a2) < 1e-12 * max(abs(a1), 1.0)) {           // near-linear fallback
    return clamp(-c / a1, 0.0, 1.0);
  }
  let disc = max(a1 * a1 - 4.0 * a2 * c, 0.0);
  let sq = sqrt(disc);
  let qq = -0.5 * (a1 + select(-sq, sq, a1 >= 0.0));   // numerically stable quadratic
  // r1's derivative is −sign(a1)·sq, so its sign selects the root; choose operands before one safe divide.
  let use_r1 = (a1 < 0.0) == rising;
  let num = select(c, qq, use_r1);
  let den = select(qq, a2, use_r1);
  let valid = den != 0.0;
  return clamp(select(0.0, num / select(1.0, den, valid), valid), 0.0, 1.0);
}

// Exact INSIDE integral: expand cubic (x(t)+hx)·y′(t) about the interval midpoint; x0 is x(t)'s constant.
fn integrate_inside(a2 : vec2f, a1 : vec2f, x0 : f32, ta : f32, tb : f32, hx : f32) -> f32 {
  if (tb <= ta) { return 0.0; }
  let tm = 0.5 * (ta + tb);
  let d = 0.5 * (tb - ta);
  let x_mid = (a2.x * tm + a1.x) * tm + x0 + hx;
  let dmid = 2.0 * a2 * tm + a1;                       // (x′, y′) at the midpoint
  return 2.0 * d * x_mid * dmid.y + (2.0 * d * d * d / 3.0) * (a2.x * dmid.y + 2.0 * a2.y * dmid.x);
}

// Integrate clamp(x(t), −hx, hx)+hx over the y-window through LEFT (0), INSIDE (exact), and RIGHT
// (full-width) zones (ALGORITHM.md §3). The four clip roots are solved as one branchless vec4 batch with
// mono_root's saturation semantics.
fn integrate_piece(q1 : vec2f, q2 : vec2f, q3 : vec2f, lo : f32, hi : f32, hx : f32) -> f32 {
  let a2 = q1 - 2.0 * q2 + q3;
  let a1 = 2.0 * (q2 - q1);
  let y_rising = q3.y >= q1.y;
  let x_rising = q3.x >= q1.x;
  let A2 = a2.yyxx;
  let A1 = a1.yyxx;
  let V  = vec4f(select(vec2f(hi, lo), vec2f(lo, hi), y_rising), -hx, hx);
  let R  = vec2<bool>(y_rising, x_rising).xxyy;
  let SG = select(vec4f(-1.0), vec4f(1.0), R);

  let C = q1.yyxx - V;
  let sat0 = (C * SG) >= vec4f(0.0);
  let sat1 = ((q3.yyxx - V) * SG) <= vec4f(0.0);
  let disc = max(A1 * A1 - 4.0 * A2 * C, vec4f(0.0));
  let sq = sqrt(disc);
  let qq = -0.5 * (A1 + select(-sq, sq, A1 >= vec4f(0.0)));
  let use_r1 = (A1 < vec4f(0.0)) == R;
  let num = select(C, qq, use_r1);
  let den = select(qq, A2, use_r1);
  let valid = den != vec4f(0.0);
  let t_raw = clamp(
    select(vec4f(0.0), num / select(vec4f(1.0), den, valid), valid),
    vec4f(0.0), vec4f(1.0),
  );
  let T = select(select(t_raw, vec4f(1.0), sat1), vec4f(0.0), sat0);

  if (T.y <= T.x) { return 0.0; }
  let tc = clamp(T.zw, T.xx, T.yy);
  // Zones in sweep order: x rising ⇒ LEFT · INSIDE · RIGHT; mirrored if not.
  let t1 = select(tc.y, tc.x, x_rising);
  let t2 = max(select(tc.x, tc.y, x_rising), t1);
  var acc = integrate_inside(a2, a1, q1.x, t1, t2, hx);
  let rab = select(vec2f(T.x, t1), vec2f(t2, T.y), x_rising);
  let d = max(rab.y - rab.x, 0.0);
  let tm = 0.5 * (rab.x + rab.y);
  acc += d * (2.0 * a2.y * tm + a1.y) * (2.0 * hx);   // RIGHT zone: full width × Δy
  return acc;
}

// Signed clipped y-span; endpoint clamps make adjacent pieces telescope exactly.
fn clipped_dy(y1 : f32, y3 : f32, wlo : f32, whi : f32) -> f32 {
  return clamp(y3, wlo, whi) - clamp(y1, wlo, whi);
}

// Accumulate one row band's pieces over the rc-relative y-window [wlo, whi].
fn integrate_band(start : u32, count : u32, rc : vec2f, wlo : f32, whi : f32, sx : f32) -> f32 {
  var acc : f32 = 0.0;
  let hx = sx * 0.5;
  let sorted = count > SORT_MIN;
  let coord_ulp = max(abs(rc.x), abs(rc.y)) * 1.2e-7;
  for (var i : u32 = 0u; i < count; i = i + 1u) {
    let base = (start + i) * 3u;
    let q1 = curves[base] - rc;
    let q3 = curves[base + 2u] - rc;
    // In an xy-monotone quadratic q2 lies in the endpoint hull, so load it only after culling.
    let x_hull_max = max(q1.x, q3.x);
    if (x_hull_max <= -hx) {              // fully LEFT of the box → no area
      if (sorted) { break; }
      continue;
    }
    let py_lo = min(q1.y, q3.y);          // piece y-span (endpoint-exact when monotone)
    let py_hi = max(q1.y, q3.y);
    let lo = max(wlo, py_lo);
    let hi = min(whi, py_hi);
    if (hi <= lo) { continue; }
    let x_hull_min = min(q1.x, q3.x);
    if (x_hull_min >= hx) {               // fully RIGHT of the box → full width × clipped y-span
      acc += sx * clipped_dy(q1.y, q3.y, wlo, whi);
      continue;
    }
    // For a hull within a few coordinate ULPs, avoid noisy t-solves; midpoint-clamp is exact to ~span²
    // and telescopes.
    if (x_hull_max - x_hull_min + (py_hi - py_lo) <= coord_ulp * 16.0) {
      let xm = clamp((q1.x + q3.x) * 0.5, -hx, hx) + hx;
      acc += xm * clipped_dy(q1.y, q3.y, wlo, whi);
      continue;
    }
    let q2 = curves[base + 1u] - rc;
    acc += integrate_piece(q1, q2, q3, lo, hi, hx);
  }
  return acc;
}

// Band index for a y-offset from the band origin: floor(dy·invH) clamped — the same mapping bands.js files with.
fn band_index(dy : f32, invH : f32, R : u32) -> u32 {
  return u32(clamp(floor(dy * invH), 0.0, f32(R) - 1.0));
}

// Band ri's y-range relative to `base`. R ≤ 64, so f32(ri) + 1.0 is exact.
fn band_edges(base : f32, ri : u32, bandH : f32) -> vec2f {
  return base + vec2f(f32(ri), f32(ri) + 1.0) * bandH;
}

// Length of the overlap of intervals [a0, a1] and [b0, b1] (0 when disjoint).
fn overlap1d(a0 : f32, a1 : f32, b0 : f32, b1 : f32) -> f32 {
  return max(min(a1, b1) - max(a0, b0), 0.0);
}

// One glyph's winding integral over the pixel box (rc ± s/2), gathered through the row bands its y-slab
// touches. Windows are kept rc-RELATIVE for deep-zoom stability, and tile exactly across bands so duplicated
// pieces never double-count (ALGORITHM.md §6).
fn integrate_face(band : vec4f, y0 : f32, rc : vec2f, s : vec2f) -> f32 {
  let rowBase = u32(band.x);
  let R = u32(band.y);
  let bandH = band.z;
  let invH = band.w;
  let sy2 = s.y * 0.5;
  let dy0 = y0 - rc.y;          // band origin relative to the pixel center
  // R == 1 stores invH = 0, so band_index degenerates to 0 on its own.
  let ri0 = band_index(-dy0 - sy2, invH, R);
  let ri1 = band_index(-dy0 + sy2, invH, R);
  var f_int : f32 = 0.0;
  for (var ri = ri0; ri <= ri1; ri = ri + 1u) {
    var w_lo = -sy2;
    var w_hi = sy2;
    if (R > 1u) {
      let e = band_edges(dy0, ri, bandH);
      w_lo = max(w_lo, e.x);
      w_hi = min(w_hi, e.y);
    }
    if (w_hi <= w_lo) { continue; }
    let rIdx = (rowBase + ri) * ROW_STRIDE;
    f_int += integrate_band(rows[rIdx], rows[rIdx + 1u], rc, w_lo, w_hi, s.x);
  }
  return f_int;
}

// Approximate minification twin of integrate_face: integrate each band's precomputed winding density over
// its overlap with the pixel. A few table taps, no curve reads.
fn profile_face(band : vec4f, bbox : vec4f, rc : vec2f, s : vec2f) -> f32 {
  let pixLo = rc - s * 0.5;
  let pixHi = rc + s * 0.5;
  if (overlap1d(pixLo.x, pixHi.x, bbox.x, bbox.z) <= 0.0) { return 0.0; }
  let rowBase = u32(band.x);
  let R = u32(band.y);
  let bandH = band.z;
  let invH = band.w;
  let y0 = bbox.y;
  let ri0 = band_index(pixLo.y - y0, invH, R);
  let ri1 = band_index(pixHi.y - y0, invH, R);
  var ink : f32 = 0.0;
  for (var ri = ri0; ri <= ri1; ri = ri + 1u) {
    let rIdx = (rowBase + ri) * ROW_STRIDE;
    let e = band_edges(y0, ri, bandH);
    let oy = overlap1d(pixLo.y, pixHi.y, e.x, e.y);
    let ox = overlap1d(pixLo.x, pixHi.x, bitcast<f32>(rows[rIdx + ROW_XMIN]), bitcast<f32>(rows[rIdx + ROW_XMAX]));
    ink += bitcast<f32>(rows[rIdx + ROW_DENSITY]) * oy * ox;
  }
  return ink;
}

// Signed winding W and crossing count K of a +x ray from the rc-relative point `pr` (EXACT_MODE only).
fn winding_at(band : vec4f, y0 : f32, rc : vec2f, pr : vec2f) -> vec2i {
  let ri = band_index(pr.y + rc.y - y0, band.w, u32(band.y));
  let rIdx = (u32(band.x) + ri) * ROW_STRIDE;
  let start = rows[rIdx];
  let count = rows[rIdx + 1u];
  var W : i32 = 0;
  var K : i32 = 0;
  for (var i : u32 = 0u; i < count; i = i + 1u) {
    let base = (start + i) * 3u;
    let q1 = curves[base] - rc;
    let q3 = curves[base + 2u] - rc;
    let rising = q3.y > q1.y;
    // half-open: joins count once, extrema not at all
    if (pr.y < min(q1.y, q3.y) || pr.y >= max(q1.y, q3.y)) { continue; }
    let q2 = curves[base + 1u] - rc;
    let a2 = q1 - 2.0 * q2 + q3;
    let a1 = 2.0 * (q2 - q1);
    let t = mono_root(a2.y, a1.y, q1.y, q3.y, pr.y, rising);
    let x = (a2.x * t + a1.x) * t + q1.x;
    let hit = x > pr.x;
    K = K + select(0, 1, hit);
    W = W + select(0, select(-1, 1, rising), hit);
  }
  return vec2i(W, K);
}

// Fraction of an EXACT_GRID² grid over the pixel footprint whose true winding satisfies the fill rule.
fn exact_coverage(band : vec4f, y0 : f32, fillRule : f32, rc : vec2f, s : vec2f) -> f32 {
  let inv = 1.0 / f32(EXACT_GRID);
  let evenodd = fillRule > 0.5;
  var inside : u32 = 0u;
  for (var j : u32 = 0u; j < EXACT_GRID; j = j + 1u) {
    for (var i : u32 = 0u; i < EXACT_GRID; i = i + 1u) {
      let off = (vec2f(f32(i), f32(j)) + 0.5) * inv - 0.5;
      let wk = winding_at(band, y0, rc, off * s);
      let hit = select(wk.x != 0, (wk.y & 1) == 1, evenodd);
      inside = inside + select(0u, 1u, hit);
    }
  }
  return f32(inside) / f32(EXACT_GRID * EXACT_GRID);
}

@fragment
fn fs(in : VsOut) -> @location(0) vec4f {
  let I = instances[in.inst];
  let rc = in.rc;
  // units_per_pixel from the screen-space gradients — the device pixel's preimage under scale/translation.
  let s = max(fwidth(rc), vec2f(1e-9));

  if (EXACT_MODE) {
    let cov = exact_coverage(I.band, I.bbox.y, I.place.w, rc, s);
    return shade(I.color, style_coverage(cov, U.style.x, U.style.y));
  }

  if (MINIFICATION_GUARD && all(s * GUARD_PX >= I.bbox.zw - I.bbox.xy)) {
    return fold_shade(profile_face(I.band, I.bbox, rc, s) / (s.x * s.y), I.place.w, I.color);
  }
  return fold_shade(integrate_face(I.band, I.bbox.y, rc, s) / (s.x * s.y), I.place.w, I.color);
}
