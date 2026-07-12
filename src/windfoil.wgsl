// windfoil.wgsl — box-filter coverage as a per-pixel winding integral. Math + derivation: docs/ALGORITHM.md.
// The vertex stage expands a unit quad per instance; the fragment stage integrates the winding number over
// the pixel's footprint in closed form (F = ∫∫_box w dA / area) and folds it to coverage.

struct Uniforms {
  res : vec2<f32>,    // render-target size in pixels
  style : vec2<f32>,  // (gamma, sharp) coverage transform; (1, 1) = exact
  cam : vec4<f32>,    // camera: device px = worldPx·(scaleX, scaleY) + (transX, transY)
};

struct Instance {
  place : vec4<f32>, // originX, originY (device px), unitsToPx, fillRule (0 = nonzero, 1 = even-odd)
  bbox  : vec4<f32>, // ink box loX, loY, hiX, hiY (glyph units, Y-down)
  color : vec4<f32>, // straight-alpha RGBA
  band  : vec4<f32>, // rowBase, bandCount, y0, invH
};

// Bands with count > SORT_MIN are x-sorted on the CPU so the gather can break at the first piece fully left
// of the box. MUST equal BAND_SORT_MIN in bands.js.
const SORT_MIN : u32 = 4u;

// Row-table layout — MUST match bands.js's rowOut.push(start, count, areaBits, xMinBits, xMaxBits).
const ROW_STRIDE : u32 = 5u;
const ROW_AREA : u32 = 2u;
const ROW_XMIN : u32 = 3u;
const ROW_XMAX : u32 = 4u;

// Below GUARD_PX device pixels (whole glyph, both axes — illegible sizes) coverage comes from the banded ink
// profile (profile_face) instead of the exact gather. Threshold rationale + the quality/perf trade of raising
// it: bench/README.md, bench/ACCEL-NOTES.md.
const MINIFICATION_GUARD = true;
const GUARD_PX = 3.7;

// Kernel support plus 0.125px derivative slack; adjust support per axis for other kernels.
const KERNEL_SUPPORT_PX = vec2<f32>(0.5);
const KERNEL_SKIRT_PX = KERNEL_SUPPORT_PX + vec2<f32>(0.125);

@group(0) @binding(0) var<uniform> U : Uniforms;
@group(0) @binding(1) var<storage, read> instances : array<Instance>;
// Curve atlas: three consecutive vec2 per xy-monotone piece (endpoints + control).
@group(0) @binding(2) var<storage, read> curves : array<vec2<f32>>;
// Row-band table: ROW_STRIDE u32s per band (see the ROW_* constants and bands.js).
@group(0) @binding(3) var<storage, read> rows : array<u32>;

struct VsOut {
  @builtin(position) pos : vec4<f32>,
  @location(0) rc : vec2<f32>,                       // glyph-space position of this fragment (Y-down)
  @location(1) @interpolate(flat) inst : u32,
};

@vertex
fn vs(@builtin(vertex_index) vi : u32, @builtin(instance_index) ii : u32) -> VsOut {
  let I = instances[ii];
  let unitsToPx = I.place.z;
  let camScale = U.cam.xy;
  let pad = KERNEL_SKIRT_PX / (unitsToPx * max(abs(camScale), vec2<f32>(1e-6)));
  let lo = I.bbox.xy - pad;
  let hi = I.bbox.zw + pad;
  // Unit-quad corners for a triangle-strip; vi ∈ {0..3}.
  let uv = vec2<f32>(f32(vi & 1u), f32(vi >> 1u));
  let em = mix(lo, hi, uv);
  let worldPx = I.place.xy + em * unitsToPx;
  let devicePx = worldPx * camScale + U.cam.zw;
  let ndc = devicePx / U.res * 2.0 - 1.0;             // Y-down NDC; clip space flips y
  var o : VsOut;
  o.pos = vec4<f32>(ndc.x, -ndc.y, 0.0, 1.0);
  o.rc = em;
  o.inst = ii;
  return o;
}

// Period-2 triangle wave 1 − |1 − (t mod 2)|: folds signed winding to even-odd coverage, range [0, 1].
fn tri_wave(t : f32) -> f32 {
  let m = t - 2.0 * floor(t * 0.5);
  return 1.0 - abs(1.0 - m);
}

// Opt-in perceptual styling (--gamma / --sharp); (1, 1) leaves the exact coverage bit-for-bit untouched.
fn style_coverage(cov : f32, gamma : f32, sharp : f32) -> f32 {
  if (gamma == 1.0 && sharp == 1.0) { return cov; }
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

// Shared fragment tail: fold the pixel-averaged winding number by fill rule, style, shade.
fn fold_shade(f : f32, fillRule : f32, color : vec4<f32>) -> vec4<f32> {
  var cov : f32;
  if (fillRule > 0.5) {
    cov = tri_wave(f);                // even-odd
  } else {
    cov = clamp(abs(f), 0.0, 1.0);    // nonzero (saturating)
  }
  return shade(color, style_coverage(cov, U.style.x, U.style.y));
}

// Solve the monotone quadratic component A2·t² + A1·t + A0 = v on [0,1], saturating to the endpoints
// (a0 = value at t = 0, e1 = value at t = 1).
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
  let r1 = qq / a2;
  let r2 = select(0.0, c / qq, qq != 0.0);
  // The derivative at r1 is −sign(a1)·sq, so the branch pick reduces to a sign test on a1.
  let t = select(r2, r1, (a1 < 0.0) == rising);
  return clamp(t, 0.0, 1.0);
}

// The INSIDE zone's exact integral of (x(t)+hx)·y′(t) over [ta,tb]: midpoint rule on a symmetric interval,
// exact for this cubic integrand. `x0` is the piece's constant x coefficient.
fn integrate_inside(a2 : vec2<f32>, a1 : vec2<f32>, x0 : f32, ta : f32, tb : f32, hx : f32) -> f32 {
  if (tb <= ta) { return 0.0; }
  let tm = 0.5 * (ta + tb);
  let d = 0.5 * (tb - ta);
  let x_mid = (a2.x * tm + a1.x) * tm + x0 + hx;
  let dmid = 2.0 * a2 * tm + a1;                       // (x′, y′) at the midpoint
  return 2.0 * d * x_mid * dmid.y + (2.0 * d * d * d / 3.0) * (a2.x * dmid.y + 2.0 * a2.y * dmid.x);
}

// One xy-monotone piece's contribution over the y-window [lo, hi]: integrate clamp(x(t), −hx, hx) + hx by
// splitting the crossing interval into LEFT (0) / INSIDE (exact) / RIGHT (full width) zones (ALGORITHM.md §3).
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
  // Zones in sweep order: x rising ⇒ LEFT · INSIDE · RIGHT; mirrored if not.
  let t1 = select(t_right, t_left, x_rising);
  let t2 = max(select(t_left, t_right, x_rising), t1);
  var acc = integrate_inside(a2, a1, q1.x, t1, t2, hx);
  let ra = select(t_lo, t2, x_rising);
  let rb = select(t1, t_hi, x_rising);
  if (rb > ra) {
    let tm = 0.5 * (ra + rb);
    acc += (rb - ra) * (2.0 * a2.y * tm + a1.y) * (2.0 * hx);   // RIGHT zone: full width × Δy
  }
  return acc;
}

// A piece's y-span clipped to the window, as a difference of clamped ENDPOINTS so it telescopes over piece
// chains (shared endpoints cancel exactly). Signed: clamp(y3) − clamp(y1).
fn clipped_dy(y1 : f32, y3 : f32, wlo : f32, whi : f32) -> f32 {
  return clamp(y3, wlo, whi) - clamp(y1, wlo, whi);
}

// Accumulate one row band's pieces over the rc-relative y-window [wlo, whi].
fn integrate_band(start : u32, count : u32, rc : vec2<f32>, wlo : f32, whi : f32, sx : f32) -> f32 {
  var acc : f32 = 0.0;
  let hx = sx * 0.5;
  let sorted = count > SORT_MIN;
  let coord_ulp = max(abs(rc.x), abs(rc.y)) * 1.2e-7;
  for (var i : u32 = 0u; i < count; i = i + 1u) {
    let base = (start + i) * 3u;
    let q1 = curves[base] - rc;
    let q2 = curves[base + 1u] - rc;
    let q3 = curves[base + 2u] - rc;
    let x_hull_max = max(q1.x, max(q2.x, q3.x));
    if (x_hull_max <= -hx) {              // fully LEFT of the box → no area
      if (sorted) { break; }
      continue;
    }
    let py_lo = min(q1.y, q3.y);          // piece y-span (endpoint-exact when monotone)
    let py_hi = max(q1.y, q3.y);
    let lo = max(wlo, py_lo);
    let hi = min(whi, py_hi);
    if (hi <= lo) { continue; }
    let x_hull_min = min(q1.x, min(q2.x, q3.x));
    if (x_hull_min >= hx) {               // fully RIGHT of the box → full width × clipped y-span
      acc += sx * clipped_dy(q1.y, q3.y, wlo, whi);
      continue;
    }
    // f32-degenerate piece (hull within a few coordinate-ULPs, e.g. flattened content at deep zoom): the
    // t-solves would divide noise; use the midpoint-clamp form, exact to ~span² and telescoping.
    if (x_hull_max - x_hull_min + (py_hi - py_lo) <= coord_ulp * 16.0) {
      let xm = clamp((q1.x + q3.x) * 0.5, -hx, hx) + hx;
      acc += xm * clipped_dy(q1.y, q3.y, wlo, whi);
      continue;
    }
    acc += integrate_piece(q1, q2, q3, lo, hi, hx);
  }
  return acc;
}

// Band index for a y-offset from the band origin: floor(dy·invH) clamped — the same mapping bands.js files with.
fn band_index(dy : f32, invH : f32, R : u32) -> u32 {
  return u32(clamp(floor(dy * invH), 0.0, f32(R) - 1.0));
}

// Band ri's y-range relative to `base`. R ≤ 64, so f32(ri) + 1.0 is exact.
fn band_edges(base : f32, ri : u32, invH : f32) -> vec2<f32> {
  let r = f32(ri);
  return vec2<f32>(base + r / invH, base + (r + 1.0) / invH);
}

// Length of the overlap of intervals [a0, a1] and [b0, b1] (0 when disjoint).
fn overlap1d(a0 : f32, a1 : f32, b0 : f32, b1 : f32) -> f32 {
  return max(min(a1, b1) - max(a0, b0), 0.0);
}

// One glyph's winding integral over the pixel box (rc ± s/2), gathered through the row bands its y-slab
// touches. Windows are kept rc-RELATIVE for deep-zoom stability, and tile exactly across bands so duplicated
// pieces never double-count (ALGORITHM.md §6).
fn integrate_face(band : vec4<f32>, rc : vec2<f32>, s : vec2<f32>) -> f32 {
  let rowBase = u32(band.x);
  let R = u32(band.y);
  let invH = band.w;
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
    var w_lo = -sy2;
    var w_hi = sy2;
    if (invH > 0.0) {
      let e = band_edges(dy0, ri, invH);
      w_lo = max(w_lo, e.x);
      w_hi = min(w_hi, e.y);
    }
    if (w_hi <= w_lo) { continue; }
    let rIdx = (rowBase + ri) * ROW_STRIDE;
    f_int += integrate_band(rows[rIdx], rows[rIdx + 1u], rc, w_lo, w_hi, s.x);
  }
  return f_int;
}

// The minification guard's twin of integrate_face: the same ∫∫_box w dA, approximated from the precomputed
// banded ink profile — each band's strip integral × the pixel's y-share of the strip × its x-share of the
// band's ink hull. A few table taps, no curve reads.
fn profile_face(band : vec4<f32>, bbox : vec4<f32>, rc : vec2<f32>, s : vec2<f32>) -> f32 {
  let pixLo = rc - s * 0.5;
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
  var ink : f32 = 0.0;
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
  // units_per_pixel from the screen-space gradients — the device pixel's preimage under scale/translation.
  let s = max(fwidth(rc), vec2<f32>(1e-9));

  if (MINIFICATION_GUARD && all(s * GUARD_PX >= I.bbox.zw - I.bbox.xy)) {
    return fold_shade(profile_face(I.band, I.bbox, rc, s) / (s.x * s.y), I.place.w, I.color);
  }
  return fold_shade(integrate_face(I.band, rc, s) / (s.x * s.y), I.place.w, I.color);
}
