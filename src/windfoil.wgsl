// windfoil.wgsl — box-filter coverage as a per-pixel winding integral. Math + derivation: docs/ALGORITHM.md.
// The vertex stage expands a unit quad per instance; the fragment stage integrates the winding number over
// the pixel's footprint in closed form (F = ∫∫_box w dA / area) and folds it to coverage.
//
// The gather is built for warp coherence: pieces are stored band-CLIPPED (tight per-band hulls), split into
// an F segment (spans its whole band) and an E segment (has an endpoint inside the band), each with an
// 8-byte packed cull word per piece. A piece either culls on that word or takes the one branchless exact
// integral — the fully-right F prefix collapses to a single signed-count multiply first.

struct Uniforms {
  res : vec2<f32>,    // render-target size in pixels
  style : vec2<f32>,  // (gamma, sharp) coverage transform; (1, 1) = exact
  cam : vec4<f32>,    // camera: device px = worldPx·(scaleX, scaleY) + (transX, transY)
};

struct Instance {
  place : vec4<f32>, // originX, originY (device px), unitsToPx, fillRule (0 = nonzero, 1 = even-odd)
  bbox  : vec4<f32>, // ink box loX, loY, hiX, hiY (glyph units, Y-down)
  color : vec4<f32>, // straight-alpha RGBA
  band  : vec4<f32>, // rowBase (vec4 units), bandCount, bandH, invH
};

// Row-table layout — TWO vec4<u32> per band (MUST match bands.js):
//   H0 = (start, fCount, eCount, wF) : start = vec2-index of the band's F-meta block in `curves`;
//        fCount/eCount = full-band-span / endpoint piece counts; wF = bit-punned f32 max F hull width.
//   H1 = (density, xMin, xMax, 0)   : bit-punned f32 banded-ink guard profile (profile_face only).
// A band's record in `curves`: metaF[fCount+1] · metaE[eCount] · piece data (3 vec2 per piece, F then E).
// Meta words hold conservative f16 hulls (lows rounded down, highs up), so a cull can never drop a piece
// that matters — borderline pieces just fall through to the exact integral.

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
const KERNEL_SUPPORT_PX = vec2<f32>(0.5);
const KERNEL_SKIRT_PX = KERNEL_SUPPORT_PX + vec2<f32>(0.125);

@group(0) @binding(0) var<uniform> U : Uniforms;
@group(0) @binding(1) var<storage, read> instances : array<Instance>;
// Curve atlas: per band, the packed meta words then three consecutive vec2 per xy-monotone clipped piece.
@group(0) @binding(2) var<storage, read> curves : array<vec2<f32>>;
// Row-band table: two vec4<u32> per band (see the layout note above and bands.js).
@group(0) @binding(3) var<storage, read> rows : array<vec4<u32>>;

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
    cov = min(abs(f), 1.0);           // nonzero (saturating)
  }
  return shade(color, style_coverage(cov, U.style.x, U.style.y));
}

// Solve A2·t² + A1·t + A0 = v for t∈[0,1], saturating at a0=q(0) and e1=q(1). (EXACT_MODE's winding_at.)
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
  let t = select(0.0, num / select(1.0, den, valid), valid);
  return clamp(t, 0.0, 1.0);
}

// Exact INSIDE integral: expand cubic (x(t)+hx)·y′(t) about the interval midpoint; x0 is x(t)'s constant.
fn integrate_inside(a2 : vec2<f32>, a1 : vec2<f32>, x0 : f32, ta : f32, tb : f32, hx : f32) -> f32 {
  if (tb <= ta) { return 0.0; }
  let tm = 0.5 * (ta + tb);
  let d = 0.5 * (tb - ta);
  let x_mid = (a2.x * tm + a1.x) * tm + x0 + hx;
  let dmid = 2.0 * a2 * tm + a1;                       // (x′, y′) at the midpoint
  return 2.0 * d * x_mid * dmid.y + (2.0 * d * d * d / 3.0) * (a2.x * dmid.y + 2.0 * a2.y * dmid.x);
}

// Integrate clamp(x(t), −hx, hx)+hx over the y-window through LEFT (0), INSIDE (exact), and RIGHT
// (full-width) zones (ALGORITHM.md §3). All four clip roots (y-window entry/exit, x = ∓hx) are solved as
// one branchless vec4 batch — the same stable q-form and saturation semantics as mono_root, component-wise.
fn integrate_piece(q1 : vec2<f32>, q2 : vec2<f32>, q3 : vec2<f32>, lo : f32, hi : f32, hx : f32) -> f32 {
  let a2 = q1 - 2.0 * q2 + q3;
  let a1 = 2.0 * (q2 - q1);
  let y_rising = q3.y >= q1.y;
  let x_rising = q3.x >= q1.x;
  let vy = select(vec2<f32>(hi, lo), vec2<f32>(lo, hi), y_rising); // y levels in sweep order (entry, exit)

  let A2 = vec4<f32>(a2.y, a2.y, a2.x, a2.x);
  let A1 = vec4<f32>(a1.y, a1.y, a1.x, a1.x);
  let A0 = vec4<f32>(q1.y, q1.y, q1.x, q1.x);
  let E1 = vec4<f32>(q3.y, q3.y, q3.x, q3.x);
  let V  = vec4<f32>(vy.x, vy.y, -hx, hx);
  let R  = vec4<bool>(y_rising, y_rising, x_rising, x_rising);
  let SG = select(vec4<f32>(-1.0), vec4<f32>(1.0), R);

  let C = A0 - V;
  let sat0 = (C * SG) >= vec4<f32>(0.0);        // starts at/past the level → t = 0
  let sat1 = ((E1 - V) * SG) <= vec4<f32>(0.0); // never reaches the level → t = 1
  let disc = max(A1 * A1 - 4.0 * A2 * C, vec4<f32>(0.0));
  let sq = sqrt(disc);
  let qq = -0.5 * (A1 + select(-sq, sq, A1 >= vec4<f32>(0.0)));
  // For an xy-monotone piece sign(a1) matches the sweep direction (or a1 = 0 at a vertex start), so the
  // stable branch is always den = qq, whose magnitude is ≥ |a1|/2 — near-linear pieces need no special
  // case here (a2 = 0 exactly folds to −c/a1 through the same expressions).
  let use_r1 = (A1 < vec4<f32>(0.0)) == R;
  let num = select(C, qq, use_r1);
  let den = select(qq, A2, use_r1);
  let valid = den != vec4<f32>(0.0);
  var T = clamp(
    select(vec4<f32>(0.0), num / select(vec4<f32>(1.0), den, valid), valid),
    vec4<f32>(0.0), vec4<f32>(1.0),
  );
  T = select(T, vec4<f32>(1.0), sat1);
  T = select(T, vec4<f32>(0.0), sat0);

  let t_lo = T.x;
  let t_hi = T.y;
  if (t_hi <= t_lo) { return 0.0; }
  let t_left = clamp(T.z, t_lo, t_hi);
  let t_right = clamp(T.w, t_lo, t_hi);
  // Zones in sweep order: x rising ⇒ LEFT · INSIDE · RIGHT; mirrored if not.
  let t1 = select(t_right, t_left, x_rising);
  let t2 = max(select(t_left, t_right, x_rising), t1);
  var acc = integrate_inside(a2, a1, q1.x, t1, t2, hx);
  let ra = select(t_lo, t2, x_rising);
  let rb = select(t1, t_hi, x_rising);
  let d = max(rb - ra, 0.0);
  let tm = 0.5 * (ra + rb);
  acc += d * (2.0 * a2.y * tm + a1.y) * (2.0 * hx);   // RIGHT zone: full width × Δy (d = 0 when empty)
  return acc;
}

// Signed clipped y-span; endpoint clamps make adjacent pieces telescope exactly.
fn clipped_dy(y1 : f32, y3 : f32, wlo : f32, whi : f32) -> f32 {
  return clamp(y3, wlo, whi) - clamp(y1, wlo, whi);
}

// First index in the descending-f16-xmin metadata run [base+lo0, base+hi0) whose xmin, taken relative to
// rc.x, drops below `level` — everything before it is certainly right of `level`. The f16 keys are stored
// weakly descending, so a binary search replaces a serial dependent-load walk.
fn meta_lower_bound(base : u32, stride : u32, lo0 : u32, hi0 : u32, rcx : f32, level : f32) -> u32 {
  var lo = lo0;
  var hi = hi0;
  while (lo < hi) {
    let mid = (lo + hi) >> 1u;
    let xmn = unpack2x16float(bitcast<u32>(curves[base + mid * stride].x)).x;
    if (xmn - rcx >= level) { lo = mid + 1u; } else { hi = mid; }
  }
  return lo;
}

// Accumulate one row band's pieces over the rc-relative y-window [wlo, whi]. `covered` = the window spans
// the whole band, which lets the E segment's fully-right prefix aggregate through precomputed span sums.
fn integrate_band(h0 : vec4<u32>, rc : vec2<f32>, wlo : f32, whi : f32, sx : f32, covered : bool) -> f32 {
  let start = h0.x;
  let fCount = h0.y;
  let eCount = h0.z;
  let wFE = unpack2x16float(h0.w);                      // (max F hull width, max E hull width)
  let hx = sx * 0.5;
  let oy = whi - wlo;
  var acc : f32 = 0.0;

  // ── F segment (sorted by hull x-min, desc). Every F piece spans the whole band, so — when fully right of
  // the box — it contributes exactly ±(window height) under ANY window: the fully-right prefix [0, kF)
  // collapses to one signed-count multiply. Beyond bF every piece is certainly fully left (x-min below the
  // box minus the band's max hull width), so only [kF, bF) is ever touched per-piece.
  let kF = meta_lower_bound(start, 1u, 0u, fCount, rc.x, hx);
  acc += sx * oy * curves[start + kF].y;                // signed count of the skipped prefix
  let pieceBase = start + fCount + 1u + 2u * (eCount + 1u);
  for (var i = kF; i < fCount; i = i + 1u) {
    let cx = unpack2x16float(bitcast<u32>(curves[start + i].x));
    if (cx.y - rc.x <= -hx) {                           // fully left → no area
      if (cx.x - rc.x < -hx - wFE.x) { break; }
      continue;
    }
    let b = pieceBase + i * 3u;
    acc += integrate_piece(curves[b] - rc, curves[b + 1u] - rc, curves[b + 2u] - rc, wlo, whi, hx);
  }

  // ── E segment (sorted by hull x-min, desc): the fully-right prefix [0, kE) needs each piece's exact
  // y-span — a covered window takes it as one precomputed span-sum read (P_k rides in meta word 0), a
  // partial window as a branchless run of one-load clipped-span adds (the exact endpoint y's are meta
  // word 1). Survivors of [kE, bE) cull on the packed hull or the exact y-span, else integrate exactly.
  let eBase = start + fCount + 1u;
  let kE = meta_lower_bound(eBase, 2u, 0u, eCount, rc.x, hx);
  let pB = pieceBase + fCount * 3u;
  if (covered) {
    acc += sx * curves[eBase + kE * 2u].y;              // P_k: span sum of the skipped prefix
  } else {
    for (var j = 0u; j < kE; j = j + 1u) {
      let ys = curves[eBase + j * 2u + 1u];
      acc += sx * clipped_dy(ys.x - rc.y, ys.y - rc.y, wlo, whi);
    }
  }
  for (var j = kE; j < eCount; j = j + 1u) {
    let cx = unpack2x16float(bitcast<u32>(curves[eBase + j * 2u].x));
    if (cx.y - rc.x <= -hx) {                           // fully left → no area
      if (cx.x - rc.x < -hx - wFE.y) { break; }
      continue;
    }
    let ys = curves[eBase + j * 2u + 1u];
    let ylo = min(ys.x, ys.y) - rc.y;
    let yhi = max(ys.x, ys.y) - rc.y;
    if (yhi <= wlo || ylo >= whi) { continue; }          // y-disjoint from the window
    let b = pB + j * 3u;
    acc += integrate_piece(curves[b] - rc, curves[b + 1u] - rc, curves[b + 2u] - rc, wlo, whi, hx);
  }
  return acc;
}

// Band index for a y-offset from the band origin: floor(dy·invH) clamped — the same mapping bands.js files with.
fn band_index(dy : f32, invH : f32, R : u32) -> u32 {
  return u32(clamp(floor(dy * invH), 0.0, f32(R) - 1.0));
}

// Band ri's y-range relative to `base`. R ≤ 64, so f32(ri) + 1.0 is exact.
fn band_edges(base : f32, ri : u32, bandH : f32) -> vec2<f32> {
  let r = f32(ri);
  return vec2<f32>(base) + vec2<f32>(r, r + 1.0) * bandH;
}

// Length of the overlap of intervals [a0, a1] and [b0, b1] (0 when disjoint).
fn overlap1d(a0 : f32, a1 : f32, b0 : f32, b1 : f32) -> f32 {
  return max(min(a1, b1) - max(a0, b0), 0.0);
}

// One glyph's winding integral over the pixel box (rc ± s/2), gathered through the row bands its y-slab
// touches. Windows are kept rc-RELATIVE for deep-zoom stability, and tile exactly across bands so clipped
// pieces never double-count (ALGORITHM.md §6).
fn integrate_face(band : vec4<f32>, y0 : f32, rc : vec2<f32>, s : vec2<f32>) -> f32 {
  let rowBase = u32(band.x);
  let R = u32(band.y);
  let bandH = band.z;
  let invH = band.w;
  let sy2 = s.y * 0.5;
  let dy0 = y0 - rc.y;          // band origin relative to the pixel center
  var ri0 : u32 = 0u;
  var ri1 : u32 = 0u;
  if (R > 1u) {
    ri0 = band_index(-dy0 - sy2, invH, R);
    ri1 = band_index(-dy0 + sy2, invH, R);
  }
  var f_int : f32 = 0.0;
  for (var ri = ri0; ri <= ri1; ri = ri + 1u) {
    var w_lo = -sy2;
    var w_hi = sy2;
    var covered = false;
    if (R > 1u) {
      let e = band_edges(dy0, ri, bandH);
      covered = w_lo <= e.x && w_hi >= e.y;
      w_lo = max(w_lo, e.x);
      w_hi = min(w_hi, e.y);
    }
    if (w_hi <= w_lo) { continue; }
    f_int += integrate_band(rows[rowBase + 2u * ri], rc, w_lo, w_hi, s.x, covered);
  }
  return f_int;
}

// Approximate minification twin of integrate_face: integrate each band's precomputed winding density over
// its overlap with the pixel. A few table taps, no curve reads.
fn profile_face(band : vec4<f32>, bbox : vec4<f32>, rc : vec2<f32>, s : vec2<f32>) -> f32 {
  let pixLo = rc - s * 0.5;
  let pixHi = rc + s * 0.5;
  if (overlap1d(pixLo.x, pixHi.x, bbox.x, bbox.z) <= 0.0) { return 0.0; }
  let rowBase = u32(band.x);
  let R = u32(band.y);
  let bandH = band.z;
  let invH = band.w;
  let y0 = bbox.y;
  var ri0 : u32 = 0u;
  var ri1 : u32 = 0u;
  if (R > 1u) {
    ri0 = band_index(pixLo.y - y0, invH, R);
    ri1 = band_index(pixHi.y - y0, invH, R);
  }
  var ink : f32 = 0.0;
  for (var ri = ri0; ri <= ri1; ri = ri + 1u) {
    let h1 = rows[rowBase + 2u * ri + 1u];
    let e = band_edges(y0, ri, bandH);
    let oy = overlap1d(pixLo.y, pixHi.y, e.x, e.y);
    let ox = overlap1d(pixLo.x, pixHi.x, bitcast<f32>(h1.y), bitcast<f32>(h1.z));
    ink += bitcast<f32>(h1.x) * oy * ox;
  }
  return ink;
}

// Signed winding W and crossing count K of a +x ray from the rc-relative point `pr` (EXACT_MODE only).
fn winding_at(band : vec4<f32>, y0 : f32, rc : vec2<f32>, pr : vec2<f32>) -> vec2<i32> {
  let rowBase = u32(band.x);
  let R = u32(band.y);
  let invH = band.w;
  var ri : u32 = 0u;
  if (R > 1u) { ri = band_index(pr.y + rc.y - y0, invH, R); }
  let h0 = rows[rowBase + 2u * ri];
  let count = h0.y + h0.z;
  let pieceBase = h0.x + h0.y + 1u + 2u * (h0.z + 1u);
  var W : i32 = 0;
  var K : i32 = 0;
  for (var i : u32 = 0u; i < count; i = i + 1u) {
    let base = pieceBase + i * 3u;
    let q1 = curves[base] - rc;
    let q3 = curves[base + 2u] - rc;
    let rising = q3.y > q1.y;
    let ylo = min(q1.y, q3.y);
    let yhi = max(q1.y, q3.y);
    if (pr.y < ylo || pr.y >= yhi) { continue; } // half-open: joins count once, extrema not at all
    let q2 = curves[base + 1u] - rc;
    let a2 = q1 - 2.0 * q2 + q3;
    let a1 = 2.0 * (q2 - q1);
    let t = mono_root(a2.y, a1.y, q1.y, q3.y, pr.y, rising);
    let x = (a2.x * t + a1.x) * t + q1.x;
    if (x > pr.x) {
      K = K + 1;
      W = W + select(-1, 1, rising);
    }
  }
  return vec2<i32>(W, K);
}

// Fraction of an EXACT_GRID² grid over the pixel footprint whose true winding satisfies the fill rule.
fn exact_coverage(band : vec4<f32>, y0 : f32, fillRule : f32, rc : vec2<f32>, s : vec2<f32>) -> f32 {
  let inv = 1.0 / f32(EXACT_GRID);
  let evenodd = fillRule > 0.5;
  var inside : u32 = 0u;
  for (var j : u32 = 0u; j < EXACT_GRID; j = j + 1u) {
    for (var i : u32 = 0u; i < EXACT_GRID; i = i + 1u) {
      let off = (vec2<f32>(f32(i), f32(j)) + 0.5) * inv - 0.5;
      let wk = winding_at(band, y0, rc, off * s);
      let hit = select(wk.x != 0, (wk.y & 1) == 1, evenodd);
      inside = inside + select(0u, 1u, hit);
    }
  }
  return f32(inside) / f32(EXACT_GRID * EXACT_GRID);
}

@fragment
fn fs(in : VsOut) -> @location(0) vec4<f32> {
  let I = instances[in.inst];
  let rc = in.rc;
  // units_per_pixel from the screen-space gradients — the device pixel's preimage under scale/translation.
  let s = max(fwidth(rc), vec2<f32>(1e-9));

  if (EXACT_MODE) {
    let cov = exact_coverage(I.band, I.bbox.y, I.place.w, rc, s);
    return shade(I.color, style_coverage(cov, U.style.x, U.style.y));
  }

  if (MINIFICATION_GUARD && all(s * GUARD_PX >= I.bbox.zw - I.bbox.xy)) {
    return fold_shade(profile_face(I.band, I.bbox, rc, s) / (s.x * s.y), I.place.w, I.color);
  }
  return fold_shade(integrate_face(I.band, I.bbox.y, rc, s) / (s.x * s.y), I.place.w, I.color);
}
