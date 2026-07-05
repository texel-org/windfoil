// windfoil-variable.wgsl — windfoil's box-filter coverage, but each shape carries a per-anchor COLOR and
// BLUR field instead of one flat color and one crisp edge. Two things vary continuously across a single fill:
//
//   • color — each anchor holds an OKLab color; a pixel blends the shape's anchors by an inverse-distance
//     (Shepard) weight, so the surface reads as an organic mesh-gradient rather than a linear ramp. The
//     blended OKLab is taken to sRGB in-shader with the SAME matrices texel/color uses on the CPU (passed as
//     a uniform), so a single-color shape is bit-equivalent to the flat pipeline.
//   • blur — each anchor holds a blur_scale in [0,1]; the same Shepard blend gives a per-pixel blur_scale,
//     and the box filter simply widens: s_eff = s + blur_scale·max_blur. A wider box IS a wider box blur
//     (ALGORITHM.md §2 — the integrand already takes the box size), so one shape runs from crisp to soft
//     around itself with no new integration path. max_blur is in shape units, so the softness is
//     resolution-independent and scale-invariant like the rest of windfoil.
//
// This is the exact analytic gather only — no minification guard, no approximation tiers (see docs/VARIABLE.md).
// It is built for static images and will get slower as heavier kernels drop in; that is by design.
// The winding integral itself (mono_root / integrate_* / the row-band gather) is copied verbatim from
// windfoil.wgsl — only the box size is now per-pixel and the shade is the anchor blend. Math: docs/ALGORITHM.md.

struct Uniforms {
  res : vec2<f32>,           // render-target size in pixels
  style : vec2<f32>,         // (gamma, sharp) coverage transform; (1, 1) = exact
  cam : vec4<f32>,           // camera: device px = worldPx·(scaleX, scaleY) + (transX, transY)
  okToLms : mat3x3<f32>,     // OKLab → LMS' (texel/color OKLab_to_LMS_M, columns padded to vec4)
  lmsToRgb : mat3x3<f32>,    // LMS  → linear sRGB (texel/color LMS_to_linear_sRGB_M)
};

struct Instance {
  place : vec4<f32>, // originX, originY (device px), unitsToPx, fillRule (0 = nonzero, 1 = even-odd)
  bbox  : vec4<f32>, // ink box loX, loY, hiX, hiY (shape units, Y-down)
  blur  : vec4<f32>, // maxBlur (shape units), falloffPower, anchorBase, anchorCount
  band  : vec4<f32>, // rowBase, bandCount, y0, invH
};

// One on-curve anchor: a position in shape space plus the (color, blur) it contributes to the field.
struct Anchor {
  pos : vec4<f32>,   // posX, posY (shape units), blurScale [0,1], alpha (straight)
  lab : vec4<f32>,   // OKLab L, a, b, (unused)
};

// Bands with count > SORT_MIN are x-sorted on the CPU so the gather can break at the first piece fully left
// of the box. MUST equal BAND_SORT_MIN in bands.js.
const SORT_MIN : u32 = 4u;

// Row-table layout — MUST match bands.js's rowOut.push(start, count, areaBits, xMinBits, xMaxBits). The area /
// hull fields ride along for free (they're the guard's, unused here) so bands.js stays shared, unchanged.
const ROW_STRIDE : u32 = 5u;

// Shepard falloff smoothing: near an anchor the inverse-distance weight would blow up; clamp the distance to
// this fraction of the shape's bbox diagonal so the field stays finite and the anchor's color spreads into a
// small neighborhood instead of a single hot texel. Larger = smoother/flatter blend.
const EPS_FRAC : f32 = 0.06;

@group(0) @binding(0) var<uniform> U : Uniforms;
@group(0) @binding(1) var<storage, read> instances : array<Instance>;
// Curve atlas: three consecutive vec2 per xy-monotone piece (endpoints + control).
@group(0) @binding(2) var<storage, read> curves : array<vec2<f32>>;
// Row-band table: ROW_STRIDE u32s per band (see bands.js).
@group(0) @binding(3) var<storage, read> rows : array<u32>;
// Anchor field: two vec4 per anchor (see Anchor), indexed per instance by blur.zw.
@group(0) @binding(4) var<storage, read> anchors : array<Anchor>;

struct VsOut {
  @builtin(position) pos : vec4<f32>,
  @location(0) rc : vec2<f32>,                       // shape-space position of this fragment (Y-down)
  @location(1) @interpolate(flat) inst : u32,
};

@vertex
fn vs(@builtin(vertex_index) vi : u32, @builtin(instance_index) ii : u32) -> VsOut {
  let I = instances[ii];
  let unitsToPx = I.place.z;
  let camScale = U.cam.xy;
  // Pad the quad by the AA skirt (1 device px) PLUS the blur skirt: a fully-blurred edge's box reaches
  // maxBlur/2 past the ink, so the widened coverage ramp needs that much extra room or it clips.
  let pad = 1.0 / (unitsToPx * max(camScale.x, 1e-6)) + I.blur.x * 0.5;
  let lo = I.bbox.xy - vec2<f32>(pad);
  let hi = I.bbox.zw + vec2<f32>(pad);
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

// sRGB transfer function (linear → gamma-encoded), matching texel/color's sRGB space and IEC 61966-2-1. Runs
// per channel on a value already clamped to [0,1] (pow of a negative base is undefined in WGSL).
fn srgb_encode(c : vec3<f32>) -> vec3<f32> {
  let lo = c * 12.92;
  let hi = 1.055 * pow(c, vec3<f32>(1.0 / 2.4)) - 0.055;
  return select(hi, lo, c <= vec3<f32>(0.0031308));
}

// OKLab → sRGB with the uniform matrices, mirroring texel/color exactly: linear map to LMS', cube back to
// LMS (sign-preserving, so an explicit product not pow()), linear map to linear sRGB, then the sRGB curve.
fn oklab_to_srgb(lab : vec3<f32>) -> vec3<f32> {
  let lms_ = U.okToLms * lab;
  let lms = lms_ * lms_ * lms_;
  let rgb_lin = clamp(U.lmsToRgb * lms, vec3<f32>(0.0), vec3<f32>(1.0));
  return srgb_encode(rgb_lin);
}

// The per-pixel anchor field: an inverse-distance (Shepard) blend of the shape's anchors. Weight
// w = 1/(d² + eps²)^(p/2) is smooth everywhere (no nearest-anchor seam) and, with a small eps, pulls each
// anchor's value toward it — so color and blur read as an organic field. Returns (OKLab, blurScale, alpha).
struct Field {
  lab : vec3<f32>,
  blur : f32,
  alpha : f32,
};
fn sample_field(base : u32, count : u32, rc : vec2<f32>, power : f32, eps : f32) -> Field {
  var wSum : f32 = 0.0;
  var labSum : vec3<f32> = vec3<f32>(0.0);
  var blurSum : f32 = 0.0;
  var alphaSum : f32 = 0.0;
  let eps2 = eps * eps;
  let half_p = 0.5 * power;
  for (var i : u32 = 0u; i < count; i = i + 1u) {
    let A = anchors[base + i];
    let d = rc - A.pos.xy;
    let w = pow(dot(d, d) + eps2, -half_p);
    wSum += w;
    labSum += w * A.lab.xyz;
    blurSum += w * A.pos.z;
    alphaSum += w * A.pos.w;
  }
  let inv = 1.0 / max(wSum, 1e-30);
  var f : Field;
  f.lab = labSum * inv;
  f.blur = blurSum * inv;
  f.alpha = alphaSum * inv;
  return f;
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

// Accumulate one row band's pieces over the rc-relative y-window [wlo, whi]. `sx` is the (per-pixel) box width.
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
    // f32-degenerate piece (hull within a few coordinate-ULPs): the t-solves would divide noise; use the
    // midpoint-clamp form, exact to ~span² and telescoping.
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

// One shape's winding integral over the pixel box (rc ± s/2), gathered through the row bands its y-slab
// touches. `s` is the PER-PIXEL box size (footprint widened by the local blur), so a blurrier pixel selects a
// taller slab and a wider x-reach automatically — the band additivity (ALGORITHM.md §6) is unchanged.
fn integrate_face(band : vec4<f32>, rc : vec2<f32>, s : vec2<f32>) -> f32 {
  let rowBase = u32(band.x);
  let R = u32(band.y);
  let invH = band.w;
  let sy2 = s.y * 0.5;
  let dy0 = band.z - rc.y;      // band origin y0, relative to the pixel center
  var ri0 : u32 = 0u;
  var ri1 : u32 = 0u;
  if (invH > 0.0) {             // invH > 0 only for multi-band shapes (bands.js stores 0 when R == 1)
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

@fragment
fn fs(in : VsOut) -> @location(0) vec4<f32> {
  let I = instances[in.inst];
  let rc = in.rc;
  // units_per_pixel from the screen-space gradients — the device pixel's preimage under scale/translation.
  let s = max(fwidth(rc), vec2<f32>(1e-9));

  // Resolve the per-pixel field (color + blur) from the shape's anchors first: the blur it returns sets the
  // box size the winding integral runs at.
  let eps = length(I.bbox.zw - I.bbox.xy) * EPS_FRAC;
  let field = sample_field(u32(I.blur.z), u32(I.blur.w), rc, I.blur.y, eps);

  // Widen the box by the local softness. maxBlur is in shape units, so a wider s_eff is a wider box blur that
  // scales with the shape (blur_scale·maxBlur added to the crisp 1-pixel footprint).
  let s_eff = s + vec2<f32>(field.blur * I.blur.x);

  let F = integrate_face(I.band, rc, s_eff) / (s_eff.x * s_eff.y);
  var cov : f32;
  if (I.place.w > 0.5) {
    cov = tri_wave(F);                 // even-odd
  } else {
    cov = clamp(abs(F), 0.0, 1.0);     // nonzero (saturating)
  }
  cov = style_coverage(cov, U.style.x, U.style.y);

  // Straight-alpha OKLab color × coverage → premultiplied RGBA (the pipeline blends premultiplied-over).
  let rgb = oklab_to_srgb(field.lab);
  let a = field.alpha * cov;
  return vec4<f32>(rgb * a, a);
}
