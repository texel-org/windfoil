// windfoil-ext.wgsl — the windfoil gather generalized from the box filter to PLUGGABLE kernels.
//
// src/windfoil.wgsl stays the production reference: the exact box filter plus its measured accelerations.
// THIS file is the EXACTNESS reference for every other kernel: the same algorithm with the footprint weight
// generalized (docs/NOTES.md "Filter Kernels" is the sketch; docs/KERNELS.md the derivation), evaluated by
// the plain analytic path at every size — deliberately NO minification guard and no approximation tiers, so
// what this shader draws IS the kernel-filtered winding, all the way down. It could in principle adopt the
// core shader's accelerations (the guard generalizes cleanly: each band's precomputed ink cell weighted by
// the kernel's rect mass reduces term-for-term to the shipped box guard — derivation note in KERNELS.md);
// they are kept out so this file stays small and focused on the kernel machinery itself.
//
// The generalization: with the kernel's horizontal cumulative Φ(u,v) = ∫_{−∞}^{u} k(p,v) dp, the filtered
// winding is
//
//   F = Σ_pieces ∫ Φ(u(t), v(t)) · v′(t) dt        (u, v = curve in PIXEL units, relative to the pixel center)
//
// Box is the degree-0 case (Φ = the clamp ramp windfoil.wgsl integrates in closed form). Per piece:
//
//   • fully LEFT of the kernel support  → Φ = 0                    (same skip + sorted early-break)
//   • fully RIGHT                       → Φ = the full row mass    → the marginal CDF M(v), telescoping
//                                         over clamped endpoints exactly like the box shader's far path
//   • CROSSING                          → Gauss–Legendre on Φ·v′ between the support-edge roots, segmented
//                                         at the kernel's knots so each segment is one polynomial (for the
//                                         piecewise-polynomial kernels the quadrature is EXACT — GL-N covers
//                                         degree 2N−1; per-kernel accounting in kernels.js)
//
// The gather runs in pixel units (curve − rc, divided by the footprint s), so the kernel is a fixed unit-mass
// shape and the accumulated F is already the normalized filtered winding — no ÷(sx·sy) at the end.
//
// The kernel block below is SPLICED by src/kernels.js (loadKernelShaderCode); the inline default is the tent
// filter so the file reads standalone. Selecting the box filter never loads this file at all — it loads the
// untouched windfoil.wgsl, so the default path pays zero for this extension. Each kernel compiles its own
// specialized pipeline; there is deliberately no runtime kernel switch — a runtime branch taxes the whole
// shader's occupancy even when never taken (measured: bench/ACCEL-NOTES.md).

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
// of the support (the threshold moves from −sx/2 to −KERNEL_RX·sx, still monotone under the same sort).
// MUST equal BAND_SORT_MIN in bands.js.
const SORT_MIN : u32 = 4u;

// Row-table layout — MUST match bands.js. Only start/count are read here; the area/hull slots exist for the
// core shader's minification guard, which this reference deliberately omits.
const ROW_STRIDE : u32 = 5u;

@group(0) @binding(0) var<uniform> U : Uniforms;
@group(0) @binding(1) var<storage, read> instances : array<Instance>;
// Curve atlas: three consecutive vec2 per xy-monotone piece (endpoints + control).
@group(0) @binding(2) var<storage, read> curves : array<vec2<f32>>;
// Row-band table: ROW_STRIDE u32s per band (see bands.js).
@group(0) @binding(3) var<storage, read> rows : array<u32>;

// ─── KERNEL ───────────────────────────────────────────────────────────────────────────────────────────────
// The contract, all in PIXEL units relative to the pixel center (kernel mass 1):
//   KERNEL_RX/RY    support half-extents. The vertex pad, band slab and cull thresholds derive from these.
//   k_xcum(u, v)    Φ — the horizontal cumulative ∫_{−∞}^{u} k(p,v) dp. Must saturate: 0 at u ≤ −RX, the
//                   full row mass m(v) at u ≥ +RX. This is the ONLY function the crossing quadrature needs,
//                   so non-separable kernels (the bokeh disc, the polygonal iris) plug in as easily as
//                   separable ones.
//   k_ycdf(v)       the marginal CDF M(v) = ∫∫_{v′≤v} k — the far-right weight; saturates to 0/1.
//   Y_KNOTS         interior v where the kernel's y-profile changes polynomial piece (support edges need no
//                   entry — the slab already clips to ±RY). Ascending. Only the crossing quadrature cares:
//                   the far-field CDF paths evaluate exact antiderivatives, knots or not, so the knots are
//                   applied as per-PIECE t-splits inside integrate_piece — never as window splits, which
//                   would re-scan every far piece in the band once per sub-window (measured 54× at 16px
//                   text for the 4×4 kernels before this was restructured).
//   X_SPLITS        interior u where Φ kinks hard enough that the crossing quadrature must not straddle it
//                   (for the piecewise kernels this is what makes GL exact). Ascending.
//   GL_X/GL_W       Gauss–Legendre nodes/weights on [−1,1], N_GL of them.
//   N_SUB           composite factor: each quadrature segment is further cut into N_SUB equal t-slices. The
//                   knots bound polynomial pieces, but a STEEP piece can still sweep the kernel's whole
//                   y-reach in one segment, where a lone GL rule under-resolves the smooth kernels (measured
//                   up to 1e-1 for the Gaussian before this existed — see kernels.js). Uniform t-slicing
//                   fixes that resolution problem with zero extra root solves; 1 for the exact kernels.
//<kernel-block> — everything to the matching close marker is replaced by src/kernels.js; inline default: tent
const KERNEL_RX : f32 = 1.0;
const KERNEL_RY : f32 = 1.0;
const N_SUB : u32 = 1u;
const N_GL : u32 = 4u;
const GL_X = array<f32, 4>(-0.8611363115940526, -0.3399810435848563, 0.3399810435848563, 0.8611363115940526);
const GL_W = array<f32, 4>(0.34785484513745385, 0.6521451548625461, 0.6521451548625461, 0.34785484513745385);
const N_YKNOTS : u32 = 1u;
const Y_KNOTS = array<f32, 1>(0.0);
const N_XSPLITS : u32 = 1u;
const X_SPLITS = array<f32, 1>(0.0);

// tent (bilinear): k(t) = max(1 − |t|, 0). CDF quadratic per side; with the {0} knots above every quadrature
// segment is a single polynomial of degree ≤ 7, so GL-4 integrates it EXACTLY (see kernels.js).
fn kcdf(t : f32) -> f32 {
  let a = clamp(t, -1.0, 1.0);
  let h = 0.5 * (1.0 - abs(a)) * (1.0 - abs(a));
  return select(1.0 - h, h, a < 0.0);
}
fn kpdf(t : f32) -> f32 {
  return max(1.0 - abs(t), 0.0);
}
fn k_xcum(u : f32, v : f32) -> f32 { return kcdf(u) * kpdf(v); }
fn k_ycdf(v : f32) -> f32 { return kcdf(v); }
//</kernel-block>
// ─── /KERNEL ──────────────────────────────────────────────────────────────────────────────────────────────

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
  // The AA skirt reaches as far as the kernel does: (max support + 0.5) device px pad — the same 2× margin
  // discipline as the core's 1px box pad (R + 0.5 with R = 0.5), converted to glyph units by the on-screen
  // scale so it stays that many device px at any zoom.
  let pad = (max(KERNEL_RX, KERNEL_RY) + 0.5) / (unitsToPx * max(camScale.x, 1e-6));
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

// Straight-alpha color × coverage → premultiplied RGBA (the pipeline blends premultiplied-over).
fn shade(color : vec4<f32>, cov : f32) -> vec4<f32> {
  let a = color.a * cov;
  return vec4<f32>(color.rgb * a, a);
}

// Shared fragment tail: fold the kernel-filtered winding by fill rule, style, shade. The fold is the SAME
// saturating model as the core shader; for kernels with negative lobes it shows the (bounded, documented)
// ringing as a faint halo rather than clipping it — see docs/KERNELS.md.
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
// (a0 = value at t = 0, e1 = value at t = 1). Verbatim from windfoil.wgsl.
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

// Gauss–Legendre on one INSIDE segment: ∫ Φ(u(t), v(t)) · v′(t) dt over [ta, tb], the curve in pixel units.
// The segment boundaries (support-edge roots plus the X_SPLITS / Y_KNOTS cuts) guarantee Φ and the kernel's
// y-profile are each one polynomial piece across the whole segment, so for the piecewise-polynomial kernels
// the rule is exact, not approximate (degree ≤ 2·N_GL − 1; per-kernel accounting in kernels.js).
fn gl_segment(a2 : vec2<f32>, a1 : vec2<f32>, q1 : vec2<f32>, ta : f32, tb : f32) -> f32 {
  if (tb <= ta) { return 0.0; }
  let dsub = (tb - ta) / f32(N_SUB);
  var acc : f32 = 0.0;
  for (var k : u32 = 0u; k < N_SUB; k = k + 1u) {   // composite: N_SUB equal slices (const bounds — unrolled)
    let tm = ta + (f32(k) + 0.5) * dsub;
    let dt = 0.5 * dsub;
    for (var i : u32 = 0u; i < N_GL; i = i + 1u) {
      let t = tm + dt * GL_X[i];
      let p = (a2 * t + a1) * t + q1;               // (u, v) at the node
      acc += GL_W[i] * k_xcum(p.x, p.y) * (2.0 * a2.y * t + a1.y) * dt;
    }
  }
  return acc;
}

// One xy-monotone piece's contribution over the pixel-unit v-window [lo, hi] (already ∩ the piece's v-span).
// The same LEFT / INSIDE / RIGHT sweep as windfoil.wgsl's integrate_piece, at the kernel's support edges ±RX:
// LEFT contributes 0, RIGHT the marginal CDF difference (the kernel's whole row mass swept over the zone's
// v-extent), INSIDE the segmented quadrature.
fn integrate_piece(q1 : vec2<f32>, q2 : vec2<f32>, q3 : vec2<f32>, lo : f32, hi : f32) -> f32 {
  let a2 = q1 - 2.0 * q2 + q3;
  let a1 = 2.0 * (q2 - q1);
  let y_rising = q3.y >= q1.y;
  let t_lo = mono_root(a2.y, a1.y, q1.y, q3.y, select(hi, lo, y_rising), y_rising);
  let t_hi = mono_root(a2.y, a1.y, q1.y, q3.y, select(lo, hi, y_rising), y_rising);
  if (t_hi <= t_lo) { return 0.0; }
  let x_rising = q3.x >= q1.x;
  let t_left = clamp(mono_root(a2.x, a1.x, q1.x, q3.x, -KERNEL_RX, x_rising), t_lo, t_hi);
  let t_right = clamp(mono_root(a2.x, a1.x, q1.x, q3.x, KERNEL_RX, x_rising), t_lo, t_hi);
  // Zones in sweep order: x rising ⇒ LEFT · INSIDE · RIGHT; mirrored if not.
  let t1 = select(t_right, t_left, x_rising);
  let t2 = max(select(t_left, t_right, x_rising), t1);
  // INSIDE [t1,t2]: split wherever u(t) crosses an X_SPLIT or v(t) crosses a Y_KNOT, so every quadrature
  // segment sees a single polynomial piece of Φ AND of the y-profile. Each split value is one root
  // (xy-monotone piece), solved ONLY when the piece's extent actually straddles it — a knot outside the
  // extent would clamp to an empty segment anyway, and the skipped sqrt is most of a crossing's cost for
  // the 4×4 kernels. The few live boundaries are insertion-sorted; the array bound covers the widest
  // shipped knot set with room.
  let u_lo = min(q1.x, q3.x);                      // endpoint extents (monotone ⇒ exact)
  let u_hi = max(q1.x, q3.x);
  var cuts : array<f32, 8>;
  var nc : u32 = 0u;
  for (var j : u32 = 0u; j < N_XSPLITS; j = j + 1u) {
    if (X_SPLITS[j] > u_lo && X_SPLITS[j] < u_hi) {
      cuts[nc] = clamp(mono_root(a2.x, a1.x, q1.x, q3.x, X_SPLITS[j], x_rising), t1, t2);
      nc = nc + 1u;
    }
  }
  for (var j : u32 = 0u; j < N_YKNOTS; j = j + 1u) {
    if (Y_KNOTS[j] > lo && Y_KNOTS[j] < hi) {
      cuts[nc] = clamp(mono_root(a2.y, a1.y, q1.y, q3.y, Y_KNOTS[j], y_rising), t1, t2);
      nc = nc + 1u;
    }
  }
  for (var j : u32 = 1u; j < nc; j = j + 1u) {          // insertion sort the few live cuts
    let v0 = cuts[j];
    var i2 : u32 = j;
    while (i2 > 0u && cuts[i2 - 1u] > v0) {
      cuts[i2] = cuts[i2 - 1u];
      i2 = i2 - 1u;
    }
    cuts[i2] = v0;
  }
  var acc : f32 = 0.0;
  var ta = t1;
  for (var j : u32 = 0u; j < nc; j = j + 1u) {
    acc += gl_segment(a2, a1, q1, ta, cuts[j]);
    ta = max(ta, cuts[j]);
  }
  acc += gl_segment(a2, a1, q1, ta, t2);
  let ra = select(t_lo, t2, x_rising);
  let rb = select(t1, t_hi, x_rising);
  if (rb > ra) {
    // RIGHT zone: the row mass m(v) swept over the zone's v-extent = a marginal-CDF difference. The clamp
    // pins the window-edge evaluations so adjacent windows telescope.
    let va = clamp((a2.y * ra + a1.y) * ra + q1.y, lo, hi);
    let vb = clamp((a2.y * rb + a1.y) * rb + q1.y, lo, hi);
    acc += k_ycdf(vb) - k_ycdf(va);
  }
  return acc;
}

// Accumulate one row band's pieces over the rc-relative y-window [wlo, whi] in GLYPH units (band ∩ slab).
// Culls and the degenerate test run in glyph units exactly like windfoil.wgsl (same coord-ULP reasoning);
// only survivors pay the pixel-unit scale + the kernel math.
fn integrate_band(start : u32, count : u32, rc : vec2<f32>, wlo : f32, whi : f32,
                  s : vec2<f32>, inv_s : vec2<f32>) -> f32 {
  var acc : f32 = 0.0;
  let sorted = count > SORT_MIN;
  let rxg = KERNEL_RX * s.x;                            // kernel x half-support in glyph units
  let coord_ulp = max(abs(rc.x), abs(rc.y)) * 1.2e-7;
  for (var i : u32 = 0u; i < count; i = i + 1u) {
    let base = (start + i) * 3u;
    let q1 = curves[base] - rc;
    let q2 = curves[base + 1u] - rc;
    let q3 = curves[base + 2u] - rc;
    let x_hull_max = max(q1.x, max(q2.x, q3.x));
    if (x_hull_max <= -rxg) {             // fully LEFT of the support → no weight
      if (sorted) { break; }
      continue;
    }
    let py_lo = min(q1.y, q3.y);          // piece y-span (endpoint-exact when monotone)
    let py_hi = max(q1.y, q3.y);
    let lo = max(wlo, py_lo);
    let hi = min(whi, py_hi);
    if (hi <= lo) { continue; }
    let x_hull_min = min(q1.x, min(q2.x, q3.x));
    if (x_hull_min >= rxg) {
      // fully RIGHT of the support → the marginal CDF over the clipped y-span. Difference of clamped
      // endpoints, so piece chains telescope exactly as in the box shader's far path.
      acc += k_ycdf(clamp(q3.y, wlo, whi) * inv_s.y) - k_ycdf(clamp(q1.y, wlo, whi) * inv_s.y);
      continue;
    }
    // f32-degenerate piece: Φ at the hull midpoint × the clipped v-span (the kernel analog of the box
    // shader's midpoint-clamp form; exact to ~span²).
    if (x_hull_max - x_hull_min + (py_hi - py_lo) <= coord_ulp * 16.0) {
      let um = clamp((q1.x + q3.x) * 0.5, -rxg, rxg) * inv_s.x;
      let va = clamp(q1.y, wlo, whi) * inv_s.y;
      let vb = clamp(q3.y, wlo, whi) * inv_s.y;
      acc += k_xcum(um, 0.5 * (va + vb)) * (vb - va);
      continue;
    }
    acc += integrate_piece(q1 * inv_s, q2 * inv_s, q3 * inv_s, lo * inv_s.y, hi * inv_s.y);
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

// One glyph's kernel-filtered winding, gathered through the row bands its slab touches. Identical window
// plumbing to windfoil.wgsl (rc-relative for deep-zoom stability) with one generalization: the slab is
// ±KERNEL_RY pixels (the kernel's y-reach, not the pixel's half-height). The windows tile the slab exactly
// as in the box shader, so band-duplicated pieces integrate over disjoint windows and the sum stays exact.
fn integrate_face(band : vec4<f32>, rc : vec2<f32>, s : vec2<f32>) -> f32 {
  let rowBase = u32(band.x);
  let R = u32(band.y);
  let invH = band.w;
  let ry = KERNEL_RY * s.y;                             // slab half-height, glyph units
  let dy0 = band.z - rc.y;                              // band origin y0, relative to the pixel center
  var ri0 : u32 = 0u;
  var ri1 : u32 = 0u;
  if (invH > 0.0) {             // invH > 0 only for multi-band glyphs (bands.js stores 0 when R == 1)
    ri0 = band_index(-dy0 - ry, invH, R);
    ri1 = band_index(-dy0 + ry, invH, R);
  }
  let inv_s = 1.0 / s;
  var f_cov : f32 = 0.0;
  for (var ri = ri0; ri <= ri1; ri = ri + 1u) {
    var w_lo = -ry;
    var w_hi = ry;
    if (invH > 0.0) {
      let e = band_edges(dy0, ri, invH);
      w_lo = max(w_lo, e.x);
      w_hi = min(w_hi, e.y);
    }
    if (w_hi <= w_lo) { continue; }
    let rIdx = (rowBase + ri) * ROW_STRIDE;
    f_cov += integrate_band(rows[rIdx], rows[rIdx + 1u], rc, w_lo, w_hi, s, inv_s);
  }
  return f_cov;                                         // pixel units ⇒ already normalized: no ÷(sx·sy)
}

@fragment
fn fs(in : VsOut) -> @location(0) vec4<f32> {
  let I = instances[in.inst];
  let rc = in.rc;
  // units_per_pixel from the screen-space gradients — the device pixel's preimage under scale/translation.
  let s = max(fwidth(rc), vec2<f32>(1e-9));
  // One gather, the plain analytic path at every size — this shader is the exactness reference.
  return fold_shade(integrate_face(I.band, rc, s), I.place.w, I.color);
}
