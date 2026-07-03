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
// see the tuning note there (8 = the median band occupancy for the Lato glyph set).
const SORT_MIN : u32 = 8u;

// when a glyph shrinks below ~1px, every pixel's footprint spans the entire glyph, so the gather integrates
// all of its curves at every pixel and per-pixel cost begins to peak. You could cull these entirely, or just
// compute a rough coverage from the area of the glyph contours. This could be per-glyph for correctness, or
// you can use a rough average (or a uniform that changes per icon set for example). This number roughly sits
// in an average across alphanumeric glyphs of the current font set, so for a more exact AA at small sizes, you
// may wish to tune this.
const MINIFICATION_GUARD = true;
const INK_AVERAGE = 0.42;

@group(0) @binding(0) var<uniform> U : Uniforms;
@group(0) @binding(1) var<storage, read> instances : array<Instance>;
// The deduped, band-duplicated curve atlas: three consecutive vec2 per xy-monotone piece (endpoints + control).
@group(0) @binding(2) var<storage, read> curves : array<vec2<f32>>;
// Row-band table: a flat [start, count] pair per band, indexing into `curves`.
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
  // Pad the quad outward by ~2 device px so an edge's anti-aliased skirt is never clipped. The pad is in
  // glyph units, so divide by the ON-SCREEN scale (unitsToPx·camScale) to keep it ~2 device px at any zoom.
  let pad = 2.0 / (unitsToPx * max(camScale.x, 1e-6));
  let lo = I.bbox.xy - vec2<f32>(pad);
  let hi = I.bbox.zw + vec2<f32>(pad);
  // Unit-quad corners for a triangle-strip: (0,0) (1,0) (0,1) (1,1).
  let uv = vec2<f32>(f32(vi & 1u), f32((vi >> 1u) & 1u));
  let em = mix(lo, hi, uv);
  let worldPx = I.place.xy + em * unitsToPx;          // Y-down "world" pixels (the layout space)
  let devicePx = worldPx * camScale + U.cam.zw;       // apply the camera (pan + zoom) → device px
  let clip = vec2<f32>(devicePx.x / U.res.x * 2.0 - 1.0, 1.0 - devicePx.y / U.res.y * 2.0);
  var o : VsOut;
  o.pos = vec4<f32>(clip, 0.0, 1.0);
  o.rc = em;
  o.inst = ii;
  return o;
}

// Period-2 triangle wave T(t) = 1 − |1 − (t mod 2)|: folds a signed winding MAGNITUDE to its even-odd
// coverage (a value of 2 — an enclosed hole — folds to 0; fractional edges ramp linearly in and out).
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

// Solve the monotone quadratic component A2·t² + A1·t + A0 = v on [0,1]. `e0`/`e1` are the component's
// values at t = 0 / 1 (endpoint-exact for a monotone piece): saturate to 0 if the piece starts past `v`,
// to 1 if it never reaches `v`. The root branch is picked by the derivative sign matching `rising`.
fn mono_root(a2 : f32, a1 : f32, a0 : f32, e0 : f32, e1 : f32, v : f32, rising : bool) -> f32 {
  if (rising) {
    if (e0 >= v) { return 0.0; }
    if (e1 <= v) { return 1.0; }
  } else {
    if (e0 <= v) { return 0.0; }
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
  let d1 = 2.0 * a2 * r1 + a1;
  let want = select(-1.0, 1.0, rising);
  let t = select(r2, r1, d1 * want >= 0.0);
  return clamp(t, 0.0, 1.0);
}

// The INSIDE zone's exact integral of (x(t)+hx)·y′(t) over [ta,tb]: the midpoint rule on a symmetric
// interval, which is exact for this cubic integrand (odd powers about the midpoint vanish). `x(t)+hx` is
// box-local (0..sx), so there is no large-magnitude cancellation.
fn integrate_inside(a2 : vec2<f32>, a1 : vec2<f32>, q1 : vec2<f32>, ta : f32, tb : f32, hx : f32) -> f32 {
  if (tb <= ta) { return 0.0; }
  let tm = 0.5 * (ta + tb);
  let d = 0.5 * (tb - ta);
  let x_mid = (a2.x * tm + a1.x) * tm + q1.x + hx;
  let xp = 2.0 * a2.x * tm + a1.x;
  let yp = 2.0 * a2.y * tm + a1.y;
  return 2.0 * d * x_mid * yp + (2.0 * d * d * d / 3.0) * (a2.x * yp + 2.0 * a2.y * xp);
}

// One xy-monotone piece's contribution to ∫∫_box w over the rc-relative y-window [lo, hi] (already
// intersected with the piece's y-span), box half-width hx. The winding integral integrates out x as
// clamp(x(t), −hx, hx) + hx, splitting the crossing t-interval into LEFT (0) / INSIDE (exact) / RIGHT
// (full box width) zones at the two x-edge crossings — in a statically known order set by the x direction.
fn integrate_piece(q1 : vec2<f32>, q2 : vec2<f32>, q3 : vec2<f32>, lo : f32, hi : f32, hx : f32) -> f32 {
  let a2 = q1 - 2.0 * q2 + q3;
  let a1 = 2.0 * (q2 - q1);
  let y_rising = q3.y >= q1.y;
  let t_lo = mono_root(a2.y, a1.y, q1.y, q1.y, q3.y, select(hi, lo, y_rising), y_rising);
  let t_hi = mono_root(a2.y, a1.y, q1.y, q1.y, q3.y, select(lo, hi, y_rising), y_rising);
  if (t_hi <= t_lo) { return 0.0; }
  let x_rising = q3.x >= q1.x;
  let t_left = clamp(mono_root(a2.x, a1.x, q1.x, q1.x, q3.x, -hx, x_rising), t_lo, t_hi);
  let t_right = clamp(mono_root(a2.x, a1.x, q1.x, q1.x, q3.x, hx, x_rising), t_lo, t_hi);
  // Zones in sweep order: x rising ⇒ LEFT [t_lo,t_left] · INSIDE · RIGHT [t_right,t_hi]; mirrored if not.
  let t1 = select(t_right, t_left, x_rising);
  let t2 = max(select(t_left, t_right, x_rising), t1);
  var acc = integrate_inside(a2, a1, q1, t1, t2, hx);
  let ra = select(t_lo, t2, x_rising);
  let rb = select(t1, t_hi, x_rising);
  if (rb > ra) {
    // RIGHT zone: full box width × Δy, with Δy = Δt · y′(midpoint) (exact for the quadratic).
    let tm = 0.5 * (ra + rb);
    acc += (rb - ra) * (2.0 * a2.y * tm + a1.y) * (2.0 * hx);
  }
  return acc;
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
    let lo = max(wlo, min(q1.y, q3.y));                   // window ∩ piece y-span (endpoint-exact)
    let hi = min(whi, max(q1.y, q3.y));
    if (hi <= lo) { continue; }
    let x_hull_min = min(q1.x, min(q2.x, q3.x));
    if (x_hull_min >= hx) {
      // fully RIGHT of the box → full box width × the clipped y-span. As a difference of clamped endpoints
      // (not sign·overlap) so it telescopes over piece chains: shared endpoints cancel exactly, so a run of
      // ULP-scale segments sums to its span rather than accumulating per-piece rounding.
      acc += sx * (clamp(q3.y, wlo, whi) - clamp(q1.y, wlo, whi));
      continue;
    }
    // f32-degenerate piece (whole hull within a few coordinate-ULPs — flattened content at deep zoom far
    // from the origin): integrate_piece's t-solves would divide ULP-scale coefficients into noise. The midpoint-
    // clamp form is exact to ~span² and telescopes like the fully-right path.
    if (x_hull_max - x_hull_min + (max(q1.y, q3.y) - min(q1.y, q3.y)) <= coord_ulp * 16.0) {
      let xm = clamp((q1.x + q3.x) * 0.5, -hx, hx) + hx;
      acc += xm * (clamp(q3.y, wlo, whi) - clamp(q1.y, wlo, whi));
      continue;
    }
    acc += integrate_piece(q1, q2, q3, lo, hi, hx);
  }
  return acc;
}

// One glyph's winding integral over the pixel box (rc ± s/2), gathered through its ROW BANDS. The pixel's
// y-slab selects the band range it touches; each band is read clipped to its own y-range. The bands tile
// the slab, so a piece duplicated across adjacent bands integrates over disjoint windows and the sum is
// exact without a dedupe test.
fn integrate_face(I : Instance, rc : vec2<f32>, s : vec2<f32>) -> f32 {
  let rowBase = u32(I.band.x);
  let R = u32(I.band.y);
  let invH = I.band.w;

  // Build the slab rc-RELATIVE ([−sy2, +sy2]). At deep zoom on far-from-origin coordinates sy/2 drops below
  // ULP(rc.y), so an absolute slab (rc.y ± sy2, then − rc.y) would quantize to zero height and drop whole
  // pixel rows (horizontal banding). The one absolute term (dy0 = y0 − rc.y) feeds only band selection and
  // the boundary clips, whose ULP wobble just nudges the split between adjacent (still exactly tiling)
  // windows. See docs/ALGORITHM.md §6.
  let sy2 = s.y * 0.5;
  let dy0 = I.band.z - rc.y;      // band origin y0, relative to the pixel center
  var ri0 : u32 = 0u;
  var ri1 : u32 = 0u;
  if (invH > 0.0 && R > 1u) {
    // floor(((rc.y ± sy2) − y0)·invH) = floor((∓? …)); rc-relative: (slab − y0) = −dy0 ± sy2
    ri0 = u32(clamp(floor((-dy0 - sy2) * invH), 0.0, f32(R) - 1.0));
    ri1 = u32(clamp(floor((-dy0 + sy2) * invH), 0.0, f32(R) - 1.0));
  }
  var f_int : f32 = 0.0;
  for (var ri = ri0; ri <= ri1; ri = ri + 1u) {
    var w_lo = -sy2;             // rc-relative window edges — stable at any zoom
    var w_hi = sy2;
    if (invH > 0.0) {
      w_lo = max(w_lo, dy0 + f32(ri) / invH);         // band ri's y-range, rc-relative
      w_hi = min(w_hi, dy0 + (f32(ri) + 1.0) / invH);
    }
    if (w_hi <= w_lo) { continue; }
    let rIdx = (rowBase + ri) * 2u;
    f_int += integrate_band(rows[rIdx], rows[rIdx + 1u], rc, w_lo, w_hi, s.x);
  }
  return f_int;
}

@fragment
fn fs(in : VsOut) -> @location(0) vec4<f32> {
  let I = instances[in.inst];
  let rc = in.rc;
  // units_per_pixel: the length of each axis' screen-space gradient — under pure scale/translation this
  // makes the integration box exactly the device pixel's preimage.
  let s = max(
    vec2<f32>(
      length(vec2<f32>(dpdx(rc.x), dpdy(rc.x))),
      length(vec2<f32>(dpdx(rc.y), dpdy(rc.y))),
    ),
    vec2<f32>(1e-9),
  );

  // improve performance by culling/averaging minified glyphs
  if (MINIFICATION_GUARD) {
    // glyph size in font units
    let gw = I.bbox.z - I.bbox.x;
    let gh = I.bbox.w - I.bbox.y;
    // if one pixel is bigger than the whole glyph,
    // the letter is a sub-pixel smudge: coverage = ink area / window area
    if (s.x >= gw && s.y >= gh) {
      let pixLo = rc - s * 0.5; // this pixel's box, in glyph units
      let pixHi = rc + s * 0.5;
      let ovLo = max(pixLo, I.bbox.xy); // overlap of pixel box ∩ glyph bbox
      let ovHi = min(pixHi, I.bbox.zw);
      let ov   = max(ovHi - ovLo, vec2<f32>(0.0));
      let cov  = clamp(INK_AVERAGE * ov.x * ov.y / (s.x * s.y), 0.0, 1.0);
      let a = I.color.a * cov;
      return vec4<f32>(I.color.rgb * a, a);
    }
    // for more performance gains:
    // there is also a middle ground where the glyph is above some small speck, but still
    // small enough that it has to scan more bands than is ideal, and we could provide
    // a special case acceleration here, either using a baked asset or per-band moments (+2 floats per band)
    // else if (s.y * I.band.y >= gh) {
      // per-band moments or pre-rendered texture
    // }
  }

  // One gather → the winding integral ∫∫_box w over the pixel box, normalized to coverage.
  let f_cov = integrate_face(I, rc, s) / max(s.x * s.y, 1e-30);

  var cov : f32;
  if (I.place.w > 0.5) {
    cov = clamp(tri_wave(f_cov), 0.0, 1.0);   // even-odd
  } else {
    cov = clamp(abs(f_cov), 0.0, 1.0);        // nonzero (saturating winding integral)
  }
  cov = style_coverage(cov, U.style.x, U.style.y);  // opt-in --gamma / --sharp tuning; (1,1) exact ⇒ identity
  let a = I.color.a * cov;
  return vec4<f32>(I.color.rgb * a, a);       // premultiplied — pipeline blends premultiplied-over
}
