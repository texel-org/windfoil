// kernels.js — the pluggable filter-kernel registry for the windfoil coverage shader.
//
// Selecting a kernel selects a SHADER, not a branch: `box` (the default) loads the untouched
// src/windfoil.wgsl reference, so the common case pays literally nothing for this module existing. Any other
// kernel loads src/windfoil-ext.wgsl with that kernel's block spliced in between the //<kernel-block> markers,
// giving each kernel its own specialized pipeline — the compiler constant-folds the radii, knot lists and
// quadrature order, and dead-code-eliminates every other kernel. (A runtime kernel uniform was ruled out up
// front: bench/ACCEL-NOTES.md measured that merely CARRYING an untaken fast path costs the whole shader
// ~15–20% occupancy; the same applies to carrying five unused CDFs.)
//
// Every kernel supplies, in PIXEL units (see the contract comment in windfoil-ext.wgsl):
//   k_xcum(u,v)  Φ — the horizontal cumulative, the only thing the crossing quadrature needs
//   k_ycdf(v)    the marginal CDF (far-right weight)
//   k_cell(...)  rect mass (minification guard)
//   radii, Y_KNOTS / X_SPLITS (piecewise boundaries), and a Gauss–Legendre order N_GL.
//
// Accuracy accounting (why each N_GL / N_SUB / split list is what it is): inside a quadrature segment the
// integrand is Φ(u(t), v(t))·v′(t) with u, v quadratic in t. Windows are pre-split at Y_KNOTS and segments at
// X_SPLITS, so for the piecewise-polynomial kernels the integrand is ONE polynomial per segment and GL-N
// (exact through degree 2N−1) makes the gather exact, not approximate. The smooth kernels are not polynomial
// in t, and one GL rule can under-resolve a segment that sweeps the whole support (a steep edge's ky-bump
// compressed into a sliver of t — an adversarial derivation pass measured up to 1e-1 for a knotless GL-5
// Gaussian); N_SUB composite slicing fixes that resolution failure without extra root solves. Worst
// |ΔF| per piece, f64, random + adversarial monotone pieces vs a GL-8×32 reference (the sweep is
// tools/kernel-tune.js — rerun it when touching any budget):
//   box-ext    degree 3  → GL-2 exact            (measured 1.1e-15)
//   tent       degree 7  → GL-4 exact            (measured 1.0e-15)
//   mblur      degree 5  → GL-3 exact            (measured 1.1e-15)
//   BC family  degree 15, C²-joins at u=0,±1: X_SPLITS {−1,0,1} + Y_KNOTS {−1,0,1}, GL-5 → 3.4e-6
//              (without the u=0 split: 1.0e-4; GL-5×2 reaches 8.9e-9 if ever needed)
//   gaussian   smooth (a degree-13 polynomial CDF): GL-8 × N_SUB 2 → 2.0e-6 (GL-5×1 was 2.6e-2!)
//   disc       the rim's slope is unbounded (√): GL-8 × N_SUB 2 → ~2e-3 WORST CASE at rim-grazing edges
//              (sub-visible: ~0.6/255, and only where an edge runs tangent to the bokeh circle; typical
//              pixels are orders of magnitude better). The price of a genuinely non-polynomial kernel.
//
// Kernels with parameters take them after '=': e.g. 'mblur=12' (motion length px), 'disc=2.5' (radius px).

import { loadShaderCode } from './gpu.js';

const EXT_WGSL_URL = new URL('./windfoil-ext.wgsl', import.meta.url);

// ── Gauss–Legendre nodes/weights on [−1, 1] ─────────────────────────────────────────────────────────────
export const GL_TABLES = {
  2: { x: [-0.5773502691896257, 0.5773502691896257], w: [1, 1] },
  3: { x: [-0.7745966692414834, 0, 0.7745966692414834], w: [0.5555555555555556, 0.8888888888888888, 0.5555555555555556] },
  4: {
    x: [-0.8611363115940526, -0.3399810435848563, 0.3399810435848563, 0.8611363115940526],
    w: [0.34785484513745385, 0.6521451548625461, 0.6521451548625461, 0.34785484513745385],
  },
  5: {
    x: [-0.9061798459386640, -0.5384693101056831, 0, 0.5384693101056831, 0.9061798459386640],
    w: [0.23692688505618908, 0.47862867049936647, 0.5688888888888889, 0.47862867049936647, 0.23692688505618908],
  },
  8: {
    x: [-0.9602898564975363, -0.7966664774136267, -0.5255324099163290, -0.1834346424956498,
      0.1834346424956498, 0.5255324099163290, 0.7966664774136267, 0.9602898564975363],
    w: [0.10122853629037626, 0.22238103445337448, 0.31370664587788727, 0.36268378337836198,
      0.36268378337836198, 0.31370664587788727, 0.22238103445337448, 0.10122853629037626],
  },
};

// WGSL float literal: keep f64 round-trip digits, force a decimal point so it never parses as AbstractInt.
const lit = (x) => {
  const s = String(x);
  return /[.e]/.test(s) ? s : `${s}.0`;
};
const farr = (vals) => `array<f32, ${Math.max(vals.length, 1)}>(${(vals.length ? vals : [0]).map(lit).join(', ')})`;

// Assemble one kernel block (the text between the //<kernel-block> markers in windfoil-ext.wgsl).
function block({ name, rx, ry, glOrder, nSub = 1, yKnots = [], xSplits = [], fns }) {
  const gl = GL_TABLES[glOrder];
  return [
    `// ${name} (spliced by src/kernels.js)`,
    `const KERNEL_RX : f32 = ${lit(rx)};`,
    `const KERNEL_RY : f32 = ${lit(ry)};`,
    `const N_SUB : u32 = ${nSub}u;`,
    `const N_GL : u32 = ${glOrder}u;`,
    `const GL_X = ${farr(gl.x)};`,
    `const GL_W = ${farr(gl.w)};`,
    `const N_YKNOTS : u32 = ${yKnots.length}u;`,
    `const Y_KNOTS = ${farr(yKnots)};`,
    `const N_XSPLITS : u32 = ${xSplits.length}u;`,
    `const X_SPLITS = ${farr(xSplits)};`,
    fns.trim(),
  ].join('\n');
}

// The standard interface for a separable kernel from a 1D cdf/pdf pair (kcdf/kpdf defined by `fns1d`).
const separable = (fns1d) => `
${fns1d.trim()}
fn k_xcum(u : f32, v : f32) -> f32 { return kcdf(u) * kpdf(v); }
fn k_ycdf(v : f32) -> f32 { return kcdf(v); }
fn k_cell(u0 : f32, u1 : f32, v0 : f32, v1 : f32) -> f32 {
  return (kcdf(u1) - kcdf(u0)) * (kcdf(v1) - kcdf(v0));
}`;

// JS twin of `separable`: the same kernel as plain functions, for the validation tools' ground truth
// (density) and CPU replica (xcum/ycdf). Keeping shader text and JS ref side by side per kernel makes a
// mismatch a one-file diff.
const sepRef = (cdf, pdf) => ({
  density: (u, v) => pdf(u) * pdf(v),
  xcum: (u, v) => cdf(u) * pdf(v),
  ycdf: (v) => cdf(v),
});

// ── BC family (Mitchell–Netravali two-parameter cubics, radius 2) ────────────────────────────────────────
// k(t) for |t|<1: ((12−9B−6C)|t|³ + (−18+12B+6C)|t|² + (6−2B))/6; for 1≤|t|<2:
// ((−B−6C)|t|³ + (6B+30C)|t|² + (−12B−48C)|t| + (8B+24C))/6. Unit mass for every (B,C). The CDF is
// 0.5 + sign(t)·Q(min(|t|,2)) with Q(t) = ∫₀ᵗ k — quartics computed here so each variant folds to literals.
function bcKernel(B, C) {
  const n3 = (12 - 9 * B - 6 * C) / 6, n2 = (-18 + 12 * B + 6 * C) / 6, n0 = (6 - 2 * B) / 6;
  const f3 = (-B - 6 * C) / 6, f2 = (6 * B + 30 * C) / 6, f1 = (-12 * B - 48 * C) / 6, f0 = (8 * B + 24 * C) / 6;
  const Q1 = n0 + n2 / 3 + n3 / 4; // Q(1), near piece integrated from 0
  const F = (t) => ((f3 / 4 * t + f2 / 3) * t + f1 / 2) * t * t + f0 * t; // ∫ far, antiderivative at t
  const qf0 = Q1 - F(1); // far Q(t) = qf0 + F(t)
  const k1d = (t) => {
    const a = Math.abs(t);
    if (a >= 2) return 0;
    return a < 1 ? (n3 * a + n2) * a * a + n0 : ((f3 * a + f2) * a + f1) * a + f0;
  };
  const cdf1d = (t) => {
    const a = Math.min(Math.abs(t), 2);
    const q = a < 1 ? (n3 / 4 * a + n2 / 3) * a * a * a + n0 * a : qf0 + F(a);
    return 0.5 + Math.sign(t) * q;
  };
  const fns = separable(`
fn kcdf(t : f32) -> f32 {
  let a = min(abs(t), 2.0);
  let qn = ((${lit(n3 / 4)} * a + ${lit(n2 / 3)}) * a * a + ${lit(n0)}) * a;
  let qf = ${lit(qf0)} + (((${lit(f3 / 4)} * a + ${lit(f2 / 3)}) * a + ${lit(f1 / 2)}) * a + ${lit(f0)}) * a;
  let q = select(qf, qn, a < 1.0);
  return 0.5 + select(q, -q, t < 0.0);
}
fn kpdf(t : f32) -> f32 {
  let a = min(abs(t), 2.0);
  let kn = (${lit(n3)} * a + ${lit(n2)}) * a * a + ${lit(n0)};
  let kf = ((${lit(f3)} * a + ${lit(f2)}) * a + ${lit(f1)}) * a + ${lit(f0)};
  return select(kf, kn, a < 1.0);
}`);
  return {
    rx: 2, ry: 2, glOrder: 5, nSub: 1, yKnots: [-1, 0, 1], xSplits: [-1, 0, 1], fns,
    ref: sepRef(cdf1d, k1d),
  };
}

// ── truncated Gaussian (σ = 0.5 px, hard-truncated at R = 1.5 px = 3σ, renormalized) ─────────────────────
// The CDF is a degree-13 odd-polynomial least-squares fit to the renormalized truncated-normal CDF on
// [0, 1.5] with the endpoint pinned (P(1.5) = 1/2 exactly, so it saturates continuously); the density is the
// fit's exact derivative, so the pair is self-consistent and integrates to exactly 1. Fit residual
// max |ΔCDF| = 9.0e-6, density everywhere positive (min 9.6e-3 at the truncation edge vs the true 8.9e-3).
// This makes the smoothest kernel also the cheapest wide one: a single polynomial — no knots, no splits.
const GAUSS_C = [0.7999950900962238, -0.5324662228431235, 0.3152844408313249, -0.14116796092318987,
  0.04507519665174872, -0.00899761477436207, 0.0008236909634496945]; // c1, c3, … c13
const GAUSS_R = 1.5;
const gaussP = (t) => { const a2 = t * t; return t * GAUSS_C.reduceRight((acc, c) => acc * a2 + c, 0); };
const gaussD = (t) => { // dP/dt
  const a2 = t * t;
  return GAUSS_C.reduceRight((acc, c, i) => acc * a2 + (2 * i + 1) * c, 0);
};
const gaussCdf = (t) => 0.5 + gaussP(Math.max(-GAUSS_R, Math.min(GAUSS_R, t)));
const gaussPdf = (t) => (Math.abs(t) < GAUSS_R ? gaussD(t) : 0);
const gaussFns = separable(`
fn kcdf(t : f32) -> f32 {
  let a = clamp(t, ${lit(-GAUSS_R)}, ${lit(GAUSS_R)});
  let a2 = a * a;
  return 0.5 + a * ${GAUSS_C.reduceRight((s, c) => (s ? `(${lit(c)} + a2 * ${s})` : lit(c)), '')};
}
fn kpdf(t : f32) -> f32 {
  let a2 = t * t;
  let d = ${GAUSS_C.map((c, i) => (2 * i + 1) * c).reduceRight((s, c) => (s ? `(${lit(c)} + a2 * ${s})` : lit(c)), '')};
  return select(0.0, d, abs(t) < ${lit(GAUSS_R)});
}`);

// ── the registry ─────────────────────────────────────────────────────────────────────────────────────────
// Each entry: WGSL block pieces + a JS `ref` (the same kernel as plain functions) that the validation tools
// weight their point-sampled ground truth with. `blurb` is the one-line selection guide (long form:
// docs/KERNELS.md).
const boxCdf = (t) => Math.max(0, Math.min(1, t + 0.5));
const boxPdf = (t) => (Math.abs(t) < 0.5 ? 1 : 0);

export const KERNELS = {
  box: {
    blurb: 'the exact box filter — the reference. Loads the untouched windfoil.wgsl (zero cost).',
    core: true,
    rx: 0.5, ry: 0.5, glOrder: 2,
    ref: sepRef(boxCdf, boxPdf),
  },
  'box-ext': {
    blurb: 'the box filter through the ext machinery — a debug/cross-check kernel, not for users.',
    rx: 0.5, ry: 0.5, glOrder: 2,
    fns: separable(`
fn kcdf(t : f32) -> f32 { return clamp(t + 0.5, 0.0, 1.0); }
fn kpdf(t : f32) -> f32 { return select(0.0, 1.0, abs(t) < 0.5); }`),
    ref: sepRef(boxCdf, boxPdf),
  },
  tent: {
    blurb: 'tent / bilinear (2×2 px) — the gentle default upgrade: kills stair-stepping shimmer, mild blur. Exact.',
    rx: 1, ry: 1, glOrder: 4, yKnots: [0], xSplits: [0],
    fns: null, // the inline default block in windfoil-ext.wgsl IS tent; built below for uniformity
    ref: sepRef(
      (t) => { const a = Math.max(-1, Math.min(1, t)); const h = 0.5 * (1 - Math.abs(a)) ** 2; return a < 0 ? h : 1 - h; },
      (t) => Math.max(1 - Math.abs(t), 0),
    ),
  },
  gaussian: {
    blurb: 'truncated Gaussian σ=0.5px (3×3 px) — the film-like/print choice: smoothest gradients, no ringing, no Moiré.',
    rx: GAUSS_R, ry: GAUSS_R, glOrder: 8, nSub: 2,
    fns: gaussFns,
    ref: sepRef(gaussCdf, gaussPdf),
  },
  mitchell: {
    blurb: 'Mitchell–Netravali B=C=⅓ (4×4 px) — mild sharpening for static/print output; small negative lobes (≤1.7% halo under the orientation-agnostic fold).',
    ...bcKernel(1 / 3, 1 / 3),
  },
  bspline: {
    blurb: 'cubic B-spline B=1,C=0 (4×4 px) — the softest kernel, strictly positive: heavy-minification stills, thumbnails, film looks.',
    ...bcKernel(1, 0),
  },
  catmullrom: {
    blurb: 'Catmull–Rom B=0,C=½ (4×4 px) — strongest sharpening; negative lobes ≤4.2% (visible halo risk on reversed-winding content — see docs/KERNELS.md).',
    ...bcKernel(0, 0.5),
  },
  mblur: {
    blurb: 'analytic horizontal motion blur (box pixel ⊛ box shutter, length L px; mblur=L, default 8) — exact linear motion blur for static geometry.',
    param: { name: 'L', default: 8, min: 1 },
    make: (L) => {
      const W = (L + 1) / 2;
      const trapCdf = (t) => {
        const a = Math.max(-W, Math.min(W, t));
        if (a < -(W - 1)) return (a + W) ** 2 / (2 * L);
        if (a > W - 1) return 1 - (W - a) ** 2 / (2 * L);
        return 0.5 + a / L;
      };
      const trapPdf = (t) => { const a = Math.abs(t); return a >= W ? 0 : a <= W - 1 ? 1 / L : (W - a) / L; };
      return {
        // L=1 degenerates to the tent in x (both kinks collapse to u=0) — keep that one split or the
        // GL-3 rule straddles the kink and 'exact' breaks at the parameter floor (review finding: 7.5e-3).
        rx: W, ry: 0.5, glOrder: 3, yKnots: [], xSplits: L === 1 ? [0] : [-(W - 1), W - 1],
        fns: `
fn kxcdf(t : f32) -> f32 {
  let a = clamp(t, ${lit(-W)}, ${lit(W)});
  let cLo = (a + ${lit(W)}) * (a + ${lit(W)}) * ${lit(1 / (2 * L))};
  let cHi = 1.0 - (${lit(W)} - a) * (${lit(W)} - a) * ${lit(1 / (2 * L))};
  let mid = 0.5 + a * ${lit(1 / L)};
  return select(select(mid, cHi, a > ${lit(W - 1)}), cLo, a < ${lit(-(W - 1))});
}
fn k_xcum(u : f32, v : f32) -> f32 { return kxcdf(u) * select(0.0, 1.0, abs(v) < 0.5); }
fn k_ycdf(v : f32) -> f32 { return clamp(v + 0.5, 0.0, 1.0); }
fn k_cell(u0 : f32, u1 : f32, v0 : f32, v1 : f32) -> f32 {
  return (kxcdf(u1) - kxcdf(u0)) * (clamp(v1 + 0.5, 0.0, 1.0) - clamp(v0 + 0.5, 0.0, 1.0));
}`,
        ref: {
          density: (u, v) => trapPdf(u) * boxPdf(v),
          xcum: (u, v) => trapCdf(u) * boxPdf(v),
          ycdf: boxCdf,
        },
      };
    },
  },
  disc: {
    blurb: 'uniform disc (bokeh; disc=R px, default 1.5) — a real lens-blur circle, and the proof the interface takes NON-separable kernels. Artistic/moonshot demo.',
    param: { name: 'R', default: 1.5, min: 0.5 },
    make: (R) => ({
      rx: R, ry: R, glOrder: 8, nSub: 2,
      fns: `
const PI_ : f32 = 3.141592653589793;
fn k_xcum(u : f32, v : f32) -> f32 {
  let w = sqrt(max(${lit(R * R)} - v * v, 0.0));
  return (clamp(u, -w, w) + w) * ${lit(1 / (Math.PI * R * R))};
}
fn k_ycdf(v : f32) -> f32 {
  let a = clamp(v, ${lit(-R)}, ${lit(R)});
  let w = sqrt(max(${lit(R * R)} - a * a, 0.0));
  return 0.5 + (a * w + ${lit(R * R)} * asin(clamp(a * ${lit(1 / R)}, -1.0, 1.0))) * ${lit(1 / (Math.PI * R * R))};
}
// Guard stand-in: the disc's marginals are semicircles; a smoothstep CDF per axis is a close separable
// approximation, and the guard only runs on sub-4px instances where the difference is unresolvable.
fn scdf(t : f32) -> f32 {
  let s = clamp(t * ${lit(1 / (2 * R))} + 0.5, 0.0, 1.0);
  return s * s * (3.0 - 2.0 * s);
}
fn k_cell(u0 : f32, u1 : f32, v0 : f32, v1 : f32) -> f32 {
  return (scdf(u1) - scdf(u0)) * (scdf(v1) - scdf(v0));
}`,
      ref: {
        density: (u, v) => (u * u + v * v <= R * R ? 1 / (Math.PI * R * R) : 0),
        xcum: (u, v) => {
          const w = Math.sqrt(Math.max(R * R - v * v, 0));
          return (Math.max(-w, Math.min(w, u)) + w) / (Math.PI * R * R);
        },
        ycdf: (v) => {
          const a = Math.max(-R, Math.min(R, v));
          const w = Math.sqrt(Math.max(R * R - a * a, 0));
          return 0.5 + (a * w + R * R * Math.asin(Math.max(-1, Math.min(1, a / R)))) / (Math.PI * R * R);
        },
      },
    }),
  },
};

// The inline default block in windfoil-ext.wgsl is tent; regenerate it here so splicing is uniform (and so
// a drifted inline copy can be caught by tests diffing the two).
KERNELS.tent.fns = separable(`
fn kcdf(t : f32) -> f32 {
  let a = clamp(t, -1.0, 1.0);
  let h = 0.5 * (1.0 - abs(a)) * (1.0 - abs(a));
  return select(1.0 - h, h, a < 0.0);
}
fn kpdf(t : f32) -> f32 {
  return max(1.0 - abs(t), 0.0);
}`);

/**
 * Resolve a kernel name (with optional '=param', e.g. 'mblur=12', 'disc=2') to its full spec:
 * { name, rx, ry, glOrder, yKnots, xSplits, fns, ref, blurb, core? }.
 */
export function resolveKernel(spec = 'box') {
  const [name, paramRaw] = String(spec).split('=');
  const k = KERNELS[name];
  if (!k) {
    throw new Error(`unknown kernel "${name}" — available: ${Object.keys(KERNELS).join(', ')}`);
  }
  if (k.make) {
    const p = paramRaw === undefined ? k.param.default : Number(paramRaw);
    if (!Number.isFinite(p) || p < k.param.min) {
      throw new Error(`kernel "${name}" needs ${k.param.name} ≥ ${k.param.min}, got "${paramRaw}"`);
    }
    return { name: spec, blurb: k.blurb, ...k.make(p) };
  }
  if (paramRaw !== undefined) throw new Error(`kernel "${name}" takes no parameter`);
  return { name, ...k };
}

/** The spliced windfoil-ext.wgsl source for a kernel spec (throws for 'box' — that is the core shader). */
export function extShaderCode(baseCode, spec) {
  const k = resolveKernel(spec);
  if (k.core) throw new Error(`kernel "${k.name}" is the core shader — load src/windfoil.wgsl instead`);
  const open = '//<kernel-block>';
  const close = '//</kernel-block>';
  const i0 = baseCode.indexOf(open);
  const i1 = baseCode.indexOf(close);
  if (i0 < 0 || i1 < 0) throw new Error('windfoil-ext.wgsl kernel-block markers not found');
  return baseCode.slice(0, i0) + block(k) + '\n' + baseCode.slice(i1 + close.length);
}

/**
 * Load the shader source for a kernel: 'box' (default) → the untouched core windfoil.wgsl; anything else →
 * windfoil-ext.wgsl specialized to that kernel. Same Uniforms/bindings/instance layout either way, so the
 * result drops into createGlyphRenderer unchanged — a kernel is a pipeline choice, cached per kernel.
 */
export async function loadKernelShaderCode(spec = 'box') {
  const k = resolveKernel(spec); // validate the name before any I/O
  if (k.core) return loadShaderCode();
  return extShaderCode(await loadShaderCode(EXT_WGSL_URL), spec);
}
