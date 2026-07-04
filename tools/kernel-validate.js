// kernel-validate.js — check every filter kernel against an INDEPENDENT ground truth, three ways.
//
//   deno run --unstable-webgpu -A tools/kernel-validate.js              # all kernels, all shapes
//   deno run --unstable-webgpu -A tools/kernel-validate.js --kernels tent,mitchell --shapes 0,3
//
// For each (shape, kernel):
//   • gt   — ground truth: the fill indicator point-sampled on a fine sub-pixel grid (winding by ray casting
//            the raw curves — zero shared code with the shader's area integral), then weighted by the kernel
//            density around each pixel center. The true filtered coverage, up to ~1/F sampling noise.
//   • cpu  — an f64 replica of windfoil-ext.wgsl's gather (same window/zone/quadrature structure, the
//            registry's JS twin of each kernel). |cpu − gt| isolates the QUADRATURE scheme's error — it sits
//            below the sampling noise when the N_GL / X_SPLITS accounting in kernels.js is right.
//   • gpu  — the actual shader render (8-bit target, so quantization floors this at ~1/510).
//
// |gpu − cpu| then isolates f32-vs-f64 + quantization; a WGSL port bug shows up there and nowhere else.
// The self-crossing stars keep the fold-model caveat visible (docs/ALGORITHM.md §4): the winding FOLD is the
// same saturating model as the box shader, so the stars' crossing pixels deviate from the indicator ground
// truth identically for every kernel — that deviation is the fold, not the kernel machinery.

import { renderToRGBA } from '../src/gpu.js';
import { pushMonotonePieces } from '../src/geometry.js';
import { bandPieces } from '../src/bands.js';
import { loadFont, glyphQuads } from '../src/font.js';
import { KERNELS, resolveKernel, loadKernelShaderCode, GL_TABLES } from '../src/kernels.js';

const S = 96; // cell size in px (smaller than validate.js's 128 — the wide-kernel ground truth is O(S²·(2RF)²))
const F = 16; // sub-sample grid per pixel for the ground truth
const argValue = (name) => {
  const i = Deno.args.indexOf(`--${name}`);
  if (i >= 0 && i + 1 < Deno.args.length) return Deno.args[i + 1];
  const eq = Deno.args.find((a) => a.startsWith(`--${name}=`));
  return eq ? eq.slice(name.length + 3) : null;
};

const font = await loadFont(new URL('../assets/Lato-Regular.ttf', import.meta.url));

// ── shapes (same constructions as tools/validate.js, sized to S) ────────────────────────────────────────
function line(x0, y0, x1, y1) {
  return [x0, y0, (x0 + x1) / 2, (y0 + y1) / 2, x1, y1];
}
function polygon(pts) {
  const out = [];
  for (let i = 0; i < pts.length; i++) out.push(...line(...pts[i], ...pts[(i + 1) % pts.length]));
  return out;
}
function rotate(pts, deg, cx = S / 2, cy = S / 2) {
  const a = (deg * Math.PI) / 180, c = Math.cos(a), s = Math.sin(a);
  return pts.map(([x, y]) => [cx + (x - cx) * c - (y - cy) * s, cy + (x - cx) * s + (y - cy) * c]);
}
function circle(cx, cy, r, n = 64) {
  const out = [];
  for (let i = 0; i < n; i++) {
    const a0 = (i / n) * 2 * Math.PI, a1 = ((i + 1) / n) * 2 * Math.PI, am = (a0 + a1) / 2;
    const k = 1 / Math.cos((a1 - a0) / 2);
    out.push(cx + r * Math.cos(a0), cy + r * Math.sin(a0), cx + r * k * Math.cos(am), cy + r * k * Math.sin(am),
      cx + r * Math.cos(a1), cy + r * Math.sin(a1));
  }
  return out;
}
function starPts(cx, cy, r, points, step) {
  const p = [];
  for (let k = 0; k < points; k++) {
    const a = -Math.PI / 2 + ((k * step) % points) * (2 * Math.PI / points);
    p.push([cx + r * Math.cos(a), cy + r * Math.sin(a)]);
  }
  return p;
}
function glyphShape(ch) {
  const g = glyphQuads(font, ch);
  const [x0, y0, x1, y1] = g.bbox;
  const gw = x1 - x0, gh = y1 - y0, pad = 12, box = S - 2 * pad;
  const k = Math.min(box / gw, box / gh);
  const ox = pad + (box - gw * k) / 2 - x0 * k, oy = pad + (box - gh * k) / 2 - y0 * k;
  return g.quads.map((v, i) => (i % 2 === 0 ? ox + v * k : oy + v * k));
}

const q = (f) => (f * S) / 128; // validate.js geometry was authored for a 128 cell
const SHAPES = [
  ['rotated square 30°', polygon(rotate([[q(28), q(28)], [q(100), q(28)], [q(100), q(100)], [q(28), q(100)]], 30)), false],
  ['thin diagonal sliver', polygon(rotate([[q(12), q(63.5)], [q(116), q(63.5)], [q(116), q(64.5)], [q(12), q(64.5)]], 27)), false],
  ['circle', circle(S / 2, S / 2, q(44)), false],
  ["glyph 'o' (hole)", glyphShape('o'), false],
  ['star {5/2} nonzero', polygon(starPts(S / 2, S / 2, q(52), 5, 2)), false],
  ['star {5/2} even-odd', polygon(starPts(S / 2, S / 2, q(52), 5, 2)), true],
];

// ── ground truth: point-sampled fill indicator ⊛ kernel density ─────────────────────────────────────────
// Signed winding W and crossing count K of a rightward ray from (px,py) against the raw quads (identical
// construction to tools/validate.js — ray casting, no area integral anywhere).
function windingAt(px, py, quads) {
  let W = 0, K = 0;
  for (let i = 0; i < quads.length; i += 6) {
    const x0 = quads[i], y0 = quads[i + 1], cx = quads[i + 2], cy = quads[i + 3], x1 = quads[i + 4], y1 = quads[i + 5];
    if ((y0 < py && cy < py && y1 < py) || (y0 > py && cy > py && y1 > py)) continue; // hull y-reject
    const a = y0 - 2 * cy + y1, b = 2 * (cy - y0), c = y0 - py;
    let t0 = -1, t1 = -1;
    if (Math.abs(a) < 1e-9) {
      if (Math.abs(b) > 1e-12) t0 = -c / b;
    } else {
      const disc = b * b - 4 * a * c;
      if (disc >= 0) { const sq = Math.sqrt(disc); t0 = (-b + sq) / (2 * a); t1 = (-b - sq) / (2 * a); }
    }
    for (const t of [t0, t1]) {
      if (t < 0 || t > 1) continue;
      const xt = (1 - t) * (1 - t) * x0 + 2 * (1 - t) * t * cx + t * t * x1;
      if (xt > px) { K++; const dy = 2 * a * t + b; W += dy >= 0 ? 1 : -1; }
    }
  }
  return { W, K };
}

// One shared indicator grid per shape (F samples per pixel axis, padded by the widest kernel's reach), then
// each kernel is a weighted sum over it — the expensive ray casting is paid once, not per kernel.
function indicatorGrid(quads, evenodd, padPx) {
  const G = (S + 2 * padPx) * F;
  const grid = new Uint8Array(G * G);
  for (let j = 0; j < G; j++) {
    const py = (j + 0.5) / F - padPx;
    for (let i = 0; i < G; i++) {
      const { W, K } = windingAt((i + 0.5) / F - padPx, py, quads);
      grid[j * G + i] = (evenodd ? (K & 1) === 1 : W !== 0) ? 1 : 0;
    }
  }
  return { grid, G, padPx };
}

function groundTruth({ grid, G, padPx }, density, rx, ry) {
  // per-pixel: Σ indicator(sample) · k(u,v) / F²  over samples within the support. Pixel x's first sample
  // column is (x + padPx)·F, so sample (base + d) sits at offset u = (d + 0.5)/F − 0.5 from the pixel center.
  const off = (d) => (d + 0.5) / F - 0.5;
  const dLo = (r) => Math.floor((0.5 - r) * F - 0.5), dHi = (r) => Math.ceil((0.5 + r) * F - 0.5);
  const w = [];
  for (let dj = dLo(ry); dj <= dHi(ry); dj++) {
    for (let di = dLo(rx); di <= dHi(rx); di++) {
      const kv = density(off(di), off(dj));
      if (kv !== 0) w.push([dj, di, kv / (F * F)]);
    }
  }
  const out = new Float64Array(S * S);
  for (let y = 0; y < S; y++) {
    const bj = (y + padPx) * F;
    for (let x = 0; x < S; x++) {
      const bi = (x + padPx) * F;
      let acc = 0;
      for (const [dj, di, kv] of w) acc += grid[(bj + dj) * G + (bi + di)] * kv;
      out[y * S + x] = acc;
    }
  }
  return out;
}

// ── the f64 CPU replica of windfoil-ext.wgsl ─────────────────────────────────────────────────────────────
function monoRoot(a2, a1, a0, e1, v, rising) {
  if (rising ? a0 >= v : a0 <= v) return 0;
  if (rising ? e1 <= v : e1 >= v) return 1;
  const c = a0 - v;
  if (Math.abs(a2) < 1e-12 * Math.max(Math.abs(a1), 1)) return Math.min(Math.max(-c / a1, 0), 1);
  const disc = Math.max(a1 * a1 - 4 * a2 * c, 0);
  const sq = Math.sqrt(disc);
  const qq = -0.5 * (a1 + (a1 >= 0 ? sq : -sq));
  const r1 = qq / a2, r2 = qq !== 0 ? c / qq : 0;
  const t = (a1 < 0) === rising ? r1 : r2;
  return Math.min(Math.max(t, 0), 1);
}

export function cpuReplica(spec, curves, rows, header, rc, s) {
  const { rx, ry, glOrder, nSub = 1, yKnots = [], xSplits = [], ref } = spec;
  const gl = GL_TABLES[glOrder];
  const invS = [1 / s[0], 1 / s[1]];

  const glSegment = (a2x, a2y, a1x, a1y, q1x, q1y, ta, tb) => {
    if (tb <= ta) return 0;
    const dsub = (tb - ta) / nSub;
    let acc = 0;
    for (let k = 0; k < nSub; k++) {
      const tm = ta + (k + 0.5) * dsub, dt = 0.5 * dsub;
      for (let i = 0; i < gl.x.length; i++) {
        const t = tm + dt * gl.x[i];
        acc += gl.w[i] * ref.xcum((a2x * t + a1x) * t + q1x, (a2y * t + a1y) * t + q1y) * (2 * a2y * t + a1y) * dt;
      }
    }
    return acc;
  };

  const integratePiece = (q1x, q1y, q2x, q2y, q3x, q3y, lo, hi) => {
    const a2x = q1x - 2 * q2x + q3x, a2y = q1y - 2 * q2y + q3y;
    const a1x = 2 * (q2x - q1x), a1y = 2 * (q2y - q1y);
    const yR = q3y >= q1y;
    const tLo = monoRoot(a2y, a1y, q1y, q3y, yR ? lo : hi, yR);
    const tHi = monoRoot(a2y, a1y, q1y, q3y, yR ? hi : lo, yR);
    if (tHi <= tLo) return 0;
    const xR = q3x >= q1x;
    const clampT = (t) => Math.min(Math.max(t, tLo), tHi);
    const tLeft = clampT(monoRoot(a2x, a1x, q1x, q3x, -rx, xR));
    const tRight = clampT(monoRoot(a2x, a1x, q1x, q3x, rx, xR));
    const t1 = xR ? tLeft : tRight;
    const t2 = Math.max(xR ? tRight : tLeft, t1);
    // INSIDE splits at X_SPLITS (in u) and Y_KNOTS (in v), hull-prefiltered and sorted in t — mirrors
    // integrate_piece exactly
    const uLo = Math.min(q1x, q3x), uHi = Math.max(q1x, q3x);
    const cuts = [];
    for (const sp of xSplits) {
      if (sp > uLo && sp < uHi) cuts.push(Math.min(Math.max(monoRoot(a2x, a1x, q1x, q3x, sp, xR), t1), t2));
    }
    for (const kn of yKnots) {
      if (kn > lo && kn < hi) cuts.push(Math.min(Math.max(monoRoot(a2y, a1y, q1y, q3y, kn, yR), t1), t2));
    }
    cuts.sort((a, b) => a - b);
    let acc = 0, ta = t1;
    for (const tc of cuts) {
      acc += glSegment(a2x, a2y, a1x, a1y, q1x, q1y, ta, tc);
      ta = Math.max(ta, tc);
    }
    acc += glSegment(a2x, a2y, a1x, a1y, q1x, q1y, ta, t2);
    const ra = xR ? t2 : tLo, rb = xR ? tHi : t1;
    if (rb > ra) {
      const va = Math.min(Math.max((a2y * ra + a1y) * ra + q1y, lo), hi);
      const vb = Math.min(Math.max((a2y * rb + a1y) * rb + q1y, lo), hi);
      acc += ref.ycdf(vb) - ref.ycdf(va);
    }
    return acc;
  };

  const integrateBand = (start, count, wlo, whi) => {
    let acc = 0;
    const rxg = rx * s[0];
    for (let i = 0; i < count; i++) {
      const b = (start + i) * 6;
      const q1x = curves[b] - rc[0], q1y = curves[b + 1] - rc[1];
      const q2x = curves[b + 2] - rc[0], q2y = curves[b + 3] - rc[1];
      const q3x = curves[b + 4] - rc[0], q3y = curves[b + 5] - rc[1];
      const xMax = Math.max(q1x, q2x, q3x);
      if (xMax <= -rxg) continue; // replica skips the sort-order break (order-independent in f64)
      const lo = Math.max(wlo, Math.min(q1y, q3y));
      const hi = Math.min(whi, Math.max(q1y, q3y));
      if (hi <= lo) continue;
      const xMin = Math.min(q1x, q2x, q3x);
      if (xMin >= rxg) {
        acc += ref.ycdf(Math.min(Math.max(q3y, wlo), whi) * invS[1]) - ref.ycdf(Math.min(Math.max(q1y, wlo), whi) * invS[1]);
        continue;
      }
      acc += integratePiece(q1x * invS[0], q1y * invS[1], q2x * invS[0], q2y * invS[1],
        q3x * invS[0], q3y * invS[1], lo * invS[1], hi * invS[1]);
    }
    return acc;
  };

  const { rowBase, bandCount: R, y0, invH } = header;
  const ryg = ry * s[1];
  const dy0 = y0 - rc[1];
  let ri0 = 0, ri1 = 0;
  if (invH > 0 && R > 1) {
    ri0 = Math.min(Math.max(Math.floor((-dy0 - ryg) * invH), 0), R - 1);
    ri1 = Math.min(Math.max(Math.floor((-dy0 + ryg) * invH), 0), R - 1);
  }
  let f = 0;
  for (let ri = ri0; ri <= ri1; ri++) {
    let wLo = -ryg, wHi = ryg;
    if (invH > 0) {
      wLo = Math.max(wLo, dy0 + ri / invH);
      wHi = Math.min(wHi, dy0 + (ri + 1) / invH);
    }
    if (wHi <= wLo) continue;
    const rIdx = (rowBase + ri) * 5;
    f += integrateBand(rows[rIdx], rows[rIdx + 1], wLo, wHi);
  }
  return f;
}

// ── per-shape atlas + the three coverages ────────────────────────────────────────────────────────────────
export function buildAtlas(quads) {
  const pieces = [];
  for (let i = 0; i < quads.length; i += 6) pushMonotonePieces(quads.slice(i, i + 6), pieces);
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (let i = 0; i < pieces.length; i += 2) {
    x0 = Math.min(x0, pieces[i]); x1 = Math.max(x1, pieces[i]);
    y0 = Math.min(y0, pieces[i + 1]); y1 = Math.max(y1, pieces[i + 1]);
  }
  const curveOut = [], rowOut = [];
  const header = bandPieces(pieces, y0, y1, curveOut, rowOut);
  return { curves: new Float64Array(curveOut), rows: new Uint32Array(rowOut), header, bbox: [x0, y0, x1, y1] };
}

export function fold(f, evenodd) {
  if (evenodd) {
    const m = f - 2 * Math.floor(f * 0.5);
    return Math.min(Math.max(1 - Math.abs(1 - m), 0), 1);
  }
  return Math.min(Math.abs(f), 1);
}

const stats = (a, b) => {
  let sum = 0, max = 0;
  for (let i = 0; i < a.length; i++) {
    const e = Math.abs(a[i] - b[i]);
    sum += e;
    if (e > max) max = e;
  }
  return { mean: sum / a.length, max };
};

if (!import.meta.main) {
  // imported for cpuReplica / buildAtlas / fold (e.g. by the quadrature-tuning harness) — skip the run
} else {
  await main();
}

async function main() {
const kernelArg = argValue('kernels');
const KLIST = kernelArg ? kernelArg.split(',').map((k) => k.trim()) : Object.keys(KERNELS);
const shapeArg = argValue('shapes');
const SLIST = shapeArg ? shapeArg.split(',').map(Number) : SHAPES.map((_, i) => i);
const specs = KLIST.map((name) => resolveKernel(name));
const maxR = Math.max(...specs.map((k) => Math.max(k.rx, k.ry)));
const padPx = Math.ceil(maxR) + 1;

console.log(`kernel-validate · ${S}px cell · ground truth = ${F}×${F} point-sampled indicator ⊛ kernel density`);
console.log(`kernels: ${KLIST.join(', ')}\n`);
console.log(`${'shape'.padEnd(22)} ${'kernel'.padEnd(11)}  ${'cpu vs gt'.padStart(17)}   ${'gpu vs gt'.padStart(17)}   ${'gpu vs cpu'.padStart(17)}`);
console.log(`${''.padEnd(22)} ${''.padEnd(11)}  ${'mean'.padStart(8)} ${'max'.padStart(8)}   ${'mean'.padStart(8)} ${'max'.padStart(8)}   ${'mean'.padStart(8)} ${'max'.padStart(8)}`);

let worstSimple = { cpu: 0, gpu: 0 }; // worst across non-self-crossing shapes — the machinery gate
for (const si of SLIST) {
  const [label, quads, evenodd] = SHAPES[si];
  const atlas = buildAtlas(quads);
  const ind = indicatorGrid(quads, evenodd, padPx);
  const selfCrossing = label.startsWith('star');
  for (const spec of specs) {
    const gt = groundTruth(ind, spec.ref.density, spec.rx, spec.ry);
    // Fold the ground truth exactly like the shader folds its winding integral (clamp∘abs): for the
    // negative-lobe kernels the RAW filtered value rings negative outside edges and past 1 inside, and the
    // shader's orientation-agnostic fold maps both to coverage. Folding both sides isolates MACHINERY error;
    // the ringing-vs-fold semantics (the documented ≤1.75% Mitchell / ≤4.17% Catmull-Rom halo) is a model
    // property quantified in docs/KERNELS.md, not an implementation defect this tool should flag.
    for (let i = 0; i < gt.length; i++) gt[i] = Math.min(Math.abs(gt[i]), 1);
    // cpu replica (box goes through box-ext's spec — same machinery the ext shader would run)
    const rSpec = spec.core ? resolveKernel('box-ext') : spec;
    const cpu = new Float64Array(S * S);
    for (let y = 0; y < S; y++) {
      for (let x = 0; x < S; x++) {
        cpu[y * S + x] = fold(cpuReplica(rSpec, atlas.curves, atlas.rows, atlas.header, [x + 0.5, y + 0.5], [1, 1]), evenodd);
      }
    }
    // gpu render, white on black → byte = 255·coverage
    const rule = evenodd ? 1 : 0;
    const [x0, y0, x1, y1] = atlas.bbox;
    const instances = new Float32Array([0, 0, 1, rule, x0, y0, x1, y1, 1, 1, 1, 1,
      atlas.header.rowBase, atlas.header.bandCount, atlas.header.y0, atlas.header.invH]);
    const rgba = await renderToRGBA({
      width: S, height: S, background: [0, 0, 0, 1],
      curves: new Float32Array(atlas.curves), rows: atlas.rows, instances, instanceCount: 1,
      code: await loadKernelShaderCode(spec.name),
    });
    const gpu = new Float64Array(S * S);
    for (let i = 0; i < gpu.length; i++) gpu[i] = rgba[i * 4] / 255;

    const cg = stats(cpu, gt), gg = stats(gpu, gt), gc = stats(gpu, cpu);
    if (!selfCrossing) {
      worstSimple.cpu = Math.max(worstSimple.cpu, cg.max);
      worstSimple.gpu = Math.max(worstSimple.gpu, gc.max);
    }
    const f = (v) => v.toFixed(5);
    console.log(
      `${label.padEnd(22)} ${spec.name.padEnd(11)}  ${f(cg.mean).padStart(8)} ${f(cg.max).padStart(8)}   ` +
        `${f(gg.mean).padStart(8)} ${f(gg.max).padStart(8)}   ${f(gc.mean).padStart(8)} ${f(gc.max).padStart(8)}`,
    );
  }
}

console.log(
  `\nGates (non-self-crossing shapes): worst |cpu−gt| ${worstSimple.cpu.toFixed(5)} — quadrature scheme vs the ` +
    `true filter (floor: ~1/F=${(1 / F).toFixed(3)} point-sample noise on edge pixels, so ≲0.02 is at the floor);` +
    `\nworst |gpu−cpu| ${worstSimple.gpu.toFixed(5)} — WGSL f32 + 8-bit quantization vs the f64 replica (floor ~1/255≈0.004).` +
    `\nThe stars' larger deviations are the shared saturating winding-fold model (docs/ALGORITHM.md §4), kernel-independent by design.`,
);
}
