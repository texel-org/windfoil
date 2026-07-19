// validate.js — the Deno boot for the validation suite (deno task validate). The suite itself — shapes,
// the coverage sources, the box-filter oracle, stats, error maps — is the shared, environment-agnostic
// tools/validate/harness.js; the same harness also runs in a browser against the browser's own canvas2d
// rasterizer (`deno task serve`, then open http://localhost:8080/tools/validate/). This boot supplies the
// two host-specific pieces (the 2D canvas — @napi-rs/canvas, i.e. Skia — and a WebGPU device), prints the
// table, and writes the comparison PNGs that only a filesystem host can.
//
// Flags:
//   --exact        render ours with the shader's EXACT_MODE override (in-shader true-fill sampling, no fold)
//   --convergence  run the reference-refinement study instead of the suite: the old F×F point-sampled
//                  reference vs the oracle at growing F, next to ours vs the oracle — showing the point
//                  sample converge to the oracle while ours sits at its floor long before F=24
//   --selftest     check the oracle itself against two independent f64 constructions (exact interval
//                  products on the axis-aligned bars; Sutherland–Hodgman clip areas on the rotated square)
//                  that share no code with the oracle's ray-cast/quadrature path — so the docs' oracle
//                  numbers regenerate from committed code

import { loadFont } from '../src/font.js';
import { requestDevice } from '../src/gpu.js';
import { encodePNG } from '../src/png.js';
import { createCanvas } from '@napi-rs/canvas';
import {
  AMP,
  boxCoverage,
  buildShapes,
  diffRGBA,
  edgeMaskFor,
  grayRGBA,
  oracleCoverage,
  ourCoverage,
  S,
  slug,
  statsVs,
  validateShapes,
} from './validate/harness.js';

const font = await loadFont(new URL('../assets/Lato-Regular.ttf', import.meta.url));
const createContext2D = (w, h) => createCanvas(w, h).getContext('2d');
const exact = Deno.args.includes('--exact');

// Numbers span 1e-14 (oracle residual) to 3e-1 (fold maxes): fixed-point hides the small end, so print
// 1-significant-digit scientific throughout.
const f = (v) => (v === 0 ? '0' : v.toExponential(1));
const col = (v) => f(v).padStart(8);

// ── the oracle self-test (--selftest, no GPU) ───────────────────────────────────────────────────────────
if (Deno.args.includes('--selftest')) {
  console.log(
    'selftest · the oracle vs two independent f64 constructions (no shared crossing/quadrature code)\n',
  );
  const shapes = buildShapes(font);
  {
    // Axis-aligned bars: each 24-float polygon is a rectangle, and the bars are disjoint, so exact coverage
    // is a sum of x-overlap · y-overlap interval products.
    const { quads } = shapes.find((s) => s.label === 'hairlines 4..0.125px');
    const rects = [];
    for (let i = 0; i < quads.length; i += 24) {
      let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
      for (let j = i; j < i + 24; j += 2) {
        x0 = Math.min(x0, quads[j]);
        x1 = Math.max(x1, quads[j]);
        y0 = Math.min(y0, quads[j + 1]);
        y1 = Math.max(y1, quads[j + 1]);
      }
      rects.push([x0, y0, x1, y1]);
    }
    const { cov, refErr } = oracleCoverage(quads, false);
    let max = 0;
    for (let r = 0; r < S; r++) {
      for (let c = 0; c < S; c++) {
        let v = 0;
        for (const [x0, y0, x1, y1] of rects) {
          v += Math.max(0, Math.min(x1, c + 1) - Math.max(x0, c)) *
            Math.max(0, Math.min(y1, r + 1) - Math.max(y0, r));
        }
        max = Math.max(max, Math.abs(cov[r * S + c] - v));
      }
    }
    console.log(
      `hairlines (axis-aligned bars) — oracle vs exact interval products:   max |Δ| = ${f(max)}  (ref≤ ${
        f(refErr)
      })`,
    );
  }
  {
    // Convex polygon: per-pixel coverage is the area of the pixel square clipped by the four edges
    // (Sutherland–Hodgman + shoelace, f64).
    const { quads } = shapes.find((s) => s.label === 'rotated square 30°');
    const pts = [];
    for (let i = 0; i < quads.length; i += 6) pts.push([quads[i], quads[i + 1]]);
    const clip = (poly, [ax, ay], [bx, by]) => {
      const out = [];
      const side = ([px, py]) => (bx - ax) * (py - ay) - (by - ay) * (px - ax);
      for (let i = 0; i < poly.length; i++) {
        const P = poly[i], Q = poly[(i + 1) % poly.length], sp = side(P), sq = side(Q);
        if (sp >= 0) out.push(P);
        if (sp > 0 !== sq > 0 && sp !== sq) {
          const t = sp / (sp - sq);
          out.push([P[0] + t * (Q[0] - P[0]), P[1] + t * (Q[1] - P[1])]);
        }
      }
      return out;
    };
    const area = (poly) => {
      let a = 0;
      for (let i = 0; i < poly.length; i++) {
        const [x0, y0] = poly[i], [x1, y1] = poly[(i + 1) % poly.length];
        a += x0 * y1 - x1 * y0;
      }
      return Math.abs(a) / 2;
    };
    const { cov, refErr } = oracleCoverage(quads, false);
    let max = 0;
    for (let r = 0; r < S; r++) {
      for (let c = 0; c < S; c++) {
        let poly = [[c, r], [c + 1, r], [c + 1, r + 1], [c, r + 1]];
        for (let k = 0; k < 4 && poly.length; k++) poly = clip(poly, pts[k], pts[(k + 1) % 4]);
        max = Math.max(max, Math.abs(cov[r * S + c] - (poly.length ? area(poly) : 0)));
      }
    }
    console.log(
      `rotated square 30° (convex polygon) — oracle vs Sutherland–Hodgman:  max |Δ| = ${f(max)}  (ref≤ ${
        f(refErr)
      })`,
    );
  }
  Deno.exit(0);
}

const device = await requestDevice();

// ── the reference-refinement study (--convergence) ──────────────────────────────────────────────────────
if (Deno.args.includes('--convergence')) {
  const LABELS = [
    'rotated square 30°',
    'thin diagonal sliver',
    'circle r=44',
    "glyph 'o'",
    'star {5/2} even-odd',
  ];
  console.log(
    'convergence · F×F zero-AA point sample vs the box-filter oracle, next to ours vs the oracle\n' +
      'The point sample can only change a pixel in steps of 1/F² (its per-pixel quantum), so it converges to\n' +
      'the oracle as F grows; ours holds one level, below every point sample shown — the old F=24 reference\n' +
      'measured its own noise, not ours. Stats over the edge pixels the shape actually touches.\n',
  );
  for (const { label, quads, evenodd = false } of buildShapes(font)) {
    if (!LABELS.includes(label)) continue;
    const { cov: oracle, refErr } = oracleCoverage(quads, evenodd);
    const mask = edgeMaskFor(quads);
    const ours = await ourCoverage(device, quads, evenodd, exact);
    const os = statsVs(ours, oracle, mask);
    // The integer-bbox render puts the instance quad's vertices on the rasterizer's sub-pixel grid (see
    // harness buildScene): the remaining row is the integral's own f32 floor, with the vertex-snap
    // displacement of the default render removed.
    const snapped = statsVs(await ourCoverage(device, quads, evenodd, exact, true), oracle, mask);
    console.log(`${label}   (oracle residual ≤ ${f(refErr)}, ${os.edgeN} edge px)`);
    console.log(
      `  ${'source'.padEnd(16)} ${'quantum'.padStart(8)} ${'edgeμ'.padStart(8)} ${'p99'.padStart(8)} ${
        'max'.padStart(8)
      }`,
    );
    for (const F of [6, 12, 24, 48, 96]) {
      const st = statsVs(boxCoverage(quads, evenodd, F), oracle, mask);
      console.log(
        `  ${`box ${F}×${F}`.padEnd(16)} ${col(1 / (F * F))} ${col(st.edgeMean)} ${col(st.p99)} ${
          col(st.max)
        }`,
      );
    }
    console.log(
      `  ${'ours'.padEnd(16)} ${'—'.padStart(8)} ${col(os.edgeMean)} ${col(os.p99)} ${col(os.max)}`,
    );
    console.log(
      `  ${'ours (int bbox)'.padEnd(16)} ${'—'.padStart(8)} ${col(snapped.edgeMean)} ${col(snapped.p99)} ${
        col(snapped.max)
      }\n`,
    );
  }
  Deno.exit(0);
}

// ── the suite ───────────────────────────────────────────────────────────────────────────────────────────
// All renderers measured against the same box-filter oracle. Per renderer: cell-wide mean |Δ|, then mean,
// p99 and max restricted to the edge pixels the shape touches (interior/exterior pixels agree trivially and
// would pad a cell-wide mean toward zero). `ref≤` is the oracle's own convergence residual for the shape.
console.log(
  `validate · ${S}px cell · oracle = box filter, exact-in-x + adaptive Gauss–Legendre in y (f64) · ` +
    `ours/slug = f32 readback · skia = @napi-rs/canvas (8-bit API)` +
    `${exact ? ' · ours = EXACT_MODE (8×8 true-fill sampling, no fold)' : ''}\n`,
);
const G = 4 * 8 + 3; // one renderer group: four 8-char stats + inner pad
console.log(
  `${''.padEnd(24)}   ${'ours vs oracle'.padStart(G)}   ${'skia vs oracle'.padStart(G)}   ${
    'slug vs oracle'.padStart(G)
  }`,
);
const sub = `${'mean'.padStart(8)} ${'edgeμ'.padStart(8)} ${'p99'.padStart(8)} ${'max'.padStart(8)}`;
console.log(`${'shape'.padEnd(24)}   ${sub}   ${sub}   ${sub}   ${'ref≤'.padStart(8)}`);

// Two aggregates: `all` is the whole dataset; `common` excludes the †-marked fold rows (the documented
// winding-fold limits, expected to deviate) but keeps the stars — their sliver deviation is shared by every
// single-sample renderer, not a true failure. Quantiles pool every edge pixel of every shape in the set.
const rend = () => ({ meanSum: 0, max: 0, pools: [] });
const agg = () => ({ n: 0, refErr: 0, ours: rend(), skia: rend(), slug: rend() });
const all = agg(), common = agg();
const add = (a, refErr, ob, sb, lb) => {
  a.n++;
  a.refErr = Math.max(a.refErr, refErr);
  for (const [r, st] of [[a.ours, ob], [a.skia, sb], [a.slug, lb]]) {
    r.meanSum += st.mean;
    r.max = Math.max(r.max, st.max);
    r.pools.push(st.edgeErrs);
  }
};
const panels = [];
for await (
  const {
    label,
    fold,
    ours,
    slug: slugCov,
    canvas: skia,
    oracle,
    refErr,
    oursVsOracle: ob,
    canvasVsOracle: sb,
    slugVsOracle: lb,
  } of validateShapes({ font, createContext2D, device, exact })
) {
  panels.push({ label, ours, slug: slugCov, skia, oracle });
  add(all, refErr, ob, sb, lb);
  if (!fold) add(common, refErr, ob, sb, lb);
  const group = (st) => `${col(st.mean)} ${col(st.edgeMean)} ${col(st.p99)} ${col(st.max)}`;
  const bad = ob.bad + sb.bad + lb.bad;
  console.log(
    `${label.padEnd(24)}   ${group(ob)}   ${group(sb)}   ${group(lb)}   ${col(refErr)}${fold ? '  †' : ''}` +
      `${bad ? `  ⚠ ${bad} non-finite px excluded` : ''}`,
  );
}

// Pool a renderer's edge errors across the aggregate's shapes for whole-set quantiles.
const pooled = (r) => {
  let n = 0;
  for (const p of r.pools) n += p.length;
  const errs = new Float64Array(n);
  let o = 0, sum = 0;
  for (const p of r.pools) {
    errs.set(p, o);
    o += p.length;
    for (const e of p) sum += e;
  }
  errs.sort();
  const q = (p) => (n ? errs[Math.floor(p * (n - 1))] : 0);
  return { edgeN: n, edgeMean: sum / n, p50: q(0.5), p90: q(0.9), p99: q(0.99), max: r.max };
};
const line = (a) =>
  ['ours', 'skia', 'slug'].map((k) => {
    const p = pooled(a[k]);
    return `  ${k}: cell mean ${f(a[k].meanSum / a.n)} · edge (${p.edgeN} px) mean ${f(p.edgeMean)} ` +
      `p50 ${f(p.p50)} p90 ${f(p.p90)} p99 ${f(p.p99)} max ${f(p.max)}`;
  }).join('\n');
console.log(
  `\nwhole dataset (${all.n} shapes, oracle residual ≤ ${f(all.refErr)}):\n${line(all)}` +
    `\ncommon shapes (${common.n}, no † rows):\n${line(common)}` +
    `\n\n† = winding-fold limit cases (tools/failure.js, docs/ALGORITHM.md §4/§8): 'ours vs oracle' deviates there` +
    `\n  BY DESIGN — the fold cannot recover coverage once a pixel spans more than two adjacent winding levels.` +
    `\nThe oracle is the box filter windfoil TARGETS, computed independently of the shader (interval coverage over` +
    `\n  ray-cast crossings of the raw curves; \`ref≤\` bounds its own residual). "skia vs oracle" is therefore the` +
    `\n  distance between Skia's AA — its own, deliberate reconstruction model — and this target, NOT a general` +
    `\n  fidelity ranking; on the † fold cases Skia's coverage rasterizer is the more correct one. slug vs oracle —` +
    `\n  the other analytic model, a scalar per-pixel estimate like ours, so it shares the † fold-family deviations.` +
    `\nOn common shapes the ours-vs-oracle residual has two parts: the rasterizer's sub-pixel vertex snapping of` +
    `\n  the instance quad (≤ ~1e-3 on a unit-gradient edge; zero on integer-bbox shapes — circle, spokes — and` +
    `\n  bit-identical in slug, so it is the GPU quad path, not the integral), and the integral's own f32 floor` +
    `\n  (~1e-6). The star maxes are the §4 fold limit at sub-pixel self-intersection slivers.` +
    `\nRun \`deno task validate --convergence\` for the reference-refinement study: the old F×F point-sampled` +
    `\n  reference approaching this oracle as F grows, and the snap-free \`ours (int bbox)\` decomposition.`,
);

// ── comparison images: one PNG per shape per view, in output/validation/ ─────────────────────────────
// For each shape: the four coverage renders (white = covered) and the pairwise error maps (|Δcoverage|
// amplified so faint differences show). Files: <shape>_{ours,skia,slug,oracle,ours_oracle_diff,
// skia_oracle_diff,slug_oracle_diff,ours_skia_diff,ours_slug_diff}.png.
const Z = 4, C = S * Z; // 4× nearest-neighbour upscale so individual pixels stay crisp
const outDir = new URL('../output/validation/', import.meta.url);
Deno.mkdirSync(outDir, { recursive: true });

const upscale = (src) => {
  const d = new Uint8Array(C * C * 4);
  for (let y = 0; y < C; y++) {
    for (let x = 0; x < C; x++) {
      const o = (y * C + x) * 4, s = (((y / Z) | 0) * S + ((x / Z) | 0)) * 4;
      d[o] = src[s];
      d[o + 1] = src[s + 1];
      d[o + 2] = src[s + 2];
      d[o + 3] = 255;
    }
  }
  return d;
};
const write = (name, rgba) =>
  Deno.writeFileSync(new URL(`${name}.png`, outDir), encodePNG(upscale(rgba), C, C));

for (const { label, ours, slug: slugCov, skia, oracle } of panels) {
  const s = slug(label);
  write(`${s}_ours`, grayRGBA(ours));
  write(`${s}_skia`, grayRGBA(skia));
  write(`${s}_slug`, grayRGBA(slugCov));
  write(`${s}_oracle`, grayRGBA(oracle));
  write(`${s}_ours_oracle_diff`, diffRGBA(ours, oracle));
  write(`${s}_skia_oracle_diff`, diffRGBA(skia, oracle));
  write(`${s}_slug_oracle_diff`, diffRGBA(slugCov, oracle));
  write(`${s}_ours_skia_diff`, diffRGBA(ours, skia));
  write(`${s}_ours_slug_diff`, diffRGBA(ours, slugCov));
}
console.log(
  `\nwrote ${panels.length * 9} PNGs to ${Deno.realPathSync(outDir)}` +
    `\n  <shape>_{ours,skia,slug,oracle,ours_oracle_diff,skia_oracle_diff,slug_oracle_diff,ours_skia_diff,ours_slug_diff}.png` +
    `\n  (diffs amplified x${AMP})`,
);
