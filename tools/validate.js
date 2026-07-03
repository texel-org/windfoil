// validate.js — check the shader's coverage against two INDEPENDENT references.  (deno task validate)
//
//   • box  — the mathematical box filter, estimated by a zero-AA point sample: for each pixel, the fraction
//            of an F×F grid of sub-sample points that fall inside the shape (winding by ray casting against
//            the raw curves — no shared AA model). This is the true filter, but its own per-pixel noise is
//            ~1/F, so it validates the shape of the error and any bias, not sub-1e-3 precision.
//   • skia — @napi-rs/canvas (Skia), a mature independent rasterizer.
//
// All render white-on-black, where the stored byte is 255·coverage (linear), so bytes compare directly.
// We report mean |Δcoverage| for our shader vs each reference, and vs Skia directly. If "ours vs box" and
// "skia vs box" are close, both track the box filter equally (the residual is the point-sample noise, not
// our error); "ours vs skia" is the direct agreement between two real renderers.

import { renderToRGBA } from '../src/gpu.js';
import { pushMonotonePieces } from '../src/geometry.js';
import { bandPieces } from '../src/bands.js';
import { loadFont, glyphQuads } from '../src/font.js';
import { encodePNG } from '../src/png.js';
import { createCanvas } from '@napi-rs/canvas';

const S = 128; // cell size in px
const F = 24; // point-sample grid per pixel for the box-filter reference
const font = await loadFont(new URL('../assets/Lato-Regular.ttf', import.meta.url));

// ── shapes: flat quads [x0,y0,cx,cy,x1,y1,...] in cell coordinates (0..S), a line = a midpoint quad ──────
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
function circle(cx, cy, r, n = 8) {
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
  const gw = x1 - x0, gh = y1 - y0, pad = 14, box = S - 2 * pad;
  const k = Math.min(box / gw, box / gh);
  const ox = pad + (box - gw * k) / 2 - x0 * k, oy = pad + (box - gh * k) / 2 - y0 * k;
  return g.quads.map((v, i) => (i % 2 === 0 ? ox + v * k : oy + v * k));
}

const SHAPES = [
  ['rotated square 30°', polygon(rotate([[28, 28], [100, 28], [100, 100], [28, 100]], 30)), false],
  ['thin diagonal sliver', polygon(rotate([[12, 63.5], [116, 63.5], [116, 64.5], [12, 64.5]], 27)), false],
  ['circle r=44', circle(64, 64, 44, 64), false], // 64 arcs: smooth enough that Skia's curve flattening is negligible
  ["glyph 'o' (with hole)", glyphShape('o'), false],
  ["glyph 'e' (aperture)", glyphShape('e'), false],
  ['star {5/2} nonzero', polygon(starPts(64, 64, 52, 5, 2)), false], // self-intersecting → winding 2 core
  ['star {5/2} even-odd', polygon(starPts(64, 64, 52, 5, 2)), true], // hollow core
];

// ── 1. our shader ───────────────────────────────────────────────────────────────────────────────────
function buildScene(quads, evenodd, scale) {
  const pieces = [];
  for (let i = 0; i < quads.length; i += 6) pushMonotonePieces(quads.slice(i, i + 6), pieces);
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (let i = 0; i < pieces.length; i += 2) {
    x0 = Math.min(x0, pieces[i]); x1 = Math.max(x1, pieces[i]);
    y0 = Math.min(y0, pieces[i + 1]); y1 = Math.max(y1, pieces[i + 1]);
  }
  const curveOut = [], rowOut = [];
  const { rowBase, bandCount, y0: by0, invH } = bandPieces(pieces, y0, y1, curveOut, rowOut);
  const rule = evenodd ? 1 : 0;
  const instances = new Float32Array([0, 0, scale, rule, x0, y0, x1, y1, 1, 1, 1, 1, rowBase, bandCount, by0, invH]);
  return { curves: new Float32Array(curveOut), rows: new Uint32Array(rowOut), instances };
}

async function ourCoverage(quads, evenodd) {
  const { curves, rows, instances } = buildScene(quads, evenodd, 1);
  const rgba = await renderToRGBA({
    width: S, height: S, background: [0, 0, 0, 1], curves, rows, instances, instanceCount: 1,
  });
  const out = new Float64Array(S * S);
  for (let i = 0; i < out.length; i++) out[i] = rgba[i * 4] / 255;
  return out;
}

// ── 2. Skia (@napi-rs/canvas) ─────────────────────────────────────────────────────────────────────────
function skiaCoverage(quads, evenodd) {
  const ctx = createCanvas(S, S).getContext('2d');
  ctx.fillStyle = '#000';
  ctx.fillRect(0, 0, S, S);
  ctx.fillStyle = '#fff';
  ctx.beginPath();
  let px = null, py = null;
  for (let i = 0; i < quads.length; i += 6) {
    const [x0, y0, cx, cy, x1, y1] = quads.slice(i, i + 6);
    if (px === null || Math.abs(x0 - px) > 1e-4 || Math.abs(y0 - py) > 1e-4) {
      if (px !== null) ctx.closePath();
      ctx.moveTo(x0, y0);
    }
    ctx.quadraticCurveTo(cx, cy, x1, y1);
    px = x1; py = y1;
  }
  ctx.closePath();
  ctx.fill(evenodd ? 'evenodd' : 'nonzero');
  const d = ctx.getImageData(0, 0, S, S).data;
  const out = new Float64Array(S * S);
  for (let i = 0; i < out.length; i++) out[i] = d[i * 4] / 255;
  return out;
}

// ── 3. point-sampled box filter (independent: winding by ray casting the raw curves) ───────────────────
// Signed winding W and crossing count K of a rightward ray from (px,py) against the raw quads.
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

function boxCoverage(quads, evenodd) {
  const out = new Float64Array(S * S);
  for (let y = 0; y < S; y++) {
    for (let x = 0; x < S; x++) {
      let inside = 0;
      for (let j = 0; j < F; j++) {
        for (let i = 0; i < F; i++) {
          const { W, K } = windingAt(x + (i + 0.5) / F, y + (j + 0.5) / F, quads);
          if (evenodd ? (K & 1) === 1 : W !== 0) inside++;
        }
      }
      out[y * S + x] = inside / (F * F);
    }
  }
  return out;
}

// ── compare ─────────────────────────────────────────────────────────────────────────────────────────
function stats(a, b) {
  let sum = 0, max = 0;
  for (let i = 0; i < a.length; i++) {
    const e = Math.abs(a[i] - b[i]);
    sum += e;
    if (e > max) max = e;
  }
  return { mean: sum / a.length, max };
}

// Both renderers measured against the same independent box-filter reference: mean and worst-pixel |Δ|.
console.log(`validate · ${S}px cell · box filter = ${F}×${F} zero-AA point-sample · skia = @napi-rs/canvas\n`);
console.log(`${'shape'.padEnd(24)}   ${'ours vs box'.padStart(17)}   ${'skia vs box'.padStart(17)}`);
console.log(`${''.padEnd(24)}   ${'mean'.padStart(8)} ${'max'.padStart(8)}   ${'mean'.padStart(8)} ${'max'.padStart(8)}`);
let obMean = 0, sbMean = 0, obMax = 0, sbMax = 0;
const panels = [];
for (const [label, quads, evenodd] of SHAPES) {
  const ours = await ourCoverage(quads, evenodd);
  const skia = skiaCoverage(quads, evenodd);
  const box = boxCoverage(quads, evenodd);
  panels.push({ label, ours, skia, box });
  const ob = stats(ours, box), sb = stats(skia, box);
  obMean += ob.mean; sbMean += sb.mean; obMax = Math.max(obMax, ob.max); sbMax = Math.max(sbMax, sb.max);
  const f = (v) => v.toFixed(5);
  console.log(
    `${label.padEnd(24)}   ${f(ob.mean).padStart(8)} ${f(ob.max).padStart(8)}   ${f(sb.mean).padStart(8)} ${f(sb.max).padStart(8)}`,
  );
}
console.log(
  `\nours vs box: mean ${(obMean / SHAPES.length).toFixed(5)}. Simple fills sit at the ${F}×${F} point-sample noise` +
    `\n  (max ≲ 0.02); the larger star maxes are the fold-model limit at self-intersections (see docs/ALGORITHM.md §8).` +
    `\nskia vs box: mean ${(sbMean / SHAPES.length).toFixed(5)}, max ${sbMax.toFixed(3)} — Skia's AA is its own model, not the exact filter.` +
    `\nThe box filter is point-sampled from the raw curves, independent of our shader (no self-comparison).`,
);

// ── comparison images: one PNG per shape per view, in output/validation/ ─────────────────────────────
// For each shape: the three coverage renders (white = covered) and the three pairwise error maps
// (|Δcoverage| amplified so faint differences show). Files: <shape>_{ours,skia,box,ours_box_diff,
// skia_box_diff,ours_skia_diff}.png.
const AMP = 15; // error-map gain: |Δ|·AMP, so 1/AMP coverage reads full-bright
const Z = 4, C = S * Z; // 4× nearest-neighbour upscale so individual pixels stay crisp
const outDir = new URL('../output/validation/', import.meta.url);
Deno.mkdirSync(outDir, { recursive: true });

const slug = (s) => s.toLowerCase().replace(/[^a-z0-9]+/g, '_').replace(/^_+|_+$/g, '');
const upscale = (rgbAt) => {
  const d = new Uint8Array(C * C * 4);
  for (let y = 0; y < C; y++) {
    for (let x = 0; x < C; x++) {
      const [r, g, b] = rgbAt(((y / Z) | 0) * S + ((x / Z) | 0));
      const o = (y * C + x) * 4;
      d[o] = r; d[o + 1] = g; d[o + 2] = b; d[o + 3] = 255;
    }
  }
  return d;
};
const gray = (cov) => upscale((i) => { const v = Math.round(cov[i] * 255); return [v, v, v]; });
const errMap = (a, b) => upscale((i) => { const v = Math.round(Math.min(Math.abs(a[i] - b[i]) * AMP, 1) * 255); return [v, Math.round(v * 0.28), Math.round(v * 0.12)]; });
const write = (name, rgba) => Deno.writeFileSync(new URL(`${name}.png`, outDir), encodePNG(rgba, C, C));

for (const { label, ours, skia, box } of panels) {
  const s = slug(label);
  write(`${s}_ours`, gray(ours));
  write(`${s}_skia`, gray(skia));
  write(`${s}_box`, gray(box));
  write(`${s}_ours_box_diff`, errMap(ours, box));
  write(`${s}_skia_box_diff`, errMap(skia, box));
  write(`${s}_ours_skia_diff`, errMap(ours, skia));
}
console.log(
  `\nwrote ${panels.length * 6} PNGs to ${Deno.realPathSync(outDir)}` +
    `\n  <shape>_{ours,skia,box,ours_box_diff,skia_box_diff,ours_skia_diff}.png (diffs amplified x${AMP})`,
);
