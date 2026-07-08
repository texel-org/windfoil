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
//
// The geometry, coverage, box filter and stats all live in tools/util.js so the browser comparison tool
// (tools/chrome/) can reuse them verbatim. This script also writes output/validation/skia-ref.json — the
// napi-rs coverage bytes per shape — so that tool can diff Chrome's <canvas> directly against Skia here.

import { S, F, makeShapes, ourCoverage, canvasCoverage, boxCoverage, stats, slug } from './util.js';
import { loadFont } from '../src/font.js';
import { encodePNG } from '../src/png.js';
import { createCanvas } from '@napi-rs/canvas';

const font = await loadFont(new URL('../assets/Lato-Regular.ttf', import.meta.url));
const SHAPES = makeShapes(font);

// Skia via @napi-rs/canvas: a fresh S×S canvas per shape, fed through the shared Canvas 2D path.
const skiaCoverage = (quads, evenodd) => canvasCoverage(quads, evenodd, createCanvas(S, S).getContext('2d'));

// ── compare ─────────────────────────────────────────────────────────────────────────────────────────────
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

// ── napi-rs reference for the browser comparison (tools/chrome/) ─────────────────────────────────────────
// Store 255·coverage as a byte per pixel per shape (keyed by slug), so tools/chrome/ can fetch this and diff
// Chrome's own <canvas> Skia against this build's Skia directly, pixel-for-pixel.
const ref = { S, shapes: {} };
for (const { label, skia, box } of panels) {
  const [, , evenodd] = SHAPES.find(([l]) => l === label);
  ref.shapes[slug(label)] = {
    label,
    evenodd,
    skia: Array.from(skia, (v) => Math.round(v * 255)),
    box: Array.from(box, (v) => Math.round(v * 255)),
  };
}
Deno.writeTextFileSync(new URL('skia-ref.json', outDir), JSON.stringify(ref));
console.log(`\nwrote output/validation/skia-ref.json — napi-rs reference for tools/chrome/ (open that page to diff Chrome).`);
