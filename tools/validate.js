// validate.js — the Deno boot for the validation suite (deno task validate). The suite itself — shapes,
// the three coverage sources, stats, error maps — is the shared, environment-agnostic
// tools/validate/harness.js; the same harness also runs in a browser against the browser's own canvas2d
// rasterizer (`deno task serve`, then open http://localhost:8080/tools/validate/). This boot supplies the
// two host-specific pieces (the 2D canvas — @napi-rs/canvas, i.e. Skia — and a WebGPU device), prints the
// table, and writes the comparison PNGs that only a filesystem host can.

import { loadFont } from '../src/font.js';
import { requestDevice } from '../src/gpu.js';
import { encodePNG } from '../src/png.js';
import { createCanvas } from '@napi-rs/canvas';
import { AMP, diffRGBA, F, grayRGBA, S, slug, validateShapes } from './validate/harness.js';

const font = await loadFont(new URL('../assets/Lato-Regular.ttf', import.meta.url));
const device = await requestDevice();
const createContext2D = (w, h) => createCanvas(w, h).getContext('2d');

// "exact mode" (curiosity knob, see ourCoverage): `deno task validate --exact` (16×) or `--ss=N`.
// The shader renders at N× and box-averages down, so the fold's documented failures shrink ~1/N.
const ssArg = Deno.args.find((a) => a.startsWith('--ss='))?.slice(5);
const supersample = Deno.args.includes('--exact') ? 16 : Math.max(1, Number(ssArg) || 1);

// All renderers measured against the same independent box-filter reference: mean and worst-pixel |Δ|.
console.log(
  `validate · ${S}px cell · box filter = ${F}×${F} zero-AA point-sample · skia = @napi-rs/canvas · slug = bench/slug.wgsl` +
    `${supersample > 1 ? ` · ours ×${supersample} supersampled (fold per sub-pixel)` : ''}\n`,
);
console.log(
  `${'shape'.padEnd(24)}   ${'ours vs box'.padStart(17)}   ${'skia vs box'.padStart(17)}   ${'slug vs box'.padStart(17)}`,
);
console.log(
  `${''.padEnd(24)}   ${'mean'.padStart(8)} ${'max'.padStart(8)}   ${'mean'.padStart(8)} ${'max'.padStart(8)}   ${
    'mean'.padStart(8)
  } ${'max'.padStart(8)}`,
);

// Two aggregates: `all` is the whole dataset; `common` excludes the †-marked fold rows (the documented
// winding-fold limits, expected to deviate) but keeps the stars — their sliver deviation is shared by every
// single-sample renderer, not a true failure.
const agg = () => ({ n: 0, obMean: 0, sbMean: 0, lbMean: 0, obMax: 0, sbMax: 0, lbMax: 0 });
const all = agg(), common = agg();
const add = (a, ob, sb, lb) => {
  a.n++; a.obMean += ob.mean; a.sbMean += sb.mean; a.lbMean += lb.mean;
  a.obMax = Math.max(a.obMax, ob.max); a.sbMax = Math.max(a.sbMax, sb.max); a.lbMax = Math.max(a.lbMax, lb.max);
};
const panels = [];
for await (
  const { label, fold, ours, slug, canvas: skia, box, oursVsBox: ob, canvasVsBox: sb, slugVsBox: lb }
    of validateShapes({ font, createContext2D, device, supersample })
) {
  panels.push({ label, ours, slug, skia, box });
  add(all, ob, sb, lb);
  if (!fold) add(common, ob, sb, lb);
  const f = (v) => v.toFixed(5);
  console.log(
    `${label.padEnd(24)}   ${f(ob.mean).padStart(8)} ${f(ob.max).padStart(8)}   ${f(sb.mean).padStart(8)} ${
      f(sb.max).padStart(8)
    }   ${f(lb.mean).padStart(8)} ${f(lb.max).padStart(8)}${fold ? '  †' : ''}`,
  );
}
const line = (a) =>
  `ours vs box mean ${(a.obMean / a.n).toFixed(5)} max ${a.obMax.toFixed(3)} · ` +
  `skia vs box mean ${(a.sbMean / a.n).toFixed(5)} max ${a.sbMax.toFixed(3)} · ` +
  `slug vs box mean ${(a.lbMean / a.n).toFixed(5)} max ${a.lbMax.toFixed(3)}`;
console.log(
  `\nwhole dataset (${all.n} shapes):  ${line(all)}` +
    `\ncommon shapes (${common.n}, no † rows):  ${line(common)}` +
    `\n\n† = winding-fold limit cases (tools/failure.js, docs/ALGORITHM.md §4/§8): 'ours vs box' deviates there` +
    `\n  BY DESIGN — the fold cannot recover coverage once a pixel spans more than two adjacent winding levels.` +
    `\nOn common shapes, ours sits at the ${F}×${F} point-sample noise of the box reference (max ≲ 0.02); the` +
    `\n  star maxes are the same fold limit at sub-pixel self-intersection slivers, where Skia deviates too.` +
    `\nskia vs box — Skia's AA is its own model, not the exact filter. slug vs box — the other analytic model,` +
    `\n  a scalar per-pixel estimate like ours, so expect it to share the † fold-family deviations.` +
    `\nThe box filter is point-sampled from the raw curves, independent of our shader (no self-comparison).`,
);

// ── comparison images: one PNG per shape per view, in output/validation/ ─────────────────────────────
// For each shape: the four coverage renders (white = covered) and the pairwise error maps (|Δcoverage|
// amplified so faint differences show). Files: <shape>_{ours,skia,slug,box,ours_box_diff,skia_box_diff,
// slug_box_diff,ours_skia_diff,ours_slug_diff}.png.
const Z = 4, C = S * Z; // 4× nearest-neighbour upscale so individual pixels stay crisp
const outDir = new URL('../output/validation/', import.meta.url);
Deno.mkdirSync(outDir, { recursive: true });

const upscale = (src) => {
  const d = new Uint8Array(C * C * 4);
  for (let y = 0; y < C; y++) {
    for (let x = 0; x < C; x++) {
      const o = (y * C + x) * 4, s = (((y / Z) | 0) * S + ((x / Z) | 0)) * 4;
      d[o] = src[s]; d[o + 1] = src[s + 1]; d[o + 2] = src[s + 2]; d[o + 3] = 255;
    }
  }
  return d;
};
const write = (name, rgba) => Deno.writeFileSync(new URL(`${name}.png`, outDir), encodePNG(upscale(rgba), C, C));

for (const { label, ours, slug: slugCov, skia, box } of panels) {
  const s = slug(label);
  write(`${s}_ours`, grayRGBA(ours));
  write(`${s}_skia`, grayRGBA(skia));
  write(`${s}_slug`, grayRGBA(slugCov));
  write(`${s}_box`, grayRGBA(box));
  write(`${s}_ours_box_diff`, diffRGBA(ours, box));
  write(`${s}_skia_box_diff`, diffRGBA(skia, box));
  write(`${s}_slug_box_diff`, diffRGBA(slugCov, box));
  write(`${s}_ours_skia_diff`, diffRGBA(ours, skia));
  write(`${s}_ours_slug_diff`, diffRGBA(ours, slugCov));
}
console.log(
  `\nwrote ${panels.length * 9} PNGs to ${Deno.realPathSync(outDir)}` +
    `\n  <shape>_{ours,skia,slug,box,ours_box_diff,skia_box_diff,slug_box_diff,ours_skia_diff,ours_slug_diff}.png` +
    `\n  (diffs amplified x${AMP})`,
);
