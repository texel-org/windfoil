// comparison.js — one scene, rendered every way this repo can render it, one PNG per way. (deno task comparison)
//
// The validation suite (tools/validate.js) answers "how close is windfoil to the exact box filter, over a
// whole dataset, in numbers". This answers the other question: what does ONE shape actually look like out of
// each rasteriser, big enough to see. Same scene, same geometry, same output format — five renders and, for
// each of them, the error map against the ideal:
//
//   truth     the box filter — a pixel's coverage IS the fraction of its area inside the shape — estimated
//             by point-sampling an N×N grid per pixel (--samples, default 64 ⇒ 4096 samples/px) with winding
//             from ray casting the raw curves. No AA model, no shared code with any renderer below, and
//             right on the cases the winding fold gets wrong (docs/ALGORITHM.md §4/§8), so it is the ideal
//             every other image is measured against rather than a fourth opinion.
//   binary    the same sampler at N=1: one sample at the pixel centre, in or out. No anti-aliasing at all —
//             the baseline that shows what the other four are buying.
//   windfoil  this repo's shader (src/windfoil.wgsl).
//   slug      the benchmark's Slug port (bench/slug.wgsl) — the other analytic AA model.
//   chrome    @napi-rs/canvas, i.e. Skia: the production rasteriser Chrome ships.
//
// Plus scene.svg, the same geometry as a path, so an SVG engine (Figma, a browser, Illustrator) can be
// dropped into the comparison as one more renderer: it emits the identical quadratics with the identical
// fill rule on the identical black ground, so its export diffs against truth.png like everything else.
//
//   deno task comparison                                            # glyph 'G' at 512px
//   deno task comparison --scene glyph:Q --size 1024
//   deno task comparison --scene "shape:fold A ±1 cancellation" --size 256 --offset 0.5
//   deno task comparison --scene ~/tmp/rosette.svg --size 1024      # any SVG file
//   deno task comparison --list                                     # every scene name
//
// Flags: --scene <spec> (default glyph:G, see common/scenes.js) · --size <px> (512) · --samples <n> (64)
//        --amp <n> (15) · --offset <px> (0) · --exact · --list
//        --fit viewbox|ink  svg scenes only: fit the file's viewBox (default) or its ink bbox
//        --out <name>  the folder under output/comparison/ to write into (default: the scene's slug)

import { loadFont } from '../src/font.js';
import { requestDevice } from '../src/gpu.js';
import { encodePNG } from '../src/png.js';
import { createCanvas } from '@napi-rs/canvas';
import { canvasCoverage, pointCoverage, slugCoverage, stats, windfoilCoverage } from './common/coverage.js';
import { AMP, diffRGBA, grayRGBA, WHITE } from './common/images.js';
import { svgDocument } from './common/svg.js';
import { listScenes, resolveScene } from './common/scenes.js';
import { args } from './common/args.js';

const argv = args(Deno.args);
const size = argv.number('size', 512); // square render size, in px, shared by every renderer
const samples = argv.number('samples', 64); // ground-truth sub-sample grid: samples² points per pixel
const amp = argv.number('amp', AMP); // error-map gain (--amp 1 for the true, unamplified |Δ|)
const offset = argv.number('offset', 0); // sub-pixel phase of the scene against the grid (see scenes.js)
const exact = argv.has('exact'); // render windfoil with the shader's EXACT_MODE override
const spec = argv.string('scene', 'glyph:G');

const font = await loadFont(new URL('../assets/Lato-Regular.ttf', import.meta.url));
if (argv.has('list')) {
  console.log(listScenes(font).join('\n'));
  Deno.exit(0);
}

const scene = await resolveScene(spec, { font, size, offset, fit: argv.string('fit', 'viewbox') });
const { quads, evenodd, label } = scene;
for (const w of scene.warnings ?? []) console.warn(`  ! ${w}`);
const common = { size, evenodd };

const device = await requestDevice();
const createContext2D = (w, h) => createCanvas(w, h).getContext('2d');

// The one place a renderer is named. Add an entry and it picks up its own PNG, its own error map against the
// truth, and its own row in the table — nothing below is per-renderer.
const RENDERERS = [
  {
    name: 'truth',
    title: `ideal box filter, ${samples}×${samples} point-sampled`,
    reference: true, // what every other renderer's error map is measured against
    run: () => pointCoverage(quads, { ...common, samples }),
  },
  {
    name: 'binary',
    title: 'winding at the pixel centre, no AA',
    run: () => pointCoverage(quads, { ...common, samples: 1 }),
  },
  {
    name: 'windfoil',
    title: `src/windfoil.wgsl${exact ? ' · EXACT_MODE' : ''}`,
    run: () => windfoilCoverage(quads, { ...common, device, exact }),
  },
  { name: 'slug', title: 'bench/slug.wgsl', run: () => slugCoverage(quads, { ...common, device }) },
  { name: 'chrome', title: '@napi-rs/canvas (Skia)', run: () => canvasCoverage(createContext2D, quads, common) },
];

console.log(
  `comparison · ${label} · ${size}×${size} · truth = ${samples}×${samples} point-sampled box filter` +
    ` · error maps ×${amp}${offset ? ` · offset ${offset}px` : ''}\n`,
);

const outDir = new URL(`../output/comparison/${argv.string('out', scene.slug)}/`, import.meta.url);
Deno.mkdirSync(outDir, { recursive: true });
const write = (name, bytes) => Deno.writeFileSync(new URL(name, outDir), bytes);
const writePNG = (name, rgba) => write(`${name}.png`, encodePNG(rgba, size, size));

// The scene itself, for an SVG engine to rasterise: white on black, exactly like the coverage PNGs, so its
// export can be diffed against truth.png with no colour-space or polarity fiddling.
write(
  'scene.svg',
  new TextEncoder().encode(svgDocument({
    quads,
    width: size,
    height: size,
    evenodd,
    comment: `${label} · windfoil comparison scene · render at ${size}×${size} to compare with the PNGs`,
  })),
);

const results = [];
for (const r of RENDERERS) {
  const t0 = performance.now();
  const cov = await r.run();
  results.push({ ...r, cov, ms: performance.now() - t0 });
}

const truth = results.find((r) => r.reference).cov;
for (const r of results) {
  r.stats = r.reference ? null : stats(r.cov, truth);
  writePNG(r.name, grayRGBA(r.cov));
  // |renderer − truth|: black where they agree, white where they don't. Amplified ×amp, because a renderer
  // that tracks the box filter differs by a few percent at most — at gain 1 its map reads as solid black.
  if (!r.reference) writePNG(`${r.name}_diff`, diffRGBA(r.cov, truth, amp, WHITE));
}

// stats.json: the same numbers, machine-readable, so a sheet/report can be built over many runs without
// re-rendering or scraping stdout.
write(
  'stats.json',
  new TextEncoder().encode(JSON.stringify({
    scene: spec,
    label,
    slug: scene.slug,
    size,
    samples,
    amp,
    offset,
    exact,
    quads: quads.length / 6,
    renderers: results.map((r) => ({
      name: r.name,
      title: r.title,
      reference: !!r.reference,
      mean: r.stats?.mean ?? null,
      max: r.stats?.max ?? null,
      ms: +r.ms.toFixed(1),
    })),
  }, null, 2) + '\n'),
);

const col = (s, n) => String(s).padStart(n);
console.log(`${'renderer'.padEnd(10)} ${'source'.padEnd(38)} ${col('mean |Δ|', 10)} ${col('max |Δ|', 9)} ${col('ms', 8)}`);
for (const r of results) {
  console.log(
    `${r.name.padEnd(10)} ${r.title.padEnd(38)} ` +
      `${col(r.stats ? r.stats.mean.toFixed(6) : '—', 10)} ${col(r.stats ? r.stats.max.toFixed(6) : '—', 9)} ` +
      `${col(r.ms.toFixed(0), 8)}`,
  );
}

console.log(
  `\nwrote ${results.length * 2 - 1} PNGs + scene.svg + stats.json to ${Deno.realPathSync(outDir)}` +
    `\n  <renderer>.png       coverage, white = covered` +
    `\n  <renderer>_diff.png  |<renderer> − truth| ×${amp}, white = mismatch (there is no truth_diff — it IS the truth)` +
    `\n  scene.svg            the same geometry as a path; render it (Figma, a browser) and diff it against truth.png` +
    `\n\nmean/max |Δ| are against truth, on the raw coverage — unamplified, so they read the same at any --amp.` +
    `\nThe box filter's own point-sample noise is ~1/${samples}, which floors what any renderer can score here.`,
);
