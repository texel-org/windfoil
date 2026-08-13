// report.js — finish a staged scene: delta every render against the truth, crop everything, tabulate.
// (deno task report)
//
// This is the second half of a two-step workflow:
//
//   deno task comparison --scene <spec>     stage: scene.svg + truth/binary/windfoil/slug/chrome renders
//   …hand-export scene.svg from Figma / Photoshop / a browser into the same folder…
//   deno task report --dir <that folder>    finish: <name>_diff.png for every render, crops, one table
//
// Splitting it this way is what lets a hand-exported renderer be measured at all, and it puts every delta in
// one place so they are all computed the same way rather than some in-process and some from disk.
//
// A NOTE ON PRECISION, because the table mixes two kinds of number. The renderers this repo drives are
// measured in float, in-process, and those figures are read back from stats.json. A hand-exported PNG has
// necessarily been quantised to 1/255, so it can only be measured from disk. That difference is not
// cosmetic: windfoil's per-AA-pixel error is ~0.001, well under the 1/255 = 0.0039 an 8-bit file can even
// represent, so its float figure is the only meaningful one. The `source` column says which is which.
//
// Flags: --dir <folder> (or a bare path) · --amp <n> (15) · --tint white|heat (white)
//        --zoom <n> (2) · --size <px> (source size) · --inset <px> (40) · --corner tr|tl|br|bl (tr)
//        --no-crop  deltas only · --force  redo existing crops

import { edgeStats, stats } from './common/coverage.js';
import { AMP, diffRGBA, HEAT, WHITE } from './common/images.js';
import { coverageOf, deltaOf, loadRGBA, writePNGFile } from './common/imageio.js';
import { cropRaster, cropSVG } from './common/cropper.js';
import { args } from './common/args.js';

const argv = args(Deno.args);
const dir = (argv.string('dir') ?? argv.positionals()[0])?.replace(/\/$/, '');
const amp = argv.number('amp', AMP);
const tint = argv.string('tint', 'white') === 'heat' ? HEAT : WHITE;
const doCrop = !argv.has('no-crop');
const force = argv.has('force');
const cropOpts = {
  zoom: Math.max(1, Math.round(argv.number('zoom', 2))),
  outSize: argv.number('size', 0),
  inset: argv.number('inset', 40),
  corner: (argv.string('corner', 'tr') ?? 'tr').toLowerCase(),
};

if (!dir) {
  console.error('usage: deno task report --dir <scene folder> [--amp 15] [--zoom 2] [--no-crop] [--force]');
  Deno.exit(1);
}

let entries;
try {
  entries = [...Deno.readDirSync(dir)].filter((e) => e.isFile).map((e) => e.name).sort();
} catch (err) {
  console.error(`cannot read folder "${dir}": ${err?.message ?? err}`);
  Deno.exit(1);
}
if (!entries.includes('truth.png')) {
  console.error(`"${dir}" has no truth.png — stage it first with: deno task comparison --scene <spec>`);
  Deno.exit(1);
}

// Renders to judge: every PNG that is not the reference and not something this tool produced.
const renders = entries.filter((n) =>
  n.endsWith('.png') && n !== 'truth.png' && !n.endsWith('_diff.png') && !n.endsWith('_crop.png')
);
const truth = await loadRGBA(`${dir}/truth.png`);
const truthCov = coverageOf(truth);

// Float figures for the renderers this repo drove, if the stage left them (see the precision note above).
let staged = null;
try {
  staged = JSON.parse(await Deno.readTextFile(`${dir}/stats.json`));
} catch { /* fine — everything just gets measured from disk */ }
const stagedOf = (name) => staged?.renderers?.find((r) => r.name === name && !r.reference) ?? null;

console.log(`report · ${dir} · ${truth.w}×${truth.h} · ${renders.length} render(s) vs truth · error maps ×${amp}`);
if (staged) console.log(`  stage: ${staged.label} · truth = ${staged.samples}² samples/px`);

const rows = [];
for (const name of renders) {
  const base = name.replace(/\.png$/, '');
  const img = await loadRGBA(`${dir}/${name}`);
  if (img.w !== truth.w || img.h !== truth.h) {
    console.log(`  skip  ${name} — ${img.w}×${img.h}, truth is ${truth.w}×${truth.h}; re-export at the same size`);
    continue;
  }
  const delta = deltaOf(truth, img);
  const zero = new Float64Array(delta.length);
  writePNGFile(`${dir}/${base}_diff.png`, diffRGBA(delta, zero, amp, tint), truth.w, truth.h);

  const disk = edgeStats(delta, zero, truthCov); // |Δ| over the band the truth defines
  const bias = edgeStats(coverageOf(img), truthCov).signed; // one-directional ⇒ colour/compositing, not AA
  const whole = stats(delta, zero);
  const f = stagedOf(base);
  rows.push({
    name: base,
    edgeMean: f ? f.edgeMean : disk.mean,
    edgeMax: f ? f.edgeMax : disk.max,
    mean: f ? f.mean : whole.mean,
    max: f ? f.max : whole.max,
    band: f ? f.band : disk.band,
    bias,
    source: f ? 'float' : '8-bit',
  });
}

rows.sort((a, b) => a.edgeMean - b.edgeMean);
const col = (s, n) => String(s).padStart(n);
console.log(
  `\n${'render'.padEnd(12)} ${col('per-AA-px', 10)} ${col('mean', 9)} ${col('max', 9)} ${col('bias', 10)}  source`,
);
for (const r of rows) {
  console.log(
    `${r.name.padEnd(12)} ${col(r.edgeMean.toFixed(6), 10)} ${col(r.mean.toFixed(6), 9)} ` +
      `${col(r.max.toFixed(6), 9)} ${col((r.bias >= 0 ? '+' : '') + r.bias.toFixed(5), 10)}  ${r.source}`,
  );
}
console.log(
  `\nper-AA-px = mean |Δ| over the ${rows[0]?.band ?? 0} pixels the truth puts strictly between empty and full;\n` +
    `  it is the size-independent figure. 'mean' is diluted by the background and falls ~1/size.\n` +
    `bias = signed (render − truth) on the same band. Near zero is ordinary AA disagreement; a value close to\n` +
    `  per-AA-px means every edge is pushed one way — a compositing/colour-space mismatch in the export, not\n` +
    `  the renderer's anti-aliasing. Blending in linear light and encoding to sRGB puts 50% coverage at\n` +
    `  188/255 instead of 128/255.\n` +
    `source: float = measured in-process before 8-bit output (from stats.json); 8-bit = measured from the PNG,\n` +
    `  so it cannot resolve below 1/255 = 0.0039.`,
);
for (const r of rows) {
  if (r.edgeMean > 0.1 && Math.abs(r.bias) > 0.6 * r.edgeMean) {
    console.log(
      `\n  ! ${r.name}: edges are systematically ${r.bias > 0 ? 'BRIGHTER' : 'DARKER'} than truth ` +
        `(bias ${r.bias.toFixed(3)} of a ${r.edgeMean.toFixed(3)} mean) — check the export for colour\n` +
        `    conversion, non-8-bit depth, a non-black ground, or rescaling before you record this number.`,
    );
  }
}

// ── crops ───────────────────────────────────────────────────────────────────────────────────────────────
if (doCrop) {
  const targets = [...new Set([...entries, ...renders.map((n) => n.replace(/\.png$/, '_diff.png')), 'truth.png'])]
    .filter((n) => (n.endsWith('.png') || n.endsWith('.svg')) && !/_crop\.(png|svg)$/.test(n));
  let wrote = 0, skipped = 0;
  for (const name of targets) {
    const isSVG = name.endsWith('.svg');
    const outPath = `${dir}/${name.replace(/\.(png|svg)$/, `_crop.$1`)}`;
    if (!force) {
      try {
        Deno.statSync(outPath);
        skipped++;
        continue;
      } catch { /* not there yet */ }
    }
    if (isSVG) {
      const r = cropSVG(await Deno.readTextFile(`${dir}/${name}`), cropOpts);
      if (!r || r.tooSmall) {
        skipped++;
        continue;
      }
      Deno.writeFileSync(outPath, new TextEncoder().encode(r.svg));
    } else {
      const r = cropRaster(await loadRGBA(`${dir}/${name}`), cropOpts);
      if (r.tooSmall) {
        skipped++;
        continue;
      }
      writePNGFile(outPath, r.rgba, r.cw, r.ch);
    }
    wrote++;
  }
  const win = cropWindowInfo();
  console.log(
    `\ncrops: wrote ${wrote}${skipped ? `, skipped ${skipped}${force ? '' : ' (already there — use --force)'}` : ''}` +
      ` · zoom ×${cropOpts.zoom}, ${win} · nearest-neighbour`,
  );
}
function cropWindowInfo() {
  const w = truth.w, out = cropOpts.outSize > 0 ? cropOpts.outSize : w;
  return `out ${out}px covering ${out / cropOpts.zoom}px of source, ${cropOpts.inset}px inset from ${cropOpts.corner}`;
}

console.log(`\n${Deno.realPathSync(dir)}`);
