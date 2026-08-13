// diff.js — error map between any two rendered images. (deno task diff)
//
// The comparison tool can only diff renderers it can drive itself. This diffs two PNGs from anywhere: a
// Figma or Illustrator export of output/comparison/<scene>/scene.svg against that scene's truth.png, a
// browser screenshot against a native render, or one build's output against the last. It is the same error
// map the comparison writes — same |Δ|, same gain, same white-on-black — so a diff from here sits next to a
// <renderer>_diff.png and means the same thing.
//
//   deno task diff --target truth.png --test figma-export.png
//     → writes figma-export_diff.png beside the test image, and prints mean / max / how many pixels are off
//
// Flags: --target <png> the reference · --test <png> the image being judged · --amp <n> (15)
//        --tint white|heat (white) · --out <png> write somewhere other than <test>_diff.png
//
// Both images must be the same pixel size — a mismatch is an error, not something to paper over by scaling,
// because a resample would invent coverage that neither renderer produced. Alpha is composited over black,
// matching the coverage convention. Per pixel the difference is the LARGEST of the three channel
// differences, so a discrepancy in any channel shows at full strength instead of being averaged away.

import { edgeStats, stats } from './common/coverage.js';
import { AMP, diffRGBA, HEAT, WHITE } from './common/images.js';
import { coverageOf, deltaOf, loadRGBA, writePNGFile } from './common/imageio.js';
import { args } from './common/args.js';

const argv = args(Deno.args);
const targetPath = argv.string('target');
const testPath = argv.string('test');
const amp = argv.number('amp', AMP);
const tint = argv.string('tint', 'white') === 'heat' ? HEAT : WHITE;

if (!targetPath || !testPath) {
  console.error(
    'usage: deno task diff --target <reference.png> --test <image.png> [--amp 15] [--tint white|heat] [--out <png>]',
  );
  Deno.exit(1);
}

const target = await loadRGBA(targetPath);
const test = await loadRGBA(testPath);
if (target.w !== test.w || target.h !== test.h) {
  console.error(
    `size mismatch: target is ${target.w}×${target.h}, test is ${test.w}×${test.h}.\n` +
      `Re-render one of them at the other's size — rescaling here would invent coverage neither produced.`,
  );
  Deno.exit(1);
}

// `zero` is the "no difference" operand that lets the shared error-map and stats code render |delta − 0|.
const n = target.w * target.h;
const delta = deltaOf(target, test);
const zero = new Float64Array(n);
const targetCov = coverageOf(target); // the reference's own coverage, for the AA-band metric

const outPath = argv.string('out') ?? testPath.replace(/(\.png)?$/i, '_diff.png');
writePNGFile(outPath, diffRGBA(delta, zero, amp, tint), target.w, target.h);

const s = stats(delta, zero);
// |Δ| over the band the TARGET defines: the difference is delta-vs-zero, the mask is the target's coverage.
const e = edgeStats(delta, zero, targetCov);
// Signed test − target on the coverage channel, over that same band, to tell a systematic bias from noise.
const bias = edgeStats(coverageOf(test), targetCov).signed;
const over = (t) => {
  let c = 0;
  for (let i = 0; i < n; i++) if (delta[i] > t) c++;
  return c;
};
const pct = (c) => `${((100 * c) / n).toFixed(3)}%`;
console.log(`target  ${targetPath}`);
console.log(`test    ${testPath}`);
console.log(`         ${target.w}×${target.h}`);
console.log(
  `per-AA-pixel |Δ| ${e.mean.toFixed(6)}   over ${e.band} band px   signed bias ${bias >= 0 ? '+' : ''}` +
    `${bias.toFixed(6)}  ← per-AA-pixel is the size-independent figure to report`,
);
console.log(`whole-image mean ${s.mean.toFixed(6)}   max |Δ| ${s.max.toFixed(6)}  (${Math.round(s.max * 255)}/255)`);
console.log(
  `pixels differing at all: ${over(0)} (${pct(over(0))})   ` +
    `by >2/255: ${over(2 / 255)} (${pct(over(2 / 255))})   by >0.1: ${over(0.1)} (${pct(over(0.1))})`,
);
// A renderer that composites in the wrong space fails as a smooth bias across the whole ramp, not as noise
// at a few hard pixels — e.g. blending in linear light and encoding to sRGB turns 50% coverage into 188/255
// instead of 128/255. That is a colour-management bug in the export, not the renderer's anti-aliasing, and
// it would otherwise be reported as a huge AA error.
if (e.band > 0 && e.mean > 0.1 && Math.abs(bias) > 0.6 * e.mean) {
  console.log(
    `\n  ! edges are systematically ${bias > 0 ? 'BRIGHTER' : 'DARKER'} than the target ` +
      `(signed bias ${bias.toFixed(3)} of a ${e.mean.toFixed(3)} mean).\n` +
      `    A one-directional error of this size is a COMPOSITING/COLOUR mismatch, not anti-aliasing quality:\n` +
      `    blending in linear light and encoding to sRGB turns 50% coverage into 188/255 instead of 128/255.\n` +
      `    Check the export: no colour-profile conversion, 8-bit sRGB, #000 ground, #fff fill, no rescaling.`,
  );
}
console.log(`wrote ${Deno.realPathSync(outPath)}  (|Δ| ×${amp}, white = mismatch)`);
