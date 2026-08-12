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

import { createCanvas, loadImage } from '@napi-rs/canvas';
import { encodePNG } from '../src/png.js';
import { stats } from './common/coverage.js';
import { AMP, diffRGBA, HEAT, WHITE } from './common/images.js';
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

// A PNG → one RGB plane per channel, alpha composited over black.
async function loadRGB(path) {
  let img;
  try {
    img = await loadImage(path);
  } catch (err) {
    console.error(`cannot read image "${path}": ${err?.message ?? err}`);
    Deno.exit(1);
  }
  const w = img.width, h = img.height;
  const ctx = createCanvas(w, h).getContext('2d');
  ctx.fillStyle = '#000';
  ctx.fillRect(0, 0, w, h);
  ctx.drawImage(img, 0, 0);
  return { data: ctx.getImageData(0, 0, w, h).data, w, h, path };
}

const target = await loadRGB(targetPath);
const test = await loadRGB(testPath);
if (target.w !== test.w || target.h !== test.h) {
  console.error(
    `size mismatch: target is ${target.w}×${target.h}, test is ${test.w}×${test.h}.\n` +
      `Re-render one of them at the other's size — rescaling here would invent coverage neither produced.`,
  );
  Deno.exit(1);
}

// Per-pixel |Δ| as the max over R, G, B (identical to the channel value for the grayscale coverage images
// the comparison writes). Held as a plain coverage buffer so the shared error-map and stats code applies
// unchanged — `zero` is the "no difference" operand that lets diffRGBA render |delta − 0| = delta.
const n = target.w * target.h;
const delta = new Float64Array(n);
const zero = new Float64Array(n);
for (let i = 0; i < n; i++) {
  const o = i * 4;
  delta[i] = Math.max(
    Math.abs(target.data[o] - test.data[o]),
    Math.abs(target.data[o + 1] - test.data[o + 1]),
    Math.abs(target.data[o + 2] - test.data[o + 2]),
  ) / 255;
}

const outPath = argv.string('out') ?? testPath.replace(/(\.png)?$/i, '_diff.png');
Deno.writeFileSync(outPath, encodePNG(diffRGBA(delta, zero, amp, tint), target.w, target.h));

const s = stats(delta, zero);
const over = (t) => {
  let c = 0;
  for (let i = 0; i < n; i++) if (delta[i] > t) c++;
  return c;
};
const pct = (c) => `${((100 * c) / n).toFixed(3)}%`;
console.log(`target  ${targetPath}`);
console.log(`test    ${testPath}`);
console.log(`         ${target.w}×${target.h}`);
console.log(`mean |Δ| ${s.mean.toFixed(6)}    max |Δ| ${s.max.toFixed(6)}  (${Math.round(s.max * 255)}/255)`);
console.log(
  `pixels differing at all: ${over(0)} (${pct(over(0))})   ` +
    `by >2/255: ${over(2 / 255)} (${pct(over(2 / 255))})   by >0.1: ${over(0.1)} (${pct(over(0.1))})`,
);
console.log(`wrote ${Deno.realPathSync(outPath)}  (|Δ| ×${amp}, white = mismatch)`);
