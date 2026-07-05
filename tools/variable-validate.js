// variable-validate.js — correctness checks for windfoil-variable.wgsl (`deno task validate:variable`).
//
// The variable shader adds two things over the core box gather: a per-pixel OKLab→sRGB decode through the
// uniform matrices, and a per-pixel box widening. This tool checks the parts that are new:
//
//   1. Color round-trip. A flat, crisp, fully-opaque fill must reproduce its authored sRGB EXACTLY (within
//      one 8-bit level): sRGB → OKLab (CPU, @texel/color) → blend → OKLab → sRGB (GPU, uniform matrices) is
//      the identity for a single color. This is what makes a flat variable shape match the plain pipeline.
//   2. Blur mass. Widening the box is a box blur, which must CONSERVE coverage: the total ink of a blurred
//      shape equals the total ink of the same shape rendered crisp (a box filter has unit mass). We render a
//      disc crisp and heavily blurred and compare summed coverage.
//
// Independent of the GPU area integral itself, which the core `deno task validate` already checks bit-for-bit.

import { convert, OKLab, sRGB } from '@texel/color';
import { buildVariableScene } from '../src/variable.js';
import { renderVariableToRGBA } from '../src/variable-gpu.js';

const BG = [0, 0, 0, 1]; // opaque black; crisp opaque fills are unaffected, blurred alpha reads cleanly

function circle(colors, blurs, n = colors.length) {
  const anchors = [], controls = [];
  const half = Math.PI / n, rc = 1 / Math.cos(half);
  for (let k = 0; k < n; k++) {
    const th = (2 * Math.PI * k) / n;
    anchors.push({
      x: Math.cos(th),
      y: Math.sin(th),
      color: colors[k % colors.length],
      blur: blurs[k % blurs.length],
    });
    controls.push({ x: rc * Math.cos(th + half), y: rc * Math.sin(th + half) });
  }
  return { anchors, controls };
}

async function renderOne(shape, width, height, background = BG) {
  const scene = buildVariableScene([shape]);
  return renderVariableToRGBA({
    width,
    height,
    background,
    curves: scene.curves,
    rows: scene.rows,
    instances: scene.instances,
    anchors: scene.anchors,
    instanceCount: scene.instanceCount,
  });
}

// ── 1. Color round-trip ───────────────────────────────────────────────────────────────────────────────────
// A crisp square (straight edges) of a single color, sampled well inside so coverage is exactly 1.
const COLORS = ['#12a5b0', '#3a7bd5', '#e8384f', '#f9d423', '#101418', '#c9184a', '#6a4bd6'];
const S = 80;
let worstColor = 0;
for (const hex of COLORS) {
  const square = {
    anchors: [
      { x: -1, y: -1, color: hex, blur: 0 },
      { x: 1, y: -1, color: hex, blur: 0 },
      { x: 1, y: 1, color: hex, blur: 0 },
      { x: -1, y: 1, color: hex, blur: 0 },
    ],
    place: { x: S / 2, y: S / 2, scale: S * 0.32 },
    maxBlur: 0,
    falloff: 2,
  };
  const rgba = await renderOne(square, S, S);
  // Expected 8-bit sRGB from the authored color, via the same library the anchors were built with.
  const [r, g, b] = convert(convert([...hexToRgb(hex)], sRGB, OKLab), OKLab, sRGB);
  const exp = [r, g, b].map((v) => Math.round(clamp01(v) * 255));
  // Sample a 10×10 block at the centre (fully interior → coverage 1).
  let worst = 0;
  for (let y = S / 2 - 5; y < S / 2 + 5; y++) {
    for (let x = S / 2 - 5; x < S / 2 + 5; x++) {
      const i = (y * S + x) * 4;
      for (let c = 0; c < 3; c++) worst = Math.max(worst, Math.abs(rgba[i + c] - exp[c]));
    }
  }
  worstColor = Math.max(worstColor, worst);
  console.log(`  ${hex} → sRGB8 [${exp.join(', ')}]  worst |Δ| = ${worst} level(s)`);
}
console.log(`color round-trip: worst ${worstColor} of 255 (pass ≤ 1)\n`);

// ── 2. Blur mass conservation ─────────────────────────────────────────────────────────────────────────────
// Same disc, once crisp and once heavily blurred; a box blur has unit mass, so summed coverage must match.
// TRANSPARENT background so the premultiplied alpha channel reads coverage directly (an opaque bg would pin
// every pixel to α = 1 and measure nothing).
const D = 200;
function discSum(maxBlur) {
  const shape = {
    ...circle(['#ffffff'], [maxBlur > 0 ? 1 : 0], 24),
    place: { x: D / 2, y: D / 2, scale: D * 0.28 },
    maxBlur,
    falloff: 2,
  };
  return renderOne(shape, D, D, [0, 0, 0, 0]).then((rgba) => {
    let sum = 0;
    for (let i = 0; i < rgba.length; i += 4) sum += rgba[i + 3]; // alpha = coverage (white disc on empty bg)
    return sum;
  });
}
const crisp = await discSum(0);
const blurred = await discSum(0.5); // ~28 px box widening
const rel = Math.abs(blurred - crisp) / crisp;
console.log(
  `blur mass: crisp Σα = ${crisp}, blurred Σα = ${blurred}, relative Δ = ${
    (rel * 100).toFixed(3)
  }% (pass < 1%)`,
);

// ── 3. Scale invariance ───────────────────────────────────────────────────────────────────────────────────
// Blur is authored in SHAPE units, so a shape drawn 2× larger must have a 2× wider soft edge (in px) — the
// resolution-independence claim. Render a uniformly-blurred disc at scale k and 2k and measure the width of
// its edge coverage ramp (10%→90%) on the centre scanline; that width should double.
const W = 520;
async function edgeWidthPx(scale) {
  const shape = {
    ...circle(['#ffffff'], [1], 40),
    place: { x: W / 2, y: W / 2, scale },
    maxBlur: 0.5,
    falloff: 2,
  };
  const rgba = await renderOne(shape, W, W, [0, 0, 0, 0]);
  const row = W / 2; // centre scanline; walk from the left edge inward through the soft ramp
  const a = (x) => rgba[(row * W + x) * 4 + 3];
  let x10 = -1, x90 = -1;
  for (let x = 0; x < W / 2; x++) {
    if (x10 < 0 && a(x) >= 0.1 * 255) x10 = x;
    if (a(x) >= 0.9 * 255) {
      x90 = x;
      break;
    }
  }
  return x90 - x10;
}
const w1 = await edgeWidthPx(90);
const w2 = await edgeWidthPx(180);
const ratio = w2 / w1;
console.log(
  `scale invariance: edge ramp ${w1}px @1× → ${w2}px @2×, ratio ${ratio.toFixed(3)} (pass 1.9–2.1)`,
);

const ok = worstColor <= 1 && rel < 0.01 && ratio > 1.9 && ratio < 2.1;
console.log(`\n${ok ? 'PASS' : 'FAIL'}`);
if (!ok) Deno.exit(1);

function hexToRgb(hex) {
  const h = hex.replace('#', '');
  return [0, 2, 4].map((o) => parseInt(h.slice(o, o + 2), 16) / 255);
}
function clamp01(v) {
  return Math.min(Math.max(v, 0), 1);
}
