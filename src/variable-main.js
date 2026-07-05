// variable-main.js — demo for windfoil-variable.wgsl (`deno task render:variable`).
//
// Four filled shapes that each vary two independent fields across one crisp analytic fill:
//   1. big disc  — an OKLab hue wheel (color varies) AND a crisp→soft blur ramp (blur varies), together.
//   2. disc      — one teal, blur ramps top(crisp) → bottom(soft): the BLUR axis alone.
//   3. disc      — vivid corners, blur 0 everywhere: the COLOR axis alone (still pixel-crisp edges).
//   4. blob      — a warm palette with patchy per-anchor blur: both, on a wobbly outline.
//
// Everything is resolution-independent: blur is authored in shape units, so it scales with the shape exactly
// like the geometry. Colors are authored in sRGB, blended per pixel in OKLab. Writes an anti-aliased PNG.

import { buildVariableScene } from './variable.js';
import { renderVariableToRGBA } from './variable-gpu.js';
import { encodePNG } from './png.js';

const BG = [244, 241, 236, 0xff].map((x) => x / 0xff); // soft warm paper
const MARGIN = 64;

// A circle as N quadratic arcs (exact tangent-intersection controls), one anchor per arc endpoint. `colors`
// and `blurs` are length-N, assigned around the ring starting at angle 0 (3 o'clock), going clockwise (Y-down).
function circle(cx, cy, r, { colors, blurs, n = colors.length }) {
  const anchors = [];
  const controls = [];
  const half = Math.PI / n;
  const rc = r / Math.cos(half); // tangent-intersection radius for a quadratic arc of half-angle `half`
  for (let k = 0; k < n; k++) {
    const th = (2 * Math.PI * k) / n;
    anchors.push({
      x: cx + r * Math.cos(th),
      y: cy + r * Math.sin(th),
      color: colors[k % colors.length],
      blur: blurs[k % blurs.length],
    });
    const tc = th + half;
    controls.push({ x: cx + rc * Math.cos(tc), y: cy + rc * Math.sin(tc) });
  }
  return { anchors, controls };
}

// A wobbly closed blob: N arcs on a circle whose radius is perturbed per anchor by `wob` (deterministic).
function blob(cx, cy, r, { colors, blurs, n, wob }) {
  const anchors = [];
  const controls = [];
  const half = Math.PI / n;
  const rad = (k) => r * (1 + wob[k % wob.length]);
  for (let k = 0; k < n; k++) {
    const th = (2 * Math.PI * k) / n;
    anchors.push({
      x: cx + rad(k) * Math.cos(th),
      y: cy + rad(k) * Math.sin(th),
      color: colors[k % colors.length],
      blur: blurs[k % blurs.length],
    });
    // Control on the bisector; radius averaged from the two neighbours' tangent radii for a smooth join.
    const rc = 0.5 * (rad(k) + rad(k + 1)) / Math.cos(half);
    const tc = th + half;
    controls.push({ x: cx + rc * Math.cos(tc), y: cy + rc * Math.sin(tc) });
  }
  return { anchors, controls };
}

// Blur that ramps with vertical position of each anchor around a ring: top anchors crisp, bottom soft.
function vRamp(n, lo = 0, hi = 1) {
  return Array.from({ length: n }, (_, k) => {
    const y = Math.sin((2 * Math.PI * k) / n); // −1 at top, +1 at bottom (Y-down)
    return lo + (hi - lo) * (0.5 + 0.5 * y);
  });
}

const WHEEL = ['#e8384f', '#f6a02d', '#f9d423', '#3fbf6f', '#17a2b8', '#2a6fdb', '#6a4bd6', '#d6459b'];

// Shapes are authored in LOCAL unit space centred on the origin; `place` positions them (pixel centre + scale).
const shapes = [
  // 1 — hue wheel + crisp→soft ramp. Both fields vary at once; the interior blends to a soft neutral (the
  //     mean of the ring), which is the mesh-gradient character a linear gradient can't give you.
  {
    ...circle(0, 0, 1, { colors: WHEEL, blurs: vRamp(8, 0.0, 1.0) }),
    place: { x: 540, y: 330, scale: 230 },
    maxBlur: 0.26,
    falloff: 2.0,
  },
  // 2 — BLUR only: one color, blur ramps top→bottom. Nothing about the color changes; the edge goes from
  //     pixel-crisp at the top to a wide soft skirt at the bottom.
  {
    ...circle(0, 0, 1, { colors: ['#12a5b0'], blurs: vRamp(10, 0.0, 1.0), n: 10 }),
    place: { x: 260, y: 820, scale: 150 },
    maxBlur: 0.34,
    falloff: 2.0,
  },
  // 3 — COLOR only: vivid anchors, blur 0 everywhere → edges stay exactly as crisp as the box filter, while
  //     the fill runs an organic 4-way OKLab blend. Higher falloff tightens each anchor's zone.
  {
    ...circle(0, 0, 1, { colors: ['#d6459b', '#f6a02d', '#2a6fdb', '#3fbf6f'], blurs: [0], n: 4 }),
    place: { x: 560, y: 820, scale: 150 },
    maxBlur: 0.0,
    falloff: 3.0,
  },
  // 4 — both, on a wobbly outline with patchy blur (alternating crisp / soft anchors).
  {
    ...blob(0, 0, 1, {
      colors: ['#ff6b6b', '#ffd93d', '#ff924c', '#c9184a', '#ff8fab', '#ffb703'],
      blurs: [0.0, 0.9, 0.1, 1.0, 0.0, 0.7],
      n: 6,
      wob: [0.12, -0.14, 0.08, -0.05, 0.16, -0.1],
    }),
    place: { x: 860, y: 820, scale: 150 },
    maxBlur: 0.3,
    falloff: 2.2,
  },
];

const scene = buildVariableScene(shapes);
const width = Math.ceil(scene.bounds.maxX + MARGIN);
const height = Math.ceil(scene.bounds.maxY + MARGIN);

console.log(
  `Rendering ${scene.instanceCount} variable shapes ` +
    `(${scene.anchors.length / 8} anchors, ${scene.curves.length / 6} banded pieces) → ${width}×${height}`,
);
const t0 = performance.now();
const rgba = await renderVariableToRGBA({
  width,
  height,
  background: BG,
  curves: scene.curves,
  rows: scene.rows,
  instances: scene.instances,
  anchors: scene.anchors,
  instanceCount: scene.instanceCount,
});
const t1 = performance.now();

const png = encodePNG(rgba, width, height);
await Deno.mkdir(new URL('../output/', import.meta.url), { recursive: true });
const outPath = new URL('../output/windfoil-variable.png', import.meta.url);
await Deno.writeFile(outPath, png);

console.log(`  ${(t1 - t0).toFixed(1)} ms on the GPU`);
console.log(`  wrote ${Deno.realPathSync(outPath)} (${(png.length / 1024).toFixed(1)} KB)`);
