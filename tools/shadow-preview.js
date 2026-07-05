// shadow-preview.js — render the soft-shadow canopy to a PNG on the CPU, using the analytic twin of the shader
// (src/cpu-coverage.js). No GPU required, so it runs anywhere (CI, a headless box) and is how the repo's
// preview image is generated. The interactive, GPU version of the same scene is demo/shadows/.
//
//   deno run -A tools/shadow-preview.js            → assets/shadow-canopy.png
//   node tools/shadow-preview.js                   (with npm deps installed)
// Flags: --width N  --seed N  --count N  --density F  --softness F  --length F  --leaves (draw the leaf layer)

import { buildCanopy, packShadows, packLeaves, DEFAULT_SUN } from '../demo/shadows/scene.js';
import { FLOATS_PER_INSTANCE } from '../src/layout.js';
import { coverageAt } from '../src/cpu-coverage.js';
import { encodePNG } from '../src/png.js';

const isDeno = typeof Deno !== 'undefined';
const rawArgs = isDeno ? Deno.args : process.argv.slice(2);
function arg(name, fallback) {
  const i = rawArgs.indexOf(`--${name}`);
  if (i >= 0 && (i + 1 >= rawArgs.length || rawArgs[i + 1].startsWith('--'))) return true; // boolean flag
  return i >= 0 ? Number(rawArgs[i + 1]) : fallback;
}

const worldW = 1600, worldH = 1000;
const imgW = Math.round(arg('width', 1440));
const Z = imgW / worldW;
const imgH = Math.round(worldH * Z);
const seed = Math.round(arg('seed', 7));
const count = Math.round(arg('count', 240));
const density = arg('density', 0.52);
const drawLeaves = arg('leaves', false) === true;
const sun = { ...DEFAULT_SUN, softness: arg('softness', DEFAULT_SUN.softness), length: arg('length', DEFAULT_SUN.length) };

const BG = [233 / 255, 227 / 255, 213 / 255];

const { curves, rows, leaves } = buildCanopy({ seed, worldW, worldH, count });
const shadows = packShadows(leaves, { zoom: Z, sun, density });
const leafInsts = drawLeaves ? packLeaves(leaves, { alpha: 1 }) : null;

// straight-alpha RGB accumulation buffer, initialised to the ground colour
const img = new Float64Array(imgW * imgH * 3);
for (let i = 0; i < imgW * imgH; i++) { img[i * 3] = BG[0]; img[i * 3 + 1] = BG[1]; img[i * 3 + 2] = BG[2]; }

// Composite one instanced layer (shadows or leaves) with premultiplied "over".
function drawLayer(data, n) {
  for (let k = 0; k < n; k++) {
    const o = k * FLOATS_PER_INSTANCE;
    const ox = data[o], oy = data[o + 1], unitsToPx = data[o + 2];
    const x0 = data[o + 4], y0 = data[o + 5], x1 = data[o + 6], y1 = data[o + 7];
    const cr = data[o + 8], cg = data[o + 9], cb = data[o + 10], ca = data[o + 11];
    const band = { rowBase: data[o + 12], bandCount: data[o + 13], y0: data[o + 14], invH: data[o + 15] };
    const blur = [data[o + 16], data[o + 17], data[o + 18], data[o + 19]];
    const sLocal = 1 / (unitsToPx * Z); // local units per device px
    const padLocal = (1 + blur[3]) * 0.5 * sLocal + sLocal; // skirt in local units
    // device-px bbox for this instance (world = origin + em·unitsToPx, device = world·Z)
    const toDevX = (emx) => (ox + emx * unitsToPx) * Z;
    const toDevY = (emy) => (oy + emy * unitsToPx) * Z;
    let dx0 = Math.floor(Math.min(toDevX(x0 - padLocal), toDevX(x1 + padLocal)));
    let dx1 = Math.ceil(Math.max(toDevX(x0 - padLocal), toDevX(x1 + padLocal)));
    let dy0 = Math.floor(Math.min(toDevY(y0 - padLocal), toDevY(y1 + padLocal)));
    let dy1 = Math.ceil(Math.max(toDevY(y0 - padLocal), toDevY(y1 + padLocal)));
    dx0 = Math.max(0, dx0); dy0 = Math.max(0, dy0); dx1 = Math.min(imgW - 1, dx1); dy1 = Math.min(imgH - 1, dy1);
    const s = [sLocal, sLocal];
    for (let py = dy0; py <= dy1; py++) {
      const wy = (py + 0.5) / Z;
      const rcy = (wy - oy) / unitsToPx;
      for (let px = dx0; px <= dx1; px++) {
        const wx = (px + 0.5) / Z;
        const rcx = (wx - ox) / unitsToPx;
        const cov = coverageAt(curves, rows, band, [x0, y0], rcx, rcy, s, blur);
        if (cov <= 0) continue;
        const a = cov * ca;
        if (a <= 0) continue;
        const idx = (py * imgW + px) * 3;
        img[idx] = cr * a + img[idx] * (1 - a);
        img[idx + 1] = cg * a + img[idx + 1] * (1 - a);
        img[idx + 2] = cb * a + img[idx + 2] * (1 - a);
      }
    }
  }
}

const t0 = performance.now();
drawLayer(shadows, leaves.length);
if (leafInsts) drawLayer(leafInsts, leaves.length);
const t1 = performance.now();

const rgba = new Uint8Array(imgW * imgH * 4);
for (let i = 0; i < imgW * imgH; i++) {
  rgba[i * 4] = Math.round(Math.min(1, Math.max(0, img[i * 3])) * 255);
  rgba[i * 4 + 1] = Math.round(Math.min(1, Math.max(0, img[i * 3 + 1])) * 255);
  rgba[i * 4 + 2] = Math.round(Math.min(1, Math.max(0, img[i * 3 + 2])) * 255);
  rgba[i * 4 + 3] = 255;
}

const png = encodePNG(rgba, imgW, imgH);
const outUrl = new URL('../assets/shadow-canopy.png', import.meta.url);
if (isDeno) {
  await Deno.writeFile(outUrl, png);
} else {
  const { writeFileSync } = await import('node:fs');
  writeFileSync(outUrl, png);
}
console.log(`rendered ${count} leaf shadows${drawLeaves ? ' + leaves' : ''} → ${imgW}×${imgH} in ${(t1 - t0).toFixed(0)} ms`);
console.log(`wrote ${isDeno ? outUrl.pathname : new URL(outUrl).pathname} (${(png.length / 1024).toFixed(1)} KB)`);
