// tree-shadow.js — render a dappled tree-canopy shadow on the ground, the way it actually looks: ONE foliage
// silhouette (a union of hundreds of small vector leaves; nonzero fill, so the gaps between leaves ARE the
// bright dapples), convolved with the SUN'S DISC at a per-pixel penumbra radius that grows from sharp at the
// bottom (near / low foliage, contact) to soft at the top (far / high canopy). Shadow only — no visible leaves,
// just the shadow cast on textured ground, seen at a grazing angle.
//
// The disc kernel is the physically-right one (a real area light is a disc, not a box): its penumbra is a
// smooth S-curve, and gaps smaller than the penumbra round off into the round sun-dapples you see under a real
// tree. Rendered on the CPU by the analytic boundary integral (src/kernel-coverage.js) — the twin of windfoil's
// pluggable-kernel gather — so it needs no GPU. Deterministic in --seed.
//
//   deno task tree-shadow            → assets/shadow-canopy.png
//   node tools/tree-shadow.js        (with npm deps installed)
// Flags: --width N --seed N --leaves N --near F --far F --floor F

import { makeLeaf, mulberry32 } from '../src/shapes.js';
import { pushMonotonePieces } from '../src/geometry.js';
import { buildPieceGrid, discCoverage } from '../src/kernel-coverage.js';
import { encodePNG } from '../src/png.js';

const isDeno = typeof Deno !== 'undefined';
const rawArgs = isDeno ? Deno.args : process.argv.slice(2);
const arg = (name, d) => {
  const i = rawArgs.indexOf(`--${name}`);
  return i >= 0 && i + 1 < rawArgs.length ? Number(rawArgs[i + 1]) : d;
};

const imgW = Math.round(arg('width', 1500));
const imgH = Math.round(imgW * 0.66);
const seed = Math.round(arg('seed', 11));
const nLeaves = Math.round(arg('leaves', 2600));
const rNear = arg('near', 2.2);   // penumbra radius (px) at the bottom (near / contact — sharp)
const rFar = arg('far', 22);      // penumbra radius (px) at the top (far / high canopy — soft)
const floorL = arg('floor', 0.34); // shadow floor (ambient sky light in the umbra), 0 = black

// ── value noise (hash → bilinear), a couple octaves — used for clustering foliage and speckling the ground ──
function makeNoise(seed) {
  const perm = new Uint8Array(512);
  const rng = mulberry32(seed);
  for (let i = 0; i < 256; i++) perm[i] = i;
  for (let i = 255; i > 0; i--) { const j = (rng() * (i + 1)) | 0; [perm[i], perm[j]] = [perm[j], perm[i]]; }
  for (let i = 0; i < 256; i++) perm[i + 256] = perm[i];
  const h = (x, y) => perm[(perm[x & 255] + y) & 255] / 255;
  const smooth = (t) => t * t * (3 - 2 * t);
  return (x, y) => {
    const xi = Math.floor(x), yi = Math.floor(y), xf = x - xi, yf = y - yi;
    const u = smooth(xf), v = smooth(yf);
    const a = h(xi, yi), b = h(xi + 1, yi), c = h(xi, yi + 1), d = h(xi + 1, yi + 1);
    return (a * (1 - u) + b * u) * (1 - v) + (c * (1 - u) + d * u) * v;
  };
}
const fbm = (noise, x, y, oct = 4) => {
  let s = 0, a = 0.5, f = 1, norm = 0;
  for (let o = 0; o < oct; o++) { s += a * noise(x * f, y * f); norm += a; a *= 0.5; f *= 2; }
  return s / norm;
};

// ── build the canopy: clustered small leaves, foreshortened (smaller/denser toward the far top), thinning
// toward the top-right so a corner of ground stays sunlit (as in the reference) ──
const rng = mulberry32(seed);
const clusterNoise = makeNoise(seed * 7 + 1);
const quads = [];
// scale factor for perspective: leaves shrink toward the top (far)
const persp = (y) => 0.5 + 0.5 * (y / imgH);          // 0.5 (top) → 1.0 (bottom)
// Jittered-grid (blue-noise-like) placement: even porosity EVERYWHERE, unlike Poisson scatter which clumps
// into solid patches and big holes. Each cell places one leaf a little larger than the spacing (so leaves
// overlap into a connected canopy) with a keep-probability < 1 — the dropped cells plus the inter-leaf gaps
// are the sunflecks. A noise field modulates the keep-probability so open/closed regions clump organically,
// and the sunlit top-right corner is opened up (as in the reference).
const spacing0 = arg('spacing', 15);
let placed = 0;
for (let gy = -1; gy * spacing0 < imgH + spacing0; gy++) {
  for (let gx = -1; gx * spacing0 < imgW + spacing0; gx++) {
    const cellY = gy * spacing0, cellX = gx * spacing0;
    const s = persp(cellY);
    const sp = spacing0; // grid is in image px; perspective scales the leaf size, not the grid
    const x = cellX + (rng() - 0.5) * sp * 1.1;
    const y = cellY + (rng() - 0.5) * sp * 1.1;
    if (x < -20 || x > imgW + 20 || y < -20 || y > imgH + 20) continue;
    const dens = fbm(clusterNoise, x / 165, y / 165, 4);
    // Sunlit opening only in the extreme top-right corner (as in the reference); elsewhere the canopy fills.
    const corner = Math.max(0, (x / imgW - 0.62) * 2.4) * Math.max(0, (1 - y / imgH - 0.5) * 2.2);
    // High keep + leaves well larger than the cell → a CONNECTED dark canopy that fills the frame; the
    // dropped cells and inter-leaf gaps are the bright sunflecks. Density noise clumps the openings so the
    // dapple sizes vary organically instead of being a regular screen.
    const keep = 0.79 + 0.17 * (dens - 0.5) - corner;
    if (rng() > keep) continue;
    const size = sp * (1.5 + 0.6 * rng());                    // >> cell → overlaps into a connected silhouette
    const leaf = makeLeaf({ rng, length: size, width: 0.52, angle: rng() * Math.PI * 2 });
    for (let i = 0; i < leaf.length; i += 2) { quads.push(leaf[i] + x, leaf[i + 1] * 0.82 + y); }
    placed++;
  }
}

// split into xy-monotone pieces for the boundary integral
const pieces = [];
for (let i = 0; i < quads.length; i += 6) pushMonotonePieces(quads.slice(i, i + 6), pieces);
const grid = buildPieceGrid(pieces, 26);

// penumbra field: sharp (small r) at the bottom, soft (large r) at the top; stretched vertically toward the
// top for the grazing angle; local variation so dapples aren't uniformly soft.
const softNoise = makeNoise(seed * 13 + 5);
function radii(px, py) {
  const tY = py / imgH;                       // 0 top (far) → 1 bottom (near)
  const base = rFar + (rNear - rFar) * tY;    // far→near
  const local = 0.6 + 0.9 * fbm(softNoise, px / 140, py / 140, 3);
  const rx = Math.max(0.9, base * local);
  const ry = rx * (1.7 - 0.55 * tY);          // taller kernel toward the far top (foreshortening)
  return [rx, ry];
}

// ── render ──
const groundNoise = makeNoise(seed * 3 + 2);
const groundFine = makeNoise(seed * 5 + 9);
const litRGB = [0.80, 0.785, 0.735];   // sunlit concrete, warm gray
const shMul = [floorL * 0.95, floorL, floorL * 1.14]; // umbra multiplier per channel (slightly cool)
const rgba = new Uint8Array(imgW * imgH * 4);

const t0 = performance.now();
for (let py = 0; py < imgH; py++) {
  for (let px = 0; px < imgW; px++) {
    const [rx, ry] = radii(px + 0.5, py + 0.5);
    const cov = discCoverage(grid, px + 0.5, py + 0.5, rx, ry);
    // ground texture: coarse mottle + fine aggregate speckle
    const mott = (fbm(groundNoise, px / 120, py / 120, 3) - 0.5) * 0.10;
    const spec = (groundFine(px / 1.7, py / 1.7) - 0.5) * 0.13;
    const tex = 1 + mott + spec;
    const idx = (py * imgW + px) * 4;
    for (let c = 0; c < 3; c++) {
      const lit = litRGB[c] * tex;
      const v = lit * (1 - cov * (1 - shMul[c])); // coverage darkens toward the (ambient, cool) umbra floor
      rgba[idx + c] = Math.round(Math.min(1, Math.max(0, v)) * 255);
    }
    rgba[idx + 3] = 255;
  }
}
const t1 = performance.now();

const png = encodePNG(rgba, imgW, imgH);
const outUrl = new URL('../assets/shadow-canopy.png', import.meta.url);
if (isDeno) await Deno.writeFile(outUrl, png);
else { const { writeFileSync } = await import('node:fs'); writeFileSync(outUrl, png); }
console.log(`${placed} leaves → ${pieces.length / 6} monotone pieces; ${imgW}×${imgH} in ${((t1 - t0) / 1000).toFixed(1)}s`);
console.log(`wrote assets/shadow-canopy.png (${(png.length / 1024).toFixed(1)} KB)`);
