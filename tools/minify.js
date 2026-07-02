// minify.js — show that the area integral IS the exact prefilter under minification.  (deno task minify)
//
// Renders the word "area" at a few tiny em sizes, three ways, magnified with nearest-neighbour so the pixels
// are visible:
//
//   • point — a single point sample at each pixel centre (the cheapest possible gather: inside/outside, no
//             prefilter). This is what you get WITHOUT an area integral — stems shimmer and drop out.
//   • area  — our shader: ONE closed-form box-integral evaluation per pixel.
//   • box   — the ground truth: a 24×24 supersampled box filter (independent code path, from validate.js).
//
// The claim of §1 (see the brainstorm): because coverage is ∫∫_box w dA, minification is not a special case —
// the same one evaluation already equals the many-sample box filter. So the `area` column should be
// indistinguishable from the `box` column, while `point` is visibly broken. We also print mean |Δ| to the
// ground truth for each, so it is a number, not just a vibe.

import { renderToRGBA } from '../src/gpu.js';
import { pushMonotonePieces } from '../src/geometry.js';
import { bandPieces } from '../src/bands.js';
import { loadFont, glyphQuads } from '../src/font.js';
import { encodePNG } from '../src/png.js';

const TEXT = 'area';
const EMS = [10, 16, 26]; // em sizes in px — small enough that the x-height is only ~5–13px (true minification)
const F = 24; // point-sample grid per pixel for the box-filter ground truth (matches validate.js)
const MARGIN = 2; // px of breathing room around the word in each little target
const ROW_TARGET = 150; // each ladder row is magnified to about this many px tall

const INK = [0.11, 0.11, 0.17];
const PAPER = [0.96, 0.95, 0.92];
const CARD = [0.82, 0.81, 0.78];
const HEAD = { point: [0.80, 0.30, 0.22], area: [0.16, 0.55, 0.52], box: [0.45, 0.45, 0.48] };

const font = await loadFont(new URL('../assets/Lato-Regular.ttf', import.meta.url));

// Lay `text` out along the baseline at `pxPerEm`, returning the glyph quads translated into a tight little
// target of size W×H px (Y-down), plus that size. Straight segments already carry a midpoint control (font.js).
function wordQuads(text, pxPerEm) {
  const scale = pxPerEm / font.unitsPerEm;
  const raw = [];
  let penX = 0;
  for (const ch of text) {
    const g = glyphQuads(font, ch);
    if (g) {
      for (let i = 0; i < g.quads.length; i += 2) raw.push(g.quads[i] * scale + penX, g.quads[i + 1] * scale);
      penX += g.advance * scale;
    } else {
      penX += font.charToGlyph(ch).advanceWidth * scale; // blank (space)
    }
  }
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (let i = 0; i < raw.length; i += 2) {
    minX = Math.min(minX, raw[i]); maxX = Math.max(maxX, raw[i]);
    minY = Math.min(minY, raw[i + 1]); maxY = Math.max(maxY, raw[i + 1]);
  }
  const dx = MARGIN - minX, dy = MARGIN - minY;
  for (let i = 0; i < raw.length; i += 2) { raw[i] += dx; raw[i + 1] += dy; }
  return { quads: raw, W: Math.ceil(maxX - minX + 2 * MARGIN), H: Math.ceil(maxY - minY + 2 * MARGIN) };
}

// ── coverage, three ways (all return Float64Array[W*H], coverage 0..1) ─────────────────────────────────

// Signed winding of a rightward ray from (px,py) against the raw quads (independent of the shader).
function windingAt(px, py, quads) {
  let W = 0;
  for (let i = 0; i < quads.length; i += 6) {
    const x0 = quads[i], y0 = quads[i + 1], cx = quads[i + 2], cy = quads[i + 3], x1 = quads[i + 4], y1 = quads[i + 5];
    if ((y0 < py && cy < py && y1 < py) || (y0 > py && cy > py && y1 > py)) continue;
    const a = y0 - 2 * cy + y1, b = 2 * (cy - y0), c = y0 - py;
    let t0 = -1, t1 = -1;
    if (Math.abs(a) < 1e-9) {
      if (Math.abs(b) > 1e-12) t0 = -c / b;
    } else {
      const disc = b * b - 4 * a * c;
      if (disc >= 0) { const sq = Math.sqrt(disc); t0 = (-b + sq) / (2 * a); t1 = (-b - sq) / (2 * a); }
    }
    for (const t of [t0, t1]) {
      if (t < 0 || t > 1) continue;
      const xt = (1 - t) * (1 - t) * x0 + 2 * (1 - t) * t * cx + t * t * x1;
      if (xt > px) W += (2 * a * t + b) >= 0 ? 1 : -1;
    }
  }
  return W;
}

function pointCoverage(quads, W, H) {
  const out = new Float64Array(W * H);
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) out[y * W + x] = windingAt(x + 0.5, y + 0.5, quads) !== 0 ? 1 : 0;
  }
  return out;
}

function boxCoverage(quads, W, H) {
  const out = new Float64Array(W * H);
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      let inside = 0;
      for (let j = 0; j < F; j++) {
        for (let i = 0; i < F; i++) if (windingAt(x + (i + 0.5) / F, y + (j + 0.5) / F, quads) !== 0) inside++;
      }
      out[y * W + x] = inside / (F * F);
    }
  }
  return out;
}

async function areaCoverage(quads, W, H) {
  const pieces = [];
  for (let i = 0; i < quads.length; i += 6) pushMonotonePieces(quads.slice(i, i + 6), pieces);
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (let i = 0; i < pieces.length; i += 2) {
    x0 = Math.min(x0, pieces[i]); x1 = Math.max(x1, pieces[i]);
    y0 = Math.min(y0, pieces[i + 1]); y1 = Math.max(y1, pieces[i + 1]);
  }
  const curveOut = [], rowOut = [];
  const { rowBase, bandCount, y0: by0, invH } = bandPieces(pieces, y0, y1, curveOut, rowOut);
  const instances = new Float32Array([0, 0, 1, 0, x0, y0, x1, y1, 1, 1, 1, 1, rowBase, bandCount, by0, invH]);
  const rgba = await renderToRGBA({
    width: W, height: H, background: [0, 0, 0, 1],
    curves: new Float32Array(curveOut), rows: new Uint32Array(rowOut), instances, instanceCount: 1,
  });
  const out = new Float64Array(W * H);
  for (let i = 0; i < out.length; i++) out[i] = rgba[i * 4] / 255; // white ink on black ⇒ red channel = coverage
  return out;
}

function meanAbs(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += Math.abs(a[i] - b[i]);
  return s / a.length;
}

// ── compose one magnified ink-on-paper grid: rows = sizes, columns = point / area / box ────────────────

const rows = [];
for (const em of EMS) {
  const { quads, W, H } = wordQuads(TEXT, em);
  const point = pointCoverage(quads, W, H);
  const area = await areaCoverage(quads, W, H);
  const box = boxCoverage(quads, W, H);
  const Z = Math.max(4, Math.round(ROW_TARGET / H));
  rows.push({ em, W, H, Z, cells: { point, area, box } });
  console.log(
    `em ${String(em).padStart(3)}px (${W}×${H})  mean |Δ| vs box:  area ${meanAbs(area, box).toFixed(4)}` +
      `   point ${meanAbs(point, box).toFixed(4)}`,
  );
}

const G = 18; // gutter
const HEADER = 10; // coloured column-header strip
const COLS = ['point', 'area', 'box'];
const colW = Math.max(...rows.map((r) => r.W * r.Z));
const canvasW = G + COLS.length * (colW + G);
const canvasH = HEADER + G + rows.reduce((s, r) => s + r.H * r.Z + G, 0);

const canvas = new Uint8Array(canvasW * canvasH * 4);
const put = (x, y, [r, g, b]) => {
  if (x < 0 || y < 0 || x >= canvasW || y >= canvasH) return;
  const o = (y * canvasW + x) * 4;
  canvas[o] = r * 255; canvas[o + 1] = g * 255; canvas[o + 2] = b * 255; canvas[o + 3] = 255;
};
for (let y = 0; y < canvasH; y++) for (let x = 0; x < canvasW; x++) put(x, y, CARD); // card background

const colX = (c) => G + c * (colW + G);
for (let c = 0; c < COLS.length; c++) { // coloured header per column
  for (let y = 0; y < HEADER - 3; y++) for (let x = colX(c); x < colX(c) + colW; x++) put(x, y, HEAD[COLS[c]]);
}

let rowY = HEADER + G;
for (const r of rows) {
  const dispW = r.W * r.Z, dispH = r.H * r.Z;
  for (let c = 0; c < COLS.length; c++) {
    const cov = r.cells[COLS[c]];
    const ox = colX(c) + ((colW - dispW) >> 1); // centre the panel in its column
    for (let y = 0; y < dispH; y++) {
      for (let x = 0; x < dispW; x++) {
        const v = cov[((y / r.Z) | 0) * r.W + ((x / r.Z) | 0)]; // nearest-neighbour magnify
        put(ox + x, rowY + y, [
          PAPER[0] + (INK[0] - PAPER[0]) * v,
          PAPER[1] + (INK[1] - PAPER[1]) * v,
          PAPER[2] + (INK[2] - PAPER[2]) * v,
        ]);
      }
    }
  }
  rowY += dispH + G;
}

await Deno.mkdir(new URL('../output/', import.meta.url), { recursive: true });
const outPath = new URL('../output/minify-compare.png', import.meta.url);
await Deno.writeFile(outPath, encodePNG(canvas, canvasW, canvasH));
console.log(
  `\ncolumns: [point sample · area (ours) · box truth 24×24].  ours should match the truth; point should not.` +
    `\nwrote ${Deno.realPathSync(outPath)} (${canvasW}×${canvasH})`,
);
