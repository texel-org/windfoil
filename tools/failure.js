// failure.js — construct the winding-fold failure cases and measure how bad they are.  (deno task failure)
//
// The algorithm (docs/ALGORITHM.md §4, §8) computes the pixel-AVERAGED signed winding number F exactly, then
// folds it to coverage (nonzero → min(|F|,1); even-odd → triangle wave). The fold is exact only where a pixel's
// winding is effectively {0,k}. Everything else loses information. This reproduces the distinct mechanisms on
// the real shader — the many "failure cases" collapse to these:
//
//   A · opposite-sign cancellation — +1 abuts −1; both filled (true 1) but F=0 → fold 0. A black hairline.
//   B · winding multiplicity >1    — a +N region's edge; F ramps 0→N so min(|F|,1) saturates → AA lost, ½px fat.
//   C · three levels {0,1,2}       — overlapping subpaths; a corner pixel over-counts (skia stays exact here).
//   D · even-odd of a doubled path — interior correctly empties, but a false 1px halo appears (tri_wave peak).
//   E · same F, different coverage — two shapes the shader renders identically but whose true coverage differs.
//   + a {5/2}-star companion (both single-sample renderers deviate) and size/multiplicity/minification sweeps.
//
// Each case emits SEPARATE, suffixed files (no combined montage) into output/failure/: an .svg of the shape,
// three whole-shape PNGs (so the geometry is legible), and four magnified PNGs of the failing pixels — one per
// coverage source, so it's unambiguous which is which:
//   • ours — the windfoil WGSL shader (the real code path, via renderToRGBA)
//   • box  — the mathematical box filter, point-sampled 24×24 from the raw curves (the ground truth)
//   • skia — @napi-rs/canvas, an independent production rasterizer (does it make the same trade? §4 claims yes)
//   • diff — |ours − box|, the error, as a heat map
// It also prints exact coverage at the worst pixel + how the error degrades with size and winding multiplicity.

import { renderToRGBA } from '../src/gpu.js';
import { pushMonotonePieces } from '../src/geometry.js';
import { bandPieces } from '../src/bands.js';
import { encodePNG } from '../src/png.js';
import { createCanvas } from '@napi-rs/canvas';

const S = 128; // cell size in px (whole-shape render)
const F = 24; // point-sample grid per pixel for the box-filter ground truth
const EXACT_GRID_NOTE = "the shader's 8×8 sub-sample quantisation, → 0 as EXACT_GRID rises"; // see windfoil.wgsl

// ── geometry helpers (flat quads [x0,y0,cx,cy,x1,y1,...], a line = a midpoint quad; same convention as validate) ─
function line(x0, y0, x1, y1) {
  return [x0, y0, (x0 + x1) / 2, (y0 + y1) / 2, x1, y1];
}
function polygon(pts) {
  const out = [];
  for (let i = 0; i < pts.length; i++) out.push(...line(...pts[i], ...pts[(i + 1) % pts.length]));
  return out;
}
// Axis-aligned rectangle as a closed contour. dir = +1 or −1 flips the traversal, flipping its winding sign.
function rect(x0, y0, x1, y1, dir = 1) {
  const cs = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]];
  return polygon(dir >= 0 ? cs : cs.slice().reverse());
}

// ── our shader coverage (identical scene build to validate.js) ─────────────────────────────────────────
function buildScene(quads, evenodd) {
  const pieces = [];
  for (let i = 0; i < quads.length; i += 6) pushMonotonePieces(quads.slice(i, i + 6), pieces);
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (let i = 0; i < pieces.length; i += 2) {
    x0 = Math.min(x0, pieces[i]);
    x1 = Math.max(x1, pieces[i]);
    y0 = Math.min(y0, pieces[i + 1]);
    y1 = Math.max(y1, pieces[i + 1]);
  }
  const curveOut = [], rowOut = [];
  const { rowBase, bandCount, y0: by0, invH } = bandPieces(pieces, y0, y1, curveOut, rowOut);
  const rule = evenodd ? 1 : 0;
  const instances = new Float32Array([
    0,
    0,
    1,
    rule,
    x0,
    y0,
    x1,
    y1,
    1,
    1,
    1,
    1,
    rowBase,
    bandCount,
    by0,
    invH,
  ]);
  return { curves: new Float32Array(curveOut), rows: new Uint32Array(rowOut), instances };
}
async function ourCoverage(quads, evenodd = false, size = S, exact = false) {
  const { curves, rows, instances } = buildScene(quads, evenodd);
  const rgba = await renderToRGBA({
    width: size,
    height: size,
    background: [0, 0, 0, 1],
    curves,
    rows,
    instances,
    instanceCount: 1,
    exact,
  });
  const out = new Float64Array(size * size);
  for (let i = 0; i < out.length; i++) out[i] = rgba[i * 4] / 255;
  return out;
}

// ── ground-truth box filter: winding by ray-casting the raw curves, averaged over an F×F sub-sample grid ─
function windingAt(px, py, quads) {
  let W = 0, K = 0;
  for (let i = 0; i < quads.length; i += 6) {
    const x0 = quads[i],
      y0 = quads[i + 1],
      cx = quads[i + 2],
      cy = quads[i + 3],
      x1 = quads[i + 4],
      y1 = quads[i + 5];
    if ((y0 < py && cy < py && y1 < py) || (y0 > py && cy > py && y1 > py)) continue;
    const a = y0 - 2 * cy + y1, b = 2 * (cy - y0), c = y0 - py;
    let t0 = -1, t1 = -1;
    if (Math.abs(a) < 1e-9) {
      if (Math.abs(b) > 1e-12) t0 = -c / b;
    } else {
      const disc = b * b - 4 * a * c;
      if (disc >= 0) {
        const sq = Math.sqrt(disc);
        t0 = (-b + sq) / (2 * a);
        t1 = (-b - sq) / (2 * a);
      }
    }
    for (const t of [t0, t1]) {
      if (t < 0 || t > 1) continue;
      const xt = (1 - t) * (1 - t) * x0 + 2 * (1 - t) * t * cx + t * t * x1;
      if (xt > px) {
        K++;
        const dy = 2 * a * t + b;
        W += dy >= 0 ? 1 : -1;
      }
    }
  }
  return { W, K };
}
function boxCoverage(quads, evenodd = false, size = S) {
  const out = new Float64Array(size * size);
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      let inside = 0;
      for (let j = 0; j < F; j++) {
        for (let i = 0; i < F; i++) {
          const { W, K } = windingAt(x + (i + 0.5) / F, y + (j + 0.5) / F, quads);
          if (evenodd ? (K & 1) === 1 : W !== 0) inside++;
        }
      }
      out[y * size + x] = inside / (F * F);
    }
  }
  return out;
}

// ── skia (@napi-rs/canvas) coverage ────────────────────────────────────────────────────────────────────
function skiaCoverage(quads, evenodd = false, size = S) {
  const ctx = createCanvas(size, size).getContext('2d');
  ctx.fillStyle = '#000';
  ctx.fillRect(0, 0, size, size);
  ctx.fillStyle = '#fff';
  ctx.beginPath();
  let px = null, py = null;
  for (let i = 0; i < quads.length; i += 6) {
    const [x0, y0, cx, cy, x1, y1] = quads.slice(i, i + 6);
    if (px === null || Math.abs(x0 - px) > 1e-4 || Math.abs(y0 - py) > 1e-4) {
      if (px !== null) ctx.closePath();
      ctx.moveTo(x0, y0);
    }
    ctx.quadraticCurveTo(cx, cy, x1, y1);
    px = x1;
    py = y1;
  }
  ctx.closePath();
  ctx.fill(evenodd ? 'evenodd' : 'nonzero');
  const d = ctx.getImageData(0, 0, size, size).data;
  const out = new Float64Array(size * size);
  for (let i = 0; i < out.length; i++) out[i] = d[i * 4] / 255;
  return out;
}

// ── image compositing: coverage buffers → labelled, magnified PNG panels ────────────────────────────────
const GRID = [70, 74, 96], MARK = [232, 120, 40]; // pixel-grid line, failing-pixel outline
const gray = (v) => {
  const c = Math.round(Math.max(0, Math.min(1, v)) * 255);
  return [c, c, c];
};
const heat = (d) => {
  const v = Math.min(1, d);
  return [Math.round(v * 255), Math.round(v * 90), Math.round(v * 32)];
};

// Magnify a rectangular region of a coverage buffer, one source pixel → mag×mag block. `mask` (optional, same
// length as the buffer) outlines a block in MARK; a GRID separator is drawn between blocks when mag is large.
function magnify(cov, w, region, mag, color, mask) {
  const { x: rx, y: ry, w: rw, h: rh } = region;
  const grid = mag >= 6;
  const W = rw * mag, H = rh * mag;
  const out = new Uint8Array(W * H * 4);
  for (let j = 0; j < rh; j++) {
    for (let i = 0; i < rw; i++) {
      const sx = rx + i, sy = ry + j, idx = sy * w + sx;
      const [r, g, b] = color(cov[idx]);
      const flagged = mask && mask[idx];
      for (let dy = 0; dy < mag; dy++) {
        for (let dx = 0; dx < mag; dx++) {
          let R = r, G = g, B = b;
          if (flagged && (dx < 2 || dy < 2 || dx >= mag - 2 || dy >= mag - 2)) [R, G, B] = MARK;
          else if (grid && (dx === 0 || dy === 0)) [R, G, B] = GRID;
          const o = ((j * mag + dy) * W + (i * mag + dx)) * 4;
          out[o] = R;
          out[o + 1] = G;
          out[o + 2] = B;
          out[o + 3] = 255;
        }
      }
    }
  }
  return { rgba: out, w: W, h: H };
}

// The quad path as an SVG `d` string: each contour opens with M at its start, then one Q per edge (the
// straight edges are exact degenerate quadratics — the same representation the shader consumes). A new
// contour starts wherever an edge's start point doesn't continue the previous edge's end.
const num = (n) => (Number.isInteger(n) ? `${n}` : `${+n.toFixed(3)}`);
function svgPath(quads) {
  let d = '', px = null, py = null;
  for (let i = 0; i < quads.length; i += 6) {
    const [x0, y0, cx, cy, x1, y1] = quads.slice(i, i + 6);
    if (px === null || Math.abs(x0 - px) > 1e-4 || Math.abs(y0 - py) > 1e-4) {
      d += `${d ? 'Z ' : ''}M ${num(x0)} ${num(y0)} `;
    }
    d += `Q ${num(cx)} ${num(cy)} ${num(x1)} ${num(y1)} `;
    px = x1;
    py = y1;
  }
  return `${d}Z`;
}

const outDir = new URL('../output/failure/', import.meta.url);
Deno.mkdirSync(outDir, { recursive: true });
const writePNG = (name, panel) => {
  Deno.writeFileSync(new URL(`${name}.png`, outDir), encodePNG(panel.rgba, panel.w, panel.h));
  return `${name}.png (${panel.w}×${panel.h})`;
};
// A minimal standalone SVG of the failure shape: one <path d> under the same fill rule the case uses. Rendered
// by any conformant SVG engine this shows the CORRECT fill (a solid square for A, a left rect for B) — i.e. what
// the winding fold gets wrong. Checkerboard backdrop so anti-aliased edges read on both light and dark.
const writeSVG = (name, quads, evenodd, note) => {
  const rule = evenodd ? 'evenodd' : 'nonzero';
  const svg =
    `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${S} ${S}" width="${S}" height="${S}">\n` +
    `  <!-- ${note} · fill-rule: ${rule} -->\n` +
    `  <path d="${svgPath(quads)}" fill="#111" fill-rule="${rule}"/>\n` +
    `</svg>\n`;
  Deno.writeFileSync(new URL(`${name}.svg`, outDir), new TextEncoder().encode(svg));
  return `${name}.svg`;
};

function absDiff(a, b) {
  const d = new Float64Array(a.length);
  for (let i = 0; i < a.length; i++) d[i] = Math.abs(a[i] - b[i]);
  return d;
}
function worstPixel(diff, w, h) {
  let m = -1, mi = 0;
  for (let i = 0; i < diff.length; i++) {
    if (diff[i] > m) {
      m = diff[i];
      mi = i;
    }
  }
  return { x: mi % w, y: (mi / w) | 0, d: m };
}
function count(diff, thr) {
  let n = 0;
  for (const d of diff) if (d > thr) n++;
  return n;
}
function mean(diff) {
  let s = 0;
  for (const d of diff) s += d;
  return s / diff.length;
}
const fx = (v, n = 3) => v.toFixed(n);
const pct = (v) => `${(v * 100).toFixed(1)}%`;

// Render one case as a set of SEPARATE, clearly-suffixed files (no combined montage):
//   {name}.svg                          — the shape itself, one <path d>
//   {name}_overview_{ours,box,skia}.png — the whole shape at 3× (real pixels, no marker)
//   {name}_zoom_{ours,box,skia,diff}.png — the failing pixels magnified; the pixels ours gets wrong are
//                                          outlined in orange (same crop in every file, so they line up)
// `zoom` = {cx, cy, w, h, mag} for the crop; cx/cy default to the worst |ours−box| pixel.
async function renderCase(name, title, quads, zoom, evenodd = false) {
  const ours = await ourCoverage(quads, evenodd);
  const oursExact = await ourCoverage(quads, evenodd, S, true); // shader {exact:true} — supersampled fill
  const box = boxCoverage(quads, evenodd);
  const skia = skiaCoverage(quads, evenodd);
  const dOB = absDiff(ours, box), dSB = absDiff(skia, box), dEB = absDiff(oursExact, box);
  const worst = worstPixel(dOB, S, S);
  const mask = dOB.map((d) => (d > 0.25 ? 1 : 0));
  const wrote = [];

  wrote.push(writeSVG(name, quads, evenodd, title));

  // whole-shape views at 3× — NO marker, so each file shows its real pixels (ours grows the artifact;
  // box/skia stay clean).
  const OV = 3, full = { x: 0, y: 0, w: S, h: S };
  wrote.push(writePNG(`${name}_overview_ours`, magnify(ours, S, full, OV, gray)));
  wrote.push(writePNG(`${name}_overview_box`, magnify(box, S, full, OV, gray)));
  wrote.push(writePNG(`${name}_overview_skia`, magnify(skia, S, full, OV, gray)));

  // magnified crop around the failing pixels (floor the origin: the crop centre can be a half-pixel like MID)
  const cx = Math.floor(zoom.cx ?? worst.x), cy = Math.floor(zoom.cy ?? worst.y);
  const rx = Math.max(0, Math.min(S - zoom.w, cx - (zoom.w >> 1)));
  const ry = Math.max(0, Math.min(S - zoom.h, cy - (zoom.h >> 1)));
  const crop = { x: rx, y: ry, w: zoom.w, h: zoom.h }, mag = zoom.mag;
  wrote.push(writePNG(`${name}_zoom_ours`, magnify(ours, S, crop, mag, gray, mask)));
  wrote.push(writePNG(`${name}_zoom_box`, magnify(box, S, crop, mag, gray, mask)));
  wrote.push(writePNG(`${name}_zoom_skia`, magnify(skia, S, crop, mag, gray, mask)));
  wrote.push(writePNG(`${name}_zoom_diff`, magnify(dOB, S, crop, mag, heat, mask)));
  wrote.push(writePNG(`${name}_zoom_exact`, magnify(oursExact, S, crop, mag, gray, mask))); // {exact:true} fix

  // report
  const wi = worst.y * S + worst.x, dsb = worstPixel(dSB, S, S).d, we = worstPixel(dEB, S, S);
  console.log(`\n▐ ${title}`);
  console.log(
    `  worst pixel (${worst.x},${worst.y}):  ours ${fx(ours[wi])}   box/truth ${fx(box[wi])}   skia ${
      fx(skia[wi])
    }   exact ${fx(oursExact[wi])}   →  fast off by ${fx(worst.d)}`,
  );
  console.log(
    `  |ours−box|:  mean ${fx(mean(dOB), 5)}   max ${fx(worst.d)}   pixels off >0.5: ${
      count(dOB, 0.5)
    }   >0.1: ${count(dOB, 0.1)}`,
  );
  console.log(
    `  |exact−box|: mean ${fx(mean(dEB), 5)}   max ${fx(we.d)}   ← {exact:true} supersample fixes it ` +
      `(residual is ${EXACT_GRID_NOTE})`,
  );
  console.log(
    `  |skia−box|:  mean ${fx(mean(dSB), 5)}   max ${fx(dsb)}   ` +
      `(does skia share the failure? ${dsb > 0.3 ? 'YES' : 'no — skia matches the box filter here'})`,
  );
  console.log(`  wrote: ${wrote.join(', ')}`);
  return { ours, box, skia, dOB, worst };
}

// ── the four fractional-pixel offsets that put a seam / edge through a pixel CENTRE (so it can't hide on a
//    pixel boundary): S is even, so S/2 is a boundary; S/2+0.5 is the centre of column S/2. ────────────────
const MID = S / 2 + 0.5;

console.log(`failure · ${S}px cell · box-filter truth = ${F}×${F} point-sample · skia = @napi-rs/canvas`);
console.log(
  `the shader computes F = avg signed winding EXACTLY, then folds it; cases A–E are where that fold ≠ true`,
);
console.log(`coverage, one per distinct mechanism (docs/ALGORITHM.md §4, §8).`);

// ══ CASE A — opposite-sign cancellation ════════════════════════════════════════════════════════════════
// A solid square built as a +1 left half abutting a −1 right half. Nonzero fills the whole square; the seam
// column (footprint spans both signs) averages to F = 0 → a black hairline where the two windings meet.
const M = 16, N = S - 16;
const cancelSquare = [
  ...rect(M, M, MID, N, +1), // left half, winding +1
  ...rect(MID, M, N, N, -1), // right half, winding −1
];
{
  const inL = windingAt((M + MID) / 2, S / 2, cancelSquare).W;
  const inR = windingAt((MID + N) / 2, S / 2, cancelSquare).W;
  console.log(
    `\n[measured winding] case A left half = ${inL >= 0 ? '+' : ''}${inL}, right half = ${
      inR >= 0 ? '+' : ''
    }${inR} ` +
      `(opposite signs ⇒ they cancel on the seam)`,
  );
}
const A = await renderCase(
  'A_cancellation',
  'CASE A · opposite-sign winding cancellation (true coverage 1, fold gives 0)',
  cancelSquare,
  { cx: S / 2, cy: S / 2, w: 9, h: 13, mag: 26 },
);

// ══ CASE B — winding multiplicity > 1 ══════════════════════════════════════════════════════════════════
// The left region is wound TWICE (two identical +1 rects → +2); the right side is empty. The +2→0 edge should
// ramp 0→1 like any edge, but F ramps 0→2 so min(|F|,1) saturates after half the edge: AA lost, edge fattened.
const doubleRegion = [
  ...rect(M, M, MID, N, +1),
  ...rect(M, M, MID, N, +1), // duplicate, same direction → winding +2 on the left region
];
{
  const inL = windingAt((M + MID) / 2, S / 2, doubleRegion).W;
  console.log(`\n[measured winding] case B left region = +${inL} (duplicated same-direction contour)`);
}
const B = await renderCase(
  'B_double_winding',
  'CASE B · winding multiplicity 2 (edge AA saturates, edge displaced ~½px)',
  doubleRegion,
  { cx: MID, cy: S / 2, w: 11, h: 9, mag: 26 },
);

// ══ CASE C — three winding levels {0,1,2} in one footprint (overlapping subpaths / self-intersection) ═════
// Two overlapping same-direction rectangles: arms are w=1, the overlap is w=2, outside is w=0. Under nonzero
// the union is filled once (C=1 inside), so interiors are fine — but a pixel at the overlap CORNER sees all of
// {0,1,2} at once and the average over-counts. This is the practical "layered artwork" case. NOTE the measured
// result below: on this CLEAN axis-aligned overlap skia stays correct — it's windfoil's fold that over-counts.
const rectA = rect(M, M, 80.5, N, +1); // right edge at a pixel centre
const rectB = rect(48, 40.5, N + 16, 88.5, +1); // bottom edge at a pixel centre; overlaps rectA's lower-right
const overlap = [...rectA, ...rectB];
{
  const w2 = windingAt(64, 64, overlap).W; // inside both
  console.log(
    `\n[measured winding] case C overlap region = +${w2} (two stacked +1 subpaths); corner sees {0,1,2}`,
  );
}
const C = await renderCase(
  'C_overlap_multilevel',
  'CASE C · three winding levels {0,1,2} at an overlap corner (fold over-counts by ~0.25)',
  overlap,
  { w: 13, h: 13, mag: 22 }, // auto-centre on the worst (triple-point) pixel
);

// Where DOES windfoil "tie" skia (§4/§5)? Only sub-pixel slivers at a SHARP self-intersection. Measure the
// {5/2} star (nonzero) — a single self-intersecting contour with a w=2 core and thin points — with no images,
// just the worst-pixel deltas, to show both renderers deviate there (unlike the clean overlap above).
function starPts(cx, cy, r, points, step) {
  const p = [];
  for (let k = 0; k < points; k++) {
    const a = -Math.PI / 2 + ((k * step) % points) * (2 * Math.PI / points);
    p.push([cx + r * Math.cos(a), cy + r * Math.sin(a)]);
  }
  return p;
}
{
  const star = polygon(starPts(64, 64, 52, 5, 2));
  const o = await ourCoverage(star), b = boxCoverage(star), sk = skiaCoverage(star);
  const wo = worstPixel(absDiff(o, b), S, S), ws = worstPixel(absDiff(sk, b), S, S);
  console.log(
    `\n▐ COMPANION · {5/2} star (nonzero, sharp self-intersection) — where BOTH single-sample renderers deviate`,
  );
  console.log(
    `  worst |ours−box| ${fx(wo.d)} at (${wo.x},${wo.y})   worst |skia−box| ${fx(ws.d)} at (${ws.x},${ws.y})`,
  );
  console.log(
    `  ⇒ both deviate at the star points (sub-pixel slivers no single sample resolves; here ours is even closer`,
  );
  console.log(
    `    than skia). Contrast Case C: the overlap is clean, so skia is exact and only ours is off — §4's "same`,
  );
  console.log(
    `    as skia/vello" is narrow (the sharp-sliver regime), not overlaps/cancellation/multiplicity in general.`,
  );
}

// ══ CASE D — even-odd parity is not recoverable from the average ═════════════════════════════════════════
// EVEN-ODD fill folds with a triangle wave, not min(|F|,1) — so ChatGPT's "100%·w=2 ⇒ clamp gives 1" is NOT
// what windfoil does (tri_wave(2)=0, correct; a uniform w=2 region correctly renders EMPTY). The real even-odd
// failure is a NON-ADJACENT parity jump: a doubled contour steps winding 0→2 with no w=1 in between, so the
// interior is correctly empty but the boundary sweeps F through 1 (tri_wave's peak) → a false 1px HALO. A
// correct even-odd renderer (box, skia) draws nothing at all. Put all four edges on pixel CENTRES (half-ints)
// so every edge straddles a pixel and the halo forms a complete outline (an edge on a pixel boundary is sharp,
// F jumps 0→2 with no pixel landing on the F=1 peak, so it wouldn't halo).
const eoSquare = [...rect(24.5, 24.5, 96.5, 96.5, +1), ...rect(24.5, 24.5, 96.5, 96.5, +1)]; // doubled → w=2
const D = await renderCase(
  'D_evenodd_halo',
  'CASE D · even-odd of a doubled contour: interior empties (correct) but a false halo outline appears',
  eoSquare,
  { cx: 24, cy: 24, w: 11, h: 11, mag: 24 }, // zoom a corner: two halo edges meet, interior + exterior both empty
  true,
);

// ══ CASE E — the fundamental ambiguity: one scalar F cannot encode coverage ══════════════════════════════
// Two DIFFERENT shapes engineered to give the SAME averaged winding F=0.5 in the target column, but different
// true coverage. No scalar fold g(F) can be right for both — the shader renders them identically.
//   E1: a single edge covering 50% of the column  → w=1 over 50% → F=0.5,  true C=0.5  (fold is exact)
//   E2: a doubled edge covering 25% of the column → w=2 over 25% → F=0.5,  true C=0.25 (fold is 2× too high)
async function renderAmbiguity() {
  const col = 64, row = S / 2;
  const e1 = rect(M, M, col + 0.5, N, +1); // right edge mid-column → 50% of column, w=1
  const e2 = [...rect(M, M, col + 0.25, N, +1), ...rect(M, M, col + 0.25, N, +1)]; // 25% of column, w=2
  const o1 = await ourCoverage(e1), b1 = boxCoverage(e1);
  const o2 = await ourCoverage(e2), b2 = boxCoverage(e2);
  const x1 = await ourCoverage(e1, false, S, true), x2 = await ourCoverage(e2, false, S, true); // {exact:true}
  const i = row * S + col;
  const crop = { x: col - 4, y: 48, w: 9, h: 24 }, mag = 18;
  const wrote = [
    writeSVG('E_ambiguity_shapeA', e1, false, 'E1 · single edge, 50% of column, w=1 (true coverage 0.5)'),
    writeSVG('E_ambiguity_shapeB', e2, false, 'E2 · doubled edge, 25% of column, w=2 (true coverage 0.25)'),
    writePNG('E_ambiguity_shapeA_ours', magnify(o1, S, crop, mag, gray)),
    writePNG('E_ambiguity_shapeA_box', magnify(b1, S, crop, mag, gray)),
    writePNG('E_ambiguity_shapeB_ours', magnify(o2, S, crop, mag, gray)),
    writePNG('E_ambiguity_shapeB_box', magnify(b2, S, crop, mag, gray)),
    writePNG('E_ambiguity_shapeB_exact', magnify(x2, S, crop, mag, gray)),
  ];
  console.log(`\n▐ CASE E · same averaged winding F, different true coverage (no scalar fold can fix this)`);
  console.log(`  target column ${col}, F=0.5 for BOTH shapes:`);
  console.log(`    E1 (edge, w=1, 50%):   fast ${fx(o1[i])}   exact ${fx(x1[i])}   box/truth ${fx(b1[i])}`);
  console.log(`    E2 (doubled, w=2, 25%): fast ${fx(o2[i])}   exact ${fx(x2[i])}   box/truth ${fx(b2[i])}`);
  console.log(
    `  ⇒ the FAST fold renders E1 and E2 IDENTICALLY (${fx(o1[i])} vs ${
      fx(o2[i])
    }) though the truths differ ` +
      `(${fx(b1[i])} vs ${fx(b2[i])}) —`,
  );
  console.log(
    `    the info is gone before a scalar fold. {exact:true} does NOT fold, so it tells them apart: ` +
      `${fx(x1[i])} vs ${fx(x2[i])} (matches truth).`,
  );
  console.log(`  wrote: ${wrote.join(', ')}`);
}
await renderAmbiguity();

// ══ DEGRADATION 1 — how CASE A scales: the hairline stays full-contrast but thin (error ∝ perimeter, not area) ══
console.log(`\n▐ DEGRADATION · case-A cancellation vs shape size (the crack is 1px wide at every size)`);
console.log(
  `  ${'cell'.padStart(6)}  ${'max Δ'.padStart(7)}  ${'mean Δ'.padStart(8)}  ${
    'px off >0.5'.padStart(11)
  }   how it reads`,
);
for (const sz of [32, 64, 128, 256]) {
  const mid = sz / 2 + 0.5, m = Math.round(sz * 0.12), n = sz - Math.round(sz * 0.12);
  const sq = [...rect(m, m, mid, n, +1), ...rect(mid, m, n, n, -1)];
  const o = await ourCoverage(sq, false, sz), b = boxCoverage(sq, false, sz), d = absDiff(o, b);
  const w = worstPixel(d, sz, sz);
  console.log(
    `  ${(sz + 'px').padStart(6)}  ${fx(w.d).padStart(7)}  ${fx(mean(d), 5).padStart(8)}  ${
      String(count(d, 0.5)).padStart(11)
    }   ` +
      `${w.d > 0.9 ? 'solid black hairline' : 'faint'}, ${pct(count(d, 0.5) / (sz * sz))} of pixels`,
  );
}

// ══ DEGRADATION 2 — how CASE B worsens with winding multiplicity N ═════════════════════════════════════
// Sample the coverage ramp across a +N region's right edge and compare to the box filter's linear ramp.
console.log(`\n▐ DEGRADATION · edge AA vs winding multiplicity N (N=1 is exact; error grows with N)`);
console.log(
  `  ${'N'.padStart(3)}  ${'max edge Δ'.padStart(10)}  ${'edge shift'.padStart(10)}  ${
    'ramp width'.padStart(10)
  }   verdict`,
);
const ramps = [];
for (const Nw of [1, 2, 3, 4, 6]) {
  const quads = [];
  for (let k = 0; k < Nw; k++) quads.push(...rect(M, M, MID, N, +1));
  const o = await ourCoverage(quads, false), b = boxCoverage(quads, false);
  // sample the row through the middle, across the right edge at x≈MID
  const row = Math.round(S / 2);
  let maxD = 0, halfOurs = null, halfBox = null;
  for (let x = M; x < N; x++) {
    const i = row * S + x;
    maxD = Math.max(maxD, Math.abs(o[i] - b[i]));
    if (halfOurs === null && o[i] <= 0.5) {
      halfOurs = x + (o[row * S + x - 1] - 0.5) / Math.max(1e-6, o[row * S + x - 1] - o[i]);
    }
    if (halfBox === null && b[i] <= 0.5) {
      halfBox = x + (b[row * S + x - 1] - 0.5) / Math.max(1e-6, b[row * S + x - 1] - b[i]);
    }
  }
  // ramp width = pixels where 0.02 < ours < 0.98 along the edge row
  let rampW = 0;
  for (let x = M; x < N; x++) {
    const v = o[row * S + x];
    if (v > 0.02 && v < 0.98) rampW++;
  }
  const shift = halfOurs !== null && halfBox !== null ? halfOurs - halfBox : NaN;
  ramps.push({ N: Nw, o, b, maxD });
  console.log(
    `  ${String(Nw).padStart(3)}  ${fx(maxD).padStart(10)}  ${(fx(shift, 2) + 'px').padStart(10)}  ${
      (rampW + 'px').padStart(10)
    }   ` +
      `${Nw === 1 ? 'exact' : maxD > 0.4 ? 'hard/aliased edge' : 'degraded'}`,
  );
}
// one magnified edge crop per N (same crop each time, so flipping between them shows the ramp collapsing to a
// hard edge), plus the box-filter's correct soft edge as the reference (identical for every N).
{
  const edge = { x: Math.floor(MID) - 6, y: Math.round(S / 2) - 3, w: 12, h: 6 }, mag = 20;
  const wrote = [writePNG('B_multiplicity_box', magnify(ramps[0].b, S, edge, mag, gray))];
  for (const r of ramps) {
    wrote.push(writePNG(`B_multiplicity_ours_N${r.N}`, magnify(r.o, S, edge, mag, gray)));
  }
  console.log(`  wrote: ${wrote.join(', ')}`);
  console.log(
    `  (B_multiplicity_box = the correct soft edge; ours_N1 matches it, ours_N≥2 collapses to a hard edge)`,
  );
}

// ══ DEGRADATION 3 — minification: a REGIME that turns Case A on for ordinary shapes ══════════════════════
// A "picket fence" of alternating +1/−1 filled bars: solid at full size (each bar filled; thin black cracks on
// the seams), but as it shrinks, one footprint spans many bars so the SIGNED average → 0 while true coverage
// stays ~1 → the whole shape fades to black. Alignment-robust (many seams), unlike a single-seam square. This
// is the practical "zoomed-out vector art" failure; below GUARD_PX (~3.7) the ink-profile guard, which also
// averages signed winding per band, inherits and amplifies it.
function stripes(sz, k) {
  const x0 = sz * 0.15, y0 = sz * 0.15, y1 = sz * 0.85, bw = (sz * 0.7) / k;
  const out = [];
  for (let i = 0; i < k; i++) out.push(...rect(x0 + i * bw, y0, x0 + (i + 1) * bw, y1, i % 2 ? -1 : 1));
  return { quads: out, bbox: [x0, y0, x0 + k * bw, y1] };
}
console.log(
  `\n▐ DEGRADATION · minification of a +1/−1 picket fence (8 bars; true coverage ≈ 1.0 at every size)`,
);
console.log(
  `  ${'size'.padStart(6)}  ${'bar w'.padStart(6)}  ${'ours mean'.padStart(9)}  ${'reads as'.padStart(14)}`,
);
const minFrames = [];
for (const sz of [96, 64, 32, 16, 10, 6]) {
  const { quads, bbox } = stripes(sz, 8);
  const o = await ourCoverage(quads, false, sz), b = boxCoverage(quads, false, sz);
  // mean coverage over the ink bbox (truth ≈ 1); ours collapses as the footprint eats whole bars
  let so = 0, sb = 0, nn = 0;
  for (let y = Math.ceil(bbox[1]); y < bbox[3]; y++) {
    for (let x = Math.ceil(bbox[0]); x < bbox[2]; x++) {
      so += o[y * sz + x];
      sb += b[y * sz + x];
      nn++;
    }
  }
  const om = so / nn, bm = sb / nn;
  minFrames.push({ sz, o });
  console.log(
    `  ${(sz + 'px').padStart(6)}  ${(fx((sz * 0.7) / 8, 1) + 'px').padStart(6)}  ${fx(om).padStart(9)}  ` +
      `${(om > 0.9 ? 'solid (~ok)' : om > 0.5 ? 'dim/patchy' : 'fades to BLACK').padStart(14)}  (truth ${
        fx(bm)
      })`,
  );
}
{
  // filmstrip: the fence at three sizes, each upscaled to a common width so the fade-out is visible
  const wrote = [];
  for (const { sz, o } of minFrames.filter((f) => [96, 32, 10].includes(f.sz))) {
    wrote.push(
      writePNG(
        `F_minify_fence_${sz}px_ours`,
        magnify(o, sz, { x: 0, y: 0, w: sz, h: sz }, Math.round(288 / sz), gray),
      ),
    );
  }
  console.log(`  wrote: ${wrote.join(', ')}  (96px = mostly solid; 10px = collapsed to near-black)`);
}
console.log(
  `  ⇒ signed-winding averaging can't survive minification of opposite-oriented detail; the exact fill`,
);
console.log(
  `    rule needs the winding DISTRIBUTION, not its mean. (Ordinary same-orientation art minifies fine.)`,
);

// ── verdict ─────────────────────────────────────────────────────────────────────────────────────────────
console.log(`\n▐ VERDICT`);
console.log(
  `  The integral is exact for F (the averaged winding); every failure is in the scalar fold F→cover,`,
);
console.log(
  `  which cannot recover coverage once a pixel's winding field is more than {0,k}. ChatGPT's 11 points`,
);
console.log(`  are all instances of just FOUR mechanisms (+ one regime that makes them likely):`);
console.log(
  `   1. SIGN CANCELLATION  (Case A)  — ± windings average below either; #1. Fully wrong (Δ up to 1).`,
);
console.log(
  `   2. MAGNITUDE >1       (Case B)  — |w|>1 over-counts; #2, #7 bad-hole, #8 coincident, #9 overlap.`,
);
console.log(
  `   3. MULTI-LEVEL {0,1,2}(Case C)  — 3+ levels in one footprint; #3, #6 self-intersect. Over-counts ~0.25.`,
);
console.log(
  `   4. FILL-RULE / EVEN-ODD(Case D) — parity isn't in the average; #5, #11. tri_wave halos on 0→2 jumps.`,
);
console.log(
  `   ★ ROOT: same F ⇏ same coverage (Case E; #4) — a scalar fold is lossy in principle, not just in code.`,
);
console.log(
  `   ◦ REGIME: footprint size / minification (#10) multiplies 1–4 — small/zoomed-out geometry fails more.`,
);
console.log(
  `  Corrections to the notes: windfoil folds even-odd with a TRIANGLE WAVE, not clamp — so "100%·w=2 ⇒`,
);
console.log(
  `  clamp=1" and "50/50 w=1,w=2 ⇒ fails" are WRONG here (tri_wave gives 0 and 0.5, both correct). The`,
);
console.log(
  `  real even-odd failure is the non-adjacent 0→2 halo (Case D). And on the SKIA claim: measured here, skia`,
);
console.log(
  `  matches the box filter on A, B AND the clean overlap C (it does NOT share those failures). Both only`,
);
console.log(
  `  deviate together at sharp self-intersections (the star companion) — so §4's "same as skia/vello" is narrow.`,
);
console.log(
  `  Fast fold is fine for: simple contours, holes with consistent opposite orientation, no pathological`,
);
console.log(
  `  overlap, no duplicated same-direction contours, nonzero fill on normalized paths (most fonts/icons/UI).`,
);
console.log(
  `  Exact fill matters for: arbitrary SVG, self-intersection, overlapping subpaths, even-odd, nested`,
);
console.log(
  `  contours of inconsistent direction, duplicated/coincident contours, uncontrolled CAD/map/vector data.`,
);
console.log(
  `  THE FIX (opt-in): the shader's {exact:true} path supersamples the true fill instead of folding a scalar,`,
);
console.log(
  `  so A–E collapse to ~0 error above (residual = the 8×8 sub-sample grid). It is offline-only — much slower`,
);
console.log(
  `  per pixel — and the fast fold is untouched when off (a uniform branch). See _zoom_exact.png per case.`,
);
console.log(
  `\n  wrote ${
    Deno.realPathSync(outDir)
  }/  ·  open the PNGs to see each failure whole, magnified, and exact-fixed.`,
);
