// tools/v3stats.js — CPU replication of windfoil3.wgsl's integrate_face / integrate_band traversal over the
// bench scenes, counting (not rendering) exactly what each real fragment does against the NEW bandPieces
// atlas (src/bands.js): per band TWO vec4<u32> headers, and in `curves` (vec2 units)
//   metaF[fCount+1] · metaE[eCount] · prefixE pairs · piece data (3 vec2 per piece, F then E).
//
// Per exact (non-guard) fragment it counts: bands visited, covered bands, F meta words walked in the prefix
// jump-find, F pieces scanned post-jump (split fully-left-cull / integrated / cut-by-wF-break), E meta words
// walked in the covered-only jump-find, E pieces scanned (split fully-left-cull / cheap-right / y-cull /
// integrated / cut-by-wE-break), and how many walked pieces an IDEAL exact break (perfect per-piece hull
// suffix knowledge) would have cut that the conservative wF/wE break failed to. Roll-ups: integrate_piece
// calls / 8-byte meta-word loads / 24-byte-equivalent piece-data touches per fragment.
//
//   deno run -A tools/v3stats.js
//
// Camera / culling / instance placement replicate bench/main.js bit-for-bit (same builders, same grid extents,
// EM_WORLD 100, 720×720, 0.625px skirt pad on the quad, fwidth = 1/(sc·camS) glyph units per device px).
// Cull decisions use the ACTUAL f16 meta words decoded from the built atlas, so classifications match the
// shader; coordinate math is f64 (boundary-insensitive). Above ~400k fragments a level uniformly samples
// instances (seeded shuffle) and reports means over the sample.

import { loadFont } from '../src/font.js';
import { buildGlyphAtlas } from '../src/bands.js';
import { buildSlugAtlas } from '../bench/slug.js';
import { buildScene, SCENE_TEXT, INK } from '../bench/scene.js';
import { buildShapeScene } from '../bench/shape.js';
import { buildTigerScene } from '../bench/tiger.js';

// ── bench-identical constants ─────────────────────────────────────────────────────────────────────────
const W = 720, H = 720;
const EM_WORLD = 100;
const SKIRT_PX = 0.625; // KERNEL_SKIRT_PX = 0.5 support + 0.125 slack
const GUARD_PX = 3.7;
const argNum = (name, fallback) => {
  const a = Deno.args.find((x) => x.startsWith(`--${name}=`));
  return a ? Number(a.split('=')[1]) : fallback;
};
const MAX_FRAGS = argNum('frags', 400_000);
const MIN_INST = 600; // ≥ 2× tiger's 304 distinct shapes, so heavy rare shapes can't skew a small sample
const SEED = argNum('seed', 0x9e3779b9) >>> 0;
// The bench sizes each grid for the FULL default ladder's smallest level (main.js):
const LADDER_MIN = { glyphs: 2, shape: 12, tiger: 64 };

// ── f16 half decoding (bit-exact twin of the shader's unpack2x16float) ────────────────────────────────
const f16buf = new Float16Array(1);
const f16u16 = new Uint16Array(f16buf.buffer);
function f16(bits) {
  f16u16[0] = bits;
  return f16buf[0];
}

// ── per-band decoded cache: f16 hulls as the shader sees them + exact suffix-max for the ideal break ──
function makeBandGetter(scene) {
  const curves = scene.wCurves;
  const bits = new Uint32Array(curves.buffer, curves.byteOffset, curves.length);
  const rows = scene.wRows;
  const cache = new Map();
  return function get(hdr) { // hdr = rowBase + 2*ri, vec4 index of the band's H0
    let B = cache.get(hdr);
    if (B) return B;
    const o = hdr * 4;
    const start = rows[o], fCount = rows[o + 1], eCount = rows[o + 2], wPacked = rows[o + 3];
    const wF = f16(wPacked & 0xffff), wE = f16(wPacked >>> 16);
    const pieceBase = start + fCount + 1 + eCount + ((eCount + 2) >> 1);
    const fXmin = new Float64Array(fCount), fXmax = new Float64Array(fCount);
    for (let i = 0; i < fCount; i++) {
      const w = bits[(start + i) * 2];
      fXmin[i] = f16(w & 0xffff);
      fXmax[i] = f16(w >>> 16);
    }
    const eBase = start + fCount + 1;
    const eXmin = new Float64Array(eCount), eXmax = new Float64Array(eCount);
    const eYlo = new Float64Array(eCount), eYhi = new Float64Array(eCount);
    for (let j = 0; j < eCount; j++) {
      const wx = bits[(eBase + j) * 2], wy = bits[(eBase + j) * 2 + 1];
      eXmin[j] = f16(wx & 0xffff);
      eXmax[j] = f16(wx >>> 16);
      eYlo[j] = f16(wy & 0xffff);
      eYhi[j] = f16(wy >>> 16);
    }
    // Ideal-break oracle: sufF[p] = exact max piece-hull x-max over positions p..fCount-1 (non-increasing).
    const sufF = new Float64Array(fCount + 1);
    sufF[fCount] = -Infinity;
    for (let i = fCount - 1; i >= 0; i--) {
      const b = (pieceBase + i * 3) * 2;
      sufF[i] = Math.max(sufF[i + 1], curves[b], curves[b + 4]);
    }
    const sufE = new Float64Array(eCount + 1);
    sufE[eCount] = -Infinity;
    for (let j = eCount - 1; j >= 0; j--) {
      const b = (pieceBase + (fCount + j) * 3) * 2;
      sufE[j] = Math.max(sufE[j + 1], curves[b], curves[b + 4]);
    }
    B = { fCount, eCount, wF, wE, fXmin, fXmax, eXmin, eXmax, eYlo, eYhi, sufF, sufE };
    cache.set(hdr, B);
    return B;
  };
}

// Smallest p with suf[p] <= T (suf is non-increasing; suf[last] = -Inf so p always exists).
function firstLE(suf, T) {
  let lo = 0, hi = suf.length - 1;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (suf[mid] <= T) hi = mid;
    else lo = mid + 1;
  }
  return lo;
}

function newC() {
  return {
    frags: 0, guardFrags: 0, exactFrags: 0,
    bands: 0, covered: 0, occF: 0, occE: 0,
    fJump: 0, fScan: 0, fCullLeft: 0, fInt: 0, fCut: 0, fFail: 0,
    eJump: 0, eScan: 0, eCullLeft: 0, eCheap: 0, eYCull: 0, eInt: 0, eCut: 0, eFail: 0,
    meta8: 0, invar: 0,
  };
}

// ── exact counting twin of integrate_band (windfoil3.wgsl) ────────────────────────────────────────────
function integrateBandCount(C, B, rcx, rcy, wlo, whi, sx, covered) {
  const hx = sx * 0.5;
  const fCount = B.fCount, eCount = B.eCount;
  C.occF += fCount;
  C.occE += eCount;

  // F jump-find: walk metaF while the (f16, rounded-down) hull x-min is certainly fully right.
  let i = 0;
  while (i < fCount) {
    C.fJump++;
    C.meta8++;
    if (B.fXmin[i] - rcx < hx) break; // first piece not certainly fully-right
    i++;
  }
  C.meta8++; // signed prefix-count read: curves[start + i].y (terminator word when i == fCount)
  const iJump = i;

  // F scan.
  let fIters = 0, fBrokeAt = -1;
  for (; i < fCount; i++) {
    fIters++;
    C.meta8++;
    if (B.fXmax[i] - rcx <= -hx) { // fully left
      C.fCullLeft++;
      if (B.fXmin[i] - rcx < -hx - B.wF) { fBrokeAt = i; break; } // wF break
      continue;
    }
    C.fInt++; // integrate_piece (F pieces have no y-cull / cheap-right path)
  }
  C.fScan += fIters;
  const fCutA = fBrokeAt >= 0 ? fCount - 1 - fBrokeAt : 0;
  C.fCut += fCutA;
  if (iJump + fIters + fCutA !== fCount) C.invar++;
  const T = rcx - hx; // exact fully-left threshold on hull x-max
  {
    const p = Math.max(firstLE(B.sufF, T), iJump);
    const cutI = Math.max(0, fCount - 1 - p); // ideal break walks position p, cuts the rest
    if (cutI > fCutA) C.fFail += cutI - fCutA;
  }

  // E jump-find (covered windows only): prefix of fully-right pieces collapses to one span-sum read.
  let j = 0;
  if (covered) {
    while (j < eCount) {
      C.eJump++;
      C.meta8++;
      if (B.eXmin[j] - rcx < hx) break;
      j++;
    }
    C.meta8++; // prefixE span-sum word read (always, even when eCount == 0)
  }
  const jStart = j;

  // E scan.
  let eIters = 0, eBrokeAt = -1;
  for (; j < eCount; j++) {
    eIters++;
    C.meta8++;
    if (B.eXmax[j] - rcx <= -hx) { // fully left
      C.eCullLeft++;
      if (B.eXmin[j] - rcx < -hx - B.wE) { eBrokeAt = j; break; } // wE break
      continue;
    }
    if (B.eXmin[j] - rcx >= hx) { // fully right → cheap clipped-span add (2 of the 3 piece words)
      C.eCheap++;
      continue;
    }
    if (B.eYhi[j] - rcy <= wlo || B.eYlo[j] - rcy >= whi) { // y-disjoint from the window
      C.eYCull++;
      continue;
    }
    C.eInt++; // integrate_piece
  }
  C.eScan += eIters;
  const eCutA = eBrokeAt >= 0 ? eCount - 1 - eBrokeAt : 0;
  C.eCut += eCutA;
  if (jStart + eIters + eCutA !== eCount) C.invar++;
  {
    const p = Math.max(firstLE(B.sufE, T), jStart);
    const cutI = Math.max(0, eCount - 1 - p);
    if (cutI > eCutA) C.eFail += cutI - eCutA;
  }
}

// ── one scene+level measurement ───────────────────────────────────────────────────────────────────────
function runLevel(scene, emPx, bandGet) {
  const camS = emPx / EM_WORLD;
  const tx = W / 2 - camS * scene.center.x;
  const ty = H / 2 - camS * scene.center.y;
  const wI = scene.wInstances;

  // Per-level culling — identical to bench/main.js visibleIndices().
  const halfW = (W / 2) / camS, halfH = (H / 2) / camS;
  const margin = Math.min(EM_WORLD, Math.max(halfW, halfH));
  const loX = scene.center.x - halfW - margin, hiX = scene.center.x + halfW + margin;
  const loY = scene.center.y - halfH - margin, hiY = scene.center.y + halfH + margin;
  const visible = [];
  for (let i = 0; i < scene.count; i++) {
    const b = i * 16;
    const px = wI[b], py = wI[b + 1], sc = wI[b + 2];
    const gx0 = px + wI[b + 4] * sc, gy0 = py + wI[b + 5] * sc;
    const gx1 = px + wI[b + 6] * sc, gy1 = py + wI[b + 7] * sc;
    if (gx1 >= loX && gx0 <= hiX && gy1 >= loY && gy0 <= hiY) visible.push(i);
  }

  // Device-pixel footprint of instance i (padded quad clipped to the viewport), as in the vertex shader.
  const bounds = (i) => {
    const b = i * 16;
    const px = wI[b], py = wI[b + 1], sc = wI[b + 2];
    const pad = SKIRT_PX / (sc * camS); // glyph units
    const dx0 = (px + (wI[b + 4] - pad) * sc) * camS + tx;
    const dx1 = (px + (wI[b + 6] + pad) * sc) * camS + tx;
    const dy0 = (py + (wI[b + 5] - pad) * sc) * camS + ty;
    const dy1 = (py + (wI[b + 7] + pad) * sc) * camS + ty;
    const x0 = Math.max(0, Math.ceil(dx0 - 0.5)), x1 = Math.min(W - 1, Math.ceil(dx1 - 0.5) - 1);
    const y0 = Math.max(0, Math.ceil(dy0 - 0.5)), y1 = Math.min(H - 1, Math.ceil(dy1 - 0.5) - 1);
    return { x0, x1, y0, y1, n: Math.max(0, x1 - x0 + 1) * Math.max(0, y1 - y0 + 1) };
  };

  let totalFrags = 0;
  const areas = new Array(visible.length);
  for (let k = 0; k < visible.length; k++) {
    areas[k] = bounds(visible[k]).n;
    totalFrags += areas[k];
  }

  // Uniform instance sampling above the fragment budget (seeded shuffle; keep ≥ MIN_INST instances).
  const order = visible.map((_, k) => k);
  const sampled = totalFrags > MAX_FRAGS;
  if (sampled) {
    let seed = SEED;
    const rnd = () => ((seed = (seed * 1664525 + 1013904223) >>> 0) / 2 ** 32);
    for (let k = order.length - 1; k > 0; k--) {
      const j = Math.floor(rnd() * (k + 1));
      const t = order[k]; order[k] = order[j]; order[j] = t;
    }
  }
  const picked = [];
  let measuredFrags = 0;
  for (const k of order) {
    if (sampled && measuredFrags >= MAX_FRAGS && picked.length >= MIN_INST) break;
    if (areas[k] === 0) continue;
    picked.push(visible[k]);
    measuredFrags += areas[k];
  }

  const C = newC();
  const winLo = new Float64Array(64), winHi = new Float64Array(64);
  const winCov = new Uint8Array(64);
  const rowBands = new Array(64);

  for (const i of picked) {
    const b = i * 16;
    const ox = wI[b], oy = wI[b + 1], sc = wI[b + 2];
    const bb0 = wI[b + 4], bb1 = wI[b + 5], bb2 = wI[b + 6], bb3 = wI[b + 7];
    const rowBase = wI[b + 12], R = wI[b + 13], bandH = wI[b + 14], invH = wI[b + 15];
    const s = 1 / (sc * camS); // fwidth(rc): glyph units per device pixel, both axes (isotropic cam)
    const bo = bounds(i);
    if (bo.n === 0) continue;
    const guarded = s * GUARD_PX >= bb2 - bb0 && s * GUARD_PX >= bb3 - bb1;
    if (guarded) { // profile_face path: no gather work at all
      C.frags += bo.n;
      C.guardFrags += bo.n;
      continue;
    }
    const sy2 = s * 0.5;
    for (let py = bo.y0; py <= bo.y1; py++) {
      const rcy = ((py + 0.5 - ty) / camS - oy) / sc;
      const dy0 = bb1 - rcy; // y0 − rc.y, as in integrate_face
      let ri0 = 0, ri1 = 0;
      if (R > 1) {
        ri0 = Math.min(Math.max(Math.floor((-dy0 - sy2) * invH), 0), R - 1);
        ri1 = Math.min(Math.max(Math.floor((-dy0 + sy2) * invH), 0), R - 1);
      }
      // Band windows are x-independent: precompute once per pixel row.
      let nb = 0;
      for (let ri = ri0; ri <= ri1; ri++) {
        let wlo = -sy2, whi = sy2, covered = false;
        if (R > 1) {
          const ex = dy0 + ri * bandH, ey = dy0 + (ri + 1) * bandH;
          covered = wlo <= ex && whi >= ey;
          if (ex > wlo) wlo = ex;
          if (ey < whi) whi = ey;
        }
        if (whi <= wlo) continue; // zero-height window: integrate_band never runs
        winLo[nb] = wlo;
        winHi[nb] = whi;
        winCov[nb] = covered ? 1 : 0;
        rowBands[nb] = bandGet(rowBase + 2 * ri);
        nb++;
      }
      for (let px = bo.x0; px <= bo.x1; px++) {
        const rcx = ((px + 0.5 - tx) / camS - ox) / sc;
        C.frags++;
        C.exactFrags++;
        C.bands += nb;
        for (let k = 0; k < nb; k++) {
          if (winCov[k]) C.covered++;
          integrateBandCount(C, rowBands[k], rcx, rcy, winLo[k], winHi[k], s, winCov[k] === 1);
        }
      }
    }
  }

  return { C, totalFrags, measuredFrags, sampled, visible: visible.length, pickedCount: picked.length };
}

// ── scenes (bench-identical builders + extents) ───────────────────────────────────────────────────────
async function buildScenes() {
  const extentFor = (minLevel) => Math.max((W / 2) / (minLevel / EM_WORLD) * 1.05, 4 * EM_WORLD);
  const out = {};
  const font = await loadFont(new URL('../assets/Lato-Regular.ttf', import.meta.url));
  const wA = buildGlyphAtlas(font, SCENE_TEXT);
  const sA = buildSlugAtlas(font, SCENE_TEXT);
  const grid = buildScene(font, wA.table, sA.table, { emWorld: EM_WORLD, extent: extentFor(LADDER_MIN.glyphs), color: INK });
  out.glyphs = {
    wCurves: wA.curves, wRows: wA.rows,
    wInstances: grid.wInstances, count: grid.count, center: grid.center,
  };
  out.shape = buildShapeScene({ emWorld: EM_WORLD, extent: extentFor(LADDER_MIN.shape), fillRule: 0 });
  out.tiger = await buildTigerScene({ emWorld: EM_WORLD, extent: extentFor(LADDER_MIN.tiger) });
  return out;
}

// ── main ──────────────────────────────────────────────────────────────────────────────────────────────
const PLAN = [
  ['shape', [12, 32, 128]],
  ['glyphs', [8, 16, 64]],
  ['tiger', [64, 256]],
];

const scenes = await buildScenes();
const getters = Object.fromEntries(Object.keys(scenes).map((k) => [k, makeBandGetter(scenes[k])]));

const results = [];
for (const [name, levels] of PLAN) {
  for (const emPx of levels) {
    const t0 = performance.now();
    const r = runLevel(scenes[name], emPx, getters[name]);
    const C = r.C;
    const ef = Math.max(C.exactFrags, 1);
    const row = {
      key: `${name}@${emPx}`,
      totalFrags: r.totalFrags,
      sampleNote: r.sampled ? `${r.pickedCount}/${r.visible}i` : `all ${r.visible}i`,
      guardPct: 100 * C.guardFrags / Math.max(C.frags, 1),
      bandsPerFrag: C.bands / ef,
      covPerFrag: C.covered / ef,
      occF: C.occF / Math.max(C.bands, 1),
      occE: C.occE / Math.max(C.bands, 1),
      fJump: C.fJump / ef,
      fCullLeft: C.fCullLeft / ef,
      fInt: C.fInt / ef,
      fCut: C.fCut / ef,
      fFail: C.fFail / ef,
      eJump: C.eJump / ef,
      eCullLeft: C.eCullLeft / ef,
      eCheap: C.eCheap / ef,
      eYCull: C.eYCull / ef,
      eInt: C.eInt / ef,
      eCut: C.eCut / ef,
      eFail: C.eFail / ef,
      ip: (C.fInt + C.eInt) / ef,
      meta8: C.meta8 / ef,
      piece24: (C.fInt + C.eInt + (2 / 3) * C.eCheap) / ef,
      invar: C.invar,
    };
    results.push(row);
    console.error(
      `${row.key} done in ${((performance.now() - t0) / 1000).toFixed(1)}s ` +
      `(${row.sampleNote}, ${C.exactFrags} exact frags, invariant misses ${C.invar})`,
    );
  }
}

// ── print ─────────────────────────────────────────────────────────────────────────────────────────────
const fmt = (v, d = 2) => (Number.isFinite(v) ? v.toFixed(d) : 'n/a');
const pad = (s, n) => String(s).padStart(n);
console.log('\nPer-fragment MEANS over exact (non-guard) fragments. guard% = share of real fragments on the');
console.log('profile_face path (no gather). b/f bands visited, cov/f covered bands; F/b, E/b = mean band');
console.log('occupancy (fCount, eCount) per visited band. F: jump-find meta words walked (fJmp), then scan');
console.log('split cull-left / integrated / cut-by-wF-break; fFail = walked pieces an ideal exact break would');
console.log('have cut. E: covered-only jump words (eJmp), scan split cull-left / cheap-right / y-cull /');
console.log('integrated / cut-by-wE-break; eFail likewise. ip/f integrate_piece calls; meta8/f 8-byte meta+');
console.log('prefix word loads; pc24/f 24-byte-equivalent piece-data touches (integrate = 1, cheap-right = 2/3).\n');
const hdr =
  'scene@px     frags smpl        guard%   b/f cov/f   F/b   E/b | fJmp/f  fL/f fInt/f fCut/f fFail/f' +
  ' | eJmp/f  eL/f  eR/f  eY/f eInt/f eCut/f eFail/f |  ip/f meta8/f pc24/f';
console.log(hdr);
for (const r of results) {
  console.log(
    pad(r.key, 10) + pad((r.totalFrags / 1000).toFixed(0) + 'k', 7) + ' ' + r.sampleNote.padEnd(11) +
    pad(fmt(r.guardPct, 1), 7) + pad(fmt(r.bandsPerFrag), 6) + pad(fmt(r.covPerFrag), 6) +
    pad(fmt(r.occF, 1), 6) + pad(fmt(r.occE, 1), 6) + ' |' +
    pad(fmt(r.fJump), 7) + pad(fmt(r.fCullLeft), 6) + pad(fmt(r.fInt), 7) + pad(fmt(r.fCut), 7) + pad(fmt(r.fFail), 8) + ' |' +
    pad(fmt(r.eJump), 7) + pad(fmt(r.eCullLeft), 6) + pad(fmt(r.eCheap), 6) + pad(fmt(r.eYCull), 6) +
    pad(fmt(r.eInt), 7) + pad(fmt(r.eCut), 7) + pad(fmt(r.eFail), 8) + ' |' +
    pad(fmt(r.ip), 6) + pad(fmt(r.meta8, 1), 8) + pad(fmt(r.piece24), 7),
  );
}
const bad = results.filter((r) => r.invar > 0);
if (bad.length) console.log(`\nWARNING: traversal invariant misses in: ${bad.map((r) => r.key).join(', ')}`);
