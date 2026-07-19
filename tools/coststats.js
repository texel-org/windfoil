// tools/coststats.js — CPU replication of the exact per-fragment traversal WORK of windfoil.wgsl vs
// bench/slug.wgsl, over the same scenes / camera / culling as bench/main.js. Counts (not renders) what each
// fragment would do: bands touched, pieces visited, far-left/far-right/y-skip/ulp/straddle classification,
// mono_root early-out vs sqrt, slug's per-ray band curves / far breaks / root-code skips / full solves.
// Also classifies, per straddle visit, whether a band-CLIPPED x-hull (tight hull of the piece restricted to
// the band's y-range) would have reclassified the visit (false straddles), and whether the pixel's y-window
// fully covers each visited band (gates a per-band prefix-sum optimization).
//
// TWO cost roll-ups:
//   • lane ALU  — every lane pays only its own path (the "counts" model from the task).
//   • warp ALU  — 8×4 = 32-lane SIMD-group lockstep: each piece iteration costs the UNION of the branch
//     classes taken across active lanes (branches serialize), loops run to the max lane trip count, band
//     setup is paid once per warp, and partially-covered tiles still pay full warp issue. This is what a
//     real GPU pays; the lane/warp gap is measured divergence + lane-utilization loss, not a fudge.
//
//   deno run -A tools/coststats.js
//
// All curve/band data comes from the real builders (bands.js, bench/slug.js, bench/{scene,shape,tiger}.js)
// so counts match the bench atlases bit-for-bit; traversal math is f64 (counts are boundary-insensitive).

import { loadFont } from '../src/font.js';
import { buildGlyphAtlas } from '../src/bands.js';
import { buildSlugAtlas } from '../bench/slug.js';
import { buildScene, SCENE_TEXT, INK } from '../bench/scene.js';
import { buildShapeScene } from '../bench/shape.js';
import { buildTigerScene } from '../bench/tiger.js';

// ── bench-identical constants (bench/main.js defaults) ────────────────────────────────────────────────
const W = 720, H = 720;
const EM_WORLD = 100;
const SKIRT_PX = 0.625; // KERNEL_SKIRT_PX, both shaders pad the quad by this
const GUARD_PX = 3.7; // windfoil minification guard
const SORT_MIN = 4; // BAND_SORT_MIN
const MAX_FRAGS = 400_000; // sample instances above this many fragments per level
const MIN_INST = 40; // keep at least this many instances in a sample (variance control for tiger)
const TW = 8, TH = 4; // SIMD-group tile (32 lanes, Apple-style)

// The bench builds each scene's grid for the FULL default ladder's smallest level — replicate that so
// instance placement matches the bench exactly (main.js: extent = max((720/2)/(levels[0]/100)*1.05, 400)).
const LADDER_MIN = { glyphs: 2, shape: 12, tiger: 64 };

// ALU cost model (from the task): per-event costs in "ALU units".
const COST = {
  straddle: 60, farRight: 10, ulp: 10, farLeftSkip: 4, ySkip: 6, bandSetup: 15,
  slugSolve: 35, slugSkip: 6, slugCodeSkip: 10,
};
// Branch-class bit → cost, for the warp-union roll-up.
const W_CLASS = [COST.farLeftSkip, COST.ySkip, COST.farRight, COST.ulp, COST.straddle]; // bits 0..4
const S_CLASS = [COST.slugSkip, COST.slugCodeSkip, COST.slugSolve]; // bits 0..2
const wUnion = new Float64Array(32), sUnion = new Float64Array(8);
for (let m = 0; m < 32; m++) for (let b = 0; b < 5; b++) if (m & (1 << b)) wUnion[m] += W_CLASS[b];
for (let m = 0; m < 8; m++) for (let b = 0; b < 3; b++) if (m & (1 << b)) sUnion[m] += S_CLASS[b];

// Baseline measured ms/frame (Apple GPU) for the self-consistency check: [windfoil, slug].
const MEASURED = {
  'glyphs@8': [3.25, 1.00], 'glyphs@16': [1.24, 0.58], 'glyphs@32': [0.578, 0.388], 'glyphs@64': [0.328, 0.298],
  'shape@12': [35.9, 5.71], 'shape@32': [9.59, 3.56], 'shape@128': [3.45, 2.52], 'shape@512': null,
  'tiger@64': [40.0, 8.44], 'tiger@256': [6.94, 4.89], 'tiger@1024': null,
};

// ── f64 twin of windfoil.wgsl's mono_root, with root-kind counting into counters C ────────────────────
function monoRootCount(C, a2, a1, a0, e1, v, rising) {
  if (rising ? a0 >= v : a0 <= v) { C.rootEarly++; return 0; }
  if (rising ? e1 <= v : e1 >= v) { C.rootEarly++; return 1; }
  const c = a0 - v;
  if (Math.abs(a2) < 1e-12 * Math.max(Math.abs(a1), 1)) {
    C.rootLinear++;
    return Math.min(Math.max(-c / a1, 0), 1);
  }
  C.rootSqrt++;
  const sq = Math.sqrt(Math.max(a1 * a1 - 4 * a2 * c, 0));
  const qq = -0.5 * (a1 + (a1 >= 0 ? sq : -sq));
  const useR1 = (a1 < 0) === rising;
  const num = useR1 ? qq : c;
  const den = useR1 ? a2 : qq;
  const t = den !== 0 ? num / den : 0;
  return Math.min(Math.max(t, 0), 1);
}

// Same solve, no counting (used by the band-clip reclassification analysis).
function monoRootPlain(a2, a1, a0, e1, v, rising) {
  if (rising ? a0 >= v : a0 <= v) return 0;
  if (rising ? e1 <= v : e1 >= v) return 1;
  const c = a0 - v;
  if (Math.abs(a2) < 1e-12 * Math.max(Math.abs(a1), 1)) return Math.min(Math.max(-c / a1, 0), 1);
  const sq = Math.sqrt(Math.max(a1 * a1 - 4 * a2 * c, 0));
  const qq = -0.5 * (a1 + (a1 >= 0 ? sq : -sq));
  const useR1 = (a1 < 0) === rising;
  const t = (useR1 ? a2 : qq) !== 0 ? (useR1 ? qq : c) / (useR1 ? a2 : qq) : 0;
  return Math.min(Math.max(t, 0), 1);
}

function newCounters() {
  return {
    // fragment classes
    frags: 0, exactFrags: 0, guardFrags: 0, guardTaps: 0, warps: 0,
    // windfoil per-band (per lane)
    bandIter: 0, bands: 0, covered: 0, bandCurves: 0,
    // windfoil per-piece (per lane)
    visits: 0, farLeft: 0, breakHits: 0, breakSkipped: 0, ySkip: 0, farRight: 0, ulp: 0,
    straddle: 0, degStraddle: 0,
    rootEarly: 0, rootLinear: 0, rootSqrt: 0,
    // band-clip reclassification of straddle visits
    falseLeft: 0, falseRight: 0, falseNone: 0,
    // slug (both rays summed, per lane)
    sBandCurves: 0, sVisited: 0, sFarSkip: 0, sBreak: 0, sBreakSkipped: 0, sCode0: 0, sSolved: 0,
    // warp-lockstep serialized ALU
    warpW: 0, warpS: 0,
  };
}

// Per-warp lane scratch (32 lanes max).
const L = 32;
const laneRcx = new Float64Array(L), laneRcy = new Float64Array(L);
const laneRi0 = new Int32Array(L), laneRi1 = new Int32Array(L);
const laneWlo = new Float64Array(L), laneWhi = new Float64Array(L);
const laneBLo = new Float64Array(L), laneBHi = new Float64Array(L);
const laneUlp = new Float64Array(L);
const laneOn = new Uint8Array(L); // active for the current band
const laneBrk = new Uint8Array(L); // broke out of the current band's piece loop
const laneBi = new Int32Array(L), laneStart = new Int32Array(L), laneCount = new Int32Array(L);
const laneSort = new Uint8Array(L);

// One slug ray for a whole warp, lockstep. Returns nothing; increments per-lane counters + C.warpS.
function slugRayWarp(C, curves, rows, lanesN, rcxArr, rcyArr, sign, rowBase, R, y0, invH, half) {
  // sign: +1 → ray uses (rcx, rcy) plain; -1 → vertical frame (rc.y, −rc.x) with the arrays pre-swapped by caller.
  C.warpS += COST.bandSetup; // per-warp band header read + setup
  let maxCount = 0;
  for (let l = 0; l < lanesN; l++) {
    const ry = rcyArr[l];
    let bi = 0;
    if (invH > 0 && R > 1) bi = Math.min(Math.max(Math.floor((ry - y0) * invH), 0), R - 1);
    const rIdx = (rowBase + bi) * 5;
    laneBi[l] = bi;
    laneStart[l] = rows[rIdx];
    laneCount[l] = rows[rIdx + 1];
    laneSort[l] = laneCount[l] > SORT_MIN ? 1 : 0;
    laneBrk[l] = 0;
    C.sBandCurves += laneCount[l];
    if (laneCount[l] > maxCount) maxCount = laneCount[l];
  }
  for (let i = 0; i < maxCount; i++) {
    let mask = 0, alive = 0;
    for (let l = 0; l < lanesN; l++) {
      if (laneBrk[l] || i >= laneCount[l]) continue;
      alive++;
      C.sVisited++;
      const b = (laneStart[l] + i) * 6;
      const rx = rcxArr[l], ry = rcyArr[l];
      const p1x = curves[b] - rx, p1y = curves[b + 1] - ry;
      const p2x = curves[b + 2] - rx, p2y = curves[b + 3] - ry;
      const p3x = curves[b + 4] - rx, p3y = curves[b + 5] - ry;
      const mx = Math.max(p1x, p2x, p3x);
      if (mx <= -half) { // fully behind the ray's near clamp edge
        mask |= 1;
        if (laneSort[l]) { laneBrk[l] = 1; C.sBreak++; C.sBreakSkipped += laneCount[l] - 1 - i; } else C.sFarSkip++;
        continue;
      }
      // calc_root_code from the three y-sign bits (f64 twin of the 0x2E74 table).
      const shift = (p1y < 0 ? 1 : 0) | (p2y < 0 ? 2 : 0) | (p3y < 0 ? 4 : 0);
      if (((0x2E74 >> shift) & 0x0101) === 0) { C.sCode0++; mask |= 2; } else { C.sSolved++; mask |= 4; }
    }
    if (!alive) break;
    C.warpS += sUnion[mask];
  }
}

// ── one scene+level measurement ───────────────────────────────────────────────────────────────────────
function runLevel(scene, emPx) {
  const camS = emPx / EM_WORLD;
  const tx = W / 2 - camS * scene.center.x;
  const ty = H / 2 - camS * scene.center.y;
  const wI = scene.wInstances, sI = scene.sInstances;
  const wCurves = scene.wCurves, wRows = scene.wRows;
  const sCurves = scene.sCurves, sRows = scene.sRows;

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

  // Device-pixel footprint of instance i (padded quad clipped to the viewport), as in both vertex shaders.
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
  for (let k = 0; k < visible.length; k++) { areas[k] = bounds(visible[k]).n; totalFrags += areas[k]; }

  // Uniform instance sampling above the fragment budget (seeded shuffle, take until >= MAX_FRAGS and at
  // least MIN_INST instances so per-shape variance stays bounded on tiger).
  const order = visible.map((_, k) => k);
  const sampled = totalFrags > MAX_FRAGS;
  if (sampled) {
    let seed = 0x9e3779b9 >>> 0;
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
    if (sampled && measuredFrags >= 5 * MAX_FRAGS) break;
    if (areas[k] === 0) continue;
    picked.push(visible[k]);
    measuredFrags += areas[k];
  }
  const scale = measuredFrags > 0 ? totalFrags / measuredFrags : 1;

  const C = newCounters();

  for (const i of picked) {
    const b = i * 16, sb = i * 20;
    const ox = wI[b], oy = wI[b + 1], sc = wI[b + 2];
    const bb0 = wI[b + 4], bb1 = wI[b + 5], bb2 = wI[b + 6], bb3 = wI[b + 7];
    const rowBase = wI[b + 12], R = wI[b + 13], bandH = wI[b + 14], invH = wI[b + 15];
    const hRowBase = sI[sb + 12], hR = sI[sb + 13], hY0 = sI[sb + 14], hInvH = sI[sb + 15];
    const vRowBase = sI[sb + 16], vR = sI[sb + 17], vY0 = sI[sb + 18], vInvW = sI[sb + 19];
    const s = 1 / (sc * camS); // units per device pixel (fwidth(rc), both axes — cam is isotropic)
    const sy2 = s * 0.5, hx = s * 0.5;
    const guarded = s * GUARD_PX >= bb2 - bb0 && s * GUARD_PX >= bb3 - bb1;
    const bo = bounds(i);
    if (bo.n === 0) continue;

    // Screen-aligned 8×4 SIMD tiles over the footprint.
    const tileY0 = Math.floor(bo.y0 / TH) * TH, tileX0 = Math.floor(bo.x0 / TW) * TW;
    for (let tyv = tileY0; tyv <= bo.y1; tyv += TH) {
      for (let txv = tileX0; txv <= bo.x1; txv += TW) {
        // Gather this warp's live lanes.
        let lanesN = 0;
        for (let dy = 0; dy < TH; dy++) {
          const pyI = tyv + dy;
          if (pyI < bo.y0 || pyI > bo.y1) continue;
          const rcy = ((pyI + 0.5 - ty) / camS - oy) / sc;
          for (let dx = 0; dx < TW; dx++) {
            const pxI = txv + dx;
            if (pxI < bo.x0 || pxI > bo.x1) continue;
            laneRcx[lanesN] = ((pxI + 0.5 - tx) / camS - ox) / sc;
            laneRcy[lanesN] = rcy;
            lanesN++;
          }
        }
        if (!lanesN) continue;
        C.frags += lanesN;
        C.warps++;

        // ── windfoil ────────────────────────────────────────────────────────────────
        if (guarded) {
          C.guardFrags += lanesN;
          let maxTaps = 0;
          for (let l = 0; l < lanesN; l++) {
            const rcx = laneRcx[l], rcy = laneRcy[l];
            let taps = 0;
            if (Math.min(rcx + hx, bb2) - Math.max(rcx - hx, bb0) > 0) {
              if (R > 1) {
                const r0 = Math.min(Math.max(Math.floor((rcy - sy2 - bb1) * invH), 0), R - 1);
                const r1 = Math.min(Math.max(Math.floor((rcy + sy2 - bb1) * invH), 0), R - 1);
                taps = r1 - r0 + 1;
              } else taps = 1;
            }
            C.guardTaps += taps;
            if (taps > maxTaps) maxTaps = taps;
          }
          C.warpW += COST.bandSetup * maxTaps;
        } else {
          C.exactFrags += lanesN;
          let riMin = 1 << 30, riMax = -1;
          for (let l = 0; l < lanesN; l++) {
            const dy0 = bb1 - laneRcy[l];
            let r0 = 0, r1 = 0;
            if (R > 1) {
              r0 = Math.min(Math.max(Math.floor((-dy0 - sy2) * invH), 0), R - 1);
              r1 = Math.min(Math.max(Math.floor((-dy0 + sy2) * invH), 0), R - 1);
            }
            laneRi0[l] = r0; laneRi1[l] = r1;
            if (r0 < riMin) riMin = r0;
            if (r1 > riMax) riMax = r1;
            laneUlp[l] = Math.max(Math.abs(laneRcx[l]), Math.abs(laneRcy[l])) * 1.2e-7 * 16;
          }
          for (let ri = riMin; ri <= riMax; ri++) {
            let activeCount = 0;
            for (let l = 0; l < lanesN; l++) {
              laneOn[l] = 0;
              if (ri < laneRi0[l] || ri > laneRi1[l]) continue;
              C.bandIter++;
              const dy0 = bb1 - laneRcy[l];
              let wlo = -sy2, whi = sy2, bLo, bHi;
              if (R > 1) {
                bLo = dy0 + ri * bandH; bHi = dy0 + (ri + 1) * bandH;
                if (bLo > wlo) wlo = bLo;
                if (bHi < whi) whi = bHi;
              } else { bLo = bb1 - laneRcy[l]; bHi = bb3 - laneRcy[l]; }
              if (whi <= wlo) continue;
              C.bands++;
              if (bLo >= -sy2 && bHi <= sy2) C.covered++; // pixel y-window covers the whole band
              laneOn[l] = 1;
              laneBrk[l] = 0;
              laneWlo[l] = wlo; laneWhi[l] = whi;
              laneBLo[l] = bLo; laneBHi[l] = bHi;
              activeCount++;
            }
            if (!activeCount) continue;
            C.warpW += COST.bandSetup;
            const rIdx = (rowBase + ri) * 5;
            const start = wRows[rIdx], count = wRows[rIdx + 1];
            C.bandCurves += count * activeCount;
            const sorted = count > SORT_MIN;
            let alive = activeCount;
            for (let k = 0; k < count && alive > 0; k++) {
              let mask = 0;
              const cb = (start + k) * 6;
              const c1x = wCurves[cb], c1y = wCurves[cb + 1];
              const c3x = wCurves[cb + 4], c3y = wCurves[cb + 5];
              for (let l = 0; l < lanesN; l++) {
                if (!laneOn[l] || laneBrk[l]) continue;
                C.visits++;
                const rcx = laneRcx[l], rcy = laneRcy[l];
                const q1x = c1x - rcx, q1y = c1y - rcy;
                const q3x = c3x - rcx, q3y = c3y - rcy;
                const xmax = q1x > q3x ? q1x : q3x;
                if (xmax <= -hx) { // fully LEFT
                  mask |= 1;
                  if (sorted) { laneBrk[l] = 1; alive--; C.breakHits++; C.breakSkipped += count - 1 - k; } else C.farLeft++;
                  continue;
                }
                const pyLo = q1y < q3y ? q1y : q3y;
                const pyHi = q1y < q3y ? q3y : q1y;
                const lo = laneWlo[l] > pyLo ? laneWlo[l] : pyLo;
                const hi = laneWhi[l] < pyHi ? laneWhi[l] : pyHi;
                if (hi <= lo) { C.ySkip++; mask |= 2; continue; }
                const xmin = q1x < q3x ? q1x : q3x;
                if (xmin >= hx) { C.farRight++; mask |= 4; continue; } // fully RIGHT: cheap add
                if (xmax - xmin + (pyHi - pyLo) <= laneUlp[l]) { C.ulp++; mask |= 8; continue; }
                // STRADDLE → integrate_piece: replicate its 4 mono_root calls.
                C.straddle++;
                mask |= 16;
                const q2x = wCurves[cb + 2] - rcx, q2y = wCurves[cb + 3] - rcy;
                const a2x = q1x - 2 * q2x + q3x, a2y = q1y - 2 * q2y + q3y;
                const a1x = 2 * (q2x - q1x), a1y = 2 * (q2y - q1y);
                const yr = q3y >= q1y;
                const tLo = monoRootCount(C, a2y, a1y, q1y, q3y, yr ? lo : hi, yr);
                const tHi = monoRootCount(C, a2y, a1y, q1y, q3y, yr ? hi : lo, yr);
                if (tHi <= tLo) C.degStraddle++;
                else {
                  const xr = q3x >= q1x;
                  monoRootCount(C, a2x, a1x, q1x, q3x, -hx, xr);
                  monoRootCount(C, a2x, a1x, q1x, q3x, hx, xr);
                }
                // Band-clip reclassification: tight x-hull of the piece restricted to THIS band's y-range.
                const cl = laneBLo[l] > pyLo ? laneBLo[l] : pyLo;
                const ch = laneBHi[l] < pyHi ? laneBHi[l] : pyHi;
                if (ch <= cl) C.falseNone++; // piece has no y-extent inside the band at all
                else {
                  const tA = monoRootPlain(a2y, a1y, q1y, q3y, yr ? cl : ch, yr);
                  const tB = monoRootPlain(a2y, a1y, q1y, q3y, yr ? ch : cl, yr);
                  const xa = (a2x * tA + a1x) * tA + q1x;
                  const xb = (a2x * tB + a1x) * tB + q1x;
                  const hMin = xa < xb ? xa : xb, hMax = xa < xb ? xb : xa;
                  if (hMax <= -hx) C.falseLeft++; // would have been far-left (or sorted-break sooner)
                  else if (hMin >= hx) C.falseRight++; // would have been the cheap far-right add
                }
              }
              C.warpW += wUnion[mask];
            }
          }
        }

        // ── slug: two rays, one band each, lockstep across the warp ─────────────────
        slugRayWarp(C, sCurves, sRows, lanesN, laneRcx, laneRcy, 1, hRowBase, hR, hY0, hInvH, hx);
        // Vertical frame: rc' = (rc.y, −rc.x). Reuse scratch by materializing the rotated coords.
        for (let l = 0; l < lanesN; l++) { laneWlo[l] = laneRcy[l]; laneWhi[l] = -laneRcx[l]; }
        slugRayWarp(C, sCurves, sRows, lanesN, laneWlo, laneWhi, -1, vRowBase, vR, vY0, vInvW, hx);
      }
    }
  }

  return { C, totalFrags, measuredFrags, scale, sampled, visible: visible.length, pickedCount: picked.length };
}

// ── ALU roll-ups ──────────────────────────────────────────────────────────────────────────────────────
function laneAluOf(C) {
  const w = C.bands * COST.bandSetup +
    C.straddle * COST.straddle +
    C.farRight * COST.farRight +
    C.ulp * COST.ulp +
    (C.farLeft + C.breakHits) * COST.farLeftSkip +
    C.ySkip * COST.ySkip +
    C.guardTaps * COST.bandSetup; // guard profile taps ≈ a band setup each (row read + overlap math)
  const s = C.frags * 2 * COST.bandSetup +
    C.sSolved * COST.slugSolve +
    C.sCode0 * COST.slugCodeSkip +
    (C.sFarSkip + C.sBreak) * COST.slugSkip;
  return { w, s };
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
    wCurves: wA.curves, wRows: wA.rows, sCurves: sA.curves, sRows: sA.rows,
    wInstances: grid.wInstances, sInstances: grid.sInstances, count: grid.count, center: grid.center,
  };

  out.shape = buildShapeScene({ emWorld: EM_WORLD, extent: extentFor(LADDER_MIN.shape), fillRule: 0 });
  out.tiger = await buildTigerScene({ emWorld: EM_WORLD, extent: extentFor(LADDER_MIN.tiger) });
  return out;
}

// ── main ──────────────────────────────────────────────────────────────────────────────────────────────
const PLAN = [
  ['glyphs', [8, 16, 32, 64]],
  ['shape', [12, 32, 128, 512]],
  ['tiger', [64, 256, 1024]],
];

const scenes = await buildScenes();
const rows = [];
for (const [name, levels] of PLAN) {
  for (const emPx of levels) {
    const t0 = performance.now();
    const r = runLevel(scenes[name], emPx);
    const C = r.C;
    const ef = Math.max(C.exactFrags, 1);
    const alu = laneAluOf(C);
    const meas = MEASURED[`${name}@${emPx}`];
    const row = {
      key: `${name}@${emPx}`,
      totalFrags: r.totalFrags,
      sampled: r.sampled,
      sampleNote: r.sampled ? `${r.pickedCount}/${r.visible} inst x${r.scale.toFixed(2)}` : `all ${r.visible} inst`,
      guardPct: 100 * C.guardFrags / Math.max(C.frags, 1),
      lanesPerWarp: C.frags / Math.max(C.warps, 1),
      bandsPerFrag: C.bands / ef,
      visitsPerFrag: C.visits / ef,
      straddlesPerFrag: C.straddle / ef,
      falseStraddlePct: 100 * (C.falseLeft + C.falseRight + C.falseNone) / Math.max(C.straddle, 1),
      falseLeftPct: 100 * (C.falseLeft + C.falseNone) / Math.max(C.straddle, 1),
      falseRightPct: 100 * C.falseRight / Math.max(C.straddle, 1),
      sqrtPerStraddle: C.rootSqrt / Math.max(C.straddle, 1),
      coveredPct: 100 * C.covered / Math.max(C.bands, 1),
      slugSolvesPerFrag: C.sSolved / Math.max(C.frags, 1),
      slugVisitedPerFrag: C.sVisited / Math.max(C.frags, 1),
      slugBandCurvesPerFrag: C.sBandCurves / Math.max(C.frags, 1),
      wBandOcc: C.bandCurves / Math.max(C.bands, 1),
      farRightPerFrag: C.farRight / ef,
      farLeftPerFrag: (C.farLeft + C.breakHits) / ef,
      ySkipPerFrag: C.ySkip / ef,
      ulpPerFrag: C.ulp / ef,
      breakSkippedPerFrag: C.breakSkipped / ef,
      sBreakSkippedPerFrag: C.sBreakSkipped / Math.max(C.frags, 1),
      sCode0PerFrag: C.sCode0 / Math.max(C.frags, 1),
      wAluPerFrag: alu.w / Math.max(C.frags, 1),
      sAluPerFrag: alu.s / Math.max(C.frags, 1),
      wAluTotal: alu.w * r.scale,
      sAluTotal: alu.s * r.scale,
      aluRatio: alu.w / Math.max(alu.s, 1),
      wWarpTotal: C.warpW * r.scale,
      sWarpTotal: C.warpS * r.scale,
      warpRatio: C.warpW / Math.max(C.warpS, 1),
      msRatio: meas ? meas[0] / meas[1] : NaN,
      // Straddle work's share of windfoil lane ALU, and the arithmetic estimate of what band-clipped
      // hulls would save (false-left straddles → far-left skip, false-right → cheap far-right add).
      straddleAluPct: 100 * C.straddle * COST.straddle / Math.max(alu.w, 1),
      bandClipSavePct: 100 * ((C.falseLeft + C.falseNone) * (COST.straddle - COST.farLeftSkip) +
        C.falseRight * (COST.straddle - COST.farRight)) / Math.max(alu.w, 1),
      degPerStraddle: 100 * C.degStraddle / Math.max(C.straddle, 1),
      earlyPerStraddle: C.rootEarly / Math.max(C.straddle, 1),
      linearPerStraddle: C.rootLinear / Math.max(C.straddle, 1),
      C,
    };
    // Divergence + utilization factor: serialized warp issue vs a perfectly packed, divergence-free warp
    // doing the same lane-summed work (laneALU / 32 issue slots). >1 = branch serialization, loop tails,
    // and empty lanes in partial tiles.
    row.wDivF = C.warpW / Math.max(alu.w / 32, 1e-9);
    row.sDivF = C.warpS / Math.max(alu.s / 32, 1e-9);
    rows.push(row);
    console.error(`${row.key} done in ${((performance.now() - t0) / 1000).toFixed(1)}s  (${row.sampleNote})`);
  }
}

// ── print ─────────────────────────────────────────────────────────────────────────────────────────────
const fmt = (v, d = 2) => Number.isFinite(v) ? v.toFixed(d) : 'n/a';
const pad = (s, n) => String(s).padStart(n);
console.log('\nscene@px      frags  sample                 b/f   vis/f   str/f  fstr%  (->L%  ->R%)  sqrt/str  cov%  slugSolv/f  wALU/f  sALU/f  laneR  warpR  msw/s  guard%  lanes/warp');
for (const r of rows) {
  console.log(
    pad(r.key, 11) + pad((r.totalFrags / 1000).toFixed(0) + 'k', 8) + '  ' + r.sampleNote.padEnd(20) +
    pad(fmt(r.bandsPerFrag), 6) + pad(fmt(r.visitsPerFrag, 1), 8) + pad(fmt(r.straddlesPerFrag), 8) +
    pad(fmt(r.falseStraddlePct, 1), 7) + pad(fmt(r.falseLeftPct, 1), 7) + pad(fmt(r.falseRightPct, 1), 6) +
    pad(fmt(r.sqrtPerStraddle), 10) + pad(fmt(r.coveredPct, 1), 6) +
    pad(fmt(r.slugSolvesPerFrag), 12) + pad(fmt(r.wAluPerFrag, 0), 8) + pad(fmt(r.sAluPerFrag, 0), 8) +
    pad(fmt(r.aluRatio), 7) + pad(fmt(r.warpRatio), 7) + pad(fmt(r.msRatio), 7) + pad(fmt(r.guardPct, 1), 8) +
    pad(fmt(r.lanesPerWarp, 1), 12),
  );
}
console.log('\nper-frag detail (windfoil): farL/f farR/f ySkip/f ulp/f brkSkip/f wOcc | slug: vis/f code0/f brkSkip/f bandCurves/f');
for (const r of rows) {
  console.log(
    pad(r.key, 11) +
    pad(fmt(r.farLeftPerFrag, 1), 8) + pad(fmt(r.farRightPerFrag, 1), 8) + pad(fmt(r.ySkipPerFrag, 1), 8) +
    pad(fmt(r.ulpPerFrag, 2), 7) + pad(fmt(r.breakSkippedPerFrag, 1), 10) + pad(fmt(r.wBandOcc, 1), 6) + '   |' +
    pad(fmt(r.slugVisitedPerFrag, 1), 8) + pad(fmt(r.sCode0PerFrag, 1), 8) + pad(fmt(r.sBreakSkippedPerFrag, 1), 10) +
    pad(fmt(r.slugBandCurvesPerFrag, 1), 13),
  );
}
console.log('\nstraddle economics: straddle share of windfoil lane ALU, band-clip savings estimate, root kinds:');
console.log('scene@px     stradALU%  clipSave%  deg/str%  early/str  lin/str  sqrt/str');
for (const r of rows) {
  console.log(
    pad(r.key, 11) + pad(fmt(r.straddleAluPct, 1), 10) + pad(fmt(r.bandClipSavePct, 1), 11) +
    pad(fmt(r.degPerStraddle, 1), 10) + pad(fmt(r.earlyPerStraddle), 11) + pad(fmt(r.linearPerStraddle, 3), 9) +
    pad(fmt(r.sqrtPerStraddle), 10),
  );
}
console.log('\ntotal estimated ALU per frame — lane model (each lane pays its own path) and warp model (32-lane');
console.log('SIMD lockstep: branch classes serialize, loops run to the max lane, partial tiles pay full issue):');
console.log('scene@px       laneW(M)  laneS(M)  laneR |  warpW(M)  warpS(M)  warpR |  msw/s   wDiv   sDiv');
for (const r of rows) {
  console.log(
    pad(r.key, 11) + pad((r.wAluTotal / 1e6).toFixed(1), 10) + pad((r.sAluTotal / 1e6).toFixed(1), 10) +
    pad(fmt(r.aluRatio), 7) + ' |' + pad((r.wWarpTotal / 1e6).toFixed(1), 10) + pad((r.sWarpTotal / 1e6).toFixed(1), 10) +
    pad(fmt(r.warpRatio), 7) + ' |' + pad(fmt(r.msRatio), 7) + pad(fmt(r.wDivF), 7) + pad(fmt(r.sDivF), 7),
  );
}
