// validate-blur.js — check the soft-shadow footprint widening against a closed form, with no GPU.
//
// Box-filtering a step edge with a box of width w gives a LINEAR coverage ramp of width w centred on the edge:
//   coverage(x) = clamp(0.5 − x/w, 0, 1)         (edge at x = 0, filled side x < 0)
// windfoil's blur just widens that box to w = s·(1 + blurPx), so the penumbra width must track (1 + blurPx)
// exactly. We evaluate the CPU port (src/cpu-coverage.js, the faithful twin of the WGSL fragment) on a shape
// with one vertical edge and compare. Also sanity-checks the generic leaf atlas.
//
//   deno run -A tools/validate-blur.js         (or `node tools/validate-blur.js` with deps installed)

import { coverageAt } from '../src/cpu-coverage.js';
import { buildShapeAtlas, makeLeaf, mulberry32 } from '../src/shapes.js';

const isDeno = typeof Deno !== 'undefined';
const exit = (code) => (isDeno ? Deno.exit(code) : process.exit(code));

let failures = 0;
function check(name, ok, detail = '') {
  console.log(`  ${ok ? '✓' : '✗'} ${name}${detail ? `  ${detail}` : ''}`);
  if (!ok) failures++;
}

// A tall rectangle filling x ∈ [−50, 0]: its only near-horizon edge at y≈0 is the vertical one at x = 0.
function rectRightEdge() {
  const P = [[-50, -50], [0, -50], [0, 50], [-50, 50]];
  const quads = [];
  for (let i = 0; i < 4; i++) {
    const [x0, y0] = P[i], [x1, y1] = P[(i + 1) % 4];
    quads.push(x0, y0, (x0 + x1) / 2, (y0 + y1) / 2, x1, y1);
  }
  return quads;
}

const { curves, rows, table } = buildShapeAtlas([rectRightEdge()]);
const band = table[0];
const bboxMin = [table[0].bbox[0], table[0].bbox[1]];
const s = [1, 1]; // 1 unit per device px

function rampError(blurPx) {
  const w = 1 + blurPx;
  const blur = [blurPx, 0, 0, blurPx]; // constant penumbra of diameter blurPx (px)
  let maxErr = 0;
  for (let x = -3; x <= 3; x += 0.05) {
    const cov = coverageAt(curves, rows, band, bboxMin, x, 0, s, blur);
    const want = Math.min(Math.max(0.5 - x / w, 0), 1);
    maxErr = Math.max(maxErr, Math.abs(cov - want));
  }
  return maxErr;
}

console.log('blur widens the penumbra ramp exactly (edge → linear ramp of width 1 + blurPx):');
for (const b of [0, 1, 3, 8, 24]) {
  const err = rampError(b);
  check(`blurPx = ${b} → ramp width ${1 + b}`, err < 2e-3, `max|Δ| = ${err.toExponential(2)}`);
}

// blur == 0 must be the exact 1px box filter: coverage 1 half a pixel inside, 0 half a pixel outside, 0.5 on it.
console.log('\nblur == 0 is the exact 1px box filter:');
const c0 = coverageAt(curves, rows, band, bboxMin, 0, 0, s, [0, 0, 0, 0]);
const cIn = coverageAt(curves, rows, band, bboxMin, -0.5, 0, s, [0, 0, 0, 0]);
const cOut = coverageAt(curves, rows, band, bboxMin, 0.5, 0, s, [0, 0, 0, 0]);
check('coverage on the edge ≈ 0.5', Math.abs(c0 - 0.5) < 1e-4, `= ${c0.toFixed(6)}`);
check('coverage half a px inside ≈ 1', Math.abs(cIn - 1) < 1e-4, `= ${cIn.toFixed(6)}`);
check('coverage half a px outside ≈ 0', Math.abs(cOut - 0) < 1e-4, `= ${cOut.toFixed(6)}`);

// A wide blur conserves the edge's mean position (box blur is symmetric): coverage(−a) + coverage(+a) ≈ 1.
console.log('\nbox blur is symmetric about the edge (mean-preserving):');
{
  const blur = [12, 0, 0, 12];
  let maxAsym = 0;
  for (let a = 0; a <= 6; a += 0.1) {
    const lo = coverageAt(curves, rows, band, bboxMin, -a, 0, s, blur);
    const hi = coverageAt(curves, rows, band, bboxMin, a, 0, s, blur);
    maxAsym = Math.max(maxAsym, Math.abs(lo + hi - 1));
  }
  check('coverage(−a) + coverage(+a) ≈ 1', maxAsym < 2e-3, `max|Δ| = ${maxAsym.toExponential(2)}`);
}

// Generic leaf atlas is well-formed (finite curves + row table, non-degenerate bbox and band areas).
console.log('\nleaf atlas is well-formed:');
{
  const rng = mulberry32(1234);
  const leaves = Array.from({ length: 8 }, (_, i) => makeLeaf({ rng, angle: i }));
  const atlas = buildShapeAtlas(leaves);
  const curvesFinite = atlas.curves.every(Number.isFinite);
  const rowsOk = atlas.rows.length % 5 === 0 && atlas.rows.length > 0;
  const bboxOk = atlas.table.every((t) => {
    const [x0, y0, x1, y1] = t.bbox;
    return Number.isFinite(x0) && x1 > x0 && y1 > y0;
  });
  check('curve atlas all finite', curvesFinite, `${atlas.curves.length / 6} pieces`);
  check('row table is a multiple of 5 (start,count,area,xMin,xMax)', rowsOk, `${atlas.rows.length / 5} bands`);
  check('every leaf has a non-degenerate bbox', bboxOk);
}

console.log(`\n${failures === 0 ? 'all checks passed' : `${failures} check(s) FAILED`}`);
exit(failures === 0 ? 0 : 1);
