// validate-shadow-kernel.js — check the analytic disc-kernel coverage (src/kernel-coverage.js) against an
// INDEPENDENT ground truth: stratified point sampling of the same silhouette inside the same disc (a different
// code path — ray-cast winding, no shared math with the boundary integral). If the analytic soft shadow
// matches the sampled one, the penumbra is right.
//
//   deno run -A tools/validate-shadow-kernel.js     (or node, with deps installed)

import { buildPieceGrid, discCoverage, discCoverageSampled } from '../src/kernel-coverage.js';

const isDeno = typeof Deno !== 'undefined';
const exit = (c) => (isDeno ? Deno.exit(c) : process.exit(c));
let failures = 0;
const check = (name, ok, detail = '') => { console.log(`  ${ok ? '✓' : '✗'} ${name}${detail ? `  ${detail}` : ''}`); if (!ok) failures++; };

const poly = (pts) => { const q = []; for (let i = 0; i < pts.length; i++) { const [a, b] = pts[i], [c, d] = pts[(i + 1) % pts.length]; q.push(a, b, (a + c) / 2, (b + d) / 2, c, d); } return q; };
const circle = (cx, cy, r, n) => { const p = []; for (let i = 0; i < n; i++) { const a = i / n * 2 * Math.PI; p.push([cx + r * Math.cos(a), cy + r * Math.sin(a)]); } return poly(p); };

// Compare against the mean of several independent high-resolution sample sets (averaging drives the sampler's
// own noise down so a real analytic error would stand out; a matching pair confirms the boundary integral).
function agree(name, pieces, tests, tol) {
  const grid = buildPieceGrid(pieces, 20);
  let e = 0, worst = null;
  for (const [px, py, rx, ry] of tests) {
    const a = discCoverage(grid, px, py, rx, ry ?? rx);
    let s = 0;
    for (let k = 0; k < 4; k++) s += discCoverageSampled(pieces, px, py, rx, ry ?? rx, 60 + k, 120 + 2 * k);
    s /= 4;
    if (Math.abs(a - s) > e) { e = Math.abs(a - s); worst = [px, py, rx, ry ?? rx, +a.toFixed(4), +s.toFixed(4)]; }
  }
  check(name, e < tol, `maxΔ=${e.toExponential(2)}  worst(x,y,rx,ry,ana,samp)=${JSON.stringify(worst)}`);
}

console.log('analytic disc-kernel coverage vs independent point-sampling (the soft shadow is correct):');
// a straight edge → the disc CDF S-curve (the ideal penumbra), round and stretched radii
const edge = poly([[-800, -800], [0, -800], [0, 800], [-800, 800]]);
const tE = [];
for (let d = -60; d <= 60; d += 3) for (const [rx, ry] of [[8, 8], [24, 24], [30, 50]]) tE.push([d, 0, rx, ry]);
agree("edge → disc S-curve penumbra", edge, tE, 1.2e-2);
// a convex rim
const circ = circle(0, 0, 160, 256);
const tC = [];
for (let d = -60; d <= 60; d += 4) for (const [rx, ry] of [[10, 10], [30, 30], [24, 44]]) tC.push([160 + d, 0, rx, ry]);
agree("circle rim penumbra", circ, tC, 1.5e-2);

// interior/exterior saturate, and a tiny gap dims toward its area average (sub-penumbra detail washes out)
const g = buildPieceGrid(circ, 20);
check('deep interior coverage ≈ 1', Math.abs(discCoverage(g, 0, 0, 20) - 1) < 1e-6, `= ${discCoverage(g, 0, 0, 20).toFixed(4)}`);
check('far exterior coverage ≈ 0', discCoverage(g, 400, 0, 20) < 1e-6, `= ${discCoverage(g, 400, 0, 20).toFixed(4)}`);

console.log(`\n${failures === 0 ? 'all checks passed' : `${failures} check(s) FAILED`}`);
exit(failures === 0 ? 0 : 1);
