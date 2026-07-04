// kernel-tune.js — pick a kernel's quadrature budget (glOrder × nSub × split lists): worst |ΔF| per piece
// over random + adversarial xy-monotone pieces, f64 CPU replica, against the same structure at GL-8 × 32
// slices (effectively exact). This is how the shipped configs in src/kernels.js were chosen; run it when
// adding a kernel or questioning a budget.  (deno run -A tools/kernel-tune.js [trials])
//
// The adversarial families matter more than the random pool: steep near-vertical pieces sweep the kernel's
// whole y-profile in a sliver of t (the case that made a knotless GL-5 Gaussian err by 1e-1), shallow ones do
// the same in x, and full-support diagonals stress both. Scale reference: full-pixel coverage is 1, one 8-bit
// output level is 3.9e-3, and the shipped configs sit at 1e-6-ish (exact kernels at f64 epsilon).

import { cpuReplica } from './kernel-validate.js';
import { resolveKernel } from '../src/kernels.js';

let seed = 12345;
const rnd = () => (seed = (seed * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff;

function randPiece(R) {
  const E = R + 1;
  const q1 = [rnd() * 2 * E - E, rnd() * 2 * E - E];
  const q3 = [rnd() * 2 * E - E, rnd() * 2 * E - E];
  const q2 = [q1[0] + rnd() * (q3[0] - q1[0]), q1[1] + rnd() * (q3[1] - q1[1])];
  return [q1[0], q1[1], q2[0], q2[1], q3[0], q3[1]];
}
function* adversarial(R) {
  const E = R + 0.2;
  for (let i = 0; i <= 60; i++) {
    const u = -R - 0.1 + (i / 60) * (2 * R + 0.2);
    yield [u, -E, u + 0.02 * (rnd() - 0.5), rnd() * 0.2 - 0.1, u + 0.01, E]; // steep
    yield [-E, u, rnd() * 0.2 - 0.1, u + 0.02 * (rnd() - 0.5), E, u + 0.01]; // shallow
    yield [-E, -E, (rnd() - 0.5) * R, (rnd() - 0.5) * R, E, E]; // diagonal
    yield [-E, E, (rnd() - 0.5) * R, (rnd() - 0.5) * R, E, -E]; // anti-diagonal
  }
}

function evalPiece(spec, piece) {
  const curves = new Float64Array(piece);
  const rows = new Uint32Array([0, 1, 0, 0, 0]);
  const lo = Math.min(piece[1], piece[5]) - 1;
  const header = { rowBase: 0, bandCount: 1, y0: lo, invH: 0 };
  return cpuReplica(spec, curves, rows, header, [0, 0], [1, 1]);
}

function worst(spec, ref, R, trials) {
  let maxErr = 0;
  const test = (piece) => {
    const p = [...piece]; // hull-monotonize: control clamped between endpoints per axis
    p[2] = Math.min(Math.max(p[2], Math.min(p[0], p[4])), Math.max(p[0], p[4]));
    p[3] = Math.min(Math.max(p[3], Math.min(p[1], p[5])), Math.max(p[1], p[5]));
    const e = Math.abs(evalPiece(spec, p) - evalPiece(ref, p));
    if (e > maxErr) maxErr = e;
  };
  for (let i = 0; i < trials; i++) test(randPiece(R));
  for (const p of adversarial(R)) test(p);
  return maxErr;
}

const TRIALS = Number(Deno.args[0] ?? 4000);
// The shipped configs, plus the runner-up budgets each one beat — kept so a rerun shows the whole trade.
const CASES = [
  ['box-ext', [{}]],
  ['tent', [{}]],
  ['mblur=8', [{}]],
  ['mblur=1', [{}]],
  ['gaussian', [{}, { glOrder: 5, nSub: 3 }, { glOrder: 5, nSub: 1 }]],
  ['mitchell', [{}, { xSplits: [-1, 1] }, { yKnots: [-1, 1], xSplits: [-1, 1] }]],
  ['bspline', [{}]],
  ['catmullrom', [{}]],
  ['disc=1.5', [{}, { nSub: 3 }]],
  ['disc=3', [{}]],
  ['iris=1.5', [{}]],
  ['iris=3', [{}]],
  ['iris=3,5,18', [{}]],
  ['iris=3,3', [{}]],
];

console.log(`kernel-tune · worst |ΔF| per piece vs GL-8×32 reference · ${TRIALS} random + 244 adversarial pieces`);
for (const [name, overrides] of CASES) {
  const base = resolveKernel(name);
  const ref = { ...base, glOrder: 8, nSub: 32 };
  for (const over of overrides) {
    const spec = { ...base, ...over };
    const tag = `y{${(spec.yKnots ?? []).join(',')}} x{${(spec.xSplits ?? []).join(',')}} GL${spec.glOrder}×${spec.nSub ?? 1}`;
    const shipped = Object.keys(over).length === 0 ? '  ← shipped' : '';
    console.log(`${name.padEnd(11)} ${tag.padEnd(36)} ${worst(spec, ref, Math.max(spec.rx, spec.ry), TRIALS).toExponential(1)}${shipped}`);
  }
}
