// renderers.js — every way this repo can turn a shape into coverage, as ONE list, plus the two things every
// tool then does with it: run them all over a shape, and pool the results across many shapes.
//
// tools/comparison.js renders one scene big and writes a PNG per entry. tools/validate.js renders a whole
// dataset small and prints a number per entry. They used to keep separate lists of the same five renderers,
// which is how a tool ends up measuring something subtly different from what the other one pictures. One
// list, one order, one set of names: a row in the validation table and a file in a comparison folder are now
// the same measurement, and adding a renderer adds it to both.
//
// Every entry is { name, title, run(quads, opts) } and every `run` returns the shared coverage buffer
// (./coverage.js), so nothing downstream is per-renderer. Two extra marks:
//   reference — what every other entry's error is measured against. Exactly one entry has it, and it is
//               first, so a `derive` entry can be built from it.
//   control   — not a renderer being judged but a row that calibrates the judging (see `controls` below).
//               Tools that draw pictures skip these; tools that print numbers want them.

import { canvasCoverage, pointCoverage, quantize8, slugCoverage, stats, windfoilCoverage } from './coverage.js';

/**
 * @param {object} o
 * @param {GPUDevice} [o.device]           shared WebGPU device for the two shader renderers
 * @param {Function} [o.createContext2D]   (w, h) → a 2D canvas context in this host
 * @param {number} [o.samples]             the truth grid: samples² points per pixel
 * @param {boolean} [o.exact]              render windfoil with the shader's EXACT_MODE override
 * @param {boolean} [o.controls]           include the control rows
 * @param {{name?: string, title?: string}} [o.canvas]  rename the host-canvas entry (the browser boot calls
 *   it `canvas`, because there it is whatever the engine ships — Skia in Chrome, CoreGraphics in Safari)
 */
export function buildRenderers(
  { device, createContext2D, samples = 64, exact = false, controls = false, canvas = {} } = {},
) {
  const half = Math.max(1, samples >> 1);
  return [
    {
      name: 'truth',
      title: `ideal box filter, ${samples}×${samples} point-sampled`,
      reference: true, // the one entry everything else is measured against
      run: (quads, o) => pointCoverage(quads, { ...o, samples }),
    },
    // ── controls: rows that answer "what does this table's own precision allow it to say?" ───────────────
    // They cost one extra point-sample and one map, and between them they bound the two floors under every
    // other number: how far the reference itself has converged, and what the readback quantises away. A
    // renderer's number only means something once it is read against these two.
    ...(controls
      ? [
        {
          name: 'truth/2',
          title: `the same at ${half}×${half} — has the reference converged?`,
          control: true,
          // Halving the grid measures the reference against a coarser copy of itself. Under the ~O(1/F²)
          // convergence of the sample, this over-states truth's own residual by ~3× — so it is an upper
          // bound on how much of any row below is the reference rather than the renderer.
          run: (quads, o) => pointCoverage(quads, { ...o, samples: half }),
        },
        {
          name: '8-bit',
          title: 'truth through an 8-bit readback — the floor',
          control: true,
          // The exact box filter, quantised the way an RGBA8 readback quantises every renderer below. This
          // is the score a PERFECT renderer gets on this table: no 8-bit row can go under it, and a row
          // that sits ON it is exact as far as an 8-bit image can tell.
          derive: (reference) => quantize8(reference),
        },
      ]
      : []),
    {
      name: 'binary',
      title: 'winding at the pixel centre, no AA at all',
      control: controls, // in a numbers table it is the top of the scale; in a picture it is a renderer
      run: (quads, o) => pointCoverage(quads, { ...o, samples: 1 }),
    },
    // ── the renderers under test ─────────────────────────────────────────────────────────────────────────
    {
      name: 'windfoil',
      title: `src/windfoil.wgsl${exact ? ' · EXACT_MODE' : ''}`,
      run: (quads, o) => windfoilCoverage(quads, { ...o, device, exact }),
    },
    {
      name: 'slug',
      title: 'bench/slug.wgsl',
      run: (quads, o) => slugCoverage(quads, { ...o, device }),
    },
    {
      // `skia`, matching tools/comparison.js — the two tools' tables end up side by side, and a renderer that
      // changes name between them is one a reader has to reconcile. Named for the library rather than the
      // browser because that is what is being measured: Skia through Node bindings, not Chrome.
      name: canvas.name ?? 'skia',
      title: canvas.title ?? '@napi-rs/canvas (Skia)',
      run: (quads, o) => canvasCoverage(createContext2D, quads, o),
    },
  ];
}

/**
 * Run every renderer over one shape, in list order, and measure each against the reference.
 *
 * @param {Array} list           from `buildRenderers`
 * @param {number[]} quads       flat quadratics (./shapes.js)
 * @param {object} opts          { size, evenodd, segments } — passed to every renderer untouched, so an
 *                               option only some of them understand (`segments`: stroke these centerlines
 *                               instead of filling) needs no special case here.
 * @returns {Promise<Array>}     the list, each entry with { cov, stats, ms } added (stats null on reference)
 */
export async function renderAll(list, quads, opts) {
  const out = [];
  for (const r of list) {
    const t0 = performance.now();
    // `derive` entries are built from the reference rather than rendered — it is first in the list, so it is
    // already in `out` by the time one of them is reached.
    const cov = r.derive ? r.derive(out.find((x) => x.reference).cov) : await r.run(quads, opts);
    out.push({ ...r, cov, ms: performance.now() - t0 });
  }
  const reference = out.find((r) => r.reference).cov;
  for (const r of out) r.stats = r.reference ? null : stats(r.cov, reference);
  return out;
}

/**
 * Pool per-shape stats into one row per renderer — the whole point of the suite being a suite. Band means and
 * bias are pooled over PIXELS, not averaged over shapes, so a shape with 3000 band pixels counts thirty times
 * a shape with 100 rather than equally; whole-image means are averaged over shapes (every shape is the same
 * cell here, so that is the pooled mean too); max and `off` are a max and a total.
 */
export function aggregate(list) {
  const acc = list.map((r) => ({
    name: r.name, title: r.title, control: !!r.control,
    n: 0, sum: 0, bandSum: 0, biasSum: 0, band: 0, max: 0, off: 0,
  }));
  let shapes = 0;
  return {
    /** How many shapes have been pooled — what the aggregate's "(n shapes)" caption reports. */
    get n() {
      return shapes;
    },
    add(results) {
      shapes++;
      results.forEach((r, i) => {
        if (!r.stats) return;
        const a = acc[i];
        a.n++;
        a.sum += r.stats.mean;
        a.bandSum += r.stats.bandSum;
        a.biasSum += r.stats.bias * r.stats.band;
        a.band += r.stats.band;
        a.off += r.stats.off;
        if (r.stats.max > a.max) a.max = r.stats.max;
      });
    },
    /** One row per measured renderer: { name, title, control, bandMean, band, bias, mean, max, off }. */
    rows: () =>
      acc.filter((a) => a.n).map((a) => ({
        name: a.name,
        title: a.title,
        control: a.control,
        bandMean: a.band ? a.bandSum / a.band : 0,
        band: a.band, // how many anti-aliased pixels the band mean is over
        bias: a.band ? a.biasSum / a.band : 0,
        mean: a.sum / a.n,
        max: a.max,
        off: a.off,
      })),
  };
}
