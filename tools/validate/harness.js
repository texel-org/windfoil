// harness.js — the environment-agnostic core of the validation suite: it walks the shared shape dataset
// (../common/shapes.js), renders every shape through all four coverage sources (../common/coverage.js) and
// yields the images plus the stats. The same code runs under Deno (`deno task validate`, via ../validate.js)
// and in a browser (`deno task serve`, then /tools/validate/); each boot supplies only what differs between
// the two hosts — a 2D-canvas context factory and the WebGPU device.
//
// The four coverage sources per shape (all white-on-black, so the stored byte is 255·coverage, linear):
//   • ours   — the windfoil shader.
//   • slug   — the benchmark's Slug port (bench/slug.wgsl), the other analytic AA model.
//   • canvas — the host's 2D canvas rasterizer: @napi-rs/canvas (Skia) under Deno; whatever the engine
//              uses in a browser (Skia in Chrome, CoreGraphics in Safari, WebRender in Firefox).
//   • box    — the mathematical box filter, point-sampled F×F from the raw curves. This is the true filter,
//              but its own per-pixel noise is ~1/F, so it validates the shape of the error and any bias, not
//              sub-1e-3 precision.
//
// We report mean |Δcoverage| for each renderer vs the box reference. If "ours vs box" and "canvas vs box"
// are close, both track the box filter equally (the residual is the point-sample noise, not our error).

import { canvasCoverage, pointCoverage, slugCoverage, stats, windfoilCoverage } from '../common/coverage.js';
import { buildShapes, CELL } from '../common/shapes.js';

export const S = CELL; // cell size in px
export const F = 24; // point-sample grid per pixel for the box-filter reference

export { ALPHABET, buildShapes } from '../common/shapes.js';
export { AMP, diffRGBA, grayRGBA, slugify as slug } from '../common/images.js';
export { stats };

/**
 * Run the whole suite, yielding one result per shape as it completes — so a boot can stream rows into a
 * console table (Deno) or the page (browser) while later shapes are still rendering.
 *
 * @param {object} o
 * @param {object} o.font              a parsed font (see font.js loadFont/parseFont)
 * @param {Function} o.createContext2D (w, h) → a 2D canvas context in the host environment
 * @param {GPUDevice} o.device         a shared WebGPU device (see gpu.js requestDevice)
 * @param {boolean} [o.exact]          render ours with the shader's EXACT_MODE override (see coverage.js
 *                                     windfoilCoverage); slug, canvas and box are unaffected — they're the
 *                                     yardstick, so exact mode applies only to ours.
 */
export async function* validateShapes({ font, createContext2D, device, exact = false }) {
  for (const { label, quads, evenodd = false, segments, fold = false } of buildShapes(font)) {
    const ours = await windfoilCoverage(quads, { size: S, evenodd, device, exact });
    const slug = await slugCoverage(quads, { size: S, evenodd, device });
    const canvas = canvasCoverage(createContext2D, quads, { size: S, evenodd, segments });
    const box = pointCoverage(quads, { size: S, evenodd, samples: F });
    yield {
      label, fold, ours, slug, canvas, box,
      oursVsBox: stats(ours, box), slugVsBox: stats(slug, box), canvasVsBox: stats(canvas, box),
    };
  }
}
