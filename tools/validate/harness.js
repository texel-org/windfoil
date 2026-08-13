// harness.js — the environment-agnostic core of the validation suite: the shape dataset (../common/shapes.js)
// crossed with the renderer list (../common/renderers.js), at one fixed cell size and one fixed reference
// grid. The same core runs under Deno (`deno task validate`, via ../validate.js) and in a browser (`deno task
// serve`, then /tools/validate/); each boot supplies only what differs between the two hosts — a 2D-canvas
// context factory and the WebGPU device — and decides how to present what comes back.
//
// The suite is a table of numbers, so it runs with the control rows on: `truth/2` (the reference measured
// against a coarser copy of itself) and `8-bit` (the reference through the same readback quantisation every
// renderer goes through) bound what this table's own precision can certify, and every renderer's number is
// only meaningful read against them. See ../common/renderers.js.

import { buildRenderers, renderAll } from '../common/renderers.js';
import { buildShapes, CELL } from '../common/shapes.js';

export const S = CELL; // cell size in px

// The reference grid: F² sample points per pixel. Not a taste setting — it is the smallest grid at which the
// `truth/2` control comes back clean, i.e. at which HALVING the grid no longer moves a single pixel of the
// dataset by more than one 8-bit code value. Below it (F=192 vs 96 leaves ~800 such pixels) part of what the
// table reports as a renderer's error is really the reference's own; at F=384 the reference has converged
// past the precision anything here is compared at, and the control row in every block says so.
export const F = 384;

export { aggregate } from '../common/renderers.js';
export { AMP, diffRGBA, grayRGBA } from '../common/images.js';

/**
 * @param {object} o
 * @param {object} o.font                a parsed font (src/font.js loadFont/parseFont)
 * @param {Function} o.createContext2D   (w, h) → a 2D canvas context in this host
 * @param {GPUDevice} o.device           a shared WebGPU device (src/gpu.js requestDevice)
 * @param {number} [o.samples]           override the reference grid (F² samples per pixel)
 * @param {boolean} [o.exact]            render windfoil with the shader's EXACT_MODE override (see
 *   ../common/coverage.js windfoilCoverage). Only windfoil — everything else is the yardstick.
 * @param {{name?: string, title?: string}} [o.canvas]  rename the host-canvas entry for this host
 * @returns {{ renderers: Array, shapes: Array, render: (shape) => Promise<Array> }}
 */
export function validationSuite({ font, createContext2D, device, samples = F, exact = false, canvas }) {
  const renderers = buildRenderers({ device, createContext2D, samples, exact, controls: true, canvas });
  return {
    renderers,
    shapes: buildShapes(font),
    /** One shape through every renderer, each measured against the reference. */
    render: ({ quads, evenodd = false, segments }) =>
      renderAll(renderers, quads, { size: S, evenodd, segments }),
  };
}
