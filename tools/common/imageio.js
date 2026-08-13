// imageio.js — reading rendered images back off disk, for the tools that compare renders they did not make.
//
// Everything else in tools/common works on float coverage produced in-process. These helpers cover the other
// direction: an image that arrived as a file — a Figma or Photoshop export, a browser screenshot, a previous
// run — which has necessarily been through an 8-bit PNG and so is quantised to 1/255 steps.

import { createCanvas, loadImage } from '@napi-rs/canvas';
import { encodePNG } from '../../src/png.js';

/** Decode an image file to RGBA8, compositing any alpha over black (the coverage-render convention). */
export async function loadRGBA(path) {
  const img = await loadImage(path);
  const w = img.width, h = img.height;
  const ctx = createCanvas(w, h).getContext('2d');
  ctx.fillStyle = '#000';
  ctx.fillRect(0, 0, w, h);
  ctx.drawImage(img, 0, 0);
  return { data: ctx.getImageData(0, 0, w, h).data, w, h, path };
}

/** The coverage plane of an RGBA8 buffer: white-on-black renders carry it in every channel, so red will do. */
export function coverageOf({ data, w, h }) {
  const out = new Float64Array(w * h);
  for (let i = 0; i < out.length; i++) out[i] = data[i * 4] / 255;
  return out;
}

/**
 * Per-pixel difference between two RGBA8 buffers as the LARGEST of the three channel differences, so a
 * discrepancy in any one channel shows at full strength instead of being diluted by the two that agree.
 * Identical to the channel value for the grayscale coverage images these tools produce.
 */
export function deltaOf(a, b) {
  const n = a.w * a.h;
  const out = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const o = i * 4;
    out[i] = Math.max(
      Math.abs(a.data[o] - b.data[o]),
      Math.abs(a.data[o + 1] - b.data[o + 1]),
      Math.abs(a.data[o + 2] - b.data[o + 2]),
    ) / 255;
  }
  return out;
}

export const writePNGFile = (path, rgba, w, h) => Deno.writeFileSync(path, encodePNG(rgba, w, h));
