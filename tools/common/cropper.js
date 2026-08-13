// cropper.js — the magnified detail window, shared by `deno task crop` and `deno task report`.
//
// The zoom comes from the window being small, not from the output being big:
//
//     source 512×512  ──×2──▶  1024×1024 (conceptually)  ──window 512×512 at 40px inset──▶  out 512×512
//
// so at zoom 2 the window covers a 256×256 region of the source — the upper-right quadrant — and at zoom 4 a
// 128×128 region. Output matches the source size, so a crop drops into a figure beside the full renders with
// nothing rescaled. There is only ever ONE resampling step: `upscaleCrop` reads just the source pixels the
// window needs and writes them magnified, so the full upscale is never built and nothing is downscaled.
// Because the window is written 1:1, `inset` is the same count in the upscaled image and in the output.

import { upscaleCrop } from './images.js';

/**
 * Where the window lands, in upscaled-image coordinates.
 * @returns {{cx, cy, cw, ch, W, H, sx, sy, sw, sh} | {tooSmall: true, ...}}
 */
export function cropWindow(w, h, { zoom = 2, outSize = 0, inset = 40, corner = 'tr' } = {}) {
  const W = w * zoom, H = h * zoom;
  const cw = outSize > 0 ? outSize : w;
  const ch = outSize > 0 ? outSize : h;
  if (cw + inset > W || ch + inset > H) return { tooSmall: true, W, H, cw, ch };
  const cx = corner === 'tr' || corner === 'br' ? W - cw - inset : inset;
  const cy = corner === 'tr' || corner === 'tl' ? inset : H - ch - inset;
  // the source region the window covers — what says which part of the picture a figure is showing
  return { cx, cy, cw, ch, W, H, sx: cx / zoom, sy: cy / zoom, sw: cw / zoom, sh: ch / zoom };
}

/** Crop an RGBA8 image. `img` is {data, w, h} from imageio.loadRGBA. */
export function cropRaster(img, opts) {
  const win = cropWindow(img.w, img.h, opts);
  if (win.tooSmall) return win;
  return { ...win, rgba: upscaleCrop(img.data, img.w, img.h, opts.zoom ?? 2, win.cx, win.cy, win.cw, win.ch) };
}

/**
 * The SAME window applied to an SVG, as a viewBox rather than pixels — so rendering the result in Figma or
 * Photoshop at the output size gives the region the raster crops show, and the comparison survives cropping.
 * Without this the SVG would still cover the whole frame and could not be placed beside the crops.
 */
export function cropSVG(src, opts) {
  const root = src.match(/<svg\b[^>]*>/i)?.[0];
  if (!root) return null;
  const attr = (n) => root.match(new RegExp(`\\b${n}\\s*=\\s*"([^"]*)"`, 'i'))?.[1];
  const vb = (attr('viewBox')?.match(/[+-]?(?:\d*\.\d+|\d+\.?)(?:[eE][+-]?\d+)?/g) ?? []).map(Number);
  const w = parseFloat(attr('width')) || (vb.length === 4 ? vb[2] : 0);
  const h = parseFloat(attr('height')) || (vb.length === 4 ? vb[3] : 0);
  if (!(w > 0 && h > 0)) return null;

  const win = cropWindow(w, h, opts);
  if (win.tooSmall) return win;

  // window → document px → viewBox units (they differ if width/height disagree with the viewBox)
  const kx = vb.length === 4 ? vb[2] / w : 1, ky = vb.length === 4 ? vb[3] / h : 1;
  const vx = (vb.length === 4 ? vb[0] : 0) + win.sx * kx;
  const vy = (vb.length === 4 ? vb[1] : 0) + win.sy * ky;
  const vw = win.sw * kx, vh = win.sh * ky;
  const num = (n) => (Number.isInteger(n) ? `${n}` : `${+n.toFixed(4)}`);

  let out = root
    .replace(/\bviewBox\s*=\s*"[^"]*"/i, `viewBox="${num(vx)} ${num(vy)} ${num(vw)} ${num(vh)}"`)
    .replace(/\bwidth\s*=\s*"[^"]*"/i, `width="${num(win.cw)}"`)
    .replace(/\bheight\s*=\s*"[^"]*"/i, `height="${num(win.ch)}"`);
  if (!/viewBox/i.test(out)) {
    out = out.replace(/<svg\b/i, `<svg viewBox="${num(vx)} ${num(vy)} ${num(vw)} ${num(vh)}"`);
  }
  return { ...win, svg: src.replace(root, out) };
}
