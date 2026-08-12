// images.js — turning coverage buffers into pixels, shared by every tool that writes or shows one.
//
// A "coverage buffer" throughout the tools is a Float64Array of width*height values in 0..1, linear (NOT
// gamma-encoded): 1 = fully inside the shape. Every renderer in ./coverage.js produces one, so all of them
// are directly comparable and all of them display the same way.

/** Default error-map gain: |Δ|·AMP, so a coverage difference of 1/AMP already reads full-bright. */
export const AMP = 15;

/** Tints for `diffRGBA`. HEAT reads as "error" next to grey coverage; WHITE keeps a diff purely tonal. */
export const HEAT = [1, 0.28, 0.12];
export const WHITE = [1, 1, 1];

// Build an RGBA8 buffer of `n` pixels from a per-index [r, g, b] function (opaque alpha).
const mapRGBA = (n, rgbAt) => {
  const d = new Uint8Array(n * 4);
  for (let i = 0; i < n; i++) {
    const [r, g, b] = rgbAt(i);
    const o = i * 4;
    d[o] = r;
    d[o + 1] = g;
    d[o + 2] = b;
    d[o + 3] = 255;
  }
  return d;
};

/** Coverage as grayscale RGBA8 (white = covered). */
export const grayRGBA = (cov) =>
  mapRGBA(cov.length, (i) => {
    const v = Math.round(Math.max(0, Math.min(1, cov[i])) * 255);
    return [v, v, v];
  });

/**
 * |a − b| as an RGBA8 error map: black where the two agree, `tint` where they don't. `gain` amplifies the
 * difference (clamped at 1), because renderers that track the same filter differ by only a few percent —
 * at gain 1 an accurate renderer's error map is indistinguishable from black.
 */
export const diffRGBA = (a, b, gain = AMP, tint = HEAT) =>
  mapRGBA(a.length, (i) => {
    // quantise to 8-bit first, then tint, so the brightest channel is exactly the grayscale error
    const v = Math.round(Math.min(Math.abs(a[i] - b[i]) * gain, 1) * 255);
    return [Math.round(v * tint[0]), Math.round(v * tint[1]), Math.round(v * tint[2])];
  });

/**
 * A rectangle of the nearest-neighbour ×z upscale of an RGBA8 image, without building the whole upscale.
 * Coordinates are in UPSCALED space, so a crop can be positioned to sub-source-pixel precision (an inset of
 * 40 upscaled px is 10 source px at z=4, but 41 is not a whole source pixel and still works here).
 * Nearest-neighbour is the point: every output pixel is one exact source value, so a magnified figure shows
 * the coverage a renderer actually produced instead of interpolation invented afterwards.
 */
export function upscaleCrop(rgba, w, h, z, cx, cy, cw, ch) {
  const d = new Uint8Array(cw * ch * 4);
  for (let y = 0; y < ch; y++) {
    const sy = Math.min(h - 1, Math.max(0, ((cy + y) / z) | 0));
    for (let x = 0; x < cw; x++) {
      const sx = Math.min(w - 1, Math.max(0, ((cx + x) / z) | 0));
      const o = (y * cw + x) * 4, s = (sy * w + sx) * 4;
      d[o] = rgba[s];
      d[o + 1] = rgba[s + 1];
      d[o + 2] = rgba[s + 2];
      d[o + 3] = 255;
    }
  }
  return d;
}

/** Nearest-neighbour upscale of an RGBA8 image by an integer factor, so individual pixels stay square. */
export const upscale = (rgba, w, h, z) => (z === 1 ? rgba : upscaleCrop(rgba, w, h, z, 0, 0, w * z, h * z));

/** A label → a filename-safe slug ('star {5/2} nonzero' → 'star_5_2_nonzero'). */
export const slugify = (s) => s.toLowerCase().replace(/[^a-z0-9]+/g, '_').replace(/^_+|_+$/g, '');
