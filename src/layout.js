// layout.js — turn a string into per-glyph shader instances, using the font's own advance widths and
// kerning. Single line, left-to-right; no shaping. Each non-space character becomes one instance pointing
// at its glyph's row-band table, so repeated letters reuse one banded copy.

import { advanceOf, kerningOf } from './font.js';

export const FLOATS_PER_INSTANCE = 16;

/** Width of a laid-out string in device pixels at the given size (used to size the canvas). */
export function measureText(text, font, fontSizePx) {
  const scale = fontSizePx / font.unitsPerEm;
  let w = 0;
  let prev = null;
  for (const ch of text) {
    if (prev !== null) w += kerningOf(font, prev, ch) * scale;
    w += advanceOf(font, ch) * scale;
    prev = ch;
  }
  return w;
}

/**
 * Append one line of text as instances to `out` (a flat number[] of FLOATS_PER_INSTANCE per glyph).
 * `x`/`baselineY` are device pixels; `color` is straight-alpha [r,g,b,a] in 0..1; `fillRule` is
 * 'nonzero' | 'evenodd'. Returns the pen's end x.
 */
export function layoutLine(out, text, table, font, { x, baselineY, fontSizePx, color, fillRule = 'nonzero' }) {
  const scale = fontSizePx / font.unitsPerEm;
  const rule = fillRule === 'evenodd' ? 1 : 0;
  const [r, g, b, a = 1] = color;
  let pen = x;
  let prev = null;
  for (const ch of text) {
    if (prev !== null) pen += kerningOf(font, prev, ch) * scale;
    const gl = table[ch];
    if (gl) {
      out.push(
        pen, baselineY, scale, rule, // place: origin px, units→px, fill rule
        gl.bbox[0], gl.bbox[1], gl.bbox[2], gl.bbox[3], // ink box (font units)
        r, g, b, a, // color
        gl.rowBase, gl.bandCount, gl.y0, gl.invH, // row-band table + y-origin / bands-per-unit
      );
    }
    pen += advanceOf(font, ch) * scale; // advance for glyphs and spaces alike
    prev = ch;
  }
  return pen;
}
