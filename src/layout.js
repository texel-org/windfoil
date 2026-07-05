// layout.js — turn a string into per-glyph shader instances, using the font's own advance widths and
// kerning. Single line, left-to-right; no shaping. Each non-space character becomes one instance pointing
// at its glyph's row-band table, so repeated letters reuse one banded copy.

import { advanceOf, kerningOf } from './font.js';

export const FLOATS_PER_INSTANCE = 20;

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
        0, 0, 0, 0, // blur: text is never blurred (sEff == s → exact box filter, bit-for-bit)
      );
    }
    pen += advanceOf(font, ch) * scale; // advance for glyphs and spaces alike
    prev = ch;
  }
  return pen;
}

/**
 * Stack a run of lines top-to-bottom, each at its own size, left-aligned at `x`, and append every glyph to a
 * single instance array (so the whole stack is one instanced draw). The row rhythm — x-height headroom above
 * the baseline, descender depth, and the gap to the next row — all scale with each line's size, so the spacing
 * grows with a geometric size ladder. This is the shared layout both the PNG ladder and the pan/zoom client
 * use.
 *
 * @param {{text: string, size: number}[]} lines
 * @param {object} table  the per-glyph band table from buildGlyphAtlas
 * @param {object} font
 * @param {object} o  { x, top, color, fillRule } — top-left origin (device/world px), ink color, fill rule
 * @returns {{ instances: number[], bounds: {minX,minY,maxX,maxY} }} flat instances + the content's world bbox
 */
export function layoutStack(lines, table, font, { x, top, color, fillRule = 'nonzero' }) {
  const instances = [];
  let maxWidth = 0;
  let y = top;
  for (const { text, size } of lines) {
    const inkAbove = 0.56 * size; // x-height headroom above the baseline
    const inkBelow = 0.28 * size; // descender depth ('g')
    const gap = 0.34 * size; // space to the next row
    const baselineY = y + inkAbove;
    layoutLine(instances, text, table, font, { x, baselineY, fontSizePx: size, color, fillRule });
    maxWidth = Math.max(maxWidth, measureText(text, font, size));
    y = baselineY + inkBelow + gap;
  }
  const bottom = y - 0.34 * lines[lines.length - 1].size; // drop the trailing gap after the last line
  return { instances, bounds: { minX: x, minY: top, maxX: x + maxWidth, maxY: bottom } };
}
