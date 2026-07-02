// font.js — load a TrueType font and read glyph outlines + metrics via opentype.js.
//
// TrueType outlines are quadratic Béziers, which is what the shader consumes, so no curve conversion is
// needed for such a font. Outlines come out in font units (see `unitsPerEm`), Y-down, baseline at y = 0.

import opentype from 'opentype.js';

export async function loadFont(url) {
  const bytes = await Deno.readFile(url);
  return opentype.parse(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
}

// A cubic → two quadratics (only reached if the font carries cubic outlines; TrueType does not).
function cubicToQuads(x0, y0, c1x, c1y, c2x, c2y, x1, y1, out) {
  const m = (a, b) => (a + b) / 2;
  const ax = m(x0, c1x), ay = m(y0, c1y), bx = m(c1x, c2x), by = m(c1y, c2y);
  const cx = m(c2x, x1), cy = m(c2y, y1), dx = m(ax, bx), dy = m(ay, by), ex = m(bx, cx), ey = m(by, cy);
  const mx = m(dx, ex), my = m(dy, ey);
  out.push(x0, y0, 1.5 * dx - 0.25 * (x0 + mx), 1.5 * dy - 0.25 * (y0 + my), mx, my);
  out.push(mx, my, 1.5 * ex - 0.25 * (mx + x1), 1.5 * ey - 0.25 * (my + y1), x1, y1);
}

/**
 * A glyph's outline as flat quadratics [x0,y0, cx,cy, x1,y1] (font units, Y-down, baseline at 0). Straight
 * segments become quads with the control point at their midpoint, so the outline is one uniform list.
 * Returns null for a blank glyph (e.g. space). Also returns the glyph's advance and ink bbox.
 */
export function glyphQuads(font, ch) {
  const g = font.charToGlyph(ch);
  const path = g.getPath(0, 0, font.unitsPerEm);
  const quads = [];
  const line = (x0, y0, x1, y1) => quads.push(x0, y0, (x0 + x1) / 2, (y0 + y1) / 2, x1, y1);
  let cx = 0, cy = 0, sx = 0, sy = 0;
  for (const c of path.commands) {
    if (c.type === 'M') { cx = c.x; cy = c.y; sx = c.x; sy = c.y; }
    else if (c.type === 'L') { line(cx, cy, c.x, c.y); cx = c.x; cy = c.y; }
    else if (c.type === 'Q') { quads.push(cx, cy, c.x1, c.y1, c.x, c.y); cx = c.x; cy = c.y; }
    else if (c.type === 'C') { cubicToQuads(cx, cy, c.x1, c.y1, c.x2, c.y2, c.x, c.y, quads); cx = c.x; cy = c.y; }
    else if (c.type === 'Z') { if (cx !== sx || cy !== sy) line(cx, cy, sx, sy); cx = sx; cy = sy; }
  }
  if (quads.length === 0) return null;
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (let i = 0; i < quads.length; i += 2) {
    x0 = Math.min(x0, quads[i]); x1 = Math.max(x1, quads[i]);
    y0 = Math.min(y0, quads[i + 1]); y1 = Math.max(y1, quads[i + 1]);
  }
  return { quads, advance: g.advanceWidth, bbox: [x0, y0, x1, y1] };
}

/** Advance width of a character in font units. */
export function advanceOf(font, ch) {
  return font.charToGlyph(ch).advanceWidth;
}

/** Kerning between two characters in font units (0 if the font has no pair for them). */
export function kerningOf(font, a, b) {
  return font.getKerningValue(font.charToGlyph(a), font.charToGlyph(b));
}
