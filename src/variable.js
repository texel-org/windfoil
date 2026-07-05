// variable.js — build a scene for windfoil-variable.wgsl: closed contours whose on-curve ANCHORS each carry a
// color (authored in sRGB, stored in OKLab) and a blur_scale. It reuses the core pipeline unchanged — split
// to xy-monotone pieces (geometry.js) and file them into row bands (bands.js) exactly like a glyph — and adds
// one parallel buffer: the anchor field the shader blends per pixel.
//
// Color lives in OKLab so the per-pixel Shepard blend interpolates perceptually (no muddy sRGB midpoints).
// We convert each anchor sRGB → OKLab here with @texel/color, and hand the shader the SAME OKLab→sRGB matrices
// (packed for a WGSL uniform) so its per-pixel decode is bit-equivalent to the library — a flat single-color
// shape renders identically to the plain windfoil pipeline.

import { convert, LMS_to_linear_sRGB_M, OKLab, OKLab_to_LMS_M, sRGB } from '@texel/color';
import { pushMonotonePieces } from './geometry.js';
import { bandPieces } from './bands.js';

export const FLOATS_PER_INSTANCE = 16;
export const FLOATS_PER_ANCHOR = 8; // vec4(posX, posY, blurScale, alpha) + vec4(L, a, b, pad)

// Accept a color as a '#rgb'/'#rrggbb' hex string or an [r,g,b] / [r,g,b,a] array in 0..1 → [r,g,b,a] sRGB.
function normalizeColor(c) {
  if (typeof c === 'string') {
    let h = c.replace('#', '');
    if (h.length === 3) h = [...h].map((d) => d + d).join('');
    const r = parseInt(h.slice(0, 2), 16) / 255;
    const g = parseInt(h.slice(2, 4), 16) / 255;
    const b = parseInt(h.slice(4, 6), 16) / 255;
    const a = h.length >= 8 ? parseInt(h.slice(6, 8), 16) / 255 : 1;
    return [r, g, b, a];
  }
  return [c[0], c[1], c[2], c[3] ?? 1];
}

const clamp01 = (v) => Math.min(Math.max(v, 0), 1);

// Pack a 3×3 math matrix (row-major M[i][j]) as a WGSL mat3x3<f32>: three COLUMNS, each vec3 padded to vec4,
// so `mat * v` in the shader computes Σ_j M[i][j]·v[j]. 12 floats.
export function packMat3(M) {
  const out = new Float32Array(12);
  for (let col = 0; col < 3; col++) {
    out[col * 4 + 0] = M[0][col];
    out[col * 4 + 1] = M[1][col];
    out[col * 4 + 2] = M[2][col];
    out[col * 4 + 3] = 0;
  }
  return out;
}

// The two matrices windfoil-variable.wgsl needs, packed for the uniform (OKLab → LMS' and LMS → linear sRGB).
// The cube nonlinearity sits between them, so they can't be pre-multiplied into one — the shader keeps both.
export function oklabUniformMatrices() {
  return { okToLms: packMat3(OKLab_to_LMS_M), lmsToRgb: packMat3(LMS_to_linear_sRGB_M) };
}

// Normalize a shape's contours: accept either `shape.contours` (array of { anchors, controls }) or a single
// top-level `shape.anchors`/`shape.controls`. Each contour is a CLOSED loop of on-curve anchors; segment j
// runs anchor[j] → control[j] → anchor[j+1], with a null/absent control meaning a straight line (control at
// the midpoint, matching geometry.js's line convention).
function contoursOf(shape) {
  if (shape.contours) return shape.contours;
  return [{ anchors: shape.anchors, controls: shape.controls }];
}

/**
 * Build the GPU buffers for a list of variable shapes.
 *
 * A shape:
 *   {
 *     anchors | contours,             // on-curve points; each { x, y, color, blur } (color sRGB, blur 0..1)
 *     controls?,                      // per-segment control points ({x,y} | null); null ⇒ straight segment
 *     fillRule?: 'nonzero'|'evenodd',
 *     maxBlur:  number,               // shape-space box widening at blur_scale = 1 (softness range)
 *     falloff?: number,               // Shepard power p (higher ⇒ tighter per-anchor zones; default 2)
 *     place:    { x, y, scale },      // origin in device px + shape-units→px scale
 *   }
 *
 * Returns the four storage buffers the shader binds, the instance count, and the content bounds (device px,
 * blur skirt included) for canvas sizing.
 */
export function buildVariableScene(shapes) {
  const curves = []; // flat 6-float band-duplicated pieces (bands.js appends here)
  const rows = []; // ROW_STRIDE u32-ish per band
  const anchors = []; // FLOATS_PER_ANCHOR per anchor
  const instances = [];
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;

  for (const shape of shapes) {
    const pieces = [];
    let loX = Infinity, loY = Infinity, hiX = -Infinity, hiY = -Infinity;
    const anchorBase = anchors.length / FLOATS_PER_ANCHOR;

    for (const contour of contoursOf(shape)) {
      const A = contour.anchors;
      const n = A.length;
      const controls = contour.controls ?? new Array(n).fill(null);
      for (let j = 0; j < n; j++) {
        const p0 = A[j];
        const p1 = A[(j + 1) % n];
        const ctrl = controls[j];
        const cx = ctrl ? ctrl.x : (p0.x + p1.x) / 2;
        const cy = ctrl ? ctrl.y : (p0.y + p1.y) / 2;
        pushMonotonePieces([p0.x, p0.y, cx, cy, p1.x, p1.y], pieces);
      }
      // Anchor field: convert sRGB → OKLab once, on the CPU, and store L,a,b + blur + alpha.
      for (const p of A) {
        const [r, g, b, a] = normalizeColor(p.color);
        const [L, aa, bb] = convert([r, g, b], sRGB, OKLab);
        anchors.push(p.x, p.y, clamp01(p.blur ?? 0), a, L, aa, bb, 0);
      }
    }

    // Exact ink bbox: every piece is xy-monotone, so its endpoints (and the enclosed control) bound it.
    for (let k = 0; k < pieces.length; k += 6) {
      for (let o = 0; o < 6; o += 2) {
        loX = Math.min(loX, pieces[k + o]);
        hiX = Math.max(hiX, pieces[k + o]);
        loY = Math.min(loY, pieces[k + o + 1]);
        hiY = Math.max(hiY, pieces[k + o + 1]);
      }
    }

    const header = bandPieces(pieces, loY, hiY, curves, rows);
    const anchorCount = anchors.length / FLOATS_PER_ANCHOR - anchorBase;

    const { x, y, scale } = shape.place;
    const rule = shape.fillRule === 'evenodd' ? 1 : 0;
    instances.push(
      x,
      y,
      scale,
      rule, // place: origin px, units→px, fill rule
      loX,
      loY,
      hiX,
      hiY, // ink box (shape units)
      shape.maxBlur,
      shape.falloff ?? 2,
      anchorBase,
      anchorCount, // blur: maxBlur, falloff, anchor range
      header.rowBase,
      header.bandCount,
      header.y0,
      header.invH, // row-band table
    );

    // World bounds (device px) including the blur skirt, so the canvas + margins never clip a soft edge.
    const skirt = shape.maxBlur * 0.5;
    minX = Math.min(minX, x + (loX - skirt) * scale);
    minY = Math.min(minY, y + (loY - skirt) * scale);
    maxX = Math.max(maxX, x + (hiX + skirt) * scale);
    maxY = Math.max(maxY, y + (hiY + skirt) * scale);
  }

  return {
    curves: new Float32Array(curves),
    rows: new Uint32Array(rows),
    anchors: new Float32Array(anchors),
    instances: new Float32Array(instances),
    instanceCount: instances.length / FLOATS_PER_INSTANCE,
    bounds: { minX, minY, maxX, maxY },
  };
}
