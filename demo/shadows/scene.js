// scene.js — the soft-shadow canopy: a field of vector leaves floating above a ground plane, each casting an
// analytic soft shadow with a physically-plausible, per-pixel-variable penumbra (docs/SHADOWS.md).
//
// This is pure data + math (no WebGPU, no DOM), so it is shared by the interactive client (demo/shadows/main.js)
// and the offscreen preview tools. The shadow of a planar occluder under an area light is the occluder's
// silhouette BLURRED by an amount that grows with the occluder→ground gap ("contact hardening"): sharp where a
// leaf nearly touches the ground, soft where it's high. windfoil box-filters the exact vector silhouette at a
// per-pixel footprint, so we get that penumbra for free by widening the footprint — no blur pass, no SDF.

import { buildShapeAtlas, makeLeaf, mulberry32 } from '../../src/shapes.js';
import { FLOATS_PER_INSTANCE } from '../../src/layout.js';

/**
 * Build a canopy: `count` leaves scattered over a world of `worldW`×`worldH` px, each at its own position,
 * size, rotation, and height above the ground (0 = on the ground, 1 = as high as the canopy gets). Heights are
 * skewed so most leaves sit low (crisp shadows) with a few high ones (broad, soft shadows) — the mix that makes
 * contact hardening read. Deterministic in `seed`.
 *
 * @returns {{ curves: Float32Array, rows: Uint32Array, leaves: object[], worldW: number, worldH: number }}
 */
export function buildCanopy({ seed = 7, worldW = 1600, worldH = 1000, count = 220 } = {}) {
  const rng = mulberry32(seed);
  const shapes = [];
  const meta = [];
  for (let i = 0; i < count; i++) {
    const angle = rng() * Math.PI * 2;
    const length = 46 + 120 * rng() * rng(); // local units; leaf drawn ~this tall before scaling
    shapes.push(makeLeaf({ rng, length, angle }));
    // height skewed low: rng²·rng puts the bulk near the ground, a tail up high
    const height = Math.pow(rng(), 1.7);
    const tiltAng = rng() * Math.PI * 2;
    meta.push({
      x: rng() * worldW,
      y: rng() * worldH,
      scale: 0.7 + 0.6 * rng(), // world px per local unit
      height,
      tiltDir: [Math.cos(tiltAng), Math.sin(tiltAng)],
      tiltFrac: 0.35 + 0.4 * rng(), // how much the penumbra widens across this leaf (depth tilt)
      hue: 96 + 40 * rng(), // green-ish, for the optional leaf layer
      light: 0.42 + 0.3 * rng(),
    });
  }
  const { curves, rows, table } = buildShapeAtlas(shapes);
  const leaves = meta.map((m, i) => ({ ...m, table: table[i] }));
  return { curves, rows, leaves, worldW, worldH };
}

// Default sun: a low-ish sun throwing shadows down-right, with a soft disc. `length` is the shadow offset in
// world px at height 1; `softness` the penumbra diameter in world px at height 1 (the light's angular size).
export const DEFAULT_SUN = { dirX: 0.42, dirY: 0.36, length: 230, softness: 30 };

const SHADOW_FILL = 0; // nonzero fill rule

/**
 * Pack the leaves' SHADOW instances (20 floats each; see layout.js). The shadow is the leaf silhouette,
 * translated by the sun parallax (∝ height) and box-blurred by a penumbra whose diameter is height·softness at
 * the leaf's reference point and tilts across the leaf, so one shadow can sharpen at one edge and soften at the
 * other. Blur is a screen-space width, so it scales with `zoom`.
 *
 * @param {object[]} leaves     from buildCanopy
 * @param {object} o  { zoom, sun, density }
 * @returns {Float32Array}
 */
export function packShadows(leaves, { zoom = 1, sun = DEFAULT_SUN, density = 0.5 } = {}) {
  const out = new Float32Array(leaves.length * FLOATS_PER_INSTANCE);
  let p = 0;
  for (const lf of leaves) {
    const [x0, y0, x1, y1] = lf.table.bbox;
    const cx = (x0 + x1) * 0.5, cy = (y0 + y1) * 0.5;
    const diag = Math.hypot(x1 - x0, y1 - y0) || 1;
    // Shadow origin: leaf's world position, pushed along the sun's ground direction, further the higher it is.
    const ox = lf.x + sun.dirX * lf.height * sun.length;
    const oy = lf.y + sun.dirY * lf.height * sun.length;
    // Penumbra diameter at the leaf centre, in device px (screen-space → scales with zoom).
    const base = lf.height * sun.softness * zoom;
    // Depth tilt: widen the penumbra across the leaf by ±tiltFrac over its half-diagonal (device px / local unit).
    const gmag = (base * lf.tiltFrac) / (0.5 * diag);
    const gradX = gmag * lf.tiltDir[0];
    const gradY = gmag * lf.tiltDir[1];
    // blur.x is the value at the bbox min corner; convert from the centre-referenced model.
    const atMin = base + gradX * (x0 - cx) + gradY * (y0 - cy);
    const maxBlur = base * (1 + lf.tiltFrac) + 0.75; // clamp ceiling; sizes the skirt so no penumbra clips
    out[p++] = ox; out[p++] = oy; out[p++] = lf.scale; out[p++] = SHADOW_FILL; // place
    out[p++] = x0; out[p++] = y0; out[p++] = x1; out[p++] = y1;                // bbox
    out[p++] = 0; out[p++] = 0; out[p++] = 0; out[p++] = density;              // color (black, straight alpha)
    out[p++] = lf.table.rowBase; out[p++] = lf.table.bandCount; out[p++] = lf.table.y0; out[p++] = lf.table.invH;
    out[p++] = atMin; out[p++] = gradX; out[p++] = gradY; out[p++] = maxBlur;  // blur
  }
  return out;
}

// HSL→straight RGB (0..1), for the optional crisp leaf layer above the shadows.
function hsl(h, s, l) {
  h = ((h % 360) + 360) % 360 / 360;
  const a = s * Math.min(l, 1 - l);
  const f = (n) => {
    const k = (n + h * 12) % 12;
    return l - a * Math.max(-1, Math.min(k - 3, 9 - k, 1));
  };
  return [f(0), f(8), f(4)];
}

/** Pack the crisp (unblurred) leaves themselves, drawn above the shadows. blur is all-zero → exact fill. */
export function packLeaves(leaves, { alpha = 1 } = {}) {
  const out = new Float32Array(leaves.length * FLOATS_PER_INSTANCE);
  let p = 0;
  for (const lf of leaves) {
    const [x0, y0, x1, y1] = lf.table.bbox;
    const [r, g, b] = hsl(lf.hue, 0.5, lf.light);
    out[p++] = lf.x; out[p++] = lf.y; out[p++] = lf.scale; out[p++] = SHADOW_FILL;
    out[p++] = x0; out[p++] = y0; out[p++] = x1; out[p++] = y1;
    out[p++] = r; out[p++] = g; out[p++] = b; out[p++] = alpha;
    out[p++] = lf.table.rowBase; out[p++] = lf.table.bandCount; out[p++] = lf.table.y0; out[p++] = lf.table.invH;
    out[p++] = 0; out[p++] = 0; out[p++] = 0; out[p++] = 0;
  }
  return out;
}
