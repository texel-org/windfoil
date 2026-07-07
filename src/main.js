// main.js — the demo entry point (`deno task render`).
//
// Renders the phrase "The five boxing wizards jump quickly" at a ladder of geometrically increasing sizes, every glyph of every
// row in one instanced draw, and writes an anti-aliased PNG. The sizes share one banded glyph atlas, so the
// geometry is stored once however many times a letter repeats.

import { loadFont } from "./font.js";
import { buildGlyphAtlas } from "./bands.js";
import { layoutStack, FLOATS_PER_INSTANCE } from "./layout.js";
import { renderToRGBA } from "./gpu.js";
import { encodePNG } from "./png.js";

// --gamma / --sharp: an opt-in perceptual coverage curve applied on top of the exact coverage (gamma = stem
// weight, <1 bolder / >1 thinner; sharp = edge contrast about 0.5, >1 crisper / <1 softer). Both default to
// 1.0 — the identity — so the plain render is the bit-for-bit exact box filter.
//   e.g. `deno task render --gamma 0.72 --sharp 1.1`
function argValue(name) {
  const i = Deno.args.indexOf(`--${name}`);
  if (i >= 0 && i + 1 < Deno.args.length) return Deno.args[i + 1];
  const eq = Deno.args.find((a) => a.startsWith(`--${name}=`));
  return eq ? eq.slice(name.length + 3) : null;
}
function argNumber(name, fallback) {
  const raw = argValue(name);
  if (raw === null) return fallback;
  const v = Number(raw);
  if (!Number.isFinite(v)) {
    console.error(`--${name} must be a number, got "${raw}"`);
    Deno.exit(1);
  }
  return v;
}
const gamma = argNumber("gamma", 1.0);
const sharp = argNumber("sharp", 1.0);
const style = [gamma, sharp];

const TEXT = "The five boxing wizards jump quickly";
const INK = [12, 15, 28, 0xff].map((x) => x / 0xff); // near-black ink
const BG = [233, 227, 213, 0xff].map((x) => x / 0xff); // warm off-white
const MARGIN = 64;

// The zoom ladder: STEPS sizes in geometric progression from MIN to MAX (a constant ratio between rows),
// preceded by a couple of extra tiny rows to show the box-integral coverage stays clean and evenly weighted
// as the type degrades to a few px (at 8px the x-height is only ~4px, well below one pixel of stem detail).
const STEPS = 10;
const MIN_SIZE = 20;
const MAX_SIZE = 200;
const TINY = [8, 13];
const ratio = (MAX_SIZE / MIN_SIZE) ** (1 / (STEPS - 1));
const sizes = [
  ...TINY,
  ...Array.from({ length: STEPS }, (_, i) => MIN_SIZE * ratio ** i),
];

const font = await loadFont(
  new URL("../assets/Lato-Regular.ttf", import.meta.url),
);
const { curves, rows, table, stats } = buildGlyphAtlas(font, TEXT);

// Lay out one row per size, left-aligned, stacked with spacing proportional to each size so the rhythm
// scales with the geometric ladder (the gaps grow at the same ratio as the type). Same TEXT every row.
const { instances, bounds } = layoutStack(
  sizes.map((size) => ({ text: TEXT, size })),
  table,
  font,
  { x: MARGIN, top: MARGIN, color: INK },
);
const width = Math.ceil(bounds.maxX + MARGIN); // content box + the right/bottom margins
const height = Math.ceil(bounds.maxY + MARGIN);

const instanceData = new Float32Array(instances);
const instanceCount = instanceData.length / FLOATS_PER_INSTANCE;

console.log(
  `Rendering "${TEXT}" [gamma ${gamma}, sharp ${sharp}] at ${sizes.length} sizes (${sizes[0]}–${MAX_SIZE}px) → ${width}×${height}`,
);
const t0 = performance.now();
const rgba = await renderToRGBA({
  width,
  height,
  background: BG,
  curves,
  rows,
  instances: instanceData,
  instanceCount,
  style,
});
const t1 = performance.now();

const png = encodePNG(rgba, width, height);
await Deno.mkdir(new URL("../output/", import.meta.url), { recursive: true });
const outPath = new URL(`../output/windfoil.png`, import.meta.url);
await Deno.writeFile(outPath, png);

console.log(
  `  ${instanceCount} glyph instances, one draw call, ${(t1 - t0).toFixed(1)} ms on the GPU`,
);
console.log(
  `  atlas: ${stats.uniqueGlyphs} unique glyphs → ${stats.monotonePieces} monotone pieces in ` +
    `${stats.bandCount} row bands (${stats.bandedPieces} banded, ${stats.duplication.toFixed(2)}× dup)`,
);
console.log(
  `  wrote ${Deno.realPathSync(outPath)} (${(png.length / 1024).toFixed(1)} KB)`,
);
