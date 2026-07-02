// main.js — the demo entry point (`deno task render`).
//
// Renders the phrase "area coverage" at a ladder of geometrically increasing sizes, every glyph of every
// row in one instanced draw, and writes an anti-aliased PNG. The sizes share one banded glyph atlas, so the
// geometry is stored once however many times a letter repeats.

import { loadFont } from "./font.js";
import { buildGlyphAtlas } from "./bands.js";
import { layoutLine, measureText } from "./layout.js";
import { renderToRGBA } from "./gpu.js";
import { encodePNG } from "./png.js";

const TEXT = "area coverage";
const INK = [0.11, 0.11, 0.17, 1]; // near-black ink
const BG = [0.96, 0.95, 0.92, 1]; // warm off-white
const MARGIN = 64;

// The zoom ladder: STEPS sizes in geometric progression from MIN to MAX (a constant ratio between rows),
// preceded by a couple of extra tiny rows to show the box-integral coverage stays clean and evenly weighted
// as the type degrades to a few px (at 8px the x-height is only ~4px, well below one pixel of stem detail).
const STEPS = 10;
const MIN_SIZE = 20;
const MAX_SIZE = 200;
const TINY = [8, 13];
const ratio = (MAX_SIZE / MIN_SIZE) ** (1 / (STEPS - 1));
const sizes = [...TINY, ...Array.from({ length: STEPS }, (_, i) => MIN_SIZE * ratio ** i)];

const font = await loadFont(
  new URL("../assets/Lato-Regular.ttf", import.meta.url),
);
const { curves, rows, table, stats } = buildGlyphAtlas(font, TEXT);

// Lay out one row per size, left-aligned, stacked with spacing proportional to each size so the rhythm
// scales with the geometric ladder (the gaps grow at the same ratio as the type).
const instances = [];
let maxWidth = 0;
let y = MARGIN;
for (const size of sizes) {
  const inkAbove = 0.56 * size; // x-height headroom above the baseline
  const inkBelow = 0.28 * size; // descender depth ('g')
  const gap = 0.34 * size; // space to the next row
  const baselineY = y + inkAbove;
  layoutLine(instances, TEXT, table, font, {
    x: MARGIN,
    baselineY,
    fontSizePx: size,
    color: INK,
  });
  maxWidth = Math.max(maxWidth, measureText(TEXT, font, size));
  y = baselineY + inkBelow + gap;
}
const width = Math.ceil(maxWidth + MARGIN * 2);
const height = Math.ceil(y - 0.34 * sizes.at(-1) + MARGIN); // drop the trailing gap, add the bottom margin

const instanceData = new Float32Array(instances);
const instanceCount = instanceData.length / 16;

console.log(
  `Rendering "${TEXT}" at ${sizes.length} sizes (${sizes[0]}–${MAX_SIZE}px) → ${width}×${height}`,
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
});
const t1 = performance.now();

const png = encodePNG(rgba, width, height);
await Deno.mkdir(new URL("../output/", import.meta.url), { recursive: true });
const outPath = new URL("../output/area-coverage.png", import.meta.url);
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
