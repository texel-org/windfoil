// bench/main.js — windfoil vs Slug, per-frame GPU time across a ladder of zoom levels, in Deno.
//
//   deno run --unstable-webgpu -A bench/main.js                    # all three scenes
//   deno run --unstable-webgpu -A bench/main.js --scene glyphs     # just the text grid
//   deno run --unstable-webgpu -A bench/main.js --scene shape --check
//   deno run --unstable-webgpu -A bench/main.js --levels 4,16,64 --size 900
//
// Three scenes, each rendered by BOTH algorithms into the same offscreen target (only the coverage technique
// differs — windfoil.wgsl vs bench/slug.wgsl):
//   • glyphs — a dense grid of real text: sparse curves, ~1 edge per pixel. Slug's sweet spot.
//   • shape  — one self-crossing shape of ~240 whole quadratics that all span the extent and overlap into a
//              high-winding core: many edges per pixel, and packed bands full of FAR curves. windfoil's regime
//              (it compares — doesn't solve — far curves, and reads one band axis vs Slug's two).
//   • tiger  — a real SVG drawing (the Ghostscript tiger): 304 overlapping shapes with painter's-order
//              overdraw, each shape one instance with its own bands.
//
// A "zoom level" is the on-screen size in device px (pixels-per-em): the camera scales the fixed world scene so
// a unit is `emPx` tall. Each scene's grid is sized to fill the viewport at its smallest level; larger levels
// zoom in on the same dense content, so every level renders a full screen — no empty frames. Off-screen units
// are culled per level (as a real renderer would), keeping the measurement fragment-bound.
//
// Metric: encode N identical passes into one command buffer, submit once, time submit → onSubmittedWorkDone,
// report the median per-frame over a few batches. N is sized so each batch runs ~TARGET_MS. (Timestamp queries
// were tried but Deno/wgpu doesn't normalise the timestamp period, so submit→done wall time is used instead.)

import { loadFont } from '../src/font.js';
import { buildGlyphAtlas } from '../src/bands.js';
import { createGlyphRenderer, loadShaderCode } from '../src/gpu.js';
import { encodePNG } from '../src/png.js';
import { buildSlugAtlas, loadSlugShaderCode } from './slug.js';
import { buildScene, SCENE_TEXT, INK } from './scene.js';
import { buildShapeScene } from './shape.js';
import { buildTigerScene } from './tiger.js';

// ── args ──────────────────────────────────────────────────────────────────────────────────────────────
function argValue(name) {
  const i = Deno.args.indexOf(`--${name}`);
  if (i >= 0 && i + 1 < Deno.args.length) return Deno.args[i + 1];
  const eq = Deno.args.find((a) => a.startsWith(`--${name}=`));
  return eq ? eq.slice(name.length + 3) : null;
}
function argNumber(name, fallback) {
  const raw = argValue(name);
  return raw === null ? fallback : Number(raw);
}
const TARGET = argNumber('size', 720); // offscreen square, device px
const EM_WORLD = argNumber('em', 100); // world units per em (the camera zoom turns this into device px)
const TARGET_MS = argNumber('target-ms', 250); // aim each measurement at ~this long (adaptive frame count)
const MIN_FRAMES = 15, MAX_FRAMES = 400; // MIN keeps the very heavy levels (dense shape, deep minification) bounded
const CHECK = Deno.args.includes('--check');
const CHECK_PX = argNumber('check-px', 0); // override the --check render size (0 = each scene's default)
const IMAGES = Deno.args.includes('--images'); // dump a PNG per level (windfoil + slug), skip timing
const REF_PX = 16; // "1× zoom" reference: a 16px em is normal reading size
// --scene: any of glyphs,shape,tiger (comma list), or `all` / `both` (=glyphs,shape). Default all three.
const sceneArg = (argValue('scene') || 'all').toLowerCase();
const SCENES = sceneArg === 'all'
  ? ['glyphs', 'shape', 'tiger']
  : sceneArg === 'both'
  ? ['glyphs', 'shape']
  : sceneArg.split(',').map((x) => x.trim()).filter((x) => ['glyphs', 'shape', 'tiger'].includes(x));
const SHAPE_FILL = argValue('shape-fill') === 'evenodd' ? 1 : 0;

const levelsArg = argValue('levels');
const parseLevels = (def) =>
  (levelsArg ? levelsArg.split(',').map(Number) : def).filter((v) => Number.isFinite(v) && v > 0).sort((a, b) => a - b);
// On-screen size in device px per level (px = whole-unit height on screen). The shape/tiger ladders skip the
// deepest minification (windfoil's known worst case, already shown by glyphs, and slow) but reach deep
// magnification, where windfoil's footprint collapses to a single band and its advantages surface.
const GLYPH_LEVELS = parseLevels([2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]);
const SHAPE_LEVELS = parseLevels([12, 16, 24, 32, 48, 64, 128, 256, 512, 1024, 2048, 4096, 8192]);
const TIGER_LEVELS = parseLevels([64, 128, 256, 512, 1024, 2048, 4096, 8192]); // px = whole-drawing height on screen
const levelsFor = (which) => (which === 'glyphs' ? GLYPH_LEVELS : which === 'shape' ? SHAPE_LEVELS : TIGER_LEVELS);

const W = TARGET, H = TARGET;
const BG = [233, 227, 213, 0xff].map((x) => x / 0xff); // warm off-white

// ── device + shared GPU plumbing ────────────────────────────────────────────────────────────────────────
const adapter = await navigator.gpu?.requestAdapter();
if (!adapter) {
  console.error('No WebGPU adapter — needs a GPU-capable host (run with --unstable-webgpu).');
  Deno.exit(1);
}
const device = await adapter.requestDevice();
device.addEventListener?.('uncapturederror', (e) => console.error('WebGPU error:', e.error?.message));

const format = 'rgba8unorm';
const target = device.createTexture({
  size: [W, H],
  format,
  usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
});
const view = target.createView();
const [wCode, sCode] = await Promise.all([loadShaderCode(), loadSlugShaderCode()]);

function passDesc() {
  return {
    colorAttachments: [{ view, clearValue: { r: BG[0], g: BG[1], b: BG[2], a: 1 }, loadOp: 'clear', storeOp: 'store' }],
  };
}

// Encode `n` identical passes into one command buffer, submit once, return submit→onSubmittedWorkDone wall (ms).
// Encoding is off the clock and the single submit amortises over all n passes, so wall/n is a clean per-frame
// GPU throughput number (no dependence on the flaky timestamp-query period).
async function runBatch(renderer, n) {
  const enc = device.createCommandEncoder();
  for (let f = 0; f < n; f++) {
    const pass = enc.beginRenderPass(passDesc());
    renderer.draw(pass);
    pass.end();
  }
  const cmd = enc.finish();
  const t0 = performance.now();
  device.queue.submit([cmd]);
  await device.queue.onSubmittedWorkDone();
  return performance.now() - t0;
}

const median = (a) => (a.length ? [...a].sort((x, y) => x - y)[a.length >> 1] : NaN);

// Curve + row buffer size — windfoil's single band axis stores ~half of Slug's dual bands (the memory / read
// bandwidth advantage the algorithm was designed for; see the repo README's tiger-SVG figures).
const atlasKB = (curves, rows) => `${((curves.byteLength + rows.byteLength) / 1024).toFixed(0)}KB`;

// Short scene tag for output filenames, derived from the ladder title.
const sceneTag = (title) => (title.startsWith('complex') ? 'shape' : title.startsWith('tiger') ? 'tiger' : 'glyphs');

// One measurement: warm up, size the batch so it runs ~TARGET_MS, then median per-frame over a few batches.
async function measure(renderer, cam) {
  renderer.setUniforms({ width: W, height: H, cam });
  const est = (await runBatch(renderer, 20)) / 20;
  const frames = Math.max(MIN_FRAMES, Math.min(MAX_FRAMES, Math.round(TARGET_MS / Math.max(est, 1e-3))));
  const perFrame = [];
  for (let rep = 0; rep < 3; rep++) perFrame.push((await runBatch(renderer, frames)) / frames);
  return { perFrame: median(perFrame), frames };
}

// ── one scene, one full zoom ladder ─────────────────────────────────────────────────────────────────────
// `scene` is the normalized shape both builders return: { wCurves, wRows, sCurves, sRows, wInstances,
// sInstances, count, center, worldSpan, statsLine, checkEmPx, fillRule }.
async function runLadder(title, scene, levels) {
  const ALGOS = [
    { name: 'windfoil', code: wCode, curves: scene.wCurves, rows: scene.wRows, instances: scene.wInstances, floats: 16 },
    { name: 'slug', code: sCode, curves: scene.sCurves, rows: scene.sRows, instances: scene.sInstances, floats: 20 },
  ];
  const camForEmPx = (emPx) => {
    const s = emPx / EM_WORLD;
    return [s, s, W / 2 - s * scene.center.x, H / 2 - s * scene.center.y];
  };
  // Indices of instances whose world footprint overlaps the viewport at `emPx` (+1-em margin). Positions are
  // identical between the two layouts, so one index set fits both.
  const visibleIndices = (emPx) => {
    const s = emPx / EM_WORLD;
    // Margin so partly-visible units count: a full unit at normal zoom, but at deep zoom shrink it to the view
    // span so the count stays honest (a fixed unit-wide margin would pull in whole off-screen neighbours).
    const halfW = (W / 2) / s, halfH = (H / 2) / s, margin = Math.min(EM_WORLD, Math.max(halfW, halfH));
    const loX = scene.center.x - halfW - margin, hiX = scene.center.x + halfW + margin;
    const loY = scene.center.y - halfH - margin, hiY = scene.center.y + halfH + margin;
    const inst = scene.wInstances;
    const idx = [];
    for (let i = 0; i < scene.count; i++) {
      const b = i * 16;
      const px = inst[b], py = inst[b + 1], sc = inst[b + 2];
      const gx0 = px + inst[b + 4] * sc, gy0 = py + inst[b + 5] * sc;
      const gx1 = px + inst[b + 6] * sc, gy1 = py + inst[b + 7] * sc;
      if (gx1 >= loX && gx0 <= hiX && gy1 >= loY && gy0 <= hiY) idx.push(i);
    }
    return idx;
  };
  const makeRenderer = (algo, idx) => {
    const f = algo.floats;
    const sub = new Float32Array(idx.length * f);
    for (let k = 0; k < idx.length; k++) sub.set(algo.instances.subarray(idx[k] * f, idx[k] * f + f), k * f);
    return createGlyphRenderer(device, {
      code: algo.code, format, curves: algo.curves, rows: algo.rows, instances: sub, instanceCount: idx.length,
    });
  };

  console.log(`\n════════ ${title} ════════`);
  console.log(`  ${scene.statsLine}`);
  console.log(`  ${scene.count.toLocaleString()} instances tiled over ${(scene.worldSpan | 0).toLocaleString()} world units into ${W}×${H}`);

  // --images: dump a PNG per level (both algorithms) so you can flip through the zoom ladder. No timing.
  if (IMAGES) {
    const tag = sceneTag(title);
    const dir = new URL('../output/bench/levels/', import.meta.url);
    await Deno.mkdir(dir, { recursive: true });
    for (const emPx of levels) {
      const cam = camForEmPx(emPx);
      const idx = visibleIndices(emPx);
      const px = String(emPx).padStart(5, '0'); // zero-pad so files sort by zoom
      for (const algo of ALGOS) {
        const rgba = await readback(makeRenderer(algo, idx), cam);
        await Deno.writeFile(new URL(`${tag}_${px}px_${algo.name}.png`, dir), encodePNG(rgba, W, H));
      }
    }
    console.log(`  wrote ${levels.length * 2} images → output/bench/levels/${tag}_<px>px_{windfoil,slug}.png`);
    return;
  }

  console.log('');
  console.log(
    `${'px'.padStart(6)}  ${'zoom'.padStart(6)}  ${'on-screen'.padStart(9)}  ${'windfoil'.padStart(10)}  ${'slug'.padStart(10)}   faster`,
  );
  const fmt = (ms) => (ms >= 10 ? ms.toFixed(2) : ms.toFixed(4));
  const rows = [];
  for (const emPx of levels) {
    const cam = camForEmPx(emPx);
    const idx = visibleIndices(emPx);
    const w = await measure(makeRenderer(ALGOS[0], idx), cam);
    const s = await measure(makeRenderer(ALGOS[1], idx), cam);
    const zoom = emPx / REF_PX;
    const faster = w.perFrame < s.perFrame
      ? `windfoil ${(s.perFrame / w.perFrame).toFixed(2)}×`
      : `slug ${(w.perFrame / s.perFrame).toFixed(2)}×`;
    rows.push({ emPx, zoom, visible: idx.length, w, s, faster });
    console.log(
      `${String(emPx).padStart(6)}  ${(zoom < 1 ? zoom.toFixed(2) : zoom.toFixed(1)).padStart(5)}×  ` +
        `${idx.length.toLocaleString().padStart(9)}  ${fmt(w.perFrame).padStart(9)}m  ${fmt(s.perFrame).padStart(9)}m   ${faster}`,
    );
  }

  // Takeaway: windfoil's worst regime, its minification-guard recovery, and the magnification crossover.
  // Below ~GUARD-covered sizes (a whole glyph ≤ GUARD_PX ≈ 3.7 device px → glyph ems ≤ ~5px) windfoil renders
  // from the banded ink profile; between that and ~16px is its remaining exact-path worst case.
  const GUARDED_EM_PX = 5;
  const mid = rows.filter((r) => r.emPx > GUARDED_EM_PX && r.emPx <= 16);
  const worst = mid.reduce((a, r) => (!a || r.s.perFrame / r.w.perFrame < a.s.perFrame / a.w.perFrame ? r : a), null);
  if (worst && worst.w.perFrame > worst.s.perFrame) {
    console.log(
      `\n  minification: slug up to ${(worst.w.perFrame / worst.s.perFrame).toFixed(1)}× faster (at ${worst.emPx}px) — ` +
        `windfoil's footprint spans many bands, integrating many curves each.`,
    );
  }
  const guard = rows.filter((r) => r.emPx <= GUARDED_EM_PX && r.w.perFrame < r.s.perFrame);
  if (guard.length) {
    const g = guard[guard.length - 1];
    console.log(`  illegible: at ≤${g.emPx}px windfoil's MINIFICATION_GUARD renders from the banded ink profile → ${(g.s.perFrame / g.w.perFrame).toFixed(1)}× faster than slug.`);
  }
  const mag = rows.filter((r) => r.emPx > 16);
  const cross = mag.find((r) => r.w.perFrame < r.s.perFrame);
  if (cross && mag.slice(mag.indexOf(cross)).every((r) => r.w.perFrame <= r.s.perFrame * 1.02)) {
    const gap = mag[mag.length - 1];
    console.log(
      `  magnified: windfoil pulls ahead from ~${cross.emPx}px, up to ${(gap.s.perFrame / gap.w.perFrame).toFixed(2)}× ` +
        `at ${gap.emPx}px — dense bands of FAR curves it compares (not solves), on one axis not two.`,
    );
  }

  if (CHECK) await check(scene, ALGOS, makeRenderer, camForEmPx, visibleIndices, title);
}

// ── correctness / quality check: dump PNGs + a windfoil-vs-slug coverage diff at a readable size ──────────
async function readback(renderer, cam) {
  renderer.setUniforms({ width: W, height: H, cam });
  const enc = device.createCommandEncoder();
  const pass = enc.beginRenderPass(passDesc());
  renderer.draw(pass);
  pass.end();
  const bytesPerRow = Math.ceil((W * 4) / 256) * 256;
  const buf = device.createBuffer({ size: bytesPerRow * H, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  enc.copyTextureToBuffer({ texture: target }, { buffer: buf, bytesPerRow }, [W, H]);
  device.queue.submit([enc.finish()]);
  await buf.mapAsync(GPUMapMode.READ);
  const padded = new Uint8Array(buf.getMappedRange());
  const rgba = new Uint8Array(W * H * 4);
  for (let y = 0; y < H; y++) rgba.set(padded.subarray(y * bytesPerRow, y * bytesPerRow + W * 4), y * W * 4);
  buf.unmap();
  buf.destroy();
  return rgba;
}

async function check(scene, ALGOS, makeRenderer, camForEmPx, visibleIndices, title) {
  const emPx = CHECK_PX || scene.checkEmPx;
  const cam = camForEmPx(emPx);
  const idx = visibleIndices(emPx);
  const wRGBA = await readback(makeRenderer(ALGOS[0], idx), cam);
  const sRGBA = await readback(makeRenderer(ALGOS[1], idx), cam);
  const bgR = Math.round(BG[0] * 255);
  let sum = 0, wInk = 0, sInk = 0;
  for (let i = 0; i < W * H; i++) {
    sum += Math.abs(wRGBA[i * 4] - sRGBA[i * 4]) + Math.abs(wRGBA[i * 4 + 1] - sRGBA[i * 4 + 1]) + Math.abs(wRGBA[i * 4 + 2] - sRGBA[i * 4 + 2]);
    if (wRGBA[i * 4] < bgR - 8) wInk++;
    if (sRGBA[i * 4] < bgR - 8) sInk++;
  }
  const meanDiff = sum / (W * H * 3) / 255;
  const tag = sceneTag(title);
  await Deno.mkdir(new URL('../output/bench/', import.meta.url), { recursive: true });
  await Deno.writeFile(new URL(`../output/bench/${tag}_windfoil.png`, import.meta.url), encodePNG(wRGBA, W, H));
  await Deno.writeFile(new URL(`../output/bench/${tag}_slug.png`, import.meta.url), encodePNG(sRGBA, W, H));
  console.log(
    `  check @ ${emPx}px:  windfoil inked ${(100 * wInk / (W * H)).toFixed(1)}% · slug inked ${(100 * sInk / (W * H)).toFixed(1)}%` +
      ` · mean |Δrgb| ${meanDiff.toFixed(4)}  → output/bench/${tag}_{windfoil,slug}.png`,
  );
}

// ── build the requested scenes and run ──────────────────────────────────────────────────────────────────
console.log('windfoil vs slug · Deno WebGPU benchmark');
console.log(`  timing: submit→done ms/frame (median of 3 batches, ~${TARGET_MS}ms each) · off-screen units culled per level`);

for (const which of SCENES) {
  const levels = levelsFor(which);
  // Grid half-span: fills the viewport at the smallest level, but never smaller than a few units — otherwise an
  // all-magnified ladder would size the grid below one unit and leave nothing to render.
  const extent = Math.max((TARGET / 2) / (levels[0] / EM_WORLD) * 1.05, 4 * EM_WORLD);

  if (which === 'glyphs') {
    const font = await loadFont(new URL('../assets/Lato-Regular.ttf', import.meta.url));
    const wA = buildGlyphAtlas(font, SCENE_TEXT);
    const sA = buildSlugAtlas(font, SCENE_TEXT);
    const grid = buildScene(font, wA.table, sA.table, { emWorld: EM_WORLD, extent, color: INK });
    await runLadder('glyphs (text grid)', {
      wCurves: wA.curves, wRows: wA.rows, sCurves: sA.curves, sRows: sA.rows,
      wInstances: grid.wInstances, sInstances: grid.sInstances, count: grid.count,
      center: grid.center, worldSpan: grid.worldSpan, checkEmPx: 48,
      statsLine: `windfoil ${wA.stats.monotonePieces} monotone pieces / ${wA.stats.bandCount} bands / ${atlasKB(wA.curves, wA.rows)} · ` +
        `slug ${sA.stats.curves} whole quads / ${sA.stats.bandCount} bands (dual) / ${atlasKB(sA.curves, sA.rows)}`,
    }, levels);
  } else if (which === 'shape') {
    const sh = buildShapeScene({ emWorld: EM_WORLD, extent, fillRule: SHAPE_FILL });
    await runLadder(`complex shape · ${sh.stats.quads} self-crossing quads · ${SHAPE_FILL ? 'even-odd' : 'nonzero'}`, {
      ...sh, checkEmPx: 256,
      statsLine: `${sh.stats.quads} quads · windfoil ${sh.stats.wBanded} banded / ${sh.stats.wBands} bands / ${atlasKB(sh.wCurves, sh.wRows)} · ` +
        `slug ${sh.stats.sBanded} banded / ${sh.stats.sBands} bands (dual) / ${atlasKB(sh.sCurves, sh.sRows)}`,
    }, levels);
  } else {
    const tg = await buildTigerScene({ emWorld: EM_WORLD, extent });
    await runLadder(`tiger SVG · ${tg.stats.shapes} shapes / ${tg.stats.rawCurves} quads · nonzero`, {
      ...tg, checkEmPx: 512,
      statsLine: `${tg.stats.shapes} shapes, ${tg.stats.rawCurves} quads · ` +
        `windfoil ${tg.stats.wBanded} banded / ${tg.stats.wBands} bands / ${atlasKB(tg.wCurves, tg.wRows)} · ` +
        `slug ${tg.stats.sBanded} banded / ${tg.stats.sBands} bands (dual) / ${atlasKB(tg.sCurves, tg.sRows)}`,
    }, levels);
  }
}
