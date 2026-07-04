// bench/kernels.js — filter-kernel cost and look, windfoil vs itself across kernels, in Deno.
//
//   deno run --unstable-webgpu -A bench/kernels.js                          # perf ladders, default kernels
//   deno run --unstable-webgpu -A bench/kernels.js --scene hairlines --kernels box,tent,gaussian,mitchell
//   deno run --unstable-webgpu -A bench/kernels.js --images                 # a PNG per (scene, level, kernel)
//   deno run --unstable-webgpu -A bench/kernels.js --montage                # curated labeled comparison strips
//
// Same scenes, camera ladder and batch-timing methodology as bench/main.js, but the two "algorithms" being
// compared are windfoil-with-box (the untouched core shader — the baseline every kernel is priced against)
// and windfoil-with-<kernel> (windfoil-ext.wgsl specialized by src/kernels.js). Slug is out of the picture
// here; bench/main.js remains the cross-algorithm harness.
//
// The interesting outputs:
//   • the ladder table — each kernel's per-frame cost as a multiple of box at every zoom level. The support
//     story is mechanical (a 2·R px slab touches ~2·R× the bands and culls at ±R·sx). Below ~4px the CORE
//     shader switches to its minification guard while the ext shader (the exactness reference) keeps
//     gathering, so the small-size multipliers measure guard-vs-exact, not kernel cost.
//   • --images / --montage — the zone plate (Moiré), the sub-pixel fan (thin-stroke tone), and small text
//     make the kernels' visual trade visible by eye; mblur/disc strips demo the analytic-effects direction.

import { loadFont } from '../src/font.js';
import { buildGlyphAtlas } from '../src/bands.js';
import { createGlyphRenderer } from '../src/gpu.js';
import { loadKernelShaderCode, resolveKernel } from '../src/kernels.js';
import { layoutLine, measureText } from '../src/layout.js';
import { encodePNG } from '../src/png.js';
import { buildScene, SCENE_TEXT, INK } from './scene.js';
import { buildShapeScene } from './shape.js';
import { buildTigerScene } from './tiger.js';
import { buildHairlineScene } from './hairlines.js';
import { buildSlugAtlas } from './slug.js';

// ── args (same conventions as bench/main.js) ────────────────────────────────────────────────────────────
function argValue(name) {
  const i = Deno.args.indexOf(`--${name}`);
  if (i >= 0 && i + 1 < Deno.args.length) return Deno.args[i + 1];
  const eq = Deno.args.find((a) => a.startsWith(`--${name}=`));
  return eq ? eq.slice(name.length + 3) : null;
}
const argNumber = (name, fallback) => {
  const raw = argValue(name);
  return raw === null ? fallback : Number(raw);
};
const TARGET = argNumber('size', 720);
const EM_WORLD = argNumber('em', 100);
const TARGET_MS = argNumber('target-ms', 250);
const MIN_FRAMES = 15, MAX_FRAMES = 400;
const IMAGES = Deno.args.includes('--images');
const MONTAGE = Deno.args.includes('--montage');
const REF_PX = 16;

const sceneArg = (argValue('scene') || 'glyphs,hairlines,tiger').toLowerCase();
const SCENES = sceneArg.split(',').map((x) => x.trim())
  .filter((x) => ['glyphs', 'shape', 'tiger', 'hairlines'].includes(x));
// box first: it is the baseline every other column is priced against.
const KERNEL_LIST = (argValue('kernels') || 'box,tent,gaussian,mitchell,bspline,catmullrom')
  .split(',').map((x) => x.trim()).filter(Boolean);
for (const k of KERNEL_LIST) resolveKernel(k); // fail fast on typos

const levelsArg = argValue('levels');
const parseLevels = (def) =>
  (levelsArg ? levelsArg.split(',').map(Number) : def).filter((v) => Number.isFinite(v) && v > 0).sort((a, b) => a - b);
// Shorter ladders than bench/main.js — each level runs once per KERNEL. The ext shader is the exactness
// reference and deliberately has NO minification guard, so its minified cost grows with the footprint; the
// 4px row shows that honestly and the 2px row (all guard on the core side, all gather on the ext side) is
// left out of the default ladder to keep runtimes sane.
const LEVELS = {
  glyphs: parseLevels([4, 8, 16, 32, 64, 256, 1024, 4096]),
  shape: parseLevels([16, 32, 64, 256, 1024, 4096]),
  tiger: parseLevels([64, 128, 256, 512, 1024, 4096]),
  hairlines: parseLevels([128, 256, 512, 1024, 2048]),
};

const W = TARGET, H = TARGET;
const BG = [233, 227, 213, 0xff].map((x) => x / 0xff);

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
const kernelCode = Object.fromEntries(await Promise.all(KERNEL_LIST.map(async (k) => [k, await loadKernelShaderCode(k)])));

function passDesc() {
  return {
    colorAttachments: [{ view, clearValue: { r: BG[0], g: BG[1], b: BG[2], a: 1 }, loadOp: 'clear', storeOp: 'store' }],
  };
}

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

async function measure(renderer, cam) {
  renderer.setUniforms({ width: W, height: H, cam });
  const est = (await runBatch(renderer, 20)) / 20;
  const frames = Math.max(MIN_FRAMES, Math.min(MAX_FRAMES, Math.round(TARGET_MS / Math.max(est, 1e-3))));
  const perFrame = [];
  for (let rep = 0; rep < 3; rep++) perFrame.push((await runBatch(renderer, frames)) / frames);
  return median(perFrame);
}

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

// ── scene plumbing (windfoil side of bench/main.js's normalized scene shape) ────────────────────────────
async function buildFor(which, levels) {
  const extent = Math.max((TARGET / 2) / (levels[0] / EM_WORLD) * 1.05, 4 * EM_WORLD);
  if (which === 'glyphs') {
    const font = await loadFont(new URL('../assets/Lato-Regular.ttf', import.meta.url));
    const wA = buildGlyphAtlas(font, SCENE_TEXT);
    const sA = buildSlugAtlas(font, SCENE_TEXT); // buildScene wants both tables; slug side is discarded
    const grid = buildScene(font, wA.table, sA.table, { emWorld: EM_WORLD, extent, color: INK });
    return {
      title: 'glyphs (text grid)', curves: wA.curves, rows: wA.rows,
      instances: grid.wInstances, count: grid.count, center: grid.center,
      stats: `${wA.stats.monotonePieces} monotone pieces / ${wA.stats.bandCount} bands`,
    };
  }
  if (which === 'shape') {
    const sh = buildShapeScene({ emWorld: EM_WORLD, extent, fillRule: 0 });
    return {
      title: `complex shape (${sh.stats.quads} self-crossing quads)`, curves: sh.wCurves, rows: sh.wRows,
      instances: sh.wInstances, count: sh.count, center: sh.center, stats: `${sh.stats.wBanded} banded pieces`,
    };
  }
  if (which === 'hairlines') {
    const hl = buildHairlineScene({ emWorld: EM_WORLD, extent });
    return {
      title: 'hairlines (fan + spikes + zone plate)', curves: hl.wCurves, rows: hl.wRows,
      instances: hl.wInstances, count: hl.count, center: hl.center, stats: `${hl.stats.quads} quads`,
    };
  }
  const tg = await buildTigerScene({ emWorld: EM_WORLD, extent });
  return {
    title: `tiger SVG (${tg.stats.shapes} shapes)`, curves: tg.wCurves, rows: tg.wRows,
    instances: tg.wInstances, count: tg.count, center: tg.center, stats: `${tg.stats.rawCurves} quads`,
  };
}

const camForEmPx = (scene, emPx) => {
  const s = emPx / EM_WORLD;
  return [s, s, W / 2 - s * scene.center.x, H / 2 - s * scene.center.y];
};
const visibleIndices = (scene, emPx) => {
  const s = emPx / EM_WORLD;
  const halfW = (W / 2) / s, halfH = (H / 2) / s, margin = Math.min(EM_WORLD, Math.max(halfW, halfH));
  const loX = scene.center.x - halfW - margin, hiX = scene.center.x + halfW + margin;
  const loY = scene.center.y - halfH - margin, hiY = scene.center.y + halfH + margin;
  const inst = scene.instances;
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
const makeRenderer = (scene, kernel, idx) => {
  const sub = new Float32Array(idx.length * 16);
  for (let k = 0; k < idx.length; k++) sub.set(scene.instances.subarray(idx[k] * 16, idx[k] * 16 + 16), k * 16);
  return createGlyphRenderer(device, {
    code: kernelCode[kernel], format, curves: scene.curves, rows: scene.rows, instances: sub, instanceCount: idx.length,
  });
};

// ── labeled montage strips ───────────────────────────────────────────────────────────────────────────────
// Crops of the same view rendered under each kernel, side by side, each with a caption bar — rendered by the
// engine itself (layoutLine + the box kernel), which is both dogfooding and the only text rasterizer here.
const LABEL_H = 22, SEP = 2;
const LABEL_CHARS = 'abcdefghijklmnopqrstuvwxyz0123456789=.-,· ';
let labelKit = null;
async function ensureLabelKit() {
  if (!labelKit) {
    const font = await loadFont(new URL('../assets/Lato-Regular.ttf', import.meta.url));
    labelKit = { font, atlas: buildGlyphAtlas(font, LABEL_CHARS + SCENE_TEXT) };
  }
  return labelKit;
}
async function renderLabel(text, w) {
  const { atlas, font } = await ensureLabelKit();
  const size = 13;
  const inst = [];
  const tw = measureText(text, font, size);
  layoutLine(inst, text, atlas.table, font, {
    x: Math.max(4, (w - tw) / 2), baselineY: LABEL_H - 7, fontSizePx: size, color: INK,
  });
  const lt = device.createTexture({
    size: [w, LABEL_H], format, usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
  });
  const r = createGlyphRenderer(device, {
    code: kernelCode['box'] ?? await loadKernelShaderCode('box'), format,
    curves: atlas.curves, rows: atlas.rows, instances: new Float32Array(inst), instanceCount: inst.length / 16,
  });
  r.setUniforms({ width: w, height: LABEL_H });
  const enc = device.createCommandEncoder();
  const pass = enc.beginRenderPass({
    colorAttachments: [{
      view: lt.createView(), clearValue: { r: BG[0] * 0.92, g: BG[1] * 0.92, b: BG[2] * 0.92, a: 1 },
      loadOp: 'clear', storeOp: 'store',
    }],
  });
  r.draw(pass);
  pass.end();
  const bpr = Math.ceil((w * 4) / 256) * 256;
  const buf = device.createBuffer({ size: bpr * LABEL_H, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  enc.copyTextureToBuffer({ texture: lt }, { buffer: buf, bytesPerRow: bpr }, [w, LABEL_H]);
  device.queue.submit([enc.finish()]);
  await buf.mapAsync(GPUMapMode.READ);
  const padded = new Uint8Array(buf.getMappedRange());
  const rgba = new Uint8Array(w * LABEL_H * 4);
  for (let y = 0; y < LABEL_H; y++) rgba.set(padded.subarray(y * bpr, y * bpr + w * 4), y * w * 4);
  buf.unmap();
  buf.destroy();
  lt.destroy();
  return rgba;
}

function blit(dst, dW, src, sW, sH, dx, dy) {
  for (let y = 0; y < sH; y++) dst.set(src.subarray(y * sW * 4, (y + 1) * sW * 4), ((dy + y) * dW + dx) * 4);
}

// The bokeh-lights strip: a scatter of period/middle-dot glyphs as warm POINT LIGHTS on near-black — the
// one setting where an aperture's SHAPE is unmistakable (each sub-pixel-ish dot renders as the kernel:
// circle under `disc`, hexagon/triangle/pentagon under `iris`). Dark-on-light text only ever shows generic
// blur, so this panel is light-on-dark by construction.
async function bokehLightsStrip(kernels) {
  const { font, atlas } = await ensureLabelKit();
  const crop = 250;
  const inst = [];
  let seed = 7;
  const rnd = () => (seed = (seed * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff;
  for (let i = 0; i < 46; i++) {
    const size = 16 + rnd() * 42; // period ink ≈ 0.1em → ~1.5–6px dots
    const warm = 0.75 + rnd() * 0.25;
    layoutLine(inst, rnd() < 0.5 ? '.' : '·', atlas.table, font, {
      x: 6 + rnd() * (crop - 18), baselineY: 10 + rnd() * (crop - 16), fontSizePx: size,
      color: [warm, warm * 0.94, warm * 0.8, 1],
    });
  }
  const instances = new Float32Array(inst);
  const cols = kernels.length;
  const MW = cols * crop + (cols - 1) * SEP, MH = LABEL_H + crop;
  const out = new Uint8Array(MW * MH * 4).fill(255);
  const pt = device.createTexture({
    size: [crop, crop], format, usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
  });
  for (let c = 0; c < cols; c++) {
    const k = kernels[c];
    if (!kernelCode[k]) kernelCode[k] = await loadKernelShaderCode(k);
    const r = createGlyphRenderer(device, {
      code: kernelCode[k], format, curves: atlas.curves, rows: atlas.rows,
      instances, instanceCount: instances.length / 16,
    });
    r.setUniforms({ width: crop, height: crop });
    const enc = device.createCommandEncoder();
    const pass = enc.beginRenderPass({
      colorAttachments: [{
        view: pt.createView(), clearValue: { r: 0.05, g: 0.05, b: 0.07, a: 1 }, loadOp: 'clear', storeOp: 'store',
      }],
    });
    r.draw(pass);
    pass.end();
    const bpr = Math.ceil((crop * 4) / 256) * 256;
    const buf = device.createBuffer({ size: bpr * crop, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    enc.copyTextureToBuffer({ texture: pt }, { buffer: buf, bytesPerRow: bpr }, [crop, crop]);
    device.queue.submit([enc.finish()]);
    await buf.mapAsync(GPUMapMode.READ);
    const padded = new Uint8Array(buf.getMappedRange());
    const panel = new Uint8Array(crop * crop * 4);
    for (let y = 0; y < crop; y++) panel.set(padded.subarray(y * bpr, y * bpr + crop * 4), y * crop * 4);
    buf.unmap();
    buf.destroy();
    const x0 = c * (crop + SEP);
    blit(out, MW, await renderLabel(k, crop), crop, LABEL_H, x0, 0);
    blit(out, MW, panel, crop, crop, x0, LABEL_H);
  }
  pt.destroy();
  const dir = new URL('../output/bench/kernels/', import.meta.url);
  await Deno.mkdir(dir, { recursive: true });
  await Deno.writeFile(new URL('bokeh_lights.png', dir), encodePNG(out, MW, MH));
  console.log(`  wrote output/bench/kernels/bokeh_lights.png (${MW}×${MH})`);
}

// One strip: the scene at `emPx`, cropped to crop×crop centred at (W/2 + offX, H/2 + offY), one panel per
// kernel. Writes <name>.png under output/bench/kernels/.
async function montageStrip(scene, { name, emPx, crop, offX = 0, offY = 0, kernels }) {
  const cam = camForEmPx(scene, emPx);
  const idx = visibleIndices(scene, emPx);
  const cols = kernels.length;
  crop = Math.min(crop, W, H); // a crop larger than the target would wrap subarray indices silently
  const MW = cols * crop + (cols - 1) * SEP, MH = LABEL_H + crop;
  const out = new Uint8Array(MW * MH * 4).fill(255);
  const cx = Math.min(Math.max(Math.round(W / 2 + offX - crop / 2), 0), W - crop);
  const cy = Math.min(Math.max(Math.round(H / 2 + offY - crop / 2), 0), H - crop);
  for (let c = 0; c < cols; c++) {
    const k = kernels[c];
    if (!kernelCode[k]) kernelCode[k] = await loadKernelShaderCode(k);
    const rgba = await readback(makeRenderer(scene, k, idx), cam);
    const panel = new Uint8Array(crop * crop * 4);
    for (let y = 0; y < crop; y++) {
      panel.set(rgba.subarray(((cy + y) * W + cx) * 4, ((cy + y) * W + cx + crop) * 4), y * crop * 4);
    }
    const x0 = c * (crop + SEP);
    blit(out, MW, await renderLabel(k, crop), crop, LABEL_H, x0, 0);
    blit(out, MW, panel, crop, crop, x0, LABEL_H);
  }
  const dir = new URL('../output/bench/kernels/', import.meta.url);
  await Deno.mkdir(dir, { recursive: true });
  await Deno.writeFile(new URL(`${name}.png`, dir), encodePNG(out, MW, MH));
  console.log(`  wrote output/bench/kernels/${name}.png (${MW}×${MH})`);
}

// ── run ──────────────────────────────────────────────────────────────────────────────────────────────────
console.log('windfoil filter kernels · Deno WebGPU benchmark');
console.log(`  kernels: ${KERNEL_LIST.join(', ')} · baseline = box (the untouched core shader)`);

for (const which of SCENES) {
  const levels = LEVELS[which];
  const scene = await buildFor(which, levels);
  console.log(`\n════════ ${scene.title} ════════`);
  console.log(`  ${scene.stats} · ${scene.count.toLocaleString()} instances`);

  if (MONTAGE) {
    if (which === 'hairlines') {
      // whole pattern at 256px: zone plate rings (Moiré), fan (sub-pixel tone), spikes — all in frame
      await montageStrip(scene, { name: 'hairlines_0256px', emPx: 256, crop: 250, kernels: KERNEL_LIST });
      // fan/crossing close-up at 1024px: strokes ~1.3px — edge gradient shapes visible
      await montageStrip(scene, { name: 'hairlines_1024px_center', emPx: 1024, crop: 250, kernels: KERNEL_LIST });
    } else if (which === 'glyphs') {
      await montageStrip(scene, { name: 'glyphs_0013px', emPx: 13, crop: 250, kernels: KERNEL_LIST });
      await montageStrip(scene, { name: 'glyphs_0032px', emPx: 32, crop: 250, kernels: KERNEL_LIST });
      // the analytic-effects strip: motion blur, round bokeh, and shaped (N-blade iris) bokeh — same
      // radius for disc vs iris so the aperture SHAPE is the only difference
      await montageStrip(scene, {
        name: 'glyphs_0064px_effects', emPx: 64, crop: 250,
        kernels: ['box', 'mblur=12', 'disc=4', 'iris=4', 'iris=4,5,18'],
      });
      // aperture shapes on point LIGHTS (the only setting where bokeh shape truly reads)
      await bokehLightsStrip(['box', 'disc=5', 'iris=5', 'iris=5,3,30', 'iris=5,5,18']);
    } else if (which === 'tiger') {
      await montageStrip(scene, { name: 'tiger_0256px', emPx: 256, crop: 250, kernels: KERNEL_LIST });
    }
    continue;
  }

  if (IMAGES) {
    const dir = new URL('../output/bench/kernels/', import.meta.url);
    await Deno.mkdir(dir, { recursive: true });
    for (const emPx of levels) {
      const cam = camForEmPx(scene, emPx);
      const idx = visibleIndices(scene, emPx);
      const px = String(emPx).padStart(5, '0');
      for (const k of KERNEL_LIST) {
        const rgba = await readback(makeRenderer(scene, k, idx), cam);
        await Deno.writeFile(new URL(`${which}_${px}px_${k}.png`, dir), encodePNG(rgba, W, H));
      }
    }
    console.log(`  wrote ${levels.length * KERNEL_LIST.length} images → output/bench/kernels/${which}_<px>px_<kernel>.png`);
    continue;
  }

  const head = ['px'.padStart(6), 'zoom'.padStart(6), 'shown'.padStart(7)];
  for (const k of KERNEL_LIST) head.push(k.padStart(k === 'box' ? 9 : 16));
  console.log('\n' + head.join('  '));
  const fmt = (ms) => (ms >= 10 ? ms.toFixed(2) : ms.toFixed(3));
  for (const emPx of levels) {
    const cam = camForEmPx(scene, emPx);
    const idx = visibleIndices(scene, emPx);
    const cells = [];
    let boxMs = 0;
    for (const k of KERNEL_LIST) {
      const ms = await measure(makeRenderer(scene, k, idx), cam);
      if (k === 'box') {
        boxMs = ms;
        cells.push(`${fmt(ms)}m`.padStart(9));
      } else {
        // no box baseline measured (yet) at this level → absolute time only, not a bogus ratio
        const ratio = boxMs > 0 ? ` ${(ms / boxMs).toFixed(2)}×` : '';
        cells.push(`${fmt(ms)}m${ratio}`.padStart(16));
      }
    }
    const zoom = emPx / REF_PX;
    console.log(
      `${String(emPx).padStart(6)}  ${(zoom < 1 ? zoom.toFixed(2) : zoom.toFixed(1)).padStart(5)}×  ` +
        `${idx.length.toLocaleString().padStart(7)}  ${cells.join('  ')}`,
    );
  }
}
