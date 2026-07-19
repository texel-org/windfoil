// tools/ab.js — A/B two windfoil shader variants on the bench scenes, interleaved to resist GPU contention.
//
//   deno run --unstable-webgpu -A tools/ab.js --b src/windfoil2.wgsl --scene glyphs --levels 8,16,32 --diff
//
// Both variants read the SAME atlas + instances (only the WGSL differs), so --diff byte-compares their
// readbacks per level: 0 differing bytes = bit-exact on that view. Timing alternates A/B batches within a
// level and reports medians, so slow drift or a concurrently-busy GPU biases both sides equally.

import { loadFont } from '../src/font.js';
import { buildGlyphAtlas } from '../src/bands.js';
import { createGlyphRenderer, loadShaderCode } from '../src/gpu.js';
import { buildSlugAtlas } from '../bench/slug.js';
import { buildScene, INK, SCENE_TEXT } from '../bench/scene.js';
import { buildShapeScene } from '../bench/shape.js';
import { buildTigerScene } from '../bench/tiger.js';

function argValue(name) {
  const i = Deno.args.indexOf(`--${name}`);
  if (i >= 0 && i + 1 < Deno.args.length) return Deno.args[i + 1];
  const eq = Deno.args.find((a) => a.startsWith(`--${name}=`));
  return eq ? eq.slice(name.length + 3) : null;
}
const ROOT = new URL('../', import.meta.url);
const A_URL = new URL(argValue('a') ?? 'src/windfoil.wgsl', ROOT);
const B_URL = new URL(argValue('b') ?? 'src/windfoil.wgsl', ROOT);
const SCENE = argValue('scene') ?? 'glyphs';
const DIFF = Deno.args.includes('--diff');
const REPS = Number(argValue('reps') ?? 5);
const W = Number(argValue('size') ?? 720), H = W;
const EM_WORLD = 100;
const TARGET_MS = Number(argValue('target-ms') ?? 250);
const MIN_FRAMES = 15, MAX_FRAMES = 400;
const DEFAULT_LEVELS = { glyphs: [8, 16, 32, 64], shape: [12, 32, 128], tiger: [64, 256] };
const LEVELS = (argValue('levels')?.split(',').map(Number) ?? DEFAULT_LEVELS[SCENE]).sort((a, b) => a - b);

const adapter = await navigator.gpu?.requestAdapter();
if (!adapter) throw new Error('No WebGPU adapter');
const device = await adapter.requestDevice();
device.addEventListener?.('uncapturederror', (e) => console.error('WebGPU error:', e.error?.message));

const format = 'rgba8unorm';
const target = device.createTexture({
  size: [W, H],
  format,
  usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
});
const view = target.createView();
const BG = [233 / 255, 227 / 255, 213 / 255];
const passDesc = () => ({
  colorAttachments: [{
    view,
    clearValue: { r: BG[0], g: BG[1], b: BG[2], a: 1 },
    loadOp: 'clear',
    storeOp: 'store',
  }],
});

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
const median = (a) => [...a].sort((x, y) => x - y)[a.length >> 1];

async function readback(renderer, cam) {
  renderer.setUniforms({ width: W, height: H, cam });
  const enc = device.createCommandEncoder();
  const pass = enc.beginRenderPass(passDesc());
  renderer.draw(pass);
  pass.end();
  const bytesPerRow = Math.ceil((W * 4) / 256) * 256;
  const buf = device.createBuffer({
    size: bytesPerRow * H,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
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

// ── scene build (mirrors bench/main.js) ─────────────────────────────────────────────────────────────────
const extent = Math.max((W / 2) / (LEVELS[0] / EM_WORLD) * 1.05, 4 * EM_WORLD);
let scene;
if (SCENE === 'glyphs') {
  const font = await loadFont(new URL('assets/Lato-Regular.ttf', ROOT));
  const wA = buildGlyphAtlas(font, SCENE_TEXT);
  const sA = buildSlugAtlas(font, SCENE_TEXT);
  const grid = buildScene(font, wA.table, sA.table, { emWorld: EM_WORLD, extent, color: INK });
  scene = {
    wCurves: wA.curves,
    wRows: wA.rows,
    wInstances: grid.wInstances,
    count: grid.count,
    center: grid.center,
  };
} else if (SCENE === 'shape') {
  scene = buildShapeScene({ emWorld: EM_WORLD, extent, fillRule: 0 });
} else if (SCENE === 'tiger') {
  scene = await buildTigerScene({ emWorld: EM_WORLD, extent });
} else {
  throw new Error(`unknown scene ${SCENE}`);
}

const camForEmPx = (emPx) => {
  const s = emPx / EM_WORLD;
  return [s, s, W / 2 - s * scene.center.x, H / 2 - s * scene.center.y];
};
const visibleIndices = (emPx) => {
  const s = emPx / EM_WORLD;
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

const [codeA, codeB] = await Promise.all([loadShaderCode(A_URL), loadShaderCode(B_URL)]);
console.log(
  `A = ${A_URL.pathname.split('/').slice(-2).join('/')}   B = ${
    B_URL.pathname.split('/').slice(-2).join('/')
  }`,
);
console.log(`scene = ${SCENE} · levels = ${LEVELS.join(',')} · reps = ${REPS}${DIFF ? ' · diff' : ''}\n`);
console.log(
  `${'px'.padStart(6)}  ${'A ms'.padStart(9)}  ${'B ms'.padStart(9)}  ${'B vs A'.padStart(8)}  diff`,
);

for (const emPx of LEVELS) {
  const cam = camForEmPx(emPx);
  const idx = visibleIndices(emPx);
  const sub = new Float32Array(idx.length * 16);
  for (let k = 0; k < idx.length; k++) {
    sub.set(scene.wInstances.subarray(idx[k] * 16, idx[k] * 16 + 16), k * 16);
  }
  const mk = (code) =>
    createGlyphRenderer(device, {
      code,
      format,
      curves: scene.wCurves,
      rows: scene.wRows,
      instances: sub,
      instanceCount: idx.length,
    });
  const rA = mk(codeA), rB = mk(codeB);
  rA.setUniforms({ width: W, height: H, cam });
  rB.setUniforms({ width: W, height: H, cam });
  const estA = (await runBatch(rA, 12)) / 12;
  const estB = (await runBatch(rB, 12)) / 12;
  const frames = Math.max(
    MIN_FRAMES,
    Math.min(MAX_FRAMES, Math.round(TARGET_MS / Math.max(estA, estB, 1e-3))),
  );
  const tA = [], tB = [];
  for (let rep = 0; rep < REPS; rep++) {
    tA.push((await runBatch(rA, frames)) / frames);
    tB.push((await runBatch(rB, frames)) / frames);
  }
  const mA = median(tA), mB = median(tB);
  let diffTxt = '';
  if (DIFF) {
    const a = await readback(rA, cam), b = await readback(rB, cam);
    let bytes = 0, maxD = 0;
    for (let i = 0; i < a.length; i++) {
      const d = Math.abs(a[i] - b[i]);
      if (d) {
        bytes++;
        if (d > maxD) maxD = d;
      }
    }
    diffTxt = bytes === 0 ? 'bit-exact' : `${bytes} bytes differ (max ${maxD})`;
  }
  const ratio = mA / mB;
  console.log(
    `${String(emPx).padStart(6)}  ${mA.toFixed(4).padStart(9)}  ${mB.toFixed(4).padStart(9)}  ` +
      `${(ratio >= 1 ? '+' : '') + ((ratio - 1) * 100).toFixed(1)}%  ${diffTxt}`,
  );
}
