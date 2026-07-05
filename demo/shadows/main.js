// demo/shadows/main.js — the interactive soft-shadow canopy. A field of vector leaves floats above a ground
// plane; each casts an analytic soft shadow whose penumbra widens with the leaf's height (contact hardening),
// evaluated per pixel by widening windfoil's coverage footprint — no blur pass, no SDF. Pan/zoom to inspect;
// the penumbra is a screen-space width, so it's recomputed against the zoom every frame (renderer.updateInstances).
//
// Shares the scene math (scene.js) with the offscreen preview (tools/shadow-preview.js) and the whole GPU
// pipeline (gpu.js / windfoil.wgsl) with the text demo — this file only adds the camera, input, and HUD.
//
// Serve from the repo ROOT (WebGPU needs a secure context; localhost counts): `deno task serve`, then open
//   http://localhost:8080/demo/shadows/

import { loadShaderCode, requestDevice, createGlyphRenderer } from "../../src/gpu.js";
import { FLOATS_PER_INSTANCE } from "../../src/layout.js";
import { buildCanopy, packShadows, packLeaves, DEFAULT_SUN } from "./scene.js";

const BG = [233, 227, 213, 0xff].map((x) => x / 0xff); // warm off-white ground

const MIN_ZOOM = 0.15;
const MAX_ZOOM = 12;

// ── Camera: world-centre + zoom (device px = worldPx·z + (W/2 − z·cam), same model as the text demo). ──
const cam = { x: 800, y: 500, z: 1 };
let dpr = 1, W = 1, H = 1;
let canvas, rect = { left: 0, top: 0, width: 1, height: 1 };

function clampZoom(z) {
  return Math.max(MIN_ZOOM * dpr, Math.min(MAX_ZOOM * dpr, z));
}
function screenToWorld(sx, sy) {
  return { x: (sx - W / 2) / cam.z + cam.x, y: (sy - H / 2) / cam.z + cam.y };
}
function panBy(dx, dy) {
  cam.x -= dx / cam.z;
  cam.y -= dy / cam.z;
}
function zoomAt(sx, sy, f) {
  const w = screenToWorld(sx, sy);
  cam.z = clampZoom(cam.z * f);
  cam.x = w.x - (sx - W / 2) / cam.z;
  cam.y = w.y - (sy - H / 2) / cam.z;
}
function cameraUniform() {
  return [cam.z, cam.z, W / 2 - cam.z * cam.x, H / 2 - cam.z * cam.y];
}

function resize() {
  dpr = Math.min(globalThis.devicePixelRatio || 1, 2);
  W = Math.max(1, Math.round(canvas.clientWidth * dpr));
  H = Math.max(1, Math.round(canvas.clientHeight * dpr));
  canvas.width = W;
  canvas.height = H;
  cam.z = clampZoom(cam.z);
  rect = canvas.getBoundingClientRect();
}
function devicePos(e) {
  return {
    x: (e.clientX - rect.left) * (canvas.width / rect.width),
    y: (e.clientY - rect.top) * (canvas.height / rect.height),
  };
}

function recenter(worldW, worldH) {
  cam.x = worldW / 2;
  cam.y = worldH / 2;
  cam.z = clampZoom(Math.min(W / worldW, H / worldH) * 1.02);
}

function installInput() {
  const pointers = new Map();
  let pinchPrev = null;
  canvas.addEventListener("pointerdown", (e) => {
    canvas.setPointerCapture(e.pointerId);
    pointers.set(e.pointerId, devicePos(e));
    pinchPrev = null;
  });
  canvas.addEventListener("pointermove", (e) => {
    if (!pointers.has(e.pointerId)) return;
    const p = devicePos(e);
    const prev = pointers.get(e.pointerId);
    pointers.set(e.pointerId, p);
    if (pointers.size === 1) {
      panBy(p.x - prev.x, p.y - prev.y);
    } else if (pointers.size === 2) {
      const [a, b] = [...pointers.values()];
      const mid = { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 };
      const dist = Math.hypot(a.x - b.x, a.y - b.y);
      if (pinchPrev) {
        panBy(mid.x - pinchPrev.mid.x, mid.y - pinchPrev.mid.y);
        if (pinchPrev.dist > 0) zoomAt(mid.x, mid.y, dist / pinchPrev.dist);
      }
      pinchPrev = { mid, dist };
    }
  });
  const release = (e) => {
    pointers.delete(e.pointerId);
    pinchPrev = null;
  };
  canvas.addEventListener("pointerup", release);
  canvas.addEventListener("pointercancel", release);
  canvas.addEventListener(
    "wheel",
    (e) => {
      e.preventDefault();
      const p = devicePos(e);
      zoomAt(p.x, p.y, Math.exp(-e.deltaY * 0.0015));
    },
    { passive: false },
  );
  const swallow = (e) => {
    if (e.touches.length === 1) e.preventDefault();
  };
  canvas.addEventListener("touchstart", swallow, { passive: false });
  canvas.addEventListener("touchmove", swallow, { passive: false });
  globalThis.addEventListener("resize", resize);
}

async function main() {
  canvas = document.getElementById("gpu");
  const fpsEl = document.getElementById("fps");
  if (!navigator.gpu) throw new Error("WebGPU is not available in this browser.");

  const worldW = 1600, worldH = 1000;
  const { curves, rows, leaves } = buildCanopy({ seed: 7, worldW, worldH, count: 230 });

  // One combined instance buffer: all shadows first (drawn under), then all leaves (drawn over).
  const n = leaves.length;
  const combined = new Float32Array(2 * n * FLOATS_PER_INSTANCE);

  const code = await loadShaderCode();
  const device = await requestDevice();
  const context = canvas.getContext("webgpu");
  const format = navigator.gpu.getPreferredCanvasFormat();
  context.configure({ device, format, alphaMode: "opaque" });

  const renderer = createGlyphRenderer(device, {
    code, format, curves, rows, instances: combined, instanceCount: 2 * n,
  });

  resize();
  recenter(worldW, worldH);
  installInput();

  // HUD controls
  const $ = (id) => document.getElementById(id);
  const ui = { soft: $("soft"), len: $("len"), dens: $("dens"), leaves: $("leaves") };
  $("reset").addEventListener("click", () => recenter(worldW, worldH));

  const [br, bg, bb, ba] = BG;
  const shadowStride = n * FLOATS_PER_INSTANCE;

  let fpsDt = 1000 / 60, prevTs = 0;
  function frame(now) {
    requestAnimationFrame(frame);
    const dt = prevTs ? now - prevTs : 1000 / 60;
    prevTs = now;
    fpsDt = fpsDt * 0.9 + dt * 0.1;
    const fps = Math.min(60, Math.round(1000 / fpsDt));

    const sun = { ...DEFAULT_SUN, softness: +ui.soft.value, length: +ui.len.value };
    const density = +ui.dens.value;
    // Penumbra is a screen-space width → rebuild the shadow instances against the current zoom every frame.
    combined.set(packShadows(leaves, { zoom: cam.z, sun, density }), 0);
    combined.set(packLeaves(leaves, { alpha: ui.leaves.checked ? 1 : 0 }), shadowStride);
    renderer.updateInstances(combined);

    renderer.setUniforms({ width: W, height: H, cam: cameraUniform() });

    const encoder = device.createCommandEncoder();
    const pass = encoder.beginRenderPass({
      colorAttachments: [
        {
          view: context.getCurrentTexture().createView(),
          clearValue: { r: br * ba, g: bg * ba, b: bb * ba, a: ba },
          loadOp: "clear",
          storeOp: "store",
        },
      ],
    });
    renderer.draw(pass);
    pass.end();
    device.queue.submit([encoder.finish()]);

    fpsEl.textContent = `${fps} fps • ${(cam.z / dpr).toFixed(2)}x`;
  }
  requestAnimationFrame(frame);
}

main().catch((err) => {
  for (const el of document.querySelectorAll(".hud")) el.style.display = "none";
  const c = document.getElementById("gpu");
  if (c) c.style.display = "none";
  const el = document.getElementById("error");
  el.style.display = "block";
  el.textContent = err.message || String(err);
  console.error(err);
});
