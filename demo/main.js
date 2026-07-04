// demo/main.js — the web demo: a tall run of ~128 lorem-ipsum lines, each bigger than the last, rendered to a
// WebGPU canvas in a realtime rAF loop you can pan and pinch/zoom around (0.05× out to 3000× in). Every glyph
// of every line is one instance in a SINGLE instanced draw (the same banded atlas the PNG demo uses — repeated
// letters across all lines share one banded copy), and the coverage shader re-anti-aliases per pixel, so the
// type stays crisp at any zoom. There are so many lines because you can zoom so far out.
//
// It is DRY with the offscreen renderer: the atlas (bands.js), layout (layout.js/layoutStack), font parsing
// (font.js) and the GPU pipeline + shader (gpu.js/windfoil.wgsl) are all the shared src/*.js modules — this file
// only adds the browser-specific parts (a camera, pointer/wheel input, a canvas swapchain, and the HUD).
//
// Serve from the repo ROOT (WebGPU needs a secure context — localhost counts) so /src/*.js, /src/windfoil.wgsl and
// /assets/*.ttf all resolve:  `deno task serve`  then open  http://localhost:8080/demo/

import { parseFont } from "../src/font.js";
import { buildGlyphAtlas } from "../src/bands.js";
import { layoutStack } from "../src/layout.js";
import {
  loadShaderCode,
  requestDevice,
  createGlyphRenderer,
} from "../src/gpu.js";

const INK = [12, 15, 28, 0xff].map((x) => x / 0xff); // near-black ink
// const BG = [233, 227, 213, 0xff].map((x) => x / 0xff); // warm off-white
const BG = [233, 227, 213, 0xff].map((x) => x / 0xff); // warm off-white

// Zoom range, expressed as the user-facing "zoom level" (1× = one world unit per CSS px). The camera's device
// scale is dpr × this, so the readout matches what the eye sees regardless of the display's pixel ratio.
const MIN_ZOOM = 0.005;
const MAX_ZOOM = 100;

// A size ladder over many lines, each bigger than the last. A plain geometric ramp spends equal lines per
// octave, which leaves too many giant lines; skewing the normalized index by GROWTH_SKEW (>1) packs most of
// the lines into the small end — a long run of readable body-text sizes with only a short tail of big ones.
const N_LINES = 180;
const MIN_SIZE = 2;
const MAX_SIZE = 2200;
const GROWTH_SKEW = 1.35; // >1 → more small lines, fewer big ones (1 = plain geometric)
const SIZES = Array.from(
  { length: N_LINES },
  (_, i) =>
    MIN_SIZE * (MAX_SIZE / MIN_SIZE) ** ((i / (N_LINES - 1)) ** GROWTH_SKEW),
);

// Deterministic lorem lines (screenshot-stable, no RNG state): the first line is literally "lorem ipsum", the
// rest flow through a lorem corpus at a varying 3–8 words per line. Each line is one row of glyphs.
const LOREM = (
  "lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod tempor incididunt ut labore et " +
  "dolore magna aliqua enim ad minim veniam quis nostrud exercitation ullamco laboris nisi aliquip ex ea " +
  "commodo consequat duis aute irure in reprehenderit voluptate velit esse cillum eu fugiat nulla pariatur " +
  "excepteur sint occaecat cupidatat non proident sunt culpa qui officia deserunt mollit anim id est " +
  "laborum perspiciatis unde omnis iste natus error voluptatem accusantium doloremque laudantium totam rem " +
  "aperiam eaque ipsa quae ab illo inventore veritatis quasi architecto beatae vitae dicta explicabo nemo"
).split(" ");
function makeLines(n) {
  const lines = ["lorem ipsum"];
  let w = 0;
  for (let i = 1; i < n; i++) {
    const count = 3 + ((i * 3) % 6); // 3..8 words, deterministic
    const words = [];
    for (let k = 0; k < count; k++) words.push(LOREM[w++ % LOREM.length]);
    lines.push(words.join(" "));
  }
  return lines;
}
const LINES = makeLines(N_LINES);

// ---------------------------------------------------------------------------------------------------
// Camera — a world-center + zoom model (mirrors the network-stress example). `z` is DEVICE px per world unit;
// `(x, y)` is the world point under the screen center. Resize-robust (a resize only moves the W/2,H/2 offset,
// not what you're looking at) and makes zoom-about-cursor trivial. The user-facing "zoom level" is z / dpr.
// ---------------------------------------------------------------------------------------------------
const cam = { x: 0, y: 0, z: 1 }; // the input TARGET — pointer/gesture/wheel handlers write this
const view = { x: 0, y: 0, z: 1 }; // the RENDERED camera — eased toward `cam` only during a gesture's opening
let attackT = -Infinity; // performance.now() the current gesture began; camera smoothing fades out ~ATTACK_MS after
let dpr = 1;
let W = 1; // canvas backing-store px
let H = 1;
let contentBounds = { minX: 0, minY: 0, maxX: 1, maxY: 1 };

// Kinetic panning — a flick keeps gliding and eases to a stop, for a native feel (velX/velY are device px per
// ms, sampled from the drag). `dragging` suspends the glide while a finger/mouse is down. The glide itself is
// applied once per frame in the rAF loop; a new touch cancels it (tap-to-stop).
let velX = 0;
let velY = 0;
let dragging = false;

const zoomLevel = () => view.z / dpr; // world-units-per-CSS-px → the number shown in the HUD (what's on screen)
function clampZoom(z) {
  return Math.max(MIN_ZOOM * dpr, Math.min(MAX_ZOOM * dpr, z));
}
function screenToWorld(sx, sy) {
  return { x: (sx - W / 2) / cam.z + cam.x, y: (sy - H / 2) / cam.z + cam.y };
}
function panBy(dxDev, dyDev) {
  cam.x -= dxDev / cam.z;
  cam.y -= dyDev / cam.z;
}
function zoomAt(sx, sy, factor) {
  const w = screenToWorld(sx, sy); // hold the world point under the cursor fixed across the zoom
  cam.z = clampZoom(cam.z * factor);
  cam.x = w.x - (sx - W / 2) / cam.z;
  cam.y = w.y - (sy - H / 2) / cam.z;
}
// Default view: zoom in on the big lower lines so they FILL the screen, rather than fitting the whole sparse
// triangle (which leaves the readable big text tiny). We size the zoom so ~ROWS_TO_SHOW of the biggest rows
// fill the height, then pin the content's bottom-left near the screen's bottom-left so you read into the run.
const ROWS_TO_SHOW = 10; // how many of the biggest rows fill the viewport height (smaller = more zoomed in)
function recenter() {
  velX = velY = 0; // stop any glide when snapping back to the default view
  const rowH = 1.18 * MAX_SIZE; // world-px a biggest row occupies (inkAbove+inkBelow+gap; see layoutStack)
  cam.z = clampZoom(H / (ROWS_TO_SHOW * rowH));
  const pad = 24 * dpr; // small inset from the screen edges (device px)
  cam.x = contentBounds.minX + (W / 2 - pad) / cam.z; // content left ~pad from the left edge
  cam.y = contentBounds.maxY - (H / 2 - pad) / cam.z; // content bottom ~pad from the bottom edge
}

// The camera as the shader's uniform: device px = worldPx · (z, z) + (transX, transY), where the translate
// folds in both the pan (−z·cam) and the screen-center offset (W/2, H/2).
function cameraUniform() {
  return [view.z, view.z, W / 2 - view.z * view.x, H / 2 - view.z * view.y];
}

// ---------------------------------------------------------------------------------------------------
// Sizing — keep the backing store at device resolution; capped dpr keeps the fill-rate sane on retina.
// ---------------------------------------------------------------------------------------------------
let canvas;
let rect = { left: 0, top: 0, width: 1, height: 1 }; // cached canvas bounds — refreshed on resize, not per event
function resize() {
  dpr = Math.min(globalThis.devicePixelRatio || 1, 2);
  W = Math.max(1, Math.round(canvas.clientWidth * dpr));
  H = Math.max(1, Math.round(canvas.clientHeight * dpr));
  canvas.width = W;
  canvas.height = H;
  cam.z = clampZoom(cam.z); // a smaller window can push us past the zoom floor/ceiling
  rect = canvas.getBoundingClientRect();
}

// Map a pointer/wheel event to device px (handles CSS sizing + dpr in one shot). Uses the cached rect so we
// don't force a synchronous layout on every pointermove (that reflow can itself jank a fast drag on mobile).
function devicePos(e) {
  return {
    x: (e.clientX - rect.left) * (canvas.width / rect.width),
    y: (e.clientY - rect.top) * (canvas.height / rect.height),
  };
}

// ---------------------------------------------------------------------------------------------------
// Input — pointer drag (1 finger = pan, 2 fingers = pinch-zoom + pan), wheel zoom about the cursor, and a
// ---------------------------------------------------------------------------------------------------
function installInput() {
  const pointers = new Map();
  let pinchPrev = null;
  let nativeGesture = false; // a WebKit gesture* pinch is driving (Safari iOS/macOS) — mute the pointer fallback
  let lastMoveT = 0; // performance.now() of the previous pan sample, used for the release (fling) velocity

  canvas.addEventListener("pointerdown", (e) => {
    canvas.setPointerCapture(e.pointerId);
    const first = pointers.size === 0;
    pointers.set(e.pointerId, devicePos(e));
    pinchPrev = null;
    dragging = true;
    velX = velY = 0; // a new touch stops any ongoing glide (tap-to-stop, like native scroll views)
    lastMoveT = performance.now();
    // Arm the fading camera smoother on a *touch* gesture's first finger (see the rAF loop). Mouse/trackpad
    // have no landing jitter, so leave them crisp 1:1 (attackT stays in the past → the smoothing weight is 1).
    if (first && e.pointerType !== "mouse") attackT = lastMoveT;
  });
  canvas.addEventListener("pointermove", (e) => {
    if (!pointers.has(e.pointerId)) return;
    const p = devicePos(e);
    const prev = pointers.get(e.pointerId);
    pointers.set(e.pointerId, p);
    if (pointers.size === 1) {
      const dx = p.x - prev.x;
      const dy = p.y - prev.y;
      panBy(dx, dy); // move the target 1:1 — the rAF loop smooths the rendered view only during the opening
      // Track a smoothed pointer velocity (device px per ms) so a flick keeps gliding after release.
      const t = performance.now();
      const dt = t - lastMoveT;
      if (dt > 0) {
        velX = velX ? velX * 0.7 + (dx / dt) * 0.3 : dx / dt;
        velY = velY ? velY * 0.7 + (dy / dt) * 0.3 : dy / dt;
        lastMoveT = t;
      }
    } else if (pointers.size === 2 && !nativeGesture) {
      // Fallback pinch, reconstructed from two pointers — for browsers without gesture* events (Chrome/
      // Firefox, Android). On Safari the native handlers below take over via the nativeGesture gate.
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
    if (pointers.size === 0) {
      dragging = false;
      // Released after a pause (finger held still) → no fling, so it stops where you left it.
      if (performance.now() - lastMoveT > 80) velX = velY = 0;
    }
  };
  canvas.addEventListener("pointerup", release);
  canvas.addEventListener("pointercancel", release);

  // Native pinch-zoom via WebKit gesture events (Safari on iOS + the macOS trackpad). They hand us a
  // cumulative `scale` and the gesture centroid, so pinch-zoom and its pan track the OS's own gesture — much
  // smoother than reconstructing it from two pointers. Where they fire they win (nativeGesture gate);
  // elsewhere the pointer pinch above is the fallback. Desktop scroll-to-zoom (the wheel handler) is untouched.
  let gPrev = null;
  canvas.addEventListener("gesturestart", (e) => {
    e.preventDefault();
    nativeGesture = true;
    pinchPrev = null;
    velX = velY = 0;
    attackT = performance.now(); // a pinch is a fresh gesture too — fade-smooth its opening
    const p = devicePos(e);
    gPrev = { scale: e.scale || 1, x: p.x, y: p.y };
  });
  canvas.addEventListener("gesturechange", (e) => {
    e.preventDefault();
    if (!gPrev) return;
    const p = devicePos(e);
    panBy(p.x - gPrev.x, p.y - gPrev.y); // follow the centroid (on iOS it moves; on a Mac trackpad it ~holds)
    if (gPrev.scale > 0) zoomAt(p.x, p.y, e.scale / gPrev.scale);
    gPrev = { scale: e.scale, x: p.x, y: p.y };
  });
  const gestureEnd = (e) => {
    if (e && e.preventDefault) e.preventDefault();
    nativeGesture = false;
    gPrev = null;
  };
  canvas.addEventListener("gestureend", gestureEnd);

  canvas.addEventListener(
    "wheel",
    (e) => {
      e.preventDefault();
      const p = devicePos(e);
      zoomAt(p.x, p.y, Math.exp(-e.deltaY * 0.0015)); // trackpad/mouse wheel → smooth exponential zoom
    },
    { passive: false },
  );

  // iOS: even with `touch-action: none`, Safari still runs its own scroll/overscroll recognizer for the first
  // moments of a *fast* swipe — which is why a slow "touch, pause, then drag" is smooth but a quick flick
  // judders (the browser is contesting the gesture and delivers pointer events late/bunched until it yields).
  // preventDefault on single-finger touches claims the gesture immediately, so the pointermoves arrive clean.
  // Two-finger touches are left alone so the native pinch (the gesture* handlers above) still runs.
  const swallowTouch = (e) => {
    if (e.touches.length === 1) e.preventDefault();
  };
  canvas.addEventListener("touchstart", swallowTouch, { passive: false });
  canvas.addEventListener("touchmove", swallowTouch, { passive: false });

  document.getElementById("hint").textContent = "pan & zoom around";
  globalThis.addEventListener("resize", resize);
}

// Format the zoom level like the HUD spec: "0.05x", "1x", "2.4x", "3000x".
function fmtZoom(z) {
  if (z >= 100) return z.toFixed(0);
  if (z >= 1) return z.toFixed(1).replace(/\.0$/, "");
  return z.toFixed(3);
}

// ---------------------------------------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------------------------------------
async function main() {
  canvas = document.getElementById("gpu");
  const fpsEl = document.getElementById("fps");

  if (!navigator.gpu)
    throw new Error("WebGPU is not available in this browser.");

  // Load the font bytes + shader source in parallel, then build the shared atlas and lay out the stack.
  const fontUrl = new URL("../assets/Lato-Regular.ttf", import.meta.url);
  const [fontBuf, code] = await Promise.all([
    fetch(fontUrl).then((r) => r.arrayBuffer()),
    loadShaderCode(),
  ]);
  const font = parseFont(fontBuf);

  const text = LINES.join(""); // every char that appears, so the atlas covers the whole scene
  const { curves, rows, table } = buildGlyphAtlas(font, text);
  const { instances, bounds } = layoutStack(
    LINES.map((t, i) => ({ text: t, size: SIZES[i] })),
    table,
    font,
    { x: 0, top: 0, color: INK },
  );
  contentBounds = bounds;

  const instanceData = new Float32Array(instances);
  const instanceCount = instanceData.length / 16;

  // WebGPU device + canvas swapchain, configured for the preferred format.
  const device = await requestDevice();
  const context = canvas.getContext("webgpu");
  const format = navigator.gpu.getPreferredCanvasFormat();
  context.configure({ device, format, alphaMode: "opaque" });

  const renderer = createGlyphRenderer(device, {
    code,
    format,
    curves,
    rows,
    instances: instanceData,
    instanceCount,
  });

  resize();
  recenter();
  view.x = cam.x; // start the rendered camera exactly on the target (no ease-in from the default view)
  view.y = cam.y;
  view.z = cam.z;
  installInput();

  const [br, bg, bb, ba] = BG;

  // One continuous rAF loop, and the ONLY place anything is drawn — input handlers just move the camera, they
  // never render. We render on every vsync (no throttle): iOS holds ProMotion at 60Hz when idle but ramps to
  // 120Hz *during touch*, and matching that is what keeps panning smooth. An earlier 60fps cap skipped frames,
  // which made the surviving ProMotion frames land at uneven intervals — that was the pan "jitter". (The >60
  // reading was always just the display's refresh rate, never a double render: draw happens once, right here.)
  const FRAME_MS = 1000 / 60; // ~16.67ms — reference interval for the momentum decay and the fps-meter seed
  let fpsDt = FRAME_MS; // EMA of the interval between rendered frames; we average the interval THEN invert it
  //                       (not an EMA of 1000/dt, which Jensen-biases the readout above the true frame rate).
  let prevTs = 0;

  // Fade-out camera smoothing: for a gesture's opening ~ATTACK_MS, ease the rendered `view` toward the input
  // `cam`, with a strength that decays to nothing — after that, view == cam (exact 1:1). This low-passes the
  // jittery first frames of a touch pan/pinch (iOS delivers events unevenly against the 120Hz render clock);
  // it does NOT keep smoothing steady-state motion. Mouse/wheel never arm it, so desktop stays crisp.
  const ATTACK_MS = 300; // how long the opening smoothing lasts before it has fully faded to 1:1
  const SMOOTH_K = 0.3; // per-60Hz-frame catch-up at full strength (smaller = smoother, softer opening)
  function frame(now) {
    requestAnimationFrame(frame);

    const dt = prevTs ? now - prevTs : FRAME_MS;
    prevTs = now;
    fpsDt = fpsDt * 0.9 + dt * 0.1;
    const fps = Math.min(60, Math.round(1000 / fpsDt)); // honest average, shown capped at 60 (won't read >60)

    // Kinetic panning: after a flick, keep gliding and ease to a stop. The decay is per-ms so it feels the
    // same regardless of frame interval; below a hair of a pixel per frame we snap to rest.
    if (!dragging && (velX || velY) && dt > 0) {
      panBy(velX * dt, velY * dt);
      const decay = Math.pow(0.8, dt / FRAME_MS);
      velX *= decay;
      velY *= decay;
      if (Math.abs(velX) < 0.002 && Math.abs(velY) < 0.002) velX = velY = 0;
    }

    // Ease the rendered `view` toward the input `cam`, fading the smoothing out over the gesture's opening.
    const s = now - attackT >= ATTACK_MS ? 0 : 1 - (now - attackT) / ATTACK_MS; // 1 at finger-down → 0 by ATTACK_MS
    if (s > 0) {
      const k = 1 - s * (1 - SMOOTH_K); // per-60Hz-frame catch-up: SMOOTH_K (strong) → 1 (none) as it fades
      const kf = 1 - Math.pow(1 - k, dt / FRAME_MS); // frame-rate-normalized so 60/120Hz feel the same
      view.x += (cam.x - view.x) * kf;
      view.y += (cam.y - view.y) * kf;
      view.z *= Math.pow(cam.z / view.z, kf); // zoom eases in log space (it's multiplicative)
    } else {
      view.x = cam.x; // past the attack window → exact 1:1, zero added lag
      view.y = cam.y;
      view.z = cam.z;
    }

    renderer.setUniforms({
      width: W,
      height: H,
      cam: cameraUniform(),
    });

    const encoder = device.createCommandEncoder();
    const pass = encoder.beginRenderPass({
      colorAttachments: [
        {
          view: context.getCurrentTexture().createView(),
          clearValue: { r: br * ba, g: bg * ba, b: bb * ba, a: ba }, // premultiplied clear
          loadOp: "clear",
          storeOp: "store",
        },
      ],
    });
    renderer.draw(pass);
    pass.end();
    device.queue.submit([encoder.finish()]);

    fpsEl.textContent = `${fps} fps • ${fmtZoom(zoomLevel())}x`;
  }
  requestAnimationFrame(frame);
}

main().catch((err) => {
  // Fallback: hide the HUD + canvas and show a simple red message in the center of the page.
  for (const el of document.querySelectorAll(".hud")) el.style.display = "none";
  const canvasEl = document.getElementById("gpu");
  if (canvasEl) canvasEl.style.display = "none";
  const el = document.getElementById("error");
  el.style.display = "block";
  el.textContent = err.message || String(err);
  console.error(err);
});
