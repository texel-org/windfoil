// browser.js — the browser boot for the validation suite: the same shared harness as `deno task validate`,
// but the 2D-canvas reference is THIS browser's own canvas2d rasterizer (Skia in Chrome, CoreGraphics in
// Safari, WebRender in Firefox), so the page shows how the engine's AA sits against the box filter next to
// ours. Results stream in shape by shape: a stats table up top, then a panel of the six views per shape.
//
// Serve from the repo ROOT so /src/*.js and /assets/*.ttf resolve (WebGPU needs a secure context —
// localhost counts):  `deno task serve`  then open  http://localhost:8080/tools/validate/

import { loadFont } from '../../src/font.js';
import { requestDevice } from '../../src/gpu.js';
import { AMP, diffRGBA, F, grayRGBA, S, validateShapes } from './harness.js';

const $ = (sel) => document.querySelector(sel);
const el = (tag, className, text) => {
  const e = document.createElement(tag);
  if (className) e.className = className;
  if (text !== undefined) e.textContent = text;
  return e;
};
const f5 = (v) => v.toFixed(5);

// An S×S RGBA image as a crisp canvas (CSS scales it up, image-rendering: pixelated keeps pixels square).
function imageCell(rgba, caption) {
  const fig = el('figure');
  const c = el('canvas');
  c.width = S;
  c.height = S;
  c.getContext('2d').putImageData(
    new ImageData(new Uint8ClampedArray(rgba.buffer, rgba.byteOffset, rgba.length), S, S),
    0,
    0,
  );
  fig.append(c, el('figcaption', '', caption));
  return fig;
}

function statsRow(label, fold, ob, cb, lb) {
  const tr = el('tr', fold ? 'fold' : '');
  tr.append(...[label + (fold ? ' †' : ''), f5(ob.mean), f5(ob.max), f5(cb.mean), f5(cb.max), f5(lb.mean), f5(lb.max)]
    .map((t) => el('td', '', t)));
  return tr;
}

function shapePanel({ label, fold, ours, slug, canvas, box, oursVsBox: ob, canvasVsBox: cb, slugVsBox: lb }) {
  const panel = el('section', 'panel');
  panel.append(el('h2', '', label + (fold ? ' † (winding-fold limit — deviation expected)' : '')));
  panel.append(el('div', 'panel-stats',
    `ours vs box: mean ${f5(ob.mean)} max ${f5(ob.max)} · canvas vs box: mean ${f5(cb.mean)} max ${f5(cb.max)}` +
      ` · slug vs box: mean ${f5(lb.mean)} max ${f5(lb.max)}`));
  const row = el('div', 'row');
  row.append(
    imageCell(grayRGBA(ours), 'ours'),
    imageCell(grayRGBA(canvas), 'canvas'),
    imageCell(grayRGBA(slug), 'slug'),
    imageCell(grayRGBA(box), 'box'),
    imageCell(diffRGBA(ours, box), `|ours−box| ×${AMP}`),
    imageCell(diffRGBA(canvas, box), `|canvas−box| ×${AMP}`),
    imageCell(diffRGBA(slug, box), `|slug−box| ×${AMP}`),
    imageCell(diffRGBA(ours, canvas), `|ours−canvas| ×${AMP}`),
    imageCell(diffRGBA(ours, slug), `|ours−slug| ×${AMP}`),
  );
  panel.append(row);
  return panel;
}

// "exact mode" (curiosity knob, see harness ourCoverage): ?exact (16×) or ?ss=N — the shader renders at
// N× and box-averages down, so the winding fold's documented failures shrink ~1/N.
const params = new URLSearchParams(location.search);
const ssParam = Number(params.get('ss'));
const supersample = params.has('exact') && params.get('exact') !== 'false' ? 16 : Math.max(1, ssParam || 1);

$('#params').textContent =
  `${S}px cell · box filter = ${F}×${F} zero-AA point-sample · canvas = this browser's canvas2d` +
  `${supersample > 1 ? ` · ours ×${supersample} supersampled (fold per sub-pixel)` : ''}`;

try {
  const device = await requestDevice();
  const font = await loadFont(new URL('../../assets/Lato-Regular.ttf', import.meta.url));
  const createContext2D = (w, h) => {
    const c = el('canvas');
    c.width = w;
    c.height = h;
    return c.getContext('2d', { willReadFrequently: true });
  };

  // Two aggregates: the whole dataset, and the common shapes (no † fold rows — the documented winding-fold
  // limits, expected to deviate; the stars stay in common, their sliver deviation isn't a true failure).
  const agg = () => ({ n: 0, obMean: 0, cbMean: 0, lbMean: 0, obMax: 0, cbMax: 0, lbMax: 0 });
  const all = agg(), common = agg();
  const add = (a, ob, cb, lb) => {
    a.n++; a.obMean += ob.mean; a.cbMean += cb.mean; a.lbMean += lb.mean;
    a.obMax = Math.max(a.obMax, ob.max); a.cbMax = Math.max(a.cbMax, cb.max); a.lbMax = Math.max(a.lbMax, lb.max);
  };
  for await (const result of validateShapes({ font, createContext2D, device, supersample })) {
    const { label, fold, oursVsBox: ob, canvasVsBox: cb, slugVsBox: lb } = result;
    add(all, ob, cb, lb);
    if (!fold) add(common, ob, cb, lb);
    $('#stats tbody').append(statsRow(label, fold, ob, cb, lb));
    $('#panels').append(shapePanel(result));
    $('#status').textContent = `running… ${all.n} shapes done`;
    await new Promise(requestAnimationFrame); // let the new row paint before the next shape blocks the thread
  }

  const line = (a) =>
    `ours vs box mean ${f5(a.obMean / a.n)} max ${a.obMax.toFixed(3)} · ` +
    `canvas vs box mean ${f5(a.cbMean / a.n)} max ${a.cbMax.toFixed(3)} · ` +
    `slug vs box mean ${f5(a.lbMean / a.n)} max ${a.lbMax.toFixed(3)}`;
  const summary = `done · whole dataset (${all.n} shapes): ${line(all)} — ` +
    `common shapes (${common.n}, no † fold-limit rows): ${line(common)}`;
  $('#status').textContent = summary;
  console.log(`validate: ${summary}`); // greppable from headless runs
} catch (err) {
  $('#status').textContent = `error: ${err?.message ?? err}`;
  console.error(err);
  throw err;
}
