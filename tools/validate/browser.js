// browser.js — the browser boot for the validation suite: the same shared harness as `deno task validate`,
// but the 2D-canvas reference is THIS browser's own canvas2d rasterizer (Skia in Chrome, CoreGraphics in
// Safari, WebRender in Firefox), so the page shows how the engine's AA sits against the box-filter oracle
// next to ours. Results stream in shape by shape: a stats table up top, then a panel of the views per shape.
//
// Serve from the repo ROOT so /src/*.js and /assets/*.ttf resolve (WebGPU needs a secure context —
// localhost counts):  `deno task serve`  then open  http://localhost:8080/tools/validate/

import { loadFont } from '../../src/font.js';
import { requestDevice } from '../../src/gpu.js';
import { AMP, diffRGBA, grayRGBA, S, validateShapes } from './harness.js';

const $ = (sel) => document.querySelector(sel);
const el = (tag, className, text) => {
  const e = document.createElement(tag);
  if (className) e.className = className;
  if (text !== undefined) e.textContent = text;
  return e;
};
// Values span 1e-14 (oracle residual) to 3e-1 (fold maxes): 1-significant-digit scientific throughout.
const f = (v) => (v === 0 ? '0' : v.toExponential(1));

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

function statsRow(label, fold, ob, cb, lb, refErr) {
  const tr = el('tr', fold ? 'fold' : '');
  const bad = ob.bad + cb.bad + lb.bad;
  const cells = [label + (fold ? ' †' : '') + (bad ? ` ⚠ ${bad} non-finite px excluded` : '')];
  for (const st of [ob, cb, lb]) cells.push(f(st.mean), f(st.edgeMean), f(st.p99), f(st.max));
  cells.push(f(refErr));
  tr.append(...cells.map((t) => el('td', '', t)));
  return tr;
}

function shapePanel(
  { label, fold, ours, slug, canvas, oracle, oursVsOracle: ob, canvasVsOracle: cb, slugVsOracle: lb },
) {
  const panel = el('section', 'panel');
  panel.append(el('h2', '', label + (fold ? ' † (winding-fold limit — deviation expected)' : '')));
  const s = (name, st) => `${name}: edgeμ ${f(st.edgeMean)} p99 ${f(st.p99)} max ${f(st.max)}`;
  panel.append(
    el('div', 'panel-stats', `${s('ours', ob)} · ${s('canvas', cb)} · ${s('slug', lb)}  (vs oracle)`),
  );
  const row = el('div', 'row');
  row.append(
    imageCell(grayRGBA(ours), 'ours'),
    imageCell(grayRGBA(canvas), 'canvas'),
    imageCell(grayRGBA(slug), 'slug'),
    imageCell(grayRGBA(oracle), 'oracle'),
    imageCell(diffRGBA(ours, oracle), `|ours−oracle| ×${AMP}`),
    imageCell(diffRGBA(canvas, oracle), `|canvas−oracle| ×${AMP}`),
    imageCell(diffRGBA(slug, oracle), `|slug−oracle| ×${AMP}`),
    imageCell(diffRGBA(ours, canvas), `|ours−canvas| ×${AMP}`),
    imageCell(diffRGBA(ours, slug), `|ours−slug| ×${AMP}`),
  );
  panel.append(row);
  return panel;
}

// Exact mode (see harness ourCoverage): ?exact renders ours with the shader's EXACT_MODE override —
// in-shader true-fill sampling instead of the winding fold.
const params = new URLSearchParams(location.search);
const exact = params.has('exact') && params.get('exact') !== 'false';

$('#params').textContent =
  `${S}px cell · oracle = box filter, exact-in-x + adaptive Gauss–Legendre in y (f64) · ` +
  `ours/slug = f32 readback · canvas = this browser's canvas2d (8-bit API)` +
  `${exact ? ' · ours = EXACT_MODE (8×8 true-fill sampling, no fold)' : ''}`;

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
  // Edge quantiles pool every edge pixel of every shape in the set.
  const rend = () => ({ meanSum: 0, max: 0, pools: [] });
  const agg = () => ({ n: 0, refErr: 0, ours: rend(), canvas: rend(), slug: rend() });
  const all = agg(), common = agg();
  const add = (a, refErr, ob, cb, lb) => {
    a.n++;
    a.refErr = Math.max(a.refErr, refErr);
    for (const [r, st] of [[a.ours, ob], [a.canvas, cb], [a.slug, lb]]) {
      r.meanSum += st.mean;
      r.max = Math.max(r.max, st.max);
      r.pools.push(st.edgeErrs);
    }
  };
  for await (const result of validateShapes({ font, createContext2D, device, exact })) {
    const { label, fold, refErr, oursVsOracle: ob, canvasVsOracle: cb, slugVsOracle: lb } = result;
    add(all, refErr, ob, cb, lb);
    if (!fold) add(common, refErr, ob, cb, lb);
    $('#stats tbody').append(statsRow(label, fold, ob, cb, lb, refErr));
    $('#panels').append(shapePanel(result));
    $('#status').textContent = `running… ${all.n} shapes done`;
    await new Promise(requestAnimationFrame); // let the new row paint before the next shape blocks the thread
  }

  const pooled = (r) => {
    let n = 0;
    for (const p of r.pools) n += p.length;
    const errs = new Float64Array(n);
    let o = 0, sum = 0;
    for (const p of r.pools) {
      errs.set(p, o);
      o += p.length;
      for (const e of p) sum += e;
    }
    errs.sort();
    const q = (p) => (n ? errs[Math.floor(p * (n - 1))] : 0);
    return { edgeMean: sum / n, p50: q(0.5), p99: q(0.99) };
  };
  const line = (a) =>
    ['ours', 'canvas', 'slug'].map((k) => {
      const p = pooled(a[k]);
      return `${k} cell mean ${f(a[k].meanSum / a.n)} edgeμ ${f(p.edgeMean)} p50 ${f(p.p50)} p99 ${
        f(p.p99)
      } max ${f(a[k].max)}`;
    }).join(' · ');
  const summary =
    `done · whole dataset (${all.n} shapes, oracle residual ≤ ${f(all.refErr)}): ${line(all)} — ` +
    `common shapes (${common.n}, no † fold-limit rows): ${line(common)}`;
  $('#status').textContent = summary;
  console.log(`validate: ${summary}`); // greppable from headless runs
} catch (err) {
  $('#status').textContent = `error: ${err?.message ?? err}`;
  console.error(err);
  throw err;
}
