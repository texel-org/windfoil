// browser.js — the browser boot for the validation suite: the same shared harness as `deno task validate`,
// but the host canvas is THIS browser's own canvas2d rasterizer (Skia in Chrome, CoreGraphics in Safari,
// WebRender in Firefox), so the page shows how the engine's AA sits against the box filter next to ours.
// Results stream in shape by shape: a stats table up top, then a panel of coverage + error maps per shape.
//
// Same renderers, same controls, same numbers as the Deno boot (tools/common/renderers.js) — only the host
// canvas and the presentation differ.
//
// Serve from the repo ROOT so /src/*.js and /assets/*.ttf resolve (WebGPU needs a secure context —
// localhost counts):  `deno task serve`  then open  http://localhost:8080/tools/validate/
//
//   ?exact          render windfoil with the shader's EXACT_MODE override
//   ?samples=192    a coarser (faster) reference grid; the truth/2 row says whether it still holds up
//   ?scope=shapes   synthetic stress + fold shapes only, skipping the ASCII range (a much quicker run)
//   ?rows           a table row per glyph as well (default: the non-glyph rows and the worst three glyphs)
//   ?panels         an image panel per glyph as well (default: the non-glyph shapes — 94 more panels is
//                   ~660 more canvases, and the glyph rows are what the aggregate is for)

import { loadFont } from '../../src/font.js';
import { requestDevice } from '../../src/gpu.js';
import { aggregate, AMP, diffRGBA, F, grayRGBA, S, validationSuite } from './harness.js';

const $ = (sel) => document.querySelector(sel);
const el = (tag, className, text) => {
  const e = document.createElement(tag);
  if (className) e.className = className;
  if (text !== undefined) e.textContent = text;
  return e;
};
const e2 = (v) => v.toExponential(2);
const f4 = (v) => v.toFixed(4);

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

const params = new URLSearchParams(location.search);
const flag = (name) => params.has(name) && params.get(name) !== 'false';
const exact = flag('exact');
const samples = Number(params.get('samples')) || F;

try {
  const device = await requestDevice();
  const font = await loadFont(new URL('../../assets/Lato-Regular.ttf', import.meta.url));
  const suite = validationSuite({
    font,
    device,
    samples,
    exact,
    // In a browser this is not @napi-rs/canvas but whatever the engine ships, so the entry says so.
    canvas: { name: 'canvas', title: `this browser's canvas2d` },
    createContext2D: (w, h) => {
      const c = el('canvas');
      c.width = w;
      c.height = h;
      return c.getContext('2d', { willReadFrequently: true });
    },
  });

  // The renderers on trial (everything that is neither the reference nor a control row) drive the per-shape
  // table columns; the controls appear in the aggregate, where they belong.
  const measured = suite.renderers.filter((r) => !r.reference && !r.control);
  const head = $('#stats thead');
  head.replaceChildren();
  head.append(
    (() => {
      const tr = el('tr');
      tr.append(el('th', '', 'shape'), ...measured.map((r) => {
        const th = el('th', '', r.name);
        th.colSpan = 2;
        return th;
      }));
      return tr;
    })(),
    (() => {
      const tr = el('tr');
      tr.append(el('th'), ...measured.flatMap(() => [el('th', '', 'mean'), el('th', '', 'max')]));
      return tr;
    })(),
  );

  $('#params').textContent = `${S}px cell · truth = ${samples}×${samples} point-sampled box filter · ` +
    `canvas = this browser's canvas2d${exact ? ' · windfoil = EXACT_MODE (no winding fold)' : ''}`;

  const GROUPS = {
    glyph: 'printable ASCII of Lato-Regular',
    stress: 'synthetic stress shapes',
    fold: 'winding-fold limit cases †',
  };
  const aggs = Object.fromEntries(Object.keys(GROUPS).map((k) => [k, aggregate(suite.renderers)]));
  const shapes = suite.shapes.filter((s) => s.group !== 'glyph' || params.get('scope') !== 'shapes');
  const statOf = (results, name) => results.find((r) => r.name === name).stats;

  const statsRow = (label, results, fold) => {
    const tr = el('tr', fold ? 'fold' : '');
    tr.append(
      el('td', '', label + (fold ? ' †' : '')),
      ...measured.flatMap((r) => [
        el('td', '', e2(statOf(results, r.name).mean)),
        el('td', '', f4(statOf(results, r.name).max)),
      ]),
    );
    return tr;
  };

  const shapePanel = (shape, results) => {
    const fold = shape.group === 'fold';
    const panel = el('section', 'panel');
    panel.append(el('h2', '', shape.label + (fold ? ' † (winding-fold limit — deviation expected)' : '')));
    panel.append(el(
      'div',
      'panel-stats',
      measured.map((r) => {
        const s = statOf(results, r.name);
        return `${r.name}: ${e2(s.mean)} max ${f4(s.max)} · ${s.off} px >1/255`;
      }).join(' — '),
    ));
    const row = el('div', 'row');
    const reference = results.find((r) => r.reference);
    // The coverage renders (control rows are calibration, not pictures), then one error map each.
    for (const r of results) if (!r.control) row.append(imageCell(grayRGBA(r.cov), r.name));
    for (const r of measured) {
      const cov = results.find((x) => x.name === r.name).cov;
      row.append(imageCell(diffRGBA(cov, reference.cov), `|${r.name}−truth| ×${AMP}`));
    }
    panel.append(row);
    return panel;
  };

  const glyphs = []; // held back so the table can show the worst few rather than 94 rows
  let done = 0;
  for (const shape of shapes) {
    const results = await suite.render(shape);
    aggs[shape.group].add(results);
    const isGlyph = shape.group === 'glyph';
    if (isGlyph && !params.has('rows')) glyphs.push({ shape, results });
    else $('#stats tbody').append(statsRow(shape.label, results, shape.group === 'fold'));
    if (!isGlyph || params.has('panels')) $('#panels').append(shapePanel(shape, results));

    $('#status').textContent = `running… ${++done} / ${shapes.length} shapes`;
    await new Promise(requestAnimationFrame); // let the new row paint before the next shape blocks the thread
  }
  if (glyphs.length) {
    const worst = (e) => statOf(e.results, 'windfoil').mean;
    const tr = el('tr');
    const td = el('td', '', `${glyphs.length} ASCII glyphs — worst 3 by windfoil mean (?rows for one each):`);
    td.colSpan = 1 + measured.length * 2;
    tr.append(td);
    $('#stats tbody').append(tr);
    for (const g of glyphs.slice().sort((a, b) => worst(b) - worst(a)).slice(0, 3)) {
      $('#stats tbody').append(statsRow(g.shape.label, g.results, false));
    }
  }

  // Aggregates: one block per group, controls included — the whole point of the table (see harness.js).
  const out = el('div', 'aggregates');
  for (const [key, title] of Object.entries(GROUPS)) {
    const agg = aggs[key];
    if (!agg.n) continue;
    const rows = agg.rows();
    out.append(el('h2', '', `${title} · ${agg.n} shapes, ${rows[0].band.toLocaleString('en-US')} anti-aliased px`));
    const table = el('table', 'agg');
    const hdr = el('tr');
    // whole-cell mean first — it is the headline quantity, and unlike the band mean beside it, it counts a
    // wrong interior or exterior pixel too (see tools/validate.js).
    hdr.append(...['', 'what it is', 'mean |Δ|', 'AA-px', 'max |Δ|', 'px >1/255'].map((t) => el('th', '', t)));
    table.append(hdr);
    for (const r of rows) {
      const tr = el('tr', r.control ? 'control' : '');
      tr.append(...[r.name, r.title, e2(r.mean), e2(r.bandMean), f4(r.max), String(r.off)].map((t) => el('td', '', t)));
      table.append(tr);
    }
    out.append(table);
  }
  $('#panels').before(out);

  const wf = aggs.glyph.rows().find((r) => r.name === 'windfoil');
  const floor = aggs.glyph.rows().find((r) => r.name === '8-bit');
  const summary = wf
    ? `done · ${GROUPS.glyph}: windfoil ${e2(wf.mean)} against an 8-bit floor of ${e2(floor.mean)}, ` +
      `${wf.off} px more than one code value from the reference`
    : `done · ${aggs.stress.n + aggs.fold.n} shapes`;
  $('#status').textContent = summary;
  console.log(`validate: ${summary}`); // greppable from headless runs
} catch (err) {
  $('#status').textContent = `error: ${err?.message ?? err}`;
  console.error(err);
  throw err;
}
