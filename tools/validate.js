// validate.js — one number: mean |Δ| from ground truth, over a dataset. (deno task validate)
//
// The comparison tool (tools/comparison.js) shows what ONE shape looks like out of each rasteriser. This is
// the other half: the same renderers over a whole corpus, reduced to a figure per renderer that can be
// quoted. Both quote the same quantity under the same names, so the two tables read together.
//
// The corpus is the printable-ASCII range of the bundled font plus the synthetic stress shapes, and — kept
// separate, see below — the winding-fold cases from tools/failure.js.
//
// The figure is mean |Δ| over EVERY pixel of the cell. Restricting it to the anti-aliased band — the pixels
// the truth puts strictly between empty and full — is tempting, because that is where the error usually is
// and because the band mean does not change with render size. But it can only see errors that land on an
// edge, and a renderer's worst failures are often the ones that do not: a hole that fills in, a stray blob,
// interior winding that fades out. The winding-fold cases here are exactly that shape — 431 band pixels, yet
// 8,322 pixels more than one code value out — so a band mean misses ~99% of what goes wrong there. The cell
// is a fixed CELL×CELL for every shape (../common/shapes.js), so a whole-image mean is directly comparable
// across renderers and shapes; it is only across RENDER SIZES that it would not be, and nothing here varies
// that. The band mean is still measured — `--full` and stats.json carry it, next to the whole-image one.
//
// `--full` prints everything underneath: a row per shape, and per-group tables including the control rows
// that say what this measurement's own precision can certify (tools/common/renderers.js). Those controls run
// on every invocation whether or not they are printed — if the reference grid is too coarse to support the
// number, the default output says so instead of quoting it.
//
//   deno task validate                    # the number
//   deno task validate --full             # …with every shape and every control row
//   deno task validate --font ~/x.ttf     # the same corpus in another font
//   deno task validate --samples 192      # a coarser (faster) reference grid
//   deno task validate --exact            # render windfoil with the shader's EXACT_MODE override
//   deno task validate --images all|none  # PNG panels for every shape / for none (default: the non-glyphs)

import { loadFont } from '../src/font.js';
import { requestDevice } from '../src/gpu.js';
import { encodePNG } from '../src/png.js';
import { createCanvas } from '@napi-rs/canvas';
import { AMP, diffRGBA, grayRGBA, upscale } from './common/images.js';
import { args } from './common/args.js';
import { aggregate, F, S, validationSuite } from './validate/harness.js';

const argv = args(Deno.args);
const samples = argv.number('samples', F);
const exact = argv.has('exact');
const full = argv.has('full') || argv.has('glyphs');
const images = argv.string('images', 'shapes'); // none | shapes | all
const fontPath = argv.string('font');
const fontURL = fontPath
  ? new URL(fontPath, `file://${Deno.cwd()}/`)
  : new URL('../assets/Lato-Regular.ttf', import.meta.url);
const fontName = decodeURIComponent(fontURL.pathname).split('/').pop().replace(/\.[ot]tf$/i, '');

const font = await loadFont(fontURL);
const device = await requestDevice();
const suite = validationSuite({
  font,
  device,
  createContext2D: (w, h) => createCanvas(w, h).getContext('2d'),
  samples,
  exact,
});

// The renderers actually on trial: everything that is neither the reference nor a control row.
const measured = suite.renderers.filter((r) => !r.reference && !r.control);
const int = (n) => n.toLocaleString('en-US');
const e2 = (v) => v.toExponential(2);

// `corpus` is what the headline is over; the fold cases are pooled apart. They are adversarial by
// construction (docs/ALGORITHM.md §4/§8), so averaging them in would let a
// handful of hand-made pathologies set most of a number that is supposed to describe ordinary drawing. They
// are not dropped — the default output prints them on their own line, right underneath.
const corpus = aggregate(suite.renderers);
const folds = aggregate(suite.renderers);
const groups = { glyph: aggregate(suite.renderers), stress: aggregate(suite.renderers), fold: folds };

console.log(
  `validate · ${fontName} · ${S}px cell · truth = ${samples}² samples/px` +
    `${exact ? ' · windfoil = EXACT_MODE (no winding fold)' : ''}`,
);

// ── run ──────────────────────────────────────────────────────────────────────────────────────────────────
const shapeRows = [];
const panels = [];
const offenders = []; // shapes with a pixel more than one 8-bit code value out, for --full
for (const shape of suite.shapes) {
  const results = await suite.render(shape);
  groups[shape.group].add(results);
  if (shape.group !== 'fold') corpus.add(results);
  const off = results.find((r) => r.name === 'windfoil').stats.off;
  if (off) offenders.push({ ...shape, off });
  if (images === 'all' || (images === 'shapes' && shape.group !== 'glyph')) panels.push({ shape, results });
  if (full) shapeRows.push({ shape, results });
}

// ── the number ───────────────────────────────────────────────────────────────────────────────────────────
const row = (agg, name) => agg.rows().find((r) => r.name === name);
const e6 = (v) => v.toFixed(6);
console.log(
  `\ndataset · ${corpus.n} shapes: ${groups.glyph.n} printable-ASCII glyphs + ${groups.stress.n} synthetic ` +
    `stress shapes\n${''.padEnd(11)}${int(corpus.n * S * S)} pixels (${S}×${S} per shape)\n`,
);
console.log(`${'renderer'.padEnd(11)}mean |Δ| from ground truth`);
for (const r of measured) console.log(`${r.name.padEnd(11)}${e6(row(corpus, r.name).mean).padStart(16)}`);

console.log(
  `\nmean |Δ| over every pixel of a fixed ${S}×${S} cell, so it counts a wrong interior or exterior pixel — a` +
    `\nfilled hole, a stray blob, interior winding that fades — and not only the anti-aliased edge.`,
);
console.log(
  `\nThe ${folds.n} winding-fold cases are measured but kept out of that mean: adversarial by construction` +
    `\n(docs/ALGORITHM.md §4/§8), and averaged in they would set most of it. There:` +
    `\n${measured.map((r) => `${r.name} ${e6(row(folds, r.name).mean)}`).join(' · ')}`,
);

// The controls run whether or not they are printed. If the reference grid is too coarse to resolve what the
// table claims, say so here rather than let the number be quoted — see tools/common/renderers.js.
const ref = row(corpus, 'truth/2');
if (ref.off > 0) {
  console.log(
    `\n  ! the reference is NOT converged at --samples ${samples}: halving its grid moves ${int(ref.off)} pixels by` +
      `\n    more than one 8-bit code value, so part of the number above is the sample, not the renderers.` +
      `\n    Raise --samples until this warning goes away before quoting it.`,
  );
}

if (!full) console.log(`\n--full for a row per shape, the control rows, and what they certify.`);

// ── --full: everything underneath ────────────────────────────────────────────────────────────────────────
if (full) {
  const e1 = (v) => v.toExponential(1).padStart(7);
  const f4 = (v) => v.toFixed(4).padStart(8);
  const centre = (s, w) => s.padStart(Math.floor((w + s.length) / 2)).padEnd(w);
  console.log(
    `\n${''.padEnd(24)} ${measured.map((r) => `  ${centre(r.name, 16)}`).join('')}\n` +
      `${''.padEnd(24)} ${measured.map(() => `  ${'whole'.padStart(7)} ${'max'.padStart(8)}`).join('')}`,
  );
  for (const { shape, results } of shapeRows) {
    console.log(
      `${(shape.label + (shape.group === 'fold' ? ' †' : '')).padEnd(24)} ` +
        measured.map((r) => {
          const s = results.find((x) => x.name === r.name).stats;
          return `  ${e1(s.mean)} ${f4(s.max)}`;
        }).join(''),
    );
  }

  const W = 50;
  const signed = (v) => `${v >= 0 ? '+' : ''}${v.toExponential(1)}`.padStart(9);
  // `whole` leads — it is the headline quantity. `AA-px` beside it is the band-only mean: where the two
  // diverge, the error is NOT on the anti-aliased edge, which is the interesting case (see the † block).
  const HEAD = `${''.padEnd(9)} ${''.padEnd(W)} ${'whole'.padStart(9)} ${'AA-px'.padStart(8)} ` +
    `${'bias'.padStart(9)} ${'max'.padStart(7)} ${'px >1/255'.padStart(9)}`;
  const TITLES = {
    glyph: `printable ASCII of ${fontName}`,
    stress: 'synthetic stress shapes',
    fold: 'winding-fold limit cases † (windfoil is EXPECTED to deviate — docs/ALGORITHM.md §4/§8)',
  };
  for (const [key, title] of Object.entries(TITLES)) {
    const agg = groups[key], rows = agg.rows();
    console.log(`\n── ${title} ${'─'.repeat(Math.max(2, 99 - title.length))}`);
    console.log(
      `   ${agg.n} shapes · ${int(rows[0].band)} anti-aliased px of ${int(agg.n * S * S)}`,
    );
    console.log(HEAD);
    for (const r of rows) {
      console.log(
        `${r.name.padEnd(9)} ${r.title.slice(0, W).padEnd(W)} ${e2(r.mean).padStart(9)} ` +
          `${e2(r.bandMean).padStart(8)} ${signed(r.bias)} ${r.max.toFixed(4).padStart(7)} ` +
          `${String(r.off).padStart(9)}${r.control ? '  ·' : ''}`,
      );
    }
  }

  const wf = row(corpus, 'windfoil'), floor = row(corpus, '8-bit');
  const named = offenders.filter((s) => s.group !== 'fold').sort((a, b) => b.off - a.off)
    .slice(0, 4).map((s) => `${s.label} (${s.off})`).join(', ');
  console.log(
    `\n· = a control row: not a renderer on trial, but a floor under the ones that are.` +
      `\n  truth/2  the reference measured against a coarser copy of itself — has it converged?` +
      `\n  8-bit    the exact box filter through the same readback — what a PERFECT renderer scores here.` +
      `\n  binary   one sample at the pixel centre — the scale of the thing being measured.` +
      `\n\nwindfoil's ${e2(wf.mean)} sits on the 8-bit row's ${e2(floor.mean)} — the score the exact box filter` +
      `\nITSELF gets once quantised the way the readback quantises every renderer here. ${
        wf.off === 0
          ? `No pixel of the corpus lands\nmore than one code value from the reference, so`
          : `All but ${int(wf.off)} pixels of\n${int(wf.band)} land within one code value of the reference, so away from those`
      } at the precision the` +
      `\noutput has this cannot separate windfoil from the ideal box filter: it certifies that any gap is under` +
      `\nthe 8-bit quantum, not that it is zero.` +
      (named ? `\nThe exceptions outside the † rows: ${named} — the winding fold at a sub-pixel self-intersection.` : '') +
      `\n\nslug and skia are measured against the same target, which is not a fidelity ranking: each aims at` +
      `\nits own reconstruction, and on the † rows skia is the more correct renderer.`,
  );
}

// ── PNG panels + machine-readable stats ──────────────────────────────────────────────────────────────────
const outDir = new URL('../output/validation/', import.meta.url);
Deno.mkdirSync(outDir, { recursive: true });
const Z = 4, C = S * Z; // 4× nearest-neighbour upscale so individual pixels stay crisp
const write = (name, rgba) =>
  Deno.writeFileSync(new URL(`${name}.png`, outDir), encodePNG(upscale(rgba, S, S, Z), C, C));

let files = 0;
for (const { shape, results } of panels) {
  const reference = results.find((r) => r.reference);
  for (const r of results) {
    if (r.control) continue; // the control rows are calibration, not pictures
    write(`${shape.slug}_${r.name}`, grayRGBA(r.cov));
    if (!r.reference) write(`${shape.slug}_${r.name}_diff`, diffRGBA(r.cov, reference.cov));
    files += r.reference ? 1 : 2;
  }
}

Deno.writeFileSync(
  new URL('stats.json', outDir),
  new TextEncoder().encode(JSON.stringify({
    font: fontName,
    cell: S,
    samples,
    exact,
    corpus: { shapes: corpus.n, renderers: corpus.rows() },
    groups: Object.fromEntries(Object.entries(groups).map(([k, a]) => [k, { shapes: a.n, renderers: a.rows() }])),
  }, null, 2) + '\n'),
);

console.log(
  `\nwrote ${files ? `${files} PNGs + ` : ''}stats.json to ${Deno.realPathSync(outDir)}` +
    (files ? `  (<shape>_<renderer>.png and _diff.png ×${AMP}, at ${Z}× zoom)` : ''),
);
