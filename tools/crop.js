// crop.js — magnified detail crops of a folder, standalone. (deno task crop)
//
// `deno task report` already crops the scene it reports on; this is the same operation on its own, for a
// folder that is not a staged scene (a figure directory, a set of screenshots). The window maths and the
// SVG viewBox rewrite live in common/cropper.js so both commands cut exactly the same region.
//
//   deno task crop --dir figures/rosette
//     512×512 ×2 → out 512×512 covering a 256×256 source region, 40px inside the top-right
//
// Re-runnable: anything already named *_crop is skipped, so a second run adds only what is new.
//
// Flags: --dir <folder> (or a bare path) · --zoom <n> (2) · --size <px> (the source's own size)
//        --inset <px> (40) · --corner tr|tl|br|bl (tr) · --recursive · --force
//
// Scaling is NEAREST-NEIGHBOUR at an INTEGER zoom: every output pixel is one exact value the renderer
// produced rather than an interpolation invented afterwards, and every source pixel is the same size on the
// page.

import { loadRGBA, writePNGFile } from './common/imageio.js';
import { cropRaster, cropSVG } from './common/cropper.js';
import { args } from './common/args.js';

const IMAGE_EXT = /\.(png|jpe?g|webp)$/i;
const SVG_EXT = /\.svg$/i;
const CROPPED = /_crop$/i;

const argv = args(Deno.args);
const dir = argv.string('dir') ?? argv.positionals()[0];
const recursive = argv.has('recursive');
const force = argv.has('force');
const opts = {
  zoom: Math.max(1, Math.round(argv.number('zoom', 2))),
  outSize: argv.number('size', 0),
  inset: argv.number('inset', 40),
  corner: (argv.string('corner', 'tr') ?? 'tr').toLowerCase(),
};

if (!dir) {
  console.error(
    'usage: deno task crop --dir <folder> [--zoom 2] [--size <px>] [--inset 40]\n' +
      '                     [--corner tr|tl|br|bl] [--recursive] [--force]',
  );
  Deno.exit(1);
}
if (!['tr', 'tl', 'br', 'bl'].includes(opts.corner)) {
  console.error(`--corner must be one of tr, tl, br, bl (got "${opts.corner}")`);
  Deno.exit(1);
}

function* walk(root) {
  for (const e of Deno.readDirSync(root)) {
    const path = `${root}/${e.name}`;
    if (e.isDirectory) {
      if (recursive) yield* walk(path);
    } else if (
      (IMAGE_EXT.test(e.name) || SVG_EXT.test(e.name)) &&
      !CROPPED.test(e.name.replace(IMAGE_EXT, '').replace(SVG_EXT, ''))
    ) {
      yield path;
    }
  }
}

let files;
try {
  files = [...walk(dir)].sort();
} catch (err) {
  console.error(`cannot read folder "${dir}": ${err?.message ?? err}`);
  Deno.exit(1);
}
if (!files.length) {
  console.error(`no images to crop in "${dir}"${recursive ? '' : ' (try --recursive)'}`);
  Deno.exit(1);
}

console.log(
  `crop · ${files.length} file(s) in ${dir} · zoom ×${opts.zoom} · inset ${opts.inset}px · ` +
    `corner ${opts.corner} · output ${opts.outSize ? `${opts.outSize}px` : 'same as source'} · nearest-neighbour`,
);

let wrote = 0, skipped = 0;
for (const path of files) {
  const isSVG = SVG_EXT.test(path);
  const outPath = isSVG ? path.replace(SVG_EXT, '_crop.svg') : path.replace(IMAGE_EXT, '') + '_crop.png';
  if (!force) {
    try {
      Deno.statSync(outPath);
      skipped++;
      continue; // already cropped; --force to redo
    } catch { /* not there yet — go ahead */ }
  }

  const name = path.split('/').pop();
  const r = isSVG
    ? cropSVG(await Deno.readTextFile(path), opts)
    : cropRaster(await loadRGBA(path), opts);
  if (!r || r.tooSmall) {
    console.log(
      `  skip  ${name} — ${r ? `a ${r.cw}×${r.ch} window inset ${opts.inset}px does not fit in ${r.W}×${r.H}` : 'no usable <svg> size'}`,
    );
    skipped++;
    continue;
  }
  if (isSVG) Deno.writeFileSync(outPath, new TextEncoder().encode(r.svg));
  else writePNGFile(outPath, r.rgba, r.cw, r.ch);
  wrote++;
  console.log(`  ${name.padEnd(22)} out ${r.cw}×${r.ch}  covers source (${r.sx},${r.sy}) ${r.sw}×${r.sh}`);
}

console.log(
  `\nwrote ${wrote}${skipped ? `, skipped ${skipped}` : ''}` +
    `${skipped && !force ? ' (already cropped — use --force to redo)' : ''}`,
);
