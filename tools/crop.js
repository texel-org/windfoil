// crop.js — magnified detail crops of a whole folder of renders, for figures. (deno task crop)
//
// A 512px coverage render shows the shape; it does not show the pixels. This magnifies a small region of
// each image so a figure can show the same detail of every renderer at a size where individual pixel values
// are readable. Output is the SAME SIZE as the source, so a crop drops straight into a figure beside the
// full renders without rescaling anything.
//
// The zoom comes from the crop being small, not from the output being big:
//
//     source 512×512  ──×2──▶  1024×1024 (conceptually)  ──window 512×512 at 40px inset──▶  output 512×512
//
// so at --zoom 2 the window covers a 256×256 region of the source — the upper-right quadrant — and at
// --zoom 4 a 128×128 region. There is only ever ONE resampling step: `upscaleCrop` reads the source pixels
// the window needs and writes them magnified, so the full upscale is never built and nothing is downscaled.
// Because the window is written 1:1, --inset is the same number of pixels in the upscaled image and in the
// output.
//
//   deno task crop --dir output/comparison/svg_shape8
//     512×512 ×2 → window 512×512, its top-right corner 40px inside the upscaled image's
//     → written as <name>_crop.png beside each source
//
// Re-runnable: anything already named *_crop is skipped, so a second run adds only what is new.
//
// Flags: --dir <folder> (or a bare path) · --zoom <n> (2) · --size <px> (the source's own size)
//        --inset <px> (40) · --corner tr|tl|br|bl (tr) · --recursive · --force
//
// Scaling is NEAREST-NEIGHBOUR at an INTEGER zoom. Both matter for a paper: nearest-neighbour means every
// output pixel is one exact coverage value the renderer produced rather than an interpolation invented
// afterwards, and an integer factor keeps every source pixel the same size on the page.

import { createCanvas, loadImage } from '@napi-rs/canvas';
import { encodePNG } from '../src/png.js';
import { upscaleCrop } from './common/images.js';
import { args } from './common/args.js';

const IMAGE_EXT = /\.(png|jpe?g|webp)$/i;
const SVG_EXT = /\.svg$/i;
const CROPPED = /_crop$/i;

// An SVG gets the SAME window, expressed as a viewBox instead of pixels — so rendering scene_crop.svg in
// Figma or a browser at the output size gives the region the raster crops show, and the comparison survives
// cropping. Without this the SVG would still cover the whole frame and could not be put beside the crops.
function cropSVG(src, { zoom, outSize, inset, corner }) {
  const root = src.match(/<svg\b[^>]*>/i)?.[0];
  if (!root) return null;
  const attr = (n) => root.match(new RegExp(`\\b${n}\\s*=\\s*"([^"]*)"`, 'i'))?.[1];
  const vb = (attr('viewBox')?.match(/[+-]?(?:\d*\.\d+|\d+\.?)(?:[eE][+-]?\d+)?/g) ?? []).map(Number);
  const w = parseFloat(attr('width')) || (vb.length === 4 ? vb[2] : 0);
  const h = parseFloat(attr('height')) || (vb.length === 4 ? vb[3] : 0);
  if (!(w > 0 && h > 0)) return null;

  const W = w * zoom, H = h * zoom;
  const cw = outSize > 0 ? outSize : w, ch = outSize > 0 ? outSize : h;
  if (cw + inset > W || ch + inset > H) return { tooSmall: true, W, H, cw, ch };
  const cx = corner === 'tr' || corner === 'br' ? W - cw - inset : inset;
  const cy = corner === 'tr' || corner === 'tl' ? inset : H - ch - inset;

  // window → document px → viewBox units (they differ if width/height disagree with the viewBox)
  const kx = vb.length === 4 ? vb[2] / w : 1, ky = vb.length === 4 ? vb[3] / h : 1;
  const vx = (vb.length === 4 ? vb[0] : 0) + (cx / zoom) * kx;
  const vy = (vb.length === 4 ? vb[1] : 0) + (cy / zoom) * ky;
  const vw = (cw / zoom) * kx, vh = (ch / zoom) * ky;
  const num = (n) => (Number.isInteger(n) ? `${n}` : `${+n.toFixed(4)}`);

  let out = root
    .replace(/\bviewBox\s*=\s*"[^"]*"/i, `viewBox="${num(vx)} ${num(vy)} ${num(vw)} ${num(vh)}"`)
    .replace(/\bwidth\s*=\s*"[^"]*"/i, `width="${num(cw)}"`)
    .replace(/\bheight\s*=\s*"[^"]*"/i, `height="${num(ch)}"`);
  if (!/viewBox/i.test(out)) out = out.replace(/<svg\b/i, `<svg viewBox="${num(vx)} ${num(vy)} ${num(vw)} ${num(vh)}"`);
  return { svg: src.replace(root, out), sx: cx / zoom, sy: cy / zoom, sw: cw / zoom, sh: ch / zoom, cw, ch };
}

const argv = args(Deno.args);
const dir = argv.string('dir') ?? argv.positionals()[0];
const zoom = Math.max(1, Math.round(argv.number('zoom', 2)));
const outSize = argv.number('size', 0); // 0 ⇒ match each source image's own dimensions
const inset = argv.number('inset', 40);
const corner = (argv.string('corner', 'tr') ?? 'tr').toLowerCase();
const recursive = argv.has('recursive');
const force = argv.has('force');

if (!dir) {
  console.error(
    'usage: deno task crop --dir <folder> [--zoom 2] [--size <px>] [--inset 40]\n' +
      '                     [--corner tr|tl|br|bl] [--recursive] [--force]',
  );
  Deno.exit(1);
}
if (!['tr', 'tl', 'br', 'bl'].includes(corner)) {
  console.error(`--corner must be one of tr, tl, br, bl (got "${corner}")`);
  Deno.exit(1);
}

async function* walk(root) {
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
  files = [];
  for await (const f of walk(dir)) files.push(f);
} catch (err) {
  console.error(`cannot read folder "${dir}": ${err?.message ?? err}`);
  Deno.exit(1);
}
files.sort();
if (!files.length) {
  console.error(`no images to crop in "${dir}"${recursive ? '' : ' (try --recursive)'}`);
  Deno.exit(1);
}

console.log(
  `crop · ${files.length} image(s) in ${dir} · zoom ×${zoom} · inset ${inset}px · corner ${corner} · ` +
    `output ${outSize ? `${outSize}px` : 'same as source'} · nearest-neighbour`,
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

  if (isSVG) {
    const r = cropSVG(await Deno.readTextFile(path), { zoom, outSize, inset, corner });
    if (!r || r.tooSmall) {
      console.log(`  skip  ${path.split('/').pop()} — ${r ? 'window does not fit' : 'no usable <svg> size'}`);
      skipped++;
      continue;
    }
    Deno.writeFileSync(outPath, new TextEncoder().encode(r.svg));
    wrote++;
    console.log(
      `  ${path.split('/').pop().padEnd(22)} viewBox → out ${r.cw}×${r.ch}` +
        `  covers source (${r.sx},${r.sy}) ${r.sw}×${r.sh}`,
    );
    continue;
  }

  const img = await loadImage(path);
  const w = img.width, h = img.height;
  const ctx = createCanvas(w, h).getContext('2d');
  ctx.drawImage(img, 0, 0);
  const rgba = ctx.getImageData(0, 0, w, h).data;

  const W = w * zoom, H = h * zoom; // the upscaled image, never actually built
  const cw = outSize > 0 ? outSize : w;
  const ch = outSize > 0 ? outSize : h;

  if (cw + inset > W || ch + inset > H) {
    console.log(
      `  skip  ${path.split('/').pop()} — a ${cw}×${ch} window inset ${inset}px does not fit in ` +
        `${W}×${H} (raise --zoom or lower --size/--inset)`,
    );
    skipped++;
    continue;
  }
  // Inset from the chosen corner: the window's own corner sits `inset` px inside the upscaled image's.
  const cx = corner === 'tr' || corner === 'br' ? W - cw - inset : inset;
  const cy = corner === 'tr' || corner === 'tl' ? inset : H - ch - inset;

  Deno.writeFileSync(outPath, encodePNG(upscaleCrop(rgba, w, h, zoom, cx, cy, cw, ch), cw, ch));
  wrote++;
  // Report the covered SOURCE region too — that is what says which part of the picture a figure is showing.
  const sx = cx / zoom, sy = cy / zoom, sw = cw / zoom, sh = ch / zoom;
  console.log(
    `  ${path.split('/').pop().padEnd(22)} ${w}×${h} ×${zoom} → out ${cw}×${ch}` +
      `  covers source (${sx},${sy}) ${sw}×${sh}`,
  );
}

console.log(
  `\nwrote ${wrote} *_crop.png${skipped ? `, skipped ${skipped}` : ''}` +
    `${skipped && !force ? ' (already cropped — use --force to redo)' : ''}`,
);
