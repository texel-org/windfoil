// scenes.js — turn a scene SPEC (one string, e.g. from `--scene`) into geometry at a given render size.
//
// A scene is just { label, slug, quads, evenodd }: flat quads (../common/shapes.js) already in pixel
// coordinates for the requested size, so every renderer takes it as-is. Keeping the resolution in one place
// is what lets a demo swap what it draws without touching how it draws it.
//
// Three families of spec:
//   glyph:<char>   one glyph of the loaded font, scaled to fit the frame  (e.g. `glyph:G`, `glyph:@`)
//   shape:<name>   any shape from the validation dataset, scaled up from its authoring cell — the stress
//                  shapes, the winding-fold failure cases, and every lowercase letter (e.g. `shape:circle
//                  r=44`, `shape:fold A ±1 cancellation`, `shape:star {5/2} even-odd`). Names match on the
//                  label or its slug, so `shape:star_5_2_even_odd` works too.
//   svg:<file>     an SVG file's filled geometry, fitted to the frame (see ./svg-parse.js for exactly what
//                  is read and what is ignored). A bare path ending in .svg also works.
//
// To add a family, add a case here — everything downstream (renderers, diffs, SVG, filenames) is generic.

import { buildShapes, CELL, glyphShape, scaleQuads, translateQuads } from './shapes.js';
import { slugify } from './images.js';
import { parseSVG } from './svg-parse.js';

/** Every spec this resolver accepts for the given font, as printable lines. */
export function listScenes(font) {
  return [
    `glyph:<char>   any character of the loaded font, e.g. glyph:G`,
    `svg:<file>     an SVG file, e.g. svg:./art/rosette.svg  (or just the path, if it ends in .svg)`,
    ...buildShapes(font).map(({ label }) => `shape:${label}`),
  ];
}

/**
 * @param {string} spec  see the families above; a bare single character is shorthand for `glyph:<char>`
 * @param {object} o
 * @param {object} o.font    a parsed font (src/font.js)
 * @param {number} o.size    the square render size in px — the scene is built to fill it
 * @param {number} [o.offset] translate the whole scene by this many px in x and y, AFTER scaling.
 *   What a rasteriser does at an edge depends on where that edge falls between sample points, so a shape's
 *   sub-pixel phase is part of the test, not a detail. The `shape:` cases are authored to put their critical
 *   edge on a pixel CENTRE in the 128px cell (the half-integer coordinates in ./shapes.js) — but scaling to
 *   another size moves it: at size 256 every one of those .5s doubles to a whole number, landing the edge on
 *   a pixel BOUNDARY, where the geometry is pixel-aligned and every renderer is trivially exact. `--offset
 *   0.5` puts it back on a centre. If a comparison comes out all zeros, this is usually why.
 * @param {'viewbox'|'ink'} [o.fit] `svg:` only — fit the file's viewBox (default) or its ink bbox
 * @returns {Promise<{ label, slug, quads, evenodd, warnings? }>}
 */
export async function resolveScene(spec, { font, size, offset = 0, fit = 'viewbox' }) {
  const bare = spec.includes(':') ? null : spec;
  const kind = bare
    ? (bare.toLowerCase().endsWith('.svg') ? 'svg' : [...bare].length === 1 ? 'glyph' : 'shape')
    : spec.slice(0, spec.indexOf(':'));
  const rest = bare ?? spec.slice(spec.indexOf(':') + 1);
  const place = (quads) => translateQuads(quads, offset);

  if (kind === 'svg') {
    const url = new URL(rest, `file://${Deno.cwd()}/`);
    const parsed = parseSVG(await Deno.readTextFile(url), { size, fit, pad: 0 });
    const base = decodeURIComponent(url.pathname).split('/').pop().replace(/\.svg$/i, '');
    return {
      label: `svg '${base}' (${parsed.elements} element${parsed.elements === 1 ? '' : 's'}, ` +
        `viewBox ${parsed.viewBox.join(' ')})`,
      slug: `svg_${slugify(base) || 'scene'}`,
      quads: place(parsed.quads),
      evenodd: parsed.evenodd,
      warnings: parsed.warnings,
    };
  }

  if (kind === 'glyph') {
    if ([...rest].length !== 1) throw new Error(`glyph scene needs exactly one character, got "${rest}"`);
    // The codepoint is always in the slug: most interesting glyphs slugify to nothing ('@', 'ø'), and case
    // alone would collide ('G' and 'g' are different scenes that must not share an output folder).
    const hex = rest.codePointAt(0).toString(16).padStart(4, '0');
    const name = slugify(rest);
    return { label: `glyph '${rest}' U+${hex.toUpperCase()}`, slug: `glyph_${name ? `${name}_` : ''}u${hex}`,
      quads: place(glyphShape(font, rest, size)), evenodd: false };
  }

  if (kind === 'shape') {
    const want = slugify(rest);
    // Match on the label first, then on the dataset's own slug — the two differ where a label would slugify
    // ambiguously ('a' and 'A' both give `glyph_a`), which is exactly why shapes carry an explicit slug.
    const hit = buildShapes(font).find((s) => s.label === rest || s.slug === rest || s.slug === want);
    if (!hit) {
      throw new Error(`no shape named "${rest}" — run with --list to see them all`);
    }
    // Authored in the CELL cell; a uniform scale takes it to the render size. The comparison always FILLS,
    // so the `segments` of the stroked validate variants are ignored — they describe the same geometry.
    return { label: hit.label, slug: hit.slug,
      quads: place(scaleQuads(hit.quads, size / CELL)), evenodd: hit.evenodd ?? false };
  }

  throw new Error(`unknown scene kind "${kind}" — expected glyph:<char> or shape:<name> (--list to see them)`);
}
