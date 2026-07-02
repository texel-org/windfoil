// png.js — encode an RGBA8 pixel buffer to a PNG using @mattdesl's `png-tools`.
//
// png-tools keeps the (de)compressor pluggable so it stays runtime-agnostic; we hand it Deno's built-in
// zlib. `None` scanline filtering is both the cheapest to apply and the smallest for flat / vector-ish art
// (solid fills deflate into long runs), which is exactly what this demo produces.

import { encode, ColorType, FilterMethod } from 'png-tools';
import { deflateSync } from 'node:zlib';

/** Encode `rgba` (Uint8Array, width*height*4, straight alpha) to a PNG byte array. */
export function encodePNG(rgba, width, height) {
  return encode(
    {
      data: rgba,
      width,
      height,
      depth: 8,
      colorType: ColorType.RGBA,
      filter: FilterMethod.None,
    },
    (buf, opts) => deflateSync(buf, opts),
  );
}
