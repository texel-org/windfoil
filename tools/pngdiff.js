// tools/pngdiff.js — decode two PNGs (as written by src/png.js: RGBA8, filter None) and report pixel-value
// differences (count, max |Δ|, mean |Δ|).
//   deno run -A tools/pngdiff.js a.png b.png
import { inflateSync } from 'node:zlib';

function decodePNG(bytes) {
  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  let off = 8; // signature
  let width = 0, height = 0;
  const idat = [];
  while (off < bytes.length) {
    const len = dv.getUint32(off);
    const type = String.fromCharCode(bytes[off + 4], bytes[off + 5], bytes[off + 6], bytes[off + 7]);
    const data = bytes.subarray(off + 8, off + 8 + len);
    if (type === 'IHDR') {
      width = dv.getUint32(off + 8);
      height = dv.getUint32(off + 12);
      if (bytes[off + 16] !== 8 || bytes[off + 17] !== 6) throw new Error('expected RGBA8');
    } else if (type === 'IDAT') idat.push(data);
    off += 12 + len;
  }
  const zcat = new Uint8Array(idat.reduce((s, d) => s + d.length, 0));
  let zo = 0;
  for (const d of idat) {
    zcat.set(d, zo);
    zo += d.length;
  }
  const raw = inflateSync(zcat);
  const stride = width * 4;
  const out = new Uint8Array(width * height * 4);
  for (let y = 0; y < height; y++) {
    if (raw[y * (stride + 1)] !== 0) throw new Error('expected filter None');
    out.set(raw.subarray(y * (stride + 1) + 1, (y + 1) * (stride + 1)), y * stride);
  }
  return { width, height, data: out };
}

const [aPath, bPath] = Deno.args;
const a = decodePNG(await Deno.readFile(aPath));
const b = decodePNG(await Deno.readFile(bPath));
if (a.width !== b.width || a.height !== b.height) {
  console.log(`size mismatch: ${a.width}x${a.height} vs ${b.width}x${b.height}`);
  Deno.exit(1);
}
let n = 0, maxD = 0, sum = 0;
for (let i = 0; i < a.data.length; i++) {
  const d = Math.abs(a.data[i] - b.data[i]);
  if (d) {
    n++;
    if (d > maxD) maxD = d;
  }
  sum += d;
}
const name = aPath.split('/').pop();
console.log(
  `${name}: ${
    n === 0
      ? 'IDENTICAL'
      : `${n} bytes differ / ${a.data.length} (max ${maxD}, mean |Δ| ${(sum / a.data.length).toFixed(7)})`
  }`,
);
