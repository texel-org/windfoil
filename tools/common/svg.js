// svg.js — emit a scene as SVG, so a conformant SVG engine (a browser, Figma, Illustrator, resvg) becomes
// one more renderer in the comparison. The geometry is written out losslessly as quadratics — the same
// representation the shader consumes — so what the SVG engine rasterises is exactly the shape every other
// renderer here was handed, not an approximation of it.

const num = (n) => (Number.isInteger(n) ? `${n}` : `${+n.toFixed(4)}`);

/**
 * Flat quads [x0,y0,cx,cy,x1,y1,...] as an SVG path `d` string: each contour opens with M at its start, then
 * one Q per edge (straight edges are exact degenerate quadratics, control point at the midpoint). A new
 * contour starts wherever an edge's start point doesn't continue the previous edge's end.
 */
export function svgPath(quads) {
  let d = '', px = null, py = null;
  for (let i = 0; i < quads.length; i += 6) {
    const x0 = quads[i], y0 = quads[i + 1], cx = quads[i + 2], cy = quads[i + 3];
    const x1 = quads[i + 4], y1 = quads[i + 5];
    if (px === null || Math.abs(x0 - px) > 1e-4 || Math.abs(y0 - py) > 1e-4) {
      d += `${d ? 'Z ' : ''}M ${num(x0)} ${num(y0)} `;
    }
    d += `Q ${num(cx)} ${num(cy)} ${num(x1)} ${num(y1)} `;
    px = x1;
    py = y1;
  }
  return `${d}Z`;
}

/**
 * A standalone SVG document of one shape.
 *
 * @param {object} o
 * @param {number[]} o.quads      flat quads in pixel coordinates (Y-down, origin top-left — SVG's own space)
 * @param {number} o.width  @param {number} o.height   the viewBox, in the same pixel units
 * @param {boolean} [o.evenodd]   fill rule
 * @param {string} [o.fill]       shape colour (default white, matching the coverage PNGs)
 * @param {string|null} [o.background] backdrop rect colour, or null for a transparent document
 * @param {string} [o.comment]    a note written into the file as an XML comment
 */
export function svgDocument({ quads, width, height, evenodd = false, fill = '#fff', background = '#000', comment }) {
  const rule = evenodd ? 'evenodd' : 'nonzero';
  return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${num(width)} ${num(height)}" ` +
    `width="${num(width)}" height="${num(height)}">\n` +
    (comment ? `  <!-- ${comment} · fill-rule: ${rule} -->\n` : '') +
    (background ? `  <rect width="100%" height="100%" fill="${background}"/>\n` : '') +
    `  <path d="${svgPath(quads)}" fill="${fill}" fill-rule="${rule}"/>\n` +
    `</svg>\n`;
}
