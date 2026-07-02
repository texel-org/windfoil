# area coverage

> **README stub — to be written.**

A small, standalone demo of a box-filter coverage / anti-aliasing algorithm for 2D vector rendering. It
renders the phrase "area coverage" at a ladder of increasing sizes on the GPU in a single instanced draw,
anti-aliasing each glyph with the method described in [`docs/ALGORITHM.md`](docs/ALGORITHM.md).

![area coverage](output/area-coverage.png)

## Run it

Requires [Deno](https://deno.com/) 2.x on a machine with a WebGPU-capable GPU.

```sh
deno task render     # → output/area-coverage.png
deno task validate   # compare coverage vs a point-sampled box filter and Skia
```

## What's here

- [`docs/ALGORITHM.md`](docs/ALGORITHM.md) — the algorithm.
- [`docs/COMPARISON.md`](docs/COMPARISON.md) — how it relates to other coverage methods.
- [`src/area.wgsl`](src/area.wgsl) — the shader: the winding-integral box filter + the row-band gather.
- [`src/bands.js`](src/bands.js) — the row-band acceleration structure (see ALGORITHM §6).
- [`src/font.js`](src/font.js) — glyph outlines + metrics from the bundled font (opentype.js, at runtime).
- [`src/`](src/) — geometry, layout, WebGPU plumbing, PNG output.

<!--
TODO (things to expand when filling this in):
  - the pitch: what problem this solves for generative art (conflation-free coverage, high-quality AA at print scale)
  - a couple of comparison images / numbers
  - motivation for publishing openly
  - credits
-->

## License

Apache-2.0 — see [`LICENSE`](LICENSE). Bundles the Lato font (SIL OFL 1.1); see [`NOTICE`](NOTICE) and
[`assets/OFL.txt`](assets/OFL.txt).
