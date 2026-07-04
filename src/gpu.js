// gpu.js — WebGPU plumbing for the windfoil coverage shader, shared by the offscreen PNG renderer and the
// realtime web client. The core is one instanced draw of the glyph atlas under a camera (`createGlyphRenderer`);
// `renderToRGBA` wraps it for a one-shot offscreen render + readback, while the web client draws it into a
// canvas swapchain every frame with a moving camera. Four bindings (uniforms, instances, curve atlas, row
// table), so all glyphs of all sizes render in a single draw(4, instanceCount).

const WGSL_URL = new URL('./windfoil.wgsl', import.meta.url);

// Read a WGSL source in either environment: Deno reads it off disk, the browser fetches it. Defaults to the
// windfoil shader; pass a URL to load a different one (e.g. the benchmark's Slug shader), so the loader stays
// shared rather than duplicated. Both resolve relative to their module.
export async function loadShaderCode(url = WGSL_URL) {
  if (typeof Deno !== 'undefined') return Deno.readTextFile(url);
  return fetch(url).then((r) => r.text());
}

// Request a WebGPU device (throws with a helpful message if there's no adapter).
export async function requestDevice() {
  const adapter = await navigator.gpu?.requestAdapter();
  if (!adapter) throw new Error('No WebGPU adapter — needs a GPU-capable host (Deno: run with --unstable-webgpu).');
  const device = await adapter.requestDevice();
  device.addEventListener?.('uncapturederror', (e) => console.error('WebGPU error:', e.error?.message));
  return device;
}

function storage(device, floats) {
  const buf = device.createBuffer({
    size: floats.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(buf, 0, floats);
  return buf;
}

/**
 * Build the retained glyph renderer: the pipeline plus the uploaded atlas + instance buffers for one scene.
 * The instances are static (the camera moves, not the geometry), so this is upload-once / draw-forever —
 * `setUniforms` updates the target size + camera each frame and `draw` records the single instanced draw.
 *
 * @param {GPUDevice} device
 * @param {object} o
 * @param {string} o.code            the WGSL source (see `loadShaderCode`)
 * @param {GPUTextureFormat} o.format the render target's format ('rgba8unorm' offscreen; the canvas preferred format live)
 * @param {Float32Array} o.curves    the deduped, band-duplicated curve atlas (3 vec2 per monotone piece)
 * @param {Uint32Array} o.rows       the row-band table ([start, count, area, xMin, xMax] per band; see bands.js)
 * @param {Float32Array} o.instances packed instance data (16 floats each)
 * @param {number} o.instanceCount
 */
export function createGlyphRenderer(device, { code, format, curves, rows, instances, instanceCount }) {
  const module = device.createShaderModule({ code });

  // Uniforms: res (vec2) + style (gamma, sharp) + camera (scaleX, scaleY, transX, transY) = 8 floats.
  const uniform = device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const curveBuf = storage(device, curves);
  const rowBuf = storage(device, rows);
  const instBuf = storage(device, instances);

  const pipeline = device.createRenderPipeline({
    layout: 'auto',
    vertex: { module, entryPoint: 'vs' },
    fragment: {
      module,
      entryPoint: 'fs',
      targets: [
        {
          format,
          // premultiplied-alpha "over": out = src + dst·(1 − src.a)
          blend: {
            color: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
            alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
          },
        },
      ],
    },
    primitive: { topology: 'triangle-strip' },
  });

  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniform } },
      { binding: 1, resource: { buffer: instBuf } },
      { binding: 2, resource: { buffer: curveBuf } },
      { binding: 3, resource: { buffer: rowBuf } },
    ],
  });

  return {
    // Update the per-frame uniforms. `cam` is [scaleX, scaleY, transX, transY] (identity by default, so the
    // offscreen path passes only width/height/style).
    setUniforms({ width, height, style = [1, 1], cam = [1, 1, 0, 0] }) {
      device.queue.writeBuffer(
        uniform,
        0,
        // res, style, cam
        new Float32Array([width, height, style[0], style[1], cam[0], cam[1], cam[2], cam[3]]),
      );
    },
    // Record the one instanced draw for every glyph into an open render pass.
    draw(pass) {
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bindGroup);
      pass.draw(4, instanceCount);
    },
  };
}

/**
 * One-shot offscreen render → RGBA8 readback (the PNG path). Renders the scene with an identity camera into a
 * texture the size of the layout, then copies it back to straight-alpha RGBA8. Uses the same pipeline + shader
 * as the live client via `createGlyphRenderer`.
 *
 * @param {object} o
 * @param {number} o.width  @param {number} o.height
 * @param {number[]} o.background straight-alpha [r,g,b,a] in 0..1 (kept opaque so readback needs no unpremultiply)
 * @param {Float32Array} o.curves    @param {Uint32Array} o.rows    @param {Float32Array} o.instances
 * @param {number} o.instanceCount
 * @param {[number, number]} [o.style] coverage-style (gamma, sharp); [1, 1] = exact (identity)
 * @param {string} [o.code] WGSL source override (e.g. a kernel shader from src/kernels.js); defaults to windfoil
 * @returns {Promise<Uint8Array>} width*height*4 RGBA8, straight alpha
 */
export async function renderToRGBA({ width, height, background, curves, rows, instances, instanceCount, style = [1, 1], code }) {
  const device = await requestDevice();
  const format = 'rgba8unorm';
  code = code ?? await loadShaderCode();

  const target = device.createTexture({
    size: [width, height],
    format,
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
  });

  const renderer = createGlyphRenderer(device, { code, format, curves, rows, instances, instanceCount });
  renderer.setUniforms({ width, height, style }); // identity camera → device px === layout px

  const [br, bg, bb, ba = 1] = background;
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginRenderPass({
    colorAttachments: [
      {
        view: target.createView(),
        // clear to the (opaque) background, premultiplied so it composites consistently with the glyphs
        clearValue: { r: br * ba, g: bg * ba, b: bb * ba, a: ba },
        loadOp: 'clear',
        storeOp: 'store',
      },
    ],
  });
  renderer.draw(pass);
  pass.end();

  // Copy the texture into a readback buffer (rows padded to 256 bytes, per WebGPU rules) and de-pad.
  const bytesPerRow = Math.ceil((width * 4) / 256) * 256;
  const readback = device.createBuffer({
    size: bytesPerRow * height,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  encoder.copyTextureToBuffer({ texture: target }, { buffer: readback, bytesPerRow }, [width, height]);
  device.queue.submit([encoder.finish()]);

  await readback.mapAsync(GPUMapMode.READ);
  const padded = new Uint8Array(readback.getMappedRange());
  const rgba = new Uint8Array(width * height * 4);
  for (let y = 0; y < height; y++) {
    rgba.set(padded.subarray(y * bytesPerRow, y * bytesPerRow + width * 4), y * width * 4);
  }
  readback.unmap();
  return rgba;
}
