// gpu.js — WebGPU plumbing: upload the atlas + instance buffers, run one instanced draw of the shader into
// an offscreen texture, read the pixels back as RGBA8. Four bindings (target size, instances, curve atlas,
// row table), so all glyphs of all sizes render in a single draw(4, instanceCount).

const WGSL_URL = new URL('./area.wgsl', import.meta.url);

/**
 * @param {object} o
 * @param {number} o.width  @param {number} o.height
 * @param {number[]} o.background straight-alpha [r,g,b,a] in 0..1 (kept opaque so readback needs no unpremultiply)
 * @param {Float32Array} o.curves    the deduped, band-duplicated curve atlas (3 vec2 per monotone piece)
 * @param {Uint32Array} o.rows       the row-band table ([start, count] per band)
 * @param {Float32Array} o.instances packed instance data (16 floats each)
 * @param {number} o.instanceCount
 * @returns {Promise<Uint8Array>} width*height*4 RGBA8, straight alpha
 */
export async function renderToRGBA({ width, height, background, curves, rows, instances, instanceCount }) {
  const adapter = await navigator.gpu?.requestAdapter();
  if (!adapter) throw new Error('No WebGPU adapter — run Deno with --unstable-webgpu on a GPU-capable host.');
  const device = await adapter.requestDevice();
  device.addEventListener?.('uncapturederror', (e) => console.error('WebGPU error:', e.error?.message));

  const format = 'rgba8unorm';
  const code = await Deno.readTextFile(WGSL_URL);
  const module = device.createShaderModule({ code });

  const target = device.createTexture({
    size: [width, height],
    format,
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
  });

  // Uniforms: target size (vec2) padded to 16 bytes.
  const uniform = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(uniform, 0, new Float32Array([width, height, 0, 0]));

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
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.draw(4, instanceCount); // one instanced draw call for every glyph
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

function storage(device, floats) {
  const buf = device.createBuffer({
    size: floats.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(buf, 0, floats);
  return buf;
}
