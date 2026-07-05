// variable-gpu.js — WebGPU plumbing for windfoil-variable.wgsl. Same instanced-draw model as gpu.js, plus a
// fifth binding (the anchor field) and two OKLab matrices in the uniform. Kept separate from the core glyph
// renderer so the variable feature stays self-contained (and the core shader path is untouched — see NOTES).
//
// This is the offscreen PNG path only: one draw into an rgba8unorm target, read back to straight-alpha RGBA8.

import { loadShaderCode, requestDevice } from './gpu.js';
import { oklabUniformMatrices } from './variable.js';

const WGSL_URL = new URL('./windfoil-variable.wgsl', import.meta.url);

function storage(device, arr) {
  const buf = device.createBuffer({
    size: arr.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(buf, 0, arr);
  return buf;
}

/**
 * One-shot offscreen render of a variable scene → straight-alpha RGBA8.
 *
 * @param {object} o
 * @param {number} o.width  @param {number} o.height
 * @param {number[]} o.background straight-alpha [r,g,b,a] in 0..1 (opaque reads back as-is; a transparent bg yields premultiplied RGBA)
 * @param {Float32Array} o.curves @param {Uint32Array} o.rows @param {Float32Array} o.instances @param {Float32Array} o.anchors
 * @param {number} o.instanceCount
 * @param {[number, number]} [o.style] coverage-style (gamma, sharp); [1, 1] = exact (identity)
 * @returns {Promise<Uint8Array>} width*height*4 RGBA8, straight alpha
 */
export async function renderVariableToRGBA({
  width,
  height,
  background,
  curves,
  rows,
  instances,
  anchors,
  instanceCount,
  style = [1, 1],
}) {
  const device = await requestDevice();
  const format = 'rgba8unorm';
  const code = await loadShaderCode(WGSL_URL);
  const module = device.createShaderModule({ code });

  // Uniforms: res(2) + style(2) + cam(4) + okToLms(mat3x3 → 12) + lmsToRgb(mat3x3 → 12) = 32 floats = 128 bytes.
  const { okToLms, lmsToRgb } = oklabUniformMatrices();
  const uni = new Float32Array(32);
  uni.set([width, height, style[0], style[1], 1, 1, 0, 0], 0); // identity camera → device px === layout px
  uni.set(okToLms, 8);
  uni.set(lmsToRgb, 20);
  const uniform = device.createBuffer({
    size: uni.byteLength,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(uniform, 0, uni);

  const curveBuf = storage(device, curves);
  const rowBuf = storage(device, rows);
  const instBuf = storage(device, instances);
  const anchorBuf = storage(device, anchors);

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
      { binding: 4, resource: { buffer: anchorBuf } },
    ],
  });

  const target = device.createTexture({
    size: [width, height],
    format,
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
  });

  const [br, bg, bb, ba = 1] = background;
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginRenderPass({
    colorAttachments: [
      {
        view: target.createView(),
        clearValue: { r: br * ba, g: bg * ba, b: bb * ba, a: ba }, // premultiplied clear
        loadOp: 'clear',
        storeOp: 'store',
      },
    ],
  });
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.draw(4, instanceCount);
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
