importScripts('vendor/ort/ort.wasm.min.js', 'detector.js?v=fresh-boxes-1');
ort.env.wasm.wasmPaths = new URL('vendor/ort/', self.location.href).href;
ort.env.wasm.numThreads = 1; // Works on GitHub Pages without cross-origin isolation.
ort.env.wasm.proxy = false;
let sessionPromise;
async function session() {
  if (!sessionPromise) {
    self.postMessage({ phase: 'loading' });
    sessionPromise = ort.InferenceSession.create(new URL('models/yolo11n.onnx', self.location.href).href, { executionProviders: ['wasm'], graphOptimizationLevel: 'all' });
  }
  return sessionPromise;
}
self.onmessage = async ({ data }) => {
  try {
    const model = await session();
    const inputTensor = new ort.Tensor('float32', data.input, [1, 3, 640, 640]);
    let output;
    try { output = await model.run({ [model.inputNames[0]]: inputTensor }); }
    finally { inputTensor.dispose?.(); }
    const tensor = output[model.outputNames[0]];
    const detections = YoloBoxes.decode(tensor.data, tensor.dims, data.geometry);
    for (const result of Object.values(output)) result.dispose?.();
    self.postMessage({ id: data.id, detections });
  } catch (error) { self.postMessage({ id: data.id, error: error.message || String(error) }); }
};
