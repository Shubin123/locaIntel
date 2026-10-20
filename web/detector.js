/* Fresh-frame YOLO11 detections, independent of the scheduled motion/road pipeline. */
(function (root) {
  const MODEL_VERSION='yolo11n-640-opset17-iou45-v1';
  const labels = { 0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck' };
  function letterbox(width, height, size = 640) {
    const scale = Math.min(size / width, size / height);
    const resizedWidth = Math.round(width * scale), resizedHeight = Math.round(height * scale);
    return { width, height, scale, resizedWidth, resizedHeight, padX: Math.round((size - resizedWidth) / 2 - .1), padY: Math.round((size - resizedHeight) / 2 - .1), size };
  }
  function iou(a, b) {
    const area = r => Math.max(0, r[2] - r[0]) * Math.max(0, r[3] - r[1]);
    const overlap = Math.max(0, Math.min(a[2], b[2]) - Math.max(a[0], b[0])) * Math.max(0, Math.min(a[3], b[3]) - Math.max(a[1], b[1]));
    return overlap / Math.max(1e-9, area(a) + area(b) - overlap);
  }
  function decode(data, dims, geometry, confidence = .3) {
    if (dims.length !== 3 || dims[0] !== 1 || (dims[1] !== 84 && dims[2] !== 84)) throw Error('Unsupported YOLO output shape: ' + dims.join('×'));
    const channelsFirst = dims[1] === 84, count = channelsFirst ? dims[2] : dims[1];
    const read = (i, channel) => data[channelsFirst ? channel * count + i : i * 84 + channel];
    const candidates = [], { scale, padX, padY, width, height } = geometry;
    for (let i = 0; i < count; i++) {
      let score = 0, category = -1;
      for (let k = 0; k < 80; k++) if (read(i, k + 4) > score) { score = read(i, k + 4); category = k; }
      if (score < confidence || !labels[category]) continue;
      const cx = read(i, 0), cy = read(i, 1), w = read(i, 2), h = read(i, 3);
      const clamp = value => Math.max(0, Math.min(1, value));
      const box = [clamp((cx - w / 2 - padX) / scale / width), clamp((cy - h / 2 - padY) / scale / height), clamp((cx + w / 2 - padX) / scale / width), clamp((cy + h / 2 - padY) / scale / height)];
      if (!box.every(Number.isFinite) || box[2] <= box[0] || box[3] <= box[1]) continue;
      candidates.push({ label: labels[category], confidence: score, box, on_road: null, motion: { state: 'unknown', reason: 'single_source_frame' } });
    }
    candidates.sort((a, b) => b.confidence - a.confidence);
    const kept = [];
    for (const candidate of candidates.slice(0, 1000)) {
      if (kept.some(d => d.label === candidate.label && iou(d.box, candidate.box) > .45)) continue;
      kept.push(candidate); if (kept.length >= 300) break;
    }
    return kept;
  }
  async function preprocess(source) {
    let image;
    if (typeof createImageBitmap === 'function') image = await createImageBitmap(source.blob);
    else { image = new Image(); image.src = source.url; await image.decode(); }
    try {
      const geometry = letterbox(image.width, image.height);
      const canvas = document.createElement('canvas'); canvas.width = canvas.height = geometry.size;
      const ctx = canvas.getContext('2d', { willReadFrequently: true });
      if (!ctx) throw Error('Image canvas unavailable');
      ctx.fillStyle = 'rgb(114,114,114)'; ctx.fillRect(0, 0, geometry.size, geometry.size);
      ctx.drawImage(image, geometry.padX, geometry.padY, geometry.resizedWidth, geometry.resizedHeight);
      const pixels = ctx.getImageData(0, 0, geometry.size, geometry.size).data;
      const plane = geometry.size * geometry.size, input = new Float32Array(plane * 3);
      for (let i = 0; i < plane; i++) { input[i] = pixels[i * 4] / 255; input[i + plane] = pixels[i * 4 + 1] / 255; input[i + 2 * plane] = pixels[i * 4 + 2] / 255; }
      return { input, geometry };
    } finally { image.close?.(); }
  }
  class BrowserDetector {
    constructor({ workerFactory = () => new Worker('detect-worker.js?v=local-1'), prepare = preprocess, onChange = () => {}, onResult = () => {}, isWanted = () => true, isPaused = () => false, delayMs = 0, timeout = 120000, onModelUpdate = () => {} } = {}) {
      Object.assign(this, { workerFactory, prepare, onChange, onResult, isWanted, isPaused, delayMs, timeout, onModelUpdate });
      this.entries = new Map(); this.pending = new Map(); this.queue = []; this.worker = null; this.waiting = new Map(); this.serial = 0; this.running = false; this.state = 'idle';this.modelVersion=MODEL_VERSION; this.cooldown = null;
    }
    get(hash) { return this.entries.get(hash); }
    request(source, priority = false) {
      if (this.pending.has(source.hash)) {
        if (priority) { const index = this.queue.findIndex(t => t.source.hash === source.hash); if (index > 0) this.queue.unshift(this.queue.splice(index, 1)[0]); }
        return this.pending.get(source.hash);
      }
      if (this.entries.has(source.hash)) return Promise.resolve(this.entries.get(source.hash));
      let resolve;
      const promise = new Promise(r => { resolve = r; }); this.pending.set(source.hash, promise);
      const task = { source, resolve }; if (priority) this.queue.unshift(task); else this.queue.push(task);
      this.entries.set(source.hash, { status: 'queued' }); this.onChange(); this.pump(); return promise;
    }
    ensureWorker() {
      if (this.worker) return;
      this.worker = this.workerFactory();
      this.worker.onmessage = ({ data }) => {
        if(data.model_cache_update){this.onModelUpdate(data.model_cache_update);return;}
        if (data.phase) { this.state = data.phase; this.onChange(); return; }
        const waiter = this.waiting.get(data.id); if (!waiter) return;
        this.waiting.delete(data.id); clearTimeout(waiter.timer);
        if (data.error) { this.state = 'error'; waiter.reject(Error(data.error)); }
        else { this.state = 'ready';this.backend=data.backend||'wasm';this.lastInferenceMs=data.inference_ms; waiter.resolve(data); }
      };
      this.worker.onerror = error => {
        this.state = 'error';
        for (const waiter of this.waiting.values()) { clearTimeout(waiter.timer); waiter.reject(Error(error.message || 'Detection worker failed')); }
        this.waiting.clear(); this.worker?.terminate(); this.worker = null;
      };
    }
    infer(prepared) {
      this.ensureWorker(); const id = ++this.serial;
      return new Promise((resolve, reject) => {
        const timer = setTimeout(() => { this.waiting.delete(id); this.worker?.terminate(); this.worker = null; reject(Error('Browser detection timed out')); }, this.timeout);
        this.waiting.set(id, { resolve, reject, timer });
        try { this.worker.postMessage({ id, input: prepared.input, geometry: prepared.geometry }, [prepared.input.buffer]); }
        catch (error) { clearTimeout(timer); this.waiting.delete(id); reject(error); }
      });
    }
    async pump() {
      if (this.running || this.cooldown || !this.queue.length || this.isPaused()) return;
      this.running = true; const task = this.queue.shift(), { source } = task;let didWork=false;
      try {
        if (!this.isWanted(source.hash)) { this.entries.delete(source.hash); task.resolve({ status: 'skipped' }); return; }
        didWork=true;this.entries.set(source.hash, { status: 'processing' }); this.onChange();
        const { detections,backend,inference_ms,fallback } = await this.infer(await this.prepare(source));
        const counts = {}; for (const d of detections) counts[d.label] = (counts[d.label] || 0) + 1;
        const result = { status: 'ready', hash: source.hash, version:MODEL_VERSION, detections, counts, observed: new Date().toISOString(), model: 'YOLO11n / ONNX browser',backend:backend||'wasm',inference_ms,fallback };
        this.entries.set(source.hash, result); this.onResult(source,result); task.resolve(result);
      } catch (error) {
        const result = { status: 'error', error: error.message, hash: source.hash }; this.entries.set(source.hash, result); task.resolve(result);
      } finally {
        this.pending.delete(source.hash); this.running = false;
        // Keep a bounded cache of completed frame analyses.
        if (this.entries.size > 128) for (const [hash, entry] of this.entries) { if (!this.pending.has(hash) && hash !== source.hash) { this.entries.delete(hash); break; } }
        this.onChange();
        if(didWork&&this.delayMs>0&&this.queue.length)this.cooldown=setTimeout(()=>{this.cooldown=null;this.pump();},(this.backend==='webgpu'||this.backend==='webgl')?this.delayMs*.15:this.delayMs);
        else Promise.resolve().then(()=>this.pump());
      }
    }
    setBudget({delayMs=this.delayMs}={}) {this.delayMs=Math.max(0,delayMs);if(this.cooldown){clearTimeout(this.cooldown);this.cooldown=null;}if(!this.running&&this.queue.length&&this.delayMs>0)this.cooldown=setTimeout(()=>{this.cooldown=null;this.pump();},(this.backend==='webgpu'||this.backend==='webgl')?this.delayMs*.15:this.delayMs);else this.pump();}
    wake() {this.pump();}
    retryFailed() { for (const [hash, entry] of this.entries) if (entry.status === 'error') this.entries.delete(hash); if (this.state === 'error' && !this.running) { this.worker?.terminate(); this.worker = null; this.state = 'idle'; } }
  }
  root.YoloBoxes = { letterbox, iou, decode, preprocess };
  root.BrowserDetector = BrowserDetector;
})(globalThis);
