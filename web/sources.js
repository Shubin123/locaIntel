/* Direct upstream snapshot refresh; no GitHub credentials or inference server required. */
globalThis.CameraSourceCache = class CameraSourceCache {
  constructor({ fetcher = (...args) => fetch(...args), digest = bytes => crypto.subtle.digest('SHA-256', bytes),
    makeURL = blob => URL.createObjectURL(blob), revokeURL = url => URL.revokeObjectURL(url),
    validate = async blob => { if (typeof createImageBitmap === 'function') { const bitmap = await createImageBitmap(blob); const size={width:bitmap.width,height:bitmap.height};bitmap.close();return size; } },
    concurrency = 6, timeout = 12000, onUpdate = () => {} } = {}) {
    Object.assign(this, { fetcher, digest, makeURL, revokeURL, validate, concurrency, timeout, onUpdate });
    this.entries = new Map(); this.inflight = new Map(); this.active = 0; this.waiters = []; this.retired = new Set();
  }
  get(id) { return this.entries.get(String(id)); }
  async slot() { if (this.active < this.concurrency) { this.active++; return; } await new Promise(resolve => this.waiters.push(resolve)); }
  wake() {while(this.active<this.concurrency&&this.waiters.length){this.active++;this.waiters.shift()();}}
  release() { this.active=Math.max(0,this.active-1);this.wake(); }
  setConcurrency(value) {this.concurrency=Math.max(1,Math.min(6,value));this.wake();}
  releaseUnused(keep) {for(const url of this.retired)if(!keep.has(url)){this.revokeURL(url);this.retired.delete(url);}}
  async load(camera, force = false) {
    const id = String(camera.id);
    if (this.inflight.has(id)) return this.inflight.get(id);
    if (!force && this.entries.has(id)) return this.entries.get(id);
    const task = this.request(camera).finally(() => this.inflight.delete(id));
    this.inflight.set(id, task); return task;
  }
  async request(camera) {
    await this.slot();
    const id = String(camera.id), previous = this.entries.get(id), controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), this.timeout);
    try {
      const source = new URL(camera.image);
      if (source.protocol !== 'https:' || !(source.hostname === 'drivebc.ca' || source.hostname.endsWith('.drivebc.ca'))) throw Error('Unexpected camera image host');
      source.searchParams.set('_locaintel', Date.now().toString());
      const response = await this.fetcher(source.href, { cache: 'no-store', credentials: 'omit', signal: controller.signal });
      if (!response.ok) throw Error('DriveBC HTTP ' + response.status);
      const blob = await response.blob();
      if (!blob.size || !blob.type.startsWith('image/')) throw Error('DriveBC did not return an image');
      const dimensions=await this.validate(blob);
      const hash = Array.from(new Uint8Array(await this.digest(await blob.arrayBuffer())), b => b.toString(16).padStart(2, '0')).join('');
      const url = previous?.hash === hash && previous.url ? previous.url : this.makeURL(blob);
      const entry = { hash, url, blob, cameraId:id, width:dimensions?.width,height:dimensions?.height, checked: new Date().toISOString(), modified: response.headers.get('last-modified'), error: null };
      this.entries.set(id, entry);
      // Let existing rendered <img> elements finish swapping to their new URLs.
      if (previous?.url && previous.url !== url) this.retired.add(previous.url);
      try { this.onUpdate(camera,entry); } catch (_) { /* Storage errors do not invalidate a successful image fetch. */ }
      return entry;
    } catch (error) {
      const entry = { ...previous, error: error.name === 'AbortError' ? 'DriveBC request timed out' : error.message, attempted: new Date().toISOString() };
      this.entries.set(id, entry); return entry;
    } finally { clearTimeout(timer); this.release(); }
  }
  async refresh(cameras, force = false) {
    const unique = [...new Map(cameras.map(c => [String(c.id), c])).values()];
    return Promise.all(unique.map(camera => this.load(camera, force)));
  }
  clear() { for (const entry of this.entries.values()) if (entry.url) this.revokeURL(entry.url); this.entries.clear();for(const url of this.retired)this.revokeURL(url);this.retired.clear(); }
}
