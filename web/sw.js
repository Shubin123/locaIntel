/* Static app shell only. Photos/evidence live in IndexedDB; model weights use
 * the runtime's checksummed model cache. No server API stores user data. */
const SHELL='locaintel-shell-routes-8';
const ASSETS=['./','index.html','style.css','app.js','sources.js','motion.js','detector.js','monitor.js','smooth.js','temporal.js','orientation.js','road.js','heading.js','paired-facing.js','facing-cache.js','capture-pair.js','routes.js','routes-ui.js','models/paired-facing-v1.json','models/vehicle-facing-v3.json','orientation-worker.js','detect-worker.js','road-worker.js','inference-runtime.js','vendor/leaflet.js','vendor/leaflet.css','vendor/leaflet-heat.js','vendor/ort/ort.webgpu.min.js','vendor/ort/ort.wasm.min.js','vendor/ort/ort.webgl.min.js'];
self.addEventListener('install',event=>event.waitUntil(caches.open(SHELL).then(cache=>cache.addAll(ASSETS.map(asset=>new Request(asset,{cache:'reload'})))).then(()=>self.skipWaiting())));
self.addEventListener('activate',event=>event.waitUntil((async()=>{for(const key of await caches.keys())if(key.startsWith('locaintel-shell-')&&key!==SHELL)await caches.delete(key);await self.clients.claim();})()));
self.addEventListener('fetch',event=>{
 const request=event.request,url=new URL(request.url);if(request.method!=='GET'||url.origin!==self.location.origin)return;
 // These weights are already stored (and verified) by ModelRuntime.bytes.
 if(/\/models\/.*\.(onnx|bin)$/.test(url.pathname)||/\/(api|frames)\//.test(url.pathname))return;
 const relative=url.pathname.slice(new URL(self.registration.scope).pathname.length);
 if(!ASSETS.includes(relative)&&relative!==''&&!/^data\/(latest|cameras|patterns)\.json$/.test(relative)&&!/^models\/[^/]+\.json$/.test(relative)&&!/^vendor\/ort\/[^/]+\.(wasm|mjs)$/.test(relative))return;
 event.respondWith((async()=>{
  // Network first: a deploy must never be shadowed by a stale cached script.
  // The cache is only the offline fallback.
  const cache=await caches.open(SHELL),cached=await cache.match(request,{ignoreSearch:true});
  try{const response=await fetch(request);if(response.ok){await cache.put(request,response.clone());return response;}if(cached)return cached;return response;}
  catch(error){if(cached)return cached;if(request.mode==='navigate'){const home=await cache.match('index.html');if(home)return home;}throw error;}
 })());
});
