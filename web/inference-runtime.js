/* Shared worker runtime: request real WebGPU, retain a working WASM fallback. */
(function(root){
  let boot;
  async function start(){
    if(!boot)boot=(async()=>{
      let adapter=null;try{adapter=await root.navigator?.gpu?.requestAdapter();}catch(_){}
      root.importScripts(adapter?'vendor/ort/ort.webgpu.min.js':'vendor/ort/ort.wasm.min.js');
      root.ort.env.wasm.wasmPaths=new URL('vendor/ort/',root.location.href).href;
      root.ort.env.wasm.numThreads=1;root.ort.env.wasm.proxy=false;
      if(adapter)root.ort.env.webgpu.adapter=adapter;
      return {gpu:!!adapter,fp16:!!adapter?.features.has('shader-f16'),adapter};
    })();return boot;
  }
  async function bytes(spec){
    const url=new URL(spec.file,root.location.href).href;let cache;try{cache=await root.caches.open('locaintel-models-v1');}catch(_){}
    let response=await cache?.match(url),cached=!!response;if(!response){response=await root.fetch(url);if(!response.ok)throw Error('Model HTTP '+response.status);}
    const buffer=await response.arrayBuffer();
    if(spec.sha256){const digest=[...new Uint8Array(await root.crypto.subtle.digest('SHA-256',buffer))].map(v=>v.toString(16).padStart(2,'0')).join('');if(digest!==spec.sha256){await cache?.delete(url);throw Error('Model checksum mismatch');}}
    if(cache&&!cached)try{await cache.put(url,new Response(buffer,{headers:{'content-type':'application/octet-stream'}}));}catch(_){}
    return new Uint8Array(buffer);
  }
  class Session{
    constructor(cpu,gpu){this.cpu=cpu;this.gpuSpec=gpu;this.backend='wasm';this.fallback=null;this.session=null;}
    async initialize(){
      const runtime=await start();
      if(runtime.gpu&&(!this.gpuSpec?.requiresF16||runtime.fp16)){
        try{this.session=await root.ort.InferenceSession.create(await bytes(this.gpuSpec||this.cpu),{executionProviders:['webgpu','wasm'],graphOptimizationLevel:'all'});this.backend='webgpu';return this;}catch(error){this.fallback=error.message;}
      }
      await this.useCPU();return this;
    }
    async useCPU(){try{await this.session?.release();}catch(_){}this.session=await root.ort.InferenceSession.create(await bytes(this.cpu),{executionProviders:['wasm'],graphOptimizationLevel:'all'});this.backend='wasm';}
    async run(feeds){try{return await this.session.run(feeds);}catch(error){if(this.backend!=='webgpu')throw error;this.fallback=error.message;await this.useCPU();return this.session.run(feeds);}}
    get inputNames(){return this.session.inputNames;}get outputNames(){return this.session.outputNames;}
  }
  root.ModelRuntime={start,bytes,Session};
})(globalThis);
