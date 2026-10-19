/* Shared worker runtime: request real WebGPU, retain a working WASM fallback. */
(function(root){
  let boot;
  async function start(){
    if(!boot)boot=(async()=>{
      let adapter=null;
      try{adapter=await root.navigator?.gpu?.requestAdapter({powerPreference:'high-performance'});}catch(_){}
      if(!adapter)try{adapter=await root.navigator?.gpu?.requestAdapter();}catch(_){}
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
  // Static GPU IO lets ORT replay its command graph rather than rebuilding
  // hundreds of per-car dispatches. Buffers stay owned by this session.
  class CapturedIO{
    constructor(session,feeds,shapes){
      this.session=session;this.device=root.ort.env.webgpu.device;this.buffers=[];this.feeds={};this.fetches={};this.outputs=[];let offset=0;
      const allocate=(dims)=>{const count=dims.reduce((a,b)=>a*b,1),size=Math.ceil(count*4/16)*16,buffer=this.device.createBuffer({size,usage:root.GPUBufferUsage.STORAGE|root.GPUBufferUsage.COPY_SRC|root.GPUBufferUsage.COPY_DST});this.buffers.push(buffer);return {buffer,size,count,tensor:root.ort.Tensor.fromGpuBuffer(buffer,{dataType:'float32',dims})};};
      try{
        this.inputs=Object.entries(feeds).map(([name,tensor])=>{if(tensor.type!=='float32')throw Error('Capture requires float32 IO');const slot=allocate(tensor.dims);this.feeds[name]=slot.tensor;return {name,...slot};});
        for(const [name,dims]of Object.entries(shapes)){const slot=allocate(dims);this.fetches[name]=slot.tensor;this.outputs.push({name,dims,offset,...slot});offset+=slot.size;}
        this.readback=this.device.createBuffer({size:offset,usage:root.GPUBufferUsage.MAP_READ|root.GPUBufferUsage.COPY_DST});this.buffers.push(this.readback);
      }catch(error){this.dispose();throw error;}
    }
    async run(feeds){
      for(const slot of this.inputs){const data=feeds[slot.name].data;if(data.length!==slot.count)throw Error('GPU capture input shape changed');this.device.queue.writeBuffer(slot.buffer,0,data);}
      await this.session.run(this.feeds,this.fetches);
      const encoder=this.device.createCommandEncoder();for(const slot of this.outputs)encoder.copyBufferToBuffer(slot.buffer,0,this.readback,slot.offset,slot.count*4);this.device.queue.submit([encoder.finish()]);
      await this.readback.mapAsync(root.GPUMapMode.READ);
      try{const mapped=this.readback.getMappedRange(),results={};for(const slot of this.outputs)results[slot.name]=new root.ort.Tensor('float32',new Float32Array(mapped,slot.offset,slot.count).slice(),slot.dims);return results;}finally{this.readback.unmap();}
    }
    dispose(){for(const buffer of this.buffers)buffer.destroy();}
  }
  class Session{
    constructor(cpu,gpu){this.cpu=cpu;this.gpuSpec=gpu;this.backend='wasm';this.fallback=null;this.session=null;this.variant=null;this.gpuErrors=[];}
    async initialize(){
      const runtime=await start();
      // Lack of shader-f16 is not lack of a GPU. Try the compatible full-
      // precision graph on WebGPU before considering CPU execution.
      if(runtime.gpu){
        const candidates=[];
        for(const spec of (Array.isArray(this.gpuSpec)?this.gpuSpec:[this.gpuSpec]))if(spec&&(!spec.requiresF16||runtime.fp16))candidates.push(spec);
        if(!candidates.some(spec=>spec.file===this.cpu.file))candidates.push(this.cpu);
        for(const spec of candidates)try{await this.useGPU(spec,!!spec.captureShapes);return this;}catch(error){this.gpuErrors.push(error.message);if(spec.captureShapes)try{await this.useGPU(spec,false);return this;}catch(retry){this.gpuErrors.push(retry.message);}}
        this.fallback='GPU initialization failed: '+this.gpuErrors.join('; ');
      }else this.fallback='WebGPU adapter unavailable in this browser';
      await this.useCPU();return this;
    }
    async release(){try{await this.session?.release();}catch(_){}this.captureIO?.dispose();this.captureIO=null;this.session=null;}
    async useGPU(spec,capture){await this.release();const [graph,externalData]=await Promise.all([bytes(spec),Promise.all((spec.externalData||[]).map(async part=>({path:part.path,data:await bytes(part)})))]);this.session=await root.ort.InferenceSession.create(graph,{executionProviders:['webgpu'],graphOptimizationLevel:'all',enableGraphCapture:capture,...(externalData.length?{externalData}:{})});this.backend='webgpu';this.variant=spec.file;this.activeSpec=spec;this.capture=capture;this.lost=null;const session=this.session;this.loss=root.ort.env.webgpu.device?.lost?.then(info=>{const error=Error('GPU device lost: '+(info.message||info.reason||'unavailable'));if(this.session===session)this.lost=error;return error;});}
    async useCPU(){await this.release();this.session=await root.ort.InferenceSession.create(await bytes(this.cpu),{executionProviders:['wasm'],graphOptimizationLevel:'all'});this.backend='wasm';this.variant=this.cpu.file;this.capture=false;}
    async runGPU(feeds){
      if(this.backend==='webgpu'&&this.lost)throw this.lost;
      if(this.capture&&!this.captureIO)this.captureIO=new CapturedIO(this.session,feeds,this.activeSpec.captureShapes);
      const work=this.capture?this.captureIO.run(feeds):this.session.run(feeds);
      return this.backend==='webgpu'&&this.loss?Promise.race([work,this.loss.then(error=>{throw error;})]):work;
    }
    async run(feeds,{cpuFeeds}={}){
      try{return await this.runGPU(feeds);}catch(error){
        if(this.backend!=='webgpu')throw error;
        // An adapter may support the model but not command capture. Retry
        // ordinary GPU execution before falling back to the CPU provider.
        if(this.capture&&!this.lost)try{await this.useGPU(this.activeSpec,false);return await this.runGPU(feeds);}catch(retry){error=retry;}
        this.fallback=error.message;await this.useCPU();return this.session.run(cpuFeeds?cpuFeeds():feeds);
      }
    }
    get inputNames(){return this.session.inputNames;}get outputNames(){return this.session.outputNames;}
  }
  root.ModelRuntime={start,bytes,Session};
})(globalThis);
