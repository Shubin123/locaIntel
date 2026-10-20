/* Shared worker runtime: WebGPU first, then WebGL, then WASM. */
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
  // The WebGL build is a separate ONNX Runtime instance; keep it under its own
  // global so the WebGPU/WASM `ort` stays intact for CPU fallback.
  let webglLib;
  function webgl(){
    if(webglLib===undefined){
      webglLib=null;
      try{
        if(typeof root.OffscreenCanvas==='function'&&new root.OffscreenCanvas(1,1).getContext('webgl2')){
          const main=root.ort;root.importScripts('vendor/ort/ort.webgl.min.js');webglLib=root.ort;root.ort=main;
        }
      }catch(_){webglLib=null;}
    }
    return webglLib;
  }
  let cacheUpdates=Promise.resolve();
  async function bytes(spec){
    const update=(state,extra={})=>root.postMessage?.({model_cache_update:{file:spec.file,state,at:Date.now(),scope:root.navigator?.locks?.request?'origin':'worker_only',...extra}});
    update('queued');
    const operation=async()=>{
      try{
        const url=new URL(spec.file,root.location.href).href;let cache;
        try{cache=await root.caches.open('locaintel-models-v1');}catch(_){}
        const verified=async buffer=>{
          if(!spec.sha256)return true;
          const digest=[...new Uint8Array(await root.crypto.subtle.digest('SHA-256',buffer))].map(v=>v.toString(16).padStart(2,'0')).join('');return digest===spec.sha256;
        };
        let response=await cache?.match(url),buffer;
        if(response){update('checking_local_cache');buffer=await response.arrayBuffer();if(await verified(buffer)){update('cached',{bytes:buffer.byteLength});return new Uint8Array(buffer);}update('repairing_cache');}
        const download=new URL(url);if(spec.sha256)download.searchParams.set('sha256',spec.sha256);
        update('downloading');response=await root.fetch(download.href,{cache:'reload'});if(!response.ok)throw Error('Model HTTP '+response.status);buffer=await response.arrayBuffer();
        update('verifying',{bytes:buffer.byteLength});if(!await verified(buffer))throw Error('Model checksum mismatch');
        let stored=false;if(cache)try{await cache.put(url,new Response(buffer,{headers:{'content-type':'application/octet-stream'}}));stored=true;}catch(_){}
        update(stored?'saved_locally':'cache_unavailable',{bytes:buffer.byteLength});return new Uint8Array(buffer);
      }catch(error){update('error',{error:error.message});throw error;}
    };
    // Origin-wide Web Locks serialize updates across model workers and tabs.
    // Browsers lacking this API serialize within each worker and report it.
    if(root.navigator?.locks?.request)return root.navigator.locks.request('locaintel-core-model-updates',operation);
    update('queued',{scope:'worker_only'});const pending=cacheUpdates.then(operation);cacheUpdates=pending.catch(()=>{});return pending;
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
  // One device-loss subscription, with explicit removal of released sessions.
  // Per-session promise callbacks otherwise retain packed batch buffers after
  // every slider change until the entire GPU device is destroyed.
  const observedDevices=new WeakMap();
  function observeDevice(owner,device){
    if(!device?.lost)return;
    let state=observedDevices.get(device);
    if(!state){state={sessions:new Set(),error:null};observedDevices.set(device,state);device.lost.then(info=>{
      state.error=Error('GPU device lost: '+(info.message||info.reason||'unavailable'));
      for(const session of state.sessions){session.lost=state.error;session.rejectLost?.(state.error);}state.sessions.clear();
    });}
    state.sessions.add(owner);owner.lossState=state;if(state.error)owner.lost=state.error;
  }
  class Session{
    constructor(cpu,gpu,webglSpec,{gpuOnly=false}={}){this.cpu=cpu;this.gpuSpec=gpu;this.webglSpec=webglSpec;this.gpuOnly=gpuOnly;this.backend='wasm';this.fallback=null;this.session=null;this.variant=null;this.gpuErrors=[];}
    async initialize(){
      const runtime=await start();
      // Lack of shader-f16 is not lack of a GPU. Try the compatible full-
      // precision graph on WebGPU before considering CPU execution.
      if(runtime.gpu){
        const candidates=[];
        for(const spec of (Array.isArray(this.gpuSpec)?this.gpuSpec:[this.gpuSpec]))if(spec&&(!spec.requiresF16||runtime.fp16))candidates.push(spec);
        if(!this.gpuOnly&&!candidates.some(spec=>spec.file===this.cpu.file))candidates.push(this.cpu);
        for(const spec of candidates)try{await this.useGPU(spec,!!spec.captureShapes);return this;}catch(error){this.gpuErrors.push(error.message);if(spec.captureShapes)try{await this.useGPU(spec,false);return this;}catch(retry){this.gpuErrors.push(retry.message);}}
        this.fallback='WebGPU initialization failed: '+this.gpuErrors.join('; ');
      }else this.fallback='WebGPU adapter unavailable in this browser';
      if(this.gpuOnly)throw Error(this.fallback);
      if(this.webglSpec)try{await this.useWebGL();return this;}catch(error){this.fallback+='; WebGL failed: '+error.message;}
      await this.useCPU();return this;
    }
    async useWebGL(){
      const lib=webgl();if(!lib)throw Error('WebGL2 unavailable in workers');
      await this.release();const spec=this.webglSpec;
      lib.env.webgl.pack=!!spec.pack;
      this.session=await lib.InferenceSession.create(await bytes(spec),{executionProviders:['webgl']});
      this.backend='webgl';this.variant=spec.file;this.capture=false;this.webglLib=lib;
    }
    async runWebGL(feeds){
      const lib=this.webglLib,inputs={},outputs={};
      for(const [name,t] of Object.entries(feeds))inputs[name]=new lib.Tensor(t.type,t.data,t.dims);
      for(const [name,t] of Object.entries(await this.session.run(inputs))){outputs[name]=new root.ort.Tensor('float32',Float32Array.from(t.data),t.dims);t.dispose?.();}
      return outputs;
    }
    async release(){this.lossState?.sessions.delete(this);this.lossState=null;try{await this.session?.release();}catch(_){}this.captureIO?.dispose();this.captureIO=null;this.session=null;}
    async useGPU(spec,capture){await this.release();const [graph,externalData]=await Promise.all([bytes(spec),Promise.all((spec.externalData||[]).map(async part=>({path:part.path,data:await bytes(part)})))]);this.session=await root.ort.InferenceSession.create(graph,{executionProviders:['webgpu'],graphOptimizationLevel:'all',enableGraphCapture:capture,...(externalData.length?{externalData}:{})});this.backend='webgpu';this.variant=spec.file;this.activeSpec=spec;this.capture=capture;this.lost=null;observeDevice(this,root.ort.env.webgpu.device);}
    async useCPU(){await this.release();this.session=await root.ort.InferenceSession.create(await bytes(this.cpu),{executionProviders:['wasm'],graphOptimizationLevel:'all'});this.backend='wasm';this.variant=this.cpu.file;this.capture=false;}
    async runGPU(feeds){
      if(this.backend==='webgpu'&&this.lost)throw this.lost;
      if(this.capture&&!this.captureIO)this.captureIO=new CapturedIO(this.session,feeds,this.activeSpec.captureShapes);
      const work=this.capture?this.captureIO.run(feeds):this.session.run(feeds);
      if(this.backend!=='webgpu'||!this.lossState)return work;
      let rejectLost;const lost=new Promise((_,reject)=>{rejectLost=reject;this.rejectLost=reject;});
      try{return await Promise.race([work,lost]);}finally{if(this.rejectLost===rejectLost)this.rejectLost=null;}
    }
    async run(feeds,{cpuFeeds}={}){
      if(this.backend==='webgl')try{return await this.runWebGL(feeds);}catch(error){
        // Lost contexts or unsupported shaders: keep working on the CPU.
        this.fallback='WebGL failed: '+error.message;await this.useCPU();return this.session.run(feeds);
      }
      try{return await this.runGPU(feeds);}catch(error){
        if(this.backend!=='webgpu')throw error;
        // An adapter may support the model but not command capture. Retry
        // ordinary GPU execution before falling back to the CPU provider.
        if(this.capture&&!this.lost)try{await this.useGPU(this.activeSpec,false);return await this.runGPU(feeds);}catch(retry){error=retry;}
        this.fallback=error.message;if(this.gpuOnly)throw error;await this.useCPU();return this.session.run(cpuFeeds?cpuFeeds():feeds);
      }
    }
    get inputNames(){return this.session.inputNames;}get outputNames(){return this.session.outputNames;}
  }
  root.ModelRuntime={start,bytes,webgl,Session};
})(globalThis);
