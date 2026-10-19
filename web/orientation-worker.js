importScripts('vendor/ort/ort.wasm.min.js','orientation.js?v=facing-1');
ort.env.wasm.wasmPaths=new URL('vendor/ort/',self.location.href).href;ort.env.wasm.numThreads=1;ort.env.wasm.proxy=false;
let ready;
async function load(){
  if(!ready)ready=(async()=>{
    self.postMessage({phase:'loading'});
    const response=await fetch('models/vehicle-facing-v1.json');if(!response.ok)throw Error('Facing manifest HTTP '+response.status);const manifest=await response.json();
    const url=new URL('models/cfv-convnext-facing-int8.onnx',self.location.href).href;
    let cache;try{cache=await caches.open('locaintel-models-v1');}catch(_){}
    let modelResponse=await cache?.match(url),cached=!!modelResponse;
    if(!modelResponse){modelResponse=await fetch(url);if(!modelResponse.ok)throw Error('Facing model HTTP '+modelResponse.status);}
    const bytes=await modelResponse.arrayBuffer(),digest=[...new Uint8Array(await crypto.subtle.digest('SHA-256',bytes))].map(v=>v.toString(16).padStart(2,'0')).join('');
    if(digest!==manifest.sha256){await cache?.delete(url);throw Error('Facing model checksum mismatch');}
    if(cache&&!cached)try{await cache.put(url,new Response(bytes,{headers:{'content-type':'application/octet-stream'}}));}catch(_){}
    const model=await ort.InferenceSession.create(new Uint8Array(bytes),{executionProviders:['wasm'],graphOptimizationLevel:'all'});return {manifest,model};
  })().catch(error=>{ready=null;throw error;});return ready;
}
async function predict(model,input){const tensor=new ort.Tensor('float32',input,[1,3,224,224]);let outputs;try{outputs=await model.run({[model.inputNames[0]]:tensor});return {logits:Array.from(outputs.view_logits.data),angle:outputs.angle_fraction.data[0]};}finally{tensor.dispose?.();if(outputs)for(const t of Object.values(outputs))t.dispose?.();}}
self.onmessage=async({data})=>{
  try{
    const {manifest,model}=await load(),start=performance.now(),prediction=await predict(model,data.input);
    let result=VehicleFacing.classify(prediction.logits,prediction.angle,manifest,data.lights);
    if(result.state==='estimated'){
      const flipped=new Float32Array(data.input.length),size=224,plane=size*size;for(let c=0;c<3;c++)for(let y=0;y<size;y++)for(let x=0;x<size;x++)flipped[c*plane+y*size+x]=data.input[c*plane+y*size+size-1-x];
      result=VehicleFacing.classify(prediction.logits,prediction.angle,manifest,data.lights,await predict(model,flipped));
    }
    self.postMessage({phase:'ready'});self.postMessage({id:data.id,result:{...result,crop:data.crop,inference_ms:Math.round(performance.now()-start)}});
  }catch(error){self.postMessage({id:data.id,error:error.message||String(error)});}
};
