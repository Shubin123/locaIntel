importScripts('inference-runtime.js?v=gpu-batch-1','orientation.js?v=facing-v3-1');
// One DriveBC-adapted model. Each car contributes its crop and mirror as two
// rows; up to eight cars share a single dynamic-batch dispatch.
let ready;
async function load(){
  if(!ready)ready=(async()=>{
    self.postMessage({phase:'loading'});
    const response=await fetch('models/vehicle-facing-v3.json?v=facing-v3-1');if(!response.ok)throw Error('Facing manifest HTTP '+response.status);const manifest=await response.json();
    const spec={file:'models/'+manifest.file,sha256:manifest.sha256},model=await new ModelRuntime.Session(spec,spec).initialize();
    return {manifest,model};
  })().catch(error=>{ready=null;throw error;});return ready;
}
async function run(jobs){
  const {manifest,model}=await load(),stride=3*224*224,input=new Float32Array(jobs.length*2*stride),start=performance.now();
  for(let i=0;i<jobs.length;i++){
    const source=jobs[i].input,offset=i*2*stride;input.set(source,offset);
    for(let c=0;c<3;c++)for(let y=0;y<224;y++){const row=c*224*224+y*224;for(let x=0;x<224;x++)input[offset+stride+row+x]=source[row+223-x];}
  }
  const tensor=new ort.Tensor('float32',input,[jobs.length*2,3,224,224]);let outputs;
  try{
    outputs=await model.run({[model.inputNames[0]]:tensor});const logits=Array.from(outputs.view_logits.data),elapsed=performance.now()-start;
    self.postMessage({phase:'ready',backend:model.backend,fallback:model.fallback,model:manifest.model,path:'mirror-averaged',batchCars:jobs.length});
    return jobs.map((job,i)=>({...VehicleFacing.estimate(logits.slice(i*16,i*16+16),manifest,job.lights,{truncated:job.truncated}),
      backend:model.backend,fallback:model.fallback,path:'mirror-averaged',crop:job.crop,gpu_batch_cars:jobs.length,batch_count:jobs.length,
      inference_ms:Math.round(elapsed/jobs.length),batch_inference_ms:Math.round(elapsed),model_ran:true}));
  }finally{tensor.dispose?.();if(outputs)for(const t of Object.values(outputs))t.dispose?.();}
}
// Serial messages keep one dispatch in flight.
let work=Promise.resolve();
self.onmessage=({data})=>work=work.then(async()=>{
  try{
    if(data.jobs){
      if(!data.jobs.length||data.jobs.length>8)throw Error('Expected 1–8 car crops per batch');
      self.postMessage({id:data.id,results:await run(data.jobs),batchError:null});
    }else self.postMessage({id:data.id,result:(await run([data]))[0]});
  }catch(error){self.postMessage({id:data.id,error:error.message||String(error)});}
});
