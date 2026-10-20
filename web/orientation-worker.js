importScripts('inference-runtime.js?v=gpu-batch-1','orientation.js?v=paired-1');
let ready,teacherReady,quickReady;
async function load(){
  if(!ready)ready=(async()=>{
    self.postMessage({phase:'loading'});
    const response=await fetch('models/vehicle-facing-v2.json?v=gpu-batch-1');if(!response.ok)throw Error('Facing manifest HTTP '+response.status);const manifest=await response.json();
    return {manifest};
  })().catch(error=>{ready=null;throw error;});return ready;
}
async function quickModel(manifest){
  if(!quickReady)quickReady=(async()=>{
    let quick=null,quickError=null;
    if(manifest.fast)try{quick=await new ModelRuntime.Session({file:'models/'+manifest.fast.file,sha256:manifest.fast.sha256},{file:'models/'+manifest.fast.file,sha256:manifest.fast.sha256,captureShapes:{view_logits:[2,8],angle_vector:[2,2]}}).initialize();}catch(error){quickError=error.message;}
    return {quick,quickError};
  })();return quickReady;
}

async function teacher(manifest){
  if(!teacherReady)teacherReady=(async()=>{
    self.postMessage({phase:'loading',detail:'Checking difficult crop with full-resolution model'});
    const gpu=[manifest.gpu&&{...manifest.gpu,requiresF16:true},manifest.gpu32].filter(Boolean).map(spec=>({...spec,
      file:'models/'+spec.file,externalData:spec.externalData?.map(part=>({...part,file:'models/'+part.file})),
      captureShapes:spec.batch===2?{view_logits:[2,8],angle_fraction:[2,1]}:undefined}));
    return new ModelRuntime.Session({file:'models/cfv-convnext-facing-int8.onnx',sha256:manifest.sha256},gpu).initialize();
  })().catch(error=>{teacherReady=null;throw error;});return teacherReady;
}
function mirror(input){const flipped=new Float32Array(input.length),size=224,plane=size*size;for(let c=0;c<3;c++)for(let y=0;y<size;y++)for(let x=0;x<size;x++)flipped[c*plane+y*size+x]=input[c*plane+y*size+size-1-x];return flipped;}
async function predict(model,input){const tensor=new ort.Tensor('float32',input,[1,3,224,224]);let outputs;try{outputs=await model.run({[model.inputNames[0]]:tensor});return {logits:Array.from(outputs.view_logits.data),angle:outputs.angle_fraction.data[0]};}finally{tensor.dispose?.();if(outputs)for(const t of Object.values(outputs))t.dispose?.();}}
async function fullPredict(model,input,manifest,lights){
  let prediction,flipped;
  if([manifest.gpu,manifest.gpu32].some(spec=>spec?.batch===2&&model.variant==='models/'+spec.file)){
    const pair=new Float32Array(input.length*2);pair.set(input);pair.set(mirror(input),input.length);
    const tensor=new ort.Tensor('float32',pair,[2,3,224,224]),single=new ort.Tensor('float32',input,[1,3,224,224]);let outputs;
    try{
      outputs=await model.run({[model.inputNames[0]]:tensor},{cpuFeeds:()=>({[model.inputNames[0]]:single})});
      prediction={logits:Array.from(outputs.view_logits.data).slice(0,8),angle:outputs.angle_fraction.data[0]};
      if(outputs.view_logits.data.length===16)flipped={logits:Array.from(outputs.view_logits.data).slice(8),angle:outputs.angle_fraction.data[1]};
    }finally{tensor.dispose?.();single.dispose?.();if(outputs)for(const t of Object.values(outputs))t.dispose?.();}
  }else prediction=await predict(model,input);
  return VehicleFacing.classify(prediction.logits,prediction.angle,manifest,lights,flipped||await predict(model,mirror(input)));
}
async function fastPredict(model,input,manifest,lights){
  const pair=new Float32Array(input.length*2);pair.set(input);pair.set(mirror(input),input.length);
  const tensor=new ort.Tensor('float32',pair,[2,3,224,224]);let outputs;
  try{outputs=await model.run({[model.inputNames[0]]:tensor});return VehicleFacing.fastClassify(Array.from(outputs.view_logits.data),Array.from(outputs.angle_vector.data),manifest,lights);}
  finally{tensor.dispose?.();if(outputs)for(const t of Object.values(outputs))t.dispose?.();}
}
async function single(data){
    for(const kind of Object.keys(batchModels)){await batchModels[kind].model.release();delete batchModels[kind];}
    const {manifest}=await load(),{quick,quickError}=await quickModel(manifest),start=performance.now();let result,model,path='full-resolution',fastMs=null,fastError=quickError;
    if(quick){const quickStart=performance.now();try{result=await fastPredict(quick,data.input,manifest,data.lights);if(result.state==='estimated'){model=quick;path='fast-verified';}}catch(error){fastError=error.message;}fastMs=Math.round(performance.now()-quickStart);}
    if(!model){
      model=await teacher(manifest);result=await fullPredict(model,data.input,manifest,data.lights);
    }
    const modelName=path==='fast-verified'?manifest.fast.model:model.variant?.includes('fp16')?'CFV ConvNeXt-Small (FP16)':model.variant?.includes('batch2-fp32')?'CFV ConvNeXt-Small (FP32)':manifest.model;
    self.postMessage({phase:'ready',backend:model.backend,fallback:model.fallback,model:modelName,path,batchCars:1});
    return {...result,model:modelName,backend:model.backend,gpu_capture:!!model.capture,fallback:model.fallback,path,fast_inference_ms:fastMs,fast_error:fastError,crop:data.crop,inference_ms:Math.round(performance.now()-start),gpu_batch_cars:1,model_ran:true};
}

// One resident graph per stage bounds GPU memory when the slider changes. All
// batch sizes share immutable weights in Cache Storage; only shapes differ.
const batchModels={},failedBatches=new Set();
const capacity=n=>n<=2?2:n<=4?4:8;
function gpuSpec(spec,shapes){return {...spec,file:'models/'+spec.file,externalData:spec.externalData?.map(p=>({...p,file:'models/'+p.file})),captureShapes:shapes};}
async function batchModel(manifest,kind,cars){
  const key=kind+':'+cars;
  if(failedBatches.has(key))throw Error('GPU batch '+cars+' unavailable; using individual crops');
  if(batchModels[kind]?.cars===cars)return batchModels[kind].model;
  if(batchModels[kind]){await batchModels[kind].model.release();delete batchModels[kind];}
  if(kind==='teacher'&&teacherReady){await (await teacherReady).release();teacherReady=null;}
  const specs=manifest.gpuBatches?.[cars];if(!specs)throw Error('GPU batch graphs unavailable');
  const shapes={view_logits:[cars*2,8],...(kind==='fast'?{angle_vector:[cars*2,2]}:{angle_fraction:[cars*2,1]})};
  const candidates=kind==='fast'?[gpuSpec(specs.fast,shapes)]:[{...gpuSpec(specs.gpu,shapes),requiresF16:true},gpuSpec(specs.gpu32,shapes)];
  const model=new ModelRuntime.Session({file:'models/cfv-convnext-facing-int8.onnx',sha256:manifest.sha256},candidates,null,{gpuOnly:true});
  try{await model.initialize();}catch(error){failedBatches.add(key);await model.release();throw error;}
  model.batchKey=key;batchModels[kind]={cars,model};return model;
}
async function batchOutputs(model,jobs,cars){
  const stride=3*224*224,input=model.packedInput??=new Float32Array(cars*2*stride);if(jobs.length<cars)input.fill(0);
  for(let i=0;i<jobs.length;i++){const source=jobs[i].input,offset=i*2*stride;input.set(source,offset);for(let c=0;c<3;c++)for(let y=0;y<224;y++){const row=c*224*224+y*224;for(let x=0;x<224;x++)input[offset+stride+row+x]=source[row+223-x];}}
  const tensor=new ort.Tensor('float32',input,[cars*2,3,224,224]);let output;
  try{output=await model.run({[model.inputNames[0]]:tensor});return Object.fromEntries(Object.entries(output).map(([k,t])=>[k,Array.from(t.data)]));}
  catch(error){failedBatches.add(model.batchKey);throw error;}
  finally{tensor.dispose?.();if(output)for(const t of Object.values(output))t.dispose?.();}
}
async function batch(data){
  const {manifest}=await load(),jobs=data.jobs,cars=data.batchSize||capacity(jobs.length),start=performance.now(),results=Array(jobs.length),pending=[];
  let fastMs=null,fastError=null;
  try{
    const model=await batchModel(manifest,'fast',cars),t=performance.now(),out=await batchOutputs(model,jobs,cars);fastMs=performance.now()-t;
    for(let i=0;i<jobs.length;i++){
      const r=VehicleFacing.fastClassify(out.view_logits.slice(i*16,i*16+16),out.angle_vector.slice(i*4,i*4+4),manifest,jobs[i].lights);
      if(r.state==='estimated')results[i]={...r,model:manifest.fast.model,backend:model.backend,gpu_capture:!!model.capture,path:'fast-verified',gpu_batch_cars:cars};else pending.push(i);
    }
  }catch(error){fastError=error.message;pending.length=0;results.fill(undefined);for(let i=0;i<jobs.length;i++)pending.push(i);}
  if(pending.length){
    const size=cars,model=await batchModel(manifest,'teacher',size),out=await batchOutputs(model,pending.map(i=>jobs[i]),size);
    for(let j=0;j<pending.length;j++){
      const i=pending[j],a=j*16,b=j*2;
      const r=VehicleFacing.classify(out.view_logits.slice(a,a+8),out.angle_fraction[b],manifest,jobs[i].lights,{logits:out.view_logits.slice(a+8,a+16),angle:out.angle_fraction[b+1]});
      results[i]={...r,model:'CFV ConvNeXt-Small ('+(model.variant.includes('gpu32')?'FP32':'FP16')+')',backend:model.backend,gpu_capture:!!model.capture,path:'full-resolution',gpu_batch_cars:size};
    }
  }
  const elapsed=performance.now()-start;
  return results.map((r,i)=>({...r,crop:jobs[i].crop,fast_inference_ms:fastMs,fast_error:fastError,inference_ms:Math.round(elapsed/jobs.length),batch_inference_ms:Math.round(elapsed),batch_count:jobs.length,model_ran:true}));
}
// Serial messages prevent graph/buffer mutation during a captured GPU run.
let work=Promise.resolve();
self.onmessage=({data})=>work=work.then(async()=>{
  try{
    if(data.jobs){
      if(!data.jobs.length||data.jobs.length>8||data.batchSize&&(![1,2,4,8].includes(data.batchSize)||data.jobs.length>data.batchSize))throw Error('Expected 1–8 car crops within the selected batch size');
      let results,batchError=null;
      if(data.batchSize!==1)try{results=await batch(data);}catch(error){batchError=error.message;}
      if(!results){results=[];for(const job of data.jobs)results.push(await single(job));}
      const last=results.at(-1);self.postMessage({phase:'ready',backend:last.backend,path:last.path,model:last.model,fallback:batchError||last.fallback,batchCars:Math.max(...results.map(r=>r.gpu_batch_cars||1))});
      self.postMessage({id:data.id,results,batchError});
    }else self.postMessage({id:data.id,result:await single(data)});
  }catch(error){self.postMessage({id:data.id,error:error.message||String(error)});}
});
