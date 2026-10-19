importScripts('inference-runtime.js?v=facing-v2','orientation.js?v=facing-v2');
let ready,teacherReady;
async function load(){
  if(!ready)ready=(async()=>{
    self.postMessage({phase:'loading'});
    const response=await fetch('models/vehicle-facing-v2.json?v=facing-v2');if(!response.ok)throw Error('Facing manifest HTTP '+response.status);const manifest=await response.json();
    let quick=null,quickError=null;
    if(manifest.fast)try{quick=await new ModelRuntime.Session({file:'models/'+manifest.fast.file,sha256:manifest.fast.sha256}).initialize();}catch(error){quickError=error.message;}
    return {manifest,quick,quickError};
  })().catch(error=>{ready=null;throw error;});return ready;
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
  let result=VehicleFacing.classify(prediction.logits,prediction.angle,manifest,lights);
  if(result.state==='estimated')result=VehicleFacing.classify(prediction.logits,prediction.angle,manifest,lights,flipped||await predict(model,mirror(input)));
  return result;
}
async function fastPredict(model,input,manifest,lights){
  const pair=new Float32Array(input.length*2);pair.set(input);pair.set(mirror(input),input.length);
  const tensor=new ort.Tensor('float32',pair,[2,3,224,224]);let outputs;
  try{outputs=await model.run({[model.inputNames[0]]:tensor});return VehicleFacing.fastClassify(Array.from(outputs.view_logits.data),Array.from(outputs.angle_vector.data),manifest,lights);}
  finally{tensor.dispose?.();if(outputs)for(const t of Object.values(outputs))t.dispose?.();}
}
self.onmessage=async({data})=>{
  try{
    const {manifest,quick,quickError}=await load(),start=performance.now();let result,model,path='full-resolution',fastMs=null,fastError=quickError;
    if(quick){const quickStart=performance.now();try{result=await fastPredict(quick,data.input,manifest,data.lights);if(result.state==='estimated'){model=quick;path='fast-verified';}}catch(error){fastError=error.message;}fastMs=Math.round(performance.now()-quickStart);}
    if(!model){
      model=await teacher(manifest);result=await fullPredict(model,data.input,manifest,data.lights);
    }
    const modelName=path==='fast-verified'?manifest.fast.model:model.variant?.includes('fp16')?'CFV ConvNeXt-Small (FP16)':model.variant?.includes('batch2-fp32')?'CFV ConvNeXt-Small (FP32)':manifest.model;
    self.postMessage({phase:'ready',backend:model.backend,fallback:model.fallback,model:modelName,path});
    self.postMessage({id:data.id,result:{...result,model:modelName,backend:model.backend,gpu_capture:!!model.capture,fallback:model.fallback,path,fast_inference_ms:fastMs,fast_error:fastError,crop:data.crop,inference_ms:Math.round(performance.now()-start)}});
  }catch(error){self.postMessage({id:data.id,error:error.message||String(error)});}
};
