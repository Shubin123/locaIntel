importScripts('inference-runtime.js?v=gpu-road-1','orientation.js?v=gpu-road-1');
let ready;
async function load(){
  if(!ready)ready=(async()=>{
    self.postMessage({phase:'loading'});
    const response=await fetch('models/vehicle-facing-v1.json?v=gpu-road-1');if(!response.ok)throw Error('Facing manifest HTTP '+response.status);const manifest=await response.json();
    const gpu=manifest.gpu?{file:'models/'+manifest.gpu.file,sha256:manifest.gpu.sha256,requiresF16:true}:null;
    const model=await new ModelRuntime.Session({file:'models/cfv-convnext-facing-int8.onnx',sha256:manifest.sha256},gpu).initialize();return {manifest,model};
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
    self.postMessage({phase:'ready',backend:model.backend});self.postMessage({id:data.id,result:{...result,model:model.backend==='webgpu'?'CFV ConvNeXt-Small (FP16 / WebGPU)':manifest.model,backend:model.backend,fallback:model.fallback,crop:data.crop,inference_ms:Math.round(performance.now()-start)}});
  }catch(error){self.postMessage({id:data.id,error:error.message||String(error)});}
};
