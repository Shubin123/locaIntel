importScripts('inference-runtime.js?v=webgl-1','detector.js?v=webgl-1');
let sessionPromise;
async function webglSpec(name){try{const r=await fetch('models/webgl-v1.json?v=webgl-1');if(!r.ok)return null;const spec=(await r.json())[name];return spec&&{...spec,file:'models/'+spec.file};}catch(_){return null;}}
async function session(){
  if(!sessionPromise){self.postMessage({phase:'loading'});sessionPromise=webglSpec('yolo').then(webgl=>new ModelRuntime.Session({file:'models/yolo11n.onnx'},null,webgl).initialize()).catch(error=>{sessionPromise=null;throw error;});}return sessionPromise;
}
self.onmessage=async({data})=>{
  try{
    const model=await session(),inputTensor=new ort.Tensor('float32',data.input,[1,3,640,640]),start=performance.now();let output;
    try{output=await model.run({[model.inputNames[0]]:inputTensor});}finally{inputTensor.dispose?.();}
    const tensor=output[model.outputNames[0]],detections=YoloBoxes.decode(tensor.data,tensor.dims,data.geometry);
    for(const result of Object.values(output))result.dispose?.();
    self.postMessage({id:data.id,detections,backend:model.backend,inference_ms:Math.round(performance.now()-start),fallback:model.fallback});
  }catch(error){self.postMessage({id:data.id,error:error.message||String(error)});}
};
