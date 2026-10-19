importScripts('inference-runtime.js?v=facing-v2','detector.js?v=gpu-road-1');
let sessionPromise;
async function session(){
  if(!sessionPromise){self.postMessage({phase:'loading'});sessionPromise=new ModelRuntime.Session({file:'models/yolo11n.onnx'}).initialize().catch(error=>{sessionPromise=null;throw error;});}return sessionPromise;
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
