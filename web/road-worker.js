importScripts('inference-runtime.js?v=facing-v2','road.js?v=gpu-road-1');
let ready;
async function load(){if(!ready)ready=(async()=>{self.postMessage({phase:'loading'});const response=await fetch('models/road-v1.json');if(!response.ok)throw Error('Road manifest HTTP '+response.status);const manifest=await response.json(),model=await new ModelRuntime.Session({file:'models/'+manifest.file,sha256:manifest.sha256}).initialize();return {manifest,model};})().catch(error=>{ready=null;throw error;});return ready;}
self.onmessage=async({data})=>{try{
  const {manifest,model}=await load(),input=new ort.Tensor('float32',data.input,[1,3,256,256]),start=performance.now();let outputs;
  try{outputs=await model.run({[model.inputNames[0]]:input});const values=outputs.road_probability.data,size=64,mask=Array.from({length:24},(_,y)=>Array.from({length:32},(_,x)=>{let sum=0,n=0;for(let yy=Math.floor(y*size/24);yy<Math.floor((y+1)*size/24);yy++)for(let xx=x*2;xx<x*2+2;xx++){sum+=values[yy*size+xx];n++;}return Math.round(sum/n*100)/100;}));
    self.postMessage({phase:'ready',backend:model.backend});self.postMessage({id:data.id,result:{version:RoadVision.VERSION,state:'ready',model:manifest.model,road_mask:mask,skeleton:RoadVision.skeleton(mask),width:data.width,height:data.height,backend:model.backend,inference_ms:Math.round(performance.now()-start),fallback:model.fallback}});
  }finally{input.dispose?.();if(outputs)for(const tensor of Object.values(outputs))tensor.dispose?.();}
}catch(error){self.postMessage({id:data.id,error:error.message||String(error)});}};
