/* Vehicle facing from isolated crops. These estimates never set motion.state. */
(function(root){
  const VERSION='cfv-facing-cascade-v2',SIZE=224,supported=new Set(['car']);
  const fingerprint=d=>JSON.stringify([d.label,d.box]);
  function lampPairs(rgba,width,height){
    const find=kind=>{
      const mask=new Uint8Array(width*height),components=[];
      for(let y=Math.floor(height*.5);y<height*.92;y++)for(let x=0;x<width;x++){
        const i=y*width+x,p=i*4,r=rgba[p],g=rgba[p+1],b=rgba[p+2];
        mask[i]=kind==='red'?(r>115&&r>g*1.65&&r>b*1.65?1:0):((r>205&&g>190&&b>160&&Math.max(r,g,b)-Math.min(r,g,b)<65)?1:0);
      }
      for(let i=0;i<mask.length;i++)if(mask[i]){
        const stack=[i];mask[i]=0;let n=0,sx=0,sy=0;
        while(stack.length){const p=stack.pop(),x=p%width,y=Math.floor(p/width);n++;sx+=x;sy+=y;for(const q of [x>0?p-1:-1,x<width-1?p+1:-1,y>0?p-width:-1,y<height-1?p+width:-1])if(q>=0&&mask[q]){mask[q]=0;stack.push(q);}}
        if(n>=2&&n<=width*height*.055)components.push({n,x:sx/n/width,y:sy/n/height});
      }
      let best=null;
      for(let i=0;i<components.length;i++)for(let j=i+1;j<components.length;j++){
        const a=components[i],b=components[j],dx=Math.abs(a.x-b.x),dy=Math.abs(a.y-b.y),ratio=Math.min(a.n,b.n)/Math.max(a.n,b.n);
        if(dx<.22||dx>.85||dy>.10||ratio<.55||(a.y+b.y)/2<.55)continue;
        const score=(1-dy/.1)*ratio;if(score<.55)continue;if(!best||score>best.score)best={score,points:[[a.x,a.y],[b.x,b.y]]};
      }return best;
    };
    const red=find('red'),white=find('white');
    if(red&&white)return {type:'ambiguous',reason:'both_light_colors',points:[]};
    const pair=red||white;return pair?{type:red?'taillight_pair':'headlight_pair',score:pair.score,points:pair.points,method:'paired-color-components; heuristic'}:{type:'none',points:[]};
  }
  const circularError=(a,b)=>Math.abs(((a-b+540)%360)-180);
  function decode(logits,angle){
    if(logits.length!==8||!Array.from(logits).every(Number.isFinite)||!Number.isFinite(angle)||angle<0||angle>1)throw Error('Invalid facing model output');
    const maximum=Math.max(...logits),values=Array.from(logits,v=>Math.exp(v-maximum)),sum=values.reduce((a,b)=>a+b),probabilities=values.map(v=>v/sum);
    const index=probabilities.indexOf(Math.max(...probabilities)),groups={front:probabilities[0]+probabilities[1]+probabilities[7],rear:probabilities[3]+probabilities[4]+probabilities[5],side:probabilities[2]+probabilities[6]},order=Object.keys(groups).sort((a,b)=>groups[b]-groups[a]);
    return {index,probabilities,groups,group:order[0],score:groups[order[0]],margin:groups[order[0]]-groups[order[1]],angle:angle*360};
  }
  function classify(logits,angle,manifest,lights={type:'none'},mirrored=null){
    const p=decode(logits,angle),lightView=lights.type==='headlight_pair'?'front':lights.type==='taillight_pair'?'rear':null;
    const result={version:VERSION,model:manifest.model,state:'unknown',facing:'unknown',view:manifest.views[p.index],score:p.score,margin:p.margin,view_scores:p.probabilities,angle_degrees:p.angle,angle_convention:manifest.angle_convention,lights,method:'cfv_supervised_view_and_angle_plus_lamp_evidence',measured_motion:false};
    if(p.score<.68||p.margin<.3){result.reason='ambiguous_model_scores';return result;}
    if(circularError(p.angle,p.index*45)>67.5){result.reason='angle_view_disagreement';return result;}
    if(lightView&&lightView!==p.group&&p.group!=='side'){result.reason='lamp_model_disagreement';return result;}
    if(mirrored){
      const flip=decode(mirrored.logits,mirrored.angle),error=circularError(flip.angle,(360-p.angle)%360);
      result.mirror_error_degrees=error;result.mirror_consistent=flip.group===p.group&&flip.score>=.6&&error<=45;
      if(!result.mirror_consistent){result.reason='mirror_disagreement';return result;}
    }
    result.state='estimated';result.reason=lightView===p.group?'model_and_light_pair_agree':'trained_crop_appearance';
    result.facing=p.group==='front'?'toward_camera':p.group==='rear'?'away_from_camera':'side_view';
    if(result.mirror_consistent&&p.probabilities[p.index]>=.65){result.image_direction=['down','down-right','right','up-right','up','up-left','left','down-left'][p.index];result.facing_glyph=['↓','↘','→','↗','↑','↖','←','↙'][p.index];if(p.group==='side')result.facing=p.index===2?'image_right':'image_left';}
    return result;
  }
  function fastClassify(logits,vectors,manifest,lights={type:'none'}){
    const policy=manifest.fast?.policy;
    if(!policy||logits.length!==16||vectors.length!==4||!vectors.every(Number.isFinite)||!logits.every(Number.isFinite))throw Error('Invalid fast facing output or policy');
    const temperature=policy.temperature,threshold=policy.threshold;
    if(!(temperature>0)||!(threshold>0&&threshold<=1))throw Error('Invalid fast facing policy');
    const angle=offset=>((Math.atan2(vectors[offset],vectors[offset+1])*180/Math.PI+360)%360)/360;
    const first=logits.slice(0,8).map(v=>v/temperature),second=logits.slice(8).map(v=>v/temperature),a=angle(0),b=angle(2);
    const p=decode(first,a),flip=decode(second,b),result=classify(first,a,manifest,lights,{logits:second,angle:b});
    const strong=Math.hypot(...vectors.slice(0,2))>=.1&&Math.hypot(...vectors.slice(2))>=.1&&p.probabilities[p.index]>=threshold&&flip.probabilities[flip.index]>=threshold&&flip.index===(8-p.index)%8&&circularError(p.angle,p.index*45)<=45&&circularError(flip.angle,(360-p.angle)%360)<=policy.mirror_max_error;
    if(!strong)return {...result,state:'unknown',facing:'unknown',image_direction:undefined,facing_glyph:undefined,reason:'fast_model_requires_full_resolution_check'};
    return {...result,method:'cfv_low_resolution_supervised_mobilenet_mirror_verified',calibration_temperature:temperature};
  }
  async function prepare(source,d){
    const image=await createImageBitmap(source.blob);
    try{
      const [x,y,xx,yy]=d.box,w=(xx-x)*image.width,h=(yy-y)*image.height;
      const crop={width:Math.round(w),height:Math.round(h)};
      if(!supported.has(d.label)||w<20||h<16||w*h<600||x<=.001||y<=.001||xx>=.999||yy>=.999)return {skip:{version:VERSION,state:'unknown',facing:'unknown',reason:!supported.has(d.label)?'unsupported_vehicle':w<20||h<16||w*h<600?'crop_too_small':'truncated_crop',crop,measured_motion:false}};
      const canvas=document.createElement('canvas');canvas.width=canvas.height=SIZE;const ctx=canvas.getContext('2d',{willReadFrequently:true});ctx.fillStyle='rgb(114,114,114)';ctx.fillRect(0,0,SIZE,SIZE);
      ctx.drawImage(image,x*image.width,y*image.height,w,h,0,0,SIZE,SIZE);
      const pixels=ctx.getImageData(0,0,SIZE,SIZE).data,visualKey=typeof crypto!=='undefined'&&crypto.subtle?[...new Uint8Array(await crypto.subtle.digest('SHA-256',pixels))].map(v=>v.toString(16).padStart(2,'0')).join(''):null,input=new Float32Array(SIZE*SIZE*3),plane=SIZE*SIZE;
      for(let i=0;i<plane;i++)for(let c=0;c<3;c++)input[c*plane+i]=pixels[i*4+c]/255;
      const lampCanvas=document.createElement('canvas');lampCanvas.width=Math.max(1,Math.round(Math.min(1,160/Math.max(w,h))*w));lampCanvas.height=Math.max(1,Math.round(Math.min(1,160/Math.max(w,h))*h));
      const lc=lampCanvas.getContext('2d',{willReadFrequently:true});lc.drawImage(image,x*image.width,y*image.height,w,h,0,0,lampCanvas.width,lampCanvas.height);
      const lights=lampPairs(lc.getImageData(0,0,lampCanvas.width,lampCanvas.height).data,lampCanvas.width,lampCanvas.height);
      lights.points=lights.points.map(([px,py])=>[x+px*(xx-x),y+py*(yy-y)]);
      return {input,label:d.label,lights,crop,visualKey};
    }finally{image.close();}
  }
  class CropOrientation{
    constructor({workerFactory=()=>new Worker('orientation-worker.js?v=facing-v2'),prepareCrop=prepare,onChange=()=>{},onResult=()=>{},isPaused=()=>false,isWanted=()=>true,delayMs=350,timeout=90000}={}){
      Object.assign(this,{workerFactory,prepareCrop,onChange,onResult,isPaused,isWanted,delayMs,timeout});this.version=VERSION;this.entries=new Map();this.visualEntries=new Map();this.queue=[];this.running=false;this.worker=null;this.serial=0;this.waiter=null;this.cooldown=null;this.phase='idle';
    }
    key(source,d){return source.cameraId+':'+source.hash+':'+fingerprint(d);}
    get(source,d){return this.entries.get(this.key(source,d));}
    request(source,d,index,priority=false){
      const key=this.key(source,d);if(this.entries.has(key)){if(priority){const task=this.queue.find(t=>t.key===key);if(task){task.priority=Number(priority)||0;this.queue.sort((a,b)=>b.priority-a.priority);}}return;}
      this.entries.set(key,{state:'queued',version:VERSION});const task={source,d,index,key,priority:Number(priority)||0};this.queue.push(task);this.queue.sort((a,b)=>b.priority-a.priority);this.pump();
    }
    ensureWorker(){if(this.worker)return;this.worker=this.workerFactory();this.worker.onmessage=({data})=>{if(data.phase){this.phase=data.phase;if(data.backend){this.backend=data.backend;this.fallback=data.fallback||null;}if(data.path)this.path=data.path;if(data.model)this.model=data.model;this.onChange();return;}const waiter=this.waiter;if(!waiter||data.id!==waiter.id)return;clearTimeout(waiter.timer);this.waiter=null;if(data.error)waiter.reject(Error(data.error));else waiter.resolve(data.result);};this.worker.onerror=e=>this.fail(Error(e.message||'Facing worker failed'));}
    fail(error){if(this.waiter){clearTimeout(this.waiter.timer);this.waiter.reject(error);this.waiter=null;}this.worker?.terminate();this.worker=null;this.phase='error';}
    infer(prepared){this.ensureWorker();const id=++this.serial;return new Promise((resolve,reject)=>{this.waiter={id,resolve,reject,timer:setTimeout(()=>this.fail(Error('Facing model timed out')),this.timeout)};try{this.worker.postMessage({id,...prepared},[prepared.input.buffer]);}catch(e){this.fail(e);}});}
    async pump(){
      if(this.running||this.cooldown||this.isPaused()||!this.queue.length)return;
      const task=this.queue.shift();if(!this.isWanted(task.source.hash)){this.entries.delete(task.key);queueMicrotask(()=>this.pump());return;}
      let skip=false;this.running=true;this.entries.set(task.key,{state:'processing',version:VERSION});this.onChange();
      try{const prepared=await this.prepareCrop(task.source,task.d),visualKey=prepared.visualKey?task.source.cameraId+':'+prepared.visualKey:null,cached=visualKey?this.visualEntries.get(visualKey):null;skip=!!prepared.skip||!!cached;const result=prepared.skip||(cached?{...cached,lights:prepared.lights,crop:prepared.crop,inference_ms:0,backend:'cache',reused_exact_crop:true}:await this.infer(prepared));if(visualKey&&!cached&&!prepared.skip){this.visualEntries.set(visualKey,result);if(this.visualEntries.size>256)this.visualEntries.delete(this.visualEntries.keys().next().value);}if(Number.isFinite(result.inference_ms))this.lastInferenceMs=result.inference_ms;this.lastReused=!!result.reused_exact_crop;const record={...result,fingerprint:fingerprint(task.d),observedAt:Date.now()};this.entries.set(task.key,record);this.onResult(task.source,task.index,record);}
      catch(error){this.entries.set(task.key,{version:VERSION,state:'error',reason:error.message,fingerprint:fingerprint(task.d)});this.phase='error';}
      finally{this.running=false;while(this.entries.size>512){const key=[...this.entries].find(([,v])=>!['queued','processing'].includes(v.state))?.[0];if(!key)break;this.entries.delete(key);}this.onChange();if(this.queue.length)this.cooldown=setTimeout(()=>{this.cooldown=null;this.pump();},skip?0:this.backend==='webgpu'?this.delayMs*.03:this.delayMs);}
    }
    setBudget({delayMs=this.delayMs}={}){this.delayMs=Math.max(0,delayMs);if(this.cooldown){clearTimeout(this.cooldown);this.cooldown=null;}this.pump();}
    wake(){this.pump();}
    retryFailed(){for(const [k,v]of this.entries)if(v.state==='error')this.entries.delete(k);if(this.phase==='error'){this.worker?.terminate();this.worker=null;this.phase='idle';}}
  }
  function description(o){if(!o)return 'Facing: waiting for crop analysis';if(o.state==='queued'||o.state==='processing')return 'Facing: analyzing isolated crop…';if(o.state==='error')return 'Facing unavailable: '+o.reason;if(o.state!=='estimated')return 'Facing unresolved · '+(o.reason||'insufficient evidence').replaceAll('_',' ');return 'Estimated facing: '+({toward_camera:'toward camera · front visible',away_from_camera:'away from camera · rear visible',image_left:'← left on image',image_right:'→ right on image',side_view:'side view · nose direction unresolved'}[o.facing]||o.facing)+(o.facing_glyph?' · approximate image facing '+o.facing_glyph:'')+(o.reason==='model_and_light_pair_agree'?' · light pair agrees':'');}
  root.VehicleFacing={VERSION,SIZE,fingerprint,lampPairs,circularError,decode,classify,fastClassify,prepare,CropOrientation,description};
})(globalThis);
