/* Conservative three-observation tracking. Image directions are not compass bearings. */
(function(root){
  const labels=new Set(['car','truck','bus','motorcycle','bicycle']);
  const center=d=>[(d.box[0]+d.box[2])/2,(d.box[1]+d.box[3])/2];
  const distance=(a,b)=>Math.hypot(...a.map((v,i)=>v-b[i]));
  const difference=(a,b)=>{const pairs=a.map((v,i)=>[v,b[i]]).filter(p=>p.every(Number.isFinite));return pairs.length<a.length*.4?1:pairs.reduce((s,p)=>s+Math.abs(p[0]-p[1]),0)/pairs.length;};
  function matches(a,b){
    const result=new Map(),used=new Set();
    for(let j=0;j<b.detections.length;j++){
      const d=b.detections[j];if(!labels.has(d.label))continue;
      const candidates=a.detections.map((p,i)=>({i,p,error:p.label===d.label&&p.appearance&&d.appearance?difference(p.appearance,d.appearance):1})).filter(x=>x.error<.065&&distance(center(x.p),center(d))<.12&&Math.abs(Math.log(((x.p.box[2]-x.p.box[0])*(x.p.box[3]-x.p.box[1]))/((d.box[2]-d.box[0])*(d.box[3]-d.box[1]))))<.55).sort((x,y)=>x.error-y.error);
      if(candidates.length&&(!candidates[1]||candidates[1].error-candidates[0].error>.025)&&!b.detections.some((other,k)=>k!==j&&other.label===d.label&&other.appearance&&distance(center(candidates[0].p),center(other))<.12&&difference(candidates[0].p.appearance,other.appearance)<=candidates[0].error+.025)&&!used.has(candidates[0].i)){result.set(j,candidates[0].i);used.add(candidates[0].i);}
    }return result;
  }
  function analyze(frames){
    const c=frames.at(-1);if(!c)return null;
    const detections=c.detections.map(({appearance,...d})=>({...d,motion:{state:'unknown',reason:'need_three_distinct_frames'}}));
    const log={version:1,frameHashes:frames.map(f=>f.hash),observedAt:c.observedAt,status:'need_three_distinct_frames',states:{moving:0,stationary:0,unknown:0}};
    if(frames.length===3&&new Set(frames.map(f=>f.hash)).size===3){
      const [a,b]=frames,gaps=[(b.observedAt-a.observedAt)/1000,(c.observedAt-b.observedAt)/1000];
      const stable=frames.every(f=>f.width===c.width&&f.height===c.height&&f.background)&&difference(a.background,b.background)<.035&&difference(b.background,c.background)<.035;
      log.status=frames.some((f,i)=>i&&f.pixelSignature&&f.pixelSignature===frames[i-1].pixelSignature)?'duplicate_frame':!stable?'unstable_background':gaps.some(g=>g<=0||g>1800)?'insufficient_cadence':'temporal_analyzed';
      if(log.status==='temporal_analyzed'){
        const ab=matches(a,b),bc=matches(b,c);
        log.turnover={arrivals:c.detections.filter(d=>labels.has(d.label)).length-bc.size,departures:b.detections.filter(d=>labels.has(d.label)).length-new Set(bc.values()).size};
        for(const [ci,bi]of bc){const ai=ab.get(bi);if(ai==null)continue;
          const p=center(a.detections[ai]),q=center(b.detections[bi]),r=center(c.detections[ci]),v=q.map((x,i)=>x-p[i]),w=r.map((x,i)=>x-q[i]),lengths=[Math.hypot(...v),Math.hypot(...w)];
          const stationary=lengths.every(n=>n<.003)&&distance(p,r)<.004;
          const moving=gaps.every(g=>g<=15)&&lengths.every(n=>n>.006)&&v.reduce((s,x,i)=>s+x*w[i],0)/(lengths[0]*lengths[1])>.9&&Math.max(...lengths)/Math.min(...lengths)<2;
          detections[ci].motion={state:stationary?'stationary':moving?'moving':'unknown',reason:stationary?'persistent_position_three_frames':moving?'consistent_three_frame_track':'ambiguous_displacement',evidence_kind:'three_frame_appearance',vector:moving?w:null,gap_seconds:gaps[1],evidence:{frames:3,window_seconds:gaps[0]+gaps[1],background_stable:true},confidence:(stationary||moving)? .8:0};
        }
      }
    }
    for(const d of detections)if(labels.has(d.label))log.states[d.motion.state]++;
    return {detections,log};
  }
  async function features(blob,detections){
    const bitmap=await createImageBitmap(blob),canvas=document.createElement('canvas');canvas.width=96;canvas.height=64;const ctx=canvas.getContext('2d',{willReadFrequently:true});ctx.drawImage(bitmap,0,0,96,64);
    const pixels=ctx.getImageData(0,0,96,64).data;
    const sample=(x,y)=>{const i=(Math.min(63,Math.max(0,Math.floor(y)))*96+Math.min(95,Math.max(0,Math.floor(x))))*4;return [pixels[i]/255,pixels[i+1]/255,pixels[i+2]/255];};
    const crop=document.createElement('canvas');crop.width=12;crop.height=12;const cc=crop.getContext('2d',{willReadFrequently:true});
    const ds=detections.map(d=>{const [x,y,xx,yy]=d.box;cc.clearRect(0,0,12,12);cc.drawImage(bitmap,x*bitmap.width,y*bitmap.height,(xx-x)*bitmap.width,(yy-y)*bitmap.height,0,0,12,12);const rgba=cc.getImageData(0,0,12,12).data,appearance=[];for(let i=0;i<rgba.length;i++)if(i%4!==3)appearance.push(rgba[i]/255);return {...d,appearance};});bitmap.close();
    const background=[];for(let y=8;y<64;y+=4)for(let x=2;x<96;x+=4){if(detections.some(d=>x/96>d.box[0]-.02&&x/96<d.box[2]+.02&&y/64>d.box[1]-.02&&y/64<d.box[3]+.02))background.push(null);else background.push(sample(x,y).reduce((a,b)=>a+b)/3);}
    let signature=2166136261;for(const value of pixels)signature=Math.imul(signature^value,16777619);return {detections:ds,background,pixelSignature:String(signature>>>0)};
  }
  root.TemporalTraffic={analyze,features,matches};
})(globalThis);
