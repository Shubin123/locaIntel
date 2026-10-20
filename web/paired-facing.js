/* Learned two-view confirmation over distinct cached screenshots. No pseudo-labels. */
(function(root){
  const VERSION='paired-facing-v1',BASE_VERSION='cfv-facing-cascade-v4',MAX_GAP_MS=15000,MAX_FRAMES=5;
  const fingerprint=d=>JSON.stringify([d.label,d.box]);
  const detections=row=>row?.browserResult?.detections||row?.detections||[];
  const signature=ds=>JSON.stringify(ds.map(fingerprint));
  const circular=(a,b)=>Math.abs(((a-b+540)%360)-180);
  const validBox=d=>d?.label==='car'&&Array.isArray(d.box)&&d.box.length===4&&d.box.every(v=>Number.isFinite(v)&&v>=0&&v<=1)&&d.box[2]>d.box[0]&&d.box[3]>d.box[1];
  function validFeatures(row){
    const f=row?.features,valid=d=>d&&typeof d.label==='string'&&Array.isArray(d.box)&&d.box.length===4&&d.box.every(v=>Number.isFinite(v)&&v>=0&&v<=1)&&d.box[2]>d.box[0]&&d.box[3]>d.box[1]&&Array.isArray(d.appearance)&&d.appearance.length>=108&&d.appearance.every(v=>Number.isFinite(v)&&v>=0&&v<=1);
    const ds=detections(row);
    return typeof row?.hash==='string'&&row.hash.length>0&&['string','number'].includes(typeof row.cameraId)&&String(row.cameraId).length>0&&Number.isFinite(row.observedAt)&&f?.version===1&&f.hash===row.hash&&Array.isArray(ds)&&ds.every(d=>d&&typeof d.label==='string'&&Array.isArray(d.box))&&Array.isArray(f.detections)&&f.detections.every(valid)&&f.detectionSignature===signature(ds)&&signature(f.detections)===f.detectionSignature&&Number.isInteger(f.width)&&f.width>0&&Number.isInteger(f.height)&&f.height>0&&typeof f.pixelSignature==='string'&&f.pixelSignature.length>0&&Array.isArray(f.background)&&f.background.length>=16&&f.background.every(v=>v===null||Number.isFinite(v)&&v>=0&&v<=1);
  }

  function difference(a,b){if(!Array.isArray(a)||!Array.isArray(b)||a.length!==b.length||!a.length)return Infinity;let sum=0,n=0;for(let i=0;i<a.length;i++)if(Number.isFinite(a[i])&&Number.isFinite(b[i])){sum+=Math.abs(a[i]-b[i]);n++;}return n>=a.length*.4?sum/n:Infinity;}
  function rowsFor(current,history){return !current||!Array.isArray(history)?[]:history.filter(r=>r&&String(r.cameraId)===String(current.cameraId)&&r.hash!==current.hash&&Number.isFinite(r.observedAt)&&r.observedAt<current.observedAt&&current.observedAt-r.observedAt<=MAX_GAP_MS).sort((a,b)=>b.observedAt-a.observedAt).slice(0,MAX_FRAMES-1);}
  function context(current,history){
    const state={current,candidates:[],reason:'need_second_screenshot',frames:current?[current.hash]:[]};
    if(!current)return state;if(!validFeatures(current)){state.reason='screenshot_features_pending';return state;}
    if(typeof root.TemporalTraffic?.matches!=='function'){state.reason='screenshot_association_unavailable';return state;}
    const seen=new Set([current.features.pixelSignature]);
    for(const row of rowsFor(current,history)){
      if(!validFeatures(row)){state.reason='screenshot_features_pending';continue;}
      const a=row.features,b=current.features;if(seen.has(a.pixelSignature)){state.reason='duplicate_screenshot_pixels';continue;}seen.add(a.pixelSignature);
      if(a.width!==b.width||a.height!==b.height||difference(a.background,b.background)>=.035){state.reason='camera_view_changed';continue;}
      // Association is intentionally independent of the facing predictions.
      const clean=f=>({...f,detections:f.detections.map(d=>validBox(d)&&Array.isArray(d.appearance)&&d.appearance.length>=108&&d.appearance.every(Number.isFinite)?d:{...d,appearance:null})});
      const matches=root.TemporalTraffic.matches(clean(a),clean(b));
      state.candidates.push({row,matches,gapMs:current.observedAt-row.observedAt});state.frames.push(row.hash);
    }
    if(state.candidates.length)state.reason='vehicle_unmatched_across_screenshots';return state;
  }
  function baseFor(row,d){const key=fingerprint(d),r=row?.facing?.byFingerprint?.[key]||Object.values(row?.facing?.byDetection||{}).find(r=>r?.fingerprint===key);return r?.version===BASE_VERSION&&r.fingerprint===key?r:null;}
  function features(current,previous){
    const log=p=>Math.log(Math.max(1e-6,p)),r=Math.PI/180;
    return [...current.view_scores.map(log),...previous.view_scores.map(log),Math.sin(current.angle_degrees*r),Math.cos(current.angle_degrees*r),Math.sin(previous.angle_degrees*r),Math.cos(previous.angle_degrees*r),current.score,previous.score,circular(current.angle_degrees,previous.angle_degrees)/180];
  }
  function predict(current,previous,model){
    const x=features(current,previous).map((v,i)=>(v-model.mean[i])/model.std[i]),logits=current.view_scores.map(p=>Math.log(Math.max(1e-6,p)));
    for(let j=0;j<8;j++){logits[j]+=model.bias[j];for(let i=0;i<x.length;i++)logits[j]+=x[i]*model.weights[i][j];}
    const values=logits.map(v=>Math.exp(v-Math.max(...logits))),sum=values.reduce((a,b)=>a+b),scores=values.map(v=>v/sum),index=scores.indexOf(Math.max(...scores));return {index,score:scores[index],scores};
  }
  function validateModel(model){
    if(model?.version!==VERSION||model.base_version!==BASE_VERSION||model.minimum_frames!==2||model.max_frames!==MAX_FRAMES||model.max_gap_ms!==MAX_GAP_MS||![model.mean,model.std,model.weights,model.bias].every(Array.isArray)||model.mean.length!==23||model.std.length!==23||model.weights.length!==23||model.bias.length!==8||!model.weights.every(r=>Array.isArray(r)&&r.length===8)||![...model.mean,...model.std,...model.bias,...model.weights.flat()].every(Number.isFinite)||model.std.some(v=>v<=0)||!Number.isFinite(model.threshold)||!(model.threshold>=0&&model.threshold<=1))throw Error('Invalid paired facing model');return model;
  }
  const probabilities=p=>Array.isArray(p)&&p.length===8&&p.every(v=>Number.isFinite(v)&&v>=0&&v<=1)&&Math.abs(p.reduce((a,b)=>a+b,0)-1)<1e-5;
  function summarize(p){const groups={front:p[0]+p[1]+p[7],rear:p[3]+p[4]+p[5],side:p[2]+p[6]},order=Object.keys(groups).sort((a,b)=>groups[b]-groups[a]);return {index:p.indexOf(Math.max(...p)),group:order[0],score:groups[order[0]],margin:groups[order[0]]-groups[order[1]]};}
  function eligible(r){
    if(r?.version!==BASE_VERSION||r.model_ran!==true||r.state!=='estimated'||r.quality_abstention||r.paired||r.mirror_consistent!==true||!probabilities(r.view_scores)||!probabilities(r.mirror_view_scores)||![r.angle_degrees,r.mirror_angle_degrees].every(v=>Number.isFinite(v)&&v>=0&&v<=360))return false;
    const p=summarize(r.view_scores),flip=summarize(r.mirror_view_scores),error=circular(r.mirror_angle_degrees,(360-r.angle_degrees)%360);
    if(p.score<.68||p.margin<.3||circular(r.angle_degrees,p.index*45)>67.5||p.group!==flip.group||flip.score<.6||error>45||!Number.isFinite(r.score)||Math.abs(r.score-p.score)>1e-5||!Number.isFinite(r.margin)||Math.abs(r.margin-p.margin)>1e-5)return false;
    const lamp=r.lights?.type==='headlight_pair'?'front':r.lights?.type==='taillight_pair'?'rear':null;
    if(lamp&&lamp!==p.group&&p.group!=='side')return false;
    const direction=p.group==='front'?'toward_camera':p.group==='rear'?'away_from_camera':r.view_scores[p.index]>=.65?(p.index===2?'image_right':'image_left'):'side_view';
    if(r.facing!==direction)return false;
    // Recheck the stored mirror relationship instead of trusting the cached flag.
    // The fast path additionally requires exact eight-view mirror agreement.
    if(r.path==='fast-verified'||r.method==='cfv_low_resolution_supervised_mobilenet_mirror_verified')return r.view_scores[p.index]>=.95&&r.mirror_view_scores[flip.index]>=.95&&flip.index===(8-p.index)%8&&circular(r.angle_degrees,p.index*45)<=45&&error<=25;
    return true;
  }
  function combine(base,d,state,model){
    if(!base||['queued','processing','error'].includes(base.state)||!validBox(d))return base;
    state=state||{current:null,candidates:[],reason:'need_second_screenshot'};
    const result={...base,state:'unknown',facing:'unknown',image_direction:undefined,facing_glyph:undefined,single_frame_state:base.state,reason:state.reason,paired:{version:VERSION,state:'pending',count:1,frameHashes:[state.current?.hash],frames:[],model_version:model?.version||null,measured_motion:false}};
    if(!eligible(base)||base.fingerprint!==fingerprint(d)){result.reason=base.quality_abstention?base.reason:'single_frame_facing_unverified';return result;}
    if(!model){result.reason='paired_facing_model_loading';return result;}
    try{validateModel(model);}catch(_){result.reason='paired_facing_model_invalid';return result;}
    if(!validFeatures(state.current)){result.reason='screenshot_features_pending';return result;}
    const index=state.current?.features?.detections?.findIndex(p=>fingerprint(p)===fingerprint(d));if(index==null||index<0)return result;
    const checks=[];let pending=false;
    for(const candidate of state.candidates||[]){
      const previousIndex=candidate.matches.get(index);if(previousIndex==null)continue;
      const old=candidate.row.features.detections[previousIndex],prior=baseFor(candidate.row,old);if(!eligible(prior)){result.reason='previous_facing_pending';pending=true;continue;}
      const latestView=base.view_scores.indexOf(Math.max(...base.view_scores)),priorView=prior.view_scores.indexOf(Math.max(...prior.view_scores));
      const prediction=predict(base,prior,model),agree=latestView===priorView&&base.facing===prior.facing&&circular(base.angle_degrees,prior.angle_degrees)<=22.5&&prediction.index===latestView&&prediction.score>=model.threshold;
      checks.push({agree,prediction,frame:{key:candidate.row.key,hash:candidate.row.hash,cameraId:String(candidate.row.cameraId),observedAt:candidate.row.observedAt,pixelSignature:candidate.row.features.pixelSignature,detectionSignature:candidate.row.features.detectionSignature,fingerprint:fingerprint(old)}});
    }
    if(!checks.length)return result;
    if(checks.some(c=>!c.agree)){result.reason='cached_facing_disagreement';result.paired.state='conflicting';return result;}
    if(pending)return result;
    const current=state.current,frames=[...checks.map(c=>c.frame),{key:current.key,hash:current.hash,cameraId:String(current.cameraId),observedAt:current.observedAt,pixelSignature:current.features.pixelSignature,detectionSignature:current.features.detectionSignature,fingerprint:fingerprint(d)}];
    return {...base,single_frame_state:base.state,reason:'two_screenshot_facing_confirmed',paired:{version:VERSION,state:'confirmed',count:frames.length,frameHashes:frames.map(f=>f.hash),frames,model_version:model.version,min_model_score:Math.min(...checks.map(c=>c.prediction.score)),measured_motion:false}};
  }
  root.PairedFacing={VERSION,BASE_VERSION,MAX_GAP_MS,MAX_FRAMES,validFeatures,rowsFor,context,baseFor,features,predict,validateModel,eligible,combine};
})(globalThis);
