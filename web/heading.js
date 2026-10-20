/* Conservative spatial evidence fusion. User arrows are references, not motion. */
(function(root){
  const VERSION='spatial-heading-v2',names=['right','down-right','down','down-left','left','up-left','up','up-right'];
  const directions={right:[1,0],'down-right':[1,1],down:[0,1],'down-left':[-1,1],left:[-1,0],'up-left':[-1,-1],up:[0,-1],'up-right':[1,-1]};
  const fingerprint=d=>JSON.stringify([d.label,d.box]);
  const detectionSignature=ds=>JSON.stringify(ds.map(fingerprint));
  const frameDetections=row=>row.browserResult?.detections||row.detections||[];
  function mergeTemporal(ds,temporal){
    const records=new Map((temporal?.detections||[]).map(d=>[fingerprint(d),d]));
    return ds.map(d=>{const t=records.get(fingerprint(d));return t?{...d,motion:t.motion?.state!=='unknown'||!d.motion? t.motion:d.motion}:d;});
  }
  function facing(row,d,index){const key=fingerprint(d),saved=row?.facing;const r=saved?.byFingerprint?.[key]||saved?.byDetection?.[index];return r?.fingerprint===key?r:null;}
  function normalize(vector,aspect){if(!Array.isArray(vector)||vector.length!==2||!vector.every(Number.isFinite))return null;const n=Math.hypot(vector[0]*aspect,vector[1]);return n>1e-8?[vector[0]*aspect/n,vector[1]/n]:null;}
  const dot=(a,b)=>a[0]*b[0]+a[1]*b[1];
  function sanitize(vectors){
    if(!Array.isArray(vectors)||vectors.length>64)throw Error('Use at most 64 direction references per camera');
    return vectors.map(v=>{if(![v.start,v.end].every(p=>Array.isArray(p)&&p.length===2&&p.every(n=>Number.isFinite(n)&&n>=0&&n<=1))||Math.hypot(v.end[0]-v.start[0],v.end[1]-v.start[1])<.02)throw Error('Each arrow needs two distinct image points between 0 and 1');return {start:[...v.start],end:[...v.end]};});
  }
  function stableView(reference,scene){
    if(!reference)return false;
    if(reference.frameHash===scene.hash)return true;
    if(!(reference.width>0&&reference.height>0&&scene.width>0&&scene.height>0)||Math.abs(reference.width/reference.height/(scene.width/scene.height)-1)>.01)return false;
    const a=reference.background,b=scene.background;if(!Array.isArray(a)||!Array.isArray(b)||a.length!==b.length)return false;
    const pairs=a.map((v,i)=>[v,b[i]]).filter(p=>p.every(Number.isFinite));
    return pairs.length>=a.length*.4&&pairs.reduce((s,p)=>s+Math.abs(p[0]-p[1]),0)/pairs.length<.035;
  }
  function nearby(reference,scene,box,aspect){
    if(!stableView(reference,scene))return {state:'unavailable',reason:'reference_view_unverified',vectors:[]};
    const p=[(box[0]+box[2])/2*aspect,box[3]],candidates=[];
    for(const [index,v]of reference.vectors.entries()){
      const a=[v.start[0]*aspect,v.start[1]],b=[v.end[0]*aspect,v.end[1]],delta=b.map((x,i)=>x-a[i]),length2=dot(delta,delta);if(!length2)continue;
      const t=Math.max(0,Math.min(1,dot(p.map((x,i)=>x-a[i]),delta)/length2)),distance=Math.hypot(...p.map((x,i)=>x-a[i]-t*delta[i]));
      if(distance<=.2)candidates.push({index,distance,unit:delta.map(x=>x/Math.sqrt(length2))});
    }
    candidates.sort((a,b)=>a.distance-b.distance);const close=candidates.filter(c=>c.distance<=candidates[0].distance+.04);
    if(!close.length)return {state:'unavailable',reason:'no_nearby_reference',vectors:[]};
    if(close.some(c=>dot(c.unit,close[0].unit)<.8))return {state:'unknown',reason:'conflicting_reference_vectors',vectors:close.map(c=>c.index)};
    const sum=close.reduce((s,c)=>s.map((x,i)=>x+c.unit[i]/(.02+c.distance)),[0,0]),n=Math.hypot(...sum);
    return {state:'ready',unit:sum.map(x=>x/n),vectors:close.map(c=>c.index)};
  }
  function classify(d,{width=640,height=400,hash,background,reference}={}){
    const aspect=width/height,o=d.orientation,scene={width,height,hash,background};
    const result={version:VERSION,state:'unknown',reason:'insufficient_direction_evidence',measured_motion:false,referenceRevision:reference?.revision||0,sources:[],fingerprint:fingerprint(d)};
    const estimated=o?.state==='estimated'&&o.mirror_consistent===true&&directions[o.image_direction];
    const appearance=estimated?normalize(directions[o.image_direction],1):null;
    const guides=reference?.vectors?.length?nearby(reference,scene,d.box,aspect):{state:'unavailable',vectors:[]};
    const road=o?.road_direction?.state==='estimated'?normalize(o.road_direction.vector,aspect):null;
    const finish=(unit,state,reason,sources,measured=false)=>({...result,state,reason,sources,measured_motion:measured,vector:[unit[0]/aspect,unit[1]],direction:names[Math.floor(((Math.atan2(unit[1],unit[0])*180/Math.PI+382.5)%360)/45)%8],referenceVectors:guides.vectors});
    if(typeof root.validVector==='function'&&root.validVector(d)){
      const motion=normalize(d.motion.vector,aspect);
      return {...finish(motion,'measured','validated_temporal_motion',['temporal_motion'],true),facing_conflict:!!appearance&&dot(motion,appearance)<0};
    }
    if(!appearance)return {...result,referenceStatus:guides.state,referenceReason:guides.reason};
    if(o.road_direction?.state==='unknown')return {...result,reason:'facing_road_axis_disagree',sources:['crop_facing','road_axis']};
    if(guides.state==='unknown')return {...result,reason:guides.reason,sources:['crop_facing','user_vectors']};
    if(guides.state==='ready'){
      if(dot(appearance,guides.unit)<.55||road&&dot(road,guides.unit)<.8)return {...result,reason:'reference_facing_disagreement',sources:['crop_facing','user_vectors'],referenceVectors:guides.vectors};
      // References refine geometry only after the trained facing estimate
      // agrees; nearby cars never supply a missing vehicle identity or label.
      return finish(road||guides.unit,'estimated','facing_and_spatial_vectors_agree',['crop_facing',...(road?['road_axis']:[]),'user_vectors']);
    }
    return {...finish(road||appearance,'estimated',road?'road_aligned_facing':'appearance_only',['crop_facing',...(road?['road_axis']:[])]),referenceStatus:guides.state,referenceReason:guides.reason};
  }
  function attach(c,row,reference){
    const scene={width:c._photoWidth||row?.width||640,height:c._photoHeight||row?.height||400,hash:c.hash,background:row?.temporal?.scene?.background,reference};
    return {...c,detections:(c.detections||[]).map(d=>({...d,heading:classify(d,scene)}))};
  }
  root.SpatialHeading={VERSION,fingerprint,detectionSignature,frameDetections,mergeTemporal,facing,sanitize,stableView,nearby,classify,attach};
})(globalThis);
