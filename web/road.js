/* Fast scene road support and image-space road axes, never one-way traffic rules. */
(function(root){
  const projectionCache=new Map();
  const VERSION='segformer-b0-road-256-v1',scene={label:'road-scene',box:[0,0,1,1]};
  function skeleton(mask){
    const h=mask.length,w=mask[0].length,grid=mask.map(row=>row.map(p=>p>=.55?1:0));
    for(let iteration=0;iteration<64;iteration++){let changed=0;
      for(let pass=0;pass<2;pass++){const remove=[];
        for(let y=1;y<h-1;y++)for(let x=1;x<w-1;x++){if(!grid[y][x])continue;
          const p=[grid[y-1][x],grid[y-1][x+1],grid[y][x+1],grid[y+1][x+1],grid[y+1][x],grid[y+1][x-1],grid[y][x-1],grid[y-1][x-1]],n=p.reduce((a,b)=>a+b),transitions=p.reduce((n,v,i)=>n+(!v&&p[(i+1)%8]?1:0),0);
          if(n<2||n>6||transitions!==1)continue;
          if(pass===0?p[0]*p[2]*p[4]===0&&p[2]*p[4]*p[6]===0:p[0]*p[2]*p[6]===0&&p[0]*p[4]*p[6]===0)remove.push([x,y]);
        }for(const [x,y]of remove)grid[y][x]=0;changed+=remove.length;
      }if(!changed)break;
    }
    const points=[];for(let y=0;y<h;y++)for(let x=0;x<w;x++)if(grid[y][x])points.push([(x+.5)/w,(y+.5)/h]);return points;
  }
  function footprint(mask,box){
    const h=mask.length,w=mask[0].length,x=(box[0]+box[2])/2,y=box[3];let sum=0,n=0;
    for(const dy of [.005,.02,.035])for(const dx of [-.015,0,.015]){const xx=Math.floor((x+dx)*w),yy=Math.floor((y+dy)*h);if(xx>=0&&xx<w&&yy>=0&&yy<h){sum+=mask[yy][xx];n++;}}
    return n?sum/n:null;
  }
  function axis(points,box,width=640,height=400){
    if(points.length<4)return null;const foot=[(box[0]+box[2])/2,Math.min(.99,box[3]+.02)],aspect=width/height;
    const distance=(a,b)=>Math.hypot((a[0]-b[0])*aspect,a[1]-b[1]);
    const nearest=[...points].sort((a,b)=>distance(a,foot)-distance(b,foot))[0];if(distance(nearest,foot)>.4)return null;
    const local=points.filter(p=>distance(p,nearest)<.22);if(local.length<4)return null;
    const xs=local.map(p=>p[0]*aspect),ys=local.map(p=>p[1]),mx=xs.reduce((a,b)=>a+b)/xs.length,my=ys.reduce((a,b)=>a+b)/ys.length;
    let xx=0,yy=0,xy=0;for(let i=0;i<xs.length;i++){xx+=(xs[i]-mx)**2;yy+=(ys[i]-my)**2;xy+=(xs[i]-mx)*(ys[i]-my);}
    const trace=xx+yy,delta=Math.hypot(xx-yy,2*xy),confidence=trace?(trace+delta)/(2*trace):0;if(confidence<.8)return null;
    const angle=.5*Math.atan2(2*xy,xx-yy),vector=[Math.cos(angle)/aspect,Math.sin(angle)],norm=Math.hypot(...vector);return {vector:vector.map(v=>v/norm),confidence,points:local.length,image_aspect:aspect};
  }
  function corridor(anchors,box,mask,width=640,height=400){
    const aspect=width/height,foot=[(box[0]+box[2])/2*aspect,box[3]],local=anchors.map(p=>[p[0]*aspect,p[1]]).filter(p=>Math.hypot(p[0]-foot[0],p[1]-foot[1])<.45).sort((a,b)=>Math.hypot(a[0]-foot[0],a[1]-foot[1])-Math.hypot(b[0]-foot[0],b[1]-foot[1])).slice(0,24);
    if(local.length<3)return null;let best=null;
    for(const point of local){const dx=point[0]-foot[0],dy=point[1]-foot[1],length=Math.hypot(dx,dy);if(length<.05)continue;const unit=[dx/length,dy/length],inliers=local.filter(p=>Math.abs((p[0]-foot[0])*unit[1]-(p[1]-foot[1])*unit[0])<.022);
      if(inliers.length<3)continue;const projections=inliers.map(p=>(p[0]-foot[0])*unit[0]+(p[1]-foot[1])*unit[1]),span=Math.max(...projections)-Math.min(...projections);if(span<.1)continue;
      let supported=0,samples=0;for(let step=-6;step<=6;step++){const x=(foot[0]+unit[0]*step*.02)/aspect,y=foot[1]+unit[1]*step*.02,ix=Math.floor(x*mask[0].length),iy=Math.floor(y*mask.length);if(ix>=0&&ix<mask[0].length&&iy>=0&&iy<mask.length){samples++;if(mask[iy][ix]>=.4)supported++;}}if(samples<5||supported/samples<.6)continue;
      const score=inliers.length*Math.min(.4,span);if(!best||score>best.score){const vector=[unit[0]/aspect,unit[1]],norm=Math.hypot(...vector);best={score,vector:vector.map(v=>v/norm),confidence:Math.min(.99,.7+.04*inliers.length),points:inliers.length,image_aspect:aspect,method:'road_supported_vehicle_corridor'};}
    }return best;
  }
  function attach(c,result){
    const mask=result.road_mask,key=c.id+':'+c.hash,signature=JSON.stringify([result.version,result.backend,result.width,result.height,(c.detections||[]).map(d=>[d.label,d.box])]);
    let cached=projectionCache.get(key);
    if(cached?.signature!==signature){const points=result.skeleton||skeleton(mask),projected=(c.detections||[]).map(d=>{const score=footprint(mask,d.box),on=score==null?null:score>=.6?true:score<=.2?false:null;return {on_road:on,road_score:score,road_axis:null};}),anchors=(c.detections||[]).filter((d,i)=>projected[i].on_road&&['car','truck','bus','motorcycle','bicycle'].includes(d.label)).map(d=>[(d.box[0]+d.box[2])/2,d.box[3]]);projected.forEach((p,i)=>{if(p.on_road)p.road_axis=corridor(anchors,c.detections[i].box,mask,result.width||640,result.height||400)||axis(points,c.detections[i].box,result.width||640,result.height||400);});cached={signature,projected};projectionCache.set(key,cached);if(projectionCache.size>128)projectionCache.delete(projectionCache.keys().next().value);}
    const ds=(c.detections||[]).map((d,i)=>({...d,...cached.projected[i]}));
    return {...c,detections:ds,analysis:{...c.analysis,road_model:result.model||VERSION,road_mask:mask,on_road:ds.filter(d=>d.on_road&&['car','truck','bus','motorcycle','bicycle'].includes(d.label)).length,road_backend:result.backend,road_inference_ms:result.inference_ms}};
  }
  function direction(d,orientation){
    if(d.on_road!==true||!d.road_axis||orientation?.state!=='estimated'||!orientation.image_direction)return null;
    const directions={down:[0,1],'down-right':[1,1],right:[1,0],'up-right':[1,-1],up:[0,-1],'up-left':[-1,-1],left:[-1,0],'down-left':[-1,1]},facing=directions[orientation.image_direction],axis=d.road_axis.vector,aspect=d.road_axis.image_aspect||1,dot=axis[0]*aspect*facing[0]+axis[1]*facing[1],alignment=Math.abs(dot)/(Math.hypot(axis[0]*aspect,axis[1])*Math.hypot(...facing));
    if(alignment<.55)return {state:'unknown',reason:'facing_road_axis_disagree',alignment};
    const vector=axis.map(v=>v*(dot<0?-1:1)),angle=(Math.atan2(vector[1],vector[0]*aspect)*180/Math.PI+360)%360,name=['right','down-right','down','down-left','left','up-left','up','up-right'][Math.floor((angle+22.5)/45)%8];
    return {state:'estimated',vector,direction:name,alignment,road_score:d.road_score,method:(d.road_axis.method||'road_mask_medial_axis')+'_and_crop_facing',measured_motion:false};
  }
  async function prepare(source){const bitmap=await createImageBitmap(source.blob);try{const canvas=document.createElement('canvas');canvas.width=canvas.height=256;const ctx=canvas.getContext('2d',{willReadFrequently:true});ctx.drawImage(bitmap,0,0,256,256);const pixels=ctx.getImageData(0,0,256,256).data,input=new Float32Array(3*65536),mean=[.485,.456,.406],std=[.229,.224,.225];for(let c=0;c<3;c++)for(let i=0;i<65536;i++)input[c*65536+i]=(pixels[i*4+c]/255-mean[c])/std[c];return {input,width:bitmap.width,height:bitmap.height};}finally{bitmap.close();}}
  root.RoadVision={VERSION,scene,skeleton,footprint,axis,corridor,attach,direction,prepare};
})(globalThis);
