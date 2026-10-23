/* Run the browser's facing policy and route linker over a pipeline.route_survey file.
 * Usage: node scripts/route-survey.cjs .cache/route-survey/1.jsonl [report.json] */
const fs=require('node:fs'),vm=require('node:vm'),path=require('node:path');
const context=vm.createContext({});for(const f of ['orientation.js','routes.js'])vm.runInContext(fs.readFileSync(path.join(__dirname,'../web',f),'utf8'),context);
const {VehicleFacing,VehicleRoutes}=context,manifest=JSON.parse(fs.readFileSync(path.join(__dirname,'../web/models/vehicle-facing-v3.json')));
const [input,output]=process.argv.slice(2);
const rows=fs.readFileSync(input,'utf8').trim().split('\n').map(l=>JSON.parse(l)).map(r=>{
  const facing={},features={};
  r.detections.forEach((d,i)=>{
    if(!d.logits)return;facing[i]=VehicleFacing.estimate(d.logits,manifest,{type:'none'},{truncated:d.truncated});
    const rgba=new Uint8ClampedArray(d.patch.length*4);d.patch.forEach((p,j)=>{rgba.set(p,j*4);rgba[j*4+3]=255;});features[i]=VehicleRoutes.describePixels(rgba);
  });
  return {key:r.camera.id+':'+r.hash,hash:r.hash,cameraId:r.camera.id,observedAt:r.observedAt,sourceModified:r.sourceModified,camera:r.camera,detections:r.detections.map(({label,box})=>({label,box})),facing:{byDetection:facing},routeFeatures:{byDetection:features}};
});
const sightings=VehicleRoutes.sightings(rows),found=VehicleRoutes.routes(sightings),flows=VehicleRoutes.flows(sightings);
const report={frames:rows.length,cameras:new Set(rows.map(r=>r.cameraId)).size,sightings:sightings.length,with_heading:sightings.filter(s=>s.heading!=null).length,
  routes:found.length,by_confidence:{high:found.filter(r=>r.confidence>=.6).length,medium:found.filter(r=>r.confidence>=.35&&r.confidence<.6).length,low:found.filter(r=>r.confidence<.35).length},
  examples:found.slice(0,40).map(r=>({colour:r.colour,label:r.label,heading:r.heading,km:+(r.distance/1000).toFixed(1),kmh:r.speed?Math.round(r.speed*3.6):null,confidence:+r.confidence.toFixed(2),
    hops:r.hops.map(h=>({camera:h.camera.name,cameraId:h.cameraId,hash:h.hash,box:h.box,time:new Date(h.t).toISOString().slice(11,19),facing:h.facing}))})),
  flows:flows.filter(f=>f.dominant).map(f=>({camera:f.camera.name,headed:f.headed,dominant:f.dominant.direction,share:+f.dominant.share.toFixed(2)}))};
if(output)fs.writeFileSync(output,JSON.stringify(report,null,1));
console.log(JSON.stringify({...report,examples:report.examples.slice(0,8).map(r=>`${r.colour} ${r.label} → ${r.heading} · ${r.hops.length} cams · ${r.km} km · ${r.kmh} km/h · conf ${r.confidence} · ${r.hops.map(h=>h.camera+' '+h.time).join(' → ')}`),flows:report.flows.length},null,1));
// Decoy check: shift each camera's clock by a random ±10–20 min. True vehicles no
// longer line up, so links found here estimate how many real links are chance.
if(process.env.DECOYS){
  let seed=7;const rand=()=>(seed=(seed*16807)%2147483647)/2147483647,counts=[];
  for(let k=0;k<Number(process.env.DECOYS);k++){
    const shift=new Map([...new Set(rows.map(r=>r.cameraId))].map(id=>[id,(rand()<.5?-1:1)*(10+rand()*10)*60000]));
    const decoy=rows.map(r=>({...r,observedAt:r.observedAt+shift.get(r.cameraId),sourceModified:r.sourceModified?new Date(Date.parse(r.sourceModified)+shift.get(r.cameraId)).toUTCString():null}));
    counts.push(VehicleRoutes.routes(VehicleRoutes.sightings(decoy)).length);
  }
  console.log(JSON.stringify({real_routes:found.length,decoy_routes:counts,decoy_mean:counts.reduce((a,b)=>a+b,0)/counts.length}));
}
// Same-site handoffs between views of one interchange, against shifted-clock decoys.
const hand=VehicleRoutes.handoffs(sightings);
console.log(JSON.stringify({handoffs:hand.length,handoff_chance:+VehicleRoutes.handoffChance(sightings,{},20).toFixed(2),
  examples:hand.slice(0,6).map(h=>`${h.a.camera.name} → ${h.b.camera.name} · ${h.a.colourName} ${h.a.label} heading ${VehicleRoutes.compassName(h.a.heading)} · ${h.seconds}s · ${h.a.facing} → ${h.b.facing}`)},null,1));
