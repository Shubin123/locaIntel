const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm');
const context=vm.createContext({setTimeout,clearTimeout,Date});vm.runInContext(fs.readFileSync('web/facing-cache.js','utf8'),context);
const d={label:'car',box:[.2,.2,.4,.4]},fp=JSON.stringify([d.label,d.box]);
function setup(rows,{paused=false,records={}}={}){
 const fetched=[],requested=[],entries=new Map(),analyzer={version:'current',running:false,queue:[],entries,key:(s,d)=>s.cameraId+':'+s.hash+':'+JSON.stringify([d.label,d.box]),get(s,d){return entries.get(this.key(s,d));},request(s,d,index,priority){requested.push({s,d,index,priority,previous:this.get(s,d)});this.entries.set(this.key(s,d),{version:'current',state:'estimated',model_ran:true});},wake(){}};
 const q=new context.LocalFacingQueue({rows:()=>rows,photo:async hash=>{fetched.push(hash);return records[hash]===null?null:new Blob(['local']);},analyzer,isPaused:()=>paused,delayMs:10000});
 return {q,fetched,requested,analyzer};
}
const row=(hash='a',facing=null)=>({key:'archived:'+hash,cameraId:'archived',hash,observedAt:Date.now()-50000,detections:[d],...(facing?{facing:{byFingerprint:{[fp]:facing}}}:{})});
test('nonvisible archived photos are inferred using local bytes, exact detector fingerprints and low priority',async()=>{
 const {q,fetched,requested}=setup([row()]);try{await q.tick();assert.deepEqual(fetched,['a']);assert.equal(requested.length,1);assert.equal(requested[0].s.blob.size,5);assert.equal(requested[0].s.cameraId,'archived');assert.equal(requested[0].priority,-10);assert(q.wants('a'));}finally{q.dispose();}
});
test('completed quality abstentions are retained while stale model versions are recomputed',async()=>{
 const a=row('done',{version:'current',state:'unknown',model_ran:true,quality_abstention:true}),b=row('stale',{version:'old',state:'estimated',model_ran:true});const {q,fetched}=setup([a,b]);try{await q.tick();assert.deepEqual(fetched,['stale']);}finally{q.dispose();}
});
test('future retry and exhausted failures persist across a fresh queue; due failures retain their attempt count',async()=>{
 const future=row('future',{version:'current',state:'error',retry_attempts:1,next_retry_at:Date.now()+60000}),exhausted=row('exhausted',{version:'current',state:'error',retry_attempts:3,retry_exhausted:true}),due=row('due',{version:'current',state:'error',retry_attempts:1,next_retry_at:Date.now()-1000});
 const {q,fetched,analyzer,requested}=setup([future,exhausted,due]);try{await q.tick();assert.deepEqual(fetched,['due']);assert.equal(analyzer.entries.size,1);assert.equal(requested[0].previous.retry_attempts,1);}finally{q.dispose();}
});
test('paused work, active GPU work and absent local photo bytes never fetch a remote fallback',async()=>{
 const paused=setup([row()],{paused:true});await paused.q.tick();assert.deepEqual(paused.fetched,[]);paused.q.dispose();
 const busy=setup([row()]);busy.analyzer.running=true;await busy.q.tick();assert.deepEqual(busy.fetched,[]);busy.q.dispose();
 const absent=setup([row()],{records:{a:null}});try{await absent.q.tick();await absent.q.tick();assert.deepEqual(absent.fetched,['a']);assert.deepEqual(absent.requested,[]);}finally{absent.q.dispose();}
});
