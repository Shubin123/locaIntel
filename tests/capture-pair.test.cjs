const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm');
const context=vm.createContext({});vm.runInContext(fs.readFileSync('web/capture-pair.js','utf8'),context);const PairCapture=context.PairCapture;
function clock(){
  let time=100000,serial=0;const timers=new Map();
  return {now:()=>time,setTimer:(fn,delay)=>{timers.set(++serial,{fn,at:time+delay});return serial;},clearTimer:id=>timers.delete(id),size:()=>timers.size,
    async advance(ms){const end=time+ms;while(true){const next=[...timers].filter(([,v])=>v.at<=end).sort((a,b)=>a[1].at-b[1].at)[0];if(!next)break;time=next[1].at;timers.delete(next[0]);await next[1].fn();}time=end;}};
}
test('one distinct follow-up completes and duplicate callbacks cannot create a capture loop',async()=>{
  const time=clock(),calls=[],pair=new PairCapture({...time,capture:async id=>{calls.push(id);return {hash:'second'};}});
  assert.equal(pair.request(238,'first',time.now()),true);assert.equal(pair.request('238','first',time.now()),false);
  await time.advance(4999);assert.equal(calls.length,0);await time.advance(1);assert.deepEqual(calls,['238']);
  assert.equal(pair.request(238,'second',time.now()),false);await time.advance(30000);
  assert.equal(pair.request(238,'first',time.now()),false);assert.equal(pair.request(238,'second',time.now()),false);assert.equal(time.size(),0);
});
test('unchanged upstream photos get at most two attempts inside the freshness window',async()=>{
  const time=clock();let calls=0;const pair=new PairCapture({...time,capture:async()=>{calls++;return 'same';}});
  pair.request('1','same',time.now());await time.advance(60000);assert.equal(calls,2);assert.equal(time.size(),0);assert.equal(pair.request('1','same',time.now()),false);
  assert.equal(pair.request('2','stale',time.now()-11000),false);assert.equal(pair.request('2','future',time.now()+1),false);
});
test('pause cancels pending work, wake resumes fresh requests, and expired requests stop',async()=>{
  const time=clock();let calls=0,hidden=false;const pair=new PairCapture({...time,isPaused:()=>hidden,capture:async()=>{calls++;return 'other';}});
  pair.request('1','first',time.now());pair.pause();await time.advance(1000);assert.equal(calls,0);assert.equal(time.size(),0);
  pair.wake();await time.advance(5000);assert.equal(calls,1);
  pair.request('2','first',time.now());hidden=true;await time.advance(5000);assert.equal(calls,1);hidden=false;await time.advance(11000);pair.wake();await time.advance(5000);assert.equal(calls,1);
});
test('per-camera requests never overlap and cancelled in-flight completions cannot retry',async()=>{
  const time=clock();let release,calls=0;const pair=new PairCapture({...time,capture:()=>{calls++;return new Promise(resolve=>{release=resolve;});}});
  pair.request('1','first',time.now());const advancing=time.advance(5000);await Promise.resolve();assert.equal(calls,1);
  assert.equal(pair.request('1','new-first',time.now()),false);pair.cancel('1');release('first');await advancing;await time.advance(20000);assert.equal(calls,1);assert.equal(time.size(),0);
});
test('errors are reported with bounded retries and disposal cancels scheduled timers',async()=>{
  const time=clock(),errors=[];let calls=0;const pair=new PairCapture({...time,capture:async()=>{calls++;throw Error('offline');},onError:(e,id)=>errors.push([e.message,id])});
  pair.request(1,'first',time.now());await time.advance(10000);assert.equal(calls,2);assert.deepEqual(errors,[['offline','1'],['offline','1']]);
  pair.request(2,'first',time.now());pair.dispose();assert.equal(time.size(),0);await time.advance(10000);assert.equal(calls,2);assert.equal(pair.request(3,'first',time.now()),false);
});
test('tracked camera state is bounded without starving already scheduled cameras',async()=>{
  const time=clock(),calls=[],pair=new PairCapture({...time,maxCameras:2,capture:async id=>{calls.push(id);return 'other';}});
  pair.request('1','first',time.now());pair.request('2','first',time.now());assert.equal(pair.request('3','first',time.now()),false);assert.equal(pair.records.size,2);assert.equal(time.size(),2);
  await time.advance(5000);assert.deepEqual(calls,['1','2']);assert.equal(pair.request('3','first',time.now()),true);assert.equal(pair.records.size,2);
  pair.dispose();assert.equal(time.size(),0);
});
