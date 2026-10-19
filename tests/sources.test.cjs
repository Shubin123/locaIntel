const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const { webcrypto } = require('node:crypto');
const context = vm.createContext({ URL, Uint8Array, AbortController, setTimeout, clearTimeout, Date, crypto: webcrypto });
vm.runInContext(fs.readFileSync('web/sources.js', 'utf8'), context);
const Cache = context.CameraSourceCache;
const camera = {id:'864',image:'https://www.drivebc.ca/images/864.jpg'};
function options(fetcher) { let i=0; return { fetcher, makeURL:()=>`blob:test-${++i}`,revokeURL:()=>{},validate:async()=>{} }; }
function image(content='frame') { return new Response(new Blob([content],{type:'image/jpeg'}),{headers:{'last-modified':'Mon, 19 Oct 2026 12:00:00 GMT'}}); }

test('manual source refresh bypasses caches and reuses URL only for identical bytes', async () => {
  const calls=[];let content='first';
  const cache=new Cache(options(async(url,opts)=>{calls.push({url,opts});return image(content);}));
  const first=await cache.load(camera,true);assert.ok(first.hash);assert.ok(first.url);
  assert.equal(new URL(calls[0].url).hostname,'www.drivebc.ca');
  assert.ok(new URL(calls[0].url).searchParams.has('_locaintel'));
  assert.equal(calls[0].opts.cache,'no-store');assert.equal(calls[0].opts.credentials,'omit');
  await cache.load(camera);assert.equal(calls.length,1);
  const same=await cache.load(camera,true);assert.equal(calls.length,2);assert.equal(first.url,same.url);
  content='newer';const changed=await cache.load(camera,true);assert.notEqual(changed.hash,first.hash);assert.notEqual(changed.url,first.url);
});
test('failed source fetch keeps last successful image without claiming it was refreshed', async () => {
  let fail=false;const cache=new Cache(options(async()=>{if(fail)throw Error('offline');return image();}));
  const good=await cache.load(camera,true);fail=true;const failed=await cache.load(camera,true);
  assert.equal(failed.url,good.url);assert.equal(failed.hash,good.hash);assert.equal(failed.checked,good.checked);assert.equal(failed.error,'offline');
});
test('bad response and unsafe hosts never replace cached frames', async () => {
  let html=false;const cache=new Cache(options(async()=>html?new Response('not an image',{headers:{'content-type':'text/html'}}):image()));
  const good=await cache.load(camera,true);html=true;
  assert.equal((await cache.load(camera,true)).url,good.url);
  const blocked=await cache.load({id:'x',image:'https://example.com/image.jpg'},true);assert.match(blocked.error,/Unexpected/);assert.equal(blocked.url,undefined);
});
test('multiple views of a camera share one request and downloads stay bounded', async () => {
  let active=0,peak=0,calls=0;
  const cache=new Cache({...options(async()=>{calls++;active++;peak=Math.max(active,peak);await new Promise(r=>setTimeout(r,5));active--;return image();}),concurrency:3});
  const first=cache.load(camera,true),duplicate=cache.load(camera,true);
  await Promise.all([first,duplicate]);assert.equal(calls,1);
  await cache.refresh(Array.from({length:16},(_,i)=>({...camera,id:String(i)})),true);
  assert.ok(peak<=3);assert.equal(cache.active,0);assert.equal(cache.waiters.length,0);
});
test('a source timeout is surfaced and releases its concurrency slot', async () => {
  const cache=new Cache({...options((url,opts)=>new Promise((resolve,reject)=>opts.signal.addEventListener('abort',()=>reject(Object.assign(Error('timeout'),{name:'AbortError'}))))),timeout:10});
  const entry=await cache.load(camera,true);assert.match(entry.error,/timed out/);assert.equal(cache.active,0);
});
test('lowering the download budget applies to waiting requests without interrupting active transfers',async()=>{
 let calls=0,active=0;const later=[];
 const cache=new Cache({...options(async()=>{calls++;active++;if(calls>3)later.push(active);await new Promise(r=>setTimeout(r,8));active--;return image();}),concurrency:3});
 const job=cache.refresh(Array.from({length:10},(_,i)=>({...camera,id:String(i)})),true);await new Promise(r=>setTimeout(r,1));cache.setConcurrency(1);await job;
 assert(later.every(n=>n===1));assert.equal(cache.active,0);
});
