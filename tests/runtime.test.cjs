const test=require('node:test'),assert=require('node:assert/strict'),fs=require('fs'),vm=require('vm');
function runtime({gpu=false,half=true,createFails=false,runFails=false,halfFails=false,preferredMissing=false,hangGPU=false}={}){
 const created=[],imports=[],requests=[];let loseGPU;const device={lost:new Promise(resolve=>loseGPU=resolve)};
 const ort={env:{wasm:{},webgpu:{device}},InferenceSession:{create:async(bytes,options)=>{
  const backend=options.executionProviders[0];created.push({backend,bytes:[...bytes],providers:options.executionProviders,externalData:options.externalData});
  if(backend==='webgpu'&&(createFails||(halfFails&&bytes[0]===2)))throw Error('unsupported GPU graph');
  return {inputNames:['image'],outputNames:['out'],release:async()=>{},run:async feeds=>{if(backend==='webgpu'&&hangGPU)return new Promise(()=>{});if(backend==='webgpu'&&runFails)throw Error('GPU device lost');return {out:backend,size:feeds?.image?.dims?.[0]};}};
 }}};
 const ctx=vm.createContext({URL,Response,Uint8Array,ort,location:{href:'https://example.test/app/worker.js'},importScripts:p=>imports.push(p),navigator:gpu?{gpu:{requestAdapter:async options=>{requests.push(options);return preferredMissing&&options?null:{features:new Set(half?['shader-f16']:[])};}}}:{},fetch:async url=>new Response(new Uint8Array([url.includes('half')?2:1]))});
 vm.runInContext(fs.readFileSync('web/inference-runtime.js','utf8'),ctx);return {m:ctx.ModelRuntime,created,imports,requests,loseGPU};
}
const cpu={file:'full.onnx'},half={file:'half.onnx',requiresF16:true};
test('GPU workers request high performance and explicitly select WebGPU',async()=>{const {m,created,imports,requests}=runtime({gpu:true}),s=await new m.Session(cpu,half).initialize();assert.equal(s.backend,'webgpu');assert.equal(requests[0].powerPreference,'high-performance');assert.deepEqual(created.map(c=>c.backend),['webgpu']);assert.equal(created[0].providers.length,1);assert(imports[0].includes('ort.webgpu'));});
test('missing FP16 support still executes the compatible graph on GPU',async()=>{const {m,created}=runtime({gpu:true,half:false}),s=await new m.Session(cpu,half).initialize();assert.equal(s.backend,'webgpu');assert.equal(s.variant,'full.onnx');assert.deepEqual(created.map(c=>c.bytes[0]),[1]);});
test('unsupported FP16 graph tries full precision on GPU before CPU',async()=>{const {m,created}=runtime({gpu:true,halfFails:true}),s=await new m.Session(cpu,half).initialize();assert.equal(s.backend,'webgpu');assert.equal(s.variant,'full.onnx');assert.deepEqual(created.map(c=>c.bytes[0]),[2,1]);assert(created.every(c=>c.backend==='webgpu'));});
test('missing preferred adapter retries the browser default adapter',async()=>{const {m,requests}=runtime({gpu:true,preferredMissing:true}),s=await new m.Session(cpu).initialize();assert.equal(s.backend,'webgpu');assert.equal(requests.length,2);assert.equal(requests[1],undefined);});
test('no adapter uses CPU with an explicit reason',async()=>{const {m,created}=runtime(),s=await new m.Session(cpu,half).initialize();assert.equal(s.backend,'wasm');assert.match(s.fallback,/adapter unavailable/);assert.deepEqual(created.map(c=>c.backend),['wasm']);});
test('GPU graph failures and device loss fall back to a working CPU session',async()=>{for(const options of [{gpu:true,createFails:true},{gpu:true,runFails:true}]){const {m,created}=runtime(options),s=await new m.Session(cpu).initialize(),result=await s.run({});assert.equal(result.out,'wasm');assert.equal(s.backend,'wasm');assert(s.fallback);assert.deepEqual(created.map(c=>c.backend),['webgpu','wasm']);}});

test('capture incompatibility retries ordinary GPU execution before CPU',async()=>{const {m,created}=runtime({gpu:true}),s=await new m.Session(cpu,{...half,captureShapes:{out:[1]}}).initialize();const result=await s.run({image:{type:'float32',dims:[1],data:new Float32Array([1])}});assert.equal(result.out,'webgpu');assert.equal(s.capture,false);assert.deepEqual(created.map(c=>c.backend),['webgpu','webgpu']);});
test('GPU device loss maps batched inputs to the compatible CPU shape',async()=>{const {m}=runtime({gpu:true,runFails:true}),s=await new m.Session(cpu,half).initialize();const result=await s.run({image:{dims:[2]}},{cpuFeeds:()=>({image:{dims:[1]}})});assert.equal(result.size,1);assert.equal(s.backend,'wasm');});

test('device loss interrupts a stuck GPU operation and recovers on CPU',async()=>{const {m,loseGPU}=runtime({gpu:true,hangGPU:true}),s=await new m.Session(cpu).initialize();const pending=s.run({image:{dims:[2]}},{cpuFeeds:()=>({image:{dims:[1]}})});loseGPU({message:'Device was destroyed.'});let timer;try{const result=await Promise.race([pending,new Promise((_,reject)=>timer=setTimeout(()=>reject(Error('GPU loss hung')),100))]);assert.equal(result.out,'wasm');assert.equal(result.size,1);assert.match(s.fallback,/Device was destroyed/);}finally{clearTimeout(timer);}});

test('non-FP16 adapters choose the FP32 GPU graph and supply cached external weights',async()=>{const {m,created}=runtime({gpu:true,half:false}),spec={file:'fp32.onnx',externalData:[{file:'weights.bin',path:'weights.bin'}]},s=await new m.Session(cpu,[half,spec]).initialize();assert.equal(s.backend,'webgpu');assert.equal(s.variant,'fp32.onnx');assert.equal(created[0].externalData[0].path,'weights.bin');assert.deepEqual([...created[0].externalData[0].data],[1]);});

function webglRuntime({gpu=false,webgl2=true,createFails=false,runFails=false}={}){
 const created=[],runs=[];let ctx;
 class Tensor{constructor(type,data,dims){Object.assign(this,{type,data,dims});}}
 const session=(backend,fail)=>({inputNames:['image'],outputNames:['logits'],release:async()=>{},run:async feeds=>{runs.push({backend,dims:feeds.image.dims,data:Array.from(feeds.image.data)});if(fail)throw Error('WebGL context lost');const x=feeds.image.data;return {logits:new Tensor('float32',Float32Array.from([x[0]*10,x[0]*10+1]),[feeds.image.dims[0]===1?1:feeds.image.dims[0],2])};}});
 const webglLib={env:{webgl:{}},Tensor,InferenceSession:{create:async(bytes,options)=>{created.push(options.executionProviders[0]);if(createFails)throw Error('cannot resolve operator');return session('webgl',runFails);}}};
 const main={env:{wasm:{},webgpu:{}},Tensor,InferenceSession:{create:async(bytes,options)=>{created.push(options.executionProviders[0]);return session(options.executionProviders[0],false);}}};
 ctx=vm.createContext({URL,Response,Uint8Array,Float32Array,ort:main,location:{href:'https://example.test/app/worker.js'},
  importScripts:p=>{if(p.includes('ort.webgl'))ctx.ort=webglLib;},
  OffscreenCanvas:class{getContext(kind){return kind==='webgl2'&&webgl2?{}:null;}},
  navigator:gpu?{gpu:{requestAdapter:async()=>({features:new Set()})}}:{},fetch:async()=>new Response(new Uint8Array([1]))});
 vm.runInContext(fs.readFileSync('web/inference-runtime.js','utf8'),ctx);return {m:ctx.ModelRuntime,ctx,main,webglLib,created,runs};
}
const glSpec={file:'model-webgl.onnx',pack:true};
test('without WebGPU, WebGL runs before CPU and leaves the CPU runtime global intact',async()=>{const {m,ctx,main,webglLib,created}=webglRuntime(),s=await new m.Session(cpu,null,glSpec).initialize();assert.equal(s.backend,'webgl');assert.deepEqual(created,['webgl']);assert.equal(ctx.ort,main);assert.equal(webglLib.env.webgl.pack,true);assert.match(s.fallback,/WebGPU adapter unavailable/);const out=await s.run({image:new main.Tensor('float32',Float32Array.from([2]),[1,1])});assert(out.logits instanceof main.Tensor);assert.deepEqual([...out.logits.data],[20,21]);});
test('WebGL graph or context failures fall back to the CPU model',async()=>{for(const options of [{createFails:true},{runFails:true}]){const {m,main,created}=webglRuntime(options),s=await new m.Session(cpu,null,glSpec).initialize(),out=await s.run({image:new main.Tensor('float32',Float32Array.from([1]),[1,1])});assert.equal(s.backend,'wasm');assert.match(s.fallback,/WebGL/);assert.deepEqual(created,['webgl','wasm']);assert.deepEqual([...out.logits.data],[10,11]);}});
test('browsers without WebGL2 in workers go straight to CPU',async()=>{const {m,created}=webglRuntime({webgl2:false}),s=await new m.Session(cpu,null,glSpec).initialize();assert.equal(s.backend,'wasm');assert.match(s.fallback,/WebGL2 unavailable/);assert.deepEqual(created,['wasm']);});
test('WebGPU stays preferred over WebGL when available',async()=>{const {m,created}=webglRuntime({gpu:true}),s=await new m.Session(cpu,null,glSpec).initialize();assert.equal(s.backend,'webgpu');assert.deepEqual(created,['webgpu']);});
