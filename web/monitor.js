/* Local monitoring history and explicit work budgets. Photos never leave this browser. */
(function(root){
  const MB=1024*1024,DAY=86400000;
  const profiles=[
    {level:1,name:'Eco',refreshMs:300000,downloads:1,inferenceDelay:1500,paintMs:400},
    {level:2,name:'Low',refreshMs:180000,downloads:2,inferenceDelay:900,paintMs:250},
    {level:3,name:'Balanced',refreshMs:90000,downloads:3,inferenceDelay:350,paintMs:120},
    {level:4,name:'Fast',refreshMs:60000,downloads:4,inferenceDelay:100,paintMs:80},
    {level:5,name:'Maximum',refreshMs:30000,downloads:6,inferenceDelay:0,paintMs:60}
  ];
  const profile=level=>profiles[Math.max(0,Math.min(4,Math.round(Number(level)||3)-1))];
  const bytes=value=>value<MB?(value/1024).toFixed(0)+' KB':value<1024*MB?(value/MB).toFixed(1)+' MB':(value/1024/MB).toFixed(2)+' GB';
  const request=r=>new Promise((resolve,reject)=>{r.onsuccess=()=>resolve(r.result);r.onerror=()=>reject(r.error);});
  const complete=tx=>new Promise((resolve,reject)=>{tx.oncomplete=resolve;tx.onabort=tx.onerror=()=>reject(tx.error||Error('Photo storage transaction failed'));});
  function latestAt(rows,time){const found=new Map();for(const row of rows){if(row.observedAt<=time&&(!found.has(row.cameraId)||found.get(row.cameraId).observedAt<row.observedAt))found.set(row.cameraId,row);}return [...found.values()];}
  function observationTimes(rows,cameraId){return [...new Set(rows.filter(r=>!cameraId||r.cameraId===cameraId).map(r=>r.observedAt))].sort((a,b)=>a-b);}
  function patternsAt(rows,time){
    const selected=rows.filter(r=>r.observedAt>=time-DAY&&r.observedAt<=time&&r.detections);
    const hours=new Map(),cameras={};
    for(const row of selected){
      const h=new Date(row.observedAt).toISOString().slice(0,13)+':00:00Z';
      if(!hours.has(h))hours.set(h,new Map());const cameraHours=hours.get(h);if(!cameraHours.has(row.cameraId))cameraHours.set(row.cameraId,[]);
      const ds=row.detections.filter(d=>['car','truck','bus','motorcycle','bicycle'].includes(d.label));cameraHours.get(row.cameraId).push(ds.length);
      if(!cameras[row.cameraId])cameras[row.cameraId]={samples:0,occupancy_grid:Array.from({length:24},()=>Array(24).fill(0)),first:row.observedAt/1000,last:row.observedAt/1000};
      const c=cameras[row.cameraId];c.samples++;c.first=Math.min(c.first,row.observedAt/1000);c.last=Math.max(c.last,row.observedAt/1000);
      for(const d of ds){const x=Math.max(0,Math.min(23,Math.floor((d.box[0]+d.box[2])*12))),y=Math.max(0,Math.min(23,Math.floor(d.box[3]*24)));c.occupancy_grid[y][x]++;}
    }
    for(const c of Object.values(cameras)){let count=0;for(const row of c.occupancy_grid)for(let x=0;x<row.length;x++){count+=row[x];row[x]/=c.samples;}c.mean_vehicles=count/c.samples;}
    const hourly=[...hours].sort(([a],[b])=>a.localeCompare(b)).map(([hour,perCamera])=>({hour,cameras:perCamera.size,samples:[...perCamera.values()].reduce((n,v)=>n+v.length,0),mean_vehicles:[...perCamera.values()].reduce((n,v)=>n+v.reduce((a,b)=>a+b,0)/v.length,0)/perCamera.size}));
    return {generated:new Date(time).toISOString(),cameras,hourly,window_hours:24};
  }
  class PhotoArchive{
    constructor({indexedDB=root.indexedDB,name='locaintel-monitor-v1',budgetBytes=500*MB,retentionMs=7*DAY,onChange=()=>{}}={}){
      Object.assign(this,{indexedDB,name,budgetBytes,retentionMs,onChange});this.chain=Promise.resolve();this.dbPromise=null;this.cachedBytes=null;this.lastPrune=0;
    }
    open(){
      if(this.dbPromise)return this.dbPromise;
      this.dbPromise=new Promise((resolve,reject)=>{
        if(!this.indexedDB){reject(Error('Browser photo storage is unavailable'));return;}
        const r=this.indexedDB.open(this.name,1);
        r.onupgradeneeded=()=>{
          const db=r.result,frames=db.createObjectStore('frames',{keyPath:'key'});frames.createIndex('observedAt','observedAt');frames.createIndex('cameraId','cameraId');frames.createIndex('hash','hash');
          db.createObjectStore('photos',{keyPath:'hash'});db.createObjectStore('settings');
        };
        r.onerror=()=>reject(r.error);r.onblocked=()=>reject(Error('Close another app tab to update local photo storage'));
        r.onsuccess=()=>{const db=r.result;db.onversionchange=()=>db.close();resolve(db);};
      });return this.dbPromise;
    }
    enqueue(fn){const result=this.chain.then(fn);this.chain=result.catch(()=>{});return result;}
    async initialize(){const db=await this.open(),tx=db.transaction('settings','readonly');const prefs=await request(tx.objectStore('settings').get('preferences'));if(prefs){this.budgetBytes=prefs.budgetBytes;this.retentionMs=prefs.retentionMs;}return this.prune();}
    saveFrame(camera,source,{observedAt,analysisMode='none',detectorVersion,detections,counts,analysis}={}){
      return this.enqueue(async()=>{
        if(!source?.blob||!source.hash)return null;
        const db=await this.open(),tx=db.transaction(['frames','photos'],'readwrite'),done=complete(tx),frames=tx.objectStore('frames'),photos=tx.objectStore('photos');
        const key=String(camera.id)+':'+source.hash,existing=await request(frames.get(key));
        const now=Date.now(),time=Number.isFinite(observedAt)?observedAt:Date.parse(source.checked)||now;
        const incomingRank={none:0,browser:1,published:2}[analysisMode]||0,existingRank={none:0,browser:1,published:2}[existing?.analysisMode]||0;
        const row={...existing,key,cameraId:String(camera.id),hash:source.hash,observedAt:existing?Math.min(existing.observedAt,time):time,lastSeen:now,width:source.width||existing?.width||null,height:source.height||existing?.height||null,sourceModified:source.modified||existing?.sourceModified||null,
          camera:{id:String(camera.id),name:camera.name,caption:camera.caption,region:camera.region,highway:camera.highway,lat:camera.lat,lon:camera.lon,source:camera.source,image:camera.image}};
        if(!existing||incomingRank>=existingRank){row.analysisMode=analysisMode;if(detectorVersion)row.detectorVersion=detectorVersion;if(detections!==undefined)row.detections=detections;if(counts!==undefined)row.counts=counts;if(analysis!==undefined)row.analysis=analysis;}
        const newPhoto=!await request(photos.get(source.hash));
        if(newPhoto)photos.put({hash:source.hash,blob:source.blob,size:source.blob.size});
        frames.put(row);await done;
        if(this.cachedBytes!=null&&newPhoto)this.cachedBytes+=source.blob.size;
        const pruning=this.cachedBytes==null||this.cachedBytes>this.budgetBytes||now-this.lastPrune>60000||time<now-this.retentionMs?await this.pruneNow():null;
        this.onChange({row,removedKeys:pruning?.removedKeys||[]});return row;
      });
    }
    saveAnalysis(cameraId,hash,result){return this.enqueue(async()=>{
      const db=await this.open(),tx=db.transaction('frames','readwrite'),done=complete(tx),store=tx.objectStore('frames'),rows=await request(store.index('hash').getAll(hash)),updated=[];
      for(const row of rows){
        const next=row.analysisMode==='published'?{...row,browserResult:{version:result.version||null,detections:result.detections,counts:result.counts,observed:result.observed||new Date().toISOString()}}:{...row,detections:result.detections,counts:result.counts,analysisMode:'browser',detectorVersion:result.version||null,analysis:{motion:{status:'insufficient_cadence',gap_seconds:null},road_model:null}};
        store.put(next);updated.push(next);
      }
      await done;for(const row of updated)this.onChange({row,removedKeys:[]});
    });}

    async metadata(){await this.chain;const db=await this.open(),tx=db.transaction('frames','readonly');return request(tx.objectStore('frames').getAll());}
    async photo(hash){const db=await this.open(),tx=db.transaction('photos','readonly');return (await request(tx.objectStore('photos').get(hash)))?.blob||null;}
    prune(){return this.enqueue(()=>this.pruneNow());}
    async pruneNow(){
      const db=await this.open(),tx=db.transaction(['frames','photos'],'readwrite'),done=complete(tx),frames=tx.objectStore('frames'),photos=tx.objectStore('photos');
      const rows=(await request(frames.getAll())).sort((a,b)=>a.observedAt-b.observedAt),allPhotos=await request(photos.getAll());
      const sizes=new Map(allPhotos.map(p=>[p.hash,p.size])),refs=new Map();for(const r of rows)refs.set(r.hash,(refs.get(r.hash)||0)+1);
      const removedKeys=[];
      let used=allPhotos.reduce((n,p)=>n+p.size,0),removed=0;const cutoff=Date.now()-this.retentionMs;
      for(const row of rows){
        if(row.observedAt>=cutoff&&used<=this.budgetBytes)break;
        frames.delete(row.key);removedKeys.push(row.key);removed++;refs.set(row.hash,refs.get(row.hash)-1);
        if(!refs.get(row.hash)){photos.delete(row.hash);used-=sizes.get(row.hash)||0;}
      }
      // A photo whose sole frame was removed by a previous interrupted write is not retained.
      for(const p of allPhotos)if(!refs.has(p.hash)){photos.delete(p.hash);used-=p.size;}
      await done;this.cachedBytes=Math.max(0,used);this.lastPrune=Date.now();return {removedKeys,usedBytes:Math.max(0,used),frames:rows.length-removed,photos:[...refs.values()].filter(n=>n>0).length,budgetBytes:this.budgetBytes,retentionMs:this.retentionMs};
    }
    async stats(){await this.chain;if(this.cachedBytes==null)await this.prune();const db=await this.open(),tx=db.transaction(['frames','photos'],'readonly'),frames=tx.objectStore('frames');
      const [count,photos,oldest,newest]=await Promise.all([request(frames.count()),request(tx.objectStore('photos').count()),request(frames.index('observedAt').openCursor()),request(frames.index('observedAt').openCursor(null,'prev'))]);
      return {usedBytes:this.cachedBytes,frames:count,photos,oldest:oldest?.value.observedAt||null,newest:newest?.value.observedAt||null,budgetBytes:this.budgetBytes,retentionMs:this.retentionMs};}

    configure({budgetBytes=this.budgetBytes,retentionMs=this.retentionMs}={}){return this.enqueue(async()=>{
      this.budgetBytes=Math.max(1,budgetBytes);this.retentionMs=Math.max(1,retentionMs);
      const db=await this.open(),tx=db.transaction('settings','readwrite'),done=complete(tx);tx.objectStore('settings').put({budgetBytes:this.budgetBytes,retentionMs:this.retentionMs},'preferences');await done;
      const stats=await this.pruneNow();this.onChange({reload:true,removedKeys:stats.removedKeys});return stats;
    });}
    clear(){return this.enqueue(async()=>{const db=await this.open(),tx=db.transaction(['frames','photos'],'readwrite'),done=complete(tx);tx.objectStore('frames').clear();tx.objectStore('photos').clear();await done;this.cachedBytes=0;this.onChange({clear:true});});}
  }
  root.Monitoring={MB,DAY,profiles,profile,bytes,latestAt,observationTimes,patternsAt,PhotoArchive};
})(globalThis);
