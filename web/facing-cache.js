/* Retry missed facing jobs only from IndexedDB photos. Never fetch image URLs. */
(function(root){
  const fingerprint=d=>JSON.stringify([d.label,d.box]);
  const valid=d=>d?.label==='car'&&Array.isArray(d.box)&&d.box.length===4&&d.box.every(v=>Number.isFinite(v)&&v>=0&&v<=1)&&d.box[2]>d.box[0]&&d.box[3]>d.box[1];
  class LocalFacingQueue{
    constructor({rows,photo,analyzer,isPaused=()=>false,onError=()=>{},delayMs=500}={}){Object.assign(this,{rows,photo,analyzer,isPaused,onError,delayMs});this.active=null;this.loading=false;this.timer=null;this.missing=new Set();this.disposed=false;}
    wants(hash){return this.active?.hash===hash;}
    wake(delay=0){if(this.disposed||this.timer||this.loading||this.isPaused())return;this.timer=setTimeout(()=>{this.timer=null;this.tick();},delay);this.timer.unref?.();}
    pause(){clearTimeout(this.timer);this.timer=null;}
    dispose(){this.disposed=true;this.pause();}
    async tick(){
      if(this.disposed||this.loading||this.isPaused())return;
      if(this.analyzer.running||this.analyzer.queue.length){this.wake(this.delayMs);return;}
      this.active=null;let nextRetry=Infinity;
      for(const row of this.rows().slice().sort((a,b)=>b.observedAt-a.observedAt)){
        if(this.missing.has(row.key))continue;
        const source={cameraId:String(row.cameraId),hash:row.hash},ds=row.browserResult?.detections||row.detections||[],work=[];
        for(const [index,d]of ds.entries()){
          if(!valid(d))continue;
          const fp=fingerprint(d),saved=row.facing?.byFingerprint?.[fp]||Object.values(row.facing?.byDetection||{}).find(r=>r?.fingerprint===fp),cached=this.analyzer.get(source,d),record=cached||(saved?.version===this.analyzer.version?saved:null);
          if(record&&record.version===this.analyzer.version){
            if(record.model_ran===true||['queued','processing'].includes(record.state)||record.retry_exhausted)continue;
            if(record.state==='error'&&(record.next_retry_at||0)>Date.now()){nextRetry=Math.min(nextRetry,record.next_retry_at);continue;}
          }
          work.push({d,index,record});
        }
        if(!work.length)continue;
        this.loading=true;this.active=source;
        try{
          const blob=await this.photo(row.hash);
          if(!blob){this.missing.add(row.key);while(this.missing.size>256)this.missing.delete(this.missing.values().next().value);continue;}
          if(this.disposed||this.isPaused())return;
          for(const {d,index,record}of work){if(record?.state==='error'&&!this.analyzer.get(source,d))this.analyzer.entries.set(this.analyzer.key(source,d),record);this.analyzer.request({...source,blob,width:row.width,height:row.height},d,index,-10);}
          this.analyzer.wake();return;
        }catch(error){this.onError(error);this.missing.add(row.key);}
        finally{this.loading=false;this.wake(this.delayMs);}
      }
      if(Number.isFinite(nextRetry))this.wake(Math.max(this.delayMs,nextRetry-Date.now()));
    }
  }
  root.LocalFacingQueue=LocalFacingQueue;
})(globalThis);
