/* Retry missed facing jobs only from IndexedDB photos. Never fetch image URLs. */
(function(root){
  const fingerprint=d=>JSON.stringify([d.label,d.box]);
  const valid=d=>['car','truck','bus'].includes(d?.label)&&Array.isArray(d.box)&&d.box.length===4&&d.box.every(v=>Number.isFinite(v)&&v>=0&&v<=1)&&d.box[2]>d.box[0]&&d.box[3]>d.box[1];
  class LocalFacingQueue{
    constructor({rows,photo,analyzer,isPaused=()=>false,onError=()=>{},delayMs=500}={}){Object.assign(this,{rows,photo,analyzer,isPaused,onError,delayMs});this.active=null;this.activeHashes=new Set();this.loading=false;this.timer=null;this.timerAt=Infinity;this.missing=new Set();this.disposed=false;}
    markMissing(key){this.missing.add(key);while(this.missing.size>256)this.missing.delete(this.missing.values().next().value);}
    wants(hash){return this.activeHashes.has(hash);}
    wake(delay=0){
      if(this.disposed||this.loading||this.isPaused())return;
      const at=Date.now()+Math.max(0,delay);if(this.timer&&this.timerAt<=at)return;
      clearTimeout(this.timer);this.timerAt=at;this.timer=setTimeout(()=>{this.timer=null;this.timerAt=Infinity;this.tick();},Math.max(0,delay));this.timer.unref?.();
    }
    pause(){clearTimeout(this.timer);this.timer=null;this.timerAt=Infinity;}
    dispose(){this.disposed=true;this.pause();this.activeHashes.clear();}
    async tick(){
      if(this.disposed||this.loading||this.isPaused())return;
      if(this.analyzer.running||this.analyzer.queue.length){this.wake(this.delayMs);return;}
      this.loading=true;this.active=null;this.activeHashes.clear();let nextRetry=Infinity;
      const jobs=[],capacity=Math.max(1,Math.min(8,Number(this.analyzer.batchSize)||1));
      try{
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
            work.push({d,index,record});if(work.length>=capacity-jobs.length)break;
          }
          if(!work.length)continue;
          try{
            const blob=await this.photo(row.hash);
            if(!blob){this.markMissing(row.key);continue;}
            if(this.disposed||this.isPaused())return;
            for(const item of work)jobs.push({...item,source:{...source,blob,width:row.width,height:row.height}});
          }catch(error){this.onError(error);this.markMissing(row.key);}
          if(jobs.length>=capacity)break;
        }
        // Load at most eight local photos before queueing the entire small
        // batch synchronously. No 1,000-photo blob queue or single-frame idle gap.
        if(jobs.length){
          this.active=jobs[0].source;for(const job of jobs)this.activeHashes.add(job.source.hash);
          for(const {source,d,index,record}of jobs){if(record?.state==='error'&&!this.analyzer.get(source,d))this.analyzer.entries.set(this.analyzer.key(source,d),record);this.analyzer.request(source,d,index,-10);}
          this.analyzer.wake();
        }
      }finally{
        this.loading=false;
        if(jobs.length)this.wake(this.delayMs);else if(Number.isFinite(nextRetry))this.wake(Math.max(this.delayMs,nextRetry-Date.now()));
      }
    }
  }
  root.LocalFacingQueue=LocalFacingQueue;
})(globalThis);
