/* Bounded follow-up fetches. Fetch times do not establish upstream capture times. */
(function(root){
  class PairCapture {
    constructor({capture,isPaused=()=>false,onError=()=>{},delayMs=5000,maxAttempts=2,cooldownMs=30000,maxAgeMs=15000,maxCameras=32,now=()=>Date.now(),setTimer=(fn,ms)=>setTimeout(fn,ms),clearTimer=id=>clearTimeout(id)}={}){
      if(typeof capture!=='function')throw Error('A camera capture callback is required');
      Object.assign(this,{capture,isPaused,onError,now,setTimer,clearTimer});
      this.delayMs=Math.max(1,delayMs);this.maxAttempts=Math.max(1,Math.floor(maxAttempts));
      this.cooldownMs=Math.max(0,cooldownMs);this.maxAgeMs=Math.max(1,maxAgeMs);
      this.maxCameras=Math.max(1,Math.floor(maxCameras));this.records=new Map();this.disposed=false;this.paused=false;
    }
    request(cameraId,firstHash,observedAt=this.now()){
      const id=String(cameraId),time=Number(observedAt),now=this.now(),previous=this.records.get(id);
      if(this.disposed||this.paused||this.isPaused()||!firstHash||!Number.isFinite(time)||time>now||now-time+this.delayMs>this.maxAgeMs)return false;
      if(previous&&(previous.seen.has(firstHash)||previous.capturing||['scheduled','paused'].includes(previous.state)||now<previous.nextEligibleAt))return false;
      if(!previous&&this.records.size>=this.maxCameras){
        const oldest=[...this.records].find(([,record])=>!record.capturing&&!['scheduled','paused'].includes(record.state));
        if(!oldest)return false;
        this.cancel(oldest[0]);this.records.delete(oldest[0]);
      }
      const seen=new Set(previous?.seen||[]);seen.add(firstHash);while(seen.size>8)seen.delete(seen.values().next().value);
      const record={firstHash,observedAt:time,attempts:0,state:'scheduled',timer:null,capturing:false,nextEligibleAt:now+this.cooldownMs,seen};
      this.records.delete(id);this.records.set(id,record);this.schedule(id,record);return true;
    }
    schedule(id,record){
      if(this.now()-record.observedAt+this.delayMs>this.maxAgeMs){record.state='expired';return;}
      record.state='scheduled';record.timer=this.setTimer(()=>{record.timer=null;return this.run(id,record);},this.delayMs);
    }
    async run(id,record){
      if(this.disposed||this.records.get(id)!==record)return;
      if(this.paused||this.isPaused()){record.state='paused';return;}
      if(this.now()-record.observedAt>this.maxAgeMs){record.state='expired';return;}
      record.capturing=true;record.state='capturing';record.attempts++;
      try{
        const result=await this.capture(id);
        if(this.disposed||this.records.get(id)!==record||record.state==='cancelled')return;
        if(result?.error)throw Error(result.error);
        const hash=typeof result==='string'?result:result?.hash;
        if(!hash)throw Error('Camera fetch returned no photo hash');
        if(hash!==record.firstHash){record.seen.add(hash);record.state='complete';return;}
        record.state='unchanged';
      }catch(error){
        if(this.disposed||this.records.get(id)!==record||record.state==='cancelled')return;
        record.state='error';this.onError(error,id);
      }finally{
        record.capturing=false;
        if(!this.disposed&&this.records.get(id)===record&&['unchanged','error'].includes(record.state)&&record.attempts<this.maxAttempts){
          if(this.paused||this.isPaused())record.state='paused';else this.schedule(id,record);
        }
      }
    }
    // pause() cancels timers; active fetches may finish. wake() explicitly resumes
    // only still-fresh requests. The capture callback owns its fetch cancellation.
    pause(){
      this.paused=true;
      for(const record of this.records.values())if(record.timer!==null){this.clearTimer(record.timer);record.timer=null;record.state='paused';}
    }
    wake(){
      if(this.disposed)return;this.paused=false;if(this.isPaused())return;
      for(const [id,record]of this.records)if(record.state==='paused'&&!record.capturing)this.schedule(id,record);
    }
    cancel(cameraId){
      const record=this.records.get(String(cameraId));if(!record)return;
      if(record.timer!==null)this.clearTimer(record.timer);record.timer=null;record.state='cancelled';
    }
    dispose(){this.disposed=true;for(const id of this.records.keys())this.cancel(id);this.records.clear();}
  }
  root.PairCapture=PairCapture;
})(globalThis);
