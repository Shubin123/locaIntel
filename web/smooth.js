/* Keyed DOM updates: retain image, focus, scroll position and stable pane elements. */
(function(root){
  const swaps=new WeakMap();
  function patch(current,next){
    if(current===next)return current;
    if(current.nodeType!==next.nodeType||current.nodeName!==next.nodeName){current.replaceWith(next);return next;}
    if(current.nodeType===3){if(current.nodeValue!==next.nodeValue)current.nodeValue=next.nodeValue;return current;}
    if(current.nodeType!==1)return current;
    for(const attr of [...current.attributes])if(!next.hasAttribute(attr.name))current.removeAttribute(attr.name);
    for(const attr of [...next.attributes])if(current.getAttribute(attr.name)!==attr.value)current.setAttribute(attr.name,attr.value);
    if('onclick'in next)current.onclick=next.onclick;
    if('onload'in next)current.onload=next.onload;if('onerror'in next)current.onerror=next.onerror;
    const keyed=new Map([...current.children].filter(n=>n.dataset.key).map(n=>[n.dataset.key,n]));
    let cursor=current.firstChild;
    for(const desired of [...next.childNodes]){
      let found=desired.nodeType===1&&desired.dataset.key?keyed.get(desired.dataset.key):cursor;
      if(found&&desired.nodeType===1&&desired.classList.contains('picture')&&found.classList?.contains('picture')){
        if(found!==cursor)current.insertBefore(found,cursor);picture(found,desired);cursor=found.nextSibling;
      }else if(found){if(found!==cursor)current.insertBefore(found,cursor);const actual=patch(found,desired);cursor=actual.nextSibling;}
      else{current.insertBefore(desired,cursor);cursor=desired.nextSibling;}
    }
    while(cursor){const nextSibling=cursor.nextSibling;cursor.remove();cursor=nextSibling;}
    return current;
  }
  function picture(current,next){
    const token={};swaps.set(current,token);
    if(next.dataset.loading==='true'){current.setAttribute('aria-busy','true');if(!current.querySelector('.photo-pending')){const status=document.createElement('span');status.className='photo-pending';status.textContent='Loading saved photo…';current.append(status);}return current;}
    const oldImage=current.querySelector('img'),newImage=next.querySelector('img');
    const commit=()=>{
      if(swaps.get(current)!==token||!current.isConnected)return;
      const changed=current.dataset.frameKey!==next.dataset.frameKey;
      patch(current,next);
      if(changed&&current.animate&&!root.matchMedia?.('(prefers-reduced-motion: reduce)').matches)current.animate([{opacity:.94},{opacity:1}],{duration:180,easing:'ease-out'});
      root.dispatchEvent?.(new Event('monitor-frame-commit'));
    };
    if(oldImage&&newImage&&oldImage.getAttribute('src')!==newImage.getAttribute('src')&&newImage.getAttribute('src')){
      const preload=new Image();preload.src=newImage.src;
      const ready=preload.decode?preload.decode():new Promise((resolve,reject)=>{preload.onload=resolve;preload.onerror=reject;});
      ready.then(commit).catch(()=>{if(swaps.get(current)===token)current.dataset.swapError='New image unavailable; previous frame retained';});
    }else commit();
    return current;
  }
  function list(container,nodes){
    const keyed=new Map([...container.children].map(n=>[n.dataset.key,n]));let cursor=container.firstElementChild;const keep=new Set();
    for(const desired of nodes){const key=desired.dataset.key;let element=keyed.get(key);if(element)element=patch(element,desired);else element=desired;keep.add(element);if(element!==cursor)container.insertBefore(element,cursor);cursor=element.nextElementSibling;}
    for(const old of [...container.children])if(!keep.has(old))old.remove();
  }
  function options(select,items,value){const version=JSON.stringify(items);if(select.dataset.optionsVersion!==version){select.replaceChildren(...items.map(([label,id])=>new Option(label,id)));select.dataset.optionsVersion=version;}if(select.value!==value)select.value=value;}
  root.SmoothUI={patch,picture,list,options};
})(globalThis);
