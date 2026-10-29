/* The one map everything is pinned to. Camera pins are always shown; vehicle tracking,
   density, facing and routes are layers switched from the map's layer panel. Other
   modules draw into AppMap.layer(name); each layer has its own pane so tracking always
   sits above density and arrows. The map moves only when you move it or ask it to: the
   tile server sees the area you look at, so nothing pans to your GPS track by itself. */
(function(root){
  const LAYERS=[{id:'tracking',on:true,z:460},{id:'density',on:false},{id:'facing',on:false,z:430},{id:'routes',on:false,z:440}];
  const groups=new Map(),listeners=new Set();let map=null,heat=null;
  const saved=()=>{try{return JSON.parse(localStorage.getItem('locaintel.layers')||'{}')||{};}catch(_){return {};}};
  const box=id=>document.getElementById('layer-'+id);
  const emit=(kind,id)=>{for(const f of listeners)try{f(kind,id);}catch(error){console.warn('Map listener failed',error);}};
  function init(){
    if(map||!root.L||!document.getElementById('map'))return map;
    map=L.map('map',{preferCanvas:true,scrollWheelZoom:true,zoomSnap:.5}).setView([53.7,-125],5);
    L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png',{maxZoom:18,attribution:'&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'}).addTo(map);
    map.createPane('cameras').style.zIndex=450;groups.set('cameras',L.layerGroup().addTo(map));
    for(const l of LAYERS){if(l.z)map.createPane(l.id).style.zIndex=l.z;groups.set(l.id,L.layerGroup());}
    if(L.heatLayer)heat=L.heatLayer([],{radius:28,blur:22,maxZoom:10,minOpacity:.2,gradient:{.2:'#467bd0',.4:'#49c7ad',.6:'#e7e775',.8:'#f19a4b',1:'#ed5252'}});
    const state=saved();
    for(const l of LAYERS){const input=box(l.id);if(!input)continue;input.checked=state[l.id]??l.on;input.addEventListener('change',()=>apply(l.id,true));apply(l.id,false);}
    let moveTimer;map.on('moveend',()=>{clearTimeout(moveTimer);moveTimer=setTimeout(()=>emit('move'),200);});
    // The map's box changes with the layout (phone rotation, panel resize).
    if(root.ResizeObserver)new ResizeObserver(()=>map.invalidateSize({pan:false})).observe(document.getElementById('map'));
    return map;
  }
  function apply(id,persist){
    const on=!!box(id)?.checked,group=groups.get(id);
    if(group){if(on&&!map.hasLayer(group))group.addTo(map);else if(!on&&map.hasLayer(group))group.remove();}
    if(id==='density'&&heat){if(on&&!map.hasLayer(heat))heat.addTo(map);else if(!on&&map.hasLayer(heat))heat.remove();}
    for(const n of document.querySelectorAll(`[data-layer-option="${id}"]`))n.hidden=!on;
    if(persist){try{localStorage.setItem('locaintel.layers',JSON.stringify(Object.fromEntries(LAYERS.map(l=>[l.id,!!box(l.id)?.checked]))));}catch(_){}}
    emit('layer',id);
  }
  // Frame a set of points once, at most at the given zoom.
  function frame(points,{maxZoom=13,padding=[40,40]}={}){if(map&&points.length)map.fitBounds(points,{padding,maxZoom});}
  root.AppMap={init,frame,
    get map(){return map;},get heat(){return heat;},
    layer:id=>groups.get(id),
    enabled:id=>!!box(id)?.checked,
    onChange:f=>listeners.add(f),
    // The visible area plus a small margin, or null before the map exists.
    bounds:()=>map?map.getBounds().pad(.05):null};
})(globalThis);
