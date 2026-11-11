import {createSceneDisplay} from './recording_scene.js';

const $ = id => document.getElementById(id);
const apiRoot = '/api/point-tracking/';
let sessions = [], lastPreview = 0, currentStatus = {}, result = null, mode = 'preview', fitNext = true;
let playing = false, resultBase = '', requestBusy = false, pollBusy = false, lastPlayback = 0;
let cachedPreview=null, frozenPreview=null, optionsSynced=false, displayTimer=null;
let livePreview=null, clip=null, clipBase='', availableClipBase='', clipIndex=0, clipPlaying=false, clipLoading=false;
let clipRequest=0,lastClipTick=0,autoOpenClip=false,previousDetail=null;
let clipStartTime=0,clipStartWall=0;
const clipFrames=new Map();
const preferenceKey='anatobreast3d.recording.views.v2';
try {
  const saved=JSON.parse(localStorage.getItem(preferenceKey)||'{}');
  for(const id of ['viewMode','scope','detail','pointSize','opacity','feedMode']) {
    if(saved[id]!==undefined && (!$(id).options || [...$(id).options].some(o=>o.value===saved[id]))) $(id).value=saved[id];
  }
} catch {}
function savePreferences() {
  try { localStorage.setItem(preferenceKey,JSON.stringify(Object.fromEntries(['viewMode','scope','detail','pointSize','opacity','feedMode'].map(id=>[id,id==='detail'&&previousDetail?previousDetail:$(id).value])))); } catch {}
}

function error(message) { $('error').textContent = message; $('error').hidden = !message; }
async function json(url, payload) {
  const response = await fetch(url, payload === undefined ? {cache:'no-store'} : {
    method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(payload),
  });
  const value = url.split('?')[0].endsWith('.gz')
    ? await new Response(response.body.pipeThrough(new DecompressionStream('gzip'))).json()
    : await response.json();
  if (!response.ok || value.ok === false) throw new Error(value.error || `Request failed (${response.status})`);
  return value;
}
async function action(path, payload = {}) {
  if (requestBusy) return;
  requestBusy = true; error('');
  try { await json(apiRoot + path, payload); await poll(); }
  catch (e) { error(e.message); }
  finally { requestBusy = false; }
}

const sceneDisplay=createSceneDisplay($('scene'),now=>{
  if (playing && result && now - lastPlayback > 120) {
    lastPlayback = now;
    $('frame').value = (Number($('frame').value) + 1) % result.frame_count;
    drawTracks();
  }
  if(clipPlaying&&clip&&!clipLoading) {
    const target=clipStartTime+(now-clipStartWall)/1000*Number($('clipSpeed').value);
    let next=clipIndex;
    while(next+1<clip.frame_count&&clip.times_s[next+1]<=target) next++;
    if(clipIndex===clip.frame_count-1) setClipPlaying(false);
    else if(next>clipIndex) seekClip(next);
  }
});

function viewOptions() {
  return {view:$('viewMode').value,pointSize:Number($('pointSize').value),opacity:Number($('opacity').value)/100,
          axes:$('axes').checked,grid:$('grid').checked,visible:[1,2].map(i=>$(`cam${i}visible`).checked),
          frustums:$('frustums').checked,frustumDepth:Number($('frustumDepth').value),
          scope:$('scope').value,near:Number($('near').value),far:Number($('far').value)};
}
const hints={
  color:'Photographic colour on the observed depth surface. Each camera contributes its own colour; hide one to inspect it separately.',
  camera:'Cyan and orange identify the two cameras. Look for double edges and separated surfaces.',
  'color-points':'Measured RGB-D samples with photographic colours. Point size is adjustable under Display settings.',
  shape:'Neutral lighting reveals the captured shape without photographic colour differences.',
  depth:'Colour represents camera Z depth, in metres. The two cameras measure depth from their own positions.',
  gap:'Nearest sampled surface separation between cameras. Grey means no surface from the other camera within 30 mm. Occlusion and sampling affect this check.',
  transparent:'Transparent camera surfaces help reveal layers and offsets. Adjust opacity under Display settings.',
  normals:'Colour shows the orientation of the observed surface relative to the viewing direction.',
  wire:'Triangles connect neighbouring valid depth samples. Missing depth and large depth jumps remain open.',
};
function draw3D() {
  if(!['preview','playback'].includes(mode)) return;
  const preview=frozenPreview||cachedPreview; if(!preview) return;
  const options=viewOptions(), counts=sceneDisplay.preview(preview,options);
  const pending=mode==='preview'&&preview.geometry_version===2 && (preview.scope!==$('scope').value||preview.detail!==$('detail').value);
  $('sceneInfo').textContent=(frozenPreview?'3D paused · ':'')+(pending?'Updating area and detail… · ':'')+
    `${counts.points.toLocaleString()} depth samples · ${counts.triangles.toLocaleString()} surface triangles`;
  Object.assign($('sceneInfo').dataset,{view:options.view,scope:preview.full_scene?options.scope:preview.scope||'',
    detail:preview.detail||'',points:counts.points,faces:counts.triangles,timestamp:preview.timestamp});
  $('sceneEmpty').hidden=counts.points>0;
  if(!counts.points) $('sceneEmpty').textContent=options.visible.some(Boolean)?'No valid depth in this area. Try Whole scene or widen the depth range.':'Select a camera to display.';
  $('geometryHint').textContent=(hints[options.view]||'')+
    ($('scope').value==='breast_area'?' Full breast area includes a margin around the detected breast.':'')+
    ' Live depth shows visible surfaces; uncaptured areas stay open.';
  $('sceneLegend').hidden=!['depth','gap','normals'].includes(options.view);
  if(options.view==='depth') {
    const [near,far]=mode==='playback'?[options.near,options.far]:(preview.depth_range_m||[.3,1]);
    $('sceneLegend').innerHTML=`<span>${near.toFixed(2)} m</span><i class="legend-bar"></i><span>${far.toFixed(2)} m</span><span class="legend-note">Camera Z depth</span>`;
  } else if(options.view==='gap') $('sceneLegend').innerHTML='<span>0 mm</span><i class="legend-bar gap"></i><span>10+ mm</span><span class="legend-note">Grey: no nearby surface</span>';
  else if(options.view==='normals') $('sceneLegend').textContent='Surface orientation · rotate to inspect the shape';
  if(fitNext&&!pending) { sceneDisplay.fit();fitNext=false; }
  $('pointSizeText').textContent=`${options.pointSize} mm`;$('opacityText').textContent=`${Math.round(options.opacity*100)}%`;
  $('frustumDepthText').textContent=`${options.frustumDepth.toFixed(2)} m`;
}
function applyFeedMode() {
  const feed=$('feedMode').value;
  for(const i of [1,2]) {
    const slot=`cam${i}`;
    $(slot+'rgb').hidden=!$(slot+'rgb').src||feed==='depth';
    $(slot+'depth').hidden=!$(slot+'depth').src||!['depth','blend'].includes(feed);
    $(slot+'depth').style.opacity=feed==='depth'?1:Number($('overlay').value)/100;
    $(slot+'mask').hidden=feed!=='mask'||!$(slot+'mask').getAttribute('src');
  }
}
function updateDisplayOptions() {
  if(mode==='playback') {draw3D();return;}
  clearTimeout(displayTimer);
  displayTimer=setTimeout(()=>{
    if(requestBusy) { updateDisplayOptions();return; }
    if(!currentStatus.preview?.running) return;
    action('command',{action:'range',min_depth_m:Number($('near').value),max_depth_m:Number($('far').value),
      scope:$('scope').value,detail:$('detail').value,mask_overlay:$('feedMode').value==='mask'});
  },150);
}
function freeze(value) {
  frozenPreview=value?cachedPreview:null;
  $('frozenBadge').hidden=!frozenPreview;$('freeze').textContent=frozenPreview?'Resume 3D':'Freeze 3D';
  $('freeze').setAttribute('aria-pressed',String(Boolean(frozenPreview)));
  for(const id of ['scope','detail','near','far','range']) $(id).disabled=Boolean(frozenPreview);
  draw3D();
}

function drawPreview(preview, base, recorded=false) {
  cachedPreview=preview;
  $('connection').textContent = recorded?(clipPlaying?'Playing recorded RGB-D':'Recorded RGB-D · paused'):
    preview.state === 'recording' ? '● Recording both cameras' : 'Both cameras connected';
  $('drift').textContent = `${preview.drift_ms.toFixed(1)} ms`;
  $('calibration').textContent = 'Saved rig loaded';
  $('calibration').title = preview.calibration;
  for (let i=0;i<preview.cameras.length;i++) {
    const view = preview.cameras[i], slot = `cam${i+1}`;
    $(slot+'rgb').src = base + view.rgb + '?t=' + preview.timestamp;
    $(slot+'depth').src = base + view.depth + '?t=' + preview.timestamp;
    if(view.mask) $(slot+'mask').src=base+view.mask+'?t='+preview.timestamp;
    else $(slot+'mask').removeAttribute('src');
    $(slot+'empty').hidden = true;
    $(slot+'meta').textContent = `${view.id} · ${recorded?'recorded':view.serial} · depth ${view.valid_depth_percent}%`;
  }
  applyFeedMode();
  if(!recorded&&!optionsSynced&&preview.geometry_version===2) {
    optionsSynced=true;
    if(preview.scope!==$('scope').value||preview.detail!==$('detail').value||$('feedMode').value==='mask') updateDisplayOptions();
  }
  if (!['preview','playback'].includes(mode)) return;
  if(!frozenPreview) draw3D();
}

function drawTracks() {
  if (!result) return;
  const t = Number($('frame').value), count=sceneDisplay.tracks(result,t);
  const start = result.timestamps_ns[0], elapsed = (result.timestamps_ns[t] - start) / 1e9;
  $('frameText').textContent = `Frame ${t+1}/${result.frame_count} · ${elapsed.toFixed(2)} s · ${count} visible`;
  $('sceneInfo').textContent=`${count} visible tracked points · ${result.camera_id}`;
  $('geometryHint').textContent = 'Tracked points are coloured by movement from their initial position: cyan = small, orange = 20 mm or more. Points predicted as hidden are not displayed. Coordinates use the saved calibration; this is not an accuracy measurement.';
  $('sceneEmpty').hidden = true;
}

async function showTracks() {
  try {
    result = await json(resultBase + 'result.json');
    setClipPlaying(false); $('clipPlayback').hidden=true;
    mode = 'tracks'; playing = false; $('play').textContent = 'Play';
    for(const id of ['viewControls','displaySettings','freeze','sceneLegend']) $(id).hidden=true;
    document.querySelector('.scene-tools').hidden=true;$('frozenBadge').hidden=true;
    document.querySelector('.frustum-settings').hidden=true;
    $('showOverlap').hidden = false;
    $('sceneTitle').textContent = `3D surface tracking · ${result.point_count} points`;
    $('playback').hidden = false; $('frame').max = result.frame_count - 1; $('frame').value = 0;
    drawTracks(); sceneDisplay.fit(result.query_points.map(q=>q.slice(1)));
  } catch(e) { error(e.message); }
}

function setClipPlaying(value) {
  clipPlaying=value;lastClipTick=performance.now();$('clipPlay').textContent=value?'Pause':'Play';
  clipStartWall=lastClipTick;clipStartTime=clip?.times_s[clipIndex]||0;
  if(mode==='playback') $('connection').textContent=value?'Playing recorded RGB-D':'Recorded RGB-D · paused';
}
async function seekClip(index) {
  if(!clip) return;
  index=Math.max(0,Math.min(clip.frame_count-1,index));
  const request=++clipRequest;clipLoading=true;
  try {
    let frame=clipFrames.get(index);
    if(!frame) {frame=await json(clipBase+clip.frames[index]);clipFrames.set(index,frame);}
    for(const view of frame.cameras) if(!view.frustum&&clip.frustums?.[view.id]) view.frustum=clip.frustums[view.id];
    if(request!==clipRequest||mode!=='playback') return;
    clipIndex=index;$('clipFrame').value=index;
    drawPreview(frame,clipBase,true);
    $('sceneTitle').textContent='Recorded RGB-D · 3D inspection';
    $('clipTime').textContent=`${clip.times_s[index].toFixed(2)} / ${clip.duration_s.toFixed(2)} s`;
    $('clipFrameInfo').textContent=`Frame ${index+1} of ${clip.frame_count} · ${clip.cameras.length} cameras · ~5 fps inspection preview`+(clip.truncated?' · first minute':'');
    while(clipFrames.size>8) clipFrames.delete(clipFrames.keys().next().value);
  } catch(e) {setClipPlaying(false);error(e.message);}
  finally {if(request===clipRequest) clipLoading=false;}
}
async function showClip() {
  if(!availableClipBase) return;
  try {
    error('');const manifest=await json(availableClipBase+'manifest.json');
    if(clipBase!==availableClipBase) clipFrames.clear();clipBase=availableClipBase;clip=manifest;
    previousDetail=previousDetail||$('detail').value;
    mode='playback';playing=false;frozenPreview=null;fitNext=true;setClipPlaying(false);
    $('clipPlayback').hidden=false;$('playback').hidden=true;$('showOverlap').hidden=false;
    for(const id of ['viewControls','displaySettings']) $(id).hidden=false;
    document.querySelector('.scene-tools').hidden=false;
    document.querySelector('.frustum-settings').hidden=false;
    $('freeze').hidden=true;$('frozenBadge').hidden=true;
    for(const id of ['scope','near','far','range']) $(id).disabled=false;
    $('detail').value='fine';$('detail').disabled=true;
    $('clipFrame').max=clip.frame_count-1;
    await seekClip(0);
    $('record').disabled=true;
  } catch(e) {error(e.message);}
}

function renderCameras() {
  const session = sessions.find(s=>s.id === $('sessions').value);
  const previous = $('camera').value;
  $('camera').replaceChildren();
  for (const name of session?.cameras || []) {
    const option = document.createElement('option'); option.value = name; option.textContent = name;
    $('camera').append(option);
  }
  if ([...$('camera').options].some(o=>o.value === previous)) $('camera').value = previous;
  else if (session?.cameras?.includes('cam2')) $('camera').value = 'cam2';
}
async function refreshSessions(preferred = null) {
  const response = await json('/api/replay/sessions');
  sessions = response.sessions;
  const previous = preferred || $('sessions').value;
  $('sessions').replaceChildren();
  for (const session of sessions) {
    const option = document.createElement('option'); option.value=session.id;
    option.textContent=`${session.name} · ${Number(session.duration_s || 0).toFixed(1)} s`;
    $('sessions').append(option);
  }
  if (!sessions.length) { const option = document.createElement('option'); option.textContent='No recordings yet';option.value='';$('sessions').append(option); }
  else if (sessions.some(s=>s.id===previous)) $('sessions').value=previous;
  renderCameras();
}

async function poll() {
  if (pollBusy) return;
  pollBusy = true;
  try {
    currentStatus = await json(apiRoot + 'status');
    const preview = currentStatus.preview || {}, tracking = currentStatus.tracking || {}, playback=currentStatus.playback||{};
    const active = ['starting','preview','recording','saving'].includes(preview.state) && preview.running;
    $('preview').disabled = active;
    $('stop').disabled = !active || preview.state === 'saving';
    $('stop').textContent = preview.state === 'recording' ? 'Stop & save' : 'Stop preview';
    $('record').disabled = preview.state !== 'preview' || !active || mode!=='preview';
    $('recordState').textContent = preview.state === 'recording' ? `${preview.elapsed_s?.toFixed(1) || '0'} s` : (preview.state || 'Ready when you are');
    $('captureStatus').textContent = preview.message || 'Start preview and check your framing before recording.';
    $('track').disabled = !sessions.length || tracking.running;
    $('trackingState').textContent = tracking.state || 'TAPIP3D';
    $('trackingStatus').textContent = tracking.message || 'Tracks one camera over time using measured depth and the saved calibration.';
    $('loadClip').disabled=!sessions.length||playback.running;
    $('clipStatus').textContent=playback.state||'RGB-D + 3D';
    $('clipMessage').textContent=playback.message||'Replay both camera views and the 3D surface together. Pause, scrub, and change the view.';
    if(playback.state==='failed') {error(playback.message);autoOpenClip=false;}
    if(playback.state==='complete') {
      availableClipBase=playback.base;$('showClip').hidden=false;
      if(autoOpenClip) {autoOpenClip=false;await showClip();}
    }
    if (preview.state === 'failed') error(preview.message);
    if (tracking.state === 'failed') error(tracking.message);
    if (tracking.state === 'complete') {
      resultBase = tracking.base;
      $('showTracks').hidden = $('downloads').hidden = false;
      $('csv').href = resultBase+'tracks.csv'; $('npz').href = resultBase+'tracks.npz';
    }
    if (preview.state === 'saving'&&mode!=='playback') $('connection').textContent = 'Saving recording…';
    if (active && preview.base && mode!=='playback') {
      const frame = await json(preview.base+'preview.json').catch(()=>null);
      if (frame && frame.timestamp !== lastPreview) { lastPreview=frame.timestamp; livePreview={frame,base:preview.base};drawPreview(frame, preview.base); }
    } else if (preview.state === 'recorded' && preview.session && !sessions.some(s=>s.id===preview.session)) {
      await refreshSessions(preview.session);
    } else if (!active&&mode!=='playback') $('connection').textContent = preview.state === 'recorded' ? 'Recording saved · cameras released' : 'Cameras stopped';
  } catch(e) { error(e.message); }
  finally { pollBusy = false; }
}

$('preview').onclick = () => {
  showOverlap();optionsSynced=false;
  action('preview');
};
$('stop').onclick = () => action('command', {action:'stop'});
$('record').onclick = () => action('command', {action:'record', duration_s:Number($('duration').value)});
$('range').onclick = () => {
  if(!(.1<=Number($('near').value)&&Number($('near').value)<Number($('far').value)&&Number($('far').value)<=3)) {error('Choose a depth range between 0.1 and 3 metres, with near smaller than far.');return;}
  fitNext=true;updateDisplayOptions();
};
for(const id of ['scope','detail']) $(id).onchange=()=>{fitNext=true;savePreferences();updateDisplayOptions();};
for(const id of ['viewMode','axes','grid','pointSize','opacity','cam1visible','cam2visible']) {
  $(id).addEventListener(id==='pointSize'||id==='opacity'?'input':'change',()=>{savePreferences();draw3D();});
}
$('frustums').onchange=()=>{draw3D();if($('frustums').checked) sceneDisplay.fit(null,true);};
$('fitCameras').onclick=()=>{$('frustums').checked=true;draw3D();sceneDisplay.fit(null,true);};
$('frustumDepth').oninput=draw3D;
$('feedMode').onchange=()=>{
  if($('feedMode').value==='blend'&&Number($('overlay').value)===0) $('overlay').value=35;
  savePreferences();applyFeedMode();updateDisplayOptions();
};
$('freeze').onclick=()=>freeze(!frozenPreview);
for(const button of document.querySelectorAll('[data-view]')) button.onclick=()=>sceneDisplay.preset(button.dataset.view);
$('expand').onclick=async()=>{
  try {
    if(document.fullscreenElement) await document.exitFullscreen();
    else await $('geometryPanel').requestFullscreen();
  } catch(e) { error('Use the Expand button in a browser that supports fullscreen views.'); }
};
document.addEventListener('fullscreenchange',()=>{$('expand').textContent=document.fullscreenElement?'Shrink':'Expand';});
$('track').onclick = () => action('start', {session:$('sessions').value, camera:$('camera').value});
$('sessions').onchange = renderCameras;
$('showTracks').onclick = showTracks;
const loadClip=async()=>{
  autoOpenClip=true;await action('playback',{session:$('sessions').value});
};
$('loadClip').onclick=loadClip;$('showClip').onclick=showClip;
$('clipPlay').onclick=async()=>{if(!clipPlaying&&clipIndex===clip.frame_count-1) await seekClip(0);setClipPlaying(!clipPlaying);};
$('clipPrevious').onclick=()=>{setClipPlaying(false);seekClip(clipIndex-1);};
$('clipNext').onclick=()=>{setClipPlaying(false);seekClip(clipIndex+1);};
$('clipFrame').oninput=()=>{setClipPlaying(false);seekClip(Number($('clipFrame').value));};
$('clipSpeed').onchange=()=>{if(clipPlaying) setClipPlaying(true);};
function showOverlap() {
  setClipPlaying(false);++clipRequest;clipLoading=false;$('clipPlayback').hidden=true;
  if(previousDetail) {$('detail').value=previousDetail;previousDetail=null;}
  mode='preview'; fitNext=true; playing=false; $('playback').hidden=true;
  for(const id of ['viewControls','displaySettings','freeze']) $(id).hidden=false;
  document.querySelector('.scene-tools').hidden=false;freeze(false);
  document.querySelector('.frustum-settings').hidden=false;
  $('showOverlap').hidden=true; $('sceneTitle').textContent='3D camera overlap'; lastPreview=0;
  draw3D();
  if(livePreview) drawPreview(livePreview.frame,livePreview.base);
}
$('showOverlap').onclick = () => { showOverlap();poll(); };
$('overlay').oninput=()=>{ $('feedMode').value=Number($('overlay').value)>0?'blend':'color';savePreferences();applyFeedMode(); };
$('reset').onclick = () => sceneDisplay.fit(mode==='tracks' && result ? result.query_points.map(q=>q.slice(1)) : null);
$('frame').oninput = () => { playing=false; $('play').textContent='Play'; drawTracks(); };
$('play').onclick = () => { playing=!playing; $('play').textContent=playing?'Pause':'Play'; };
window.addEventListener('error', event=>error(event.message));
window.addEventListener('unhandledrejection', event=>error(String(event.reason?.message || event.reason)));
await refreshSessions(); await poll(); setInterval(poll, 600);
