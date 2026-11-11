import * as THREE from 'three';
import {OrbitControls} from './vendor/OrbitControls.js';

const $ = id => document.getElementById(id);
const apiRoot = '/api/point-tracking/';
let sessions = [], lastPreview = 0, currentStatus = {}, result = null, mode = 'preview', fitNext = true;
let playing = false, resultBase = '', requestBusy = false, pollBusy = false, lastPlayback = 0;

function error(message) { $('error').textContent = message; $('error').hidden = !message; }
async function json(url, payload) {
  const response = await fetch(url, payload === undefined ? {cache:'no-store'} : {
    method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(payload),
  });
  const value = await response.json();
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

const scene = new THREE.Scene();
scene.background = new THREE.Color('#10252e');
const camera = new THREE.PerspectiveCamera(45, 1, .001, 100);
camera.up.set(0, 0, 1);
camera.position.set(.5, -.7, .8);
const renderer = new THREE.WebGLRenderer({antialias:true});
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
$('scene').append(renderer.domElement);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
scene.add(new THREE.AxesHelper(.08));
const clouds = new Map();

function cloud(name, xyz, color, size = .002) {
  let points = clouds.get(name);
  if (!points) {
    points = new THREE.Points(new THREE.BufferGeometry(), new THREE.PointsMaterial({color, size, sizeAttenuation:true}));
    scene.add(points); clouds.set(name, points);
  }
  points.geometry.dispose();
  points.geometry = new THREE.BufferGeometry();
  points.geometry.setAttribute('position', new THREE.Float32BufferAttribute(xyz.flat(), 3));
  points.geometry.computeBoundingSphere();
  points.material.color.set(color);
  points.visible = true;
  return points;
}
function hideClouds() { for (const object of clouds.values()) object.visible = false; }
function fit(points = null) {
  const values = points || [...clouds.values()].filter(p => p.visible).flatMap(p => {
    const a = p.geometry.getAttribute('position'); return Array.from({length:a.count}, (_,i)=>[a.getX(i),a.getY(i),a.getZ(i)]);
  });
  if (!values.length) return;
  const bounds = new THREE.Box3().setFromPoints(values.map(p => new THREE.Vector3(...p)));
  const centre = bounds.getCenter(new THREE.Vector3());
  const length = Math.max(.15, bounds.getSize(new THREE.Vector3()).length());
  controls.target.copy(centre);
  camera.position.copy(centre).add(new THREE.Vector3(.5, -.65, .8).multiplyScalar(length));
  controls.update(); fitNext = false;
}
new ResizeObserver(() => {
  const width = $('scene').clientWidth, height = $('scene').clientHeight;
  camera.aspect = width / height; camera.updateProjectionMatrix(); renderer.setSize(width, height, false);
}).observe($('scene'));
function animate(now) {
  requestAnimationFrame(animate); controls.update(); renderer.render(scene, camera);
  if (playing && result && now - lastPlayback > 120) {
    lastPlayback = now;
    $('frame').value = (Number($('frame').value) + 1) % result.frame_count;
    drawTracks();
  }
}
requestAnimationFrame(animate);

function drawPreview(preview, base) {
  $('foreground').closest('label').hidden = !Object.hasOwn(preview, 'foreground_only');
  $('connection').textContent = preview.state === 'recording' ? '● Recording both cameras' : 'Both cameras connected';
  $('drift').textContent = `${preview.drift_ms.toFixed(1)} ms`;
  $('calibration').textContent = 'Saved rig loaded';
  $('calibration').title = preview.calibration;
  for (let i=0;i<preview.cameras.length;i++) {
    const view = preview.cameras[i], slot = `cam${i+1}`;
    $(slot+'rgb').src = base + view.rgb + '?t=' + preview.timestamp;
    $(slot+'depth').src = base + view.depth + '?t=' + preview.timestamp;
    $(slot+'rgb').hidden = $(slot+'depth').hidden = false;
    $(slot+'empty').hidden = true;
    $(slot+'meta').textContent = `${view.id} · ${view.serial} · depth ${view.valid_depth_percent}%`;
  }
  if (mode !== 'preview') return;
  hideClouds();
  preview.cameras.forEach((view,i) => {
    const object = cloud(`camera${i+1}`, view.xyz, i === 0 ? '#41cfe6' : '#ffab5c');
    object.visible = $(`cam${i+1}visible`).checked;
  });
  $('sceneEmpty').hidden = true;
  if (fitNext) fit();
}

function drawTracks() {
  if (!result) return;
  hideClouds();
  const t = Number($('frame').value), coords = result.coords[t], visible = result.visible[t];
  cloud('context', result.context[t].xyz, '#52747f', .0015);
  const points = [], colors = [];
  coords.forEach((xyz, i) => {
    if (!visible[i] || xyz.some(value => value === null)) return;
    points.push(xyz);
    const color = new THREE.Color().setHSL(.52 - Math.min(result.displacement_mm[t][i] / 20, 1) * .5, .9, .6);
    colors.push(color.r, color.g, color.b);
  });
  const tracks = cloud('tracks', points, 'white', .004);
  tracks.material.depthTest = false; tracks.renderOrder = 10;
  tracks.geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
  tracks.material.vertexColors = true; tracks.material.needsUpdate = true;
  const start = result.timestamps_ns[0], elapsed = (result.timestamps_ns[t] - start) / 1e9;
  $('frameText').textContent = `Frame ${t+1}/${result.frame_count} · ${elapsed.toFixed(2)} s · ${points.length} visible`;
  $('geometryHint').textContent = 'Tracked points are coloured by movement from their initial position: cyan = small, orange = 20 mm or more. Points predicted as hidden are not displayed. Coordinates use the saved calibration; this is not an accuracy measurement.';
  $('sceneEmpty').hidden = true;
}

async function showTracks() {
  try {
    result = await json(resultBase + 'result.json');
    mode = 'tracks'; playing = false; $('play').textContent = 'Play';
    $('showOverlap').hidden = false;
    $('sceneTitle').textContent = `3D surface tracking · ${result.point_count} points`;
    $('playback').hidden = false; $('frame').max = result.frame_count - 1; $('frame').value = 0;
    drawTracks(); fit(result.query_points.map(q=>q.slice(1)));
  } catch(e) { error(e.message); }
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
    const preview = currentStatus.preview || {}, tracking = currentStatus.tracking || {};
    const active = ['starting','preview','recording','saving'].includes(preview.state) && preview.running;
    $('preview').disabled = active;
    $('stop').disabled = !active || preview.state === 'saving';
    $('stop').textContent = preview.state === 'recording' ? 'Stop & save' : 'Stop preview';
    $('record').disabled = preview.state !== 'preview' || !active;
    $('recordState').textContent = preview.state === 'recording' ? `${preview.elapsed_s?.toFixed(1) || '0'} s` : (preview.state || 'Ready when you are');
    $('captureStatus').textContent = preview.message || 'Start preview and check your framing before recording.';
    $('track').disabled = !sessions.length || tracking.running;
    $('trackingState').textContent = tracking.state || 'TAPIP3D';
    $('trackingStatus').textContent = tracking.message || 'Tracks one camera over time using measured depth and the saved calibration.';
    if (preview.state === 'failed') error(preview.message);
    if (tracking.state === 'failed') error(tracking.message);
    if (tracking.state === 'complete') {
      resultBase = tracking.base;
      $('showTracks').hidden = $('downloads').hidden = false;
      $('csv').href = resultBase+'tracks.csv'; $('npz').href = resultBase+'tracks.npz';
      $('reviewTracks').hidden = false;
      $('reviewTracks').href = 'tracks.html?base='+encodeURIComponent(resultBase);
    }
    if (preview.state === 'saving') $('connection').textContent = 'Saving recording…';
    if (active && preview.base) {
      const frame = await json(preview.base+'preview.json').catch(()=>null);
      if (frame && frame.timestamp !== lastPreview) { lastPreview=frame.timestamp; drawPreview(frame, preview.base); }
    } else if (preview.state === 'recorded' && preview.session && !sessions.some(s=>s.id===preview.session)) {
      await refreshSessions(preview.session);
    } else if (!active) $('connection').textContent = preview.state === 'recorded' ? 'Recording saved · cameras released' : 'Cameras stopped';
  } catch(e) { error(e.message); }
  finally { pollBusy = false; }
}

$('preview').onclick = () => {
  mode='preview'; fitNext=true; playing=false; $('playback').hidden=true; $('sceneTitle').textContent='3D camera overlap';
  $('geometryHint').textContent='Look for overlapping surfaces. A doubled edge can indicate calibration error, depth noise, or unsynchronised motion.';
  action('preview');
};
$('stop').onclick = () => action('command', {action:'stop'});
$('record').onclick = () => action('command', {action:'record', duration_s:Number($('duration').value)});
$('range').onclick = () => action('command', {action:'range', min_depth_m:Number($('near').value), max_depth_m:Number($('far').value), foreground_only:$('foreground').checked});
$('foreground').onchange = () => $('range').click();
$('track').onclick = () => action('start', {session:$('sessions').value, camera:$('camera').value, full_recording:$('trackingMode').value==='full'});
$('sessions').onchange = renderCameras;
$('showTracks').onclick = showTracks;
$('showOverlap').onclick = () => {
  mode='preview'; fitNext=true; playing=false; $('playback').hidden=true;
  $('showOverlap').hidden=true; $('sceneTitle').textContent='3D camera overlap'; lastPreview=0;
  $('geometryHint').textContent='Look for overlapping surfaces. A doubled edge can indicate calibration error, depth noise, or unsynchronised motion.';
  poll();
};
$('overlay').oninput = () => document.querySelectorAll('.depth').forEach(image=>image.style.opacity=Number($('overlay').value)/100);
$('reset').onclick = () => fit(mode==='tracks' && result ? result.query_points.map(q=>q.slice(1)) : null);
for (const i of [1,2]) $(`cam${i}visible`).onchange = () => { if(mode==='preview' && clouds.has(`camera${i}`)) clouds.get(`camera${i}`).visible=$(`cam${i}visible`).checked; };
$('frame').oninput = () => { playing=false; $('play').textContent='Play'; drawTracks(); };
$('play').onclick = () => { playing=!playing; $('play').textContent=playing?'Pause':'Play'; };
window.addEventListener('error', event=>error(event.message));
window.addEventListener('unhandledrejection', event=>error(String(event.reason?.message || event.reason)));
await refreshSessions(); await poll(); setInterval(poll, 600);
