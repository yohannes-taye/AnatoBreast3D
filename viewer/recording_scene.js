import * as THREE from 'three';
import {OrbitControls} from './vendor/OrbitControls.js';

const cameraColors = ['#41cfe6', '#ffab5c'];
const linear = Array.from({length:256}, (_,v) => {
  const x=v/255; return x<=.04045 ? x/12.92 : ((x+.055)/1.055)**2.4;
});
const depthPalette = Array.from({length:256}, (_,i)=>new THREE.Color().setHSL((1-i/255)*.67,.9,.55));
const gapPalette = Array.from({length:256}, (_,i)=>new THREE.Color().lerpColors(
  new THREE.Color(i<128?'#30c799':'#ffe072'), new THREE.Color(i<128?'#ffe072':'#fa5d5a'), (i%128)/127));
const unsupported = new THREE.Color('#617783');

export function createSceneDisplay(host, onAnimation = ()=>{}) {
  const scene = new THREE.Scene(); scene.background = new THREE.Color('#10252e');
  const camera = new THREE.PerspectiveCamera(45,1,.001,100);
  camera.up.set(0,0,-1); camera.position.set(.5,-.7,-.8);
  const renderer = new THREE.WebGLRenderer({antialias:true});
  renderer.setPixelRatio(Math.min(devicePixelRatio,1.5)); host.append(renderer.domElement);
  let controls;
  function orbit(target, up=new THREE.Vector3(0,0,-1)) {
    controls?.dispose(); camera.up.copy(up);
    controls = new OrbitControls(camera,renderer.domElement); controls.enableDamping=true;
    controls.target.copy(target); controls.update();
  }
  orbit(new THREE.Vector3());
  scene.add(new THREE.HemisphereLight(0xffffff,0x4a5961,1.8));
  for(const [position,intensity] of [[[1,-1,-2],2.2],[[-1,1,1],1.2]]) {
    const light=new THREE.DirectionalLight(0xffffff,intensity); light.position.set(...position); scene.add(light);
  }
  const axes=new THREE.AxesHelper(.08); scene.add(axes);
  const grid=new THREE.GridHelper(1.2,24,0x67818c,0x36515b); grid.rotation.x=Math.PI/2;
  grid.material.transparent=true; grid.material.opacity=.3; grid.visible=false; scene.add(grid);
  const objects=new Map(); let latestPreview=null;
  const frustums=new THREE.Group();scene.add(frustums);
  function hide() { for(const object of objects.values()) object.visible=false; }
  function geometry(xyz, triangles=null) {
    const result=new THREE.BufferGeometry();
    result.setAttribute('position',new THREE.Float32BufferAttribute(xyz.flat(),3));
    if(triangles?.length) result.setIndex(triangles);
    result.computeBoundingSphere(); return result;
  }
  function put(key, geo, material, mesh=false) {
    let object=objects.get(key);
    if(!object) { object=mesh?new THREE.Mesh():new THREE.Points(); objects.set(key,object); scene.add(object); }
    object.geometry?.dispose(); object.material?.dispose();
    object.geometry=geo; object.material=material; object.visible=true; return object;
  }
  function bounds(points=null) {
    if(points) return new THREE.Box3().setFromPoints(points.map(x=>new THREE.Vector3(...x)));
    const box=new THREE.Box3();
    for(const object of objects.values()) if(object.visible) {
      object.geometry.computeBoundingBox(); box.union(object.geometry.boundingBox);
    }
    return box;
  }
  function fit(points=null,includeCameras=false) {
    const box=bounds(points);
    if(includeCameras&&frustums.children.length) box.union(new THREE.Box3().setFromObject(frustums));
    if(box.isEmpty()) return;
    const centre=box.getCenter(new THREE.Vector3()), length=Math.max(.15,box.getSize(new THREE.Vector3()).length());
    camera.fov=45; camera.updateProjectionMatrix();
    const poses=latestPreview?.cameras.map(view=>view.T_world_cam).filter(Boolean)||[];
    const direction=new THREE.Vector3(.5,-.65,-.8);
    const up=new THREE.Vector3(0,0,-1);
    if(poses.length) {
      direction.set(0,0,0);
      for(const pose of poses) direction.add(new THREE.Vector3(pose[0][3],pose[1][3],pose[2][3]));
      direction.divideScalar(poses.length).sub(centre).normalize().multiplyScalar(1.35);
      up.set(0,0,0);
      for(const pose of poses) up.add(new THREE.Vector3(-pose[0][1],-pose[1][1],-pose[2][1]));
      if(up.lengthSq()<1e-6) up.set(0,1,0);else up.normalize();
    }
    camera.position.copy(centre).add(direction.multiplyScalar(length)); orbit(centre,up);
  }
  function preset(name) {
    if(name==='orbit') { fit(); return; }
    const box=bounds(); if(box.isEmpty()) return;
    const centre=box.getCenter(new THREE.Vector3()), length=Math.max(.15,box.getSize(new THREE.Vector3()).length());
    if(name.startsWith('camera')) {
      const pose=latestPreview?.cameras[Number(name.slice(-1))-1]?.T_world_cam;
      if(!pose) return;
      camera.position.set(pose[0][3],pose[1][3],pose[2][3]);
      orbit(centre,new THREE.Vector3(-pose[0][1],-pose[1][1],-pose[2][1]));return;
    }
    const direction={top:[0,0,-1.35],front:[0,-1.35,-.05],side:[1.35,0,-.05]}[name];
    if(!direction) return;
    camera.position.copy(centre).add(new THREE.Vector3(...direction).multiplyScalar(length));
    orbit(centre,name==='top'?new THREE.Vector3(0,1,0):new THREE.Vector3(0,0,-1));
  }
  function vertexColors(view, mode, near, far) {
    const colors=new Float32Array(view.xyz.length*3);
    for(let i=0;i<view.xyz.length;i++) {
      if(mode==='color'||mode==='color-points') {
        const rgb=view.colors?.[i] || [150,180,185];
        colors.set(rgb.map(x=>linear[x]),i*3);
      } else {
        let color;
        if(mode==='depth') color=depthPalette[Math.round(255*THREE.MathUtils.clamp(((view.depth_m?.[i]||near)-near)/(far-near),0,1))];
        else {
          const gap=view.gap_mm?.[i] ?? -1;
          color=gap<0 ? unsupported : gapPalette[Math.round(255*THREE.MathUtils.clamp(gap/10,0,1))];
        }
        colors.set([color.r,color.g,color.b],i*3);
      }
    }
    return colors;
  }
  function preview(frame, options) {
    latestPreview=frame; hide();
    axes.visible=options.axes; grid.visible=options.grid;
    for(const object of [...frustums.children]) {
      object.geometry?.dispose();object.material?.map?.dispose();object.material?.dispose();frustums.remove(object);
    }
    frustums.visible=Boolean(options.frustums);
    if(options.frustums) frame.cameras.forEach((view,i)=>{
      if(!options.visible[i]||!view.frustum) return;
      const origin=new THREE.Vector3(...view.frustum.origin_world_m);
      const corners=view.frustum.rays_world.map(ray=>new THREE.Vector3(...ray).multiplyScalar(options.frustumDepth).add(origin));
      const lines=[];
      for(let j=0;j<4;j++) lines.push(origin.toArray(),corners[j].toArray(),corners[j].toArray(),corners[(j+1)%4].toArray());
      const axis=corners.reduce((sum,p)=>sum.add(p),new THREE.Vector3()).divideScalar(4);
      lines.push(origin.toArray(),axis.toArray());
      frustums.add(new THREE.LineSegments(geometry(lines),new THREE.LineBasicMaterial({color:cameraColors[i%2],transparent:true,opacity:.8,depthTest:false})));
      const marker=new THREE.Mesh(new THREE.SphereGeometry(.006,10,6),new THREE.MeshBasicMaterial({color:cameraColors[i%2],depthTest:false}));
      marker.position.copy(origin);frustums.add(marker);
      const canvas=document.createElement('canvas');canvas.width=256;canvas.height=64;
      const context=canvas.getContext('2d');context.fillStyle='#10252ecc';context.fillRect(0,0,256,64);
      context.fillStyle=cameraColors[i%2];context.font='bold 32px Segoe UI';context.textAlign='center';context.fillText(view.id,128,44);
      const texture=new THREE.CanvasTexture(canvas);texture.colorSpace=THREE.SRGBColorSpace;
      const label=new THREE.Sprite(new THREE.SpriteMaterial({map:texture,depthTest:false}));
      label.position.copy(origin).add(new THREE.Vector3(0,0,-.025));label.scale.set(.09,.0225,1);frustums.add(label);
    });
    let points=0, triangles=0;
    frame.cameras.forEach((original,i)=>{
      if(!options.visible[i]) return;
      let view=original;
      if(frame.full_scene) {
        const indices=[], remap=new Int32Array(original.xyz.length).fill(-1);
        original.xyz.forEach((_,j)=>{
          const region=original.region?.[j]||0, depth=original.depth_m[j];
          const inside=options.scope==='scene'||Boolean(region&(options.scope==='breast'?1:2));
          if(inside&&depth>=options.near&&depth<=options.far) {remap[j]=indices.length;indices.push(j);}
        });
        const triangles=[];
        for(let j=0;j<(original.triangles?.length||0);j+=3) {
          const a=remap[original.triangles[j]],b=remap[original.triangles[j+1]],c=remap[original.triangles[j+2]];
          if(a>=0&&b>=0&&c>=0) triangles.push(a,b,c);
        }
        view={...original,triangles};
        for(const key of ['xyz','colors','depth_m','gap_mm']) view[key]=indices.map(j=>original[key]?.[j]);
      }
      const style=options.view, vertex=['color','color-points','depth','gap'].includes(style);
      const wantsMesh=!['camera','color-points','gap'].includes(style);
      const isMesh=wantsMesh && Boolean(view.triangles?.length);
      const geo=geometry(view.xyz,isMesh?view.triangles:null);
      const range=frame.full_scene?[options.near,options.far]:(frame.depth_range_m||[.3,1]);
      if(vertex) geo.setAttribute('color',new THREE.BufferAttribute(vertexColors(view,style,...range),3));
      const opacity=options.opacity*(style==='transparent' ? .4 : 1);
      const common={color:vertex?'white':cameraColors[i%2],vertexColors:vertex,opacity,
                    transparent:opacity<1,depthWrite:opacity>=1};
      let material;
      if(!isMesh) material=new THREE.PointsMaterial({...common,size:options.pointSize/1000,sizeAttenuation:true});
      else if(style==='shape') {
        geo.computeVertexNormals();
        material=new THREE.MeshStandardMaterial({...common,vertexColors:false,color:'#c4d4d8',roughness:.88,metalness:0,side:THREE.DoubleSide});
      } else if(style==='normals') {
        geo.computeVertexNormals(); material=new THREE.MeshNormalMaterial({side:THREE.DoubleSide,transparent:opacity<1,opacity});
      } else material=new THREE.MeshBasicMaterial({...common,side:THREE.DoubleSide,wireframe:style==='wire'});
      put(`view${i}-${isMesh?'mesh':'points'}`,geo,material,isMesh);
      points+=view.xyz.length; if(isMesh) triangles+=view.triangles.length/3;
    });
    return {points,triangles};
  }
  function tracks(result,t) {
    hide();frustums.visible=false;
    put('track-context',geometry(result.context[t].xyz),new THREE.PointsMaterial({color:'#52747f',size:.0015}));
    const points=[],colors=[];
    result.coords[t].forEach((xyz,i)=>{
      if(!result.visible[t][i]||xyz.some(x=>x===null)) return;
      points.push(xyz);const color=new THREE.Color().setHSL(.52-Math.min(result.displacement_mm[t][i]/20,1)*.5,.9,.6);
      colors.push(color.r,color.g,color.b);
    });
    const geo=geometry(points); geo.setAttribute('color',new THREE.Float32BufferAttribute(colors,3));
    const object=put('tracks',geo,new THREE.PointsMaterial({vertexColors:true,size:.004,depthTest:false}));object.renderOrder=10;
    return points.length;
  }
  new ResizeObserver(()=>{
    const width=host.clientWidth,height=host.clientHeight;
    camera.aspect=width/height;camera.updateProjectionMatrix();renderer.setSize(width,height,false);
  }).observe(host);
  function animate(now) { requestAnimationFrame(animate);controls.update();renderer.render(scene,camera);onAnimation(now); }
  requestAnimationFrame(animate);
  return {preview,tracks,fit,preset};
}
