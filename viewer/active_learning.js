(function () {
  "use strict";

  const live = window.liveReconState || {};
  const cams = live.cams || ["cam0", "cam1", "cam2"];
  const defaultBase = live.base || "../runs/recon3d_live/";
  const mount = document.getElementById("activeLearningMount");
  if (!mount) return;

  const state = {
    cam: cams[0],
    tool: "add",
    brush: 24,
    image: null,
    maskCanvas: document.createElement("canvas"),
    tintCanvas: document.createElement("canvas"),
    boxes: [],
    undo: [],
    drawing: false,
    boxStart: null,
    draftBox: null,
    frozen: null,
  };

  mount.innerHTML = `
    <section class="al-card">
      <div class="al-head">
        <div><b>Active Learning</b><span id="alSummary">loading</span></div>
        <span class="al-badge" id="alState">ready</span>
      </div>
      <div class="al-tabs" id="alCamTabs"></div>
      <div class="al-actions">
        <button class="primary" id="alAnnotate">Annotate</button>
        <button class="warn" id="alFineTune">Fine-tune</button>
      </div>
      <div class="al-status" id="alStatus">dataset idle</div>
    </section>
  `;

  document.body.insertAdjacentHTML("beforeend", `
    <div class="al-modal" id="alModal" hidden>
      <div class="al-shell">
        <div class="al-top">
          <div class="al-title">
            <b>Annotation Studio</b>
            <span id="alModalMeta">no frame</span>
          </div>
          <button class="al-close" id="alClose">X</button>
        </div>
        <div class="al-work">
          <div class="al-canvas-wrap"><canvas id="alCanvas"></canvas></div>
          <aside class="al-tools">
            <div class="al-tool-grid">
              <button id="alToolAdd" class="active">Add</button>
              <button id="alToolErase">Erase</button>
              <button id="alToolBox">Box</button>
              <button id="alUndo">Undo</button>
            </div>
            <div class="al-field">
              <label for="alBrush">Brush</label>
              <input id="alBrush" type="range" min="4" max="90" step="2" value="24">
            </div>
            <div class="al-tool-grid">
              <button id="alReset">Prediction</button>
              <button id="alClear" class="al-danger">Clear</button>
              <button id="alDeleteBox">Delete box</button>
              <button id="alInvert">Invert</button>
            </div>
            <div class="al-stat-line">
              <span id="alMaskStat">mask 0 px</span>
              <span id="alBoxStat">boxes 0</span>
            </div>
            <button class="al-save" id="alSave">Save Label</button>
            <div class="al-muted" id="alEditorStatus">ready</div>
          </aside>
        </div>
      </div>
    </div>
  `);

  const ui = {
    tabs: document.getElementById("alCamTabs"),
    annotate: document.getElementById("alAnnotate"),
    fineTune: document.getElementById("alFineTune"),
    summary: document.getElementById("alSummary"),
    state: document.getElementById("alState"),
    status: document.getElementById("alStatus"),
    modal: document.getElementById("alModal"),
    modalMeta: document.getElementById("alModalMeta"),
    close: document.getElementById("alClose"),
    canvas: document.getElementById("alCanvas"),
    brush: document.getElementById("alBrush"),
    toolAdd: document.getElementById("alToolAdd"),
    toolErase: document.getElementById("alToolErase"),
    toolBox: document.getElementById("alToolBox"),
    undo: document.getElementById("alUndo"),
    reset: document.getElementById("alReset"),
    clear: document.getElementById("alClear"),
    deleteBox: document.getElementById("alDeleteBox"),
    invert: document.getElementById("alInvert"),
    save: document.getElementById("alSave"),
    maskStat: document.getElementById("alMaskStat"),
    boxStat: document.getElementById("alBoxStat"),
    editorStatus: document.getElementById("alEditorStatus"),
  };

  function setStatus(text) {
    ui.status.textContent = text;
  }

  function setEditorStatus(text) {
    ui.editorStatus.textContent = text;
  }

  function currentManifest() {
    return (window.liveReconState && window.liveReconState.manifest) || null;
  }

  function currentBase() {
    const liveBase = window.liveReconState && window.liveReconState.base;
    const value = String(liveBase || defaultBase).replace(/\\/g, "/").replace(/\/+$/, "");
    return value + "/";
  }

  function cloneManifest(man) {
    try {
      return JSON.parse(JSON.stringify(man));
    } catch (err) {
      return man;
    }
  }

  function bust(man) {
    return "?t=" + Math.round((man.timestamp || Date.now() / 1000) * 1000);
  }

  function loadImage(src) {
    return new Promise((resolve, reject) => {
      const image = new Image();
      image.onload = () => resolve(image);
      image.onerror = () => reject(new Error("failed to load " + src));
      image.src = src;
    });
  }

  function buildTabs() {
    ui.tabs.innerHTML = "";
    for (const cam of cams) {
      const button = document.createElement("button");
      button.textContent = cam;
      button.className = cam === state.cam ? "active" : "";
      button.onclick = () => {
        state.cam = cam;
        buildTabs();
      };
      ui.tabs.append(button);
    }
  }

  async function refreshSummary() {
    try {
      const res = await fetch("/api/active-learning/summary", { cache: "no-store" });
      const data = await res.json();
      ui.summary.textContent = `${data.samples} samples · ${data.boxes} boxes`;
    } catch (err) {
      ui.summary.textContent = "server offline";
    }
  }

  function setTool(tool) {
    state.tool = tool;
    for (const [name, button] of [["add", ui.toolAdd], ["erase", ui.toolErase], ["box", ui.toolBox]]) {
      button.classList.toggle("active", name === tool);
    }
  }

  function editorPoint(event) {
    const rect = ui.canvas.getBoundingClientRect();
    return {
      x: (event.clientX - rect.left) * (ui.canvas.width / rect.width),
      y: (event.clientY - rect.top) * (ui.canvas.height / rect.height),
    };
  }

  function pushUndo() {
    const ctx = state.maskCanvas.getContext("2d");
    state.undo.push({
      mask: ctx.getImageData(0, 0, state.maskCanvas.width, state.maskCanvas.height),
      boxes: state.boxes.map((box) => box.slice()),
    });
    if (state.undo.length > 18) state.undo.shift();
  }

  function undo() {
    const last = state.undo.pop();
    if (!last) return;
    state.maskCanvas.getContext("2d").putImageData(last.mask, 0, 0);
    state.boxes = last.boxes;
    renderEditor();
  }

  function drawBrush(point) {
    const ctx = state.maskCanvas.getContext("2d");
    ctx.save();
    ctx.beginPath();
    ctx.arc(point.x, point.y, state.brush, 0, Math.PI * 2);
    if (state.tool === "erase") {
      ctx.globalCompositeOperation = "destination-out";
      ctx.fillStyle = "rgba(0,0,0,1)";
    } else {
      ctx.globalCompositeOperation = "source-over";
      ctx.fillStyle = "rgba(255,255,255,1)";
    }
    ctx.fill();
    ctx.restore();
  }

  function normalizeBox(a, b) {
    const x1 = Math.max(0, Math.min(ui.canvas.width - 1, Math.min(a.x, b.x)));
    const x2 = Math.max(0, Math.min(ui.canvas.width - 1, Math.max(a.x, b.x)));
    const y1 = Math.max(0, Math.min(ui.canvas.height - 1, Math.min(a.y, b.y)));
    const y2 = Math.max(0, Math.min(ui.canvas.height - 1, Math.max(a.y, b.y)));
    return [x1, y1, x2, y2];
  }

  function prepareTint() {
    const tint = state.tintCanvas;
    tint.width = state.maskCanvas.width;
    tint.height = state.maskCanvas.height;
    const tctx = tint.getContext("2d");
    tctx.clearRect(0, 0, tint.width, tint.height);
    tctx.fillStyle = "#21d49b";
    tctx.fillRect(0, 0, tint.width, tint.height);
    tctx.globalCompositeOperation = "destination-in";
    tctx.drawImage(state.maskCanvas, 0, 0);
    tctx.globalCompositeOperation = "source-over";
    return tint;
  }

  function maskPixels() {
    const ctx = state.maskCanvas.getContext("2d");
    const data = ctx.getImageData(0, 0, state.maskCanvas.width, state.maskCanvas.height).data;
    let count = 0;
    for (let i = 3; i < data.length; i += 4) {
      if (data[i] > 127) count += 1;
    }
    return count;
  }

  function renderEditor() {
    if (!state.image) return;
    const ctx = ui.canvas.getContext("2d");
    ctx.clearRect(0, 0, ui.canvas.width, ui.canvas.height);
    ctx.drawImage(state.image, 0, 0, ui.canvas.width, ui.canvas.height);
    ctx.save();
    ctx.globalAlpha = 0.46;
    ctx.drawImage(prepareTint(), 0, 0);
    ctx.restore();

    ctx.lineWidth = 3;
    ctx.strokeStyle = "#27f2b4";
    ctx.shadowColor = "rgba(39,242,180,.55)";
    ctx.shadowBlur = 8;
    ctx.strokeRect(0.5, 0.5, ui.canvas.width - 1, ui.canvas.height - 1);
    ctx.shadowBlur = 0;

    const allBoxes = state.draftBox ? state.boxes.concat([state.draftBox]) : state.boxes;
    for (const box of allBoxes) {
      const [x1, y1, x2, y2] = box;
      ctx.strokeStyle = "#ff6464";
      ctx.lineWidth = 3;
      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
      ctx.fillStyle = "rgba(255,100,100,.18)";
      ctx.fillRect(x1, y1, x2 - x1, y2 - y1);
    }
    ui.maskStat.textContent = `mask ${maskPixels().toLocaleString()} px`;
    ui.boxStat.textContent = `boxes ${state.boxes.length}`;
  }

  async function loadPredictionMask(src, width, height) {
    const maskImage = await loadImage(src);
    state.maskCanvas.width = width;
    state.maskCanvas.height = height;
    const ctx = state.maskCanvas.getContext("2d");
    ctx.clearRect(0, 0, width, height);
    ctx.drawImage(maskImage, 0, 0, width, height);
    const imageData = ctx.getImageData(0, 0, width, height);
    const data = imageData.data;
    for (let i = 0; i < data.length; i += 4) {
      const on = data[i] > 127 || data[i + 1] > 127 || data[i + 2] > 127;
      data[i] = 255;
      data[i + 1] = 255;
      data[i + 2] = 255;
      data[i + 3] = on ? 255 : 0;
    }
    ctx.putImageData(imageData, 0, 0);
  }

  function initialBoxes(man, cam, imageWidth, imageHeight) {
    const raw = ((man.perception || {})[cam] || {}).nipple_boxes_xyxy || [];
    const display = (man.display || {})[cam] || {};
    const sx = imageWidth / (display.source_width || imageWidth);
    const sy = imageHeight / (display.source_height || imageHeight);
    return raw.map((box) => [box[0] * sx, box[1] * sy, box[2] * sx, box[3] * sy]);
  }

  async function openEditor() {
    const man = cloneManifest(currentManifest());
    if (!man || !man.rgb || !man.rgb[state.cam]) {
      setStatus("no live frame");
      return;
    }
    const frozenBase = currentBase();
    const frozenCam = state.cam;
    const token = bust(man);
    setStatus("freezing frame");
    state.image = await loadImage(frozenBase + man.rgb[frozenCam] + token);
    ui.canvas.width = state.image.naturalWidth || state.image.width;
    ui.canvas.height = state.image.naturalHeight || state.image.height;
    const maskPath = man.mask && man.mask[frozenCam];
    if (maskPath) {
      await loadPredictionMask(frozenBase + maskPath + token, ui.canvas.width, ui.canvas.height);
    } else {
      state.maskCanvas.width = ui.canvas.width;
      state.maskCanvas.height = ui.canvas.height;
      state.maskCanvas.getContext("2d").clearRect(0, 0, ui.canvas.width, ui.canvas.height);
    }
    state.boxes = initialBoxes(man, frozenCam, ui.canvas.width, ui.canvas.height);
    state.undo = [];
    state.draftBox = null;
    state.frozen = {
      base: frozenBase,
      cam: frozenCam,
      manifest: man,
      frame: man.frame || 0,
      timestamp: man.timestamp || Date.now() / 1000,
    };
    ui.modalMeta.textContent = `${frozenCam} · frame ${man.frame}`;
    ui.modal.hidden = false;
    setTool("add");
    setEditorStatus("ready");
    renderEditor();
    setStatus("frame frozen");
  }

  function maskDataUrl() {
    const out = document.createElement("canvas");
    out.width = state.maskCanvas.width;
    out.height = state.maskCanvas.height;
    const octx = out.getContext("2d");
    const src = state.maskCanvas.getContext("2d").getImageData(0, 0, out.width, out.height);
    const data = src.data;
    for (let i = 0; i < data.length; i += 4) {
      const on = data[i + 3] > 127;
      data[i] = on ? 255 : 0;
      data[i + 1] = on ? 255 : 0;
      data[i + 2] = on ? 255 : 0;
      data[i + 3] = 255;
    }
    octx.putImageData(src, 0, 0);
    return out.toDataURL("image/png");
  }

  function imageDataUrl() {
    const out = document.createElement("canvas");
    out.width = ui.canvas.width;
    out.height = ui.canvas.height;
    out.getContext("2d").drawImage(state.image, 0, 0, out.width, out.height);
    return out.toDataURL("image/png");
  }

  async function saveLabel() {
    if (!state.image) return;
    const frozen = state.frozen || {};
    const man = frozen.manifest || currentManifest() || {};
    const cam = frozen.cam || state.cam;
    setEditorStatus("saving");
    const payload = {
      camera_id: cam,
      frame: frozen.frame || man.frame || 0,
      timestamp: frozen.timestamp || man.timestamp || Date.now() / 1000,
      image_png: imageDataUrl(),
      mask_png: maskDataUrl(),
      nipple_boxes_xyxy: state.boxes,
      source_manifest: {
        frame: frozen.frame || man.frame,
        timestamp: frozen.timestamp || man.timestamp,
        perception: man.perception && man.perception[cam],
        display: man.display && man.display[cam],
      },
    };
    const res = await fetch("/api/active-learning/annotation", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    if (!res.ok || !data.ok) throw new Error(data.error || "save failed");
    setEditorStatus(`saved ${data.file_name}`);
    setStatus(`${data.samples} samples · ${data.boxes} boxes`);
    await refreshSummary();
  }

  async function fineTune() {
    ui.state.textContent = "training";
    setStatus("launching fine-tune");
    const res = await fetch("/api/active-learning/fine-tune", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ min_samples: 1 }),
    });
    const data = await res.json();
    if (!data.ok) {
      ui.state.textContent = "ready";
      setStatus(data.message || data.error || "not started");
      return;
    }
    setStatus(`training pid ${data.pid}`);
  }

  ui.canvas.addEventListener("pointerdown", (event) => {
    if (!state.image) return;
    const point = editorPoint(event);
    ui.canvas.setPointerCapture(event.pointerId);
    pushUndo();
    state.drawing = true;
    if (state.tool === "box") {
      state.boxStart = point;
      state.draftBox = normalizeBox(point, point);
    } else {
      drawBrush(point);
    }
    renderEditor();
  });

  ui.canvas.addEventListener("pointermove", (event) => {
    if (!state.drawing) return;
    const point = editorPoint(event);
    if (state.tool === "box") {
      state.draftBox = normalizeBox(state.boxStart, point);
    } else {
      drawBrush(point);
    }
    renderEditor();
  });

  ui.canvas.addEventListener("pointerup", (event) => {
    if (!state.drawing) return;
    state.drawing = false;
    if (state.tool === "box" && state.draftBox) {
      const [x1, y1, x2, y2] = state.draftBox;
      if (x2 - x1 > 4 && y2 - y1 > 4) state.boxes.push(state.draftBox);
    }
    state.draftBox = null;
    renderEditor();
  });

  ui.annotate.onclick = () => openEditor().catch((err) => setStatus(err.message));
  ui.close.onclick = () => {
    ui.modal.hidden = true;
    state.image = null;
    state.frozen = null;
  };
  ui.toolAdd.onclick = () => setTool("add");
  ui.toolErase.onclick = () => setTool("erase");
  ui.toolBox.onclick = () => setTool("box");
  ui.undo.onclick = undo;
  ui.brush.oninput = () => {
    state.brush = Number(ui.brush.value);
  };
  ui.reset.onclick = () => openEditor().catch((err) => setEditorStatus(err.message));
  ui.clear.onclick = () => {
    pushUndo();
    state.maskCanvas.getContext("2d").clearRect(0, 0, state.maskCanvas.width, state.maskCanvas.height);
    renderEditor();
  };
  ui.deleteBox.onclick = () => {
    pushUndo();
    state.boxes.pop();
    renderEditor();
  };
  ui.invert.onclick = () => {
    pushUndo();
    const ctx = state.maskCanvas.getContext("2d");
    const imageData = ctx.getImageData(0, 0, state.maskCanvas.width, state.maskCanvas.height);
    const data = imageData.data;
    for (let i = 3; i < data.length; i += 4) data[i] = data[i] > 127 ? 0 : 255;
    ctx.putImageData(imageData, 0, 0);
    renderEditor();
  };
  ui.save.onclick = () => saveLabel().catch((err) => setEditorStatus(err.message));
  ui.fineTune.onclick = () => fineTune().catch((err) => {
    ui.state.textContent = "ready";
    setStatus(err.message);
  });

  buildTabs();
  refreshSummary();
  setInterval(refreshSummary, 8000);
})();
