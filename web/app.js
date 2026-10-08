"use strict";
const $ = id => document.getElementById(id);
const labels = {motorcycle:"Xe máy",car:"Ô tô",truck:"Xe tải",bus:"Xe buýt",other:"Khác / xe đạp"};
const canvas = $("canvas"), ctx = canvas.getContext("2d"), background = new Image();
let zones = [], selected = 0, drawing = false, draft = [], drag = -1, jobId = null, job = null, allEvents = [], seen = new Set(), audio = null, polling = false, sourceVersion = 0, evidenceEvent = null;
const statusLabels = {queued:"Đang chuẩn bị",running:"Đang giám sát",completed:"Đã hoàn tất",stopped:"Đã dừng",error:"Có lỗi"};

function tell(message, error=false) { $("message").textContent=message; $("message").classList.toggle("error",error); $("message").hidden=false; }
async function api(path, options={}) {
  const response = await fetch(path, options);
  if (!response.ok) {
    const body = await response.json().catch(()=>({}));
    const message = Array.isArray(body.detail) ? body.detail.map(e=>`${e.loc.slice(1).join(".")}: ${e.msg}`).join("; ") : body.detail;
    throw new Error(message || `Lỗi HTTP ${response.status}`);
  }
  return response.json();
}
function jsonOptions(method, value) { return {method, headers:{"Content-Type":"application/json"},body:JSON.stringify(value)}; }
function guarded(fn) { return async (...args)=>{try {await fn(...args);} catch(e){tell(e.message,true);}}; }
function element(tag, text, cls) { const e=document.createElement(tag); if(text!==undefined)e.textContent=text; if(cls)e.className=cls; return e; }
function option(value,text) { const e=element("option",text);e.value=value;return e; }
function page(name) {
  for (const section of document.querySelectorAll(".page")) section.hidden=section.id!==name;
  for (const button of document.querySelectorAll(".nav")) button.classList.toggle("active",button.dataset.page===name);
  if(name==="history") guarded(loadHistory)();
}
document.querySelectorAll(".nav").forEach(button=>button.onclick=()=>page(button.dataset.page));
function mapValue() { syncForm(); if(drawing)throw new Error("Hoàn tất hoặc hủy nét vẽ trước khi lưu/chạy."); return {zones:structuredClone(zones)}; }
function syncForm() {
  const z=zones[selected];if(!z)return;
  Object.assign(z,{name:$("zoneName").value,color:$("zoneColor").value,severity:$("severity").value,
    dwell_seconds:Number($("dwell").value),min_frames:Number($("minFrames").value),margin:Number($("margin").value),exit_seconds:Number($("exitTime").value),
    alert_initial:$("alertInitial").checked,classes:[...document.querySelectorAll("#classChoices input:checked")].map(e=>e.value)});
}
function renderZoneList() {
  $("zoneList").replaceChildren();
  zones.forEach((z,i)=>{const b=element("button",undefined,"zone-chip"+(selected===i?" selected":""));
    const dot=element("i");dot.style.background=z.color;b.append(dot,document.createTextNode(z.name||"Vùng chưa đặt tên"));
    b.onclick=()=>{if(drawing)return tell("Hoàn tất hoặc hủy nét vẽ trước.",true);syncForm();selected=i;renderForm();renderCanvas();};$("zoneList").append(b);});
  $("zoneCount").textContent=zones.length;
}
function renderForm() {
  const z=zones[selected]; if(!z)return;
  $("zoneName").value=z.name;$("zoneColor").value=z.color;$("severity").value=z.severity;
  $("dwell").value=z.dwell_seconds;$("minFrames").value=z.min_frames;$("margin").value=z.margin;$("exitTime").value=z.exit_seconds;$("alertInitial").checked=z.alert_initial;
  document.querySelectorAll("#classChoices input").forEach(e=>e.checked=z.classes.includes(e.value));renderZoneList();
}
for (const [key,label] of Object.entries(labels)) {
  const wrapper=element("label"), box=element("input");box.type="checkbox";box.value=key;box.onchange=syncForm;wrapper.append(box,document.createTextNode(label));$("classChoices").append(wrapper);
  $("filterClass").append(option(key,label));
}
for(const id of ["zoneName","zoneColor","severity","dwell","minFrames","margin","exitTime","alertInitial"]) $(id).oninput=()=>{syncForm();renderZoneList();renderCanvas();};
function renderCanvas() {
  ctx.clearRect(0,0,canvas.width,canvas.height);
  if(background.complete&&background.naturalWidth)ctx.drawImage(background,0,0,canvas.width,canvas.height);
  zones.forEach((z,i)=>{if(drawing&&i===selected)return;const pts=z.points;if(!pts.length)return;
    ctx.beginPath();pts.forEach(([x,y],k)=>k?ctx.lineTo(x*canvas.width,y*canvas.height):ctx.moveTo(x*canvas.width,y*canvas.height));ctx.closePath();
    ctx.fillStyle=z.color+"26";ctx.fill();ctx.strokeStyle=z.color;ctx.lineWidth=i===selected?3:1.5;ctx.stroke();
    ctx.font="bold 14px Segoe UI";ctx.fillStyle=z.color;ctx.fillText(z.name,pts[0][0]*canvas.width+8,Math.max(20,pts[0][1]*canvas.height-10));
    if(i===selected)pts.forEach(([x,y])=>{ctx.beginPath();ctx.arc(x*canvas.width,y*canvas.height,5,0,Math.PI*2);ctx.fillStyle="#fff";ctx.fill();ctx.stroke();});
  });
  if(drawing&&draft.length){ctx.beginPath();draft.forEach(([x,y],k)=>k?ctx.lineTo(x*canvas.width,y*canvas.height):ctx.moveTo(x*canvas.width,y*canvas.height));ctx.strokeStyle="#51dab1";ctx.lineWidth=2;ctx.stroke();draft.forEach(([x,y])=>{ctx.beginPath();ctx.arc(x*canvas.width,y*canvas.height,5,0,Math.PI*2);ctx.fillStyle="#51dab1";ctx.fill();});}
}
function showMap() {$("liveFrame").hidden=true;$("resultVideo").hidden=true;$("resultVideo").pause();canvas.hidden=false;renderCanvas();}
function resetSession() {jobId=null;job=null;allEvents=[];seen=new Set();renderEvents();$("alertCount").textContent="0";$("activeCount").textContent="0";$("fps").textContent="—";$("sessionBadge").textContent="CHƯA CHẠY";$("sessionBadge").classList.remove("running");$("progress").value=0;$("phase").textContent="Sẵn sàng phân tích";$("progressText").textContent="Kiểm tra vùng rồi bắt đầu.";$("downloads").hidden=true;$("watchResult").hidden=true;$("stop").disabled=true;}
function drawingState(value) {drawing=value;$("drawingTools").hidden=!value;$("drawHint").textContent=value?"Click thêm đỉnh · Enter hoàn tất · Escape hủy.":"Kéo các đỉnh để chỉnh vùng. Lưu bản đồ sau khi điều chỉnh.";renderCanvas();}
$("drawZone").onclick=()=>{showMap();draft=[];drawingState(true);canvas.focus();};
$("cancelDraw").onclick=()=>{draft=[];drawingState(false);};
$("undoPoint").onclick=()=>{draft.pop();renderCanvas();};
$("finishDraw").onclick=()=>{if(draft.length<3)return tell("Cần ít nhất 3 đỉnh.",true);zones[selected].points=draft.map(p=>p.slice());draft=[];drawingState(false);};
function coordinate(e) {const rect=canvas.getBoundingClientRect();return [Math.max(0,Math.min(1,(e.clientX-rect.left)/rect.width)),Math.max(0,Math.min(1,(e.clientY-rect.top)/rect.height))];}
canvas.onpointerdown=e=>{if(!background.naturalWidth)return;const p=coordinate(e);if(drawing){draft.push(p);renderCanvas();return;}
  drag=zones[selected]?.points.findIndex(q=>Math.hypot((q[0]-p[0])*canvas.clientWidth,(q[1]-p[1])*canvas.clientHeight)<14)??-1;
  if(drag>=0)canvas.setPointerCapture(e.pointerId);};
canvas.onpointermove=e=>{if(drag>=0){zones[selected].points[drag]=coordinate(e);renderCanvas();}};
canvas.onpointerup=()=>drag=-1;canvas.onpointercancel=()=>drag=-1;
canvas.onkeydown=e=>{if(!drawing)return;if(e.key==="Enter"){$("finishDraw").click();e.preventDefault();}if(e.key==="Escape")$("cancelDraw").click();if(e.key==="Backspace"){$("undoPoint").click();e.preventDefault();}};
$("editMap").onclick=showMap;
$("newZone").onclick=()=>{if(drawing)return tell("Hoàn tất nét vẽ trước.",true);if(zones.length>=10)return tell("Tối đa 10 vùng.",true);syncForm();zones.push({id:"zone_"+Date.now().toString(36),name:"Vùng cấm "+(zones.length+1),color:["#ff665e","#f4bf6b","#5ab4ff","#b799ff"][zones.length%4],points:[[.3,.3],[.7,.3],[.7,.8],[.3,.8]],classes:Object.keys(labels),dwell_seconds:.4,min_frames:3,exit_seconds:.6,margin:.006,alert_initial:true,severity:"high"});selected=zones.length-1;renderForm();showMap();};
$("deleteZone").onclick=()=>{if(drawing)return tell("Hủy nét vẽ trước.",true);if(zones.length<=1)return tell("Cần giữ ít nhất một vùng.",true);zones.splice(selected,1);selected=Math.max(0,selected-1);renderForm();renderCanvas();};
$("saveMap").onclick=guarded(async()=>{await api("/api/map?source_id="+encodeURIComponent($("source").value),jsonOptions("PUT",mapValue()));tell("Đã lưu bản đồ cho nguồn đang chọn.");});
$("exportMap").onclick=guarded(()=>{const url=URL.createObjectURL(new Blob([JSON.stringify(mapValue(),null,2)],{type:"application/json"}));const a=element("a");a.href=url;a.download="zone-map.json";a.click();URL.revokeObjectURL(url);});
$("importMap").onchange=guarded(async e=>{const file=e.target.files[0];if(!file)return;if(file.size>100000)throw new Error("Bản đồ JSON tối đa 100 KB.");const value=JSON.parse(await file.text());
  const valid=await api("/api/map?source_id="+encodeURIComponent($("source").value),jsonOptions("PUT",value));zones=valid.zones;selected=0;drawingState(false);renderForm();showMap();tell("Đã nhập và lưu bản đồ.");e.target.value="";});

async function loadSources(chosen) {const value=chosen||$("source").value;const entries=await api("/api/sources");$("source").replaceChildren(...entries.map(e=>option(e.id,e.name)));if(value.startsWith("camera:"))$("source").append(option(value,"Camera USB "+value.split(":")[1]));$("source").value=entries.some(e=>e.id===value)||value.startsWith("camera:")?value:"parking";}
async function loadSource() {
  if(job&&["running","queued"].includes(job.status))throw new Error("Dừng phiên hiện tại trước khi đổi nguồn.");
  resetSession();$("start").disabled=true;
  const current=++sourceVersion, source=$("source").value;$("viewerEmpty").textContent="Đang tải ảnh camera…";$("viewerEmpty").hidden=false;showMap();
  const map=await api("/api/map?source_id="+encodeURIComponent(source));
  // Fetch before image assignment so camera failures have a useful error message.
  const response=await fetch("/api/sources/"+encodeURIComponent(source)+"/preview?t="+Date.now());
  if(!response.ok){const error=await response.json();throw new Error(error.detail||"Không mở được nguồn.");}
  const url=URL.createObjectURL(await response.blob());
  if(current!==sourceVersion){URL.revokeObjectURL(url);return;}
  await new Promise((resolve,reject)=>{background.onload=()=>{canvas.width=background.naturalWidth;canvas.height=background.naturalHeight;URL.revokeObjectURL(url);resolve();};background.onerror=()=>{URL.revokeObjectURL(url);reject(new Error("Không tải được ảnh nguồn."));};background.src=url;});
  if(current!==sourceVersion)return;
  zones=map.zones;selected=0;draft=[];drawingState(false);renderForm();renderCanvas();$("viewerEmpty").hidden=true;$("start").disabled=false;
}
$("source").onchange=guarded(loadSource);
$("upload").onchange=guarded(async e=>{const file=e.target.files[0];if(!file)return;if(file.size>500*1024*1024)throw new Error("Video tối đa 500 MB.");tell("Đang tải và kiểm tra video…");const data=new FormData();data.append("file",file);const source=await api("/api/sources",{method:"POST",body:data});await loadSources(source.id);await loadSource();tell("Video đã sẵn sàng. Hãy chỉnh vùng theo góc quay.");e.target.value="";});
$("cameraButton").onclick=guarded(async()=>{if($("cameraIndexLabel").hidden){$("cameraIndexLabel").hidden=false;tell("Chọn chỉ số camera rồi bấm Camera USB lần nữa.");return;}const index=Number($("cameraIndex").value);if(!Number.isInteger(index)||index<0||index>5)throw new Error("Chỉ số camera từ 0 đến 5.");await loadSources("camera:"+index);await loadSource();});
function initializeAudio() {audio ||= new (window.AudioContext||window.webkitAudioContext)();return audio.resume();}
$("sound").onchange=guarded(async()=>{if($("sound").checked)await initializeAudio();});
function beep() {if(!$("sound").checked||!audio)return;const oscillator=audio.createOscillator(),gain=audio.createGain();oscillator.connect(gain);gain.connect(audio.destination);oscillator.frequency.value=880;gain.gain.setValueAtTime(.08,audio.currentTime);gain.gain.exponentialRampToValueAtTime(.001,audio.currentTime+.3);oscillator.start();oscillator.stop(audio.currentTime+.3);}
$("start").onclick=guarded(async()=>{const data={source_id:$("source").value,zone_map:mapValue(),model_key:$("modelKey").value,confidence:Number($("confidence").value),imgsz:Number($("imgsz").value),device:$("device").value,max_seconds:Number($("maxSeconds").value)};
  if($("sound").checked)await initializeAudio();const created=await api("/api/jobs",jsonOptions("POST",data));jobId=created.id;job=created;seen=new Set();allEvents=[];$("message").hidden=true;$("watchResult").hidden=true;$("downloads").hidden=true;renderJob();renderEvents();await poll();});
$("stop").onclick=guarded(async()=>{await api(`/api/jobs/${jobId}/stop`,{method:"POST"});$("stop").disabled=true;tell("Đang dừng và lưu video/kết quả của phần đã xử lý.");});
function renderJob() {
  if(!job)return;const running=["running","queued"].includes(job.status);
  $("sessionBadge").textContent=statusLabels[job.status]||job.status;$("sessionBadge").classList.toggle("running",running);
  $("alertCount").textContent=job.event_count||0;$("activeCount").textContent=running?job.active_count||0:0;
  const rate=job.processing_fps||(job.processed_frames&&job.elapsed_seconds?job.processed_frames/job.elapsed_seconds:null);$("fps").textContent=rate?rate.toFixed(1):"—";
  $("progress").value=job.progress||0;$("phase").textContent=job.phase==="encoding"?"Đang mã hóa video kết quả":statusLabels[job.status]||job.status;
  $("progressText").textContent=`${job.processed_frames||0} khung · ${(job.video_seconds||0).toFixed(1)} giây nguồn${job.metadata?.live?" · Camera trực tiếp":""}`;
  for(const id of ["start","source","upload","cameraButton"])$(id).disabled=running;$("stop").disabled=!running;
  if(job.status==="error")tell(job.error||"Lỗi phân tích.",true);
  const done=["completed","stopped"].includes(job.status);$("downloads").hidden=!done;$("watchResult").hidden=!done||!job.processed_frames;
  if(done){const root=`/api/jobs/${jobId}/files/`;for(const [id,name] of [["csvLink","events.csv"],["jsonLink","result.json"],["configLink","config.json"],["videoLink","annotated.mp4"]]){$(id).href=root+name;$(id).download=name;}$("videoLink").hidden=!job.processed_frames;}
}
function renderEvents() {
  $("eventBadge").textContent=allEvents.length;const filter=$("filterClass").value;
  const events=allEvents.filter(e=>!filter||e.class===filter).slice().reverse();$("events").replaceChildren();
  if(!events.length){$("events").append(element("div","Chưa có cảnh báo phù hợp. Cảnh báo cần đủ số khung và thời gian xác nhận.","empty"));return;}
  for(const event of events){const card=element("button",undefined,"event"), image=element("img");image.src=`/api/jobs/${jobId}/evidence/${event.id}`;image.alt="Bằng chứng "+event.zone_name;image.loading="lazy";
    const info=element("div"), time=element("span",`${event.confirmed_seconds.toFixed(2)}s · #${event.track_id}`,"time");info.append(time,element("strong",`${labels[event.class]} · ${event.zone_name}`),element("small",event.kind==="entry"?"Vào vùng từ bên ngoài":"Đã có trong vùng khi bắt đầu"),element("small",`${event.severity==="high"?"Vi phạm":"Lưu ý"} · Tin cậy ${(event.confidence*100).toFixed(0)}%`));card.append(image,info);
    card.onclick=()=>{evidenceEvent=event;$("reviewEvidence").disabled=!["completed","stopped"].includes(job?.status);$("evidenceImage").src=image.src;$("evidenceCaption").textContent=`${event.zone_name} · ${labels[event.class]} · track #${event.track_id} · vào vùng ${event.entered_seconds}s, xác nhận ${event.confirmed_seconds}s`;$("evidenceDialog").showModal();};$("events").append(card);}
}
$("filterClass").onchange=renderEvents;$("closeEvidence").onclick=()=>$("evidenceDialog").close();
function watchVideo(seconds=0) {canvas.hidden=true;$("liveFrame").hidden=true;const video=$("resultVideo");video.hidden=false;video.onloadedmetadata=()=>{video.currentTime=Math.min(Math.max(0,seconds),Math.max(0,video.duration-.05));video.play().catch(()=>{});};video.src=`/api/jobs/${jobId}/files/annotated.mp4`;}
$("watchResult").onclick=()=>watchVideo();
$("reviewEvidence").onclick=()=>{$("evidenceDialog").close();watchVideo(evidenceEvent.frame_index/job.metadata.fps-2);$("resultVideo").scrollIntoView({block:"center"});};
async function poll() {
  if(!jobId||polling)return;polling=true;
  try {const selectedJob=jobId;const [current,events]=await Promise.all([api(`/api/jobs/${selectedJob}`),api(`/api/jobs/${selectedJob}/events`)]);if(selectedJob!==jobId)return;
    job=current;const fresh=events.some(e=>!seen.has(e.id));events.forEach(e=>seen.add(e.id));if(fresh)beep();
    if(events.length!==allEvents.length){allEvents=events;renderEvents();}renderJob();
    if(job.processed_frames&&!drawing&&$("resultVideo").hidden&&(["queued","running"].includes(job.status)||!canvas.hidden)){
      if(["queued","running"].includes(job.status)){canvas.hidden=true;$("liveFrame").hidden=false;}
      if(!$("liveFrame").hidden)$("liveFrame").src=`/api/jobs/${jobId}/files/preview.jpg?t=${Date.now()}`;
    }
  } catch(e){tell(e.message,true);} finally{polling=false;}
}
setInterval(()=>{if(job&&["running","queued"].includes(job.status))poll();},1000);
async function loadHistory() {
  const jobs=await api("/api/jobs");$("historyList").replaceChildren();if(!jobs.length){$("historyList").append(element("div","Chưa có phiên giám sát.","empty"));return;}
  for(const session of jobs){const row=element("div",undefined,"history-row"),info=element("div",undefined,"info");info.append(element("strong",session.name),element("small",new Date(session.created).toLocaleString("vi-VN",{timeZone:"Asia/Ho_Chi_Minh"})+" · "+(statusLabels[session.status]||session.status)+" · "+(session.model||session.request.model_key)));
    const open=element("button","Mở phiên","secondary");open.onclick=guarded(async()=>{
      if(job&&["running","queued"].includes(job.status)&&jobId!==session.id)throw new Error("Dừng hoặc chờ phiên hiện tại trước khi mở phiên khác.");
      if(!["queued","running"].includes(session.status)) {job=null;await loadSources(session.request.source_id);await loadSource();}
      jobId=session.id;job=session;seen=new Set();allEvents=await api(`/api/jobs/${jobId}/events`);allEvents.forEach(e=>seen.add(e.id));
      zones=structuredClone(session.request.zone_map.zones);selected=0;renderForm();
      for(const [id,key] of [["modelKey","model_key"],["confidence","confidence"],["imgsz","imgsz"],["device","device"],["maxSeconds","max_seconds"]])$(id).value=session.request[key];
      renderJob();renderEvents();page("workspace");canvas.hidden=true;$("resultVideo").hidden=true;$("liveFrame").hidden=false;
      $("liveFrame").src=`/api/jobs/${jobId}/files/preview.jpg?t=${Date.now()}`;$("drawHint").textContent="Đang xem cấu hình và kết quả của phiên đã lưu.";
    });
    const remove=element("button","Xóa","danger-text");remove.onclick=guarded(async()=>{if(!confirm("Xóa phiên và toàn bộ bằng chứng của phiên này?"))return;await api(`/api/jobs/${session.id}`,{method:"DELETE"});if(jobId===session.id){jobId=null;job=null;allEvents=[];renderEvents();await loadSource();}await loadHistory();});
    row.append(info,element("span",`${session.event_count||0} cảnh báo`,"history-count"),open,remove);$("historyList").append(row);}
}
$("refreshHistory").onclick=guarded(loadHistory);
async function trainingStatus() {
  const health=await api("/api/health"),training=health.training;
  $("hardware").textContent=training&&["verifying","waiting_gpu","waiting_memory","preparing_train","training","evaluating"].includes(training.status)?"CPU giám sát · GPU dành cho train":health.device_name;
  if(health.available_models?.includes("large")&&!document.querySelector('#modelKey option[value="large"]'))$("modelKey").prepend(option("large","YOLO11n · Camera + COCO (đã nghiệm thu)"));
  if(!training){$("trainingStatus").textContent="Chưa khởi chạy. Quy trình GPU tại máy: scripts/prepare_compact.py → start-training.ps1. Checkpoint đang dùng vẫn là bản VisDrone đã được kiểm chứng.";return health;}
  const names={verifying:"Đang kiểm tra checksum dữ liệu",waiting_gpu:"Đang chờ GPU trống ít nhất 2,5 GB VRAM",waiting_memory:"Đang chờ RAM và bộ nhớ cấp phát trống ít nhất 2 GB",preparing_train:"Đang khởi tạo mô hình và đọc nhãn train",training:"Đang huấn luyện trên GPU tại máy",evaluating:"Đang đánh giá mô hình",completed:"Huấn luyện và đánh giá đã hoàn tất",error:"Phiên huấn luyện gặp lỗi"};
  const detail=training.status==="verifying"?` · ${training.verified_images||0}/${training.dataset_images||0} ảnh đã kiểm`:training.status==="training"?` · epoch ${training.current_epoch||training.epoch||0}/${training.epochs||0} · batch ${training.batch||0}/${training.batches||0}`:training.status==="evaluating"?` · ${training.evaluation_source||"baseline"}/${training.evaluation_split||"val"} · batch ${training.batch||0}/${training.batches||0}`:training.status==="waiting_memory"?` · ${training.memory_headroom_mb||0} MB trống`:"";
  $("trainingStatus").textContent=`${names[training.status]||training.status}${detail} · ${training.train_images||0} ảnh train · ${training.device||"GPU"}${training.status==="completed"?(training.accepted?" · Mô hình mới đã qua tiêu chí nghiệm thu.":" · Giữ mô hình cũ vì ứng viên chưa đạt tiêu chí."):""}${training.error?" · "+training.error:""}`;
  return health;
}
setInterval(()=>guarded(trainingStatus)(),15000);
guarded(async()=>{const health=await trainingStatus();if(health.recommended_model)$("modelKey").value=health.recommended_model;if(health.training&&["verifying","waiting_gpu","waiting_memory","preparing_train","training","evaluating"].includes(health.training.status))$("device").value="cpu";await loadSources();await loadSource();if(health.active_job){jobId=health.active_job;job=await api(`/api/jobs/${jobId}`);allEvents=await api(`/api/jobs/${jobId}/events`);allEvents.forEach(e=>seen.add(e.id));renderEvents();await poll();}})();
