const $ = (id) => document.getElementById(id);
const input = $("imageInput"), drop = $("dropArea"), preview = $("imagePreview"), button = $("classifyButton");
const key = new URLSearchParams(location.search).get("key") || "";
let file = null, objectUrl = null, noticeTimer = null, readCount = 0;
const historyItems = [];

function notice(text) {
  const box = $("notice"); box.textContent = text; box.classList.add("show");
  clearTimeout(noticeTimer); noticeTimer = setTimeout(() => box.classList.remove("show"), 4200);
}
function clearImage() {
  if (objectUrl) URL.revokeObjectURL(objectUrl);
  objectUrl = null; file = null; input.value = ""; preview.removeAttribute("src"); preview.hidden = true;
  $("previewOverlay").hidden = true; $("dropEmpty").hidden = false; $("fileStatus").textContent = "NO IMAGE SELECTED";
  button.disabled = true; $("emptyResult").hidden = false; $("predictionResult").hidden = true; $("loadingResult").hidden = true;
}
function chooseImage(next) {
  if (!next) return;
  if (!next.type.startsWith("image/")) return notice("Choose an image file to continue.");
  if (next.size > 8 * 1024 * 1024) return notice("Choose an image smaller than 8 MB.");
  if (objectUrl) URL.revokeObjectURL(objectUrl);
  file = next; objectUrl = URL.createObjectURL(next); preview.src = objectUrl; preview.hidden = false;
  $("previewOverlay").hidden = false; $("dropEmpty").hidden = true; $("previewName").textContent = next.name;
  $("fileStatus").textContent = `READY / ${next.name}`.toUpperCase(); button.disabled = false;
  $("emptyResult").hidden = false; $("predictionResult").hidden = true; $("loadingResult").hidden = true;
}
function addHistory(label, confidence, filename) {
  historyItems.unshift({label, confidence, filename, time: new Date().toLocaleTimeString([], {hour:"2-digit",minute:"2-digit"})});
  historyItems.splice(5); const list = $("historyList"); list.replaceChildren();
  for (const item of historyItems) {
    const row = document.createElement("div"); row.className = "history-row";
    const name = document.createElement("span"); name.textContent = `${item.label} · ${item.filename}`;
    const score = document.createElement("span"); score.textContent = `${(item.confidence * 100).toFixed(1)}%`;
    const time = document.createElement("time"); time.textContent = item.time; row.append(name, score, time); list.append(row);
  }
  $("clearHistory").hidden = !historyItems.length;
}
async function loadStatus() {
  try {
    const response = await fetch("/api/status", {cache:"no-store"}); if (!response.ok) throw new Error();
    const status = await response.json(); $("stateDot").classList.add("ready"); $("stateText").textContent = "MODEL READY";
    $("categoryTotal").textContent = `${status.categoryCount} CATEGORIES`;
    $("categoryCount").textContent = String(status.categoryCount).padStart(2,"0");
    const list = $("categoryList"); list.replaceChildren();
    status.categories.forEach((name) => { const chip = document.createElement("span"); chip.className = "category-chip"; chip.textContent = name; list.append(chip); });
  } catch (_) {
    $("stateDot").classList.add("error"); $("stateText").textContent = "MODEL OFFLINE";
    $("categoryList").textContent = "Restart the image desk and open its current session link.";
  }
}

input.addEventListener("change", () => chooseImage(input.files && input.files[0]));
$("removeImage").addEventListener("click", (event) => {event.preventDefault(); event.stopPropagation(); clearImage();});
drop.addEventListener("keydown", (event) => {if ((event.key === "Enter" || event.key === " ") && event.target === drop) {event.preventDefault(); input.click();}});
["dragenter","dragover"].forEach((name) => drop.addEventListener(name, (event) => {event.preventDefault(); drop.classList.add("is-over");}));
["dragleave","drop"].forEach((name) => drop.addEventListener(name, (event) => {event.preventDefault(); drop.classList.remove("is-over");}));
drop.addEventListener("drop", (event) => chooseImage(event.dataTransfer && event.dataTransfer.files[0]));
$("clearHistory").addEventListener("click", () => {historyItems.length = 0; $("historyList").innerHTML = '<p class="history-empty">Recent reads stay only in this browser session.</p>'; $("clearHistory").hidden = true;});

button.addEventListener("click", async () => {
  if (!file) return;
  if (!key) return notice("Open the complete session link printed in the launcher window.");
  const chosen = file, data = new FormData(); data.append("image", chosen);
  button.disabled = true; $("emptyResult").hidden = true; $("predictionResult").hidden = true; $("loadingResult").hidden = false;
  try {
    const response = await fetch("/api/predict", {method:"POST", headers:{"X-Access-Key":key}, body:data});
    const result = await response.json(); if (!response.ok) throw new Error(result.error || "The image could not be classified.");
    $("predictionLabel").textContent = result.label;
    const value = Math.max(0,Math.min(100,result.confidence*100)); $("confidenceValue").textContent = `${value.toFixed(1)}%`; $("confidenceBar").style.width = `${value}%`;
    const matches = $("matchList"); matches.replaceChildren();
    result.matches.slice(1).forEach((match) => {const row=document.createElement("li"),name=document.createElement("span"),score=document.createElement("span");name.textContent=match.label;score.className="match-confidence";score.textContent=`${(match.confidence*100).toFixed(1)}%`;row.append(name,score);matches.append(row);});
    readCount++; $("readNumber").textContent = `READ ${String(readCount).padStart(2,"0")}`;
    $("loadingResult").hidden = true; $("predictionResult").hidden = false; addHistory(result.label,result.confidence,chosen.name);
    $("fileStatus").textContent = `READ ${String(readCount).padStart(2,"0")} / ${chosen.name}`.toUpperCase();
  } catch (error) {
    $("loadingResult").hidden = true; $("emptyResult").hidden = false; notice(error.message || "Try another image.");
  } finally {button.disabled = !file;}
});
window.addEventListener("beforeunload", () => {if (objectUrl) URL.revokeObjectURL(objectUrl);});
loadStatus();
