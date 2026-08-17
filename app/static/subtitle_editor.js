(() => {
  const stage = document.querySelector("#subtitle-stage");
  if (!stage) return;

  const video = document.querySelector("#subtitle-video");
  const overlay = document.querySelector("#subtitle-overlay");
  const subtitleText = document.querySelector("#subtitle-text");
  const resizeHandle = document.querySelector("#subtitle-resize");
  const fontFamily = document.querySelector("#font-family");
  const fontSize = document.querySelector("#font-size");
  const fontSizeOutput = document.querySelector("#font-size-output");
  const textColor = document.querySelector("#text-color");
  const backgroundColor = document.querySelector("#background-color");
  const saveStatus = document.querySelector("#save-status");
  const cues = JSON.parse(document.querySelector("#subtitle-cues").textContent);

  const state = {
    x_ratio: Number(overlay.dataset.x),
    y_ratio: Number(overlay.dataset.y),
    width_ratio: Number(overlay.dataset.width),
    height_ratio: Number(overlay.dataset.height),
    font_family: overlay.dataset.fontFamily,
    font_size_ratio: Number(overlay.dataset.fontSize),
    text_color: overlay.dataset.textColor,
    background_color: overlay.dataset.backgroundColor,
  };

  const clamp = (value, minimum, maximum) => Math.min(maximum, Math.max(minimum, value));
  const rounded = (value) => Math.round(value * 10000) / 10000;

  function applyStyle() {
    overlay.style.left = `${state.x_ratio * 100}%`;
    overlay.style.top = `${state.y_ratio * 100}%`;
    overlay.style.width = `${state.width_ratio * 100}%`;
    overlay.style.height = `${state.height_ratio * 100}%`;
    overlay.style.fontFamily = state.font_family;
    overlay.style.fontSize = `${stage.clientHeight * state.font_size_ratio}px`;
    overlay.style.color = state.text_color;
    overlay.style.backgroundColor = state.background_color;
    fontSizeOutput.value = `${(state.font_size_ratio * 100).toFixed(1)}% chiều cao`;
    document.querySelector("#value-x").textContent = state.x_ratio.toFixed(4);
    document.querySelector("#value-y").textContent = state.y_ratio.toFixed(4);
    document.querySelector("#value-width").textContent = state.width_ratio.toFixed(4);
    document.querySelector("#value-height").textContent = state.height_ratio.toFixed(4);
  }

  let saveTimer;
  function scheduleSave() {
    saveStatus.textContent = "Đang lưu…";
    clearTimeout(saveTimer);
    saveTimer = setTimeout(saveStyle, 250);
  }

  async function saveStyle() {
    try {
      const response = await fetch(stage.dataset.styleUrl, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(state),
      });
      if (!response.ok) throw new Error("save failed");
      saveStatus.textContent = "Đã lưu";
      return true;
    } catch (_) {
      saveStatus.textContent = "Lưu thất bại — thử thay đổi lại";
      return false;
    }
  }

  function syncSubtitle() {
    const currentMs = video.currentTime * 1000;
    const cue = cues.find((item) => currentMs >= item.start_ms && currentMs < item.end_ms);
    if (cue) {
      subtitleText.textContent = cue.text;
      overlay.classList.add("is-visible");
    } else {
      subtitleText.textContent = "";
      overlay.classList.remove("is-visible");
    }
  }

  function beginPointer(event, mode) {
    event.preventDefault();
    event.stopPropagation();
    const startX = event.clientX;
    const startY = event.clientY;
    const initial = { ...state };
    const target = mode === "resize" ? resizeHandle : overlay;
    target.setPointerCapture(event.pointerId);

    const move = (moveEvent) => {
      const dx = (moveEvent.clientX - startX) / stage.clientWidth;
      const dy = (moveEvent.clientY - startY) / stage.clientHeight;
      if (mode === "move") {
        state.x_ratio = rounded(clamp(initial.x_ratio + dx, 0, 1 - state.width_ratio));
        state.y_ratio = rounded(clamp(initial.y_ratio + dy, 0, 1 - state.height_ratio));
      } else {
        state.width_ratio = rounded(clamp(initial.width_ratio + dx, 0.1, 1 - state.x_ratio));
        state.height_ratio = rounded(clamp(initial.height_ratio + dy, 0.05, 1 - state.y_ratio));
      }
      applyStyle();
    };
    const finish = () => {
      target.removeEventListener("pointermove", move);
      target.removeEventListener("pointerup", finish);
      target.removeEventListener("pointercancel", finish);
      scheduleSave();
    };
    target.addEventListener("pointermove", move);
    target.addEventListener("pointerup", finish);
    target.addEventListener("pointercancel", finish);
  }

  overlay.addEventListener("pointerdown", (event) => beginPointer(event, "move"));
  resizeHandle.addEventListener("pointerdown", (event) => beginPointer(event, "resize"));
  video.addEventListener("timeupdate", syncSubtitle);
  video.addEventListener("seeked", syncSubtitle);
  window.addEventListener("resize", applyStyle);

  fontFamily.addEventListener("change", () => {
    state.font_family = fontFamily.value;
    applyStyle();
    scheduleSave();
  });
  fontSize.addEventListener("input", () => {
    state.font_size_ratio = Number(fontSize.value);
    applyStyle();
    scheduleSave();
  });
  textColor.addEventListener("input", () => {
    state.text_color = textColor.value;
    applyStyle();
    scheduleSave();
  });
  backgroundColor.addEventListener("input", () => {
    state.background_color = backgroundColor.value;
    applyStyle();
    scheduleSave();
  });

  const currentTimeButton = document.querySelector("#use-current-time");
  const previewStart = document.querySelector("#preview-start");
  currentTimeButton.addEventListener("click", () => {
    previewStart.value = Math.max(0, video.currentTime).toFixed(1);
  });

  const referenceTimeButton = document.querySelector("#use-current-reference-time");
  const referenceStart = document.querySelector("#reference-start");
  referenceTimeButton.addEventListener("click", () => {
    referenceStart.value = Math.max(0, video.currentTime).toFixed(1);
  });

  const ttsGenerateForm = document.querySelector(".tts-generate-form");
  ttsGenerateForm.addEventListener("submit", (event) => {
    if (!window.confirm(ttsGenerateForm.dataset.confirm)) {
      event.preventDefault();
      return;
    }
    const button = ttsGenerateForm.querySelector("button");
    button.disabled = true;
    button.textContent = "Đang tạo TTS…";
  });

  document.querySelectorAll(".render-form").forEach((form) => {
    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      if (form.dataset.confirm && !window.confirm(form.dataset.confirm)) return;
      clearTimeout(saveTimer);
      const saved = await saveStyle();
      if (!saved) return;
      form.querySelectorAll("button").forEach((button) => { button.disabled = true; });
      const submitButton = form.querySelector('button[type="submit"]');
      submitButton.textContent = "Đang render…";
      form.submit();
    });
  });

  const autoBuildForm = document.querySelector(".auto-build-form");
  autoBuildForm?.addEventListener("submit", async (event) => {
    event.preventDefault();
    clearTimeout(saveTimer);
    if (!(await saveStyle())) return;
    const button = autoBuildForm.querySelector("button");
    button.disabled = true;
    button.textContent = "Đang xếp job…";
    autoBuildForm.submit();
  });

  const finalSection = document.querySelector("#final-output");
  const finalJobId = finalSection.dataset.jobId;
  const finalJobStatus = finalSection.dataset.jobStatus;
  if (finalJobId && ["queued", "running"].includes(finalJobStatus)) {
    const progressElement = document.querySelector("#job-progress");
    const percentElement = document.querySelector("#job-percent");
    const messageElement = document.querySelector("#job-message");
    const errorElement = document.querySelector("#job-error");
    const poll = async () => {
      try {
        const response = await fetch(`/jobs/${finalJobId}`, { cache: "no-store" });
        if (!response.ok) throw new Error("poll failed");
        const job = await response.json();
        progressElement.value = job.progress;
        percentElement.textContent = `${job.progress}%`;
        messageElement.textContent = job.message;
        errorElement.textContent = job.error || "";
        if (job.status === "completed") {
          window.location.reload();
          return;
        }
        if (job.status === "failed") {
          finalSection.classList.add("job-failed");
          window.location.reload();
          return;
        }
        window.setTimeout(poll, 1000);
      } catch (_) {
        messageElement.textContent = "Mất kết nối progress — đang thử lại";
        window.setTimeout(poll, 2000);
      }
    };
    window.setTimeout(poll, 500);
  }

  applyStyle();
  if (cues.length) {
    video.addEventListener("loadedmetadata", () => {
      video.currentTime = Math.min(cues[0].start_ms / 1000 + 0.01, video.duration || Infinity);
      syncSubtitle();
    }, { once: true });
  }
})();
