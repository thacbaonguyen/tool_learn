(() => {
  const activeCard = [...document.querySelectorAll("[data-ocr-job-id]")].find(
    (card) => ["queued", "running"].includes(card.dataset.ocrJobStatus),
  );
  if (!activeCard) return;

  const jobId = activeCard.dataset.ocrJobId;
  const message = activeCard.querySelector(".ocr-message");
  const progress = activeCard.querySelector("progress");

  async function poll() {
    try {
      const response = await fetch(`/jobs/${jobId}`, { cache: "no-store" });
      if (!response.ok) throw new Error("poll failed");
      const job = await response.json();
      if (message) message.textContent = job.message;
      if (progress) progress.value = job.progress;
      if (["completed", "failed"].includes(job.status)) {
        window.location.reload();
        return;
      }
      window.setTimeout(poll, 1500);
    } catch (_) {
      if (message) message.textContent = "Mất kết nối progress — đang thử lại";
      window.setTimeout(poll, 2500);
    }
  }

  window.setTimeout(poll, 700);
})();
