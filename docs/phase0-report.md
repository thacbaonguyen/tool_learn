# Báo cáo Phase 0

## Môi trường

- CPU host benchmark: AMD Ryzen 7 5800H with Radeon Graphics
- CPU quota của container: 3; host có 16 logical CPU
- RAM nhìn thấy: 13.5 GB
- Chế độ: CPU-only, xử lý tuần tự, toàn bộ dữ liệu local

## Kết quả đo

| Bước | Thời gian | Peak RAM | Kết quả |
|---|---:|---:|---|
| OCR | 32.111 giây | 392.39 MB | 19 cue |
| TTS | 30.374 giây | 1565.03 MB | v-tts-zero-shot |
| Render | 3.767 giây | 884.27 MB | 15.0 giây video |

## OCR

- Video mẫu: 54.73 giây, 1920x1080.
- Tần suất lấy mẫu: 5.0 FPS.
- Transcript WER so với subtitle track tham chiếu: 0.0.
- Số cue OCR và subtitle nhúng có thể khác nhau vì subtitle đóng cứng hiển thị nhiều cụm từ trong cùng một banner.

## TTS và render

- Voice cloning thực sự: đạt.
- Chạy khi container bị tắt network: đạt.
- Dung lượng model cache sau lần tải đầu: 851.1 MB.
- Thời gian TTS trong bảng bao gồm load model và sinh cue, nhưng không bao gồm tải model lần đầu.
- Audio gốc đã bị loại bỏ: có.
- File video: `artifacts/phase0_preview.mp4`.
- Cue có tốc độ lớn hơn 1.2x được đánh dấu `needs_review` để người dùng rút gọn bản dịch.

## Kết luận

OCR nền xanh, voice cloning CPU và render local đều đạt yêu cầu POC trên video mẫu. Kết quả benchmark được đo trên host hiện tại với quota 3 CPU; máy i3 Gen 10 cần được benchmark lại và dự kiến chậm hơn.
