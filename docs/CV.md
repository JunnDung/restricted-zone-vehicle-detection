# ZoneGuard — nội dung đưa vào CV

**Hệ thống phát hiện phương tiện đi vào khu vực cấm** · Python, PyTorch, YOLO11n, OpenCV, ByteTrack, FastAPI, SQLite, Colab, GitHub Actions.

- Xây dựng pipeline phát hiện/theo dõi phương tiện và xác nhận vi phạm vùng đa giác; tích hợp dashboard, ảnh bằng chứng, video H.264 và nhật ký SQLite/CSV/JSON.
- Chuẩn hóa 15.000 ảnh MIO/VisDrone/COCO thành 5 lớp, tách train/validation/test, loại trùng chính xác và kiểm nguồn gốc bằng SHA-256; fine-tune YOLO11n trên Colab T4.
- Nâng mAP50–95 trung bình đều ba test chọn lọc từ 10,78% lên 36,21% (+25,44 điểm phần trăm); xử lý video mẫu 377 khung ở 28,10 FPS toàn pipeline trên T4.
- Thiết lập chọn checkpoint theo từng miền dữ liệu, phục hồi optimizer/checkpoint và CI kiểm API/quy tắc/tính toàn vẹn dữ liệu.

## Giải thích khi phỏng vấn

36,21% là mAP50–95 detector trên tập con nghiên cứu, không phải “độ chính xác cảnh báo”. VisDrone test giảm 2,08 điểm %. FPS là một phép đo trên T4 tính cả xuất video, không cam kết realtime trên mọi máy. F1 video chỉ có hai lượt dùng hiệu chỉnh; không mô tả thành “100% chính xác”.

Các quyết định kỹ thuật có thể trình bày: điểm đáy hộp bao để xét ROI, dwell/min_frames, hysteresis, ổn định nhãn track, tái cảnh báo sau khi ra ngoài, nghiệm thu bằng validation trước test. Hướng tiếp theo là video camera khác/góc xa/ban đêm có nhãn độc lập.

Đối chiếu: [case study tiếng Anh](CASE_STUDY.md), [JSON kết quả](../reports/compact-training.json), [model card](../models/MODEL_CARD.md). Repository private cần cấp quyền xem hoặc dùng video demo khi ứng tuyển; quyền riêng tư repository chưa được thay đổi.
