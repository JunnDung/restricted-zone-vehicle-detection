# Huấn luyện ZoneGuard trên Google Colab

Notebook: [ZoneGuard_Colab.ipynb](../notebooks/ZoneGuard_Colab.ipynb). Dùng T4 miễn phí; không yêu cầu GPU trả phí hoặc token GitHub trong notebook. Hạn mức và thời gian phiên Colab do Google quyết định.

## Chạy từ bản clone

```powershell
git clone https://github.com/JunnDung/restricted-zone-vehicle-detection.git
cd restricted-zone-vehicle-detection
python scripts/package_colab.py
```

Mở notebook trên Colab, chọn **Thời gian chạy → Thay đổi loại thời gian chạy → GPU T4**. Tải `dist/zoneguard-colab.zip` trong ô upload rồi chạy ô khởi động. Gói chỉ chứa file dự án và checkpoint cần thiết; không chứa `.git`, credential, môi trường Python hoặc dataset lớn. `COLAB_PACKAGE.json` ghi commit, trạng thái working tree và hash các file.

Script tái tạo VisDrone, MIO và COCO từ nguồn gốc, giữ CRLF của file nhãn/manifest Windows để kiểm được SHA-256 run cũ. Chỉ resume khi manifest compact khớp `99d49b745fad70f1e40b0d5408fdd262976defecdab7187e91f4d112128db2b0`. Không bỏ qua kiểm tra nếu khác hash. Bản sao checkpoint gốc được kiểm hash, giữ nguyên optimizer và 5 epoch đã hoàn thành; đường dẫn dataset/output được chuyển sang Colab trên một bản sao.

## Thiết lập và đánh giá

- YOLO11n, Ultralytics 8.3.253, 640 px, AMP, Mosaic=0, SGD; resume hướng tới tổng 15 epoch.
- Batch 16, workers 2 trên T4; đây là thay đổi so với batch 4/workers 0 trên laptop. Optimizer/scheduler được phục hồi; không phải tái hiện bitwise giữa hai GPU.
- Baseline được đo lại trên cùng T4/batch. Checkpoint được xét bằng validation riêng từng nguồn; chỉ giữ checkpoint có trung bình mAP50–95 cải thiện >0,005 và không nguồn nào giảm quá 0,02.
- Quyết định nghiệm thu trước khi nhìn test; sau đó đo cả baseline và ứng viên trên ba test split. Video bãi xe chỉ kiểm hồi quy 2 lượt, không phải số đo độ chính xác ngoài thực địa.
- Báo cáo lưu mAP theo lớp/nguồn, thời gian suy luận, FPS toàn pipeline có tính xuất video, số tham số, kích thước và SHA-256 trọng số. Không so sánh FPS giữa các máy như cùng một thí nghiệm.

## Tiến độ và phục hồi

Chạy lại ô status để lấy snapshot mới; output cũ không tự cập nhật. `reports/training-live.json` ghi môi trường `google_colab` để dashboard không khóa GPU của máy Windows.

Chạy ô download sau khi checkpoint được lưu, và sau khi đánh giá hoàn tất. Giải nén bundle vào checkout tương ứng. Gói Colab lần sau sẽ kèm `last.pt`, `best.pt`, kết quả CSV và lựa chọn checkpoint nếu tồn tại, để tiếp tục checkpoint mới nhất thay vì quay lại epoch 5. Không thể phục hồi chính xác phần batch chưa lưu trong epoch dở.

Không đóng tab notebook hoặc tắt máy khi đang phụ thuộc vào MCP để nhận kết quả. Không dùng thao tác giả lập hoạt động để né hạn mức Colab. Notebook không tự lưu vào Drive khi đang dùng scratchpad; file `.ipynb` được giữ trong Git để có thể mở lại.

## Nguồn

[Ultralytics resume](https://docs.ultralytics.com/modes/train/#resuming-interrupted-trainings), [Google Colab MCP](https://github.com/googlecolab/colab-mcp), [Colab FAQ](https://research.google.com/colaboratory/faq.html). Các giấy phép nguồn dữ liệu vẫn áp dụng; xem [model card](../models/MODEL_CARD.md).
