# Model card — ZoneGuard

## Model mặc định được nghiệm thu · 02/11/2026

`vehicle-large.pt` hiện được chọn mặc định trên dashboard: **YOLO11n**, 2.590.815 tham số, 5.479.112 byte (5,23 MiB), SHA-256 `4affce6c60f7d7c4b8b75b172209938e6673fa0404037862f3395797be72cab3`. Khóa API `large` giữ để tương thích; không biểu thị YOLO11s. Bản inference mới đã bỏ optimizer; checkpoint huấn luyện được lưu riêng.

Hoàn tất 15 epoch: 5 trên laptop, 10 tiếp tục trên Tesla T4; batch 4→16, workers 0→2, 640px, Mosaic=0, AMP, SGD lr0=0,002. Colab: Ultralytics 8.3.253, torch 2.11.0+cu130, Python 3.13.16. Không coi đây là tái hiện bitwise giữa Windows/Linux.

Test chọn lọc 500 ảnh mỗi nguồn: mAP50–95 MIO 59,16%, VisDrone 20,14%, COCO 29,34%; trung bình đều nguồn 36,21% so với baseline 10,78%. **VisDrone test giảm 2,08 điểm %**; giới hạn giảm 0,02 trong nghiệm thu áp dụng validation. Không bảo đảm từng test nguồn tăng. Chi tiết từng lớp/nguồn ở `reports/compact-training.json`.

Video bãi xe 377 khung: cả hai lượt gán car, F1 có xét lớp 1,00 so với baseline 0,50. Pipeline T4 28,10 FPS tính xuất H.264, một lần chạy. Hai lượt chỉ kiểm hồi quy; cần dữ liệu có nhãn từ camera triển khai để đo độ chính xác cảnh báo. COCO/VisDrone từng xuất hiện trong phát triển model gốc, split MIO chưa chứng minh độc lập camera.

[Case study](../docs/CASE_STUDY.md) · [Colab](../docs/COLAB.md) · [Manifest nén](../data/compact-manifest.json.gz). Đợt replay dự kiến được bỏ qua vì run chính đạt validation; không có thêm epoch hoặc chọn model bằng test. Các phần dưới giữ lịch sử checkpoint ban đầu.

## Checkpoint dùng được ngay

`vehicle-yolo11n.pt` là YOLO11n, fine-tune từ COCO trong dự án cùng máy `D:\ComputerVision\NhanDangPhuongTien` (commit nguồn `3b5ce0645801d5a4f4aa0e31dadb990cbea57f9c`). Bản sao giữ nguyên trọng số, SHA-256 `819f8c0c2864ea6efc64e974fb5de9609f3a958dfe7f88acd4b8961cc27f5efc`.

- Dữ liệu nguồn: VisDrone2019 DET, 6.471 train / 548 validation / 1.610 test-dev; vùng ignore được che; nhãn được gộp thành 5 nhóm.
- Lớp theo đúng thứ tự: `motorcycle`, `car`, `truck`, `bus`, `other`.
- `other` ở VisDrone gồm bicycle, van, tricycle, awning-tricycle; không phải bộ nhận diện mọi vật thể chưa biết.
- Huấn luyện 30 epoch, ảnh 640, batch 8, seed 32, RTX 3060 Laptop. Epoch validation tốt nhất: 29.
- Test-dev sau chuyển đổi: precision 0,4976; recall 0,3784; mAP50 0,3754; mAP50–95 0,2224. Chi tiết từng lớp ở `reports/upstream-training.json`.
- Đây là kết quả của checkpoint dự án nguồn, không phải phép đo mới về tracking hoặc vi phạm vùng cấm, cũng không phải benchmark VisDrone chính thức.

## Trọng số đối chứng

`yolo11n.pt` và `yolo11s.pt` là trọng số COCO chính thức từ [Ultralytics assets v8.3.0](https://github.com/ultralytics/assets/releases/tag/v8.3.0). Chỉ giữ motorcycle/car/truck/bus/bicycle khi suy luận. SHA-256:

| Tệp | SHA-256 |
|---|---|
| yolo11n.pt | 0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1 |
| yolo11s.pt | 85a76fe86dd8afe384648546b56a7a78580c7cb7b404fc595f97969322d502d5 |

COCO không đạt cảnh báo nào trên video bãi xe góc nhìn từ trên trong phép thử hiện tại; tăng kích thước mô hình đơn thuần chưa giải quyết được lệch miền dữ liệu.

## Run dữ liệu lớn ngày 06/10/2026 (đã dừng)

Run lịch sử của `scripts/train_large.py` khởi tạo YOLO11s từ COCO, học trên MIO-TCD Localization kết hợp VisDrone. Máy tự kiểm checksum dữ liệu, đo baseline và huấn luyện trên GPU cục bộ. Thiết lập khi đó: 30 epoch tối đa, early stopping 8 epoch, batch 8, ảnh 640, seed 42, AMP, workers 0. Run lỗi RAM trước khi hoàn thành epoch đầu; xem cấu hình thay thế bên dưới.

`other` ở dữ liệu kết hợp thêm work_van và non-motorized_vehicle. MIO pickup_truck được gộp vào truck; đây là định nghĩa của tác giả dataset. Các motorized_vehicle không phân biệt được loại bị che xám, pedestrian là background. Không lấy nhãn demo ngẫu nhiên trong gói code làm ground truth.

Validation chọn checkpoint gồm số ảnh bằng nhau từ hai nguồn. Sau train, đánh giá toàn bộ validation riêng theo nguồn. Chỉ tạo `vehicle-large.pt` nếu:

1. Trung bình mAP50–95 hai nguồn tăng hơn 0,005 so với baseline, và VisDrone không giảm quá 0,02.
2. F1 cảnh báo và F1 có xét loại xe trên video kiểm tra bãi xe đều không giảm.
3. Hoàn thành đánh giá test và ghi báo cáo. Quyết định đạt tiêu chí được tính trước khi xem test.

Nếu không đạt, dashboard tiếp tục dùng checkpoint hiện tại. Kết quả cuối cùng nằm ở `reports/large-training.json`; không được coi mô hình mới là tốt hơn khi tệp này chưa tồn tại hoặc `accepted` là false. Video kiểm tra chỉ có hai lượt và đã dùng để hiệu chỉnh; không đại diện cho độ chính xác ngoài thực địa.

## Giới hạn và giấy phép

Xe nhỏ, bị che, cảnh đêm, rung camera và góc nhìn khác dữ liệu có thể gây lỗi. Track ID đổi có thể gây cảnh báo lặp; ID giữ sai có thể làm bỏ sót. Cần video có nhãn từ camera triển khai và đánh giá theo lượt xe trước khi vận hành thực tế. MIO chia theo nhóm ID ảnh; chưa chứng minh các camera độc lập giữa train/val/test, dù ảnh trùng hệt bị loại.

Mã dự án và Ultralytics dùng AGPL-3.0. MIO-TCD: CC BY-NC-SA 4.0 theo [tác giả](https://tcd.miovision.com/challenge/dataset.html). VisDrone: theo [điều khoản dataset](https://github.com/VisDrone/VisDrone-Dataset). Trọng số kết hợp cần được sử dụng phù hợp với điều khoản các nguồn dữ liệu này.

## Cấu hình thay thế ngày 08/10/2026

YOLO11s/MIO+VisDrone ở trên là run lịch sử, đã lỗi RAM tại epoch 1 batch 255, không tạo model được nghiệm thu. Cấu hình mặc định mới fine-tune `vehicle-yolo11n.pt` trên compact COCO+MIO+VisDrone (tối đa 12.000 train), 15 epoch, batch 4, imgsz 640, AMP, mosaic=0, SGD lr0=0,002. Dataset/provenance mới: `reports/compact-dataset.json`; baseline/kết quả: `reports/compact-baseline-validation.json`, `reports/compact-training.json` khi tương ứng hoàn tất. Không kế thừa số mAP của bộ lớn làm kết quả compact.

Quy tắc nghiệm thu mới áp dụng cho cả ba nguồn: trung bình validation mAP50–95 tăng >0,005, từng nguồn giảm không quá 0,02, F1 video hồi quy không giảm; quyết định trước test. Giữ khóa model `large` và tên `vehicle-large.pt` để tương thích API; nội dung ứng viên mới là YOLO11n. Test COCO là phần val2017 không fine-tune, nhưng pretrained/development từng dùng COCO; không gọi đây là dữ liệu độc lập hoàn toàn với model gốc. Split MIO chưa độc lập camera được chứng minh. Các giấy phép ảnh nguồn giữ nguyên. Xem `docs/TRAIN_NHANH.md`.
