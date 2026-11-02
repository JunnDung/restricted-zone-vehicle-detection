# Model card — ZoneGuard

## Mô hình mặc định

`vehicle-large.pt` là **YOLO11n** với 2.590.815 tham số. Khóa API `large` được giữ để tương thích, không biểu thị kiến trúc YOLO11s. File inference đã bỏ optimizer; không dùng để resume training.

- Kích thước: 5.479.112 byte (5,23 MiB).
- SHA-256: `4affce6c60f7d7c4b8b75b172209938e6673fa0404037862f3395797be72cab3`.
- Thứ tự lớp: `motorcycle`, `car`, `truck`, `bus`, `other`.
- Huấn luyện: 15 epoch, gồm 5 trên RTX 3060 Laptop và 10 tiếp tục trên Tesla T4.
- Thiết lập: 640 px, SGD, lr0=0,002, AMP, Mosaic=0, seed 42; batch 4→16, workers 0→2 khi chuyển sang T4.
- Môi trường Colab: Ultralytics 8.3.253, PyTorch 2.11.0+cu130, Python 3.13.16; xem [runtime](../reports/colab-runtime.json).

Chuyển batch và phần cứng không bảo đảm tái hiện bitwise. Checkpoint phục hồi có optimizer nằm tại `checkpoints/compact-last.pt`, kèm [metadata và checksum](checkpoints/compact-last.json); bản lưu này hoàn thành **5 epoch**, không phải trọng số inference cuối cùng 15 epoch.

## Dữ liệu và định nghĩa nhãn

Dataset compact gồm 12.000 ảnh train (MIO 5.000, VisDrone 2.000, COCO 5.000), 1.500 ảnh validation và 1.500 ảnh test. Mỗi nguồn đóng góp 500 ảnh cho từng split validation/test. [Manifest nén](../data/compact-manifest.json.gz) và [provenance](../reports/compact-dataset.json) ghi nguồn, split, chuyển đổi nhãn và hash.

`other` gộp một số nhóm như bicycle, van, tricycle, awning-tricycle, work_van và non-motorized_vehicle tùy dataset; không phải lớp nhận diện mọi vật thể chưa biết. MIO `pickup_truck` được gộp vào `truck` theo định nghĩa dataset. Nhãn mơ hồ và vùng ignore được xử lý trong bước chuẩn bị dữ liệu.

## Đánh giá

| Test chọn lọc · mAP50–95 | Baseline | Fine-tuned |
|---|---:|---:|
| MIO-TCD | 7,80% | 59,16% |
| VisDrone | 22,22% | 20,14% |
| COCO vehicles | 2,31% | 29,34% |
| Trung bình đều nguồn | 10,78% | 36,21% |

Checkpoint được chọn theo validation trước khi đo test. VisDrone test giảm 2,08 điểm phần trăm. Video hồi quy hai lượt xe cho F1 có xét loại xe 1,00; kết quả này không đại diện cho độ chính xác thực địa và được đo trước chính sách tổng hợp nhãn hiện tại.

Xem [phương pháp và giới hạn](../docs/EVALUATION.md), [kết quả chi tiết](../reports/compact-training.json) và [quyết định chọn checkpoint](../reports/colab-checkpoint-selection.json).

## Baseline và trọng số đối chứng

`vehicle-yolo11n.pt` là YOLO11n fine-tune VisDrone2019 DET, từ [dự án nhận dạng phương tiện](https://github.com/JunnDung/vehicle-type-recognition), commit `3b5ce0645801d5a4f4aa0e31dadb990cbea57f9c`. SHA-256: `819f8c0c2864ea6efc64e974fb5de9609f3a958dfe7f88acd4b8961cc27f5efc`.

Baseline dùng 6.471 train / 548 validation / 1.610 test-dev, 30 epoch, ảnh 640, batch 8, seed 32. [Báo cáo training nguồn](../reports/upstream-training.json) và [dữ liệu nguồn](../reports/upstream-dataset.json) được giữ để truy xuất nguồn gốc; đó không phải phép đo mới về cảnh báo vùng cấm.

`yolo11n.pt` và `yolo11s.pt` là các trọng số COCO đối chứng từ [Ultralytics assets v8.3.0](https://github.com/ultralytics/assets/releases/tag/v8.3.0):

| Tệp | SHA-256 |
|---|---|
| `yolo11n.pt` | `0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1` |
| `yolo11s.pt` | `85a76fe86dd8afe384648546b56a7a78580c7cb7b404fc595f97969322d502d5` |

## Giới hạn và giấy phép

Sai nhãn vẫn có thể xảy ra với xe nhỏ, bị che, cảnh đêm và camera khác miền dữ liệu. Track ID đổi có thể gây cảnh báo lặp; ID giữ sai có thể gây bỏ sót. Cần đánh giá theo lượt xe trên camera triển khai. COCO/VisDrone từng được sử dụng khi phát triển model gốc; split MIO chưa chứng minh độc lập camera.

Mã dự án và Ultralytics theo AGPL-3.0. MIO-TCD theo [CC BY-NC-SA 4.0](https://tcd.miovision.com/challenge/dataset.html); VisDrone theo [điều khoản dataset](https://github.com/VisDrone/VisDrone-Dataset); ảnh COCO theo giấy phép ảnh gốc. Trọng số kết hợp phải được sử dụng phù hợp với các điều khoản này.
