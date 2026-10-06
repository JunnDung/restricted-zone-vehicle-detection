# Đối chiếu nghiên cứu và cải tiến ZoneGuard

Tra cứu ngày 06/10/2026. Chỉ sử dụng tài liệu chính thức, bài báo và repository của tác giả; không lấy tỷ lệ chính xác quảng cáo của demo khác làm mục tiêu đã đạt của dự án.

## Người khác giải quyết bài toán như thế nào?

| Nguồn | Phương pháp được mô tả | Áp dụng vào ZoneGuard |
|---|---|---|
| [Roboflow Supervision PolygonZone](https://supervision.roboflow.com/latest/detection/tools/polygon_zone/) | Kiểm tra anchor của detection trong đa giác; anchor mặc định là giữa cạnh đáy. | Giữ anchor giữa cạnh đáy đang dùng. Bổ sung trạng thái theo lượt, dwell và hysteresis thay vì chỉ đếm xe đang ở trong. OpenCV hiện có đáp ứng phép kiểm tra hình học. |
| [ByteTrack — tác giả](https://github.com/FoundationVision/ByteTrack) | Kết hợp detection điểm tin cậy thấp với tracklet để khôi phục vật thể bị che và lọc nền. | Giữ detection thấp để nối track; yêu cầu detection đạt ngưỡng người dùng để tích lũy xác nhận vi phạm. Đây là chính sách cảnh báo của dự án, không phải quy tắc vùng cấm của bài báo ByteTrack. |
| [Vehicle Detection and Tracking From Surveillance Cameras in Urban Scenes](https://arxiv.org/abs/2109.12414) | Kết hợp tracker dựa trên IoU và appearance ReID để xử lý che khuất dài, chuyển động nhanh; thí nghiệm trên UA-DETRAC. | Ghi nhận hướng khắc phục đổi ID. Chưa thêm ReID vì chưa có phép đo lỗi đổi ID trên camera triển khai và GPU laptop còn dùng train. Track ID hiện tại không bảo đảm định danh vật lý xuyên che khuất dài. |
| [Vehicle detection for embedded platforms — repository nghiên cứu](https://github.com/sktedro/vehicle_detection_for_embedded_platforms) | Chuẩn hóa nhiều dataset giao thông, ánh xạ lớp/masking; đánh giá model theo nguồn dữ liệu và cấu hình phần cứng. | Kết hợp dữ liệu mặt đường MIO với VisDrone, ghi mapping/hash, đo riêng từng nguồn và có baseline trước fine-tune. Không gộp mAP của nhiều domain thành lời hứa chính xác thực địa. |
| [MIO-TCD — tác giả dataset](https://tcd.miovision.com/challenge/dataset.html) | Dataset camera giao thông, gồm phần localization và classification riêng. | Chỉ dùng localization có ground truth; phần test chính thức không nhãn được loại khỏi phép đo. Kiểm nhãn theo tài liệu/gói tác giả. |

Phát hiện xe, theo dõi xe và xác nhận vi phạm là ba phép đo khác nhau. Detector mAP cao không tự chứng minh cảnh báo đúng lượt, đúng vùng và không lặp. Dự án giữ cả đánh giá detector theo domain và đánh giá cảnh báo trên video.

## Những thay đổi đã thực hiện sau tra cứu

1. **Tách độ tin cậy nối track khỏi xác nhận vi phạm.** `pipeline.py` vẫn cho ByteTrack nhận box điểm thấp; `zones.py` chỉ cộng chuỗi dwell/min_frames khi detection đủ ngưỡng. Detection yếu đặt lại chuỗi chưa xác nhận nhưng không mở lượt cảnh báo mới cho xe đã được cảnh báo và vẫn trong vùng. Khi độ tin cậy dao động, xác nhận có thể chậm hơn; cần đo tradeoff trên video có nhãn độc lập.
2. **Rà soát diễn biến quanh bằng chứng.** Trong cửa sổ ảnh, nút “Xem diễn biến trước cảnh báo” mở video của phiên và tua tới hai giây trước khung cảnh báo. Dùng frame_index/FPS của video xuất, phù hợp cả phiên USB có đồng hồ thời gian khác. Chỉ bật khi phiên đã kết thúc và có thể xem kết quả.
3. **Tiến độ training phản ánh công việc thật.** Xác minh hash ảnh/nhãn bằng bốn luồng với hàng đợi giới hạn; vẫn kiểm đủ manifest. Dashboard hiển thị số ảnh đã kiểm, nguồn/batch đánh giá và epoch/batch train. Phân biệt chuẩn bị dataloader với tối ưu model. Callback được đối chiếu [tài liệu Ultralytics](https://docs.ultralytics.com/usage/callbacks/) và mã nguồn phiên bản 8.3.253 cài tại máy.

Không thêm dependency cho ba thay đổi trên: dùng ThreadPoolExecutor của Python, callback Ultralytics và trình phát video HTML hiện có.

## Kết quả kiểm chứng

- 20 kiểm thử unittest đạt, trong đó có kiểm dữ liệu bị sửa phải bị từ chối và detection yếu không được xác nhận vi phạm.
- Chạy lại toàn bộ video bãi xe Intel 377 khung trên CPU sau thay đổi: hai cảnh báo tại 6,56 s và 17,12 s, khớp kết quả trước đó. F1 lượt cảnh báo 1,0; F1 có xét loại xe 0,5 do ô tô đỏ vẫn bị nhận thành truck. Chỉ có hai lượt trên video hiệu chỉnh, không phải benchmark độc lập.
- Kiểm tra trên trình duyệt: cảnh báo 6,56 s mở video đúng mốc 4,56 s; video H.264 giải mã được ở 768×432, readyState=4. Ảnh kiểm chứng: `reports/review-evidence.png`.
- Báo cáo máy đọc: `reports/research-improvements.json`. Nhật ký training đang chạy: `reports/training-live.json` và `runtime/training.stdout.log` (không commit vì thay đổi liên tục).

Các cải tiến này được kiểm chứng về hành vi và khả năng quan sát tiến độ. Chưa có kết luận tăng độ chính xác của model mới khi training chưa hoàn tất. Trọng số mới chỉ được đưa vào dashboard nếu đạt điều kiện admission đã đặt trước khi nhìn test, xem [MODEL_CARD](../models/MODEL_CARD.md).

Đã đo lại checkpoint hiện tại trên toàn bộ validation mở rộng trước fine-tune, cùng imgsz=640 và batch=8:

| Validation | Số ảnh | mAP50 | mAP50–95 |
|---|---:|---:|---:|
| MIO, camera giao thông | 12.089 | 11,51% | 7,35% |
| VisDrone, góc drone | 548 | 40,76% | 25,10% |

Nguồn: `reports/large-baseline-validation.json`. Đây là mốc đối chứng của checkpoint cũ, không phải kết quả YOLO11s mới. Chênh lệch trên hai dataset cho thấy chất lượng hiện tại khác nhiều theo nguồn; không thể quy hoàn toàn cho góc camera vì phân bố lớp và ảnh cũng khác nhau. Số validation VisDrone này khác phép đo test-dev 37,54% ghi trong model card do split khác.

## Phép đo tiếp theo cần ưu tiên

Thu video có nhãn theo camera/ngày, gồm ngày/đêm, xe sát biên, che khuất, vào-ra lại và xe không vi phạm. Đo precision/recall/F1 theo lượt, sai loại, đổi ID/cảnh báo lặp và độ trễ; đo FPS khi máy không chạy tác vụ khác. Split MIO theo nhóm ID chưa chứng minh độc lập camera; ảnh gần trùng có thể còn. Chỉ cân nhắc ReID khi có bằng chứng đổi ID là nguyên nhân chính của cảnh báo lặp hoặc bỏ sót.
