# Fine-tune nhanh trên RTX 3060 Laptop

Cập nhật 08/10/2026. Run YOLO11s trên 81.923 ảnh đã lỗi cấp phát RAM ở epoch 1, batch 255, chưa hoàn thành epoch và chưa có checkpoint. Trạng thái lỗi được giữ trong `reports/training-failure-20261006.json`; các báo cáo dataset/baseline lớn là lịch sử, không phải kết quả run mới.

## Cấu hình thay thế

| Thành phần | Run cũ | Run mới |
|---|---|---|
| Khởi tạo | YOLO11s COCO | YOLO11n 5 lớp đã fine-tune VisDrone |
| Train | 81.923 ảnh | Tối đa 12.000 ảnh chọn theo lớp |
| Nguồn | MIO + VisDrone | COCO vehicles 5.000 + MIO 5.000 + VisDrone 2.000 |
| Validation / test | Bộ lớn, hai nguồn | Tối đa 500 ảnh/nguồn/split, ba nguồn |
| Epoch tối đa / patience | 30 / 8 | 15 / 5 |
| Batch / kích thước | 8 / 640 | 4 / 640 |
| Mosaic / cache ảnh / workers | 0,5 / không / 0 | 0 / không / 0 |
| Optimizer | Tự động | SGD, lr0=0,002, AMP |

Giữ 640 px để không giảm thêm khả năng phát hiện xe nhỏ. Model đã học phương tiện được fine-tune toàn bộ, không huấn luyện từ đầu. Tắt Mosaic tránh dựng ảnh ghép 1280×1280; vẫn có biến đổi màu, scale và flip của Ultralytics. Batch nhỏ giảm bộ nhớ mỗi bước; Ultralytics tích lũy gradient theo nbs mặc định. Chưa có benchmark tốc độ hoàn chỉnh hoặc kết luận model mới chính xác hơn.

## Chọn dữ liệu và đánh giá

`scripts/prepare_compact.py` lấy lớp hiếm theo vòng, thứ tự SHA-256 cố định và không lặp ảnh. Giữ một phần ảnh nền không có lớp xe nếu nguồn có sẵn. Đây là cân bằng độ phủ lớp theo ảnh, không bảo đảm mọi lớp có số box bằng nhau. COCO ánh xạ category ID: motorcycle 4→0, car 3→1, truck 8→2, bus 6→3, bicycle 2→4. Chỉ tải các JPEG cần dùng và annotations gốc; ảnh có crowd box phương tiện bị loại thay vì giả định nhãn đó là xe đơn.

MIO/VisDrone giữ nguyên phân vùng từ manifest cũ. COCO dùng ảnh có xe của train2017 cho train; val2017 được chia theo ID chẵn/lẻ để chọn validation và test riêng. Sau loại crowd, hai nửa val2017 chỉ có 452/396 ảnh chứa xe; phần còn lại được chọn từ ảnh nền có annotation chính thức để đủ 500 ảnh/split, qua đó kiểm tra báo nhầm. Khi thiếu ảnh dương, bước chọn mẫu bổ sung ảnh nền vượt ngân sách khởi đầu. Loại trùng chính xác giữa các split trước khi chốt manifest. Ảnh và nhãn được sao chép độc lập vào `datasets/compact`; xóa dữ liệu cũ không ảnh hưởng dữ liệu mới hoặc dự án bên cạnh.

Đánh giá checkpoint cũ trước fine-tune trên cùng ba validation. Chỉ đưa ứng viên vào dashboard nếu trung bình mAP50–95 tăng >0,005, từng nguồn không giảm quá 0,02 và cả F1 cảnh báo có/không xét loại xe không giảm trên video bãi xe. Quyết định trước khi nhìn test. Model hiện tại được giữ khi không đạt.

Giới hạn: đây là tập chọn lọc, không phải benchmark COCO/MIO/VisDrone chính thức. Model khởi tạo từng dùng COCO trong quá trình pretrain/phát triển; test COCO không chứng minh tổng quát hóa sang camera mới. Split MIO chưa được chứng minh độc lập camera; ảnh gần trùng có thể còn. Hai lượt trong video bãi xe là kiểm tra hồi quy, không đủ đo độ chính xác thực địa.

## Chạy, phục hồi và lưu kết quả

Chạy `start-training.ps1` tại thư mục dự án. Dữ liệu hiện tại chỉ cần kiểm bằng `python scripts/prepare_compact.py`; khi manifest đã có, script không cần bộ dữ liệu lớn cũ. Nếu tái tạo từ đầu, chạy `prepare_visdrone.py`, `prepare_large.py`, rồi `prepare_compact.py` để tái hiện chọn mẫu từ các nguồn gốc; sau khi kiểm tra thành công có thể dọn dữ liệu trung gian.

Run mới: `runs/zoneguard_compact`. `last.pt` được lưu sau mỗi epoch; `start-training.ps1 -Resume` yêu cầu checkpoint thật và dataset/model/batch/imgsz khớp. Không coi batch của epoch chưa hoàn tất là epoch hoàn thành. Log ở `runtime/training.*.log`, tiến độ ở dashboard và `reports/training-live.json`. Báo cáo cuối: `reports/compact-training.json`, curve và baseline ba nguồn. Tùy chọn backup của launcher tự commit/push kết quả cuối; dataset không được đẩy GitHub.

Trên Windows, kiểm tra cả RAM vật lý có sẵn và dung lượng có thể commit. Chờ khi một trong hai dưới 2 GB; kiểm tra sau mỗi batch train/val và trước bước chính. GPU cần 2,5 GB VRAM trống lúc khởi động. Không đóng ứng dụng khác hoặc đổi pagefile của máy. Kiểm tra này giảm nguy cơ hết bộ nhớ nhưng không loại bỏ tình huống ứng dụng khác đột ngột chiếm RAM giữa hai lần kiểm tra. Máy cần tiếp tục chạy; không có ETA chắc chắn trước epoch đầu.

## Nguồn

- [COCO — tác giả](https://cocodataset.org/#download), [COCO dataset trong Ultralytics](https://docs.ultralytics.com/datasets/detect/coco/): annotation chính thức, ảnh tải qua endpoint S3 HTTPS có chứng chỉ hợp lệ. License ID từng ảnh nằm trong manifest; ảnh giữ giấy phép gốc, không được đẩy GitHub.
- [MIO-TCD](https://tcd.miovision.com/challenge/dataset.html), [VisDrone](https://github.com/VisDrone/VisDrone-Dataset): giữ nguồn gốc và điều khoản nghiên cứu, MIO CC BY-NC-SA 4.0.
- [Ultralytics training tips](https://docs.ultralytics.com/guides/model-training-tips/): pretrained, batch phù hợp và validation để đánh giá; API được đối chiếu bản 8.3.253 cài tại máy.
- [Microsoft MEMORYSTATUSEX](https://learn.microsoft.com/en-us/windows/win32/api/sysinfoapi/ns-sysinfoapi-memorystatusex): phân biệt RAM trống và khả năng commit.
