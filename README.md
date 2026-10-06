# Xây dựng hệ thống phát hiện phương tiện đi vào khu vực cấm

**ZoneGuard**: YOLO11 + ByteTrack, bản đồ đa giác trên camera, cảnh báo có ảnh bằng chứng và nhật ký SQLite. Phân tích video hoặc camera USB; giao diện tiếng Việt chạy tại máy.

Đề tài: **Xây dựng hệ thống phát hiện phương tiện đi vào khu vực cấm**. Sản phẩm gồm checkpoint AI dùng được ngay, trình vẽ vùng cấm theo góc camera và cảnh báo kèm bằng chứng. Mã nguồn được lưu theo từng phần tại [GitHub](https://github.com/JunnDung/restricted-zone-vehicle-detection).

## Chạy tại Windows

Máy hiện tại đã có môi trường CUDA tại `.venv`. Nhấp `run.cmd`, mở **http://127.0.0.1:8003**. Server dùng một nguồn và một tiến trình; giữ cửa sổ server khi sử dụng.

Trên máy mới cần Python 3.12, internet khi cài dependency, sau đó chạy PowerShell tại thư mục dự án:

```powershell
powershell -ExecutionPolicy Bypass -File setup.ps1 -Cuda
.\run.cmd
```

Bỏ `-Cuda` nếu dùng CPU. Có thể truyền `-Python C:\duong-dan\python.exe`. Setup tạo venv riêng, cài torch/torchvision từ kho chính thức, cài các phiên bản khóa và chạy kiểm thử. Trọng số và hai video mẫu đã nằm trong repository; chạy demo không cần tải dataset hoặc tài khoản cloud. Không chạy hai server cùng trỏ một thư mục runtime.

## Sử dụng

1. Chọn **Bãi xe · Intel**, video khác hoặc Camera USB (chỉ số 0–5). Video tối đa 500 MB, 15 phút, 54.000 khung và 4096 px mỗi chiều/9 megapixel.
2. Kiểm tra đa giác trên mặt đường. **Thêm vùng**, click đỉnh theo thứ tự, Enter/Hoàn tất để kết thúc; kéo đỉnh để chỉnh; Escape để hủy. Tối đa 10 vùng, 3–32 đỉnh/vùng. Có thể nhập/xuất JSON.
3. Chọn loại xe bị hạn chế, thời gian xác nhận, số lần phát hiện, vùng đệm biên và thời gian xác nhận ra ngoài. **Lưu bản đồ** lưu riêng cho nguồn đang chọn.
4. Bấm **Bắt đầu giám sát**; chọn Âm báo để bật tiếng cảnh báo trong trình duyệt. Mở ảnh cảnh báo để rà soát.
5. Sau phiên tải CSV UTF-8, JSON, cấu hình và video MP4/H.264. Lịch sử và ảnh còn lại sau khi khởi động lại; cấu hình phiên là bản chụp độc lập với bản đồ sửa sau đó.

Một cảnh báo cho mỗi cặp track/vùng/lượt. Xe phải xuất hiện liên tục đủ dwell và min_frames; mất quan sát hơn 1 giây không được cộng nối thời gian. Ra ngoài đủ exit_seconds mới cho cảnh báo lượt tiếp theo. Tùy chọn “Cảnh báo xe đã có…” tắt thì cần quan sát xe ở ngoài trước khi vào. Điểm giữa cạnh đáy bounding box xấp xỉ vị trí xe chạm đường; không xét đơn thuần góc hộp bao giao đa giác.

Khi GPU đang train, **Tự động** dùng CPU cho giám sát; chọn GPU trực tiếp bị chặn để tránh hết VRAM. Có thể giảm kích thước AI/độ phân giải nguồn nếu CPU chậm. Bản đồ là ROI trong ảnh, chưa có tọa độ GIS; hệ thống cảnh báo cục bộ bằng giao diện/âm báo/nhật ký, không gửi email hay webhook.

## Huấn luyện dataset lớn tại máy, không mất phí cloud

Đã bổ sung pipeline dữ liệu **MIO-TCD Localization + VisDrone DET**, không nhầm với tập classification của MIO. MIO Localization tổng cộng 137.743 ảnh; chỉ phần train có ground truth được dùng và chia thành train/val/test nội bộ. Phần test chính thức không có nhãn không được dùng làm phép đo giả. Số lượng sau loại trùng, phân bố lớp, hash và nguồn tải ở `reports/large-dataset.json` khi chuyển đổi hoàn tất.

```powershell
# Cài CUDA nếu chưa có; chuẩn bị VisDrone ở data/vehicles.
.\.venv\Scripts\python.exe scripts\prepare_visdrone.py
# Tải MIO từ tác giả, chuyển nhãn thật, kiểm checksum và loại ảnh trùng hệt.
.\.venv\Scripts\python.exe scripts\prepare_large.py
# Chạy GPU cục bộ ở nền, tự đánh giá và commit/push kết quả khi hoàn tất.
powershell -ExecutionPolicy Bypass -File start-training.ps1
```

Máy hiện tại đã có VisDrone và gói MIO tải đủ; không cần chạy lại chuẩn bị khi `datasets/large/manifest.json` tồn tại. 30 epoch trên dữ liệu lớn có thể kéo dài nhiều giờ hoặc hơn một ngày, tùy tốc độ thực tế; chưa có ETA đáng tin trước epoch đầu. Máy cần tiếp tục hoạt động. Checkpoint `runs/zoneguard_large/weights/last.pt` được lưu mỗi epoch; sau khi gián đoạn:

```powershell
powershell -ExecutionPolicy Bypass -File start-training.ps1 -Resume
```

Xem **Mô hình & hướng dẫn** trong dashboard, `reports/training-live.json`, `runtime/training.stdout.log` và `runtime/training.stderr.log`. Tiến độ epoch được cập nhật sau mỗi epoch. Nếu tác vụ khác đang giữ VRAM, train chờ ít nhất 4 GB trống và không dừng tiến trình đó. Có thể kiểm tra GPU bằng `nvidia-smi`. Giữ dữ liệu nguyên checksum khi resume. Không xóa run/checkpoint đang huấn luyện. Nếu train lỗi, xem log rồi resume sau khi xử lý nguyên nhân; không lấy checkpoint lỗi làm mô hình chính.

Dataset được chia theo nhóm ID, giữ split chính thức VisDrone, loại ảnh trùng chính xác giữa các split. Chưa chứng minh split MIO độc lập theo camera; cảnh gần giống vẫn có thể còn. Validation chọn checkpoint cân bằng hai nguồn; phép đo cuối dùng toàn bộ validation/test theo từng nguồn. YOLO11s mới chỉ xuất hiện trong dashboard sau khi đạt các tiêu chí chống suy giảm mô tả tại [MODEL_CARD](models/MODEL_CARD.md).

Ảnh dữ liệu, venv, log và runtime không đẩy GitHub. Repository lưu script tái tạo, checksum, báo cáo, trọng số mẫu; train hoàn tất với `--backup-github` sẽ commit riêng báo cáo và trọng số mới nếu đạt. Script không lưu token; Git dùng đăng nhập Credential Manager có sẵn của máy. Nếu push thất bại, kết quả vẫn ở máy; chạy `git push` sau khi kết nối lại.

## Kết quả đã kiểm chứng

| Kiểm tra | Kết quả |
|---|---|
| Quy tắc vùng, validation API, lưu lịch sử, lớp nhãn, chống tranh GPU, ảnh preview và khóa tệp Windows | 18 kiểm thử unittest đạt |
| Checkpoint nguồn VisDrone 5 lớp, test-dev sau chuyển đổi | mAP50 37,54%; mAP50–95 22,24% |
| Video bãi xe Intel, 377 khung, cấu hình mặc định | Hai cảnh báo tại 6,56 s và 17,12 s |
| Phân loại hai cảnh báo | Xe trắng đúng car; xe đỏ bị nhầm truck |
| YOLO11n/YOLO11s COCO trên cùng video | 0 cảnh báo |
| Dashboard | Vẽ ROI, lịch sử, ảnh bằng chứng, phát MP4/H.264, bố cục desktop/mobile được kiểm tra trong trình duyệt |

Nguồn số đo: `reports/upstream-training.json`, `reports/model-comparison.json`, `reports/parking-reference.json`. Mốc video được gán tay với cửa sổ thời gian, chỉ gồm hai lượt và đã dùng hiệu chỉnh; F1 trên đó không thể suy rộng thành độ chính xác thực địa. FPS ban đầu có tiến trình dashboard chạy đồng thời nên chỉ tham khảo, chưa phải benchmark phần cứng có kiểm soát. Mô hình train lớn chưa được coi là cải thiện cho đến khi có báo cáo cuối và `accepted: true`.

Camera USB có đường chạy và thông báo lỗi thiết bị, chưa được kiểm chứng bằng camera vật lý trong đợt bàn giao này. Ban đêm, camera rung, xe bị che và ID đổi vẫn là hạn chế. Không có nhận diện biển số, nhiều camera đồng thời hoặc RTSP trong phiên bản này.

## Chạy không qua dashboard và kiểm thử

```powershell
.\.venv\Scripts\python.exe pipeline.py data\samples\car-detection.mp4 --zones config\parking.json --output runtime\cli-demo --device 0
.\.venv\Scripts\python.exe pipeline.py camera:0 --output runtime\camera-demo --device cpu --max-seconds 60
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Mỗi lần CLI cần output mới để không ghi đè phiên cũ. API docs ở http://127.0.0.1:8003/docs. Server mặc định chỉ bind 127.0.0.1; chưa có đăng nhập cho triển khai ra mạng.

## Cấu trúc

| Đường dẫn | Vai trò |
|---|---|
| app.py / web/ | API và giao diện tiếng Việt |
| pipeline.py / zones.py / store.py | Detector/tracker, quy tắc lượt vào vùng, SQLite |
| config/parking.json | Bản đồ mẫu ROI và chính sách |
| models/ | Checkpoint hiện tại, COCO khởi tạo và model card |
| scripts/ | Chuẩn bị dữ liệu, train, đánh giá cảnh báo |
| reports/ | Nguồn gốc và số đo tái kiểm tra |
| runtime/ | Video tải lên, bản đồ từng nguồn, phiên, bằng chứng, log (không commit) |
| datasets/large/ / runs/ | Dataset manifest và checkpoint training (không commit) |
| docs/BAO_CAO.md | Mô tả phương pháp và kế hoạch đánh giá thực địa |

## Tham khảo và ghi công

- [Ultralytics YOLO11](https://docs.ultralytics.com/models/yolo11/) và [tracking](https://docs.ultralytics.com/modes/track/): detector/tracker và pretrained COCO, AGPL-3.0.
- [ByteTrack — FoundationVision](https://github.com/FoundationVision/ByteTrack): theo dõi kết hợp detection điểm tin cậy thấp.
- [OpenCV pointPolygonTest](https://docs.opencv.org/4.x/dc/d48/tutorial_point_polygon_test.html): khoảng cách có dấu tới biên đa giác.
- [VisDrone Dataset](https://github.com/VisDrone/VisDrone-Dataset): drone detector, giữ các split gốc. `prepare_visdrone.py` và checkpoint 5 lớp tái sử dụng từ dự án cùng máy nêu trong model card.
- [MIO-TCD — tác giả](https://tcd.miovision.com/challenge/dataset.html): Luo và cộng sự, dữ liệu camera giao thông, CC BY-NC-SA 4.0.
- [Intel sample videos](https://github.com/intel-iot-devkit/sample-videos): CC BY 4.0; attribution và URL/hash từng video trong `data/samples/NOTICE.md` và `sources.json`.

Mã nguồn dự án: AGPL-3.0, xem LICENSE. Điều khoản dataset và mẫu video được giữ riêng theo nguồn; xem model card trước khi sử dụng trọng số kết hợp ngoài phạm vi nghiên cứu.
