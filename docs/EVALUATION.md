# Đánh giá mô hình

## Giao thức

Dataset compact gồm 12.000 ảnh train, 1.500 validation và 1.500 test từ MIO-TCD, VisDrone và COCO. Validation/test lấy 500 ảnh mỗi nguồn. Các số đo là kết quả trên split chọn lọc, không phải benchmark dataset đầy đủ.

Baseline và ứng viên được đo trên cùng Tesla T4. Quy tắc lựa chọn dùng validation: trung bình đều nguồn mAP50–95 phải cải thiện hơn 0,005, từng nguồn không giảm quá 0,02 và F1 trên video hồi quy không giảm. Quyết định được tính trước khi đánh giá test; không dùng test để chọn checkpoint.

Run hoàn thành 15 epoch (5 laptop và 10 Colab), chọn checkpoint epoch 15. Trung bình validation mAP50–95 tăng từ 12,18% lên 36,52%. Cấu hình và nguồn dữ liệu được lưu trong [run settings](../reports/compact-run-settings.json), [provenance](../reports/compact-dataset.json), [manifest](../data/compact-manifest.json.gz) và [quyết định chọn checkpoint](../reports/colab-checkpoint-selection.json).

## Kết quả test

| mAP50–95 | Baseline | Fine-tuned | Thay đổi (điểm %) |
|---|---:|---:|---:|
| MIO-TCD | 7,80% | 59,16% | +51,36 |
| VisDrone | 22,22% | 20,14% | −2,08 |
| COCO vehicles | 2,31% | 29,34% | +27,03 |
| Trung bình đều ba nguồn | 10,78% | 36,21% | +25,44 |

Trung bình đều nguồn là trung bình cộng của ba mAP, không phải mAP tính gộp tất cả ảnh. Mức giảm cho phép 0,02 áp dụng validation; kết quả test VisDrone vẫn giảm 2,08 điểm phần trăm.

![Learning curve và kết quả test](../reports/evaluation.png)

[Báo cáo theo lớp và nguồn](../reports/compact-training.json) · [Baseline validation/test](../reports/colab-baseline-validation.json) · [Learning curve CSV](../reports/compact-training-curve.csv)

## Kiểm tra pipeline

Video bãi xe có 377 khung hình và hai lượt xe có nhãn trong [parking reference](../reports/parking-reference.json). F1 cảnh báo là 1,00 cho cả baseline và model fine-tuned; F1 có xét loại xe tăng từ 0,50 lên 1,00. Pipeline đầy đủ, tính xuất H.264, đạt lần lượt 27,93 và 28,10 FPS trên T4 trong một lần chạy.

Video này đã dùng để hiệu chỉnh nên chỉ là kiểm tra hồi quy. Kết quả được đo trước khi bổ sung chính sách tổng hợp nhãn qua nhiều khung hình. Chính sách hiện tại có thể trả về nhóm chung khi các nhãn xung đột, thay vì luôn gán một loại xe cụ thể. Chưa đo lại toàn bộ benchmark tốc độ và F1 cho chính sách mới.

[Video kết quả](../reports/demo-colab-model.mp4) · [Thông tin môi trường](../reports/colab-runtime.json)

## Giới hạn diễn giải

- COCO/VisDrone từng được sử dụng trong phát triển model gốc; không gọi test hiện tại là độc lập hoàn toàn với lịch sử phát triển.
- Split MIO chưa chứng minh độc lập camera; loại ảnh trùng hệt không loại được mọi tương quan giữa camera và cảnh.
- Chưa có benchmark thực địa đủ lớn theo lượt xe, ban đêm, thời tiết hoặc camera triển khai.
- FPS phụ thuộc video, phần cứng và chi phí xuất kết quả; một lần chạy không phải phân phối hiệu năng.

Xem [model card](../models/MODEL_CARD.md) để biết checksum và giấy phép, [hướng dẫn Colab](COLAB.md) để tái tạo dữ liệu và chạy huấn luyện.
