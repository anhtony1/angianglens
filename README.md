# AnGiang Lens — source APK demo

Bộ source này đã chứa chính 2 file AI của bạn:

- `assets/ai/angiang_lens_model.tflite`
- `assets/ai/angiang_lens_database.json`

## Đã có sẵn

- Camera/chọn ảnh.
- AI TFLite chạy offline trên điện thoại.
- Resize 224×224, embedding, cosine similarity.
- Top-5 + majority vote.
- Threshold lấy từ JSON (hiện là 0.75).
- Nhận Miếu Bà Chúa Xứ / Hòn Phụ Tử.
- Ảnh lạ dưới ngưỡng → “Không nhận diện được địa điểm”.
- Chỉ khi nhận diện thành công mới mở trang chi tiết.
- Hiển thị tên, vị trí, % tương đồng, giới thiệu.
- Nút đọc thuyết minh tiếng Việt bằng Text-to-Speech.
- Nút video (chưa có URL vì chưa được cung cấp video).
- Nút quét lại.

## Build APK KHÔNG cần cài Android Studio trên Mac

1. Tạo repository mới trên GitHub.
2. Giải nén ZIP này.
3. Upload **toàn bộ nội dung bên trong thư mục `AnGiangLens_Flutter`** lên repo, giữ nguyên `.github/workflows/`.
4. Vào tab **Actions**.
5. Chọn **Build Android APK** → **Run workflow**.
6. Khi workflow xanh ✓, mở lần chạy đó.
7. Phần **Artifacts** → tải `AnGiangLens-APK`.
8. Giải nén sẽ có `app-release.apk` để cài lên Android.

Workflow tự tạo phần Android, đặt minSdk 26 rồi build APK release.

## Thêm ảnh/video sau

Sửa `assets/content/place_catalog.json`.

- `videoUrl`: điền URL video HTTPS.
- `images`: danh sách đường dẫn ảnh asset. Nếu thêm ảnh offline, nhớ khai báo thư mục ảnh trong `pubspec.yaml`.

Hiện app dùng ảnh vừa quét làm ảnh đại diện nên có thể test AI ngay bằng 2 file bạn đã gửi.
