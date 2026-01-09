# BiM-VFI Web Demo Interface

Đây là module giao diện web tương tác cho dự án [BiM-VFI](../README.md). Module này cho phép người dùng trải nghiệm mô hình nội suy khung hình thông qua trình duyệt một cách trực quan.

[Quay lại Project Chính](../README.md)

## Tính năng
- **Giao diện trực quan**: Upload và xử lý ảnh/video ngay trên trình duyệt.
- **Hỗ trợ đa dạng**: Nội suy cặp ảnh, video ngắn, hoặc chuỗi khung hình.
- **Tùy chỉnh**: Lựa chọn model (Pretrained/Reproduced), số lượng khung hình trung gian (2x, 4x, 8x...).
- **Export**: Tải xuống kết quả dưới dạng Video.

## Cài đặt

### 1. Chuẩn bị môi trường gốc
Trước tiên, hãy đảm bảo đã cài đặt môi trường `bimvfi` và các thư viện cốt lõi (PyTorch, CUDA) theo hướng dẫn tại **[README chính của dự án](../README.md#cai-dat-moi-truong)**.

### 2. Cài đặt thư viện Web
Kích hoạt môi trường và cài thêm các gói cần thiết cho giao diện web:

```bash
conda activate bimvfi
pip install flask werkzeug pillow scikit-image
```

## Hướng dẫn sử dụng

1. **Di chuyển vào thư mục demo**:
   ```bash
   cd web_demo
   ```

2. **Khởi chạy Server**:
   ```bash
   python app.py
   ```
   *Lưu ý: Đảm bảo không có tiến trình nào khác đang chạy trên cổng 5000.*

3. **Truy cập**:
   Mở trình duyệt và vào địa chỉ: `http://localhost:5000`

## Cấu trúc thư mục

```
web_demo/
├── app.py                 # File khởi chạy Flask Server
├── templates/             # Giao diện HTML
│   └── index.html
├── static/
│   ├── uploads/           # Nơi lưu file người dùng upload (Tự động dọn dẹp)
│   ├── results/           # Nơi lưu kết quả xử lý (Tự động dọn dẹp)
│   ├── css/               # Stylesheet
│   └── js/                # Script xử lý frontend
└── README.md              # Tài liệu hướng dẫn này
```

## Khắc phục sự cố thường gặp

*   **Lỗi "ModuleNotFoundError"**:
    *   Đảm bảo đã chạy `conda activate bimvfi` trước khi chạy `python app.py`.
    *   Đảm bảo đang đứng đúng thư mục `web_demo` (hoặc cấu hình đường dẫn import đúng trong code).

*   **Lỗi CUDA Out of Memory**:
    *   Khi chạy trên GPU yếu (như GTX 1650 4GB), hãy hạn chế upload video độ phân giải quá cao (trên Full HD) hoặc giảm số lượng khung hình nội suy.

*   **Lỗi không load được Model**:
    *   Kiểm tra lại đường dẫn file `.pth` trong `README.md` chính xem đã cấu hình đường dẫn tuyệt đối chưa.

## Bản quyền
Module này là một phần mở rộng được phát triển bởi nhóm sinh viên thực hiện đồ án, dựa trên mã nguồn cốt lõi của **KAIST-VICLab**.
Vui lòng tham khảo [Giấy phép chung](../LICENSE) của dự án.