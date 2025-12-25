# BiM-VFI Web Demo

Web demo cho thuật toán nội suy khung hình BiM-VFI (Bilateral Motion-based Video Frame Interpolation).

## Tính năng

- **Nội suy khung hình**: Tạo các khung hình trung gian mượt mà giữa hai ảnh
- **Lựa chọn model**: Chọn giữa pretrained model hoặc custom trained models
- **Xuất video**: Tự động tạo video từ các khung hình đã nội suy
- **Giao diện thân thiện**: Giao diện web đơn giản, dễ sử dụng

## Cài đặt

### Yêu cầu

- Python 3.8+
- PyTorch
- Flask
- OpenCV
- Các thư viện khác trong project BiM-VFI

### Hướng dẫn cài đặt

1. Đảm bảo bạn đã cài đặt tất cả dependencies cho BiM-VFI:

```bash
pip install torch torchvision
pip install opencv-python
pip install flask
pip install numpy
```

2. Đảm bảo bạn có các model weights:
   - `pretrained/bim_vfi.pth` - Pretrained model từ tác giả
   - `save/bim_vfi_train_new__400_epochs_NEW/checkpoints/model_best.pth` - Custom trained model (400 epochs)

## Sử dụng

1. Chạy web server:

```bash
cd web_demo
python app.py
```

2. Mở trình duyệt và truy cập: `http://localhost:5000`

3. Sử dụng web demo:
   - Upload hai ảnh (ảnh bắt đầu và ảnh kết thúc)
   - Chọn model muốn sử dụng
   - Chọn số lượng khung hình trung gian (8, 16, hoặc 32)
   - Nhấn "Tạo video"
   - Xem kết quả và tải video xuống

## Cấu trúc thư mục

```
web_demo/
├── app.py                 # Flask application
├── README.md             # File hướng dẫn này
├── templates/
│   └── index.html        # Giao diện web
└── static/
    ├── uploads/          # Ảnh upload
    ├── results/          # Video và frames kết quả
    └── images/           # Assets
```

## Models

### Pretrained (Original)
Model gốc từ tác giả BiM-VFI, được train trên dataset chuẩn.

### Custom Trained (400 epochs)
Model được train lại với 400 epochs trên dataset tùy chỉnh.


## Lưu ý

- Các file ảnh được hỗ trợ: `.jpg`, `.jpeg`, `.png`
- Kích thước file tối đa: 16MB
- Model sẽ tự động padding ảnh để đảm bảo kích thước chia hết cho 32
- Kết quả sẽ được lưu trong thư mục `static/results/`

## Khắc phục sự cố

### Lỗi: "Không tìm thấy module 'modules'"
Đảm bảo bạn đang chạy app.py từ thư mục "web demo" và project root đã được thêm vào sys.path.

### Lỗi: "Không thể load model"
Kiểm tra xem file model có tồn tại tại đường dẫn được chỉ định không.

### Lỗi CUDA out of memory
Giảm kích thước ảnh đầu vào hoặc số lượng khung hình trung gian.

## Tham khảo

- [BiM-VFI GitHub](https://github.com/KAIST-VICLab/BiM-VFI)
- [BiM-VFI Paper](https://arxiv.org/abs/2306.15111)
