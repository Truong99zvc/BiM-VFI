<p align="center">
  <a href="https://www.uit.edu.vn/" title="Trường Đại học Công nghệ Thông tin" style="border: 5;">
    <img src="https://i.imgur.com/WmMnSRt.png" alt="Trường Đại học Công nghệ Thông tin | University of Information Technology">
  </a>
</p>

<!-- Title -->
<h1 align="center"><b>CS420.Q12 - CÁC VẤN ĐỀ CHỌN LỌC TRONG THỊ GIÁC MÁY TÍNH</b></h1>

## MỤC LỤC
* [Giới thiệu môn học](#gioi-thieu-mon-hoc)
* [Giảng viên hướng dẫn](#giang-vien-huong-dan)
* [Sinh viên thực hiện](#sinh-vien-thuc-hien)
* [Đồ án](#do-an)
* [Cài đặt môi trường](#cai-dat-moi-truong)
* [Lưu ý cấu hình quan trọng](#luu-y-cau-hinh-quan-trong)
* [Dữ liệu](#du-lieu)
* [Mô hình huấn luyện sẵn](#mo-hinh-huan-luyen-san)
* [Đánh giá](#danh-gia)
* [Huấn luyện](#huan-luyen)
* [Demo](#demo)
  - [Demo dòng lệnh](#demo-dong-lenh)
  - [Web Demo](#web-demo)
* [Kaggle Notebook](#kaggle-notebook)
* [Tham khảo](#tham-khao)

## GIỚI THIỆU MÔN HỌC
<a name="gioi-thieu-mon-hoc"></a>
* **Tên môn học**: Các vấn đề chọn lọc trong Thị giác máy tính
* **Mã môn học**: CS420
* **Mã lớp**: CS420.Q12
* **Năm học**: 2025 - 2026

## GIẢNG VIÊN HƯỚNG DẪN
<a name="giang-vien-huong-dan"></a>
* **TS. Mai Tiến Dũng** - *dungmt@uit.edu.vn*

## SINH VIÊN THỰC HIỆN
<a name="sinh-vien-thuc-hien"></a>
| MSSV | Họ và tên | Github | Email |
|:----------:|:-------------------:|:----------------------------------------------------:|:-----------------------:|
| 22521587   | Trương Phúc Trường  | [Truong99zvc](https://github.com/Truong99zvc/)      | 22521587@gm.uit.edu.vn  |
| 22521571   | Võ Đình Trung       | [votrung654](https://github.com/votrung654/)         | 22521571@gm.uit.edu.vn  |

## ĐỒ ÁN
<a name="do-an"></a>
**Tên đồ án**: BiM-VFI - NỘI SUY KHUNG HÌNH VIDEO DỰA TRÊN TRƯỜNG CHUYỂN ĐỘNG HAI CHIỀU

**Dự án này là kết quả của quá trình tái lập (reproduce) và phát triển ứng dụng dựa trên nghiên cứu:**
> **BiM-VFI: Bilateral Motion Field-Guided Video Frame Interpolation for Non-Uniform Motions (CVPR 2025)**
> *Tác giả: Wonyong Seo, Jihyong Oh, Munchurl Kim*

**Đóng góp của nhóm sinh viên:**
1.  **Tái lập huấn luyện (Reproducibility):** Huấn luyện lại mô hình từ đầu trên GPU giới hạn (GTX 1650) và GPU T4x2 free của kaggle để kiểm chứng kết quả trong bài báo.
2.  **Khắc phục lỗi tương thích:** Chỉnh sửa mã nguồn để hoạt động ổn định trên các môi trường mới (Torch mới, CUDA mới) và sửa các lỗi đặc thù trên Windows (đường dẫn, thư viện OpenMP).
3.  **Phát triển Ứng dụng:** Xây dựng thêm module **Web Demo** (Flask) cho phép người dùng tương tác trực quan với mô hình.
4.  **Tài liệu hóa:** Việt hóa và chi tiết hóa tài liệu hướng dẫn sử dụng.

Mã nguồn cốt lõi (Core Model Architecture) thuộc về nhóm tác giả KAIST-VICLab.

Repository này chứa mã nguồn cài đặt của BiM-VFI, một phương pháp nội suy khung hình video được hướng dẫn bởi trường chuyển động hai chiều dành cho video có chuyển động không đồng nhất.  Dự án dựa trên bài báo CVPR 2025 của Wonyong Seo, Jihyong Oh và Munchurl Kim. Repository này sử dụng code gốc của nhóm tác giả đồng thời chỉnh sửa một số thiết lập về cấu hình, phiên bản,... để phù hợp tương thích đa số các thiết bị ở thời điểm hiện tại để thực hiện quá trình reproduce dễ dàng hơn và tránh lỗi, xung đột. Ngoài ra, nhóm cũng bổ sung thêm chức năng web demo để chạy thử mô hình như 1 ứng dụng.

## CÀI ĐẶT MÔI TRƯỜNG
<a name="cai-dat-moi-truong"></a>

### Yêu cầu tiên quyết
Trước khi thiết lập môi trường, hãy đảm bảo đã cài đặt **Conda** trên hệ thống của mình. Có thể tải xuống và cài đặt Conda từ:
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) (Khuyên dùng - nhẹ)
- [Anaconda](https://www.anaconda.com/products/distribution) (Bản đầy đủ)

### Thiết lập môi trường và cài đặt thư viện

> **Lưu ý quan trọng**: Các phiên bản thư viện trong repository này khác với repository gốc của BiM-VFI. Vì quá trình tái lập huấn luyện (reproduce) của nhóm được thực hiện trên **GTX 1650**, nhóm sử dụng phiên bản PyTorch mới nhất hỗ trợ CUDA 13.0 (`torch torchvision --index-url https://download.pytorch.org/whl/cu130`) để tối ưu hóa khả năng tương thích.

```bash
conda create -n bimvfi python=3.11
conda activate bimvfi
pip install basicsr-fixed Ipython torchsummary moviepy pyyaml imageio packaging tqdm opencv-python tensorboardx ptflops pyiqa lpips stlpips_pytorch dists_pytorch torch torchvision --index-url https://download.pytorch.org/whl/cu130
conda install cupy -c conda-forge
```

### Thư viện bổ sung cho Web Demo
Để chạy web demo, cần cài đặt thêm các thư viện sau:

```bash
pip install flask werkzeug pillow scikit-image
```

**Lưu ý**: `opencv-python` và `torch` đã được bao gồm trong thiết lập môi trường chính ở trên. Hãy đảm bảo đã cài đặt đầy đủ trong quá trình thiết lập môi trường để tránh lỗi.

## LƯU Ý CẤU HÌNH QUAN TRỌNG
<a name="luu-y-cau-hinh-quan-trong"></a>

### Biến môi trường KMP_DUPLICATE_LIB_OK
Trong `main.py`, nhóm đã thêm dòng sau không có trong repository gốc:
```python
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
```
Biến môi trường này giải quyết lỗi "OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized" có thể xảy ra khi nhiều bản sao của runtime OpenMP được liên kết vào chương trình. Đây là vấn đề phổ biến trên hệ thống Windows khi sử dụng các thư viện như NumPy, PyTorch và OpenCV cùng nhau.

### Cấu hình đường dẫn tuyệt đối

Sau khi clone repository, **bắt buộc** phải sửa đổi đường dẫn dataset và model trong các file cấu hình nằm trong thư mục `cfgs/`. Các đường dẫn tương đối mặc định sẽ không hoạt động và cần được thay đổi thành **đường dẫn tuyệt đối** tương ứng với hệ thống cục bộ.

#### Cho huấn luyện (`cfgs/bim_vfi_train_new.yaml`):
Thay đổi đường dẫn dataset từ tương đối sang tuyệt đối:
```yaml
# Trước (sẽ KHÔNG hoạt động)
root_path: ../data/vimeo_triplet

# Sau (ví dụ - điều chỉnh theo đường dẫn thực tế)
root_path: C:/Users/YourUsername/BiM-VFI/data/vimeo_triplet
```

#### Cho đánh giá (`cfgs/bim_vfi_benchmark.yaml`):
1. **Đường dẫn model** - Thay đổi `resume` thành đường dẫn tuyệt đối:
```yaml
# Trước
resume: ./save/train_new__400_epochs_NEW/checkpoints/model_best.pth

# Sau (ví dụ)
resume: C:/Users/YourUsername/BiM-VFI/save/train_new__400_epochs_NEW/checkpoints/model_best.pth
```

2. **Đường dẫn dataset** - Thay đổi `root_path` thành đường dẫn tuyệt đối:
```yaml
# Trước
root_path: ../data/vimeo_triplet

# Sau (ví dụ)
root_path: C:/Users/YourUsername/BiM-VFI/data/vimeo_triplet
```

## DỮ LIỆU
<a name="du-lieu"></a>
### Tải xuống
Có thể tải xuống dataset Vimeo90K được sử dụng để huấn luyện và kiểm thử từ liên kết sau:
> - [Vimeo90K](https://cove.thecvf.com/datasets/875)

### Chuẩn bị
Sau khi tải xuống dataset, hãy sắp xếp nó theo cấu trúc dự án. Dataset nên được đặt trong thư mục `data`. Nhóm sử dụng dataset Vimeo 90K-Triplet cho quá trình reproduce và đánh giá.

## MÔ HÌNH HUẤN LUYỆN SẴN
<a name="mo-hinh-huan-luyen-san"></a>

Repository này bao gồm hai model:

### 1. Pretrained model gốc (từ paper)
- **Đường dẫn**: `pretrained/bim_vfi.pth`
- **Mô tả**: Đây là pretrained model gốc từ bài báo BiM-VFI. Nó đã được bao gồm trong repository này.

### 2. Model được huấn luyện lại (reproduce)
- **Đường dẫn**: `save/train_new__400_epochs_NEW/checkpoints/model_best.pth`
- **Mô tả**: Model này được nhóm huấn luyện lại từ đầu trên dataset Vimeo Triplet. Quá trình huấn luyện được cấu hình cho 400 epoch nhưng **đã dừng sớm ở epoch 330** do cơ chế Early Stopping (đây cũng là tinh chỉnh khác biệt so với repository gốc của nhóm tác giả paper).

### Cấu trúc thư mục
```
BiM-VFI/
├── pretrained/
│   └── bim_vfi.pth                    # Pretrained model gốc của bài báo
└── save/
    ├── eval_pretrained_model/         # Kết quả đánh giá của pretrained model
    │   └── logs/
    │       └── log_benchmark_['vimeo']_[['test']].txt
    ├── eval_train_330_epochs/         # Kết quả đánh giá của model huấn luyện lại
    │   └── logs/
    │       └── log_benchmark_['vimeo']_[['test']].txt
    └── train_new__400_epochs_NEW/     # Model được huấn luyện lại của nhóm
        └── checkpoints/
            └── model_best.pth         # Model tốt nhất (dừng sớm ở epoch 330)
```

## ĐÁNH GIÁ
<a name="danh-gia"></a>

### Kết quả đánh giá
Thư mục `save/` chứa kết quả đánh giá cho cả hai model:

#### 1. Đánh giá pretrained model
- **Vị trí**: `save/eval_pretrained_model/logs/log_benchmark_['vimeo']_[['test']].txt`
- **Mô tả**: Chứa kết quả benchmark (PSNR, SSIM, LPIPS, STLPIPS, NIQE) của **pretrained model gốc** từ bài báo, được đánh giá trên tập test Vimeo Triplet.

#### 2. Đánh giá reproduce model
- **Vị trí**: `save/eval_train_330_epochs/logs/log_benchmark_['vimeo']_[['test']].txt`
- **Mô tả**: Chứa kết quả benchmark (PSNR, SSIM, LPIPS, STLPIPS, NIQE) của **model được huấn luyện lại** (huấn luyện trong 330 epoch), được đánh giá trên tập test Vimeo Triplet.

### Chạy đánh giá
Việc đánh giá mong muốn có thể được thực hiện bằng cách thay thế phần `benchmark_dataset` trong `cfgs/bim_vfi_benchmark.yaml`.
* `name`: Tên của các dataset benchmark. Các dataset có thể benchmark là [_vimeo_, _vimeo\_septuplet_, _snu\_film_, _snu\_film\_arb_, _xtest_].
* `args`:
  * `root_path`: Đường dẫn của từng dataset. **Phải là đường dẫn tuyệt đối!**
  * `split`: Các split mong muốn để đánh giá. [_test_, _val_] cho _vimeo_ và _vimeo\_septuplet_, [(_easy_), _medium_, _hard_, _extreme_] cho _snu\_film_ và _snu\_film\_arb_, và [_single_, _multiple_] cho _xtest_.
  * `pyr_lvl`: 3 cho vimeo, 5 cho snu_film, và 7 cho xtest.
* `save_imgs`: `True` nếu muốn lưu kết quả nội suy, ngược lại là `False`. Việc lưu ảnh sẽ tốn nhiều thời gian hơn.

Sau đó, chạy lệnh bên dưới:
```bash
python main.py --cfg cfgs/bim_vfi_benchmark.yaml
```

## HUẤN LUYỆN
<a name="huan-luyen"></a>

Để huấn luyện model:
```bash
python main.py --cfg cfgs/bim_vfi_train_new.yaml
```

**Lưu ý**: Hãy chắc chắn cấu hình đường dẫn tuyệt đối trong `cfgs/bim_vfi_train_new.yaml` trước khi chạy (xem [Lưu ý cấu hình quan trọng](#luu-y-cau-hinh-quan-trong)).

## DEMO
<a name="demo"></a>
### Demo qua dòng lệnh
<a name="demo-dong-lenh"></a>
Các video tùy chỉnh ở định dạng nhiều ảnh hoặc video có thể được nội suy như sau.

Đầu tiên, thiết lập thư mục gốc demo như sau:
  - video1.mp4 
  - video2.mp4
  - video3
    - img0.png
    - img1.png
    - ...
  - ...

Sau đó, thay thế `root_path` trong `cfgs/bim_vfi_demo.yaml` thành đường dẫn dữ liệu mong muốn, và chạy:
```bash
python main.py --cfg cfgs/bim_vfi_demo.yaml
```

### Web Demo
<a name="web-demo"></a>
Dự án bao gồm một giao diện demo dựa trên web để dễ dàng nội suy khung hình video. Không như [Demo dòng lệnh](#demo-dong-lenh) ở trên vốn từ repository gốc, web demo này được nhóm xây dựng mới hoàn toàn với đa dạng chức năng và tiện dụng hơn. Để chạy web demo:

1. **Điều hướng đến thư mục web_demo**:
   ```bash
   cd web_demo
   ```

2. **Chạy ứng dụng Flask**:
   ```bash
   python app.py
   ```

3. **Truy cập giao diện web**:
   Mở trình duyệt web và truy cập `http://localhost:5000` (hoặc `http://127.0.0.1:5000`)

Web demo cung cấp các tính năng sau:
- **Nội suy cặp ảnh**: Tải lên hai ảnh và tạo khung hình nội suy giữa chúng
- **Nội suy video**: Tải lên tệp video và nội suy khung hình giữa các khung hình liên tiếp
- **Nội suy chuỗi khung hình**: Tải lên nhiều khung hình và nội suy giữa chúng
- **Lựa chọn Model**: Chọn giữa các model được huấn luyện trước khác nhau (pretrained, reproduce)
- **Thông số tùy chỉnh**: Điều chỉnh số lượng khung hình nội suy và FPS đầu ra

**Lưu ý**: Hãy chắc chắn rằng đã cài đặt tất cả các thư viện cần thiết được đề cập trong phần [Cài đặt môi trường](#cai-dat-moi-truong), bao gồm các thư viện bổ sung cho web demo (Flask, werkzeug, Pillow, scikit-image).

## KAGGLE NOTEBOOK
<a name="kaggle-notebook"></a>

Đối với người dùng **có GPU không tương thích, không có GPU**, gặp các lỗi về phần cứng, thiết lập, cài đặt hoặc những người muốn huấn luyện/đánh giá model trên tài nguyên đám mây, nhóm cung cấp một Kaggle notebook (đã chạy sẵn ra log và model):

> **Kaggle Notebook**: [https://www.kaggle.com/code/truong9/bim-vfi?scriptVersionId=289083440](https://www.kaggle.com/code/truong9/bim-vfi?scriptVersionId=289083440)

**Lưu ý**: Khi sử dụng Kaggle notebook, cũng sẽ cần điều chỉnh một số cấu hình trong các tệp YAML để khớp với đường dẫn môi trường Kaggle (ví dụ: `/kaggle/input/` cho dataset). Cần chọn đúng dataset Vimeo 90K-Triplet (đã có sẵn trên kaggle) và GPU phù hợp (kaggle có cho free 30 tiếng/tuần cho P100 hoặc T4x2).

## THAM KHẢO
<a name="tham-khao"></a>
Đồ án sử dụng mã nguồn gốc từ repository [KAIST-VICLab/BiM-VFI](https://github.com/KAIST-VICLab/BiM-VFI). Nhóm thực hiện xin gửi lời cảm ơn chân thành đến các tác giả.

Mã nguồn được tham khảo, tinh chỉnh và phát triển nhằm mục đích nghiên cứu khoa học và học tập, làm đồ án, hoàn toàn không mang tính thương mại.

Nếu sử dụng mã nguồn này cho nghiên cứu, vui lòng trích dẫn bài báo gốc:

```bibtex
@inproceedings{seo2025bimvfi,
  title={BiM-VFI: Bilateral Motion Field-Guided Video Frame Interpolation for Non-Uniform Motions},
  author={Seo, Wonyong and Oh, Jihyong and Kim, Munchurl},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2025}
}

