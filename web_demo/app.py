# app.py
import os
import sys
import uuid
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify, send_from_directory
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import warnings
warnings.filterwarnings("ignore")

# Thêm thư mục gốc vào đường dẫn
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

app = Flask(__name__)

# Cấu hình
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')
app.config['RESULT_FOLDER'] = os.path.join('static', 'results')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Giới hạn kích thước file 16MB

# Đảm bảo thư mục uploads, results và images tồn tại
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)
os.makedirs(os.path.join('static', 'images'), exist_ok=True)

# Import BiM-VFI modules
from modules.components import make_components
from utils.padder import InputPadder

# Khởi tạo thiết bị
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Sử dụng thiết bị: {device}")
torch.set_grad_enabled(False)
if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

# Cấu hình model paths
MODEL_PATHS = {
    'pretrained': os.path.join(project_root, 'pretrained', 'bim_vfi.pth'),
    'trained_330': os.path.join(project_root, 'save', 'bim_vfi_train_new__400_epochs_NEW', 'checkpoints', 'model_best.pth'),
}

# Dictionary để lưu trữ các models đã load
loaded_models = {}

def load_bim_vfi_model(model_path):
    """Load BiM-VFI model từ checkpoint"""
    if model_path in loaded_models:
        return loaded_models[model_path]
    
    # Khởi tạo model
    model_cfg = {
        'name': 'bim_vfi',
        'args': {
            'pyr_level': 3,
            'feat_channels': 32
        }
    }
    model = make_components(model_cfg)
    
    # Load checkpoint
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        if 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
            print(f"Đã tải model từ checkpoint: {model_path}")
        else:
            model.load_state_dict(checkpoint)
            print(f"Đã tải model từ state dict: {model_path}")
    else:
        print(f"Cảnh báo: Không tìm thấy file model: {model_path}")
        return None
    
    model.to(device)
    model.eval()
    loaded_models[model_path] = model
    return model

# Load pretrained model mặc định
default_model = load_bim_vfi_model(MODEL_PATHS['pretrained'])
if default_model is None:
    # Thử load model khác nếu pretrained không tồn tại
    for key, path in MODEL_PATHS.items():
        default_model = load_bim_vfi_model(path)
        if default_model is not None:
            print(f"Đã load model {key} làm mặc định")
            break

# Các định dạng ảnh được phép
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def make_inference(model, img0, img1, n):
    """
    Thực hiện nội suy giữa hai ảnh bằng thuật toán BiM-VFI
    
    model: BiM-VFI model
    img0, img1: Tensor ảnh đầu vào, đã chuẩn hóa [0-1]
    n: Số khung hình trung gian cần tạo (tạo đúng n frames, không phải n-1)
    """
    frames = []
    
    # Tạo n frames với time steps đều đặn từ 1/(n+1) đến n/(n+1)
    # Tương tự như inference_demo: range(1, ratio) / ratio
    for i in range(1, n + 1):
        time_step = i / (n + 1)
        with torch.no_grad():
            input_dict = {
                'img0': img0,
                'img1': img1,
                'time_step': torch.tensor([time_step]).to(device),
                'pyr_level': 5
            }
            result = model(**input_dict)
            frame = result['imgt_pred']
            frames.append(frame)
    
    return frames

def generate_interpolation(model, img0_path, img1_path, num_frames=16):
    """
    Tạo các khung hình trung gian giữa hai ảnh bằng BiM-VFI
    
    model: BiM-VFI model
    img0_path, img1_path: Đường dẫn đến hai ảnh gốc
    num_frames: Số lượng khung hình trung gian cần tạo
    """
    # Đọc ảnh đầu vào (BGR format)
    img0_bgr = cv2.imread(img0_path)
    img1_bgr = cv2.imread(img1_path)
    
    if img0_bgr is None or img1_bgr is None:
        raise ValueError(f"Không thể đọc ảnh đầu vào. Kiểm tra đường dẫn: {img0_path}, {img1_path}")
    
    # Chuyển sang RGB
    img0_rgb = cv2.cvtColor(img0_bgr, cv2.COLOR_BGR2RGB)
    img1_rgb = cv2.cvtColor(img1_bgr, cv2.COLOR_BGR2RGB)
    
    # Đảm bảo kích thước ảnh giống nhau
    h, w, _ = img0_rgb.shape
    img1_rgb = cv2.resize(img1_rgb, (w, h))
    
    # Chuẩn hóa và chuyển đổi định dạng sang tensor
    img0 = torch.from_numpy(img0_rgb.transpose(2, 0, 1)).to(device).unsqueeze(0).float() / 255.
    img1 = torch.from_numpy(img1_rgb.transpose(2, 0, 1)).to(device).unsqueeze(0).float() / 255.
    
    # Padding ảnh để đảm bảo kích thước chia hết cho 32
    padder = InputPadder(img0.shape, 32)
    img0_padded, img1_padded = padder.pad(img0, img1)
    
    # Tạo các khung hình trung gian
    frames = make_inference(model, img0_padded, img1_padded, num_frames)
    
    # Chuyển các khung hình về định dạng numpy và unpad
    result_frames = []
    for frame in frames:
        frame_unpadded = padder.unpad(frame)
        # Chuyển về numpy và RGB
        frame_np = (frame_unpadded[0] * 255).clamp(0, 255).byte().cpu().numpy().transpose(1, 2, 0)
        # Chuyển về BGR để lưu với cv2
        frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
        result_frames.append(frame_bgr)
    
    return result_frames

@app.route('/favicon.ico')
def favicon():
    # Tạo một favicon đơn giản nếu chưa có
    if not os.path.exists(os.path.join(app.root_path, 'static', 'images', 'favicon.ico')):
        # Tạo hình vuông xanh lá cây đơn giản
        img = Image.new('RGB', (32, 32), color=(76, 175, 80))
        img.save(os.path.join(app.root_path, 'static', 'images', 'favicon.ico'))
    
    return send_from_directory(os.path.join(app.root_path, 'static', 'images'),
                              'favicon.ico', mimetype='image/vnd.microsoft.icon')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/interpolate', methods=['POST'])
def interpolate():
    if 'image1' not in request.files or 'image2' not in request.files:
        return jsonify({'error': 'Thiếu file ảnh'}), 400
    
    image1 = request.files['image1']
    image2 = request.files['image2']
    
    if image1.filename == '' or image2.filename == '':
        return jsonify({'error': 'Không có file được chọn'}), 400
    
    if not (allowed_file(image1.filename) and allowed_file(image2.filename)):
        return jsonify({'error': 'Định dạng file không được hỗ trợ. Chỉ chấp nhận .jpg, .jpeg, .png'}), 400
    
    # Tạo ID duy nhất cho phiên làm việc này
    session_id = str(uuid.uuid4())
    
    # Lưu ảnh đầu vào
    img0_path = os.path.join(app.config['UPLOAD_FOLDER'], f'{session_id}_0.png')
    img1_path = os.path.join(app.config['UPLOAD_FOLDER'], f'{session_id}_1.png')
    
    try:
        image1.save(img0_path)
        image2.save(img1_path)
        
        # Số lượng khung hình trung gian
        try:
            num_frames = int(request.form.get('frames', 16))
            if num_frames <= 0:
                num_frames = 16
        except:
            num_frames = 16
        
        # Lấy model được chọn
        selected_model = request.form.get('model', 'pretrained')
        if selected_model not in MODEL_PATHS:
            selected_model = 'pretrained'
        
        # Load model nếu chưa load
        model_path = MODEL_PATHS[selected_model]
        model = load_bim_vfi_model(model_path)
        
        if model is None:
            return jsonify({'error': f'Không thể load model: {selected_model}'}), 500
        
        # Tạo các khung hình trung gian
        frames = generate_interpolation(model, img0_path, img1_path, num_frames)
        
        # Lưu các khung hình kết quả
        frame_paths = []
        for i, frame in enumerate(frames):
            frame_path = os.path.join(app.config['RESULT_FOLDER'], f'{session_id}_{i}.png')
            cv2.imwrite(frame_path, frame)
            frame_paths.append(os.path.join('static', 'results', f'{session_id}_{i}.png'))
        
        # Tạo video từ các khung hình
        video_path = os.path.join(app.config['RESULT_FOLDER'], f'{session_id}.mp4')
        
        # Kích thước khung hình
        h, w = frames[0].shape[:2]
        
        # Tạo video writer với codec h264
        try:
            # Thử sử dụng codec h264 cho khả năng tương thích tốt hơn
            fourcc = cv2.VideoWriter_fourcc(*'H264')
            video = cv2.VideoWriter(video_path, fourcc, 24, (w, h))
        except:
            try:
                # Nếu không thành công, thử avc1
                fourcc = cv2.VideoWriter_fourcc(*'avc1')
                video = cv2.VideoWriter(video_path, fourcc, 24, (w, h))
            except:
                # Cuối cùng, sử dụng mp4v
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video = cv2.VideoWriter(video_path, fourcc, 24, (w, h))
        
        # Thêm khung hình đầu tiên
        img0_bgr = cv2.imread(img0_path)
        img0_bgr = cv2.resize(img0_bgr, (w, h))
        video.write(img0_bgr)
        
        # Thêm các khung hình trung gian
        for frame in frames:
            video.write(frame)
        
        # Thêm khung hình cuối cùng
        img1_bgr = cv2.imread(img1_path)
        img1_bgr = cv2.resize(img1_bgr, (w, h))
        video.write(img1_bgr)
        
        video.release()
        
        return jsonify({
            'success': True, 
            'frames': frame_paths,
            'video': f'/video/{session_id}.mp4',
            'download_video': os.path.join('static', 'results', f'{session_id}.mp4'),
            'original1': os.path.join('static', 'uploads', f'{session_id}_0.png'),
            'original2': os.path.join('static', 'uploads', f'{session_id}_1.png')
        })
    
    except Exception as e:
        print(f"Lỗi xử lý: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory('static', path)

@app.route('/video/<path:filename>')
def serve_video(filename):
    """Phục vụ video với headers phù hợp để phát trực tiếp trên trình duyệt"""
    video_path = os.path.join(app.config['RESULT_FOLDER'], filename)
    response = send_from_directory(os.path.dirname(video_path), 
                                  os.path.basename(video_path), 
                                  mimetype='video/mp4',
                                  as_attachment=False)
    # Thêm các headers cần thiết
    response.headers['Content-Disposition'] = f'inline; filename={filename}'
    response.headers['Accept-Ranges'] = 'bytes'
    return response

if __name__ == '__main__':
    app.run(debug=True)