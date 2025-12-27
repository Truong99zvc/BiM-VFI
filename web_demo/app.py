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
from skimage.metrics import structural_similarity as ssim
import shutil
import subprocess
import time
warnings.filterwarnings("ignore")

# Thêm thư mục gốc vào đường dẫn
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

app = Flask(__name__)

# Cấu hình
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')
app.config['RESULT_FOLDER'] = os.path.join('static', 'results')
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB

# Đảm bảo thư mục tồn tại
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)
os.makedirs(os.path.join('static', 'images'), exist_ok=True)

# Import BiM-VFI modules
from modules.components import make_components
from utils.padder import InputPadder

# --- THÊM: Import class kiến trúc lai (đảm bảo bạn đã tạo file modules/components/bim_vfi/bim_ifnet.py) ---
try:
    from modules.components.bim_vfi.bim_ifnet import BiM_IFNet
except ImportError:
    print("Cảnh báo: Không tìm thấy module BiM_IFNet. Hãy kiểm tra lại file bim_ifnet.py")

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
    'bim-ifnet': os.path.join(project_root, 'save', 'bim_vfi_train_new__400_epochs_NEW', 'checkpoints', 'bim-ifnet.pth'),
}

# Dictionary để lưu trữ các models đã load
loaded_models = {}

# Class giả lập arguments (vì BiM_IFNet __init__ nhận args)
class ModelArgs:
    def __init__(self, pyr_level=3, feat_channels=32):
        self.pyr_level = pyr_level
        self.feat_channels = feat_channels

def load_bim_vfi_model(model_key):
    """
    Load model dựa trên key (tên loại model).
    model_key: 'pretrained', 'trained_330', hoặc 'bim-ifnet'
    """
    model_path = MODEL_PATHS.get(model_key)
    if not model_path:
        print(f"Lỗi: Không tìm thấy đường dẫn cho key {model_key}")
        return None
    
    # Kiểm tra cache
    if model_key in loaded_models:
        return loaded_models[model_key]
    print(f"Đang khởi tạo model: {model_key}...")

    # --- LOGIC CHỌN KIẾN TRÚC ---
    if model_key == 'bim-ifnet':
        # 1. Nếu là Model Lai -> Load class BiM_IFNet
        try:
            args = ModelArgs(pyr_level=3, feat_channels=32)
            model = BiM_IFNet(args)
            print("-> Đã khởi tạo kiến trúc BiM-IFNet (Hybrid)")
        except Exception as e:
            print(f"Lỗi khi khởi tạo BiM_IFNet: {e}")
            return None
    else:
        # 2. Nếu là Model Gốc (Pretrained / Reproduced) -> Dùng make_components cũ
        model_cfg = {
            'name': 'bim_vfi', # Tên registered trong code gốc
            'args': {
                'pyr_level': 3,
                'feat_channels': 32
            }
        }
        try:
            model = make_components(model_cfg)
            print("-> Đã khởi tạo kiến trúc BiM-VFI (Gốc)")
        except Exception as e:
            print(f"Lỗi khi khởi tạo BiM-VFI gốc: {e}")
            return None
    
    # --- LOAD TRỌNG SỐ (WEIGHTS) ---
    if os.path.exists(model_path):
        try:
            # weights_only=False để tránh warning với pickle phức tạp, hoặc True nếu chỉ lưu state_dict đơn giản
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False) 
            
            # Xử lý các trường hợp lưu checkpoint khác nhau
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint

            # Xử lý prefix "module." nếu train bằng DataParallel
            new_state_dict = {}
            for k, v in state_dict.items():
                name = k.replace("module.", "") 
                new_state_dict[name] = v
            
            # Load vào model
            model.load_state_dict(new_state_dict, strict=False) # strict=False để linh hoạt hơn nếu thừa thiếu key nhỏ
            print(f"-> Đã tải trọng số từ: {model_path}")
            
        except Exception as e:
            print(f"Lỗi khi load file checkpoint {model_path}: {e}")
            return None
    else:
        print(f"Cảnh báo: Không tìm thấy file checkpoint tại: {model_path}")
        # Vẫn trả về model (random weights) hoặc None tùy bạn, ở đây mình trả về None cho an toàn
        return None
    
    model.to(device)
    model.eval()
    loaded_models[model_key] = model # Lưu cache theo key
    return model

# Load model mặc định (ví dụ pretrained) khi khởi động app
print("--- Đang tải model mặc định ---")
current_model = load_bim_vfi_model('pretrained')

# Các định dạng file
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def allowed_video(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_VIDEO_EXTENSIONS

def compute_ssim(img1, img2):
    """Tính SSIM giữa hai ảnh để kiểm tra độ tương đồng"""
    if len(img1.shape) == 3:
        img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    else:
        img1_gray = img1
    
    if len(img2.shape) == 3:
        img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    else:
        img2_gray = img2
    
    if img1_gray.shape != img2_gray.shape:
        h, w = min(img1_gray.shape[0], img2_gray.shape[0]), min(img1_gray.shape[1], img2_gray.shape[1])
        img1_gray = cv2.resize(img1_gray, (w, h))
        img2_gray = cv2.resize(img2_gray, (w, h))
    
    similarity = ssim(img1_gray, img2_gray)
    return similarity

def resize_images_for_interpolation(img1, img2):
    """
    Resize 2 ảnh về cùng kích thước tối ưu cho nội suy
    Chiến lược: Resize về kích thước nhỏ hơn nhưng đảm bảo chia hết cho 32
    """
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    # Lấy kích thước nhỏ hơn
    target_h = min(h1, h2)
    target_w = min(w1, w2)
    
    # Làm tròn về bội số của 32 (xuống)
    target_h = (target_h // 32) * 32
    target_w = (target_w // 32) * 32
    
    # Đảm bảo kích thước tối thiểu
    target_h = max(target_h, 128)
    target_w = max(target_w, 128)
    
    # Resize cả 2 ảnh
    img1_resized = cv2.resize(img1, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
    img2_resized = cv2.resize(img2, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
    
    return img1_resized, img2_resized, target_w, target_h

def smart_resize_for_interpolation(img, max_dim=1024):
    """
    Resize thông minh để tối ưu tốc độ cho GTX 1650 4GB
    - Giữ nguyên nếu ảnh nhỏ hơn max_dim (chỉ làm tròn về bội 32)
    - Resize xuống nếu ảnh lớn hơn max_dim
    - KHÔNG resize lên nếu ảnh nhỏ
    - Trả về ảnh đã resize và scale factor để upscale lại sau
    """
    h, w = img.shape[:2]
    original_h, original_w = h, w
    
    # Nếu cả hai chiều đều nhỏ hơn max_dim, chỉ làm tròn về bội 32
    if h <= max_dim and w <= max_dim:
        # Làm tròn về bội số của 32 (xuống để không resize lên)
        target_h = (h // 32) * 32
        target_w = (w // 32) * 32
        
        # Nếu kích thước sau khi làm tròn quá nhỏ (< 32), giữ nguyên
        if target_h < 32:
            target_h = 32
        if target_w < 32:
            target_w = 32
        
        # Nếu kích thước không đổi, trả về ảnh gốc
        if target_h == h and target_w == w:
            return img, 1.0, original_h, original_w
        else:
            # Chỉ resize nếu cần làm tròn (resize xuống, không resize lên)
            if target_h <= h and target_w <= w:
                resized = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)
                return resized, 1.0, original_h, original_w
            else:
                # Nếu làm tròn làm ảnh to lên, giữ nguyên ảnh gốc
                return img, 1.0, original_h, original_w
    
    # Chỉ resize xuống nếu ảnh lớn hơn max_dim
    # Tính scale factor dựa trên chiều lớn nhất
    max_current = max(h, w)
    scale_factor = max_dim / max_current
    
    # Tính kích thước mới
    new_h = int(h * scale_factor)
    new_w = int(w * scale_factor)
    
    # Làm tròn về bội số của 32 (xuống)
    target_h = (new_h // 32) * 32
    target_w = (new_w // 32) * 32
    
    # Đảm bảo không nhỏ hơn 32 pixels
    target_h = max(target_h, 32)
    target_w = max(target_w, 32)
    
    # Resize xuống
    resized = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)
    
    # Tính scale factor thực tế để upscale lại sau
    actual_scale_h = original_h / target_h
    actual_scale_w = original_w / target_w
    
    return resized, max(actual_scale_h, actual_scale_w), original_h, original_w

def upscale_frames(frames, original_h, original_w):
    """
    Upscale các frames về kích thước gốc
    """
    upscaled_frames = []
    for frame in frames:
        upscaled = cv2.resize(frame, (original_w, original_h), interpolation=cv2.INTER_LANCZOS4)
        upscaled_frames.append(upscaled)
    return upscaled_frames

def make_inference(model, img0, img1, n):
    """Thực hiện nội suy giữa hai ảnh bằng thuật toán BiM-VFI"""
    frames = []
    
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
    """Tạo các khung hình trung gian giữa hai ảnh bằng BiM-VFI - OPTIMIZED"""
    img0_bgr = cv2.imread(img0_path)
    img1_bgr = cv2.imread(img1_path)
    
    if img0_bgr is None or img1_bgr is None:
        raise ValueError(f"Không thể đọc ảnh đầu vào")
    
    # Resize ảnh về cùng kích thước trước
    h0, w0 = img0_bgr.shape[:2]
    h1, w1 = img1_bgr.shape[:2]
    
    if h0 != h1 or w0 != w1:
        # Resize về kích thước nhỏ hơn
        target_h = min(h0, h1)
        target_w = min(w0, w1)
        img0_bgr = cv2.resize(img0_bgr, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
        img1_bgr = cv2.resize(img1_bgr, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
    
    # Smart resize để tối ưu tốc độ
    img0_resized, scale_factor, original_h, original_w = smart_resize_for_interpolation(img0_bgr)
    img1_resized, _, _, _ = smart_resize_for_interpolation(img1_bgr)
    
    # Ghi log để debug
    print(f"Original size: {original_w}x{original_h}")
    print(f"Resized size: {img0_resized.shape[1]}x{img0_resized.shape[0]}")
    print(f"Scale factor: {scale_factor:.2f}x")
    
    # Chuyển sang RGB
    img0_rgb = cv2.cvtColor(img0_resized, cv2.COLOR_BGR2RGB)
    img1_rgb = cv2.cvtColor(img1_resized, cv2.COLOR_BGR2RGB)
    
    # Chuyển sang tensor
    img0 = torch.from_numpy(img0_rgb.transpose(2, 0, 1)).to(device).unsqueeze(0).float() / 255.
    img1 = torch.from_numpy(img1_rgb.transpose(2, 0, 1)).to(device).unsqueeze(0).float() / 255.
    
    padder = InputPadder(img0.shape, 32)
    img0_padded, img1_padded = padder.pad(img0, img1)
    
    # Nội suy
    frames = make_inference(model, img0_padded, img1_padded, num_frames)
    
    # Chuyển về numpy
    result_frames = []
    for frame in frames:
        frame_unpadded = padder.unpad(frame)
        frame_np = (frame_unpadded[0] * 255).clamp(0, 255).byte().cpu().numpy().transpose(1, 2, 0)
        frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
        result_frames.append(frame_bgr)
    
    # Upscale lại về kích thước gốc nếu đã resize
    if scale_factor > 1.0:
        print(f"Upscaling {len(result_frames)} frames back to original size...")
        result_frames = upscale_frames(result_frames, original_h, original_w)
    
    return result_frames

def extract_frames_from_video(video_path):
    """Trích xuất tất cả các khung hình từ video"""
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    
    cap.release()
    return frames

def compute_frame_hash(frame):
    """Tính hash của khung hình để so sánh"""
    small = cv2.resize(frame, (8, 8))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    return hash(gray.tobytes())

def remove_duplicate_frames(frames):
    """Loại bỏ các khung hình giống nhau 100%"""
    if not frames:
        return []
    
    unique_frames = [frames[0]]
    prev_hash = compute_frame_hash(frames[0])
    
    for frame in frames[1:]:
        curr_hash = compute_frame_hash(frame)
        if curr_hash != prev_hash:
            unique_frames.append(frame)
            prev_hash = curr_hash
    
    return unique_frames

def interpolate_video_frames(model, frames, num_interpolations):
    """Nội suy giữa tất cả các cặp khung hình liên tiếp"""
    if len(frames) < 2:
        return frames
    
    result_frames = []
    
    for i in range(len(frames) - 1):
        result_frames.append(frames[i])
        
        frame0 = frames[i]
        frame1 = frames[i + 1]
        
        h, w = frame0.shape[:2]
        frame1_resized = cv2.resize(frame1, (w, h))
        
        frame0_rgb = cv2.cvtColor(frame0, cv2.COLOR_BGR2RGB)
        frame1_rgb = cv2.cvtColor(frame1_resized, cv2.COLOR_BGR2RGB)
        
        img0 = torch.from_numpy(frame0_rgb.transpose(2, 0, 1)).to(device).unsqueeze(0).float() / 255.
        img1 = torch.from_numpy(frame1_rgb.transpose(2, 0, 1)).to(device).unsqueeze(0).float() / 255.
        
        padder = InputPadder(img0.shape, 32)
        img0_padded, img1_padded = padder.pad(img0, img1)
        
        interpolated = make_inference(model, img0_padded, img1_padded, num_interpolations)
        
        for interp_frame in interpolated:
            interp_unpadded = padder.unpad(interp_frame)
            interp_np = (interp_unpadded[0] * 255).clamp(0, 255).byte().cpu().numpy().transpose(1, 2, 0)
            interp_bgr = cv2.cvtColor(interp_np, cv2.COLOR_RGB2BGR)
            result_frames.append(interp_bgr)
    
    result_frames.append(frames[-1])
    
    return result_frames

def convert_to_web_compatible(input_path, output_path=None):
    """Chuyển đổi video sang định dạng tương thích với web (H.264)"""
    if output_path is None:
        output_path = input_path.replace('.mp4', '_web.mp4')
    
    try:
        cmd = [
            'ffmpeg', '-y', '-i', input_path,
            '-c:v', 'libx264',
            '-preset', 'fast',
            '-crf', '23',
            '-pix_fmt', 'yuv420p',
            '-movflags', '+faststart',
            output_path
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        
        if output_path != input_path:
            os.remove(input_path)
            os.rename(output_path, input_path)
        
        print(f"Đã chuyển đổi video sang H.264: {input_path}")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"Không thể dùng ffmpeg, giữ nguyên video gốc: {e}")
        return False

def create_video_writer(output_path, fps, width, height):
    """Tạo video writer với codec tương thích nhất"""
    codecs = [
        ('avc1', 'mp4'),
        ('mp4v', 'mp4'),
        ('XVID', 'avi'),
    ]
    
    for codec, ext in codecs:
        try:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            test_path = output_path if ext == 'mp4' else output_path.replace('.mp4', f'.{ext}')
            writer = cv2.VideoWriter(test_path, fourcc, fps, (width, height))
            if writer.isOpened():
                print(f"Sử dụng codec: {codec}")
                return writer, test_path
            writer.release()
        except Exception as e:
            print(f"Codec {codec} không khả dụng: {e}")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    return cv2.VideoWriter(output_path, fourcc, fps, (width, height)), output_path

@app.route('/favicon.ico')
def favicon():
    if not os.path.exists(os.path.join(app.root_path, 'static', 'images', 'favicon.ico')):
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
        return jsonify({'error': 'Định dạng file không được hỗ trợ'}), 400
    
    session_id = str(uuid.uuid4())
    
    img0_path = os.path.join(app.config['UPLOAD_FOLDER'], f'{session_id}_0.png')
    img1_path = os.path.join(app.config['UPLOAD_FOLDER'], f'{session_id}_1.png')
    
    # Bắt đầu đo thời gian tổng
    total_start_time = time.time()
    
    try:
        # Đo thời gian upload/save
        upload_start = time.time()
        image1.save(img0_path)
        image2.save(img1_path)
        upload_time = time.time() - upload_start
        
        img0_bgr = cv2.imread(img0_path)
        img1_bgr = cv2.imread(img1_path)
        
        similarity = compute_ssim(img0_bgr, img1_bgr)
        
        SSIM_THRESHOLD = 0.3
        force_process = request.form.get('force_process', 'false') == 'true'
        
        h0, w0 = img0_bgr.shape[:2]
        h1, w1 = img1_bgr.shape[:2]
        need_resize = (h0 != h1) or (w0 != w1)
        
        if (similarity < SSIM_THRESHOLD or need_resize) and not force_process:
            warning_msg = []
            if similarity < SSIM_THRESHOLD:
                warning_msg.append(f"Hai ảnh có độ tương đồng thấp (SSIM = {similarity:.2f}), có thể không phải frames liên tiếp.")
            if need_resize:
                warning_msg.append(f"Hai ảnh có kích thước khác nhau ({w0}x{h0} vs {w1}x{h1}).")
            
            return jsonify({
                'warning': True,
                'message': ' '.join(warning_msg) + ' Bạn có muốn tiếp tục xử lý?',
                'similarity': float(similarity),
                'need_resize': need_resize,
                'original_sizes': f"{w0}x{h0} và {w1}x{h1}" if need_resize else None,
                'resized_to': f"{min(w0,w1)}x{min(h0,h1)}" if need_resize else None
            })
        
        # Đo thời gian preprocessing (resize)
        preprocess_start = time.time()
        # Resize nếu cần
        if need_resize:
            img0_bgr, img1_bgr, final_w, final_h = resize_images_for_interpolation(img0_bgr, img1_bgr)
            cv2.imwrite(img0_path, img0_bgr)
            cv2.imwrite(img1_path, img1_bgr)
        else:
            final_h, final_w = img0_bgr.shape[:2]
        preprocess_time = time.time() - preprocess_start
        
        num_frames = int(request.form.get('frames', 16))
        selected_model = request.form.get('model', 'pretrained')
        
        if selected_model not in MODEL_PATHS:
            selected_model = 'pretrained'
        
        # Đo thời gian load model
        model_load_start = time.time()
        model = load_bim_vfi_model(selected_model)
        model_load_time = time.time() - model_load_start
        
        if model is None:
            return jsonify({'error': f'Không thể load model: {selected_model}'}), 500
        
        # Đo thời gian inference (quan trọng nhất)
        inference_start = time.time()
        frames = generate_interpolation(model, img0_path, img1_path, num_frames)
        inference_time = time.time() - inference_start

        # Đo thời gian postprocessing (save frames + create video)
        postprocess_start = time.time()

        frame_paths = []
        for i, frame in enumerate(frames):
            frame_path = os.path.join(app.config['RESULT_FOLDER'], f'{session_id}_{i}.png')
            cv2.imwrite(frame_path, frame)
            frame_paths.append(os.path.join('static', 'results', f'{session_id}_{i}.png'))
        
        # Tạo video với codec tương thích
        video_path = os.path.join(app.config['RESULT_FOLDER'], f'{session_id}.mp4')
        h, w = frames[0].shape[:2]
        
        video, actual_path = create_video_writer(video_path, 24, w, h)
        
        img0_final = cv2.imread(img0_path)
        img0_final = cv2.resize(img0_final, (w, h))
        video.write(img0_final)
        
        for frame in frames:
            video.write(frame)
        
        img1_final = cv2.imread(img1_path)
        img1_final = cv2.resize(img1_final, (w, h))
        video.write(img1_final)
        
        video.release()
        
        # Chuyển đổi sang H.264 nếu cần
        if actual_path != video_path:
            os.rename(actual_path, video_path)
        convert_to_web_compatible(video_path)

        postprocess_time = time.time() - postprocess_start

        # Tính tổng thời gian
        total_time = time.time() - total_start_time

        # Tính FPS inference
        fps_inference = num_frames / inference_time if inference_time > 0 else 0
        
        return jsonify({
            'success': True, 
            'frames': frame_paths,
            'video': f'/video/{session_id}.mp4',
            'download_video': os.path.join('static', 'results', f'{session_id}.mp4'),
            'original1': os.path.join('static', 'uploads', f'{session_id}_0.png'),
            'original2': os.path.join('static', 'uploads', f'{session_id}_1.png'),
            'similarity': float(similarity),
            'final_size': f"{w}x{h}",
            'timing': {
                'upload_time': round(upload_time, 3),
                'preprocess_time': round(preprocess_time, 3),
                'model_load_time': round(model_load_time, 3),
                'inference_time': round(inference_time, 3),
                'postprocess_time': round(postprocess_time, 3),
                'total_time': round(total_time, 3),
                'fps_inference': round(fps_inference, 2),
                'time_per_frame': round(inference_time / num_frames, 3) if num_frames > 0 else 0
            }
        })
    
    except Exception as e:
        print(f"Lỗi xử lý: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/interpolate_video', methods=['POST'])
def interpolate_video():
    if 'video' not in request.files:
        return jsonify({'error': 'Thiếu file video'}), 400
    
    video_file = request.files['video']
    
    if video_file.filename == '':
        return jsonify({'error': 'Không có file được chọn'}), 400
    
    if not allowed_video(video_file.filename):
        return jsonify({'error': 'Định dạng video không được hỗ trợ'}), 400
    
    session_id = str(uuid.uuid4())

    # Bắt đầu đo thời gian tổng
    total_start_time = time.time()
    
    try:
        # Đo thời gian upload
        upload_start = time.time()
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], f'{session_id}_input.mp4')
        video_file.save(video_path)
        upload_time = time.time() - upload_start
        
        num_interpolations = int(request.form.get('interpolations', 2))
        output_fps = int(request.form.get('fps', 30))
        selected_model = request.form.get('model', 'pretrained')
        
        if selected_model not in MODEL_PATHS:
            selected_model = 'pretrained'
        # Đo thời gian load model
        model_load_start = time.time()
        model = load_bim_vfi_model(selected_model)
        model_load_time = time.time() - model_load_start
        
        if model is None:
            return jsonify({'error': f'Không thể load model: {selected_model}'}), 500
        # Đo thời gian extract frames
        extract_start = time.time()
        print("Đang trích xuất frames từ video...")
        frames = extract_frames_from_video(video_path)
        extract_time = time.time() - extract_start
        
        if len(frames) == 0:
            return jsonify({'error': 'Không thể trích xuất frames từ video'}), 400
        
        print(f"Đã trích xuất {len(frames)} frames")

        # Đo thời gian dedup
        dedup_start = time.time()
        print("Đang loại bỏ frames trùng lặp...")
        unique_frames = remove_duplicate_frames(frames)
        dedup_time = time.time() - dedup_start
        print(f"Còn lại {len(unique_frames)} frames sau khi loại bỏ trùng lặp")
        
        if len(unique_frames) < 2:
            return jsonify({'error': 'Video cần có ít nhất 2 khung hình khác nhau'}), 400
        
        # Đo thời gian inference
        inference_start = time.time()
        print(f"Đang nội suy với {num_interpolations} frames trung gian...")
        interpolated_frames = interpolate_video_frames(model, unique_frames, num_interpolations)
        inference_time = time.time() - inference_start
        print(f"Đã tạo {len(interpolated_frames)} frames sau nội suy")
        
        # Đo thời gian postprocess (tạo video)
        postprocess_start = time.time()
        output_video_path = os.path.join(app.config['RESULT_FOLDER'], f'{session_id}_output.mp4')
        h, w = interpolated_frames[0].shape[:2]
        
        out_video, actual_path = create_video_writer(output_video_path, output_fps, w, h)
        
        for frame in interpolated_frames:
            out_video.write(frame)
        
        out_video.release()
        
        # Chuyển đổi sang H.264 nếu cần
        if actual_path != output_video_path:
            os.rename(actual_path, output_video_path)
        convert_to_web_compatible(output_video_path)
        
        sample_indices = [0, len(interpolated_frames)//4, len(interpolated_frames)//2, 
                         3*len(interpolated_frames)//4, len(interpolated_frames)-1]
        sample_paths = []
        
        for idx in sample_indices:
            if idx < len(interpolated_frames):
                sample_path = os.path.join(app.config['RESULT_FOLDER'], f'{session_id}_sample_{idx}.png')
                cv2.imwrite(sample_path, interpolated_frames[idx])
                sample_paths.append(os.path.join('static', 'results', f'{session_id}_sample_{idx}.png'))
        postprocess_time = time.time() - postprocess_start

        # Tính tổng thời gian
        total_time = time.time() - total_start_time

        # Tính số frame được nội suy (không tính frame gốc)
        total_interpolated = len(interpolated_frames) - len(unique_frames)
        fps_inference = total_interpolated / inference_time if inference_time > 0 else 0
        return jsonify({
            'success': True,
            'video': f'/video/{session_id}_output.mp4',
            'download_video': os.path.join('static', 'results', f'{session_id}_output.mp4'),
            'samples': sample_paths,
            'stats': {
                'original_frames': len(frames),
                'unique_frames': len(unique_frames),
                'final_frames': len(interpolated_frames),
                'fps': output_fps
            },
            'timing': {
                'upload_time': round(upload_time, 3),
                'model_load_time': round(model_load_time, 3),
                'extract_time': round(extract_time, 3),
                'dedup_time': round(dedup_time, 3),
                'inference_time': round(inference_time, 3),
                'postprocess_time': round(postprocess_time, 3),
                'total_time': round(total_time, 3),
                'fps_inference': round(fps_inference, 2),
                'avg_time_per_pair': round(inference_time / (len(unique_frames) - 1), 3) if len(unique_frames) > 1 else 0
            }
        })
    
    except Exception as e:
        print(f"Lỗi xử lý video: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/extract_frames_from_video', methods=['POST'])
def extract_frames_from_video_endpoint():
    """Endpoint để trích xuất frames từ video và lưu vào folder"""
    if 'video' not in request.files:
        return jsonify({'error': 'Thiếu file video'}), 400
    
    video_file = request.files['video']
    
    if video_file.filename == '':
        return jsonify({'error': 'Không có file được chọn'}), 400
    
    if not allowed_video(video_file.filename):
        return jsonify({'error': 'Định dạng video không được hỗ trợ'}), 400
    
    session_id = str(uuid.uuid4())
    
    try:
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], f'{session_id}_extract.mp4')
        video_file.save(video_path)
        
        # Tạo folder để lưu frames
        frames_folder = os.path.join(app.config['RESULT_FOLDER'], f'frames_{session_id}')
        os.makedirs(frames_folder, exist_ok=True)
        
        # Trích xuất frames
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_path = os.path.join(frames_folder, f'frame_{frame_count:06d}.png')
            cv2.imwrite(frame_path, frame)
            frame_count += 1
        
        cap.release()
        
        return jsonify({
            'success': True,
            'total_frames': frame_count,
            'fps': fps,
            'frames_folder': frames_folder
        })
    
    except Exception as e:
        print(f"Lỗi trích xuất frames: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/interpolate_frame_sequence', methods=['POST'])
def interpolate_frame_sequence():
    """Endpoint để nội suy chuỗi frames"""
    if 'frames' not in request.files:
        return jsonify({'error': 'Thiếu frames đầu vào'}), 400
    
    uploaded_frames = request.files.getlist('frames')
    
    if len(uploaded_frames) < 2:
        return jsonify({'error': 'Cần ít nhất 2 frames'}), 400
    
    session_id = str(uuid.uuid4())
    
    try:
        # Lưu và đọc các frames
        input_frames = []
        for idx, frame_file in enumerate(uploaded_frames):
            if not allowed_file(frame_file.filename):
                continue
            
            frame_path = os.path.join(app.config['UPLOAD_FOLDER'], f'{session_id}_seq_{idx}.png')
            frame_file.save(frame_path)
            
            frame = cv2.imread(frame_path)
            if frame is not None:
                input_frames.append(frame)
        
        if len(input_frames) < 2:
            return jsonify({'error': 'Cần ít nhất 2 frames hợp lệ'}), 400
        
        # Lấy params
        num_interpolations = int(request.form.get('interpolations', 2))
        output_fps = int(request.form.get('fps', 30))
        selected_model = request.form.get('model', 'pretrained')
        
        if selected_model not in MODEL_PATHS:
            selected_model = 'pretrained'
        
        model = load_bim_vfi_model(selected_model)
        
        if model is None:
            return jsonify({'error': f'Không thể load model: {selected_model}'}), 500
        
        # Nội suy giữa các cặp frames liên tiếp
        print(f"Đang nội suy {len(input_frames)} frames...")
        interpolated_frames = interpolate_video_frames(model, input_frames, num_interpolations)
        print(f"Đã tạo {len(interpolated_frames)} frames sau nội suy")
        
        # Tạo video
        output_video_path = os.path.join(app.config['RESULT_FOLDER'], f'{session_id}_sequence.mp4')
        h, w = interpolated_frames[0].shape[:2]
        
        out_video, actual_path = create_video_writer(output_video_path, output_fps, w, h)
        
        for frame in interpolated_frames:
            out_video.write(frame)
        
        out_video.release()
        
        # Chuyển đổi sang H.264 nếu cần
        if actual_path != output_video_path:
            os.rename(actual_path, output_video_path)
        convert_to_web_compatible(output_video_path)
        
        # Lưu một số samples
        sample_indices = [0, len(interpolated_frames)//4, len(interpolated_frames)//2, 
                         3*len(interpolated_frames)//4, len(interpolated_frames)-1]
        sample_paths = []
        
        for idx in sample_indices:
            if idx < len(interpolated_frames):
                sample_path = os.path.join(app.config['RESULT_FOLDER'], f'{session_id}_seq_sample_{idx}.png')
                cv2.imwrite(sample_path, interpolated_frames[idx])
                sample_paths.append(os.path.join('static', 'results', f'{session_id}_seq_sample_{idx}.png'))
        
        return jsonify({
            'success': True,
            'video': f'/video/{session_id}_sequence.mp4',
            'download_video': os.path.join('static', 'results', f'{session_id}_sequence.mp4'),
            'samples': sample_paths,
            'stats': {
                'input_frames': len(input_frames),
                'output_frames': len(interpolated_frames),
                'fps': output_fps
            }
        })
    
    except Exception as e:
        print(f"Lỗi nội suy chuỗi frames: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/prepare_comparison', methods=['POST'])
def prepare_comparison():
    if 'video1' not in request.files or 'video2' not in request.files:
        return jsonify({'error': 'Thiếu file video'}), 400
    
    video1_file = request.files['video1']
    video2_file = request.files['video2']
    
    if video1_file.filename == '' or video2_file.filename == '':
        return jsonify({'error': 'Không có file được chọn'}), 400
    
    if not (allowed_video(video1_file.filename) and allowed_video(video2_file.filename)):
        return jsonify({'error': 'Định dạng video không được hỗ trợ'}), 400
    
    session_id = str(uuid.uuid4())
    
    try:
        video1_path = os.path.join(app.config['UPLOAD_FOLDER'], f'{session_id}_compare1.mp4')
        video2_path = os.path.join(app.config['UPLOAD_FOLDER'], f'{session_id}_compare2.mp4')
        
        video1_file.save(video1_path)
        video2_file.save(video2_path)
        
        cap1 = cv2.VideoCapture(video1_path)
        cap2 = cv2.VideoCapture(video2_path)
        
        fps1 = cap1.get(cv2.CAP_PROP_FPS)
        fps2 = cap2.get(cv2.CAP_PROP_FPS)
        frames1 = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))
        frames2 = int(cap2.get(cv2.CAP_PROP_FRAME_COUNT))
        width1 = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
        height1 = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width2 = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
        height2 = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        cap1.release()
        cap2.release()
        
        return jsonify({
            'success': True,
            'video1': f'/comparison_video/{session_id}_compare1.mp4',
            'video2': f'/comparison_video/{session_id}_compare2.mp4',
            'stats': {
                'video1': {
                    'fps': round(fps1, 2),
                    'frames': frames1,
                    'resolution': f'{width1}x{height1}'
                },
                'video2': {
                    'fps': round(fps2, 2),
                    'frames': frames2,
                    'resolution': f'{width2}x{height2}'
                }
            }
        })
    
    except Exception as e:
        print(f"Lỗi xử lý so sánh video: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/comparison_video/<path:filename>')
def serve_comparison_video(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, 
                              mimetype='video/mp4', as_attachment=False)

@app.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory('static', path)

@app.route('/video/<path:filename>')
def serve_video(filename):
    video_path = os.path.join(app.config['RESULT_FOLDER'], filename)
    
    # Kiểm tra file tồn tại
    if not os.path.exists(video_path):
        return jsonify({'error': 'Video không tồn tại'}), 404
    
    response = send_from_directory(
        app.config['RESULT_FOLDER'], 
        filename, 
        mimetype='video/mp4'
    )
    response.headers['Content-Type'] = 'video/mp4'
    response.headers['Accept-Ranges'] = 'bytes'
    response.headers['Cache-Control'] = 'no-cache'
    return response

@app.route('/gallery_add', methods=['POST'])
def gallery_add():
    if 'original' not in request.files or 'bimvfi' not in request.files:
        return jsonify({'error': 'Thiếu video gốc hoặc video BiM-VFI'}), 400
    
    original_file = request.files['original']
    bimvfi_file = request.files['bimvfi']
    other_file = request.files.get('other')
    
    if original_file.filename == '' or bimvfi_file.filename == '':
        return jsonify({'error': 'Không có file được chọn'}), 400
    
    if not (allowed_video(original_file.filename) and allowed_video(bimvfi_file.filename)):
        return jsonify({'error': 'Định dạng video không được hỗ trợ'}), 400
    
    if other_file and other_file.filename != '' and not allowed_video(other_file.filename):
        return jsonify({'error': 'Định dạng video model khác không được hỗ trợ'}), 400
    
    session_id = str(uuid.uuid4())
    
    try:
        title = request.form.get('title', 'Không có tiêu đề')
        layout = request.form.get('layout', 'row')
        
        original_path = os.path.join(app.config['UPLOAD_FOLDER'], f'{session_id}_gallery_original.mp4')
        bimvfi_path = os.path.join(app.config['UPLOAD_FOLDER'], f'{session_id}_gallery_bimvfi.mp4')
        
        original_file.save(original_path)
        bimvfi_file.save(bimvfi_path)
        
        other_path = None
        if other_file and other_file.filename != '':
            other_path = os.path.join(app.config['UPLOAD_FOLDER'], f'{session_id}_gallery_other.mp4')
            other_file.save(other_path)
        
        result = {
            'id': session_id,
            'title': title,
            'layout': layout,
            'videos': {
                'original': f'/comparison_video/{session_id}_gallery_original.mp4',
                'bimvfi': f'/comparison_video/{session_id}_gallery_bimvfi.mp4',
                'other': f'/comparison_video/{session_id}_gallery_other.mp4' if other_path else None
            }
        }
        
        return jsonify(result)
    
    except Exception as e:
        print(f"Lỗi thêm vào trưng bày: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)