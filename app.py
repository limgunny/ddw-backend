# backend/app.py

import os
import subprocess
import secrets
import cv2
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_jwt_extended import (
    create_access_token,
    create_refresh_token,
    jwt_required,
    JWTManager,
    get_jwt_identity
)
from datetime import datetime, timezone
import time
import base64
from cryptography.fernet import Fernet, InvalidToken
import json
import cloudinary
import cloudinary.uploader
from functools import wraps
import hashlib
from dotenv import load_dotenv
import threading
import uuid

# 환경 변수 로드
load_dotenv()

# --- 성능 최적화를 위한 캐시 시스템 ---
class SimpleCache:
    def __init__(self, max_size=100, ttl=300):  # 5분 TTL
        self.cache = {}
        self.max_size = max_size
        self.ttl = ttl
    
    def get(self, key):
        if key in self.cache:
            data, timestamp = self.cache[key]
            if time.time() - timestamp < self.ttl:
                return data
            else:
                del self.cache[key]
        return None
    
    def set(self, key, value):
        if len(self.cache) >= self.max_size:
            # 가장 오래된 항목 제거
            oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k][1])
            del self.cache[oldest_key]
        
        self.cache[key] = (value, time.time())
    
    def clear(self):
        self.cache.clear()

# 전역 캐시 인스턴스
cache = SimpleCache()

# 비동기 작업 관리 시스템
class AsyncTaskManager:
    def __init__(self):
        self.tasks = {}
        self.lock = threading.Lock()
    
    def create_task(self, task_id, task_func, *args, **kwargs):
        """새로운 비동기 작업 생성"""
        with self.lock:
            self.tasks[task_id] = {
                'status': 'running',
                'progress': 0,
                'result': None,
                'error': None,
                'created_at': datetime.now(timezone.utc)
            }
        
        def task_wrapper():
            try:
                result = task_func(*args, **kwargs)
                with self.lock:
                    self.tasks[task_id]['status'] = 'completed'
                    self.tasks[task_id]['progress'] = 100
                    self.tasks[task_id]['result'] = result
            except Exception as e:
                with self.lock:
                    self.tasks[task_id]['status'] = 'failed'
                    self.tasks[task_id]['error'] = str(e)
        
        thread = threading.Thread(target=task_wrapper)
        thread.daemon = True
        thread.start()
        return task_id
    
    def get_task_status(self, task_id):
        """작업 상태 조회"""
        with self.lock:
            return self.tasks.get(task_id, None)
    
    def cleanup_old_tasks(self, max_age_hours=24):
        """오래된 작업 정리"""
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)
        with self.lock:
            to_remove = [
                task_id for task_id, task in self.tasks.items()
                if task['created_at'] < cutoff_time
            ]
            for task_id in to_remove:
                del self.tasks[task_id]

# 전역 작업 관리자
task_manager = AsyncTaskManager()

# JWT 토큰 검증 헬퍼 함수
def verify_token_or_refresh():
    """
    토큰이 유효한지 확인하고, 만료된 경우 갱신을 시도합니다.
    """
    try:
        # 현재 토큰이 유효한지 확인
        get_jwt_identity()
        return True, None
    except Exception as e:
        # 토큰이 만료되었거나 유효하지 않은 경우
        return False, str(e)

def create_token_response(user_data):
    """
    사용자 데이터로부터 토큰 응답을 생성합니다.
    """
    identity_data = {'username': user_data.username, 'id': user_data.id, 'role': user_data.role}
    access_token = create_access_token(identity=json.dumps(identity_data))
    refresh_token = create_refresh_token(identity=json.dumps(identity_data))
    return jsonify({
        'access_token': access_token, 
        'refresh_token': refresh_token,
        'expires_in': 4 * 60 * 60  # 4시간을 초 단위로
    })

def async_watermark_task(input_path, master_path, embeddable_watermark, task_id):
    """
    비동기 워터마킹 작업 함수
    """
    def progress_callback(progress):
        with task_manager.lock:
            if task_id in task_manager.tasks:
                task_manager.tasks[task_id]['progress'] = progress
    
    try:
        success, message = embed_watermark(input_path, master_path, embeddable_watermark, progress_callback)
        return {
            'success': success,
            'message': message,
            'master_path': master_path if success else None
        }
    except Exception as e:
        return {
            'success': False,
            'message': f'워터마킹 작업 중 오류 발생: {str(e)}',
            'master_path': None
        }

def cached_response(ttl=300):
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            # 캐시 키 생성
            cache_key = f"{f.__name__}_{hashlib.md5(str(args).encode()).hexdigest()}"
            
            # 캐시에서 확인
            cached_result = cache.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # 캐시에 없으면 함수 실행
            result = f(*args, **kwargs)
            
            # 결과를 캐시에 저장
            cache.set(cache_key, result)
            
            return result
        return decorated_function
    return decorator

# --- 기존 유틸리티 함수 (변경 없음) ---
def text_to_binary(text):
    return ''.join(format(ord(char), '08b') for char in text)

def binary_to_text(binary_string):
    chars = [binary_string[i:i+8] for i in range(0, len(binary_string), 8)]
    text = []
    for char_bin in chars:
        if len(char_bin) == 8:
            try:
                text.append(chr(int(char_bin, 2)))
            except ValueError:
                pass
    return "".join(text)

# --- Flask 애플리케이션 설정 ---
app = Flask(__name__)
# [수정] CORS 설정을 강화하여 인증 헤더(Authorization)를 포함한 요청을 허용합니다.
# supports_credentials=True 옵션이 없으면 브라우저가 Authorization 헤더가 포함된 요청을 차단합니다.
CORS(app, resources={
    r"/api/*": {
        "origins": [
            "http://localhost:3000", 
            "http://127.0.0.1:3000",
            "https://ddw-frontend-zpvi.vercel.app",
            "https://ddw-backend.onrender.com"
        ], 
        "supports_credentials": True,
        "allow_headers": ["Content-Type", "Authorization"] # [추가] DELETE와 같은 요청에 포함된 Authorization 헤더를 허용
    }
})

# [추가] 데이터베이스 및 인증 설정
# [수정] 배포 환경에서는 환경변수에서 시크릿 키를 가져옵니다.
SECRET_KEY = os.getenv('SECRET_KEY', 'a-super-secret-and-static-key-for-development-only')
app.config['SECRET_KEY'] = SECRET_KEY
app.config['JWT_SECRET_KEY'] = SECRET_KEY
# 토큰 만료 시간 설정 (접근 4시간, 리프레시 30일)
from datetime import timedelta
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(hours=4)  # 60분에서 4시간으로 연장
app.config['JWT_REFRESH_TOKEN_EXPIRES'] = timedelta(days=30)

# [성능 최적화] 데이터베이스 설정 개선
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {
    'pool_pre_ping': True,
    'pool_recycle': 300,
    'connect_args': {'check_same_thread': False}
}

db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
jwt = JWTManager(app)

# [추가] 암호화 설정
# Fernet 키는 32바이트여야 합니다. SECRET_KEY를 기반으로 일관된 키를 생성합니다.
# 키가 32바이트보다 길 경우 잘라내고, 짧을 경우 패딩합니다.
key_32_bytes = SECRET_KEY.encode().ljust(32)[:32]
fernet_key = base64.urlsafe_b64encode(key_32_bytes)
cipher_suite = Fernet(fernet_key)

# [추가] 사용자 모델 정의
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), unique=True, nullable=False)
    password = db.Column(db.String(60), nullable=False)
    role = db.Column(db.String(20), nullable=False, default='user') # [추가] 사용자 역할 (user, admin)

    def __repr__(self):
        return f"User('{self.username}')"

    def to_dict(self):
        return {
            'id': self.id,
            'username': self.username,
            'role': self.role
        }

# [추가] 비디오 모델 정의
class Video(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(150), nullable=True)
    original_filename = db.Column(db.String(100), nullable=False)
    master_filename = db.Column(db.String(100), unique=True, nullable=False)
    playback_filename = db.Column(db.String(100), unique=True, nullable=False)
    upload_timestamp = db.Column(db.DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))
    views = db.Column(db.Integer, nullable=False, default=0)
    thumbnail_filename = db.Column(db.String(150), nullable=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    user = db.relationship('User', backref=db.backref('videos', lazy=True, cascade="all, delete-orphan"))

    def to_dict(self):
        return {
            'id': self.id,
            'title': self.title,
            'original_filename': self.original_filename,
            'playback_filename': self.playback_filename,
            'upload_timestamp': self.upload_timestamp.isoformat(),
            'views': self.views,
            'thumbnail_filename': self.thumbnail_filename,
            'user': {'username': self.user.username, 'id': self.user.id} # [추가] 비디오를 업로드한 사용자 정보
        }

# [추가] 앱 시작 시 데이터베이스 테이블 자동 생성
# 이 코드는 매번 서버를 시작할 때마다 데이터베이스와 테이블이 존재하는지 확인하고,
# 없으면 자동으로 생성해줍니다. 수동으로 db.create_all()을 실행할 필요가 없습니다.
with app.app_context():
    db.create_all()
    # 부족한 컬럼 자동 추가 (SQLite 전용 간단 처리)
    try:
        info = db.session.execute(db.text("PRAGMA table_info(video)")).fetchall()
        columns = {row[1] for row in info}
        if 'title' not in columns:
            db.session.execute(db.text("ALTER TABLE video ADD COLUMN title VARCHAR(150)"))
        if 'views' not in columns:
            db.session.execute(db.text("ALTER TABLE video ADD COLUMN views INTEGER NOT NULL DEFAULT 0"))
        if 'thumbnail_filename' not in columns:
            db.session.execute(db.text("ALTER TABLE video ADD COLUMN thumbnail_filename VARCHAR(150)"))
        db.session.commit()
    except Exception as e:
        app.logger.error(f"DB migration check failed: {e}")

UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

# 폴더가 없으면 생성
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Cloudinary 설정 (환경변수에서 읽기)
CLOUDINARY_CLOUD_NAME = os.getenv('CLOUDINARY_CLOUD_NAME')
CLOUDINARY_API_KEY = os.getenv('CLOUDINARY_API_KEY')
CLOUDINARY_API_SECRET = os.getenv('CLOUDINARY_API_SECRET')
if CLOUDINARY_CLOUD_NAME and CLOUDINARY_API_KEY and CLOUDINARY_API_SECRET:
    cloudinary.config(
        cloud_name=CLOUDINARY_CLOUD_NAME,
        api_key=CLOUDINARY_API_KEY,
        api_secret=CLOUDINARY_API_SECRET,
        secure=True
    )

# --- 기존 함수를 API에 맞게 수정 ---
def embed_watermark(input_path, output_path, watermark_text, progress_callback=None):
    """
    워터마크 삽입 함수 - 성능 최적화 버전
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        return False, f"Error: '{input_path}' 동영상을 열 수 없습니다."

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # [성능 최적화] 더 빠른 코덱 사용 (H.264)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    watermark_binary = text_to_binary(watermark_text + '$$END$$')
    watermark_len = len(watermark_binary)

    # 워터마크 크기 검증
    available_bits = width * height
    if watermark_len > available_bits:
        cap.release()
        out.release()
        return False, f"Error: 워터마크가 너무 깁니다. (최대 {available_bits} 비트, 현재 {watermark_len} 비트)"

    frame_count = 0
    watermark_inserted = False
    
    # [성능 최적화] 배치 처리로 프레임 읽기 최적화
    batch_size = 10
    frames_buffer = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frames_buffer.append(frame)
        
        # 배치가 가득 찼거나 마지막 프레임인 경우 처리
        if len(frames_buffer) >= batch_size or frame_count == total_frames - 1:
            for i, buffered_frame in enumerate(frames_buffer):
                # 첫 번째 프레임에만 워터마크를 삽입합니다.
                if frame_count - len(frames_buffer) + i == 0 and not watermark_inserted:
                    # [성능 최적화] 벡터화된 연산 사용
                    watermark_inserted = True
                    watermark_insertion_optimized(buffered_frame, watermark_binary, width, height)
                
                out.write(buffered_frame)
            
            frames_buffer.clear()
            
            # 진행률 콜백 호출
            if progress_callback and total_frames > 0:
                progress = min(100, int((frame_count + 1) / total_frames * 100))
                progress_callback(progress)
        
        frame_count += 1
    
    cap.release()
    out.release()
    return True, "워터마크 삽입 완료"

def watermark_insertion_optimized(frame, watermark_binary, width, height):
    """
    워터마크 삽입 최적화 함수 - 벡터화된 연산 사용
    """
    watermark_len = len(watermark_binary)
    
    # [성능 최적화] NumPy를 사용한 벡터화된 연산
    frame_flat = frame.reshape(-1, 3)  # (height*width, 3)
    
    # 워터마크 비트를 배열로 변환
    watermark_bits = np.array([int(bit) for bit in watermark_binary[:watermark_len]])
    
    # 파딩이 필요한 경우 0으로 채움
    if len(watermark_bits) < len(frame_flat):
        padding = np.zeros(len(frame_flat) - len(watermark_bits), dtype=int)
        watermark_bits = np.concatenate([watermark_bits, padding])
    
    # LSB 방식으로 워터마크 삽입 (벡터화된 연산)
    frame_flat[:len(watermark_bits), 0] = (frame_flat[:len(watermark_bits), 0] & 0b11111110) | watermark_bits
    
    # 원래 형태로 복원
    frame[:] = frame_flat.reshape(height, width, 3)

def extract_watermark(input_path):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        return f"Error: '{input_path}' 동영상을 열 수 없습니다."

    ret, frame = cap.read()
    if not ret:
        cap.release()
        return "Error: 동영상에서 프레임을 읽을 수 없습니다."

    height, width, _ = frame.shape
    extracted_binary = ""
    terminator_binary = text_to_binary("$$END$$")

    # [수정] 프레임의 첫 번째 행만 읽던 것에서 프레임 전체를 순회하며 워터마크 비트를 추출하도록 변경합니다.
    for i in range(height * width):
        row = i // width
        col = i % width
        pixel = frame[row, col]
        blue_val = pixel[0]
        # [수정] 삽입 방식과 동일하게 최하위 비트(LSB)에서 데이터를 추출합니다.
        lsb = blue_val & 1
        extracted_binary += str(lsb)
        # [추가] 종료 문자를 찾으면 더 이상 읽지 않고 중단하여 효율성을 증대시킵니다.
        if extracted_binary.endswith(terminator_binary):
            break

    extracted_text = binary_to_text(extracted_binary)
    terminator = "$$END$$"
    if terminator in extracted_text:
        watermark = extracted_text.split(terminator)[0]
    else:
        watermark = "워터마크를 찾지 못했거나 데이터가 손상되었습니다."
    
    cap.release()
    return watermark

# --- [추가] 인증 API 엔드포인트 ---
@app.route('/api/signup', methods=['POST'])
def signup():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')

    if not username or not password:
        return jsonify({'error': '사용자 이름과 비밀번호를 모두 입력해주세요.'}), 400

    if User.query.filter_by(username=username).first():
        return jsonify({'error': '이미 존재하는 사용자 이름입니다.'}), 409

    # [추가] 첫 번째 가입자를 admin으로 자동 지정
    role = 'user'
    if User.query.first() is None:
        app.logger.info(f"First user '{username}' is being set as admin.")
        role = 'admin'

    hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
    user = User(username=username, password=hashed_password, role=role)
    db.session.add(user)
    db.session.commit()
    return jsonify({'message': '회원가입 성공!'}), 201

@app.route('/api/login', methods=['POST'])
def login():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    user = User.query.filter_by(username=username).first()

    if user and bcrypt.check_password_hash(user.password, password):
        # [수정] JWT 토큰에 역할(role) 정보도 포함시킴
        identity_data = {'username': user.username, 'id': user.id, 'role': user.role}
        access_token = create_access_token(identity=json.dumps(identity_data))
        refresh_token = create_refresh_token(identity=json.dumps(identity_data))
        return jsonify({
            'access_token': access_token, 
            'refresh_token': refresh_token,
            'expires_in': 4 * 60 * 60,  # 4시간을 초 단위로
            'token_type': 'Bearer'
        })

    return jsonify({'error': '사용자 이름 또는 비밀번호가 일치하지 않습니다.'}), 401

# 리프레시 토큰으로 액세스 토큰 재발급 (개선된 버전)
@app.route('/api/refresh', methods=['POST'])
@jwt_required(refresh=True)
def refresh_access_token():
    try:
        identity_json = get_jwt_identity()
        new_access_token = create_access_token(identity=identity_json)
        return jsonify({
            'access_token': new_access_token,
            'expires_in': 4 * 60 * 60,  # 4시간을 초 단위로
            'message': '토큰이 성공적으로 갱신되었습니다.'
        })
    except Exception as e:
        app.logger.error(f"Token refresh failed: {e}")
        return jsonify({'error': '토큰 갱신에 실패했습니다.'}), 401

# [추가] 프로필 관리 API - 비밀번호 변경
@app.route('/api/profile/change-password', methods=['POST'])
@jwt_required()
def change_password():
    identity_json = get_jwt_identity()
    current_user_data = json.loads(identity_json)
    user_id = current_user_data['id']
    
    data = request.get_json()
    current_password = data.get('current_password')
    new_password = data.get('new_password')

    if not current_password or not new_password:
        return jsonify({'error': '현재 비밀번호와 새 비밀번호를 모두 입력해주세요.'}), 400

    user = User.query.get(user_id)

    if not user or not bcrypt.check_password_hash(user.password, current_password):
        return jsonify({'error': '현재 비밀번호가 일치하지 않습니다.'}), 401

    user.password = bcrypt.generate_password_hash(new_password).decode('utf-8')
    db.session.commit()

    return jsonify({'message': '비밀번호가 성공적으로 변경되었습니다.'})

# --- API 엔드포인트 ---
@app.route('/api/embed', methods=['POST'])
@jwt_required()
def embed_route():
    identity_json = get_jwt_identity()
    current_user = json.loads(identity_json)
    original_watermark = f"user_id:{current_user['id']},username:{current_user['username']}"
    app.logger.info(f"Watermark embed request by {original_watermark}")

    # [추가] 워터마크 암호화
    encrypted_watermark_bytes = cipher_suite.encrypt(original_watermark.encode())
    # 암호화된 바이트를 URL-safe Base64 문자열로 변환하여 LSB 삽입에 사용
    embeddable_watermark = base64.urlsafe_b64encode(encrypted_watermark_bytes).decode('utf-8')

    if 'video' not in request.files:
        return jsonify({'error': '비디오 파일이 없습니다.'}), 400

    video_file = request.files['video']
    title = request.form.get('title')

    # 파일 유효성 검사
    if not video_file or video_file.filename == '':
        return jsonify({'error': '파일이 선택되지 않았습니다.'}), 400
    
    # 파일명 처리 (안전한 방식)
    try:
        if '.' not in video_file.filename:
            return jsonify({'error': '파일 확장자가 없습니다.'}), 400
        
        name, ext = video_file.filename.rsplit('.', 1)
        if not name or not ext:
            return jsonify({'error': '잘못된 파일명입니다.'}), 400
            
        # 고유한 파일명 생성
        base_filename = f"{int(time.time())}_{name}"
        filename = f"{base_filename}.{ext}"
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # 디렉토리 존재 확인 및 생성
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        
        # 파일 저장
        video_file.save(input_path)
        
        # 파일 저장 확인
        if not os.path.exists(input_path):
            return jsonify({'error': '파일 저장에 실패했습니다.'}), 500
            
    except Exception as e:
        app.logger.error(f"파일 처리 중 오류: {e}")
        return jsonify({'error': f'파일 처리 중 오류가 발생했습니다: {str(e)}'}), 500

    # [성능 최적화] 비동기 워터마킹 작업 시작
    try:
        task_id = str(uuid.uuid4())
        master_filename = f"watermarked_{base_filename}.mp4"  # mp4v 코덱 사용으로 확장자 변경
        master_path = os.path.join(app.config['OUTPUT_FOLDER'], master_filename)
        
        # 출력 디렉토리 생성
        os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)
        
        # 비동기 작업 시작
        task_manager.create_task(
            task_id, 
            async_watermark_task, 
            input_path, 
            master_path, 
            embeddable_watermark, 
            task_id
        )

        return jsonify({
            'task_id': task_id,
            'message': '워터마킹 작업이 시작되었습니다.',
            'status': 'processing'
        })
        
    except Exception as e:
        app.logger.error(f"워터마킹 작업 시작 중 오류: {e}")
        # 업로드된 파일 정리
        try:
            if os.path.exists(input_path):
                os.remove(input_path)
        except:
            pass
        return jsonify({'error': f'워터마킹 작업 시작에 실패했습니다: {str(e)}'}), 500

# 작업 상태 조회 API
@app.route('/api/embed/status/<task_id>', methods=['GET'])
@jwt_required()
def get_embed_status(task_id):
    task_status = task_manager.get_task_status(task_id)
    
    if not task_status:
        return jsonify({'error': '작업을 찾을 수 없습니다.'}), 404
    
    response = {
        'task_id': task_id,
        'status': task_status['status'],
        'progress': task_status['progress']
    }
    
    if task_status['status'] == 'completed':
        result = task_status['result']
        if result['success']:
            # 워터마킹 완료 후 추가 처리
            master_path = result['master_path']
            base_filename = os.path.splitext(os.path.basename(master_path))[0].replace('watermarked_', '')
            
            # 재생 가능한 MP4 파일 생성 및 썸네일 생성
            playback_filename = f"playback_{base_filename}.mp4"
            playback_path = os.path.join(app.config['OUTPUT_FOLDER'], playback_filename)
            thumbnail_filename = f"thumb_{base_filename}.jpg"
            thumbnail_path = os.path.join(app.config['OUTPUT_FOLDER'], thumbnail_filename)
            
            try:
                # FFmpeg를 사용하여 최적화된 MP4 변환
                subprocess.run([
                    'ffmpeg', '-i', master_path, 
                    '-c:v', 'libx264', '-preset', 'fast', '-crf', '23', 
                    '-c:a', 'aac', '-movflags', '+faststart',
                    '-y', playback_path
                ], check=True, capture_output=True, text=True, encoding='utf-8')
                
                # 썸네일 생성
                subprocess.run([
                    'ffmpeg', '-ss', '00:00:01.000', '-i', playback_path, 
                    '-vframes', '1', '-q:v', '2', '-y', thumbnail_path
                ], check=True, capture_output=True, text=True, encoding='utf-8')
                
                # 데이터베이스에 비디오 정보 저장
                identity_json = get_jwt_identity()
                current_user = json.loads(identity_json)
                
                new_video = Video(
                    title=request.args.get('title', ''),
                    original_filename=f"{base_filename}.mp4",
                    master_filename=os.path.basename(master_path),
                    playback_filename=playback_filename,
                    thumbnail_filename=thumbnail_filename,
                    user_id=current_user['id']
                )
                db.session.add(new_video)
                db.session.commit()
                
                # 캐시 무효화
                cache.clear()
                
                response.update({
                    'message': '워터마킹이 완료되었습니다.',
                    'playback_file': playback_filename,
                    'master_file': os.path.basename(master_path),
                    'video_id': new_video.id
                })
                
            except subprocess.CalledProcessError as e:
                app.logger.error(f"FFmpeg error: {e.stderr}")
                response.update({
                    'status': 'failed',
                    'error': f'비디오 변환 실패: {e.stderr}'
                })
            except Exception as e:
                app.logger.error(f"Post-processing error: {e}")
                response.update({
                    'status': 'failed',
                    'error': f'후처리 중 오류: {str(e)}'
                })
        else:
            response.update({
                'status': 'failed',
                'error': result['message']
            })
    elif task_status['status'] == 'failed':
        response['error'] = task_status['error']
    
    return jsonify(response)

@app.route('/api/my-videos', methods=['GET'])
@jwt_required()
def my_videos():
    identity_json = get_jwt_identity()
    current_user = json.loads(identity_json)
    
    # 페이지네이션 파라미터
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 20, type=int)
    per_page = min(per_page, 50)  # 최대 50개로 제한
    
    # 사용자별 비디오 조회 최적화
    user_videos_query = Video.query.filter_by(user_id=current_user['id']).order_by(Video.upload_timestamp.desc())
    user_videos = user_videos_query.paginate(
        page=page, 
        per_page=per_page, 
        error_out=False
    ).items
    
    return jsonify([video.to_dict() for video in user_videos])

# 공개 비디오 목록 API (최신 업로드 순) - 성능 최적화 + 캐싱
@app.route('/api/videos', methods=['GET'])
@cached_response(ttl=60)  # 1분 캐시
def public_videos():
    # 페이지네이션 파라미터
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 12, type=int)
    per_page = min(per_page, 50)  # 최대 50개로 제한
    
    # 데이터베이스 쿼리 최적화 - 페이지네이션 적용
    videos_query = Video.query.order_by(Video.upload_timestamp.desc())
    videos = videos_query.paginate(
        page=page, 
        per_page=per_page, 
        error_out=False
    ).items
    
    # 썸네일 누락 시 자동 생성 (비동기 처리로 변경)
    thumbnail_tasks = []
    for video in videos:
        if not video.thumbnail_filename:
            base_filename = os.path.splitext(video.playback_filename)[0]
            thumb_name = f"thumb_{base_filename}.jpg"
            thumb_path = os.path.join(app.config['OUTPUT_FOLDER'], thumb_name)
            playback_path = os.path.join(app.config['OUTPUT_FOLDER'], video.playback_filename)
            
            if os.path.exists(playback_path) and not os.path.exists(thumb_path):
                # 비동기로 썸네일 생성 (백그라운드에서 처리)
                try:
                    subprocess.Popen([
                        'ffmpeg', '-ss', '00:00:01.000', '-i', playback_path, 
                        '-vframes', '1', '-q:v', '2', '-y', thumb_path
                    ])
                    video.thumbnail_filename = thumb_name
                except Exception as e:
                    app.logger.error(f"Background thumbnail generation failed (video_id={video.id}): {e}")
    
    # 데이터베이스 업데이트는 배치로 처리
    try:
        db.session.commit()
    except Exception as e:
        app.logger.error(f"Commit thumbnails failed: {e}")
        db.session.rollback()
    
    return jsonify([video.to_dict() for video in videos])

# 조회수 증가 API
@app.route('/api/videos/<int:video_id>/view', methods=['POST'])
def increase_view(video_id):
    video = Video.query.get(video_id)
    if not video:
        return jsonify({'error': '비디오를 찾을 수 없습니다.'}), 404
    
    # 중복 조회 방지: 세션 기반으로 같은 사용자가 같은 비디오를 중복 조회하는 것을 방지
    session_key = f"viewed_video_{video_id}"
    client_ip = request.environ.get('HTTP_X_FORWARDED_FOR', request.environ.get('REMOTE_ADDR', ''))
    user_agent = request.environ.get('HTTP_USER_AGENT', '')
    
    # 고유한 식별자 생성 (IP + User-Agent 조합)
    unique_identifier = f"{client_ip}_{user_agent}"
    viewed_key = f"{session_key}_{hash(unique_identifier)}"
    
    # 세션에 이미 조회 기록이 있는지 확인
    if viewed_key in session:
        # 이미 조회한 경우 현재 조회수만 반환
        return jsonify({'views': video.views, 'already_viewed': True})
    
    try:
        # 조회수 증가
        video.views = (video.views or 0) + 1
        db.session.commit()
        
        # 세션에 조회 기록 저장 (24시간 유지)
        session[viewed_key] = True
        
        return jsonify({'views': video.views, 'already_viewed': False})
    except Exception as e:
        app.logger.error(f"Increase view failed: {e}")
        return jsonify({'error': '조회수 증가 실패'}), 500

# [추가] 관리자 전용 API - 모든 비디오 목록 조회 (성능 최적화)
@app.route('/api/admin/all-videos', methods=['GET'])
@jwt_required()
def all_videos():
    identity_json = get_jwt_identity()
    current_user = json.loads(identity_json)

    # 관리자 역할인지 확인
    if current_user.get('role') != 'admin':
        return jsonify({"msg": "관리자 권한이 필요합니다."}), 403

    # 페이지네이션 파라미터
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 20, type=int)
    per_page = min(per_page, 100)  # 관리자는 더 많이 볼 수 있음
    
    # 모든 비디오 조회 최적화
    all_vids_query = Video.query.order_by(Video.upload_timestamp.desc())
    all_vids = all_vids_query.paginate(
        page=page, 
        per_page=per_page, 
        error_out=False
    ).items
    
    return jsonify([video.to_dict() for video in all_vids])

# [추가] 관리자 전용 API - 모든 사용자 목록 조회 (성능 최적화)
@app.route('/api/admin/all-users', methods=['GET'])
@jwt_required()
def all_users():
    identity_json = get_jwt_identity()
    current_user = json.loads(identity_json)

    if current_user.get('role') != 'admin':
        return jsonify({"msg": "관리자 권한이 필요합니다."}), 403

    # 페이지네이션 파라미터
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 20, type=int)
    per_page = min(per_page, 100)  # 관리자는 더 많이 볼 수 있음
    
    # 사용자 목록 조회 최적화
    users_query = User.query.order_by(User.id.asc())
    users = users_query.paginate(
        page=page, 
        per_page=per_page, 
        error_out=False
    ).items
    
    return jsonify([user.to_dict() for user in users])

# [추가] 관리자 전용 API - 비디오 삭제
@app.route('/api/admin/videos/<int:video_id>', methods=['DELETE'])
@jwt_required()
def delete_video(video_id):
    identity_json = get_jwt_identity()
    current_user = json.loads(identity_json)

    # 관리자 역할인지 확인
    if current_user.get('role') != 'admin':
        return jsonify({"msg": "관리자 권한이 필요합니다."}), 403

    video_to_delete = Video.query.get(video_id)

    if not video_to_delete:
        return jsonify({"error": "비디오를 찾을 수 없습니다."}), 404

    # 파일 시스템에서 실제 파일 삭제 (모든 관련 파일 포함)
    deleted_files = []
    failed_files = []
    
    try:
        # 마스터 파일 삭제
        master_path = os.path.join(app.config['OUTPUT_FOLDER'], video_to_delete.master_filename)
        if os.path.exists(master_path):
            os.remove(master_path)
            deleted_files.append(video_to_delete.master_filename)
        
        # 재생용 파일 삭제
        playback_path = os.path.join(app.config['OUTPUT_FOLDER'], video_to_delete.playback_filename)
        if os.path.exists(playback_path):
            os.remove(playback_path)
            deleted_files.append(video_to_delete.playback_filename)
        
        # 썸네일 파일 삭제
        if video_to_delete.thumbnail_filename:
            thumbnail_path = os.path.join(app.config['OUTPUT_FOLDER'], video_to_delete.thumbnail_filename)
            if os.path.exists(thumbnail_path):
                os.remove(thumbnail_path)
                deleted_files.append(video_to_delete.thumbnail_filename)
        
        # 원본 업로드 파일 삭제 (원본 파일명으로 찾기)
        original_path = os.path.join(app.config['UPLOAD_FOLDER'], video_to_delete.original_filename)
        if os.path.exists(original_path):
            os.remove(original_path)
            deleted_files.append(video_to_delete.original_filename)
            
    except Exception as e:
        app.logger.error(f"파일 삭제 중 오류 발생 (video_id: {video_id}): {e}")
        failed_files.append(str(e))

    # 데이터베이스에서 비디오 정보 삭제
    try:
        db.session.delete(video_to_delete)
        db.session.commit()
        
        # 삭제 결과 메시지 구성
        message = f"비디오가 성공적으로 삭제되었습니다."
        if deleted_files:
            message += f" 삭제된 파일: {', '.join(deleted_files)}"
        if failed_files:
            message += f" 삭제 실패: {', '.join(failed_files)}"
            
        return jsonify({"message": message})
        
    except Exception as e:
        app.logger.error(f"데이터베이스 삭제 중 오류 발생 (video_id: {video_id}): {e}")
        db.session.rollback()
        return jsonify({"error": "데이터베이스에서 비디오 정보 삭제 중 오류가 발생했습니다."}), 500

# 소유자 비디오 삭제 API
@app.route('/api/videos/<int:video_id>', methods=['DELETE'])
@jwt_required()
def delete_own_video(video_id):
    identity_json = get_jwt_identity()
    current_user = json.loads(identity_json)
    video = Video.query.get(video_id)
    if not video:
        return jsonify({"error": "비디오를 찾을 수 없습니다."}), 404
    if video.user_id != current_user.get('id'):
        return jsonify({"error": "본인의 비디오만 삭제할 수 있습니다."}), 403

    # 파일 시스템에서 실제 파일 삭제 (모든 관련 파일 포함)
    deleted_files = []
    failed_files = []
    
    try:
        # 마스터 파일 삭제
        master_path = os.path.join(app.config['OUTPUT_FOLDER'], video.master_filename)
        if os.path.exists(master_path):
            os.remove(master_path)
            deleted_files.append(video.master_filename)
        
        # 재생용 파일 삭제
        playback_path = os.path.join(app.config['OUTPUT_FOLDER'], video.playback_filename)
        if os.path.exists(playback_path):
            os.remove(playback_path)
            deleted_files.append(video.playback_filename)
        
        # 썸네일 파일 삭제
        if video.thumbnail_filename:
            thumbnail_path = os.path.join(app.config['OUTPUT_FOLDER'], video.thumbnail_filename)
            if os.path.exists(thumbnail_path):
                os.remove(thumbnail_path)
                deleted_files.append(video.thumbnail_filename)
        
        # 원본 업로드 파일 삭제
        original_path = os.path.join(app.config['UPLOAD_FOLDER'], video.original_filename)
        if os.path.exists(original_path):
            os.remove(original_path)
            deleted_files.append(video.original_filename)
            
    except Exception as e:
        app.logger.error(f"파일 삭제 중 오류 발생 (video_id: {video_id}): {e}")
        failed_files.append(str(e))

    # 데이터베이스에서 비디오 정보 삭제
    try:
        db.session.delete(video)
        db.session.commit()
        
        # 삭제 결과 메시지 구성
        message = "비디오가 성공적으로 삭제되었습니다."
        if deleted_files:
            message += f" 삭제된 파일: {', '.join(deleted_files)}"
        if failed_files:
            message += f" 삭제 실패: {', '.join(failed_files)}"
            
        return jsonify({"message": message})
        
    except Exception as e:
        app.logger.error(f"데이터베이스 삭제 중 오류 발생 (video_id: {video_id}): {e}")
        db.session.rollback()
        return jsonify({"error": "데이터베이스에서 비디오 정보 삭제 중 오류가 발생했습니다."}), 500

# [추가] 관리자 전용 API - 사용자 삭제
@app.route('/api/admin/users/<int:user_id>', methods=['DELETE'])
@jwt_required()
def delete_user(user_id):
    identity_json = get_jwt_identity()
    current_user = json.loads(identity_json)

    # 1. 관리자 권한 확인
    if current_user.get('role') != 'admin':
        return jsonify({"msg": "관리자 권한이 필요합니다."}), 403

    # 2. 자기 자신을 삭제하려는지 확인
    if current_user.get('id') == user_id:
        return jsonify({"error": "자기 자신을 삭제할 수 없습니다."}), 400

    user_to_delete = User.query.get(user_id)

    if not user_to_delete:
        return jsonify({"error": "사용자를 찾을 수 없습니다."}), 404

    # 3. 다른 관리자를 삭제하려는지 확인 (선택적이지만 좋은 정책)
    if user_to_delete.role == 'admin':
        return jsonify({"error": "다른 관리자를 삭제할 수 없습니다."}), 403

    # 4. 사용자와 관련된 모든 비디오 및 파일 삭제
    videos_to_delete = Video.query.filter_by(user_id=user_id).all()
    deleted_files_count = 0
    failed_files_count = 0
    
    for video in videos_to_delete:
        try:
            # 마스터 파일 삭제
            master_path = os.path.join(app.config['OUTPUT_FOLDER'], video.master_filename)
            if os.path.exists(master_path):
                os.remove(master_path)
                deleted_files_count += 1
            
            # 재생용 파일 삭제
            playback_path = os.path.join(app.config['OUTPUT_FOLDER'], video.playback_filename)
            if os.path.exists(playback_path):
                os.remove(playback_path)
                deleted_files_count += 1
            
            # 썸네일 파일 삭제
            if video.thumbnail_filename:
                thumbnail_path = os.path.join(app.config['OUTPUT_FOLDER'], video.thumbnail_filename)
                if os.path.exists(thumbnail_path):
                    os.remove(thumbnail_path)
                    deleted_files_count += 1
            
            # 원본 업로드 파일 삭제
            original_path = os.path.join(app.config['UPLOAD_FOLDER'], video.original_filename)
            if os.path.exists(original_path):
                os.remove(original_path)
                deleted_files_count += 1
                
        except Exception as e:
            app.logger.error(f"사용자 삭제 중 파일 삭제 오류 (video_id: {video.id}): {e}")
            failed_files_count += 1

    # 5. 사용자 및 관련 비디오 DB 기록 삭제 (SQLAlchemy의 cascade를 이용)
    # Video 모델의 user 관계에 cascade='all, delete-orphan'을 추가하면 user만 지워도 video가 지워지지만,
    # 파일 삭제를 위해 명시적으로 처리하는 것이 더 안전합니다.
    try:
        db.session.delete(user_to_delete)
        db.session.commit()
        
        # 삭제 결과 메시지 구성
        message = f"사용자 '{user_to_delete.username}' 및 관련 데이터가 모두 삭제되었습니다."
        if deleted_files_count > 0:
            message += f" 삭제된 파일 수: {deleted_files_count}개"
        if failed_files_count > 0:
            message += f" 삭제 실패 파일 수: {failed_files_count}개"
            
        return jsonify({"message": message})
        
    except Exception as e:
        app.logger.error(f"사용자 삭제 중 데이터베이스 오류 발생 (user_id: {user_id}): {e}")
        db.session.rollback()
        return jsonify({"error": "데이터베이스에서 사용자 정보 삭제 중 오류가 발생했습니다."}), 500

# [추가] 사용자 본인 회원 탈퇴 API
@app.route('/api/profile/delete-account', methods=['POST'])
@jwt_required()
def delete_account():
    identity_json = get_jwt_identity()
    current_user_data = json.loads(identity_json)
    user_id = current_user_data['id']

    data = request.get_json()
    password = data.get('password')

    if not password:
        return jsonify({'error': '비밀번호를 입력해주세요.'}), 400

    user_to_delete = User.query.get(user_id)

    if not user_to_delete or not bcrypt.check_password_hash(user_to_delete.password, password):
        return jsonify({'error': '비밀번호가 일치하지 않습니다.'}), 401

    # 사용자와 관련된 모든 비디오 및 파일 삭제
    videos_to_delete = Video.query.filter_by(user_id=user_id).all()
    deleted_files_count = 0
    failed_files_count = 0
    
    for video in videos_to_delete:
        try:
            # 마스터 파일 삭제
            master_path = os.path.join(app.config['OUTPUT_FOLDER'], video.master_filename)
            if os.path.exists(master_path):
                os.remove(master_path)
                deleted_files_count += 1
            
            # 재생용 파일 삭제
            playback_path = os.path.join(app.config['OUTPUT_FOLDER'], video.playback_filename)
            if os.path.exists(playback_path):
                os.remove(playback_path)
                deleted_files_count += 1
            
            # 썸네일 파일 삭제
            if video.thumbnail_filename:
                thumbnail_path = os.path.join(app.config['OUTPUT_FOLDER'], video.thumbnail_filename)
                if os.path.exists(thumbnail_path):
                    os.remove(thumbnail_path)
                    deleted_files_count += 1
            
            # 원본 업로드 파일 삭제
            original_path = os.path.join(app.config['UPLOAD_FOLDER'], video.original_filename)
            if os.path.exists(original_path):
                os.remove(original_path)
                deleted_files_count += 1
                
        except Exception as e:
            app.logger.error(f"파일 삭제 중 오류 발생 (video_id: {video.id}): {e}")
            failed_files_count += 1

    try:
        db.session.delete(user_to_delete)
        db.session.commit()
        
        # 삭제 결과 메시지 구성
        message = '회원 탈퇴가 성공적으로 처리되었습니다.'
        if deleted_files_count > 0:
            message += f' 삭제된 파일 수: {deleted_files_count}개'
        if failed_files_count > 0:
            message += f' 삭제 실패 파일 수: {failed_files_count}개'
            
        return jsonify({'message': message})
        
    except Exception as e:
        app.logger.error(f"회원 탈퇴 중 데이터베이스 오류 발생 (user_id: {user_id}): {e}")
        db.session.rollback()
        return jsonify({'error': '데이터베이스에서 사용자 정보 삭제 중 오류가 발생했습니다.'}), 500

@app.route('/api/extract', methods=['POST'])
def extract_route():
    if 'video' not in request.files:
        return jsonify({'error': '비디오 파일이 없습니다.'}), 400

    video_file = request.files['video']
    filename = f"{int(time.time())}_{video_file.filename}"
    input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    video_file.save(input_path)

    # [수정] 워터마크 추출 및 복호화
    extracted_b64_watermark = extract_watermark(input_path)

    if "Error:" in extracted_b64_watermark or "워터마크를 찾지 못했" in extracted_b64_watermark:
        return jsonify({'watermark': extracted_b64_watermark})

    try:
        encrypted_watermark_bytes = base64.urlsafe_b64decode(extracted_b64_watermark.encode('utf-8'))
        decrypted_watermark = cipher_suite.decrypt(encrypted_watermark_bytes).decode('utf-8')
        return jsonify({'watermark': decrypted_watermark})
    except (InvalidToken, ValueError, TypeError) as e:
        app.logger.error(f"Watermark decryption failed for extracted data: {e}")
        return jsonify({'watermark': "워터마크 복호화에 실패했습니다. 데이터가 손상되었거나 유효하지 않습니다."})

# 헬스 체크 엔드포인트
@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'cache_size': len(cache.cache)
    })

@app.route('/outputs/<filename>')
def get_output_file(filename):
    # [수정] MP4 파일은 스트리밍을 위해 인라인으로 제공, 그 외에는 다운로드
    if filename.endswith('.mp4'):
        return send_from_directory(app.config['OUTPUT_FOLDER'], filename, as_attachment=False)
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename, as_attachment=True)

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
