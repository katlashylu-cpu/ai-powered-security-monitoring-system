import os
import cv2
import numpy as np
import face_recognition
import pygame
import threading
import time
import re
import hashlib
from collections import deque, defaultdict
from datetime import datetime, timedelta
from flask import Flask, Response, render_template_string, jsonify, request, session, redirect, url_for, send_file
from functools import wraps
from werkzeug.utils import secure_filename

# Only keep log parsing from summary_generator (no report generation)
from summary_generator import parse_log_file

# ========== CONFIGURATION ==========


class Config:
    FACE_TOLERANCE = 0.55
    RECOGNITION_BUFFER_SIZE = 10
    LOG_COOLDOWN_AUTHORIZED = 30
    LOG_COOLDOWN_INTRUDER = 5
    SUSPICIOUS_AREA_THRESHOLD = 3000   # pixels for movement detection
    FRAME_SKIP = 2
    SCALE_FACTOR = 0.5
    FACE_MODEL = "hog"
    FLASK_PORT = 5000
    FLASK_HOST = "0.0.0.0"
    MAX_FPS = 15
    AUTH_USERNAME = "admin"
    AUTH_PASSWORD = "security123"
    SECRET_KEY = "your-secret-key-here-change-this"
    CAPTURE_COOLDOWN = 15
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}
    MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB
    SESSION_LIFETIME_MINUTES = 5      # Admin session expires after 5 minutes
    MULTIPLE_FACES_COOLDOWN = 15
    MULTIPLE_FACES_MIN_COUNT = 2
    MOTION_FRAME_INTERVAL = 3

# ========== AUTHENTICATION ==========


def check_auth(username, password):
    return username == Config.AUTH_USERNAME and password == Config.AUTH_PASSWORD


def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('logged_in'):
            return redirect(url_for('login_page'))
        return f(*args, **kwargs)
    return decorated_function

# ========== UTILITY ==========


def sanitize_filename(filename):
    if not filename:
        return "unknown"
    return re.sub(r'[^a-zA-Z0-9_\-\.]', '_', str(filename))[:100]


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS


def boxes_overlap(box1, box2):
    """box = (x, y, w, h)"""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    return not (x1 + w1 < x2 or x2 + w2 < x1 or y1 + h1 < y2 or y2 + h2 < y1)

# ========== FACE RECOGNITION SETUP ==========
folder_path = "authorized_faces"
authorized_encodings = []
authorized_names = []


def load_authorized_faces():
    global authorized_encodings, authorized_names
    authorized_encodings = []
    authorized_names = []
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        return
    for filename in os.listdir(folder_path):
        if filename.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
            image_path = os.path.join(folder_path, filename)
            try:
                image = face_recognition.load_image_file(image_path)
                encodings = face_recognition.face_encodings(image)
                if encodings:
                    authorized_encodings.append(encodings[0])
                    name = sanitize_filename(os.path.splitext(filename)[0])
                    authorized_names.append(name)
                    print(f"✅ Loaded face: {filename} -> {name}")
                else:
                    print(f"⚠️ No face found in: {filename} (will be skipped)")
            except Exception as e:
                print(f"❌ Error loading {filename}: {e}")


# Initial load
load_authorized_faces()


def reload_authorized_faces():
    """Call this after adding/deleting faces to refresh recognition data."""
    load_authorized_faces()
    print(f"🔄 Face database reloaded. Now {len(authorized_names)} authorized faces.")


# ========== FLASK APP ==========
app = Flask(__name__)
app.config['SECRET_KEY'] = Config.SECRET_KEY
app.config['MAX_CONTENT_LENGTH'] = Config.MAX_FILE_SIZE
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(minutes=Config.SESSION_LIFETIME_MINUTES)

# ========== CAMERA ==========
camera = None
camera_lock = threading.Lock()
camera_initialized = False
streaming_enabled = True


def init_camera():
    global camera, camera_initialized
    with camera_lock:
        if camera is not None:
            camera.release()
        camera = cv2.VideoCapture(0)
        if camera.isOpened():
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            camera.set(cv2.CAP_PROP_FPS, 30)
            camera_initialized = True
            return True
        camera_initialized = False
        return False


init_camera()

# ========== AUDIO ==========
pygame.mixer.init()
alert_sound_loaded = False
alert_sound_path = "alert.mp3"
if os.path.exists(alert_sound_path):
    try:
        pygame.mixer.music.load(alert_sound_path)
        alert_sound_loaded = True
    except:
        pass

# ========== GLOBAL STATE ==========
recognition_buffer = defaultdict(lambda: deque(maxlen=Config.RECOGNITION_BUFFER_SIZE))
last_logged = {}
last_alert_time = 0
captured_faces = {}
fgbg = cv2.createBackgroundSubtractorMOG2()   # for suspicious movement
motion_counter = 0
last_multiple_faces_time = 0

# ========== LOGGING (full frame) ==========


def log_event(frame, location, name, event_type=None, force_log=False):
    global last_logged, captured_faces

    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
    date_folder = now.strftime("%Y-%m-%d")

    if event_type is None:
        event_type = "Authorized" if name != "Unauthorized" else "Intruder"

    name = sanitize_filename(name)
    event_type = sanitize_filename(event_type)

    current_time = time.time()
    should_capture = False
    if name not in captured_faces:
        should_capture = True
        captured_faces[name] = current_time
    elif (current_time - captured_faces[name]) >= Config.CAPTURE_COOLDOWN:
        should_capture = True
        captured_faces[name] = current_time

    if not force_log:
        cooldown = Config.LOG_COOLDOWN_AUTHORIZED if event_type == "Authorized" else Config.LOG_COOLDOWN_INTRUDER
        key = (name, event_type)
        if key in last_logged:
            if (current_time - last_logged[key]) < cooldown:
                return

    day_log_folder = os.path.join("logs", date_folder)
    event_folder = os.path.join(day_log_folder, event_type)
    os.makedirs(event_folder, exist_ok=True)

    log_file_path = os.path.join(day_log_folder, "detection_log.txt")
    try:
        with open(log_file_path, "a") as log_file:
            log_line = f"{timestamp} - Event: {event_type} - Name: {name}, Location: {location}\n"
            log_file.write(log_line)
        last_logged[(name, event_type)] = current_time
    except IOError as e:
        print(f"Log error: {e}")
        return

    if should_capture and frame is not None:
        try:
            snapshot_path = os.path.join(event_folder, f"{timestamp}_{event_type}_{name}.jpg")
            cv2.imwrite(snapshot_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            print(f"📸 Full frame snapshot saved: {snapshot_path}")
        except Exception as e:
            print(f"Snapshot error: {e}")

# ========== SUSPICIOUS MOVEMENT DETECTION (object only, no overlap with faces) ==========


def detect_suspicious_movement(frame, face_rects_scaled):
    """Detect large motion, draw yellow box only if it does NOT overlap any face rectangle."""
    global motion_counter
    motion_counter += 1
    if motion_counter % Config.MOTION_FRAME_INTERVAL != 0:
        return
    try:
        fgmask = fgbg.apply(frame)
        thresh = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=2)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > Config.SUSPICIOUS_AREA_THRESHOLD:
                x, y, w, h = cv2.boundingRect(contour)
                # Check overlap with any face
                overlap = False
                for (fx, fy, fw, fh) in face_rects_scaled:
                    if boxes_overlap((x, y, w, h), (fx, fy, fw, fh)):
                        overlap = True
                        break
                if overlap:
                    continue
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
                cv2.putText(frame, "Suspicious Object", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                log_event(frame, (y, x+w, y+h, x), "Unknown", event_type="Suspicious")
    except Exception as e:
        print(f"Movement detection error: {e}")


def detect_multiple_faces(frame, face_rects_scaled, current_time):
    """Log a suspicious event if multiple faces are present (cooldown applied)."""
    global last_multiple_faces_time
    if len(face_rects_scaled) >= Config.MULTIPLE_FACES_MIN_COUNT:
        if (current_time - last_multiple_faces_time) > Config.MULTIPLE_FACES_COOLDOWN:
            last_multiple_faces_time = current_time
            # Draw grouped box around all faces
            if face_rects_scaled:
                xs = [r[0] for r in face_rects_scaled]
                ys = [r[1] for r in face_rects_scaled]
                x2s = [r[0]+r[2] for r in face_rects_scaled]
                y2s = [r[1]+r[3] for r in face_rects_scaled]
                group_x = min(xs)
                group_y = min(ys)
                group_w = max(x2s) - group_x
                group_h = max(y2s) - group_y
                cv2.rectangle(frame, (group_x, group_y), (group_x+group_w, group_y+group_h), (0, 255, 255), 3)
                cv2.putText(frame, f"Multiple Faces ({len(face_rects_scaled)})", (group_x, group_y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            log_event(frame, (0,0,0,0), "MultipleFaces", event_type="Suspicious", force_log=True)

# ========== FRAME PROCESSING ==========


def annotate_frame(frame):
    global last_alert_time

    if frame is None or not streaming_enabled:
        return frame

    small_frame = cv2.resize(frame, (0, 0), fx=Config.SCALE_FACTOR, fy=Config.SCALE_FACTOR)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    try:
        face_locations = face_recognition.face_locations(rgb_small_frame, model=Config.FACE_MODEL)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            name = "Unauthorized"
            if authorized_encodings:
                matches = face_recognition.compare_faces(authorized_encodings, face_encoding,
                                                         tolerance=Config.FACE_TOLERANCE)
                face_distances = face_recognition.face_distance(authorized_encodings, face_encoding)
                if len(face_distances) > 0:
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index] and face_distances[best_match_index] < Config.FACE_TOLERANCE:
                        name = authorized_names[best_match_index]

            # stable recognition buffer
            buffer_key = hashlib.md5(face_encoding.tobytes()).hexdigest()[:16]
            recognition_buffer[buffer_key].append(name)
            name = max(set(recognition_buffer[buffer_key]), key=recognition_buffer[buffer_key].count)
            face_names.append(name)

        scale_back = int(1 / Config.SCALE_FACTOR)

        face_rects_original = []
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            top *= scale_back
            right *= scale_back
            bottom *= scale_back
            left *= scale_back

            h, w = frame.shape[:2]
            top = max(0, min(top, h))
            bottom = max(0, min(bottom, h))
            left = max(0, min(left, w))
            right = max(0, min(right, w))

            face_rects_original.append((left, top, right-left, bottom-top))

            is_authorized = name != "Unauthorized"
            color = (0, 255, 0) if is_authorized else (0, 0, 255)

            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
            cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

            if not is_authorized:
                current_time = time.time()
                if alert_sound_loaded and (current_time - last_alert_time) > 3:
                    try:
                        pygame.mixer.music.play()
                        last_alert_time = current_time
                        print("🔊 Intruder alert!")
                    except:
                        pass

            log_event(frame, (top, right, bottom, left), name)

        # Detect suspicious object motion (non‑overlapping)
        detect_suspicious_movement(frame, face_rects_original)

        # Detect multiple faces as suspicious
        current_time = time.time()
        detect_multiple_faces(frame, face_rects_original, current_time)

    except Exception as e:
        print(f"Frame processing error: {e}")

    return frame


def generate_frames():
    global camera, camera_initialized, streaming_enabled
    frame_count = 0
    while True:
        if not streaming_enabled:
            paused_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(paused_frame, "STREAMING PAUSED", (150, 240),
                        cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 0, 255), 2)
            ret, buffer = cv2.imencode('.jpg', paused_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if ret:
                yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(0.1)
            continue

        with camera_lock:
            if camera is None or not camera_initialized:
                if not init_camera():
                    time.sleep(0.1)
                    continue
            success, frame = camera.read()
            if not success:
                print("Failed to read frame, reconnecting...")
                init_camera()
                time.sleep(0.1)
                continue

        frame_count += 1
        if frame_count % Config.FRAME_SKIP == 0:
            frame = annotate_frame(frame)

        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if ret:
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

        time.sleep(1 / Config.MAX_FPS)

# ========== FLASK ROUTES ==========


@app.route('/')
def index():
    return render_template_string("""<!DOCTYPE html>
    <html>
    <head><title>AI Security Monitoring System | NeuralGuard</title>
    <meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
        * { margin:0; padding:0; box-sizing:border-box; }
        body {
            font-family: 'Inter', sans-serif;
            background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 100%);
            color: #fff; overflow-x: hidden; min-height: 100vh;
            display: flex; align-items: center; justify-content: center;
        }
        .container { max-width: 1200px; width:95%; margin:20px auto; padding:20px; background:rgba(0,0,0,0.3); border-radius:30px; z-index:1; }
        .header { background:rgba(20,20,40,0.8); backdrop-filter:blur(10px); border-radius:20px; padding:20px 25px; margin-bottom:20px; border:1px solid rgba(0,255,0,0.2); }
        .header h1 { font-size:28px; font-weight:700; background:linear-gradient(135deg,#00ff00,#00ccff); -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
        .status-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(160px,1fr)); gap:15px; margin-bottom:20px; }
        .status-card { background:rgba(20,20,40,0.8); backdrop-filter:blur(10px); border-radius:15px; padding:15px; text-align:center; border:1px solid rgba(0,255,0,0.2); transition:0.3s; }
        .status-icon { font-size:28px; margin-bottom:8px; }
        .status-value { font-size:20px; font-weight:700; color:#00ff00; }
        .video-container { background:rgba(0,0,0,0.5); border-radius:20px; padding:8px; margin-bottom:20px; border:2px solid rgba(0,255,0,0.3); max-width:800px; margin-left:auto; margin-right:auto; }
        .video-wrapper { border-radius:15px; overflow:hidden; max-height:450px; }
        .video-wrapper img { width:100%; height:auto; display:block; max-height:450px; object-fit:contain; }
        .controls { display:flex; gap:15px; justify-content:center; margin-bottom:20px; flex-wrap:wrap; }
        .btn { padding:8px 24px; font-size:14px; font-weight:600; border:none; border-radius:50px; cursor:pointer; transition:0.3s; }
        .btn-play { background:linear-gradient(135deg,#28a745,#20c997); color:white; }
        .btn-pause { background:linear-gradient(135deg,#dc3545,#c82333); color:white; }
        .btn-admin { background:linear-gradient(135deg,#007bff,#0056b3); color:white; }
        .stream-status { text-align:center; padding:10px; background:rgba(0,0,0,0.5); border-radius:10px; margin-top:15px; }
        .status-badge.live { background:#ff0000; animation:pulse 1.5s infinite; display:inline-block; padding:4px 12px; border-radius:20px; font-size:11px; font-weight:600; }
        @keyframes pulse { 0%,100%{opacity:1;} 50%{opacity:0.5;} }
        .footer { text-align:center; padding:15px; background:rgba(0,0,0,0.5); border-radius:10px; margin-top:20px; font-size:10px; color:#666; }
        @media (max-width:768px){ .header h1{font-size:20px;} .btn{padding:6px 16px;} }
    </style>
    </head>
    <body>
        <div class="container">
            <div class="header"><h1>🔒 NEURALGUARD AI SECURITY SYSTEM</h1><p>Advanced Facial Recognition & Threat Detection Platform</p></div>
            <div class="status-grid">
                <div class="status-card"><div class="status-icon">👥</div><div class="status-label">Authorized Personnel</div><div class="status-value">{{ face_count }}</div></div>
                <div class="status-card"><div class="status-icon">🎥</div><div class="status-label">Camera Status</div><div class="status-value">ACTIVE</div></div>
                <div class="status-card"><div class="status-icon">🤖</div><div class="status-label">AI Model</div><div class="status-value">FaceNet</div></div>
                <div class="status-card"><div class="status-icon">🛡️</div><div class="status-label">Security Level</div><div class="status-value">MAXIMUM</div></div>
            </div>
            <div class="video-container"><div class="video-wrapper"><img id="videoFeed" src="/video" alt="Security Feed"></div></div>
            <div class="controls">
                <button class="btn btn-play" onclick="resumeStream()">▶ PLAY STREAM</button>
                <button class="btn btn-pause" onclick="pauseStream()">⏸ PAUSE STREAM</button>
                <button class="btn btn-admin" onclick="window.location.href='/admin'">🔐 ADMIN ACCESS</button>
            </div>
            <div class="stream-status"><span class="status-badge live">● LIVE</span> Neural Network Active | Real-time Monitoring</div>
            <div class="footer"><p>Powered by Advanced AI | Real-time Facial Recognition | Intelligent Threat Detection</p><p>© 2025 NeuralGuard Security Systems</p></div>
        </div>
        <script>
            var videoFeed = document.getElementById('videoFeed');
            function pauseStream() { fetch('/pause_stream',{method:'POST'}).then(()=>{ videoFeed.src='/video?'+new Date().getTime(); }); }
            function resumeStream() { fetch('/resume_stream',{method:'POST'}).then(()=>{ videoFeed.src='/video?'+new Date().getTime(); }); }
        </script>
    </body>
    </html>""", face_count=len(authorized_names))


@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/pause_stream', methods=['POST'])
def pause_stream():
    global streaming_enabled
    streaming_enabled = False
    return jsonify({"status": "paused"})


@app.route('/resume_stream', methods=['POST'])
def resume_stream():
    global streaming_enabled
    streaming_enabled = True
    return jsonify({"status": "streaming"})

# ========== ADMIN AUTH ==========


@app.route('/admin')
def login_page():
    if session.get('logged_in'):
        return redirect(url_for('admin_dashboard'))
    return render_template_string("""<!DOCTYPE html>
    <html><head><title>Admin Login</title><style>
        body { font-family:Inter; background:linear-gradient(135deg,#0a0a0a,#1a1a2e); min-height:100vh; display:flex; align-items:center; justify-content:center; }
        .login-container { background:rgba(20,20,40,0.95); padding:50px; border-radius:20px; width:450px; border:1px solid rgba(0,255,0,0.3); }
        h2 { color:#00ff00; text-align:center; margin-bottom:10px; }
        input { width:100%; padding:15px; margin:10px 0; background:rgba(0,0,0,0.5); border:1px solid rgba(0,255,0,0.3); border-radius:10px; color:white; }
        button { width:100%; padding:15px; margin-top:20px; background:linear-gradient(135deg,#007bff,#0056b3); color:white; border:none; border-radius:10px; cursor:pointer; }
        .error { color:#ff0000; text-align:center; margin-top:15px; }
        .back-link { text-align:center; margin-top:20px; }
        .back-link a { color:#00ff00; text-decoration:none; }
    </style></head>
    <body><div class="login-container"><h2>🔐 ADMIN ACCESS</h2><form method="POST" action="/admin/login">
        <input type="text" name="username" placeholder="Username" required autofocus>
        <input type="password" name="password" placeholder="Password" required>
        <button type="submit">Authenticate</button>
        {% if error %}<div class="error">{{ error }}</div>{% endif %}
    </form><div class="back-link"><a href="/">← Return to Dashboard</a></div></div></body></html>""")


@app.route('/admin/login', methods=['POST'])
def admin_login():
    username = request.form.get('username')
    password = request.form.get('password')
    if check_auth(username, password):
        session.permanent = True  # session will expire after SESSION_LIFETIME_MINUTES
        session['logged_in'] = True
        session['username'] = username
        return redirect(url_for('admin_dashboard'))
    return render_template_string("""<!DOCTYPE html>
    <html><head><title>Admin Login</title><style>
        body { font-family:Inter; background:linear-gradient(135deg,#0a0a0a,#1a1a2e); display:flex; align-items:center; justify-content:center; }
        .login-container { background:rgba(20,20,40,0.95); padding:50px; border-radius:20px; width:450px; border:1px solid rgba(0,255,0,0.3); }
        h2 { color:#00ff00; text-align:center; }
        input { width:100%; padding:15px; margin:10px 0; background:rgba(0,0,0,0.5); border:1px solid rgba(0,255,0,0.3); border-radius:10px; color:white; }
        button { width:100%; padding:15px; margin-top:20px; background:linear-gradient(135deg,#007bff,#0056b3); color:white; border:none; border-radius:10px; cursor:pointer; }
        .error { color:#ff0000; text-align:center; margin-top:15px; }
        .back-link { text-align:center; margin-top:20px; }
    </style></head>
    <body><div class="login-container"><h2>🔐 ACCESS DENIED</h2><form method="POST" action="/admin/login">
        <input type="text" name="username" placeholder="Username" required>
        <input type="password" name="password" placeholder="Password" required>
        <button type="submit">Try Again</button>
        <div class="error">Invalid username or password</div>
    </form><div class="back-link"><a href="/">← Return to Dashboard</a></div></div></body></html>""")


@app.route('/admin/dashboard')
@login_required
def admin_dashboard():
    return render_template_string("""<!DOCTYPE html>
    <html><head><title>Admin Dashboard | NeuralGuard</title>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        * { margin:0; padding:0; box-sizing:border-box; }
        body { font-family:'Inter',sans-serif; background:linear-gradient(135deg,#0a0a0a,#1a1a2e); color:white; }
        .container { max-width:1400px; margin:0 auto; padding:20px; }
        .header { background:rgba(20,20,40,0.8); backdrop-filter:blur(10px); border-radius:20px; padding:25px; margin-bottom:30px; border:1px solid rgba(0,255,0,0.2); }
        h1 { background:linear-gradient(135deg,#00ff00,#00ccff); -webkit-background-clip:text; -webkit-text-fill-color:transparent; font-size:28px; }
        .admin-badge { background:#007bff; display:inline-block; padding:5px 15px; border-radius:20px; font-size:12px; margin-left:15px; }
        .dashboard-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(280px,1fr)); gap:30px; margin:40px 0; }
        .dashboard-card { background:rgba(20,20,40,0.8); backdrop-filter:blur(10px); border-radius:20px; padding:40px; text-align:center; cursor:pointer; transition:0.3s; border:1px solid rgba(0,255,0,0.2); }
        .dashboard-card:hover { transform:translateY(-10px); border-color:rgba(0,255,0,0.5); box-shadow:0 10px 30px rgba(0,255,0,0.1); }
        .icon { font-size:64px; margin-bottom:20px; }
        .dashboard-card h2 { margin-bottom:10px; color:#00ff00; }
        .logout-btn { background:#dc3545; color:white; padding:12px 30px; border:none; border-radius:50px; cursor:pointer; font-size:16px; font-weight:600; transition:0.3s; }
        .logout-btn:hover { background:#c82333; transform:translateY(-2px); }
        .back-link { display:inline-block; margin-top:20px; color:#00ff00; text-decoration:none; margin-left:20px; }
    </style></head>
    <body><div class="container"><div class="header"><h1>Admin Control Panel <span class="admin-badge">Logged in as {{ username }}</span></h1><p>System Administration & Configuration</p></div>
    <div class="dashboard-grid">
        <div class="dashboard-card" onclick="window.location.href='/admin/logs'"><div class="icon">📋</div><h2>Security Logs</h2><p>View detailed event logs and detection history</p></div>
        <div class="dashboard-card" onclick="window.location.href='/admin/statistics'"><div class="icon">📊</div><h2>Analytics & Statistics</h2><p>Comprehensive security metrics and insights</p></div>
        <div class="dashboard-card" onclick="window.location.href='/admin/image-logs'"><div class="icon">🖼️</div><h2>Image Logs</h2><p>View captured snapshots (Authorized / Intruder / Suspicious)</p></div>
        <div class="dashboard-card" onclick="window.location.href='/admin/faces'"><div class="icon">👤</div><h2>Face Management</h2><p>Add, remove, or view authorized faces</p></div>
    </div>
    <div style="text-align:center;"><form action="/admin/logout" method="POST" style="display:inline;"><button type="submit" class="logout-btn">🚪 Logout</button></form><a href="/" class="back-link">← Back to Live Dashboard</a></div></div></body></html>""", username=session.get('username', 'Admin'))

# ========== FACE MANAGEMENT ROUTES ==========


@app.route('/admin/faces')
@login_required
def manage_faces():
    faces = []
    if os.path.exists(folder_path):
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                name = os.path.splitext(filename)[0]
                img_path = os.path.join(folder_path, filename)
                try:
                    img = face_recognition.load_image_file(img_path)
                    encodings = face_recognition.face_encodings(img)
                    valid = len(encodings) > 0
                except:
                    valid = False
                faces.append({
                    'name': name,
                    'filename': filename,
                    'valid': valid,
                    'image_url': f'/admin/face-image/{filename}'
                })
    return render_template_string("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Face Management | NeuralGuard</title>
        <style>
            * { margin:0; padding:0; box-sizing:border-box; }
            body { font-family:'Inter',sans-serif; background:linear-gradient(135deg,#0a0a0a,#1a1a2e); color:white; padding:20px; }
            .container { max-width:1200px; margin:0 auto; }
            .header { background:rgba(20,20,40,0.8); padding:20px; border-radius:10px; margin-bottom:20px; }
            .card { background:rgba(20,20,40,0.8); border-radius:15px; padding:20px; margin-bottom:20px; }
            .faces-grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(200px,1fr)); gap:20px; margin-top:20px; }
            .face-card { background:rgba(30,30,50,0.9); border-radius:10px; padding:15px; text-align:center; border:2px solid #00ff00; }
            .face-card img { width:100%; height:150px; object-fit:cover; border-radius:10px; margin-bottom:10px; }
            .face-name { font-weight:bold; margin:10px 0; color:#00ff00; }
            .delete-btn { background:#dc3545; color:white; border:none; padding:5px 15px; border-radius:5px; cursor:pointer; margin-top:5px; }
            .delete-btn:hover { background:#c82333; }
            .add-form { background:rgba(20,20,40,0.8); padding:20px; border-radius:15px; margin-bottom:20px; }
            .add-form input, .add-form button { padding:10px; margin:10px 0; border-radius:5px; border:none; }
            .add-form input { width:100%; background:#333; color:white; }
            .add-form button { background:#28a745; color:white; cursor:pointer; font-weight:bold; }
            .add-form button:hover { background:#218838; }
            .message { padding:10px; border-radius:5px; margin-bottom:15px; }
            .success { background:#28a745; }
            .error { background:#dc3545; }
            .back-link { color:#00ff00; text-decoration:none; display:inline-block; margin-top:20px; }
            button { padding:10px 20px; margin:5px; background:#333; color:white; border:none; border-radius:5px; cursor:pointer; }
            .warning { color:#ffc107; font-size:12px; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header"><h1>👤 Face Management</h1><p>Add, remove, or view authorized faces</p></div>
            {% if message %}
            <div class="message {{ message_type }}">{{ message }}</div>
            {% endif %}
            <div class="add-form">
                <h3>➕ Add New Authorized Face</h3>
                <form method="POST" action="/admin/faces/add" enctype="multipart/form-data">
                    <input type="text" name="person_name" placeholder="Person's Full Name" required>
                    <input type="file" name="face_image" accept="image/jpeg,image/png,image/webp" required>
                    <button type="submit">Save Face</button>
                </form>
                <div class="warning">⚠️ Image must contain exactly one clear face. Only JPG, PNG, WEBP up to 5MB.</div>
            </div>
            <div class="card">
                <h3>📋 Existing Authorized Faces ({{ faces|length }})</h3>
                <div class="faces-grid">
                    {% for face in faces %}
                    <div class="face-card">
                        <img src="{{ face.image_url }}" alt="{{ face.name }}">
                        <div class="face-name">{{ face.name }}</div>
                        {% if not face.valid %}<div class="warning">⚠️ No face detected in this image</div>{% endif %}
                        <form method="POST" action="/admin/faces/delete" onsubmit="return confirm('Delete {{ face.name }}?')">
                            <input type="hidden" name="filename" value="{{ face.filename }}">
                            <button type="submit" class="delete-btn">🗑️ Delete</button>
                        </form>
                    </div>
                    {% else %}
                    <div>No authorized faces added yet. Use the form above to add one.</div>
                    {% endfor %}
                </div>
            </div>
            <div style="text-align:center;">
                <button onclick="window.location.href='/admin/dashboard'">Back to Dashboard</button>
                <button onclick="window.location.reload()">Refresh</button>
            </div>
            <a href="/" class="back-link">← Back to Live Dashboard</a>
        </div>
    </body>
    </html>
    """, faces=faces)


@app.route('/admin/faces/add', methods=['POST'])
@login_required
def add_face():
    if 'face_image' not in request.files:
        return redirect(url_for('manage_faces', message='No file uploaded', message_type='error'))
    file = request.files['face_image']
    person_name = request.form.get('person_name', '').strip()
    if not person_name:
        return redirect(url_for('manage_faces', message='Name is required', message_type='error'))
    if file.filename == '':
        return redirect(url_for('manage_faces', message='No file selected', message_type='error'))
    if not allowed_file(file.filename):
        return redirect(url_for('manage_faces', message='Invalid file type. Use JPG, PNG, WEBP', message_type='error'))

    safe_name = sanitize_filename(person_name)
    filename = f"{safe_name}.jpg"
    filepath = os.path.join(folder_path, filename)

    file.save(filepath)

    try:
        img = face_recognition.load_image_file(filepath)
        encodings = face_recognition.face_encodings(img)
        if not encodings:
            os.remove(filepath)
            return redirect(url_for('manage_faces', message='No face detected in the image. Please upload a clear face photo.', message_type='error'))
    except Exception as e:
        os.remove(filepath)
        return redirect(url_for('manage_faces', message=f'Error processing image: {e}', message_type='error'))

    reload_authorized_faces()
    return redirect(url_for('manage_faces', message=f'Face "{person_name}" added successfully!', message_type='success'))


@app.route('/admin/faces/delete', methods=['POST'])
@login_required
def delete_face():
    filename = request.form.get('filename')
    if not filename:
        return redirect(url_for('manage_faces', message='No file specified', message_type='error'))
    filepath = os.path.join(folder_path, filename)
    if os.path.exists(filepath):
        os.remove(filepath)
        reload_authorized_faces()
        return redirect(url_for('manage_faces', message=f'Deleted {filename}', message_type='success'))
    else:
        return redirect(url_for('manage_faces', message='File not found', message_type='error'))


@app.route('/admin/face-image/<filename>')
@login_required
def serve_face_image(filename):
    base_dir = os.path.abspath(folder_path)
    req_path = os.path.abspath(os.path.join(base_dir, filename))
    if not req_path.startswith(base_dir):
        return "Access denied", 403
    if os.path.exists(req_path) and os.path.isfile(req_path):
        return send_file(req_path)
    return "Image not found", 404

# ========== LOGS, STATISTICS, IMAGE LOGS ROUTES ==========


@app.route('/admin/logs')
@login_required
def admin_logs():
    date = request.args.get('date', datetime.now().strftime("%Y-%m-%d"))
    log_file_path = os.path.join("logs", date, "detection_log.txt")
    available_dates = []
    if os.path.exists("logs"):
        for d in os.listdir("logs"):
            if os.path.isdir(os.path.join("logs", d)) and len(d) == 10:
                available_dates.append(d)
    available_dates.sort(reverse=True)
    logs_content = []
    if os.path.exists(log_file_path):
        with open(log_file_path, "r") as f:
            logs_content = f.readlines()[-200:]
    return render_template_string("""<!DOCTYPE html>
    <html><head><title>Security Logs</title><style>
        body { font-family:'Courier New',monospace; background:linear-gradient(135deg,#0a0a0a,#1a1a2e); color:#00ff00; padding:20px; }
        .container { max-width:1400px; margin:0 auto; }
        .header { background:rgba(20,20,40,0.8); padding:20px; border-radius:10px; margin-bottom:20px; }
        .controls { background:rgba(20,20,40,0.8); padding:20px; border-radius:10px; margin-bottom:20px; }
        select, button { padding:10px; margin:5px; background:#333; color:white; border:none; border-radius:5px; cursor:pointer; }
        .logs-container { background:#111; padding:20px; border-radius:10px; max-height:600px; overflow-y:auto; }
        .log-entry { padding:8px; margin:5px 0; border-left:3px solid #00ff00; background:#1a1a1a; font-size:12px; }
        .authorized { border-left-color:#00ff00; color:#aaffaa; }
        .intruder { border-left-color:#ff0000; color:#ffaaaa; }
        .suspicious { border-left-color:#ffff00; color:#ffffaa; }
        .back-link { color:#00ff00; text-decoration:none; }
    </style></head>
    <body><div class="container"><div class="header"><h1>📋 System Security Logs</h1></div>
    <div class="controls"><form method="GET" style="display:inline;"><select name="date" onchange="this.form.submit()">{% for d in available_dates %}<option value="{{ d }}" {% if d==date %}selected{% endif %}>{{ d }}</option>{% endfor %}</select></form>
    <button onclick="window.location.reload()">Refresh</button><button onclick="window.location.href='/admin/dashboard'">Back</button></div>
    <div class="logs-container">{% if logs %}{% for line in logs %}{% if 'Authorized' in line %}<div class="log-entry authorized">✅ {{ line.strip() }}</div>{% elif 'Intruder' in line %}<div class="log-entry intruder">🔴 {{ line.strip() }}</div>{% elif 'Suspicious' in line %}<div class="log-entry suspicious">⚠️ {{ line.strip() }}</div>{% else %}<div class="log-entry">{{ line.strip() }}</div>{% endif %}{% endfor %}{% else %}<div>No logs found for {{ date }}</div>{% endif %}</div>
    <a href="/admin/dashboard" class="back-link">← Back to Dashboard</a></div></body></html>""", logs=logs_content, date=date, available_dates=available_dates)


@app.route('/admin/statistics')
@login_required
def admin_statistics():
    today = datetime.now().strftime("%Y-%m-%d")
    today_log = os.path.join("logs", today, "detection_log.txt")
    today_auth = today_intruder = today_suspicious = 0
    if os.path.exists(today_log):
        today_auth, today_intruder, today_suspicious, _ = parse_log_file(today_log)
    week_auth = week_intruder = week_suspicious = 0
    week_data = []
    for i in range(7):
        day = (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d")
        day_log = os.path.join("logs", day, "detection_log.txt")
        if os.path.exists(day_log):
            auth, intr, susp, _ = parse_log_file(day_log)
            week_auth += auth
            week_intruder += intr
            week_suspicious += susp
            week_data.append({'date': day, 'auth': auth, 'intruder': intr, 'susp': susp})
    total_auth = total_intruder = total_suspicious = 0
    if os.path.exists("logs"):
        for date_folder in os.listdir("logs"):
            if os.path.isdir(os.path.join("logs", date_folder)):
                log_file = os.path.join("logs", date_folder, "detection_log.txt")
                if os.path.exists(log_file):
                    auth, intr, susp, _ = parse_log_file(log_file)
                    total_auth += auth
                    total_intruder += intr
                    total_suspicious += susp
    return render_template_string("""<!DOCTYPE html>
    <html><head><title>Statistics</title><style>
        body { font-family:'Inter',sans-serif; background:linear-gradient(135deg,#0a0a0a,#1a1a2e); color:white; padding:20px; }
        .stats-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(300px,1fr)); gap:20px; margin-bottom:30px; }
        .stat-card { background:rgba(20,20,40,0.8); padding:30px; border-radius:15px; text-align:center; }
        .stat-number { font-size:48px; font-weight:bold; margin:10px 0; }
        .authorized { border-top:4px solid #00ff00; } .intruder { border-top:4px solid #ff0000; } .suspicious { border-top:4px solid #ffff00; } .total { border-top:4px solid #007bff; }
        .data-table { background:rgba(20,20,40,0.8); border-radius:15px; padding:20px; margin-top:20px; }
        table { width:100%; border-collapse:collapse; } th,td { padding:12px; text-align:left; border-bottom:1px solid #444; }
        th { color:#00ff00; }
        button { padding:10px 20px; margin:5px; background:#333; color:white; border:none; border-radius:5px; cursor:pointer; }
        .back-link { color:#00ff00; text-decoration:none; display:inline-block; margin-top:20px; }
    </style></head>
    <body><div class="container"><div class="header"><h1>📊 Security Analytics Dashboard</h1></div>
    <div class="stats-grid">
        <div class="stat-card authorized"><div class="stat-title">Today's Statistics</div><div class="stat-number">{{ today_auth }}</div><div>✅ Authorized</div><div class="stat-number" style="color:#ff0000">{{ today_intruder }}</div><div>🔴 Intruder</div><div class="stat-number" style="color:#ffff00">{{ today_suspicious }}</div><div>⚠️ Suspicious</div></div>
        <div class="stat-card intruder"><div class="stat-title">Last 7 Days</div><div class="stat-number">{{ week_auth }}</div><div>✅ Authorized</div><div class="stat-number" style="color:#ff0000">{{ week_intruder }}</div><div>🔴 Intruder</div><div class="stat-number" style="color:#ffff00">{{ week_suspicious }}</div><div>⚠️ Suspicious</div></div>
        <div class="stat-card total"><div class="stat-title">All Time Total</div><div class="stat-number">{{ total_auth }}</div><div>✅ Authorized</div><div class="stat-number" style="color:#ff0000">{{ total_intruder }}</div><div>🔴 Intruder</div><div class="stat-number" style="color:#ffff00">{{ total_suspicious }}</div><div>⚠️ Suspicious</div></div>
    </div>
    <div class="data-table"><h3>Last 7 Days Breakdown</h3><table><thead><tr><th>Date</th><th>Authorized</th><th>Intruder</th><th>Suspicious</th></tr></thead><tbody>{% for day in week_data %}<tr><td>{{ day.date }}</td><td style="color:#00ff00">{{ day.auth }}</td><td style="color:#ff0000">{{ day.intruder }}</td><td style="color:#ffff00">{{ day.susp }}</td></tr>{% endfor %}</tbody></table></div>
    <div style="text-align:center;"><button onclick="window.location.href='/admin/dashboard'">Back to Dashboard</button><button onclick="window.location.reload()">Refresh</button></div>
    <a href="/" class="back-link">← Back to Live Dashboard</a></div></body></html>""",
                                  today_auth=today_auth, today_intruder=today_intruder, today_suspicious=today_suspicious,
                                  week_auth=week_auth, week_intruder=week_intruder, week_suspicious=week_suspicious,
                                  total_auth=total_auth, total_intruder=total_intruder, total_suspicious=total_suspicious,
                                  week_data=week_data)

# ========== IMAGE LOGS WITH THREE TABS ==========


@app.route('/admin/image-logs')
@login_required
def image_logs():
    selected_date = request.args.get('date', '')
    tab = request.args.get('tab', 'authorized')
    logs_dir = "logs"

    date_folders = []
    if os.path.exists(logs_dir):
        for d in os.listdir(logs_dir):
            if os.path.isdir(os.path.join(logs_dir, d)) and len(d) == 10:
                date_folders.append(d)
        date_folders.sort(reverse=True)

    if not selected_date and date_folders:
        selected_date = date_folders[0]

    images = []
    if selected_date:
        date_path = os.path.join(logs_dir, selected_date)
        if os.path.isdir(date_path):
            for root, dirs, files in os.walk(date_path):
                for file in files:
                    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        rel_path = os.path.relpath(os.path.join(root, file), logs_dir)
                        name_parts = os.path.splitext(file)[0].split('_')
                        timestamp = ""
                        event_type = "Unknown"
                        name = "Unknown"
                        if len(name_parts) >= 7:
                            date_part = "_".join(name_parts[:3])
                            time_part = "_".join(name_parts[3:6])
                            timestamp = f"{date_part} {time_part.replace('-', ':')}"
                            event_type = name_parts[6]
                            name = " ".join(name_parts[7:]) if len(name_parts) > 7 else ""
                            if name:
                                name = name.replace('_', ' ')
                            else:
                                name = "Unknown"
                        else:
                            event_type = os.path.basename(root)
                            name = "Unknown"

                        if event_type == "Authorized":
                            category = "authorized"
                            border_color = "green"
                            display_name = name if name != "Unknown" else "Authorized Person"
                        elif event_type == "Intruder":
                            category = "intruder"
                            border_color = "red"
                            display_name = "Intruder"
                        elif event_type == "Suspicious":
                            category = "suspicious"
                            border_color = "yellow"
                            display_name = "Suspicious Movement"
                        else:
                            category = "intruder"
                            border_color = "red"
                            display_name = "Unknown"

                        images.append({
                            'path': rel_path,
                            'date': selected_date,
                            'event_type': event_type,
                            'display_name': display_name,
                            'timestamp': timestamp,
                            'category': category,
                            'border_color': border_color
                        })

    filtered_images = [img for img in images if img['category'] == tab]
    page = request.args.get('page', 1, type=int)
    per_page = 20
    total = len(filtered_images)
    start = (page - 1) * per_page
    end = start + per_page
    paginated_images = filtered_images[start:end]

    return render_template_string("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Image Logs | NeuralGuard</title>
        <style>
            * { margin:0; padding:0; box-sizing:border-box; }
            body { font-family:'Inter',sans-serif; background:linear-gradient(135deg,#0a0a0a,#1a1a2e); color:white; padding:20px; }
            .container { max-width:1400px; margin:0 auto; }
            .header { background:rgba(20,20,40,0.8); padding:20px; border-radius:10px; margin-bottom:20px; }
            .date-nav { background:rgba(20,20,40,0.6); border-radius:10px; padding:15px; margin-bottom:20px; overflow-x:auto; white-space:nowrap; }
            .date-nav a { display:inline-block; padding:8px 16px; margin:0 5px; background:#333; color:white; text-decoration:none; border-radius:20px; font-size:14px; transition:0.2s; }
            .date-nav a:hover, .date-nav a.active { background:#00ff00; color:#000; }
            .tabs { display:flex; gap:10px; margin-bottom:20px; border-bottom:1px solid #444; }
            .tab { padding:10px 20px; cursor:pointer; background:rgba(20,20,40,0.6); border-radius:10px 10px 0 0; transition:0.2s; }
            .tab.active { background:#00ff00; color:#000; font-weight:bold; }
            .tab a { color:inherit; text-decoration:none; }
            .gallery { display:grid; grid-template-columns:repeat(auto-fill,minmax(280px,1fr)); gap:20px; margin:20px 0; }
            .image-card { background:rgba(20,20,40,0.9); border-radius:10px; overflow:hidden; transition:transform 0.2s; cursor:pointer; border:2px solid; }
            .image-card:hover { transform:scale(1.02); }
            .image-card img { width:100%; height:200px; object-fit:cover; display:block; }
            .image-info { padding:12px; font-size:12px; }
            .event-type { display:inline-block; padding:2px 8px; border-radius:12px; font-size:10px; font-weight:bold; margin-bottom:8px; }
            .Authorized { background:#28a745; }
            .Intruder { background:#dc3545; }
            .Suspicious { background:#ffc107; color:#000; }
            .date { color:#aaa; font-size:10px; margin-top:5px; }
            .name { font-weight:bold; color:#00ff00; }
            .pagination { text-align:center; margin:20px 0; }
            .pagination a, .pagination span { display:inline-block; padding:8px 12px; margin:0 4px; background:#333; color:white; text-decoration:none; border-radius:5px; }
            .pagination .current { background:#007bff; }
            .back-link { color:#00ff00; text-decoration:none; display:inline-block; margin-top:20px; }
            button { padding:10px 20px; margin:5px; background:#333; color:white; border:none; border-radius:5px; cursor:pointer; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header"><h1>🖼️ Captured Images Gallery</h1><p>Authorized (green) | Intruder (red) | Suspicious (yellow)</p></div>
            {% if date_folders %}
            <div class="date-nav">
                {% for d in date_folders %}
                <a href="?date={{ d }}&tab={{ tab }}" class="{% if d == selected_date %}active{% endif %}">{{ d }}</a>
                {% endfor %}
            </div>
            {% endif %}
            <div class="tabs">
                <div class="tab {% if tab == 'authorized' %}active{% endif %}"><a href="?date={{ selected_date }}&tab=authorized">✅ Authorized Images</a></div>
                <div class="tab {% if tab == 'intruder' %}active{% endif %}"><a href="?date={{ selected_date }}&tab=intruder">🔴 Intruder Images</a></div>
                <div class="tab {% if tab == 'suspicious' %}active{% endif %}"><a href="?date={{ selected_date }}&tab=suspicious">⚠️ Suspicious Images</a></div>
            </div>
            {% if selected_date %}
                <h3>Images for {{ selected_date }} ({{ 'Authorized' if tab == 'authorized' else 'Intruder' if tab == 'intruder' else 'Suspicious' }})</h3>
                <div class="gallery">
                    {% for img in images %}
                    <div class="image-card" style="border-color: {{ img.border_color }};" onclick="window.open('/admin/image/{{ img.path }}', '_blank')">
                        <img src="/admin/image/{{ img.path }}" alt="Snapshot">
                        <div class="image-info">
                            <div class="event-type {{ img.event_type }}">{{ img.event_type }}</div>
                            <div class="name">{{ img.display_name }}</div>
                            <div class="date">{{ img.timestamp }}</div>
                        </div>
                    </div>
                    {% else %}
                    <div>No images found for this date in this category.</div>
                    {% endfor %}
                </div>
                <div class="pagination">
                    {% if page > 1 %}<a href="?date={{ selected_date }}&tab={{ tab }}&page={{ page-1 }}">Previous</a>{% endif %}
                    {% for p in range(1, (total // per_page) + 2) if p <= (total // per_page) + 1 %}
                        {% if p == page %}<span class="current">{{ p }}</span>{% else %}<a href="?date={{ selected_date }}&tab={{ tab }}&page={{ p }}">{{ p }}</a>{% endif %}
                    {% endfor %}
                    {% if page < (total // per_page) + 1 %}<a href="?date={{ selected_date }}&tab={{ tab }}&page={{ page+1 }}">Next</a>{% endif %}
                </div>
            {% else %}
                <div>No images found in any log folder.</div>
            {% endif %}
            <div style="text-align:center;"><button onclick="window.location.href='/admin/dashboard'">Back to Dashboard</button></div>
            <a href="/" class="back-link">← Back to Live Dashboard</a>
        </div>
    </body>
    </html>
    """, images=paginated_images, page=page, per_page=per_page, total=total,
                                  selected_date=selected_date, date_folders=date_folders, tab=tab)


@app.route('/admin/image/<path:filename>')
@login_required
def serve_image(filename):
    base_dir = os.path.abspath("logs")
    req_path = os.path.abspath(os.path.join(base_dir, filename))
    if not req_path.startswith(base_dir):
        return "Access denied", 403
    if os.path.exists(req_path) and os.path.isfile(req_path):
        return send_file(req_path)
    return "Image not found", 404


@app.route('/admin/logout', methods=['POST'])
def admin_logout():
    session.clear()
    return redirect(url_for('index'))


@app.route('/health')
def health():
    return jsonify({
        "status": "running",
        "camera": camera_initialized,
        "streaming": streaming_enabled,
        "authorized_faces": len(authorized_names),
        "alert_sound": alert_sound_loaded
    })


if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 NEURALGUARD AI SECURITY SYSTEM")
    print("="*60)
    print(f"✅ Authorized faces loaded: {len(authorized_names)}")
    print(f"✅ Camera status: {'OK' if camera_initialized else 'FAILED'}")
    audio_status = "Enabled for intruders only" if alert_sound_loaded else "Disabled"
    print(f"✅ Audio alerts: {audio_status}")
    print("="*60)
    print(f"🌐 Web interface: http://{Config.FLASK_HOST}:{Config.FLASK_PORT}")
    print(f"🔐 Admin Login: http://{Config.FLASK_HOST}:{Config.FLASK_PORT}/admin")
    print(f"👤 Username: {Config.AUTH_USERNAME}")
    print(f"🔑 Password: {Config.AUTH_PASSWORD}")
    print("="*60)
    print("🎯 Features: Full‑frame snapshots | Suspicious object (non‑overlapping) | Multiple faces detection | 3‑tab image galleries | Face Management | Auto logout (5min)")
    print("="*60)
    print("⚠️  Press CTRL+C to stop the server")
    print("="*60 + "\n")
    try:
        app.run(host=Config.FLASK_HOST, port=Config.FLASK_PORT, debug=False, threaded=True)
    except KeyboardInterrupt:
        print("\n👋 Shutting down...")
    finally:
        if camera is not None:
            camera.release()
        pygame.mixer.quit()