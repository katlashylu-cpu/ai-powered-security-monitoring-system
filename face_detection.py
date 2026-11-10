# Optimized face detection with suspicious movement and full‑frame snapshots
import face_recognition
import cv2
import numpy as np
import os
import re
import hashlib
from collections import deque, defaultdict
import pygame
from datetime import datetime
import threading
import time


class Config:
    FACE_TOLERANCE = 0.55
    RECOGNITION_BUFFER_SIZE = 10
    LOG_COOLDOWN_AUTHORIZED = 30
    LOG_COOLDOWN_INTRUDER = 5
    LOG_COOLDOWN_SUSPICIOUS = 10
    SUSPICIOUS_AREA_THRESHOLD = 3000
    FRAME_SKIP = 2
    SCALE_FACTOR = 0.5
    FACE_MODEL = "hog"
    MAX_LOG_ENTRIES = 10000
    SNAPSHOT_QUALITY = 85
    CLEANUP_OLD_LOGS_DAYS = 30
    MOTION_FRAME_INTERVAL = 3
    MULTIPLE_FACES_COOLDOWN = 15          # seconds between multiple‑face alerts
    MULTIPLE_FACES_MIN_COUNT = 2          # trigger when >=2 faces


def sanitize_filename(filename):
    if not filename:
        return "unknown"
    return re.sub(r'[^a-zA-Z0-9_\-\.]', '_', str(filename))[:100]


def cleanup_old_logs(base_log_dir="logs", days_old=30):
    try:
        cutoff = time.time() - (days_old * 86400)
        for root, dirs, files in os.walk(base_log_dir):
            for file in files:
                path = os.path.join(root, file)
                try:
                    if os.path.getmtime(path) < cutoff:
                        os.remove(path)
                except:
                    pass
            for d in dirs:
                dir_path = os.path.join(root, d)
                try:
                    if not os.listdir(dir_path):
                        os.rmdir(dir_path)
                except:
                    pass
    except Exception as e:
        print(f"⚠️ Log cleanup error: {e}")


def check_disk_space(path="logs", required_mb=100):
    try:
        import shutil
        free_mb = shutil.disk_usage(path).free / (1024*1024)
        if free_mb < required_mb:
            print(f"⚠️ Low disk space: {free_mb:.1f}MB free")
            return False
        return True
    except:
        return True


def boxes_overlap(box1, box2):
    """Check if two rectangles overlap.
       box format: (x, y, w, h) or (x1, y1, x2, y2) - we'll use (x,y,w,h)."""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    return not (x1 + w1 < x2 or x2 + w2 < x1 or y1 + h1 < y2 or y2 + h2 < y1)


folder_path = "authorized_faces"
authorized_encodings = []
authorized_names = []
if not os.path.exists(folder_path):
    os.makedirs(folder_path)
    print(f"📁 Created {folder_path} folder. Add authorized face images.")
else:
    for filename in os.listdir(folder_path):
        if filename.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
            path = os.path.join(folder_path, filename)
            try:
                image = face_recognition.load_image_file(path)
                encodings = face_recognition.face_encodings(image)
                if encodings:
                    for enc in encodings:
                        authorized_encodings.append(enc)
                        authorized_names.append(sanitize_filename(
                            os.path.splitext(filename)[0]))
                    print(f"✅ Loaded: {filename}")
                else:
                    print(f"⚠️ No face in: {filename}")
            except Exception as e:
                print(f"❌ Error {filename}: {e}")

print("\n--- Encoding Summary ---")
print("Authorized names:", authorized_names)
print("Number of faces:", len(authorized_encodings))

video_capture = None
camera_lock = threading.Lock()


def init_camera():
    global video_capture
    with camera_lock:
        if video_capture is not None:
            video_capture.release()
        video_capture = cv2.VideoCapture(0)
        if not video_capture.isOpened():
            return False
        video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        video_capture.set(cv2.CAP_PROP_FPS, 30)
        return True


if not init_camera():
    print("❌ Cannot open camera. Exiting.")
    exit(1)

pygame.mixer.init()
alert_sound_loaded = False
if os.path.exists("alert.mp3"):
    try:
        pygame.mixer.music.load("alert.mp3")
        alert_sound_loaded = True
    except:
        pass

process_this_frame = True
last_face_locations = []
last_face_names = []
recognition_buffer = defaultdict(
    lambda: deque(maxlen=Config.RECOGNITION_BUFFER_SIZE))
last_logged = {}
fgbg = cv2.createBackgroundSubtractorMOG2()
last_alert_time = 0
frame_count = 0
running = True
motion_counter = 0
last_multiple_faces_time = 0


def log_event(frame, location, name, event_type=None, force_log=False):
    global last_logged
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
    date_folder = now.strftime("%Y-%m-%d")
    if event_type is None:
        event_type = "Authorized" if name != "Unauthorized" else "Intruder"
    name = sanitize_filename(name)
    event_type = sanitize_filename(event_type)
    cooldown = 0 if force_log else (Config.LOG_COOLDOWN_AUTHORIZED if event_type == "Authorized"
                                    else Config.LOG_COOLDOWN_INTRUDER if event_type == "Intruder" else Config.LOG_COOLDOWN_SUSPICIOUS)
    key = (name, event_type)
    if not force_log and key in last_logged and (now.timestamp() - last_logged[key]) < cooldown:
        return False
    if not check_disk_space():
        return False
    day_folder = os.path.join("logs", date_folder)
    event_folder = os.path.join(day_folder, event_type)
    os.makedirs(event_folder, exist_ok=True)
    log_path = os.path.join(day_folder, "detection_log.txt")
    try:
        with open(log_path, "a") as f:
            f.write(
                f"{timestamp} - Event: {event_type} - Name: {name}, Location: {location}\n")
        last_logged[key] = now.timestamp()
        if len(last_logged) > Config.MAX_LOG_ENTRIES:
            for _ in range(int(Config.MAX_LOG_ENTRIES*0.2)):
                if last_logged:
                    del last_logged[min(last_logged, key=last_logged.get)]
    except IOError as e:
        print(f"Log error: {e}")
        return False
    if frame is not None:
        try:
            snapshot_path = os.path.join(
                event_folder, f"{timestamp}_{event_type}_{name}.jpg")
            cv2.imwrite(snapshot_path, frame, [
                        cv2.IMWRITE_JPEG_QUALITY, Config.SNAPSHOT_QUALITY])
        except Exception as e:
            print(f"Snapshot error: {e}")
    return True


def detect_suspicious_movement(frame, face_rects_scaled):
    """
    Detect large motion, draw yellow box only if it does NOT overlap any face rectangle.
    face_rects_scaled: list of (x, y, w, h) in original frame coordinates.
    """
    global motion_counter
    motion_counter += 1
    if motion_counter % Config.MOTION_FRAME_INTERVAL != 0:
        return
    try:
        fgmask = fgbg.apply(frame)
        thresh = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=2)
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > Config.SUSPICIOUS_AREA_THRESHOLD:
                x, y, w, h = cv2.boundingRect(cnt)
                # Check if this motion box overlaps any face box
                overlap = False
                for (fx, fy, fw, fh) in face_rects_scaled:
                    if boxes_overlap((x, y, w, h), (fx, fy, fw, fh)):
                        overlap = True
                        break
                if overlap:
                    continue  # skip – motion caused by a person's movement
                # Draw yellow box for non‑face object
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
                cv2.putText(frame, "Suspicious Object", (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                log_event(frame, (y, x+w, y+h, x),
                          "Unknown", event_type="Suspicious")
    except Exception as e:
        print(f"Movement error: {e}")


def detect_multiple_faces(frame, face_locations_scaled, current_time):
    """Log a suspicious event if multiple faces are present (cooldown applied)."""
    global last_multiple_faces_time
    if len(face_locations_scaled) >= Config.MULTIPLE_FACES_MIN_COUNT:
        if (current_time - last_multiple_faces_time) > Config.MULTIPLE_FACES_COOLDOWN:
            last_multiple_faces_time = current_time
            # Draw a grouped bounding box around all faces (optional)
            if face_locations_scaled:
                xs = [r[0] for r in face_locations_scaled]
                ys = [r[1] for r in face_locations_scaled]
                x2s = [r[0]+r[2] for r in face_locations_scaled]
                y2s = [r[1]+r[3] for r in face_locations_scaled]
                group_x = min(xs)
                group_y = min(ys)
                group_w = max(x2s) - group_x
                group_h = max(y2s) - group_y
                cv2.rectangle(frame, (group_x, group_y), (group_x +
                              group_w, group_y+group_h), (0, 255, 255), 3)
                cv2.putText(frame, f"Multiple Faces ({len(face_locations_scaled)})", (group_x, group_y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            log_event(frame, (0, 0, 0, 0), "MultipleFaces",
                      event_type="Suspicious", force_log=True)


print("\n🚀 Starting security monitoring... Press ESC to quit\n")
cleanup_old_logs()
cv2.namedWindow("Security Monitoring System", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Security Monitoring System",
                      cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

try:
    while running:
        with camera_lock:
            if video_capture is None or not video_capture.isOpened():
                if not init_camera():
                    time.sleep(1)
                    continue
            ret, frame = video_capture.read()
        if not ret:
            time.sleep(0.1)
            continue
        frame_count += 1
        if frame_count % Config.FRAME_SKIP != 0:
            cv2.imshow('Security Monitoring System', frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
            continue

        small = cv2.resize(
            frame, (0, 0), fx=Config.SCALE_FACTOR, fy=Config.SCALE_FACTOR)
        rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        face_locations = []
        face_names = []
        if process_this_frame:
            try:
                face_locations = face_recognition.face_locations(
                    rgb_small, model=Config.FACE_MODEL)
                face_encodings = face_recognition.face_encodings(
                    rgb_small, face_locations)
                for enc in face_encodings:
                    name = "Unauthorized"
                    if authorized_encodings:
                        matches = face_recognition.compare_faces(
                            authorized_encodings, enc, tolerance=Config.FACE_TOLERANCE)
                        distances = face_recognition.face_distance(
                            authorized_encodings, enc)
                        if len(distances) > 0:
                            best = np.argmin(distances)
                            if matches[best] and distances[best] < Config.FACE_TOLERANCE:
                                name = authorized_names[best]
                    buf_key = hashlib.md5(enc.tobytes()).hexdigest()[:16]
                    recognition_buffer[buf_key].append(name)
                    name = max(
                        set(recognition_buffer[buf_key]), key=recognition_buffer[buf_key].count)
                    face_names.append(name)
                last_face_locations = face_locations
                last_face_names = face_names
            except Exception as e:
                print(f"Face error: {e}")
                face_locations = last_face_locations
                face_names = last_face_names
        else:
            face_locations = last_face_locations
            face_names = last_face_names
        process_this_frame = not process_this_frame

        scale = int(1/Config.SCALE_FACTOR)
        # Build list of face rectangles in original frame coordinates (x, y, w, h)
        face_rects_original = []
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            top, right, bottom, left = top*scale, right*scale, bottom*scale, left*scale
            h, w = frame.shape[:2]
            top = max(0, min(top, h))
            bottom = max(0, min(bottom, h))
            left = max(0, min(left, w))
            right = max(0, min(right, w))
            face_rects_original.append((left, top, right-left, bottom-top))
            is_auth = name != "Unauthorized"
            color = (0, 255, 0) if is_auth else (0, 0, 255)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.rectangle(frame, (left, bottom-35),
                          (right, bottom), color, cv2.FILLED)
            cv2.putText(frame, name, (left+6, bottom-6),
                        cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)
            if not is_auth:
                now_time = time.time()
                if alert_sound_loaded and (now_time - last_alert_time) > 3:
                    try:
                        pygame.mixer.music.play()
                        last_alert_time = now_time
                    except:
                        pass
            log_event(frame, (top, right, bottom, left), name)

        # Detect suspicious movement (object) – skip if overlaps any face
        detect_suspicious_movement(frame, face_rects_original)

        # Detect multiple faces as suspicious
        current_time = time.time()
        detect_multiple_faces(frame, face_rects_original, current_time)

        cv2.putText(frame, f"Faces: {len(face_names)} | Authorized: {sum(1 for n in face_names if n != 'Unauthorized')}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow('Security Monitoring System', frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
except KeyboardInterrupt:
    print("\n⚠️ Interrupted")
finally:
    with camera_lock:
        if video_capture:
            video_capture.release()
    cv2.destroyAllWindows()
    pygame.mixer.quit()
    print("✅ System shutdown")
