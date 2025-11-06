import os
import uuid
import threading
from collections import Counter
from datetime import datetime
from pathlib import Path
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, Response, flash
import cv2
from imutils.video import FPS
from twilio.rest import Client
import playsound
import subprocess

# ===============================================================
# CONFIGURATION
# ===============================================================
BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "static" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

SIREN_PATH = str(BASE_DIR / "static" / "siren" / "Siren.wav")
TWILIO_ACCOUNT_SID = "xxxxxxxxxxxxxxxxxxxxxx"
TWILIO_AUTH_TOKEN = "yyyyyyyyyyyyyyyyyyyyyyy"
TWILIO_PHONE_NUMBER = "+12xxxxxxxx"
FARM_OWNER_NUMBER = "+91xxxxxxxxxxxx"

MODEL_PATH = r"D:\ANIMAL-DETECTION-main\ANIMAL-DETECTION-main\runs\detect\train5\weights\best.pt"
CONF_THRESH = 0.35

app = Flask(__name__)
app.secret_key = "wildvision-secret-key"
twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

# ===============================================================
# LOAD YOLO MODEL
# ===============================================================
yolo_enabled = False
try:
    from ultralytics import YOLO
    print("[INFO] Loading YOLO model...")
    model = YOLO(MODEL_PATH)
    print("[INFO] YOLO model loaded successfully.")
    yolo_enabled = True
except Exception as e:
    print("[WARN] YOLO failed to load:", e)
    print("[INFO] Running UI-only mode (no detection).")

# ===============================================================
# STATE VARIABLE
# ===============================================================
live_streaming = {"running": False}

# ===============================================================
# NEWS SECTION
# ===============================================================
TN_NEWS = [
    {
        "date": "2025-10-03",
        "title": "Elephant herd blocks highway near Coimbatore; traffic diverted",
        "summary": "A herd of elephants moved across the Salem–Coimbatore road at dawn, forcing authorities to divert traffic for two hours."
    },
    {
        "date": "2025-09-25",
        "title": "Leopard sighted in outskirts of Erode; villagers alerted",
        "summary": "Local residents reported a leopard near farmland. Forest department teams responded and mounted a camera trap survey."
    },
    {
        "date": "2025-08-12",
        "title": "Wild boar damages crops near Madurai villages",
        "summary": "Repeated crop damage by wild boar has affected farmers; compensation and preventive fencing discussed."
    },
    {
        "date": "2025-06-30",
        "title": "Stray tiger calf rescued near Kalrayan Hills",
        "summary": "Forest personnel rescued an injured tiger cub and transferred it to the wildlife care centre."
    }
]

# ===============================================================
# HELPERS
# ===============================================================
def send_sms(phone_number, detected_time, detected_class):
    """Send SMS alert via Twilio"""
    try:
        body = f"🚨 Wild Vision Alert: {detected_class} detected at {detected_time}"
        message = twilio_client.messages.create(
            body=body, from_=TWILIO_PHONE_NUMBER, to=phone_number)
        print("[INFO] SMS sent:", message.sid)
        return True
    except Exception as e:
        print("[ERROR] SMS failed:", e)
        return False


def play_siren(path=SIREN_PATH):
    """Play siren alert sound"""
    if os.path.exists(path):
        try:
            threading.Thread(target=playsound.playsound, args=(path,), daemon=True).start()
        except Exception as e:
            print("[ERROR] Siren playback failed:", e)
    else:
        print("[WARN] Siren file missing:", path)


def annotate_frame(frame, results):
    """Draw detection boxes"""
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = f"{r.names[cls_id]} ({conf:.2f})"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 128), 2)
            cv2.putText(frame, label, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 128), 2)
    return frame


import subprocess

def process_uploaded_video(in_path, out_path):
    if not yolo_enabled:
        import shutil
        shutil.copy(in_path, out_path)
        return ["Demo Mode"], CONF_THRESH

    cap = cv2.VideoCapture(str(in_path))
    if not cap.isOpened():
        raise RuntimeError("Could not open uploaded video")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    temp_out = str(out_path).replace(".mp4", "_raw.mp4")
    out = cv2.VideoWriter(temp_out, fourcc, fps, (width, height))

    detected_classes = set()
    avg_conf = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = model.predict(frame, conf=CONF_THRESH, verbose=False)
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                detected_classes.add(r.names[cls_id])
                avg_conf.append(conf)
        frame = annotate_frame(frame, results)
        out.write(frame)

    cap.release()
    out.release()

    # ✅ Re-encode using FFmpeg for browser compatibility
    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", temp_out,
                "-vcodec", "libx264", "-acodec", "aac",
                "-movflags", "faststart", str(out_path)
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT
        )
        os.remove(temp_out)
    except Exception as e:
        print("[WARN] FFmpeg conversion failed:", e)
        os.rename(temp_out, out_path)

    avg_conf_val = round(sum(avg_conf) / len(avg_conf), 2) if avg_conf else CONF_THRESH
    return list(detected_classes), avg_conf_val


def generate_live_frames(conf_thresh=CONF_THRESH):
    """Generate frames for live detection"""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Camera not available.")
        return

    fps = FPS().start()
    count = []
    flag = False
    detected_class = None

    while live_streaming.get("running", True):
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(frame, conf=conf_thresh, verbose=False)
        det_flag = 0
        for r in results:
            for b in r.boxes:
                detected_class = r.names[int(b.cls[0])]
                det_flag = 1

        frame = annotate_frame(frame, results)
        count.append(det_flag)

        if Counter(count[-36:])[1] > 15 and not flag and detected_class:
            detected_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[ALERT] {detected_class} detected at {detected_time}")
            play_siren()
            send_sms(FARM_OWNER_NUMBER, detected_time, detected_class)
            flag = True

        ret2, jpeg = cv2.imencode('.jpg', frame)
        if not ret2:
            continue
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
        fps.update()

    cap.release()
    fps.stop()
    print("[INFO] Live stream ended.")


# ===============================================================
# FLASK ROUTES
# ===============================================================
@app.route("/")
def index():
    abstract = ("Unexpected wild animal appearances near forests often lead to accidents and conflict. "
                "Wild Vision uses AI (YOLOv11) to detect and alert in real-time.")
    scope = ("Enhance safety using AI-driven detection, minimizing animal-human conflicts.")
    problem = ("Wild animal intrusions threaten safety and agriculture in Tamil Nadu.")
    return render_template("index.html", abstract=abstract, scope=scope, problem=problem, news=TN_NEWS)


@app.route("/upload", methods=["GET", "POST"])
def upload():
    if request.method == "GET":
        return render_template("upload.html")

    if 'file' not in request.files:
        flash("No file part")
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash("No selected file")
        return redirect(request.url)

    filename = f"{uuid.uuid4().hex}_{file.filename}"
    in_path = UPLOAD_DIR / filename
    file.save(str(in_path))
    out_filename = f"out_{filename}"
    out_path = UPLOAD_DIR / out_filename

    try:
        detected_list, avg_conf = process_uploaded_video(in_path, out_path)
    except Exception as e:
        flash(f"Video processing error: {e}")
        return redirect(request.url)

    return render_template(
        "result.html",
        video_file=out_filename,
        detected=detected_list,
        CONF_THRESH=avg_conf
    )


@app.route("/live")
def live():
    global live_streaming
    live_streaming["running"] = True
    return render_template("live.html", FARM_OWNER_NUMBER=FARM_OWNER_NUMBER, CONF_THRESH=CONF_THRESH)


@app.route("/stop_live")
def stop_live():
    global live_streaming
    live_streaming["running"] = False
    return redirect(url_for("index"))


@app.route("/video_feed")
def video_feed():
    return Response(generate_live_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    """Serve uploaded or processed video files"""
    file_path = UPLOAD_DIR / filename
    if not file_path.exists():
        return "File not found", 404
    return send_from_directory(UPLOAD_DIR, filename, mimetype="video/mp4", as_attachment=False)


# ===============================================================
# MAIN ENTRY
# ===============================================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
