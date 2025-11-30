# app.py
import streamlit as st
from video_processor import VideoProcessor
from mlflow_tracker import start_run, log_hyperparams, log_metrics, log_artifact
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoTransformerBase, RTCConfiguration
import tempfile
import cv2
import numpy as np
import time
from ultralytics import YOLO
import av
import os
from fractions import Fraction
from PIL import Image
import threading

import datetime
import time
import mlflow


# ---------------- Page Config ----------------
st.set_page_config(page_title="YOLO Real-Time Detection with MLflow", layout="wide")
st.title("🚗 YOLO Object Detection with MLflow Tracking")

# ---------------- Mode Selection ----------------
mode = st.sidebar.radio("Processing Mode", ["Fast Preview", "Full Quality"])
frame_skip = 10 if mode == "Fast Preview" else 1
scale_factor = 0.5 if mode == "Fast Preview" else 1.0


# ---------------- Start MLflow Run ----------------
if st.button("Start MLflow Run"):
    run = start_run()
    st.session_state["mlflow_run"] = run
    st.session_state["mlflow_run_id"] = run.info.run_id    
    st.session_state["mlflow_run_active"] = True
    st.session_state["file_logged"] = False
    

    log_hyperparams({
        "model_file": "best.pt",
        "mode": mode,
        "frame_skip": frame_skip,
        "scale_factor": scale_factor,
        "confidence_threshold": 0.25
    })
    log_artifact("best.pt")
    st.success("MLflow Run Started!")

# ---------------- Initialize VideoProcessor ----------------
vp_upload = VideoProcessor(model_path="best.pt")

# ------------------- Save Video -------------------

def save_video(frames, original_fps=20, frame_skip=1, output_path="processed_output.mp4"):
    """
    Save processed frames to a video while maintaining correct playback speed.

    frames: list of frames (BGR)
    original_fps: FPS of the original video
    frame_skip: how many frames were skipped during processing
    output_path: output video file path
    """
    if len(frames) == 0:
        return None

    # Compute effective FPS for output video
    effective_fps = original_fps / frame_skip if frame_skip > 0 else original_fps

    # PyAV expects a rational-like object (has numerator/denominator) or an integer.
    # Convert float fps to int when it's effectively integral, otherwise use Fraction.
    try:
        if abs(effective_fps - round(effective_fps)) < 1e-6:
            rate_value = int(max(1, round(effective_fps)))
        else:
            rate_value = Fraction(effective_fps).limit_denominator(1001)
            if rate_value == 0:
                rate_value = 1
    except Exception:
        rate_value = int(max(1, round(effective_fps)))

    container = av.open(output_path, mode="w")
    stream = container.add_stream("h264", rate=rate_value)
    stream.width = frames[0].shape[1]
    stream.height = frames[0].shape[0]
    stream.pix_fmt = "yuv420p"

    for frame in frames:
        # Convert BGR to RGB for PyAV
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        av_frame = av.VideoFrame.from_ndarray(frame_rgb, format="rgb24")
        for packet in stream.encode(av_frame):
            container.mux(packet)

    # Flush encoder
    for packet in stream.encode():
        container.mux(packet)

    container.close()
    return output_path

# ------------------- Process Video Upload -------------------

def process_video_upload(uploaded_file, model, class_names, scale_factor=1.0, frame_skip=1):
    """
    Process uploaded video: detect objects frame by frame, draw boxes, save processed video,
    show progress bar, and display in Streamlit. Logs artifact + metrics to MLflow when run_id is available.
    """
    if not uploaded_file or not uploaded_file.type.startswith("video"):
        return

    if "processed_videos" not in st.session_state:
        st.session_state["processed_videos"] = {}

    if uploaded_file.name in st.session_state["processed_videos"]:
        st.success("Playing previously processed video")
        st.video(st.session_state["processed_videos"][uploaded_file.name])
        return

    st.info("⏳ Processing video...")
    temp_path = f"temp_{uploaded_file.name}"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    cap = cv2.VideoCapture(temp_path)
    if not cap.isOpened():
        st.error("Could not open video file")
        return

    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    fps = cap.get(cv2.CAP_PROP_FPS) or 20
    progress = st.progress(0)
    processed = 0

    # Metrics collectors
    total_detections = 0
    conf_sum = 0.0
    class_counts = {}
    inference_times = []

    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        if i % frame_skip != 0:
            continue

        small_frame = cv2.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor) if scale_factor < 1 else frame.copy()
        rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        t0 = time.time()
        results = model(rgb_frame)
        t1 = time.time()
        inference_times.append((t1 - t0) * 1000.0)

        for r in results:
            for box in r.boxes:
                xyxy_arr = np.array(box.xyxy).flatten()
                if xyxy_arr.size < 4:
                    continue
                coords = [int(round(float(x))) for x in xyxy_arr[:4]]

                conf = float(box.conf) if hasattr(box, "conf") else 0.0
                class_id = int(box.cls) if hasattr(box, "cls") else -1

                total_detections += 1
                conf_sum += conf
                class_counts[class_id] = class_counts.get(class_id, 0) + 1

                if scale_factor < 1:
                    coords = [int(round(c / scale_factor)) for c in coords]

                x1, y1, x2, y2 = coords
                h, w = frame.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w - 1, x2), min(h - 1, y2)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{class_names.get(class_id, 'Unknown')}:{conf:.2f}"
                cv2.putText(frame, label, (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        frames.append(frame)
        processed += 1
        progress.progress(min(processed / max(total_frames // frame_skip, 1), 1.0))

    cap.release()
    st.success("Video processed successfully!")

    os.makedirs("videos", exist_ok=True)
    output_path = os.path.join("videos", f"processed_{uploaded_file.name}")
    save_video(frames, original_fps=fps, frame_skip=frame_skip, output_path=output_path)

    avg_conf = (conf_sum / total_detections) if total_detections > 0 else 0.0
    avg_latency_ms = (sum(inference_times) / len(inference_times)) if inference_times else 0.0
    total_inference_seconds = sum(inference_times) / 1000.0 if inference_times else 0.0
    processing_fps = (processed / total_inference_seconds) if total_inference_seconds > 0 else 0.0

    # Safe MLflow logging: use stored run_id from session_state or skip with a warning
    run_id = st.session_state.get("mlflow_run_id")
    if not run_id:
        st.warning("No active MLflow run. Press 'Start MLflow Run' before logging artifacts/metrics.")
    else:
        try:
            # artifact_path and run_id are passed to mlflow_tracker helpers (client fallback supported)
            log_artifact(output_path, artifact_path="videos", run_id=run_id)
            metrics = {
                "Avg_Confidence": float(avg_conf),
                "Total_Detections": int(total_detections),
                "Processed_Frames": int(processed),
                "Avg_Inference_ms": float(avg_latency_ms),
                "Processing_FPS": float(processing_fps),
            }
            for cls_id, cnt in class_counts.items():
                name = class_names.get(cls_id, str(cls_id))
                metrics[f"count_{name}"] = int(cnt)

            log_metrics(metrics, run_id=run_id)
            st.session_state["file_logged"] = True
            st.success("Logged processed video and metrics to MLflow.")
        except Exception as e:
            st.error(f"MLflow logging failed: {e}")

    st.video(output_path)
    st.session_state["processed_videos"][uploaded_file.name] = output_path
    return output_path

# ---------------- Pre-warm Model ----------------
if "vp_loaded" not in st.session_state:
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    vp_upload.process_frame(dummy_frame)
    st.session_state["vp_loaded"] = True

# ---------------- File Uploader ----------------
uploaded_file = st.file_uploader(
    "Upload Image/Video",
    type=["png","jpg","jpeg","mp4","mov","avi","mkv"],
    key="upload_image_video"
)

if uploaded_file:
    if uploaded_file.type.startswith("image"):
        # Process image
        frame = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
        result_img, saved_path = vp_upload.process_frame(frame, save_name="annotated_image.jpg")
        st.image(result_img, channels="BGR", use_container_width=True)

        if st.session_state.get("mlflow_run_active") and not st.session_state.get("file_logged"):
            #log_artifact(saved_path)
            run_id = st.session_state.get("mlflow_run_id")
            if run_id:
                try:
                    log_artifact(saved_path, artifact_path="images", run_id=run_id)
                    st.success("Image logged to MLflow.")
                except Exception as e:
                    st.error(f"Failed to log image to MLflow: {e}")
            else:
                st.warning("No active MLflow run — press 'Start MLflow Run' before uploading if you want to log artifacts.")
            counts, avg_conf, fps, latency = vp_upload.get_metrics()
            log_metrics({"Avg_Confidence": avg_conf, "FPS": fps, "Latency_ms": latency})
            st.session_state["file_logged"] = True

    elif uploaded_file.type.startswith("video"):
            # Process video
            model = vp_upload.model
            class_names = vp_upload.get_class_names()
            process_video_upload(
                uploaded_file,
                model=model,
                class_names=class_names,
                scale_factor=scale_factor,
                frame_skip=frame_skip)
# ---------------- Live Webcam ----------------
st.title("YOLO Real-Time Webcam Detection")

# ------------------ YOLO Transformer with MLflow ------------------
class YOLOTransformer(VideoTransformerBase):
    def __init__(self, class_names=None):
        self.model = YOLO("best.pt")
        self.avg_conf = 0
        self.object_counts = {}
        self.fps = 0
        self.latency = 0
        self.frame_counter = 0
        # class_names: dict {cls_id: name}
        self.class_names = class_names if class_names else {}

        # Warm-up
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        self.model(dummy)

    def transform(self, frame):
        start = time.time()
        img = frame.to_ndarray(format="bgr24")
        results = self.model(img)
        end = time.time()

        self.latency = (end - start) * 1000
        self.fps = 1 / (end - start) if (end - start) > 0 else 0

        # Map counts to names
        confs = []
        counts = {}
        for r in results:
            for box in r.boxes:
                confs.append(float(box.conf))
                cls_id = int(box.cls)
                cls_name = self.class_names.get(cls_id, str(cls_id))
                counts[cls_name] = counts.get(cls_name, 0) + 1

        self.avg_conf = sum(confs)/len(confs) if confs else 0
        self.object_counts = counts

        # Log metrics to MLflow
        self.frame_counter += 1
        if self.frame_counter % 10 == 0:
            mlflow.log_metric("FPS", self.fps)
            mlflow.log_metric("Latency_ms", self.latency)
            mlflow.log_metric("Avg_Confidence", self.avg_conf)
            for cls_name, count in self.object_counts.items():
                mlflow.log_metric(f"Object_{cls_name}_Count", count)

        annotated = results[0].plot() if results else img
        return annotated


# ------------------ WebRTC Config ------------------
RTC_CONFIG = RTCConfiguration({"iceServers":[{"urls":["stun:stun.l.google.com:19302"]}]})

class_names = vp_upload.get_class_names()  # same as used for video upload
ctx = webrtc_streamer(
    key="yolo-webcam",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIG,
    video_transformer_factory=lambda: YOLOTransformer(class_names=class_names),
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)


# ------------------ Streamlit Live Metrics ------------------
st.markdown("### Live Metrics")
if ctx.video_transformer:
    counts, avg_conf, fps, latency = (
        ctx.video_transformer.object_counts,
        ctx.video_transformer.avg_conf,
        ctx.video_transformer.fps,
        ctx.video_transformer.latency,
    )
    st.write(f"FPS: {fps:.1f}, Latency: {latency:.1f} ms, Avg Confidence: {avg_conf*100:.1f}%")
    st.write("Object Counts:", counts)
else:
    st.info("Start the camera")
