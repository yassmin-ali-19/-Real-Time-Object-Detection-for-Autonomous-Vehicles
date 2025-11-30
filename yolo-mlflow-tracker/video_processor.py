import cv2
import time
from ultralytics import YOLO
import os

class VideoProcessor:
    def __init__(self, model_path="best.pt", save_dir="annotated"):
        self.model = YOLO(model_path)
        self.fps = 0
        self.latency = 0
        self.avg_conf = 0
        self.object_counts = {}
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        
    def get_class_names(self):
        """Return YOLO class names dictionary {id: name}"""
        return self.model.names

    def process_frame(self, frame, save_name=None):
        start_time = time.time()
        results = self.model(frame)
        end_time = time.time()

        # Update metrics
        self.latency = (end_time - start_time) * 1000
        self.fps = 1 / (end_time - start_time) if (end_time - start_time) > 0 else 0

        # Object counts and confidence
        confs, counts = [], {}
        for r in results:
            for box in r.boxes:
                confs.append(float(box.conf))
                cls = int(box.cls)
                counts[cls] = counts.get(cls, 0) + 1

        self.avg_conf = sum(confs)/len(confs) if confs else 0
        self.object_counts = counts

        annotated_frame = results[0].plot() if results else frame

        # Save if needed
        save_path = None
        if save_name:
            save_path = os.path.join(self.save_dir, save_name)
            cv2.imwrite(save_path, annotated_frame)

        return annotated_frame, save_path

    def get_metrics(self):
        return self.object_counts, self.avg_conf, self.fps, self.latency
    
    def transform(self, frame):
        """Used for live webcam streaming"""
        annotated_frame, _ = self.process_frame(frame)
        return annotated_frame
    
    def process_video(self, video_path, output_name="annotated_video.mp4", scale_factor=1.0):
        os.makedirs(self.save_dir, exist_ok=True)
        output_path = os.path.join(self.save_dir, output_name)
        
        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * scale_factor)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * scale_factor)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if scale_factor != 1.0:
                frame = cv2.resize(frame, (width, height))
            annotated_frame, _ = self.process_frame(frame)
            out.write(annotated_frame)

        cap.release()
        out.release()
        return output_path
