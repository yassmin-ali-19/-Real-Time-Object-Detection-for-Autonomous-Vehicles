from ultralytics import YOLO
from PIL import Image

class ObjectDetectionModel:
    """
    Wrapper around YOLO model to handle loading and predictions.
    """
    def __init__(self, model_path="best.pt"):
        self.model = YOLO(model_path)

    def predict(self, image: Image.Image, conf: float = 0.25):
        """
        Run YOLO inference on a PIL image.
        Returns detections: [x1, y1, x2, y2, conf, class_id]
        """
        results = self.model.predict(image, conf=conf, verbose=False)
        detections = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf_val = float(box.conf[0])
                class_id = int(box.cls[0])
                detections.append([x1, y1, x2, y2, conf_val, class_id])
        return detections

    def get_class_names(self):
        """Return YOLO class names dictionary {id: name}"""
        return self.model.names
