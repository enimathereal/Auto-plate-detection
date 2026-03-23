import cv2
import numpy as np
from ultralytics import YOLO

class PlateDetector:
    def load_model(self, weight_path: str):
        self.model = YOLO(weight_path)

    def load_image(self, img_path):
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Image introuvable : {img_path}")
        h, w, c = img.shape
        return img, h, w, c

    def detect_plates(self, img):
        # Inférence YOLOv8
        results = self.model.predict(img, imgsz=640, conf=0.25, verbose=False)
        return None, results
        
    def get_boxes(self, outputs, width, height, threshold=0.25):
        boxes, confidences, class_ids = [], [], []
        if not outputs or len(outputs) == 0:
            return boxes, confidences, class_ids
        
        result = outputs[0]
        if not hasattr(result, 'boxes') or result.boxes is None:
            return boxes, confidences, class_ids

        for box in result.boxes:
            conf = float(box.conf[0].item())
            if conf > threshold:
                xyxy = box.xyxy[0].tolist()
                x1, y1, x2, y2 = map(int, xyxy)
                boxes.append([x1, y1, x2 - x1, y2 - y1])
                confidences.append(conf)
                class_ids.append(int(box.cls[0].item()))
        return boxes, confidences, class_ids

    def draw_labels(self, boxes, confidences, class_ids, img):
        plats = []
        img_h, img_w = img.shape[:2]
        for i in range(len(boxes)):
            x, y, w, h = boxes[i]
            x, y = max(0, x), max(0, y)
            x2, y2 = min(img_w, x + w), min(img_h, y + h)
            
            crop_img = img[y:y2, x:x2]
            if crop_img.size > 0:
                try:
                    crop_resized = cv2.resize(crop_img, (470, 110))
                    plats.append(crop_resized)
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
                except:
                    pass
        return img, plats
