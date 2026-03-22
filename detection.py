import cv2
import numpy as np
from ultralytics import YOLO

class PlateDetector:
    def load_model(self, weight_path: str, cfg_path: str = None):
        # On charge ton fichier YOLOv8 (.pt) et on ignore le paramètre cfg_path
        self.model = YOLO(weight_path)
        with open("classes-detection.names", "r") as f:
            self.classes = [line.strip() for line in f.readlines()]

    def load_image(self, img_path):
        img = cv2.imread(img_path)
        height, width, channels = img.shape
        return img, height, width, channels

    def detect_plates(self, img):
        # YOLOv8 fait le travail du "blob" automatiquement
        results = self.model.predict(img, conf=0.3, verbose=False)
        return None, results
        
    def get_boxes(self, outputs, width, height, threshold=0.3):
        boxes = []
        confidences = []
        class_ids = []
        
        # 'outputs' est l'objet results de YOLOv8
        for result in outputs:
            for box in result.boxes:
                conf = float(box.conf[0])
                if conf > threshold:
                    # Coordonnées (x_min, y_min, x_max, y_max)
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    w = x2 - x1
                    h = y2 - y1
                    
                    boxes.append([x1, y1, w, h])
                    confidences.append(conf)
                    class_ids.append(int(box.cls[0]))
                    
        return boxes, confidences, class_ids

    def draw_labels(self, boxes, confidences, class_ids, img):
        # Ton code original de dessin et redimensionnement
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.1, 0.1)
        font = cv2.FONT_HERSHEY_PLAIN
        plats = []
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                color_green = (0, 255, 0)
                crop_img = img[y:y+h, x:x+w]
                try:
                    crop_resized = cv2.resize(crop_img, dsize=(470, 110))
                    plats.append(crop_resized)
                    cv2.rectangle(img, (x, y), (x + w, y + h), color_green, 8)
                    confidence = round(confidences[i], 3) * 100
                    cv2.putText(img, str(confidence) + "%", (x + 20, y - 20), font, 12, (0, 255, 0), 6)
                except cv2.error as err:
                    print(err)

        return img, plats
