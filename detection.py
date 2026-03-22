import cv2
from ultralytics import YOLO
import numpy as np

class PlateDetector:
    def __init__(self):
        self.model = None

    def load_model(self, weight_path: str):
        # On remplace l'ancien cv2.dnn par la librairie moderne Ultralytics
        self.model = YOLO(weight_path)

    def load_image(self, img_path):
        img = cv2.imread(img_path)
        height, width, channels = img.shape
        return img, height, width, channels

    def process_image(self, img):
        # Détection avec YOLOv8
        results = self.model.predict(img, conf=0.5, verbose=False)
        plats = []
        
        for result in results:
            for box in result.boxes.xyxy.cpu().numpy():
                x1, y1, x2, y2 = map(int, box)
                
                # Découpage et redimensionnement (Logique exacte de Hamza)
                crop_img = img[y1:y2, x1:x2]
                try:
                    crop_resized = cv2.resize(crop_img, dsize=(470, 110))
                    plats.append(crop_resized)
                    # Dessin du rectangle sur l'image originale
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 8)
                except cv2.error as err:
                    print(err)

        return img, plats
