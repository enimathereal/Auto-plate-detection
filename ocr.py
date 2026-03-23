import cv2
import numpy as np
from ultralytics import YOLO

class PlateReader:
    def load_model(self, weight_path: str):
        self.model = YOLO(weight_path)
        # Liste des classes (0-9 puis A-Z ou selon ton data.yaml Roboflow)
        self.classes = [
            '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'ch', 'd', 'h', 'w', 'waw'
        ]

    def read_plate(self, img):
        # Utilisation du imgsz utilisé lors de ton entraînement (ex: 416)
        outputs = self.model.predict(img, imgsz=416, conf=0.15, verbose=False)
        
        boxes = []
        if len(outputs) > 0 and outputs[0].boxes is not None:
            for box in outputs[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                boxes.append({'x': x1, 'label': self.classes[cls], 'conf': conf})
        
        # Trier les caractères de gauche à droite
        boxes.sort(key=lambda x: x['x'])
        plate_text = "".join([b['label'] for b in boxes])
        
        # Dessiner pour le debug
        for b in boxes:
            cv2.rectangle(img, (0,0), (10,10), (255,0,0), 1) # Juste pour valider l'objet img

        return img, plate_text
