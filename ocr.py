import cv2
import numpy as np
from ultralytics import YOLO

class PlateReader:
    def load_model(self, weight_path: str):
        """Charge le modèle YOLOv8 pour l'OCR."""
        self.model = YOLO(weight_path)
        self.classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 
                        'a', 'b', 'ch', 'd', 'h', 'w', 'waw']

    def read_plate(self, img):
        # --- PRÉTRAITEMENT SIMPLE ET EFFICACE ---
        # On repasse au gris simple sans forcer le noir et blanc
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_resized = cv2.resize(gray, (640, 160)) 
        
        # Conversion en RGB pour plaire à YOLOv8
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)
        
        # --- PRÉDICTION ---
        results = self.model.predict(img_rgb, imgsz=640, conf=0.5, verbose=False)
        
        chars = []
        if len(results) > 0 and results[0].boxes is not None:
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                
                # On dessine les petits rectangles verts sur les lettres
                cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                chars.append({'x': x1, 'val': self.classes[cls], 'conf': conf})
        
        # --- TRI ET FILTRAGE ---
        chars.sort(key=lambda x: x['x'])
        
        final_chars = []
        if chars:
            final_chars.append(chars[0])
            for i in range(1, len(chars)):
                if abs(chars[i]['x'] - final_chars[-1]['x']) < 20:
                    if chars[i]['conf'] > final_chars[-1]['conf']:
                        final_chars[-1] = chars[i]
                else:
                    final_chars.append(chars[i])
        
        plate_text = "".join([c['val'] for c in final_chars])
        
        return img_rgb, plate_text
