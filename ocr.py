import cv2
import numpy as np
from ultralytics import YOLO

class PlateReader:
    def __init__(self):
        self.model = None

    def load_model(self, weight_path: str):
        self.model = YOLO(weight_path)

    def arabic_chars(self, label):
        # Conversion des labels YOLO en lettres arabes (Logique de Hamza)
        mapping = {'a': 'أ', 'b': 'ب', 'w': 'و', 'waw': 'و', 'd': 'د', 'h': 'ه', 'ch': 'ش'}
        return mapping.get(label, label)

    def read_plate(self, plate_img):
        results = self.model.predict(plate_img, conf=0.4, verbose=False)
        characters = []
        
        for result in results:
            for box, cls in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.cls.cpu().numpy()):
                x1 = int(box[0])
                # On récupère le nom de la classe (ex: '3', 'd', '1')
                label = self.model.names[int(cls)] 
                characters.append((label, x1))
                
        # Tri de gauche à droite
        characters.sort(key=lambda x: x[1])
        
        # Reconstruction de la plaque
        left_nums = ""
        letter = ""
        right_nums = ""
        found_letter = False

        for label, _ in characters:
            if label.isdigit():
                if not found_letter:
                    left_nums += label
                else:
                    right_nums += label
            else:
                letter = self.arabic_chars(label)
                found_letter = True

        # Fallback de sécurité
        if not letter: letter = "?"
        if not left_nums: left_nums = "XXXXX"
        if not right_nums: right_nums = "XX"

        return f"{left_nums} | {letter} | {right_nums}"
