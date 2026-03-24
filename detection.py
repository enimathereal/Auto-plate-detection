import cv2
import numpy as np
from ultralytics import YOLO

class PlateDetector:
    def __init__(self):
        self.model = None

    def load_model(self, weight_path: str):
        """Charge le modèle YOLOv8 entraîné pour la détection de plaques."""
        self.model = YOLO(weight_path)

    def load_image(self, img_path):
        """Charge l'image et retourne ses dimensions."""
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Impossible de lire l'image : {img_path}")
        h, w, c = img.shape
        return img, h, w, c

    def detect_plates(self, img):
        """Exécute la prédiction YOLOv8 sur l'image."""
        # imgsz=640 est standard pour YOLOv8 et offre un bon compromis vitesse/précision
        results = self.model.predict(img, imgsz=640, conf=0.25, verbose=False)
        return None, results
        
    def get_boxes(self, outputs, width, height, threshold=0.25):
        """Extrait les coordonnées des plaques détectées."""
        boxes, confidences, class_ids = [], [], []
        
        if not outputs or len(outputs) == 0:
            return boxes, confidences, class_ids
        
        result = outputs[0]
        if result.boxes is not None:
            for box in result.boxes:
                conf = float(box.conf[0].item())
                if conf > threshold:
                    # Coordonnées au format [x1, y1, x2, y2]
                    xyxy = box.xyxy[0].tolist()
                    x1, y1, x2, y2 = map(int, xyxy)
                    
                    # On stocke sous format [x, y, w, h] pour draw_labels
                    boxes.append([x1, y1, x2 - x1, y2 - y1])
                    confidences.append(conf)
                    class_ids.append(int(box.cls[0].item()))
                    
        return boxes, confidences, class_ids

    def draw_labels(self, boxes, confidences, class_ids, img):
        """Dessine les rectangles et découpe la plaque pour l'OCR."""
        plats = []
        img_h, img_w = img.shape[:2]
        
        for i in range(len(boxes)):
            x, y, w, h = boxes[i]
            
            # Sécurité : S'assurer que les coordonnées ne sortent pas de l'image
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(img_w, x + w), min(img_h, y + h)
            
            # Découpage de la plaque (Crop)
            crop_img = img[y1:y2, x1:x2]
            
            if crop_img.size > 0:
                # Redimensionnement optionnel pour uniformiser l'entrée de l'OCR
                crop_resized = cv2.resize(crop_img, (470, 110))
                plats.append(crop_resized)
                
                # Dessin du rectangle sur l'image originale (pour le débug/affichage)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, f"Plaque {confidences[i]:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                            
        return img, plats
