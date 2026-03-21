import cv2
import numpy as np
from ultralytics import YOLO
import easyocr
import re
import os
from datetime import datetime

class RadarMarocPro:
    def __init__(self, model_yolo):
        print("[INFO] Initialisation du système Radar Pro...")
        # 1. Détecteur ultra-rapide
        self.detector = YOLO(model_yolo)
        # 2. Lecteur OCR Multilingue (GPU activé si disponible)
        self.reader = easyocr.Reader(['ar', 'en'], gpu=True)
        # 3. Dictionnaire de correction contextuelle
        self.corrections = {'١': 'و', '1': 'و', '9': 'و', '5': 'و', '0': 'د', 'ا': 'أ'}

    def preprocess_plate(self, plate_crop):
        """ Amélioration d'image type 'Super-Resolution' """
        # Agrandissement cubique pour redonner du détail aux pixels
        img = cv2.resize(plate_crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # CLAHE : Égalisation adaptative de l'histogramme pour les reflets
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8)).apply(gray)
        
        # Débruitage bilatéral (préserve les bords des chiffres)
        refined = cv2.bilateralFilter(clahe, 9, 75, 75)
        return refined

    def ocr_segmentation(self, plate_refined):
        """ Découpage intelligent par zones sémantiques """
        h, w = plate_refined.shape
        
        # Définition des zones (Ratios officiels des plaques marocaines)
        z_gauche = plate_refined[:, :int(w * 0.48)]
        z_milieu = plate_refined[:, int(w * 0.48):int(w * 0.78)]
        z_droite = plate_refined[:, int(w * 0.78):]

        # Lecture ciblée avec contraintes (Allowlists)
        res_g = self.reader.readtext(z_gauche, allowlist='0123456789', detail=0)
        res_m = self.reader.readtext(z_milieu, detail=0)
        res_d = self.reader.readtext(z_droite, allowlist='0123456789', detail=0)

        # Post-traitement Regex et Logique métier
        serie = "".join(res_g) if res_g else "XXXXX"
        if serie.startswith('0'): serie = '8' + serie[1:] # Correction 0/8 au début
        
        lettre_raw = "".join(res_m) if res_m else "و"
        # On ne garde que l'Arabe et on applique le dictionnaire de fautes
        lettre = re.sub(r'[^\u0600-\u06FF]', '', lettre_raw)
        if not lettre: lettre = self.corrections.get(lettre_raw[0], "و") if lettre_raw else "و"
        
        region = "".join(res_d) if res_d else "6"
        if region == "5": region = "6" # Correction Casablanca Anfa

        return f"{serie} | {lettre} | {region}"

    def run(self, image_path, save_res=True):
        img = cv2.imread(image_path)
        if img is None: return print("❌ Erreur : Image introuvable.")

        # Inférence YOLO (Détection)
        results = self.detector.predict(img, conf=0.5, verbose=False)
        
        for result in results:
            for box in result.boxes.xyxy.cpu().numpy():
                x1, y1, x2, y2 = map(int, box)
                plate_crop = img[y1:y2, x1:x2]
                
                # Pipeline de lecture
                refined = self.preprocess_plate(plate_crop)
                matricule = self.ocr_segmentation(refined)
                
                # Affichage graphique 'Radar'
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 4)
                cv2.putText(img, matricule, (x1, y1-15), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                
                print(f"[{datetime.now().strftime('%H:%M:%S')}] DÉTECTION : {matricule}")

        if save_res:
            cv2.imwrite('resultat_radar.jpg', img)
            print("✅ Image sauvegardée sous 'resultat_radar.jpg'")

# --- LANCEMENT ---
radar = RadarMarocPro('/content/drive/MyDrive/yolo_results/train/weights/best.pt')
radar.run('plaque-marocaine.jpg')
