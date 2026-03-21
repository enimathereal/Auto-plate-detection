import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
import numpy as np

# 1. Chargement du modèle entraîné
# Assure-toi que le fichier 'best.pt' est dans le même dossier
model = YOLO('/content/drive/MyDrive/yolo_results/train/weights/best.pt') 

# 2. Chargement de l'image de la voiture
image_path = 'plaque-marocaine.jpg' # Remplace par ton nom de fichier
image = cv2.imread(image_path)
if image is None:
    print("Erreur : Impossible de charger l'image.")
else:
    # 3. Détection de la plaque avec YOLO
    results = model.predict(source=image, conf=0.45) # Seuil de confiance à 45%

    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            
            # --- ÉTAPE 1 : RECADRAGE (CROP) ---
            plate_crop = image[y1:y2, x1:x2]
            
            # --- ÉTAPE 2 : PRÉTRAITEMENT COMPLET (OPENCV) ---
            # a. Niveaux de gris
            gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
            # b. Filtre bilatéral (réduction de bruit en gardant les bords nets)
            blurred = cv2.bilateralFilter(gray, 11, 17, 17)
            # c. Égalisation de contraste (CLAHE)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(blurred)
            # d. Binarisation Adaptative (Noir et Blanc pur)
            thresh = cv2.adaptiveThreshold(clahe, 255, 
                                          cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                          cv2.THRESH_BINARY, 11, 2)
            # e. Nettoyage final
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

            # --- ÉTAPE 3 : AFFICHAGE AVEC MATPLOTLIB ---
            # On prépare une figure avec 3 zones pour ta présentation
            plt.figure(figsize=(15, 5))

            # Image originale avec le carré de détection
            plt.subplot(1, 3, 1)
            img_detected = image.copy()
            cv2.rectangle(img_detected, (x1, y1), (x2, y2), (0, 255, 0), 10)
            plt.imshow(cv2.cvtColor(img_detected, cv2.COLOR_BGR2RGB))
            plt.title("1. Détection YOLOv8")
            plt.axis('off')

            # Zoom sur la plaque brute
            plt.subplot(1, 3, 2)
            plt.imshow(cv2.cvtColor(plate_crop, cv2.COLOR_BGR2RGB))
            plt.title("2. Recadrage (Crop)")
            plt.axis('off')

            # Plaque nettoyée pour l'OCR
            plt.subplot(1, 3, 3)
            plt.imshow(cleaned, cmap='gray')
            plt.title("3. Prétraitement Final (OCR Ready)")
            plt.axis('off')

            plt.tight_layout()
            plt.show()

            # Message de succès pour la console
            print(f"Plaque n°{i+1} détectée et traitée avec succès.")
