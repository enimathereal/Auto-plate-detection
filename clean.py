import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

# --- 1. CONFIGURATION ---
MODEL_PATH = '/content/drive/MyDrive/yolo_results/train/weights/best.pt'
IMAGE_INPUT = 'plaque-marocaine.jpg' # L'image avec plusieurs voitures
OUTPUT_DIR = 'DETECTION_FINALE'

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# --- 2. CHARGEMENT DU MODÈLE ---
model = YOLO(MODEL_PATH)
image_originale = cv2.imread(IMAGE_INPUT)

if image_originale is None:
    print("❌ Erreur : Image introuvable. Vérifiez le nom du fichier.")
else:
    # --- 3. DÉTECTION MULTI-PLAQUES (YOLO) ---
    results = model.predict(source=image_originale, conf=0.40)
    img_viz = image_originale.copy()

    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            
            # --- ÉTAPE A : EXTRACTION (CROP) ---
            plate_crop = image_originale[y1:y2, x1:x2]

            # --- ÉTAPE B : PIPELINE DE NETTOYAGE ROBUSTE (OPENCV) ---
            
            # 1. UPSCALING (Gérer la distance) : On multiplie la taille par 3
            # L'interpolation cubique recrée les détails manquants
            plate_zoom = cv2.resize(plate_crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
            
            # 2. GRIS & CONTRASTE (Gérer l'angle et l'ombre)
            gray = cv2.cvtColor(plate_zoom, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8)).apply(gray)
            
            # 3. FILTRE BILATÉRAL (Gérer le flou et le bruit)
            # Lisse le fond mais garde les chiffres TRÈS NETS
            smoothed = cv2.bilateralFilter(clahe, 9, 75, 75)
            
            # 4. BINARISATION D'OTSU (Seuil automatique)
            # Sépare parfaitement le noir du blanc selon l'éclairage
            _, thresh = cv2.threshold(smoothed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # 5. NETTOYAGE MORPHOLOGIQUE (Supprimer les derniers parasites)
            kernel = np.ones((3,3), np.uint8)
            # Opening : Supprime les petits points blancs isolés (bruit)
            # Closing : Reconnecte les morceaux de caractères brisés
            cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
            final_output = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=1)

            # --- ÉTAPE C : SAUVEGARDE ---
            cv2.imwrite(os.path.join(OUTPUT_DIR, f'plaque_{i+1}_brute.jpg'), plate_crop)
            cv2.imwrite(os.path.join(OUTPUT_DIR, f'plaque_{i+1}_nettoyee.jpg'), final_output)

            # --- ÉTAPE D : AFFICHAGE COMPARATIF ---
            plt.figure(figsize=(12, 4))
            plt.subplot(1, 3, 1)
            plt.imshow(cv2.cvtColor(plate_crop, cv2.COLOR_BGR2RGB))
            plt.title(f"Plaque {i+1} : Brute")
            plt.axis('off')

            plt.subplot(1, 3, 2)
            plt.imshow(final_output, cmap='gray')
            plt.title(f"Plaque {i+1} : Nettoyage Robuste")
            plt.axis('off')
            
            plt.show()

            # Dessiner sur l'image globale
            cv2.rectangle(img_viz, (x1, y1), (x2, y2), (0, 255, 0), 4)
            cv2.putText(img_viz, f"Plaque {i+1}", (x1, y1-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Sauvegarde finale de la scène complète
    cv2.imwrite(os.path.join(OUTPUT_DIR, 'resultat_scene_complete.jpg'), img_viz)
    print(f"✅ Pipeline terminé avec succès. Dossier : {OUTPUT_DIR}")
