import os
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO

# 1. Création du dossier de sortie
output_dir = 'RESULTATS_PLAQUES'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Dossier '{output_dir}' créé avec succès.")

# 2. Chargement du modèle et de l'image
model = YOLO('/content/drive/MyDrive/yolo_results/train/weights/best.pt') 
image_path = 'plaque-marocaine.jpg'
image = cv2.imread(image_path)

# 3. Détection
results = model.predict(source=image, conf=0.45)

# Copie de l'image pour dessiner dessus sans modifier l'originale
image_annotated = image.copy()

for result in results:
    boxes = result.boxes.xyxy.cpu().numpy()
    
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        
        # --- RECADRAGE ET NETTOYAGE ---
        plate_crop = image[y1:y2, x1:x2]
        gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

        # --- ENREGISTREMENT DES FICHIERS ---
        # Sauvegarde de la plaque nettoyée (Noir et Blanc)
        filename_clean = os.path.join(output_dir, f'plaque_{i+1}_clean.jpg')
        cv2.imwrite(filename_clean, thresh)
        
        # Sauvegarde de la plaque brute (Couleur)
        filename_crop = os.path.join(output_dir, f'plaque_{i+1}_crop.jpg')
        cv2.imwrite(filename_crop, plate_crop)

        # Dessiner le rectangle sur l'image globale
        cv2.rectangle(image_annotated, (x1, y1), (x2, y2), (0, 255, 0), 5)
        cv2.putText(image_annotated, f"Plaque {i+1}", (x1, y1-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

# 4. Enregistrer l'image globale avec toutes les détections
cv2.imwrite(os.path.join(output_dir, 'resultat_global.jpg'), image_annotated)

print(f"Terminé ! Regarde dans le dossier '{output_dir}' à gauche dans Colab.")e et traitée avec succès.")
