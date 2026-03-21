import cv2
from ultralytics import YOLO

# 1. Charger ton modèle entraîné (remplace 'best.pt' par le chemin de ton fichier)
model = YOLO('/content/drive/MyDrive/yolo_results/train/weights/best.pt') 

# 2. Charger l'image de la voiture marocaine
image_path = 'test_car.jpg'
image = cv2.imread(image_path)
h, w, _ = image.shape

# 3. Exécuter la détection
results = model.predict(source=image, conf=0.5) # conf=0.5 pour éviter les faux positifs

# 4. Parcourir les résultats détectés
for result in results:
    boxes = result.boxes.xyxy.cpu().numpy() # Récupère les coordonnées [x1, y1, x2, y2]
    
    for box in boxes:
        # Extraire les coordonnées
        x1, y1, x2, y2 = map(int, box)
        
        # --- ÉTAPE DE RECADRAGE (CROP) ---
        # On découpe la plaque de l'image originale
        plate_crop = image[y1:y2, x1:x2]
        
        # --- ÉTAPE DE PRÉTRAITEMENT (OPENCV) ---
        gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
        blurred = cv2.bilateralFilter(gray, 11, 17, 17)
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        
        # --- AFFICHAGE ---
        # Dessiner le rectangle sur l'image d'origine pour la présentation
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, "Plaque Maroc", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Montrer la plaque recadrée et nettoyée
        cv2.imshow('Plaque Recadree et Nettoyee', thresh)

# Afficher l'image finale avec la détection
cv2.imshow('Detection YOLOv8', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
