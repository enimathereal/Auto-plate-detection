import cv2
import numpy as np

def pre_process_plate(plate_crop):
    # 1. Passage en niveaux de gris
    # On simplifie l'image pour ne garder que l'intensité lumineuse
    gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)

    # 2. Réduction du bruit (Filtre Bilatéral)
    # Lisse les surfaces mais garde les bords des chiffres très nets
    blurred = cv2.bilateralFilter(gray, 11, 17, 17)

    # 3. Égalisation du contraste (CLAHE)
    # Améliore la visibilité des chiffres même si la plaque est à l'ombre
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    contrast = clahe.apply(blurred)

    # 4. Binarisation Adaptative
    # Transforme l'image en Noir et Blanc pur (0 ou 255)
    # Très efficace contre les reflets du soleil marocain sur le métal
    thresh = cv2.adaptiveThreshold(contrast, 255, 
                                  cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY, 11, 2)

    # 5. Nettoyage Morphologique (Closing)
    # Bouche les petits trous à l'intérieur des chiffres pour qu'ils soient pleins
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    return cleaned

# --- Execution ---
plaque_recadree = cv2.imread('votre_crop_yolo.jpg')
image_finale = pre_process_plate(plaque_recadree)
cv2.imshow('Plaque Prete pour OCR', image_finale)
cv2.waitKey(0)