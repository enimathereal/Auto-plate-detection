import cv2
from datetime import datetime
from detection import PlateDetector
from ocr import PlateReader

def main():
    print("[INFO] Lancement du Radar Pro (Architecture Modulaire)...")
    
    # --- 1. CONFIGURATION DES CHEMINS ---
    chemin_yolo = '/content/drive/MyDrive/yolo_results/train/weights/best.pt'
    chemin_image = 'plaque-marocaine.jpg' 
    save_name = 'resultat_radar.jpg'
    
    # --- 2. INITIALISATION DES MODULES ---
    detector = PlateDetector(chemin_yolo)
    reader = PlateReader()
    
    # Dictionnaire pour éviter le bug des "??" sur l'image avec OpenCV
    affichage_latin = {'أ': 'A', 'ب': 'B', 'ت': 'T', 'ج': 'J', 'د': 'D', 'ه': 'H', 'و': 'W'}

    # --- 3. LECTURE DE L'IMAGE ---
    img = cv2.imread(chemin_image)
    if img is None:
        print("❌ Erreur : Image introuvable.")
        return

    # --- 4. DÉTECTION (YOLO) ---
    crops, coords = detector.detect(img)
    
    if not crops:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] DÉTECTION : Aucune plaque détectée")
        return

    # --- 5. LECTURE (OCR) ET AFFICHAGE ---
    for crop, (x1, y1, x2, y2) in zip(crops, coords):
        
        # Appel de ton fichier ocr.py
        matricule_officiel, serie, lettre, region = reader.read_plate(crop)
        
        # Traduction visuelle juste pour le dessin sur l'image
        lettre_visuelle = affichage_latin.get(lettre, lettre)
        matricule_dessin = f"{serie} | {lettre_visuelle} | {region}"
        
        # Dessin graphique
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 4)
        cv2.putText(img, matricule_dessin, (x1, y1-15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        
        # TON PRINT ORIGINAL (Intact)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] DÉTECTION : {matricule_officiel}")

    # --- 6. SAUVEGARDE ---
    cv2.imwrite(save_name, img)
    print(f"✅ Image sauvegardée sous '{save_name}'")

if __name__ == "__main__":
    main()
