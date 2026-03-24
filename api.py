from flask import Flask, request, render_template_string
import os, cv2, traceback, arabic_reshaper, base64, re
from bidi.algorithm import get_display
from detection import PlateDetector
from ocr import PlateReader
import numpy as np

app = Flask(__name__)

# --- CONFIGURATION ---
PATH_DET = "/content/drive/MyDrive/dataset/runs/detect/train/weights/best.pt"
PATH_OCR = "/content/drive/MyDrive/dataset/runs/detect/train/weights/best_ocr_v2.pt"

ARABIC_MAP = {
    'a': ' أ ', 'b': ' ب ', 'd': ' د ', 
    'h': ' هـ ', 'w': ' و ', 'waw': ' و ',
    'ch': ' ش ', 'j': ' ج '
}

# --- HTML/CSS DE LA PAGE D'ACCUEIL ---
HTML_HOME = '''
<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <title>PFE - LPR Maroc</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body { background-color: #f0f2f5; font-family: sans-serif; }
        .main-card { background: white; border-radius: 15px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); padding: 40px; margin-top: 50px; }
        .radar-loader { display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(255, 255, 255, 0.95); z-index: 1000; justify-content: center; align-items: center; flex-direction: column; }
        .radar { width: 120px; height: 120px; border-radius: 50%; background: conic-gradient(#1a73e8 0% 15%, transparent 15% 100%); animation: scan 1.2s linear infinite; box-shadow: 0 0 15px rgba(26, 115, 232, 0.4); position: relative; border: 2px solid #ddd; }
        .radar:before { content: ''; position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); width: 8px; height: 8px; background: #1a73e8; border-radius: 50%; }
        @keyframes scan { to { transform: rotate(360deg); } }
    </style>
</head>
<body>
    <div id="radarLoader" class="radar-loader">
        <div class="radar"></div>
        <h3 class="text-primary mt-4">Analyse en cours...</h3>
    </div>

    <div class="container text-center">
        <div class="row justify-content-center">
            <div class="col-md-8 main-card">
                <h1 class="text-primary mb-3">🔍 Système LPR Maroc</h1>
                <form action="/upload" method="post" enctype="multipart/form-data" id="uploadForm">
                    <div class="mb-4"><input class="form-control form-control-lg" type="file" name="image" required></div>
                    <button type="submit" class="btn btn-primary btn-lg px-5">Lancer l'Analyse</button>
                </form>
            </div>
        </div>
    </div>
    <script>
        document.getElementById('uploadForm').onsubmit = function() {
            document.getElementById('radarLoader').style.display = 'flex';
        };
    </script>
</body>
</html>
'''

def format_moroccan_plate(raw_text):
    """
    Sépare le texte OCR en 3 blocs (Gauche, Milieu, Droite)
    et génère un affichage HTML qui imite une vraie plaque marocaine.
    """
    if not raw_text: return "<p>Non reconnu</p>"
    
    # On cherche le format : Chiffres + Lettres(s) + Chiffres (ex: 29003a17)
    match = re.match(r'^(\d+)([a-zA-Z]+)(\d+)$', raw_text)
    
    if match:
        part1 = match.group(1)       # ex: 29003
        letter_latin = match.group(2) # ex: a
        part2 = match.group(3)       # ex: 17
        
        # Traduction de la lettre en Arabe
        letter_arabe = ARABIC_MAP.get(letter_latin.lower(), letter_latin)
        
        # Création du bloc HTML imitant la vraie plaque (avec les barres de séparation)
        return f'''
        <div style="display: flex; justify-content: center; align-items: center; gap: 30px; font-size: 1.2em;">
            <div>{part1}</div>
            <div style="border-left: 5px solid #1a73e8; border-right: 5px solid #1a73e8; padding: 0 30px;">
                {letter_arabe}
            </div>
            <div>{part2}</div>
        </div>
        '''
    else:
        # Fallback de sécurité si l'IA n'a pas lu la plaque parfaitement
        translated = "".join([ARABIC_MAP.get(char.lower(), char) for char in raw_text])
        return f'<div>{get_display(arabic_reshaper.reshape(translated))}</div>'

@app.route('/')
def home(): return render_template_string(HTML_HOME)

@app.route('/upload', methods=['POST'])
def upload():
    try:
        file = request.files.get('image')
        if not file: return "Image manquante"
        path = os.path.join('./recieved', file.filename)
        os.makedirs('./recieved', exist_ok=True); file.save(path)

        det = PlateDetector(); det.load_model(PATH_DET)
        reader = PlateReader(); reader.load_model(PATH_OCR)

        img, h, w, _ = det.load_image(path)
        _, results = det.detect_plates(img)
        boxes, confs, ids = det.get_boxes(results, w, h)
        img_res, LpImg = det.draw_labels(boxes, confs, ids, img)

        if len(LpImg) > 0:
            # OCR + Dessin des petits rectangles sur la plaque
            img_with_segmentation, raw_text = reader.read_plate(LpImg[0])
            
            # Conversion en Base64 de l'IMAGE ORIGINALE AVEC LE GRAND RECTANGLE VERT
            _, buffer = cv2.imencode('.jpg', img_res)
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            
            # Utilisation de la fonction de formatage HTML (Flexbox pour la plaque marocaine)
            plate_html = format_moroccan_plate(raw_text)
            
            # --- NOUVEAU HTML/CSS POUR LA MISE EN PAGE EN DEUX COLONNES ---
            return f'''
            <!DOCTYPE html>
            <html lang="fr">
            <head>
                <meta charset="UTF-8">
                <title>Résultats PFE - LPR Maroc</title>
                <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
                <style>
                    body {{ background-color: #f0f2f5; font-family: sans-serif; padding-top: 30px; }}
                    .res-card {{ background: white; border-radius: 15px; padding: 40px; box-shadow: 0 4px 15px rgba(0,0,0,0.08); }}
                    
                    /* Style spécifique pour le bloc de la plaque d'immatriculation */
                    .plate-display {{ border: 5px solid #1a73e8; background: #fff; border-radius: 12px; padding: 20px; display: inline-block; width: 100%; }}
                    .plate-text {{ font-size: 3.5em; color: #1a73e8; font-weight: bold; margin: 0; font-family: 'Arial', sans-serif; }}
                    
                    /* Style pour l'image de la voiture */
                    .car-img {{ border-radius: 12px; border: 3px solid #ddd; max-width: 100%; height: auto; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
                    
                    /* Centre verticalement le contenu de la colonne de droite */
                    .align-vh-center {{ display: flex; flex-direction: column; justify-content: center; height: 100%; }}
                </style>
            </head>
            <body>
                <div class="container-fluid px-5">
                    <div class="row justify-content-center">
                        <div class="col-xl-10 res-card">
                            <h2 class="text-muted text-center mb-5">Analyse Automatique de l'Immatriculation</h2>
                            
                            <div class="row g-5">
                                
                                <div class="col-md-6 text-center">
                                    <h4 class="text-primary mb-3">Image Source (Zone Détectée)</h4>
                                    <img src="data:image/jpeg;base64,{img_base64}" alt="Voiture analysée" class="car-img">
                                </div>
                                
                                <div class="col-md-6 text-center">
                                    <div class="align-vh-center">
                                        <h4 class="text-primary mb-3">Résultat de l'Extraction (OCR)</h4>
                                        <div class="plate-display">
                                            <div class="plate-text">{plate_html}</div>
                                        </div>
                                        <div class="mt-5">
                                            <a href="/" class="btn btn-outline-primary btn-lg px-4">← Analyser une autre voiture</a>
                                        </div>
                                    </div>
                                </div>
                                
                            </div> </div>
                    </div>
                </div>
            </body>
            </html>
            '''
        return "Aucune plaque détectée sur cette image."
    except Exception:
        return f"<pre>{traceback.format_exc()}</pre>"

if __name__ == '__main__':
    os.system("fuser -k 5000/tcp > /dev/null 2>&1")
    app.run(host='0.0.0.0', port=5000)
