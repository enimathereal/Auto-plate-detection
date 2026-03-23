from flask import Flask, request, render_template_string
import os, cv2, traceback, arabic_reshaper
from bidi.algorithm import get_display
from detection import PlateDetector
from ocr import PlateReader

app = Flask(__name__)

# --- CHEMINS À VÉRIFIER ---
PATH_DET = "/content/drive/MyDrive/dataset/runs/detect/train/weights/best.pt"
PATH_OCR = "/content/drive/MyDrive/dataset/runs/detect/train/weights/best_ocr_v2.pt"

HTML = '''
<!DOCTYPE html>
<html>
<head><title>LPR Morocco</title></head>
<body style="text-align:center; font-family:sans-serif; background:#f4f4f9;">
    <div style="margin-top:50px; display:inline-block; background:white; padding:30px; border-radius:10px; shadow:0 0 10px rgba(0,0,0,0.1);">
        <h1>🚗 Plate Recognition System</h1>
        <form action="/upload" method="post" enctype="multipart/form-data">
            <input type="file" name="image" required><br><br>
            <button type="submit" style="padding:10px 20px; background:#007bff; color:white; border:none; border-radius:5px;">Lancer l'Analyse</button>
        </form>
    </div>
</body>
</html>
'''

@app.route('/')
def home(): return render_template_string(HTML)

@app.route('/upload', methods=['POST'])
def upload():
    try:
        file = request.files.get('image')
        if not file: return "Image manquante"
        
        os.makedirs('./recieved', exist_ok=True)
        path = os.path.join('./recieved', file.filename)
        file.save(path)

        # Initialisation
        det = PlateDetector()
        det.load_model(PATH_DET)
        reader = PlateReader()
        reader.load_model(PATH_OCR)

        # Process
        img, h, w, _ = det.load_image(path)
        _, results = det.detect_plates(img)
        boxes, confs, ids = det.get_boxes(results, w, h)
        img_res, LpImg = det.draw_labels(boxes, confs, ids, img)
        cv2.imwrite('/content/debug_detection.jpg', img_res)

        if len(LpImg) > 0:
            img_ocr, text = reader.read_plate(LpImg[0])
            cv2.imwrite('/content/debug_ocr.jpg', img_ocr)
            
            # Gestion de l'affichage Arabe
            display_text = get_display(arabic_reshaper.reshape(text)) if text else "Non reconnu"
            
            return f'''
                <div style="text-align:center; margin-top:100px;">
                    <h2>Résultat :</h2>
                    <h1 style="border:3px solid green; display:inline-block; padding:20px;">{display_text}</h1>
                    <br><br><a href="/">Retour</a>
                </div>
            '''
        return "Aucune plaque détectée."
    except:
        return f"<pre>{traceback.format_exc()}</pre>"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
