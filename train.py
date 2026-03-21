from ultralytics import YOLO

# 1. Charger le modèle
# On a utilise 'yolov8n.pt' car c'est le plus rapide
model = YOLO('yolov8n.pt') 

# 2. Lancer l'entraînement
# data='data.yaml' pointe vers notre fichier de configuration Roboflow
# epochs=50 signifie que le modèle va s'entraîner 50 fois sur tout le dataset
# imgsz=640 est la taille des images pour le modele
results = model.train(data='data.yaml', epochs=50, imgsz=640)