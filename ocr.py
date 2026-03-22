import cv2
import numpy as np
from ultralytics import YOLO

class PlateReader:
    def load_model(self, weight_path: str, cfg_path: str = None):
        self.model = YOLO(weight_path)
        with open("classes-ocr.names", "r") as f:
            self.classes = [line.strip() for line in f.readlines()]
        self.colors = np.random.uniform(0, 255, size=(len(self.classes), 3))

    def load_image(self, img_path):
        img = cv2.imread(img_path)
        height, width, channels = img.shape
        return img, height, width, channels

    def read_plate(self, img):
        results = self.model.predict(img, conf=0.3, verbose=False)
        return None, results
    
    def get_boxes(self, outputs, width, height, threshold=0.3):
        boxes = []
        confidences = []
        class_ids = []
        
        for result in outputs:
            for box in result.boxes:
                conf = float(box.conf[0])
                if conf > threshold:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    w = x2 - x1
                    h = y2 - y1
                    
                    boxes.append([x1, y1, w, h])
                    confidences.append(conf)
                    class_ids.append(int(box.cls[0]))
                    
        return boxes, confidences, class_ids
    
    def draw_labels(self, boxes, confidences, class_ids, img): 
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.1, 0.1)
        font = cv2.FONT_HERSHEY_PLAIN
        characters = []
        
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                # Lier l'ID détecté par YOLOv8 à ton fichier classes-ocr.names
                label = str(self.classes[class_ids[i]])
                
                color = self.colors[i % len(self.colors)]
                cv2.rectangle(img, (x,y), (x+w, y+h), color, 3)
                confidence = round(confidences[i], 3) * 100
                cv2.putText(img, str(confidence) + "%", (x, y - 6), font, 1, color, 2)
                characters.append((label, x))
                
        characters.sort(key=lambda x:x[1])
        plate = ""
        for l in characters:
            plate += l[0]
            
        # Ta logique originale de mise en forme des caractères arabes
        chg = 0
        for i in range(len(plate)):
            if plate[i] in ['b', 'h', 'd', 'a']:
                if plate[i-1] == 'w':
                    ar = i-1
                    chg = 2
                elif plate[i-1] == 'c':
                    ar = i - 1
                    chg = 3
                else:
                    ar = i
                    chg = 1

        if chg == 1:
            plate = plate[:ar] + ' | ' + str(self.arabic_chars(ord(plate[ar])), encoding="utf-8") + ' | ' + plate[ar+1:]
        if chg == 2:
            index = 0
            for i in range(3):
                index = index + plate[ar+i]
            plate = plate[:ar] + ' | ' + str(self.arabic_chars(index), encoding="utf-8") + ' | ' + plate[ar+3:]
        if chg == 3:
            index = 0
            for i in range(2):
                index = index + plate[ar+i]
            plate = plate[:ar] + ' | ' + str(self.arabic_chars(index), encoding="utf-8") + ' | ' + plate[ar+2:]

        return img, plate

    def arabic_chars(self, index):
        if (index == ord('a')): return "أ".encode("utf-8")
        if (index == ord('b')): return "ب".encode("utf-8")
        if (index == 2 * ord('w') + ord('a') or index == ord('w')): return "و".encode("utf-8")
        if (index == ord('d')): return "د".encode("utf-8")
        if (index == ord('h')): return "ه".encode("utf-8")
        if (index == ord('c') + ord('h')): return "ش".encode("utf-8")

    def tesseract_ocr(self, image, lang="eng", psm=7):
        import pytesseract
        alphanumeric = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        options = "-l {} --psm {} -c tessedit_char_whitelist={}".format(lang, psm, alphanumeric)
        return pytesseract.image_to_string(image, config=options)
