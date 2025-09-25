import os
import cv2
import dlib
import pandas as pd
import datetime
from pathlib import Path

def detect_faces(img, detector, upsample=1):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, upsample)
    faces = []
    for r in rects:
        faces.append({
            "left": r.left(),
            "top": r.top(),
            "right": r.right(),
            "bottom": r.bottom(),
            "width": r.width(),
            "height": r.height()
        })
    return faces

def draw_boxes(img, faces, color=(0,255,0), thickness=2):
    for f in faces:
        cv2.rectangle(img, (f["left"], f["top"]), (f["right"], f["bottom"]), color, thickness)
    return img

def process_folder(input_folder="known_images", output_folder="results"):
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    detector = dlib.get_frontal_face_detector()
    results = []

    for fname in os.listdir(input_folder):
        if not fname.lower().endswith(('.jpg','.jpeg','.png')):
            continue
        path = os.path.join(input_folder, fname)
        img = cv2.imread(path)
        if img is None:
            print("⚠️ Could not read:", path)
            continue

        faces = detect_faces(img, detector)
        annotated = draw_boxes(img.copy(), faces)

        out_path = os.path.join(output_folder, fname)
        cv2.imwrite(out_path, annotated)

        results.append({
            "filename": fname,
            "n_faces": len(faces),
            "processed_at": datetime.datetime.now().isoformat()
        })

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_folder, "detection_log.csv"), index=False)
    print("✅ Detection completed. Results saved to:", output_folder)

if __name__ == "__main__":
    process_folder()
