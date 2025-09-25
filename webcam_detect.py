import cv2
import dlib
import os
import datetime

detector = dlib.get_frontal_face_detector()
cap = cv2.VideoCapture(0)  # 0 = default webcam
os.makedirs("results/webcam_snapshots", exist_ok=True)

print("▶ Press 's' to save snapshot, 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)

    for r in rects:
        cv2.rectangle(frame, (r.left(), r.top()), (r.right(), r.bottom()), (0,255,0), 2)

    cv2.imshow("Webcam Face Detection", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):
        fname = datetime.datetime.now().strftime("snap_%Y%m%d_%H%M%S.jpg")
        cv2.imwrite(os.path.join("results/webcam_snapshots", fname), frame)
        print("📸 Saved:", fname)

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
