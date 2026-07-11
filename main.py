print("hi sid how are you")
import cv2
import numpy as np
from ultralytics import YOLO
print("sab changa si")
model = YOLO("yolov8m.pt")

# Load 4 different videos
video1 = cv2.VideoCapture("crowd.mp4.mp4")
video2 = cv2.VideoCapture("crowd2.mp4.mp4")
video3 = cv2.VideoCapture("crowd3.mp4")
video4 = cv2.VideoCapture("crowd4.mp4")

videos = [video1, video2, video3, video4]

while True:
    frames = []
    counts = []

    for vid in videos:
        ret, frame = vid.read()

        if not ret:
            frame = None
            counts.append(0)
            frames.append(None)
            continue

        frame = cv2.resize(frame, (800, 400))

        results = model(frame)

        people_count = 0

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])

                if cls == 0 and conf > 0.3:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    head_y2 = int(y1 + (y2 - y1) * 0.3)

                    cv2.rectangle(frame, (x1, y1), (x2, head_y2), (255, 0, 0), 2)

                    people_count += 1

        counts.append(people_count)
        frames.append(frame)

    # Replace None frames with black screen
    for i in range(4):
        if frames[i] is None:
            frames[i] = 255 * np.ones((800,400, 3), dtype=np.uint8)

    # Combine frames into grid
    top = cv2.hconcat([frames[0], frames[1]])
    bottom = cv2.hconcat([frames[2], frames[3]])
    final = cv2.vconcat([top, bottom])

    # Display counts
    cv2.putText(final, f"C1: {counts[0]}", (50, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

    cv2.putText(final, f"C2: {counts[1]}", (450, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

    cv2.putText(final, f"C3: {counts[2]}", (50, 350),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

    cv2.putText(final, f"C4: {counts[3]}", (450, 350),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

    cv2.imshow("Multi-Coach (4 Cameras)", final)

    if cv2.waitKey(1) & 0xFF == ord('s'):
        break

for v in videos:
    v.release()

cv2.destroyAllWindows()