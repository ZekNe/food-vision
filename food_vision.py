import torch
import cv2

model = torch.hub.load("ultralytics/yolov5", "yolov5s")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Error: Failed to grab frame")
        break


    results = model(frame)

    labels, cords = results.names, results.xywh[0]

    for *xywh, conf, cls in cords:
        x1, y1, x2, y2 = map(int, xywh)
        label = labels[int(cls)]
        confidence = conf.item()
        color = (0, 255, 0)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        cv2.putText(frame, f"{label} {confidence:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)


    cv2.imshow("Webcam", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows