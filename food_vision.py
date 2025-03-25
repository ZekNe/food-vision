import cv2 as cv
import torch
import os

def process_frame(frame):
    # Convert BGR to RGB
    frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    # Convert to PyTorch tensor and normalize to [0,1], (H, W, C) -> (C, H, W)
    tensor_frame = torch.tensor(frame, dtype=torch.float32).permute(2, 0, 1) / 255.0

    return tensor_frame

os.makedirs("saved_frames", exist_ok=True)

cap = cv.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

frame_count = 0

while True:
    ret, frame =cap.read()

    if not ret:
        print("Error: Failed to grab frame")
        break

    tensor_frame = process_frame(frame)

    # Save regular frame
    cv.imwrite(f"saved_frames/regular_frame_{frame_count}.jpg", frame)

    # Save tensor frame
    torch.save(tensor_frame, f"saved_frames/tensor_frame_{frame_count}.pt")

    frame_count += 1
    
    # Convert tensors back to RGB 
    frame_rgb = tensor_frame.permute(1, 2, 0).numpy()
    frame_bgr = cv.cvtColor(frame_rgb, cv.COLOR_RGB2BGR)

    # Show live feed
    cv.imshow("Live Feed", frame_rgb)

    if cv.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv.destroyAllWindows()