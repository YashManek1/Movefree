from ultralytics import YOLO
import cv2
import argparse

model = YOLO("runs/detect/movefree_finetune_fast/weights/best.pt")

print("🔍 MODEL'S CLASS NAMES:")
print(model.names)
print()

parser = argparse.ArgumentParser()
parser.add_argument("--source", default="0")
args = parser.parse_args()

try:
    source = int(args.source)
except:
    source = args.source

if isinstance(source, str) and source.startswith("https://"):
    source = source.replace("https://", "http://")

cap = cv2.VideoCapture(source)

print(f"✅ Camera: {source}")
print("📷 Press SPACE to detect, Q to quit\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    display_frame = cv2.resize(frame, (640, 480))
    cv2.putText(
        display_frame,
        "SPACE=detect Q=quit",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )
    cv2.imshow("Test", display_frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord(" "):
        print("\n" + "=" * 50)
        print("🎯 DETECTING (Enhanced Mode)...")

        # APPLY IMAGE ENHANCEMENTS BEFORE DETECTION
        # 1. Denoise
        enhanced = cv2.fastNlMeansDenoisingColored(frame, None, 10, 10, 7, 21)

        # 2. Increase contrast
        lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

        # 3. Detect with VERY LOW threshold + augment=True (TTA)
        results = model(enhanced, conf=0.10, iou=0.3, augment=True, verbose=False)

        if results[0].boxes and len(results[0].boxes) > 0:
            print(f"✅ Found {len(results[0].boxes)} objects:\n")

            for i, box in enumerate(results[0].boxes, 1):
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                conf = float(box.conf[0])

                # Only show if confidence > 0.15
                if conf > 0.15:
                    print(f"  {i}. {class_name:15s} | Conf: {conf:.2f}")

                    xyxy = box.xyxy[0].cpu().numpy()
                    cv2.rectangle(
                        display_frame,
                        (int(xyxy[0]), int(xyxy[1])),
                        (int(xyxy[2]), int(xyxy[3])),
                        (0, 255, 0),
                        2,
                    )
                    cv2.putText(
                        display_frame,
                        f"{class_name} {conf:.2f}",
                        (int(xyxy[0]), int(xyxy[1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 255),
                        2,
                    )

            cv2.imshow("Result", display_frame)
            cv2.waitKey(0)
            cv2.destroyWindow("Result")
        else:
            print("❌ NO DETECTIONS")

        print("=" * 50 + "\n")

    elif key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
