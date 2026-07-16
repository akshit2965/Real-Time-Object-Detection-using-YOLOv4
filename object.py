from pathlib import Path
import sys

import cv2
import numpy as np


BASE_DIR = Path(__file__).resolve().parent
WEIGHTS_PATH = BASE_DIR / "yolov4.weights"
CONFIG_PATH = BASE_DIR / "yolov4.cfg"
CLASSES_PATH = BASE_DIR / "coco.names"
CONFIDENCE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.4
FRAME_SKIP = 2


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(
            f"Required file not found: {path.name}. Place it in {BASE_DIR}."
        )


def main() -> None:
    for path in (WEIGHTS_PATH, CONFIG_PATH, CLASSES_PATH):
        require_file(path)

    net = cv2.dnn.readNet(str(WEIGHTS_PATH), str(CONFIG_PATH))
    with CLASSES_PATH.open(encoding="utf-8") as names_file:
        classes = [line.strip() for line in names_file if line.strip()]

    layer_names = net.getLayerNames()
    output_layers = [
        layer_names[index - 1]
        for index in net.getUnconnectedOutLayers().flatten()
    ]
    colors = np.random.default_rng(42).integers(0, 256, size=(len(classes), 3))

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise ConnectionError(
            f"Unable to open IP camera at {cap}. Check the camera address and network."
        )

    frame_id = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                raise RuntimeError("The camera stream ended or a frame could not be read.")

            frame_id += 1
            if frame_id % FRAME_SKIP:
                continue

            height, width = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(
                frame, 1 / 255.0, (416, 416), swapRB=True, crop=False
            )
            net.setInput(blob)
            outputs = net.forward(output_layers)

            boxes, confidences, class_ids = [], [], []
            for output in outputs:
                for detection in output:
                    scores = detection[5:]
                    class_id = int(np.argmax(scores))
                    confidence = float(detection[4] * scores[class_id])
                    if confidence < CONFIDENCE_THRESHOLD:
                        continue

                    center_x, center_y = (detection[:2] * [width, height]).astype(int)
                    box_width, box_height = (detection[2:4] * [width, height]).astype(int)
                    x = max(0, center_x - box_width // 2)
                    y = max(0, center_y - box_height // 2)
                    boxes.append([x, y, box_width, box_height])
                    confidences.append(confidence)
                    class_ids.append(class_id)

            indexes = cv2.dnn.NMSBoxes(
                boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD
            )
            if len(indexes) > 0:
                for index in indexes.flatten():
                    x, y, box_width, box_height = boxes[index]
                    class_id = class_ids[index]
                    color = tuple(int(value) for value in colors[class_id])
                    label = f"{classes[class_id]}: {confidences[index]:.2f}"
                    cv2.rectangle(frame, (x, y), (x + box_width, y + box_height), color, 2)
                    cv2.putText(
                        frame, label, (x, max(20, y - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color, 2,
                    )

            cv2.imshow("YOLOv4 Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except (ConnectionError, FileNotFoundError, RuntimeError, cv2.error) as error:
        sys.exit(f"Error: {error}")
