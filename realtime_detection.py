import argparse
import base64
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
import requests


Box = Tuple[int, int, int, int]


@dataclass
class Detection:
    label: str
    confidence: float
    box: Box
    source: str


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def parse_video_source(source: str) -> Any:
    return int(source) if source.isdigit() else source


def resize_frame(frame: np.ndarray, width: Optional[int]) -> np.ndarray:
    if not width or frame.shape[1] <= width:
        return frame
    ratio = width / frame.shape[1]
    height = int(frame.shape[0] * ratio)
    return cv2.resize(frame, (width, height))


class WebhookAlerter:
    def __init__(
        self,
        webhook_url: Optional[str],
        cooldown_seconds: int,
        include_snapshot: bool = False,
    ) -> None:
        self.webhook_url = webhook_url
        self.cooldown_seconds = cooldown_seconds
        self.include_snapshot = include_snapshot
        self.last_sent_at: Dict[str, float] = {}

    def send(self, detection: Detection, frame: np.ndarray) -> None:
        if not self.webhook_url:
            return

        now = time.time()
        if now - self.last_sent_at.get(detection.label, 0) < self.cooldown_seconds:
            return

        payload: Dict[str, Any] = {
            "event": detection.label,
            "confidence": round(float(detection.confidence), 4),
            "source": detection.source,
            "box": {
                "x1": detection.box[0],
                "y1": detection.box[1],
                "x2": detection.box[2],
                "y2": detection.box[3],
            },
            "timestamp": int(now),
        }

        if self.include_snapshot:
            ok, buffer = cv2.imencode(".jpg", frame)
            if ok:
                payload["snapshot_jpeg_base64"] = base64.b64encode(buffer).decode("ascii")

        try:
            response = requests.post(self.webhook_url, json=payload, timeout=5)
            response.raise_for_status()
            self.last_sent_at[detection.label] = now
            print(f"Webhook sent: {detection.label} ({detection.confidence:.2f})")
        except requests.RequestException as exc:
            print(f"Webhook failed for {detection.label}: {exc}")


class YOLOCrashDetector:
    def __init__(self, config: Dict[str, Any], device: Optional[str]) -> None:
        self.enabled = bool(config.get("enabled", False))
        self.model = None
        self.confidence = float(config.get("confidence", 0.45))
        self.iou = float(config.get("iou", 0.5))
        self.image_size = int(config.get("image_size", 640))
        self.device = device
        self.crash_classes = {name.lower() for name in config.get("crash_classes", [])}

        if not self.enabled:
            return

        try:
            from ultralytics import YOLO

            self.model = YOLO(config["model_path"])
        except Exception as exc:
            self.enabled = False
            print(f"YOLO crash detector disabled: {exc}")

    def detect(self, frame: np.ndarray) -> List[Detection]:
        if not self.enabled or self.model is None:
            return []

        results = self.model.predict(
            frame,
            conf=self.confidence,
            iou=self.iou,
            imgsz=self.image_size,
            device=self.device,
            verbose=False,
        )

        detections: List[Detection] = []
        for result in results:
            names = result.names or {}
            for box in result.boxes:
                class_id = int(box.cls[0])
                label = str(names.get(class_id, class_id)).lower()
                if self.crash_classes and label not in self.crash_classes:
                    continue

                x1, y1, x2, y2 = [int(value) for value in box.xyxy[0].tolist()]
                confidence = float(box.conf[0])
                detections.append(
                    Detection(
                        label=f"crash:{label}",
                        confidence=confidence,
                        box=(x1, y1, x2, y2),
                        source="yolov8",
                    )
                )
        return detections


class TensorFlowFireDetector:
    def __init__(self, config: Dict[str, Any]) -> None:
        self.enabled = bool(config.get("enabled", False))
        self.model = None
        self.threshold = float(config.get("threshold", 0.7))
        self.input_size = tuple(config.get("input_size", [224, 224]))
        self.class_names = config.get("class_names", ["no_fire", "fire"])

        if not self.enabled:
            return

        try:
            import tensorflow as tf

            self.model = tf.keras.models.load_model(config["model_path"])
        except Exception as exc:
            self.enabled = False
            print(f"TensorFlow fire detector disabled: {exc}")

    def detect(self, frame: np.ndarray) -> List[Detection]:
        if not self.enabled or self.model is None:
            return []

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, tuple(self.input_size))
        batch = np.expand_dims(resized.astype("float32") / 255.0, axis=0)
        prediction = self.model.predict(batch, verbose=0)[0]

        if np.ndim(prediction) == 0 or len(np.atleast_1d(prediction)) == 1:
            fire_score = float(np.atleast_1d(prediction)[0])
        else:
            fire_index = self.class_names.index("fire") if "fire" in self.class_names else 1
            fire_score = float(prediction[fire_index])

        if fire_score < self.threshold:
            return []

        height, width = frame.shape[:2]
        return [
            Detection(
                label="fire",
                confidence=fire_score,
                box=(0, 0, width, height),
                source="tensorflow",
            )
        ]


class HSVFireDetector:
    def __init__(self, config: Dict[str, Any]) -> None:
        self.enabled = bool(config.get("enabled", True))
        self.lower = np.array(config.get("lower_hsv", [0, 70, 50]), dtype=np.uint8)
        self.upper = np.array(config.get("upper_hsv", [18, 255, 255]), dtype=np.uint8)
        self.min_area = int(config.get("min_area", 12000))

    def detect(self, frame: np.ndarray) -> List[Detection]:
        if not self.enabled:
            return []

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower, self.upper)
        mask = cv2.GaussianBlur(mask, (15, 15), 0)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections: List[Detection] = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_area:
                continue
            x, y, width, height = cv2.boundingRect(contour)
            confidence = min(area / max(self.min_area * 3, 1), 1.0)
            detections.append(
                Detection(
                    label="fire",
                    confidence=confidence,
                    box=(x, y, x + width, y + height),
                    source="hsv",
                )
            )
        return detections


def draw_detections(frame: np.ndarray, detections: Iterable[Detection]) -> np.ndarray:
    for detection in detections:
        x1, y1, x2, y2 = detection.box
        color = (0, 0, 255) if detection.label.startswith("fire") else (0, 165, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        text = f"{detection.label} {detection.confidence:.2f} [{detection.source}]"
        cv2.putText(frame, text, (x1, max(y1 - 8, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return frame


def open_writer(output_path: Optional[str], frame: np.ndarray, fps: float) -> Optional[cv2.VideoWriter]:
    if not output_path:
        return None

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    height, width = frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(output_path, fourcc, fps or 24.0, (width, height))


def run(config: Dict[str, Any]) -> None:
    source = parse_video_source(str(config.get("video_source", "0")))
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video source: {source}")

    yolo_detector = YOLOCrashDetector(config.get("yolo", {}), config.get("device"))
    tensorflow_fire_detector = TensorFlowFireDetector(config.get("tensorflow_fire", {}))
    hsv_fire_detector = HSVFireDetector(config.get("hsv_fire_fallback", {}))
    alerter = WebhookAlerter(
        config.get("webhook_url"),
        int(config.get("alert_cooldown_seconds", 15)),
        bool(config.get("include_snapshot_in_webhook", False)),
    )

    writer = None
    fps = capture.get(cv2.CAP_PROP_FPS) or 24.0
    display = bool(config.get("display", True))

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            frame = resize_frame(frame, config.get("resize_width"))
            detections: List[Detection] = []
            detections.extend(yolo_detector.detect(frame))

            fire_detections = tensorflow_fire_detector.detect(frame)
            if not fire_detections:
                fire_detections = hsv_fire_detector.detect(frame)
            detections.extend(fire_detections)

            for detection in detections:
                print(f"{detection.label} detected ({detection.confidence:.2f}) via {detection.source}")
                alerter.send(detection, frame)

            annotated = draw_detections(frame.copy(), detections)
            if writer is None:
                writer = open_writer(config.get("output_path"), annotated, fps)
            if writer:
                writer.write(annotated)

            if display:
                cv2.imshow("Fire and Crash Detection", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        capture.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="Real-time fire and vehicle crash detection pipeline.")
    parser.add_argument("--config", default="config.example.json", help="Path to the JSON configuration file.")
    parser.add_argument("--source", help="Override the configured video source.")
    parser.add_argument("--headless", action="store_true", help="Run without opening an OpenCV display window.")
    args = parser.parse_args()

    config = load_json(args.config)
    if args.source:
        config["video_source"] = args.source
    if args.headless:
        config["display"] = False

    run(config)


if __name__ == "__main__":
    main()
