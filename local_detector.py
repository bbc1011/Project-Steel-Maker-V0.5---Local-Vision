from pathlib import Path

class LocalMinifigureDetector:
    def __init__(
        self,
        backend="yolo_world",
        world_model="yolov8s-worldv2.pt",
        world_classes=("LEGO minifigure", "LEGO minifig"),
        custom_model="models/minifigure_detector.pt",
        device=None,
        iou=0.55,
    ):
        from ultralytics import YOLO

        self.backend = backend.lower().strip()
        self.device = device
        self.iou = float(iou)

        if self.backend == "yolo_world":
            self.model = YOLO(world_model)
            classes = list(world_classes)
            if not classes:
                raise ValueError("YOLO_WORLD_CLASSES cannot be empty.")
            self.model.set_classes(classes)
        elif self.backend == "custom_yolo":
            path = Path(custom_model)
            if not path.exists():
                raise FileNotFoundError(
                    f"Custom detector not found: {path}. "
                    "Train one or use DETECTOR_BACKEND=yolo_world."
                )
            self.model = YOLO(str(path))
        else:
            raise ValueError("DETECTOR_BACKEND must be yolo_world or custom_yolo.")

    def detect_batch(self, images, confidence=0.15, imgsz=640, batch_size=16):
        if not images:
            return []

        all_results = []
        step = max(1, int(batch_size))

        for start in range(0, len(images), step):
            chunk = images[start:start + step]
            kwargs = dict(
                source=chunk,
                conf=float(confidence),
                iou=self.iou,
                imgsz=int(imgsz),
                verbose=False,
            )
            if self.device is not None:
                kwargs["device"] = self.device

            results = self.model.predict(**kwargs)

            for result in results:
                detections = []
                boxes = getattr(result, "boxes", None)
                if boxes is None:
                    all_results.append(detections)
                    continue

                xyxy = boxes.xyxy.detach().cpu().tolist()
                confs = boxes.conf.detach().cpu().tolist()
                classes = boxes.cls.detach().cpu().tolist()
                names = getattr(result, "names", {}) or {}

                for coords, conf, cls_id in zip(xyxy, confs, classes):
                    cls_id = int(cls_id)
                    if isinstance(names, dict):
                        label = names.get(cls_id, "minifigure")
                    elif 0 <= cls_id < len(names):
                        label = names[cls_id]
                    else:
                        label = "minifigure"

                    if not str(label).strip():
                        continue

                    detections.append({
                        "pixel_box": [float(v) for v in coords],
                        "confidence": float(conf),
                        "class_id": cls_id,
                        "label": str(label),
                    })

                all_results.append(detections)

        return all_results
