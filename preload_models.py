from config import settings
from local_detector import LocalMinifigureDetector

def main():
    if settings.detector_backend != "yolo_world":
        print(
            "Model preload skipped because DETECTOR_BACKEND is not yolo_world."
        )
        return

    print("Preloading YOLO-World model and CLIP text encoder...")
    LocalMinifigureDetector(
        backend="yolo_world",
        world_model=settings.yolo_world_model,
        world_classes=settings.yolo_world_classes,
        custom_model=settings.yolo_custom_model,
        device=settings.yolo_device,
        iou=settings.yolo_iou,
    )
    print("Detector preload complete.")

if __name__ == "__main__":
    main()
