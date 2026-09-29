from pathlib import Path
from ultralytics import YOLO

DATASET_YAML = "dataset/minifigures.yaml"

def main():
    if not Path(DATASET_YAML).exists():
        raise FileNotFoundError(
            "Missing dataset/minifigures.yaml. See dataset/README.md."
        )

    model = YOLO("yolo26n.pt")
    model.train(
        data=DATASET_YAML,
        epochs=100,
        imgsz=768,
        batch=-1,
        patience=20,
        project="runs/minifigure",
        name="minifigure_detector",
    )

    print(
        "\nCopy the trained best.pt to models/minifigure_detector.pt, "
        "then set DETECTOR_BACKEND=custom_yolo."
    )

if __name__ == "__main__":
    main()
