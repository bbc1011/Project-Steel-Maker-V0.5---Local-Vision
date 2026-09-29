# Local detector roadmap

## Stage A — zero-shot test

Use:

    DETECTOR_BACKEND=yolo_world

Measure:
- minifigure recall
- false positives
- broad-sweep time
- deep-scan time
- Brickognize success after local crops

The broad detector should favor recall.

## Stage B — collect a real dataset

Train one class:

    lego_minifigure

Useful images include:
- crowded LEGO lots
- tiny figures
- partial occlusion
- multiple identical figures
- blurry/poorly lit marketplace photos
- loose parts and vehicles as confusing context
- negative photos with no minifigures

## Stage C — train the dedicated detector

Run:

    python train_detector.py

Copy the resulting best weights to:

    models/minifigure_detector.pt

Then:

    DETECTOR_BACKEND=custom_yolo

## Stage D — optimize inference

Once the custom detector is reliable:
- benchmark 416/512/640 broad input sizes
- benchmark 640/768/896 detailed input sizes
- export to ONNX or OpenVINO for CPU deployment
- use TensorRT where supported
- reuse high-confidence broad boxes instead of re-detecting the same photo

The final goal is a detector that is specialized enough that the general
zero-shot model can also be removed.
