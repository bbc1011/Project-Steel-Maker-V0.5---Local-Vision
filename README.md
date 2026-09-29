# LEGO Deal Finder v0.5 — Local Vision

v0.5 removes Gemini from the live scanner.

    Gemini API calls per cycle: 0

Pipeline:

    eBay
      ↓
    local YOLO detector
      ↓
    local crops
      ↓
    Brickognize
      ↓
    BrickLink
      ↓
    Buy It Now / auction deal engine

## The new speed-first scan

### Phase 1 — broad sweep

The scanner first discovers the complete opportunity set.

1. Run every configured eBay search.
2. Merge listings that appear in several searches.
3. Fetch listing details concurrently.
4. Reuse cached visual analyses immediately.
5. For uncached listings, sample first + middle + last photos.
6. Run all sampled photos through the local detector in batches.
7. Put only listings with minifigure detections into the promising queue.

No Brickognize or BrickLink analysis blocks this broad sweep.

### Phase 2 — deep analysis

Promising listings are processed in this order:

1. auctions ending soon
2. remaining listings from oldest → newest

The deep stage runs the local detector across the listing photos, crops each
figure, sends the crops to Brickognize, retrieves BrickLink current-average
USED prices, and evaluates the deal.

## Detector modes

### `yolo_world` — works before we have training data

    DETECTOR_BACKEND=yolo_world

The default zero-shot model is:

    yolov8s-worldv2.pt

with local prompts:

    LEGO minifigure
    LEGO minifig

On first use, Ultralytics downloads the model weights. After that, inference is
performed locally and no Gemini key is needed.

This is the bridge detector. Its recall needs to be tested on real eBay photos.

### `custom_yolo` — intended production detector

After we build a training set:

    DETECTOR_BACKEND=custom_yolo
    YOLO_CUSTOM_MODEL=models/minifigure_detector.pt

The included `train_detector.py` starts from YOLO26n and trains a one-class
`lego_minifigure` detector.

See:

    dataset/README.md

## Setup

Windows PowerShell:

    py -3 -m venv .venv
    .\.venv\Scripts\Activate.ps1
    pip install -r requirements.txt

Copy:

    .env.example

to:

    .env

Fill in:
- eBay credentials
- BrickLink credentials

There is no Gemini credential.

Run once:

    python main.py

Continuous five-minute scanning:

    CONTINUOUS_SCAN=true
    SCAN_INTERVAL_MINUTES=5

## Speed/quality controls

Broad sweep defaults:

    OVERVIEW_SAMPLE_IMAGES=3
    OVERVIEW_BATCH_SIZE=16
    OVERVIEW_IMGSZ=512
    OVERVIEW_CONFIDENCE=0.10

A low overview confidence favors recall. A false positive only causes extra
deep analysis; a false negative can hide a valuable listing.

Deep scan defaults:

    DETAIL_BATCH_SIZE=12
    DETAIL_IMGSZ=896
    DETAIL_CONFIDENCE=0.14

For a slower CPU, try:

    OVERVIEW_IMGSZ=416
    DETAIL_IMGSZ=640

For an NVIDIA GPU:

    YOLO_DEVICE=0

## Existing deal logic retained

- near-duplicate photo filtering
- conservative cross-photo figure counting
- Brickognize exact identification
- BrickLink current-average USED valuation
- cached visual analysis
- cheap repeated bid/price monitoring
- Buy It Now discount logic
- auction target maximum bid
- early-auction opportunity alerts
- ending-soon auction opportunity alerts
- reference-image target matching

## Testing

Run the offline logic test:

    python SELF_TEST.py

This does not call eBay, Brickognize, BrickLink, Gemini, or YOLO.


## Railway headless fix — v0.5.1

This build uses `ultralytics-opencv-headless` rather than desktop OpenCV.
It is intended for Railway, Docker, CI, cloud VMs, and other environments
without a display server.

If you previously deployed v0.5 and Railway cached the old dependency layer,
set `NO_CACHE=1` in Railway Variables for one rebuild.


## Railway v0.5.2 — YOLO-World CLIP dependency

YOLO-World uses a CLIP text encoder when `set_classes()` converts prompts such
as `LEGO minifigure` into detector embeddings.

For Railway, v0.5.2 installs that dependency during the Docker build rather
than allowing Ultralytics to attempt a runtime AutoUpdate.

The root `Dockerfile` also preloads the public YOLO-World weights and text
encoder so normal service startup can immediately construct the detector.
