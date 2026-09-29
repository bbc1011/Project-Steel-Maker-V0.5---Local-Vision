from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from PIL import ImageDraw

from image_pipeline import download_images_parallel, unique_image_records

def _crop_from_detection(image, detection, padding=0.08):
    width, height = image.size
    x1, y1, x2, y2 = detection["pixel_box"]

    x1 = max(0, min(width, float(x1)))
    y1 = max(0, min(height, float(y1)))
    x2 = max(0, min(width, float(x2)))
    y2 = max(0, min(height, float(y2)))

    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    px, py = bw * padding, bh * padding

    box = (
        int(max(0, x1 - px)),
        int(max(0, y1 - py)),
        int(min(width, x2 + px)),
        int(min(height, y2 + py)),
    )
    if box[2] - box[0] < 30 or box[3] - box[1] < 30:
        return None

    return {
        "pixel_box": box,
        "confidence": float(detection.get("confidence", 0)),
        "label": detection.get("label", "minifigure"),
        "image": image.crop(box).copy(),
    }

def _save_preview(image, figures, output_path):
    preview = image.copy()
    draw = ImageDraw.Draw(preview)
    for i, fig in enumerate(figures, start=1):
        x1, y1, x2, y2 = fig["pixel_box"]
        draw.rectangle((x1, y1, x2, y2), width=4)
        draw.text(
            (x1 + 4, max(0, y1 - 16)),
            f"Fig {i} {fig.get('confidence', 0):.2f}",
        )

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    preview.save(path)

def _recognize_one(brickognize, photo_number, crop):
    try:
        result = brickognize.identify_pil(crop["image"])
        best = brickognize.best_candidate(result)
        if not best or not best.get("bricklink_id"):
            return {
                "status": "unidentified",
                "photo_number": photo_number,
                "detector_confidence": crop.get("confidence", 0),
            }

        return {
            "status": "identified",
            "photo_number": photo_number,
            "bricklink_id": str(best["bricklink_id"]),
            "name": best.get("name"),
            "brickognize_score": float(best.get("score", 0) or 0),
            "detector_confidence": float(crop.get("confidence", 0) or 0),
        }
    except Exception as exc:
        return {
            "status": "error",
            "photo_number": photo_number,
            "error": str(exc),
        }

def aggregate_recognized_figures(per_photo_results):
    counts_by_id = defaultdict(dict)
    best_meta = {}

    for photo_number, recognized in per_photo_results.items():
        counts = Counter(
            item["bricklink_id"]
            for item in recognized
            if item.get("status") == "identified"
        )
        for item_id, count in counts.items():
            counts_by_id[item_id][photo_number] = count

        for item in recognized:
            if item.get("status") != "identified":
                continue
            item_id = item["bricklink_id"]
            if (
                item_id not in best_meta
                or item.get("brickognize_score", 0)
                > best_meta[item_id].get("brickognize_score", 0)
            ):
                best_meta[item_id] = dict(item)

    aggregated = []
    for item_id, by_photo in counts_by_id.items():
        meta = best_meta[item_id]
        aggregated.append({
            "status": "identified",
            "bricklink_id": item_id,
            "name": meta.get("name"),
            "brickognize_score": meta.get("brickognize_score", 0),
            "detector_confidence": meta.get("detector_confidence", 0),
            "quantity": max(by_photo.values()),
            "observed_photo_counts": {
                str(k): int(v) for k, v in sorted(by_photo.items())
            },
        })

    return sorted(aggregated, key=lambda x: x["bricklink_id"])

def refresh_figure_prices(figures, bricklink):
    refreshed = []
    for figure in figures:
        item = dict(figure)
        item_id = item.get("bricklink_id")
        if not item_id:
            refreshed.append(item)
            continue

        try:
            price = bricklink.current_average_used_price(item_id)
        except Exception as exc:
            price = None
            item["price_error"] = str(exc)

        item["bricklink_current_avg_used"] = (
            price.get("avg_price") if price else None
        )
        item["currency"] = price.get("currency_code") if price else None
        item["bricklink_inventory_count"] = (
            price.get("unit_quantity") if price else None
        )
        refreshed.append(item)

    return refreshed

def analyze_listing_local(
    listing,
    settings,
    detector,
    brickognize,
    bricklink,
    image_cache=None,
):
    urls = listing.get("image_urls", [])[:settings.max_listing_images]
    image_cache = download_images_parallel(
        urls,
        workers=settings.image_download_workers,
        existing=image_cache,
    )

    records = unique_image_records(
        urls,
        image_cache,
        max_images=settings.max_listing_images,
        max_hash_distance=settings.image_dhash_distance,
    )
    if not records:
        return {
            "status": "no_images",
            "figures": [],
            "meta": {"unique_images": 0, "detector_boxes": 0},
        }

    detections = detector.detect_batch(
        [r["image"] for r in records],
        confidence=settings.detail_confidence,
        imgsz=settings.detail_imgsz,
        batch_size=settings.detail_batch_size,
    )

    crops_by_photo = {}
    total_boxes = 0

    for record, photo_detections in zip(records, detections):
        crops = []
        for detection in photo_detections[:settings.max_figures_per_image]:
            crop = _crop_from_detection(record["image"], detection)
            if crop:
                crops.append(crop)

        crops_by_photo[record["photo_number"]] = crops
        total_boxes += len(crops)

        if settings.save_boxed_previews and crops:
            safe_id = str(listing["item_id"]).replace("|", "_")
            _save_preview(
                record["image"],
                crops,
                Path(settings.boxed_preview_dir)
                / safe_id
                / f"photo_{record['photo_number']}.jpg",
            )

    jobs = [
        (photo_number, crop)
        for photo_number, crops in crops_by_photo.items()
        for crop in crops
    ]
    per_photo_results = defaultdict(list)

    if jobs:
        with ThreadPoolExecutor(
            max_workers=max(1, settings.brickognize_workers)
        ) as pool:
            futures = {
                pool.submit(_recognize_one, brickognize, number, crop): number
                for number, crop in jobs
            }
            for future in as_completed(futures):
                number = futures[future]
                per_photo_results[number].append(future.result())

    aggregated = aggregate_recognized_figures(per_photo_results)
    priced = refresh_figure_prices(aggregated, bricklink)

    return {
        "status": "analyzed",
        "figures": priced,
        "meta": {
            "unique_images": len(records),
            "detector_boxes": total_boxes,
            "detector_backend": settings.detector_backend,
        },
    }
