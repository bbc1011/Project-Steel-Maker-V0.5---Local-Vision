import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

def _bool(name, default=False):
    return os.getenv(name, str(default)).lower() in ("1", "true", "yes", "on")

def _device(value):
    value = (value or "auto").strip()
    return None if value.lower() == "auto" else value

@dataclass
class Settings:
    ebay_client_id: str = os.getenv("EBAY_CLIENT_ID", "")
    ebay_client_secret: str = os.getenv("EBAY_CLIENT_SECRET", "")
    ebay_marketplace_id: str = os.getenv("EBAY_MARKETPLACE_ID", "EBAY_US")
    ebay_detail_workers: int = int(os.getenv("EBAY_DETAIL_WORKERS", "8"))

    detector_backend: str = os.getenv("DETECTOR_BACKEND", "yolo_world").lower()
    yolo_world_model: str = os.getenv("YOLO_WORLD_MODEL", "yolov8s-worldv2.pt")
    yolo_world_classes: tuple = tuple(
        x.strip() for x in os.getenv(
            "YOLO_WORLD_CLASSES", "LEGO minifigure;LEGO minifig"
        ).split(";") if x.strip()
    )
    yolo_custom_model: str = os.getenv(
        "YOLO_CUSTOM_MODEL", "models/minifigure_detector.pt"
    )
    yolo_device: object = _device(os.getenv("YOLO_DEVICE", "auto"))
    yolo_iou: float = float(os.getenv("YOLO_IOU", "0.55"))

    overview_sample_images: int = int(os.getenv("OVERVIEW_SAMPLE_IMAGES", "3"))
    overview_batch_size: int = int(os.getenv("OVERVIEW_BATCH_SIZE", "16"))
    overview_imgsz: int = int(os.getenv("OVERVIEW_IMGSZ", "512"))
    overview_confidence: float = float(os.getenv("OVERVIEW_CONFIDENCE", "0.10"))

    detail_batch_size: int = int(os.getenv("DETAIL_BATCH_SIZE", "12"))
    detail_imgsz: int = int(os.getenv("DETAIL_IMGSZ", "896"))
    detail_confidence: float = float(os.getenv("DETAIL_CONFIDENCE", "0.14"))

    max_listing_images: int = int(os.getenv("MAX_LISTING_IMAGES", "12"))
    max_figures_per_image: int = int(os.getenv("MAX_FIGURES_PER_IMAGE", "20"))

    image_dhash_distance: int = int(os.getenv("IMAGE_DHASH_DISTANCE", "6"))
    image_download_workers: int = int(os.getenv("IMAGE_DOWNLOAD_WORKERS", "12"))

    brickognize_workers: int = int(os.getenv("BRICKOGNIZE_WORKERS", "4"))
    brickognize_min_similarity: float = float(
        os.getenv("BRICKOGNIZE_MIN_SIMILARITY", "0.72")
    )
    brickognize_top_k: int = int(os.getenv("BRICKOGNIZE_TOP_K", "5"))

    bricklink_consumer_key: str = os.getenv("BRICKLINK_CONSUMER_KEY", "")
    bricklink_consumer_secret: str = os.getenv("BRICKLINK_CONSUMER_SECRET", "")
    bricklink_token_value: str = os.getenv("BRICKLINK_TOKEN_VALUE", "")
    bricklink_token_secret: str = os.getenv("BRICKLINK_TOKEN_SECRET", "")
    bricklink_country_code: str = os.getenv("BRICKLINK_COUNTRY_CODE", "")
    bricklink_currency: str = os.getenv("BRICKLINK_CURRENCY", "USD")
    price_cache_hours: float = float(os.getenv("PRICE_CACHE_HOURS", "6"))

    search_config: str = os.getenv("SEARCH_CONFIG", "searches.json")
    continuous_scan: bool = _bool("CONTINUOUS_SCAN", False)
    scan_interval_minutes: int = int(os.getenv("SCAN_INTERVAL_MINUTES", "5"))

    save_boxed_previews: bool = _bool("SAVE_BOXED_PREVIEWS", True)
    boxed_preview_dir: str = os.getenv("BOXED_PREVIEW_DIR", "debug_boxes")
    db_path: str = os.getenv("DB_PATH", "deal_finder.db")

settings = Settings()
