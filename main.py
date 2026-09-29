import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

from config import settings
from database import Database
from ebay_client import EbayClient
from brickognize_client import BrickognizeClient
from bricklink_client import BrickLinkClient
from reference_resolver import resolve_reference_ids
from local_detector import LocalMinifigureDetector
from image_pipeline import (
    download_images_parallel,
    image_signature,
    sample_photo_numbers,
)
from listing_analyzer import analyze_listing_local, refresh_figure_prices
from deal_engine import evaluate_deal

def load_searches():
    with open(settings.search_config, "r", encoding="utf-8") as f:
        return json.load(f)

def parse_time(value):
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except Exception:
        return None

def seconds_until(value):
    dt = parse_time(value)
    if dt is None:
        return None
    return (dt - datetime.now(timezone.utc)).total_seconds()

def local_search_skip(listing, search):
    title = listing.get("title", "").lower()
    excluded = [
        str(x).lower().strip()
        for x in search.get("pre_screen_exclude_title_terms", [])
        if str(x).strip()
    ]
    return any(term in title for term in excluded)

def build_clients(db):
    ebay = EbayClient(
        settings.ebay_client_id,
        settings.ebay_client_secret,
        settings.ebay_marketplace_id,
    )
    brickognize = BrickognizeClient(
        min_similarity=settings.brickognize_min_similarity,
        top_k=settings.brickognize_top_k,
    )
    bricklink = BrickLinkClient(
        consumer_key=settings.bricklink_consumer_key,
        consumer_secret=settings.bricklink_consumer_secret,
        token_value=settings.bricklink_token_value,
        token_secret=settings.bricklink_token_secret,
        database=db,
        cache_hours=settings.price_cache_hours,
        currency_code=settings.bricklink_currency,
        country_code=settings.bricklink_country_code,
    )
    detector = LocalMinifigureDetector(
        backend=settings.detector_backend,
        world_model=settings.yolo_world_model,
        world_classes=settings.yolo_world_classes,
        custom_model=settings.yolo_custom_model,
        device=settings.yolo_device,
        iou=settings.yolo_iou,
    )
    return ebay, brickognize, bricklink, detector

def collect_search_results(ebay, searches):
    merged = {}

    for search_index, search in enumerate(searches):
        name = search.get("name", search.get("query", "Unnamed search"))
        print(f"\nSearching: {name}")
        try:
            listings = ebay.search(
                query=search.get("query", ""),
                filters=search.get("filters", {}),
                max_results=search.get("max_results", 20),
            )
        except Exception as exc:
            print(f"  eBay search failed: {exc}")
            continue

        print(f"  Returned {len(listings)} item(s).")

        for summary in listings:
            item_id = summary.get("item_id")
            if not item_id or local_search_skip(summary, search):
                continue

            entry = merged.setdefault(
                item_id,
                {"summary": summary, "search_indexes": []},
            )
            if search_index not in entry["search_indexes"]:
                entry["search_indexes"].append(search_index)

    return merged

def enrich_all(ebay, items):
    enriched = {}

    with ThreadPoolExecutor(
        max_workers=max(1, settings.ebay_detail_workers)
    ) as pool:
        futures = {
            pool.submit(ebay.enrich_listing, entry["summary"]): item_id
            for item_id, entry in items.items()
        }
        for future in as_completed(futures):
            item_id = futures[future]
            try:
                enriched[item_id] = future.result()
            except Exception as exc:
                print(f"  Detail fetch failed for {item_id}: {exc}")
                enriched[item_id] = items[item_id]["summary"]

    return enriched

def prepare_overview(items, enriched, db):
    cached, uncached, samples = {}, {}, []

    for item_id, entry in items.items():
        listing = enriched[item_id]
        sig = image_signature(listing.get("image_urls", []))
        saved = db.get_listing_analysis(item_id, sig)

        base = {
            "listing": listing,
            "search_indexes": entry["search_indexes"],
            "signature": sig,
        }

        if saved is not None:
            cached[item_id] = {**base, "cached": saved}
            continue

        urls = listing.get("image_urls", [])
        numbers = sample_photo_numbers(
            len(urls),
            settings.overview_sample_images,
        )
        uncached[item_id] = {**base, "overview_photo_numbers": numbers}

        for number in numbers:
            samples.append({
                "item_id": item_id,
                "photo_number": number,
                "url": urls[number - 1],
            })

    return cached, uncached, samples

def broad_local_sweep(detector, uncached, samples):
    print("\n===== PHASE 1: FAST LOCAL BROAD SWEEP =====")

    if not samples:
        return [], {}

    image_cache = download_images_parallel(
        [sample["url"] for sample in samples],
        workers=settings.image_download_workers,
    )
    valid = [s for s in samples if s["url"] in image_cache]
    images = [image_cache[s["url"]] for s in valid]

    results = detector.detect_batch(
        images,
        confidence=settings.overview_confidence,
        imgsz=settings.overview_imgsz,
        batch_size=settings.overview_batch_size,
    )

    promising = set()
    for sample, boxes in zip(valid, results):
        if boxes:
            promising.add(sample["item_id"])

    print(
        f"Broad sweep: {len(uncached)} uncached listing(s), "
        f"{len(valid)} sampled photo(s), "
        f"{len(promising)} promising listing(s)."
    )
    return list(promising), image_cache

def queue_sort_key(record, searches):
    listing = record["listing"]
    options = set(listing.get("buying_options") or [])
    pure_auction = "AUCTION" in options and "FIXED_PRICE" not in options

    ending_soon = False
    remaining = seconds_until(listing.get("item_end_date"))
    if pure_auction and remaining is not None:
        thresholds = [
            float(searches[i].get("auction_ending_soon_minutes", 60)) * 60
            for i in record["search_indexes"]
        ]
        ending_soon = 0 <= remaining <= max(thresholds or [3600])

    created = parse_time(listing.get("item_creation_date"))
    created_key = created.timestamp() if created else float("inf")

    return (0 if ending_soon else 1, created_key)

def resolve_targets(search_index, searches, brickognize, cache):
    if search_index not in cache:
        search = searches[search_index]
        wanted_ids, _ = resolve_reference_ids(
            brickognize=brickognize,
            reference_paths=search.get("reference_images", []),
            explicit_ids=search.get("wanted_bricklink_ids", []),
        )
        cache[search_index] = wanted_ids
    return cache[search_index]

def evaluate_listing(
    listing,
    figures,
    search_indexes,
    searches,
    brickognize,
    target_cache,
):
    evaluated = []
    for index in search_indexes:
        search = searches[index]
        wanted_ids = resolve_targets(
            index, searches, brickognize, target_cache
        )
        result = evaluate_deal(
            listing=listing,
            figures=figures,
            min_discount_percent=search.get("min_discount_percent", 25),
            wanted_ids=wanted_ids,
            require_reference_match=search.get(
                "require_reference_match", False
            ),
            auction_early_alert_discount_percent=search.get(
                "auction_early_alert_discount_percent", 60
            ),
            auction_early_min_room_usd=search.get(
                "auction_early_min_room_usd", 15
            ),
            auction_ending_soon_minutes=search.get(
                "auction_ending_soon_minutes", 60
            ),
        )
        evaluated.append((search, result))
    return evaluated

def print_figures(figures):
    for fig in figures:
        price = fig.get("bricklink_current_avg_used")
        price_text = f"${price:.2f}" if price is not None else "no price"
        print(
            f"    {int(fig.get('quantity', 1) or 1)}x "
            f"{fig.get('name')} | {fig.get('bricklink_id')} | "
            f"BL avg used {price_text}"
        )

def print_evaluation(listing, search, result):
    name = search.get("name", search.get("query", "Search"))
    if result["listing_type"] == "AUCTION":
        print(
            f"  [{name}] auction bid=${result['current_bid']:.2f} | "
            f"max target=${result['max_bid']:.2f} | "
            f"room=${result['bid_room']:.2f} | "
            f"value=${result['estimated_bricklink_value']:.2f}"
        )
    else:
        print(
            f"  [{name}] BIN total=${result['listing_total']:.2f} | "
            f"value=${result['estimated_bricklink_value']:.2f} | "
            f"discount={result['discount_percent']:.1f}%"
        )

    if result.get("is_alert"):
        print(f"    *** {result.get('alert_type')} ***")
        print(f"    {listing.get('item_web_url', '')}")

def run_once():
    searches = load_searches()
    db = Database(settings.db_path)
    ebay, brickognize, bricklink, detector = build_clients(db)

    started = time.perf_counter()

    # Discover all opportunities before deep work starts.
    items = collect_search_results(ebay, searches)
    print(f"\nUnique listings discovered: {len(items)}")

    enriched = enrich_all(ebay, items)
    cached, uncached, samples = prepare_overview(items, enriched, db)

    promising_ids, overview_cache = broad_local_sweep(
        detector, uncached, samples
    )

    queue = []
    for item_id, record in cached.items():
        queue.append({
            "item_id": item_id,
            **record,
            "needs_deep_analysis": False,
        })

    for item_id in promising_ids:
        queue.append({
            "item_id": item_id,
            **uncached[item_id],
            "needs_deep_analysis": True,
        })

    queue.sort(key=lambda record: queue_sort_key(record, searches))

    broad_time = time.perf_counter() - started
    print(
        f"\nBroad discovery finished in {broad_time:.2f}s. "
        f"Deep/evaluation queue: {len(queue)} listing(s)."
    )
    print("Priority: ending-soon auctions, then oldest -> newest.")

    print("\n===== PHASE 2: PRIORITIZED DEEP ANALYSIS =====")
    target_cache = {}

    for position, record in enumerate(queue, start=1):
        listing = record["listing"]
        print(
            f"\n[{position}/{len(queue)}] "
            f"{listing.get('title', '')}"
        )

        if record["needs_deep_analysis"]:
            try:
                analysis = analyze_listing_local(
                    listing=listing,
                    settings=settings,
                    detector=detector,
                    brickognize=brickognize,
                    bricklink=bricklink,
                    image_cache=overview_cache,
                )
            except Exception as exc:
                print(f"  Deep analysis failed: {exc}")
                continue

            figures = analysis["figures"]
            db.save_listing_analysis(
                item_id=record["item_id"],
                image_signature=record["signature"],
                status=analysis["status"],
                figures=figures,
                meta=analysis["meta"],
            )
            print(
                f"  Local boxes: {analysis['meta'].get('detector_boxes', 0)} | "
                f"unique photos: {analysis['meta'].get('unique_images', 0)}"
            )
        else:
            figures = refresh_figure_prices(
                record["cached"]["figures"],
                bricklink,
            )
            print("  Cached visual analysis reused; no detector pass.")

        print_figures(figures)

        for search, result in evaluate_listing(
            listing,
            figures,
            record["search_indexes"],
            searches,
            brickognize,
            target_cache,
        ):
            print_evaluation(listing, search, result)

        db.save_listing_state(listing, record["signature"])

    total = time.perf_counter() - started
    print(f"\nCycle complete in {total:.2f}s. Gemini API calls: 0.")

def main():
    while True:
        run_once()
        if not settings.continuous_scan:
            break
        print(f"\nWaiting {settings.scan_interval_minutes} minute(s)...")
        time.sleep(max(1, settings.scan_interval_minutes) * 60)

if __name__ == "__main__":
    main()
