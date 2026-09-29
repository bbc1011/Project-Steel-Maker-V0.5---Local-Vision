from PIL import Image

from deal_engine import evaluate_deal
from image_pipeline import dhash, hamming_distance, sample_photo_numbers
from listing_analyzer import aggregate_recognized_figures

def run():
    assert sample_photo_numbers(10, 3) == [1, 5, 10]
    assert sample_photo_numbers(5, 3) == [1, 3, 5]
    assert sample_photo_numbers(2, 3) == [1, 2]

    a = Image.new("RGB", (100, 100), "white")
    b = a.copy()
    assert hamming_distance(dhash(a), dhash(b)) == 0

    per_photo = {
        1: [
            {
                "status": "identified",
                "bricklink_id": "sw0001",
                "name": "A",
                "brickognize_score": 0.90,
            },
            {
                "status": "identified",
                "bricklink_id": "sw0001",
                "name": "A",
                "brickognize_score": 0.91,
            },
        ],
        2: [
            {
                "status": "identified",
                "bricklink_id": "sw0001",
                "name": "A",
                "brickognize_score": 0.95,
            },
        ],
    }
    aggregated = aggregate_recognized_figures(per_photo)
    assert len(aggregated) == 1
    assert aggregated[0]["quantity"] == 2

    figures = [{
        "bricklink_id": "sw0001",
        "quantity": 2,
        "bricklink_current_avg_used": 50,
    }]

    fixed = evaluate_deal(
        listing={
            "price": 50,
            "shipping": 5,
            "buying_options": ["FIXED_PRICE"],
        },
        figures=figures,
        min_discount_percent=25,
    )
    assert round(fixed["discount_percent"], 1) == 45.0
    assert fixed["is_deal"] is True

    auction = evaluate_deal(
        listing={
            "price": 20,
            "current_bid_price": 20,
            "shipping": 5,
            "buying_options": ["AUCTION"],
            "bid_count": 0,
        },
        figures=figures,
        min_discount_percent=30,
        auction_early_alert_discount_percent=60,
        auction_early_min_room_usd=10,
    )
    assert round(auction["max_bid"], 2) == 65.0
    assert round(auction["bid_room"], 2) == 45.0
    assert auction["alert_type"] == "auction_early_opportunity"

    print("SELF_TEST passed.")

if __name__ == "__main__":
    run()
