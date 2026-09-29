import hashlib
import io
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from PIL import Image

def download_image(url):
    r = requests.get(url, timeout=25)
    r.raise_for_status()
    return Image.open(io.BytesIO(r.content)).convert("RGB")

def download_images_parallel(urls, workers=12, existing=None):
    cache = dict(existing or {})
    missing = [u for u in dict.fromkeys(urls) if u and u not in cache]

    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
        futures = {pool.submit(download_image, url): url for url in missing}
        for future in as_completed(futures):
            url = futures[future]
            try:
                cache[url] = future.result()
            except Exception as exc:
                print(f"  Image download failed: {exc}")
    return cache

def dhash(image, hash_size=8):
    gray = image.convert("L").resize(
        (hash_size + 1, hash_size), Image.Resampling.LANCZOS
    )
    pixels = list(gray.getdata())
    value = 0
    for row in range(hash_size):
        offset = row * (hash_size + 1)
        for col in range(hash_size):
            value <<= 1
            value |= pixels[offset + col] > pixels[offset + col + 1]
    return value

def hamming_distance(a, b):
    return (a ^ b).bit_count()

def image_signature(urls):
    joined = "\n".join(str(x).strip() for x in urls if x)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()

def sample_photo_numbers(count, desired=3):
    count = int(count)
    desired = max(1, int(desired))
    if count <= 0:
        return []
    if count <= desired:
        return list(range(1, count + 1))
    if desired == 1:
        return [1]

    out = []
    for i in range(desired):
        pos = round(i * (count - 1) / (desired - 1))
        number = pos + 1
        if number not in out:
            out.append(number)
    return out

def unique_image_records(urls, image_cache, max_images=12, max_hash_distance=6):
    records, hashes = [], []

    for number, url in enumerate(urls[:max_images], start=1):
        image = image_cache.get(url)
        if image is None:
            continue

        h = dhash(image)
        if any(hamming_distance(h, old) <= max_hash_distance for old in hashes):
            continue

        hashes.append(h)
        records.append({
            "photo_number": number,
            "url": url,
            "image": image,
            "dhash": f"{h:016x}",
        })

    return records
