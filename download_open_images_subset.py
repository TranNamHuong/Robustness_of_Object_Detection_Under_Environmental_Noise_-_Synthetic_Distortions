"""Download the selected Open Images validation images via the public S3 bucket."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.request import urlopen


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("datasets/open_images_v7"))
    parser.add_argument("--workers", type=int, default=5)
    args = parser.parse_args()
    ids_file = args.root / "image_ids.txt"
    out_dir = args.root / "images" / "val"
    out_dir.mkdir(parents=True, exist_ok=True)
    image_ids = [
        line.strip().split("/", 1)[1]
        for line in ids_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    def fetch(image_id: str) -> tuple[str, str]:
        path = out_dir / f"{image_id}.jpg"
        if path.exists() and path.stat().st_size > 0:
            return image_id, "cached"
        url = f"https://open-images-dataset.s3.amazonaws.com/validation/{image_id}.jpg"
        with urlopen(url, timeout=45) as response:
            data = response.read()
        if not data.startswith(b"\xff\xd8"):
            raise ValueError(f"Unexpected image content for {image_id}")
        path.write_bytes(data)
        return image_id, "downloaded"

    downloaded = 0
    failed: list[tuple[str, str]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(fetch, image_id): image_id for image_id in image_ids}
        for future in as_completed(futures):
            image_id = futures[future]
            try:
                _, status = future.result()
                downloaded += status == "downloaded"
            except Exception as exc:  # preserve per-image errors for retry
                failed.append((image_id, str(exc)))

    print(f"Requested: {len(image_ids)}; downloaded: {downloaded}; cached: {len(image_ids) - downloaded - len(failed)}; failed: {len(failed)}")
    for image_id, error in failed:
        print(f"FAILED {image_id}: {error}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
