#!/usr/bin/env python3
"""Join the 3 front views of each V3 sample into ONE horizontal panorama.

front_left | front | front_right  ->  <scene>/<idx>_CAM_JOINT.jpg
(saved in the SAME directory as the source views), and rewrite the V3 JSON's
`image` field to that single joined image.

Idempotent: skips a joined image that already exists. Parallel across samples.

Usage:
  python join_images.py \
    --json /weka/.../dvla_sft/dvlm-ad_waymo_training_v3.json \
    --out  /weka/.../dvla_sft/dvlm-ad_waymo_training_v3_joint.json \
    --image_root /weka/.../workspace/waymo --workers 16
"""
from __future__ import annotations
import argparse, json, os
from concurrent.futures import ProcessPoolExecutor, as_completed
from PIL import Image


def joined_rel_path(images):
    """Derive '<...>/<idx>_CAM_JOINT.jpg' from the center FRONT view (image[1])."""
    center = images[1]
    assert center.endswith("_CAM_FRONT.jpg"), f"image[1] not a FRONT view: {center}"
    return center[: -len("_CAM_FRONT.jpg")] + "_CAM_JOINT.jpg"


def make_joined(args_tuple):
    images, image_root, quality = args_tuple
    rel = joined_rel_path(images)
    dst = os.path.join(image_root, rel)
    if os.path.exists(dst):
        return rel, "exists"
    ims = [Image.open(os.path.join(image_root, p)).convert("RGB") for p in images]
    h = min(im.height for im in ims)
    ims = [im if im.height == h else im.resize((round(im.width * h / im.height), h))
           for im in ims]
    canvas = Image.new("RGB", (sum(im.width for im in ims), h))
    x = 0
    for im in ims:
        canvas.paste(im, (x, 0)); x += im.width
    canvas.save(dst, quality=quality)
    return rel, "made"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True, help="V3 JSON (3-image samples)")
    ap.add_argument("--out", required=True, help="output V3 JSON (single joined image)")
    ap.add_argument("--image_root", required=True,
                    help="root the image paths are relative to (joined images written here)")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--quality", type=int, default=92)
    ap.add_argument("--n", type=int, default=0, help="process only first N (0 = all)")
    args = ap.parse_args()

    data = json.load(open(args.json))
    if args.n > 0:
        data = data[: args.n]

    # Build the join work-list (samples with exactly 3 views).
    jobs, rels = [], [None] * len(data)
    skipped = 0
    for i, s in enumerate(data):
        imgs = s.get("image") or []
        if len(imgs) != 3:
            skipped += 1
            continue
        rels[i] = joined_rel_path(imgs)
        jobs.append((i, (imgs, args.image_root, args.quality)))

    made = exists = errs = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(make_joined, payload): i for i, payload in jobs}
        for n, fut in enumerate(as_completed(futs), 1):
            try:
                _, status = fut.result()
                made += status == "made"; exists += status == "exists"
            except Exception as e:
                errs += 1
                if errs <= 5:
                    print(f"  [err] {type(e).__name__}: {e}")
            if n % 2000 == 0:
                print(f"  {n}/{len(jobs)} (made={made} exists={exists} err={errs})", flush=True)

    # Rewrite image field -> single joined image.
    for i, s in enumerate(data):
        if rels[i] is not None:
            s["image"] = [rels[i]]

    with open(args.out, "w") as f:
        json.dump(data, f, ensure_ascii=False)
    print(f"\nwrote {len(data)} samples -> {args.out}")
    print(f"  joined: made={made} exists={exists} err={errs} skipped_non3={skipped}")


if __name__ == "__main__":
    main()
