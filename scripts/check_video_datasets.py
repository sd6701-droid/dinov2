#!/usr/bin/env python3
"""
Sanity-check the video datasets in a DINOv2 config before launching training.

For every "VideoFiles:..." entry in train.dataset_path it prints how many videos were
parsed, a few resolved paths, how many of a random sample exist on disk, and decodes
one frame from a few of them.

    PYTHONPATH=. python scripts/check_video_datasets.py --config dinov2/configs/train/vits14_video.yaml
"""

import argparse
import os
import random
import time

from omegaconf import OmegaConf

from dinov2.data.loaders import _parse_dataset_str
from dinov2.data.datasets.video_files import VideoFilesDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--n-exist", type=int, default=200, help="videos to check for existence per dataset")
    parser.add_argument("--n-decode", type=int, default=5, help="videos to decode one frame from per dataset")
    args = parser.parse_args()

    dataset_path = OmegaConf.load(args.config).train.dataset_path
    total = 0
    for sub in [s.strip() for s in dataset_path.split("+") if s.strip()]:
        print(f"\n=== {sub}")
        class_, kwargs = _parse_dataset_str(sub)
        if class_ is not VideoFilesDataset:
            print("  (not a VideoFiles dataset, skipped)")
            continue
        ds = class_(**kwargs)
        n_videos = len(ds._videos)
        total += len(ds)
        print(f"  videos: {n_videos:,d}   samples/epoch: {len(ds):,d}")
        for p in ds._videos[:3]:
            print(f"  e.g. {p}")

        sample = random.sample(ds._videos, min(args.n_exist, n_videos))
        missing = [p for p in sample if not os.path.isfile(p)]
        print(f"  exist on disk: {len(sample) - len(missing)}/{len(sample)} sampled")
        for p in missing[:3]:
            print(f"  MISSING: {p}")

        for i in random.sample(range(n_videos), min(args.n_decode, n_videos)):
            t0 = time.time()
            try:
                img, _ = ds[i]
                print(f"  decoded {img.size[0]}x{img.size[1]} in {time.time() - t0:.2f}s  {ds.video_path(i)}")
            except Exception as e:
                print(f"  DECODE FAILED: {ds.video_path(i)}: {e}")

    print(f"\nTotal samples per epoch: {total:,d}")


if __name__ == "__main__":
    main()
