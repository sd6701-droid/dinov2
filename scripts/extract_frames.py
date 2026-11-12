#!/usr/bin/env python3
"""
Decode videos into JPEG frames for DINOv2 image pretraining.

DINOv2 trains on single images, so SSv2 (.webm), Kinetics-400 (.mp4) and driving
videos are first turned into frames:

    out_root/<relative/video/path without extension>/000001.jpg, 000002.jpg, ...

Example:
    python scripts/extract_frames.py --src /data/ssv2/videos --dst /data/frames/ssv2 --fps 2 --short-side 256
    python scripts/extract_frames.py --src /data/k400/train  --dst /data/frames/k400 --fps 1 --short-side 256

Requires ffmpeg on PATH. Already-extracted videos are skipped, so the script can be re-run
after an interruption. Datasets that already ship as images (many driving datasets do) do
not need this step: point VideoFrames:root=... straight at the image folder.
"""

import argparse
import os
import subprocess
import sys
from multiprocessing import Pool

VIDEO_EXTS = (".mp4", ".webm", ".mkv", ".avi", ".mov", ".m4v")


def find_videos(src):
    videos = []
    for dirpath, _, filenames in os.walk(src):
        for fname in filenames:
            if fname.lower().endswith(VIDEO_EXTS):
                videos.append(os.path.join(dirpath, fname))
    videos.sort()
    return videos


def extract_one(job):
    video, src, dst, fps, short_side, quality = job
    rel = os.path.splitext(os.path.relpath(video, src))[0]
    out_dir = os.path.join(dst, rel)
    done_marker = os.path.join(out_dir, ".done")
    if os.path.exists(done_marker):
        return video, "skipped"
    os.makedirs(out_dir, exist_ok=True)

    # Resize so the shorter side is `short_side` (keeps aspect ratio, even dimensions).
    scale = f"scale='if(lt(iw,ih),{short_side},-2)':'if(lt(iw,ih),-2,{short_side})'"
    cmd = [
        "ffmpeg", "-loglevel", "error", "-nostdin", "-y",
        "-i", video,
        "-vf", f"fps={fps},{scale}",
        "-q:v", str(quality),
        os.path.join(out_dir, "%06d.jpg"),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return video, f"error: {result.stderr.strip()[:200]}"
    open(done_marker, "w").close()
    return video, "ok"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", required=True, help="Folder containing videos (searched recursively)")
    parser.add_argument("--dst", required=True, help="Output folder for frames")
    parser.add_argument("--fps", type=float, default=1.0, help="Frames per second to keep (default 1)")
    parser.add_argument("--short-side", type=int, default=256, help="Resize shorter side to this many pixels")
    parser.add_argument("--quality", type=int, default=3, help="ffmpeg JPEG quality, 2 (best) to 31 (worst)")
    parser.add_argument("--workers", type=int, default=os.cpu_count(), help="Parallel ffmpeg processes")
    args = parser.parse_args()

    videos = find_videos(args.src)
    print(f"Found {len(videos):,d} videos under {args.src}", flush=True)
    jobs = [(v, args.src, args.dst, args.fps, args.short_side, args.quality) for v in videos]

    n_ok = n_skip = n_err = 0
    with Pool(args.workers) as pool:
        for i, (video, status) in enumerate(pool.imap_unordered(extract_one, jobs, chunksize=8), 1):
            if status == "ok":
                n_ok += 1
            elif status == "skipped":
                n_skip += 1
            else:
                n_err += 1
                print(f"[{status}] {video}", file=sys.stderr, flush=True)
            if i % 1000 == 0 or i == len(jobs):
                print(f"{i:,d}/{len(jobs):,d}  ok={n_ok:,d} skipped={n_skip:,d} errors={n_err:,d}", flush=True)


if __name__ == "__main__":
    main()
