#!/usr/bin/env python3
import os
import argparse
import numpy as np
from pathlib import Path
from PIL import Image
import torch

from depth_anything_3.api import DepthAnything3

# Allowed image extensions
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def load_images(input_dir):
    """Collect all image file paths from the input directory."""
    image_paths = []
    for file in sorted(Path(input_dir).iterdir()):
        if file.is_file() and file.suffix.lower() in EXTS:
            image_paths.append(str(file))
    return image_paths


def ensure_dir(path):
    """Create the directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


def save_depth_dmb(depth, out_path):
    """
    Save float32 depth map (H×W) to APD .dmb format:
    int32[4] header = [version=1, rows, cols, type=5]
    type=5 = CV_32FC1
    """
    depth = depth.astype(np.float32)
    h, w = depth.shape
    header = np.array([1, h, w, 5], dtype=np.int32)

    with open(out_path, "wb") as f:
        f.write(header.tobytes())
        f.write(depth.tobytes())


def normalize_to_255(depth):
    """
    Normalize DA3 metric depth to 0–255 float32.
    APD expects inverse-depth-like values in 0..255 range.
    """
    d = depth.astype(np.float32)

    # Shift to zero
    d = d - d.min()

    # Avoid divide-by-zero
    maxv = d.max()
    if maxv < 1e-6:
        return np.zeros_like(d, dtype=np.float32)

    # Scale to [0,1]
    d = d / maxv

    # Scale to [0,255]
    d = d * 255.0
    return d.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(
        description="Export DepthAnything3 depth maps to APD-compatible .dmb"
    )
    parser.add_argument("--input_dir", required=True,
                        help="Folder with input images")
    parser.add_argument("--output_dir", required=True,
                        help="Folder to write .dmb files")
    parser.add_argument("--model",
                        default="depth-anything/da3-base",
                        help="DA3 model name or path")
    args = parser.parse_args()

    ensure_dir(args.output_dir)

    # Load Depth Anything 3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DepthAnything3.from_pretrained(args.model).to(device)

    # Load images
    images = load_images(args.input_dir)
    if not images:
        print(f"❌ No images found in {args.input_dir}")
        return

    print(f"Found {len(images)} images.")
    print("Running Depth Anything 3...\n")

    for img_path in images:
        # Inference
        pred = model.inference([img_path])
        depth_map = pred.depth  # shape: [1, H, W]

        # Convert tensor → numpy
        if isinstance(depth_map, torch.Tensor):
            depth_map = depth_map.detach().cpu().numpy()

        depth_map = depth_map[0]  # remove batch dimension

        # Match depth to image resolution
        with Image.open(img_path) as im:
            orig_w, orig_h = im.size

        h, w = depth_map.shape
        if (h, w) != (orig_h, orig_w):
            depth_img = Image.fromarray(depth_map, mode="F")
            depth_img = depth_img.resize(
                (orig_w, orig_h), resample=Image.BILINEAR
            )
            depth_map = np.array(depth_img, dtype=np.float32)

        # Normalize depth for APD
        depth_norm = normalize_to_255(depth_map)

        # Save .dmb
        fname = Path(img_path).stem + ".dmb"
        out_file = os.path.join(args.output_dir, fname)
        save_depth_dmb(depth_norm, out_file)
        print(f"✔ Saved depth: {out_file}")

    print("\n✅ Done! All depth maps exported in APD format.")


if __name__ == "__main__":
    main()
