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
    
    print(f"   Saved {out_path} [{h}×{w}]")


def normalize_depth_for_apd(depth):
    """
    Normalize DA3 metric depth for APD compatibility.
    
    APD expects depth values where SMALLER values = CLOSER objects.
    DA3 outputs metric depth where LARGER values = FARTHER objects.
    
    We invert and scale to 0-255 range to match APD's expected format.
    """
    d = depth.astype(np.float64)
    
    # Remove NaN and Inf
    d = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Handle edge cases
    if d.max() == d.min() or d.max() <= 0:
        print("   ⚠ Warning: Constant depth map detected")
        return np.full_like(d, 127.5, dtype=np.float32)
    
    # Invert depth (closer objects = higher values)
    d_max = d.max()
    d_inv = d_max - d
    
    # Normalize to [0, 1]
    d_min = d_inv.min()
    d_range = d_inv.max() - d_min
    
    if d_range < 1e-6:
        return np.full_like(d, 127.5, dtype=np.float32)
    
    d_norm = (d_inv - d_min) / d_range
    
    # Scale to [0, 255]
    d_scaled = d_norm * 255.0
    
    return d_scaled.astype(np.float32)


def process_single_image(img_path, model, device, output_dir, target_size=None):
    """Process a single image and save depth map."""
    
    # Load original image to get dimensions
    with Image.open(img_path) as im:
        orig_w, orig_h = im.size
        orig_mode = im.mode
    
    print(f"\n📸 Processing: {Path(img_path).name}")
    print(f"   Input size: {orig_w}×{orig_h}")
    
    # Run inference
    pred = model.inference([img_path])
    depth_map = pred.depth  # shape: [1, H, W]
    
    # Convert tensor to numpy
    if isinstance(depth_map, torch.Tensor):
        depth_map = depth_map.detach().cpu().numpy()
    
    depth_map = depth_map[0]  # remove batch dimension
    print(f"   DA3 output: {depth_map.shape[1]}×{depth_map.shape[0]}")
    print(f"   Depth range: [{depth_map.min():.2f}, {depth_map.max():.2f}]")
    
    # Resize depth to match original image if needed
    if target_size is not None:
        target_w, target_h = target_size
    else:
        target_w, target_h = orig_w, orig_h
    
    h, w = depth_map.shape
    if (w, h) != (target_w, target_h):
        print(f"   Resizing depth to: {target_w}×{target_h}")
        depth_img = Image.fromarray(depth_map, mode="F")
        depth_img = depth_img.resize(
            (target_w, target_h), 
            resample=Image.BILINEAR
        )
        depth_map = np.array(depth_img, dtype=np.float32)
    
    # Normalize for APD
    depth_norm = normalize_depth_for_apd(depth_map)
    print(f"   Normalized range: [{depth_norm.min():.2f}, {depth_norm.max():.2f}]")
    
    # Save .dmb
    fname = Path(img_path).stem + ".dmb"
    out_file = os.path.join(output_dir, fname)
    save_depth_dmb(depth_norm, out_file)
    
    return depth_norm


def verify_output(output_dir, num_expected):
    """Verify all files were created correctly."""
    dmb_files = list(Path(output_dir).glob("*.dmb"))
    
    print(f"\n{'='*60}")
    print(f"📊 Summary:")
    print(f"   Expected files: {num_expected}")
    print(f"   Created files:  {len(dmb_files)}")
    
    if len(dmb_files) == num_expected:
        print(f"✅ All depth maps exported successfully!")
    else:
        print(f"⚠ Warning: File count mismatch!")
        
    return len(dmb_files) == num_expected


def main():
    parser = argparse.ArgumentParser(
        description="Export DepthAnything3 depth maps to APD-compatible .dmb format"
    )
    parser.add_argument(
        "--input_dir", 
        required=True,
        help="Folder with input images (e.g., dense_folder/images)"
    )
    parser.add_argument(
        "--output_dir", 
        required=True,
        help="Folder to write .dmb files (e.g., dense_folder/dep)"
    )
    parser.add_argument(
        "--model",
        default="depth-anything/da3-base",
        help="DA3 model name or path"
    )
    parser.add_argument(
        "--target_width",
        type=int,
        default=None,
        help="Target width for output depth maps (optional)"
    )
    parser.add_argument(
        "--target_height",
        type=int,
        default=None,
        help="Target height for output depth maps (optional)"
    )
    
    args = parser.parse_args()
    
    # Validate input directory
    if not os.path.exists(args.input_dir):
        print(f"❌ Error: Input directory does not exist: {args.input_dir}")
        return 1
    
    # Create output directory
    ensure_dir(args.output_dir)
    print(f"📁 Output directory: {args.output_dir}")
    
    # Set target size if specified
    target_size = None
    if args.target_width and args.target_height:
        target_size = (args.target_width, args.target_height)
        print(f"🎯 Target size: {target_size[0]}×{target_size[1]}")
    
    # Load Depth Anything 3
    print(f"\n🔄 Loading model: {args.model}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   Using device: {device}")
    
    try:
        model = DepthAnything3.from_pretrained(args.model).to(device).eval()
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return 1
    
    # Load images
    images = load_images(args.input_dir)
    if not images:
        print(f"❌ No images found in {args.input_dir}")
        return 1
    
    print(f"\n✅ Found {len(images)} images")
    print(f"{'='*60}")
    
    # Process each image
    for i, img_path in enumerate(images, 1):
        print(f"\n[{i}/{len(images)}]", end=" ")
        try:
            process_single_image(img_path, model, device, args.output_dir, target_size)
        except Exception as e:
            print(f"   ❌ Error processing {Path(img_path).name}: {e}")
            continue
    
    # Verify output
    verify_output(args.output_dir, len(images))
    
    print(f"\n{'='*60}")
    print(f"✅ Done! Check output in: {args.output_dir}\n")
    return 0


if __name__ == "__main__":
    exit(main())