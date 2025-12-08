import os
import argparse
import struct
import numpy as np
from pathlib import Path
from PIL import Image

import torch
from depth_anything_3.api import DepthAnything3

def load_images(input_dir):
    """Collect all image file paths from the input directory."""
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    image_paths = []
    for file in sorted(Path(input_dir).iterdir()):
        if file.is_file() and file.suffix.lower() in exts:
            image_paths.append(str(file))
    return image_paths

def ensure_dir(path):
    """Create the directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)

def save_depth_dmb(depth_array, out_path):
    """Save a single-channel float32 depth map to .dmb format."""
    depth = depth_array.astype(np.float32)  # ensure float32
    height, width = depth.shape
    # Prepare header: version=1, rows, cols, type=5 (CV_32FC1)
    header = np.array([1, height, width, 5], dtype=np.int32)
    # Write header and data to file
    with open(out_path, "wb") as f:
        f.write(header.tobytes())
        f.write(depth.tobytes())

def main():
    parser = argparse.ArgumentParser(description="Export depth maps using DA3 to .dmb format")
    parser.add_argument("--input_dir", required=True, help="Path to input image folder")
    parser.add_argument("--output_dir", required=True, help="Path to output .dmb folder")
    parser.add_argument("--model", default="depth-anything/da3-base",
                        help="Name or path of the DA3 model (e.g. 'depth-anything/da3-base')")
    args = parser.parse_args()

    ensure_dir(args.output_dir)

    # Load the Depth Anything 3 model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DepthAnything3.from_pretrained(args.model)  # e.g., "depth-anything/da3-base"
    model = model.to(device=device)

    # Get list of image files in the input directory
    images = load_images(args.input_dir)
    if not images:
        print(f"No images found in {args.input_dir}")
        return

    # Process each image
    for img_path in images:
        # Run depth inference (the model can take a list; we use a single-item list)
        prediction = model.inference([img_path])
        # The prediction object contains the depth map as a NumPy array (shape [1, H, W])
        depth_map = prediction.depth
        if isinstance(depth_map, torch.Tensor):
            # If returned as torch tensor, move to CPU and convert to numpy
            depth_map = depth_map.detach().cpu().numpy()
        # depth_map shape is (N, H, W); for a single image, N=1
        depth_map = depth_map[0]  # extract the depth array for this image

        # Resize depth map to original image size if needed
        with Image.open(img_path) as img:
            orig_width, orig_height = img.size
        dh, dw = depth_map.shape
        if (dh, dw) != (orig_height, orig_width):
            # Resize using PIL (bilinear interpolation for float data)
            depth_img = Image.fromarray(depth_map, mode='F')  # 'F' mode: 32-bit floating point
            depth_img = depth_img.resize((orig_width, orig_height), resample=Image.BILINEAR)
            depth_map = np.array(depth_img, dtype=np.float32)

        # Determine output file path and save
        img_name = Path(img_path).stem  # filename without extension
        out_file = os.path.join(args.output_dir, img_name + ".dmb")
        save_depth_dmb(depth_map, out_file)
        print(f"Saved depth: {out_file}")

if __name__ == "__main__":
    main()
