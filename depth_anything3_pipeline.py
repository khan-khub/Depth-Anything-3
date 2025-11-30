import os
import glob
import argparse
import torch
import numpy as np
from PIL import Image
import json
from pygltflib import GLTF2
import open3d as o3d

from depth_anything_3.api import DepthAnything3
from depth_anything_3.utils.visualize import visualize_depth


def load_images(image_dir):
    exts = ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif"]
    files = []
    for e in exts:
        files.extend(glob.glob(os.path.join(image_dir, e)))
    return sorted(files)


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def save_depth_png(depth_maps, out_dir):
    """Save raw depth as 16-bit PNG (scaled)."""
    depth_png_dir = os.path.join(out_dir, "depth_png")
    ensure_dir(depth_png_dir)

    for i, depth in enumerate(depth_maps):
        # Normalize to [0, 1], scale to 16-bit range for preserving detail
        d = depth - depth.min()
        d = d / (d.max() + 1e-8)
        d_uint16 = (d * 65535).astype(np.uint16)

        img = Image.fromarray(d_uint16)
        img.save(os.path.join(depth_png_dir, f"depth_{i:04d}.png"))


def save_conf_png(conf_maps, out_dir):
    """Save confidence maps as 8-bit grayscale PNG."""
    conf_png_dir = os.path.join(out_dir, "conf_png")
    ensure_dir(conf_png_dir)

    for i, conf in enumerate(conf_maps):
        c = conf - conf.min()
        c = c / (c.max() + 1e-8)
        c_uint8 = (c * 255).astype(np.uint8)

        img = Image.fromarray(c_uint8)
        img.save(os.path.join(conf_png_dir, f"conf_{i:04d}.png"))


def save_intrinsics(intrinsics, out_dir):
    cam_dir = os.path.join(out_dir, "camera")
    ensure_dir(cam_dir)

    # Save TXT
    with open(os.path.join(cam_dir, "intrinsics.txt"), "w") as f:
        f.write("Camera Intrinsics (3x3):\n")
        f.write(str(intrinsics))

    # Save JSON
    with open(os.path.join(cam_dir, "intrinsics.json"), "w") as f:
        json.dump(intrinsics.tolist(), f, indent=4)


def save_extrinsics(extrinsics, out_dir):
    cam_dir = os.path.join(out_dir, "camera/extrinsics")
    ensure_dir(cam_dir)

    for i, ext in enumerate(extrinsics):
        # TXT
        with open(os.path.join(cam_dir, f"extr_{i:04d}.txt"), "w") as f:
            f.write("Extrinsics (World-to-Camera 3x4):\n")
            f.write(str(ext))

        # JSON
        with open(os.path.join(cam_dir, f"extr_{i:04d}.json"), "w") as f:
            json.dump(ext.tolist(), f, indent=4)


def visualize_and_save_depth(depth_maps, out_dir):
    """Save colorized depth maps as PNG."""
    vis_dir = os.path.join(out_dir, "depth_vis")
    ensure_dir(vis_dir)

    for i, depth in enumerate(depth_maps):
        depth_vis = visualize_depth(depth, cmap="Spectral")   # numpy array (H,W,3)
        depth_img = Image.fromarray(depth_vis.astype(np.uint8))
        depth_img.save(os.path.join(vis_dir, f"depth_{i:04d}.png"))


def glb_to_ply(glb_path, ply_path, min_points=400):
    """
    Convert a GLB file into a clean PLY point cloud.
    Removes tiny clusters (cameras) and keeps real scene points only.
    """

    print(f"🌀 Converting GLB → PLY: {os.path.basename(glb_path)}")

    gltf = GLTF2().load(glb_path)
    buffer = gltf.binary_blob()

    # -------------------------
    # Helper: read accessor
    # -------------------------
    def read_accessor(accessor):
        view = gltf.bufferViews[accessor.bufferView]
        start = view.byteOffset + accessor.byteOffset

        dtype_map = {5126: np.float32, 5123: np.uint16, 5121: np.uint8}
        type_map = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}

        dtype = dtype_map[accessor.componentType]
        ncomp = type_map[accessor.type]

        stride = view.byteStride or ncomp * np.dtype(dtype).itemsize

        arr = np.zeros((accessor.count, ncomp), dtype=dtype)

        for i in range(accessor.count):
            i0 = start + i * stride
            raw = buffer[i0 : i0 + ncomp * np.dtype(dtype).itemsize]
            arr[i] = np.frombuffer(raw, dtype=dtype, count=ncomp)

        return arr.astype(np.float32)

    # -------------------------
    # Extract all point clouds
    # -------------------------
    all_pts = []
    all_cols = []

    for mesh in gltf.meshes:
        for prim in mesh.primitives:

            # position required
            if prim.attributes.POSITION is None:
                continue

            pts = read_accessor(gltf.accessors[prim.attributes.POSITION])

            # filter tiny clusters (camera nodes)
            if pts.shape[0] < min_points:
                continue

            all_pts.append(pts)

            # Optional color
            if prim.attributes.COLOR_0 is not None:
                cols = read_accessor(gltf.accessors[prim.attributes.COLOR_0])

                # Normalize if 0–255
                if cols.max() > 1.5:
                    cols /= 255.0

                if cols.shape[1] == 4:
                    cols = cols[:, :3]

                all_cols.append(cols)
            else:
                all_cols.append(None)

    # -------------------------
    # Merge arrays
    # -------------------------
    if len(all_pts) == 0:
        print("❌ No valid point clusters found in GLB.")
        return False

    points = np.vstack(all_pts)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    if any(c is not None for c in all_cols):
        colors = np.vstack([c for c in all_cols if c is not None])
        pcd.colors = o3d.utility.Vector3dVector(colors)

    # -------------------------
    # Save PLY
    # -------------------------
    o3d.io.write_point_cloud(ply_path, pcd)
    print(f"✅ Saved clean PLY → {ply_path}  ({len(points)} points)")

    return True


def main(args):

    # Load images
    image_paths = load_images(args.input)
    if len(image_paths) == 0:
        raise ValueError(f"No images found in: {args.input}")

    print(f"📂 Found {len(image_paths)} images")

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Loading model on {device}: {args.model}")

    model = DepthAnything3.from_pretrained(args.model)
    model = model.to(device).eval()

    # Prepare output directory
    ensure_dir(args.output)

    # Run inference
    prediction = model.inference(
        image=image_paths,
        process_res=args.resolution,
        process_res_method="upper_bound_resize",
        export_dir=args.output,
        export_format=args.format
    )

    # Save numpy arrays
    np_dir = os.path.join(args.output, "raw_numpy")
    ensure_dir(np_dir)
    np.save(os.path.join(np_dir, "depth.npy"), prediction.depth)
    np.save(os.path.join(np_dir, "conf.npy"), prediction.conf)
    np.save(os.path.join(np_dir, "intrinsics.npy"), prediction.intrinsics)
    np.save(os.path.join(np_dir, "extrinsics.npy"), prediction.extrinsics)
    print(f"💾 Saved numpy arrays to: {np_dir}")

    # Extra image outputs
    print("🎨 Saving depth PNGs (16-bit grayscale)...")
    save_depth_png(prediction.depth, args.output)

    print("🎨 Saving confidence PNGs (8-bit grayscale)...")
    save_conf_png(prediction.conf, args.output)

    print("🎨 Saving colorized (Spectral) depth visualizations...")
    visualize_and_save_depth(prediction.depth, args.output)

    # Camera metadata
    print("📸 Saving intrinsics and extrinsics...")
    save_intrinsics(prediction.intrinsics, args.output)
    save_extrinsics(prediction.extrinsics, args.output)

    print("✅ DA3 pipeline completed successfully.")

    # Convert all exported GLBs → PLY automatically
    glb_files = sorted(glob.glob(os.path.join(args.output, "*.glb")))
    ply_out_dir = os.path.join(args.output, "ply_clean")
    ensure_dir(ply_out_dir)

    print("🧼 Converting GLB files to clean PLY format...")

    for glb in glb_files:
        name = os.path.splitext(os.path.basename(glb))[0]
        ply_out = os.path.join(ply_out_dir, f"{name}.ply")
        glb_to_ply(glb, ply_out)

    print("✨ All GLBs converted to clean PLYs.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Depth Anything 3 Pipeline")

    parser.add_argument("--input", "-i", type=str, required=True, help="Input image directory")
    parser.add_argument("--output", "-o", type=str, required=True, help="Output directory")
    parser.add_argument("--model", "-m", type=str, default="depth-anything/da3-base", help="Model name")
    parser.add_argument("--format", "-f", type=str, default="glb",
                        choices=["glb", "npz", "mini_npz", "ply", "gs_ply", "gs_video"],
                        help="Export format")
    parser.add_argument("--resolution", "-r", type=int, default=504, help="Processing resolution")

    args = parser.parse_args()
    main(args)
