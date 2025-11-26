from pygltflib import GLTF2
import numpy as np
import open3d as o3d

glb_path = "/data/da3_output/scene.glb"
ply_path = "/data/da3_output/scene_clean.ply"

gltf = GLTF2().load(glb_path)
buffer = gltf.binary_blob()

def read_accessor(acc):
    view = gltf.bufferViews[acc.bufferView]
    start = view.byteOffset + acc.byteOffset
    dtype_map = {5126: np.float32, 5123: np.uint16, 5121: np.uint8}
    dtype = dtype_map[acc.componentType]
    type_map = {"SCALAR":1, "VEC2":2, "VEC3":3, "VEC4":4}
    ncomp = type_map[acc.type]

    stride = view.byteStride or ncomp * np.dtype(dtype).itemsize

    arr = np.zeros((acc.count, ncomp), dtype=dtype)
    for i in range(acc.count):
        i0 = start + i * stride
        raw = buffer[i0:i0 + ncomp * dtype().nbytes]
        arr[i] = np.frombuffer(raw, dtype=dtype, count=ncomp)
    return arr.astype(np.float32)

all_pts = []
all_cols = []

for mesh in gltf.meshes:
    for prim in mesh.primitives:

        # Skip if no positions found
        if prim.attributes.POSITION is None:
            continue

        pts = read_accessor(gltf.accessors[prim.attributes.POSITION])

        # Skip camera nodes (small clusters)
        if pts.shape[0] < 400:      # cameras ~50–200 points
            continue

        # Keep only scene data
        all_pts.append(pts)

        # Colors if exist
        if prim.attributes.COLOR_0 is not None:
            cols = read_accessor(gltf.accessors[prim.attributes.COLOR_0])
            if cols.max() > 1.5:  
                cols /= 255.0
            if cols.shape[1] == 4:
                cols = cols[:, :3]
            all_cols.append(cols)
        else:
            all_cols.append(None)

# Merge everything
points = np.vstack(all_pts)

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)

# Merge colors if available
if any(c is not None for c in all_cols):
    colors = np.vstack([c for c in all_cols if c is not None])
    pcd.colors = o3d.utility.Vector3dVector(colors)

# Save clean scene
o3d.io.write_point_cloud(ply_path, pcd)

print("Saved:", ply_path)
print("Scene points:", len(points))
