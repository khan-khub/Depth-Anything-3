import open3d as o3d
import numpy as np

def load_ply(path):
    pcd = o3d.io.read_point_cloud(path)
    pts = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors) if pcd.has_colors() else None
    return pts, colors

file1 = "/data/da3_output/scene_clean.ply"
file2 = "/data/scene.ply"

pts1, col1 = load_ply(file1)
pts2, col2 = load_ply(file2)

print("----- POINT COUNTS -----")
print("File 1:", pts1.shape[0])
print("File 2:", pts2.shape[0])

print("\n----- COLOR PRESENCE -----")
print("File 1 colors:", col1 is not None)
print("File 2 colors:", col2 is not None)

# If point counts differ, stop early
if pts1.shape[0] != pts2.shape[0]:
    print("\n❌ Point count differs. Cannot directly compare positions.")
else:
    print("\n----- POSITION DIFFERENCES -----")

    # compute per-point L2 distance
    diff = np.linalg.norm(pts1 - pts2, axis=1)

    print("Mean difference:", diff.mean())
    print("Max difference: ", diff.max())
    print("Min difference: ", diff.min())

    if diff.max() < 1e-4:
        print("✅ The two PLY files contain identical point clouds.")
    else:
        print("⚠️ Point clouds differ.")

print("\n----- OPTIONAL: COLOR DIFFERENCES -----")
if (col1 is not None) and (col2 is not None):
    col_diff = np.linalg.norm(col1 - col2, axis=1)
    print("Mean color diff:", col_diff.mean())
    print("Max color diff:", col_diff.max())
else:
    print("At least one file has no colors.")
