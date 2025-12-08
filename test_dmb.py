import numpy as np
import matplotlib.pyplot as plt

import numpy as np

def loadDMB(dmb_path):
    """
    Reads a .dmb file and returns the depth map as a NumPy array.
    
    The format expected is:
    - 4 bytes for type (uint)
    - 8 bytes for height, width (uint x2)
    - 4 * width * height bytes for depth map data (float)
    - additional metadata strings
    """
    with open(dmb_path, 'rb') as dmb:
        # Read type (usually an identifier)
        type_val = np.frombuffer(dmb.read(4), dtype=np.dtype('I'))[0]
        
        # Read height and width
        height, width = np.frombuffer(dmb.read(8), dtype=np.dtype('I'))
        
        # Read the raw depth data into a 1D float array
        depth_data_1d = np.frombuffer(dmb.read(4 * width * height), dtype=np.dtype('f'))
        
        # Reshape to the actual 2D depth map
        depth_map = depth_data_1d.reshape(height, width)
        
        # (Optional) Read and parse calibration data if present in your specific file
        # This part of the code is commented out as it might differ based on your file's specific structure.
        # If your file is a simple depth map, the above lines are sufficient.
        
        # string_length = np.frombuffer(dmb.read(4), dtype=np.dtype('I'))[0]
        # calibration_string = dmb.read(string_length).decode().replace(' \n', '\n').split('\n')
        
    return depth_map

# Example usage:
dmb_path = "/workspace/dep2/00000000.dmb"
file_path = dmb_path
depth_image_array = loadDMB(file_path)

# You can then use libraries like OpenCV or Matplotlib to visualize or process the array:
import cv2
cv2.imwrite('depth.png', depth_image_array)
