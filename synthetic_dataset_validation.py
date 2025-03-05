
## PURPOSE:
# This script allows the user to parse through a dataset to generate overlays and ensure that 
# synthetic data is generated properly. The synthetic/truedepth image and depthmap pairs are 
# all imported, then overlaid in a matplotlib window. 

## INPUTS:
# The only first for this script is the root folder for the dataset
root = r"/Users/connorpovoledo/Downloads/Test Set 02 - iPhone 15 Pro/Gridboard Index 1"
# The second identifies if the top left image in the output window should be masked (to visualize the quality of the blender mask)
should_mask = True

## CODE: 
import cv2 as cv
import os
from glob import glob
import matplotlib.pyplot as plt
import numpy as np

# Import the image paths
truedepth_image_paths = sorted(glob(os.path.join(root, "image*")))
synthetic_image_paths = sorted(glob(os.path.join(root, "synthetic_image*")))
truedepth_depth_map_paths = sorted(glob(os.path.join(root, "depthMap*")))
synthetic_depth_map_paths = sorted(glob(os.path.join(root, "synthetic_depth_map*")))
synthetic_mask_paths = sorted(glob(os.path.join(root, "mask*")))

# Ensure the proper number of data samples are imported
n = len(truedepth_image_paths)
if n != len(synthetic_image_paths):
    print(f"Wrong number of synthetic images detected. Found: {len(synthetic_image_paths)}, expected: {n}")
    exit()
elif 15*n != len(truedepth_depth_map_paths):
    print(f"Wrong number of truedepth depth maps detected. Found: {len(truedepth_depth_map_paths)}, expected: {15*n}")
    exit()
elif n != len(synthetic_depth_map_paths):
    print(f"Wrong number of synthetic depth maps detected. Found: {len(synthetic_depth_map_paths)}, expected: {n}")
    exit()
elif n != len(synthetic_mask_paths):
    print(f"Wrong number of masks detected. Found: {len(synthetic_mask_paths)}, expected: {n}")
    exit()

# Define a waitkey function for matplotlib
def on_key(event):
    print(f"You pressed {event.key}")
    if event.key:
        plt.close(event.canvas.figure)

# Loop through each example and display the results
for i in range(n):
    # Import the images
    truedepth_image = cv.imread(truedepth_image_paths[i])
    synthetic_image = cv.imread(synthetic_image_paths[i])
    mask = cv.imread(synthetic_mask_paths[i])
    
    # Import the truedepth depth map
    truedepth_depth_map_path = truedepth_depth_map_paths[i*15]
    data_int32 = np.fromfile(truedepth_depth_map_path, dtype=np.int32)
    data_float32 = np.fromfile(truedepth_depth_map_path, dtype=np.float32)
    bytesPerRow = int(data_int32[0]/4)
    rows = int(data_int32[1])
    truedepth_depth_map = np.reshape(data_float32[2:],(rows,bytesPerRow))
    truedepth_depth_map = cv.rotate(truedepth_depth_map, cv.ROTATE_90_CLOCKWISE)

    # Import the blender depth map
    synthetic_depth_map = np.load(synthetic_depth_map_paths[i])

    # Downsize the truedepth image
    truedepth_image = cv.resize(truedepth_image, (480, 640))

    # Create the overlays
    depth_overlay = truedepth_depth_map + synthetic_depth_map
    image_overlay = cv.addWeighted(truedepth_image, 0.5, synthetic_image, 0.5, 0)

    # Display the results
    fig = plt.figure(figsize=(10, 10))
    plt.title(f"Sample: {i}")
    plt.axis('off')
    columns = 3
    rows = 2 

    plot_image = cv.cvtColor(truedepth_image, cv.COLOR_BGR2RGB)
    if should_mask:
        plot_image = cv.bitwise_and(mask, plot_image)
    fig.add_subplot(rows, columns, 1)
    plt.axis('off')
    plt.imshow(plot_image)

    fig.add_subplot(rows, columns, 2)
    plt.axis('off')
    plt.imshow(cv.cvtColor(synthetic_image, cv.COLOR_BGR2RGB))

    fig.add_subplot(rows, columns, 3)
    plt.axis('off')
    plt.imshow(cv.cvtColor(image_overlay, cv.COLOR_BGR2RGB))

    fig.add_subplot(rows, columns, 4)
    plt.axis('off')
    plt.imshow(truedepth_depth_map)

    fig.add_subplot(rows, columns, 5)
    plt.axis('off')
    plt.imshow(synthetic_depth_map)

    fig.add_subplot(rows, columns, 6)
    plt.axis('off')
    plt.imshow(depth_overlay)

    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show()
