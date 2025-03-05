
# Guard the code so multiprocessing does not fail
if __name__ == '__main__':

    ## PURPOSE:
    # This script executes postprocessing computations for synthetic depthmap datasets 
    # produced by blender

    ## INPUTS: 
    # The only input requried by this script is the path to the root directory containing
    # the dataset.
    input_dir = r"/Users/connorpovoledo/Downloads/Test Set 02 - iPhone 15 Pro/Gridboard Index 12"

    ## CODE: 
    import os
    import numpy as np
    os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
    import cv2 as cv
    from glob import glob
    from synthetic_depthmap_generation import *
    from tqdm import tqdm

    # Parse files for relivant paths
    camera_properties_paths = sorted(glob(os.path.join(input_dir, "cameraProperties*")))
    exr_paths = sorted(glob(os.path.join(input_dir, "rendered*")))
    inverse_distortion_table_paths = sorted(glob(os.path.join(input_dir, "inverseLensDistortionLookupTable*")))

    # Check to ensure the number of samples is correct
    if len(camera_properties_paths) != len(exr_paths):
        print(f"Incorrect number of EXRs, expected: {len(camera_properties_paths)}, recieved: {len(exr_paths)}")
        exit()
    elif len(exr_paths) != len(inverse_distortion_table_paths):
        print(f"Incorrect number of data samples; EXRs: {len(exr_paths)}, Distortion Tables: {len(inverse_distortion_table_paths)}")
        exit()

    # Create the filesave convention
    output_fname_image = "synthetic_image_"
    output_fname_depth_map = "synthetic_depth_map_"

    # Post process each of the files
    for i in tqdm(range(len(exr_paths))):

        # Load the camera properties
        fx, fy, cx, cy, dx, dy, referenceWidth, referenceHeight, pixelsToMMRatio = loadCameraProperties(camera_properties_paths[i])

        # Load the EXR file output by blender
        rgb_image, depth_image = load_exr(exr_paths[i])

        # Find the scale ratio to downsize the camera intrinsic properties
        scale = depth_image.shape[0] / referenceWidth
        dx = dx * scale
        dy = dy * scale

        # Load the inverse distortion table
        inverse_distortion_table = np.fromfile(inverse_distortion_table_paths[i], dtype=np.float32)

        # Distort the synthetic depth map, but skip finding the magnification at all points
        output_depth_map, xpoints, ypoints = undistortImage(depth_image, inverse_distortion_table, Point(dx, dy), return_points = True)

        # Convert floating-point RGB representation to uint representation
        rgb_image_normalized = (rgb_image - rgb_image.min()) / (rgb_image.max() - rgb_image.min())
        output_image = cv.cvtColor((rgb_image_normalized * 255).astype('uint8'), cv.COLOR_BGR2RGB)
        output_image = undistortImage(output_image, inverse_distortion_table, Point(dx, dy), xpoints=xpoints, ypoints=ypoints)

        # Save the image and depth map
        cv.imwrite(os.path.join(input_dir, output_fname_image + str(i+1).zfill(4) + ".png"), output_image)
        np.save(os.path.join(input_dir, output_fname_depth_map + str(i+1).zfill(4) + ".npy"), output_depth_map)
