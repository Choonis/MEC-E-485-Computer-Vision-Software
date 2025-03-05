
from tqdm import tqdm
import cv2 as cv
import numpy as np
from glob import glob
import os
from synthetic_depthmap_generation import *

##################################################################################################
## PURPOSE:
##################################################################################################
# This script generates a CSV containing filename and camera transformation matrix pairs 
# to pass to Blender for synthetic depthmap rendering. 

if __name__ == '__main__':
    ##################################################################################################
    ## INPUTS:
    ##################################################################################################
    # The first input for this script is the path to the directory containing the raw data from
    # the Auxilary Capture application. For each capture, the dataset should contain:
    # - an RGB image, 
    # - 15 depth maps, 
    # - cameraProperties table, 
    # - lens distortion lookup table, 
    aux_capture_data_path = r"/Users/connorpovoledo/Library/CloudStorage/GoogleDrive-connorpovoledo@wound3.com/Shared drives/1. Wound3/Engineering/Code/Backend/Python/R&D/Built Datasets/Depth Completion - ArUco x Blender Datasets/Test Set 02 - iPhone 15 Pro/Gridboard Index 14"

    # The second set of inputs is for the gridboard configuration

    # Real-world origin location (x, y, z from origin to obj point 1 in Fusion 360 coordinates)
    ArUcoOriginWorldCoordinates = [0.02405, 0.342025, 0.00635]

    # Define the shape of the grid (LASER CUT SETTINGS)
    numberOfTags = 5            # the gridboard will be NxN in shape
    markerLength = 0.040        # size in meters
    markerSeperation = 0.029493344
    worldGridSize = numberOfTags*markerLength+(numberOfTags-1)*markerSeperation
    gridIds = np.array([(4,  5,  6,  7,  8), 
                        (9,  20, 21, 22, 10),
                        (11, 23, 24, 25, 12),
                        (13, 26, 27, 28, 14),
                        (15, 17, 18, 19, 16)]).flatten()

    ##################################################################################################
    ##################################################################################################
    ## CODE:
    ##################################################################################################
    ##################################################################################################

    # Get all of the data from aux caputre
    image_paths = sorted(glob(os.path.join(aux_capture_data_path, "image*")))
    depth_map_paths = sorted(glob(os.path.join(aux_capture_data_path, "depthMap*")))
    camera_properties_paths = sorted(glob(os.path.join(aux_capture_data_path, "cameraProperties*")))
    distortion_table_paths = sorted(glob(os.path.join(aux_capture_data_path, "lensDistortionLookupTable*")))

    # Ensure that there are the correct number of files
    n = len(image_paths)
    if 15*n != len(depth_map_paths):
        print("Incorrect number of images and depth maps")
        exit()
    elif n != len(camera_properties_paths):
        print(f"Incorrect number of camera properties csvs, found: {len(camera_properties_paths)}, expected: {n}")
        exit()
    elif n != len(distortion_table_paths):
        print("Incorrect number of distortion tables")
        exit()

    # Setup a path for output
    output_transforms_path = os.path.join(aux_capture_data_path, "blender_transforms.npy")
    output_focals_path = os.path.join(aux_capture_data_path, "blender_focals.npy")
    output_transforms_array = np.zeros((n,4,4))
    output_focals_array = np.zeros((n))

    ##################################################################################################
    ## Load the dictionary for which the markers will be based
    ##################################################################################################

    dictionary = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_50)
    parameters = cv.aruco.DetectorParameters()
    parameters.polygonalApproxAccuracyRate = 0.003
    # parameters.cornerRefinementMethod = cv.aruco.CORNER_REFINE_APRILTAG
    detector = cv.aruco.ArucoDetector(dictionary, parameters)

    ##################################################################################################
    ## Generate Gridboard
    ##################################################################################################

    # Instantiate the grid
    gridSize = (numberOfTags, numberOfTags)
    grid = cv.aruco.GridBoard(gridSize, markerLength, markerSeperation, dictionary, gridIds)

    # Output grid parameters
    print(" ")
    print("GRID PARAMETERS")
    print("Grid size: {} m".format(worldGridSize))
    print("Numer of markers: {} x {}".format(numberOfTags, numberOfTags))
    print("Marker seperation: {}".format(markerSeperation))
    print("Grid Ids: {}".format(gridIds))
    print(" ")

    ##################################################################################################
    # Loop through each example to find the camera transform that will generate synthetic
    # variations of each sample
    ##################################################################################################
    for i in tqdm (range(len(image_paths))):

        # Load image
        image = cv.imread(image_paths[i])

        # Load camera properties
        cameraPropertiesCSV = camera_properties_paths[i]
        fx, fy, cx, cy, dx, dy, referenceWidth, referenceHeight, pixelsToMMRatio = loadCameraProperties(cameraPropertiesCSV)

        # Find the focal length - NOTE: this formula assumes that the camera uses square pixels, thus fx=fy
        focal_length = (fx+fy)/2

        # Apply parameters to create intrinsic matrix
        mtx = np.array([[fy, 0, cy], [0, fx, cx], [0, 0, 1]])

        # Rectification is applied to image in subsiquent steps, so this is zero
        dist = np.zeros(5)

        # Load the distortionLookup table and inverseDistortionLookupTable
        lensDistortionLookupTablePath = distortion_table_paths[i]
        lensDistortionLookupTable = np.fromfile(lensDistortionLookupTablePath, dtype=np.float32)
        
        # Rectify the image
        image = undistortImage(image, lensDistortionLookupTable, Point(dx, dy))

        ##################################################################################################
        ## Find the camera pose and output it for blender processing
        ##################################################################################################

        rvec, tvec = findTagsInGrid(image, grid, detector, mtx, dist, worldGridSize)
        blender_matrix = convertTransformsToBlender(rvec, tvec, ArUcoOriginWorldCoordinates)

        output_transforms_array[i,:,:] = blender_matrix
        output_focals_array[i] = focal_length
        
    # Write the master numpy array to a csv
    np.save(output_transforms_path, output_transforms_array)
    np.save(output_focals_path, output_focals_array)