
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from glob import glob
import os
from stl import mesh
import csv
import OpenEXR
import Imath
from multiprocessing import Pool


def loadCameraProperties(cameraPropertiesCSV):
    cameraProperties = {}

    # Open the CSV containing the camera properties
    with open(cameraPropertiesCSV, "r") as csvFile:
        reader = csv.reader(csvFile, delimiter=",")
        next(reader)

        # Loop through the rows to isolate the properties
        for row in reader:
            if not row == []:
                # Append the properties to the master dictionary
                property = row[0]
                if not property == 'Device':
                    value = float(row[1])
                else: 
                    value = row[1]
                cameraProperties[property] = value

    # Extract the image parameters: 
    fx = cameraProperties['Intrinsic Matrix [FocalX]']
    fy = cameraProperties['Intrinsic Matrix [FocalY]']
    cx = cameraProperties['Intrinsic Matrix [PrincipalPointX]']
    cy = cameraProperties['Intrinsic Matrix [PrincipalPointY]']
    dx = cameraProperties['Distortion Center [X]']
    dy = cameraProperties['Distortion Center [Y]']
    referenceWidth = cameraProperties['Reference Dimensions [Width]']
    referenceHeight = cameraProperties['Reference Dimensions [Height]']
    pixelsToMMRatio = cameraProperties['Pixel Size']

    return fx, fy, cx, cy, dx, dy, referenceWidth, referenceHeight, pixelsToMMRatio


def readLookupTableFromCSV(path):
    values = []
    
    with open(path, "r") as csvFile:
        reader = csv.reader(csvFile, delimiter=",")
        next(reader)
        for row in reader:
            values.append(np.float32(row[0]))

    return np.array(values)


class Point:
    def __init__(self, x, y) -> None:
        self.x = x
        self.y = y

class Size: 
    def __init__(self, width, height) -> None:
        self.width = width
        self.height = height

def rectifyPoint(point: Point, lookupTable: np.ndarray, opticalCenter: Point, imageSize: Size):
    
    # Determine the maximum radius 
    delta_ocx_max = max(opticalCenter.x, imageSize.width - opticalCenter.x)
    delta_ocy_max = max(opticalCenter.y, imageSize.height - opticalCenter.y)
    r_max = np.sqrt( delta_ocx_max**2 + delta_ocy_max**2 )

    # Determine the vector from the opical center to the given point
    v_point_x = point.x - opticalCenter.x
    v_point_y = point.y - opticalCenter.y

    # Determine the radius at the point
    r_point = np.sqrt( v_point_x**2 + v_point_y**2)

    # Look up the relative radial magnification to apply in the provided lookup table
    lookupTableCount = lookupTable.shape[0]

    # Linear interpolation
    if r_point < r_max: 
        val = r_point * (lookupTableCount - 1) / r_max
        idx = int(val)
        frac = val - idx

        mag_1 = lookupTable[idx]
        mag_2 = lookupTable[idx + 1]

        magnification = (1 - frac)*mag_1 + frac*mag_2
    
    else:
        magnification = lookupTable[lookupTableCount-1]

    # Apply radial magnification
    new_v_point_x = v_point_x + magnification*v_point_x
    new_v_point_y = v_point_y + magnification*v_point_y

    return Point(opticalCenter.x + new_v_point_x, opticalCenter.y + new_v_point_y)


def compute_segment(y_start, y_end, width, lensDistortionLookupTable, opticalCenter, imageSize):
    xpoints_segment = np.empty((y_end - y_start, width), dtype=np.float32)
    ypoints_segment = np.empty((y_end - y_start, width), dtype=np.float32)
    for y in range(y_start, y_end):
        for x in range(width):
            inputPoint = Point(x, y)
            transformedPoint = rectifyPoint(inputPoint, lensDistortionLookupTable, opticalCenter, imageSize)
            xpoints_segment[y - y_start, x] = transformedPoint.x
            ypoints_segment[y - y_start, x] = transformedPoint.y
    return xpoints_segment, ypoints_segment


def undistortImage(image: np.ndarray, 
                   lensDistortionLookupTable: np.ndarray, 
                   distortionCenter: Point, 
                   return_points = False, 
                   xpoints: np.ndarray = np.zeros((1)), 
                   ypoints: np.ndarray = np.zeros((1))):
    
    height, width = image.shape[:2]
    imageSize = Size(width, height)

    # Determine the optical center based on image orientation
    opticalCenter = Point(distortionCenter.x, distortionCenter.y) if width > height else Point(distortionCenter.y, distortionCenter.x)

    if (xpoints == np.zeros((1))).all() or (ypoints == np.zeros((1))).all():
        # Number of processes to use
        num_processes = os.cpu_count()
        pool = Pool(processes=num_processes)

        # Calculate segment size. This time, account for any remainder.
        segment_height = height // num_processes
        remainder = height % num_processes

        tasks = []
        start_y = 0
        for i in range(num_processes):
            # Distribute the remainder among the first 'remainder' segments
            end_y = start_y + segment_height + (1 if i < remainder else 0)
            tasks.append((start_y, min(end_y, height), width, lensDistortionLookupTable, opticalCenter, imageSize))
            start_y = end_y

        # Process each segment in parallel
        results = pool.starmap(compute_segment, tasks)
        pool.close()
        pool.join()

        # Combine the results
        xpoints = np.vstack([result[0] for result in results])
        ypoints = np.vstack([result[1] for result in results])

    # Map the image to the rectified image space
    if return_points: 
        return cv.remap(image, xpoints, ypoints, cv.INTER_LANCZOS4), xpoints, ypoints
    else: 
        return cv.remap(image, xpoints, ypoints, cv.INTER_LANCZOS4)


def undistortImageChronological(image: np.ndarray, 
                   lensDistortionLookupTable: np.ndarray, 
                   distortionCenter: Point, 
                   return_points = False, 
                   xpoints: np.ndarray = np.zeros((1)), 
                   ypoints: np.ndarray = np.zeros((1))):
    
    # Get image parameters
    width = image.shape[1]
    height = image.shape[0]
    imageSize = Size(width, height)

    # Ensure image properties are correct relative to the orientation of the image
    if width > height:
        opticalCenter = Point(distortionCenter.x, distortionCenter.y)
    else:
        opticalCenter = Point(distortionCenter.y, distortionCenter.x)

    # Save the transformed positions of all of the x and y points
    if (xpoints == np.zeros((1))).all() or (ypoints == np.zeros((1))).all():
        xpoints = np.empty((height,width), dtype=np.float32)
        ypoints = np.empty((height,width), dtype=np.float32)

        for y in range(height):
            for x in range(width): 
                inputPoint = Point(x, y)
                transformedPoint = rectifyPoint(inputPoint, lensDistortionLookupTable, opticalCenter, imageSize)
                xpoints[y,x] = transformedPoint.x
                ypoints[y,x] = transformedPoint.y

    # Map the image to the rectified image space
    if return_points: 
        return cv.remap(image, xpoints, ypoints, cv.INTER_LANCZOS4), xpoints, ypoints
    else: 
        return cv.remap(image, xpoints, ypoints, cv.INTER_LANCZOS4)


def findTagsInGrid(imageOfGrid, grid, detector, mtx, dist, axisLength):

    ## Detect tags on grid test
    corners, ids, rejected = detector.detectMarkers(cv.cvtColor(imageOfGrid, cv.COLOR_BGR2GRAY))

    if len(corners) > 0:
        
        # allocate space for master id and corner arrays for later use
        all_ids = ids.copy()
        all_corners = np.empty((len(ids),4,2), dtype=np.float32)

        # flatten the ArUco IDs list
        ids = ids.flatten()

        # loop over the detected ArUCo corners
        for i,(markerCorner, markerID) in enumerate(zip(corners, ids)):
            # extract the marker corners (which are always returned in
            # top-left, top-right, bottom-right, and bottom-left order)
            corners = markerCorner.reshape((4, 2))
            all_corners[i,:,:] = corners

        # Solve for the position of the board
        matchedObjPoints, matchedImgPoints = grid.matchImagePoints(all_corners, all_ids)
        ret, rvec, tvec = cv.solvePnP(matchedObjPoints, matchedImgPoints, mtx, dist)
        
        if ret: 
            return rvec, tvec

        else: 
            print("SolvePnP failed.")
            return
        

def convertTransformsToBlender(rvec, tvec, worldAxisCoordinate):

    # Convert axis from obj point roation relative to camera, to camera pos. relative to obj points.
    R, _ = cv.Rodrigues(rvec)
    R_inv = R.T                 # Transpose is the same as an inversion as R is symmetric

    ## FIND THE TRANSLATION VECTOR
    # Invert the translation vector so the transform represents the camera relative to obj. points instead of obj. points relative to camera
    tvec_inv = -np.dot(R_inv, tvec)

    # Compensate for the real-world coordinates of the ArUco origin in the tvec
    T_adjusted = [tvec_inv[0] + worldAxisCoordinate[0],     # x_blender = x + width_of_gridboard_padding
                  worldAxisCoordinate[1] - tvec_inv[1],     # y_blender = length_of_gridboard_padding - y
                  worldAxisCoordinate[2] - tvec_inv[2]]     # z_blender = height_of_gridboard - z
    T_adjusted = np.array(T_adjusted)

    ## FIND THE ROTATION VECTOR
    # Transform the blender coordinate system to the ArUco Gridboard Coordinate System
    blenderDatumnToGridboardDatumn = np.array([[1,  0,  0], 
                                               [0, -1,  0], 
                                               [0,  0, -1]], dtype=np.float64)
    # Transform the OpenCV camera coordinate system to the Blender camera coordinate system
    openCVCameraDatumnToBlenderCameraDatumn = np.array([[1,  0,  0], 
                                                        [0, -1,  0], 
                                                        [0,  0, -1]], dtype=np.float64)
    # First, transform the blender coordinates to ArUco coordinates, then, apply the inverse rotation to 
    # map the ArUco grid space to the openCV camera space, then finally convert from openCV camera space to 
    # blender camera space
    R_adjusted = openCVCameraDatumnToBlenderCameraDatumn @ R_inv @ blenderDatumnToGridboardDatumn

    # Create a 4x4 transformation matrix with inverted rotation and translation
    blender_matrix = np.zeros((4, 4))
    blender_matrix[:3, :3] = R_adjusted
    blender_matrix[:3, 3] = T_adjusted.flatten()
    blender_matrix[3, 3] = 1.0
    
    return blender_matrix


def load_exr(exr_path):
    # Open the EXR file
    exr_file = OpenEXR.InputFile(exr_path)

    # Get the size of the image
    dw = exr_file.header()['dataWindow']
    width, height = dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1

    # Specify the pixel type
    pt = Imath.PixelType(Imath.PixelType.FLOAT)

    # Read the RGB channels
    rgb_str = [exr_file.channel(c, pt) for c in ('R', 'G', 'B')]
    # Convert the strings to NumPy arrays
    rgb_arrays = [np.frombuffer(c, dtype=np.float32) for c in rgb_str]
    # Stack the arrays into a single multi-channel array (height, width, channels)
    rgb_image = np.stack(rgb_arrays, axis=-1).reshape(height, width, 3)

    # Read the depth channel
    depth_str = exr_file.channel('Z', pt)
    # Convert the string to a NumPy array
    depth_array = np.frombuffer(depth_str, dtype=np.float32)
    # Reshape the array to the image dimensions (height, width)
    depth_image = depth_array.reshape(height, width)

    return rgb_image, depth_image