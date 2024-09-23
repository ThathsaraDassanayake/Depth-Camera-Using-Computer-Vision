import numpy as np
import cv2 as cv
import glob
#import matplotlib.pyplot as plt

def calibrate_camera(images):
    
        ################ FIND CHESSBOARD CORNERS - OBJECT POINTS AND IMAGE POINTS #############################

        chessboardSize = (9,6)
        frameSize = (1920,1080)


        # termination criteria
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)


        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
        objp[:,:2] = np.mgrid[0:chessboardSize[0],0:chessboardSize[1]].T.reshape(-1,2)

        size_of_chessboard_squares_mm = 20
        objp = objp * size_of_chessboard_squares_mm


        # Arrays to store object points and image points from all the images.
        objpoints = [] # 3d point in real world space
        imgpoints = [] # 2d points in image plane.


        #images = glob.glob('cameraCalibration/images/*.png')

        for image in images:

            img = cv.imread(image)
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

            # Find the chess board corners
            ret, corners = cv.findChessboardCorners(gray, chessboardSize, None)

            # If found, add object points, image points (after refining them)
            if ret == True:

                objpoints.append(objp)
                corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
                imgpoints.append(corners)

                # Draw and display the corners
                cv.drawChessboardCorners(img, chessboardSize, corners2, ret)
                cv.imshow('img', img)
                cv.waitKey(1000)


        cv.destroyAllWindows()


        ############## CALIBRATION PROCESS ################

        ret, cameraMatrix, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, frameSize, None, None)

        print(ret)
        print(cameraMatrix)

        return cameraMatrix, dist 

def stereo_calibrate(mtx1, dist1, mtx2, dist2, c1_images1, c2_images1):
    #read the synched frames

    c1_images = []
    c2_images = []
    for im1 in c1_images1:
        _im = cv.imread(im1)
        if _im is None:
            print(f"Failed to load image: {im1}")
            continue
        c1_images.append(_im)

    for im2 in c2_images1:
        _im = cv.imread(im2)
        if _im is None:
            print(f"Failed to load image: {im2}")
            continue
        c2_images.append(_im)

    #change this if stereo calibration not good.
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
 
    rows = 9 #number of checkerboard rows.
    columns = 6 #number of checkerboard columns.
    world_scaling = 20. #change this to the real world square size. Or not.
 
    #coordinates of squares in the checkerboard world space
    objp = np.zeros((rows*columns,3), np.float32)
    objp[:,:2] = np.mgrid[0:rows,0:columns].T.reshape(-1,2)
    objp = world_scaling* objp
 
    #frame dimensions. Frames should be the same size.
    frameSize = (4032,3024)
 
    #Pixel coordinates of checkerboards
    imgpoints_left = [] # 2d points in image plane.
    imgpoints_right = []
 
    #coordinates of the checkerboard in checkerboard world space.
    objpoints = [] # 3d point in real world space
 
    for frame1, frame2 in zip(c1_images, c2_images):
        gray1 = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
        gray2 = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
        c_ret1, corners1 = cv.findChessboardCorners(gray1, (9, 6), None)
        c_ret2, corners2 = cv.findChessboardCorners(gray2, (9, 6), None)
 
        if c_ret1 == True and c_ret2 == True:
            corners1 = cv.cornerSubPix(gray1, corners1, (11, 11), (-1, -1), criteria)
            corners2 = cv.cornerSubPix(gray2, corners2, (11, 11), (-1, -1), criteria)
 
            cv.drawChessboardCorners(frame1, (9,6), corners1, c_ret1)
            cv.imshow('img', frame1)
 
            cv.drawChessboardCorners(frame2, (9,6), corners2, c_ret2)
            cv.imshow('img2', frame2)
            cv.waitKey(1000)
 
            objpoints.append(objp)
            imgpoints_left.append(corners1)
            imgpoints_right.append(corners2)

    print(f"Number of valid calibration pairs: {len(objpoints)}")
    if len(objpoints) == 0:
        raise Exception("No valid image points found for stereo calibration. Check image paths and checkerboard detection.")
    stereocalibration_flags = cv.CALIB_FIX_INTRINSIC
    ret, CM1, dist1, CM2, dist2, R, T, E, F = cv.stereoCalibrate(objpoints, imgpoints_left, imgpoints_right, mtx1, dist1,
                                                                 mtx2, dist2, frameSize, criteria = criteria, flags = stereocalibration_flags)
 
    print(ret)
    return R, T



# images = glob.glob('cameraCalibration/images/*.png')
# mtx, dist = calibrate_camera(images)

c1_images = glob.glob("CameraCalibration/images2/*.png")
c1_images = sorted(c1_images)
c2_images = glob.glob("CameraCalibration/images1/*.png")
c2_images = sorted(c2_images)

mtx1, dist1 = calibrate_camera(c1_images)
mtx2, dist2 = calibrate_camera(c2_images)

R, T = stereo_calibrate(mtx1, dist1, mtx2, dist2, c1_images, c2_images)

# print("R =", R)
# print("T =",T)

# ###################################################################################################################################

# #img1 = glob.glob("CameraCalibration/left.jpg")
# #img2 = glob.glob("CameraCalibration/right.jpg")

# img1 = cv.imread("CameraCalibration/left.jpg", cv.IMREAD_GRAYSCALE)  # Left image
# img2 = cv.imread("CameraCalibration/right.jpg", cv.IMREAD_GRAYSCALE)  # Right image

# #R1, R2, P1, P2, Q, roi1, roi2 = cv.stereoRectify(mtx1, dist1, mtx2, dist2, img1.shape[::-1], R, T)

# P1 = np.hstack((mtx1, np.zeros((3, 1))))  # Projection matrix for the first camera
# P2 = mtx2 @ np.hstack((R, T.reshape(-1, 1)))  # Projection matrix for the second camera


# #map1x, map1y = cv.initUndistortRectifyMap(mtx1, dist1, R1, P1, img1.shape[::-1], cv.CV_32FC1)
# #map2x, map2y = cv.initUndistortRectifyMap(mtx2, dist2, R2, P2, img2.shape[::-1], cv.CV_32FC1)

# #rectified_img1 = cv.remap(img1, map1x, map1y, cv.INTER_LINEAR)
# #rectified_img2 = cv.remap(img2, map2x, map2y, cv.INTER_LINEAR)

# sift = cv.SIFT_create()
# kp1, descriptor1 = sift.detectAndCompute(img1,None)
# kp2, descriptor2 = sift.detectAndCompute(img2,None)
# bf = cv.BFMatcher()
# nNeighbors = 2 

# # this will take awhile....(pause video)
# matches = bf.knnMatch(descriptor1,descriptor2,k=nNeighbors)
# goodMatches = []
# #testRatio = 0.75 # reference Lowe's paper 
# #for m,n in matches: # m and n are the 2 neighbors 
#  #   if m.distance < testRatio*n.distance:
#   #      goodMatches.append(m)

# imgMatch = cv.drawMatchesKnn(img1,kp1,img2,kp2,goodMatches,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# cv.imshow("Matched Features", imgMatch)
# cv.waitKey(1000)

# good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]
# pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
# pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

# # Triangulate points
# points4D = cv.triangulatePoints(P1, P2, pts1, pts2)
# points3D = points4D[:3] / points4D[3]
# depths = points3D[2, :]  # Extract Z (depth information)

# # Optional: Analyze depth information
# print("Depth statistics:", np.mean(depths), np.std(depths))