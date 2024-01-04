import numpy as np
import cv2 as cv
import glob
import numpy as np

import matplotlib.pyplot as plt

''' This part below was used to captures synchronous frames from two devices, which
are stored in folders frames and frames2 '''

# cap = cv2.VideoCapture(0) # video capture source camera (here webcam of laptop)
# num = 0

# while cap.isOpened():

#     succes, img = cap.read()
#     k = cv2.waitKey(3)

#     if k == 30:
#         break
#     elif k == ord('s'): # wait for 's' key to save and exit
#         cv2.imwrite('frames/img' + str(num) + '.png', img)
#         print("image has been saved")
#         num += 1
#     cv2.imshow('webcam', img)

# cap.release()
# cv2.destroyAllWindows()

''' Find cheesboard corners for both webcam snaphots '''

chessboardSize = (10, 7) # Number of inner corners per a chessboard width and height
frameSize = (640, 480) # Resolution of images

# Termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
size_of_square = 22 #in mm

# Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
objp[:,:2] = np.mgrid[0:chessboardSize[0], 0:chessboardSize[1]].T.reshape(-1, 2)*size_of_square

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

images = glob.glob('frames/*.png')
counter = 0

for image in images:

    img = cv.imread(image)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) # Convert to grayscale for findChessboardCorners()

    ret, corners = cv.findChessboardCorners(gray, chessboardSize, None) # Find the corners

    if ret == True:

        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria) # Refine the corners
        imgpoints.append(corners)

        cv.drawChessboardCorners(img, chessboardSize, corners2, ret)
        cv.imwrite('img_with_corners_1_' + str(counter) + '.png', img)
        counter += 1

cv.destroyAllWindows()

# Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp2 = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
objp2[:,:2] = np.mgrid[0:chessboardSize[0], 0:chessboardSize[1]].T.reshape(-1, 2)*size_of_square

# Arrays to store object points and image points from all the images.
objpoints2 = [] # 3d point in real world space
imgpoints2 = [] # 2d points in image plane.

images2 = glob.glob('frames2/*.png')
counter = 0

for image in images2:

    img = cv.imread(image)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) # Convert to grayscale for findChessboardCorners()

    ret, corners = cv.findChessboardCorners(gray, chessboardSize, None) # Find the corners

    if ret == True:

        objpoints2.append(objp2)
        corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria) # Refine the corners
        imgpoints2.append(corners)

        cv.drawChessboardCorners(img, chessboardSize, corners2, ret)
        cv.imwrite('img_with_corners_2_' + str(counter) + '.png', img)
        counter += 1

cv.destroyAllWindows()

''' Calibrate both cameras '''

ret, mtrx1, dist1, rvecs1, tvecs1 = cv.calibrateCamera(objpoints, imgpoints, frameSize, None, None)

print("First webcam calibration:")
print("--------------------------")
print("RMSE: \n")
print(ret)
print("Camera matrix: \n")
print(mtrx1)
print("Distortion coefficients: \n")
print(dist1)
print("Rotation vectors: \n")
print(rvecs1)
print("Translation vectors: \n")
print(tvecs1)
print("--------------------------")

ret, mtrx2, dist2, rvecs2, tvecs2 = cv.calibrateCamera(objpoints2, imgpoints2, frameSize, None, None)

print("Second webcam calibration:")
print("--------------------------")
print("RMSE: \n")
print(ret)
print("Camera matrix: \n")
print(mtrx2)
print("Distortion coefficients: \n")
print(dist2)
print("Rotation vectors: \n")
print(rvecs2)
print("Translation vectors: \n")
print(tvecs2)
print("--------------------------")

''' Pose estimation '''

mtx, dist = mtrx1, dist1

def draw(img, corners, imgpts):
    
    ''' Functions that draws directions on the photoes '''
    
    corner = tuple(corners[0].ravel())
    img = cv.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 5)
    img = cv.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
    img = cv.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)
    return img

def draw_cube(img, corners, imgpts):
    
    ''' Modifications for drawing 3d cube instead of lines '''
    
    imgpts = np.int32(imgpts).reshape(-1,2)
    # draw ground floor in green
    img = cv.drawContours(img, [imgpts[:4]],-1,(0,255,0),-3)
    # draw pillars in blue color
    for i,j in zip(range(4),range(4,8)):
        img = cv.line(img, tuple(imgpts[i]), tuple(imgpts[j]),(255),3)
    # draw top layer in red color
    img = cv.drawContours(img, [imgpts[4:]],-1,(0,0,255),3)
    return img

objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
objp[:,:2] = np.mgrid[0:chessboardSize[0], 0:chessboardSize[1]].T.reshape(-1, 2)

axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)
counter = 0

for fname in glob.glob('frames/*.png'):
    img = cv.imread(fname)
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    ret, corners = cv.findChessboardCorners(gray, chessboardSize, None)
    if ret == True:
        corners2 = cv.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        # Find the rotation and translation vectors.
        ret, rvecs, tvecs = cv.solvePnP(objp, corners2, mtx, dist)
        # project 3D points to image plane
        imgpts, jac = cv.projectPoints(axis, rvecs, tvecs, mtx, dist)
        # convert to integers
        imgpts = imgpts.astype(int)
        corners2 = corners2.astype(int)
        img = draw(img,corners2,imgpts)
        cv.imwrite('pose' + str(counter) + '.png', img)
        counter += 1
cv.destroyAllWindows()

objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
objp[:,:2] = np.mgrid[0:chessboardSize[0], 0:chessboardSize[1]].T.reshape(-1, 2)

axis = np.float32([[0,0,0], [0,3,0], [3,3,0], [3,0,0],
                   [0,0,-3],[0,3,-3],[3,3,-3],[3,0,-3]])

counter = 0
for fname in glob.glob('frames/*.png'):
    img = cv.imread(fname)
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    ret, corners = cv.findChessboardCorners(gray, chessboardSize, None)
    if ret == True:
        corners2 = cv.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        # Find the rotation and translation vectors.
        ret, rvecs, tvecs = cv.solvePnP(objp, corners2, mtx, dist)
        # project 3D points to image plane
        imgpts, jac = cv.projectPoints(axis, rvecs, tvecs, mtx, dist)
        # convert to integers
        imgpts = imgpts.astype(int)
        corners2 = corners2.astype(int)
        img = draw_cube(img,corners2,imgpts)
        cv.imwrite('pose_cube' + str(counter) + '.png', img)
        counter += 1
cv.destroyAllWindows()


''' Stereo calibration '''

stereocalibration_flags = cv.CALIB_FIX_INTRINSIC
ret, CM1, distS1, CM2, distS2, R, T, E, F = cv.stereoCalibrate(objpoints, imgpoints, imgpoints2, mtrx1, dist1,
mtrx2, dist2, frameSize, criteria = criteria, flags = stereocalibration_flags)

print("Stereo calibration:")
print("--------------------------")
print("RMSE: \n")
print(ret)
print("Camera matrix 1: \n")
print(CM1)
print("Distortion coefficients 1: \n")
print(distS1)
print("Camera matrix 2: \n")
print(CM2)
print("Distortion coefficients 2: \n")
print(distS2)
print("Rotation matrix: \n")
print(R)
print("Translation vector: \n")
print(T)
print("Essential matrix: \n")
print(E)
print("Fundamental matrix: \n")
print(F)
print("--------------------------")

print("Triangulation")
print("--------------------------")

''' Triangulation '''

# choosing aribtrary keypoints
uvs1 = [[317, 13], [280, 65], [350, 65],
        [445, 455], [420, 190], [116, 314],
        [156, 170], [400, 460], [150, 460],
        [320, 170]]
 
uvs2 = [[390, 5], [358, 40], [420, 48],
        [490, 418], [460, 170], [188, 260],
        [340, 140], [440, 440], [195, 440],
        [366, 158]]
 
uvs1 = np.array(uvs1)
uvs2 = np.array(uvs2)

frame1 = cv.imread('frames/img0.png')
frame2 = cv.imread('frames2/img0.png')

#RT matrix for C1 is identity.
RT1 = np.concatenate([np.eye(3), [[0],[0],[0]]], axis = -1)
P1 = mtrx1 @ RT1 #projection matrix for C1
 
#RT matrix for C2 is the R and T obtained from stereo calibration.
RT2 = np.concatenate([R, T], axis = -1)
P2 = mtrx2 @ RT2 #projection matrix for C2

def DLT(P1, P2, point1, point2):
 
    A = [point1[1]*P1[2,:] - P1[1,:],
         P1[0,:] - point1[0]*P1[2,:],
         point2[1]*P2[2,:] - P2[1,:],
         P2[0,:] - point2[0]*P2[2,:]
        ]
    A = np.array(A).reshape((4,4))
    #print('A: ')
    #print(A)
 
    B = A.transpose() @ A
    from scipy import linalg
    U, s, Vh = linalg.svd(B, full_matrices = False)
 
    print('Triangulated point: ')
    print(Vh[3,0:3]/Vh[3,3])
    return Vh[3,0:3]/Vh[3,3]

p3ds = []
for uv1, uv2 in zip(uvs1, uvs2):
    _p3d = DLT(P1, P2, uv1, uv2)
    p3ds.append(_p3d)
p3ds = np.array(p3ds)

fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Plot the first image
axes[0].imshow(frame1[:,:,[2,1,0]])
axes[0].scatter(uvs1[:,0], uvs1[:,1])
axes[0].set_title('Left webcam')

# Plot the second image
axes[1].imshow(frame2[:,:,[2,1,0]])
axes[1].scatter(uvs2[:,0], uvs2[:,1])
axes[1].set_title('Right webcam')

# Save the first image as PNG
axes[0].get_figure()

# Save the second image as PNG
axes[1].get_figure().savefig('tr_keypoints.png')

# Display the triangulated points
fig, ax = plt.subplots(figsize=(8, 6), subplot_kw={'projection': '3d'})
# Extract the x, y, and z coordinates of the triangulated points
x = p3ds[:, 0]
y = p3ds[:, 1]
z = p3ds[:, 2]

# Plot the points
ax.scatter(x, y, z, c='brown', marker='o')

# Draw connections between the points
connections = [[0,1], [1,2], [0, 2], [3, 4], [5, 6], [7, 8], [9, 4], [9, 5], [9, 7], [9, 8], [9, 0]]
for _c in connections:
    ax.plot(xs = [p3ds[_c[0],0], p3ds[_c[1],0]], ys = [p3ds[_c[0],1], p3ds[_c[1],1]], zs = [p3ds[_c[0],2], p3ds[_c[1],2]], c = 'salmon')

# Set labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Triangulated Points')

# Rotate the axis
ax.view_init(elev=65, azim=90)  # Set the elevation and azimuth angles

# Save the 3D plot as PNG
ax.get_figure().savefig('triangulated_points.png')

#Credits:

#https://docs.opencv.org/3.4/d6/d00/tutorial_py_root.html\
#https://temugeb.github.io/opencv/python/2021/02/02/stereo-camera-calibration-and-triangulation.html

print("--------------------------")
print("Output info:")
print("--------------------------")
print("img_with_corners_1 and img_with corners_2 correspond to each image in frame and frame2")
print("folders with found cheesboard patterns")
print("pose_cube and pose files are outputs of pose estimation")
print("tr_keypoints shows aribtrary keypoints selected on pair of photoes for triangulation")
print("triangulation shows the result of traingulation")
print("--------------------------")
