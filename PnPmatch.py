import numpy as np
import cv2
import os
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt


def getbg(img, msk):
    return (img*msk).astype('uint8')


def triangulate(matchp1, matchp2, P1, P2):

    points = np.zeros([4,matchp1.shape[0]])
    i = 0
    for mp1, mp2 in zip(matchp1, matchp2):
        A = np.vstack((mp1.flatten()[0]*P1[2]-P1[0],
                       mp1.flatten()[1]*P1[2]-P1[1],
                       mp2.flatten()[0]*P2[2]-P2[0],
                       mp2.flatten()[1]*P2[2]-P2[1]))
        u, s, v = np.linalg.svd(A)
        points[:,i] = v[-1]
        i += 1
#         Roughly same using eig
#         w, v = np.linalg.eig(A)
#         points.append(v[:3,-1]/v[3,-1])
    return points


def to_homog(points):
    return np.vstack((points,np.ones(points.shape[1])))


# convert points from homogeneous to euclidian
def from_homog(points_homog):
    return points_homog[:-1]/points_homog[-1]


frame_i = 1

MIN_MATCH_COUNT = 10
file_pattern = '/Users/yujieli/Documents/CFS_Video_Analysis-master/refgen.png'
img1 = cv2.imread(file_pattern, flags = cv2.IMREAD_GRAYSCALE)

size = img1.shape
focal_length = size[1]
camera_center = (size[1] / 2, size[0] / 2)
sw = 6.16
sh = 4.62
cw = 4000
ch = 3000
dx = sw/cw
dy = sh/ch
ff = 20
fx = ff/dx
fy = ff/dy
u0 = size[1] / 2 
v0 = size[0] / 2
# Initialize approximate camera intrinsic matrix
camera_intrinsic_matrix = np.array([[fx, 0, u0],
                                    [0, fy, v0],
                                    [0, 0, 1]
                                    ], dtype = "double")

P1 = np.dot(camera_intrinsic_matrix,np.eye(4)[:3]) # One critical problem is obtaining P1

mask = np.ones(img1.shape)
mask[:,1450:2650] = 0
mask[:600] = 0
mask[:900, 2650:] = 0
mask[-500:-70, :800] = 0
mask[1200:1309, 1100:1218] = 0
mask[1186:1437, 2650:3230] = 0

# Default params
# int     nfeatures = 0,
# int     nOctaveLayers = 3,
# double  contrastThreshold = 0.04,
# double  edgeThreshold = 10,
# double  sigma = 1.6 
sift_params = dict(nfeatures = 0,nOctaveLayers = 3,contrastThreshold = 0.12,edgeThreshold = 4,sigma = 1.6)
sift = cv2.xfeatures2d.SIFT_create(**sift_params)
kp1, des1 = sift.detectAndCompute(img1,None)

# Construct first 3D point set
img2 = cv2.imread("/Users/yujieli/Documents/CFS_Video_Analysis-master/NFramesRaw5/frame891.png", cv2.IMREAD_GRAYSCALE)
img2_Cut = getbg(img2,mask)
kp2, des2 = sift.detectAndCompute(img2_Cut, None)

bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

good_ini = []
for m,n in matches:
    if m.distance < 0.6*n.distance:
        good_ini.append(m)

dst_pts_ini = np.float32([ kp1[m.queryIdx].pt for m in good_ini ]).reshape(-1,1,2)
src_pts_ini = np.float32([ kp2[m.trainIdx].pt for m in good_ini ]).reshape(-1,1,2)       
des_ini = np.float32([des1[m.queryIdx] for m in good_ini])
# des_ini2 = np.float32([des2[m.trainIdx] for m in good_ini])

E ,mask0 = cv2.findEssentialMat(src_pts_ini, dst_pts_ini, camera_intrinsic_matrix)
retval, rvec, tvec, mask1 = cv2.recoverPose(E, src_pts_ini, dst_pts_ini, camera_intrinsic_matrix, 50)
# Another problem is the scalar of tvec
P2 = np.dot(camera_intrinsic_matrix,np.hstack((rvec,tvec)))
    
print "Initial pairs", len(good_ini)

X3d = cv2.triangulatePoints(P1, P2, src_pts_ini, dst_pts_ini)[:,(mask0>0).squeeze()]

with open("/Users/yujieli/Documents/CFS_Video_Analysis-master/test/CamPoseTrial.txt", "w") as f:
    frame_i = 892
    while(frame_i <= 900):

        print frame_i

        img2 = cv2.imread("/Users/yujieli/Documents/CFS_Video_Analysis-master/NFramesRaw5/frame%d.png"%(frame_i), \
            cv2.IMREAD_GRAYSCALE)
        img2_Cut = getbg(img2,mask)
        kp2, des2 = sift.detectAndCompute(img2_Cut, None)

        matches = bf.knnMatch(des_ini[(mask0>0).squeeze()], des2, k=2)
        good = []
        for m,n in matches:
            if m.distance < 0.6*n.distance:
                good.append(m)

        dst_pts_3d = np.float32([ X3d[:3,m.queryIdx] for m in good ]).reshape(-1,1,3)
        src_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
        dist_coeffs = np.zeros((4,1))
        retval, rvec, tvec, inliers = cv2.solvePnPRansac(dst_pts_3d, 
                                                         src_pts, 
                                                         camera_intrinsic_matrix, 
                                                         dist_coeffs)
        Rmat, jacobian = cv2.Rodrigues(rvec)

        f.write(str(frame_i) + "\t")
        for i in range(3):
            f.write(str(rvec.squeeze()[i]) + "\t")
        for i in range(3):
            f.write(str(tvec.squeeze()[i]) + "\t")
        f.write("\n")

        frame_i += 8


dst_pts_homog = np.vstack((dst_pts_3d.reshape(-1,3).T, np.ones((1,len(good)))))
P3 = np.dot(camera_intrinsic_matrix,np.hstack((Rmat, tvec)))
X33d = np.dot(P3,dst_pts_homog)
X3 = from_homog(X33d)
print X3.T[inliers]
print src_pts.reshape(-1,2)[inliers]
print (X3.T[inliers]-src_pts.reshape(-1,2)[inliers]).mean()
print (X3.T[inliers]-src_pts.reshape(-1,2)[inliers]).max()
