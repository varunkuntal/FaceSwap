import cv2
import argparse
import numpy as np
from utils import *

ap = argparse.ArgumentParser(description='Swap face from two images, using full face or just using facial landmarks points')

ap.add_argument("-o", "--output",
	help="path to destination image")
ap.add_argument('--flipsrc', default=False, 
    action='store_true', help='flip source image')
ap.add_argument('--fullface', default=False, 
    action='store_true', help='include complete face points in addition to landmark points')
req = ap.add_argument_group('required named arguments')
req.add_argument("-p", "--predictor", required=True,
	help="path to facial landmark predictor")
req.add_argument("-s", "--source", required=True,
	help="path to source image")
req.add_argument("-d", "--destination", required=True,
	help="path to destination image on which src face will be stuck")
args = vars(ap.parse_args())

filename1 = args['source']
filename2 = args['destination']
if args['output']:
    outputPath = 'Output\\' + args['output']
predictor_path = args['predictor']

img1 = cv2.imread(filename1);
img2 = cv2.imread(filename2);
img1Warped = np.copy(img2);  

if args['flipsrc']:
    img1 = cv2.flip(img1, 1)

points1 = facePoints(filename1, predictor_path)
points2 = facePoints(filename2, predictor_path)

if args['fullface']:
    xmin1, yl1, yr1, xmax1, ymin1, ymax1 = getFaceParts(points1)
    xmin2, yl2, yr2, xmax2, ymin2, ymax2 = getFaceParts(points2)
    newPoints1 = getConvexHullPoints(xmin1, yl1, yr1, xmax1, ymin1, ymax1)
    newPoints2 = getConvexHullPoints(xmin2, yl2, yr2, xmax2, ymin2, ymax2)
    points1 = points1 + newPoints1
    points2 = points2 + newPoints2

hull1 = []
hull2 = []

hullIndex = cv2.convexHull(np.array(points2), returnPoints = False)
for i in range(0, len(hullIndex)):
    hull1.append(points1[int(hullIndex[i])])
    hull2.append(points2[int(hullIndex[i])])

sizeImg2 = img2.shape    
rect = (0, 0, sizeImg2[1], sizeImg2[0])
dt = calculateDelaunayTriangles(rect, hull2)

if len(dt) == 0:
    quit()

# Apply affine transformation to Delaunay triangles
for i in range(0, len(dt)):
    t1 = []
    t2 = []
    
    #get points for img1, img2 corresponding to the triangles
    for j in range(0, 3):
        t1.append(hull1[dt[i][j]])
        t2.append(hull2[dt[i][j]])
    
    warpTriangle(img1, img1Warped, t1, t2)

# Calculate Mask
hull8U = []
for i in range(0, len(hull2)):
    hull8U.append((hull2[i][0], hull2[i][1]))

mask = np.zeros(img2.shape, dtype = img2.dtype)  
cv2.fillConvexPoly(mask, np.int32(hull8U), (255, 255, 255))

r = cv2.boundingRect(np.float32([hull2]))    

center = ((r[0]+int(r[2]/2), r[1]+int(r[3]/2)))

# Clone seamlessly.
output = cv2.seamlessClone(np.uint8(img1Warped), img2, mask, center, cv2.NORMAL_CLONE)

# Display Swapped Image & Save it in the Output Folder
if args['output']:
    cv2.imwrite(outputPath, output)
cv2.imshow("Face Swapped", output)
cv2.waitKey(0)
cv2.destroyAllWindows()