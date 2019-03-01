import cv2
import dlib
import numpy as np

def facePoints(img, predictor_path):
    """ 
    Inputs:
    img- Image to detect facial landmarks
    predictor_path- Pretrained facial landmarks weight file

        Returns: 
        - points: facial landmarks points
    """
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)
    image = cv2.imread(img)
    rects = detector(image, 1)
    for (i, rect) in enumerate(rects):
        shape = predictor(image, rect)
        points = [(shape.part(i).x, shape.part(i).y) for i in range(68)]
    return points

def faceDim(points, image):
    """ To calculate points around face for warping and affine transformations

        Input:
        points- facial landmarks points
        image- the image with face

        Returns a tuple of:
        - norm_points: normalised points around face
        - (x, y, w, h):  face coordinates (x, y), width and height of boundary in landmark points
        - image[y:y+h, x:x+w]: slicing of face in image
    """

    r = 10
    im_w, im_h = image.shape[:2]
    left, top = np.min(points, 0)
    right, bottom = np.max(points, 0)
    x, y = max(0, left-r), max(0, top-r)
    w, h = min(right+r, im_h)-x, min(bottom+r, im_w)-y
    norm_points = points - np.asarray([[x, y]])
    return norm_points, (x, y, w, h), image[y:y+h, x:x+w]

def getFaceParts(points):
    xmin = 10000
    xmax = -1
    ymin = 10000
    ymax = -1
    yl, yr = 0, 0
    for a, b in points:
        if ymin > b:
            ymin = b
        if ymax < b:
            ymax = b
    for a, b in points:
        if xmin > a:
            xmin, yl = a, b
        if xmax < a:
            xmax, yr = a, b
    return (xmin, yl, yr, xmax, ymin, ymax)

def getConvexHullPoints(xmin, yl, yr, xmax, ymin, ymax):
    """ Points to cover complete face in addition to facial landmarks to
        get all border points using convex Hull function.
        Considering distance from eyebrows to top of face is half of 
        the distance between eyebrows and chin landmark points
    """
    yminup = ymin - int(0.5 * (ymax - ymin))
    midx = int((xmin + xmax) / 2)
    ydiffl = int(yl - yminup) - 1
    ydiffr = int(yr - yminup) - 1
    ylist = [0] * 20
    ylist[0] = yl - int(0.21 * ydiffl) 
    ylist[1] = ylist[0] - int(0.145 * ydiffl)
    ylist[2] = ylist[1] - int(0.140 * ydiffl)
    ylist[3] = ylist[2] - int(0.135 * ydiffl)
    ylist[4] = ylist[3] - int(0.115 * ydiffl)
    ylist[5] = ylist[4] - int(0.105 * ydiffl)
    ylist[6] = ylist[5] - int(0.085 * ydiffl)
    ylist[7] = ylist[6] - int(0.058 * ydiffl)
    ylist[8] = ylist[7] - int(0.03 * ydiffl)
    ylist[9] = yminup
    ylist[19] = yr - int(0.147 * ydiffr)
    ylist[18] = ylist[19] - int(0.147 * ydiffr)
    ylist[17] = ylist[18] - int(0.147 * ydiffr)
    ylist[16] = ylist[17] - int(0.147 * ydiffr)
    ylist[15] = ylist[16] - int(0.137 * ydiffr)
    ylist[14] = ylist[15] - int(0.1111 * ydiffr)
    ylist[13] = ylist[14] - int(0.0910 * ydiffr)
    ylist[12] = ylist[13] - int(0.0605 * ydiffr)
    ylist[11] = ylist[12] - int(0.0415 * ydiffr)
    ylist[10] = ylist[11] - int(0.030 * ydiffr)

    xmidup = int((xmin + xmax) / 2)
    xlist = [0] * 20
    xdiffl = xmidup - xmin
    xdiffr = xmax - xmidup
    xlist[0] = xmin + int(0.0223 * xdiffl)
    xlist[1] = xlist[0] + int(0.0222 * xdiffl)
    xlist[2] = xlist[1] + int(0.0333 * xdiffl)
    xlist[3] = xlist[2] + int(0.0555 * xdiffl)
    xlist[4] = xlist[3] + int(0.0811 * xdiffl)
    xlist[5] = xlist[4] + int(0.1444 * xdiffl)
    xlist[6] = xlist[5] + int(0.1556 * xdiffl)
    xlist[7] = xlist[6] + int(0.2111 * xdiffl)
    xlist[8] = xlist[7] + int(0.1444 * xdiffl)
    xlist[9] = xmidup
    xlist[19] = xmax - int(0.0223 * xdiffr)
    xlist[18] = xlist[19] - int(0.0223 * xdiffr)
    xlist[17] = xlist[18] - int(0.0223 * xdiffr)
    xlist[16] = xlist[17] - int(0.0223 * xdiffr)
    xlist[15] = xlist[16] - int(0.0556 * xdiffr)
    xlist[14] = xlist[15] - int(0.1112 * xdiffr)
    xlist[13] = xlist[14] - int(0.1112 * xdiffr)
    xlist[12] = xlist[13] - int(0.16 * xdiffr)
    xlist[11] = xlist[12] - int(0.1667 * xdiffr)
    xlist[10] = xlist[11] - int(0.16 * xdiffr)

    newPoints = [[0, 0]] * 20
    for i in range(20):
        newPoints[i] = xlist[i], ylist[i]
    return newPoints

def applyAffineTransform(src, srcTri, dstTri, size) :
    # Given a pair of triangles, find the affine transform.
    warpMat = cv2.getAffineTransform( np.float32(srcTri), np.float32(dstTri) )
    # Apply the Affine Transform just found to the src image
    dst = cv2.warpAffine( src, warpMat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101 )
    return dst


# Check if a point is inside a rectangle
def rectContains(rect, point) :
    if point[0] < rect[0] :
        return False
    elif point[1] < rect[1] :
        return False
    elif point[0] > rect[0] + rect[2] :
        return False
    elif point[1] > rect[1] + rect[3] :
        return False
    return True


#calculate delanauy triangle
def calculateDelaunayTriangles(rect, points):
    #create subdiv
    subdiv = cv2.Subdiv2D(rect);
    
    # Insert points into subdiv
    for p in points:
        subdiv.insert(p) 
    
    triangleList = subdiv.getTriangleList();

    delaunayTri = []
    pt = []    
    
    for t in triangleList:        
        pt.append((t[0], t[1]))
        pt.append((t[2], t[3]))
        pt.append((t[4], t[5]))
        
        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])        
        
        if rectContains(rect, pt1) and rectContains(rect, pt2) and rectContains(rect, pt3):
            ind = []
            #Get face-points (from 68 face detector) by coordinates
            for j in range(0, 3):
                for k in range(0, len(points)):                    
                    if(abs(pt[j][0] - points[k][0]) < 1.0 and abs(pt[j][1] - points[k][1]) < 1.0):
                        ind.append(k)    
            # Three points form a triangle. Triangle array corresponds to the file tri.txt in FaceMorph 
            if len(ind) == 3:                                                
                delaunayTri.append((ind[0], ind[1], ind[2]))
        pt = []        
    return delaunayTri
        

# Warps and alpha blends triangular regions from img1 and img2 to img
def warpTriangle(img1, img2, t1, t2) :

    # Find bounding rectangle for each triangle
    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))
    # print("t1 is: ", t1)
    # print("t2 is: ", t2)
    # Offset points by left top corner of the respective rectangles
    t1Rect = [] 
    t2Rect = []
    t2RectInt = []

    for i in range(0, 3):
        t1Rect.append(((t1[i][0] - r1[0]),(t1[i][1] - r1[1])))
        t2Rect.append(((t2[i][0] - r2[0]),(t2[i][1] - r2[1])))
        t2RectInt.append(((t2[i][0] - r2[0]),(t2[i][1] - r2[1])))

    # Get mask by filling triangle
    mask = np.zeros((r2[3], r2[2], 3), dtype = np.float32)
    cv2.fillConvexPoly(mask, np.int32(t2RectInt), (1.0, 1.0, 1.0), 16, 0);

    # Apply warpImage to small rectangular patches
    img1Rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
    #img2Rect = np.zeros((r2[3], r2[2]), dtype = img1Rect.dtype)
    
    size = (r2[2], r2[3])

    img2Rect = applyAffineTransform(img1Rect, t1Rect, t2Rect, size)
    img2Rect = img2Rect * mask

    # Copy triangular region of the rectangular patch to the output image
    img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] * ( (1.0, 1.0, 1.0) - mask )
    img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] + img2Rect 

