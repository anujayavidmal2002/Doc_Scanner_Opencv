import cv2
import numpy as np


def stackImages(imgArray, scale, labels=[]):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range(0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape[:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: 
                    imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank] * rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None, scale, scale)
            if len(imgArray[x].shape) == 2: 
                imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(imgArray)
        ver = hor
    if len(labels) != 0:
        eachImgWidth = int(ver.shape[1] / cols)
        eachImgHeight = int(ver.shape[0] / rows)
        for d in range(0, rows):
            for c in range(0, cols):
                cv2.rectangle(ver, (c * eachImgWidth, eachImgHeight * d), (c * eachImgWidth + len(labels[d][c]) * 13 + 27, 30 + eachImgHeight * d), (255, 255, 255), cv2.FILLED)
                cv2.putText(ver, labels[d][c], (eachImgWidth * c + 10, eachImgHeight * d + 20), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 0, 255), 2)
    return ver

def reorder(myPoints):
    """
    Reorder points for perspective transform.
    Points are arranged in the order: top-left, top-right, bottom-left, bottom-right.
    """
    myPoints = myPoints.reshape((4, 2))
    myPointsNew = np.zeros((4, 1, 2), dtype=np.int32)
    add = myPoints.sum(1)  # Sum of x and y coordinates

    myPointsNew[0] = myPoints[np.argmin(add)]  # Top-left point
    myPointsNew[3] = myPoints[np.argmax(add)]  # Bottom-right point
    diff = np.diff(myPoints, axis=1)  # Difference between x and y coordinates
    myPointsNew[1] = myPoints[np.argmin(diff)]  # Top-right point
    myPointsNew[2] = myPoints[np.argmax(diff)]  # Bottom-left point

    return myPointsNew

def biggestContour(contours):
    """
    Find the biggest contour in the image.
    Filters based on area and ensures the contour has 4 points (a quadrilateral).
    """
    biggest = np.array([])
    max_area = 0
    for i in contours:
        area = cv2.contourArea(i)
        if area > 5000:  # Filter out small areas
            peri = cv2.arcLength(i, True)  # Calculate the perimeter
            approx = cv2.approxPolyDP(i, 0.02 * peri, True)  # Approximate to a polygon
            if area > max_area and len(approx) == 4:  # Check if it's a quadrilateral
                biggest = approx
                max_area = area

    return biggest, max_area

def drawRectangle(img, biggest, thickness):
    """
    Draw a rectangle around the biggest contour.
    Uses the four points to draw lines between them.
    """
    cv2.line(img, tuple(biggest[0][0]), tuple(biggest[1][0]), (0, 255, 0), thickness)  # Top edge
    cv2.line(img, tuple(biggest[1][0]), tuple(biggest[2][0]), (0, 255, 0), thickness)  # Right edge
    cv2.line(img, tuple(biggest[2][0]), tuple(biggest[3][0]), (0, 255, 0), thickness)  # Bottom edge
    cv2.line(img, tuple(biggest[3][0]), tuple(biggest[0][0]), (0, 255, 0), thickness)  # Left edge
    return img

def nothing(x):
    """
    A placeholder function for trackbar callback.
    """
    pass

def initializeTrackbars():
    """
    Initialize trackbars for threshold values.
    These trackbars allow real-time adjustment of thresholds for Canny edge detection.
    """
    cv2.namedWindow("Trackbars", cv2.WINDOW_NORMAL)  # Ensure the window is properly created
    cv2.resizeWindow("Trackbars", 360, 240)
    cv2.createTrackbar("Canny Low", "Trackbars", 200, 255, nothing)
    cv2.createTrackbar("Canny High", "Trackbars", 200, 255, nothing)

def valTrackbars():
    """
    Get the current values of the trackbars.
    """
    cannyLow = cv2.getTrackbarPos("Canny Low", "Trackbars")
    cannyHigh = cv2.getTrackbarPos("Canny High", "Trackbars")
    return cannyLow, cannyHigh
