import cv2
import numpy as np
import utils

################################
# Setup configurations
webCamFeed = True
pathImage = 'C:/Users/DELL/Downloads/Courses/OpenCV Projects/Doc_Scanner/1.jpg'
cap = cv2.VideoCapture(0)
cap.set(10, 160)  # Set brightness
heightImg = 640
widthImg = 480
################################

utils.initializeTrackbars()  # Initialize trackbars for threshold adjustment
count = 0

while True:
    cannyLow, cannyHigh = utils.valTrackbars()
    # Blank image for testing purposes
    imgBlank = np.zeros((heightImg, widthImg, 3), np.uint8)

    # Capture image from webcam or read from file
    if webCamFeed:
        success, img = cap.read()
        if not success:
            print("Failed to read from webcam. Check your camera index or connection.")
            break
    else:
        img = cv2.imread(pathImage)
        if img is None:
            print(f"Failed to load image from {pathImage}. Check the file path.")
            break

    img = cv2.resize(img, (widthImg, heightImg))  # Resize image
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)  # Apply Gaussian blur

    # Get trackbar values and apply Canny edge detection
    thres = utils.valTrackbars()
    imgThreshold = cv2.Canny(imgBlur, thres[0], thres[1])

    # Dilation and erosion to improve edge detection
    kernel = np.ones((5, 5))
    imgDial = cv2.dilate(imgThreshold, kernel, iterations=2)
    imgThreshold = cv2.erode(imgDial, kernel, iterations=1)

    # Find and draw all contours
    imgContours = img.copy()
    contours, hierarchy = cv2.findContours(imgThreshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 10)

    # Find the biggest contour
    biggest, maxArea = utils.biggestContour(contours)
    if biggest.size != 0:
        biggest = utils.reorder(biggest)
        cv2.drawContours(imgContours, [biggest], -1, (0, 255, 0), 20)  # Draw the biggest contour
        imgContours = utils.drawRectangle(imgContours, biggest, 2)

        # Perspective transformation
        pts1 = np.float32(biggest)
        pts2 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg))

        # Remove border noise and apply adaptive thresholding
        imgWarpColored = imgWarpColored[20:imgWarpColored.shape[0]-20, 20:imgWarpColored.shape[1]-20]
        imgWarpColored = cv2.resize(imgWarpColored, (widthImg, heightImg))
        imgWarpGray = cv2.cvtColor(imgWarpColored, cv2.COLOR_BGR2GRAY)
        imgAdaptiveThre = cv2.adaptiveThreshold(imgWarpGray, 255, 1, 1, 7, 2)
        imgAdaptiveThre = cv2.bitwise_not(imgAdaptiveThre)
        imgAdaptiveThre = cv2.medianBlur(imgAdaptiveThre, 3)

        # Image array for display
        imageArray = ([img, imgGray, imgThreshold, imgContours],
                      [imgWarpColored, imgWarpGray, imgAdaptiveThre, imgBlank])
    else:
        # If no biggest contour is found, use a blank image for imgBigContour
        imgBigContour = np.zeros_like(img)  # Create a blank image

        # Image array for display when no contour is found
        imageArray = ([img, imgGray, imgThreshold, imgContours],
                      [imgBigContour, imgWarpColored, imgWarpGray, imgAdaptiveThre])

    # Correct labels to match the images displayed
    labels = [['Original', 'Gray', 'Threshold', 'Contours'],
              ['Warp Perspective', 'Warp Gray', 'Adaptive Threshold', 'Biggest Contour']]

    # Display stacked images
    stackedImage = utils.stackImages(imageArray, 0.75, labels)
    cv2.imshow("Result", stackedImage)
    
    # Save image when 's' is pressed
    if cv2.waitKey(1) & 0xFF == ord('s'):
        cv2.imwrite(f"scanned/myImage{count}.jpg", imgWarpColored)
        cv2.putText(stackedImage, "Scan Saved", (int(stackedImage.shape[1]/2)-200, 100),
                    cv2.FONT_HERSHEY_DUPLEX, 3, (0, 0, 255), 5, cv2.LINE_AA)
        cv2.imshow("Result", stackedImage)
        cv2.waitKey(300)
        count += 1
