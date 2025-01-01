# Document Scanner using OpenCV

This project demonstrates a simple document scanner using OpenCV in Python. It utilizes edge detection, contour finding, and perspective transformation techniques to scan and straighten documents from an image or webcam feed.

## Features

- **Webcam Feed or Image Input**: Process images from a webcam or from a file path.
- **Edge Detection**: Uses Canny edge detection to find document edges.
- **Contour Detection**: Finds the biggest contour (assumed to be the document) and highlights it.
- **Perspective Transformation**: Straightens the document by applying a perspective transformation.
- **Adaptive Thresholding**: Enhances document readability with adaptive thresholding techniques.
- **Real-Time Scanning**: View the live scanning process in real-time with adjustable threshold settings.
- **Save Scanned Images**: Save the scanned (straightened) document to disk.

## Requirements

To run the project, you'll need Python 3.x and the following dependencies:

```bash
pip install opencv-python numpy
