import cv2
import numpy as np
import matplotlib.pyplot as plt

def make_coordinates(image, line_parameters):
    slope, intercept = line_parameters
    y1 = (image.shape[0])
    y2 = int(y1 * (3 / 5)) #420
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return np.array([x1, y1, x2, y2])

def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))
    # print(left_fit)
    # print(right_fit)
    left_fit_average = np.average(left_fit, axis=0)
    right_fit_average = np.average(right_fit, axis=0)
    print(left_fit_average, 'left')
    print(right_fit_average, 'right', flush=True)
    
    left_line = make_coordinates(image, left_fit_average)
    right_line = make_coordinates(image, right_fit_average) 
    return np.array([left_line, right_line])


def detect_edges(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    return edges

def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for x1, y1, x2, y2 in lines:
            # x1, y1, x2, y2 = line.reshape(4)
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)   #red color lines with thickness of 10
    return line_image

def region_of_interest(image):
    height = image.shape[0]
    triangle = np.array([[(200, height), (1100, height), (550, 250)]])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, triangle, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

cap = cv2.VideoCapture("./data/highway.mp4")
while cap.isOpened():
    ret, frame = cap.read() #decode frame by frame
    if not ret:
        break
    edges = detect_edges(frame)
    cropped_image = region_of_interest(edges)
    lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
    averaged_lines = average_slope_intercept(frame, lines)
    line_image = display_lines(frame, averaged_lines)
    combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    cv2.imshow('result', combo_image)
    if cv2.waitKey(1) & 0xFF == ord('c'):
        break
cap.release()
cv2.destroyAllWindows()


#gradient measures changes in brightnes over adjacent pixels (0-255).
#edge is a rapid change in brightness
#greyscale images have 1 channel vs 3 channels in color images (RGB), so we convert to greyscale to simplify processing
#find lines in image by using hough space transform
#hough space transform works by transforming points in image space to curves in hough space,
# canny algo to get gradient, detect lilnes with hoph transform and average them to get single line for left and right lane, per frame of video in a while loop