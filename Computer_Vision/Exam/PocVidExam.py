# Import Libraries

import cv2
import time
import numpy as np
from skimage import color
import matplotlib.pyplot as plt
from skimage import morphology
from skimage.measure import label

plt.rc('font', **{'family': 'DejaVu Sans', 'weight': 'normal'})
plt.rcParams['font.size'] = 18

def add_text_to_frame(frame, text, position=(30, 30), font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.3, color=(0, 255, 0), thickness=1):
    frame_with_text = frame.copy()
    cv2.putText(frame_with_text, text, position, font, font_scale, color, thickness)
    return frame_with_text

def get_num_of_points(img):
    _, image = cap.read(img)
    image_lab = color.rgb2lab(image)
    points = ((image_lab[:, :, 1] > 8 - 5) * (image_lab[:, :, 1] < 8 + 5)
              * (image_lab[:, :, 2] > -16 - 5) * (image_lab[:, :, 2] < -16 + 5))

    diff_thresholded_processed = morphology.dilation(morphology.erosion(morphology.closing(morphology.closing(morphology.closing(morphology.remove_small_holes(points, area_threshold=1000), morphology.square(5)))), morphology.square(5)), morphology.square(5))

    label_img = label(diff_thresholded_processed, connectivity=2)

    return diff_thresholded_processed.astype(np.uint8)

cap = cv2.VideoCapture('hra.mov')
MAX_POINTS = get_num_of_points(cap.read()[1])

cv2.namedWindow('Pac-Man', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Pac-Man', 800, 800)  # Adjust the dimensions as needed

time_frame = 0

while cap.isOpened():
    start_time = time.time()
    ret, frame = cap.read()
    if not ret:
        break

    read_time = time.time() - start_time

    mask = get_num_of_points(frame)
    mask[mask==1] = 255
    # points = MAX_POINTS - n_points

    points_time = time.time() - start_time

    # frame = add_text_to_frame(frame, f'Score: {points} out of {MAX_POINTS}', font_scale=1.5,
    #                           position=(10, 40), color=(0, 0, 255), thickness=3)

    # Display the frame with bounding boxes
    cv2.imshow('Pac-Man', mask)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    end_time = time.time() - start_time

    print(f"Read Time {time_frame}:", read_time)
    print(f"Calc Time {time_frame}:", points_time)
    print(f"Full Time {time_frame}:", end_time)

    time_frame += 1

# Release the video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
