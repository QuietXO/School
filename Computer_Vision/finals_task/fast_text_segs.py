# Import libraries
import os
import cv2
import numpy as np
from scipy import fft
from skimage import io


# Fourier Transformation
def furrier(img):
    return fft.fftshift(fft.fft2(img))


# Edge separation
def edge_finder(img):
    pixels = 200
    img_size = img.shape
    filt_spect = img

    # Set a rectangular region in the frequency domain to zero (high-pass filter)
    filt_spect[img_size[0] // 2 - pixels:img_size[0] // 2 + pixels, img_size[1] // 2 - pixels:img_size[1] // 2 + pixels] = 0 + 0j

    # inverse 2D Fourier Transform
    inverse_img = fft.ifft2(filt_spect)

    return np.abs(inverse_img).astype(np.uint8), filt_spect


# Edge thresholding
def masker(img):
    edge_img = np.abs(img) / np.max(np.abs(img))
    threshold = 0.2

    edges = edge_img > threshold

    # Delete edge noice
    edges[:5] = False
    edges[:, :5] = False
    edges[-5:] = False
    edges[:, -5:] = False

    return edges.astype(np.uint8)


# Edge Enhancing
def maskerer(img):
    lines = cv2.HoughLinesP(img, 1, np.pi / 200, threshold=10, minLineLength=40, maxLineGap=30)

    # Draw lines on a copy of the original image
    outp = img.copy()
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(outp, (x1, y1), (x2, y2), (255, 255, 255), 3)

    return outp.astype(np.uint8)


# Segmentation
def blocky(img_mask, square=20, threshold=255):
    area = 1
    sqr_size = square
    pixel_threshold = threshold
    ground_truth_outp = np.full((img_mask.shape[0], img_mask.shape[1]), 0)

    # 1st Row
    for x in range(int(img_mask.shape[0]/sqr_size)*2, -1, -1):
        square = img_mask[:sqr_size, int(sqr_size*(x/2)):int(((sqr_size*(x+2))/2))]
        if np.sum(square) < pixel_threshold:
            ground_truth_outp[:sqr_size, int(sqr_size*(x/2)):int(((sqr_size*(x+2))/2))] = area
        elif np.any(ground_truth_outp == area):
            area += 1

    # 1st Column
    for y in range(2, int(img_mask.shape[1]/sqr_size)*2+1):
        square = img_mask[int(sqr_size*(y/2)):int(((sqr_size*(y+2))/2)), :sqr_size]
        if np.sum(square) < pixel_threshold:
            ground_truth_outp[int(sqr_size*(y/2)):int(((sqr_size*(y+2))/2)), :sqr_size] = area
        elif np.any(ground_truth_outp == area):
            area += 1

    # rest of the rows, row by row
    for y in range(1, int(img_mask.shape[1]/sqr_size)*2):
        for x in range(1, int(img_mask.shape[1]/sqr_size)*2):
            square = img_mask[int(sqr_size*(y/2)):int(((sqr_size*(y+2))/2)), \
                     int(sqr_size*(x/2)):int(((sqr_size*(x+2))/2))]
            if np.sum(square) < pixel_threshold:
                # corner values
                upF = ground_truth_outp[int(sqr_size*(y/2))+int(sqr_size/2)-1][int(sqr_size*(x/2))]
                upL = ground_truth_outp[int(sqr_size*(y/2))+int(sqr_size/2)-1][int(sqr_size*(x/2))+sqr_size-1]
                leftF = ground_truth_outp[int(sqr_size*(y/2))][int(sqr_size*(x/2))+int(sqr_size/2)-1]
                leftL = ground_truth_outp[int(sqr_size*(y/2))+sqr_size-1][int(sqr_size*(x/2))+int(sqr_size/2)-1]

                # copy up if [0] == [-1] of the square above & if not 0 & if left = 0
                if upF == upL and (upF != 0 or upL != 0) and (upL == leftL or leftL == 0):
                    ground_truth_outp[int(sqr_size*(y/2)):int(((sqr_size*(y+2))/2)), \
                    int(sqr_size*(x/2)):int(((sqr_size*(x+2))/2))] = upF

                # copy left if [0] == [-1] of the square of the left & if not 0 & if up = 0
                elif leftF == leftL and (leftF != 0 or leftL != 0) and (upL == leftL or upL == 0):
                    ground_truth_outp[int(sqr_size*(y/2)):int(((sqr_size*(y+2))/2)), \
                    int(sqr_size*(x/2)):int(((sqr_size*(x+2))/2))] = leftF

                elif upL == 0 and leftL == 0:
                    new_x = x
                    try:
                        while ground_truth_outp[int(sqr_size*(y/2))+int(sqr_size/2)-1][int(sqr_size*(new_x/2))+sqr_size-1] == 0 \
                                and (np.sum(img_mask[int(sqr_size*(y/2)):int(((sqr_size*(y+2))/2)),
                                            int(sqr_size*(new_x/2)):int(((sqr_size*(new_x+2))/2))]) < pixel_threshold):
                            new_x += 1

                        if ground_truth_outp[int(sqr_size*(y/2))+int(sqr_size/2)-1][int(sqr_size*(new_x/2))+sqr_size-1] == 0:
                            ground_truth_outp[int(sqr_size*(y/2)):int(((sqr_size*(y+2))/2)), \
                            int(sqr_size*(x/2)):int(((sqr_size*(x+2))/2))] = area
                        else:
                            ground_truth_outp[int(sqr_size*(y/2)):int(((sqr_size*(y+2))/2)), \
                            int(sqr_size*(x/2)):int(((sqr_size*(x+2))/2))] = ground_truth_outp[int(sqr_size*(y/2))+int(sqr_size/2)-1] \
                                [int(sqr_size*(new_x/2))+sqr_size-1]
                    except IndexError:
                        pass

            elif np.any(ground_truth_outp == area):
                area += 1

    return ground_truth_outp.astype(np.uint8)


# Compile steps
def segmentation(img):
    fourier = furrier(img)
    edges = edge_finder(fourier)
    beta_mask = masker(edges[0])
    final_mask = maskerer(beta_mask)
    wannabe_segs = blocky(final_mask)

    return wannabe_segs


if __name__ == '__main__':
    N_OF_IMAGES = 20

    for index in range(1, N_OF_IMAGES+1):
        # Import img
        tm_path = ''.join(['tm', str(index), '_1_1.png'])
        image = io.imread(tm_path).astype(np.uint8)

        finally_segs = segmentation(image)

        # Export img
        os.mkdir('output') if not os.path.exists('output') else _
        seg_path = ''.join(['output/seg', str(index), '_1_1.png'])
        io.imsave(seg_path, finally_segs, check_contrast=False)
