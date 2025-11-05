import numpy as np
from numpy import linalg
import cv2
from PIL import Image
from pathlib import Path

def rect_sum(integral_img, x1, x2, y1, y2):
    #bottom right (x2, y2)
    a = integral_img[y2 + 1, x2 + 1]

    #top right (x2, y1 - 1)
    b = integral_img[y1, x2 + 1]

    #bottom left (x1 - 1, y2)
    c = integral_img[y2 + 1, x1]

    #top left (x1 - 1, y1 - 1)
    d = integral_img[y1, x1]

    sum = a - b - c + d
    return sum

def connected_component_analysis(mask, min_area):
    reversed_mask = cv2.bitwise_not(mask)

    (totalLabels, label_ids, values, centroid) = cv2.connectedComponentsWithStats(reversed_mask, 8, cv2.CV_32S)

    clean = np.zeros_like(reversed_mask)

    for i in range(1, totalLabels):
        component_area = values[i, cv2.CC_STAT_AREA]

        if (component_area >= min_area):
            componentMask = (label_ids == i).astype("uint8") * 255
            clean = cv2.bitwise_or(clean, componentMask)

            
    clean = cv2.bitwise_not(clean)
    return clean

def illumination_correction(gray, filter_size):
    blur = cv2.GaussianBlur(gray, (filter_size, filter_size), 0)
    corrected = (gray / (blur + 1e-5)) * np.mean(blur)
    corrected = np.clip(corrected, 0, 255).astype(np.uint8)
    return corrected

import cv2
import numpy as np

def apply_contour_smoothing(mask, epsilon_factor=0.0001):
    inverted_mask = cv2.bitwise_not(mask.astype(np.uint8))

    contours, hierarchy = cv2.findContours(inverted_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    smoothed_mask_inverted = np.zeros_like(mask, dtype=np.uint8)

    if hierarchy is None: return mask

    for i in range(len(contours)):
        contour = contours[i]
        
        perimeter = cv2.arcLength(contour, True)
        epsilon = epsilon_factor * perimeter 
        approx_contour = cv2.approxPolyDP(contour, epsilon, True)

        if hierarchy[0][i][3] == -1:
            cv2.drawContours(smoothed_mask_inverted, [approx_contour], -1, 255, thickness=cv2.FILLED, lineType = cv2.LINE_AA)
        else:
            cv2.drawContours(smoothed_mask_inverted, [approx_contour], -1, 0, thickness=cv2.FILLED)

    final_mask = cv2.bitwise_not(smoothed_mask_inverted)
    return final_mask

def sauvola(input_img, window, k, r, min_area, blur_size):
    a = cv2.imread(input_img)

    gray_scale = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)

    img_float = gray_scale.astype(np.float32)

    corrected = illumination_correction(img_float, blur_size)

    normalized = corrected / 255.0

    #integral for calculating mean and standart deviation
    integral = cv2.integral(normalized)
    integral_squared = cv2.integral(normalized * normalized)

    h, w = normalized.shape
    output_mask = np.zeros_like(normalized)
    windowHalf = window // 2

    for y in range(h):
        for x in range(w):
            print(x, y)
            x1 = max(0, x - windowHalf)
            y1 = max(0, y - windowHalf)
            x2 = min(w - 1, x + windowHalf)
            y2 = min(h - 1, y + windowHalf)

            area = ((x2 - x1 + 1) * (y2 - y1 + 1))

            sum = rect_sum(integral, x1, x2, y1, y2)
            squared_sum = rect_sum(integral_squared, x1, x2, y1, y2)

            mean = sum / area
            squared_mean = squared_sum / area

            variance = squared_mean - (mean * mean)
            s = np.sqrt(np.maximum(0, variance))

            t = mean * ( 1 + k * (s / r - 1))

            if(normalized[y, x] < t):
                output_mask[y, x] = 0
            else:
                output_mask[y, x] = 1

    final_mask = (output_mask * 255).astype(np.uint8)

    final_mask = connected_component_analysis(final_mask, min_area)

    final_mask = apply_contour_smoothing(final_mask, 0.001)

    kernel = np.ones((1,1), np.uint8)
    final_mask = cv2.erode(final_mask, kernel, iterations=1)
    result = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel)

    for i in range(1, 500):
        result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel)

    return result

def sauvola_vec(input_img, window, k, r, min_area, blur_size):
    a = cv2.imread(input_img)

    gray_scale = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)

    img_float = gray_scale.astype(np.float32)

    corrected = illumination_correction(img_float, blur_size)

    normalized = corrected / 255.0

    #integral for calculating mean and standart deviation
    integral = cv2.integral(normalized)
    integral_squared = cv2.integral(normalized * normalized)

    h, w = normalized.shape
    output_mask = np.zeros_like(normalized)
    windowHalf = window // 2

    Y_coord = np.arange(h)
    X_coord = np.arange(w)

    Y1 = np.maximum(0, Y_coord - windowHalf)
    X1 = np.maximum(0, X_coord - windowHalf)
    Y2 = np.minimum(h, Y_coord + windowHalf + 1)
    X2 = np.minimum(w, X_coord + windowHalf + 1)

    Y1_grid, X1_grid = np.meshgrid(Y1, X1, indexing='ij')
    Y2_grid, X2_grid = np.meshgrid(Y2, X2, indexing='ij')

    area = (Y2_grid - Y1_grid) * (X2_grid - X1_grid)

    sum = (integral[Y2_grid, X2_grid] - integral[Y1_grid, X2_grid] - integral[Y2_grid, X1_grid] + integral[Y1_grid, X1_grid])
    squared_sum = (integral_squared[Y2_grid, X2_grid] - integral_squared[Y1_grid, X2_grid] - integral_squared[Y2_grid, X1_grid] + integral_squared[Y1_grid, X1_grid])

    mean = sum / area
    squared_mean = squared_sum / area

    variance = squared_mean - (mean * mean)
    s = np.sqrt(np.maximum(0, variance))

    t = mean * ( 1 + k * (s / r - 1))

    output_mask = (normalized < t).astype(np.float32)
    output_mask = 1 - output_mask 
    output_mask = (normalized > t).astype(np.float32)
    

    final_mask = (output_mask * 255).astype(np.uint8)

    final_mask = connected_component_analysis(final_mask, min_area)

    final_mask = apply_contour_smoothing(final_mask, 0.001)

    kernel = np.ones((1,1), np.uint8)
    final_mask = cv2.erode(final_mask, kernel, iterations=1)
    result = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel)

    for i in range(1, 500):
        result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel)

    return result

result_mask = sauvola_vec("image_cropping/assets/dataset-cover.png", 120, 0.2, 0.5, 400, 121)
cv2.imwrite("image_cropping/assets/sauvola/dataset-cover_vec.png", result_mask)
