import numpy as np
import cv2
from matplotlib import pyplot as plt


def post_processing(b_m, image):

    kernel = np.ones((11, 11), np.uint8)
    b_m_changed = cv2.dilate(b_m, kernel, iterations=1) #morphological closing of feature map
    kernel = np.ones((9, 9), np.uint8)
    b_m_ch = cv2.erode(b_m_changed, kernel, iterations=1) #erasing feature map
    plt.imshow(b_m_ch.reshape(128, 128), cmap='hot')

    b_m1 = cv2.morphologyEx(b_m_ch, cv2.MORPH_CLOSE, kernel) #another way of morph. closing
    plt.imshow(b_m1.reshape(128, 128), cmap='hot')

    resized_b_m = cv2.resize(b_m1, dsize=(256, 256)) #resizing of feature map
    resized_b_m[np.where(resized_b_m<tr)] = 0
    resized_b_m[np.where(resized_b_m>0)] = 1
    plt.imshow(resized_b_m, cmap='hot')

    masked_image = image[resized_b_m == 1] #image with feature map
    masked_image = masked_image.reshape(-1, 3)
    mean_vector = np.mean(masked_image, axis=0) #mean pixel of image with feature map

    copy = image.copy()
    copy[resized_b_m == 0] = 0 #showing image with feature map
    plt.imshow(copy)

    mean_vector /= 255 #normalizing of mean pixel
    masked_image = np.divide(masked_image, 255) #normalizing of masked image

    eucl = np.sqrt(np.sum((masked_image - mean_vector) ** 2, axis=1)) / np.sqrt(3) #euclidian distance of pixels

    eucl = (eucl - eucl.min()) / (eucl.max() - eucl.min()) #normalizing of euclidian distance

    tr1 = 0.22 #threshold for mask
    mask2d = resized_b_m.flatten() #getting pixels of feature map

    inner_mask = np.zeros_like(eucl)
    inner_mask[eucl > tr1] = 1 #creating new mask
    mask2d[mask2d == 1] = inner_mask #inserting the new mask in the old one
    plt.imshow(mask2d.reshape(256, 256), cmap='hot')

    kernel = np.ones((3, 3), np.uint8)
    mask2d_changed = cv2.dilate(mask2d, kernel, iterations=2)
    kernel = np.ones((2, 2), np.uint8)
    mask2d_changed = cv2.erode(mask2d_changed, kernel, iterations=2)  #second morphologicsl closing
    plt.figure(figsize=(6, 6))
    plt.imshow(mask2d_changed.reshape(256, 256), cmap='hot')
    plt.axis('Off')
    plt.show()

    return mask2d_changed.reshape(256, 256)

if __name__ == '__main__':
    pass