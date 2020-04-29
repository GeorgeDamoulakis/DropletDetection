import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
from os import listdir
from os.path import isfile, join

############## I   N    F   O ##########################
# =======================================================
# new image
#
# =======================================================

# Read image
im_in = cv2.imread('/Users/georgedamoulakis/PycharmProjects/Droplets/2_Images/for_MOHA/16050-gamma-sharp.jpg',
                   cv2.IMREAD_GRAYSCALE);


def CountingCC():
    # Threshold, Set values equal to or above 220 to 0, Set values below 220 to 255.
    # asjust the thresh -- important first step
    th, im_th = cv2.threshold(im_in, 50, 150, cv2.THRESH_BINARY_INV);

    def CC(img):
        nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(img)
        label_hue = np.uint8(179 * labels / np.max(labels))
        blank_ch = 255 * np.ones_like(label_hue)
        labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])
        labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)
        labeled_img[label_hue == 0] = 0
        return labeled_img, nlabels, labels, stats, centroids

    # fixing the image
    # this is the second part of the image process
    kernel = np.ones((3, 3), np.uint8)
    erosion = cv2.erode(im_th, kernel, iterations=3)
    dilation = cv2.dilate(erosion, kernel, iterations=1)
    components, nlabels, labels, stats, centroids = CC(dilation)

    # creating the matrices
    a = np.hsplit(stats, 5)
    horizontal = a[2]
    vertical = a[3]
    area = a[4]
    b = np.hsplit(centroids, 2)
    x_centr = b[0]
    y_centr = b[1]
    horizontalNEW = np.zeros(nlabels)
    verticalNEW = np.zeros(nlabels)
    TotalAreaNEW = np.zeros(nlabels)
    NEW_dimensions = np.zeros((nlabels, 6))

    # Logic check if something is DROPLET or NOT
    d = 0
    droplet_counter = 0
    Not_Droplet = np.empty(nlabels, dtype=object)
    for i in range(nlabels):
        d = ((horizontal[i] + vertical[i]) / 2)
        d1 = 0.785 * d * d
        if abs(area[i] - (d1)) > 2500 or\
                horizontal[i] < 5 or \
                vertical[i] < 5 or \
                abs(horizontal[i] - vertical[i]) < 0 :
            Not_Droplet[i] = "NOT a droplet"
        else:
            Not_Droplet[i] = "ok"
            droplet_counter = droplet_counter + 1

    # building the new final dimensions matrix
    for row in range(nlabels):
        for column in range(8):
            if column == 0:
                NEW_dimensions[row, column] = (row + 1)
            elif column == 1:
                NEW_dimensions[row, column] = x_centr[row]
            elif column == 2:
                NEW_dimensions[row, column] = y_centr[row]
            elif column == 3:
                if horizontal[row] < 100:
                    NEW_dimensions[row, column] = horizontal[row] + 20
                else:
                    NEW_dimensions[row, column] = horizontal[row] + 40
            elif column == 4:
                if vertical[row] < 100:
                    NEW_dimensions[row, column] = vertical[row] + 20
                else:
                    NEW_dimensions[row, column] = vertical[row] + 40
            elif column == 5:
                NEW_dimensions[row, column] = ((NEW_dimensions[row][3]) + (NEW_dimensions[row][4])) * 3.14 * 0.25 * (
                            (NEW_dimensions[row][3]) + (NEW_dimensions[row][4]))
        column = column + 1
    row = row + 1
    plt.show()

    #for i in range(nlabels):
        #print(f'horiz {horizontal[i]} - vert {vertical[i]}')
        #print(f'{abs(horizontal[i] - vertical[i])}')


    # here we have to build the surface area difference
    TotalArea_Frame = 956771  # i am not sure about this number for this image - but we dont care about it now
    TotalArea_Droplets = 0
    TotalArea_Background = 0
    d3 = 0
    droplet_counter_2 = 0
    # Not_Droplet = np.empty(nlabels, dtype=object)
    for i in range(nlabels):
        d3 = ((horizontal[i] + vertical[i]) / 2)
        d4 = 0.785 * d3 * d3
        if abs(area[i] - (d4)) < 0 or horizontal[i] < 0 or \
                vertical[i] < 0 or abs(horizontal[i] - vertical[i] > 0):
            pass
        else:
            droplet_counter_2 = droplet_counter_2 + 1
            TotalArea_Droplets = int(TotalArea_Droplets + (NEW_dimensions[i][5]))

    TotalArea_Background = TotalArea_Frame - TotalArea_Droplets
    # print(f'The total area is : { TotalArea_Frame}. '
    #      f' // The droplets area is: { TotalArea_Droplets}. '
    #     f' // The free area is : { TotalArea_Background}.'
    #    f' // The droplets measured here are : {droplet_counter_2}')

    # here we draw the circles, the boxes and the numbers
    image = components
    out = image.copy()
    for row in range(1, nlabels, 1):
        for column in range(5):
            if Not_Droplet[row] == "ok":
                # print(Not_Droplet[row])
                cv2.rectangle(out, (x_centr[row] - 3, y_centr[row] - 3), (x_centr[row] + 3, y_centr[row] + 3),
                              (0, 0, 0))
                r = (math.sqrt(NEW_dimensions[row][5] * 0.31830988618) * 0.5)
                cv2.circle(out, (x_centr[row], y_centr[row]), int(r), (255, 255, 0, 4))
                cv2.putText(out, ('%d' % (row + 1)), (x_centr[row], y_centr[row]), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (255, 255, 255), 2)
            else:
                pass
        column = column + 1
    row = row + 1
    cv2.putText(out, ('%d droplets' % droplet_counter), (5, 30), cv2.FONT_ITALIC, 1.2, (220, 220, 220), 2)

    # here we will build the MatrixA

    # 1st column: Average rate of growth of each droplet in 2 minutes
    # to find the average growth you need the area and the centroid of each droplet
    # DONE!!! 2nd column: Average number of droplets in 2 minutes
    # DONE!!! 3rd column: Average  surface area of empty background in 2 minutes
    MatrixA = np.zeros((nlabels, 3))
    for row in range(nlabels):
        for column in range(0, 3, 1):
            if column == 0:
                MatrixA[row, column] = 1
            elif column == 1:
                MatrixA[row, column] = droplet_counter
            elif column == 2:
                MatrixA[row, column] = TotalArea_Background
        column = column + 1
    row = row + 1

    # show the images
    cv2.imshow("Initial", im_in)
    cv2.imshow("Final", out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


CountingCC()
