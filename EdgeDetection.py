import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

for i in range(1,2):
    temp = './UNWARP/6300/'
    temp = temp + str(i) + '.png'
    image = cv2.imread(temp)
    
    cv2.imshow('Image', image)
    image1 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    print(image1[image1 <=255].shape)
    z = 90
    print(image1[image1 >z].shape)
    
    image1 = np.where(image1 > z, 255, image1)
    image1 = np.where(image1 <= z, 0, image1)

    #cv2.imshow('Original', image)
    cv2.imwrite(str(i)+"a.png",image)

    # Canny Edge Detection ---------
    # performing the edge detection
    #gradients_sobelx = cv2.Sobel(image1, -1, 1, 0)
    #gradients_sobely = cv2.Sobel(image1, -1, 0, 1)
    #gradients_sobelxy = cv2.addWeighted(gradients_sobelx, 0.5, gradients_sobely, 0.5, 0)

    #gradients_laplacian = cv2.Laplacian(image1, -1)

    # canny_output = cv2.Canny(image1, 10, 255) # was 80 and 150
    #cv2.imshow('Sobel x', gradients_sobelx)
    #cv2.imshow('Sobel y', gradients_sobely)
    #cv2.imshow('Sobel X+y', gradients_sobelxy)
    #cv2.imshow('laplacian', gradients_laplacian)
    #cv2.imshow('Canny', canny_output)
    # cv2.waitKey()


    # Contour Detection -----
    ret, thresh = cv2.threshold(image1, 127, 255, 0)
    contours, hierarchy = cv2.findContours(
        thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    print("Number of contours = " + str(len(contours)))

    cv2.drawContours(image,  contours, -1, (255, 255, 0), 1)
    cv2.drawContours(image1, contours, -1, (0, 255, 0), 1)

    cnt = contours[0]
    area = cv2.contourArea(cnt)
    print(area)
    cv2.imwrite(str(i)+"b.png",image)
    cv2.imshow('Image-converted', image)
    #cv2.imshow('Image GRAY', image1)

    ## MIx two Images---
    #img1 = Image.open(str(i)+"a.png")
    #img2 = Image.open(str(i)+"b.png")
    #img1_size = img1.resize((250, 90))
    #img2_size = img2.resize((250, 90))

    # creating a new image and pasting
    # the images
    #img3 = Image.new("RGB", (500, 90), "white")

    # pasting the first image (image_name,
    # (position))
    #print(img1.size)
    #img3.paste(img1_size, (0, 0))

    # pasting the second image (image_name,
    # (position))
    #img3.paste(img2_size, (250, 0))
    #plt.imshow(img3)

    #cv2.imwrite(str(i)+".png",img3)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def get_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst


# for i in range(21,58):
#      im1 = Image.open(str(i)+'a.png')
#      im2 = Image.open(str(i)+'b.png')
#      get_concat_h(im1, im2).save('results/'+str(i)+'ab.png')
#      print("done")